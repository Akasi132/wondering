"""Source extractors. One per source type, each returning a Document.

Both are written against real API surfaces confirmed at build time:
  - trafilatura 2.2.0: fetch_url(url) -> str | None, extract(html, url=...) -> str | None,
    extract_metadata(html) -> settings.Document with a .title attribute.
  - youtube-transcript-api 1.2.4: YouTubeTranscriptApi().fetch(video_id, languages=(...))
    -> FetchedTranscript, which is iterable over FetchedTranscriptSnippet(text, start, duration).
    NOTE: the old static YouTubeTranscriptApi.get_transcript() does not exist in 1.x.
"""

import json
import logging
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from urllib.parse import parse_qs, urlparse

import requests
import trafilatura
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

from app import proxypool

logger = logging.getLogger(__name__)

from app.models import Document

# Single source of truth for "is this a YouTube URL". app/router.py imports this rather
# than reimplementing the check, so the router and the extractor can never disagree about
# which hosts are YouTube (they previously did, and *.youtube.com hard-failed).
YOUTUBE_HOSTS = ("youtube.com", "youtu.be")

# How often to drop a [MM:SS] marker into the transcript text. Markers are what makes a
# YouTube citation checkable: the model can only cite a timestamp it was actually shown.
TIMESTAMP_MARKER_SECONDS = 30.0


class ExtractionError(Exception):
    """Source could not be turned into a Document."""


class NoCaptionsError(ExtractionError):
    """The video exists but has no usable captions. v1 does not fall back to ASR."""


class BlockedError(ExtractionError):
    """YouTube refused *this server*, not this video.

    Kept separate from ExtractionError because the two need opposite responses. An
    ExtractionError means the submitted link was unusable and the caller should try another
    one. This means the link was fine and the server is the problem — YouTube blocks requests
    from datacenter IP ranges, which is where a deployed instance lives. Reporting it as a bad
    link tells the user to fix something they cannot fix.

    Observed on the first Wasmer deploy: URLs that extract cleanly from a residential
    connection return RequestBlocked from Edge. See `_proxy_config` for the way out.
    """


def _host(url: str) -> str:
    """Normalized hostname: lowercase, no port, no leading 'www.'."""
    netloc = urlparse(url).netloc.lower()
    if "@" in netloc:  # strip userinfo so https://youtube.com@evil.com is judged on evil.com
        netloc = netloc.rsplit("@", 1)[1]
    host = netloc.split(":", 1)[0]
    # Trailing dot is a legal absolute-DNS form ("youtube.com.") that browsers accept.
    return host.rstrip(".").removeprefix("www.")


def is_youtube_host(url: str) -> bool:
    host = _host(url)
    # `len(host) > len(h) + 1` rejects an empty leading label like ".youtube.com".
    return any(
        host == h or (host.endswith("." + h) and len(host) > len(h) + 1) for h in YOUTUBE_HOSTS
    )


def _normalize(text: str) -> str:
    """Collapse runs of blank lines and trailing whitespace, leave paragraphs intact."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+\n", "\n", text)  # any trailing whitespace, incl. NBSP
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- article


def extract_article(url: str) -> Document:
    html = trafilatura.fetch_url(url)
    if not html:
        raise ExtractionError(f"Could not fetch {url} (trafilatura.fetch_url returned nothing)")

    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    if not text or not text.strip():
        raise ExtractionError(f"No main text extracted from {url}")

    meta = trafilatura.extract_metadata(html)
    title = (getattr(meta, "title", None) or "").strip() or url

    return Document(
        source_type="article",
        url=url,
        title=title,
        text=_normalize(text),
        anchors=[],
    )


# --------------------------------------------------------------------------- youtube


def youtube_video_id(url: str) -> str:
    """Pull the 11-char video id out of the common YouTube URL shapes."""
    parsed = urlparse(url)
    host = _host(url)

    if host == "youtu.be" or host.endswith(".youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif is_youtube_host(url):
        # Accept any youtube.com subdomain, matching is_youtube_host exactly.
        path = re.sub(r"/+", "/", parsed.path).rstrip("/") or "/"
        if path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [""])[0]
        elif path.startswith(("/embed/", "/shorts/", "/live/", "/v/")):
            candidate = path.split("/")[2]
        else:
            candidate = ""
    else:
        candidate = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        raise ExtractionError(
            f"Could not find a YouTube video id in {url}. v1 handles single videos only — "
            "channel, playlist, search and @handle URLs are not supported."
        )
    return candidate


def _youtube_title(video_id: str) -> str | None:
    """Titles come from YouTube's public oEmbed endpoint (confirmed to return a 'title' key).

    Best-effort: a missing title is not a reason to fail the whole extraction.
    """
    endpoint = (
        "https://www.youtube.com/oembed?url="
        f"https://www.youtube.com/watch?v={video_id}&format=json"
    )
    try:
        req = urllib.request.Request(endpoint, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            return (json.load(response).get("title") or "").strip() or None
    except Exception:
        return None


def _timestamp(seconds: float) -> str:
    """MM:SS, or HH:MM:SS once past an hour, matching what YouTube itself displays."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _transcript_text(snippets, marker_every: float = TIMESTAMP_MARKER_SECONDS):
    """Join caption snippets, interleaving [MM:SS] (or [HH:MM:SS]) markers every `marker_every`.

    The markers are the whole point: without them the model never sees timing and any
    timestamp it cites is invented (Directive 8). Returns (text, markers_used).
    """
    parts: list[str] = []
    markers: list[str] = []
    next_mark = float("-inf")

    for snippet in snippets:
        text = snippet.text.strip()
        if not text:
            continue
        if snippet.start >= next_mark:
            stamp = _timestamp(snippet.start)
            markers.append(stamp)
            parts.append(f"[{stamp}]")
            next_mark = snippet.start + marker_every
        parts.append(text)

    return _normalize(" ".join(parts)), markers


def _proxy_config() -> GenericProxyConfig | WebshareProxyConfig | None:
    """Route transcript fetches through a proxy, if one is configured.

    YouTube blocks datacenter IP ranges, so a deployed instance gets RequestBlocked on videos
    that work fine from a laptop. The library's own remedy is a residential proxy, configured
    per-instance rather than in code:

      WEBSHARE_PROXY_USERNAME / WEBSHARE_PROXY_PASSWORD   -> WebshareProxyConfig
      YOUTUBE_PROXY_HTTPS_URL / YOUTUBE_PROXY_HTTP_URL    -> GenericProxyConfig (any provider)

    Webshare is checked first because it is what youtube-transcript-api documents for exactly
    this failure and it retries on a fresh IP when blocked; the generic form covers everything
    else. Unset means no proxy, which is correct for local use — a residential connection does
    not need one, and requiring one would break the CLI for no reason.

    Signatures confirmed against youtube-transcript-api 1.2.4:
      YouTubeTranscriptApi(proxy_config=None, http_client=None)
      GenericProxyConfig(http_url=None, https_url=None)
      WebshareProxyConfig(proxy_username, proxy_password, filter_ip_locations=None, ...)
    """
    username = (os.getenv("WEBSHARE_PROXY_USERNAME") or "").strip()
    password = (os.getenv("WEBSHARE_PROXY_PASSWORD") or "").strip()
    if username and password:
        return WebshareProxyConfig(proxy_username=username, proxy_password=password)

    https_url = (os.getenv("YOUTUBE_PROXY_HTTPS_URL") or "").strip()
    http_url = (os.getenv("YOUTUBE_PROXY_HTTP_URL") or "").strip()
    if https_url or http_url:
        # Pass both through; the library accepts either being None.
        return GenericProxyConfig(http_url=http_url or None, https_url=https_url or None)

    return None


class _TimeoutSession(requests.Session):
    """A Session with a default timeout.

    requests has no global timeout and youtube-transcript-api never passes one, so a proxy
    that accepts a connection and then goes quiet would hang the request forever. That is the
    normal behaviour of a dead free proxy, and rotation is worthless without a bound on it.
    """

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self._timeout = timeout

    def request(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        return super().request(*args, **kwargs)


def _fetch_transcript(video_id: str, languages: tuple[str, ...]):
    """Fetch a transcript, rotating through proxies only if the host itself is blocked.

    Order of preference:
      1. An explicitly configured proxy (Webshare or generic) — the operator said so.
      2. A direct connection. This is what works from a residential network, and costs one
         request to find out.
      3. If and only if direct came back RequestBlocked *and* rotation is enabled, work
         through the pool.

    Rotation is deliberately not tried for any other failure. A missing transcript or an
    unavailable video is a fact about the video, and no proxy changes it — burning eight
    proxies to be told the same thing eight times would just make the endpoint slower.

    A new YouTubeTranscriptApi (and Session) per attempt, because the library documents the
    class as not thread-safe and a fresh session also prevents a keep-alive connection to a
    dead proxy from being reused.
    """
    explicit = _proxy_config()
    if explicit is not None:
        return YouTubeTranscriptApi(
            proxy_config=explicit, http_client=_TimeoutSession(proxypool.attempt_timeout())
        ).fetch(video_id, languages=languages)

    try:
        return YouTubeTranscriptApi(
            http_client=_TimeoutSession(proxypool.attempt_timeout())
        ).fetch(video_id, languages=languages)
    except RequestBlocked:
        if not proxypool.enabled():
            raise
        logger.info("Direct fetch blocked for %s; rotating through the proxy pool", video_id)

    return _fetch_through_pool(video_id, languages)


def _attempt(address: str, video_id: str, languages: tuple[str, ...], timeout: float):
    """One proxy, one try. Returns a tagged outcome and never raises.

    Tagged rather than raising because this runs inside a worker thread, where an exception
    would only surface when the future is read and would lose the distinction between
    "this proxy is dead" and "this video has no captions".
    """
    config = GenericProxyConfig(http_url=f"http://{address}", https_url=f"http://{address}")
    try:
        fetched = YouTubeTranscriptApi(
            proxy_config=config, http_client=_TimeoutSession(timeout)
        ).fetch(video_id, languages=languages)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
        # Checked before the broad clause: these subclass the same base as RequestBlocked.
        # YouTube answered — the video is the problem, and no other proxy will differ.
        return "definitive", exc
    except RequestBlocked as exc:
        return "blocked", exc
    except CouldNotRetrieveTranscript as exc:
        return "dead", exc
    except Exception as exc:
        # Refused, timed out, TLS failure, truncated response. Expected: most free proxies
        # fail here, so it is not worth a warning.
        return "dead", exc
    return "ok", fetched


def _fetch_through_pool(video_id: str, languages: tuple[str, ...]):
    """Try pooled proxies in parallel, first success wins.

    Parallel because the attempts are independent and almost entirely idle waiting — see
    `proxypool.concurrency`. The deadline covers the whole fan-out, not each attempt, so this
    cannot extend an already-slow request beyond its budget.
    """
    pool = proxypool.POOL
    candidates = pool.candidates()[: proxypool.max_attempts()]
    if not candidates:
        logger.warning("Proxy rotation is on but the pool is empty; nothing to try")
        raise RequestBlocked(video_id)

    timeout = proxypool.attempt_timeout()
    budget = proxypool.deadline_seconds()
    began = time.monotonic()

    blocked = 0
    last_blocked: RequestBlocked | None = None

    # Not a `with` block: on success the remaining futures are abandoned rather than waited
    # on, and `with` would block until every straggler finished its timeout.
    executor = ThreadPoolExecutor(max_workers=proxypool.concurrency())
    try:
        futures = {
            executor.submit(_attempt, address, video_id, languages, timeout): address
            for address in candidates
        }
        try:
            for future in as_completed(futures, timeout=budget):
                address = futures[future]
                kind, payload = future.result()

                if kind == "ok":
                    logger.info(
                        "Proxy %s worked after %.1fs", address, time.monotonic() - began
                    )
                    pool.promote(address)
                    return payload
                if kind == "definitive":
                    pool.promote(address)  # it reached YouTube, so it is a good proxy
                    raise payload
                if kind == "blocked":
                    blocked += 1
                    last_blocked = payload
                pool.demote(address)
        except FuturesTimeout:
            logger.warning("Proxy rotation hit its %.0fs deadline for %s", budget, video_id)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    logger.warning(
        "Proxy rotation exhausted for %s: %d candidate(s), %d blocked, %.1fs elapsed",
        video_id,
        len(candidates),
        blocked,
        time.monotonic() - began,
    )
    # Re-raise a real RequestBlocked so the caller's handling is unchanged: this still ends up
    # as BlockedError -> 503, which is the honest outcome. Rotation reduces how often that
    # happens; it does not change what it means.
    raise last_blocked or RequestBlocked(video_id)


def extract_youtube(url: str, languages: tuple[str, ...] = ("en",)) -> Document:
    video_id = youtube_video_id(url)

    try:
        fetched = _fetch_transcript(video_id, languages)
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        raise NoCaptionsError(
            f"No captions available for {url} in languages {list(languages)}. "
            f"v1 has no audio-transcription fallback. ({type(exc).__name__})"
        ) from exc
    except VideoUnavailable as exc:
        raise ExtractionError(f"Video unavailable: {url} ({type(exc).__name__})") from exc
    except RequestBlocked as exc:
        # Checked before CouldNotRetrieveTranscript, which is its base class. IpBlocked
        # subclasses RequestBlocked, so this one clause covers both.
        logger.warning(
            "YouTube blocked this host while fetching %s (%s). %s",
            url,
            type(exc).__name__,
            "No proxy configured." if _proxy_config() is None else "A proxy IS configured.",
        )
        raise BlockedError(
            f"YouTube is blocking requests from this server, so it could not read {url}. "
            f"The link itself is fine. ({type(exc).__name__})"
        ) from exc
    except CouldNotRetrieveTranscript as exc:
        # Base class for RequestBlocked, IpBlocked, AgeRestricted, VideoUnplayable,
        # YouTubeRequestFailed, PoTokenRequired, ... Without this, a datacenter IP being
        # blocked by YouTube surfaces as an unhandled 500 instead of a clean error.
        raise ExtractionError(
            f"Could not retrieve the transcript for {url} ({type(exc).__name__})"
        ) from exc
    except OSError as exc:  # requests' network errors subclass OSError
        raise ExtractionError(f"Network error fetching the transcript for {url}: {exc}") from exc

    snippets = list(fetched)
    if not snippets:
        raise NoCaptionsError(f"Transcript for {url} came back empty.")

    text, markers = _transcript_text(snippets)
    if not text:
        raise NoCaptionsError(f"Transcript for {url} contained no text.")

    return Document(
        source_type="youtube",
        url=url,
        title=_youtube_title(video_id) or f"YouTube video {video_id}",
        text=text,
        anchors=markers,
    )
