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
import urllib.request
from urllib.parse import parse_qs, urlparse

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


def extract_youtube(url: str, languages: tuple[str, ...] = ("en",)) -> Document:
    video_id = youtube_video_id(url)

    try:
        fetched = YouTubeTranscriptApi(proxy_config=_proxy_config()).fetch(
            video_id, languages=languages
        )
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
