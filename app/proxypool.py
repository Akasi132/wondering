"""A rotating pool of free proxies, for when YouTube blocks the host's IP.

Why this exists: YouTube refuses transcript requests from datacenter IP ranges, so a hosted
instance gets RequestBlocked on videos that work fine from a laptop. A paid residential proxy
is the reliable answer; this is the free one.

**It is unreliable by construction, and that is measured, not assumed.** Sampling 92 proxies
from the default list on 2026-08-11: 11 could reach YouTube at all, 5 of those were blocked,
4 failed at the proxy itself, and 2 successfully fetched a transcript. A ~2% hit rate is why
rotation is required — pointing a single static proxy at a free list gives a site that works
until that one IP dies, then fails silently. Trying many and remembering the winner is the
only shape that works at that hit rate.

This module owns the *list* only: which addresses to try, in what order, and which one worked
last. The fetch loop lives in `app.extract`, so nothing here has to know about YouTube and the
pool can be tested without touching the network.

Off unless `YOUTUBE_PROXY_ROTATE` is set. Routing traffic through unvetted third-party proxies
should be a deliberate choice, not a default.
"""

import logging
import os
import random
import threading
import time

import requests

logger = logging.getLogger(__name__)

# A third-party list, refreshed by its maintainers every 30 minutes. Named here rather than
# vendored so it does not go stale in git; override it, or supply YOUTUBE_PROXY_LIST, to
# depend on something you control.
DEFAULT_LIST_URL = "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/http.txt"

DEFAULT_LIST_TTL_SECONDS = 900.0
# Tuned against measurements in NOTES.md. A bigger candidate pool costs nothing when a proxy
# is found early — success is typically under 20s — and only spends the full budget on the
# way to failing, where there is no model call to pay for afterwards anyway.
DEFAULT_MAX_ATTEMPTS = 40
DEFAULT_ATTEMPT_TIMEOUT = 5.0
DEFAULT_DEADLINE_SECONDS = 60.0
DEFAULT_CONCURRENCY = 6

# Guards against a list URL that starts returning a phone book.
MAX_CANDIDATES = 400


def _flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _number(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    return value if value > 0 else default


def enabled() -> bool:
    return _flag("YOUTUBE_PROXY_ROTATE")


def max_attempts() -> int:
    return int(_number("YOUTUBE_PROXY_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))


def attempt_timeout() -> float:
    return _number("YOUTUBE_PROXY_TIMEOUT", DEFAULT_ATTEMPT_TIMEOUT)


def deadline_seconds() -> float:
    """Total wall-clock budget for rotation.

    POST /ingest is synchronous and already slow, so the retry loop needs a ceiling that does
    not depend on how many proxies happen to hang. Attempts are capped too; whichever bites
    first wins.
    """
    return _number("YOUTUBE_PROXY_DEADLINE", DEFAULT_DEADLINE_SECONDS)


def concurrency() -> int:
    """How many proxies to try at once.

    Measured, not guessed (2026-08-11, same video, cold start each time):

        sequential, 8 attempts     1/3 succeeded, 35-50s
        6 at a time, 24 candidates 6/6 succeeded, median 10.9s
        6 at a time, 40 candidates 4/5 succeeded, from an origin IP YouTube had blocked

    Almost all of an attempt is idle waiting on a proxy that is simply dead, and the attempts
    are independent, so parallelism fits several times as many candidates into the same budget.
    That is what moved the hit rate — it is the difference between this feature being usable
    and being a coin flip that costs 40 seconds.

    Set to 1 for deterministic ordering (the tests do this).
    """
    return max(1, int(_number("YOUTUBE_PROXY_CONCURRENCY", DEFAULT_CONCURRENCY)))


def _parse(text: str) -> list[str]:
    """Pull `host:port` lines out of a proxy list, ignoring comments and junk."""
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Tolerate "1.2.3.4:8080", "http://1.2.3.4:8080", and trailing fields.
        line = line.split()[0]
        line = line.rsplit("/", 1)[-1] if "://" in line else line
        host, _, port = line.partition(":")
        if host and port.isdigit():
            found.append(f"{host}:{port}")
    return found


class ProxyPool:
    """Candidate proxies, ordered so the last known-good one is tried first.

    Thread-safe: `/ingest` is a sync endpoint, so FastAPI runs it in a threadpool and several
    requests can be rotating at once. Without the lock they would each refetch the list.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._candidates: list[str] = []
        self._fetched_at: float | None = None
        self._known_good: str | None = None

    # -------------------------------------------------------------- list management

    def _stale(self) -> bool:
        if self._fetched_at is None:
            return True
        ttl = _number("YOUTUBE_PROXY_LIST_TTL", DEFAULT_LIST_TTL_SECONDS)
        return (time.monotonic() - self._fetched_at) > ttl

    def _load(self) -> list[str]:
        """Inline list wins over the URL, so an operator can pin their own proxies."""
        inline = (os.getenv("YOUTUBE_PROXY_LIST") or "").strip()
        if inline:
            return _parse(inline.replace(",", "\n"))

        url = (os.getenv("YOUTUBE_PROXY_LIST_URL") or DEFAULT_LIST_URL).strip()
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            # Never fatal: an unreachable list means no rotation, not a failed request.
            logger.warning("Could not fetch the proxy list from %s: %s", url, exc)
            return []
        return _parse(response.text)

    def refresh(self, *, force: bool = False) -> None:
        with self._lock:
            if force or self._stale():
                found = self._load()[:MAX_CANDIDATES]
                self._candidates = found
                self._fetched_at = time.monotonic()
                logger.info("Proxy pool loaded %d candidates", len(found))

    # -------------------------------------------------------------------- ordering

    def candidates(self) -> list[str]:
        """Addresses to try, best guess first.

        Shuffled rather than taken in list order: the source list is public and everyone
        reading it top-down hammers the same few addresses, so the head of the list is the
        most likely to be rate-limited. The last known-good proxy goes first regardless — at
        a ~2% hit rate, re-finding a working proxy from scratch on every request would make
        rotation cost more than it saves.
        """
        self.refresh()
        with self._lock:
            rest = [addr for addr in self._candidates if addr != self._known_good]
            random.shuffle(rest)
            return ([self._known_good] if self._known_good else []) + rest

    def promote(self, address: str) -> None:
        with self._lock:
            self._known_good = address

    def demote(self, address: str) -> None:
        """Forget a proxy that just failed, so the next request does not lead with it."""
        with self._lock:
            if self._known_good == address:
                self._known_good = None

    # ------------------------------------------------------------------ inspection

    @property
    def known_good(self) -> str | None:
        return self._known_good

    def reset(self) -> None:
        """Drop all state. For tests, and for a future admin endpoint."""
        with self._lock:
            self._candidates = []
            self._fetched_at = None
            self._known_good = None


POOL = ProxyPool()
