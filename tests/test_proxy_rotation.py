"""Proxy rotation: pool ordering, and when the retry loop keeps going vs stops.

Offline. The pool's list source is stubbed with YOUTUBE_PROXY_LIST (inline), and the
transcript client is stubbed, so nothing here touches the network or a real proxy.

The behaviour that matters most is *when not to rotate*. A video with no captions must not
burn the whole pool being told the same thing eight times.

Run with:  python -m pytest tests/test_proxy_rotation.py -q
"""

import pytest
from youtube_transcript_api import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from app import extract, proxypool
from app.extract import BlockedError, NoCaptionsError, extract_youtube
from app.proxypool import ProxyPool, _parse

VIDEO_URL = "https://www.youtube.com/watch?v=3RwUIP9pMSo"

PROXY_ENV = (
    "WEBSHARE_PROXY_USERNAME",
    "WEBSHARE_PROXY_PASSWORD",
    "YOUTUBE_PROXY_HTTP_URL",
    "YOUTUBE_PROXY_HTTPS_URL",
    "YOUTUBE_PROXY_ROTATE",
    "YOUTUBE_PROXY_LIST",
    "YOUTUBE_PROXY_LIST_URL",
    "YOUTUBE_PROXY_MAX_ATTEMPTS",
    "YOUTUBE_PROXY_TIMEOUT",
    "YOUTUBE_PROXY_DEADLINE",
    "YOUTUBE_PROXY_LIST_TTL",
    "YOUTUBE_PROXY_CONCURRENCY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    proxypool.POOL.reset()
    yield
    proxypool.POOL.reset()


@pytest.fixture
def rotating(monkeypatch):
    """Rotation on, fixed inline list (no network), and serial so ordering is deterministic.

    Production fans out across several proxies at once, which makes the order attempts
    complete nondeterministic. Pinning concurrency to 1 lets these tests assert on sequence;
    `test_concurrent_rotation_*` cover the parallel path.
    """
    monkeypatch.setenv("YOUTUBE_PROXY_ROTATE", "1")
    monkeypatch.setenv("YOUTUBE_PROXY_LIST", "1.1.1.1:80,2.2.2.2:80,3.3.3.3:80,4.4.4.4:80")
    monkeypatch.setenv("YOUTUBE_PROXY_CONCURRENCY", "1")
    return ["1.1.1.1:80", "2.2.2.2:80", "3.3.3.3:80", "4.4.4.4:80"]


# ------------------------------------------------------------------- list parsing


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1.2.3.4:8080", ["1.2.3.4:8080"]),
        ("http://1.2.3.4:8080", ["1.2.3.4:8080"]),
        ("# a comment\n1.2.3.4:8080", ["1.2.3.4:8080"]),
        ("1.2.3.4:8080 extra fields here", ["1.2.3.4:8080"]),
        ("  \n\n1.2.3.4:8080\n\n", ["1.2.3.4:8080"]),
        ("not-a-proxy", []),
        ("1.2.3.4:notaport", []),
    ],
)
def test_parse_tolerates_real_world_list_formats(raw, expected):
    assert _parse(raw) == expected


# ---------------------------------------------------------------------- ordering


def test_known_good_proxy_is_tried_first(monkeypatch):
    monkeypatch.setenv("YOUTUBE_PROXY_LIST", "1.1.1.1:80,2.2.2.2:80,3.3.3.3:80")
    pool = ProxyPool()

    pool.promote("3.3.3.3:80")

    assert pool.candidates()[0] == "3.3.3.3:80"


def test_candidates_contains_every_address_exactly_once(monkeypatch):
    """Shuffling must not drop or duplicate the known-good entry."""
    monkeypatch.setenv("YOUTUBE_PROXY_LIST", "1.1.1.1:80,2.2.2.2:80,3.3.3.3:80")
    pool = ProxyPool()
    pool.promote("2.2.2.2:80")

    candidates = pool.candidates()

    assert sorted(candidates) == ["1.1.1.1:80", "2.2.2.2:80", "3.3.3.3:80"]
    assert len(candidates) == len(set(candidates))


def test_demote_clears_the_known_good_proxy(monkeypatch):
    monkeypatch.setenv("YOUTUBE_PROXY_LIST", "1.1.1.1:80,2.2.2.2:80")
    pool = ProxyPool()
    pool.promote("1.1.1.1:80")

    pool.demote("1.1.1.1:80")

    assert pool.known_good is None


def test_demoting_a_different_proxy_leaves_the_known_good_alone(monkeypatch):
    monkeypatch.setenv("YOUTUBE_PROXY_LIST", "1.1.1.1:80,2.2.2.2:80")
    pool = ProxyPool()
    pool.promote("1.1.1.1:80")

    pool.demote("2.2.2.2:80")

    assert pool.known_good == "1.1.1.1:80"


def test_an_unreachable_list_yields_no_candidates_rather_than_raising(monkeypatch):
    """A dead list URL must degrade to "no rotation", not break the request."""
    monkeypatch.setenv("YOUTUBE_PROXY_LIST_URL", "http://127.0.0.1:9/nope.txt")
    pool = ProxyPool()

    assert pool.candidates() == []


# ------------------------------------------------------------------- the loop


class _ScriptedApi:
    """Returns or raises per proxy address, and records the order tried."""

    script: dict = {}
    attempts: list = []

    def __init__(self, proxy_config=None, http_client=None):
        if proxy_config is None:
            self.address = None
        else:
            self.address = proxy_config.to_requests_dict()["https"].removeprefix("http://")

    def fetch(self, video_id, languages=("en",)):
        type(self).attempts.append(self.address)
        outcome = type(self).script.get(self.address, RequestBlocked)
        # Instances as well as classes: NoTranscriptFound takes three constructor args, so it
        # cannot be built from the video id alone like the others can.
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            raise outcome(video_id)
        return outcome


def scripted(monkeypatch, script):
    stub = type("Scripted", (_ScriptedApi,), {"script": script, "attempts": []})
    monkeypatch.setattr(extract, "YouTubeTranscriptApi", stub)
    return stub


def _transcript(text="hello world"):
    """Minimal stand-in for a FetchedTranscript: iterable of snippets."""
    snippet = type("Snippet", (), {"text": text, "start": 0.0, "duration": 1.0})()
    return [snippet]


def test_direct_success_never_touches_the_pool(rotating, monkeypatch):
    """A working direct connection must not pay for rotation at all."""
    stub = scripted(monkeypatch, {None: _transcript()})

    doc = extract_youtube(VIDEO_URL)

    assert stub.attempts == [None]
    assert "hello world" in doc.text


def test_rotation_stops_at_the_first_working_proxy(rotating, monkeypatch):
    stub = scripted(monkeypatch, {None: RequestBlocked, "3.3.3.3:80": _transcript()})

    doc = extract_youtube(VIDEO_URL)

    assert "hello world" in doc.text
    assert stub.attempts[0] is None  # direct is always tried first
    assert "3.3.3.3:80" in stub.attempts
    assert proxypool.POOL.known_good == "3.3.3.3:80"
    # Note: not asserting the working proxy was tried *last*. Candidates are submitted up
    # front, so a win stops the waiting, not necessarily an already-running attempt.


def test_a_working_proxy_is_remembered_for_the_next_request(rotating, monkeypatch):
    scripted(monkeypatch, {None: RequestBlocked, "2.2.2.2:80": _transcript()})
    extract_youtube(VIDEO_URL)

    stub = scripted(monkeypatch, {None: RequestBlocked, "2.2.2.2:80": _transcript()})
    extract_youtube(VIDEO_URL)

    # Direct, then straight to the remembered winner rather than re-scanning the pool.
    assert stub.attempts[:2] == [None, "2.2.2.2:80"]


def test_no_captions_stops_rotation_immediately(rotating, monkeypatch):
    """The decisive test: a video-level failure must not burn the pool.

    NoTranscriptFound subclasses CouldNotRetrieveTranscript, same as RequestBlocked, so
    without explicit ordering in the loop this would retry through every proxy to be told
    the same thing each time.
    """
    no_captions = NoTranscriptFound("3RwUIP9pMSo", ["en"], None)
    stub = scripted(monkeypatch, {None: RequestBlocked, "1.1.1.1:80": no_captions})
    proxypool.POOL.promote("1.1.1.1:80")

    with pytest.raises(NoCaptionsError):
        extract_youtube(VIDEO_URL)

    assert stub.attempts[:2] == [None, "1.1.1.1:80"]
    # The point of the test: it did not work through the rest of the pool to be told the
    # same thing four times.
    assert len(stub.attempts) < 1 + len(rotating)


@pytest.mark.parametrize("definitive", [TranscriptsDisabled, VideoUnavailable])
def test_other_video_level_failures_also_stop_rotation(rotating, monkeypatch, definitive):
    stub = scripted(monkeypatch, {None: RequestBlocked, "1.1.1.1:80": definitive})
    proxypool.POOL.promote("1.1.1.1:80")

    with pytest.raises(Exception):
        extract_youtube(VIDEO_URL)

    assert stub.attempts[:2] == [None, "1.1.1.1:80"]
    assert len(stub.attempts) < 1 + len(rotating)


def test_exhausting_the_pool_raises_blocked_error(rotating, monkeypatch):
    """Rotation changes how often blocking happens, not what it means when it does."""
    scripted(monkeypatch, {})  # everything blocks

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)


def test_max_attempts_caps_the_number_of_proxies_tried(rotating, monkeypatch):
    monkeypatch.setenv("YOUTUBE_PROXY_MAX_ATTEMPTS", "2")
    stub = scripted(monkeypatch, {})

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)

    # One direct attempt plus exactly two proxies.
    assert len(stub.attempts) == 3


def test_dead_proxies_are_skipped_rather_than_ending_rotation(rotating, monkeypatch):
    """Roughly 9 in 10 free proxies fail at the transport layer. That is not a stop condition."""

    class _Dead(Exception):
        pass

    stub = scripted(
        monkeypatch,
        {None: RequestBlocked, "1.1.1.1:80": _Dead, "2.2.2.2:80": _Dead, "3.3.3.3:80": _transcript()},
    )
    proxypool.POOL.promote("1.1.1.1:80")

    doc = extract_youtube(VIDEO_URL)

    assert "hello world" in doc.text
    assert "3.3.3.3:80" in stub.attempts


def test_rotation_is_off_unless_explicitly_enabled(monkeypatch):
    """Default must not route anyone's traffic through third-party proxies."""
    monkeypatch.setenv("YOUTUBE_PROXY_LIST", "1.1.1.1:80,2.2.2.2:80")
    stub = scripted(monkeypatch, {})

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)

    assert stub.attempts == [None]
    assert not proxypool.enabled()


def test_an_explicit_proxy_disables_rotation(rotating, monkeypatch):
    """If the operator named a proxy, use it and report its result — do not silently fan out."""
    monkeypatch.setenv("YOUTUBE_PROXY_HTTPS_URL", "http://9.9.9.9:80")
    stub = scripted(monkeypatch, {})

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)

    assert stub.attempts == ["9.9.9.9:80"]


# ------------------------------------------------------------------ concurrency


def test_concurrent_rotation_finds_a_working_proxy_among_dead_ones(monkeypatch):
    """The parallel path still returns the one good result out of many failures."""
    monkeypatch.setenv("YOUTUBE_PROXY_ROTATE", "1")
    monkeypatch.setenv("YOUTUBE_PROXY_CONCURRENCY", "6")
    monkeypatch.setenv(
        "YOUTUBE_PROXY_LIST",
        ",".join(f"10.0.0.{n}:80" for n in range(1, 21)),
    )

    class _Dead(Exception):
        pass

    script = {None: RequestBlocked}
    script.update({f"10.0.0.{n}:80": _Dead for n in range(1, 21)})
    script["10.0.0.17:80"] = _transcript()
    scripted(monkeypatch, script)

    doc = extract_youtube(VIDEO_URL)

    assert "hello world" in doc.text
    assert proxypool.POOL.known_good == "10.0.0.17:80"


def test_concurrent_rotation_reports_blocked_when_every_proxy_fails(monkeypatch):
    monkeypatch.setenv("YOUTUBE_PROXY_ROTATE", "1")
    monkeypatch.setenv("YOUTUBE_PROXY_CONCURRENCY", "6")
    monkeypatch.setenv("YOUTUBE_PROXY_LIST", ",".join(f"10.0.0.{n}:80" for n in range(1, 13)))
    scripted(monkeypatch, {})

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)


def test_an_empty_pool_reports_blocked_rather_than_hanging(monkeypatch):
    """Rotation enabled but the list came back empty — must fail fast, not spin."""
    monkeypatch.setenv("YOUTUBE_PROXY_ROTATE", "1")
    monkeypatch.setenv("YOUTUBE_PROXY_LIST_URL", "http://127.0.0.1:9/nope.txt")
    scripted(monkeypatch, {})

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)


def test_a_bad_max_attempts_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("YOUTUBE_PROXY_MAX_ATTEMPTS", "not-a-number")

    assert proxypool.max_attempts() == proxypool.DEFAULT_MAX_ATTEMPTS


def test_a_zero_deadline_falls_back_to_the_default(monkeypatch):
    """0 would mean "never rotate", which is what the off switch is for. Treat it as unset."""
    monkeypatch.setenv("YOUTUBE_PROXY_DEADLINE", "0")

    assert proxypool.deadline_seconds() == proxypool.DEFAULT_DEADLINE_SECONDS
