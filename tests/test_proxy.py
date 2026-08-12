"""Proxy selection and the blocked-request path. Offline: no network, no real YouTube call.

The live extractor tests are in tests/test_extract.py and hit the network on purpose. These
do not — the whole point is to exercise failure modes that only occur on a datacenter IP,
which is not where the suite runs.

Context: the first Wasmer Edge deploy returned RequestBlocked for a video that extracts fine
from a residential connection. `_proxy_config` is the documented remedy and `BlockedError`
keeps that failure from being reported as a bad URL.

Run with:  python -m pytest tests/test_proxy.py -q
"""

import pytest
from youtube_transcript_api import IpBlocked, PoTokenRequired, RequestBlocked
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

from app import extract
from app.extract import BlockedError, ExtractionError, _proxy_config, extract_youtube

PROXY_VARS = (
    "WEBSHARE_PROXY_USERNAME",
    "WEBSHARE_PROXY_PASSWORD",
    "YOUTUBE_PROXY_HTTP_URL",
    "YOUTUBE_PROXY_HTTPS_URL",
    # Rotation too: with it on, a "blocked" test would start walking a real proxy list over
    # the network instead of asserting on the mapping.
    "YOUTUBE_PROXY_ROTATE",
    "YOUTUBE_PROXY_LIST",
    "YOUTUBE_PROXY_LIST_URL",
)

VIDEO_URL = "https://www.youtube.com/watch?v=3RwUIP9pMSo"


@pytest.fixture(autouse=True)
def _no_ambient_proxy(monkeypatch):
    """A developer's real .env must not decide what these assert."""
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------- _proxy_config


def test_no_proxy_configured_returns_none():
    """Unset must mean "no proxy", so the CLI and local dev keep working untouched."""
    assert _proxy_config() is None


def test_webshare_credentials_build_a_webshare_config(monkeypatch):
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "user-1")
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", "secret-1")

    config = _proxy_config()

    assert isinstance(config, WebshareProxyConfig)
    assert config.proxy_username == "user-1"
    assert config.proxy_password == "secret-1"


def test_generic_https_url_builds_a_generic_config(monkeypatch):
    monkeypatch.setenv("YOUTUBE_PROXY_HTTPS_URL", "https://user:pw@proxy.example:8080")

    config = _proxy_config()

    assert isinstance(config, GenericProxyConfig)


def test_webshare_wins_when_both_are_configured(monkeypatch):
    """Documented precedence. Silently merging two proxy configs would be worse than picking."""
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "user-1")
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", "secret-1")
    monkeypatch.setenv("YOUTUBE_PROXY_HTTPS_URL", "https://proxy.example:8080")

    assert isinstance(_proxy_config(), WebshareProxyConfig)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_credentials_are_treated_as_unset(monkeypatch, blank):
    """An env var present-but-empty is the normal shape of a half-filled deploy config.

    Passing "" to WebshareProxyConfig would build a proxy that authenticates as nobody and
    fail every fetch with something far less obvious than "no proxy configured".
    """
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", blank)
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", blank)

    assert _proxy_config() is None


def test_partial_webshare_credentials_do_not_build_a_config(monkeypatch):
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "user-1")

    assert _proxy_config() is None


# ------------------------------------------------------- blocked -> BlockedError


class _StubApi:
    """Stands in for YouTubeTranscriptApi, recording what it was constructed with."""

    last_proxy_config = "not-set"
    last_http_client = "not-set"

    def __init__(self, proxy_config=None, http_client=None):
        type(self).last_proxy_config = proxy_config
        type(self).last_http_client = http_client
        self._proxy_config = proxy_config

    def fetch(self, video_id, languages=("en",)):
        raise self.to_raise(video_id)


def _stub_raising(monkeypatch, exception_type):
    stub = type("Stub", (_StubApi,), {"to_raise": staticmethod(exception_type)})
    monkeypatch.setattr(extract, "YouTubeTranscriptApi", stub)
    return stub


def test_request_blocked_becomes_blocked_error(monkeypatch):
    _stub_raising(monkeypatch, RequestBlocked)

    with pytest.raises(BlockedError) as caught:
        extract_youtube(VIDEO_URL)

    # The message has to say whose fault it is, because the front end shows it verbatim.
    assert "blocking requests from this server" in str(caught.value)
    assert "RequestBlocked" in str(caught.value)


def test_ip_blocked_also_becomes_blocked_error(monkeypatch):
    """IpBlocked subclasses RequestBlocked, so one clause covers both. Pin that."""
    assert issubclass(IpBlocked, RequestBlocked)
    _stub_raising(monkeypatch, IpBlocked)

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)


def test_other_retrieval_failures_stay_plain_extraction_errors(monkeypatch):
    """PoTokenRequired is a CouldNotRetrieveTranscript but NOT a RequestBlocked.

    It must keep mapping to 400, not 503 — it is not an IP-reputation problem and telling the
    user "this server is blocked" would be a lie.
    """
    assert not issubclass(PoTokenRequired, RequestBlocked)
    _stub_raising(monkeypatch, PoTokenRequired)

    with pytest.raises(ExtractionError) as caught:
        extract_youtube(VIDEO_URL)

    assert not isinstance(caught.value, BlockedError)


def test_blocked_error_is_an_extraction_error():
    """Callers that only know about ExtractionError must still catch this."""
    assert issubclass(BlockedError, ExtractionError)


# ------------------------------------------------- the config actually reaches the client


def test_configured_proxy_is_passed_to_the_transcript_client(monkeypatch):
    """The bug this prevents: building a proxy config and never handing it over."""
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "user-1")
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", "secret-1")
    stub = _stub_raising(monkeypatch, RequestBlocked)

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)

    assert isinstance(stub.last_proxy_config, WebshareProxyConfig)
    assert stub.last_proxy_config.proxy_username == "user-1"


def test_no_proxy_passes_none_rather_than_omitting_the_argument(monkeypatch):
    stub = _stub_raising(monkeypatch, RequestBlocked)

    with pytest.raises(BlockedError):
        extract_youtube(VIDEO_URL)

    assert stub.last_proxy_config is None
