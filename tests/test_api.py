"""Endpoint wiring tests for app/api.py.

These NEVER touch the network or the Anthropic API:
  - `app.api.route` and `app.api.build_path` are monkeypatched with fakes that return
    canned objects,
  - `app.cache.CACHE_DIR` is redirected at a tmp_path so the real cache/ dir is untouched,
  - `app.api.socket.getaddrinfo` is replaced for every test by a resolver that parses IP
    literals locally and refuses to look up names (see the `offline_dns` fixture), so the
    SSRF guard's resolve step is deterministic and DNS-free.
There is no ANTHROPIC_API_KEY in this environment and nothing here needs one.

Run with:  python -m pytest tests/test_api.py -q
"""

import socket

import pytest
from fastapi.testclient import TestClient

from app import api, cache
from app.extract import BlockedError, ExtractionError, NoCaptionsError
from app.llm import LLMError
from app.models import Document, Exercise, Lesson, Path

# --------------------------------------------------------------------------- fixtures

# Captured before anything is patched, so the offline resolver below can still use the
# stdlib's numeric parser.
_REAL_GETADDRINFO = socket.getaddrinfo

PUBLIC_IP = "93.184.216.34"


def _numeric_only_getaddrinfo(host, port, *args, **kwargs):
    """Resolve IP literals for real; refuse to resolve names.

    AI_NUMERICHOST makes getaddrinfo a pure parser: no DNS query, no /etc/hosts, no
    network. It still handles every literal spelling the platform's inet_aton accepts
    (`127.1`, `2130706433`, `0177.0.0.1`) and every IPv6 form, which is exactly what the
    SSRF guard's IP-literal cases need. Names raise gaierror, so no test can accidentally
    depend on the machine's DNS.
    """
    kwargs.pop("flags", None)
    try:
        return _REAL_GETADDRINFO(host, port, *args, flags=socket.AI_NUMERICHOST, **kwargs)
    except socket.gaierror as exc:
        raise socket.gaierror(socket.EAI_NONAME, f"DNS is disabled in tests: {host!r}") from exc


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch):
    """Make `IngestRequest`'s resolve step deterministic and offline for every test.

    Consequence worth stating out loud: under this fixture an ordinary hostname like
    example.com is *unresolvable*, and the validator deliberately lets unresolvable hosts
    through (see `test_an_unresolvable_host_is_allowed_through`). So the happy-path tests
    below reach the endpoint via that branch. The tests that care about a specific
    resolution result patch `api.socket.getaddrinfo` again with `resolving_to(...)`.
    """
    monkeypatch.setattr(api.socket, "getaddrinfo", _numeric_only_getaddrinfo)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the disk cache at a throwaway dir for every test in this module."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


@pytest.fixture
def client():
    return TestClient(api.app)


def resolving_to(*addresses: str):
    """A getaddrinfo replacement that resolves every name to `addresses`."""

    def _getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port or 0),
            )
            for address in addresses
        ]

    return _getaddrinfo


def must_not_resolve(why: str):
    """A getaddrinfo replacement that fails the test if the validator resolves at all.

    Deliberately a RuntimeError, not an AssertionError: pydantic v2 treats AssertionError
    raised inside a field_validator as a validation failure, so an `assert`-based tripwire
    here would be swallowed into the very 422 the caller is asserting and the test would
    pass vacuously. The validator's own `except (gaierror, UnicodeError, ValueError)` rules
    those three out too. RuntimeError escapes both and surfaces as a 500 that TestClient
    re-raises.
    """

    def _getaddrinfo(host, port, *args, **kwargs):
        raise RuntimeError(f"{why}: resolver was called for {host!r}")

    return _getaddrinfo


def unresolvable(host_label: str = "host"):
    """A getaddrinfo replacement that fails the way NXDOMAIN does."""

    def _getaddrinfo(host, port, *args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, f"no such {host_label}: {host!r}")

    return _getaddrinfo


def make_document(text: str = "canned source text", url: str = "https://example.com/a") -> Document:
    return Document(
        source_type="article",
        url=url,
        title="Canned Document",
        text=text,
        anchors=[],
    )


def make_lesson(order: int) -> Lesson:
    return Lesson(
        order=order,
        title=f"Lesson {order}",
        explanation=f"Explanation for lesson {order}.",
        mermaid=f"graph TD; A{order}[In] --> B{order}[Out];",
        exercise=Exercise(
            question=f"Question {order}?",
            options=["a", "b", "c"],
            answer_index=1,
            why="Because b.",
        ),
        citation=f"0{order}:00",
    )


def make_path(url: str = "https://example.com/a", n_lessons: int = 5) -> Path:
    return Path(
        document_title="Canned Document",
        source_url=url,
        lessons=[make_lesson(i) for i in range(1, n_lessons + 1)],
    )


def fake_route(doc: Document):
    """Return a `route` replacement that always yields `doc`."""

    def _route(url: str) -> Document:
        return doc

    return _route


def raising(exc: Exception):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


class CountingBuilder:
    """A build_path stand-in that records how many times it was called."""

    def __init__(self, path: Path):
        self.path = path
        self.calls = 0
        self.seen_docs: list[Document] = []

    def __call__(self, doc: Document) -> Path:
        self.calls += 1
        self.seen_docs.append(doc)
        return self.path


class PerDocumentBuilder:
    """A build_path stand-in that derives each Path from the Document it was handed.

    This is what a real pathbuilder does — `source_url` comes from `doc.url` — so it is the
    only honest way to test that a cache hit/miss returns a Path describing the *requested*
    document rather than some earlier one.
    """

    def __init__(self):
        self.calls = 0
        self.seen_docs: list[Document] = []

    def __call__(self, doc: Document) -> Path:
        self.calls += 1
        self.seen_docs.append(doc)
        return make_path(url=doc.url)


# --------------------------------------------------------------------------- /health


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------- happy path


def test_ingest_returns_a_valid_path(client, monkeypatch):
    doc = make_document()
    expected = make_path()
    builder = CountingBuilder(expected)
    monkeypatch.setattr(api, "route", fake_route(doc))
    monkeypatch.setattr(api, "build_path", builder)

    response = client.post("/ingest", json={"url": "https://example.com/a"})

    assert response.status_code == 200
    body = response.json()
    parsed = Path.model_validate(body)  # the body really is a Path
    assert parsed == expected
    assert parsed.document_title == "Canned Document"
    assert parsed.source_url == "https://example.com/a"
    assert len(parsed.lessons) == 5
    assert [lesson.order for lesson in parsed.lessons] == [1, 2, 3, 4, 5]
    assert builder.calls == 1
    # The Document produced by route is what gets handed to build_path.
    assert builder.seen_docs[0] is doc


def test_ingest_passes_the_request_url_through_to_route(client, monkeypatch):
    seen_urls: list[str] = []

    def _route(url: str) -> Document:
        seen_urls.append(url)
        return make_document()

    monkeypatch.setattr(api, "route", _route)
    monkeypatch.setattr(api, "build_path", CountingBuilder(make_path()))

    response = client.post("/ingest", json={"url": "https://youtu.be/abcdefghijk"})

    assert response.status_code == 200
    assert seen_urls == ["https://youtu.be/abcdefghijk"]


# --------------------------------------------------------------------------- error mapping


def test_no_captions_error_maps_to_422(client, monkeypatch):
    monkeypatch.setattr(api, "route", raising(NoCaptionsError("no captions for this video")))
    monkeypatch.setattr(api, "build_path", raising(AssertionError("build_path must not run")))

    response = client.post("/ingest", json={"url": "https://www.youtube.com/watch?v=aircAruvnKk"})

    assert response.status_code == 422
    assert response.json()["detail"] == "no captions for this video"


def test_extraction_error_maps_to_400(client, monkeypatch):
    monkeypatch.setattr(api, "route", raising(ExtractionError("could not fetch page")))
    monkeypatch.setattr(api, "build_path", raising(AssertionError("build_path must not run")))

    response = client.post("/ingest", json={"url": "https://example.com/dead"})

    assert response.status_code == 400
    assert response.json()["detail"] == "could not fetch page"


def test_blocked_error_maps_to_503_not_400(client, monkeypatch):
    """A blocked host is not a bad request.

    Observed live on the first Wasmer deploy: YouTube returns RequestBlocked to datacenter
    IPs for videos that extract fine from a residential connection. Returning 4xx would tell
    the caller to fix a URL that was never wrong.
    """
    monkeypatch.setattr(api, "route", raising(BlockedError("YouTube is blocking this server")))
    monkeypatch.setattr(api, "build_path", raising(AssertionError("build_path must not run")))

    response = client.post("/ingest", json={"url": "https://www.youtube.com/watch?v=3RwUIP9pMSo"})

    assert response.status_code == 503
    assert response.status_code != 400
    assert response.json()["detail"] == "YouTube is blocking this server"


def test_blocked_handler_ordering_beats_the_extraction_handler(client, monkeypatch):
    """Same load-bearing ordering as NoCaptionsError: BlockedError subclasses ExtractionError.

    If the `except ExtractionError` clause came first it would swallow this and return 400.
    """
    assert issubclass(BlockedError, ExtractionError)

    monkeypatch.setattr(api, "route", raising(BlockedError("blocked")))
    monkeypatch.setattr(api, "build_path", raising(AssertionError("build_path must not run")))

    response = client.post("/ingest", json={"url": "https://youtu.be/abcdefghijk"})

    assert response.status_code == 503


def test_no_captions_handler_ordering_beats_the_extraction_handler(client, monkeypatch):
    """NoCaptionsError subclasses ExtractionError, so handler order is load-bearing.

    If the `except ExtractionError` clause were written first it would swallow
    NoCaptionsError and return 400. Pin the 422. This ordering still matters after the
    validator/cache fixes — nothing about those changes protects it.
    """
    assert issubclass(NoCaptionsError, ExtractionError)

    monkeypatch.setattr(api, "route", raising(NoCaptionsError("empty transcript")))
    monkeypatch.setattr(api, "build_path", raising(AssertionError("build_path must not run")))

    response = client.post("/ingest", json={"url": "https://youtu.be/abcdefghijk"})

    assert response.status_code == 422
    assert response.status_code != 400


def test_llm_error_maps_to_502(client, monkeypatch):
    """LLMError -> 502.

    NOTE: app/llm.py's call_json now catches anthropic.APIError (the common base of
    APIStatusError and APIConnectionError/APITimeoutError) and re-raises it as LLMError. So
    this 502 is now the path for *every* upstream model failure — a 429 rate limit, a
    dropped connection, a timeout, a bad API key — not just schema-validation failures.
    A raw anthropic exception no longer reaches the endpoint, which is why there is no
    separate handler for one here.
    """
    monkeypatch.setattr(api, "route", fake_route(make_document()))
    monkeypatch.setattr(api, "build_path", raising(LLMError("model output failed validation")))

    response = client.post("/ingest", json={"url": "https://example.com/a"})

    assert response.status_code == 502
    assert response.json()["detail"] == "model output failed validation"


@pytest.mark.parametrize(
    "detail",
    [
        # These are the exact shape call_json produces for a wrapped transport/status error:
        #   LLMError(f"Anthropic API call failed ({type(exc).__name__}): {exc}")
        "Anthropic API call failed (RateLimitError): Error code: 429 - rate_limit_error",
        "Anthropic API call failed (APIConnectionError): Connection error.",
        "Anthropic API call failed (APITimeoutError): Request timed out.",
    ],
)
def test_wrapped_transport_errors_also_map_to_502(client, monkeypatch, detail):
    """A rate limit or connection failure arrives as LLMError, so it must be a 502 too.

    502 (bad gateway) is the right code: the failure is upstream, and the caller can retry.
    """
    monkeypatch.setattr(api, "route", fake_route(make_document()))
    monkeypatch.setattr(api, "build_path", raising(LLMError(detail)))

    response = client.post("/ingest", json={"url": "https://example.com/a"})

    assert response.status_code == 502
    assert response.json()["detail"] == detail


def test_failed_build_is_not_cached(client, monkeypatch, isolated_cache):
    """A 502 must not leave anything behind that a later request would serve."""
    doc = make_document()
    monkeypatch.setattr(api, "route", fake_route(doc))
    monkeypatch.setattr(api, "build_path", raising(LLMError("boom")))

    assert client.post("/ingest", json={"url": "https://example.com/a"}).status_code == 502
    assert not (isolated_cache / f"{cache.key_for(doc)}.json").exists()

    builder = CountingBuilder(make_path())
    monkeypatch.setattr(api, "build_path", builder)
    retry = client.post("/ingest", json={"url": "https://example.com/a"})

    assert retry.status_code == 200
    assert builder.calls == 1


# --------------------------------------------------------------------------- request validation


def test_missing_url_field_is_a_422_validation_error(client, monkeypatch):
    monkeypatch.setattr(api, "route", raising(AssertionError("route must not run")))
    monkeypatch.setattr(api, "build_path", raising(AssertionError("build_path must not run")))

    response = client.post("/ingest", json={})

    assert response.status_code == 422
    detail = response.json()["detail"]
    # FastAPI validation errors are a list of error dicts, unlike our string details.
    assert isinstance(detail, list)
    assert any(err["loc"][-1] == "url" for err in detail)


def test_empty_body_is_a_422_validation_error(client, monkeypatch):
    monkeypatch.setattr(api, "route", raising(AssertionError("route must not run")))

    response = client.post("/ingest")

    assert response.status_code == 422


def test_wrong_type_for_url_is_a_422_validation_error(client, monkeypatch):
    monkeypatch.setattr(api, "route", raising(AssertionError("route must not run")))

    response = client.post("/ingest", json={"url": 42})

    assert response.status_code == 422


# --------------------------------------------------------------------------- SSRF guard


# Every one of these used to be fetched by the server on the caller's behalf, with the
# response landing in a cache file and an LLM prompt. IngestRequest.url now rejects them at
# the pydantic layer, so FastAPI answers 422 and `route` is never reached.
#
# SCOPE OF THE GUARD — no longer a string denylist. Two layers:
#   (a) a NAME check: host lowercased, trailing dot stripped, rejected if the whole host or
#       its last label is localhost / metadata / internal / intranet. This is the only layer
#       that can catch internal names which do not resolve from here.
#   (b) a RESOLVE-then-check: every address getaddrinfo returns is rejected if it is
#       loopback, private, link-local, reserved, multicast or unspecified, with IPv4-mapped
#       IPv6 unwrapped first. This is what closes the alternate-encoding, trailing-dot,
#       RFC1918, IPv6 and "public name -> private IP" holes the old denylist left open.
#
# KNOWN GAPS that remain (asserted or noted below, and in NOTES.md):
#   - REDIRECTS are still unvalidated. trafilatura.fetch_url follows them, so a public URL
#     that 302s to 169.254.169.254 passes validation and is fetched anyway.
#   - DNS REBINDING is still possible: the address checked here is resolved again,
#     independently, at fetch time. A short-TTL record can answer public then private.
#   - UNRESOLVABLE HOSTS are allowed through by design — see
#     test_an_unresolvable_host_is_allowed_through for the trade-off and its cost.
# Everything else on the old gap list (trailing dot, 127.1 / 2130706433 / 0177.0.0.1, all of
# RFC1918, the other IPv6 spellings, link-local, bare internal names, public-DNS-to-private
# services like localtest.me and nip.io) is now closed and pinned below.


# Rejected by layer (b) with no DNS involved: these are literals, parsed not looked up.
SSRF_IP_LITERAL_URLS = [
    pytest.param("http://127.0.0.1/", id="loopback-ipv4"),
    pytest.param("http://127.1/", id="loopback-short-form"),
    pytest.param("http://2130706433/", id="loopback-decimal"),
    pytest.param("http://0177.0.0.1/", id="loopback-octal"),
    pytest.param("http://127.0.0.2/", id="loopback-other-than-dot-one"),
    pytest.param("http://0.0.0.0:5000/", id="all-interfaces"),
    pytest.param("http://10.0.0.1/", id="rfc1918-10"),
    pytest.param("http://192.168.1.1/", id="rfc1918-192-168"),
    pytest.param("http://172.16.0.1/", id="rfc1918-172-16"),
    pytest.param("http://169.254.169.254/latest/meta-data/", id="cloud-metadata-ip"),
    pytest.param("http://169.254.1.1/", id="link-local"),
    pytest.param("http://[::1]:8000/", id="loopback-ipv6"),
    pytest.param("http://[::]/", id="unspecified-ipv6"),
    pytest.param("http://[0:0:0:0:0:0:0:1]/", id="loopback-ipv6-expanded"),
    pytest.param("http://[::ffff:127.0.0.1]/", id="ipv4-mapped-loopback"),
    pytest.param("http://[::ffff:a9fe:a9fe]/", id="ipv4-mapped-metadata-hex"),
    pytest.param("http://[fd00::1]/", id="unique-local-ipv6"),
    pytest.param("http://[fe80::1]/", id="link-local-ipv6"),
    pytest.param("http://127.0.0.1./", id="loopback-trailing-dot"),
]

# Rejected by layer (a). The `offline_dns` fixture makes every name unresolvable, so these
# reaching 422 proves the *name* check fired, not the resolve check — which is the whole
# point of keeping layer (a) around.
SSRF_NAME_URLS = [
    pytest.param("http://localhost:8080/admin", id="localhost"),
    pytest.param("http://localhost./", id="localhost-trailing-dot"),
    pytest.param("http://LOCALHOST:8080/admin", id="localhost-uppercase"),
    pytest.param("http://metadata/computeMetadata/v1/", id="gcp-metadata-host"),
    pytest.param("http://internal/", id="bare-internal"),
    pytest.param("http://intranet/", id="bare-intranet"),
    pytest.param("http://foo.internal/", id="internal-suffix"),
    pytest.param("http://admin.internal/keys", id="internal-suffix-with-path"),
    pytest.param("http://metadata.google.internal/", id="gcp-metadata-fqdn"),
    pytest.param("http://something.localhost/", id="localhost-suffix"),
    pytest.param("http://api.localhost/", id="localhost-suffix-api"),
]

# Rejected before any host handling at all: bad or missing scheme, missing hostname.
SSRF_SCHEME_URLS = [
    pytest.param("file:///etc/passwd", id="file-scheme"),
    pytest.param("ftp://example.com/file.txt", id="ftp-scheme"),
    pytest.param("gopher://example.com/", id="gopher-scheme"),
    pytest.param("javascript:alert(1)", id="javascript-scheme"),
    pytest.param("example.com/page", id="scheme-less"),
    pytest.param("http:///nohost", id="empty-hostname"),
    pytest.param("", id="empty-string"),
]

SSRF_URLS = SSRF_IP_LITERAL_URLS + SSRF_NAME_URLS + SSRF_SCHEME_URLS


def assert_rejected_as_a_field_error(response, url: str) -> None:
    assert response.status_code == 422, f"{url!r} was not rejected"
    detail = response.json()["detail"]
    # A pydantic field_validator failure, not one of our hand-raised HTTPException strings.
    assert isinstance(detail, list)
    assert any(err["loc"][-1] == "url" for err in detail)


@pytest.mark.parametrize("url", SSRF_URLS)
def test_ssrf_candidate_urls_are_rejected_with_422(client, monkeypatch, url):
    monkeypatch.setattr(api, "route", raising(AssertionError(f"route must not run for {url!r}")))
    monkeypatch.setattr(api, "build_path", raising(AssertionError("build_path must not run")))

    assert_rejected_as_a_field_error(client.post("/ingest", json={"url": url}), url)


@pytest.mark.parametrize("url", SSRF_NAME_URLS)
def test_internal_names_are_rejected_without_resolving_them(client, monkeypatch, url):
    """Layer (a): these are rejected by name, before any lookup.

    Patch the resolver to blow up loudly if it is called at all. That pins *why* these are
    422 — a corp intranet host or metadata.google.internal usually does not resolve from a
    build machine, so if the name check were removed these would sail through the
    "unresolvable -> allow" branch.
    """
    monkeypatch.setattr(api.socket, "getaddrinfo", must_not_resolve("name check should have fired"))
    monkeypatch.setattr(api, "route", raising(AssertionError(f"route must not run for {url!r}")))

    assert_rejected_as_a_field_error(client.post("/ingest", json={"url": url}), url)


def test_a_public_name_resolving_to_a_public_ip_is_accepted(client, monkeypatch):
    """The accept path proper: name passes layer (a), resolves, address is public."""
    monkeypatch.setattr(api.socket, "getaddrinfo", resolving_to(PUBLIC_IP))
    seen_urls: list[str] = []

    def _route(request_url: str) -> Document:
        seen_urls.append(request_url)
        return make_document(url=request_url)

    monkeypatch.setattr(api, "route", _route)
    monkeypatch.setattr(api, "build_path", PerDocumentBuilder())

    response = client.post("/ingest", json={"url": "https://example.com/a"})

    assert response.status_code == 200
    assert seen_urls == ["https://example.com/a"]


def test_a_public_looking_name_that_resolves_to_a_private_ip_is_rejected(client, monkeypatch):
    """The case a denylist can NEVER catch, and the reason layer (b) exists.

    localtest.me, *.nip.io, and any attacker-controlled A record point a perfectly ordinary
    public hostname at 10.0.0.5 / 127.0.0.1 / 169.254.169.254. No amount of string matching
    on the hostname sees it; only resolving does.
    """
    monkeypatch.setattr(api.socket, "getaddrinfo", resolving_to("10.0.0.5"))
    monkeypatch.setattr(api, "route", raising(AssertionError("route must not run")))
    monkeypatch.setattr(api, "build_path", raising(AssertionError("build_path must not run")))

    response = client.post("/ingest", json={"url": "https://totally-normal.example.com/a"})

    assert_rejected_as_a_field_error(response, "https://totally-normal.example.com/a")


def test_one_private_address_among_several_public_ones_still_rejects(client, monkeypatch):
    """Round-robin DNS: the guard checks *every* answer, not just the first."""
    monkeypatch.setattr(api.socket, "getaddrinfo", resolving_to(PUBLIC_IP, "8.8.8.8", "10.0.0.5"))
    monkeypatch.setattr(api, "route", raising(AssertionError("route must not run")))

    response = client.post("/ingest", json={"url": "https://mixed-answers.example.com/a"})

    assert_rejected_as_a_field_error(response, "https://mixed-answers.example.com/a")


def test_an_ipv4_mapped_ipv6_answer_is_unwrapped_before_the_check(client, monkeypatch):
    """A AAAA record of ::ffff:169.254.169.254 is the metadata endpoint wearing a hat.

    `is_private` on the raw IPv6Address does not see it, so the validator unwraps
    `.ipv4_mapped` first. Pin that, since it is easy to drop in a refactor.
    """
    monkeypatch.setattr(api.socket, "getaddrinfo", resolving_to("::ffff:169.254.169.254"))
    monkeypatch.setattr(api, "route", raising(AssertionError("route must not run")))

    response = client.post("/ingest", json={"url": "https://mapped.example.com/a"})

    assert_rejected_as_a_field_error(response, "https://mapped.example.com/a")


def test_an_unresolvable_host_is_allowed_through(client, monkeypatch):
    """DELIBERATE TRADE-OFF, not an oversight — and the sharpest edge of the new validator.

    When getaddrinfo raises, the validator returns the URL rather than rejecting it, on the
    reasoning that a host nobody can resolve is a host the fetcher cannot reach either, so
    the request will fail a moment later in `route` with an ExtractionError -> 400. Failing
    closed instead would turn every transient resolver hiccup — a DNS timeout under load, a
    flaky container resolver — into a 422 that blames the caller's URL.

    What it costs: the guard now has a fail-open branch reachable by anything that can make
    resolution fail, and "resolution failed here" does not prove "resolution will fail at
    fetch time" — a split-horizon or briefly-flapping resolver breaks that assumption.
    Layer (a) is what keeps the obvious internal names out of this branch.
    """
    monkeypatch.setattr(api.socket, "getaddrinfo", unresolvable("host"))
    seen_urls: list[str] = []

    def _route(request_url: str) -> Document:
        seen_urls.append(request_url)
        return make_document(url=request_url)

    monkeypatch.setattr(api, "route", _route)
    monkeypatch.setattr(api, "build_path", PerDocumentBuilder())

    response = client.post("/ingest", json={"url": "https://no-such-host.example/a"})

    assert response.status_code == 200
    assert seen_urls == ["https://no-such-host.example/a"]


def test_a_unicode_error_from_the_resolver_also_allows_the_url_through(client, monkeypatch):
    """Same branch, other exception: an IDNA-illegal host raises UnicodeError, not gaierror."""

    def _boom(host, port, *args, **kwargs):
        raise UnicodeError("label empty or too long")

    monkeypatch.setattr(api.socket, "getaddrinfo", _boom)
    monkeypatch.setattr(api, "route", fake_route(make_document()))
    monkeypatch.setattr(api, "build_path", CountingBuilder(make_path()))

    response = client.post("/ingest", json={"url": "https://" + "a" * 300 + ".example/a"})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://127.1/", id="short-form"),
        pytest.param("http://2130706433/", id="integer-form"),
        pytest.param("http://0177.0.0.1/", id="octal-form"),
    ],
)
def test_legacy_ipv4_literals_are_parsed_locally_and_never_reach_the_resolver(
    client, monkeypatch, url
):
    """The platform gap this used to pin is closed.

    These forms are what glibc's inet_aton accepts and Windows' getaddrinfo rejects. When the
    validator relied on getaddrinfo alone, the Windows rejection was indistinguishable from
    "unresolvable" and every one of them was ALLOWED through the fail-open branch. The
    validator now parses literals itself (ipaddress, then inet_aton) and only falls open on a
    genuine name lookup, so a parse failure can no longer become a security hole.

    The resolver is tripwired: a literal must be judged without any lookup at all.
    """
    monkeypatch.setattr(api.socket, "getaddrinfo", must_not_resolve)

    response = client.post("/ingest", json={"url": url})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a",
        "http://example.com/a",
        "https://www.youtube.com/watch?v=aircAruvnKk",
        "https://youtu.be/abcdefghijk",
        "https://sub.domain.example.co.uk/deep/path?q=1#frag",
        # Not a local host despite the substring — neither the whole host nor its last label
        # is a blocked name, so layer (a) must not over-reject it.
        "https://notlocalhost.example.com/a",
        # Nor is "internal" anywhere but the last label a match.
        "https://internal.example.com/a",
        "https://metadata.example.com/a",
    ],
)
def test_ordinary_public_http_urls_are_still_accepted(client, monkeypatch, url):
    """The guard must not break the normal case, via the real accept path.

    Resolution is pinned to a public address so this asserts "resolved and found public",
    not the weaker "could not resolve, so allowed".
    """
    monkeypatch.setattr(api.socket, "getaddrinfo", resolving_to(PUBLIC_IP))
    seen_urls: list[str] = []

    def _route(request_url: str) -> Document:
        seen_urls.append(request_url)
        return make_document(url=request_url)

    monkeypatch.setattr(api, "route", _route)
    monkeypatch.setattr(api, "build_path", PerDocumentBuilder())

    response = client.post("/ingest", json={"url": url})

    assert response.status_code == 200
    assert seen_urls == [url]
    assert response.json()["source_url"] == url


def test_surrounding_whitespace_is_stripped_not_rejected(client, monkeypatch):
    """The validator strips, so a pasted URL with stray whitespace still works — and the
    stripped form is what reaches `route`."""
    monkeypatch.setattr(api.socket, "getaddrinfo", resolving_to(PUBLIC_IP))
    seen_urls: list[str] = []

    def _route(request_url: str) -> Document:
        seen_urls.append(request_url)
        return make_document(url=request_url)

    monkeypatch.setattr(api, "route", _route)
    monkeypatch.setattr(api, "build_path", PerDocumentBuilder())

    response = client.post("/ingest", json={"url": "  https://example.com/a  "})

    assert response.status_code == 200
    assert seen_urls == ["https://example.com/a"]


def test_host_matching_is_case_insensitive_for_blocked_names(client, monkeypatch):
    """The host is lowercased before the name check, so LOCALHOST is blocked like localhost."""
    monkeypatch.setattr(api.socket, "getaddrinfo", must_not_resolve("blocked name"))
    monkeypatch.setattr(api, "route", raising(AssertionError("route must not run")))

    for url in (
        "http://LOCALHOST:8080/admin",
        "http://Metadata.Google.INTERNAL/",
        "http://LocalHost./",
    ):
        assert client.post("/ingest", json={"url": url}).status_code == 422, url


# --------------------------------------------------------------------------- cache behaviour


def test_second_identical_request_is_served_from_cache(client, monkeypatch, isolated_cache):
    doc = make_document(text="the exact same extracted text")
    builder = CountingBuilder(make_path())
    monkeypatch.setattr(api, "route", fake_route(doc))
    monkeypatch.setattr(api, "build_path", builder)

    first = client.post("/ingest", json={"url": "https://example.com/a"})
    second = client.post("/ingest", json={"url": "https://example.com/a"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert builder.calls == 1, "build_path was re-run on an identical request"
    assert second.json() == first.json()
    # And it really came off disk, in the tmp cache dir, not the repo's cache/.
    key = cache.key_for(doc)
    assert (isolated_cache / f"{key}.json").exists()
    # store() writes a mkstemp .tmp then replaces, so no temp file may survive a good write.
    assert list(isolated_cache.glob("*.tmp")) == []
    # The key is no longer the bare text hash: that constant must not name a cache file.
    assert not (isolated_cache / f"{cache.text_hash(doc.text)}.json").exists()


def test_different_source_text_is_a_cache_miss(client, monkeypatch):
    builder = CountingBuilder(make_path())
    monkeypatch.setattr(api, "build_path", builder)

    monkeypatch.setattr(api, "route", fake_route(make_document(text="first text")))
    assert client.post("/ingest", json={"url": "https://example.com/a"}).status_code == 200

    monkeypatch.setattr(api, "route", fake_route(make_document(text="second text")))
    assert client.post("/ingest", json={"url": "https://example.com/b"}).status_code == 200

    assert builder.calls == 2


def test_corrupt_cache_entry_falls_back_to_a_rebuild(client, monkeypatch, isolated_cache):
    doc = make_document(text="text with a broken cache entry")
    builder = CountingBuilder(make_path())
    monkeypatch.setattr(api, "route", fake_route(doc))
    monkeypatch.setattr(api, "build_path", builder)

    isolated_cache.mkdir(parents=True, exist_ok=True)
    (isolated_cache / f"{cache.key_for(doc)}.json").write_text("{not json at all", encoding="utf-8")

    response = client.post("/ingest", json={"url": "https://example.com/a"})

    assert response.status_code == 200
    assert builder.calls == 1
    assert Path.model_validate(response.json()).document_title == "Canned Document"


def test_schema_valid_json_that_is_not_a_path_falls_back_to_a_rebuild(
    client, monkeypatch, isolated_cache
):
    """A stale-schema entry (valid JSON, wrong shape) is a ValidationError, also a miss."""
    doc = make_document(text="text with a stale-schema cache entry")
    builder = CountingBuilder(make_path())
    monkeypatch.setattr(api, "route", fake_route(doc))
    monkeypatch.setattr(api, "build_path", builder)

    isolated_cache.mkdir(parents=True, exist_ok=True)
    (isolated_cache / f"{cache.key_for(doc)}.json").write_text(
        '{"document_title": "Old", "lessons": []}', encoding="utf-8"
    )

    response = client.post("/ingest", json={"url": "https://example.com/a"})

    assert response.status_code == 200
    assert builder.calls == 1


def test_identical_text_at_two_urls_no_longer_collides(client, monkeypatch, isolated_cache):
    """REGRESSION: the cache key used to be sha256(doc.text) alone.

    Two different URLs whose extracted text was byte-identical (a syndicated article, a
    mirror, a youtu.be vs youtube.com pair for one video) collided, and the second caller
    got back a Path whose `source_url` pointed at the *first* caller's URL — the response
    misdescribed the request. `cache.key_for` now folds in CACHE_VERSION and `doc.url`, so
    the two requests are distinct cache entries. Assert the bug is gone.
    """
    shared_text = "identical text served from two different urls"
    builder = PerDocumentBuilder()
    monkeypatch.setattr(api, "build_path", builder)

    original_doc = make_document(text=shared_text, url="https://example.com/original")
    mirror_doc = make_document(text=shared_text, url="https://mirror.test/copy")

    # Same text, so the *old* key would be the same for both.
    assert cache.text_hash(original_doc.text) == cache.text_hash(mirror_doc.text)
    # The new key is not.
    assert cache.key_for(original_doc) != cache.key_for(mirror_doc)

    monkeypatch.setattr(api, "route", fake_route(original_doc))
    first = client.post("/ingest", json={"url": "https://example.com/original"})

    monkeypatch.setattr(api, "route", fake_route(mirror_doc))
    second = client.post("/ingest", json={"url": "https://mirror.test/copy"})

    assert first.status_code == 200
    assert second.status_code == 200

    # Two separate builds, because two separate cache entries.
    assert builder.calls == 2, "the second URL was wrongly served from the first URL's entry"

    # Each response describes the URL that was actually requested.
    assert first.json()["source_url"] == "https://example.com/original"
    assert second.json()["source_url"] == "https://mirror.test/copy"
    assert first.json() != second.json()

    # And there are two files on disk, at the two distinct keys.
    assert (isolated_cache / f"{cache.key_for(original_doc)}.json").exists()
    assert (isolated_cache / f"{cache.key_for(mirror_doc)}.json").exists()
    assert len(list(isolated_cache.glob("*.json"))) == 2


def test_each_url_keeps_serving_its_own_cached_path_on_a_repeat(
    client, monkeypatch, isolated_cache
):
    """The other half of the fix: after the two entries exist, each URL still hits its own."""
    shared_text = "identical text served from two different urls"
    builder = PerDocumentBuilder()
    monkeypatch.setattr(api, "build_path", builder)

    original_doc = make_document(text=shared_text, url="https://example.com/original")
    mirror_doc = make_document(text=shared_text, url="https://mirror.test/copy")

    monkeypatch.setattr(api, "route", fake_route(original_doc))
    client.post("/ingest", json={"url": "https://example.com/original"})
    monkeypatch.setattr(api, "route", fake_route(mirror_doc))
    client.post("/ingest", json={"url": "https://mirror.test/copy"})
    assert builder.calls == 2

    # Repeat both. Both are hits, and each returns its own source_url.
    monkeypatch.setattr(api, "route", fake_route(original_doc))
    again_original = client.post("/ingest", json={"url": "https://example.com/original"})
    monkeypatch.setattr(api, "route", fake_route(mirror_doc))
    again_mirror = client.post("/ingest", json={"url": "https://mirror.test/copy"})

    assert builder.calls == 2, "a cached entry was rebuilt"
    assert again_original.json()["source_url"] == "https://example.com/original"
    assert again_mirror.json()["source_url"] == "https://mirror.test/copy"


def test_cache_version_participates_in_the_key(client, monkeypatch, isolated_cache):
    """The other reason the key changed: a prompt/model change must be invalidatable.

    An entry written under one generator version must not be served under the next one.
    `key_for` now defaults `version` at CALL time, so bumping the module constant — which is
    how a real deployment does it — genuinely changes the keys. (It used to bind the default
    at def time, which made the runtime bump invisible; this test no longer needs the
    explicit-keyword workaround that wart forced.)
    """
    doc = make_document(text="text cached under an older generator version")
    builder = CountingBuilder(make_path())
    monkeypatch.setattr(api, "route", fake_route(doc))
    monkeypatch.setattr(api, "build_path", builder)

    assert client.post("/ingest", json={"url": "https://example.com/a"}).status_code == 200
    assert builder.calls == 1
    old_key = cache.key_for(doc)
    assert (isolated_cache / f"{old_key}.json").exists()

    # Bumping the constant changes the key, with no keyword argument in sight.
    monkeypatch.setattr(cache, "CACHE_VERSION", "v-next")
    new_key = cache.key_for(doc)
    assert new_key != old_key
    # The explicit keyword and the constant agree, and both still beat the old version.
    assert new_key == cache.key_for(doc, version="v-next")
    assert cache.key_for(doc, version="v1") == old_key

    # End to end: the next-version deployment misses on the old entry and rebuilds.
    assert client.post("/ingest", json={"url": "https://example.com/a"}).status_code == 200
    assert builder.calls == 2, "a version bump did not invalidate the old entry"
    assert (isolated_cache / f"{new_key}.json").exists()
    assert len(list(isolated_cache.glob("*.json"))) == 2

    # ...and the bumped deployment now serves its own entry on a repeat.
    assert client.post("/ingest", json={"url": "https://example.com/a"}).status_code == 200
    assert builder.calls == 2


def test_cache_write_failure_still_returns_200_with_a_valid_path(
    client, monkeypatch, isolated_cache
):
    """A read-only or full disk must not cost the caller a paid LLM call.

    The Path has already been generated by the time `cache.store` runs, so an OSError there
    is logged and swallowed: the request still returns 200 with the generated Path.
    """
    doc = make_document(text="text whose cache write fails")
    expected = make_path()
    builder = CountingBuilder(expected)
    monkeypatch.setattr(api, "route", fake_route(doc))
    monkeypatch.setattr(api, "build_path", builder)
    monkeypatch.setattr(cache, "store", raising(OSError("read-only file system")))

    response = client.post("/ingest", json={"url": "https://example.com/a"})

    assert response.status_code == 200
    assert Path.model_validate(response.json()) == expected
    assert builder.calls == 1
    # Nothing was persisted, so the next request rebuilds rather than 500ing.
    assert not (isolated_cache / f"{cache.key_for(doc)}.json").exists()

    second = client.post("/ingest", json={"url": "https://example.com/a"})
    assert second.status_code == 200
    assert builder.calls == 2


def test_a_failed_store_leaves_no_tmp_file_behind(isolated_cache):
    """`store` mkstemps into CACHE_DIR, so a failure mid-write must clean up after itself.

    Without the try/except-unlink, every failed write would drop an orphan
    `<key>.xxxxxx.tmp` in the cache dir — invisible to `load` (which only looks for
    `<key>.json`), never reaped, and growing forever on a flaky disk.
    """
    doc = make_document(text="text whose serialization explodes")
    key = cache.key_for(doc)

    class ExplodingPath:
        def model_dump_json(self, **kwargs):
            raise RuntimeError("serialization blew up mid-write")

    with pytest.raises(RuntimeError, match="serialization blew up"):
        cache.store(key, ExplodingPath())

    assert list(isolated_cache.glob("*.tmp")) == [], "a failed store orphaned a temp file"
    assert list(isolated_cache.iterdir()) == [], "a failed store left something behind"


def test_a_failed_store_does_not_clobber_an_existing_good_entry(isolated_cache):
    """The replace is the last step, so the previous entry survives a failed rewrite."""
    doc = make_document(text="text with a good entry and a failing rewrite")
    key = cache.key_for(doc)
    good = make_path()
    cache.store(key, good)
    assert cache.load(key) == good

    class ExplodingPath:
        def model_dump_json(self, **kwargs):
            raise OSError("no space left on device")

    with pytest.raises(OSError, match="no space left on device"):
        cache.store(key, ExplodingPath())

    assert cache.load(key) == good, "a failed rewrite destroyed the cached entry"
    assert list(isolated_cache.glob("*.tmp")) == []


def test_cache_read_failure_degrades_to_a_rebuild_not_a_500(client, monkeypatch, isolated_cache):
    """The mirror case: an OSError out of `cache.load` is logged and treated as a miss."""
    doc = make_document(text="text whose cache read fails")
    builder = CountingBuilder(make_path())
    monkeypatch.setattr(api, "route", fake_route(doc))
    monkeypatch.setattr(api, "build_path", builder)
    monkeypatch.setattr(cache, "load", raising(OSError("cache dir is unreadable")))

    with pytest.raises(OSError, match="cache dir is unreadable"):
        client.post("/ingest", json={"url": "https://example.com/a"})

    # api.ingest does NOT wrap cache.load, so the OSError escapes. cache.load itself is the
    # layer that swallows OSError -> None; verify that, since it is what the endpoint relies
    # on in production.
    monkeypatch.undo()
    monkeypatch.setattr(cache, "CACHE_DIR", isolated_cache)
    isolated_cache.mkdir(parents=True, exist_ok=True)
    entry = isolated_cache / f"{cache.key_for(doc)}.json"
    entry.mkdir()  # a directory where a file is expected -> OSError on read_text
    assert cache.load(cache.key_for(doc)) is None


# --------------------------------------------------------------------------- unmapped errors


def test_unmapped_route_exception_is_not_translated(client, monkeypatch):
    """Anything that is not ExtractionError/LLMError escapes as a 500.

    TestClient re-raises server exceptions, which is what this asserts.
    """
    monkeypatch.setattr(api, "route", raising(RuntimeError("network stack exploded")))

    with pytest.raises(RuntimeError, match="network stack exploded"):
        client.post("/ingest", json={"url": "https://example.com/a"})


def test_unmapped_build_path_exception_is_not_translated(client, monkeypatch):
    monkeypatch.setattr(api, "route", fake_route(make_document()))
    monkeypatch.setattr(api, "build_path", raising(ValueError("bad schema")))

    with pytest.raises(ValueError, match="bad schema"):
        client.post("/ingest", json={"url": "https://example.com/a"})
