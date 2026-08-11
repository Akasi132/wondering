"""FastAPI app: the website at /, and POST /ingest -> a Path of 5 lessons.

v1 runs the whole pipeline synchronously, so a cold request takes as long as the LLM call.
A job queue is the v2 upgrade (noted in NOTES.md). The front end copes by showing an elapsed
clock rather than pretending to know how far along it is.
"""

import ipaddress
import json
import logging
import socket
from pathlib import Path as FsPath
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError, field_validator

from app import cache
from app.extract import ExtractionError, NoCaptionsError
from app.llm import LLMError
from app.models import Path
from app.pathbuilder import build_path
from app.router import route

logger = logging.getLogger(__name__)

STATIC_DIR = FsPath(__file__).resolve().parent / "static"
FIXTURES_DIR = FsPath(__file__).resolve().parent.parent / "fixtures"

# Pre-built paths the site can show without an API key, so a fresh deploy is not a dead page.
# An explicit map rather than a filename parameter: the value is never taken from the caller.
EXAMPLES = {
    "chemistry": "path_chemistry.json",
    "article": "path_article.json",
}

app = FastAPI(title="Wondering Lesson Ingester", version="0.1.0")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _as_ip_literal(host: str) -> ipaddress._BaseAddress | None:
    """Parse `host` as an IP literal, or return None if it is a name.

    Handles the legacy IPv4 forms (`127.1`, `2130706433`, `0177.0.0.1`) that
    ipaddress.ip_address rejects but inet_aton — and real HTTP clients — accept.
    """
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    try:
        return ipaddress.ip_address(socket.inet_ntoa(socket.inet_aton(host)))
    except (OSError, ValueError):
        return None


def _reject_non_public(ip: ipaddress._BaseAddress) -> None:
    candidate = ip.ipv4_mapped or ip if isinstance(ip, ipaddress.IPv6Address) else ip
    if (
        candidate.is_loopback
        or candidate.is_private
        or candidate.is_link_local
        or candidate.is_reserved
        or candidate.is_multicast
        or candidate.is_unspecified
    ):
        raise ValueError(
            f"url points at a non-public address ({candidate}); refusing to fetch it"
        )


class IngestRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def must_be_public_http_url(cls, value: str) -> str:
        """Reject anything the server should not be fetching on a caller's behalf.

        Without this, POST /ingest fetches any string it is handed — cloud metadata at
        169.254.169.254, internal hostnames — and puts the result in a cache file and an LLM
        prompt. Out of PLAN.md's scope, but the surface is not worth leaving open.

        This resolves the hostname and rejects any address that is loopback, private,
        link-local, reserved, multicast, or unspecified. A hostname denylist alone is not
        enough: a trailing dot (`localhost.`), an alternate encoding (`127.1`, `2130706433`),
        all of RFC1918, and IPv4-mapped IPv6 all slip past pure string matching.

        KNOWN GAPS, deliberately not closed here (see NOTES.md):
          - Redirects are not re-validated. trafilatura follows them, so a public URL that
            302s to a private address still reaches the fetcher.
          - DNS rebinding: the address is checked here, resolved again at fetch time.
          - If resolution fails we allow the URL through, since the fetch will fail anyway.
        """
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("url must start with http:// or https://")

        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            raise ValueError("url must include a hostname")

        # Name-based check first. It is not sufficient on its own (see the resolve step
        # below), but it is the only thing that catches internal names which do not resolve
        # from here — metadata.google.internal, corp intranet hosts — where resolution
        # failure would otherwise fall through to "allow".
        blocked_names = ("localhost", "metadata", "internal", "intranet")
        labels = host.split(".")
        if host in blocked_names or labels[-1] in blocked_names:
            raise ValueError("url must not point at a local or internal hostname")

        # A literal address is parsed locally and NEVER falls into the allow-on-failure
        # branch below. Keeping these separate matters: socket.getaddrinfo rejects legacy
        # IPv4 forms ("127.1", "2130706433", "0177.0.0.1") on Windows, so treating a parse
        # failure like a lookup failure would let every one of them through.
        literal = _as_ip_literal(host)
        if literal is not None:
            _reject_non_public(literal)
            return value

        try:
            resolved = {info[4][0] for info in socket.getaddrinfo(host, None)}
        except (socket.gaierror, UnicodeError, ValueError):
            # A name nobody here can resolve is a name the fetcher cannot reach either, so
            # this fails open rather than blaming the caller for a resolver hiccup. It is the
            # weakest link in this validator — see NOTES.md.
            return value

        for address in resolved:
            try:
                _reject_non_public(ipaddress.ip_address(address))
            except ValueError as exc:
                if "refusing to fetch" in str(exc):
                    raise
        return value


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/examples/{name}", response_model=Path)
def example(name: str) -> Path:
    """Serve one of the fixture paths generated during the live runs in NOTES.md.

    These are read back through `Path` rather than streamed as raw files, so a fixture that
    has drifted from the schema fails here instead of in the browser.
    """
    filename = EXAMPLES.get(name)
    if filename is None:
        raise HTTPException(status_code=404, detail=f"No saved example named {name!r}.")
    try:
        return Path.model_validate_json((FIXTURES_DIR / filename).read_text(encoding="utf-8"))
    except OSError as exc:
        logger.error("Example %s is missing or unreadable: %s", filename, exc)
        raise HTTPException(status_code=404, detail="That example is not on this server.") from exc
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.error("Example %s does not match the Path schema: %s", filename, exc)
        raise HTTPException(status_code=500, detail="That example is no longer readable.") from exc


@app.post("/ingest", response_model=Path)
def ingest(request: IngestRequest) -> Path:
    try:
        doc = route(request.url)
    except NoCaptionsError as exc:
        # Checked before ExtractionError: NoCaptionsError subclasses it.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    key = cache.key_for(doc)
    cached = cache.load(key)
    if cached is not None:
        return cached

    try:
        path = build_path(doc)
    except LLMError as exc:
        # call_json wraps Anthropic transport/status errors in LLMError, so a 429, a dropped
        # connection, or a bad key lands here as a 502 rather than an unhandled 500.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        cache.store(key, path)
    except OSError as exc:
        # The path was generated and paid for. A failed cache write must not fail the request.
        logger.error("Could not write cache entry for %s: %s", doc.url, exc)

    return path
