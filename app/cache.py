"""Disk cache keyed by a hash of the extracted source text (plus URL and generator version).

PLAN.md Step 9 says to hash `doc.text`. The key here also folds in `doc.url` and a
CACHE_VERSION, for two reasons found during testing:
  - text alone collides across URLs (youtu.be vs youtube.com for one video, syndicated or
    mirrored articles). On a collision the cached Path carries the *first* URL's
    source_url, so the response misdescribes the request.
  - text alone survives prompt and model changes, so editing the system prompt or swapping
    models keeps serving stale generated content with no way to invalidate but rm -rf.
Both deviations are recorded in NOTES.md.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path as FsPath

from pydantic import ValidationError

from app.models import Document, Path

logger = logging.getLogger(__name__)

# Overridable because a deployed instance's package directory is not necessarily writable or
# persistent — on Wasmer Edge the durable location is a mounted volume, so the deploy sets
# WONDERING_CACHE_DIR=/data/cache. Read at import; tests patch the module attribute directly.
CACHE_DIR = FsPath(os.getenv("WONDERING_CACHE_DIR") or FsPath(__file__).resolve().parent.parent / "cache")

# Bump this whenever the prompt, the model, or the Path schema changes in a way that should
# invalidate previously generated lessons.
CACHE_VERSION = "v1"


def text_hash(text: str) -> str:
    """Plain sha256 of the source text, as PLAN.md Step 9 specifies."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _generator_id() -> str:
    """Provider + model currently configured. Imported late to keep this module standalone."""
    from app import llm

    return llm.generator_id()


def key_for(doc: Document, *, version: str | None = None, generator: str | None = None) -> str:
    """Cache key for a document: generator version + provider/model + URL + source text.

    `version` defaults to CACHE_VERSION at call time, not at def time, so patching
    cache.CACHE_VERSION actually changes the keys.

    `generator` folds in the active provider and model for the same reason CACHE_VERSION is
    here: without it, running the same URL under LLM_PROVIDER=openai serves the Claude-authored
    lessons already on disk (or vice versa), and nothing in the returned Path says which model
    wrote it. Two backends make that a routine mistake rather than a hypothetical one.
    """
    payload = "\n".join(
        (
            version or CACHE_VERSION,
            generator or _generator_id(),
            doc.url,
            text_hash(doc.text),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_file(key: str) -> FsPath:
    # Keys are hex digests, but refuse anything else so a caller passing user input can
    # never traverse out of CACHE_DIR.
    if not all(c in "0123456789abcdef" for c in key) or not key:
        raise ValueError(f"Cache key must be a hex digest, got {key!r}")
    return CACHE_DIR / f"{key}.json"


def load(key: str) -> Path | None:
    path = cache_file(key)
    if not path.exists():
        return None
    try:
        return Path.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        # Corrupt or stale-schema entry: treat as a miss and regenerate.
        logger.warning("Discarding unusable cache entry %s: %s", path.name, exc)
        return None
    except OSError as exc:
        # An unreadable cache dir is a real problem, not a cache miss. Still degrade to a
        # miss so the request succeeds, but say so loudly rather than silently.
        logger.error("Could not read cache entry %s: %s", path.name, exc)
        return None


def store(key: str, value: Path) -> FsPath:
    """Write atomically, so a crash or a concurrent writer cannot leave a truncated entry.

    Uses mkstemp rather than a name derived from id(), which is per-process and reused after
    GC — two processes writing the same key could otherwise pick the same temp filename.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_file(key)
    fd, tmp_name = tempfile.mkstemp(dir=CACHE_DIR, prefix=f"{key}.", suffix=".tmp")
    tmp = FsPath(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value.model_dump_json(indent=2))
        tmp.replace(path)  # os.replace is atomic on POSIX and Windows
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave an orphan .tmp behind
        raise
    return path
