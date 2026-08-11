"""End-to-end CLI: url -> Document -> Path -> JSON on stdout.

    python main.py https://example.com/article
    python main.py https://www.youtube.com/watch?v=... --out fixtures/path_youtube.json
    python main.py --verify-models

This module also re-exports the ASGI app as `main:app`. That is not for the CLI's benefit:
Wasmer Edge's autobuild detects a Python web app and looks for the server object itself, and
both of Wasmer's own FastAPI examples keep it in `main.py`. The obvious alternative — a root
`app.py` holding `app` — cannot work here, because a root `app.py` would be shadowed by the
existing `app/` package, so `app:app` would resolve to the package and find no attribute.
`server.py` starts the same object explicitly for hosts that want a command instead.
"""

import argparse
import logging
import sys

from app import cache
from app.api import app  # noqa: F401 — re-exported as the ASGI entrypoint, see the docstring
from app.extract import ExtractionError
from app.llm import (
    MODEL_CHEAP,
    MODEL_PATHBUILDER,
    LLMError,
    active_model,
    generator_id,
    list_models,
    provider,
    verify_model,
)
from app.pathbuilder import build_path, unverifiable_citations
from app.router import route


def show_models() -> int:
    """List what the configured endpoint actually serves.

    An OpenAI-compatible host's model slugs are its own business — this is the only way to
    learn NVIDIA's exact ID for a model rather than guessing at it.
    """
    try:
        ids = list_models()
    except Exception as exc:
        print(f"Could not list models from the {provider()} endpoint: {exc}", file=sys.stderr)
        return 1
    print(f"# {len(ids)} model(s) available via provider={provider()}", file=sys.stderr)
    for model_id in ids:
        print(model_id)
    print(f"\nSet LLM_MODEL in .env to one of these.", file=sys.stderr)
    return 0


def verify_models() -> int:
    """Directive 3: confirm model IDs against the live models list, don't trust memory."""
    ok = True
    # Anthropic has two planned models; an OpenAI-compatible endpoint only ever gets asked
    # for the one that is actually configured.
    if provider() == "openai":
        targets = (active_model(),)
    else:
        targets = (MODEL_PATHBUILDER, MODEL_CHEAP)
    for model_id in targets:
        try:
            info = verify_model(model_id)
        except LLMError as exc:
            print(f"FAIL {model_id}: {exc}", file=sys.stderr)
            ok = False
        except Exception as exc:
            print(f"FAIL {model_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            ok = False
        else:
            print(
                f"OK   {info['id']} | {info['display_name']} | "
                f"context={info['max_input_tokens']} max_output={info['max_tokens']}"
            )
    print("\nPaste the OK lines into NOTES.md under 'Model verification'.", file=sys.stderr)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Wondering lesson path from a URL.")
    parser.add_argument("url", nargs="?", help="Article or YouTube URL")
    parser.add_argument("--out", help="Also write the JSON to this file")
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the cache")
    parser.add_argument(
        "--verify-models", action="store_true", help="Check model IDs against /v1/models and exit"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List every model the endpoint serves and exit"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.list_models:
        return show_models()
    if args.verify_models:
        return verify_models()
    if not args.url:
        parser.error("a URL is required unless --verify-models or --list-models is given")

    print(f"[generator] {generator_id()}", file=sys.stderr)

    try:
        doc = route(args.url)
    except ExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"[extracted] {doc.source_type} | {doc.title!r} | {len(doc.text)} chars"
        f" | {len(doc.anchors)} timestamp anchors",
        file=sys.stderr,
    )

    key = cache.key_for(doc)
    path = None if args.no_cache else cache.load(key)
    if path is not None:
        print(f"[cache hit] {cache.cache_file(key)}", file=sys.stderr)
    else:
        try:
            path = build_path(doc)
        except LLMError as exc:
            print(f"Path building failed: {exc}", file=sys.stderr)
            return 1
        if args.no_cache:
            print("[cache skipped] --no-cache", file=sys.stderr)
        else:
            print(f"[cache write] {cache.store(key, path)}", file=sys.stderr)

    for note in unverifiable_citations(doc, path):
        print(f"[citation warning] {note}", file=sys.stderr)

    payload = path.model_dump_json(indent=2)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print(f"[wrote] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
