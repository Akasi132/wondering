# Wondering

Paste an article or a captioned YouTube video. Wondering reads the source and lays out five
lessons — each with a diagram, a short read, a multiple-choice question, and the line in the
source it came from.

The citation is the point. Every lesson carries a timestamp (for video) or a verbatim quote
(for articles), and `unverifiable_citations()` checks each one against the extracted source
before the path is returned.

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # or requirements.txt to skip pytest
cp .env.example .env                                          # then fill in a key
.venv/Scripts/python -m uvicorn app.api:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. The two "already built" examples on the landing page are served
from `fixtures/` and need no API key, so the site is usable before you configure anything.

There is also a CLI:

```bash
python main.py https://example.com/article --out path.json
python main.py --list-models      # what your endpoint actually serves
python main.py --verify-models    # confirm the configured model IDs exist
```

## Configure a model

Copy `.env.example` to `.env`. Two backends sit behind one interface, chosen by `LLM_PROVIDER`:

| | `anthropic` | `openai` |
|---|---|---|
| Needs | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| Default model | `claude-sonnet-5` | `minimaxai/minimax-m3` |
| Works with | the Anthropic API | any OpenAI-compatible endpoint (NVIDIA, OpenAI, vLLM, Ollama) |

Run `python main.py --list-models` before trusting a model ID — slugs differ by host.

## Routes

| Route | What it does |
|---|---|
| `GET /` | the site |
| `GET /health` | `{"status": "ok"}` — what the Wasmer health check hits |
| `GET /examples/{chemistry,article}` | a pre-built path from `fixtures/`, no API key needed |
| `POST /ingest` | `{"url": "..."}` → a `Path` of five lessons |

`POST /ingest` is synchronous and a cold request takes as long as the model call — 30 to 90
seconds in the runs recorded in `NOTES.md`. The front end shows an elapsed clock rather than a
fake progress bar. A job queue is the v2 upgrade.

## Deploy to Wasmer Edge

Wasmer's autobuild detects a Python web app from `requirements.txt`, installs the
dependencies — including the native ones, from <https://pythonindex.wasix.org> — and runs the
ASGI app. No Dockerfile.

Two things in this repo exist for that pipeline specifically:

- **`main.py` re-exports the ASGI app**, so `main:app` resolves. A root `app.py` would be
  shadowed by the existing `app/` package, so that conventional spelling is unavailable here.
  `server.py` starts the same object explicitly if you would rather give a command.
- **`pydantic` is pinned to 2.12.5, not 2.13.4.** pydantic 2.13 requires pydantic-core 2.46,
  and the WASIX wheel index tops out at pydantic-core 2.41.5 — which is what 2.12.5 pins.
  pydantic-core is Rust, so there is no source fallback on wasm32.

### From your machine

```bash
wasmer login
wasmer deploy                       # add --build-remote to use Wasmer's build pipeline
```

Set `owner:` in `app.yaml` to your Wasmer username first, or pass `--owner`.

### From GitHub

1. Push this repo to GitHub.
2. In the Wasmer dashboard, open the app's **Git settings**, choose GitHub, and authorize
   Wasmer for the repository.
3. Pick the repo and the branch to deploy to production (usually `main`).

Pushes to that branch deploy from then on. `app.yaml` is read from the repo on each deploy.

### Secrets

`app.yaml` is in git, so no key goes in it. Set keys on the app instead:

```bash
wasmer app secret create --app wondering OPENAI_API_KEY nvapi-...
wasmer app secret create --app wondering ANTHROPIC_API_KEY sk-ant-...
```

The non-secret settings — provider, base URL, model, cache directory — are in `app.yaml`
under `env:`.

### The cache volume

`app.yaml` mounts a volume at `/data` and points `WONDERING_CACHE_DIR` at `/data/cache`. The
packaged application directory is not a durable place to write, and without a writable cache
every request regenerates from scratch and pays the model cost again. A failed cache write is
logged and does not fail the request, so a misconfigured volume degrades rather than breaks.

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -q
```

452 tests. Only `tests/test_extract.py` touches the network; the LLM suites stub the clients
and `tests/test_api.py` stubs DNS so no test can reach a resolver.

## What is and isn't verified

`NOTES.md` is the assumptions log and is specific about this. In short: the OpenAI backend has
been proven end to end against a live NVIDIA endpoint, the Anthropic backend has never made a
live call, and the Wasmer deploy itself has not been run — see "Deploying to Wasmer Edge" in
`NOTES.md` for exactly which parts of it are checked and which are inference.
