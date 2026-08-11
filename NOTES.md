# NOTES — assumptions log

Project: **wondering** — Lesson Ingester v1 (paste a URL, get a 5-lesson path).
Built following `PLAN.md`. Every claim below was verified by running the command shown,
not recalled from memory. Where something was **not** verified, it says so explicitly.

## Environment

- Python 3.11.9 (Windows 11)
- venv at `.venv/` — all commands below use `.venv\Scripts\python.exe`

## Pinned versions (verified by `pip install` output on install day)

| Package | Version |
|---|---|
| fastapi | 0.141.1 |
| uvicorn | 0.52.1 |
| youtube-transcript-api | 1.2.4 |
| trafilatura | 2.2.0 |
| anthropic | 0.121.0 |
| pydantic | 2.13.4 |
| python-dotenv | 1.2.2 |
| pytest | 9.1.1 |

Import check (Section 2 of PLAN.md):

```
$ python -c "import fastapi, trafilatura, anthropic, pydantic, dotenv; print('ok')"
ok
```

## Real test URLs (Directive 6)

- **Article:** https://karpathy.github.io/2015/05/21/rnn-effectiveness/
  ("The Unreasonable Effectiveness of Recurrent Neural Networks", 45,102 chars extracted)
- **YouTube (captioned):** https://www.youtube.com/watch?v=aircAruvnKk
  ("But what is a neural network? | Deep learning chapter 1", 286 caption snippets,
  `is_generated=False`, 18,710 chars with 35 interleaved timestamp markers)
- **YouTube (failure path):** https://www.youtube.com/watch?v=BaW_jenozKc — unavailable video,
  raises `ExtractionError` (not `NoCaptionsError`).

## ⚠️ Model verification — NOT DONE

**This is the one PLAN.md directive that is not satisfied.** Directive 3 requires confirming
model IDs against the live list at build time. That needs an `ANTHROPIC_API_KEY`, which was
not available, so **it has not been run.**

- Path building / lesson prose: `claude-sonnet-5` — **unconfirmed**
- Cheap extraction / checks: `claude-haiku-4-5` — **unconfirmed**, and never actually called
  by v1 (the constant records PLAN.md's intent only)

To satisfy Directive 3, put a key in `.env` and run:

```
python main.py --verify-models
```

then paste the `OK` lines into this section. Until that is done, treat both IDs as unverified.

> An earlier revision of `app/llm.py` and this file claimed these IDs *had* been verified and
> pointed at a `scripts/verify_models.py` that never existed. That claim was false and has been
> removed. Recording it here because Directive 9 asks for the assumptions log to be honest, and
> a fabricated verification claim is worse than an admitted gap.

`claude-sonnet-5` behaviours the code relies on (from the SDK/docs, **not** from a live call):
- Adaptive thinking is on by default; `app/pathbuilder.py` sets `thinking={"type": "adaptive"}`
  explicitly so it is visible in the code rather than implicit.
- `max_tokens` caps thinking + response text together. Set to 16,000 for path building, which
  keeps the request under the SDK's non-streaming timeout guard.
- Manual `budget_tokens` and non-default `temperature`/`top_p`/`top_k` are rejected on this
  model, so none are used.

## Provider layer — added after v1

`app/llm.py` now has two backends behind one `call_json` contract, selected by `LLM_PROVIDER`:

| | `anthropic` (default) | `openai` |
|---|---|---|
| Call | `client.messages.parse(output_format=Path)` | `chat.completions.create` |
| Schema enforcement | inside the SDK | prompt + negotiated `response_format` + local validation |
| Default model | `claude-sonnet-5` | `minimaxai/minimax-m2` (**a guess — see below**) |
| `thinking` param | forwarded | ignored (no equivalent) |

The Anthropic path is byte-for-byte what it was; the refactor moved it behind a dispatch and
changed nothing about the request it builds. All 396 original tests still pass.

**Why not just swap SDKs.** An OpenAI-compatible endpoint has no `output_format`, so the
Anthropic backend's central guarantee — schema validation inside the call — has no equivalent.
Replacing rather than adding would have thrown that away for the Anthropic user too, and
invalidated `tests/test_llm.py` (50KB of scripted `.messages.parse` behaviour) for nothing.

**Three things the OpenAI path has to do that the Anthropic path never did:**

1. **Strip reasoning tags.** MiniMax M2 emits `<think>…</think>` around its scratchpad, and
   endpoints differ on whether that lands in `reasoning_content` or inline in the message.
   `extract_json_object()` strips closed *and* unclosed `<think>` blocks, markdown fences, and
   conversational preamble before parsing. An unclosed tag matters: a response truncated
   mid-thought otherwise swallows the JSON object.
2. **Negotiate `response_format`.** Hosts vary on whether they accept `json_schema`,
   `json_object`, or neither. `auto` mode tries them in that order, downgrades on a 400, and
   remembers the winner for the process. `strict: false` on the json_schema form because `Path`
   uses `$defs`/`$ref`, which strict mode rejects.
3. **Merge consecutive user turns.** The retry path appends a second `user` message. Anthropic
   combines same-role messages server-side; several OpenAI-compatible hosts 400 instead — which
   would have broken the retry, i.e. exactly the mechanism that fixes a 1-based `answer_index`.

**Cache keys now include provider and model** (`cache.key_for` folds in `llm.generator_id()`).
Without it, running the same URL under a different backend serves the *other* model's lessons
straight off disk with nothing in the response saying so. With two backends that stops being
hypothetical.

**`build_path(model=...)` now defaults to `None`** rather than `MODEL_PATHBUILDER`, so
`app.llm` resolves the right default per provider. Pinning it sent a Claude model ID to
whatever endpoint was configured. `tests/test_llm.py::test_build_path_passes_schema_and_request_options`
was rewritten to pin the new contract (plus a companion test asserting the resolved Anthropic
default is still `claude-sonnet-5`, so the deferral can't silently change which model runs).

### ⚠️ `minimaxai/minimax-m2` is unverified

Same status as the Claude model IDs above: not confirmed against a live endpoint. NVIDIA's
slugs are its own business and cannot be derived. Run:

```
python main.py --list-models
```

and set `LLM_MODEL` to whatever it actually reports. `--verify-models` falls back to scanning
the list endpoint when a host returns 404 for `/v1/models/{id}` — many implement `/v1/models`
only, and treating that 404 as "model missing" would have been wrong.

### ✅ LIVE RUN — the OpenAI path is proven end to end (2026-08-11)

**This closes PLAN.md Section 7's definition of done, on the OpenAI backend.** Steps 5-9 had
never been run against a real model before this; they have now.

```
$ python main.py --list-models
102 model(s) available via provider=openai

$ python main.py --verify-models
OK   minimaxai/minimax-m3 | minimaxai/minimax-m3 (listed; no per-model endpoint)

$ python main.py https://www.youtube.com/watch?v=5iTOphGnCtg --out fixtures/path_chemistry.json
[generator] openai:minimaxai/minimax-m3
[extracted] youtube | 'GENERAL CHEMISTRY explained in 19 Minutes' | 23920 chars | 36 anchors
[cache write] cache/8f2e364e...json
```

Endpoint: NVIDIA (`https://integrate.api.nvidia.com/v1`). Result, checked against the source:

| Check | Result |
|---|---|
| Lesson count | 5 ✅ |
| `answer_index` in range | all 5 ✅ — and values were 1,2,2,2,1, so not the degenerate always-0 case |
| Citations traceable to transcript | all 5 ✅ — `unverifiable_citations()` returned nothing |
| Mermaid specs | all 5 parse as `graph TD; …;` ✅ |
| Explanation length | 179-223 words (prompt asks 200-350 — slightly short, see below) |
| Retries needed | none — valid on the first attempt |

**The model slug could not have been guessed.** NVIDIA serves `minimaxai/minimax-m3`, not the
`minimax-m2` this file previously assumed. `--list-models` exists precisely because of this and
earned its place on first use; `DEFAULT_OPENAI_MODEL` has been corrected.

**NVIDIA accepts `json_schema`.** No downgrade warning was logged, so the negotiation ladder
settled on the strongest mode available. The `json_object` and no-response_format rungs remain
covered by tests but are unexercised against this host.

**`max_tokens=16_000` was sufficient** — the predicted `finish_reason=length` failure from
MiniMax's reasoning tokens did not materialise on a 23,920-char transcript. That was the
most-likely-first-failure prediction and it was wrong; no headroom measurement was taken, so
a longer source could still hit it.

### Article path — also proven live (2026-08-11)

```
$ python main.py https://www.sciencedaily.com/releases/2026/08/260807035140.htm \
    --out fixtures/path_article.json
[extracted] article | 'New fuel cell breakthrough could help power energy-hungry data centers'
            | 8480 chars | 0 timestamp anchors
```

This exercises the **other** citation branch — verbatim-quote matching against `doc.text`
rather than timestamp-marker matching. 5 lessons, all `answer_index` in range, and all five
citations were exact substrings of the source (38-62 chars, comfortably over the 12-char
"too short to verify" floor). `unverifiable_citations()` returned nothing. Both branches of
Directive 8's check are now proven against a live model.

### ~~⚠️~~ ✅ Multiple-choice answers clustered at option B — FIXED

Across both live runs — 10 lessons, every exercise with 4 options:

| index | count | |
|---|---|---|
| 0 | 0 | |
| 1 | 7 | ####### |
| 2 | 3 | ### |
| 3 | 0 | |

Uniform would be ~2.5 each. **`answer_index` was never 0 and never 3.** The article run was
`[1, 1, 1, 1, 1]` — every correct answer in the same slot.

The answers themselves are *correct*; this is a position bias, not an accuracy problem. But it
makes the exercises trivially gameable: a learner who always picks the second option scores 5/5
on the article path without reading anything. For a product whose whole value is the exercise,
that is a content defect even though every existing check passes.

It was invisible to the pipeline by construction — `check_answer_indexes()` only asks whether
the index is *in range*, which it always was.

**Fix: `pathbuilder.balance_answer_positions()`**, called from `build_path` after the
provenance correction. It permutes each exercise's `options` and rewrites `answer_index` to
follow the correct option. Rejected alternatives: asking for a distribution in `SYSTEM` (option
position bias is well documented in LLMs and does not respond reliably to prose), and routing it
through `structural_problems()` (burns a whole second generation on something a permutation
fixes for free).

Three properties that mattered more than the shuffle itself:

- **Permutes indices, not text.** Looking the correct option up by string would resolve to the
  wrong one whenever two options share text ("None of the above" twice).
- **Seeded from `doc.url`,** so a document always shuffles the same way. Without this the disk
  cache and a fresh generation would disagree about which option is correct.
- **Shuffled round-robin, not independent draws.** Five lessons over four options land in one
  slot about 1 run in 256 under independent shuffling — rare, but it is precisely the defect
  being fixed. Cycling through shuffled blocks makes it impossible rather than unlikely, and
  `test_five_four_option_lessons_never_all_share_one_position` checks 200 seeds.

Verified by regenerating both live paths with `--no-cache`:

| index | before | after |
|---|---|---|
| 0 | 0 | 2 |
| 1 | 7 | 3 |
| 2 | 3 | 2 |
| 3 | 0 | 3 |

`youtube [1,2,3,0,3]`, `article [2,0,1,3,1]` — both still 5 lessons, all indices in range, all
citations traceable. Covered by `tests/test_answer_balance.py` (17 tests); the correctness half
(the correct option text never changes) matters more than the distribution half.

Two pre-existing tests in `tests/test_llm.py` asserted `build_path` returned lesson content
byte-identical to the model's output. That contract genuinely changed, so both were updated to
compare everything *except* option order while additionally asserting the correct option text
survives — a strictly stronger check than the equality they replaced.

### Explanations run short on short sources

The article is 8,480 chars — about a third of the chemistry transcript's 23,920. Its lessons
came out at 109-183 words against a 200-350 word target (the video run managed 179-223). Five
lessons of 200+ words needs roughly 1,000+ words of source material to draw on without padding;
below that the model correctly declines to invent filler, and the "~3 minute read" promise
quietly stops holding. Either scale lesson count to source length, or treat the word range as
aspirational and say so.

### Still not verified

1. **Anthropic path has still never made a live call.** Everything above is the OpenAI backend.
   `claude-sonnet-5` / `claude-haiku-4-5` remain unconfirmed (see the section above).
2. **One live run is not a reliability measurement.** Five-lesson compliance, citation accuracy,
   and answer correctness held once, on one video. Nothing here establishes a rate.
3. **Answer correctness was spot-checked by reading, not tested.** The four answers inspected
   (protons define the element; EN difference >1.7 gives an ionic bond; metallic bonding is
   delocalised; pH 7 is neutral) are right, but no automated check verifies an answer is
   *correct* — only that its index is in range.
4. **Explanations run slightly under the 200-350 word target** (179-223). Not a failure, but if
   the ~3-minute read is load-bearing, the prompt needs tightening or the range needs revising.

## Confirmed API surfaces (Directive 1)

### youtube-transcript-api 1.2.4

The old static `YouTubeTranscriptApi.get_transcript(...)` **does not exist** in 1.x. Confirmed
by `dir()` and `inspect.signature()`:

- `YouTubeTranscriptApi()` is instantiated; `.fetch(video_id, languages=('en',), preserve_formatting=False)`
  returns a `FetchedTranscript`; `.list(video_id)` returns a `TranscriptList`.
- `FetchedTranscript` is a dataclass with fields `snippets, video_id, language, language_code,
  is_generated`, is iterable, and has `.to_raw_data()`.
- `FetchedTranscriptSnippet` has fields `text, start, duration`.
- Real first snippet on the test video: `FetchedTranscriptSnippet(text='This is a 3.', start=4.22, duration=1.18)`

Error mapping verified with real calls:
- `TranscriptsDisabled` / `NoTranscriptFound` -> `NoCaptionsError`
- `VideoUnavailable` -> `ExtractionError`
- `CouldNotRetrieveTranscript` (the base class, covering `RequestBlocked`, `IpBlocked`,
  `AgeRestricted`, `VideoUnplayable`, `YouTubeRequestFailed`, `PoTokenRequired`,
  `YouTubeDataUnparsable`) -> `ExtractionError`. Added after review: without it, YouTube
  blocking a datacenter IP — the most likely production failure — escaped as an unhandled 500.
- `OSError` (requests' network errors) -> `ExtractionError`

### trafilatura 2.2.0

- `fetch_url(url) -> str | None` (returned 81,444 chars of HTML for the test article)
- `extract(html, url=..., include_comments=False, include_tables=False) -> str | None`
  (returned 45,120 chars of clean article text)
- `extract_metadata(html)` returns a `trafilatura.settings.Document` with a `.title` attribute
  (`'The Unreasonable Effectiveness of Recurrent Neural Networks'`)

Note: trafilatura's own class is also named `Document`, which collides with our Pydantic
`Document`. `app/extract.py` avoids the collision by calling `trafilatura.extract_metadata`
through the module rather than importing the name.

### YouTube video titles

The transcript API does not return a title. Titles come from YouTube's public oEmbed endpoint
(`https://www.youtube.com/oembed?url=...&format=json`), confirmed to return a `title` key. This
is best-effort — a failure falls back to `"YouTube video <id>"` rather than failing extraction.

### anthropic 0.121.0

- `client.messages.parse(...)` accepts `output_format=<PydanticModel>` and returns a response
  with `.parsed_output`. Validation happens **inside** the SDK call
  (`anthropic/lib/_parse/_response.py` calls `TypeAdapter.validate_json`), so a `ValidationError`
  genuinely escapes `messages.parse` and the retry path in `call_json` is reachable, not dead.
- `anthropic.APIError` is the common base of `APIStatusError` and `APIConnectionError`
  (confirmed via `__mro__`). `call_json` catches it and re-raises as `LLMError`.
- `response.stop_details` is a real field defaulting to `None`, so the refusal branch cannot
  `AttributeError`.
- `client.models.retrieve(id)` exists, with `.id` / `.display_name`.

## Deviations from PLAN.md (deliberate, with reasons)

1. **Cache key is not just `sha256(doc.text)`.** Step 9 says to hash `doc.text`. The key in
   `app/cache.py` is `sha256(CACHE_VERSION + doc.url + text_hash(doc.text))`. Text alone
   collides across URLs — `youtu.be/X` and `youtube.com/watch?v=X` extract byte-identical
   transcripts — and on a collision the cached `Path` carries the *first* URL's `source_url`,
   so the response misdescribes the request. Text alone also survives prompt and model changes,
   so editing `SYSTEM` would serve stale lessons forever. `text_hash()` is still exactly the
   plain sha256 the plan specifies; `key_for()` composes it.
2. **`answer_index` range checking lives in `app/pathbuilder.py`, not `app/models.py`.**
   Section 4 mandates `answer_index: int` with no constraint and says the contracts must match
   exactly, so tightening the model would drift from the spec. But Step 6's Verify requires
   asserting the index is within range, and nothing did. `check_answer_indexes()` fills the gap
   and `build_path` raises on violation. This matters: the most common real failure is the model
   emitting a **1-based** index, which validates cleanly and silently marks the wrong option
   correct in every lesson.
3. **"Exactly 5 lessons" is enforced in code.** The schema bound is 3–6 per Section 4, and the
   Anthropic SDK drops `minItems`/`maxItems` from the JSON schema it sends
   (`anthropic/lib/_parse/_transform.py` demotes them to a description hint), so the API does
   not enforce array bounds at all. `build_path` raises if the count is not 5.
4. **Transcript text carries interleaved `[MM:SS]` markers.** See the citation section below.
5. **`main.py` gained a `--verify-models` flag** rather than a `scripts/` directory, to keep the
   Section 3 layout intact while still providing a one-command way to satisfy Directive 3.
6. **`POST /ingest` validates the URL scheme and rejects local/link-local hosts.** Not in scope,
   but the endpoint otherwise fetches any string it is handed server-side (`169.254.169.254`,
   internal hostnames) and puts the result in a cache file and an LLM prompt.

## Citations and Directive 8 — the fix that mattered most

Originally `extract_youtube` built one `MM:SS` anchor per snippet into `Document.anchors`, but
`pathbuilder` never passed `anchors` to the model — the prompt got only the flat joined
transcript with all timing discarded, while the system prompt asked for `"04:12"`-style
timestamps. **The model was being asked for information that was not in its input, so every
video citation would have been invented** — Directive 8 failing in exactly the way it exists to
prevent, and arguably fabricated data under Directive 5.

Fixed by `_transcript_text()`, which interleaves a `[MM:SS]` marker roughly every 30 seconds
into the text the model actually sees, and returns those markers as `anchors`. The prompt now
says to copy one marker verbatim. `unverifiable_citations()` then checks each citation against
`doc.anchors` (timestamped sources) or against the source text (articles).

Verified on the real video: 35 markers, spaced ~30s, and **every returned marker is present in
the text** (`markers absent from text: none`). `_timestamp` also gained an hours component —
it previously rendered a 61-minute mark as `"61:01"`, so any video over an hour would have
produced citations matching no timestamp YouTube displays.

Citation checking is reported as a warning, not a hard failure, because a citation can be
legitimately reworded while still pointing somewhere real. Out-of-range `answer_index` and a
wrong lesson count *are* hard failures, because neither has an innocent explanation.

## Assumptions I could not fully verify

1. **Model IDs are unverified.** See the Model verification section above. This is the live gap.
2. ~~**Steps 5–9 have never been run end to end.**~~ **RESOLVED 2026-08-11 for the OpenAI
   backend** — a real `Path` was generated from a live MiniMax M3 call and passed every
   structural and citation check. See "LIVE RUN" above. Still unresolved for the **Anthropic**
   backend, which has never made a live call, so Section 7's definition of done is met on one
   of the two paths.
3. **No real captionless public video found.** YouTube auto-generates English captions for
   nearly every public video, so a "captions fully disabled" video was not findable. The
   `NoCaptionsError` path is proven with a real call asking a real video for a language it has
   no track in (`languages=("zu",)`), which raises the same `NoTranscriptFound` that
   `TranscriptsDisabled` maps to. The mapping is exercised; the exact upstream exception in the
   disabled-captions case is not.
4. **oEmbed is not a documented-stable contract** for titles. It works today and returns
   `title`; treated as best-effort with a fallback.
5. **`MAX_SOURCE_CHARS = 60_000`** is a judgement call, not a measured limit. The article test
   case is 45,102 chars and fits; longer sources are truncated with an explicit marker. No
   map-reduce in v1, per scope.
6. **`max_tokens` truncation is diagnosed indirectly.** Because `messages.parse` validates
   inside the SDK call, a response truncated mid-JSON raises `ValidationError` before the
   `stop_reason == "max_tokens"` check can run — so that check only fires in the narrower case
   where no text block is emitted at all. The final `LLMError` message names truncation as a
   likely cause instead. Detecting it properly would mean dropping to `messages.create` plus
   manual validation, which could not be verified without a key.
7. **The 30-second marker interval** is a guess at a useful citation granularity, not a measured
   optimum. Finer markers cost tokens; coarser ones make citations vaguer.
8. **The SSRF guard is layered but not complete.** It parses IP literals locally (including the
   legacy `127.1` / `2130706433` / `0177.0.0.1` forms, via `inet_aton`, because Windows'
   `getaddrinfo` rejects them), rejects internal-looking names, and resolves everything else,
   refusing any loopback/private/link-local/reserved/multicast address. Remaining gaps, all
   deliberate: **redirects are not re-validated** (trafilatura follows them, so a public URL
   that 302s to a private address still reaches the fetcher); **DNS rebinding** between check
   and fetch is possible; and an **unresolvable name is allowed through** on the reasoning that
   the fetch will fail anyway — which is true for a real lookup failure, but was the mechanism
   behind the Windows literal hole before literals were parsed separately.
9. **DNS resolution happens inside a Pydantic validator**, which is an odd place for a network
   round trip. `POST /ingest` is a sync `def`, so the blocking `getaddrinfo` occupies an AnyIO
   threadpool slot (default 40) for the full resolver timeout. A caller submitting
   slow-to-resolve hostnames could exhaust that pool — cheap DoS on an endpoint that is already
   synchronous. Moving the resolve step into the fetch path, where it can share a timeout and a
   cache, is the right fix; not done in v1.
10. **The 12-character citation threshold is arbitrary.** Below it, an article citation is
   reported as "too short to verify" rather than silently passing, but a legitimately short
   real quote ("per token") gets flagged too. Warning-only, so the cost is a noisier log.
11. **Retry conversion rate is an assumption.** Post-validation failures now feed the retry, but
   nothing measures how often the second attempt actually fixes the problem, and the retry uses
   the same sampling, so a model deterministically inclined to the same mistake may repeat it.

## Structural correctness: what feeds the retry vs what fails hard

`call_json` takes an optional `post_validate` callback. `pathbuilder.structural_problems`
supplies two checks — lesson count != 5, and out-of-range `answer_index` — and their failures
go through the **same retry channel** as schema errors rather than raising immediately.

The reasoning: both are instruction-following slips a model corrects when told, and neither has
any enforcement channel other than prose plus this retry, because the SDK strips
`minItems`/`maxItems` from the schema it sends. Hard-failing threw away a whole 16k-token
generation over a fixable off-by-one. A 1-based `answer_index` is the single most predictable
model error here and the most mechanically fixable.

The cost is real and is not mitigated yet: a retry re-sends the full source as a fresh turn, so
a model that *persistently* misbehaves now burns two full calls before erroring instead of one.
**The first optimization to make once a key is available is a `cache_control` breakpoint on the
`<source_text>` block** — the retry is nearly all identical input, so caching should cut the
second call's input cost by roughly 90%. Not added yet because the interaction of
`cache_control` with `output_format` could not be verified without a live call, and Directive 1
forbids shipping unconfirmed API shapes.

## v2 upgrades noted while building

- `POST /ingest` runs synchronously; a cold request takes as long as the LLM call. A job queue
  (submit -> poll) is the v2 upgrade.
- Prompt caching on the source block (see above) — the cheapest real win.
- No in-flight dedupe: two concurrent identical requests both miss the cache and both pay for a
  full generation.
- No request timeout: the SDK's effective non-streaming timeout is ~10 minutes, so a slow run
  can pin a worker that long before failing.
- No metric distinguishing "retry converted" from "retry wasted".
- Citation checking is advisory. Making it a gate is the natural next correctness step.
- Move SSRF resolution out of the validator and into the fetch path, and validate redirects.

## The website, and deploying to Wasmer Edge — added 2026-08-11

v1 was an API and a CLI. There is now a front end at `/`, and the repo carries the
configuration a Wasmer Edge deploy needs. Split into what was actually run and what was not,
because the two halves are very different here.

### What the site is

`app/static/{index.html,style.css,app.js}`, served by the same FastAPI app:

| Route | Added | Notes |
|---|---|---|
| `GET /` | yes | the page |
| `GET /static/*` | yes | `StaticFiles` mount |
| `GET /examples/{chemistry,article}` | yes | the two fixture paths from the live runs above |
| `GET /health`, `POST /ingest` | no | unchanged |

`/examples` reads the fixture back through `Path.model_validate_json` rather than streaming the
file, so a fixture that has drifted from the schema fails on the server instead of in the
browser. The name is looked up in a fixed dict; the caller's string never reaches the
filesystem.

Those two examples exist so that **a fresh deploy with no API key is still a working site**.
Without them the landing page's only action is one that returns 502.

The waiting state shows an elapsed clock, not a progress bar. `POST /ingest` is synchronous
and there is no server-side progress to report, so anything more specific would be invented.

### ✅ Verified locally (Chrome, `uvicorn app.api:app --port 8931`)

- All six routes return the expected status: `/` 200, `/static/style.css` 200,
  `/static/app.js` 200, `/examples/chemistry` 200, `/examples/article` 200,
  `/examples/nope` 404.
- Both fixture paths render end to end: five stations, mermaid diagrams drawn, citations shown
  (`[00:32] Most of chemistry is really just the behaviour` on the video path, a verbatim quote
  on the article path).
- The exercise flow works: selecting an option and checking it marks the correct option,
  reveals `why`, disables the inputs, fills the station marker, and advances "1 of 5 answered".
- Light and dark palettes both checked, including the mermaid theme, by temporarily disabling
  the dark media query and reloading so the diagrams re-rendered against the light tokens.
- No page console errors (the one error present comes from a browser extension).
- `452 passed` after every change in this section.

### ⚠️ Not verified

1. **Nothing has been deployed to Wasmer.** That needs a Wasmer account. Every claim below
   about Edge behaviour is inference from the docs and from `wasmer 7.2.1 --help`, not from a
   deploy. This is the same class of gap as the Anthropic backend above.
2. **Which Python the deploy lands on is unresolved, and it matters.** The WASIX wheel index
   ships `cp313` wheels, but `wasmer run wasmer/python` resolves to **Python 3.12.0**. The
   registry lists a `wasmer/python` version `3.13.0`, yet `wasmer run wasmer/python@=3.13.0`
   answers "not found". `.python-version` is set to `3.13` to match Wasmer's own FastAPI
   example, and the autobuild pipeline is left to choose the runtime — but if it picks 3.12,
   the `cp313` wheels for `pydantic-core`, `jiter`, `lxml`, and `charset-normalizer` do not
   match the ABI and the install fails. **This is the most likely first failure.**
3. **Request duration versus Edge's request timeout is unknown.** A cold `/ingest` took 30-90
   seconds in the live runs above. If Edge caps a request below that, every uncached path
   fails and the fix is the job queue already listed under v2 upgrades — not a config tweak.
4. **The volume is unproven.** `app.yaml` mounts `/data` and points `WONDERING_CACHE_DIR`
   there. Whether it is writable and persists across instances has not been observed. A failed
   cache write is logged and does not fail the request, so the failure mode is "pays for every
   generation twice" rather than an error page.
5. **Mobile layout was not seen.** `resize_window` did not reflow the page, so the narrow
   breakpoint is written but unobserved. Desktop only.
6. **The page depends on two CDNs** — jsdelivr for mermaid, Google Fonts for the typefaces.
   Both degrade rather than break (a failed mermaid load opens the diagram source instead, and
   the font stacks fall back to system faces), but it is an external dependency nothing else in
   this project has.

### Deliberate changes made for the deploy

1. **`pydantic` 2.13.4 -> 2.12.5.** pydantic 2.13.x requires `pydantic-core==2.46.x`; the WASIX
   index's newest `pydantic-core` is 2.41.5, which is exactly what pydantic 2.12.5 pins.
   pydantic-core is Rust, so there is no source fallback on wasm32 — this is a hard ceiling,
   not a preference. Verified by running the full suite on 2.12.5: `452 passed`. The other
   native dependencies (`lxml`, `jiter`, `charset-normalizer`, `regex`) are all present on the
   index at versions our loose constraints already allow, so only pydantic needed moving.
2. **`main.py` re-exports the ASGI app.** Wasmer's autobuild looks for the server object, and
   both of Wasmer's own FastAPI examples keep it in `main.py`. Ours was a CLI with no `app`
   attribute. The conventional fix — a root `app.py` — is unavailable here, because it would be
   shadowed by the existing `app/` package, so `app:app` would import the package and find no
   attribute. `server.py` was added alongside for hosts that want a command rather than a
   module path.
3. **`requirements.txt` split.** It is the file autobuild reads, so it now holds runtime
   dependencies only; `pytest` moved to `requirements-dev.txt` so it is not installed into the
   wasm image.
4. **`cache.CACHE_DIR` reads `WONDERING_CACHE_DIR`.** The packaged application directory is not
   a durable place to write. Read at import; tests still patch the module attribute directly,
   so nothing in the suite changed.
5. **No hand-written `wasmer.toml`.** Wasmer's deployed Python examples ship neither a
   `wasmer.toml` nor an `app.yaml` — detection handles packaging. Writing one against a schema
   nobody here has confirmed would be exactly the unverified-API-shape problem Directive 1
   exists to prevent. `app.yaml` only carries fields read off the published configuration
   reference, and it parses as YAML.

`wasmer app secret create [name] [value] --app <APP>` was confirmed against `wasmer 7.2.1
--help` before being written into the README, rather than recalled.

### One pre-existing risk that this change makes worse

Assumption 9 above — DNS resolution inside a Pydantic validator on a synchronous endpoint —
was a note about an internal API. `/ingest` is now reachable from a public web page. A caller
feeding slow-to-resolve hostnames can occupy AnyIO's threadpool (default 40 slots) for the full
resolver timeout each, which is a cheap denial of service against a public deployment. Moving
the resolve into the fetch path was already the right fix; putting the app on the open internet
promotes it from tidiness to something worth doing before the site is shared.

## Verification log

### Step 1 — contracts

```
$ python -m pytest tests/test_models.py -q
3 passed
```

### Steps 2-4 — extractors and router (live network)

```
$ python -m pytest tests/test_extract.py -q -s
7 passed

[article] title='The Unreasonable Effectiveness of Recurrent Neural Networks' chars=45102
[youtube] title='But what is a neural network? | Deep learning chapter 1' chars=18430 anchors=286
[no-captions] No captions available for .../watch?v=aircAruvnKk in languages ['zu']. v1 has no
              audio-transcription fallback. (NoTranscriptFound)
[unavailable] Video unavailable: https://www.youtube.com/watch?v=BaW_jenozKc (VideoUnavailable)
```

Fixtures saved from these runs:
- `fixtures/article_sample.txt` — 45,102 chars
- `fixtures/youtube_sample.json` — 286 real snippets

### Step 8 — API boots under a real server

```
$ python -m uvicorn app.api:app --port 8931
HEALTH 200: {"status":"ok"}
INGEST empty-body -> HTTP 422 (FastAPI validation)
OPENAPI 200: paths = /health, /ingest
```

### Test suite

```
$ python -m pytest tests/ -q
437 passed, 1 warning in 8.91s
```

(396 before the provider layer; 40 new OpenAI-backend tests, plus one asserting the resolved
Anthropic default model is unchanged.)

(The warning is `StarletteDeprecationWarning` about httpx from the FastAPI TestClient import —
environment noise, not a failure.)

| File | What it covers |
|---|---|
| `tests/test_models.py` | Section 4 data contracts, valid and invalid |
| `tests/test_extract.py` | Live network: both real URLs, both failure paths |
| `tests/test_urls.py` | URL shapes, lookalike-domain rejection, timestamps, marker interleaving |
| `tests/test_cache.py` | Round trip, corrupt/stale entries, unicode, dir creation |
| `tests/test_llm.py` | Anthropic backend: retry logic, transport-error wrapping, post-validation, prompt assembly |
| `tests/test_llm_openai.py` | OpenAI backend: JSON extraction, response_format negotiation, retry, config |
| `tests/conftest.py` | Pins `LLM_PROVIDER=anthropic` so a real `.env` can't leak into the suite |
| `tests/test_api.py` | Status-code mapping, cache behaviour, SSRF rejection |

Only `tests/test_extract.py` touches the network. The LLM-facing suites stub the Anthropic
client entirely; `tests/test_api.py` stubs DNS via `AI_NUMERICHOST` so IP literals parse
locally and no test can reach a resolver.

Several tests deliberately pin behaviour that is *contested* rather than endorsed, with a
comment saying so — that is intentional, so a future change to those decisions shows up as a
failing test rather than a silent drift.

### Steps 5-7, 9 — LLM, path builder, CLI, cache round trip

**Pending — requires `ANTHROPIC_API_KEY` in `.env`. Not yet run.** These are the steps that
would close out Section 7's definition of done.
