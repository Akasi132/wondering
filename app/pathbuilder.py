"""Document -> Path of 5 lessons, in one schema-validated LLM call, then checked."""

import logging
import random

from app.llm import LLMError, call_json, truncate_source
from app.models import Document, Path

logger = logging.getLogger(__name__)

EXPECTED_LESSONS = 5

SYSTEM = """You turn one source document into a learning path of exactly 5 lessons.

Hard rules:
- Use ONLY the provided source text. Do not add facts, examples, numbers, or names that
  are not in it. If the source does not cover something, leave it out.
- Order the lessons so each one is a prerequisite for the ones after it. `order` runs 1..5.
- Each lesson is a self-contained unit that takes about 3 minutes to read: roughly
  200-350 words of explanation.
- `mermaid` is a valid Mermaid diagram spec as plain text (for example
  "graph TD; A[Input] --> B[Hidden layer]; B --> C[Output];"). No markdown fences.
- `exercise` is one multiple-choice question with 3 or 4 options. `answer_index` is a
  0-BASED index into `options`, so for 3 options it must be 0, 1, or 2. Never 1-based.
  `why` gives a one-line reason.
- `citation` must be checkable against the source text you were given:
  * If the source contains [MM:SS] or [HH:MM:SS] timestamp markers, copy ONE of those
    markers exactly as it appears, including the brackets. Never invent a timestamp and
    never use one that is not present in the text above.
  * Otherwise, quote a short verbatim phrase (5-12 words) copied exactly from the source.
- Return exactly 5 lessons."""


def _document_block(doc: Document) -> str:
    source = truncate_source(doc.text)
    marker_note = (
        "The source text contains timestamp markers in square brackets. Cite one of them "
        "verbatim.\n"
        if doc.anchors
        else "The source text has no timestamps. Cite a short verbatim phrase.\n"
    )
    return f"""Document title: {doc.title}
Source URL: {doc.url}
Source type: {doc.source_type}
{marker_note}
<source_text>
{source}
</source_text>

Build the 5-lesson path from this source text only."""


def check_answer_indexes(path: Path) -> list[str]:
    """Return a description of every exercise whose answer_index is out of range.

    PLAN.md Section 4 defines answer_index as a bare int, so this cannot live in the Pydantic
    contract without drifting from the spec. PLAN.md Step 6's Verify requires the check, so it
    lives here instead. The common real failure is the model emitting a 1-based index.
    """
    problems = []
    for lesson in path.lessons:
        count = len(lesson.exercise.options)
        index = lesson.exercise.answer_index
        if not 0 <= index < count:
            problems.append(
                f"lesson {lesson.order} ({lesson.title!r}): answer_index={index} "
                f"but there are {count} options (valid 0..{count - 1})"
            )
    return problems


def _target_positions(count: int, n_options: int, rng: random.Random) -> list[int]:
    """`count` answer positions, spread as evenly as n_options allows, in random order.

    A shuffled round-robin rather than independent draws: with 5 lessons and 4 options,
    independent shuffling lands every answer in the same slot roughly 1 run in 256. Rare, but
    it is the exact defect being fixed, and a learner who hits it can score full marks without
    reading. Cycling through shuffled blocks makes that outcome impossible instead of unlikely.
    """
    blocks: list[int] = []
    while len(blocks) < count:
        block = list(range(n_options))
        rng.shuffle(block)
        blocks.extend(block)
    return blocks[:count]


def balance_answer_positions(path: Path, *, seed: str) -> None:
    """Shuffle each exercise's options in place so the answer isn't always in one slot.

    Measured across the first two live runs (10 lessons, 4 options each), the model put the
    correct answer at index 1 seven times and never at index 0 or 3. The answers were right;
    the positions were not usable. Option-position bias is well documented in LLMs and does not
    respond reliably to prompting, so it is corrected deterministically here instead of being
    asked for in `SYSTEM` or enforced through the retry channel — a retry would burn a whole
    second generation to fix something a permutation fixes for free.

    Seeded from the document URL so a given document always shuffles the same way: two runs
    over the same source produce identical output, which keeps the disk cache meaningful and
    makes this testable.

    Permutes indices rather than looking the correct option up by text, so exercises with
    duplicate option strings still resolve to the right answer.
    """
    rng = random.Random(seed)
    # Grouped by option count so the round-robin is over a consistent range; the schema allows
    # 3 or 4 options and a single cycle across mixed sizes would skew toward the low indices.
    by_size: dict[int, list] = {}
    for lesson in path.lessons:
        by_size.setdefault(len(lesson.exercise.options), []).append(lesson)

    for n_options, lessons in sorted(by_size.items()):
        targets = _target_positions(len(lessons), n_options, rng)
        for lesson, target in zip(lessons, targets):
            exercise = lesson.exercise
            correct = exercise.answer_index
            if not 0 <= correct < n_options:
                # Should be unreachable: build_path only calls this on a path that already
                # passed check_answer_indexes. Skip rather than corrupt the answer.
                logger.error(
                    "Refusing to shuffle lesson %d: answer_index=%d out of range",
                    lesson.order,
                    correct,
                )
                continue
            order = list(range(n_options))
            order.remove(correct)
            rng.shuffle(order)
            order.insert(target, correct)  # correct option lands exactly on `target`
            exercise.options = [exercise.options[i] for i in order]
            exercise.answer_index = target


def unverifiable_citations(doc: Document, path: Path) -> list[str]:
    """Citations that cannot be traced back to the source text (Directive 8).

    For timestamped sources, a citation must contain one of the markers actually shown to the
    model. For articles, the quoted phrase should appear in the source. Reported rather than
    raised, because a citation can be legitimately reworded while still pointing somewhere real.
    """
    problems = []
    for lesson in path.lessons:
        citation = lesson.citation
        if doc.anchors:
            if not any(anchor in citation for anchor in doc.anchors):
                problems.append(
                    f"lesson {lesson.order}: citation {citation!r} contains no timestamp "
                    "that appears in the transcript"
                )
        else:
            probe = citation.strip().strip('"“”').lower()
            if len(probe) < 12:
                # e.g. "para 9", "Section 3" — too short to locate, so unverifiable rather
                # than verified. Silently passing these was a hole.
                problems.append(
                    f"lesson {lesson.order}: citation {citation!r} is too short to verify "
                    "against the source"
                )
            elif probe not in doc.text.lower():
                problems.append(
                    f"lesson {lesson.order}: citation {citation!r} is not a verbatim quote "
                    "from the source"
                )
    return problems


def structural_problems(path: Path) -> list[str]:
    """Everything wrong with an otherwise schema-valid Path, as correction instructions.

    Fed to call_json's retry channel rather than raised, because both failures here are
    exactly what a model can fix when told (PLAN.md Directive 7's retry-once).
    """
    problems = []
    if len(path.lessons) != EXPECTED_LESSONS:
        problems.append(
            f"You returned {len(path.lessons)} lessons; return exactly {EXPECTED_LESSONS}."
        )
    for note in check_answer_indexes(path):
        problems.append(
            f"{note}. answer_index is 0-based — the first option is 0, not 1."
        )
    return problems


def build_path(doc: Document, *, model: str | None = None) -> Path:
    # model=None lets app.llm pick the right default for whichever provider is configured;
    # pinning MODEL_PATHBUILDER here would send a Claude model ID to an OpenAI endpoint.
    path = call_json(
        SYSTEM,
        _document_block(doc),
        Path,
        model=model,
        max_tokens=16_000,
        thinking={"type": "adaptive"},
        post_validate=structural_problems,
    )

    # Provenance comes from the extractor, not the model. The model is asked to echo these,
    # but a hallucinated URL would make the path misattribute its own source. A mismatch is
    # logged before being overwritten: it is evidence the model is drawing on pretrained
    # knowledge rather than the supplied text, which would affect the lesson bodies too.
    if path.source_url != doc.url or path.document_title != doc.title:
        logger.warning(
            "Model returned provenance that does not match the document "
            "(title=%r url=%r; expected title=%r url=%r). Overwriting, but this suggests it "
            "may not be working from the supplied source text.",
            path.document_title,
            path.source_url,
            doc.title,
            doc.url,
        )
    path.document_title = doc.title
    path.source_url = doc.url

    # After the provenance fix so the seed matches the URL actually recorded on the Path, and
    # after call_json's post_validate has already confirmed every answer_index is in range.
    balance_answer_positions(path, seed=doc.url)

    for note in unverifiable_citations(doc, path):
        logger.warning("Unverifiable citation: %s", note)

    return path
