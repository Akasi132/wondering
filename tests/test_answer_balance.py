"""Option shuffling — `pathbuilder.balance_answer_positions`.

Two live runs put the correct answer at index 1 seven times out of ten and never at index 0
or 3, which made the article path's exercises answerable without reading the lesson. These
tests pin both halves of the fix: that shuffling never changes which option is correct, and
that the answers actually spread out afterwards.
"""

from collections import Counter

import pytest

from app.models import Exercise, Lesson, Path
from app.pathbuilder import balance_answer_positions, build_path, check_answer_indexes


def _lesson(order: int, *, options: list[str], answer_index: int) -> Lesson:
    return Lesson(
        order=order,
        title=f"Lesson {order}",
        explanation="An explanation long enough to look plausible.",
        mermaid="graph TD; A[In] --> B[Out];",
        exercise=Exercise(
            question=f"Question {order}?",
            options=list(options),
            answer_index=answer_index,
            why="Because the source says so.",
        ),
        citation="[00:00]",
    )


def _path(specs: list[tuple[list[str], int]]) -> Path:
    return Path(
        document_title="A Document",
        source_url="https://example.com/doc",
        lessons=[
            _lesson(i + 1, options=options, answer_index=answer)
            for i, (options, answer) in enumerate(specs)
        ],
    )


def _four(answer: int) -> tuple[list[str], int]:
    return (["A", "B", "C", "D"], answer)


def _correct_texts(path: Path) -> list[str]:
    return [lesson.exercise.options[lesson.exercise.answer_index] for lesson in path.lessons]


# --------------------------------------------------------------- correctness (the important half)


def test_the_correct_option_text_is_unchanged():
    """The whole fix is worthless if it ever remaps to the wrong answer."""
    path = _path([_four(1)] * 5)
    before = _correct_texts(path)

    balance_answer_positions(path, seed="https://example.com/doc")

    assert _correct_texts(path) == before


def test_no_options_are_added_lost_or_altered():
    path = _path([_four(1)] * 5)
    before = [sorted(lesson.exercise.options) for lesson in path.lessons]

    balance_answer_positions(path, seed="seed")

    assert [sorted(lesson.exercise.options) for lesson in path.lessons] == before


def test_answer_index_stays_in_range():
    path = _path([_four(1), (["A", "B", "C"], 2), _four(3), (["X", "Y", "Z"], 0), _four(0)])

    balance_answer_positions(path, seed="seed")

    assert check_answer_indexes(path) == []


def test_duplicate_option_text_still_resolves_to_the_right_answer():
    """Index-based permutation, not text lookup — 'None of the above' twice must not confuse it."""
    path = _path([(["Same", "Correct", "Same", "Same"], 1)] * 3)  # Path requires >= 3 lessons

    balance_answer_positions(path, seed="seed")

    for lesson in path.lessons:
        exercise = lesson.exercise
        assert exercise.options[exercise.answer_index] == "Correct"
        assert Counter(exercise.options) == Counter(["Same", "Correct", "Same", "Same"])


def test_three_option_exercises_are_handled():
    path = _path([(["A", "B", "C"], 0)] * 5)

    balance_answer_positions(path, seed="seed")

    for lesson in path.lessons:
        assert lesson.exercise.options[lesson.exercise.answer_index] == "A"
        assert 0 <= lesson.exercise.answer_index < 3


def test_mixed_option_counts_are_grouped_not_skewed():
    """A single cycle across 3- and 4-option lessons would starve index 3."""
    path = _path([_four(0), (["A", "B", "C"], 0), _four(0), (["A", "B", "C"], 0)])

    balance_answer_positions(path, seed="seed")

    for lesson in path.lessons:
        assert lesson.exercise.options[lesson.exercise.answer_index] == "A"


# --------------------------------------------------------------- distribution (the reported defect)


def test_five_four_option_lessons_never_all_share_one_position():
    """The exact reported failure: the live article run returned [1, 1, 1, 1, 1]."""
    for seed in [f"https://example.com/{i}" for i in range(200)]:
        path = _path([_four(1)] * 5)
        balance_answer_positions(path, seed=seed)
        positions = [lesson.exercise.answer_index for lesson in path.lessons]
        assert len(set(positions)) > 1, f"all answers in one slot for seed {seed}"


def test_every_position_is_used_across_five_lessons():
    """A shuffled round-robin over 4 options must cover all 4 within the first cycle."""
    path = _path([_four(1)] * 5)

    balance_answer_positions(path, seed="seed")

    positions = [lesson.exercise.answer_index for lesson in path.lessons]
    assert set(positions) == {0, 1, 2, 3}


def test_no_position_is_used_more_than_twice_across_five_lessons():
    path = _path([_four(1)] * 5)

    balance_answer_positions(path, seed="seed")

    counts = Counter(lesson.exercise.answer_index for lesson in path.lessons)
    assert max(counts.values()) <= 2


def test_position_distribution_is_roughly_uniform_across_many_documents():
    """Corrects the measured bias: index 1 seven times in ten, index 0 and 3 never."""
    counts: Counter = Counter()
    for i in range(400):
        path = _path([_four(1)] * 5)
        balance_answer_positions(path, seed=f"https://example.com/{i}")
        counts.update(lesson.exercise.answer_index for lesson in path.lessons)

    total = sum(counts.values())
    for index in range(4):
        share = counts[index] / total
        assert 0.20 < share < 0.30, f"index {index} took {share:.1%} of answers"


# --------------------------------------------------------------- determinism


def test_same_seed_produces_identical_output():
    """Two runs over one document must agree, or the disk cache starts contradicting itself."""
    a, b = _path([_four(1)] * 5), _path([_four(1)] * 5)

    balance_answer_positions(a, seed="https://example.com/doc")
    balance_answer_positions(b, seed="https://example.com/doc")

    assert a == b


def test_different_documents_shuffle_differently():
    a, b = _path([_four(1)] * 5), _path([_four(1)] * 5)

    balance_answer_positions(a, seed="https://example.com/one")
    balance_answer_positions(b, seed="https://example.com/two")

    assert a != b


# --------------------------------------------------------------- wiring


def test_build_path_balances_before_returning(monkeypatch):
    from app import pathbuilder

    biased = _path([_four(1)] * 5)
    monkeypatch.setattr(pathbuilder, "call_json", lambda *a, **k: biased)

    doc = _document()
    result = build_path(doc)

    positions = [lesson.exercise.answer_index for lesson in result.lessons]
    assert len(set(positions)) > 1
    assert _correct_texts(result) == ["B"] * 5  # still the option the model marked correct


def test_build_path_seeds_from_the_document_url(monkeypatch):
    """Seeding from the Path's own URL would use the model's echoed value, not the real one."""
    from app import pathbuilder

    def run(url):
        path = _path([_four(1)] * 5)
        path.source_url = "https://model-hallucinated-this.example"
        monkeypatch.setattr(pathbuilder, "call_json", lambda *a, **k: path)
        return build_path(_document(url=url))

    first = [lesson.exercise.answer_index for lesson in run("https://example.com/a").lessons]
    second = [lesson.exercise.answer_index for lesson in run("https://example.com/b").lessons]

    assert first != second


def test_out_of_range_answer_is_left_alone_rather_than_corrupted(caplog):
    """Unreachable via build_path, but must not silently mark a different option correct."""
    path = _path([_four(1)] * 3)  # Path requires >= 3 lessons
    path.lessons[0].exercise.answer_index = 9

    balance_answer_positions(path, seed="seed")

    assert path.lessons[0].exercise.answer_index == 9
    assert path.lessons[0].exercise.options == ["A", "B", "C", "D"]
    assert "Refusing to shuffle" in caplog.text
    # The other lessons in the same path are still shuffled normally.
    assert all(l.exercise.options[l.exercise.answer_index] == "B" for l in path.lessons[1:])


def _document(url: str = "https://example.com/doc"):
    from app.models import Document

    return Document(
        source_type="article",
        url=url,
        title="A Document",
        text="Some source text that the lessons were built from.",
    )


@pytest.fixture(autouse=True)
def _quiet(caplog):
    caplog.set_level("ERROR")
