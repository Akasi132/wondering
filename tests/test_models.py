import pytest
from pydantic import ValidationError

from app.models import Path


def _lesson(order: int) -> dict:
    return {
        "order": order,
        "title": f"Lesson {order}",
        "explanation": "A short explanation of the idea.",
        "mermaid": "graph TD; A-->B;",
        "exercise": {
            "question": "What does A point to?",
            "options": ["B", "C", "D"],
            "answer_index": 0,
            "why": "The edge runs A to B.",
        },
        "citation": "00:00-01:30",
    }


def test_valid_path_validates():
    path = Path.model_validate(
        {
            "document_title": "A Real Document",
            "source_url": "https://example.com/a",
            "lessons": [_lesson(i) for i in range(1, 6)],
        }
    )
    assert len(path.lessons) == 5
    assert path.lessons[0].exercise.options[path.lessons[0].exercise.answer_index] == "B"


def test_too_few_options_raises():
    bad = _lesson(1)
    bad["exercise"]["options"] = ["B", "C"]  # min_length=3
    with pytest.raises(ValidationError):
        Path.model_validate(
            {
                "document_title": "A Real Document",
                "source_url": "https://example.com/a",
                "lessons": [bad] + [_lesson(i) for i in range(2, 6)],
            }
        )


def test_too_few_lessons_raises():
    with pytest.raises(ValidationError):
        Path.model_validate(
            {
                "document_title": "A Real Document",
                "source_url": "https://example.com/a",
                "lessons": [_lesson(1), _lesson(2)],  # min_length=3
            }
        )
