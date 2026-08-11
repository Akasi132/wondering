"""Tests for app.cache — the sha256-keyed disk cache.

Every test monkeypatches app.cache.CACHE_DIR onto a pytest tmp_path so the real
repo-level cache/ directory is never read from or written to.
"""

import json

import pytest

from app import cache
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


def _path(title: str = "A Real Document", url: str = "https://example.com/a") -> Path:
    return Path.model_validate(
        {
            "document_title": title,
            "source_url": url,
            "lessons": [_lesson(i) for i in range(1, 4)],
        }
    )


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point app.cache.CACHE_DIR at a fresh, not-yet-created tmp directory."""
    target = tmp_path / "nested" / "cache"
    monkeypatch.setattr(cache, "CACHE_DIR", target)
    return target


# --- text_hash -------------------------------------------------------------


def test_text_hash_is_deterministic():
    text = "the same input text, twice"
    assert cache.text_hash(text) == cache.text_hash(text)


def test_text_hash_matches_known_sha256():
    # sha256("") — pins the algorithm, not just self-consistency.
    assert cache.text_hash("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_text_hash_differs_for_different_text():
    a = cache.text_hash("alpha")
    b = cache.text_hash("beta")
    assert a != b
    assert len(a) == len(b) == 64


def test_text_hash_is_sensitive_to_small_changes():
    assert cache.text_hash("hello world") != cache.text_hash("hello  world")
    assert cache.text_hash("Hello") != cache.text_hash("hello")


def test_text_hash_handles_unicode():
    # Must not raise on non-ASCII, and must stay deterministic + distinct.
    emoji = cache.text_hash("héllo wörld 🌍 — 日本語 текст")
    assert emoji == cache.text_hash("héllo wörld 🌍 — 日本語 текст")
    assert emoji != cache.text_hash("hello world")
    assert len(emoji) == 64


def test_text_hash_utf8_encoding_is_explicit():
    # sha256 of the UTF-8 bytes of "日本語", computed independently.
    import hashlib

    expected = hashlib.sha256("日本語".encode("utf-8")).hexdigest()
    assert cache.text_hash("日本語") == expected


# --- cache_file ------------------------------------------------------------


def test_cache_file_lives_under_cache_dir(cache_dir):
    key = cache.text_hash("anything")
    p = cache.cache_file(key)
    assert p.parent == cache_dir
    assert p.name == f"{key}.json"


# --- store -----------------------------------------------------------------


def test_store_creates_cache_dir_when_missing(cache_dir):
    assert not cache_dir.exists()
    key = cache.text_hash("create the dir please")
    written = cache.store(key, _path())
    assert cache_dir.is_dir()
    assert written.exists()


def test_store_returns_the_written_path(cache_dir):
    key = cache.text_hash("return value check")
    written = cache.store(key, _path())
    assert written == cache_dir / f"{key}.json"
    assert written.is_file()
    assert written.stat().st_size > 0


def test_store_writes_valid_json_matching_the_model(cache_dir):
    key = cache.text_hash("json shape")
    written = cache.store(key, _path(title="Shape Check"))
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["document_title"] == "Shape Check"
    assert len(data["lessons"]) == 3


def test_store_overwrites_an_existing_entry(cache_dir):
    key = cache.text_hash("overwrite me")
    cache.store(key, _path(title="First"))
    cache.store(key, _path(title="Second"))
    loaded = cache.load(key)
    assert loaded is not None
    assert loaded.document_title == "Second"


# --- load: happy path ------------------------------------------------------


def test_load_round_trips_a_path_exactly(cache_dir):
    original = _path(title="Round Trip", url="https://example.com/rt")
    key = cache.text_hash("round trip source text")
    cache.store(key, original)

    loaded = cache.load(key)
    assert loaded is not None
    assert isinstance(loaded, Path)
    assert loaded == original
    assert loaded.model_dump() == original.model_dump()


def test_store_and_load_round_trip_unicode(cache_dir):
    text = "Qu'est-ce que l'apprentissage ? — 機械学習 🌍"
    original = Path.model_validate(
        {
            "document_title": "Título ünico — 日本語 🌍",
            "source_url": "https://example.com/ünïcode?q=日本語",
            "lessons": [_lesson(i) for i in range(1, 4)],
        }
    )
    original.lessons[0].explanation = "Une explication avec des accents: éèêë — и кириллица."

    key = cache.text_hash(text)
    written = cache.store(key, original)
    # File is readable as UTF-8 and the non-ASCII survives the write.
    raw = written.read_text(encoding="utf-8")
    assert "日本語" in raw or "\\u65e5" in raw

    loaded = cache.load(key)
    assert loaded is not None
    assert loaded.document_title == "Título ünico — 日本語 🌍"
    assert loaded.source_url == original.source_url
    assert loaded.lessons[0].explanation == original.lessons[0].explanation
    assert loaded == original


# --- load: failure modes ---------------------------------------------------


def test_load_returns_none_for_missing_key(cache_dir):
    # Directory does not even exist yet.
    assert cache.load(cache.text_hash("never stored")) is None


def test_load_returns_none_for_missing_key_in_existing_dir(cache_dir):
    cache.store(cache.text_hash("something else"), _path())
    assert cache_dir.is_dir()
    assert cache.load(cache.text_hash("never stored")) is None


def test_load_returns_none_for_corrupt_json(cache_dir):
    key = cache.text_hash("corrupt entry")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.cache_file(key).write_text("{not valid json at all,,,", encoding="utf-8")
    assert cache.load(key) is None  # must not raise


def test_load_returns_none_for_empty_file(cache_dir):
    key = cache.text_hash("empty entry")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.cache_file(key).write_text("", encoding="utf-8")
    assert cache.load(key) is None


def test_load_returns_none_for_truncated_json(cache_dir):
    key = cache.text_hash("truncated entry")
    cache_dir.mkdir(parents=True, exist_ok=True)
    full = _path().model_dump_json()
    cache.cache_file(key).write_text(full[: len(full) // 2], encoding="utf-8")
    assert cache.load(key) is None


def test_load_returns_none_for_valid_json_wrong_schema(cache_dir):
    key = cache.text_hash("wrong schema")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.cache_file(key).write_text(
        json.dumps({"hello": "world", "lessons": "not a list"}), encoding="utf-8"
    )
    assert cache.load(key) is None


def test_load_returns_none_for_json_scalar(cache_dir):
    key = cache.text_hash("scalar json")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.cache_file(key).write_text('"just a string"', encoding="utf-8")
    assert cache.load(key) is None


def test_load_returns_none_for_stale_schema_too_few_lessons(cache_dir):
    # Valid JSON, right field names, but violates lessons min_length=3.
    key = cache.text_hash("stale schema")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.cache_file(key).write_text(
        json.dumps(
            {
                "document_title": "Stale",
                "source_url": "https://example.com/stale",
                "lessons": [_lesson(1)],
            }
        ),
        encoding="utf-8",
    )
    assert cache.load(key) is None


def test_load_returns_none_for_invalid_utf8_bytes(cache_dir):
    key = cache.text_hash("bad bytes")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.cache_file(key).write_bytes(b"\xff\xfe\x00\x01 not utf-8")
    assert cache.load(key) is None  # decode error must be swallowed too


# --- isolation guard -------------------------------------------------------


def test_tests_never_touch_the_real_cache_dir(cache_dir, tmp_path):
    """Sanity: the patched CACHE_DIR is under tmp_path, not the repo."""
    assert str(cache_dir).startswith(str(tmp_path))
    cache.store(cache.text_hash("isolation"), _path())
    assert (cache_dir / f"{cache.text_hash('isolation')}.json").exists()
