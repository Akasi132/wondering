"""The OpenAI-compatible backend.

Nothing here touches a real endpoint. `app.llm.openai_client` is monkeypatched to return a
stub whose `.chat.completions.create(**kwargs)` is fully scripted.

The point of these tests is that the structural guarantees the Anthropic backend gets from
`messages.parse` — schema validation, retry-once, exactly-5-lessons, 0-based answer_index —
still hold when the endpoint offers none of that machinery.
"""

import json

import httpx
import openai
import pytest
from pydantic import BaseModel

from app import llm
from app.models import Exercise, Lesson, Path


# --------------------------------------------------------------------------- helpers


def _lesson(order: int, *, answer_index: int = 0) -> Lesson:
    return Lesson(
        order=order,
        title=f"Lesson {order}",
        explanation="An explanation long enough to look real.",
        mermaid="graph TD; A[Atom] --> B[Molecule];",
        exercise=Exercise(
            question="What is an atom made of?",
            options=["Protons and electrons", "Only neutrons", "Nothing"],
            answer_index=answer_index,
            why="Stated directly in the source.",
        ),
        citation="[00:32]",
    )


def _path(n: int = 5, *, answer_index: int = 0) -> Path:
    return Path(
        document_title="GENERAL CHEMISTRY explained in 19 Minutes",
        source_url="https://www.youtube.com/watch?v=5iTOphGnCtg",
        lessons=[_lesson(i, answer_index=answer_index) for i in range(1, n + 1)],
    )


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeCompletion:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [FakeChoice(content, finish_reason)] if content is not None else []


class FakeCompletions:
    """Scripted create(). Each script entry is either an exception to raise or a response."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("create() called more times than the script allows")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeOpenAI:
    def __init__(self, script):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(script)


def _bad_request(message="unknown parameter response_format") -> openai.BadRequestError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": message}})
    return openai.BadRequestError(message, response=response, body=None)


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    )


@pytest.fixture(autouse=True)
def _openai_provider(monkeypatch):
    """Select the OpenAI backend and reset the negotiated-mode module global."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm, "_NEGOTIATED_JSON_MODE", None)


@pytest.fixture
def fake_api(monkeypatch):
    def _install(*script):
        api = FakeOpenAI(script)
        monkeypatch.setattr(llm, "openai_client", lambda: api)
        return api

    return _install


# --------------------------------------------------------------------------- extraction


def test_extracts_a_bare_json_object():
    assert json.loads(llm.extract_json_object('{"a": 1}')) == {"a": 1}


def test_strips_reasoning_tags_before_parsing():
    """MiniMax M2 and other reasoning models emit <think> blocks around their scratchpad."""
    raw = '<think>The user wants JSON. Let me plan.</think>\n{"a": 1}'
    assert json.loads(llm.extract_json_object(raw)) == {"a": 1}


def test_strips_an_unclosed_reasoning_tag():
    """A response cut off mid-thought leaves <think> open; it must not swallow the object."""
    raw = '{"a": 1}\n<think>still thinking when the budget ran out'
    assert json.loads(llm.extract_json_object(raw)) == {"a": 1}


def test_strips_markdown_fences():
    raw = '```json\n{"a": 1}\n```'
    assert json.loads(llm.extract_json_object(raw)) == {"a": 1}


def test_strips_conversational_preamble_and_trailer():
    raw = 'Sure! Here is the path:\n{"a": 1}\nLet me know if you want changes.'
    assert json.loads(llm.extract_json_object(raw)) == {"a": 1}


def test_keeps_nested_objects_intact():
    raw = 'text {"a": {"b": [1, 2]}} more text'
    assert json.loads(llm.extract_json_object(raw)) == {"a": {"b": [1, 2]}}


@pytest.mark.parametrize("raw", ["", "   ", "no json here at all", "}{"])
def test_rejects_output_with_no_object(raw):
    with pytest.raises(ValueError):
        llm.extract_json_object(raw)


# --------------------------------------------------------------------------- happy path


def test_valid_json_is_parsed_into_the_schema_model(fake_api):
    expected = _path()
    fake_api(FakeCompletion(expected.model_dump_json()))

    got = llm.call_json("SYS", "USER", Path)

    assert got == expected


def test_schema_is_embedded_in_the_system_prompt(fake_api):
    """The endpoint may ignore response_format entirely, so the schema must also be prose."""
    api = fake_api(FakeCompletion(_path().model_dump_json()))

    llm.call_json("SYS", "USER", Path)

    system = api.chat.completions.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("SYS")
    assert "<json_schema>" in system["content"]
    assert "lessons" in system["content"]


def test_thinking_param_is_not_forwarded(fake_api):
    """`thinking` is Anthropic-only; pathbuilder passes it unconditionally."""
    api = fake_api(FakeCompletion(_path().model_dump_json()))

    llm.call_json("SYS", "USER", Path, thinking={"type": "adaptive"})

    assert "thinking" not in api.chat.completions.calls[0]


def test_model_and_max_tokens_are_forwarded(fake_api):
    api = fake_api(FakeCompletion(_path().model_dump_json()))

    llm.call_json("SYS", "USER", Path, model="minimaxai/minimax-m2", max_tokens=4321)

    call = api.chat.completions.calls[0]
    assert call["model"] == "minimaxai/minimax-m2"
    assert call["max_tokens"] == 4321


def test_temperature_is_omitted_unless_configured(fake_api, monkeypatch):
    api = fake_api(FakeCompletion(_path().model_dump_json()))
    llm.call_json("SYS", "USER", Path)
    assert "temperature" not in api.chat.completions.calls[0]

    monkeypatch.setenv("LLM_TEMPERATURE", "1.0")
    api2 = fake_api(FakeCompletion(_path().model_dump_json()))
    llm.call_json("SYS", "USER", Path)
    assert api2.chat.completions.calls[0]["temperature"] == 1.0


# --------------------------------------------------- response_format negotiation


def test_prefers_json_schema_when_the_endpoint_accepts_it(fake_api):
    api = fake_api(FakeCompletion(_path().model_dump_json()))

    llm.call_json("SYS", "USER", Path)

    assert api.chat.completions.calls[0]["response_format"]["type"] == "json_schema"


def test_downgrades_to_json_object_when_json_schema_is_rejected(fake_api):
    api = fake_api(_bad_request(), FakeCompletion(_path().model_dump_json()))

    got = llm.call_json("SYS", "USER", Path)

    assert isinstance(got, Path)
    assert [c["response_format"]["type"] for c in api.chat.completions.calls] == [
        "json_schema",
        "json_object",
    ]


def test_downgrades_all_the_way_to_no_response_format(fake_api):
    """Plenty of self-hosted endpoints implement neither JSON mode."""
    api = fake_api(_bad_request(), _bad_request(), FakeCompletion(_path().model_dump_json()))

    got = llm.call_json("SYS", "USER", Path)

    assert isinstance(got, Path)
    assert "response_format" not in api.chat.completions.calls[2]


def test_negotiated_mode_is_remembered_across_calls(fake_api):
    """Re-probing json_schema on every request would waste a round trip each time."""
    fake_api(_bad_request(), FakeCompletion(_path().model_dump_json()))
    llm.call_json("SYS", "USER", Path)

    api = fake_api(FakeCompletion(_path().model_dump_json()))
    llm.call_json("SYS", "USER", Path)

    assert api.chat.completions.calls[0]["response_format"]["type"] == "json_object"


def test_explicit_json_mode_is_not_negotiated(fake_api, monkeypatch):
    monkeypatch.setenv("LLM_JSON_MODE", "none")
    api = fake_api(FakeCompletion(_path().model_dump_json()))

    llm.call_json("SYS", "USER", Path)

    assert "response_format" not in api.chat.completions.calls[0]


def test_every_mode_rejected_raises_llmerror(fake_api):
    fake_api(_bad_request(), _bad_request(), _bad_request())

    with pytest.raises(llm.LLMError, match="rejected every JSON output mode"):
        llm.call_json("SYS", "USER", Path)


def test_unsupported_json_mode_setting_is_rejected(fake_api, monkeypatch):
    monkeypatch.setenv("LLM_JSON_MODE", "yolo")
    fake_api(FakeCompletion(_path().model_dump_json()))

    with pytest.raises(llm.LLMError, match="not supported"):
        llm.call_json("SYS", "USER", Path)


# --------------------------------------------------------------------------- retry


def test_invalid_json_retries_once_then_succeeds(fake_api):
    expected = _path()
    api = fake_api(
        FakeCompletion("this is not json"),
        FakeCompletion(expected.model_dump_json()),
    )

    got = llm.call_json("SYS", "THE ORIGINAL PROMPT", Path)

    assert got == expected
    assert len(api.chat.completions.calls) == 2


def test_retry_feedback_reaches_the_model(fake_api):
    api = fake_api(FakeCompletion("not json"), FakeCompletion(_path().model_dump_json()))

    llm.call_json("SYS", "THE ORIGINAL PROMPT", Path)

    retry_user = api.chat.completions.calls[1]["messages"][1]["content"]
    assert "THE ORIGINAL PROMPT" in retry_user
    assert "did not validate" in retry_user


def test_retry_messages_are_merged_into_one_user_turn(fake_api):
    """Some OpenAI-compatible hosts reject consecutive same-role messages outright."""
    api = fake_api(FakeCompletion("not json"), FakeCompletion(_path().model_dump_json()))

    llm.call_json("SYS", "USER", Path)

    roles = [m["role"] for m in api.chat.completions.calls[1]["messages"]]
    assert roles == ["system", "user"]


def test_invalid_json_twice_raises_llmerror(fake_api):
    fake_api(FakeCompletion("not json"), FakeCompletion("still not json"))

    with pytest.raises(llm.LLMError, match="failed schema validation on both attempts"):
        llm.call_json("SYS", "USER", Path)


def test_schema_valid_json_that_breaks_a_rule_feeds_the_same_retry_channel(fake_api):
    """A 1-based answer_index validates fine but marks the wrong option correct."""
    bad = _path(answer_index=3)  # only 3 options, so 3 is out of range
    good = _path(answer_index=0)
    api = fake_api(
        FakeCompletion(bad.model_dump_json()),
        FakeCompletion(good.model_dump_json()),
    )

    def post_validate(path):
        return [
            f"lesson {lesson.order}: answer_index out of range"
            for lesson in path.lessons
            if not 0 <= lesson.exercise.answer_index < len(lesson.exercise.options)
        ]

    got = llm.call_json("SYS", "USER", Path, post_validate=post_validate)

    assert got == good
    assert len(api.chat.completions.calls) == 2


def test_wrong_lesson_count_is_reported_as_a_rule_break_not_a_schema_error(fake_api):
    fake_api(FakeCompletion(_path(4).model_dump_json()), FakeCompletion(_path(4).model_dump_json()))

    with pytest.raises(llm.LLMError, match="broke a stated rule on both attempts"):
        llm.call_json("SYS", "USER", Path, post_validate=lambda p: ["return exactly 5"])


# --------------------------------------------------------------------------- failures


def test_truncated_response_raises_with_a_max_tokens_hint(fake_api):
    fake_api(FakeCompletion('{"document_title": "cut off mid', finish_reason="length"))

    with pytest.raises(llm.LLMError, match="max_tokens"):
        llm.call_json("SYS", "USER", Path, max_tokens=99)


def test_content_filter_raises_rather_than_retrying(fake_api):
    api = fake_api(FakeCompletion("", finish_reason="content_filter"))

    with pytest.raises(llm.LLMError, match="refused"):
        llm.call_json("SYS", "USER", Path)
    assert len(api.chat.completions.calls) == 1


def test_transport_errors_are_wrapped_in_llmerror(fake_api):
    fake_api(_connection_error())

    with pytest.raises(llm.LLMError, match="LLM API call failed"):
        llm.call_json("SYS", "USER", Path)


def test_empty_choices_feeds_the_retry_channel(fake_api):
    fake_api(FakeCompletion(None), FakeCompletion(_path().model_dump_json()))

    assert isinstance(llm.call_json("SYS", "USER", Path), Path)


def test_missing_key_raises_a_clear_llmerror(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(llm.LLMError, match="NVIDIA_API_KEY"):
        llm.openai_client()


def test_nvidia_api_key_is_accepted_as_an_alias(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-not-real")

    assert llm.openai_client() is not None


# --------------------------------------------------------------------------- config


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")

    with pytest.raises(llm.LLMError, match="not supported"):
        llm.provider()


def test_openai_default_model_is_used_when_llm_model_is_unset(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert llm.active_model() == llm.DEFAULT_OPENAI_MODEL


def test_llm_model_overrides_the_provider_default(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
    assert llm.active_model() == "meta/llama-3.3-70b-instruct"


def test_generator_id_distinguishes_provider_and_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "minimaxai/minimax-m2")
    assert llm.generator_id() == "openai:minimaxai/minimax-m2"


def test_cache_key_changes_with_the_provider(monkeypatch):
    """Otherwise switching backends serves the other model's lessons off disk."""
    from app import cache
    from app.models import Document

    doc = Document(
        source_type="youtube",
        url="https://www.youtube.com/watch?v=5iTOphGnCtg",
        title="GENERAL CHEMISTRY explained in 19 Minutes",
        text="Everything is made of atoms.",
    )
    openai_key = cache.key_for(doc)

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    anthropic_key = cache.key_for(doc)

    assert openai_key != anthropic_key


def test_schema_instruction_is_deterministic():
    """A non-deterministic prompt would defeat prompt caching on endpoints that offer it."""

    class Tiny(BaseModel):
        a: int

    assert llm._schema_instruction(Tiny) == llm._schema_instruction(Tiny)
