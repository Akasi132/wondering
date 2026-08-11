"""Tests for app/llm.py and app/pathbuilder.py's prompt assembly and post-checks.

NOTHING here touches the real Anthropic API. `app.llm.client` is monkeypatched to
return a stub whose `.messages.parse(**kwargs)` is fully scripted, and
`app.pathbuilder.call_json` is monkeypatched wholesale, so the retry / validation /
error / post-check paths are exercised without a key and without network.
"""

import copy
import logging

import anthropic
import httpx
import pytest
from pydantic import ValidationError

from app import llm, pathbuilder
from app.models import Document, Exercise, Lesson, Path

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _exercise(*, options=None, answer_index=0) -> Exercise:
    return Exercise(
        question="What does A point to?",
        options=list(options) if options is not None else ["B", "C", "D"],
        answer_index=answer_index,
        why="The edge runs A to B.",
    )


def _lesson(order: int, *, exercise: Exercise | None = None, citation="00:00-01:30") -> Lesson:
    return Lesson(
        order=order,
        title=f"Lesson {order}",
        explanation="A short explanation of the idea.",
        mermaid="graph TD; A-->B;",
        exercise=exercise if exercise is not None else _exercise(),
        citation=citation,
    )


def _path(n: int = 5, *, title="A Real Document", url="https://example.com/a") -> Path:
    return Path(
        document_title=title,
        source_url=url,
        lessons=[_lesson(i) for i in range(1, n + 1)],
    )


def _fails_once(problems: list[str]):
    """A post_validate that reports `problems` on its first call and nothing afterwards.

    Models the case the retry channel exists for: one correctable mistake, then a fix.
    """
    state = {"calls": 0}

    def post_validate(parsed) -> list[str]:
        state["calls"] += 1
        return list(problems) if state["calls"] == 1 else []

    return post_validate


def _validation_error() -> ValidationError:
    """Build a genuine pydantic ValidationError by actually triggering one."""
    try:
        Path.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("Path.model_validate({}) unexpectedly succeeded")


def _httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _rate_limit_error() -> anthropic.RateLimitError:
    request = _httpx_request()
    return anthropic.RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )


def _auth_error() -> anthropic.AuthenticationError:
    request = _httpx_request()
    return anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx.Response(401, request=request), body=None
    )


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(message="connection reset", request=_httpx_request())


def _timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_httpx_request())


class FakeResponse:
    """Mimics the shape of anthropic's ParsedMessage that app.llm.call_json reads.

    Real ParsedMessage exposes `.parsed_output` (a property scanning content blocks),
    `.stop_reason` and `.stop_details` (which is None unless stop_reason == "refusal").
    """

    def __init__(self, *, parsed_output=None, stop_reason="end_turn", stop_details=None):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason
        self.stop_details = stop_details


class FakeMessages:
    def __init__(self, script):
        # script: list of FakeResponse | Exception, consumed one per parse() call
        self._script = list(script)
        self.calls = []  # deep-copied kwargs, one entry per parse() call

    def parse(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self._script:
            raise AssertionError(
                f"parse() called {len(self.calls)} times but only "
                f"{len(self.calls) - 1} outcomes were scripted"
            )
        outcome = self._script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


@pytest.fixture
def fake_api(monkeypatch):
    """Install a scripted fake client; returns the installer."""

    def install(*script) -> FakeClient:
        client = FakeClient(script)
        monkeypatch.setattr(llm, "client", lambda: client)
        return client

    return install


# ---------------------------------------------------------------------------
# call_json — happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_validated_model_with_exactly_one_parse_call(fake_api):
    expected = _path()
    api = fake_api(FakeResponse(parsed_output=expected, stop_reason="end_turn"))

    got = llm.call_json("SYS", "USER", Path)

    assert got is expected
    assert isinstance(got, Path)
    assert api.messages.call_count == 1, "happy path must not retry"


def test_happy_path_sends_expected_kwargs(fake_api):
    api = fake_api(FakeResponse(parsed_output=_path()))

    llm.call_json("SYS", "USER", Path, model="some-model", max_tokens=1234)

    (kwargs,) = api.messages.calls
    assert kwargs["model"] == "some-model"
    assert kwargs["max_tokens"] == 1234
    assert kwargs["system"] == "SYS"
    assert kwargs["output_format"] is Path
    assert kwargs["messages"] == [{"role": "user", "content": "USER"}]
    assert "thinking" not in kwargs, "thinking must be omitted when not requested"


def test_thinking_is_forwarded_when_provided(fake_api):
    api = fake_api(FakeResponse(parsed_output=_path()))

    llm.call_json("SYS", "USER", Path, thinking={"type": "adaptive"})

    assert api.messages.calls[0]["thinking"] == {"type": "adaptive"}


def test_defaults_use_pathbuilder_model(fake_api):
    api = fake_api(FakeResponse(parsed_output=_path()))

    llm.call_json("SYS", "USER", Path)

    assert api.messages.calls[0]["model"] == llm.MODEL_PATHBUILDER
    assert api.messages.calls[0]["max_tokens"] == 16_000


# ---------------------------------------------------------------------------
# call_json — retry on ValidationError
# ---------------------------------------------------------------------------


def test_validation_error_then_success_retries_once_with_error_feedback(fake_api):
    exc = _validation_error()
    expected = _path()
    api = fake_api(exc, FakeResponse(parsed_output=expected))

    got = llm.call_json("SYS", "THE ORIGINAL USER PROMPT", Path)

    assert got is expected
    assert api.messages.call_count == 2, "must retry exactly once"

    first, second = api.messages.calls
    assert first["messages"] == [{"role": "user", "content": "THE ORIGINAL USER PROMPT"}]

    # The retry must re-send the original prompt AND the validation feedback.
    assert len(second["messages"]) == 2
    assert second["messages"][0] == {"role": "user", "content": "THE ORIGINAL USER PROMPT"}
    feedback = second["messages"][1]
    assert feedback["role"] == "user"
    # Assert on intent, not on the exact wording, which is prompt copy and will churn.
    lowered = feedback["content"].lower()
    assert "did not validate" in lowered
    assert "schema" in lowered
    # ... and it must carry the actual pydantic error detail, not just boilerplate.
    assert "lessons" in feedback["content"]
    assert "document_title" in feedback["content"]

    # Everything else about the retry must be unchanged.
    for key in ("model", "max_tokens", "system", "output_format"):
        assert second[key] == first[key]


def test_retry_feedback_tells_the_model_to_emit_only_complete_json(fake_api):
    """New in the fix: truncation is a top cause of invalid JSON, so the retry says so."""
    api = fake_api(_validation_error(), FakeResponse(parsed_output=_path()))

    llm.call_json("SYS", "USER", Path)

    feedback = api.messages.calls[1]["messages"][1]["content"].lower()
    assert "only the json object" in feedback
    assert "complete" in feedback
    assert "closed" in feedback


def test_validation_error_twice_raises_llmerror_mentioning_schema_validation(fake_api):
    api = fake_api(_validation_error(), _validation_error())

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path)

    assert api.messages.call_count == 2, "must not retry more than once"
    message = str(excinfo.value)
    assert "schema validation" in message
    assert "both attempts" in message
    assert "lessons" in message, "last pydantic error should be surfaced"


# ---------------------------------------------------------------------------
# call_json — post_validate feeds the same retry channel as schema errors
# ---------------------------------------------------------------------------


def test_post_validate_returning_empty_passes_the_object_straight_through(fake_api):
    expected = _path()
    api = fake_api(FakeResponse(parsed_output=expected))
    seen = []

    got = llm.call_json(
        "SYS", "USER", Path, post_validate=lambda parsed: seen.append(parsed) or []
    )

    assert got is expected
    assert seen == [expected], "post_validate must see the parsed object"
    assert api.messages.call_count == 1, "a clean post-validation must not retry"


def test_post_validate_problems_retry_and_the_corrected_object_is_returned(fake_api):
    bad, good = _path(4), _path(5)
    api = fake_api(FakeResponse(parsed_output=bad), FakeResponse(parsed_output=good))
    seen = []

    def post_validate(parsed: Path) -> list[str]:
        seen.append(len(parsed.lessons))
        return ["there are too few lessons"] if len(parsed.lessons) != 5 else []

    got = llm.call_json("SYS", "USER", Path, post_validate=post_validate)

    assert got is good
    assert seen == [4, 5], "post_validate runs on every attempt, including the retry"
    assert api.messages.call_count == 2

    feedback = api.messages.calls[1]["messages"][1]["content"]
    assert "there are too few lessons" in feedback, "the problem must reach the model"
    assert "broke one of the stated rules" in feedback


def test_post_validate_joins_multiple_problems_into_one_feedback_block(fake_api):
    api = fake_api(FakeResponse(parsed_output=_path(4)), FakeResponse(parsed_output=_path()))

    llm.call_json(
        "SYS", "USER", Path, post_validate=_fails_once(["problem one", "problem two"])
    )

    feedback = api.messages.calls[1]["messages"][1]["content"]
    assert "problem one; problem two" in feedback
    assert "Problems to fix:" in feedback


def test_post_validate_failing_twice_raises_llmerror_carrying_the_problems(fake_api):
    api = fake_api(FakeResponse(parsed_output=_path(4)), FakeResponse(parsed_output=_path(4)))

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path, post_validate=lambda p: ["still only 4 lessons"])

    assert api.messages.call_count == 2, "post-validation gets exactly one retry"
    assert "still only 4 lessons" in str(excinfo.value)


def test_post_validate_is_not_called_when_the_schema_failed_to_parse(fake_api):
    """A ValidationError means there is no object to post-validate; calling it anyway
    would explode on the None/absent parse."""
    seen = []
    api = fake_api(_validation_error(), FakeResponse(parsed_output=_path()))

    llm.call_json("SYS", "USER", Path, post_validate=lambda parsed: seen.append(parsed) or [])

    assert api.messages.call_count == 2
    assert len(seen) == 1, "only the attempt that actually parsed may be post-validated"
    assert isinstance(seen[0], Path)


def test_post_validate_is_not_called_when_parsed_output_is_none(fake_api):
    seen = []
    fake_api(
        FakeResponse(parsed_output=None, stop_reason="end_turn"),
        FakeResponse(parsed_output=_path()),
    )

    llm.call_json("SYS", "USER", Path, post_validate=lambda parsed: seen.append(parsed) or [])

    assert len(seen) == 1
    assert seen[0] is not None


def test_post_validate_failure_is_logged_as_a_warning(fake_api, caplog):
    fake_api(FakeResponse(parsed_output=_path(4)), FakeResponse(parsed_output=_path()))

    with caplog.at_level(logging.WARNING, logger="app.llm"):
        llm.call_json("SYS", "USER", Path, post_validate=_fails_once(["a structural problem"]))

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("post-validation failed" in w and "a structural problem" in w for w in warnings)


# ---------------------------------------------------------------------------
# call_json — stop_reason failure modes
# ---------------------------------------------------------------------------


def test_max_tokens_raises_llmerror_and_never_returns_truncated_object(fake_api):
    truncated = _path(3)  # a shape-valid but incomplete object
    api = fake_api(FakeResponse(parsed_output=truncated, stop_reason="max_tokens"))

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path, max_tokens=99)

    assert "max_tokens" in str(excinfo.value)
    assert "99" in str(excinfo.value)
    assert api.messages.call_count == 1, "max_tokens is fatal, not retryable"


def test_refusal_raises_llmerror(fake_api):
    api = fake_api(FakeResponse(parsed_output=None, stop_reason="refusal", stop_details=None))

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path)

    assert "refused" in str(excinfo.value).lower()
    assert api.messages.call_count == 1


def test_refusal_includes_stop_details(fake_api):
    fake_api(
        FakeResponse(
            parsed_output=None,
            stop_reason="refusal",
            stop_details="category=cyber",
        )
    )

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path)

    assert "category=cyber" in str(excinfo.value)


# ---------------------------------------------------------------------------
# call_json — parsed_output is None
# ---------------------------------------------------------------------------


def test_parsed_output_none_twice_raises_instead_of_returning_none(fake_api):
    api = fake_api(
        FakeResponse(parsed_output=None, stop_reason="end_turn"),
        FakeResponse(parsed_output=None, stop_reason="end_turn"),
    )

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path)

    assert api.messages.call_count == 2, "an empty parse should be retried once"
    assert "no parseable structured output" in str(excinfo.value)


def test_parsed_output_none_then_success_recovers(fake_api):
    expected = _path()
    api = fake_api(
        FakeResponse(parsed_output=None, stop_reason="end_turn"),
        FakeResponse(parsed_output=expected, stop_reason="end_turn"),
    )

    got = llm.call_json("SYS", "USER", Path)

    assert got is expected
    assert api.messages.call_count == 2
    assert "did not validate" in api.messages.calls[1]["messages"][1]["content"]


# ---------------------------------------------------------------------------
# call_json — transport / status errors are wrapped as LLMError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_error, expected_name",
    [
        (_rate_limit_error, "RateLimitError"),
        (_auth_error, "AuthenticationError"),
        (_connection_error, "APIConnectionError"),
        (_timeout_error, "APITimeoutError"),
    ],
    ids=["rate_limit", "auth", "connection", "timeout"],
)
def test_transport_errors_are_wrapped_as_llmerror(fake_api, make_error, expected_name):
    """anthropic.APIError is the common base of APIStatusError and APIConnectionError,
    so every transport/status failure reaches callers as LLMError, never raw."""
    error = make_error()
    assert isinstance(error, anthropic.APIError), "guard: SDK hierarchy assumption"
    api = fake_api(error)

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path)

    message = str(excinfo.value)
    assert "Anthropic API call failed" in message
    assert expected_name in message, "the concrete SDK error class must be named"
    assert excinfo.value.__cause__ is error, "the original error must be chained"
    assert api.messages.call_count == 1, "transport errors are not retried"


def test_transport_error_is_not_retried_even_on_the_first_attempt(fake_api):
    """Only one outcome is scripted; a retry would trip FakeMessages' own assertion."""
    api = fake_api(_connection_error())

    with pytest.raises(llm.LLMError):
        llm.call_json("SYS", "USER", Path)

    assert api.messages.call_count == 1


def test_transport_error_on_the_retry_attempt_also_surfaces_as_llmerror(fake_api):
    api = fake_api(_validation_error(), _rate_limit_error())

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path)

    assert api.messages.call_count == 2
    assert "RateLimitError" in str(excinfo.value)


def test_non_anthropic_exceptions_still_propagate_raw(fake_api):
    """The wrapper is scoped to anthropic.APIError; a programming bug must not be
    disguised as an LLMError."""
    boom = KeyError("bug in our own kwargs assembly")
    fake_api(boom)

    with pytest.raises(KeyError):
        llm.call_json("SYS", "USER", Path)


# ---------------------------------------------------------------------------
# call_json — truncation diagnosis
# ---------------------------------------------------------------------------


def test_truncated_json_double_failure_names_max_tokens_as_the_likely_cause(fake_api):
    """messages.parse validates inside the SDK call, so a response cut off at max_tokens
    lands in the ValidationError branch and never reaches the stop_reason check. The
    final error now explains that, instead of only blaming the schema."""
    api = fake_api(_validation_error(), _validation_error())

    with pytest.raises(llm.LLMError) as excinfo:
        llm.call_json("SYS", "USER", Path, max_tokens=200)

    message = str(excinfo.value)
    assert api.messages.call_count == 2
    assert "schema validation" in message
    # The actionable diagnosis is now present.
    assert "max_tokens=200" in message
    assert "truncated" in message
    assert "invalid JSON" in message


# ---------------------------------------------------------------------------
# truncate_source
# ---------------------------------------------------------------------------


def test_truncate_source_returns_input_unchanged_when_under_budget():
    text = "x" * 100
    assert llm.truncate_source(text, budget=100) is text
    assert llm.truncate_source("short", budget=llm.MAX_SOURCE_CHARS) == "short"
    assert llm.truncate_source("", budget=10) == ""


def test_truncate_source_keeps_prefix_and_adds_marker_when_over_budget():
    text = "abcdefghij" * 20  # 200 chars
    out = llm.truncate_source(text, budget=50)

    assert out.startswith(text[:50])
    assert "[SOURCE TRUNCATED AT 50 CHARACTERS]" in out
    assert text[50:] not in out
    # nothing of the source beyond the budget survives
    assert out[:50] == text[:50]


def test_truncate_source_uses_default_budget():
    text = "y" * (llm.MAX_SOURCE_CHARS + 500)
    out = llm.truncate_source(text)

    assert out.startswith("y" * llm.MAX_SOURCE_CHARS)
    assert f"[SOURCE TRUNCATED AT {llm.MAX_SOURCE_CHARS} CHARACTERS]" in out


# ---------------------------------------------------------------------------
# client()
# ---------------------------------------------------------------------------


def test_client_raises_llmerror_with_helpful_message_when_key_missing(monkeypatch):
    # load_dotenv() ran at import time and may have populated the env, so delete
    # unconditionally with raising=False.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(llm.LLMError) as excinfo:
        llm.client()

    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert ".env" in message


def test_client_raises_llmerror_when_key_is_blank(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(llm.LLMError):
        llm.client()


def test_client_constructs_a_client_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    api = llm.client()

    # Constructing the client makes no network call.
    assert hasattr(api, "messages")
    assert hasattr(api.messages, "parse")


def test_model_ids_are_recorded_but_flagged_as_unverified():
    """Directive 3: the IDs came from PLAN.md, not from the live /v1/models list. The
    module must not claim otherwise, since `--verify-models` has never been run."""
    import inspect

    assert llm.MODEL_PATHBUILDER == "claude-sonnet-5"
    assert llm.MODEL_CHEAP == "claude-haiku-4-5"

    source = inspect.getsource(llm)
    assert "NOT YET VERIFIED" in source
    assert "--verify-models" in source


# ---------------------------------------------------------------------------
# pathbuilder — doc / path fixtures
# ---------------------------------------------------------------------------

YOUTUBE_TEXT = "[00:00] The transformer reads the whole sequence at once. [04:12] Attention follows."
ARTICLE_TEXT = (
    "Attention weights are computed per token pair. The decoder then blends those weighted "
    "values into a single representation before the feed-forward layer runs."
)


def _doc(text: str = YOUTUBE_TEXT) -> Document:
    return Document(
        source_type="youtube",
        url="https://youtu.be/abc123",
        title="Transformers Explained",
        text=text,
        anchors=["00:00", "04:12"],
    )


def _article_doc(text: str = ARTICLE_TEXT) -> Document:
    return Document(
        source_type="article",
        url="https://example.com/attention",
        title="How Attention Works",
        text=text,
        anchors=[],
    )


def _path_with_citations(
    *citations: str,
    pad: str = "[00:00]",
    title: str = "whatever",
    url: str = "https://whatever.example",
) -> Path:
    """A Path whose lesson N carries citations[N-1].

    The Path schema demands 3..6 lessons, so shorter cases are padded out with `pad`,
    which must be a citation that verifies against the doc under test (so padding never
    contributes to the problem list).

    `title`/`url` matter only to tests that watch build_path's log output: build_path warns
    when the model's provenance disagrees with the document's, so a test asserting silence
    has to hand back matching values.
    """
    filled = list(citations)
    while len(filled) < 3:
        filled.append(pad)
    return Path(
        document_title=title,
        source_url=url,
        lessons=[_lesson(i, citation=c) for i, c in enumerate(filled, start=1)],
    )


def _path_with_bad_index(
    *,
    order: int = 3,
    options: list[str] | None = None,
    answer_index: int = 3,
    title: str = "t",
    url: str = "u",
) -> Path:
    """A five-lesson path, schema-valid, whose lesson `order` has an out-of-range answer."""
    lessons = [_lesson(i) for i in range(1, 6)]
    lessons[order - 1] = _lesson(
        order,
        exercise=_exercise(
            options=options if options is not None else ["A", "B", "C"],
            answer_index=answer_index,
        ),
    )
    return Path(document_title=title, source_url=url, lessons=lessons)


@pytest.fixture
def stub_call_json(monkeypatch):
    """Replace app.pathbuilder.call_json with one that returns `returns` and records args."""

    def install(returns: Path) -> dict:
        captured: dict = {"returns": returns, "call_count": 0}

        def fake_call_json(system, user, schema_model, **kwargs):
            captured["call_count"] += 1
            captured["system"] = system
            captured["user"] = user
            captured["schema_model"] = schema_model
            captured["kwargs"] = kwargs
            return returns

        monkeypatch.setattr(pathbuilder, "call_json", fake_call_json)
        return captured

    return install


@pytest.fixture
def captured_call(stub_call_json):
    return stub_call_json(_path())


# ---------------------------------------------------------------------------
# pathbuilder.build_path — prompt assembly, no model call
# ---------------------------------------------------------------------------


def test_build_path_returns_call_json_result(captured_call):
    result = pathbuilder.build_path(_doc())

    assert result is captured_call["returns"]


def test_build_path_system_prompt_forbids_outside_facts(captured_call):
    pathbuilder.build_path(_doc())
    system = captured_call["system"]

    assert "Use ONLY the provided source text" in system
    assert "Do not add facts" in system
    assert "not in it" in system
    assert "leave it out" in system
    # the schema-shaping rules the model must honour
    assert "exactly 5 lessons" in system


def test_build_path_system_prompt_specifies_zero_based_answer_index(captured_call):
    pathbuilder.build_path(_doc())
    system = captured_call["system"]

    assert "0-BASED" in system
    assert "Never 1-based" in system
    # spelled out concretely, because "0-based" alone is what models get wrong
    assert "0, 1, or 2" in system


def test_build_path_system_prompt_splits_the_citation_rule_by_source_kind(captured_call):
    pathbuilder.build_path(_doc())
    system = captured_call["system"]

    assert "[MM:SS]" in system
    assert "copy ONE of those" in system
    assert "Never invent a timestamp" in system
    assert "verbatim phrase" in system


def test_build_path_user_prompt_contains_document_text_title_and_url(captured_call):
    doc = _doc(text="Attention weights are computed per token pair.")
    pathbuilder.build_path(doc)
    user = captured_call["user"]

    assert doc.title in user
    assert doc.url in user
    assert doc.text in user
    assert doc.source_type in user
    assert "<source_text>" in user and "</source_text>" in user
    assert "from this source text only" in user
    # the source text must sit inside the delimiters
    body = user.split("<source_text>")[1].split("</source_text>")[0]
    assert doc.text in body


def test_build_path_user_prompt_announces_timestamps_when_the_doc_has_anchors(captured_call):
    pathbuilder.build_path(_doc())
    user = captured_call["user"]

    assert "contains timestamp markers" in user
    assert "no timestamps" not in user


def test_build_path_user_prompt_announces_absence_of_timestamps_for_articles(stub_call_json):
    captured = stub_call_json(_path_with_citations(*["Attention weights are computed"] * 5))
    pathbuilder.build_path(_article_doc())
    user = captured["user"]

    assert "no timestamps" in user
    assert "verbatim phrase" in user
    assert "contains timestamp markers" not in user


def test_build_path_defers_model_choice_to_the_provider_layer(captured_call):
    """build_path passes no model, so app.llm can pick the right one for the provider.

    This changed when the OpenAI-compatible backend landed: pinning MODEL_PATHBUILDER here
    sent a Claude model ID to whatever endpoint was configured. The resolution moved into
    llm.active_model(), asserted below.
    """
    pathbuilder.build_path(_doc())

    assert captured_call["schema_model"] is Path
    kwargs = captured_call["kwargs"]
    assert kwargs["model"] is None
    assert kwargs["max_tokens"] == 16_000
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_default_model_under_the_anthropic_provider_is_unchanged():
    """The deferral above must not quietly change which Claude model actually runs."""
    assert llm.active_model() == llm.MODEL_PATHBUILDER


def test_build_path_honours_model_override(captured_call):
    pathbuilder.build_path(_doc(), model="claude-haiku-4-5")

    assert captured_call["kwargs"]["model"] == "claude-haiku-4-5"


def test_build_path_truncates_oversized_source_text(captured_call):
    long_text = "z" * (llm.MAX_SOURCE_CHARS + 5_000)
    pathbuilder.build_path(_doc(text=long_text))
    user = captured_call["user"]

    assert f"[SOURCE TRUNCATED AT {llm.MAX_SOURCE_CHARS} CHARACTERS]" in user
    assert long_text not in user
    assert "z" * llm.MAX_SOURCE_CHARS in user


# ---------------------------------------------------------------------------
# pathbuilder.build_path — lesson-count enforcement
# ---------------------------------------------------------------------------


def test_build_path_accepts_exactly_five_lessons(stub_call_json):
    doc = _doc()
    captured = stub_call_json(_path(5, title=doc.title, url=doc.url))

    result = pathbuilder.build_path(doc)

    assert len(result.lessons) == pathbuilder.EXPECTED_LESSONS == 5
    assert captured["call_count"] == 1


def _strip_option_order(dumped: dict) -> dict:
    """Drop the two fields balance_answer_positions is allowed to change.

    build_path shuffles each exercise's options to break the model's answer-position bias, so
    an exact dump comparison no longer holds. Everything else must still be untouched.
    """
    for lesson in dumped["lessons"]:
        lesson["exercise"]["options"] = sorted(lesson["exercise"]["options"])
        lesson["exercise"].pop("answer_index")
    return dumped


def test_build_path_rewrites_nothing_except_option_order(stub_call_json):
    doc = _doc()
    returned = _path(5, title=doc.title, url=doc.url)
    before = returned.model_dump()
    correct_before = [
        lesson.exercise.options[lesson.exercise.answer_index] for lesson in returned.lessons
    ]
    stub_call_json(returned)

    result = pathbuilder.build_path(doc)

    assert _strip_option_order(result.model_dump()) == _strip_option_order(before), (
        "an already-correct path must not be rewritten beyond the option shuffle"
    )
    assert [
        lesson.exercise.options[lesson.exercise.answer_index] for lesson in result.lessons
    ] == correct_before, "the shuffle must not change which option is correct"


# The tests below drive the REAL call_json (only the SDK client is faked), because the
# whole point of routing structural problems through post_validate is the retry, and a
# stubbed call_json cannot show it.


@pytest.mark.parametrize("n", [3, 4, 6], ids=["three", "four", "six"])
def test_build_path_retries_a_wrong_lesson_count_and_returns_the_correction(fake_api, n):
    """The Pydantic bounds are 3..6 and the SDK strips minItems/maxItems from the JSON
    schema, so "exactly 5" is only enforced here — as a correction, not a hard failure."""
    doc = _doc()
    good = _path(5, title=doc.title, url=doc.url)
    api = fake_api(
        FakeResponse(parsed_output=_path(n, title=doc.title, url=doc.url)),
        FakeResponse(parsed_output=good),
    )

    result = pathbuilder.build_path(doc)

    assert result is good
    assert len(result.lessons) == pathbuilder.EXPECTED_LESSONS
    assert api.messages.call_count == 2, "a wrong count must cost exactly one retry"


def test_build_path_retry_prompt_tells_the_model_the_required_lesson_count(fake_api):
    doc = _doc()
    api = fake_api(
        FakeResponse(parsed_output=_path(4, title=doc.title, url=doc.url)),
        FakeResponse(parsed_output=_path(5, title=doc.title, url=doc.url)),
    )

    pathbuilder.build_path(doc)

    feedback = api.messages.calls[1]["messages"][1]["content"]
    assert "You returned 4 lessons" in feedback
    assert "return exactly 5" in feedback
    # the original prompt is re-sent alongside the correction
    assert api.messages.calls[1]["messages"][0] == api.messages.calls[0]["messages"][0]


@pytest.mark.parametrize("n", [3, 4, 6], ids=["three", "four", "six"])
def test_build_path_raises_when_the_lesson_count_is_wrong_twice(fake_api, n):
    api = fake_api(FakeResponse(parsed_output=_path(n)), FakeResponse(parsed_output=_path(n)))

    with pytest.raises(llm.LLMError) as excinfo:
        pathbuilder.build_path(_doc())

    assert api.messages.call_count == 2, "one retry, then give up — never a third call"
    message = str(excinfo.value)
    assert f"You returned {n} lessons" in message
    assert "return exactly 5" in message


def test_build_path_never_rewrites_provenance_on_a_path_it_rejects(fake_api):
    """A double failure raises out of call_json, so the model's object is handed back
    untouched rather than half-fixed."""
    returned = _path(4, title="Hallucinated Title", url="https://invented.example/x")
    fake_api(
        FakeResponse(parsed_output=returned),
        FakeResponse(parsed_output=returned),
    )

    with pytest.raises(llm.LLMError):
        pathbuilder.build_path(_doc())

    assert returned.document_title == "Hallucinated Title"
    assert returned.source_url == "https://invented.example/x"


# ---------------------------------------------------------------------------
# pathbuilder.check_answer_indexes
# ---------------------------------------------------------------------------


def test_check_answer_indexes_returns_empty_for_a_valid_path():
    assert pathbuilder.check_answer_indexes(_path()) == []


@pytest.mark.parametrize(
    "options, answer_index",
    [
        (["A", "B", "C"], 3),  # the realistic 1-based-off-by-one
        (["A", "B", "C", "D"], 4),
        (["A", "B", "C"], -1),
        (["A", "B", "C"], 99),
    ],
    ids=["one_based_three_options", "one_based_four_options", "negative", "way_out"],
)
def test_check_answer_indexes_flags_out_of_range_values(options, answer_index):
    bad = Path(
        document_title="t",
        source_url="u",
        lessons=[
            _lesson(1),
            _lesson(2, exercise=_exercise(options=options, answer_index=answer_index)),
            _lesson(3),
        ],
    )

    problems = pathbuilder.check_answer_indexes(bad)

    assert len(problems) == 1
    (problem,) = problems
    assert "lesson 2" in problem
    assert f"answer_index={answer_index}" in problem
    assert f"{len(options)} options" in problem
    assert f"valid 0..{len(options) - 1}" in problem


def test_check_answer_indexes_returns_one_entry_per_bad_lesson():
    bad = Path(
        document_title="t",
        source_url="u",
        lessons=[
            _lesson(1, exercise=_exercise(options=["A", "B", "C"], answer_index=3)),
            _lesson(2),
            _lesson(3, exercise=_exercise(options=["A", "B", "C", "D"], answer_index=9)),
        ],
    )

    problems = pathbuilder.check_answer_indexes(bad)

    assert len(problems) == 2
    assert "lesson 1" in problems[0]
    assert "lesson 3" in problems[1]


def test_check_answer_indexes_accepts_the_last_valid_index():
    ok = Path(
        document_title="t",
        source_url="u",
        lessons=[
            _lesson(1, exercise=_exercise(options=["A", "B", "C"], answer_index=2)),
            _lesson(2, exercise=_exercise(options=["A", "B", "C", "D"], answer_index=3)),
            _lesson(3, exercise=_exercise(options=["A", "B", "C"], answer_index=0)),
        ],
    )

    assert pathbuilder.check_answer_indexes(ok) == []


# ---------------------------------------------------------------------------
# pathbuilder.structural_problems
# ---------------------------------------------------------------------------


def test_structural_problems_empty_for_a_correct_path():
    assert pathbuilder.structural_problems(_path(5)) == []


@pytest.mark.parametrize("n", [3, 4, 6], ids=["three", "four", "six"])
def test_structural_problems_flags_a_wrong_lesson_count_as_an_instruction(n):
    problems = pathbuilder.structural_problems(_path(n))

    assert len(problems) == 1
    (problem,) = problems
    assert f"You returned {n} lessons" in problem
    assert f"return exactly {pathbuilder.EXPECTED_LESSONS}" in problem


def test_structural_problems_flags_a_bad_answer_index_with_the_zero_based_hint():
    problems = pathbuilder.structural_problems(_path_with_bad_index(order=3, answer_index=3))

    assert len(problems) == 1
    (problem,) = problems
    # the underlying diagnosis from check_answer_indexes ...
    assert "lesson 3" in problem
    assert "answer_index=3" in problem
    assert "valid 0..2" in problem
    # ... plus the correction that makes it actionable for the model
    assert "answer_index is 0-based" in problem
    assert "the first option is 0, not 1" in problem


def test_structural_problems_reports_count_and_index_problems_together():
    lessons = [_lesson(i) for i in range(1, 5)]  # four lessons
    lessons[3] = _lesson(4, exercise=_exercise(options=["A", "B", "C"], answer_index=3))
    bad = Path(document_title="t", source_url="u", lessons=lessons)

    problems = pathbuilder.structural_problems(bad)

    assert len(problems) == 2
    assert "You returned 4 lessons" in problems[0]
    assert "lesson 4" in problems[1]


def test_structural_problems_reports_one_entry_per_bad_index():
    lessons = [_lesson(i) for i in range(1, 6)]
    lessons[0] = _lesson(1, exercise=_exercise(options=["A", "B", "C"], answer_index=3))
    lessons[4] = _lesson(5, exercise=_exercise(options=["A", "B", "C", "D"], answer_index=9))

    problems = pathbuilder.structural_problems(
        Path(document_title="t", source_url="u", lessons=lessons)
    )

    assert len(problems) == 2
    assert "lesson 1" in problems[0]
    assert "lesson 5" in problems[1]


def test_structural_problems_is_what_build_path_hands_to_call_json(captured_call):
    """The retry semantics only hold if build_path actually wires this in."""
    pathbuilder.build_path(_doc())

    assert captured_call["kwargs"]["post_validate"] is pathbuilder.structural_problems


# ---------------------------------------------------------------------------
# pathbuilder.build_path — answer_index enforcement
# ---------------------------------------------------------------------------


def test_build_path_retries_the_one_based_answer_index_a_model_actually_emits(fake_api):
    """3 options, answer_index=3: schema-valid, and it would silently mark a wrong answer
    correct. Exactly the mistake a model fixes when told, so it feeds the retry."""
    doc = _doc()
    good = _path(5, title=doc.title, url=doc.url)
    api = fake_api(
        FakeResponse(parsed_output=_path_with_bad_index(order=3, title=doc.title, url=doc.url)),
        FakeResponse(parsed_output=good),
    )

    result = pathbuilder.build_path(doc)

    assert result is good
    assert pathbuilder.check_answer_indexes(result) == []
    assert api.messages.call_count == 2


def test_build_path_retry_prompt_spells_out_the_zero_based_rule(fake_api):
    doc = _doc()
    api = fake_api(
        FakeResponse(parsed_output=_path_with_bad_index(order=3, title=doc.title, url=doc.url)),
        FakeResponse(parsed_output=_path(5, title=doc.title, url=doc.url)),
    )

    pathbuilder.build_path(doc)

    feedback = api.messages.calls[1]["messages"][1]["content"]
    assert "lesson 3" in feedback
    assert "answer_index=3" in feedback
    assert "valid 0..2" in feedback
    # the correction the model needs, not just the complaint
    assert "answer_index is 0-based" in feedback
    assert "the first option is 0, not 1" in feedback


def test_build_path_raises_when_the_one_based_answer_index_survives_the_retry(fake_api):
    bad = _path_with_bad_index(order=3)
    api = fake_api(FakeResponse(parsed_output=bad), FakeResponse(parsed_output=bad))

    with pytest.raises(llm.LLMError) as excinfo:
        pathbuilder.build_path(_doc())

    assert api.messages.call_count == 2
    message = str(excinfo.value)
    assert "lesson 3" in message
    assert "answer_index=3" in message
    assert "0-based" in message


def test_build_path_retries_a_negative_answer_index(fake_api):
    doc = _doc()
    good = _path(5, title=doc.title, url=doc.url)
    api = fake_api(
        FakeResponse(
            parsed_output=_path_with_bad_index(
                order=1, answer_index=-1, title=doc.title, url=doc.url
            )
        ),
        FakeResponse(parsed_output=good),
    )

    result = pathbuilder.build_path(doc)

    assert result is good
    assert "answer_index=-1" in api.messages.calls[1]["messages"][1]["content"]
    assert api.messages.call_count == 2


def test_build_path_reports_every_bad_answer_index_in_one_retry(fake_api):
    lessons = [_lesson(i) for i in range(1, 6)]
    lessons[1] = _lesson(2, exercise=_exercise(options=["A", "B", "C"], answer_index=3))
    lessons[4] = _lesson(5, exercise=_exercise(options=["A", "B", "C"], answer_index=7))
    bad = Path(document_title="t", source_url="u", lessons=lessons)
    api = fake_api(FakeResponse(parsed_output=bad), FakeResponse(parsed_output=_path()))

    pathbuilder.build_path(_doc())

    feedback = api.messages.calls[1]["messages"][1]["content"]
    assert "lesson 2" in feedback and "lesson 5" in feedback, "one retry fixes both at once"


def test_build_path_reports_a_wrong_count_and_a_bad_index_together(fake_api):
    """Both structural problems reach the model in the same correction message, so a path
    that is wrong in two ways still only costs one retry."""
    lessons = [_lesson(i) for i in range(1, 5)]
    lessons[0] = _lesson(1, exercise=_exercise(options=["A", "B", "C"], answer_index=3))
    bad = Path(document_title="t", source_url="u", lessons=lessons)
    api = fake_api(FakeResponse(parsed_output=bad), FakeResponse(parsed_output=_path()))

    pathbuilder.build_path(_doc())

    feedback = api.messages.calls[1]["messages"][1]["content"]
    assert "You returned 4 lessons" in feedback
    assert "lesson 1" in feedback and "answer_index=3" in feedback
    assert api.messages.call_count == 2


# ---------------------------------------------------------------------------
# pathbuilder.build_path — provenance overwrite
# ---------------------------------------------------------------------------


def test_build_path_overwrites_hallucinated_title_and_url_with_the_real_doc(stub_call_json):
    doc = _doc()
    stub_call_json(
        _path(5, title="Attention Is All You Need", url="https://arxiv.org/abs/1706.03762")
    )

    result = pathbuilder.build_path(doc)

    assert result.document_title == doc.title == "Transformers Explained"
    assert result.source_url == doc.url == "https://youtu.be/abc123"


def test_build_path_overwrite_is_unconditional_even_when_the_model_echoed_correctly(
    stub_call_json,
):
    doc = _doc()
    stub_call_json(_path(5, title=doc.title, url=doc.url))

    result = pathbuilder.build_path(doc)

    assert result.document_title == doc.title
    assert result.source_url == doc.url


def test_build_path_overwrite_touches_only_provenance_fields(stub_call_json):
    doc = _doc()
    returned = _path(5, title="Wrong", url="https://wrong.example")
    before = returned.model_dump()
    correct_before = [
        lesson.exercise.options[lesson.exercise.answer_index] for lesson in returned.lessons
    ]
    stub_call_json(returned)

    result = pathbuilder.build_path(doc)

    # Provenance is corrected on the Path itself; the lessons keep their content, apart from
    # the deliberate option shuffle.
    assert _strip_option_order(result.model_dump())["lessons"] == _strip_option_order(before)["lessons"]
    assert [
        lesson.exercise.options[lesson.exercise.answer_index] for lesson in result.lessons
    ] == correct_before


# ---------------------------------------------------------------------------
# pathbuilder.unverifiable_citations
# ---------------------------------------------------------------------------


def test_unverifiable_citations_empty_for_timestamps_present_in_the_transcript():
    doc = _doc()

    path = _path_with_citations("[00:00]", "[04:12]", "00:00", "around [04:12] in the talk")

    assert pathbuilder.unverifiable_citations(doc, path) == []


def test_unverifiable_citations_flags_an_invented_timestamp():
    doc = _doc()  # anchors are 00:00 and 04:12
    path = _path_with_citations("[00:00]", "[12:47]")

    problems = pathbuilder.unverifiable_citations(doc, path)

    assert len(problems) == 1
    (problem,) = problems
    assert "lesson 2" in problem
    assert "12:47" in problem
    assert "no timestamp" in problem


def test_unverifiable_citations_flags_a_prose_citation_on_a_timestamped_source():
    """A transcript citation must carry a marker; a paraphrase is not checkable."""
    doc = _doc()
    path = _path_with_citations("the speaker explains attention early on")

    problems = pathbuilder.unverifiable_citations(doc, path)

    assert len(problems) == 1
    assert "no timestamp" in problems[0]


def test_unverifiable_citations_empty_for_a_real_verbatim_article_quote():
    doc = _article_doc()
    path = _path_with_citations(
        "Attention weights are computed per token pair",
        '"blends those weighted values into a single representation"',
        "THE DECODER THEN BLENDS THOSE WEIGHTED VALUES",  # case-insensitive match
    )

    assert pathbuilder.unverifiable_citations(doc, path) == []


def test_unverifiable_citations_flags_an_invented_article_quote():
    doc = _article_doc()
    path = _path_with_citations(
        "Attention weights are computed per token pair",
        "the model learns via telepathic gradient descent",
        pad="Attention weights are computed per token pair",
    )

    problems = pathbuilder.unverifiable_citations(doc, path)

    assert len(problems) == 1
    (problem,) = problems
    assert "lesson 2" in problem
    assert "telepathic" in problem
    assert "not a verbatim quote" in problem


def test_unverifiable_citations_reports_one_entry_per_bad_lesson():
    doc = _article_doc()
    path = _path_with_citations(
        "invented phrase number one here",
        "Attention weights are computed per token pair",
        "invented phrase number two here",
    )

    problems = pathbuilder.unverifiable_citations(doc, path)

    assert len(problems) == 2
    assert "lesson 1" in problems[0]
    assert "lesson 3" in problems[1]


def test_unverifiable_citations_flags_short_article_citations():
    """A citation too short to locate is unverifiable, not verified. Silently passing
    these was the hole: "para 9" pointed nowhere and nobody heard about it."""
    doc = _article_doc()
    path = _path_with_citations(
        "para 9",  # 6 chars, nowhere in the text
        pad="Attention weights are computed per token pair",
    )
    assert "para 9" not in doc.text

    problems = pathbuilder.unverifiable_citations(doc, path)

    assert len(problems) == 1
    (problem,) = problems
    assert "lesson 1" in problem
    assert "para 9" in problem
    assert "too short to verify" in problem


def test_unverifiable_citations_flags_short_citations_even_when_they_do_appear():
    """The rule is length, not presence: an 11-char probe is too weak to be evidence
    that the model actually read the source, even if the substring happens to match."""
    doc = _article_doc()
    path = _path_with_citations(
        "per token",  # 9 chars, and genuinely in the source text
        pad="Attention weights are computed per token pair",
    )
    assert "per token" in doc.text

    problems = pathbuilder.unverifiable_citations(doc, path)

    assert len(problems) == 1
    assert "too short to verify" in problems[0]


def test_unverifiable_citations_accepts_a_twelve_character_quote():
    """Guard on the boundary itself, so the threshold cannot drift unnoticed."""
    doc = _article_doc()
    probe = "decoder then"  # exactly 12 chars, present in the source
    assert len(probe) == 12 and probe in doc.text
    path = _path_with_citations(probe, pad="Attention weights are computed per token pair")

    assert pathbuilder.unverifiable_citations(doc, path) == []


def test_unverifiable_citations_short_rule_does_not_apply_to_timestamped_sources():
    """A [00:00] marker is 7 chars and perfectly checkable; the length rule is article-only."""
    doc = _doc()

    assert pathbuilder.unverifiable_citations(doc, _path_with_citations("[00:00]")) == []


# ---------------------------------------------------------------------------
# pathbuilder.build_path — unverifiable citations are warned, not fatal
# ---------------------------------------------------------------------------


def test_build_path_warns_but_still_returns_when_a_citation_is_unverifiable(
    stub_call_json, caplog
):
    doc = _doc()
    lessons = [_lesson(i) for i in range(1, 6)]
    lessons[3] = _lesson(4, citation="[99:99]")
    # provenance matches the doc, so the only warning under test is the citation one
    stub_call_json(Path(document_title=doc.title, source_url=doc.url, lessons=lessons))

    with caplog.at_level(logging.WARNING, logger="app.pathbuilder"):
        result = pathbuilder.build_path(doc)

    assert isinstance(result, Path), "a bad citation is reported, not raised"
    assert result.lessons[3].citation == "[99:99]", "the citation is left as-is"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Unverifiable citation" in w and "99:99" in w for w in warnings)
    assert any("lesson 4" in w for w in warnings)


def test_build_path_logs_nothing_when_every_citation_checks_out(stub_call_json, caplog):
    """Silence requires BOTH clean citations and provenance the model echoed correctly —
    a mismatch is now a warning in its own right, so the canned path has to match."""
    doc = _doc()
    stub_call_json(_path_with_citations(*["[04:12]"] * 5, title=doc.title, url=doc.url))

    with caplog.at_level(logging.WARNING, logger="app.pathbuilder"):
        pathbuilder.build_path(doc)

    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


# ---------------------------------------------------------------------------
# pathbuilder.build_path — provenance mismatch is a warning
# ---------------------------------------------------------------------------


def test_build_path_warns_when_the_model_returns_provenance_that_does_not_match(
    stub_call_json, caplog
):
    """A hallucinated title/url is evidence the model may be drawing on pretrained
    knowledge instead of the supplied source, which would taint the lesson bodies too.
    It is still overwritten, but no longer silently."""
    doc = _doc()
    stub_call_json(
        _path(5, title="Attention Is All You Need", url="https://arxiv.org/abs/1706.03762")
    )

    with caplog.at_level(logging.WARNING, logger="app.pathbuilder"):
        result = pathbuilder.build_path(doc)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, "one warning, not one per field"
    (warning,) = warnings
    assert "does not match the document" in warning
    # both what was returned and what was expected, or the log is not actionable
    assert "Attention Is All You Need" in warning
    assert "https://arxiv.org/abs/1706.03762" in warning
    assert doc.title in warning and doc.url in warning
    # and it is still corrected, not merely complained about
    assert result.document_title == doc.title
    assert result.source_url == doc.url


@pytest.mark.parametrize(
    "title, url",
    [
        ("Wrong Title", "https://youtu.be/abc123"),
        ("Transformers Explained", "https://wrong.example/x"),
    ],
    ids=["title_only", "url_only"],
)
def test_build_path_warns_when_either_provenance_field_alone_is_wrong(
    stub_call_json, caplog, title, url
):
    doc = _doc()
    stub_call_json(_path(5, title=title, url=url))

    with caplog.at_level(logging.WARNING, logger="app.pathbuilder"):
        pathbuilder.build_path(doc)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("does not match the document" in w for w in warnings)


def test_build_path_checks_citations_against_the_real_doc_not_the_returned_metadata(
    stub_call_json,
):
    """Provenance is overwritten before the citation check, so the check must still be
    driven by the extractor's doc (its anchors), not by anything the model returned."""
    doc = _article_doc()
    stub_call_json(_path_with_citations(*["Attention weights are computed per token pair"] * 5))

    result = pathbuilder.build_path(doc)

    assert result.source_url == doc.url
    assert pathbuilder.unverifiable_citations(doc, result) == []
