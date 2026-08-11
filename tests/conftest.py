"""Pin the test environment.

`app.llm` reads LLM_PROVIDER / LLM_MODEL at call time, and `app.llm` calls load_dotenv() at
import, so a developer's real .env would otherwise leak into the suite: setting
LLM_PROVIDER=openai in .env would send every Anthropic-stubbed test down the OpenAI backend
and fail ~200 of them for no reason connected to the code under test.

These suites test the Anthropic backend specifically (they monkeypatch `llm.client` and script
`.messages.parse`), so the provider is pinned here rather than left to the ambient environment.
Tests that want the other backend can monkeypatch the env var themselves.
"""

import pytest


@pytest.fixture(autouse=True)
def _pinned_llm_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_JSON_MODE", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
