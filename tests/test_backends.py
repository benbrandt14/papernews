"""Tests for the OpenAI-compatible multi-provider backend."""

import pytest

from papernews.config import Settings
from papernews.core.backends import (
    OpenAICompatBackend,
    get_backend,
)
from papernews.models import LLMArticleSelection


def _chat_response(mocker, content: str, prompt_tokens: int, completion_tokens: int):
    resp = mocker.MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return resp


def test_get_backend_defaults_to_deepseek(monkeypatch):
    monkeypatch.delenv("PAPERNEWS_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    backend = get_backend(Settings())
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == "https://api.deepseek.com/v1"
    assert backend.model == "deepseek-chat"
    assert backend.api_key == "sk-test"


def test_get_backend_switches_provider_by_env(monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    backend = get_backend(Settings())
    assert backend.base_url == "https://openrouter.ai/api/v1"
    assert backend.api_key == "or-key"


def test_get_backend_local_needs_no_key(monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "local")
    backend = get_backend(Settings())
    assert backend.base_url == "http://localhost:11434/v1"
    assert backend.api_key is None


def test_get_backend_explicit_overrides_win(monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("PAPERNEWS_LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("PAPERNEWS_LLM_API_KEY", "override")
    monkeypatch.setenv("PAPERNEWS_LLM_MODEL", "custom-model")
    backend = get_backend(Settings())
    assert backend.base_url == "https://example.test/v1"
    assert backend.api_key == "override"
    assert backend.model == "custom-model"


def test_get_backend_unknown_provider_without_base_url_raises(monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "nope")
    monkeypatch.delenv("PAPERNEWS_LLM_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_backend(Settings())


def test_structured_call_uses_json_mode_and_reports_usage(mocker):
    resp = _chat_response(mocker, '{"is_selected": true}', 7, 3)
    post = mocker.patch("papernews.core.backends.requests.post", return_value=resp)

    backend = OpenAICompatBackend("https://api.x/v1", "k", "m")
    text, telemetry = backend.structured("prompt", "system", 0.1, LLMArticleSelection)

    assert text == '{"is_selected": true}'
    assert telemetry.prompt_tokens == 7
    assert telemetry.output_tokens == 3

    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "m"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"
    # The schema is injected into the system prompt to steer JSON mode.
    assert "is_selected" in payload["messages"][0]["content"]
    headers = post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer k"
    assert post.call_args.args[0] == "https://api.x/v1/chat/completions"


def test_text_call_omits_json_mode_and_auth_when_no_key(mocker):
    resp = _chat_response(mocker, "plain text", 0, 0)
    post = mocker.patch("papernews.core.backends.requests.post", return_value=resp)

    backend = OpenAICompatBackend("http://local/v1", None, "m")
    text, telemetry = backend.text("prompt", "system", 0.0)

    assert text == "plain text"
    assert telemetry.prompt_tokens == 0
    payload = post.call_args.kwargs["json"]
    assert "response_format" not in payload
    assert "Authorization" not in post.call_args.kwargs["headers"]


def test_empty_choices_returns_none(mocker):
    resp = mocker.MagicMock()
    resp.json.return_value = {"choices": [], "usage": {}}
    mocker.patch("papernews.core.backends.requests.post", return_value=resp)

    backend = OpenAICompatBackend("http://h/v1", "k", "m")
    text, _telemetry = backend.text("p", "s", 0.0)
    assert text is None


def test_http_error_propagates(mocker):
    resp = mocker.MagicMock()
    resp.raise_for_status.side_effect = RuntimeError("503")
    mocker.patch("papernews.core.backends.requests.post", return_value=resp)

    backend = OpenAICompatBackend("http://h/v1", "k", "m")
    with pytest.raises(RuntimeError, match="503"):
        backend.text("p", "s", 0.0)
