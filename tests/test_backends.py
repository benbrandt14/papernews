"""Tests for the OpenAI-compatible multi-provider backend."""

import pytest

from papernews.config import Settings
from papernews.core.backends import (
    OpenAICompatBackend,
    _extract_json,
    get_backend,
    resolve_provider,
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


# --- Robustness: fail-fast keys, JSON extraction, max_tokens, probe ----------


def test_get_backend_missing_key_raises(monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PAPERNEWS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("PAPERNEWS_LLM_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="needs an API key"):
        get_backend(Settings())


def test_resolve_custom_endpoint_needs_no_preset_key(monkeypatch):
    # An unknown provider name driven purely by a base_url override → "custom".
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "vllm")
    monkeypatch.setenv("PAPERNEWS_LLM_BASE_URL", "http://vllm.local/v1")
    monkeypatch.setenv("PAPERNEWS_LLM_MODEL", "my-model")
    resolved = resolve_provider(Settings())
    assert resolved.provider == "custom"
    assert resolved.base_url == "http://vllm.local/v1"
    assert resolved.api_key is None  # no key needed, and none demanded


def test_resolve_base_url_override_suppresses_preset_key(monkeypatch):
    # Pointing a known preset at a custom base_url means you're supplying your
    # own endpoint — the preset's key requirement no longer applies.
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("PAPERNEWS_LLM_API_KEY", raising=False)
    monkeypatch.setenv("PAPERNEWS_LLM_BASE_URL", "http://proxy.local/v1")
    resolved = resolve_provider(Settings())  # must not raise
    assert resolved.provider == "deepseek"
    assert resolved.base_url == "http://proxy.local/v1"
    assert resolved.api_key is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('Sure! Here you go: {"a": 1} — hope that helps', '{"a": 1}'),
        ('{"a": "text with } brace"}', '{"a": "text with } brace"}'),
        ("no json here", "no json here"),
    ],
)
def test_extract_json(raw, expected):
    assert _extract_json(raw) == expected


def test_structured_extracts_fenced_json(mocker):
    resp = _chat_response(mocker, '```json\n{"is_selected": true}\n```', 5, 2)
    mocker.patch("papernews.core.backends.requests.post", return_value=resp)
    backend = OpenAICompatBackend("http://h/v1", "k", "m")
    text, _t = backend.structured("p", "s", 0.1, LLMArticleSelection)
    assert text == '{"is_selected": true}'
    # And it round-trips through the model the router would validate against.
    assert LLMArticleSelection.model_validate_json(text).is_selected is True


def test_max_tokens_passed_through_when_set(mocker):
    resp = _chat_response(mocker, "ok", 1, 1)
    post = mocker.patch("papernews.core.backends.requests.post", return_value=resp)
    backend = OpenAICompatBackend("http://h/v1", "k", "m", max_tokens=256)
    backend.text("p", "s", 0.0)
    assert post.call_args.kwargs["json"]["max_tokens"] == 256


def test_check_reports_success(mocker):
    resp = _chat_response(mocker, "ok", 2, 1)
    mocker.patch("papernews.core.backends.requests.post", return_value=resp)
    backend = OpenAICompatBackend("http://h/v1", "k", "deepseek-chat")
    ok, detail = backend.check()
    assert ok is True
    assert "deepseek-chat" in detail


def test_check_swallows_errors(mocker):
    mocker.patch(
        "papernews.core.backends.requests.post",
        side_effect=ConnectionError("boom"),
    )
    backend = OpenAICompatBackend("http://h/v1", "k", "m")
    ok, detail = backend.check()
    assert ok is False
    assert "boom" in detail


def test_get_backend_applies_transport_settings(monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_PROVIDER", "local")
    monkeypatch.setenv("PAPERNEWS_LLM_TIMEOUT", "42")
    monkeypatch.setenv("PAPERNEWS_LLM_MAX_TOKENS", "128")
    backend = get_backend(Settings())
    assert backend.timeout == 42.0
    assert backend.max_tokens == 128
