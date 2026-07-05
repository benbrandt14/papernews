"""Tests for the LLM backend protocol (papernews/core/backends.py)."""

import pytest

from papernews.config import Settings
from papernews.core.backends import (
    GeminiBackend,
    OllamaBackend,
    get_backend,
)
from papernews.models import LLMArticleSelection


def test_get_backend_selects_gemini_by_default(monkeypatch):
    monkeypatch.delenv("PAPERNEWS_LLM_BACKEND", raising=False)
    backend = get_backend(Settings())
    assert isinstance(backend, GeminiBackend)
    assert backend.model == "gemini-2.5-flash"


def test_get_backend_selects_ollama_for_local(monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_BACKEND", "local")
    monkeypatch.setenv("OLLAMA_HOST", "http://myhost:1234/")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    backend = get_backend(Settings())
    assert isinstance(backend, OllamaBackend)
    assert backend.host == "http://myhost:1234"  # trailing slash stripped
    assert backend.model == "mistral"


def test_gemini_structured_call(mocker):
    response = mocker.MagicMock()
    response.text = '{"is_selected": true}'
    response.usage_metadata.prompt_token_count = 7
    response.usage_metadata.candidates_token_count = 3
    client = mocker.MagicMock()
    client.models.generate_content.return_value = response
    mocker.patch("papernews.core.backends._gemini_client", return_value=client)

    backend = GeminiBackend("gemini-2.5-flash")
    text, telemetry = backend.structured("prompt", "system", 0.1, LLMArticleSelection)

    assert text == '{"is_selected": true}'
    assert telemetry.prompt_tokens == 7
    assert telemetry.output_tokens == 3
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is LLMArticleSelection


def test_gemini_text_call_has_no_schema(mocker):
    response = mocker.MagicMock()
    response.text = "plain text"
    response.usage_metadata = None
    client = mocker.MagicMock()
    client.models.generate_content.return_value = response
    mocker.patch("papernews.core.backends._gemini_client", return_value=client)

    backend = GeminiBackend("gemini-2.5-flash")
    text, telemetry = backend.text("prompt", "system", 0.0)

    assert text == "plain text"
    assert telemetry.prompt_tokens == 0
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_schema is None


def test_ollama_structured_call(mocker):
    resp = mocker.MagicMock()
    resp.json.return_value = {
        "response": '{"is_selected": false}',
        "prompt_eval_count": 11,
        "eval_count": 5,
    }
    post = mocker.patch("papernews.core.backends.requests.post", return_value=resp)

    backend = OllamaBackend(host="http://localhost:11434", model="llama3.1")
    text, telemetry = backend.structured("prompt", "system", 0.2, LLMArticleSelection)

    assert text == '{"is_selected": false}'
    assert telemetry.prompt_tokens == 11
    assert telemetry.output_tokens == 5

    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "llama3.1"
    assert payload["system"] == "system"
    assert payload["stream"] is False
    assert payload["format"] == LLMArticleSelection.model_json_schema()
    assert post.call_args.args[0] == "http://localhost:11434/api/generate"


def test_ollama_empty_response_returns_none(mocker):
    resp = mocker.MagicMock()
    resp.json.return_value = {"response": "", "prompt_eval_count": 1, "eval_count": 0}
    mocker.patch("papernews.core.backends.requests.post", return_value=resp)

    backend = OllamaBackend(host="http://h", model="m")
    text, telemetry = backend.text("p", "s", 0.0)
    assert text is None
    assert telemetry.prompt_tokens == 1


def test_ollama_http_error_propagates(mocker):
    resp = mocker.MagicMock()
    resp.raise_for_status.side_effect = RuntimeError("503")
    mocker.patch("papernews.core.backends.requests.post", return_value=resp)

    backend = OllamaBackend(host="http://h", model="m")
    with pytest.raises(RuntimeError, match="503"):
        backend.text("p", "s", 0.0)
