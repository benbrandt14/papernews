"""LLM backends: one OpenAI-compatible transport, many providers.

Every provider papernews talks to speaks the OpenAI ``/chat/completions`` API,
so there is a single backend. Providers are **presets** — a base URL, the env
var holding the key, and a default model. Switch with
``PAPERNEWS_LLM_PROVIDER`` (default ``deepseek``), or point
``PAPERNEWS_LLM_BASE_URL`` / ``_API_KEY`` / ``_MODEL`` at anything else that
speaks the same API. No provider SDK is imported — just ``requests``.

Structured output uses JSON mode (``response_format={"type": "json_object"}``)
plus the schema injected into the system prompt; the router validates the
returned JSON against the Pydantic model.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple, Protocol

import requests
from pydantic import BaseModel

from papernews.config import Settings
from papernews.models import Telemetry


class Provider(NamedTuple):
    base_url: str
    key_env: str | None  # None → no key required (e.g. local Ollama/vLLM)
    default_model: str


# Built-in presets. Any OpenAI-compatible endpoint also works by setting
# PAPERNEWS_LLM_BASE_URL / _API_KEY / _MODEL directly.
PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"
    ),
    "openrouter": Provider(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "deepseek/deepseek-chat"
    ),
    # Ollama and vLLM both expose an OpenAI-compatible server; no key needed.
    "local": Provider("http://localhost:11434/v1", None, "llama3.1"),
}


class LLMBackend(Protocol):
    def structured(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
        schema: type[BaseModel],
    ) -> tuple[str | None, Telemetry]:
        """One model call constrained to JSON; returns raw JSON text."""
        ...

    def text(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
    ) -> tuple[str | None, Telemetry]:
        """One free-form model call; returns raw text."""
        ...


class OpenAICompatBackend:
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _call(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
        json_mode: bool,
    ) -> tuple[str | None, Telemetry]:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": contents},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()

        choices = data.get("choices") or []
        text = choices[0]["message"]["content"] if choices else None
        usage = data.get("usage") or {}
        telemetry = Telemetry(
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
        )
        return (text or None), telemetry

    def structured(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
        schema: type[BaseModel],
    ) -> tuple[str | None, Telemetry]:
        # JSON mode needs the word "json" in the prompt and benefits from the
        # target shape; the router validates the result against `schema`.
        instruction = (
            f"{system_instruction}\n\n"
            "Respond with a single JSON object matching this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}\n"
            "Output only the JSON object, no prose or code fences."
        )
        return self._call(contents, instruction, temperature, json_mode=True)

    def text(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
    ) -> tuple[str | None, Telemetry]:
        return self._call(contents, system_instruction, temperature, json_mode=False)


def get_backend(settings: Settings) -> LLMBackend:
    """Resolve the configured provider preset into a backend.

    Explicit ``PAPERNEWS_LLM_BASE_URL`` / ``_API_KEY`` / ``_MODEL`` settings win
    over the preset, so any OpenAI-compatible endpoint works without a preset.
    """
    preset = PROVIDERS.get(settings.llm_provider)
    if preset is None and not settings.llm_base_url:
        raise ValueError(
            f"Unknown LLM provider {settings.llm_provider!r}. "
            f"Known: {sorted(PROVIDERS)}, or set PAPERNEWS_LLM_BASE_URL."
        )

    base_url = settings.llm_base_url or (preset.base_url if preset else "")
    model = settings.llm_model or (preset.default_model if preset else "")
    api_key = settings.llm_api_key
    if api_key is None and preset and preset.key_env:
        api_key = os.environ.get(preset.key_env)

    return OpenAICompatBackend(base_url=base_url, api_key=api_key, model=model)
