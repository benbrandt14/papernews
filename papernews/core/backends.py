"""LLM backends: one OpenAI-compatible transport, many providers.

Every provider papernews talks to speaks the OpenAI ``/chat/completions`` API,
so there is a single backend. Providers are **presets** — a base URL, the env
var holding the key, and a default model. Switch with
``PAPERNEWS_LLM_PROVIDER`` (default ``deepseek``), or point
``PAPERNEWS_LLM_BASE_URL`` / ``_API_KEY`` / ``_MODEL`` at anything else that
speaks the same API. No provider SDK is imported — just ``requests``.

Structured output uses JSON mode (``response_format={"type": "json_object"}``)
plus the schema injected into the system prompt. The returned content is run
through ``_extract_json`` so a stray code fence or bit of prose from a less
disciplined model still yields valid JSON for the router to validate.
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
    "openai": Provider("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "groq": Provider(
        "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"
    ),
    "together": Provider(
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    "fireworks": Provider(
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
    ),
    # Ollama and vLLM both expose an OpenAI-compatible server; no key needed.
    "local": Provider("http://localhost:11434/v1", None, "llama3.1"),
}


def _extract_json(text: str) -> str:
    """Best-effort isolation of a JSON object from a model response.

    Tolerates a leading ```/```json fence and surrounding prose by returning
    the first balanced ``{...}`` object. If nothing balances, the input is
    returned unchanged so the caller's validation still runs (and fails) as it
    would have anyway — extraction only ever helps.
    """
    s = text.strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json) and any closing fence.
        s = s.split("\n", 1)[-1] if "\n" in s else ""
        if s.endswith("```"):
            s = s[: s.rfind("```")]
        s = s.strip()

    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s


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
        max_tokens: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

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
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

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
        text, telemetry = self._call(contents, instruction, temperature, json_mode=True)
        return (_extract_json(text) if text else None), telemetry

    def text(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
    ) -> tuple[str | None, Telemetry]:
        return self._call(contents, system_instruction, temperature, json_mode=False)

    def check(self) -> tuple[bool, str]:
        """One tiny liveness call. Returns (ok, human-readable detail).

        Never raises — a failed probe reports the error text so callers (CLI,
        healthz) can surface it without crashing.
        """
        try:
            text, telemetry = self._call(
                "ping",
                "Reply with the single word: ok",
                0.0,
                json_mode=False,
            )
            reply = (text or "").strip()[:40]
            return True, (
                f"{self.model} @ {self.base_url} replied "
                f"{reply!r} ({telemetry.total_tokens} tok)"
            )
        except Exception as e:  # noqa: BLE001 — probe result is data, not control flow
            return False, f"{type(e).__name__}: {e}"


class ResolvedProvider(NamedTuple):
    base_url: str
    api_key: str | None
    model: str
    provider: str  # preset name, or "custom" when driven by base_url override


def resolve_provider(settings: Settings) -> ResolvedProvider:
    """Turn settings into the concrete (base_url, api_key, model) to use.

    Explicit ``PAPERNEWS_LLM_BASE_URL`` / ``_API_KEY`` / ``_MODEL`` win over the
    preset, so any OpenAI-compatible endpoint works without a preset. Raises if
    the provider is unknown, or a preset that needs a key can't find one.
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

    # A preset that declares a key env but resolves to nothing is a
    # misconfiguration; fail loudly here instead of with a 401 mid-run.
    if api_key is None and not settings.llm_base_url and preset and preset.key_env:
        raise ValueError(
            f"Provider {settings.llm_provider!r} needs an API key in "
            f"${preset.key_env} (or set PAPERNEWS_LLM_API_KEY)."
        )

    name = settings.llm_provider if preset else "custom"
    return ResolvedProvider(
        base_url=base_url, api_key=api_key, model=model, provider=name
    )


def get_backend(settings: Settings) -> OpenAICompatBackend:
    """Build the backend for the configured provider."""
    r = resolve_provider(settings)
    return OpenAICompatBackend(
        base_url=r.base_url,
        api_key=r.api_key,
        model=r.model,
        timeout=settings.llm_timeout,
        max_tokens=(settings.llm_max_tokens or None),
    )
