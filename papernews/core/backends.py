"""LLM backends behind one small protocol.

The router only ever sees `LLMBackend`: two methods, both returning
`(raw_text, Telemetry)`. Caching lives above the backend (in the
router), so backends stay dumb transport wrappers.

Backends:
  * GeminiBackend — google-genai, structured output via response_schema.
  * OllamaBackend — local models over the Ollama HTTP API, structured
    output via its JSON-schema `format` field. Selected with
    PAPERNEWS_LLM_BACKEND=local; host/model come from OLLAMA_HOST /
    OLLAMA_MODEL.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

from papernews.config import Settings
from papernews.models import Telemetry


class LLMBackend(Protocol):
    def structured(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
        schema: type[BaseModel],
    ) -> tuple[str | None, Telemetry]:
        """One model call constrained to `schema`; returns raw JSON text."""
        ...

    def text(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
    ) -> tuple[str | None, Telemetry]:
        """One free-form model call; returns raw text."""
        ...


@lru_cache(maxsize=1)
def _gemini_client() -> genai.Client:
    """Lazy client — importing this module must not require credentials."""
    return genai.Client()


def _gemini_telemetry(response: types.GenerateContentResponse) -> Telemetry:
    if response.usage_metadata:
        return Telemetry(
            prompt_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0,
        )
    return Telemetry()


class GeminiBackend:
    def __init__(self, model: str):
        self.model = model

    def _call(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
        schema: type[BaseModel] | None,
    ) -> tuple[str | None, Telemetry]:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = schema

        response = _gemini_client().models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or None, _gemini_telemetry(response)

    def structured(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
        schema: type[BaseModel],
    ) -> tuple[str | None, Telemetry]:
        return self._call(contents, system_instruction, temperature, schema)

    def text(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
    ) -> tuple[str | None, Telemetry]:
        return self._call(contents, system_instruction, temperature, None)


class OllamaBackend:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.host = (
            host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.timeout = timeout or float(os.environ.get("OLLAMA_TIMEOUT", "1800"))

    def _call(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
        schema: type[BaseModel] | None,
    ) -> tuple[str | None, Telemetry]:
        payload: dict = {
            "model": self.model,
            "prompt": contents,
            "system": system_instruction,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if schema is not None:
            payload["format"] = schema.model_json_schema()

        r = requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()

        telemetry = Telemetry(
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
        )
        return data.get("response") or None, telemetry

    def structured(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
        schema: type[BaseModel],
    ) -> tuple[str | None, Telemetry]:
        return self._call(contents, system_instruction, temperature, schema)

    def text(
        self,
        contents: str,
        system_instruction: str,
        temperature: float,
    ) -> tuple[str | None, Telemetry]:
        return self._call(contents, system_instruction, temperature, None)


def get_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "local":
        return OllamaBackend()
    return GeminiBackend(settings.llm_model)
