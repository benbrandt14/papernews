from __future__ import annotations

import json
import os

_BACKEND = os.environ.get("LLM_BACKEND", "gemini").lower()


def chat(system: str, user: str, max_tokens: int) -> str:
    """Single-shot chat. Always streams under the hood — large rewrite batches
    can exceed the API's non-streaming deadline, and a slow Ollama instance
    benefits from bytes-flowing keepalive through any reverse proxy."""
    if _BACKEND == "ollama":
        return _ollama(system, user, max_tokens)
    return _gemini(system, user, max_tokens)


def _gemini(system: str, user: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content_stream(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
    )
    parts = []
    for chunk in response:
        parts.append(chunk.text)
    return "".join(parts)


def _ollama(system: str, user: str, max_tokens: int) -> str:
    import httpx

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "mistral")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT", "1800"))
    parts: list[str] = []
    with httpx.stream(
        "POST",
        f"{host}/api/chat",
        json={
            "model": model,
            "stream": True,
            "options": {"num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if msg := chunk.get("message"):
                parts.append(msg.get("content", ""))
            if chunk.get("done"):
                break
    return "".join(parts)
