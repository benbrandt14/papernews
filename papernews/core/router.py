# papernews/core/router.py
from collections.abc import Callable
from functools import lru_cache

import requests
from prefect import get_run_logger, task
from prefect.context import TaskRunContext
from pydantic import BaseModel

from papernews.config import Preferences, get_settings
from papernews.core.backends import get_backend
from papernews.models import (
    LLMArticleSelection,
    LLMArticleSummary,
    RawDocument,
    Telemetry,
)
from papernews.store import SimpleStore

SUMMARY_INPUT_LENGTH = 1500


@lru_cache(maxsize=1)
def _db() -> SimpleStore:
    """Lazy store — importing this module must not create database files."""
    return SimpleStore()


_TRANSIENT_CODES = {429, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    """Errors worth retrying: network trouble and 5xx/429 API responses."""
    if isinstance(exc, ConnectionError | TimeoutError | requests.RequestException):
        return True
    # A status code may sit on the exception (`.code`) or, for requests'
    # HTTPError, on its response.
    code = getattr(exc, "code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return code in _TRANSIENT_CODES


def _retries_remaining() -> bool:
    """True while Prefect still has retry attempts left for this task run.

    Transient errors are re-raised while retries remain (so Prefect's
    retry policy actually fires) and only degrade to the task's fallback
    on the final attempt. Outside a task run (unit tests calling .fn)
    there is no retry loop, so this returns False.
    """
    ctx = TaskRunContext.get()
    if ctx is None or ctx.task_run is None:
        return False
    retries = ctx.task.retries or 0
    run_count = getattr(ctx.task_run, "run_count", 1) or 1
    return run_count <= retries


def _cached_structured_call(
    cache_key: str,
    contents: str,
    system_instruction: str,
    temperature: float,
    schema: type[BaseModel] | None = None,
    transform: Callable[[str], str] = lambda s: s,
) -> tuple[str | None, Telemetry, bool]:
    """Shared skeleton for every LLM task: cache-check, call, cache-store.

    Returns (text, telemetry, cache_hit). `text` is None when the model
    produced no usable output. `transform` post-processes the raw model
    text before it is cached (e.g. stripping markdown fences). The cache
    sits above the backend, so any backend benefits from it.
    """
    cached = _db().get_cache(cache_key)
    if cached is not None:
        return cached, Telemetry(), True

    backend = get_backend(get_settings())
    if schema is not None:
        raw, telemetry = backend.structured(
            contents, system_instruction, temperature, schema
        )
    else:
        raw, telemetry = backend.text(contents, system_instruction, temperature)

    if raw is None:
        return None, telemetry, False

    text = transform(raw)
    _db().set_cache(cache_key, text)
    return text, telemetry, False


@task(name="LLM: Gatekeeper Selection", retries=3, retry_delay_seconds=10)
def llm_select_article(doc: RawDocument, prefs: Preferences) -> tuple[bool, Telemetry]:
    logger = get_run_logger()

    if not get_settings().llm_enabled:
        logger.info(f"--no-llm: Auto-accept {doc.title[:30]}...")
        return True, Telemetry()

    interests = prefs.interest or ["General high-quality news"]
    disinterests = prefs.disinterest or ["Clickbait", "Ads"]

    prompt_text = (
        f"Title: {doc.title}\n"
        f"Local Rank Score: {doc.heuristic_score}\n"
        f"User Interests: {', '.join(interests)}\n"
        f"User Disinterests: {', '.join(disinterests)}\n"
        f"Content Snippet:\n{doc.raw_text[:1500]}"
    )
    system_instruction = "As a technical expert and concierge editor evaluate the snippet. Return ONLY a boolean 'is_selected' indicating if it belongs in the digest."

    try:
        text, telemetry, cache_hit = _cached_structured_call(
            cache_key=f"select_{doc.source_id}",
            contents=prompt_text,
            system_instruction=system_instruction,
            temperature=0.1,
            schema=LLMArticleSelection,
        )
        if cache_hit:
            logger.info(f"Cache Hit: Selection for '{doc.title[:30]}...'")
        if text is None:
            return False, telemetry
        return LLMArticleSelection.model_validate_json(text).is_selected, telemetry
    except Exception as e:
        if _is_transient(e) and _retries_remaining():
            raise
        logger.error(f"Selection Error: {e}")
        return False, Telemetry()


@task(name="LLM: Article Summarization", retries=3, retry_delay_seconds=10)
def llm_summarize_article(doc: RawDocument) -> tuple[str, Telemetry]:
    logger = get_run_logger()

    if not get_settings().llm_enabled:
        logger.info(f"--no-llm: Auto-accept {doc.title[:30]}...")
        return "Summarization Disabled..", Telemetry()

    prompt_text = f"Title: {doc.title}\nSnippet:\n{doc.raw_text[:SUMMARY_INPUT_LENGTH]}"
    system_instruction = "Write a concise & engaging, while subtly sarcastic or humorous, 1-3 sentence summary of the article and it's broader context."

    try:
        text, telemetry, cache_hit = _cached_structured_call(
            cache_key=f"summary_{doc.source_id}",
            contents=prompt_text,
            system_instruction=system_instruction,
            temperature=0.3,
            schema=LLMArticleSummary,
        )
        if cache_hit:
            logger.info(f"Cache Hit: Summary for '{doc.title[:30]}...'")
        if text is None:
            return "Summary unavailable.", telemetry
        return LLMArticleSummary.model_validate_json(text).summary, telemetry
    except Exception as e:
        if _is_transient(e) and _retries_remaining():
            raise
        logger.error(f"Summarization Error: {e}")
        return "Summary unavailable due to an error.", Telemetry()


@task(name="LLM: Strict Markdown Formatter", retries=3, retry_delay_seconds=10)
def llm_format_body(doc: RawDocument) -> tuple[str, Telemetry]:
    """
    Case 3: Article formatting. asked "pretty please" not to modify content.
    Returns (formatted_markdown, Telemetry)
    """
    logger = get_run_logger()

    if not get_settings().llm_enabled:
        logger.info(f"--no-llm: Auto-accept {doc.title[:30]}...")
        return doc.raw_text, Telemetry()

    system_instruction = """
    You are a strict typography and article typsetting engine.
    Your ONLY job is to format the provided text into clean Markdown.
    - Remove unnecessary indentation, web navigation, and spacing.
    - Remove text not associated with the content (comments, external links, "see also")
    - Format quotes (`>`) and code blocks (` ``` `).
    - Correctly format hyperlinks `[text](url)`.
    - Reformat bullet points and lists cleanly.
    - For currency, escape dollar signs (like \\$5.00). Leave valid math equations enclosed in normal unescaped $
    - Identify and format section headers (`#`, `##`).

    CRITICAL RULES:
    DO NOT add any introductory or concluding text.
    DO NOT summarize.
    DO NOT change the author's words or content.
    Output ONLY the cleaned Markdown text.
    """

    def _strip_fences(text: str) -> str:
        # Strip markdown code block wrappers if the LLM includes them
        return text.strip().replace("```markdown", "").replace("```", "").strip()

    try:
        text, telemetry, cache_hit = _cached_structured_call(
            cache_key=f"format_{doc.source_id}",
            contents=doc.raw_text,
            system_instruction=system_instruction,
            temperature=0.0,
            transform=_strip_fences,
        )
        if cache_hit:
            logger.info(f"Cache Hit: Formatting for '{doc.title[:30]}...'")
        if text is None:
            return doc.raw_text, telemetry
        return text, telemetry
    except Exception as e:
        if _is_transient(e) and _retries_remaining():
            raise
        logger.warning(f"Formatting Error, falling back to deterministic raw text: {e}")
        return doc.raw_text, Telemetry()
