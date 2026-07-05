# papernews/core/router.py
from collections.abc import Callable
from functools import lru_cache

from google import genai
from google.genai import types
from prefect import get_run_logger, task

from papernews.config import Preferences, get_settings
from papernews.models import (
    LLMArticleSelection,
    LLMArticleSummary,
    RawDocument,
    Telemetry,
)
from papernews.store import SimpleStore

SUMMARY_INPUT_LENGTH = 1500


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    """Lazy Gemini client — importing this module must not require credentials."""
    return genai.Client()


@lru_cache(maxsize=1)
def _db() -> SimpleStore:
    """Lazy store — importing this module must not create database files."""
    return SimpleStore()


def _get_telemetry(response: types.GenerateContentResponse) -> Telemetry:
    """Helper to safely extract tokens from a Gemini response."""
    if response.usage_metadata:
        return Telemetry(
            prompt_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0,
        )

    return Telemetry()


def _cached_structured_call(
    cache_key: str,
    contents: str,
    system_instruction: str,
    temperature: float,
    schema: type | None = None,
    transform: Callable[[str], str] = lambda s: s,
) -> tuple[str | None, Telemetry, bool]:
    """Shared skeleton for every LLM task: cache-check, call, cache-store.

    Returns (text, telemetry, cache_hit). `text` is None when the model
    produced no usable output. `transform` post-processes the raw model
    text before it is cached (e.g. stripping markdown fences).
    """
    cached = _db().get_cache(cache_key)
    if cached is not None:
        return cached, Telemetry(), True

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
    )
    if schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = schema

    response = _client().models.generate_content(
        model=get_settings().llm_model,
        contents=contents,
        config=config,
    )
    telemetry = _get_telemetry(response)

    if not response.text:
        return None, telemetry, False

    text = transform(response.text)
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
        logger.error(f"Selection Error: {e}")
        return False, Telemetry()


@task(name="LLM: Article Summarization", retries=3, retry_delay_seconds=10)
def llm_summarize_article(doc: RawDocument) -> tuple[str, Telemetry]:
    logger = get_run_logger()

    if not get_settings().llm_enabled:
        logger.info(f"--no-llm: Auto-accept {doc.title[:30]}...")
        return "Summarization Disabled..", Telemetry()

    prompt_text = f"Title: {doc.title}\nSnippet:\n{doc.raw_text[:SUMMARY_INPUT_LENGTH]}"
    system_instruction = "Write a concise, engaging 1-3 sentence summary of this article snippet. Do not include introductory text."

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
        logger.error(f"Summarization Error: {e}")
        return "Summary unavailable.", Telemetry()


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
    You are a strict typography and formatting engine.
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
        logger.warning(f"Formatting Error, falling back to deterministic raw text: {e}")
        return doc.raw_text, Telemetry()
