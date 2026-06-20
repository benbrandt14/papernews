# papernews/core/router.py
import os
from typing import Optional
from prefect import task, get_run_logger
from google import genai
from google.genai import types

from papernews.models import RawDocument, LLMArticleSelection, LLMArticleSummary, Telemetry

client = genai.Client()

def _get_telemetry(response) -> Telemetry:
    """Helper to safely extract tokens from a Gemini response."""
    if response.usage_metadata:
        return Telemetry(
            prompt_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count
        )
    return Telemetry()

@task(name="LLM: Gatekeeper Selection", retries=3, retry_delay_seconds=10)
def llm_select_article(doc: RawDocument, prefs: dict) -> tuple[bool, Telemetry]:
    """
    Case 1: Final article selection filter. 
    Returns (is_selected, Telemetry)
    """
    logger = get_run_logger()
    
    title = doc.metadata.get('title', 'Unknown Title')
    score = doc.metadata.get('heuristic_score', 3)
    snippet = doc.raw_text[:1500] 
    
    interests = prefs.get("interest", ["General high-quality news"])
    disinterests = prefs.get("disinterest", ["Clickbait", "Ads"])
    
    prompt_text = f"Title: {title}\nLocal Rank Score: {score}\nUser Interests: {', '.join(interests)}\nUser Disinterests: {', '.join(disinterests)}\nContent Snippet:\n{snippet}"
    system_instruction = "As a technical expert and concierge editor evaluate the snippet. Return ONLY a boolean 'is_selected' indicating if it belongs in the digest."

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=LLMArticleSelection,
                temperature=0.1
            )
        )
        
        telemetry = _get_telemetry(response)

        if response.text:
            result = LLMArticleSelection.model_validate_json(response.text)
            return result.is_selected, telemetry
        return False, telemetry
    except Exception as e:
        logger.error(f"Selection Error: {e}")
        return False, Telemetry()

@task(name="LLM: Article Summarization", retries=3, retry_delay_seconds=10)
def llm_summarize_article(doc: RawDocument) -> tuple[str, Telemetry]:
    """
    Case 2: Summarization ONLY. Called only if the article survives the gatekeeper.
    Returns (summary_text, Telemetry)
    """
    logger = get_run_logger()
    title = doc.metadata.get('title', 'Unknown Title')
    snippet = doc.raw_text[:1500] 
    
    prompt_text = f"Title: {title}\nSnippet:\n{snippet}"
    system_instruction = "Write a concise, engaging 1-3 sentence summary of this article snippet. Do not include introductory text."

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=LLMArticleSummary,
                temperature=0.3
            )
        )
        
        telemetry = _get_telemetry(response)

        if response.text:
            result = LLMArticleSummary.model_validate_json(response.text)
            return result.summary, telemetry
        return "Summary unavailable.", telemetry
    except Exception as e:
        logger.error(f"Summarization Error: {e}")
        return "Summary unavailable.", Telemetry()

@task(name="LLM: Strict Markdown Formatter", retries=3, retry_delay_seconds=10)
def llm_format_body(raw_text: str) -> tuple[str, Telemetry]:
    """
    Case 3: Article formatting. asked "pretty please" not to modify content. 
    Returns (formatted_markdown, Telemetry)
    """
    logger = get_run_logger()
    
    system_instruction = """
    You are a strict typography and formatting engine.
    Your ONLY job is to format the provided text into clean Markdown.
    - Remove unnecessary indentation, web navigation, and spacing.
    - Remove text not associated with the content (comments, external links, "see also")
    - Format quotes (`>`) and code blocks (` ``` `).
    - Correctly format hyperlinks `[text](url)`.
    - Reformat bullet points and lists cleanly.
    - Identify and format section headers (`#`, `##`).
    
    CRITICAL RULES:
    DO NOT add any introductory or concluding text.
    DO NOT summarize.
    DO NOT change the author's words or content.
    Output ONLY the cleaned Markdown text.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=raw_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            )
        )
        
        telemetry = _get_telemetry(response)
        clean_text = response.text.strip().replace("```markdown", "").replace("```", "").strip() if response.text else raw_text
        return clean_text, telemetry
    except Exception as e:
        logger.warning(f"Formatting Error, falling back to deterministic raw text: {e}")
        return raw_text, Telemetry()