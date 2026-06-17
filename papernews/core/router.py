import os
from typing import Callable
from google import genai
from google.genai import types
from papernews.models import RawDocument, ArticleChunk

client = genai.Client()
ROUTER: dict[str, Callable[[RawDocument], ArticleChunk]] = {}

def register_router(content_type: str):
    def decorator(func: Callable[[RawDocument], ArticleChunk]):
        ROUTER[content_type] = func
        return func
    return decorator

@register_router("rss")
def process_rss(doc: RawDocument) -> ArticleChunk:
    prompt = f"Title: {doc.metadata.get('title', 'Unknown')}\n\nText:\n{doc.raw_text}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an expert technical writer. Summarize the following raw article into an ArticleChunk. "
                "Provide a short 1-sentence italicized 'summary' (the lede), and a 2-paragraph 'body_markdown'. "
                "Determine an appropriate single-word 'category' (e.g., Technology, Space, Cardiology, Math)."
            ),
            response_mime_type="application/json",
            response_schema=ArticleChunk,
            temperature=0.3,
        )
    )
    if not response.parsed:
        raise ValueError("LLM failed to return a valid ArticleChunk.")
    
    # Pass through the URL and Source from the raw document
    response.parsed.url = doc.metadata.get("url", "")
    response.parsed.source = doc.metadata.get("source", "Web Feed")
    response.parsed.content_type = "rss"
    return response.parsed

@register_router("academic_pdf")
def process_academic_pdf(doc: RawDocument) -> ArticleChunk:
    return ArticleChunk(
        content_type="academic_pdf",
        category="Academic Research",
        source="Unpaywall Open Access",
        title=doc.metadata.get("title", "Untitled Academic Paper"),
        summary="A newly indexed peer-reviewed paper.",
        body_markdown=f"**Abstract:** {doc.raw_text[:300]}...",
        url=doc.metadata.get("url", ""),
        priority=1
    )

# --- Wiki Routers (Pass-throughs) ---
@register_router("wiki_event")
def process_wiki_event(doc: RawDocument) -> ArticleChunk:
    return ArticleChunk(content_type="wiki_event", category="World News", source="Wikipedia", title=doc.metadata.get("title", ""), summary="", body_markdown=doc.raw_text, url="")

@register_router("wiki_quote")
def process_wiki_quote(doc: RawDocument) -> ArticleChunk:
    return ArticleChunk(content_type="wiki_quote", category="Quote", source="Wikiquote", title="Quote of the Day", summary="", body_markdown=doc.raw_text, url="")