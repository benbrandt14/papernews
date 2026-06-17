import os
from typing import Callable
from google import genai
from google.genai import types
from papernews.models import RawDocument, LayoutChunk

client = genai.Client()

# Strategy pattern registry
ROUTER: dict[str, Callable[[RawDocument], LayoutChunk]] = {}

def register_router(content_type: str):
    """Decorator to register an LLM processing strategy for a given content type."""
    def decorator(func: Callable[[RawDocument], LayoutChunk]):
        ROUTER[content_type] = func
        return func
    return decorator

@register_router("rss")
def process_rss(doc: RawDocument) -> LayoutChunk:
    """Real LLM processing function natively enforcing the LayoutChunk schema."""
    
    system_instruction = (
        "You are an expert newspaper editor. Your job is to read the following raw article "
        "and output a strictly formatted LayoutChunk. "
        "Choose the template_type based on importance ('hero_grid' for major news, "
        "'standard_article' for normal news, 'sidebar_tease' for short snippets). "
        "Write a compelling newspaper headline. "
        "Write a concise, engaging body_markdown summarizing the article (2-3 paragraphs max)."
    )
    
    prompt = f"Title: {doc.metadata.get('title', 'Unknown')}\n\nArticle Text:\n{doc.raw_text}"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=LayoutChunk,  # Natively forces the Pydantic model
            temperature=0.4,
        )
    )
    
    if not response.parsed:
        raise ValueError("LLM failed to return a valid parsed LayoutChunk.")
        
    return response.parsed

@register_router("academic_pdf")
def process_academic_pdf(doc: RawDocument) -> LayoutChunk:
    """Mock LLM processing function for academic PDFs."""
    return LayoutChunk(
        template_type="academic_digest",
        headline=doc.metadata.get("title", "Untitled Academic Paper"),
        body_markdown=f"**Abstract summary:** {doc.raw_text[:100]}...",
        priority=1
    )