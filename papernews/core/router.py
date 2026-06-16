from typing import Callable
from papernews.models import RawDocument, LayoutChunk

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
    """Mock LLM processing function for RSS feeds."""
    return LayoutChunk(
        template_type="standard_article",
        headline=doc.metadata.get("title", "Untitled RSS Article"),
        body_markdown=f"**Summary:** {doc.raw_text[:50]}...",
        priority=2
    )

@register_router("academic_pdf")
def process_academic_pdf(doc: RawDocument) -> LayoutChunk:
    """Mock LLM processing function for academic PDFs."""
    return LayoutChunk(
        template_type="academic_digest",
        headline=doc.metadata.get("title", "Untitled Academic Paper"),
        body_markdown=f"**Abstract summary:** {doc.raw_text[:100]}...",
        priority=1
    )
