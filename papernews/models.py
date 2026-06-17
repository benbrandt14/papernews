from pydantic import BaseModel, Field
from typing import Literal, Optional

class RawDocument(BaseModel):
    source_id: str
    content_type: Literal["rss", "academic_pdf", "wiki_event", "wiki_quote"]
    raw_text: str
    metadata: dict = Field(default_factory=dict)

class ArticleChunk(BaseModel):
    content_type: Literal["rss", "academic_pdf", "wiki_event", "wiki_quote"]
    category: str
    source: str
    title: str
    summary: str
    body_markdown: str
    url: str
    priority: int = 3