from pydantic import BaseModel, Field
from typing import Literal

class RawDocument(BaseModel):
    source_id: str
    content_type: str  # e.g., "rss", "wiki_event", "github"
    raw_text: str
    metadata: dict = Field(default_factory=dict)

class ArticleChunk(BaseModel):
    # Layout Designation
    region: Literal["index", "cover_feature", "sidebar", "interior"] = "interior"
    
    # Standardized Metadata
    category: str
    source: str
    title: str
    url: str
    
    # Standardized Content
    summary: str
    body_markdown: str
    
    priority: int = 3