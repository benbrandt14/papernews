from typing import Literal, Optional
from pydantic import BaseModel, Field

class RawDocument(BaseModel):
    source_id: str
    content_type: Literal["rss", "academic_pdf", "synology_log"]
    raw_text: str
    metadata: dict = Field(default_factory=dict)

class LayoutChunk(BaseModel):
    template_type: Literal["hero_grid", "sidebar_tease", "academic_digest", "standard_article"]
    headline: str
    body_markdown: str
    image_path: Optional[str] = None
    priority: int = 1
