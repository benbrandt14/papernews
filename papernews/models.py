# papernews/models.py
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# Fixed Pricing Constants (Gemini 2.5 Flash)
COST_PER_1M_PROMPT = 0.075
COST_PER_1M_OUTPUT = 0.30

class Annotation(BaseModel):
    source: str
    content: str
    completion_percentage: int
    style: Literal["standard", "snark"] = "standard"

class RawDocument(BaseModel):
    source_id: str
    content_type: Literal["rss", "academic_pdf", "wiki_event", "wiki_quote"]
    raw_text: str
    metadata: dict = Field(default_factory=dict)

class LLMArticleSelection(BaseModel):
    """Stage 3A: Strict boolean selection based on user preferences."""
    is_selected: bool = Field(description="True if the article aligns with the user's explicit interests and avoids disinterests.")

class LLMArticleSummary(BaseModel):
    """Stage 3B: Summary generation only for surviving articles."""
    summary: str = Field(description="A concise, engaging 1-3 sentence summary of the article.")

class Telemetry(BaseModel):
    prompt_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other):
        return Telemetry(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def cost_cents(self) -> float:
        dollars = (self.prompt_tokens / 1_000_000 * COST_PER_1M_PROMPT) + \
                  (self.output_tokens / 1_000_000 * COST_PER_1M_OUTPUT)
        return dollars * 100

    @property
    def formatted_tokens(self) -> str:
        if self.total_tokens == 0: return "0"
        return f"{self.total_tokens / 1000:.1f}k"

    @property
    def formatted_cost(self) -> str:
        c = self.cost_cents
        return "~ 0" if c < 0.05 else f"{c:.3f}"

class FrontpageDecorations(BaseModel):
    """Defines the allowed widgets and sidebars for the PDF template."""
    world_news: list[str] = Field(
        default_factory=lambda: ["World news currently unavailable."],
        description="Bullet points for the Wikipedia current events sidebar."
    )
    
class ArticleChunk(BaseModel):
    content_type: Literal["rss", "academic_pdf", "wiki_event", "wiki_quote"]
    category: str
    source: str
    title: str
    summary: str
    body_markdown: str
    url: str
    published_date: str = ""
    relative_time: str = ""
    telemetry: Telemetry = Field(default_factory=Telemetry) 
    annotations: List[Annotation] = Field(default_factory=list)