# papernews/models.py
from typing import Literal

from pydantic import BaseModel, Field

# Fixed Pricing Constants ($, Gemini 2.5 Flash)
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
    title: str = ""
    category: str = "Uncategorized"
    published: str = ""
    # Local-ranking score attached by triage Stage 2B (lower = better).
    heuristic_score: int = 3
    # Free-form plugin extras (e.g. feed_url); typed data belongs in fields.
    metadata: dict = Field(default_factory=dict)


class LLMArticleSelection(BaseModel):
    is_selected: bool = Field(
        description="True if the article aligns with the user's explicit interests and avoids disinterests."
    )


class LLMArticleSummary(BaseModel):
    summary: str = Field(
        description="A concise, engaging 1-3 sentence summary of the article."
    )


class Telemetry(BaseModel):
    prompt_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Telemetry") -> "Telemetry":
        return Telemetry(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def cost_cents(self) -> float:
        dollars = (self.prompt_tokens / 1_000_000 * COST_PER_1M_PROMPT) + (
            self.output_tokens / 1_000_000 * COST_PER_1M_OUTPUT
        )
        return dollars * 100

    @property
    def formatted_tokens(self) -> str:
        if self.total_tokens == 0:
            return "0"
        return f"{self.total_tokens / 1000:.1f}k"

    @property
    def formatted_cost(self) -> str:
        c = self.cost_cents
        return "~ 0" if c < 0.05 else f"{c:.3f}"


class Quote(BaseModel):
    text: str
    author: str = "Anonymous"


class FrontpageDecorations(BaseModel):
    world_news: list[str] = Field(
        default_factory=lambda: ["World news currently unavailable."],
        description="Bullet points for the Wikipedia current events sidebar.",
    )
    quote: Quote | None = None
    dyk: list[str] = Field(
        default_factory=list,
        description="'Did you know...' facts for the front page.",
    )


class Span(BaseModel):
    """Inline formatting/annotation over Block.text char offsets."""

    start: int
    end: int
    kind: Literal[
        "strong", "emph", "link", "code_inline", "math_inline", "entity", "salience"
    ]
    weight: float | None = None  # salience 0..1 → font weight/luma bucket
    href: str | None = None  # link URL, or "#label" for internal targets
    label: str | None = None  # entity id / Typst label


class ImageRef(BaseModel):
    alt: str = ""
    url: str


class Block(BaseModel):
    """One structural unit of an article body (the markdown IR)."""

    kind: Literal[
        "para", "heading", "quote", "code", "math_display", "image", "list_item"
    ]
    level: int = 0  # heading depth / list nesting
    text: str = ""  # plain text, no markdown, no Typst
    spans: list[Span] = Field(default_factory=list)
    raw: str = ""  # verbatim payload for code / math_display
    images: list[ImageRef] = Field(default_factory=list)


class EntityRef(BaseModel):
    surface: str
    label: str
    wikidata_id: str | None = None
    glossary_note: str | None = None


class Enrichment(BaseModel):
    """Sidecar data attached by Stage 3.5 enrichment plugins."""

    entities: list[EntityRef] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    comments: list[Annotation] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class ArticleChunk(BaseModel):
    content_type: Literal["rss", "academic_pdf", "wiki_event", "wiki_quote"] = "rss"
    category: str
    source: str
    title: str
    summary: str
    body_markdown: str
    url: str
    date: str = ""
    published_date: str = ""
    relative_time: str = ""
    telemetry: Telemetry = Field(default_factory=Telemetry)
    annotations: list[Annotation] = Field(default_factory=list)
    # Structured body (markdown IR); the typed emitter renders it.
    blocks: list[Block] = Field(default_factory=list)
    enrichment: Enrichment = Field(default_factory=Enrichment)


class FunnelStats(BaseModel):
    """How the triage funnel narrowed the day's intake.

    Rendered on the front-matter index page so the filtering process is
    visible in the finished paper, not just in logs.
    """

    ingested: int = 0
    after_filter: int = 0
    after_budget: int = 0
    selected: int = 0


class RenderContext(BaseModel):
    """Everything the Typst template needs for one edition.

    This is the single contract between the pipeline and the renderer;
    the adapter flattens it into template variables.
    """

    date: str
    generation_time: str
    total_tokens: str
    total_cost: str
    articles: list[ArticleChunk] = Field(default_factory=list)
    decorations: FrontpageDecorations = Field(default_factory=FrontpageDecorations)
    stats: FunnelStats = Field(default_factory=FunnelStats)
    # Index into `articles` of the front-page lead story (design refresh hook).
    lead_article_index: int | None = None
