# papernews/models.py
from typing import Literal

from pydantic import BaseModel, Field

# Approximate pricing ($ per 1M tokens) for the default provider (DeepSeek
# deepseek-chat, cache-miss rates). Token counts come from the API and are
# exact; only this cost estimate is provider-dependent — edit if you switch.
COST_PER_1M_PROMPT = 0.14
COST_PER_1M_OUTPUT = 0.28


class Annotation(BaseModel):
    source: str
    content: str
    completion_percentage: int
    style: Literal["standard", "snark"] = "standard"


class AITextMetrics(BaseModel):
    """Article-level stylometrics from the AI-likeness screen (ai_detect.py).

    Computed on the ingested source text, before the LLM rewrites anything.
    `ai_likelihood` is a noise dial (0 = human-flavored, 1 = formulaic LLM
    filler), not a forensic verdict; when `reliable` is False the sample was
    too small and the pipeline must not act on the score.
    """

    ai_likelihood: float = 0.0
    burstiness: float = 0.0  # sentence-length coefficient of variation
    lexical_diversity: float = 0.0  # moving-window type/token ratio
    stock_phrases_per_1k: float = 0.0  # LLM-tell phrase hits per 1000 words
    word_count: int = 0
    reliable: bool = False

    @property
    def formatted_likelihood(self) -> str:
        return f"{self.ai_likelihood:.0%}"

    @property
    def formatted_burstiness(self) -> str:
        return f"{self.burstiness:.2f}"

    @property
    def formatted_diversity(self) -> str:
        return f"{self.lexical_diversity:.2f}"

    @property
    def formatted_phrase_rate(self) -> str:
        return f"{self.stock_phrases_per_1k:.1f}"


class RawDocument(BaseModel):
    source_id: str
    content_type: Literal["rss", "academic_pdf", "wiki_event", "wiki_quote"]
    raw_text: str
    title: str = ""
    category: str = "Uncategorized"
    published: str = ""
    # Local-ranking score attached by triage Stage 2B (lower = better).
    heuristic_score: int = 3
    # Stylometrics attached by triage Stage 2B.5 (the AI-likeness screen).
    ai_metrics: AITextMetrics | None = None
    # Free-form plugin extras (e.g. feed_url); typed data belongs in fields.
    metadata: dict = Field(default_factory=dict)


class LLMArticleSelection(BaseModel):
    is_selected: bool = Field(
        description="True if the article aligns with the user's explicit interests and avoids disinterests."
    )


class LLMArticleSummary(BaseModel):
    summary: str = Field(
        description="A concise & engaging, while subtly sarcastic or humorous, 1-3 sentence summary of the article and it's broader context."
    )


class LLMOpenQuestions(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        description=(
            "Up to three specific, researchable questions a curious reader "
            "would want answered after reading the article."
        ),
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


class Curiosity(BaseModel):
    """A once-open reader question the pipeline has since answered.

    Questions are raised during enrichment, parked in the curiosity queue,
    and resolved on a later run via a literature lookup. The answered pairs
    surface on the front matter so the paper visibly follows up on itself.
    """

    question: str
    answer_title: str
    answer_url: str


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
    curiosities: list[Curiosity] = Field(
        default_factory=list,
        description="Answered questions surfaced from the curiosity queue.",
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
    # Stylometrics of the *source* text (pre-LLM), for the article footer.
    ai_metrics: AITextMetrics | None = None


class FunnelStats(BaseModel):
    """How the triage funnel narrowed the day's intake.

    Rendered on the front-matter index page so the filtering process is
    visible in the finished paper, not just in logs.
    """

    ingested: int = 0
    after_filter: int = 0
    after_budget: int = 0
    selected: int = 0
    # AI-likeness screen (Stage 2B.5): deranked survive with a rank penalty,
    # dropped never reach the category budget.
    ai_deranked: int = 0
    ai_dropped: int = 0


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
