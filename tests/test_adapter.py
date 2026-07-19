from papernews.adapter import article_to_dict, render_context_to_template_vars
from papernews.models import (
    Annotation,
    ArticleChunk,
    FrontpageDecorations,
    FunnelStats,
    Quote,
    RenderContext,
    Telemetry,
)


def test_article_to_dict_conversion():
    """
    Assert that a fully populated ArticleChunk model is translated flawlessly
    into the dictionary schema the Jinja template expects, including computed
    @property attributes from Telemetry.
    """
    chunk = ArticleChunk(
        content_type="rss",
        category="Technology",
        source="Hacker News",
        title="Test Article Title",
        summary="This is a test summary.",
        body_markdown="This is the full **markdown** body.",
        url="https://example.com/test",
        date="2023-10-27",
        published_date="2023-10-27T10:00:00Z",
        relative_time="2 days ago",
        telemetry=Telemetry(prompt_tokens=1500, output_tokens=500),
        annotations=[
            Annotation(source="AI", content="A great read.", completion_percentage=100)
        ],
    )

    data = article_to_dict(chunk)

    # Standard model fields
    assert data["category"] == "Technology"
    assert data["source"] == "Hacker News"
    assert data["title"] == "Test Article Title"
    assert data["summary"] == "This is a test summary."
    assert data["body_markdown"] == "This is the full **markdown** body."
    assert data["url"] == "https://example.com/test"
    assert data["date"] == "2023-10-27"

    # Verify annotations
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["source"] == "AI"

    # Computed @property fields from Telemetry that MUST exist for Jinja/Typst
    assert "telemetry" in data
    assert data["telemetry"]["prompt_tokens"] == 1500
    assert data["telemetry"]["output_tokens"] == 500
    assert data["telemetry"]["formatted_tokens"] == "^3.3"  # mag notation

    # Cost is derived from the provider price constants; assert the formatting
    # logic rather than a provider-specific number.
    from papernews.models import COST_PER_1M_OUTPUT, COST_PER_1M_PROMPT

    cents = (1500 / 1e6 * COST_PER_1M_PROMPT + 500 / 1e6 * COST_PER_1M_OUTPUT) * 100
    expected = "~ 0" if cents < 0.05 else f"{cents:.3f}"
    assert data["telemetry"]["formatted_cost"] == expected


def test_article_to_dict_cost_formatting():
    chunk = ArticleChunk(
        category="Tech",
        source="Source",
        title="Title",
        summary="Sum",
        body_markdown="Body",
        url="URL",
        telemetry=Telemetry(prompt_tokens=1_000_000, output_tokens=1_000_000),
    )
    data = article_to_dict(chunk)
    # 1M prompt + 1M output at the provider's per-1M rates, in cents.
    from papernews.models import COST_PER_1M_OUTPUT, COST_PER_1M_PROMPT

    cents = (COST_PER_1M_PROMPT + COST_PER_1M_OUTPUT) * 100
    assert data["telemetry"]["formatted_cost"] == f"{cents:.3f}"


def test_render_context_to_template_vars():
    """The adapter must assemble the exact `decorations` shape the template
    reads: front-page decorations merged with run metadata."""
    ctx = RenderContext(
        date="2026-07-05",
        generation_time="Jul 05, 2026 at 07:00 AM",
        total_tokens="1.5k",
        total_cost="~ 0",
        articles=[
            ArticleChunk(
                category="Tech",
                source="example.com",
                title="Title",
                summary="Sum",
                body_markdown="Body",
                url="https://example.com",
                telemetry=Telemetry(prompt_tokens=10, output_tokens=5),
            )
        ],
        decorations=FrontpageDecorations(
            world_news=["Something happened."],
            quote=Quote(text="Words.", author="Someone"),
            dyk=["a fact"],
        ),
        stats=FunnelStats(ingested=142, after_filter=38, after_budget=14, selected=9),
    )

    variables = render_context_to_template_vars(ctx)

    assert variables["date"] == "2026-07-05"

    # Triage-funnel telemetry is exposed to the front-matter index page.
    assert variables["stats"] == {
        "ingested": 142,
        "after_filter": 38,
        "after_budget": 14,
        "selected": 9,
    }
    deco = variables["decorations"]
    assert deco["generation_time"] == "Jul 05, 2026 at 07:00 AM"
    assert deco["total_tokens"] == "1.5k"
    assert deco["total_cost"] == "~ 0"
    assert deco["world_news"] == ["Something happened."]
    assert deco["quote"] == {"text": "Words.", "author": "Someone"}
    assert deco["dyk"] == ["a fact"]

    # Articles are dicts with the telemetry @property fields injected
    art = variables["articles"][0]
    assert art["title"] == "Title"
    assert art["telemetry"]["formatted_tokens"] == "^1.2"  # mag notation
    assert "formatted_cost" in art["telemetry"]
