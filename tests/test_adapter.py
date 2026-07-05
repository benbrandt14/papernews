from papernews.adapter import article_to_dict, render_context_to_template_vars
from papernews.models import (
    Annotation,
    ArticleChunk,
    FrontpageDecorations,
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
    assert data["telemetry"]["formatted_tokens"] == "2.0k"

    # 1500 prompt * $0.075/1M + 500 output * $0.30/1M = $0.0001125 + $0.00015 = $0.0002625
    # * 100 = 0.02625 cents (< 0.05)
    assert data["telemetry"]["formatted_cost"] == "~ 0"


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
    # 1M prompt = $0.075, 1M output = $0.30 => Total $0.375 => 37.5 cents
    assert data["telemetry"]["formatted_cost"] == "37.500"


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
    )

    variables = render_context_to_template_vars(ctx)

    assert variables["date"] == "2026-07-05"
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
    assert art["telemetry"]["formatted_tokens"] == "0.0k"
    assert "formatted_cost" in art["telemetry"]
