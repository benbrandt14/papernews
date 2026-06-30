import pytest
from papernews.models import ArticleChunk, Telemetry, Annotation
from papernews.adapter import article_to_dict

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
        url_hash="abcdef123456",
        date="2023-10-27",
        published_date="2023-10-27T10:00:00Z",
        relative_time="2 days ago",
        telemetry=Telemetry(prompt_tokens=1500, output_tokens=500),
        annotations=[
            Annotation(source="AI", content="A great read.", completion_percentage=100)
        ]
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
        category="Tech", source="Source", title="Title", summary="Sum", body_markdown="Body", url="URL",
        telemetry=Telemetry(prompt_tokens=1_000_000, output_tokens=1_000_000)
    )
    data = article_to_dict(chunk)
    # 1M prompt = $0.075, 1M output = $0.30 => Total $0.375 => 37.5 cents
    assert data["telemetry"]["formatted_cost"] == "37.500"
