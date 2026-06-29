from papernews.models import ArticleChunk, Annotation, Telemetry
from hypothesis import given, strategies as st
from pydantic import ValidationError

@given(
    st.integers(min_value=0, max_value=1_000_000),
    st.integers(min_value=0, max_value=1_000_000)
)
def test_telemetry_property(prompt_tokens, output_tokens):
    t1 = Telemetry(prompt_tokens=prompt_tokens, output_tokens=output_tokens)
    assert t1.total_tokens == prompt_tokens + output_tokens
    assert t1.formatted_tokens == (f"{t1.total_tokens / 1000:.1f}k" if t1.total_tokens > 0 else "0")

    t2 = Telemetry(prompt_tokens=100, output_tokens=100)
    t3 = t1 + t2
    assert t3.prompt_tokens == prompt_tokens + 100
    assert t3.output_tokens == output_tokens + 100

@given(
    st.builds(
        ArticleChunk,
        content_type=st.sampled_from(["rss", "academic_pdf", "wiki_event", "wiki_quote"]),
        category=st.text(),
        source=st.text(),
        title=st.text(),
        summary=st.text(),
        body_markdown=st.text(),
        url=st.text(),
        url_hash=st.text(),
        date=st.text(),
        published_date=st.text(),
        relative_time=st.text(),
        telemetry=st.builds(Telemetry, prompt_tokens=st.integers(), output_tokens=st.integers()),
        annotations=st.lists(
            st.builds(Annotation, source=st.text(), content=st.text(), completion_percentage=st.integers(), style=st.sampled_from(["standard", "snark"]))
        )
    )
)
def test_article_chunk_serialization_property(chunk):
    # Verify serialization works without crashing
    dumped = chunk.model_dump()
    assert isinstance(dumped, dict)

    # Verify round-trip deserialization
    reconstructed = ArticleChunk(**dumped)
    assert chunk.title == reconstructed.title
    assert chunk.telemetry.prompt_tokens == reconstructed.telemetry.prompt_tokens
