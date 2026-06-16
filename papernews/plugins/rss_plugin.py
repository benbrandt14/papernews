import pluggy
from typing import List
from papernews.models import RawDocument

hookimpl = pluggy.HookimplMarker("papernews")

@hookimpl
def fetch_documents(config: dict) -> List[RawDocument]:
    """Mock RSS fetcher."""
    return [
        RawDocument(
            source_id="rss_mock_1",
            content_type="rss",
            raw_text="This is a mocked RSS feed document text.",
            metadata={"title": "Mock RSS Article", "url": "http://example.com/rss-1"}
        )
    ]
