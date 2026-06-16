import pluggy
from typing import List
from papernews.models import RawDocument

hookspec = pluggy.HookspecMarker("papernews")

@hookspec
def fetch_documents(config: dict) -> List[RawDocument]:
    """Fetch documents from a source."""
