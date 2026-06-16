from __future__ import annotations

from dataclasses import dataclass

import trafilatura
from trafilatura.metadata import extract_metadata


@dataclass
class Article:
    """Container for extracted article text."""
    source: str
    url: str
    title: str
    text: str
    published: str | None = None  # ISO date from page metadata, may be None


def extract(url: str, title: str, source: str) -> Article | None:
    """Extract main body text from URL."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        include_links=True,
        include_images=True,
        favor_precision=True, # Prevent navigation cruft & short entries from being included
    )
    if not text or len(text) < 200:
        return None
    published: str | None = None
    try:
        md = extract_metadata(downloaded)
        if md and md.date:
            published = md.date  # trafilatura returns "YYYY-MM-DD"
    except Exception:
        pass
    return Article(source=source, url=url, title=title, text=text, published=published)
