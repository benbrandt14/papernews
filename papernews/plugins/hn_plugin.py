import time
import requests
import pluggy
import trafilatura
from prefect import get_run_logger
from papernews.models import RawDocument

hookimpl = pluggy.HookimplMarker("papernews")

_HN_SEARCH = "https://hn.algolia.com/api/v1/search"


@hookimpl
def fetch_sources(source_config: dict) -> list[RawDocument]:
    logger = get_run_logger()
    documents = []

    # Check if the config contains any sources requesting Hacker News
    sources = source_config.get("source", [])
    hn_sources = [s for s in sources if s.get("kind") == "hn"]

    if not hn_sources:
        return []

    for src in hn_sources:
        source_name = src.get("name", "Hacker News")
        category = src.get("category", "Hacker News")
        limit = src.get("limit", 10)
        since_hours = src.get("since_hours", 48)
        min_points = src.get("min_points", 50)

        since = int(time.time() - since_hours * 3600)
        since = int(time.time() - since_hours * 3600)
        params = {
            "tags": "story",
            # Pass as a list so 'requests' parses it as &numericFilters=X&numericFilters=Y
            "numericFilters": [f"created_at_i>{since}", f"points>{min_points}"],
            "hitsPerPage": 100,
        }

        try:
            r = requests.get(_HN_SEARCH, params=params, timeout=15)
            r.raise_for_status()
            hits = r.json().get("hits", [])

            # Sort highest points first
            hits.sort(key=lambda h: h.get("points", 0), reverse=True)

            for h in hits[:limit]:
                title = h.get("title", "Unknown Title")
                # Fallback to the HN comment thread if it's a text post (Ask HN)
                url = (
                    h.get("url")
                    or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
                )

                logger.info(f"HN Ingestion: Scraping '{title[:40]}...'")

                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    continue

                # Extract while aggressively preserving the media your layout needs
                raw_text = trafilatura.extract(
                    downloaded, include_images=True, include_links=True
                )

                # Drop empty or heavily gated articles
                if not raw_text or len(raw_text) < 800:
                    continue

                doc = RawDocument(
                    source_id=url,
                    content_type="rss",  # Route through standard markdown formatter
                    raw_text=raw_text,
                    metadata={
                        "title": title,
                        "category": category,
                        "feed_url": "https://news.ycombinator.com",
                        "points": h.get("points"),
                    },
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error fetching Hacker News: {e}")

    return documents
