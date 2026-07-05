# papernews/plugins/rss_plugin.py
import logging

import feedparser
import pluggy
import trafilatura

from papernews.config import AppConfig
from papernews.models import RawDocument

hookimpl = pluggy.HookimplMarker("papernews")
logging.getLogger("trafilatura").setLevel(logging.ERROR)


@hookimpl
def fetch_sources(source_config: AppConfig) -> list[RawDocument]:
    documents = []

    # Only RSS feeds; Hacker News is handled by hn_plugin.
    rss_sources = [s for s in source_config.sources if s.kind == "rss"]

    for source in rss_sources:
        feed = feedparser.parse(source.url)

        for entry in feed.entries:
            url = entry.link
            title = entry.get("title", "Untitled")

            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                continue

            extracted_text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                include_links=True,
                include_images=True,
                favor_precision=True,
            )

            if extracted_text:
                # Grab the date, falling back to 'updated' if 'published' is missing
                pub_date = entry.get("published", entry.get("updated", ""))

                documents.append(
                    RawDocument(
                        source_id=url,
                        content_type="rss",
                        raw_text=extracted_text,
                        title=title,
                        category=source.category,
                        published=pub_date,
                        metadata={"feed_url": source.url},
                    )
                )

    return documents
