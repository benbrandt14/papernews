# papernews/plugins/rss_plugin.py
import pluggy
import feedparser
import trafilatura
import logging
from typing import List
from papernews.models import RawDocument

hookimpl = pluggy.HookimplMarker("papernews")
logging.getLogger("trafilatura").setLevel(logging.ERROR)

@hookimpl
def fetch_sources(source_config: dict) -> List[RawDocument]:
    documents = []
    
    # 1. Extract the [[source]] array from the TOML
    sources = source_config.get("source", [])
    
    # 2. Filter for only RSS feeds (Hacker News will be handled by a separate hn_plugin)
    rss_sources = [s for s in sources if s.get("kind") == "rss"]
    
    for source in rss_sources:
        feed_url = source.get("url")
        category = source.get("category", "Uncategorized")
        feed = feedparser.parse(feed_url)
        
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
                        metadata={
                            "title": title,
                            "feed_url": feed_url,
                            "category": category,
                            "published": pub_date # Safely captured
                        }
                    )
                )
                
    return documents