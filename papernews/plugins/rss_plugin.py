import pluggy
import feedparser
import trafilatura
from typing import List
from papernews.models import RawDocument

hookimpl = pluggy.HookimplMarker("papernews")

@hookimpl
def fetch_documents(config: dict) -> List[RawDocument]:
    """Real RSS fetcher using feedparser and trafilatura."""
    documents = []
    
    # Extract feeds from the parsed sources.toml configuration.
    feed_urls = []
    for category, data in config.items():
        if isinstance(data, dict) and "feeds" in data:
            feed_urls.extend(data["feeds"])
        elif isinstance(data, list):
            feed_urls.extend(data)

    for feed_item in feed_urls:
        # Check if the feed is a rich dictionary from the TOML or a plain string
        url = feed_item.get("url") if isinstance(feed_item, dict) else feed_item
        
        # Skip if we couldn't resolve a valid string URL
        if not url or not isinstance(url, str):
            continue

        print(f"Parsing feed: {url}")
        feed = feedparser.parse(url)
        
        # Limit to top 3 entries per feed to avoid API overload during testing
        for entry in feed.entries[:3]:
            # Some RSS feeds use 'link', others put it elsewhere. fallback to empty string.
            article_url = getattr(entry, "link", "")
            if not article_url:
                continue
            
            # Bypass Academic DOIs (these belong to the academic_plugin)
            if "doi.org" in article_url:
                continue
                
            # Scrape the article body
            downloaded = trafilatura.fetch_url(article_url)
            if not downloaded:
                continue
                
            text = trafilatura.extract(downloaded)
            
            # Triage Funnel: Drop articles that are just short summaries/errors
            if text and len(text) > 200: 
                doc = RawDocument(
                    source_id=article_url,
                    content_type="rss",
                    raw_text=text,
                    metadata={
                        "title": entry.get("title", "Untitled"),
                        "url": article_url,
                        "author": entry.get("author", "Unknown"),
                    }
                )
                documents.append(doc)
            
    return documents