import pluggy
import requests
from datetime import datetime
from typing import List
from papernews.models import RawDocument

hookimpl = pluggy.HookimplMarker("papernews")

@hookimpl
def fetch_documents(config: dict) -> List[RawDocument]:
    """Fetches Current Events and Quote of the Day via APIs."""
    documents = []
    today = datetime.now()
    
    print("Fetching Wiki Events and Quotes...")

    # 1. Fetch Current/Historical Events (Wikipedia REST API)
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/{today.month}/{today.day}"
        headers = {"User-Agent": "Papernews/1.0 (Local Daily Digest)"}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            events = resp.json().get("events", [])
            # Grab the top 3 most significant events
            for event in events[:3]:
                text = event.get("text", "")
                year = event.get("year", today.year)
                if text:
                    documents.append(RawDocument(
                        source_id=f"wiki_event_{year}_{hash(text)}",
                        content_type="wiki_event",
                        raw_text=f"{year}: {text}",
                        metadata={"title": f"On This Day: {year}"}
                    ))
    except Exception as e:
        print(f"Wiki Plugin: Failed to fetch events - {e}")

    # 2. Fetch Quote of the Day (Using a reliable open API)
    try:
        resp = requests.get("https://zenquotes.io/api/today", timeout=10)
        if resp.status_code == 200:
            data = resp.json()[0]
            quote_text = f"\"{data.get('q', '')}\"\n— {data.get('a', 'Unknown')}"
            
            documents.append(RawDocument(
                source_id="wiki_quote_today",
                content_type="wiki_quote",
                raw_text=quote_text,
                metadata={"title": "Quote of the Day"}
            ))
    except Exception as e:
        print(f"Wiki Plugin: Failed to fetch quote - {e}")

    return documents