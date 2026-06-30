from __future__ import annotations

import html
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import feedparser
import requests


def _clean_title(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(html.unescape(s).split())


@dataclass
class RawItem:
    source: str
    url: str
    title: str
    surfaced: str | None = None
    rss_content: str | None = None  # ADDED: Capture the raw RSS text


_HN_SEARCH = "https://hn.algolia.com/api/v1/search"


def fetch_hn(
    source_name: str = "Hacker News",
    limit: int = 10,
    since_hours: int = 48,
    min_points: int = 50,
) -> Iterator[RawItem]:
    since = int(time.time() - since_hours * 3600)
    params = {
        "tags": "story",
        "numericFilters": f"created_at_i>{since},points>{min_points}",
        "hitsPerPage": 100,
    }
    r = requests.get(_HN_SEARCH, params=params, timeout=15)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    hits.sort(key=lambda h: h.get("points", 0), reverse=True)

    for h in hits[:limit]:
        title = _clean_title(h.get("title"))
        if not title:
            continue
        url = (
            h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        )
        ts = h.get("created_at_i")
        surfaced = (
            datetime.fromtimestamp(ts, tz=UTC).date().isoformat()
            if ts
            else None
        )
        yield RawItem(source=source_name, url=url, title=title, surfaced=surfaced)


def fetch_wikipedia_events(
    source_name: str = "World news",
    days_back: int = 1,
) -> Iterator[RawItem]:
    from datetime import date as _date
    from datetime import timedelta as _td

    from .wiki import current_events_title, current_events_url

    today = _date.today()
    for delta in range(days_back):
        day = today - _td(days=delta)
        yield RawItem(
            source=source_name,
            url=current_events_url(day),
            title=current_events_title(day),
            surfaced=day.isoformat(),
        )


def fetch_rss(source_name: str, feed_url: str, limit: int = 20) -> Iterator[RawItem]:
    d = feedparser.parse(feed_url)
    for entry in d.entries[:limit]:
        url = getattr(entry, "link", None)
        title = _clean_title(getattr(entry, "title", None))
        if not url or not title:
            continue
        parsed = getattr(entry, "published_parsed", None) or getattr(
            entry, "updated_parsed", None
        )
        surfaced = time.strftime("%Y-%m-%d", parsed) if parsed else None

        # EXTRACT FALLBACK: Many scientific feeds put the abstract directly in the RSS XML
        rss_content = None
        if hasattr(entry, "content"):
            rss_content = entry.content[0].value
        elif hasattr(entry, "summary"):
            rss_content = entry.summary

        yield RawItem(
            source=source_name,
            url=url,
            title=title,
            surfaced=surfaced,
            rss_content=rss_content,
        )
