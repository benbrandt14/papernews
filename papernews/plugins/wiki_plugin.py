# papernews/plugins/wiki_plugin.py
"""Front-page decorations: Wikipedia world news, Wikiquote quote of the
day, and 'Did you know...' nuggets.

Every fetch degrades gracefully: a failed source leaves that field at its
model default (and the house quote keeps the masthead warm) rather than
failing the run.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pluggy
import requests
import trafilatura
from bs4 import BeautifulSoup
from prefect import get_run_logger

from papernews.config import AppConfig
from papernews.models import FrontpageDecorations, Quote

hookimpl = pluggy.HookimplMarker("papernews")

_UA = "PapernewsBot/1.0"

_FALLBACK_QUOTE = Quote(
    text="Benjamin you stop pickin' the bark off of that tree!",
    author="Grandma Brandt",
)

# The QOTD page uses {{Wikiquote:Quote of the day/Template | quote = ... | author = ...}}
_QUOTE_FIELD_RE = re.compile(
    r"\|\s*quote\s*=\s*(?:<!--.*?-->)?\s*(.+?)(?=\n\s*\|\s*\w+\s*=|\n*\}\})",
    re.IGNORECASE | re.DOTALL,
)
_AUTHOR_FIELD_RE = re.compile(
    r"\|\s*author\s*=\s*(.+?)(?=\n\s*\|\s*\w+\s*=|\n*\}\})", re.IGNORECASE | re.DOTALL
)


def _strip_wiki(s: str) -> str:
    """Strip basic wikitext to plain text."""
    s = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", s)  # [[a|b]] → b
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)  # [[a]] → a
    s = re.sub(r"'''([^']+)'''", r"\1", s)  # '''bold''' → bold
    s = re.sub(r"''([^']+)''", r"\1", s)  # ''italics'' → italics
    # Tags → space so `moon<br/>Under` doesn't glue into `moonUnder`.
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # The template wraps the quote in quotation marks itself; trim any from
    # the source so we don't get doubled.
    return s.strip("\"'“”‘’").strip()


def _fetch_quote_of_day(max_words: int = 40, days_back: int = 14) -> Quote | None:
    """Wikiquote's Quote of the Day, searching back up to `days_back` days
    for one of at most `max_words` words (some picks are essay-length)."""
    for delta in range(days_back):
        d = date.today() - timedelta(days=delta)
        page = f"Wikiquote:Quote_of_the_day/{d.strftime('%B')}_{d.day},_{d.year}"
        try:
            r = requests.get(
                "https://en.wikiquote.org/w/api.php",
                params={
                    "action": "parse",
                    "page": page,
                    "format": "json",
                    "prop": "wikitext",
                },
                headers={"User-Agent": _UA},
                timeout=15,
            )
            data = r.json()
            wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
            if not wikitext:
                continue
            q = _QUOTE_FIELD_RE.search(wikitext)
            a = _AUTHOR_FIELD_RE.search(wikitext)
            if not q:
                continue
            quote = _strip_wiki(q.group(1))
            author = _strip_wiki(a.group(1)) if a else ""
            wc = len(quote.split())
            if wc < 3 or wc > max_words:
                continue
            return Quote(text=quote, author=author or "Anonymous")
        except Exception:
            continue
    return None


def _fetch_did_you_know(limit: int = 4) -> list[str]:
    """Pull 'Did you know ...' bullets from today's Wikipedia Main Page."""
    try:
        downloaded = trafilatura.fetch_url("https://en.wikipedia.org/wiki/Main_Page")
        text = trafilatura.extract(downloaded, include_comments=False) or ""
    except Exception:
        return []
    idx = text.find("Did you know")
    if idx < 0:
        return []
    items: list[str] = []
    for line in text[idx:].splitlines()[1:]:
        s = line.strip()
        if not s:
            if items:
                break
            continue
        if s.startswith("- "):
            body = s[2:].lstrip(". ").strip()
            # Strip leading "that " for terser display
            body = re.sub(r"^that\s+", "", body, flags=re.IGNORECASE)
            items.append(body)
        elif items:
            break
    return items[:limit]


def _fetch_world_news(limit: int = 4) -> list[str]:
    """Today's Wikipedia Current Events bullets."""
    bullets: list[str] = []
    headers = {"User-Agent": _UA}
    r = requests.get(
        "https://en.wikipedia.org/wiki/Portal:Current_events",
        headers=headers,
        timeout=10,
    )
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    current_day_block = soup.find(class_="current-events-content")

    if current_day_block:
        for li in current_day_block.find_all("li")[:limit]:
            text = li.get_text(strip=True)
            clean_text = " ".join(
                word for word in text.split() if not word.startswith("[")
            )
            bullets.append(clean_text)
    return bullets


@hookimpl
def fetch_decorations(source_config: AppConfig) -> FrontpageDecorations:
    logger = get_run_logger()
    logger.info("Decoration Plugin: fetching world news / quote / DYK...")

    bullets: list[str] = []
    try:
        bullets = _fetch_world_news()
    except Exception as e:
        logger.error(f"Wiki world-news error: {e}")

    quote = None
    try:
        quote = _fetch_quote_of_day()
    except Exception as e:
        logger.error(f"Wikiquote QOTD error: {e}")

    dyk: list[str] = []
    try:
        dyk = _fetch_did_you_know()
    except Exception as e:
        logger.error(f"Wiki DYK error: {e}")

    decorations = FrontpageDecorations(
        quote=quote or _FALLBACK_QUOTE,
        dyk=dyk,
    )
    if bullets:
        decorations.world_news = bullets
    return decorations
