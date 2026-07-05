# papernews/plugins/wiki_plugin.py
import pluggy
import requests
from bs4 import BeautifulSoup
from prefect import get_run_logger

from papernews.config import AppConfig
from papernews.models import FrontpageDecorations, Quote

hookimpl = pluggy.HookimplMarker("papernews")


@hookimpl
def fetch_decorations(source_config: AppConfig) -> FrontpageDecorations:
    logger = get_run_logger()
    logger.info("Decoration Plugin: Scraping Wikipedia Current Events...")
    bullets = []

    # TODO integrate functions (below) to fetch additional frontpage content

    try:
        headers = {"User-Agent": "PapernewsBot/1.0"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/Portal:Current_events",
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        current_day_block = soup.find(class_="current-events-content")

        if current_day_block:
            for li in current_day_block.find_all("li")[:4]:
                text = li.get_text(strip=True)
                clean_text = " ".join(
                    word for word in text.split() if not word.startswith("[")
                )
                bullets.append(clean_text)

    except Exception as e:
        logger.error(f"Wiki Decorator Error: {e}")

    # A proper quote-of-the-day fetcher arrives with the decorations
    # restore; until then the house quote keeps the front page warm.
    quote = Quote(
        text="Benjamin you stop pickin' the bark off of that tree!",
        author="Grandma Brandt",
    )

    # Fall back to the model's default unavailable string if bullets is empty
    if bullets:
        return FrontpageDecorations(world_news=bullets, quote=quote)
    return FrontpageDecorations(quote=quote)
