# papernews/plugins/wiki_plugin.py
import requests
from bs4 import BeautifulSoup
import pluggy
from prefect import get_run_logger
from papernews.models import FrontpageDecorations

hookimpl = pluggy.HookimplMarker("papernews")

@hookimpl
def fetch_decorations(source_config: dict) -> FrontpageDecorations:
    logger = get_run_logger()
    logger.info("Decoration Plugin: Scraping Wikipedia Current Events...")
    bullets = []

    #TODO integrate functions (below) to fetch additional frontpage content
    
    try:
        headers = {"User-Agent": "PapernewsBot/1.0"}
        r = requests.get("https://en.wikipedia.org/wiki/Portal:Current_events", headers=headers, timeout=10)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, 'html.parser')
        current_day_block = soup.find(class_="current-events-content")
        
        if current_day_block:
            for li in current_day_block.find_all('li')[:4]:
                text = li.get_text(strip=True)
                clean_text = ' '.join(word for word in text.split() if not word.startswith('['))
                bullets.append(clean_text)
                
    except Exception as e:
        logger.error(f"Wiki Decorator Error: {e}")
        
    # Pydantic will fall back to the default unavailable string if bullets is empty
    return FrontpageDecorations(world_news=bullets if bullets else None)