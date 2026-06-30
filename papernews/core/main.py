# papernews/core/main.py
import os
import re
import tomllib
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import pluggy
from prefect import flow, task

from papernews.core.router import (
    llm_format_body,
    llm_select_article,
    llm_summarize_article,
)
from papernews.models import ArticleChunk, FrontpageDecorations, RawDocument, Telemetry
from papernews.render import build_pdf

# Configuration
MAX_BUDGET = 12
# Drop raw URLs, short stubs, or noisy topics
NOISE_PATTERNS = [
    r"(?i)mice models?",
    r"(?i)rat models?",
    r"^(https?://[^\s]+)$",  # Drops articles that are literally just a URL string
]


def get_human_time(dt: datetime) -> str:
    """Converts a parsed datetime into '2 days ago', 'today', etc."""
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)  # Assume UTC if naive
    diff = now - dt

    if diff.days == 0:
        return "today"
    elif diff.days == 1:
        return "yesterday"
    elif diff.days < 30:
        return f"{diff.days} days ago"
    elif diff.days < 365:
        months = diff.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = round(diff.days / 365.0, 1)
        # Drop the .0 if it's exactly 1, 2, etc.
        return f"{int(years) if years.is_integer() else years} year{'s' if years != 1 else ''} ago"


@task(name="Stage 1: Ingestion")
def stage1_ingestion(source_config: dict) -> list[RawDocument]:
    """Dynamically loads all plugins and fetches RawDocuments."""
    pm = pluggy.PluginManager("papernews")

    # In a real app, you would load hookspecs and module plugins here
    # For now, we manually register the RSS plugin module
    from papernews.plugins import hn_plugin, rss_plugin

    pm.register(rss_plugin)
    pm.register(hn_plugin)

    # pm.hook.fetch_sources returns a list of lists (one per plugin)
    plugin_results = pm.hook.fetch_sources(source_config=source_config)

    # Flatten results
    documents = [doc for sublist in plugin_results for doc in sublist]
    print(f"Loaded {len(documents)} raw documents from plugins.")
    return documents


@task(name="Stage 2A: Deterministic Blacklist Filter")
def triage_process_a_filter(
    documents: list[RawDocument], prefs: dict
) -> list[RawDocument]:
    filtered_docs = []
    blacklist = prefs.get("blacklist_words", [])
    max_char_length = prefs.get("max_char_length", 20000)

    if blacklist:
        pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, blacklist)) + r")\b", re.IGNORECASE
        )
    else:
        pattern = None

    for doc in documents:
        if len(doc.raw_text) < 800 and doc.content_type == "rss":
            continue

        if len(doc.raw_text) > max_char_length and doc.content_type != "academic_pdf":
            continue

        if pattern:
            title = doc.metadata.get("title", "")
            if pattern.search(doc.raw_text) or pattern.search(title):
                continue

        filtered_docs.append(doc)

    return filtered_docs


@task(name="Stage 2B: Local Ranking")
def triage_process_b_rank(
    documents: list[RawDocument], prefs: dict
) -> list[RawDocument]:
    interests = prefs.get("interest", [])

    def heuristic_score(doc: RawDocument) -> int:
        text = doc.raw_text.lower()
        title = doc.metadata.get("title", "").lower()
        for interest in interests:
            keyword = interest.split()[0].lower()
            if keyword in title:
                return 1
        return 3

    # Attach the score to the metadata before returning
    for doc in documents:
        doc.metadata["heuristic_score"] = heuristic_score(doc)

    return sorted(documents, key=lambda d: d.metadata["heuristic_score"])


@task(name="Stage 2C: Category Limit Enforcer")
def triage_process_c_budget(
    documents: list[RawDocument], limits: dict, prefs: dict
) -> list[RawDocument]:
    """
    Enforces the [category_limits] strictly in Python so we never
    pay the LLM to process more articles than the PDF requires.
    """
    default_limit = prefs.get("default_category_limit", 1)
    surviving_docs = []
    category_counts = {}

    for doc in documents:
        cat = doc.metadata.get("category", "Uncategorized")
        # Look up the specific limit for this category, or fall back to default
        cat_limit = limits.get(cat, default_limit)

        current_count = category_counts.get(cat, 0)

        # If we haven't hit the cap for this category, keep the article
        if current_count < cat_limit:
            surviving_docs.append(doc)
            category_counts[cat] = current_count + 1

    return surviving_docs


@task(name="Stage 3: Hybrid Construction")
def stage3_hybrid_construction(
    documents: list[RawDocument], prefs: dict
) -> tuple[list[ArticleChunk], Telemetry]:
    processed_chunks = []
    total_run_telemetry = Telemetry()

    for doc in documents:
        # 1. Selection
        is_selected, t1 = llm_select_article(doc, prefs)
        total_run_telemetry += t1

        if not is_selected:
            continue

        # 2. Summarization & Formatting
        summary_text, t2 = llm_summarize_article(doc)
        formatted_markdown, t3 = llm_format_body(doc)

        # Aggregate
        total_run_telemetry += t2 + t3
        article_telemetry = t1 + t2 + t3

        # Parse Date
        rel_time = ""
        pub_date_str = doc.metadata.get("published", "")
        if pub_date_str:
            try:
                dt = parsedate_to_datetime(pub_date_str)
                rel_time = get_human_time(dt)
            except Exception:
                pass

        # Parse Source Domain
        domain_source = "Unknown"
        try:
            domain_source = urlparse(
                doc.metadata.get("feed_url", doc.source_id)
            ).netloc.replace("www.", "")
        except:
            pass

        chunk = ArticleChunk(
            content_type=doc.content_type,
            category=doc.metadata.get("category", "Uncategorized"),
            source=domain_source,
            title=doc.metadata.get("title", "Untitled"),
            summary=summary_text,
            body_markdown=formatted_markdown,
            url=doc.source_id,
            published_date=pub_date_str,
            relative_time=rel_time,
            telemetry=article_telemetry,
            annotations=[],
        )
        processed_chunks.append(chunk)

    return processed_chunks, total_run_telemetry


@task(name="Stage 4B: Template Decorations")
def stage4b_fetch_decorations(source_config: dict) -> dict:
    pm = pluggy.PluginManager("papernews")

    from papernews.plugins import wiki_plugin

    pm.register(wiki_plugin)

    # Execute hooks (returns a list of FrontpageDecorations models)
    results = pm.hook.fetch_decorations(source_config=source_config)

    # Start with an empty, default model
    master_decorations = FrontpageDecorations()

    # Merge all plugin models together (overwriting defaults with actual data)
    for res in results:
        for field, value in res.model_dump(exclude_unset=True).items():
            setattr(master_decorations, field, value)

    # Convert back to a dictionary for the Jinja/Typst renderer
    return master_decorations.model_dump()


@task(name="Stage 5: Bespoke Renderer")
def stage5_bespoke_render(
    articles: list[ArticleChunk], total_telemetry: Telemetry, decorations
) -> Path:
    out_dir = Path(os.getcwd()) / "output"
    out_dir.mkdir(exist_ok=True)

    today_str = date.today().strftime("%Y-%m-%d")

    generation_timestamp = datetime.now().strftime("%b %d, %Y at %I:%M %p")

    decorations = {
        "generation_time": generation_timestamp,
        "total_tokens": total_telemetry.formatted_tokens,
        "total_cost": total_telemetry.formatted_cost,
        "quote": {
            "text": "Benjamin you stop pickin' the bark off of that tree!",
            "author": "Grandma Brandt",
        },
        "world_news": [],
        "dyk": [],
    }

    pdf_path = build_pdf(
        date=today_str,
        articles=articles,
        out_dir=out_dir,
        decorations=decorations,
    )
    return pdf_path


@flow(name="Papernews Processing Flow", log_prints=True)
def run_papernews(source_config: dict):
    # Extract Configs
    prefs = source_config.get("preferences", {})
    limits = source_config.get("category_limits", {})

    raw_docs = stage1_ingestion(source_config)

    filtered = triage_process_a_filter(raw_docs, prefs)
    ranked = triage_process_b_rank(filtered, prefs)
    budgeted = triage_process_c_budget(ranked, limits, prefs)

    article_chunks, total_telemetry = stage3_hybrid_construction(budgeted, prefs)

    decorations = stage4b_fetch_decorations(source_config)

    pdf_path = stage5_bespoke_render(article_chunks, total_telemetry, decorations)

    return pdf_path


if __name__ == "__main__":
    config_path = Path("sources.toml")

    if not config_path.exists():
        print(f"Error: {config_path.absolute()} not found.")
        exit(1)

    with open(config_path, "rb") as f:
        actual_config = tomllib.load(f)

    print(f"Loaded config with {len(actual_config.get('source', []))} sources.")
    run_papernews(source_config=actual_config)
