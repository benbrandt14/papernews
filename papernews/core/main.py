# papernews/core/main.py
import re
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

from prefect import flow, task

from papernews.ai_detect import score_text
from papernews.config import AppConfig, Preferences, get_settings, load_config
from papernews.core.router import (
    llm_format_body,
    llm_select_article,
    llm_summarize_article,
)
from papernews.markdown_ir import parse_markdown
from papernews.models import (
    ArticleChunk,
    FrontpageDecorations,
    FunnelStats,
    RawDocument,
    RenderContext,
    Telemetry,
)
from papernews.plugins.registry import get_plugin_manager
from papernews.render import build_pdf
from papernews.store import SimpleStore

# Drop raw URLs, short stubs, or noisy topics before any scoring happens.
NOISE_PATTERNS = [
    r"(?i)mice models?",
    r"(?i)rat models?",
    r"^(https?://[^\s]+)$",  # Drops articles that are literally just a URL string
]
_NOISE_RES = [re.compile(p) for p in NOISE_PATTERNS]


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


@task(name="Stage 1: Ingestion", retries=2, retry_delay_seconds=5)
def stage1_ingestion(config: AppConfig) -> list[RawDocument]:
    """Dynamically loads all plugins and fetches RawDocuments."""
    pm = get_plugin_manager()

    # pm.hook.fetch_sources returns a list of lists (one per plugin)
    plugin_results = pm.hook.fetch_sources(source_config=config)

    # Flatten results
    documents = [doc for sublist in plugin_results for doc in sublist]
    print(f"Loaded {len(documents)} raw documents from plugins.")
    return documents


@task(name="Stage 2A: Deterministic Blacklist Filter")
def triage_process_a_filter(
    documents: list[RawDocument], prefs: Preferences
) -> list[RawDocument]:
    filtered_docs = []

    if prefs.blacklist_words:
        pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, prefs.blacklist_words)) + r")\b",
            re.IGNORECASE,
        )
    else:
        pattern = None

    for doc in documents:
        if len(doc.raw_text) < 800 and doc.content_type == "rss":
            continue

        if (
            len(doc.raw_text) > prefs.max_char_length
            and doc.content_type != "academic_pdf"
        ):
            continue

        if pattern and (pattern.search(doc.raw_text) or pattern.search(doc.title)):
            continue

        # Built-in noise patterns: irrelevant topics in the title, or a
        # body that is nothing but a bare URL.
        stripped = doc.raw_text.strip()
        if any(p.search(doc.title) or p.search(stripped) for p in _NOISE_RES):
            continue

        filtered_docs.append(doc)

    return filtered_docs


@task(name="Stage 2B: Local Ranking")
def triage_process_b_rank(
    documents: list[RawDocument], prefs: Preferences
) -> list[RawDocument]:
    def heuristic_score(doc: RawDocument) -> int:
        title = doc.title.lower()
        for interest in prefs.interest:
            keyword = interest.split()[0].lower()
            if keyword in title:
                return 1
        return 3

    # Return scored copies — tasks must not mutate their inputs.
    scored = [
        doc.model_copy(update={"heuristic_score": heuristic_score(doc)})
        for doc in documents
    ]
    return sorted(scored, key=lambda d: d.heuristic_score)


@task(name="Stage 2B.5: AI-Likeness Derank")
def triage_process_b5_ai_derank(
    documents: list[RawDocument], prefs: Preferences
) -> tuple[list[RawDocument], int, int]:
    """Derank (and optionally drop) statistically AI-flavored documents.

    Runs between local ranking (2B) and the category budget (2C) so that
    flagged documents fall below the budget cut line instead of costing
    LLM tokens. Scoring (papernews.ai_detect, adapted from
    lyc8503/AITextDetector) is deterministic and local; metrics attach to
    every surviving document so the renderer can print them per article.

    Returns (documents, deranked_count, dropped_count).
    """
    if not prefs.ai_detection_enabled:
        return documents, 0, 0

    deranked = 0
    dropped = 0
    scored_docs: list[RawDocument] = []
    for doc in documents:
        metrics = score_text(doc.raw_text)
        update: dict = {"ai_metrics": metrics}
        if metrics.reliable:
            if (
                prefs.ai_drop_threshold is not None
                and metrics.ai_likelihood >= prefs.ai_drop_threshold
            ):
                dropped += 1
                continue
            if metrics.ai_likelihood >= prefs.ai_derank_threshold:
                update["heuristic_score"] = (
                    doc.heuristic_score + prefs.ai_derank_penalty
                )
                deranked += 1
        # Return updated copies — tasks must not mutate their inputs.
        scored_docs.append(doc.model_copy(update=update))

    # Stable sort: deranked docs sink, everything else keeps 2B's order.
    return sorted(scored_docs, key=lambda d: d.heuristic_score), deranked, dropped


@task(name="Stage 2C: Category Limit Enforcer")
def triage_process_c_budget(
    documents: list[RawDocument], limits: dict[str, int], prefs: Preferences
) -> list[RawDocument]:
    """
    Enforces the [category_limits] strictly in Python so we never
    pay the LLM to process more articles than the PDF requires.
    """
    surviving_docs = []
    category_counts: dict[str, int] = {}

    for doc in documents:
        # Look up the specific limit for this category, or fall back to default
        cat_limit = limits.get(doc.category, prefs.default_category_limit)

        current_count = category_counts.get(doc.category, 0)

        # If we haven't hit the cap for this category, keep the article
        if current_count < cat_limit:
            surviving_docs.append(doc)
            category_counts[doc.category] = current_count + 1

    return surviving_docs


@task(name="Stage 3: Hybrid Construction")
def stage3_hybrid_construction(
    documents: list[RawDocument], prefs: Preferences
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
        if doc.published:
            try:
                dt = parsedate_to_datetime(doc.published)
                rel_time = get_human_time(dt)
            except (ValueError, TypeError):
                pass

        # Parse Source Domain
        domain_source = "Unknown"
        try:
            domain_source = urlparse(
                doc.metadata.get("feed_url", doc.source_id)
            ).netloc.replace("www.", "")
        except ValueError:
            pass

        chunk = ArticleChunk(
            content_type=doc.content_type,
            category=doc.category,
            source=domain_source,
            title=doc.title or "Untitled",
            summary=summary_text,
            body_markdown=formatted_markdown,
            url=doc.source_id,
            date=rel_time,
            published_date=doc.published,
            relative_time=rel_time,
            telemetry=article_telemetry,
            annotations=[],
            ai_metrics=doc.ai_metrics,
        )
        chunk.blocks = parse_markdown(formatted_markdown)
        processed_chunks.append(chunk)

    return processed_chunks, total_run_telemetry


@task(name="Stage 3.5: Enrichment")
def stage3_5_enrichment(
    articles: list[ArticleChunk], config: AppConfig
) -> list[ArticleChunk]:
    """Whole-day, cross-article enrichment pass.

    Plugins implementing `enrich_articles` see every surviving article at
    once and attach sidecar data in place (annotations, entities, scores).
    No built-in plugin implements it yet — this is the extension point the
    salience/interlinking/marginalia features plug into.
    """
    pm = get_plugin_manager()
    pm.hook.enrich_articles(
        articles=articles, source_config=config, store=SimpleStore()
    )
    return articles


@task(name="Stage 4B: Template Decorations")
def stage4b_fetch_decorations(config: AppConfig) -> FrontpageDecorations:
    pm = get_plugin_manager()

    # Execute hooks (returns a list of FrontpageDecorations models)
    results = pm.hook.fetch_decorations(source_config=config)

    # Merge all plugin models together (later plugins overwrite earlier ones)
    merged: dict = {}
    for res in results:
        merged.update(res.model_dump(exclude_unset=True))

    return FrontpageDecorations.model_validate(merged)


@task(name="Stage 5: Bespoke Renderer")
def stage5_bespoke_render(
    articles: list[ArticleChunk],
    total_telemetry: Telemetry,
    decorations: FrontpageDecorations,
    stats: FunnelStats,
) -> Path:
    settings = get_settings()
    out_dir = settings.output
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = RenderContext(
        date=date.today().strftime("%Y-%m-%d"),
        generation_time=datetime.now().strftime("%b %d, %Y at %I:%M %p"),
        total_tokens=total_telemetry.formatted_tokens,
        total_cost=total_telemetry.formatted_cost,
        articles=articles,
        decorations=decorations,
        stats=stats,
        lead_article_index=0 if articles else None,
    )

    return build_pdf(ctx, out_dir)


@flow(name="Papernews Processing Flow", log_prints=True)
def run_papernews(config: AppConfig) -> Path:
    raw_docs = stage1_ingestion(config)

    filtered = triage_process_a_filter(raw_docs, config.preferences)
    ranked = triage_process_b_rank(filtered, config.preferences)
    screened, ai_deranked, ai_dropped = triage_process_b5_ai_derank(
        ranked, config.preferences
    )
    budgeted = triage_process_c_budget(
        screened, config.category_limits, config.preferences
    )

    article_chunks, total_telemetry = stage3_hybrid_construction(
        budgeted, config.preferences
    )

    enriched = stage3_5_enrichment(article_chunks, config)

    decorations = stage4b_fetch_decorations(config)

    stats = FunnelStats(
        ingested=len(raw_docs),
        after_filter=len(filtered),
        after_budget=len(budgeted),
        selected=len(enriched),
        ai_deranked=ai_deranked,
        ai_dropped=ai_dropped,
    )

    pdf_path = stage5_bespoke_render(enriched, total_telemetry, decorations, stats)

    return pdf_path


if __name__ == "__main__":
    settings = get_settings()
    if not settings.config.exists():
        print(f"Error: {settings.config.absolute()} not found.")
        raise SystemExit(1)

    app_config = load_config(settings.config)
    print(f"Loaded config with {len(app_config.sources)} sources.")
    run_papernews(config=app_config)
