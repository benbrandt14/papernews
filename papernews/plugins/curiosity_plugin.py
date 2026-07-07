# papernews/plugins/curiosity_plugin.py
"""The curiosity queue: the paper asks questions, then answers them later.

Two hooks make one loop:

  * enrich_articles (Stage 3.5) — for the day's lead stories, ask the LLM
    for a few specific, researchable questions, park them in the queue, then
    try to resolve *previously* open questions against the OpenAlex corpus.
  * fetch_decorations (Stage 4) — surface the freshly-resolved pairs as a
    front-matter "Answered from the queue" box.

Both halves degrade to no-ops: no LLM configured means no new questions; an
OpenAlex outage just leaves questions open for a future run. Nothing here can
fail the edition.
"""

from __future__ import annotations

import logging
from datetime import date

import pluggy
import requests
from prefect import get_run_logger
from prefect.exceptions import MissingContextError

from papernews.config import AppConfig, get_settings
from papernews.core.backends import get_backend
from papernews.models import (
    ArticleChunk,
    Curiosity,
    FrontpageDecorations,
    LLMOpenQuestions,
)
from papernews.store import SimpleStore

hookimpl = pluggy.HookimplMarker("papernews")

# Cost guards: only the strongest few stories earn questions, and only a
# handful of open questions are chased per run.
MAX_ARTICLES_QUESTIONED = 5
MAX_QUESTIONS_PER_ARTICLE = 3
MAX_LOOKUPS_PER_RUN = 8
# OpenAlex ranks `search` hits; below this the top hit is too weak to call
# an answer, so the question stays open.
MIN_RELEVANCE = 5.0

_OPENALEX_WORKS = "https://api.openalex.org/works"
_UA = "PapernewsBot/1.0 (mailto:papernews@example.com)"


def _logger() -> logging.Logger | logging.LoggerAdapter:
    """Prefect's run logger inside a flow; a plain logger otherwise.

    Enrichment fires inside a Prefect task at runtime, but the contract and
    unit tests call the hooks directly — outside any run context — so fall
    back rather than crash.
    """
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


_QUESTION_SYSTEM = (
    "You are a curious research editor. Given an article, produce up to three "
    "specific, self-contained questions whose answers a reader would seek in "
    "the scientific literature. Each question must stand alone without the "
    "article's context. Return them as the `questions` array."
)


def _generate_questions(article: ArticleChunk, store: SimpleStore) -> list[str]:
    """Ask the backend for open questions about one article (cached by URL)."""
    cache_key = f"questions_{article.url}"
    cached = store.get_cache(cache_key)
    if cached is not None:
        return LLMOpenQuestions.model_validate_json(cached).questions

    backend = get_backend(get_settings())
    prompt = f"Title: {article.title}\n\nSummary: {article.summary}"
    raw, _telemetry = backend.structured(
        contents=prompt,
        system_instruction=_QUESTION_SYSTEM,
        temperature=0.4,
        schema=LLMOpenQuestions,
    )
    if raw is None:
        return []

    parsed = LLMOpenQuestions.model_validate_json(raw)
    questions = [q.strip() for q in parsed.questions if q.strip()][
        :MAX_QUESTIONS_PER_ARTICLE
    ]
    # Re-serialize the trimmed list so the cache matches what we return.
    store.set_cache(cache_key, LLMOpenQuestions(questions=questions).model_dump_json())
    return questions


def _lookup_openalex(question: str) -> tuple[str, str] | None:
    """Return (title, url) of the best OpenAlex work for a question, or None
    when nothing clears the relevance bar."""
    resp = requests.get(
        _OPENALEX_WORKS,
        params={"search": question, "per-page": "1"},
        headers={"User-Agent": _UA},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None

    top = results[0]
    title = (top.get("title") or "").strip()
    if not title or top.get("relevance_score", 0.0) < MIN_RELEVANCE:
        return None

    # Prefer a resolvable DOI; fall back to the canonical OpenAlex landing.
    url = top.get("doi") or top.get("id")
    if not url:
        return None
    return title, url


@hookimpl
def enrich_articles(
    articles: list[ArticleChunk],
    source_config: AppConfig,
    store: SimpleStore,
) -> None:
    logger = _logger()
    today = date.today().isoformat()

    # 1. Raise new questions on the day's strongest stories (LLM only).
    if get_settings().llm_enabled:
        for article in articles[:MAX_ARTICLES_QUESTIONED]:
            try:
                questions = _generate_questions(article, store)
            except Exception as e:  # noqa: BLE001 — enrichment never fails a run
                logger.warning(f"Curiosity: question generation failed: {e}")
                continue
            article.enrichment.open_questions = questions
            for q in questions:
                store.add_question(q, article.url, today)

    # 2. Chase down questions parked on earlier runs (not today's fresh ones).
    for question_id, question in store.open_questions(before=today)[
        :MAX_LOOKUPS_PER_RUN
    ]:
        try:
            hit = _lookup_openalex(question)
        except Exception as e:  # noqa: BLE001 — a lookup outage isn't fatal
            logger.warning(f"Curiosity: OpenAlex lookup failed: {e}")
            continue
        if hit is not None:
            title, url = hit
            store.mark_answered(question_id, today, title, url)
            logger.info(f"Curiosity: answered {question[:50]!r}")


@hookimpl
def fetch_decorations(source_config: AppConfig) -> FrontpageDecorations:
    """Surface the most recently answered questions on the front matter."""
    rows = SimpleStore().recently_answered(limit=3)
    curiosities = [
        Curiosity(question=q, answer_title=t, answer_url=u) for q, t, u in rows
    ]
    return FrontpageDecorations(curiosities=curiosities)
