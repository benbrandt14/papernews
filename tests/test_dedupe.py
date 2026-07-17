"""Tests for the article registry (store) and the already-typeset dedupe stage.

The registry separates two timestamps deliberately:
  first_seen_at — when triage first processed the article (heuristics stored)
  typeset_at    — when the article actually landed in a rendered edition

Only typeset_at gates re-publication; a run that fails before the render
must leave its articles eligible for the next edition.
"""

from __future__ import annotations

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")

import pytest

from papernews.core.main import stage6_record_edition, triage_process_dedupe
from papernews.models import ArticleChunk, RawDocument
from papernews.store import SimpleStore


@pytest.fixture
def store(tmp_path):
    return SimpleStore(str(tmp_path / "state.db"))


def _doc(url: str, title: str = "Title", score: int = 3) -> RawDocument:
    return RawDocument(
        source_id=url,
        content_type="rss",
        raw_text="Body",
        title=title,
        category="Science",
        heuristic_score=score,
    )


def _chunk(url: str) -> ArticleChunk:
    return ArticleChunk(
        category="Science",
        source="example.com",
        title="T",
        summary="S",
        body_markdown="B",
        url=url,
    )


# --- Store: article registry ------------------------------------------------


def test_record_processed_sets_first_seen_once(store):
    store.record_processed("u1", "Title A", "Science", 3, "2026-07-01T00:00:00")
    store.record_processed(
        "u1", "Title A (updated)", "Science", 1, "2026-07-02T00:00:00"
    )

    with store._connect() as conn:
        row = conn.execute(
            "SELECT first_seen_at, last_seen_at, seen_count, heuristic_score, title "
            "FROM articles WHERE url = 'u1'"
        ).fetchone()
    assert row[0] == "2026-07-01T00:00:00"  # first sighting preserved
    assert row[1] == "2026-07-02T00:00:00"  # last sighting updated
    assert row[2] == 2
    assert row[3] == 1  # computed heuristic refreshed
    assert row[4] == "Title A (updated)"


def test_typeset_urls_only_reports_stamped(store):
    store.record_processed("u1", "A", "Science", 3, "2026-07-01T00:00:00")
    store.record_processed("u2", "B", "Science", 3, "2026-07-01T00:00:00")
    store.mark_typeset(["u1"], "2026-07-01T06:00:00", "2026-07-01")

    assert store.typeset_urls(["u1", "u2", "u3"]) == {"u1"}
    assert store.typeset_urls([]) == set()


def test_mark_typeset_keeps_original_stamp(store):
    store.record_processed("u1", "A", "Science", 3, "2026-07-01T00:00:00")
    store.mark_typeset(["u1"], "2026-07-01T06:00:00", "2026-07-01")
    store.mark_typeset(["u1"], "2026-07-02T06:00:00", "2026-07-02")

    with store._connect() as conn:
        row = conn.execute(
            "SELECT typeset_at, typeset_edition FROM articles WHERE url = 'u1'"
        ).fetchone()
    assert row == ("2026-07-01T06:00:00", "2026-07-01")


def test_mark_typeset_upserts_unknown_url(store):
    # An article injected downstream of triage still gets stamped.
    store.mark_typeset(["ghost"], "2026-07-01T06:00:00", "2026-07-01")
    assert store.typeset_urls(["ghost"]) == {"ghost"}


# --- Pipeline: dedupe stage + edition recording -----------------------------


def test_dedupe_drops_already_typeset_and_records_fresh(store):
    func = triage_process_dedupe.fn
    store.mark_typeset(["old"], "2026-07-01T06:00:00", "2026-07-01")

    docs = [_doc("old"), _doc("new", score=1)]
    result = func(docs, store=store)

    assert [d.source_id for d in result] == ["new"]
    # Both sightings were recorded with their computed heuristics.
    with store._connect() as conn:
        rows = dict(
            conn.execute("SELECT url, heuristic_score FROM articles").fetchall()
        )
    assert rows["new"] == 1
    assert rows["old"] == 3


def test_second_run_excludes_first_editions_articles(store):
    """The regression: the same article must not appear in two editions."""
    dedupe = triage_process_dedupe.fn
    record = stage6_record_edition.fn

    # Run 1: article passes dedupe, edition renders, gets stamped.
    docs = [_doc("https://example.com/story")]
    survivors = dedupe(docs, store=store)
    assert len(survivors) == 1
    record([_chunk("https://example.com/story")], "2026-07-16", store=store)

    # Run 2: the feed still carries the story; dedupe now drops it.
    assert dedupe(docs, store=store) == []


def test_unrendered_articles_stay_eligible(store):
    """A run that dies before the render must not burn its articles."""
    dedupe = triage_process_dedupe.fn
    docs = [_doc("https://example.com/story")]

    dedupe(docs, store=store)  # processed, but no edition recorded (render failed)
    assert len(dedupe(docs, store=store)) == 1  # still eligible next run
