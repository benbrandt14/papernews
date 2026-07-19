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


# --- Hardened matching: URL variants, titles, within-run duplicates ----------


def test_within_run_duplicates_collapse(store):
    """Two sources carrying the same story in one run must yield one article.

    The kept occurrence is the first (documents arrive rank-sorted).
    """
    dedupe = triage_process_dedupe.fn
    docs = [
        _doc("https://example.com/story", title="A Grand Discovery", score=1),
        # Same story via an aggregator: tracking params + http + www.
        _doc(
            "http://www.example.com/story?utm_source=hn",
            title="A Grand Discovery",
            score=3,
        ),
        # Different URL entirely, but the identical (normalized) title.
        _doc("https://mirror.net/12345", title="A grand discovery!", score=3),
    ]
    result = dedupe(docs, store=store)
    assert [d.heuristic_score for d in result] == [1]


def test_url_variant_cannot_dodge_typeset_stamp(store):
    """A re-tagged link to an already-published story stays out."""
    dedupe = triage_process_dedupe.fn
    record = stage6_record_edition.fn

    dedupe([_doc("https://example.com/story")], store=store)
    record([_chunk("https://example.com/story")], "2026-07-18", store=store)

    variant = _doc("http://www.example.com/story/?utm_source=rss&fbclid=x")
    assert dedupe([variant], store=store) == []


def test_same_title_different_url_cannot_dodge_typeset_stamp(store):
    """Syndicated copies (same story, different host) stay out too."""
    dedupe = triage_process_dedupe.fn
    record = stage6_record_edition.fn

    original = _doc("https://example.com/story", title="The Definitive Account")
    dedupe([original], store=store)
    record(
        [
            ArticleChunk(
                category="Science",
                source="example.com",
                title="The Definitive Account",
                summary="S",
                body_markdown="B",
                url="https://example.com/story",
            )
        ],
        "2026-07-18",
        store=store,
    )

    syndicated = _doc("https://mirror.net/999", title="The definitive account")
    assert dedupe([syndicated], store=store) == []


def test_v4_backfill_canonicalizes_existing_rows(tmp_path):
    """A pre-v4 registry (raw URLs, no title keys) migrates in place: rows
    are re-keyed canonically and URL-variant twins merge without losing
    the typeset stamp."""
    import sqlite3

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    # Build a v3 database by hand (migrations 1-3 shipped before title_key).
    from papernews.store import MIGRATIONS

    for ddl in MIGRATIONS[:3]:
        conn.executescript(ddl)
    conn.execute("PRAGMA user_version = 3")
    conn.execute(
        "INSERT INTO articles "
        "(url, title, first_seen_at, last_seen_at, typeset_at, typeset_edition) "
        "VALUES ('http://www.example.com/story?utm_source=rss', 'The Old Story Returns', "
        "'t0', 't0', 't1', '2026-07-01')"
    )
    conn.execute(
        "INSERT INTO articles (url, title, first_seen_at, last_seen_at) "
        "VALUES ('https://example.com/story', 'The Old Story Returns', 't2', 't2')"
    )
    conn.commit()
    conn.close()

    store = SimpleStore(str(db))  # migration 4 + backfill run here

    # The two variants merged into one canonical row that kept the stamp.
    assert store.typeset_urls(["https://example.com/story"]) == {
        "https://example.com/story"
    }
    with store._connect() as conn:
        rows = conn.execute("SELECT url, title_key FROM articles").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "the old story returns"
