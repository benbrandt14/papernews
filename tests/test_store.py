"""Tests for the versioned-migration SQLite store."""

import sqlite3

from papernews.store import MIGRATIONS, SimpleStore


def test_fresh_db_migrates_to_head(tmp_path):
    store = SimpleStore(str(tmp_path / "state.db"))
    assert store.schema_version() == len(MIGRATIONS)


def test_migrations_are_idempotent_on_reopen(tmp_path):
    path = str(tmp_path / "state.db")
    SimpleStore(path)
    store = SimpleStore(path)  # reopen: nothing pending, nothing breaks
    assert store.schema_version() == len(MIGRATIONS)


def test_pre_versioning_db_upgrades_cleanly(tmp_path):
    """Databases created before the migration framework have the v1 tables
    but user_version=0; opening them must not fail and must preserve data."""
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE llm_cache (id TEXT PRIMARY KEY, response TEXT)")
        conn.execute("CREATE TABLE filters (url TEXT PRIMARY KEY, reason TEXT)")
        conn.execute("INSERT INTO llm_cache VALUES ('k', 'v')")

    store = SimpleStore(str(path))
    assert store.schema_version() == len(MIGRATIONS)
    assert store.get_cache("k") == "v"


def test_cache_roundtrip_and_overwrite(tmp_path):
    store = SimpleStore(str(tmp_path / "state.db"))
    assert store.get_cache("missing") is None
    store.set_cache("k", "v1")
    assert store.get_cache("k") == "v1"
    store.set_cache("k", "v2")
    assert store.get_cache("k") == "v2"


def test_wal_mode_enabled(tmp_path):
    store = SimpleStore(str(tmp_path / "state.db"))
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_curiosity_queue_roundtrip(tmp_path):
    store = SimpleStore(str(tmp_path / "state.db"))

    store.add_question("Why is the sky blue?", "https://a", "2026-07-01")
    store.add_question("What is dark matter?", "https://b", "2026-07-01")
    # Duplicate question is ignored, not a second row.
    store.add_question("Why is the sky blue?", "https://a", "2026-07-02")

    open_qs = store.open_questions()
    assert [q for _id, q in open_qs] == [
        "Why is the sky blue?",
        "What is dark matter?",
    ]
    assert store.recently_answered() == []

    sky_id = open_qs[0][0]
    store.mark_answered(sky_id, "2026-07-03", "Rayleigh scattering", "https://doi/1")

    # Answered rows leave the open set and appear in recently_answered.
    assert [q for _id, q in store.open_questions()] == ["What is dark matter?"]
    assert store.recently_answered() == [
        ("Why is the sky blue?", "Rayleigh scattering", "https://doi/1")
    ]
