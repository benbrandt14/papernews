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
