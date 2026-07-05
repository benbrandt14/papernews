# papernews/store.py
import os
import sqlite3
from pathlib import Path

# Versioned, append-only migrations. PRAGMA user_version tracks how many
# have been applied; opening a store applies the pending tail inside one
# transaction. NEVER edit or reorder an entry that has shipped — add a
# new one to the end instead.
MIGRATIONS: list[str] = [
    # 1: original schema — generic LLM response cache + manual filters.
    """
    CREATE TABLE IF NOT EXISTS llm_cache (
        id TEXT PRIMARY KEY,
        response TEXT
    );
    CREATE TABLE IF NOT EXISTS filters (
        url TEXT PRIMARY KEY,
        reason TEXT
    );
    """,
]


class SimpleStore:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.environ.get("PAPERNEWS_STATE", "data/state.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self, timeout: float = 10.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=timeout)
        # WAL lets concurrent Prefect task threads read while one writes.
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _migrate(self) -> None:
        with self._connect() as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            for version, ddl in enumerate(MIGRATIONS[current:], start=current + 1):
                conn.executescript(ddl)
                conn.execute(f"PRAGMA user_version = {version}")
            conn.commit()

    def schema_version(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def get_cache(self, cache_key: str) -> str | None:
        """Fetch a cached string response."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT response FROM llm_cache WHERE id = ?", (cache_key,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def set_cache(self, cache_key: str, response: str) -> None:
        """Save or overwrite a cached response. timeout=10 prevents Prefect concurrency crashes."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (id, response) VALUES (?, ?)",
                (cache_key, response),
            )
