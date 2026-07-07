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
    # 2: curiosity queue — open reader questions raised during enrichment,
    # later resolved against the literature and surfaced on the front matter.
    """
    CREATE TABLE IF NOT EXISTS curiosity_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL UNIQUE,
        article_url TEXT,
        created_date TEXT NOT NULL,
        answered_date TEXT,
        answer_title TEXT,
        answer_url TEXT,
        status TEXT NOT NULL DEFAULT 'open'
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

    # --- Curiosity queue ------------------------------------------------------

    def add_question(self, question: str, article_url: str, created_date: str) -> None:
        """Park an open reader question. Duplicate questions are ignored so a
        recurring topic doesn't accumulate identical rows."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO curiosity_queue "
                "(question, article_url, created_date, status) "
                "VALUES (?, ?, ?, 'open')",
                (question, article_url, created_date),
            )

    def open_questions(self, before: str | None = None) -> list[tuple[int, str]]:
        """Still-unanswered questions as (id, question) pairs.

        Pass `before` (an ISO date) to return only questions raised on an
        earlier run — the resolver waits a day before chasing a question so
        it isn't looked up in the same edition that raised it.
        """
        sql = "SELECT id, question FROM curiosity_queue WHERE status = 'open'"
        params: tuple = ()
        if before is not None:
            sql += " AND created_date < ?"
            params = (before,)
        sql += " ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [(int(r[0]), str(r[1])) for r in rows]

    def mark_answered(
        self,
        question_id: int,
        answered_date: str,
        answer_title: str,
        answer_url: str,
    ) -> None:
        """Resolve a queued question with the work that answers it."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE curiosity_queue SET status = 'answered', "
                "answered_date = ?, answer_title = ?, answer_url = ? "
                "WHERE id = ?",
                (answered_date, answer_title, answer_url, question_id),
            )

    def recently_answered(self, limit: int = 3) -> list[tuple[str, str, str]]:
        """Most recently resolved questions as (question, title, url) tuples."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT question, answer_title, answer_url FROM curiosity_queue "
                "WHERE status = 'answered' ORDER BY answered_date DESC, id DESC "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]
