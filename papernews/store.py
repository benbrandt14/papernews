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
    # 3: article registry — one row per article URL the triage pipeline has
    # ever processed. Separates "first seen/processed" (first_seen_at, set the
    # first time triage scores the article) from "actually typeset into an
    # edition" (typeset_at, set only after a PDF render succeeds). The
    # pipeline skips any URL with a non-NULL typeset_at, so an article can
    # appear in at most one edition. Computed attributes (heuristic_score)
    # are stored alongside so runs are auditable after the fact.
    """
    CREATE TABLE IF NOT EXISTS articles (
        url TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        category TEXT NOT NULL DEFAULT '',
        heuristic_score INTEGER,
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        seen_count INTEGER NOT NULL DEFAULT 1,
        typeset_at TEXT,
        typeset_edition TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_articles_typeset
        ON articles (typeset_at) WHERE typeset_at IS NOT NULL;
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

    # --- Article registry -----------------------------------------------------

    def record_processed(
        self,
        url: str,
        title: str,
        category: str,
        heuristic_score: int,
        seen_at: str,
    ) -> None:
        """Record that triage processed an article on this run.

        First sighting sets first_seen_at; later sightings only bump
        last_seen_at/seen_count and refresh the computed attributes
        (title and heuristic score can legitimately change between runs).
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO articles "
                "(url, title, category, heuristic_score, first_seen_at, "
                " last_seen_at, seen_count) "
                "VALUES (?, ?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(url) DO UPDATE SET "
                "  title = excluded.title, "
                "  category = excluded.category, "
                "  heuristic_score = excluded.heuristic_score, "
                "  last_seen_at = excluded.last_seen_at, "
                "  seen_count = seen_count + 1",
                (url, title, category, heuristic_score, seen_at, seen_at),
            )

    def typeset_urls(self, urls: list[str]) -> set[str]:
        """The subset of `urls` already typeset into a previous edition."""
        if not urls:
            return set()
        placeholders = ",".join("?" * len(urls))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT url FROM articles "
                f"WHERE typeset_at IS NOT NULL AND url IN ({placeholders})",
                urls,
            ).fetchall()
            return {str(r[0]) for r in rows}

    def mark_typeset(self, urls: list[str], typeset_at: str, edition: str) -> None:
        """Stamp articles as published in an edition. Called only after the
        PDF render succeeded — a failed run must leave articles eligible for
        the next edition. Already-typeset rows keep their original stamp."""
        if not urls:
            return
        with self._connect() as conn:
            # Upsert: an article that reached the renderer without passing
            # through triage (e.g. injected by a plugin) still gets stamped.
            conn.executemany(
                "INSERT INTO articles "
                "(url, first_seen_at, last_seen_at, typeset_at, typeset_edition) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(url) DO UPDATE SET "
                "  typeset_at = excluded.typeset_at, "
                "  typeset_edition = excluded.typeset_edition "
                "WHERE typeset_at IS NULL",
                [(url, typeset_at, typeset_at, typeset_at, edition) for url in urls],
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
