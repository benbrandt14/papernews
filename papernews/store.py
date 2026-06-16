from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _url_hash(url: str) -> str:
    """Generate short hash for URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _norm_title(title: str) -> str:
    """Normalize title for exact duplicate matching."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _now() -> str:
    """Return current ISO format datetime."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS article (
    url_hash         TEXT PRIMARY KEY,
    url              TEXT NOT NULL,
    title            TEXT NOT NULL,
    title_norm       TEXT NOT NULL,
    source           TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT 'Uncategorized',
    text             TEXT,              -- NULL if extraction failed
    body             TEXT,              -- NULL until rewritten
    summary          TEXT,              -- NULL until summarized
    surfaced         TEXT,              
    published        TEXT,              
    fetched_at       TEXT NOT NULL,
    extracted_at     TEXT,
    summarized_at    TEXT,
    rewritten_at     TEXT,
    rendered_at      TEXT,              
    selection_status INTEGER DEFAULT 0  -- 0: pending, 1: selected, -1: rejected
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_title_norm  ON article(title_norm);
CREATE INDEX IF NOT EXISTS idx_rendered_at ON article(rendered_at);
CREATE INDEX IF NOT EXISTS idx_selection   ON article(selection_status);
"""


def _migrate(con) -> None:
    """Initialize or update database schema."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(article)")}
    if "body" not in cols:
        con.execute("ALTER TABLE article ADD COLUMN body TEXT")
    if "rewritten_at" not in cols:
        con.execute("ALTER TABLE article ADD COLUMN rewritten_at TEXT")
    if "surfaced" not in cols:
        con.execute("ALTER TABLE article ADD COLUMN surfaced TEXT")
    if "published" not in cols:
        con.execute("ALTER TABLE article ADD COLUMN published TEXT")
    if "selection_status" not in cols:
        con.execute("ALTER TABLE article ADD COLUMN selection_status INTEGER DEFAULT 0")
    if "category" not in cols:
        con.execute("ALTER TABLE article ADD COLUMN category TEXT DEFAULT 'Uncategorized'")
    con.commit()


class Store:
    """SQLite storage for articles and metadata."""
    def __init__(self, path: Path):
        self.con = sqlite3.connect(str(path))
        self.con.row_factory = sqlite3.Row
        
        # 1. Create base tables
        self.con.executescript(_SCHEMA)
        
        # 2. Run migrations (adds missing columns to old DBs)
        _migrate(self.con)
        
        # 3. Create indexes (now guaranteed that the columns exist)
        self.con.executescript(_INDEXES)

    def sync_categories(self, source_category_map: dict[str, str]) -> None:
        """Update categories to match current configuration."""
        for source, category in source_category_map.items():
            self.con.execute("UPDATE article SET category = ? WHERE source = ?", (category, source))
        self.con.commit()

    # --- gather --------------------------------------------------------------

    def exists(self, url: str, title: str) -> bool:
        cur = self.con.execute(
            "SELECT 1 FROM article WHERE url_hash = ? OR title_norm = ? LIMIT 1",
            (_url_hash(url), _norm_title(title)),
        )
        return cur.fetchone() is not None

    def insert_raw(
        self,
        source: str,
        category: str,
        url: str,
        title: str,
        text: str | None,
        surfaced: str | None = None,
        published: str | None = None,
    ) -> None:
        now = _now()
        h = _url_hash(url)
        self.con.execute(
            """
            INSERT OR IGNORE INTO article
              (url_hash, url, title, title_norm, source, category, text,
               surfaced, published, fetched_at, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                h, url, title, _norm_title(title), source, category, text,
                surfaced, published,
                now,
                now if text is not None else None,
            ),
        )
        # Keep category updated even if the URL exists
        self.con.execute("UPDATE article SET category = ? WHERE url_hash = ?", (category, h))
        
        if surfaced:
            self.con.execute(
                "UPDATE article SET surfaced = ? WHERE url_hash = ? AND surfaced IS NULL",
                (surfaced, h),
            )
        if published:
            self.con.execute(
                "UPDATE article SET published = ? WHERE url_hash = ? AND published IS NULL",
                (published, h),
            )
        self.con.commit()

    # --- select --------------------------------------------------------------

    def pending_selection_by_category(self, category: str) -> list[sqlite3.Row]:
        cur = self.con.execute(
            """
            SELECT url_hash, source, category, url, title, text
              FROM article
             WHERE category = ?
               AND selection_status = 0
               AND text IS NOT NULL
             ORDER BY fetched_at ASC
            """,
            (category,)
        )
        return list(cur.fetchall())

    def set_selection_status(self, url_hashes: list[str], status: int) -> None:
        if not url_hashes:
            return
        self.con.executemany(
            "UPDATE article SET selection_status = ? WHERE url_hash = ?",
            [(status, h) for h in url_hashes],
        )
        self.con.commit()

    # --- summarize -----------------------------------------------------------

    def pending_summary(self) -> list[sqlite3.Row]:
        cur = self.con.execute(
            """
            SELECT url_hash, source, url, title, text
              FROM article
             WHERE summary IS NULL
               AND text    IS NOT NULL
               AND selection_status = 1
             ORDER BY fetched_at ASC
            """
        )
        return list(cur.fetchall())

    def set_summary(self, url_hash: str, summary: str) -> None:
        self.con.execute(
            "UPDATE article SET summary = ?, summarized_at = ? WHERE url_hash = ?",
            (summary, _now(), url_hash),
        )
        self.con.commit()

    # --- rewrite -------------------------------------------------------------

    def pending_rewrite(self) -> list[sqlite3.Row]:
        cur = self.con.execute(
            """
            SELECT url_hash, source, url, title, text
              FROM article
             WHERE body IS NULL
               AND text IS NOT NULL
               AND selection_status = 1
             ORDER BY fetched_at ASC
            """
        )
        return list(cur.fetchall())

    def set_body(self, url_hash: str, body: str) -> None:
        self.con.execute(
            "UPDATE article SET body = ?, rewritten_at = ? WHERE url_hash = ?",
            (body, _now(), url_hash),
        )
        self.con.commit()

    # --- render --------------------------------------------------------------

    def latest_per_category(self, category: str, limit: int) -> list[sqlite3.Row]:
        cur = self.con.execute(
            """
            SELECT url_hash, source, category, url, title, text, body, summary,
                   surfaced, published, fetched_at
              FROM article
             WHERE category = ?
               AND text     IS NOT NULL
               AND summary  IS NOT NULL
               AND rendered_at IS NULL
               AND selection_status = 1
             ORDER BY COALESCE(published, surfaced, fetched_at) DESC
             LIMIT ?
            """,
            (category, limit),
        )
        return list(cur.fetchall())

    def max_fetched_at(self) -> str:
        row = self.con.execute(
            "SELECT COALESCE(MAX(fetched_at), '') FROM article"
        ).fetchone()
        return row[0] or ""

    def mark_rendered(self, url_hashes: list[str], date: str) -> None:
        if not url_hashes:
            return
        self.con.executemany(
            "UPDATE article SET rendered_at = ? WHERE url_hash = ?",
            [(date, h) for h in url_hashes],
        )
        self.con.commit()

    def sync_renders(self, out_dir: Path, constant_name: str = "papernews.pdf") -> int:
        """
        Checks the output directory for generated PDFs. If the PDF for a rendered
        date is missing (and the constant papernews.pdf is also missing), 
        the rendered_at marking is reset so it can be rebuilt.
        """
        cur = self.con.execute("SELECT DISTINCT rendered_at FROM article WHERE rendered_at IS NOT NULL")
        dates = [row[0] for row in cur.fetchall()]
        
        unmarked = 0
        const_exists = (out_dir / constant_name).exists()
        
        for d in dates:
            if not (out_dir / f"{d}.pdf").exists() and not const_exists:
                res = self.con.execute(
                    "UPDATE article SET rendered_at = NULL WHERE rendered_at = ?",
                    (d,)
                )
                unmarked += res.rowcount
                
        self.con.commit()
        return unmarked

    # --- status --------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        c = self.con.execute
        return {
            "total":            c("SELECT COUNT(*) FROM article").fetchone()[0],
            "unreadable":       c("SELECT COUNT(*) FROM article WHERE text IS NULL").fetchone()[0],
            "pending_select":   c("SELECT COUNT(*) FROM article WHERE selection_status = 0 AND text IS NOT NULL").fetchone()[0],
            "rejected":         c("SELECT COUNT(*) FROM article WHERE selection_status = -1").fetchone()[0],
            "pending_summary":  c("SELECT COUNT(*) FROM article WHERE summary IS NULL AND text IS NOT NULL AND selection_status = 1").fetchone()[0],
            "pending_rewrite":  c("SELECT COUNT(*) FROM article WHERE body    IS NULL AND text IS NOT NULL AND selection_status = 1").fetchone()[0],
            "pending_render":   c("SELECT COUNT(*) FROM article WHERE rendered_at IS NULL AND summary IS NOT NULL AND text IS NOT NULL AND selection_status = 1").fetchone()[0],
            "rendered":         c("SELECT COUNT(*) FROM article WHERE rendered_at IS NOT NULL").fetchone()[0],
        }