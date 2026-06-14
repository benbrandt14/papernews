from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS article (
    url_hash         TEXT PRIMARY KEY,
    url              TEXT NOT NULL,
    title            TEXT NOT NULL,
    title_norm       TEXT NOT NULL,
    source           TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_title_norm  ON article(title_norm);
CREATE INDEX IF NOT EXISTS idx_rendered_at ON article(rendered_at);
CREATE INDEX IF NOT EXISTS idx_selection   ON article(selection_status);
"""


def _migrate(con) -> None:
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
    con.commit()


class Store:
    def __init__(self, path: Path):
        self.con = sqlite3.connect(str(path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA)
        _migrate(self.con)

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
              (url_hash, url, title, title_norm, source, text,
               surfaced, published, fetched_at, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                h, url, title, _norm_title(title), source, text,
                surfaced, published,
                now,
                now if text is not None else None,
            ),
        )
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

    def pending_selection(self, source: str) -> list[sqlite3.Row]:
        cur = self.con.execute(
            """
            SELECT url_hash, source, url, title, text
              FROM article
             WHERE source = ?
               AND selection_status = 0
               AND text IS NOT NULL
             ORDER BY fetched_at ASC
            """,
            (source,)
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

    def pending_render(self) -> list[sqlite3.Row]:
        cur = self.con.execute(
            """
            SELECT url_hash, source, url, title, text, body, summary,
                   surfaced, published, fetched_at
              FROM article
             WHERE rendered_at IS NULL
               AND summary     IS NOT NULL
               AND text        IS NOT NULL
               AND selection_status = 1
            """
        )
        return list(cur.fetchall())

    def latest_per_source(self, source: str, limit: int) -> list[sqlite3.Row]:
        cur = self.con.execute(
            """
            SELECT url_hash, source, url, title, text, body, summary,
                   surfaced, published, fetched_at
              FROM article
             WHERE source = ?
               AND text     IS NOT NULL
               AND summary  IS NOT NULL
               AND rendered_at IS NULL
               AND selection_status = 1
             ORDER BY COALESCE(published, surfaced, fetched_at) DESC
             LIMIT ?
            """,
            (source, limit),
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