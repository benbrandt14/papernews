# papernews/store.py
import sqlite3
from pathlib import Path
from typing import Optional

class SimpleStore:
    def __init__(self, db_path: str = "data/state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # One generic table for all LLM responses
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    id TEXT PRIMARY KEY,
                    response TEXT
                )
            """)
            # Future extension: manual blacklist or LLM rule triggers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS filters (
                    url TEXT PRIMARY KEY,
                    reason TEXT
                )
            """)

    def get_cache(self, cache_key: str) -> Optional[str]:
        """Fetch a cached string response."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT response FROM llm_cache WHERE id = ?", (cache_key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set_cache(self, cache_key: str, response: str):
        """Save or overwrite a cached response. timeout=10 prevents Prefect concurrency crashes."""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (id, response) VALUES (?, ?)", 
                (cache_key, response)
            )