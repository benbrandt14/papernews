"""Cache PDF + cover preview."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def edition_key(content_token: str, sources_config: list[dict]) -> str:
    """Stable hash representing 'which edition this is'."""
    payload = json.dumps(
        {
            "content": content_token,
            "sources": [
                {
                    "name": s.get("name"),
                    "kind": s.get("kind"),
                    "limit": s.get("limit"),
                }
                for s in sources_config
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def pdf_path(cache_dir: Path, key: str) -> Path:
    """Return path to PDF cache file."""
    return cache_dir / f"{key}.pdf"


def preview_path(cache_dir: Path, key: str) -> Path:
    """Return path to preview PNG cache file."""
    return cache_dir / f"{key}.png"


def ensure_dir(cache_dir: Path) -> Path:
    """Ensure cache directory exists."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
