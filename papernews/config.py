"""Typed, validated configuration for papernews.

Two layers:

  * `AppConfig` — the contents of `sources.toml` (sources, preferences,
    category limits), validated strictly so typos fail loudly at load
    time instead of silently degrading the pipeline.
  * `Settings` — process-level configuration from environment variables
    (``PAPERNEWS_`` prefix).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["rss", "hn"]
    category: str = "Uncategorized"
    url: str | None = None
    limit: int | None = None
    since_hours: int = 48
    min_points: int = 50

    @model_validator(mode="after")
    def _rss_requires_url(self) -> SourceSpec:
        if self.kind == "rss" and not self.url:
            raise ValueError(f"source {self.name!r}: kind='rss' requires a url")
        return self


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_category_limit: int = 1
    prefer_category: list[str] = Field(default_factory=list)
    less_prefer_category: list[str] = Field(default_factory=list)
    blacklist_words: list[str] = Field(default_factory=list)
    interest: list[str] = Field(default_factory=list)
    disinterest: list[str] = Field(default_factory=list)
    max_char_length: int = 20000


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceSpec] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)
    category_limits: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _limits_must_match_source_categories(self) -> AppConfig:
        """A [category_limits] key that matches no source category is a typo.

        Without this check, a misspelled category silently falls back to
        default_category_limit and the mistake is invisible in the output.
        """
        categories = {s.category for s in self.sources}
        unknown = sorted(set(self.category_limits) - categories)
        if unknown:
            raise ValueError(
                f"[category_limits] keys {unknown} do not match any source "
                f"category (available: {sorted(categories)})"
            )
        return self


def load_config(path: Path | str) -> AppConfig:
    """Load and validate `sources.toml` into an AppConfig."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return AppConfig(
        sources=raw.get("source", []),
        preferences=raw.get("preferences", {}),
        category_limits=raw.get("category_limits", {}),
    )


class Settings(BaseSettings):
    """Process-level settings from PAPERNEWS_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="PAPERNEWS_")

    config: Path = Path("sources.toml")
    output: Path = Path("output")
    state: Path = Path("data/state.db")
    llm_enabled: bool = False
    llm_backend: Literal["gemini", "local"] = "gemini"
    llm_model: str = "gemini-2.5-flash"
    # Render article bodies via the structured markdown IR + typed emitter
    # instead of the legacy NUL-sentinel regex path. Off until the IR path
    # has proven parity in production.
    use_ir_renderer: bool = False


def get_settings() -> Settings:
    """Fresh settings from the current environment (cheap; deliberately uncached)."""
    return Settings()
