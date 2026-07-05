"""Tests for the validated config loader (papernews/config.py)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from papernews.config import AppConfig, Preferences, Settings, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_repo_sources_toml_validates():
    """The committed sources.toml must always load cleanly.

    This is the permanent guard against the silent-typo class of bug
    (e.g. a misspelled [category_limits] key degrading a category to the
    default limit without any error).
    """
    config = load_config(REPO_ROOT / "sources.toml")
    assert len(config.sources) > 0
    assert config.preferences.default_category_limit >= 1
    assert config.category_limits


def test_unknown_category_limit_key_raises(tmp_path):
    cfg = tmp_path / "sources.toml"
    cfg.write_text(
        '[[source]]\nname = "A"\nkind = "rss"\nurl = "http://x"\ncategory = "Science"\n'
        "\n[category_limits]\n"
        '"Sciennce" = 3\n'  # deliberate typo
    )
    with pytest.raises(ValidationError, match="Sciennce"):
        load_config(cfg)


def test_unknown_source_field_raises(tmp_path):
    cfg = tmp_path / "sources.toml"
    cfg.write_text(
        '[[source]]\nname = "A"\nkind = "rss"\nurl = "http://x"\n'
        'categry = "Science"\n'  # deliberate typo
    )
    with pytest.raises(ValidationError, match="categry"):
        load_config(cfg)


def test_unknown_preferences_field_raises(tmp_path):
    cfg = tmp_path / "sources.toml"
    cfg.write_text(
        '[[source]]\nname = "A"\nkind = "rss"\nurl = "http://x"\ncategory = "S"\n'
        "\n[preferences]\nblacklist_wordz = []\n"
    )
    with pytest.raises(ValidationError, match="blacklist_wordz"):
        load_config(cfg)


def test_rss_source_requires_url():
    with pytest.raises(ValidationError, match="requires a url"):
        AppConfig(sources=[{"name": "A", "kind": "rss", "category": "S"}])


def test_hn_source_needs_no_url():
    config = AppConfig(sources=[{"name": "HN", "kind": "hn", "category": "News"}])
    assert config.sources[0].since_hours == 48
    assert config.sources[0].min_points == 50


def test_preferences_defaults():
    prefs = Preferences()
    assert prefs.default_category_limit == 1
    assert prefs.max_char_length == 20000
    assert prefs.blacklist_words == []


def test_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("PAPERNEWS_LLM_ENABLED", "1")
    monkeypatch.setenv("PAPERNEWS_OUTPUT", "/tmp/somewhere")
    s = Settings()
    assert s.llm_enabled is True
    assert s.output == Path("/tmp/somewhere")


def test_settings_llm_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PAPERNEWS_LLM_ENABLED", raising=False)
    assert Settings().llm_enabled is False
