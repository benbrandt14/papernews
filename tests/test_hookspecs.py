"""Contract tests for the pluggy hookspec layer (papernews/plugins/)."""

import pluggy
import pytest

from papernews.config import AppConfig
from papernews.models import FrontpageDecorations, RawDocument
from papernews.plugins.registry import get_plugin_manager


def test_registry_builds_manager_with_specs_and_builtins():
    pm = get_plugin_manager()
    # Specs are loaded: the relay knows every hook by name.
    assert hasattr(pm.hook, "fetch_sources")
    assert hasattr(pm.hook, "enrich_articles")
    assert hasattr(pm.hook, "fetch_decorations")
    # All built-in plugins are registered.
    names = {name for name, _ in pm.list_name_plugin()}
    assert {
        "papernews.plugins.rss_plugin",
        "papernews.plugins.hn_plugin",
        "papernews.plugins.wiki_plugin",
    } <= names


def test_all_hookimpls_match_a_spec():
    """check_pending() raises if any @hookimpl has no matching hookspec —
    the exact failure mode the old spec-less wiring hid until call time."""
    pm = get_plugin_manager()
    pm.check_pending()  # must not raise


def test_rogue_hookimpl_is_rejected():
    """A plugin whose hookimpl matches no spec must fail loudly at
    registration, not silently never fire."""
    pm = get_plugin_manager()

    class RoguePlugin:
        @pluggy.HookimplMarker("papernews")
        def fetch_sourcez(self, source_config):  # typo'd hook name
            return []

    pm.register(RoguePlugin())
    with pytest.raises(pluggy.PluginValidationError):
        pm.check_pending()


def test_fetch_sources_end_to_end_through_registry(mocker):
    """Fire the real hook relay (not a plugin module directly) and check
    typed RawDocuments come back."""

    class E:
        link = "https://example.com/a"

        def get(self, k, d=""):
            return {"title": "T", "published": "Mon, 30 Jun 2026 10:00:00 GMT"}.get(
                k, d
            )

    class F:
        entries = [E()]

    mocker.patch("feedparser.parse", return_value=F())
    mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
    mocker.patch("trafilatura.extract", return_value="body " * 300)
    mocker.patch("papernews.plugins.hn_plugin.get_run_logger")

    config = AppConfig(
        sources=[{"name": "S", "kind": "rss", "url": "http://x", "category": "Sci"}]
    )
    pm = get_plugin_manager()
    results = pm.hook.fetch_sources(source_config=config)

    docs = [d for sub in results for d in sub]
    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].title == "T"
    assert docs[0].category == "Sci"


def test_fetch_decorations_through_registry(mocker):
    class WikiResp:
        text = '<div class="current-events-content"><li>News item. [1]</li></div>'

        def raise_for_status(self):
            pass

    mocker.patch("papernews.plugins.wiki_plugin.get_run_logger")
    mocker.patch("papernews.plugins.wiki_plugin.requests.get", return_value=WikiResp())

    pm = get_plugin_manager()
    results = pm.hook.fetch_decorations(source_config=AppConfig())

    assert len(results) == 1
    assert isinstance(results[0], FrontpageDecorations)
    assert results[0].world_news == ["News item."]
    assert results[0].quote is not None  # placeholder quote present


def test_enrich_articles_hook_fires_custom_plugin(tmp_path, monkeypatch):
    """An external enrichment plugin registered against the spec receives
    the whole day's articles and can mutate them in place."""
    from papernews.models import Annotation, ArticleChunk

    monkeypatch.setenv("PAPERNEWS_STATE", str(tmp_path / "state.db"))

    hookimpl = pluggy.HookimplMarker("papernews")

    class MarginaliaStub:
        @hookimpl
        def enrich_articles(self, articles, source_config, store):
            for art in articles:
                art.annotations.append(
                    Annotation(source="stub", content="note", completion_percentage=50)
                )

    pm = get_plugin_manager()
    pm.register(MarginaliaStub())

    articles = [
        ArticleChunk(
            category="C",
            source="s",
            title="t",
            summary="s",
            body_markdown="b",
            url="u",
        )
    ]
    from papernews.store import SimpleStore

    pm.hook.enrich_articles(
        articles=articles, source_config=AppConfig(), store=SimpleStore()
    )
    assert articles[0].annotations[0].source == "stub"
