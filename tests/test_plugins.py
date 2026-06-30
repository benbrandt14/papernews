import importlib
import pkgutil

import pytest

import papernews.plugins
from papernews.models import RawDocument


def get_ingestion_plugins():
    plugins = []
    for _, name, _ in pkgutil.iter_modules(papernews.plugins.__path__):
        if name == "wiki_plugin":
            # Exclude wiki_plugin as it implements fetch_decorations, not fetch_sources
            continue
        mod = importlib.import_module(f"papernews.plugins.{name}")
        if hasattr(mod, "fetch_sources"):
            plugins.append((name, mod))
    return plugins


# We need to construct configurations that trigger the plugins.
# hn_plugin looks for kind="hn", rss_plugin looks for kind="rss"
PLUGIN_KINDS = {"hn_plugin": "hn", "rss_plugin": "rss"}


# Dummy Feedparser structures for RSS
class StubEntry:
    def __init__(self, link, title, published, updated):
        self.link = link
        self.title_attr = title
        self.published_attr = published
        self.updated_attr = updated

    def get(self, key, default=""):
        if key == "title":
            return self.title_attr
        if key == "published":
            return self.published_attr
        if key == "updated":
            return self.updated_attr
        return default


class StubFeed:
    def __init__(self, entries):
        self.entries = entries


@pytest.fixture(params=get_ingestion_plugins(), ids=lambda p: p[0])
def ingestion_plugin(request, mocker):
    name, mod = request.param

    # We will mock the external dependencies for all plugins here.
    if name == "hn_plugin":
        mocker.patch("papernews.plugins.hn_plugin.get_run_logger")

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "hits": [
                        {
                            "title": "t",
                            "url": "http://example.com",
                            "objectID": "1",
                            "points": 100,
                        }
                    ]
                }

        mocker.patch("requests.get", return_value=FakeResponse())

    elif name == "rss_plugin":
        entries = [StubEntry("http://example.com", "Title", "Date", "Date")]
        mocker.patch("feedparser.parse", return_value=StubFeed(entries))

    # Provide the mocked module and its kind
    kind = PLUGIN_KINDS.get(name, "unknown")
    return mod, kind


def test_plugin_network_failure(ingestion_plugin, mocker):
    mod, kind = ingestion_plugin

    # Mock trafilatura network failure
    mock_fetch = mocker.patch("trafilatura.fetch_url", return_value=None)

    source_config = {
        "source": [
            {"kind": kind, "url": "http://fake.com", "category": "News", "limit": 2}
        ]
    }

    docs = mod.fetch_sources(source_config)

    assert isinstance(docs, list)
    assert (
        len(docs) == 0
    )  # because fetch failed, it should continue and return empty or drop
    mock_fetch.assert_called()


def test_plugin_successful_ingestion(ingestion_plugin, mocker):
    mod, kind = ingestion_plugin

    # Mock successful trafilatura extraction
    mock_fetch = mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
    mock_extract = mocker.patch(
        "trafilatura.extract",
        return_value="Here is some extracted text that is long enough to bypass limits "
        * 20,
    )

    source_config = {
        "source": [
            {"kind": kind, "url": "http://fake.com", "category": "News", "limit": 2}
        ]
    }

    docs = mod.fetch_sources(source_config)

    assert isinstance(docs, list)
    assert len(docs) > 0
    for doc in docs:
        assert isinstance(doc, RawDocument)
        assert hasattr(doc, "raw_text")
        assert len(doc.raw_text) > 0
        assert doc.metadata is not None
        assert "title" in doc.metadata

    mock_fetch.assert_called()

    # CRITICAL REVISION: Enforce the architectural invariants for extraction
    mock_extract.assert_called()
    _, kwargs = mock_extract.call_args
    assert kwargs.get("include_images") is True, (
        f"Plugin '{mod.__name__}' failed to pass include_images=True to trafilatura"
    )
    assert kwargs.get("include_links") is True, (
        f"Plugin '{mod.__name__}' failed to pass include_links=True to trafilatura"
    )


def test_plugin_extraction_failure(ingestion_plugin, mocker):
    mod, kind = ingestion_plugin

    # Mock successful network fetch but failed extraction
    mock_fetch = mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
    mock_extract = mocker.patch("trafilatura.extract", return_value=None)

    source_config = {
        "source": [
            {"kind": kind, "url": "http://fake.com", "category": "News", "limit": 2}
        ]
    }

    docs = mod.fetch_sources(source_config)

    assert isinstance(docs, list)
    assert (
        len(docs) == 0
    )  # because extract failed, it should drop the article gracefully
    mock_fetch.assert_called()
    mock_extract.assert_called()
