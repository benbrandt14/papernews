from hypothesis import given, strategies as st, settings, HealthCheck
from papernews.plugins import hn_plugin, rss_plugin
import pytest

# Feedparser returns a dict-like structure.
# We'll construct minimal stubs that look like its output.
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

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(
        st.fixed_dictionaries({
            "title": st.text(),
            "url": st.text(),
            "objectID": st.text(),
            "points": st.integers()
        }),
        max_size=5
    ),
    st.text(),
    st.booleans()
)
def test_hn_plugin_property(mocker, hits, fake_text, fetch_success):
    # Mocking external calls
    mocker.patch("papernews.plugins.hn_plugin.get_run_logger")

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"hits": hits}

    mocker.patch("requests.get", return_value=FakeResponse())

    if fetch_success:
        mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
        mocker.patch("trafilatura.extract", return_value=fake_text)
    else:
        mocker.patch("trafilatura.fetch_url", return_value=None)

    # Valid sources config
    source_config = {
        "source": [
            {"kind": "hn", "name": "Hacker News", "category": "Tech", "limit": 2}
        ]
    }

    # We just want to ensure it doesn't crash on randomized inputs and returns a list.
    docs = hn_plugin.fetch_sources(source_config)
    assert isinstance(docs, list)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(
        st.builds(
            StubEntry,
            link=st.text(),
            title=st.text(),
            published=st.text(),
            updated=st.text()
        ),
        max_size=5
    ),
    st.text(),
    st.booleans()
)
def test_rss_plugin_property(mocker, entries, fake_text, fetch_success):
    # Mock feedparser
    mocker.patch("feedparser.parse", return_value=StubFeed(entries))

    if fetch_success:
        mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
        mocker.patch("trafilatura.extract", return_value=fake_text)
    else:
        mocker.patch("trafilatura.fetch_url", return_value=None)

    # Valid sources config
    source_config = {
        "source": [
            {"kind": "rss", "url": "http://fake.com", "category": "News"}
        ]
    }

    docs = rss_plugin.fetch_sources(source_config)
    assert isinstance(docs, list)
