"""Tests for the wiki decorations plugin (quote / DYK / world news)."""

from papernews.config import AppConfig
from papernews.models import FrontpageDecorations
from papernews.plugins import wiki_plugin

QOTD_WIKITEXT = """{{Wikiquote:Quote of the day/Template
| quote = <!-- note --> The [[imagination|imagination]] is the '''workshop''' of the mind,<br/>where ideas are ''forged''.
| author = [[Some Person|Someone Wise]]
}}"""


class FakeQotdResp:
    def json(self):
        return {"parse": {"wikitext": {"*": QOTD_WIKITEXT}}}


MAIN_PAGE_TEXT = """Welcome to Wikipedia
Some intro text.

Did you know ...
- ... that the first PDF was rendered by hand?
- that eels navigate by magnetic fields?
- ... that a third fact exists?

In the news
- Something else entirely.
"""


class FakeCurrentEventsResp:
    text = (
        '<div class="current-events-content">'
        "<li>An event happened somewhere. [1]</li>"
        "<li>Another event happened. [2]</li>"
        "</div>"
    )

    def raise_for_status(self):
        pass


def test_strip_wiki_handles_links_markup_and_tags():
    assert (
        wiki_plugin._strip_wiki("[[a|b]] and [[c]] with '''bold''' and ''ital''<br/>x")
        == "b and c with bold and ital x"
    )
    # Surrounding quotation marks are trimmed (the template adds its own).
    assert wiki_plugin._strip_wiki('"Quoted words."') == "Quoted words."


def test_fetch_quote_of_day_parses_wikitext(mocker):
    mocker.patch(
        "papernews.plugins.wiki_plugin.requests.get", return_value=FakeQotdResp()
    )
    quote = wiki_plugin._fetch_quote_of_day()
    assert quote is not None
    assert quote.text.startswith("The imagination is the workshop")
    assert quote.author == "Someone Wise"


def test_fetch_quote_of_day_skips_overlong_quotes(mocker):
    long_wikitext = QOTD_WIKITEXT.replace(
        "The [[imagination|imagination]] is the '''workshop''' of the mind,",
        "word " * 60,
    )

    class LongResp:
        def json(self):
            return {"parse": {"wikitext": {"*": long_wikitext}}}

    mocker.patch("papernews.plugins.wiki_plugin.requests.get", return_value=LongResp())
    # Every day in the window returns the same overlong quote -> None.
    assert wiki_plugin._fetch_quote_of_day(max_words=40, days_back=2) is None


def test_fetch_did_you_know_parses_main_page(mocker):
    mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
    mocker.patch("trafilatura.extract", return_value=MAIN_PAGE_TEXT)
    dyk = wiki_plugin._fetch_did_you_know()
    assert dyk == [
        "the first PDF was rendered by hand?",
        "eels navigate by magnetic fields?",
        "a third fact exists?",
    ]


def test_fetch_did_you_know_missing_section_returns_empty(mocker):
    mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
    mocker.patch("trafilatura.extract", return_value="No such section here.")
    assert wiki_plugin._fetch_did_you_know() == []


def test_fetch_decorations_populates_all_fields(mocker):
    mocker.patch("papernews.plugins.wiki_plugin.get_run_logger")
    mocker.patch(
        "papernews.plugins.wiki_plugin._fetch_world_news",
        return_value=["An event happened somewhere."],
    )
    mocker.patch(
        "papernews.plugins.wiki_plugin._fetch_quote_of_day",
        return_value=wiki_plugin.Quote(text="Words.", author="A. Author"),
    )
    mocker.patch(
        "papernews.plugins.wiki_plugin._fetch_did_you_know",
        return_value=["a fact?"],
    )

    deco = wiki_plugin.fetch_decorations(AppConfig())
    assert isinstance(deco, FrontpageDecorations)
    assert deco.world_news == ["An event happened somewhere."]
    assert deco.quote is not None and deco.quote.author == "A. Author"
    assert deco.dyk == ["a fact?"]


def test_fetch_decorations_degrades_gracefully(mocker):
    """Every source failing must still return a valid model with the
    fallback quote — never raise."""
    mocker.patch("papernews.plugins.wiki_plugin.get_run_logger")
    mocker.patch(
        "papernews.plugins.wiki_plugin._fetch_world_news",
        side_effect=OSError("offline"),
    )
    mocker.patch(
        "papernews.plugins.wiki_plugin._fetch_quote_of_day",
        side_effect=OSError("offline"),
    )
    mocker.patch(
        "papernews.plugins.wiki_plugin._fetch_did_you_know",
        side_effect=OSError("offline"),
    )

    deco = wiki_plugin.fetch_decorations(AppConfig())
    assert deco.quote == wiki_plugin._FALLBACK_QUOTE
    assert deco.world_news  # model default "unavailable" string
    assert deco.dyk == []


def test_world_news_parses_current_events_block(mocker):
    mocker.patch(
        "papernews.plugins.wiki_plugin.requests.get",
        return_value=FakeCurrentEventsResp(),
    )
    bullets = wiki_plugin._fetch_world_news()
    assert bullets == ["An event happened somewhere.", "Another event happened."]
