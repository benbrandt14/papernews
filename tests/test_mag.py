"""Mag notation (magworld.pw): the formatter, the enrichment plugin, the
emitter span, and the URL/title dedupe hardening that shipped alongside."""

from __future__ import annotations

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")

import pytest

from papernews.dedupe import canonical_url, title_key
from papernews.mag import format_mag, mag
from papernews.models import ArticleChunk, Block, Span
from papernews.plugins import mag_plugin
from papernews.typst_emit import emit_blocks

# --- format_mag / mag --------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (1000, "^3"),  # round powers of ten drop the decimal
        (2500, "^3.4"),
        (48_000, "^4.7"),
        (3.2e9, "^9.5"),
        (1, "^0"),  # human scale
        (0.01, "^-2"),  # fractions go negative
        (0, "0"),
        (-2500, "-^3.4"),  # sign of the value, magnitude of |x|
    ],
)
def test_format_mag(value, expected):
    assert format_mag(value) == expected


def test_mag_multiplication_is_addition():
    # The point of the notation: multiplying numbers adds their mags.
    assert mag(1000) + mag(100) == mag(1000 * 100)


def test_mag_undefined_for_nonpositive():
    assert mag(0) is None
    assert mag(-5) is None


# --- Enrichment plugin -------------------------------------------------------


def _article(text: str) -> ArticleChunk:
    art = ArticleChunk(
        category="Science",
        source="example.com",
        title="T",
        summary="S",
        body_markdown=text,
        url="https://example.com/a",
    )
    art.blocks = [Block(kind="para", text=text)]
    return art


def _enrich(art: ArticleChunk) -> None:
    mag_plugin.enrich_articles([art], source_config=None, store=None)


def test_plugin_glosses_big_numbers():
    art = _article("The project cost $3.2 billion and involved 48,000 people.")
    _enrich(art)
    labels = [s.label for s in art.blocks[0].spans if s.kind == "mag"]
    assert labels == ["^9.5", "^4.7"]


def test_plugin_ignores_small_numbers():
    art = _article("She wrote 3 papers over 15 years, citing 250 sources.")
    _enrich(art)
    assert [s for s in art.blocks[0].spans if s.kind == "mag"] == []


def test_plugin_respects_per_article_cap():
    text = " and ".join(f"{(n + 1) * 100_000:,} units" for n in range(20))
    art = _article(text)
    _enrich(art)
    assert (
        len([s for s in art.blocks[0].spans if s.kind == "mag"])
        == mag_plugin.MAX_GLOSSES_PER_ARTICLE
    )


def test_plugin_skips_numbers_inside_code_spans():
    text = "Run `dd bs=1048576` to copy 5,000,000 records."
    art = _article(text)
    art.blocks[0].spans.append(
        Span(start=text.index("`"), end=text.index("`", 5) + 1, kind="code_inline")
    )
    _enrich(art)
    labels = [s.label for s in art.blocks[0].spans if s.kind == "mag"]
    assert labels == ["^6.7"]  # only the prose number, not the code one


def test_mag_span_emits_magnote(tmp_path):
    text = "It cost 48,000 dollars."
    block = Block(
        kind="para",
        text=text,
        spans=[Span(start=8, end=14, kind="mag", label="^4.7")],
    )
    out = emit_blocks([block], tmp_path)
    assert out == 'It cost 48,000#magnote("^4.7")/**/ dollars.'


# --- Dedupe hardening --------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        (
            "https://www.example.com/story/?utm_source=rss&utm_medium=feed",
            "http://example.com/story",
        ),
        ("https://example.com/story#comments", "https://example.com/story"),
        (
            "https://example.com/story?id=7&utm_campaign=x",
            "https://example.com/story?id=7",
        ),
    ],
)
def test_canonical_url_folds_variants(a, b):
    assert canonical_url(a) == canonical_url(b)


def test_canonical_url_keeps_meaningful_query():
    assert "id=7" in canonical_url("https://example.com/story?id=7")


def test_canonical_url_passes_through_non_urls():
    assert canonical_url("hn:12345") == "hn:12345"


def test_title_key_matches_reformatted_titles():
    assert title_key("SpaceX launches 'Starship' — again!") == title_key(
        "SpaceX Launches Starship: Again"
    )


def test_title_key_empty_for_short_titles():
    # Short titles are not distinctive enough to dedupe on.
    assert title_key("News") == ""
