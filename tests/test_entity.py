"""Tests for the entity-interlinking enrichment plugin."""

from __future__ import annotations

from papernews.config import AppConfig
from papernews.markdown_ir import parse_markdown
from papernews.models import ArticleChunk, Block
from papernews.plugins import entity_plugin
from papernews.store import SimpleStore


def _mentions(text: str) -> list[str]:
    return [surface for _s, _e, surface in entity_plugin._iter_entities(text)]


def test_iter_entities_extracts_phrases_acronyms_camelcase():
    text = "The European Space Agency partnered with NASA and OpenAI on the probe."
    found = _mentions(text)
    assert "European Space Agency" in found  # leading "The" trimmed
    assert "NASA" in found
    assert "OpenAI" in found


def test_iter_entities_offsets_are_exact():
    text = "Reports say NASA and the James Webb telescope agree."
    for start, end, surface in entity_plugin._iter_entities(text):
        assert text[start:end] == surface


def test_iter_entities_skips_bare_single_capitalized_words():
    # Sentence-initial plain words must not become entities.
    text = "Researchers found something. Plants grow. Further study is needed."
    assert _mentions(text) == []


def test_iter_entities_trims_leading_determiner_but_keeps_multiword():
    text = "The Great Barrier Reef is shrinking."
    found = _mentions(text)
    assert "Great Barrier Reef" in found
    assert not any(f.startswith("The ") for f in found)


def _article(title: str, body: str, url: str) -> ArticleChunk:
    return ArticleChunk(
        category="News",
        source="example.com",
        title=title,
        summary="s",
        body_markdown=body,
        url=url,
        blocks=parse_markdown(body),
    )


def test_enrich_links_shared_entity_to_home_article(tmp_path):
    store = SimpleStore(str(tmp_path / "state.db"))
    # Article 0 mentions the ESA twice → it becomes the home. Article 1 mentions
    # it once and should get a link back to art1 (art index 0 → label art1).
    a0 = _article(
        "Deep dive",
        "The European Space Agency launched a probe. "
        "The European Space Agency confirmed the trajectory later that week.",
        "https://example.com/0",
    )
    a1 = _article(
        "Brief",
        "In other news, the European Space Agency budget was approved on Monday.",
        "https://example.com/1",
    )

    entity_plugin.enrich_articles([a0, a1], AppConfig(), store)

    # The home article carries no back-link to itself.
    assert all(sp.kind != "entity" for block in a0.blocks for sp in block.spans)
    # The secondary article links its mention to the home article's anchor.
    entity_spans = [
        sp for block in a1.blocks for sp in block.spans if sp.kind == "entity"
    ]
    assert len(entity_spans) == 1
    assert entity_spans[0].label == "art1"
    span = entity_spans[0]
    linked_block = next(b for b in a1.blocks if span in b.spans)
    assert linked_block.text[span.start : span.end] == "European Space Agency"
    assert a1.enrichment.entities[0].label == "art1"


def test_enrich_ignores_entities_in_a_single_article(tmp_path):
    store = SimpleStore(str(tmp_path / "state.db"))
    a0 = _article(
        "Solo",
        "The Hubble Space Telescope returned new images of a distant galaxy.",
        "https://example.com/0",
    )
    a1 = _article(
        "Unrelated",
        "A local bakery in Vermont won an award for its sourdough bread loaf.",
        "https://example.com/1",
    )
    entity_plugin.enrich_articles([a0, a1], AppConfig(), store)
    for art in (a0, a1):
        assert all(sp.kind != "entity" for block in art.blocks for sp in block.spans)


def test_enrich_records_mentions_in_knowledge_graph(tmp_path):
    store = SimpleStore(str(tmp_path / "state.db"))
    a0 = _article(
        "One",
        "NASA and the European Space Agency signed a new agreement today.",
        "https://example.com/0",
    )
    a1 = _article(
        "Two",
        "The European Space Agency also met with private launch providers.",
        "https://example.com/1",
    )
    entity_plugin.enrich_articles([a0, a1], AppConfig(), store)

    # ESA appears in both articles → mention_count 2; NASA in one → 1.
    assert store.entity_mention_count("european space agency") == 2
    assert store.entity_mention_count("nasa") == 1


def test_enrich_mentions_are_idempotent_on_rerun(tmp_path):
    store = SimpleStore(str(tmp_path / "state.db"))
    a0 = _article(
        "One",
        "The European Space Agency launched a probe successfully.",
        "https://example.com/0",
    )
    a1 = _article(
        "Two",
        "The European Space Agency reported the probe reached orbit.",
        "https://example.com/1",
    )
    entity_plugin.enrich_articles([a0, a1], AppConfig(), store)
    entity_plugin.enrich_articles([a0, a1], AppConfig(), store)  # rerun same day
    assert store.entity_mention_count("european space agency") == 2


def test_entity_spans_emit_internal_link(tmp_path):
    from papernews.typst_emit import emit_blocks

    a0 = _article(
        "Home",
        "The James Webb Space Telescope resolved the faint spiral galaxy. "
        "The James Webb Space Telescope also imaged a second target.",
        "https://example.com/0",
    )
    a1 = _article(
        "Away",
        "Separately, the James Webb Space Telescope data was released to the public.",
        "https://example.com/1",
    )
    entity_plugin.enrich_articles(
        [a0, a1], AppConfig(), SimpleStore(str(tmp_path / "state.db"))
    )
    emitted = emit_blocks(a1.blocks, tmp_path)
    assert "#link(<art1>)" in emitted


def test_article_entities_skips_code_blocks():
    art = ArticleChunk(
        category="C",
        source="s",
        title="t",
        summary="s",
        body_markdown="x",
        url="u",
        blocks=[Block(kind="code", raw="from NASA import European Space Agency")],
    )
    assert entity_plugin._article_entities(art) == {}
