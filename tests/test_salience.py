"""Tests for the salience-weighting enrichment plugin."""

from __future__ import annotations

from papernews.config import AppConfig
from papernews.markdown_ir import parse_markdown
from papernews.models import ArticleChunk, Block
from papernews.plugins import salience_plugin
from papernews.store import SimpleStore


def test_split_sentences_offsets_are_exact():
    text = "First sentence here. Second one follows! And a third?  Trailing bit"
    spans = salience_plugin._split_sentences(text)
    fragments = [text[s:e] for s, e in spans]
    assert fragments == [
        "First sentence here.",
        "Second one follows!",
        "And a third?",
        "Trailing bit",
    ]


def test_split_sentences_handles_empty_and_whitespace():
    assert salience_plugin._split_sentences("") == []
    assert salience_plugin._split_sentences("   ") == []


def test_centroid_scores_ranks_central_sentence_highest():
    # Three on-topic sentences about photosynthesis and one off-topic aside.
    sentences = [
        "Photosynthesis converts light energy into chemical energy in plants.",
        "The chemical energy from light drives photosynthesis in plant cells.",
        "Plants use photosynthesis to store energy from sunlight as sugars.",
        "By the way, the weather forecast mentioned rain on Tuesday.",
    ]
    scores = salience_plugin._centroid_scores(sentences)
    # The off-topic aside is least central.
    assert scores.index(min(scores)) == 3
    # A photosynthesis sentence is most central.
    assert scores.index(max(scores)) in {0, 1, 2}


def test_centroid_scores_all_stopwords_is_zero():
    assert salience_plugin._centroid_scores(["the and or of", "to it is"]) == [0.0, 0.0]


def test_weight_article_marks_high_and_low_sentences():
    # One clearly central topic, one clear off-topic aside, plus filler to clear
    # the MIN_SENTENCES floor.
    body = (
        "Coral reefs shelter a quarter of all marine species worldwide. "
        "Reef ecosystems depend on symbiotic algae for their vivid color. "
        "Warming oceans cause corals to expel these algae and bleach. "
        "Bleached coral reefs lose their color and slowly starve. "
        "Scientists monitor reef bleaching with satellite temperature data. "
        "Incidentally, the museum gift shop sells postcards of unrelated birds."
    )
    art = ArticleChunk(
        category="Science",
        source="example.com",
        title="Reefs",
        summary="s",
        body_markdown=body,
        url="https://example.com",
        blocks=parse_markdown(body),
    )

    salience_plugin._weight_article(art)

    salience_spans = [
        sp for block in art.blocks for sp in block.spans if sp.kind == "salience"
    ]
    assert salience_spans, "expected at least one salience span"

    weights = {round(sp.weight, 2) for sp in salience_spans}
    assert salience_plugin.HIGH_WEIGHT in weights
    assert salience_plugin.LOW_WEIGHT in weights

    # Every span must land on a real slice of its block's text.
    for block in art.blocks:
        for sp in block.spans:
            if sp.kind == "salience":
                assert 0 <= sp.start < sp.end <= len(block.text)

    # The off-topic aside should be demoted, not promoted.
    aside = next(
        sp
        for block in art.blocks
        for sp in block.spans
        if sp.kind == "salience" and "museum gift shop" in block.text[sp.start : sp.end]
    )
    assert aside.weight == salience_plugin.LOW_WEIGHT


def test_weight_article_skips_short_articles():
    art = ArticleChunk(
        category="C",
        source="s",
        title="t",
        summary="s",
        body_markdown="One short sentence. Another short one.",
        url="u",
        blocks=[Block(kind="para", text="One short sentence. Another short one.")],
    )
    salience_plugin._weight_article(art)
    assert all(sp.kind != "salience" for block in art.blocks for sp in block.spans)


def test_weight_article_ignores_non_paragraph_blocks():
    art = ArticleChunk(
        category="C",
        source="s",
        title="t",
        summary="s",
        body_markdown="x",
        url="u",
        blocks=[
            Block(kind="heading", level=1, text="A Heading That Is Fairly Long Indeed"),
            Block(kind="code", raw="print('lots of code here for length')"),
        ],
    )
    salience_plugin._weight_article(art)
    assert all(sp.kind != "salience" for block in art.blocks for sp in block.spans)


def test_enrich_articles_hook_weights_in_place(tmp_path):
    body = (
        "Neutron stars pack a sun's mass into a city-sized sphere. "
        "The density of a neutron star exceeds that of an atomic nucleus. "
        "Pulsars are neutron stars that beam radiation as they spin. "
        "Astronomers time pulsar beams to test general relativity. "
        "Some neutron stars host the strongest magnetic fields known. "
        "Unrelatedly, a nearby cafe was reviewing its lunch menu options."
    )
    art = ArticleChunk(
        category="Space",
        source="example.com",
        title="Neutron stars",
        summary="s",
        body_markdown=body,
        url="https://example.com",
        blocks=parse_markdown(body),
    )
    salience_plugin.enrich_articles(
        [art], AppConfig(), SimpleStore(str(tmp_path / "state.db"))
    )
    assert any(sp.kind == "salience" for block in art.blocks for sp in block.spans)


def test_salience_spans_emit_smart_sentence(tmp_path):
    """Attached salience spans must round-trip through the emitter into
    #smart-sentence calls."""
    from papernews.typst_emit import emit_blocks

    body = (
        "Enzymes accelerate the chemical reactions inside living cells. "
        "Each enzyme lowers the activation energy of its target reaction. "
        "Temperature and pH change how quickly enzymes catalyze reactions. "
        "Denatured enzymes lose the shape their catalysis depends on. "
        "Cells regulate enzyme activity to control their metabolism. "
        "Separately, the shipping label listed an incorrect return address."
    )
    art = ArticleChunk(
        category="Bio",
        source="example.com",
        title="Enzymes",
        summary="s",
        body_markdown=body,
        url="https://example.com",
        blocks=parse_markdown(body),
    )
    salience_plugin._weight_article(art)
    emitted = emit_blocks(art.blocks, tmp_path)
    assert "#smart-sentence(weight:" in emitted
