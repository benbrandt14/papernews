"""Tests for the markdown IR (markdown_ir.py) + typed emitter (typst_emit.py).

This is the only render path: the hostile-string gauntlet, the regression
corpus, and Hypothesis fuzz all gate it (parse → emit → compile).
"""

import json
import tempfile
from pathlib import Path

import pytest
import typst
from hypothesis import given, settings
from hypothesis import strategies as st

from papernews.markdown_ir import parse_markdown
from papernews.models import Block, ImageRef, Span
from papernews.typst_emit import PREAMBLE, emit_blocks

# Same hostile inputs as the legacy E2E gauntlet.
HOSTILE_STRINGS = [
    "Unclosed HTML: <div><p>broken",
    "Nested blockquotes: > level 1\n>> level 2\n>>> level 3",
    "Malformed LaTeX: $ x = \\frac{1}{2 $",
    "Unescaped Typst control characters: @hello #world $money",
    "Weird Unicode: ¯\\_(ツ)_/¯ 💥 👨‍👩‍👧‍👦 🚀 \x00 \x01 \x08",
    "Empty brackets: [] () {}",
    "Markdown link with empty text: [](http://example.com)",
    "Dangling backtick: `code",
    "Math with no closing: $$ \\int x dx",
    "Mixed delimiters: [ ( ] )",
]


def compile_snippet(typst_code: str) -> None:
    full = f'#import "@preview/mitex:0.2.4": mi, mitex\n{PREAMBLE}\n{typst_code}'
    with tempfile.TemporaryDirectory() as tmpdir:
        typ = Path(tmpdir) / "t.typ"
        typ.write_text(full, encoding="utf-8")
        typst.compile(str(typ), output=str(Path(tmpdir) / "t.pdf"))


def ir_render(text: str, workdir: Path) -> str:
    return emit_blocks(parse_markdown(text), workdir)


# --- Parsing ----------------------------------------------------------------


def test_parse_headings_and_paragraphs():
    # NOTE: a heading on the very first line is stripped as a redundant
    # LLM-hallucinated article title (legacy _strip_leading_metadata
    # behavior, preserved by the IR).
    blocks = parse_markdown("Body text here.\n\n## Sub\n\nMore.")
    kinds = [(b.kind, b.level) for b in blocks]
    assert kinds == [("para", 0), ("heading", 2), ("para", 0)]
    assert blocks[1].text == "Sub"


def test_parse_strips_leading_title_heading():
    blocks = parse_markdown("# Hallucinated Title\n\nActual body.")
    assert [b.kind for b in blocks] == ["para"]
    assert blocks[0].text == "Actual body."


def test_parse_list_items():
    blocks = parse_markdown("Intro:\n\n* one\n* two\n- three")
    assert [b.kind for b in blocks] == ["para", "list_item", "list_item", "list_item"]
    assert blocks[1].text == "one"


def test_parse_blockquote():
    blocks = parse_markdown("> quoted line one\n> quoted line two")
    assert blocks[0].kind == "quote"
    assert "quoted line one" in blocks[0].text


def test_parse_fenced_code_block():
    blocks = parse_markdown("Before.\n\n```python\nx = 1\n\ny = 2\n```\n\nAfter.")
    assert [b.kind for b in blocks] == ["para", "code", "para"]
    assert "x = 1\n\ny = 2" in blocks[1].raw


def test_parse_display_math():
    blocks = parse_markdown("Text.\n\n$$\\int x dx$$\n\nMore.")
    assert [b.kind for b in blocks] == ["para", "math_display", "para"]
    assert blocks[1].raw == "\\int x dx"


def test_parse_inline_spans():
    blocks = parse_markdown("Plain **bold** and *ital* and `code` and $x^2$ end.")
    para = blocks[0]
    assert para.kind == "para"
    kinds = {s.kind for s in para.spans}
    assert kinds == {"strong", "emph", "code_inline", "math_inline"}
    strong = next(s for s in para.spans if s.kind == "strong")
    assert para.text[strong.start : strong.end] == "bold"
    # The plain text carries no markdown syntax at all.
    assert "**" not in para.text and "`" not in para.text


def test_parse_link_span():
    blocks = parse_markdown("See [the docs](https://example.com) now.")
    link = next(s for s in blocks[0].spans if s.kind == "link")
    assert blocks[0].text[link.start : link.end] == "the docs"
    assert link.href == "https://example.com"


def test_parse_emph_nested_in_strong():
    blocks = parse_markdown("**bold with *inner* text**")
    spans = blocks[0].spans
    strong = next(s for s in spans if s.kind == "strong")
    emph = next(s for s in spans if s.kind == "emph")
    assert strong.start <= emph.start and emph.end <= strong.end
    assert blocks[0].text[emph.start : emph.end] == "inner"


def test_parse_hoists_images_to_blocks():
    blocks = parse_markdown(
        "Text before ![alt text](https://example.com/a.png) text after."
    )
    kinds = [b.kind for b in blocks]
    assert "image" in kinds
    img = next(b for b in blocks if b.kind == "image")
    assert img.images == [ImageRef(alt="alt text", url="https://example.com/a.png")]
    para = next(b for b in blocks if b.kind == "para")
    assert "![" not in para.text


def test_parse_is_pure_no_filesystem(tmp_path):
    before = set(tmp_path.iterdir())
    parse_markdown("![x](https://example.com/img.png)\n\n# Head\n\n`code`")
    assert set(tmp_path.iterdir()) == before


# --- Emission ---------------------------------------------------------------


def test_emit_golden_fragments(tmp_path):
    out = ir_render("Plain **bold** and *ital* and [t](https://e.com).", tmp_path)
    assert "#strong[bold]/**/" in out
    assert "#emph[ital]/**/" in out
    assert '#link("https://e.com")[t]/**/' in out


def test_emit_heading_and_list(tmp_path):
    out = ir_render("Intro.\n\n## Sub Head\n\n* item one", tmp_path)
    assert "== Sub Head" in out
    assert "- item one" in out


def test_emit_escapes_plain_text_once(tmp_path):
    out = ir_render("Costs $5 and #tag [brackets]", tmp_path)
    assert "\\$5" in out
    assert "\\#tag" in out
    assert "\\[brackets\\]" in out
    compile_snippet(out)


def test_emit_math(tmp_path):
    out = ir_render("Inline $x^2$ and display:\n\n$$\\int x dx$$", tmp_path)
    assert "#mi(` x^2 `)/**/" in out
    assert "#mitex(" in out
    compile_snippet(out)


def test_emit_salience_span_renders_smart_sentence(tmp_path):
    block = Block(
        kind="para",
        text="Important claim. Filler sentence.",
        spans=[
            Span(start=0, end=16, kind="salience", weight=0.9),
            Span(start=17, end=33, kind="salience", weight=0.1),
        ],
    )
    out = emit_blocks([block], tmp_path)
    assert "#smart-sentence(weight: 0.90)[Important claim.]/**/" in out
    assert "#smart-sentence(weight: 0.10)[Filler sentence.]/**/" in out
    compile_snippet(out)


def test_emit_entity_span_internal_link(tmp_path):
    block = Block(
        kind="para",
        text="Semiconductors are neat.",
        spans=[Span(start=0, end=14, kind="entity", label="ent-semi")],
    )
    out = emit_blocks([block], tmp_path)
    assert "#link(<ent-semi>)[Semiconductors]/**/" in out


def test_emit_overlapping_spans_split_deterministically(tmp_path):
    # [0,10) strong overlaps [5,15) emph -> emph truncated to [10,15)
    block = Block(
        kind="para",
        text="aaaaabbbbbcccccddddd",
        spans=[
            Span(start=0, end=10, kind="strong"),
            Span(start=5, end=15, kind="emph"),
        ],
    )
    out = emit_blocks([block], tmp_path)
    assert out == "#strong[aaaaabbbbb]/**/#emph[ccccc]/**/ddddd"


def test_emit_failed_image_degrades_to_link(tmp_path, mocker):
    mocker.patch(
        "papernews.typst_emit.urllib.request.urlopen",
        side_effect=OSError("no network"),
    )
    block = Block(kind="image", images=[ImageRef(alt="Alt", url="https://e.com/x.png")])
    out = emit_blocks([block], tmp_path)
    assert out == '#link("https://e.com/x.png")[Alt]/**/'
    compile_snippet(out)


# --- The parity gate ---------------------------------------------------------


@pytest.mark.parametrize("hostile", HOSTILE_STRINGS)
def test_ir_hostile_gauntlet_compiles(hostile, tmp_path):
    out = ir_render(hostile + " word" * 50, tmp_path)
    compile_snippet(out)


def test_ir_regression_corpus_compiles(tmp_path):
    fixture_file = Path(__file__).parent / "fixtures" / "test_db.json"
    if not fixture_file.exists():
        pytest.skip("No regression fixtures found.")
    cases = json.loads(fixture_file.read_text())
    for case in cases:
        out = ir_render(case["input"], tmp_path)
        compile_snippet(out)
        # The corpus pins expected escaping fidelity, not just compilability.
        assert case["expected_typst"] in out


@given(st.text())
@settings(max_examples=10, deadline=None)
def test_ir_property_safety(text):
    """Any input whatsoever must parse, emit, and compile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = ir_render(text, Path(tmpdir))
        compile_snippet(out)
