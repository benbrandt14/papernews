"""Tests for the rendering module: escaping helpers and PDF compilation.

Body-conversion behavior (markdown → Typst) lives in tests/test_ir.py —
the IR parser + typed emitter are the only render path.
"""

import re

import pytest
import typst
from hypothesis import given
from hypothesis import strategies as st

from papernews.markdown_ir import _strip_leading_metadata
from papernews.models import RenderContext
from papernews.render import (
    _TYPST_REPLACE,
    RenderError,
    build_pdf,
    typst_escape,
    typst_url,
)
from papernews.typst_emit import _render_code_block

# --- Escaping helpers --------------------------------------------------------


def test_typst_escape_basics():
    assert typst_escape("$5 and #tag [x] @user") == r"\$5 and \#tag \[x\] \@user"
    assert typst_escape(None) == ""
    assert typst_escape(123) == "123"


@given(st.text())
def test_typst_escape_property(text):
    escaped = typst_escape(text)
    # Every special character in the input must come out escaped: no bare
    # occurrences that are not preceded by a backslash.
    for ch in list(_TYPST_REPLACE) + ["#"]:
        for m in re.finditer(re.escape(ch), escaped):
            if ch == "\\":
                continue  # backslashes multiply; parity is checked by compile tests
            assert m.start() > 0 and escaped[m.start() - 1] == "\\", (
                f"unescaped {ch!r} in {escaped!r}"
            )


@given(st.text())
def test_typst_url_property(url):
    out = typst_url(url)
    assert '"' not in out.replace('\\"', "")


@given(st.text())
def test_strip_leading_metadata_property(text):
    stripped = _strip_leading_metadata(text)
    assert isinstance(stripped, str)
    assert len(stripped) <= len(text)


@given(st.text())
def test_render_code_block_property(code):
    rendered = _render_code_block(code)
    # rendered should start and end with fences that are at least 3 backticks long
    # and strictly longer than any internal backtick sequence.
    assert rendered.startswith("\n\n```")
    assert rendered.endswith("```\n\n")
    assert code in rendered


# --- build_pdf ----------------------------------------------------------------


def _ctx(**overrides) -> RenderContext:
    defaults = dict(
        date="2026-01-01",
        generation_time="Now",
        total_tokens="0",
        total_cost="0",
        articles=[],
    )
    defaults.update(overrides)
    return RenderContext(**defaults)


def test_build_pdf_raises_render_error_on_compile_failure(tmp_path, mocker):
    """A failed Typst compile must raise, never silently return a stale path."""
    mocker.patch("typst.compile", side_effect=typst.TypstError("boom"))

    with pytest.raises(RenderError, match="2026-01-01"):
        build_pdf(_ctx(), tmp_path)


def test_build_pdf_produces_pdf(tmp_path):
    """Happy path: an empty edition compiles to a real PDF."""
    pdf = build_pdf(_ctx(), tmp_path)
    assert pdf.exists()
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_build_pdf_parses_blocks_for_bodies_without_ir(tmp_path):
    """Hand-constructed articles (no pre-parsed blocks) still render their
    markdown bodies — build_pdf parses on the fly."""
    from papernews.models import ArticleChunk

    art = ArticleChunk(
        category="Sci",
        source="example.com",
        title="T",
        summary="S",
        body_markdown="Some **bold** body text.",
        url="https://example.com",
    )
    pdf = build_pdf(_ctx(articles=[art]), tmp_path)
    assert pdf.exists()
    typ = (tmp_path / ".build" / "2026-01-01.typ").read_text()
    assert "#strong[bold]/**/" in typ


def test_build_pdf_renders_frontmatter_index_with_funnel(tmp_path):
    """The front-matter index page carries the triage-funnel telemetry and a
    categorized index of the edition's articles."""
    from papernews.models import ArticleChunk, FunnelStats

    arts = [
        ArticleChunk(
            category="Science",
            source="example.com",
            title=f"Discovery Number {i}",
            summary="A concise summary of what happened in the field.",
            body_markdown="Body paragraph with detail.",
            url=f"https://example.com/{i}",
        )
        for i in range(3)
    ]
    ctx = _ctx(
        articles=arts,
        lead_article_index=0,
        stats=FunnelStats(ingested=142, after_filter=38, after_budget=14, selected=3),
    )
    pdf = build_pdf(ctx, tmp_path)
    assert pdf.exists()
    typ = (tmp_path / ".build" / "2026-01-01.typ").read_text()

    # Funnel counts surface on the index page.
    assert "The Index" in typ
    assert "142" in typ and "38" in typ and "14" in typ
    # Secondary stories appear in the categorized index.
    assert "Discovery Number 1" in typ


def test_build_pdf_renders_curiosity_box(tmp_path):
    """Answered questions from the curiosity queue render as a front-matter box."""
    from papernews.models import Curiosity, FrontpageDecorations

    ctx = _ctx(
        decorations=FrontpageDecorations(
            curiosities=[
                Curiosity(
                    question="Why do neutron stars glitch?",
                    answer_title="Superfluid vortex unpinning in pulsars",
                    answer_url="https://doi.org/10.1234/abc",
                )
            ]
        ),
    )
    pdf = build_pdf(ctx, tmp_path)
    assert pdf.exists()
    typ = (tmp_path / ".build" / "2026-01-01.typ").read_text()
    assert "Answered from the queue" in typ
    assert "Why do neutron stars glitch?" in typ
    assert "https://doi.org/10.1234/abc" in typ


def test_build_pdf_resolves_entity_interlinks(tmp_path):
    """An entity span linking to another article's <artN> anchor must compile —
    proving the emitter's internal links resolve against the real template."""
    from papernews.config import AppConfig
    from papernews.markdown_ir import parse_markdown
    from papernews.models import ArticleChunk
    from papernews.plugins import entity_plugin
    from papernews.store import SimpleStore

    a0 = ArticleChunk(
        category="Space",
        source="example.com",
        title="Deep dive",
        summary="s",
        body_markdown=(
            "The James Webb Space Telescope resolved a faint galaxy. "
            "The James Webb Space Telescope imaged a second target too."
        ),
        url="https://example.com/0",
    )
    a1 = ArticleChunk(
        category="Space",
        source="example.com",
        title="Brief",
        summary="s",
        body_markdown="Elsewhere, the James Webb Space Telescope data went public.",
        url="https://example.com/1",
    )
    for art in (a0, a1):
        art.blocks = parse_markdown(art.body_markdown)
    entity_plugin.enrich_articles(
        [a0, a1], AppConfig(), SimpleStore(str(tmp_path / "state.db"))
    )

    pdf = build_pdf(_ctx(articles=[a0, a1], lead_article_index=0), tmp_path)
    assert pdf.exists()
    assert pdf.read_bytes()[:4] == b"%PDF"
    typ = (tmp_path / ".build" / "2026-01-01.typ").read_text()
    assert "#link(<art1>)" in typ
