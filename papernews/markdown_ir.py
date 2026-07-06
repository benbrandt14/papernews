"""Markdown → structured IR parser.

`parse_markdown(text)` turns LLM/scraper markdown into a list of typed
`Block`s whose inline formatting lives in `Span` records with plain-text
character offsets. This replaces the NUL-sentinel stash pipeline for the
body path: enrichers (salience scoring, entity linking) operate on
`Block.text` with ordinary offsets and never see markdown or Typst, and
escaping happens exactly once, at emission (`papernews.typst_emit`).

The parser is pure — no network, no filesystem. Images are captured as
`ImageRef`s on dedicated image blocks; fetching happens at emit time.
"""

from __future__ import annotations

import re

from papernews.models import Block, ImageRef, Span

# --- Markdown syntax (shared vocabulary of the parser) ----------------------

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*\n?(.*?)```", re.DOTALL)
_INLINE_RE = re.compile(r"`([^`\n]+)`")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^\)]+)\)")
_LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^\)]+)\)")

_STRONG_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
_EMPH_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_STRONG_US_RE = re.compile(r"__(?!\s)(.+?)(?<!\s)__")
_EMPH_US_RE = re.compile(r"(?<![a-zA-Z0-9_])_(?!\s)(.+?)(?<!\s)_(?![a-zA-Z0-9_])")

_LEADING_DATE_RE = re.compile(
    r"^\s*("
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}"
    r"|\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r")\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_leading_metadata(text: str) -> str:
    """Robustly strips leading LLM hallucinations like dates or redundant article titles."""
    lines = text.split("\n")

    while lines:
        first = lines[0].strip()
        if not first:
            lines.pop(0)
            continue
        if _LEADING_DATE_RE.fullmatch(first):
            lines.pop(0)
            continue
        if re.match(r"^(=+|#+)\s+", first):
            lines.pop(0)
            continue
        break

    return "\n".join(lines)


# Display math only — inline math is handled during inline parsing.
_MATH_DISPLAY_RE = re.compile(r"\$\$(?P<dd>.+?)\$\$|\\\[(?P<br>.+?)\\\]", re.DOTALL)

_MATH_INLINE_RE = re.compile(
    r"(?<![\\$])\$(?P<sd>[^$\n][^$]*?)\$(?!\d)|\\\((?P<pr>.+?)\\\)",
    re.DOTALL,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM_RE = re.compile(r"^(\s*)[-*]\s+(.*)$")


def parse_markdown(text: str) -> list[Block]:
    if not text:
        return []

    text = text.replace("\x00", "")
    text = _strip_leading_metadata(text)

    # Phase A: slice the document into text / code / display-math segments
    # in source order, so paragraph splitting never runs inside a fence.
    segments = _extract_block_segments(text)

    # Match the legacy paragraph-split heuristic, computed document-wide:
    # blank-line paragraphs when any exist, otherwise one line = one para.
    plain_text = "".join(s for kind, s in segments if kind == "text")
    split_re = r"\n\s*\n+" if "\n\n" in plain_text else r"\n+"

    blocks: list[Block] = []
    for kind, payload in segments:
        if kind == "code":
            blocks.append(Block(kind="code", raw=payload))
        elif kind == "math_display":
            blocks.append(Block(kind="math_display", raw=payload.strip()))
        else:
            for para in re.split(split_re, payload.strip()):
                para = para.strip()
                if para:
                    blocks.extend(_parse_paragraph(para))
    return blocks


def _extract_block_segments(text: str) -> list[tuple[str, str]]:
    """Split into ("text"|"code"|"math_display", payload) in source order."""
    combined = re.compile(
        f"(?P<code>{_FENCE_RE.pattern})|(?P<math>{_MATH_DISPLAY_RE.pattern})",
        re.DOTALL,
    )
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in combined.finditer(text):
        if m.start() > pos:
            segments.append(("text", text[pos : m.start()]))
        if m.group("code") is not None:
            # group 2 is _FENCE_RE's capture (the code body)
            segments.append(("code", m.group(2)))
        else:
            body = m.group("dd") if m.group("dd") is not None else m.group("br")
            segments.append(("math_display", body or ""))
        pos = m.end()
    if pos < len(text):
        segments.append(("text", text[pos:]))
    return segments


def _parse_paragraph(para: str) -> list[Block]:
    """Classify one paragraph and parse its inline content."""
    hoisted_images: list[Block] = []

    # Images are hoisted to their own blocks (they render as floating
    # figures anyway, so intra-paragraph position is irrelevant).
    def hoist(m: re.Match) -> str:
        refs = [ImageRef(alt=a, url=u) for a, u in _IMAGE_RE.findall(m.group(0))]
        if refs:
            hoisted_images.append(Block(kind="image", images=refs))
        return " "

    gallery_re = re.compile(r"(?:!\[[^\]]*\]\(https?://[^\)]+\)\s*)+")
    para = gallery_re.sub(hoist, para).strip()

    out: list[Block] = []
    lines = para.split("\n")

    # Blockquote: every line starts with '>' (legacy behavior).
    if para.startswith(">") and all(ln.lstrip().startswith(">") for ln in lines):
        inner = "\n".join(ln.lstrip()[1:].strip() for ln in lines)
        text, spans = _parse_inline(inner)
        out.append(Block(kind="quote", text=text, spans=spans))
        return hoisted_images + out

    # Line-oriented headings / list items mixed with prose.
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            text, spans = _parse_inline("\n".join(prose))
            if text.strip():
                out.append(Block(kind="para", text=text, spans=spans))
            prose.clear()

    for line in lines:
        h = _HEADING_RE.match(line)
        li = _LIST_ITEM_RE.match(line)
        if h:
            flush_prose()
            text, spans = _parse_inline(h.group(2))
            out.append(
                Block(kind="heading", level=len(h.group(1)), text=text, spans=spans)
            )
        elif li:
            flush_prose()
            text, spans = _parse_inline(li.group(2))
            out.append(
                Block(
                    kind="list_item",
                    level=len(li.group(1)) // 2,
                    text=text,
                    spans=spans,
                )
            )
        else:
            prose.append(line)
    flush_prose()

    if not out and para:
        text, spans = _parse_inline(para)
        if text.strip():
            out.append(Block(kind="para", text=text, spans=spans))

    return hoisted_images + out


# --- Inline parsing ---------------------------------------------------------
#
# A segment list is progressively refined: each pass may only split "plain"
# segments, so earlier (higher-priority) constructs are never re-parsed by
# later passes. Priority mirrors the legacy stash order:
#   inline code > inline math > strong > emph > links.

_Seg = tuple[str, str, str | None]  # (kind, text, href)


def _split_pass(
    segments: list[_Seg],
    regex: re.Pattern,
    kind: str,
    text_group: int | str = 1,
    href_group: int | None = None,
) -> list[_Seg]:
    result: list[_Seg] = []
    for seg_kind, seg_text, seg_href in segments:
        if seg_kind != "plain":
            result.append((seg_kind, seg_text, seg_href))
            continue
        pos = 0
        for m in regex.finditer(seg_text):
            if m.start() > pos:
                result.append(("plain", seg_text[pos : m.start()], None))
            if isinstance(text_group, str):
                captured = next(
                    (g for g in (m.group("sd"), m.group("pr")) if g is not None), ""
                )
            else:
                captured = m.group(text_group) or ""
            href = m.group(href_group) if href_group is not None else None
            result.append((kind, captured, href))
            pos = m.end()
        if pos < len(seg_text):
            result.append(("plain", seg_text[pos:], None))
    return result


def _parse_inline(s: str) -> tuple[str, list[Span]]:
    """Parse inline markdown into (plain_text, spans)."""
    segments: list[_Seg] = [("plain", s, None)]
    segments = _split_pass(segments, _INLINE_RE, "code_inline")
    segments = _split_pass(segments, _MATH_INLINE_RE, "math_inline", text_group="named")
    segments = _split_pass(segments, _STRONG_RE, "strong")
    segments = _split_pass(segments, _STRONG_US_RE, "strong")
    segments = _split_pass(segments, _EMPH_RE, "emph")
    segments = _split_pass(segments, _EMPH_US_RE, "emph")
    segments = _split_pass(segments, _LINK_RE, "link", text_group=1, href_group=2)

    text_parts: list[str] = []
    spans: list[Span] = []
    offset = 0

    for kind, seg_text, href in segments:
        if kind == "strong":
            # Emphasis may nest inside strong: parse the inner text.
            inner_text, inner_spans = _parse_inline_nested(seg_text)
            spans.append(
                Span(start=offset, end=offset + len(inner_text), kind="strong")
            )
            for sp in inner_spans:
                spans.append(
                    sp.model_copy(
                        update={"start": sp.start + offset, "end": sp.end + offset}
                    )
                )
            text_parts.append(inner_text)
            offset += len(inner_text)
            continue

        if kind == "math_inline":
            seg_text = seg_text.strip().replace("\n", " ")

        if kind != "plain":
            spans.append(
                Span(
                    start=offset,
                    end=offset + len(seg_text),
                    kind=kind,  # type: ignore[arg-type]
                    href=href,
                )
            )
        text_parts.append(seg_text)
        offset += len(seg_text)

    return "".join(text_parts), spans


def _parse_inline_nested(s: str) -> tuple[str, list[Span]]:
    """Emphasis and links inside a strong span."""
    segments: list[_Seg] = [("plain", s, None)]
    segments = _split_pass(segments, _EMPH_RE, "emph")
    segments = _split_pass(segments, _EMPH_US_RE, "emph")
    segments = _split_pass(segments, _LINK_RE, "link", text_group=1, href_group=2)

    text_parts: list[str] = []
    spans: list[Span] = []
    offset = 0
    for kind, seg_text, href in segments:
        if kind != "plain":
            spans.append(
                Span(
                    start=offset,
                    end=offset + len(seg_text),
                    kind=kind,  # type: ignore[arg-type]
                    href=href,
                )
            )
        text_parts.append(seg_text)
        offset += len(seg_text)
    return "".join(text_parts), spans
