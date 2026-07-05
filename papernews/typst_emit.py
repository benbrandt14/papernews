"""Typed Typst emitter for the markdown IR.

`emit_blocks(blocks, workdir)` turns `Block`/`Span` records into Typst
markup. Escaping happens exactly once here, on the plain-text runs
between spans — there are no in-band sentinel tokens anywhere.

Span nesting is by containment; partially overlapping spans are
deterministically resolved by truncating the later-starting span at the
earlier one's end (pinned by tests).

Image fetching happens here (emit time), reusing the same
content-addressed assets scheme as the legacy path.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from PIL import Image

from papernews.models import Block, ImageRef, Span
from papernews.render import _render_code_block, typst_escape, typst_url

# Typst helpers the emitter's output relies on. The template preamble must
# include this (tests prepend it before compiling emitted fragments).
PREAMBLE = """
#let smart-sentence(weight: 1.0, body) = {
  if weight >= 0.75 { text(weight: "semibold", body) }
  else if weight <= 0.25 { text(fill: luma(40%), body) }
  else { body }
}
"""


def emit_blocks(blocks: list[Block], workdir: Path) -> str:
    out: list[str] = []
    for block in blocks:
        rendered = _emit_block(block, workdir)
        if rendered:
            out.append(rendered)
    return "\n\n".join(out)


def _emit_block(block: Block, workdir: Path) -> str:
    if block.kind == "code":
        return _render_code_block(block.raw).strip("\n")

    if block.kind == "math_display":
        content = block.raw
        fence_len = 3
        while "`" * fence_len in content:
            fence_len += 1
        fence = "`" * fence_len
        return f"#mitex({fence}\n{content}\n{fence})/**/"

    if block.kind == "image":
        return _emit_images(block.images, workdir)

    inline = _emit_inline(block.text, block.spans)

    if block.kind == "heading":
        return "=" * max(1, block.level) + " " + inline
    if block.kind == "quote":
        return f"#quote(block: true)[\n{inline}\n]"
    if block.kind == "list_item":
        return "  " * block.level + "- " + inline
    return inline  # para


# --- Inline emission --------------------------------------------------------


def _normalize_spans(spans: list[Span], text_len: int) -> list[Span]:
    """Clamp, drop empties, and resolve partial overlaps by truncation."""
    clamped = [
        s.model_copy(update={"start": max(0, s.start), "end": min(text_len, s.end)})
        for s in spans
    ]
    clamped = [s for s in clamped if s.start < s.end]
    # Sort outermost-first so containment nesting falls out naturally.
    clamped.sort(key=lambda s: (s.start, -(s.end - s.start)))

    result: list[Span] = []
    for span in clamped:
        for prev in result:
            if span.start < prev.end < span.end:
                # Partial overlap: truncate the later-starting span.
                span = span.model_copy(update={"start": prev.end})
                if span.start >= span.end:
                    break
        else:
            result.append(span)
    # Truncation can perturb ordering; restore outermost-first order so
    # every span's children sit consecutively after it.
    result.sort(key=lambda s: (s.start, -(s.end - s.start)))
    return result


def _emit_inline(text: str, spans: list[Span]) -> str:
    return _emit_range(text, _normalize_spans(spans, len(text)), 0, len(text))


def _emit_range(text: str, spans: list[Span], start: int, end: int) -> str:
    """Emit text[start:end], applying the given spans (all within range)."""
    out: list[str] = []
    pos = start
    i = 0
    while i < len(spans):
        span = spans[i]
        if span.start < pos:  # swallowed by a previous sibling; skip
            i += 1
            continue
        if span.start > pos:
            out.append(typst_escape(text[pos : span.start]))

        # Children are the consecutive following spans contained in this
        # one (overlap normalization guarantees containment-or-disjoint).
        j = i + 1
        while j < len(spans) and spans[j].start < span.end:
            j += 1
        out.append(_emit_span(text, span, spans[i + 1 : j]))
        pos = span.end
        i = j
    if pos < end:
        out.append(typst_escape(text[pos:end]))
    return "".join(out)


def _emit_span(text: str, span: Span, children: list[Span]) -> str:
    content = text[span.start : span.end]

    if span.kind == "code_inline":
        fence_len = 1
        while "`" * fence_len in content:
            fence_len += 1
        fence = "`" * fence_len
        safe = content
        if safe.startswith("`"):
            safe = " " + safe
        if safe.endswith("`") or safe.endswith("\\"):
            safe = safe + " "
        return f"{fence}{safe}{fence}"

    if span.kind == "math_inline":
        body = content.replace("\n", " ")
        fence_len = 1
        while "`" * fence_len in body:
            fence_len += 1
        if fence_len == 2:
            fence_len = 3
        fence = "`" * fence_len
        return f"#mi({fence} {body} {fence})/**/"

    inner = _emit_range(text, children, span.start, span.end)

    if span.kind == "strong":
        return f"#strong[{inner}]/**/"
    if span.kind == "emph":
        return f"#emph[{inner}]/**/"
    if span.kind == "link":
        return f'#link("{typst_url(span.href or "")}")[{inner}]/**/'
    if span.kind == "entity":
        return f"#link(<{span.label}>)[{inner}]/**/" if span.label else inner
    if span.kind == "salience":
        weight = span.weight if span.weight is not None else 1.0
        return f"#smart-sentence(weight: {weight:.2f})[{inner}]/**/"

    return inner


# --- Images -----------------------------------------------------------------


def _fetch_image(ref: ImageRef, assets_dir: Path) -> Path | None:
    """Download (or reuse) one image; returns the local path or None."""
    url_hash = hashlib.sha256(ref.url.encode()).hexdigest()[:16]
    existing = list(assets_dir.glob(f"{url_hash}.*"))
    if existing:
        return existing[0]

    try:
        req = urllib.request.Request(ref.url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()

            if data.startswith(b"\xff\xd8"):
                ext = ".jpg"
            elif data.startswith(b"\x89PNG\r\n\x1a\n"):
                ext = ".png"
            elif data.startswith(b"GIF8"):
                ext = ".gif"
            elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
                ext = ".webp"
            elif b"<svg" in data[:1024].lower():
                ext = ".svg"
            else:
                ctype = response.info().get_content_type()
                ext = {
                    "image/png": ".png",
                    "image/gif": ".gif",
                    "image/webp": ".webp",
                    "image/svg+xml": ".svg",
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                }.get(ctype, "")
                if not ext:
                    raise ValueError(f"Unrecognized image content-type: {ctype}")

            img_path = assets_dir / f"{url_hash}{ext}"
            img_path.write_bytes(data)
            return img_path
    except Exception as e:
        sys.stderr.write(f"  [warn] failed to fetch image {ref.url}: {e}\n")
        return None


def _emit_images(images: list[ImageRef], workdir: Path) -> str:
    assets_dir = workdir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[str] = []
    figures: list[tuple[str, float, int]] = []

    def flush_figures() -> None:
        if not figures:
            return
        if len(figures) <= 3:
            for filename, aspect, width_px in figures:
                if aspect > 1.8 and width_px > 600:
                    fig_props = 'placement: auto, scope: "parent"'
                    img_props = "width: 100%"
                elif aspect < 0.9:
                    fig_props = "placement: auto"
                    img_props = "width: 55%"
                else:
                    fig_props = "placement: auto"
                    img_props = "width: 100%"
                pieces.append(
                    f'#figure(image("assets/{filename}", {img_props}), {fig_props})/**/'
                )
        else:
            items = ", ".join(
                f'image("assets/{f}", width: 100%)' for f, _, _ in figures
            )
            pieces.append(
                f"#figure(grid(columns: 2, gutter: 6pt, {items}), placement: auto)/**/"
            )
        figures.clear()

    for ref in images:
        img_path = _fetch_image(ref, assets_dir)
        if img_path is None:
            flush_figures()
            pieces.append(f'#link("{typst_url(ref.url)}")[{typst_escape(ref.alt)}]/**/')
            continue

        aspect, width_px = 1.5, 800
        if img_path.suffix != ".svg":
            try:
                with Image.open(img_path) as img_file:
                    width_px, height_px = img_file.size
                    aspect = width_px / height_px
            except Exception:
                pass
        figures.append((img_path.name, aspect, width_px))

    flush_figures()
    return "\n".join(pieces)
