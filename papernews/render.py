from __future__ import annotations

import functools
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

import jinja2
from PIL import Image

from .adapter import render_context_to_template_vars
from .models import RenderContext


class RenderError(RuntimeError):
    """Typst compilation failed. `debug_path` points at diagnostics, if any."""

    def __init__(self, message: str, debug_path: Path | None = None):
        super().__init__(message)
        self.debug_path = debug_path


_TYPST_REPLACE = {
    "\\": r"\\",
    "*": r"\*",
    "_": r"\_",
    "$": r"\$",
    "<": r"\<",
    ">": r"\>",
    "@": r"\@",
    "`": r"\`",
    "{": r"\{",
    "}": r"\}",
    '"': r"\"",
    "[": r"\[",
    "]": r"\]",
}


def typst_escape(s: object) -> str:
    if s is None:
        return ""
    res = "".join(_TYPST_REPLACE.get(c, c) for c in str(s))
    # Unconditionally escape all hashtags so titles like "#1" don't crash the compiler
    return res.replace("#", r"\#")


def typst_url(url: str) -> str:
    if not url:
        return ""
    return url.replace("\\", "\\\\").replace('"', '\\"')


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*\n?(.*?)```", re.DOTALL)
_INLINE_RE = re.compile(r"`([^`\n]+)`")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^\)]+)\)")
_GALLERY_RE = re.compile(r"(?:!\[[^\]]*\]\(https?://[^\)]+\)\s*)+")
_LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^\)]+)\)")

_MATH_RE = re.compile(
    r"\$\$(?P<dd>.+?)\$\$"
    r"|\\\[(?P<br>.+?)\\\]"
    r"|(?<![\\$])\$(?P<sd>[^$\n][^$]*?)\$(?!\d)"
    r"|\\\((?P<pr>.+?)\\\)",
    re.DOTALL,
)

_STRONG_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
_EMPH_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_STRONG_US_RE = re.compile(r"__(?!\s)(.+?)(?<!\s)__")
_EMPH_US_RE = re.compile(r"(?<![a-zA-Z0-9_])_(?!\s)(.+?)(?<!\s)_(?![a-zA-Z0-9_])")


def _stash_typography(text: str) -> str:

    def stash_b(m: re.Match) -> str:
        return f"\x00BSTART\x00{m.group(1)}\x00BEND\x00"

    def stash_i(m: re.Match) -> str:
        return f"\x00ISTART\x00{m.group(1)}\x00IEND\x00"

    stashed = _STRONG_RE.sub(stash_b, text)
    stashed = _EMPH_RE.sub(stash_i, stashed)
    stashed = _STRONG_US_RE.sub(stash_b, stashed)
    stashed = _EMPH_US_RE.sub(stash_i, stashed)
    return stashed


def _stash_links(text: str) -> tuple[str, list[str]]:
    bits: list[str] = []

    def stash(m: re.Match) -> str:
        raw_text = m.group(1)
        url = m.group(2)

        safe_text = typst_escape(raw_text)
        safe_url = typst_url(url)

        bits.append(f'#link("{safe_url}")[{safe_text}]/**/')
        return f"\x00LNK{len(bits) - 1}\x00"

    return _LINK_RE.sub(stash, text), bits


def _stash_images(text: str, workdir: Path) -> tuple[str, list[str]]:
    bits: list[str] = []

    def stash(m: re.Match) -> str:
        raw_match = m.group(0)
        img_matches = _IMAGE_RE.findall(raw_match)

        assets_dir = workdir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        processed_images: list[str | tuple[str, float, int]] = []

        for alt, url in img_matches:
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

            existing = list(assets_dir.glob(f"{url_hash}.*"))
            filename = None
            img_path = None

            if existing:
                filename = existing[0].name
                img_path = existing[0]
            else:
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"}
                    )
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
                            if ctype == "image/png":
                                ext = ".png"
                            elif ctype == "image/gif":
                                ext = ".gif"
                            elif ctype == "image/webp":
                                ext = ".webp"
                            elif ctype == "image/svg+xml":
                                ext = ".svg"
                            elif ctype in ("image/jpeg", "image/jpg"):
                                ext = ".jpg"
                            else:
                                raise ValueError(
                                    f"Unrecognized image data and content-type: {ctype}"
                                )

                        filename = f"{url_hash}{ext}"
                        img_path = assets_dir / filename
                        with open(img_path, "wb") as f:
                            f.write(data)

                except Exception as e:
                    sys.stderr.write(f"  [warn] failed to fetch image {url}: {e}\n")
                    # If download fails, stash it immediately as a safe Typst link
                    safe_alt = typst_escape(alt)
                    safe_url = typst_url(url)
                    bits.append(f'#link("{safe_url}")[{safe_alt}]/**/')
                    processed_images.append(f"\x00IMG{len(bits) - 1}\x00")
                    continue

            aspect = 1.5
            width_px = 800
            if img_path and img_path.exists() and img_path.suffix != ".svg":
                try:
                    with Image.open(img_path) as img_file:
                        width_px, height_px = img_file.size
                        aspect = width_px / height_px
                except Exception:
                    pass

            processed_images.append((filename, aspect, width_px))

        out_str = ""
        current_valid_group: list[tuple[str, float, int]] = []

        def flush_valid_group() -> None:
            nonlocal out_str
            if not current_valid_group:
                return
            if len(current_valid_group) <= 3:
                figs = []
                for filename, aspect, width_px in current_valid_group:
                    if aspect > 1.8 and width_px > 600:
                        fig_props = 'placement: auto, scope: "parent"'
                        img_props = "width: 100%"
                    elif aspect < 0.9:
                        fig_props = "placement: auto"
                        img_props = "width: 55%"
                    else:
                        fig_props = "placement: auto"
                        img_props = "width: 100%"

                    figs.append(
                        f'#figure(image("assets/{filename}", {img_props}), {fig_props})/**/'
                    )

                bits.append("\n".join(figs))
                out_str += f"\x00IMG{len(bits) - 1}\x00"
            else:
                grid_items = []
                for filename, _, _ in current_valid_group:
                    grid_items.append(f'image("assets/{filename}", width: 100%)')

                cols = 2
                grid_str = (
                    f"grid(columns: {cols}, gutter: 6pt, {', '.join(grid_items)})"
                )
                bits.append(f"#figure({grid_str}, placement: auto)/**/")
                out_str += f"\x00IMG{len(bits) - 1}\x00"
            current_valid_group.clear()

        for item in processed_images:
            if isinstance(item, tuple):
                current_valid_group.append(item)
            else:
                # Flush existing valid images, then output the fallback link marker
                flush_valid_group()
                out_str += item

        flush_valid_group()
        return out_str

    return _GALLERY_RE.sub(stash, text), bits


def _render_code_block(code: str) -> str:
    fence_len = 3
    while "`" * fence_len in code:
        fence_len += 1
    fence = "`" * fence_len
    return f"\n\n{fence}\n{code}\n{fence}\n\n"


def _process_inline(text: str) -> str:
    parts = _INLINE_RE.split(text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(typst_escape(part))
        else:
            fence_len = 1
            while "`" * fence_len in part:
                fence_len += 1
            fence = "`" * fence_len

            safe_part = part
            if safe_part.startswith("`"):
                safe_part = " " + safe_part
            if safe_part.endswith("`") or safe_part.endswith("\\"):
                safe_part = safe_part + " "

            out.append(f"{fence}{safe_part}{fence}")
    return "".join(out)


def _stash_math(text: str) -> tuple[str, list[str]]:
    bits: list[str] = []

    def stash(m: re.Match) -> str:
        if m.group("dd") is not None:
            content = m.group("dd").strip()
            is_display = True
        elif m.group("br") is not None:
            content = m.group("br").strip()
            is_display = True
        elif m.group("sd") is not None:
            content = m.group("sd").strip()
            is_display = False
        else:  # pr
            content = m.group("pr").strip()
            is_display = False

        if not is_display:
            content = content.replace("\n", " ")

        fence_len = 3 if is_display else 1
        while "`" * fence_len in content:
            fence_len += 1
        if fence_len == 2:
            fence_len = 3

        fence = "`" * fence_len

        if is_display:
            bits.append(f"#mitex({fence}\n{content}\n{fence})/**/")
        else:
            bits.append(f"#mi({fence} {content} {fence})/**/")

        return f"\x00MB{len(bits) - 1}\x00"

    return _MATH_RE.sub(stash, text), bits


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


def typst_body(text: str, workdir: Path) -> str:
    if not text:
        return ""

    # Prevent malicious injection of our internal marker tokens
    text = text.replace("\x00", "")

    text = _strip_leading_metadata(text)

    blocks: list[str] = []

    def stash_code(m: re.Match) -> str:
        blocks.append(m.group(1))
        return f"\x00CB{len(blocks) - 1}\x00"

    stashed = _FENCE_RE.sub(stash_code, text)

    inline_code_bits: list[str] = []

    def stash_inline(m: re.Match) -> str:
        part = m.group(1)
        fence_len = 1
        while "`" * fence_len in part:
            fence_len += 1
        fence = "`" * fence_len
        safe_part = part
        if safe_part.startswith("`"):
            safe_part = " " + safe_part
        if safe_part.endswith("`") or safe_part.endswith("\\"):
            safe_part = safe_part + " "
        inline_code_bits.append(f"{fence}{safe_part}{fence}")
        return f"\x00IC{len(inline_code_bits) - 1}\x00"

    stashed = _INLINE_RE.sub(stash_inline, stashed)

    stashed, img_bits = _stash_images(stashed, workdir)
    stashed, math_bits = _stash_math(stashed)
    stashed = _stash_typography(stashed)
    stashed, lnk_bits = _stash_links(stashed)

    # --- NEW: Convert Markdown structures to Typst equivalents ---
    # Convert Markdown headers to Typst headers
    stashed = re.sub(
        r"^(#{1,6})\s+",
        lambda m: "=" * len(m.group(1)) + " ",
        stashed,
        flags=re.MULTILINE,
    )

    # Convert Markdown unordered lists (asterisks) to Typst hyphens
    stashed = re.sub(r"^(\s*)\*\s+", r"\1- ", stashed, flags=re.MULTILINE)
    # -----------------------------------------------------------

    paras = (
        re.split(r"\n\s*\n+", stashed.strip())
        if "\n\n" in stashed
        else re.split(r"\n+", stashed.strip())
    )
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue

        # --- NEW: Handle Markdown blockquotes ---
        is_blockquote = False
        if p.startswith(">"):
            lines = p.split("\n")
            # If all lines in this paragraph start with a blockquote marker
            if all(line.lstrip().startswith(">") for line in lines):
                is_blockquote = True
                p = "\n".join(line.lstrip()[1:].strip() for line in lines)
        # ----------------------------------------

        # 1. Safely escape everything outside of the stashed markers
        rendered = typst_escape(p)

        def expand_cb(mm: re.Match) -> str:
            return _render_code_block(blocks[int(mm.group(1))])

        def expand_ic(mm: re.Match) -> str:
            return inline_code_bits[int(mm.group(1))]

        def expand_img(mm: re.Match) -> str:
            return img_bits[int(mm.group(1))]

        def expand_math(mm: re.Match) -> str:
            return math_bits[int(mm.group(1))]

        def expand_lnk(mm: re.Match) -> str:
            return lnk_bits[int(mm.group(1))]

        # 2. Expand all isolated blocks back into the text stream
        while True:
            old = rendered
            rendered = re.sub(r"\x00CB(\d+)\x00", expand_cb, rendered)
            rendered = re.sub(r"\x00IC(\d+)\x00", expand_ic, rendered)
            rendered = re.sub(r"\x00IMG(\d+)\x00", expand_img, rendered)
            rendered = re.sub(r"\x00MB(\d+)\x00", expand_math, rendered)
            rendered = re.sub(r"\x00LNK(\d+)\x00", expand_lnk, rendered)

            rendered = rendered.replace("\x00BSTART\x00", "#strong[")
            rendered = rendered.replace("\x00BEND\x00", "]/**/")
            rendered = rendered.replace("\x00ISTART\x00", "#emph[")
            rendered = rendered.replace("\x00IEND\x00", "]/**/")

            if old == rendered:
                break

        # Clean up any trailing unmatched typography markers
        # if the user input happened to maliciously contain \x00ISTART\x00 without \x00IEND\x00
        # Though typst_escape will escape the input BEFORE we get here, these were originally
        # inserted by regex so it should always be paired, but if the raw text literally contained
        # \x00ISTART\x00 it would be replaced above unbalanced. Let's fix this by escaping null bytes in input.

        if is_blockquote:
            rendered = f"#quote(block: true)[\n{rendered}\n]"

        out.append(rendered)

    return "\n\n".join(out)


def _env(tpl_dir: Path, workdir: Path) -> jinja2.Environment:
    env = jinja2.Environment(
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((=",
        comment_end_string="=))",
        loader=jinja2.FileSystemLoader(str(tpl_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["typst"] = typst_escape
    env.filters["typst_url"] = typst_url
    env.filters["typst_body"] = functools.partial(typst_body, workdir=workdir)
    return env


def build_pdf(ctx: RenderContext, out_dir: Path) -> Path:
    """Render the edition described by `ctx` to a PDF in `out_dir`.

    Raises RenderError when Typst compilation fails (after writing
    diagnostics to the .build workdir).
    """
    date = ctx.date
    tpl_dir = Path(__file__).parent
    workdir = out_dir / ".build"
    workdir.mkdir(parents=True, exist_ok=True)

    env = _env(tpl_dir, workdir)
    tpl = env.get_template("template.typ.j2")

    template_vars = render_context_to_template_vars(ctx)
    # Structured-IR bodies are emitted here because emission needs the
    # workdir (image fetching). Lazy import: typst_emit uses this module's
    # escape helpers.
    if any(chunk.blocks for chunk in ctx.articles):
        from papernews.typst_emit import emit_blocks

        for chunk, art_dict in zip(ctx.articles, template_vars["articles"]):
            if chunk.blocks:
                art_dict["body_typst"] = emit_blocks(chunk.blocks, workdir)

    typst_source = tpl.render(**template_vars)

    typst_path = workdir / f"{date}.typ"
    typst_path.write_text(typst_source, encoding="utf-8")

    pdf_dst = out_dir / f"{date}.pdf"

    import typst

    try:
        typst.compile(str(typst_path), output=str(pdf_dst))
    except typst.TypstError as e:
        sys.stderr.write("\n" + "=" * 70 + "\n")
        sys.stderr.write("[FATAL] TYPST COMPILATION FAILED\n")
        sys.stderr.write("=" * 70 + "\n")
        sys.stderr.write(f"Error Message: {e}\n\n")

        debug_path: Path | None = None
        lines = typst_source.split("\n")
        line_match = re.search(r"line (\d+)", str(e), re.IGNORECASE)
        if not line_match:
            line_match = re.search(r":(\d+):\d+", str(e))

        if line_match:
            err_line = int(line_match.group(1))
            sys.stderr.write(f"[Context around line {err_line}]:\n")
            start = max(0, err_line - 5)
            end = min(len(lines), err_line + 5)
            for i in range(start, end):
                prefix = ">> " if i + 1 == err_line else "   "
                sys.stderr.write(f"{prefix}{i + 1:04d} | {lines[i]}\n")
        else:
            # If Typst drops the line number, generate a numbered debug file
            debug_path = workdir / f"{date}_DEBUG_NUMBERED.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                for i, line in enumerate(lines):
                    f.write(f"{i + 1:04d} | {line}\n")

            sys.stderr.write(
                "[Context] Typst did not provide a specific line number.\n"
            )
            sys.stderr.write(
                "(This usually means a layout rule was violated, like pagebreaks inside columns).\n\n"
            )
            sys.stderr.write(
                "--> I have generated a numbered source file for you to inspect here:\n"
            )
            sys.stderr.write(f"--> {debug_path}\n")

        sys.stderr.write("=" * 70 + "\n")
        raise RenderError(
            f"Typst compilation failed for {date}: {e}", debug_path=debug_path
        ) from e

    return pdf_dst
