from __future__ import annotations

import functools
import hashlib
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import jinja2

_TYPST_REPLACE = {
    "\\": r"\\",
    "$": r"\$",
    "<": r"\<",
    ">": r"\>",
    "@": r"\@",
    "`": r"\`",
    "{": r"\{",
    "}": r"\}",
    '"': r'\"',
}


def typst_escape(s) -> str:
    if s is None:
        return ""
    return "".join(_TYPST_REPLACE.get(c, c) for c in str(s))


def typst_url(url: str) -> str:
    if not url:
        return ""
    return url.replace('"', '\\"')


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*\n?(.*?)```", re.DOTALL)
_INLINE_RE = re.compile(r"`([^`\n]+)`")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^\)]+)\)")

_MATH_RE = re.compile(
    r"\$\$(?P<dd>.+?)\$\$"
    r"|\\\[(?P<br>.+?)\\\]"
    r"|(?<![\\$])\$(?P<sd>[^$\n][^$]*?)\$(?!\d)"
    r"|\\\((?P<pr>.+?)\\\)",
    re.DOTALL,
)

def _stash_images(text: str, workdir: Path) -> tuple[str, list[str]]:
    bits: list[str] = []

    def stash(m: re.Match) -> str:
        alt = m.group(1)
        url = m.group(2)

        # Create hash-based filename for downloaded image
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

        assets_dir = workdir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # We don't know the exact extension yet if we haven't downloaded it,
        # but we can look for existing files with this hash
        existing = list(assets_dir.glob(f"{url_hash}.*"))
        if existing:
            filename = existing[0].name
        else:
            try:
                # Add a basic User-Agent to avoid easy 403s
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    ctype = response.info().get_content_type()
                    ext = ".jpg"
                    if ctype == "image/png":
                        ext = ".png"
                    elif ctype == "image/gif":
                        ext = ".gif"
                    elif ctype == "image/webp":
                        ext = ".webp"
                    elif ctype == "image/svg+xml":
                        ext = ".svg"

                    filename = f"{url_hash}{ext}"
                    img_path = assets_dir / filename
                    with open(img_path, "wb") as f:
                        shutil.copyfileobj(response, f)
            except Exception as e:
                sys.stderr.write(f"  [warn] failed to fetch image {url}: {e}\n")
                # Fallback to just the text if image fails to download
                bits.append(f"[{alt}]({url})")
                return f"\x00IMG{len(bits) - 1}\x00"

        # Note: Typst requires image paths relative to the project root/workdir or absolute
        bits.append(f'#figure(image("assets/{filename}", width: 80%), caption: [{alt}])')
        return f"\x00IMG{len(bits) - 1}\x00"

    return _IMAGE_RE.sub(stash, text), bits

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
            out.append(f"{fence}{part}{fence}")
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

        # Use dynamic backtick fencing to safely encapsulate the raw LaTeX
        # Typst supports 1 backtick (inline code) or 3+ backticks (code block)
        fence_len = 3 if is_display else 1
        while "`" * fence_len in content:
            fence_len += 1
        # Typst requires block code strings to be at least 3 backticks, inline can be 1 or more but let's stick to 1 or 3+
        if fence_len == 2:
            fence_len = 3

        fence = "`" * fence_len
        
        if is_display:
            # We add \n so backticks don't merge if content starts/ends with `
            bits.append(f"#mitex({fence}\n{content}\n{fence})")
        else:
            space_start = " " if content.startswith("`") else ""
            space_end = " " if content.endswith("`") else ""
            bits.append(f"#mi({fence}{space_start}{content}{space_end}{fence})")

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

def _strip_leading_date_line(text: str) -> str:
    lines = text.lstrip("\n").split("\n")
    while lines and _LEADING_DATE_RE.fullmatch(lines[0].strip()):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)


def typst_body(text: str, workdir: Path) -> str:
    if not text:
        return ""
    text = _strip_leading_date_line(text)

    blocks: list[str] = []

    def stash_code(m: re.Match) -> str:
        blocks.append(m.group(1))
        return f"\x00CB{len(blocks) - 1}\x00"

    stashed = _FENCE_RE.sub(stash_code, text)
    stashed, img_bits = _stash_images(stashed, workdir)
    stashed, math_bits = _stash_math(stashed)

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
        m = re.fullmatch(r"\x00CB(\d+)\x00", p)
        if m:
            out.append(_render_code_block(blocks[int(m.group(1))]))
            continue

        def expand_code(mm: re.Match) -> str:
            return _render_code_block(blocks[int(mm.group(1))])

        p = re.sub(r"\x00CB(\d+)\x00", expand_code, p)

        rendered = _process_inline(p)

        def expand_math(mm: re.Match) -> str:
            return math_bits[int(mm.group(1))]
        rendered = re.sub(r"\x00MB(\d+)\x00", expand_math, rendered)

        def expand_img(mm: re.Match) -> str:
            return img_bits[int(mm.group(1))]
        rendered = re.sub(r"\x00IMG(\d+)\x00", expand_img, rendered)

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


def build_pdf(
    date: str,
    articles: list[dict],
    out_dir: Path,
    decorations: dict | None = None,
) -> Path:
    tpl_dir = Path(__file__).parent
    workdir = out_dir / ".build"
    workdir.mkdir(parents=True, exist_ok=True)

    env = _env(tpl_dir, workdir)
    tpl = env.get_template("template.typ.j2")
    typst_source = tpl.render(date=date, articles=articles, decorations=decorations or {})

    typst_path = workdir / f"{date}.typ"
    typst_path.write_text(typst_source, encoding="utf-8")

    pdf_dst = out_dir / f"{date}.pdf"

    import typst
    try:
        typst.compile(str(typst_path), output=str(pdf_dst))
    except typst.TypstError as e:
        sys.stderr.write(str(e))
        raise RuntimeError(f"typst failed: {e}")

    return pdf_dst
