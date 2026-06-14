from __future__ import annotations

import functools
import hashlib
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import jinja2

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
    '"': r'\"',
}

def typst_escape(s) -> str:
    if s is None:
        return ""
    res = "".join(_TYPST_REPLACE.get(c, c) for c in str(s))
    # Safely escape # unless it's the specific #table macro we want to allow
    return re.sub(r'#(?!table\b)', r'\#', res)


def typst_url(url: str) -> str:
    if not url:
        return ""
    return url.replace('"', '\\"')


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*\n?(.*?)```", re.DOTALL)
_INLINE_RE = re.compile(r"`([^`\n]+)`")

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^\)]+)\)")
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


def _stash_typography(text: str) -> tuple[str, list[str]]:
    bits: list[str] = []

    def stash_bold(m: re.Match) -> str:
        # /**/ protects the macro from accidentally absorbing trailing punctuation like ( or [
        bits.append(f"#strong[{m.group(1)}]/**/")
        return f"\x00TYP{len(bits) - 1}\x00"

    def stash_italic(m: re.Match) -> str:
        bits.append(f"#emph[{m.group(1)}]/**/")
        return f"\x00TYP{len(bits) - 1}\x00"

    stashed = _STRONG_RE.sub(stash_bold, text)
    stashed = _EMPH_RE.sub(stash_bold, stashed)
    stashed = _STRONG_US_RE.sub(stash_bold, stashed)
    stashed = _EMPH_US_RE.sub(stash_italic, stashed)
    return stashed, bits


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
        alt = m.group(1)
        url = m.group(2)

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        assets_dir = workdir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        existing = list(assets_dir.glob(f"{url_hash}.*"))
        if existing:
            filename = existing[0].name
        else:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = response.read()
                    
                    if data.startswith(b'\xff\xd8'): ext = ".jpg"
                    elif data.startswith(b'\x89PNG\r\n\x1a\n'): ext = ".png"
                    elif data.startswith(b'GIF8'): ext = ".gif"
                    elif data.startswith(b'RIFF') and data[8:12] == b'WEBP': ext = ".webp"
                    elif b'<svg' in data[:1024].lower(): ext = ".svg"
                    else:
                        ctype = response.info().get_content_type()
                        if ctype == "image/png": ext = ".png"
                        elif ctype == "image/gif": ext = ".gif"
                        elif ctype == "image/webp": ext = ".webp"
                        elif ctype == "image/svg+xml": ext = ".svg"
                        else: ext = ".jpg"

                    filename = f"{url_hash}{ext}"
                    img_path = assets_dir / filename
                    with open(img_path, "wb") as f:
                        f.write(data)
                        
            except Exception as e:
                sys.stderr.write(f"  [warn] failed to fetch image {url}: {e}\n")
                safe_alt = typst_escape(alt)
                safe_url = typst_url(url)
                bits.append(f'#link("{safe_url}")[{safe_alt}]/**/')
                return f"\x00IMG{len(bits) - 1}\x00"

        bits.append(f'#figure(image("assets/{filename}", width: 100%), placement: auto)/**/')
        return f"\x00IMG{len(bits) - 1}\x00"

    return _IMAGE_RE.sub(stash, text), bits


def _render_code_block(code: str) -> str:
    fence_len = 3
    while "`" * fence_len in code:
        fence_len += 1
    fence = "`" * fence_len
    return f"\n\n{fence}\n{code}\n{fence}\n\n"


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

    inline_code_bits: list[str] = []
    def stash_inline(m: re.Match) -> str:
        part = m.group(1)
        fence_len = 1
        while "`" * fence_len in part:
            fence_len += 1
        fence = "`" * fence_len
        safe_part = part
        if safe_part.startswith("`"): safe_part = " " + safe_part
        if safe_part.endswith("`") or safe_part.endswith("\\"): safe_part = safe_part + " "
        inline_code_bits.append(f"{fence}{safe_part}{fence}")
        return f"\x00IC{len(inline_code_bits) - 1}\x00"
    
    stashed = _INLINE_RE.sub(stash_inline, stashed)

    stashed, img_bits = _stash_images(stashed, workdir)
    stashed, math_bits = _stash_math(stashed)
    stashed, typ_bits = _stash_typography(stashed)
    stashed, lnk_bits = _stash_links(stashed)

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
        
        rendered = typst_escape(p)

        def expand_cb(mm: re.Match) -> str: return _render_code_block(blocks[int(mm.group(1))])
        def expand_ic(mm: re.Match) -> str: return inline_code_bits[int(mm.group(1))]
        def expand_img(mm: re.Match) -> str: return img_bits[int(mm.group(1))]
        def expand_math(mm: re.Match) -> str: return math_bits[int(mm.group(1))]
        def expand_lnk(mm: re.Match) -> str: return lnk_bits[int(mm.group(1))]
        def expand_typ(mm: re.Match) -> str: return typ_bits[int(mm.group(1))]

        # Iterative expansion to seamlessly handle nested formatting, like [**Bold Link**](url)
        while True:
            old = rendered
            rendered = re.sub(r"\x00CB(\d+)\x00", expand_cb, rendered)
            rendered = re.sub(r"\x00IC(\d+)\x00", expand_ic, rendered)
            rendered = re.sub(r"\x00IMG(\d+)\x00", expand_img, rendered)
            rendered = re.sub(r"\x00MB(\d+)\x00", expand_math, rendered)
            rendered = re.sub(r"\x00LNK(\d+)\x00", expand_lnk, rendered)
            rendered = re.sub(r"\x00TYP(\d+)\x00", expand_typ, rendered)
            if old == rendered:
                break

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
        sys.stderr.write(f"\n[error] Typst compilation failed!\n")
        
        # Enhanced Debug Context: Read the generated .typ file and print a snippet
        try:
            line_match = re.search(r'line (\d+)', str(e), re.IGNORECASE)
            if not line_match:
                # Some versions of Typst Python bindings output standard string formats
                line_match = re.search(r':(\d+):\d+', str(e))
                
            if line_match:
                err_line = int(line_match.group(1))
                lines = typst_source.split('\n')
                sys.stderr.write(f"[error] Context around line {err_line}:\n")
                sys.stderr.write(f"----------------------------------------\n")
                start = max(0, err_line - 4)
                end = min(len(lines), err_line + 3)
                for i in range(start, end):
                    prefix = ">> " if i + 1 == err_line else "   "
                    sys.stderr.write(f"{prefix}{i + 1:04d} | {lines[i]}\n")
                sys.stderr.write(f"----------------------------------------\n")
        except Exception as dump_e:
            sys.stderr.write(f"[error] Could not print file context: {dump_e}\n")

        sys.stderr.write(f"[error] You can inspect the generated source file here: {typst_path}\n")
        sys.stderr.write(f"[error] Typst error details:\n{e}\n\n")
        raise RuntimeError(f"typst failed on {typst_path}: {e}")

    return pdf_dst