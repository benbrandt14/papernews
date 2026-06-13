from __future__ import annotations

import re
import shutil
import subprocess
import sys
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
    "#": r"\#",
    "`": r"\`",
    "{": r"\{",
    "}": r"\}",
}


def typst_escape(s) -> str:
    if s is None:
        return ""
    return "".join(_TYPST_REPLACE.get(c, c) for c in str(s))


def typst_url(url: str) -> str:
    # URLs in Typst links don't generally need heavy escaping.
    # Just escape # which could break out of a string context in some cases,
    # though within `link("...", ...)` it's fine. We'll leave it mostly raw.
    if not url:
        return ""
    return url


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*\n?(.*?)```", re.DOTALL)
_INLINE_RE = re.compile(r"`([^`\n]+)`")

_MATH_RE = re.compile(
    r"\$\$(?P<dd>.+?)\$\$"
    r"|\\\[(?P<br>.+?)\\\]"
    r"|(?<![\\$])\$(?P<sd>[^$\n][^$]*?)\$(?!\d)"
    r"|\\\((?P<pr>.+?)\\\)",
    re.DOTALL,
)

def _render_code_block(code: str) -> str:
    # Typst has native code block support with ``` ... ```
    # If the code itself contains ```, we need to add more backticks to the fence.
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
            # Inline code: if it contains backticks, use more backticks
            fence_len = 1
            while "`" * fence_len in part:
                fence_len += 1
            fence = "`" * fence_len
            out.append(f"{fence}{part}{fence}")
    return "".join(out)


def _stash_math(text: str) -> tuple[str, list[str]]:
    """Replace math with placeholders."""
    bits: list[str] = []

    def stash(m: re.Match) -> str:
        if m.group("dd") is not None:
            bits.append(f"$ {m.group('dd').strip()} $")
        elif m.group("br") is not None:
            bits.append(f"$ {m.group('br').strip()} $")
        elif m.group("sd") is not None:
            bits.append(f"${m.group('sd').strip()}$")
        else:  # pr
            bits.append(f"${m.group('pr').strip()}$")
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


def typst_body(text: str) -> str:
    if not text:
        return ""
    text = _strip_leading_date_line(text)

    blocks: list[str] = []

    def stash_code(m: re.Match) -> str:
        blocks.append(m.group(1))
        return f"\x00CB{len(blocks) - 1}\x00"

    stashed = _FENCE_RE.sub(stash_code, text)
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

        # Need to handle potential escaping of the placeholder.
        # \x00MB{N}\x00 becomes \x00MB\{N\}\x00 because of typst_escape. Oh wait, { is not escaped.
        rendered = re.sub(r"\x00MB(\d+)\x00", expand_math, rendered)
        out.append(rendered)

    return "\n\n".join(out)

def _env(tpl_dir: Path) -> jinja2.Environment:
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
    env.filters["typst_body"] = typst_body
    return env


def build_pdf(
    date: str,
    articles: list[dict],
    out_dir: Path,
    decorations: dict | None = None,
) -> Path:
    tpl_dir = Path(__file__).parent
    env = _env(tpl_dir)
    tpl = env.get_template("template.typ.j2")
    typst_source = tpl.render(date=date, articles=articles, decorations=decorations or {})

    workdir = out_dir / ".build"
    workdir.mkdir(parents=True, exist_ok=True)
    typst_path = workdir / f"{date}.typ"
    typst_path.write_text(typst_source, encoding="utf-8")

    pdf_dst = out_dir / f"{date}.pdf"

    result = subprocess.run(
        [
            "typst",
            "compile",
            str(typst_path),
            str(pdf_dst),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-2000:])
        raise RuntimeError(f"typst failed (exit {result.returncode})")

    return pdf_dst
