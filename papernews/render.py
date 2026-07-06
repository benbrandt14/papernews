"""Jinja → Typst rendering.

Article bodies arrive as markdown IR (`Block`/`Span` records, parsed by
`papernews.markdown_ir`) and are emitted to Typst by
`papernews.typst_emit`. This module owns the Typst escaping helpers, the
Jinja environment, and PDF compilation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import jinja2

from .adapter import render_context_to_template_vars
from .markdown_ir import parse_markdown
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

    env = _env(tpl_dir)
    tpl = env.get_template("template.typ.j2")

    template_vars = render_context_to_template_vars(ctx)
    # Emit article bodies from the markdown IR. Emission lives here because
    # it needs the workdir (image fetching). Contexts built without blocks
    # (hand-constructed in tests) are parsed on the fly.
    from papernews.typst_emit import emit_blocks  # lazy: typst_emit imports us

    for chunk, art_dict in zip(ctx.articles, template_vars["articles"]):
        blocks = chunk.blocks or parse_markdown(chunk.body_markdown)
        if blocks:
            art_dict["body_typst"] = emit_blocks(blocks, workdir)

    typst_source = tpl.render(**template_vars)

    typst_path = workdir / f"{date}.typ"
    typst_path.write_text(typst_source, encoding="utf-8")

    pdf_dst = out_dir / f"{date}.pdf"

    import typst

    fonts_dir = Path(__file__).parent / "fonts"

    try:
        typst.compile(
            str(typst_path),
            output=str(pdf_dst),
            font_paths=[str(fonts_dir)] if fonts_dir.is_dir() else [],
        )
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
