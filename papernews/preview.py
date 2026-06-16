"""Render page 1 of a PDF as a PNG (cover preview for the landing page)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def render_cover_png(pdf: Path, out_png: Path, dpi: int = 180) -> Path:
    """Rasterize first page of PDF to PNG."""
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm not found (install poppler)")

    # pdftoppm writes <prefix>-<page>.png; we then rename to the requested name.
    prefix = out_png.with_suffix("").as_posix()
    result = subprocess.run(
        [
            "pdftoppm",
            "-f", "1", "-l", "1",
            "-r", str(dpi),
            "-png",
            str(pdf),
            prefix,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr.strip()}")

    # poppler typically writes prefix-01.png or prefix-1.png depending on page count.
    candidates = [
        Path(f"{prefix}-1.png"),
        Path(f"{prefix}-01.png"),
        Path(f"{prefix}-001.png"),
    ]
    for c in candidates:
        if c.exists():
            if c != out_png:
                c.replace(out_png)
            return out_png
    raise RuntimeError(f"pdftoppm produced no recognizable output for {pdf}")
