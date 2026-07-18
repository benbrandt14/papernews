"""E-ink polish tests: grayscale image pipeline, preamble sync, gray preview.

None of these compile Typst, so they run everywhere (the compile-dependent
checks live in test_ir.py / test_render.py and run in CI).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")

import pytest
from PIL import Image

from papernews.models import Block, ImageRef
from papernews.typst_emit import _MAX_EDGE_PX, PREAMBLE, _eink_process, emit_blocks


def _png_bytes(mode: str = "RGB", size=(120, 80), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(120, 80)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (40, 90, 200)).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, data: bytes, ctype: str = "application/octet-stream"):
        self._data = data
        self._ctype = ctype

    def read(self) -> bytes:
        return self._data

    def info(self):
        return SimpleNamespace(get_content_type=lambda: self._ctype)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _emit_one_image(mocker, tmp_path: Path, data: bytes, url: str) -> Path:
    """Run the real emit path with the download mocked; return the asset."""
    mocker.patch(
        "papernews.typst_emit.urllib.request.urlopen",
        return_value=_FakeResponse(data),
    )
    block = Block(kind="image", images=[ImageRef(alt="Alt", url=url)])
    out = emit_blocks([block], tmp_path)
    assert "assets/" in out
    assets = list((tmp_path / "assets").iterdir())
    assert len(assets) == 1
    return assets[0]


# --- Grayscale conversion ----------------------------------------------------


def test_color_png_becomes_grayscale_png(mocker, tmp_path):
    asset = _emit_one_image(mocker, tmp_path, _png_bytes(), "https://e.com/a.png")
    assert asset.suffix == ".png"
    with Image.open(asset) as img:
        assert img.mode == "L"


def test_color_jpeg_becomes_grayscale_jpeg(mocker, tmp_path):
    asset = _emit_one_image(mocker, tmp_path, _jpeg_bytes(), "https://e.com/b.jpg")
    assert asset.suffix == ".jpg"
    with Image.open(asset) as img:
        assert img.mode == "L"


def test_alpha_flattens_to_white_not_black():
    # A fully transparent image must come out white — PIL's default blend
    # target is black, which would print as a solid ink slab on e-ink.
    data, ext = _eink_process(_png_bytes(mode="RGBA", color=(255, 0, 0, 0)), ".png")
    with Image.open(io.BytesIO(data)) as img:
        assert img.mode == "L"
        assert img.getpixel((5, 5)) == 255


def test_oversized_image_downscales_to_page_width():
    src = io.BytesIO()
    Image.new("RGB", (_MAX_EDGE_PX * 2, 600), (10, 10, 10)).save(src, format="PNG")
    data, ext = _eink_process(src.getvalue(), ".png")
    with Image.open(io.BytesIO(data)) as img:
        assert max(img.size) == _MAX_EDGE_PX
        # Aspect preserved: 2:1 downscale halves the height too.
        assert img.size[1] == 300


def test_gif_reencodes_as_png():
    src = io.BytesIO()
    Image.new("P", (20, 20)).save(src, format="GIF")
    data, ext = _eink_process(src.getvalue(), ".gif")
    assert ext == ".png"
    with Image.open(io.BytesIO(data)) as img:
        assert img.mode == "L"


def test_webp_reencodes_as_jpeg():
    from PIL import features

    if not features.check("webp"):
        pytest.skip("Pillow built without webp")
    src = io.BytesIO()
    Image.new("RGB", (20, 20), (0, 128, 0)).save(src, format="WEBP")
    data, ext = _eink_process(src.getvalue(), ".webp")
    assert ext == ".jpg"
    with Image.open(io.BytesIO(data)) as img:
        assert img.mode == "L"


def test_svg_passes_through_untouched():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    assert _eink_process(svg, ".svg") == (svg, ".svg")


def test_corrupt_raster_degrades_to_original_bytes():
    # Sniffs as PNG but is garbage: conversion must fail soft, never raise,
    # and keep the original so the article still gets its image.
    corrupt = b"\x89PNG\r\n\x1a\n" + b"not really a png"
    assert _eink_process(corrupt, ".png") == (corrupt, ".png")


# --- Preamble / template sync ------------------------------------------------


def test_smart_sentence_definition_synced_with_template():
    """The smart-sentence buckets are defined twice (template + PREAMBLE);
    a drift between them means tests compile different styling than the
    paper actually uses."""
    template = (
        Path(__file__).parent.parent / "papernews" / "template.typ.j2"
    ).read_text(encoding="utf-8")
    for line in PREAMBLE.strip().splitlines():
        assert line.strip() in template, f"PREAMBLE line missing from template: {line}"


def test_high_weight_clears_semibold_threshold():
    """Regression: HIGH_WEIGHT=0.65 with a 0.75 threshold made promotion a
    visual no-op. The plugin's promoted weight must trigger the semibold
    branch in the emitter preamble."""
    from papernews.plugins.salience_plugin import HIGH_WEIGHT, LOW_WEIGHT

    assert "weight >= 0.6" in PREAMBLE
    assert HIGH_WEIGHT >= 0.6
    assert LOW_WEIGHT <= 0.25


# --- Grayscale preview -------------------------------------------------------


def test_preview_rasterizes_grayscale(mocker, tmp_path):
    from papernews.preview import render_cover_png

    mocker.patch("papernews.preview.shutil.which", return_value="/usr/bin/pdftoppm")

    def fake_run(cmd, **kw):
        Path(f"{(tmp_path / 'cover').as_posix()}-1.png").write_bytes(b"stub")
        return SimpleNamespace(returncode=0, stderr="")

    run = mocker.patch("papernews.preview.subprocess.run", side_effect=fake_run)
    render_cover_png(tmp_path / "x.pdf", tmp_path / "cover.png")

    cmd = run.call_args[0][0]
    assert "-gray" in cmd
