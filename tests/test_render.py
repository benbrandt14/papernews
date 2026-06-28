import pytest
from pathlib import Path
import re
from hypothesis import given, strategies as st
from papernews.render import typst_body, _stash_math, typst_escape, typst_url, _TYPST_REPLACE

def test_markdown_headers_to_typst():
    # Adding a leading paragraph so _strip_leading_metadata doesn't strip the first header
    text = "Some intro text\n\n# Header 1\n## Header 2\n### Header 3"
    result = typst_body(text, Path("/tmp"))
    assert "= Header 1" in result
    assert "== Header 2" in result
    assert "=== Header 3" in result

def test_markdown_bold_and_italic():
    text = "**bold text** and *italic text* and __bold__ and _italic_"
    result = typst_body(text, Path("/tmp"))
    assert "#strong[bold text]/**/" in result
    assert "#emph[italic text]/**/" in result
    assert "#strong[bold]/**/" in result
    assert "#emph[italic]/**/" in result

def test_math_stashing_and_rendering():
    text = "Inline math $E=mc^2$ and display math $$\\int x dx$$"
    result = typst_body(text, Path("/tmp"))
    assert "#mi(` E=mc^2 `)/**/" in result
    assert "#mitex(```\n\\int x dx\n```)/**/" in result

def test_lists():
    text = "* Item 1\n* Item 2"
    result = typst_body(text, Path("/tmp"))
    assert "- Item 1" in result
    assert "- Item 2" in result

def test_blockquotes():
    text = "> This is a quote\n> across two lines."
    result = typst_body(text, Path("/tmp"))
    assert "#quote(block: true)[\nThis is a quote\n]" in result or "#quote(block: true)[\nThis is a quote\nacross two lines.\n]" in result

def test_remote_images_fallback(mocker, tmp_path):
    import urllib.request

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError("Network unreachable")

    mocker.patch('urllib.request.urlopen', side_effect=fake_urlopen)

    text = "An image: ![AltText](http://example.com/img.png)"
    result = typst_body(text, tmp_path)

    # It should fallback to a valid typst link
    assert '#link("http://example.com/img.png")[AltText]/**/' in result


import json

def test_regression_fixtures(tmp_path):
    """
    Reads inputs from a test DB (JSON format) to ensure known edge cases
    continue to compile to expected Typst mapped syntax correctly.
    This allows users to easily add previously failing inputs to prevent regressions.
    """
    fixture_file = Path(__file__).parent / "fixtures" / "test_db.json"
    if not fixture_file.exists():
        pytest.skip("No regression fixtures found.")

    with open(fixture_file, "r") as f:
        cases = json.load(f)

    for case in cases:
        description = case.get("description", "Unknown case")
        input_text = case.get("input", "")
        expected_output = case.get("expected_typst", "")

        result = typst_body(input_text, tmp_path)

        # Verify the expected syntax is somewhere in the resulting Typst body
        assert expected_output in result, f"Regression Failed: {description}\nExpected: {expected_output}\nGot: {result}"

@given(st.text())
def test_typst_escape_property(text):
    escaped = typst_escape(text)

    # Verify no unescaped special characters exist in the output.
    # Note: typst_escape unconditionally escapes # to \#, and uses _TYPST_REPLACE for others.
    # We shouldn't find raw special chars unless they are part of an escape sequence.

    # Reconstruct original text
    reconstructed = escaped
    # \# is handled specially in typst_escape
    reconstructed = reconstructed.replace(r"\#", "#")
    # For dictionary keys in _TYPST_REPLACE
    # We must do it in the reverse order of string iteration, but actually simple un-replace
    # for each mapped escape works if we're careful.
    # The safest way to reconstruct is to iterate through the escaped string and unescape.

    i = 0
    actual_reconstructed = []
    while i < len(escaped):
        if escaped[i] == '\\' and i + 1 < len(escaped):
            # Check if it's an escaped character
            next_char = escaped[i+1]
            if next_char == '#':
                actual_reconstructed.append('#')
                i += 2
                continue

            # Check if this combination exists as a value in _TYPST_REPLACE
            found_escape = False
            for k, v in _TYPST_REPLACE.items():
                if v == "\\" + next_char:
                    actual_reconstructed.append(k)
                    i += 2
                    found_escape = True
                    break

            if found_escape:
                continue

        actual_reconstructed.append(escaped[i])
        i += 1

    assert "".join(actual_reconstructed) == text

@given(st.text())
def test_typst_url_property(url):
    escaped = typst_url(url)

    # Reconstruct
    # typst_url does: url.replace("\\", "\\\\").replace('"', '\\"')
    reconstructed = escaped.replace('\\"', '"').replace("\\\\", "\\")

    assert reconstructed == url
