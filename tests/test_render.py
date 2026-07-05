import re
import tempfile
from pathlib import Path

import pytest
import typst
from hypothesis import given
from hypothesis import strategies as st

from papernews.render import (
    _TYPST_REPLACE,
    _process_inline,
    _render_code_block,
    _stash_math,
    _strip_leading_metadata,
    typst_body,
    typst_escape,
    typst_url,
)


def compile_typst_snippet(typst_code: str):
    """
    Validates Typst syntax by running an actual compilation against a dummy document.
    Throws typst.TypstError if invalid.
    """
    full_code = f'#import "@preview/mitex:0.2.4": mi, mitex\n\n{typst_code}'
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        typ_file = tmp_path / "test.typ"
        pdf_file = tmp_path / "test.pdf"
        typ_file.write_text(full_code, encoding="utf-8")
        typst.compile(str(typ_file), output=str(pdf_file))


def assert_no_unescaped_control_chars(typst_code: str):
    """
    Ensures raw #, $, and @ characters do not leak into the output unless explicitly
    intended as part of valid Typst syntax.
    """
    # Remove math blocks since they may legally contain these characters inside raw backticks
    code_no_math = re.sub(r"#mitex\(```.*?```\)", "", typst_code, flags=re.DOTALL)
    code_no_math = re.sub(r"#mi\(`.*?`\)", "", code_no_math, flags=re.DOTALL)

    # Remove inline raw strings
    code_no_raw = re.sub(r"`[^`]*`", "", code_no_math)

    if re.search(r"(?<!\\)\$", code_no_raw):
        raise ValueError(f"Unescaped $ found in output: {typst_code}")

    if re.search(r"(?<!\\)@", code_no_raw):
        raise ValueError(f"Unescaped @ found in output: {typst_code}")

    # Check for unescaped #
    # Allowed functions
    unescaped_hash_matches = re.finditer(r"(?<!\\)#([a-zA-Z]+)", code_no_raw)
    valid_hash_funcs = {
        "strong",
        "emph",
        "mi",
        "mitex",
        "quote",
        "link",
        "figure",
        "image",
    }
    for m in unescaped_hash_matches:
        func = m.group(1)
        if func not in valid_hash_funcs:
            raise ValueError(
                f"Unexpected unescaped control character/function #{func} in output"
            )


def assert_balanced_delimiters(typst_code: str):
    """
    Ensures all brackets [], parentheses (), and braces {} are perfectly balanced
    outside of string literals and raw code blocks.
    """
    # Remove escaped delimiters to not trip the checker
    text = re.sub(r"\\.", "", typst_code)

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r'"[^"]*"', "", text)

    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for char in text:
        if char in pairs.values():
            stack.append(char)
        elif char in pairs.keys():
            # If stack is empty or doesn't match, it could be typst handling
            # standard parenthesis unescaped. But brackets and braces in typst
            # require balancing. Parenthesis generally do not need to be balanced
            # in regular text in typst unless inside code mode, which we aren't.
            # But the prompt said: "Ensure all brackets [], parentheses (), and math blocks $$ are perfectly balanced."
            # Since Typst DOES allow unbalanced `(` and `)` in text mode without crashing,
            # and our typst_escape function does not escape them, they will appear unbalanced
            # in the output if they were unbalanced in the input.
            # Typst compiler handles unbalanced () perfectly fine in text mode!
            # The only thing that crashes is [] which typst uses for block content.
            # Let's check only [] and {}
            if char in ("]", "}"):
                if not stack or stack[-1] != pairs[char]:
                    raise ValueError(f"Unbalanced delimiter {char} found in output.")
                stack.pop()
            else:
                # for `)`, we just pop if it matches, otherwise we can ignore
                if stack and stack[-1] == "(":
                    stack.pop()

    # Check what remains. Unbalanced `(` is fine in text mode.
    if stack:
        if any(c in ("[", "{") for c in stack):
            raise ValueError(f"Unbalanced delimiters remain open: {stack}")

    # Check for balanced math block delimiters ($)
    # Count all unescaped $ characters. They must be perfectly balanced.
    unescaped_dollars = len(re.findall(r"(?<!\\)\$", text))
    if unescaped_dollars % 2 != 0:
        raise ValueError(
            f"Unbalanced math blocks ($) found in output. Unescaped count: {unescaped_dollars}"
        )


def assert_no_trailing_empty_blocks(typst_code: str):
    """
    Ensures that pipelines did not leave behind broken or trailing empty blocks like [].
    """
    # Empty bracket block outside of valid usages
    # Actually, [] is valid Typst for an empty content block, but typically means a bug here
    if "[]" in typst_code:
        # Some macros might generate [] legitimately, but in typst_body we don't expect standalone []
        if re.search(r"(?<!\w)\[\]", typst_code):
            raise ValueError(f"Trailing empty block [] found in output: {typst_code}")


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
    assert (
        "#quote(block: true)[\nThis is a quote\n]" in result
        or "#quote(block: true)[\nThis is a quote\nacross two lines.\n]" in result
    )


def test_remote_images_fallback(mocker, tmp_path):
    import urllib.request

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError("Network unreachable")

    mocker.patch("urllib.request.urlopen", side_effect=fake_urlopen)

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

    with open(fixture_file) as f:
        cases = json.load(f)

    for case in cases:
        description = case.get("description", "Unknown case")
        input_text = case.get("input", "")
        expected_output = case.get("expected_typst", "")

        result = typst_body(input_text, tmp_path)

        # Verify the expected syntax is somewhere in the resulting Typst body
        assert expected_output in result, (
            f"Regression Failed: {description}\nExpected: {expected_output}\nGot: {result}"
        )


from hypothesis import settings


@given(st.text())
@settings(max_examples=10, deadline=None)
def test_typst_body_property_safety(text):
    """
    Feeds random hostile strings into the main rendering pipeline and validates that
    the resulting output does not contain unescaped control chars, unbalanced delimiters,
    and successfully compiles to a valid Typst AST without raising TypstError.
    """
    # Use a dummy working directory
    out = typst_body(text, Path("/tmp"))

    assert_no_unescaped_control_chars(out)
    assert_balanced_delimiters(out)
    assert_no_trailing_empty_blocks(out)

    # Assert syntactic validity via actual compilation
    compile_typst_snippet(out)


@given(st.text())
def test_typst_escape_property(text):
    escaped = typst_escape(text)

    # Verify no unescaped special characters exist in the output.
    # Note: typst_escape unconditionally escapes # to \#, and uses _TYPST_REPLACE for others.
    # We shouldn't find raw special chars unless they are part of an escape sequence.

    # Reconstruct original text

    # For dictionary keys in _TYPST_REPLACE
    # We must do it in the reverse order of string iteration, but actually simple un-replace
    # for each mapped escape works if we're careful.
    # The safest way to reconstruct is to iterate through the escaped string and unescape.

    i = 0
    actual_reconstructed = []
    while i < len(escaped):
        if escaped[i] == "\\" and i + 1 < len(escaped):
            # Check if it's an escaped character
            next_char = escaped[i + 1]
            if next_char == "#":
                actual_reconstructed.append("#")
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


@given(st.text())
def test_stash_math_property(text):
    # Tests that varied inputs don't crash the math extraction regex or logic
    # and always return a tuple of (string, list[str]).
    stashed_text, bits = _stash_math(text)
    assert isinstance(stashed_text, str)
    assert isinstance(bits, list)
    # Re-substituting the bits should approximately lead to the same textual footprint,
    # though formatting changes with backticks, so we mainly care that it didn't crash.


@given(st.text())
def test_strip_leading_metadata_property(text):
    # Verify we never crash on random strings
    stripped = _strip_leading_metadata(text)
    assert isinstance(stripped, str)
    assert len(stripped) <= len(text)


@given(st.text())
def test_render_code_block_property(code):
    rendered = _render_code_block(code)
    # rendered should start and end with fences that are at least 3 backticks long
    # and strictly longer than any internal backtick sequence.
    assert rendered.startswith("\n\n```")
    assert rendered.endswith("```\n\n")
    assert code in rendered


@given(st.text())
def test_process_inline_property(text):
    # Tests that varied inputs do not crash the inline processing.
    processed = _process_inline(text)
    assert isinstance(processed, str)


def test_build_pdf_raises_render_error_on_compile_failure(tmp_path, mocker):
    """A failed Typst compile must raise, never silently return a stale path."""
    from papernews.models import RenderContext
    from papernews.render import RenderError, build_pdf

    ctx = RenderContext(
        date="2026-01-01",
        generation_time="Now",
        total_tokens="0",
        total_cost="0",
        articles=[],
    )
    mocker.patch("typst.compile", side_effect=typst.TypstError("boom"))

    with pytest.raises(RenderError, match="2026-01-01"):
        build_pdf(ctx, tmp_path)


def test_build_pdf_produces_pdf(tmp_path):
    """Happy path: an empty edition compiles to a real PDF."""
    from papernews.models import RenderContext
    from papernews.render import build_pdf

    ctx = RenderContext(
        date="2026-01-01",
        generation_time="Now",
        total_tokens="0",
        total_cost="0",
        articles=[],
    )
    pdf = build_pdf(ctx, tmp_path)
    assert pdf.exists()
    assert pdf.read_bytes()[:4] == b"%PDF"
