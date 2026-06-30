import os
import re
import pytest
from pathlib import Path


HOSTILE_STRINGS = [ s + " word" * 800 for s in [
    "Unclosed HTML: <div><p>broken",
    "Nested blockquotes: > level 1\n>> level 2\n>>> level 3",
    "Malformed LaTeX: $ x = \\frac{1}{2 $",
    "Unescaped Typst control characters: @hello #world $money",
    "Weird Unicode: ¯\\_(ツ)_/¯ 💥 👨‍👩‍👧‍👦 🚀 \x00 \x01 \x08",
    "Empty brackets: [] () {}",
    "Markdown link with empty text: [](http://example.com)",
    "Markdown image with missing alt: ![](http://example.com/img.png)",
    "Dangling backtick: `code",
    "Math with no closing: $$ \\int x dx",
    "Mixed delimiters: [ ( ] )",
]]

os.environ["GEMINI_API_KEY"] = "fake-key-for-tests"
from papernews.core.main import triage_process_a_filter, triage_process_b_rank, stage3_hybrid_construction, stage5_bespoke_render
from papernews.models import Telemetry
from datetime import date
from papernews.plugins import rss_plugin

class StubEntry:
    def __init__(self, link, title, published, updated):
        self.link = link
        self.title_attr = title
        self.published_attr = published
        self.updated_attr = updated

    def get(self, key, default=""):
        if key == "title":
            return self.title_attr
        if key == "published":
            return self.published_attr
        if key == "updated":
            return self.updated_attr
        return default

class StubFeed:
    def __init__(self, entries):
        self.entries = entries

def test_e2e_gauntlet(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PREFECT_API_URL", "")
    monkeypatch.setenv("PREFECT_SERVER_ALLOW_EPHEMERAL_MODE", "false")
    monkeypatch.setenv("PREFECT_TEST_MODE", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")

    entries = [
        StubEntry(
            link=f"http://example.com/{i}",
            title=f"Hostile {i}",
            published="2023-01-01",
            updated="2023-01-01"
        )
        for i in range(len(HOSTILE_STRINGS))
    ]

    mocker.patch("feedparser.parse", return_value=StubFeed(entries))
    mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
    mocker.patch("trafilatura.extract", side_effect=HOSTILE_STRINGS)

    source_config = {
        "source": [
            {"kind": "rss", "url": "http://fake.com", "category": "News"}
        ]
    }

    raw_docs = rss_plugin.fetch_sources(source_config)
    assert len(raw_docs) == len(HOSTILE_STRINGS)

    prefs = {}

    # Stages 1, 2, 3, 4, 5 pipeline
    filtered = triage_process_a_filter.fn(raw_docs, prefs)
    ranked = triage_process_b_rank.fn(filtered, prefs)

    mocker.patch("papernews.core.main.llm_select_article", side_effect=lambda doc, prefs: (True, Telemetry()))
    mocker.patch("papernews.core.main.llm_summarize_article", side_effect=lambda doc: ("Mock summary", Telemetry()))
    mocker.patch("papernews.core.main.llm_format_body", side_effect=lambda doc: (doc.raw_text, Telemetry()))

    articles, telemetry = stage3_hybrid_construction.fn(ranked, prefs)

    # Stage 4 Pipeline
    from papernews.core.main import stage4b_fetch_decorations
    mocker.patch("papernews.plugins.wiki_plugin.fetch_decorations", return_value=[], create=True)
    # The plugin system hook itself is throwing an error since hookspec isn't loaded.
    # Let's mock stage4b_fetch_decorations entirely.
    decorations = {
        "generation_time": "Now",
        "total_tokens": "0",
        "total_cost": "0",
        "quote": None,
        "world_news": [],
        "dyk": []
    }

    monkeypatch.chdir(tmp_path)

    pdf_path = stage5_bespoke_render.fn(
        articles=articles,
        total_telemetry=telemetry,
        decorations=decorations
    )

    typst_path = tmp_path / "output" / ".build" / f"{date.today().strftime('%Y-%m-%d')}.typ"
    assert typst_path.exists()

    content = typst_path.read_text(encoding="utf-8")

    assert "\\@hello \\#world \\$money" in content, "Control characters were not properly escaped!"

    unescaped_dollars = re.findall(r'(?<!\\)\$', content)
    assert len(unescaped_dollars) == 0, "Found unescaped $ characters!"

    # Only look for orphaned empty blocks that are just standing alone,
    # not attached to a function like #link(...)[] or #quote(block: true)[]
    # 2. Balanced Delimiters
    def check_balanced_delimiters(text):
        stack = []
        i = 0
        while i < len(text):
            if text[i] == '\\':
                i += 2
                continue

            # String literals
            if text[i] == '"':
                i += 1
                while i < len(text) and text[i] != '"':
                    if text[i] == '\\':
                        i += 2
                    else:
                        i += 1
                if i < len(text):
                    i += 1
                continue

            # Typst raw text blocks (backticks)
            if text[i] == '`':
                fence_len = 0
                while i + fence_len < len(text) and text[i + fence_len] == '`':
                    fence_len += 1
                i += fence_len
                closing = '`' * fence_len
                while i < len(text) and text[i:i+fence_len] != closing:
                    i += 1
                if i < len(text):
                    i += fence_len
                continue

            # Block comments
            if text[i:i+2] == '/*':
                i += 2
                while i < len(text) and text[i:i+2] != '*/':
                    i += 1
                if i < len(text):
                    i += 2
                continue

            # Line comments
            if text[i:i+2] == '//':
                i += 2
                while i < len(text) and text[i] != '\n':
                    i += 1
                continue

            c = text[i]
            if c in ('[', '(', '{'):
                stack.append(c)
            elif c == ']':
                if not stack or stack.pop() != '[':
                    return False
            elif c == ')':
                if not stack or stack.pop() != '(':
                    return False
            elif c == '}':
                if not stack or stack.pop() != '{':
                    return False

            i += 1

        return len(stack) == 0

    assert check_balanced_delimiters(content), "Found unbalanced [], (), or {} in Typst output!"

    empty_blocks = re.findall(r'(?<!\))(?<!\[)\[\s*\]', content)
    assert len(empty_blocks) == 0, "Found empty trailing blocks []!"

    assert pdf_path.exists(), "PDF was not successfully generated!"
    assert pdf_path.stat().st_size > 0, "PDF file is empty!"
