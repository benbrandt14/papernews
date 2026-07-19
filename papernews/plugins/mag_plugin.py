"""Mag-notation enrichment: gloss big numbers with their magnitude.

Inspired by magworld.pw — the magnitude is the most informative part of a
large number, and prose formats ("$3.2 billion", "48,000") bury it. This
plugin finds large numbers in article bodies and attaches a `mag` span,
which the emitter renders as a small superscript gloss: 3.2 billion^9.5.

Deliberately conservative: only values ≥ 10^4 (below that, the number is
already intuitive), only a handful of glosses per article, and matches
inside code/math spans are left alone (the emitter drops nested
annotations there anyway).
"""

from __future__ import annotations

import re

import pluggy

from papernews.config import AppConfig
from papernews.mag import format_mag
from papernews.models import ArticleChunk, Span
from papernews.store import SimpleStore

hookimpl = pluggy.HookimplMarker("papernews")

# Below this value a number needs no gloss; per-article cap keeps the
# margins quiet even in statistics-heavy pieces.
MIN_VALUE = 10_000
MAX_GLOSSES_PER_ARTICLE = 6

_MULTIPLIERS = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}

# "$3.2 billion", "48,000", "120000", "1.5 trillion" — a digit group
# (comma-grouped or plain), optional decimals, optional scale word.
_NUMBER_RE = re.compile(
    r"(?<![\w.,])"
    r"[$€£]?\s?"
    r"(?P<digits>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<frac>\.\d+)?"
    r"(?:\s?(?P<mult>thousand|million|billion|trillion))?",
    re.IGNORECASE,
)


def _value_of(m: re.Match) -> float:
    value = float(m.group("digits").replace(",", "") + (m.group("frac") or ""))
    mult = m.group("mult")
    if mult:
        value *= _MULTIPLIERS[mult.lower()]
    return value


def _gloss_article(article: ArticleChunk) -> None:
    glossed = 0
    for block in article.blocks:
        if block.kind not in ("para", "list_item"):
            continue
        # Regions already claimed by verbatim spans — a gloss inside a code
        # or math fragment would corrupt it.
        verbatim = [
            (s.start, s.end)
            for s in block.spans
            if s.kind in ("code_inline", "math_inline")
        ]
        for m in _NUMBER_RE.finditer(block.text):
            if glossed >= MAX_GLOSSES_PER_ARTICLE:
                return
            if any(s < m.end() and m.start() < e for s, e in verbatim):
                continue
            value = _value_of(m)
            if value < MIN_VALUE:
                continue
            block.spans.append(
                Span(
                    start=m.start(),
                    end=m.end(),
                    kind="mag",
                    label=format_mag(value),
                )
            )
            glossed += 1


@hookimpl
def enrich_articles(
    articles: list[ArticleChunk],
    source_config: AppConfig,
    store: SimpleStore,
) -> None:
    for article in articles:
        _gloss_article(article)
