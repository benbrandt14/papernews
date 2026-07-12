# papernews/plugins/salience_plugin.py
"""Salience weighting: mark the sentences worth skimming.

A Stage 3.5 enrichment pass that scores every sentence of an article by how
central it is to the piece, then attaches `salience` spans to the markdown
IR. The emitter turns those into `#smart-sentence(weight: …)` — semibold for
the most central sentences, a gentle fade (still above the E-Ink contrast
floor) for the least. Most text stays normal, so the page reads calm and the
few marked sentences carry a real skim structure.

Scoring is a dependency-free lexical **centroid centrality** (the classic
Radev centroid method): a sentence scores high when its word distribution
aligns with the whole article's. It is deterministic and needs no model
download, so it runs on every edition and is fully testable. `_score_sentences`
is the seam an embedding backend (sentence-transformers, under the optional
`nlp` extra) can replace later without touching the span-attachment logic.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pluggy

from papernews.config import AppConfig
from papernews.models import ArticleChunk, Span
from papernews.store import SimpleStore

hookimpl = pluggy.HookimplMarker("papernews")

# Articles shorter than this can't be meaningfully ranked — skip them whole.
MIN_SENTENCES = 5
# Ignore very short fragments when ranking (list stubs, "See below.", etc.).
MIN_SENTENCE_CHARS = 25
# Fraction of sentences promoted / demoted. Deliberately small: the point is
# a few signposts, not a highlighter dragged over the whole page.
HIGH_FRAC = 0.10
LOW_FRAC = 0.20
# Weights map to the emitter's smart-sentence buckets (>=0.75 bold, <=0.25 fade).
HIGH_WEIGHT = 0.65
LOW_WEIGHT = 0.25

_WORD_RE = re.compile(r"[a-z0-9]+")
# A sentence ends at a run of .!? followed by whitespace or end-of-text.
_SENT_END_RE = re.compile(r"[.!?]+(?=\s|$)")

# Compact English stoplist — enough to stop function words from dominating the
# centroid without pulling in a dependency.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from had has have he her his in into is
    it its of on or that the their them they this to was were which who will with
    you your we our not are do does did than then them these those there here what
    when where why how all any can could would should may might must one two also
    """.split()
)


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Split into (start, end) char ranges, punctuation included, leading
    whitespace trimmed. Offsets index straight into `text`."""
    result: list[tuple[int, int]] = []
    pos = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.end()
        start = pos
        while start < end and text[start].isspace():
            start += 1
        if start < end:
            result.append((start, end))
        pos = end
    # Trailing remainder with no terminal punctuation.
    start = pos
    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text):
        result.append((start, len(text)))
    return result


def _tokenize(sentence: str) -> list[str]:
    return [
        w
        for w in _WORD_RE.findall(sentence.lower())
        if len(w) > 2 and w not in _STOPWORDS
    ]


def _centroid_scores(sentences: list[str]) -> list[float]:
    """Cosine similarity of each sentence's term vector to the document
    centroid. Central (representative) sentences score highest."""
    tokens = [_tokenize(s) for s in sentences]
    centroid: Counter[str] = Counter()
    for t in tokens:
        centroid.update(t)
    c_norm = math.sqrt(sum(v * v for v in centroid.values()))
    if c_norm == 0:
        return [0.0] * len(sentences)

    scores: list[float] = []
    for t in tokens:
        if not t:
            scores.append(0.0)
            continue
        vec = Counter(t)
        dot = sum(count * centroid[word] for word, count in vec.items())
        s_norm = math.sqrt(sum(v * v for v in vec.values()))
        scores.append(dot / (s_norm * c_norm) if s_norm else 0.0)
    return scores


def _score_sentences(sentences: list[str]) -> list[float]:
    """Salience score per sentence, higher = more central.

    The seam for a better scorer: an embedding backend can replace this while
    keeping the sentence splitting and span attachment below unchanged.
    """
    return _centroid_scores(sentences)


def _weight_article(article: ArticleChunk) -> None:
    """Attach salience spans to an article's paragraph blocks in place."""
    # Collect every rankable sentence across paragraph blocks, remembering
    # which block and char range each came from.
    locations: list[tuple[int, int, int]] = []  # (block_index, start, end)
    texts: list[str] = []
    for bi, block in enumerate(article.blocks):
        if block.kind != "para":
            continue
        for start, end in _split_sentences(block.text):
            if end - start < MIN_SENTENCE_CHARS:
                continue
            locations.append((bi, start, end))
            texts.append(block.text[start:end])

    if len(texts) < MIN_SENTENCES:
        return

    scores = _score_sentences(texts)
    # Rank ascending; the tail is most central, the head least.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    n_high = max(1, int(len(scores) * HIGH_FRAC))
    n_low = max(1, int(len(scores) * LOW_FRAC))
    high = set(order[-n_high:])
    low = set(order[:n_low]) - high  # a promotion always beats a demotion

    for idx, (bi, start, end) in enumerate(locations):
        if idx in high:
            weight = HIGH_WEIGHT
        elif idx in low:
            weight = LOW_WEIGHT
        else:
            continue
        article.blocks[bi].spans.append(
            Span(start=start, end=end, kind="salience", weight=weight)
        )


@hookimpl
def enrich_articles(
    articles: list[ArticleChunk],
    source_config: AppConfig,
    store: SimpleStore,
) -> None:
    for article in articles:
        _weight_article(article)
