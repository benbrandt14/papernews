# papernews/plugins/entity_plugin.py
"""Entity interlinking: connect the day's stories through what they share.

A Stage 3.5 enrichment pass that finds named entities recurring across the
edition and turns the reader loose on them. When an entity (a person, org,
place, product) is named in two or more of the day's articles, its first
mention in each secondary article becomes an internal Typst link to the
article that covers it most — so tapping "European Space Agency" in a brief
jumps to the deep-dive.

Extraction is a deterministic, dependency-free heuristic: multi-word
capitalized phrases plus acronym / CamelCase tokens (NASA, OpenAI). Plain
single-capitalized words are deliberately excluded — that is where
sentence-initial noise lives. The two-article threshold is itself a strong
precision filter. Every mention is also recorded in the entity knowledge
graph (store migration 3), which accretes across editions for future
trend/thread features. No LLM required.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from datetime import date

import pluggy

from papernews.config import AppConfig
from papernews.models import ArticleChunk, EntityRef, Span
from papernews.store import SimpleStore

hookimpl = pluggy.HookimplMarker("papernews")

# Link an entity only once it shows up in at least this many of the day's
# articles — the whole point is cross-article connective tissue.
MIN_ARTICLES = 2
# Cap the words in a phrase so a run of Title Case can't swallow a headline.
MAX_ENTITY_WORDS = 5

# Two-to-five capitalized words, OR an all-caps token, OR a CamelCase token.
_ENTITY_RE = re.compile(
    r"[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){1,4}"  # multi-word Title Case
    r"|[A-Z]{2,}"  # ALLCAPS acronym
    r"|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*"  # CamelCase (OpenAI, PyTorch)
)

# Leading words trimmed off a phrase — a capitalized determiner at a sentence
# start ("The European Space Agency") is not part of the entity.
_LEADING = frozenset(
    "the a an this that these those its their his her our and but for in on of to".split()
)


def _is_acronym_or_camel(word: str) -> bool:
    return (word.isupper() and len(word) >= 2) or (
        word[:1].isupper() and any(c.isupper() for c in word[1:])
    )


def _iter_entities(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield (start, end, surface) for each entity mention in `text`."""
    for m in _ENTITY_RE.finditer(text):
        words = m.group().split()
        if len(words) > MAX_ENTITY_WORDS:
            continue
        cursor = m.start()
        while words and words[0].lower() in _LEADING:
            cursor = text.index(words[0], cursor) + len(words[0])
            words.pop(0)
        if not words:
            continue
        # A lone survivor must be an acronym/CamelCase token to count; a bare
        # single capitalized word is almost always sentence-initial noise.
        if len(words) == 1 and not _is_acronym_or_camel(words[0]):
            continue
        start = text.index(words[0], cursor if cursor > m.start() else m.start())
        yield start, m.end(), text[start : m.end()]


def _article_entities(
    article: ArticleChunk,
) -> dict[str, tuple[str, list[tuple[int, int, int]]]]:
    """Map entity key → (surface, [(block_idx, start, end), …]) for one article."""
    found: dict[str, tuple[str, list[tuple[int, int, int]]]] = {}
    for bi, block in enumerate(article.blocks):
        if block.kind not in ("para", "heading", "quote"):
            continue
        for start, end, surface in _iter_entities(block.text):
            key = " ".join(surface.split()).lower()
            entry = found.setdefault(key, (surface, []))
            entry[1].append((bi, start, end))
    return found


@hookimpl
def enrich_articles(
    articles: list[ArticleChunk],
    source_config: AppConfig,
    store: SimpleStore,
) -> None:
    today = date.today().isoformat()

    # 1. Extract per-article, and record every mention in the knowledge graph.
    per_article = [_article_entities(art) for art in articles]
    # entity key → {article_index: (surface, mentions)}
    corpus: dict[str, dict[int, tuple[str, list[tuple[int, int, int]]]]] = defaultdict(
        dict
    )
    for ai, entities in enumerate(per_article):
        for key, (surface, mentions) in entities.items():
            corpus[key][ai] = (surface, mentions)
            store.record_entity_mention(key, surface, articles[ai].url, today)

    # 2. For each entity spanning >= MIN_ARTICLES, pick the home article (most
    # mentions, earliest on ties) and link the first mention in every other.
    for key, by_article in corpus.items():
        if len(by_article) < MIN_ARTICLES:
            continue
        home = max(by_article, key=lambda ai: (len(by_article[ai][1]), -ai))
        home_label = f"art{home + 1}"
        for ai, (surface, mentions) in by_article.items():
            if ai == home:
                continue
            bi, start, end = min(mentions)  # earliest block, then offset
            articles[ai].blocks[bi].spans.append(
                Span(start=start, end=end, kind="entity", label=home_label)
            )
            articles[ai].enrichment.entities.append(
                EntityRef(surface=surface, label=home_label)
            )
