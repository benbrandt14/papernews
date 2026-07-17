"""Article-level AI-likeness scoring for the Stage 2 triage funnel.

Adapted from lyc8503/AITextDetector, which separates human from LLM text
with a linear SVM over character n-gram TF-IDF features. That model is
trained exclusively on Chinese creative writing and does not transfer to
English news prose, so this module keeps the shape of the approach — a
deterministic lexical feature vector fed into a fixed linear decision
function — and swaps the learned n-gram weights for hand-set stylometric
features that target English LLM filler:

  * burstiness — variance of sentence length; LLM prose runs unusually
    uniform, human prose alternates short and long sentences.
  * lexical diversity — moving-window type/token ratio; formulaic text
    recycles the same vocabulary.
  * stock-phrase rate — hits per 1000 words against a fixed lexicon of
    LLM-tell phrases ("delve", "tapestry", "in today's fast-paced
    world", …).

The output is a noise dial, not a verdict: scores exist to derank
low-signal, formulaic articles below the category-budget cut line, and
false negatives are acceptable by design. Everything here is pure Python
and deterministic — no network, no model files, no third-party
dependencies — per the Stage 2 zero-API-cost invariant.
"""

from __future__ import annotations

import re
from statistics import mean, pstdev

from papernews.models import AITextMetrics

# Below these sizes the statistics are dominated by sampling noise, so the
# result is flagged unreliable and the triage funnel must never act on it.
MIN_RELIABLE_WORDS = 100
MIN_RELIABLE_SENTENCES = 4

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9']+")

# The fixed "vocabulary" of the linear model: phrases heavily over-represented
# in LLM filler relative to edited human prose. Ordinary academic connectives
# are matched only in their comma-terminated discourse-marker form so that
# arXiv abstracts aren't blanket-flagged.
_STOCK_PHRASE_PATTERNS = [
    r"\bdelv(?:e|es|ed|ing)\b",
    r"\btapestr(?:y|ies)\b",
    r"\btreasure trove\b",
    r"\bgame.?changer\b",
    r"\ba testament to\b",
    r"\bin today's (?:fast.paced|digital|modern|ever.changing) \w+",
    r"\bin the ever.evolving\b",
    r"\bever.changing landscape\b",
    r"\bnavigat(?:e|ing) the (?:complexities|landscape|world)\b",
    r"\bunlock(?:s|ing)? the (?:power|potential|secrets)\b",
    r"\b(?:harness|leverage|leveraging|harnessing) the power\b",
    r"\bseamless(?:ly)? integrat\w*",
    r"\bembark(?:s|ed|ing)? on a journey\b",
    r"\bunderscor(?:e|es|ed|ing) the (?:importance|significance)\b",
    r"\bplays? a (?:crucial|pivotal|vital|key) role\b",
    r"\b(?:valuable|actionable|key) insights\b",
    r"\bit(?:'s| is) (?:important|worth) not(?:ing|e)\b",
    r"\bin conclusion,",
    r"\bin summary,",
    r"\blook no further\b",
    r"\bwhether you(?:'re| are) a\b",
    r"\bdive (?:into the world of|deeper into)\b",
    r"\blet's dive\b",
    r"\belevate your\b",
    r"\brevolutioniz\w+",
    r"\bas an ai language model\b",
    r"\b(?:moreover|furthermore|additionally),",
]
_STOCK_PHRASES = re.compile("|".join(_STOCK_PHRASE_PATTERNS), re.IGNORECASE)

# Hand-set weights of the linear decision function (they sum to 1, so the
# combined score is already in [0, 1] — no squashing needed).
_W_STOCK_PHRASES = 0.50
_W_BURSTINESS = 0.30
_W_DIVERSITY = 0.20

_MATTR_WINDOW = 100


def _ramp(value: float, at_zero: float, at_one: float) -> float:
    """Linear map of `value` onto [0, 1], clamped.

    `at_zero`/`at_one` are the feature values that map to 0 and 1
    respectively; passing at_zero > at_one inverts the ramp.
    """
    t = (value - at_zero) / (at_one - at_zero)
    return min(1.0, max(0.0, t))


def _sentences(text: str) -> list[list[str]]:
    """Split into sentences, each a list of lowercased word tokens."""
    out = []
    for raw in _SENTENCE_SPLIT.split(text):
        words = _WORD.findall(raw.lower())
        if words:
            out.append(words)
    return out


def _burstiness(sentence_lengths: list[int]) -> float:
    """Coefficient of variation of sentence length (0 = perfectly uniform)."""
    if len(sentence_lengths) < 2:
        return 0.0
    avg = mean(sentence_lengths)
    if avg == 0:
        return 0.0
    return pstdev(sentence_lengths) / avg


def _moving_ttr(words: list[str], window: int = _MATTR_WINDOW) -> float:
    """Moving-average type/token ratio — length-robust lexical diversity."""
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    counts: dict[str, int] = {}
    for w in words[:window]:
        counts[w] = counts.get(w, 0) + 1
    ratio_sum = len(counts) / window
    n_windows = 1
    for i in range(window, len(words)):
        out_w = words[i - window]
        if counts[out_w] == 1:
            del counts[out_w]
        else:
            counts[out_w] -= 1
        in_w = words[i]
        counts[in_w] = counts.get(in_w, 0) + 1
        ratio_sum += len(counts) / window
        n_windows += 1
    return ratio_sum / n_windows


def score_text(text: str) -> AITextMetrics:
    """Score one article body. Pure and deterministic.

    Returns populated `AITextMetrics`; `reliable=False` means the sample
    was too small for the statistics to mean anything, and callers must
    not penalize the document.
    """
    sentences = _sentences(text)
    words = [w for s in sentences for w in s]
    word_count = len(words)

    burstiness = _burstiness([len(s) for s in sentences])
    diversity = _moving_ttr(words)
    phrase_hits = len(_STOCK_PHRASES.findall(text))
    phrase_rate = phrase_hits / word_count * 1000 if word_count else 0.0

    # Feature → AI-ness component ramps. Pivots are set against typical
    # English news prose (sentence-length CV ≈ 0.5, MATTR-100 ≈ 0.75).
    likelihood = (
        _W_STOCK_PHRASES * _ramp(phrase_rate, 0.5, 6.0)
        + _W_BURSTINESS * _ramp(burstiness, 0.75, 0.25)
        + _W_DIVERSITY * _ramp(diversity, 0.82, 0.58)
    )

    return AITextMetrics(
        ai_likelihood=round(likelihood, 4),
        burstiness=round(burstiness, 4),
        lexical_diversity=round(diversity, 4),
        stock_phrases_per_1k=round(phrase_rate, 4),
        word_count=word_count,
        reliable=(
            word_count >= MIN_RELIABLE_WORDS
            and len(sentences) >= MIN_RELIABLE_SENTENCES
        ),
    )
