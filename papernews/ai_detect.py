"""Article-level AI-likeness classification for the Stage 2 triage funnel.

This implements the lyc8503/AITextDetector architecture — a linear SVM
over character n-gram TF-IDF features — for English text. The upstream
trained weights couldn't be reused (they are trained exclusively on
Chinese creative writing), so the model here is trained on a labeled
English human/LLM corpus by `scripts/train_ai_classifier.py`, which
exports the fitted vectorizer + SVM + probability calibration into a
small JSON artifact (`papernews/ai_model.json.gz` by default,
overridable via PAPERNEWS_AI_MODEL).

Inference is re-implemented in pure Python (an exact replica of
scikit-learn's TfidfVectorizer(analyzer="char") → LinearSVC decision
function, pinned by parity tests) so the pipeline keeps its Stage 2
invariants: deterministic, offline, zero API cost, and no ML runtime
dependencies. scikit-learn is needed only at training time.

When no model artifact is present, articles still get descriptive
stylometrics (sentence-length burstiness, moving-window type/token
ratio) for the typeset footer, but `ai_likelihood` stays None and the
triage screen never deranks — the score comes from the trained
classifier or not at all.

The output is a noise dial, not a verdict: scores exist to derank
low-signal articles below the category-budget cut line, and false
negatives are acceptable by design.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev

from papernews.config import get_settings
from papernews.models import AITextMetrics

DEFAULT_MODEL_PATH = Path(__file__).parent / "ai_model.json.gz"

# Below these sizes the statistics (and a char-n-gram classification) are
# dominated by sampling noise, so the result is flagged unreliable and the
# triage funnel must never act on it.
MIN_RELIABLE_WORDS = 100
MIN_RELIABLE_SENTENCES = 4

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9']+")
# scikit-learn's VectorizerMixin collapses all whitespace runs to a single
# space before char n-gram extraction; inference must match exactly.
_WHITESPACE = re.compile(r"\s\s+")

_MATTR_WINDOW = 100


class LinearTextClassifier:
    """Pure-Python inference for the exported char-n-gram TF-IDF + SVM.

    Replicates scikit-learn exactly: lowercase, collapse whitespace runs,
    extract all contiguous character n-grams in the trained range, weight
    raw counts by stored IDF, L2-normalize, dot with the SVM coefficients,
    then map the decision value through the fitted Platt sigmoid to a
    probability. Parity with sklearn is pinned by tests.
    """

    def __init__(self, artifact: dict):
        self.model_id: str = artifact["model_id"]
        self.ngram_min, self.ngram_max = artifact["ngram_range"]
        self.vocabulary: dict[str, int] = artifact["vocabulary"]
        self.idf: list[float] = artifact["idf"]
        self.coef: list[float] = artifact["coef"]
        self.intercept: float = artifact["intercept"]
        # Platt scaling: p = sigmoid(a * decision + b)
        self.calib_a: float = artifact["calibration"]["a"]
        self.calib_b: float = artifact["calibration"]["b"]
        self.metadata: dict = artifact.get("metadata", {})

    @classmethod
    def load(cls, path: Path) -> LinearTextClassifier:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return cls(json.load(f))

    def _counts(self, text: str) -> dict[int, int]:
        doc = _WHITESPACE.sub(" ", text.lower())
        counts: dict[int, int] = {}
        vocab = self.vocabulary
        for n in range(self.ngram_min, min(self.ngram_max, len(doc)) + 1):
            for i in range(len(doc) - n + 1):
                idx = vocab.get(doc[i : i + n])
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
        return counts

    def decision(self, text: str) -> float:
        counts = self._counts(text)
        if not counts:
            return self.intercept
        norm = math.sqrt(sum((c * self.idf[i]) ** 2 for i, c in counts.items()))
        if norm == 0:
            return self.intercept
        dot = sum(c * self.idf[i] * self.coef[i] for i, c in counts.items())
        return dot / norm + self.intercept

    def predict_proba(self, text: str) -> float:
        """Probability the text is AI-generated, in [0, 1]."""
        z = self.calib_a * self.decision(text) + self.calib_b
        return 1.0 / (1.0 + math.exp(-z)) if z > -60 else 0.0


@lru_cache(maxsize=4)
def _load_classifier(path: str) -> LinearTextClassifier | None:
    p = Path(path)
    return LinearTextClassifier.load(p) if p.is_file() else None


def get_classifier() -> LinearTextClassifier | None:
    """The active classifier, or None when no artifact has been trained.

    Resolution order: PAPERNEWS_AI_MODEL (Settings.ai_model), then the
    packaged default path. Cached per path; scoring a whole edition loads
    the artifact once.
    """
    override = get_settings().ai_model
    return _load_classifier(str(override or DEFAULT_MODEL_PATH))


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

    `ai_likelihood` is the trained classifier's calibrated probability, or
    None when no model artifact is available. `reliable=False` means the
    sample was too small for either the classifier or the stylometrics to
    mean anything, and callers must not penalize the document.
    """
    sentences = _sentences(text)
    words = [w for s in sentences for w in s]
    word_count = len(words)

    classifier = get_classifier()
    likelihood: float | None = None
    model_id: str | None = None
    if classifier is not None:
        likelihood = round(classifier.predict_proba(text), 4)
        model_id = classifier.model_id

    return AITextMetrics(
        ai_likelihood=likelihood,
        model_id=model_id,
        burstiness=round(_burstiness([len(s) for s in sentences]), 4),
        lexical_diversity=round(_moving_ttr(words), 4),
        word_count=word_count,
        reliable=(
            word_count >= MIN_RELIABLE_WORDS
            and len(sentences) >= MIN_RELIABLE_SENTENCES
        ),
    )
