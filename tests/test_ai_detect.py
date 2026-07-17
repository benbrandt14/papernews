"""Tests for the AI-likeness classifier runtime (papernews.ai_detect).

The runtime evaluates a trained char-n-gram TF-IDF + linear SVM (the
lyc8503/AITextDetector architecture, English-trained) in pure Python.
These tests pin the contract — sklearn parity, bounds, determinism,
reliability gating, and behavior when no model artifact is installed.
The tiny fixture model (conftest) pins plumbing, not real-world
discrimination; real held-out metrics come from the training corpus and
are stored in the artifact's metadata.
"""

import sys
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from papernews.ai_detect import (
    MIN_RELIABLE_WORDS,
    LinearTextClassifier,
    score_text,
)
from papernews.models import AITextMetrics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

HUMAN_SAMPLE = (
    "The launch slipped again. Nobody at the pad seemed surprised. Engineers "
    "flagged a helium leak on Tuesday, and by Thursday the fix was still being "
    "argued over in a windowless conference room. The customer, a small "
    "startup out of Helsinki, took the delay in stride; their last ride waited "
    "five months. What worries the range office is the weather, since a front "
    "is stalling over the Gulf and the recovery ship cannot hold station in "
    "nine-foot swells. If Saturday scrubs, the next window opens on the 14th. "
    "Meanwhile the booster sat in the hangar, grid fins folded. Turnaround "
    "used to take months. Now the bottleneck is paperwork, one official "
    "joked, not hardware. Nobody laughed harder than the schedulers."
)
SLOP_SAMPLE = (
    "In today's fast-paced world, this launch is a testament to human "
    "ingenuity. The mission plays a crucial role in the ever-evolving "
    "landscape of spaceflight. Moreover, the rocket seamlessly integrates "
    "cutting-edge technology with proven practices. It's important to note "
    "that the provider has embarked on a journey to revolutionize access to "
    "orbit. Furthermore, the booster underscores the importance of "
    "sustainable solutions. Additionally, the mission provides valuable "
    "insights into the future of the industry. In conclusion, this launch is "
    "a game-changer that will elevate your understanding of rocketry. "
    "Whether you're a casual observer or a seasoned expert, the ever-changing "
    "landscape offers a rich tapestry of innovation delivered seamlessly."
)


# --- No model installed ------------------------------------------------------


def test_no_model_means_no_score(monkeypatch):
    """Without a trained artifact the screen observes: stylometrics attach,
    ai_likelihood stays None, and nothing downstream may derank."""
    monkeypatch.delenv("PAPERNEWS_AI_MODEL", raising=False)
    m = score_text(HUMAN_SAMPLE)
    assert m.ai_likelihood is None
    assert m.model_id is None
    assert m.reliable is True  # descriptive stats are still meaningful
    assert m.burstiness > 0
    assert 0 < m.lexical_diversity <= 1


# --- With the tiny fixture model ---------------------------------------------


def test_classifier_scores_registers_apart(ai_classifier_env):
    human = score_text(HUMAN_SAMPLE)
    slop = score_text(SLOP_SAMPLE)
    assert human.model_id and slop.model_id
    assert human.ai_likelihood is not None and slop.ai_likelihood is not None
    assert slop.ai_likelihood > human.ai_likelihood
    # Pin the default-threshold semantics for the derank stage.
    assert slop.ai_likelihood >= 0.6
    assert human.ai_likelihood < 0.6


def test_deterministic(ai_classifier_env):
    assert score_text(SLOP_SAMPLE) == score_text(SLOP_SAMPLE)


def test_short_text_is_unreliable(ai_classifier_env):
    m = score_text("Too short to say anything statistical about.")
    assert m.reliable is False
    assert m.word_count < MIN_RELIABLE_WORDS


def test_empty_and_degenerate_inputs(ai_classifier_env):
    for text in ["", "   ", "\n\n\n", "!!!", "。。。"]:
        m = score_text(text)
        assert isinstance(m, AITextMetrics)
        assert m.reliable is False
        assert m.word_count == 0
        assert m.ai_likelihood is not None  # classifier still returns a prob
        assert 0.0 <= m.ai_likelihood <= 1.0


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text(max_size=3000))
def test_score_text_total_function(ai_classifier_env, text):
    """Never crashes; every metric stays inside its documented bounds."""
    m = score_text(text)
    assert m.ai_likelihood is not None
    assert 0.0 <= m.ai_likelihood <= 1.0
    assert m.burstiness >= 0.0
    assert 0.0 <= m.lexical_diversity <= 1.0
    assert m.word_count >= 0


# --- sklearn parity ----------------------------------------------------------


def test_pure_python_inference_matches_sklearn(tiny_ai_model):
    """The exported artifact must reproduce sklearn's pipeline exactly:
    same decision values, same calibrated probabilities."""
    # Refit sklearn on the identical corpus/seed the fixture used.
    import random as _random

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    from tests.conftest import _AI_POOL, _HUMAN_POOL

    rng = _random.Random(7)
    samples = []
    for pool, label in ((_HUMAN_POOL, 0), (_AI_POOL, 1)):
        for _ in range(80):
            samples.append((label, " ".join(rng.sample(pool, 6))))
    fit_set = samples[:]
    _random.Random(7).shuffle(fit_set)
    fit_set = fit_set[: int(len(fit_set) * 0.8)]

    tfidf = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), min_df=2, max_features=4000
    )
    x = tfidf.fit_transform([t for _, t in fit_set])
    svc = LinearSVC(random_state=7)
    svc.fit(x, [label for label, _ in fit_set])

    clf = LinearTextClassifier.load(tiny_ai_model)
    probe_texts = [t for _, t in samples[:20]] + [HUMAN_SAMPLE, SLOP_SAMPLE]
    ours = np.array([clf.decision(t) for t in probe_texts])
    theirs = svc.decision_function(tfidf.transform(probe_texts))
    # The artifact rounds weights to 6 decimals; tolerance reflects that.
    assert np.abs(ours - theirs).max() < 1e-3


def test_artifact_carries_provenance(tiny_ai_model):
    clf = LinearTextClassifier.load(tiny_ai_model)
    assert clf.model_id.startswith("svc-char24-en-")
    assert clf.metadata["corpus"] == "tiny-test-fixture"
    assert "held_out_auc" in clf.metadata


# --- Formatted properties ----------------------------------------------------


def test_formatted_properties():
    m = AITextMetrics(
        ai_likelihood=0.6234,
        model_id="svc-char24-en-20k",
        burstiness=0.512,
        lexical_diversity=0.7345,
        word_count=500,
        reliable=True,
    )
    assert m.formatted_likelihood == "62%"
    assert m.formatted_burstiness == "0.51"
    assert m.formatted_diversity == "0.73"


def test_formatted_likelihood_without_model():
    assert AITextMetrics().formatted_likelihood == ""
