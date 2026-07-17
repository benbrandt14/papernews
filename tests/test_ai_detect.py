"""Tests for the AI-likeness scorer (papernews.ai_detect).

The scorer is a noise dial adapted from lyc8503/AITextDetector: a fixed
linear decision function over deterministic stylometric features. These
tests pin the contract (bounds, determinism, reliability gating) and the
discrimination direction — formulaic LLM filler must score above varied
human prose — not exact values.
"""

from hypothesis import given
from hypothesis import strategies as st

from papernews.ai_detect import (
    MIN_RELIABLE_WORDS,
    score_text,
)
from papernews.models import AITextMetrics

HUMAN_SAMPLE = """
The launch slipped again. Nobody at the pad seemed surprised.
Engineers had flagged a helium leak in the second stage on Tuesday, and by
Thursday the fix, a replaced seal and two days of requalification, was still
being argued over in a windowless conference room in Hawthorne.
"We fly when it's ready," the program manager said. Short. Flat. Final.
The customer, a small Earth-observation startup out of Helsinki, took the
delay in stride; their last ride waited five months.
What actually worries the range safety office is the weather. A front is
stalling over the Gulf, and the recovery ship can't hold station in
nine-foot swells. If Saturday scrubs, the next window doesn't open until
the 14th, after the range supports a classified mission.
Meanwhile the booster, on its eleventh flight, sat in the hangar with its
grid fins folded like a sleeping bird. Turnaround used to take months.
Now the bottleneck is paperwork, one official joked, not hardware.
"""

SLOP_SAMPLE = """
In today's fast-paced world of space exploration, this launch represents a
testament to human ingenuity. The mission plays a crucial role in the
ever-evolving landscape of commercial spaceflight. Moreover, the innovative
rocket seamlessly integrates cutting-edge technology with proven practices.
It's important to note that the provider has embarked on a journey to
revolutionize access to orbit. Furthermore, the reusable booster underscores
the importance of sustainable spaceflight solutions. The company continues
to delve into new methods for unlocking the potential of rapid reusability.
Additionally, the mission provides valuable insights into the future of the
industry. In conclusion, this launch is a game-changer that will elevate
your understanding of modern rocketry. Whether you're a casual observer or
a seasoned expert, the ever-changing landscape of spaceflight offers a rich
tapestry of innovation. Moreover, the team leverages the power of iterative
design to navigate the complexities of orbital mechanics.
"""


def test_slop_scores_above_human():
    human = score_text(HUMAN_SAMPLE)
    slop = score_text(SLOP_SAMPLE)
    assert human.reliable and slop.reliable
    assert slop.ai_likelihood > human.ai_likelihood
    # Pin the default-threshold semantics: with ai_derank_threshold=0.6 the
    # slop sample is deranked and the human one is not.
    assert slop.ai_likelihood >= 0.6
    assert human.ai_likelihood < 0.6


def test_features_point_the_expected_direction():
    human = score_text(HUMAN_SAMPLE)
    slop = score_text(SLOP_SAMPLE)
    # Human prose alternates sentence lengths; slop runs uniform.
    assert human.burstiness > slop.burstiness
    # The slop sample is saturated with stock phrases; the human one is clean.
    assert human.stock_phrases_per_1k == 0.0
    assert slop.stock_phrases_per_1k > 10


def test_short_text_is_unreliable():
    m = score_text("Too short to say anything statistical about.")
    assert m.reliable is False
    assert m.word_count < MIN_RELIABLE_WORDS


def test_uniform_repetition_is_not_reliable_signal_of_humanity():
    """Degenerate copy (one sentence repeated) reads as low-signal: zero
    burstiness and rock-bottom diversity land mid-scale, not at 'human'."""
    m = score_text("A paragraph with bold text and some math. " * 40)
    assert m.reliable is True
    assert m.burstiness == 0.0
    assert 0.3 <= m.ai_likelihood <= 0.7


def test_empty_and_degenerate_inputs():
    for text in ["", "   ", "\n\n\n", "!!!", "。。。"]:
        m = score_text(text)
        assert isinstance(m, AITextMetrics)
        assert m.reliable is False
        assert m.word_count == 0


def test_deterministic():
    assert score_text(SLOP_SAMPLE) == score_text(SLOP_SAMPLE)


def test_formatted_properties():
    m = AITextMetrics(
        ai_likelihood=0.6234,
        burstiness=0.512,
        lexical_diversity=0.7345,
        stock_phrases_per_1k=3.06,
        word_count=500,
        reliable=True,
    )
    assert m.formatted_likelihood == "62%"
    assert m.formatted_burstiness == "0.51"
    assert m.formatted_diversity == "0.73"
    assert m.formatted_phrase_rate == "3.1"


@given(st.text(max_size=5000))
def test_score_text_total_function(text):
    """Never crashes, and every metric stays inside its documented bounds."""
    m = score_text(text)
    assert 0.0 <= m.ai_likelihood <= 1.0
    assert m.burstiness >= 0.0
    assert 0.0 <= m.lexical_diversity <= 1.0
    assert m.stock_phrases_per_1k >= 0.0
    assert m.word_count >= 0
