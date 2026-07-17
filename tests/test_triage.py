import os

os.environ["DEEPSEEK_API_KEY"] = "dummy"
from papernews.config import Preferences
from papernews.core.main import (
    triage_process_a_filter,
    triage_process_b5_ai_derank,
    triage_process_b_rank,
    triage_process_c_budget,
)
from papernews.models import RawDocument


def test_filter_drops_short_rss():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="A" * 100,
        title="Short Document",
    )
    result = func([doc], Preferences())
    assert len(result) == 0


def test_filter_keeps_short_non_rss():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="wiki_event",
        raw_text="A" * 100,
        title="Short Document",
    )
    result = func([doc], Preferences())
    assert len(result) == 1


def test_filter_drops_blacklisted():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    prefs = Preferences(blacklist_words=["badword", "terrible"])
    doc1 = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="This has a badword in it." + ("A" * 800),
        title="Doc 1",
    )
    doc2 = RawDocument(
        source_id="2",
        content_type="rss",
        raw_text="This is fine." + ("A" * 800),
        title="This is terrible!",
    )
    doc3 = RawDocument(
        source_id="3",
        content_type="rss",
        raw_text="This is perfectly fine." + ("A" * 800),
        title="Good title",
    )
    result = func([doc1, doc2, doc3], prefs)
    assert len(result) == 1
    assert result[0].source_id == "3"


def test_filter_keeps_clean():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    prefs = Preferences(blacklist_words=["badword", "terrible"])
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="This is clean." + ("A" * 800),
        title="Clean",
    )
    result = func([doc], prefs)
    assert len(result) == 1


def test_filter_drops_url_only_body():
    """The built-in noise patterns drop docs whose body is just a bare URL."""
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="wiki_event",  # short-doc gate doesn't apply
        raw_text="https://example.com/some-page",
        title="Linkdump",
    )
    result = func([doc], Preferences())
    assert len(result) == 0


def test_filter_drops_noise_topic_in_title():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="Interesting body text. " + ("A" * 800),
        title="New results in mice models of disease",
    )
    result = func([doc], Preferences())
    assert len(result) == 0


def test_rank_prioritizes_interests():
    func = getattr(triage_process_b_rank, "fn", triage_process_b_rank)
    prefs = Preferences(interest=["Science", "AI tech"])
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="Text",
        title="A new science discovery",
    )
    result = func([doc], prefs)
    assert len(result) == 1
    assert result[0].heuristic_score == 1


def test_rank_assigns_default():
    func = getattr(triage_process_b_rank, "fn", triage_process_b_rank)
    prefs = Preferences(interest=["Science", "AI"])
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="Text",
        title="Some random topic",
    )
    result = func([doc], prefs)
    assert len(result) == 1
    assert result[0].heuristic_score == 3


def test_rank_sorts_correctly():
    func = getattr(triage_process_b_rank, "fn", triage_process_b_rank)
    prefs = Preferences(interest=["AI"])
    doc1 = RawDocument(
        source_id="1", content_type="rss", raw_text="Text", title="Random topic"
    )
    doc2 = RawDocument(
        source_id="2", content_type="rss", raw_text="Text", title="AI is cool"
    )
    doc3 = RawDocument(
        source_id="3",
        content_type="rss",
        raw_text="Text",
        title="Another random topic",
    )
    result = func([doc1, doc2, doc3], prefs)
    assert len(result) == 3
    assert result[0].source_id == "2"
    assert result[0].heuristic_score == 1
    assert result[1].heuristic_score == 3
    assert result[2].heuristic_score == 3


def test_rank_does_not_mutate_inputs():
    """Stage 2B must return scored copies, never mutate its input docs."""
    func = getattr(triage_process_b_rank, "fn", triage_process_b_rank)
    doc = RawDocument(
        source_id="1", content_type="rss", raw_text="Text", title="AI is cool"
    )
    result = func([doc], Preferences(interest=["AI"]))
    assert result[0].heuristic_score == 1
    assert doc.heuristic_score == 3  # original untouched


# Realistic-shape bodies for the AI-likeness screen: both long enough to be
# reliable; one varied and clean, one uniform and saturated with LLM filler.
_HUMAN_BODY = (
    "The launch slipped again. Nobody at the pad seemed surprised. "
    "Engineers flagged a helium leak on Tuesday, and by Thursday the fix was "
    "still being argued over in a windowless conference room. "
    '"We fly when it\'s ready," the program manager said. Short. Flat. Final. '
    "The customer, a small startup out of Helsinki, took the delay in stride; "
    "their last ride waited five months. What worries the range office is the "
    "weather, since a front is stalling over the Gulf and the recovery ship "
    "cannot hold station in nine-foot swells. If Saturday scrubs, the next "
    "window opens on the 14th. Meanwhile the booster sat in the hangar with "
    "its grid fins folded like a sleeping bird. Turnaround used to take "
    "months. Now the bottleneck is paperwork, one official joked, not "
    "hardware. Nobody laughed harder than the schedulers."
)
_SLOP_BODY = (
    "In today's fast-paced world, this launch is a testament to human "
    "ingenuity. The mission plays a crucial role in the ever-evolving "
    "landscape of spaceflight. Moreover, the rocket seamlessly integrates "
    "cutting-edge technology with proven practices. It's important to note "
    "that the provider has embarked on a journey to revolutionize access to "
    "orbit. Furthermore, the booster underscores the importance of "
    "sustainable solutions. The company continues to delve into new methods "
    "for unlocking the potential of reusability. Additionally, the mission "
    "provides valuable insights into the future of the industry. In "
    "conclusion, this launch is a game-changer that will elevate your "
    "understanding of rocketry. Whether you're a casual observer or a "
    "seasoned expert, the ever-changing landscape offers a rich tapestry of "
    "innovation. Moreover, the team leverages the power of iterative design "
    "to navigate the complexities of orbital mechanics."
)


def _doc(source_id: str, body: str, category: str = "Sci") -> RawDocument:
    return RawDocument(
        source_id=source_id,
        content_type="rss",
        raw_text=body,
        title=f"Doc {source_id}",
        category=category,
    )


def test_ai_derank_observe_only_without_model(monkeypatch):
    """No trained artifact installed: metrics attach but nothing deranks —
    the screen acts on the classifier's score or not at all."""
    monkeypatch.delenv("PAPERNEWS_AI_MODEL", raising=False)
    func = getattr(triage_process_b5_ai_derank, "fn", triage_process_b5_ai_derank)
    docs = [_doc("slop", _SLOP_BODY), _doc("h1", _HUMAN_BODY)]
    result, deranked, dropped = func(docs, Preferences(ai_drop_threshold=0.1))
    assert (deranked, dropped) == (0, 0)
    assert [d.source_id for d in result] == ["slop", "h1"]
    assert all(d.ai_metrics is not None for d in result)
    assert all(d.ai_metrics.ai_likelihood is None for d in result)
    assert all(d.heuristic_score == 3 for d in result)


def test_ai_derank_sinks_flagged_docs(ai_classifier_env):
    """A doc the classifier flags gets the penalty and sorts below clean
    docs, so the category budget cuts it first."""
    func = getattr(triage_process_b5_ai_derank, "fn", triage_process_b5_ai_derank)
    docs = [_doc("slop", _SLOP_BODY), _doc("h1", _HUMAN_BODY), _doc("h2", _HUMAN_BODY)]

    result, deranked, dropped = func(docs, Preferences())

    assert deranked == 1
    assert dropped == 0
    assert [d.source_id for d in result] == ["h1", "h2", "slop"]
    assert result[-1].heuristic_score == 3 + 2  # default penalty applied
    # Metrics attach to every surviving doc for the article footer.
    assert all(d.ai_metrics is not None for d in result)

    # Downstream: a budget of 2 now cuts the flagged doc, not a clean one.
    budget = getattr(triage_process_c_budget, "fn", triage_process_c_budget)
    surviving = budget(result, {"Sci": 2}, Preferences())
    assert [d.source_id for d in surviving] == ["h1", "h2"]


def test_ai_derank_disabled_is_a_noop(ai_classifier_env):
    func = getattr(triage_process_b5_ai_derank, "fn", triage_process_b5_ai_derank)
    docs = [_doc("slop", _SLOP_BODY)]
    result, deranked, dropped = func(docs, Preferences(ai_detection_enabled=False))
    assert (deranked, dropped) == (0, 0)
    assert result[0] is docs[0]
    assert result[0].ai_metrics is None


def test_ai_derank_hard_drop_threshold(ai_classifier_env):
    func = getattr(triage_process_b5_ai_derank, "fn", triage_process_b5_ai_derank)
    docs = [_doc("slop", _SLOP_BODY), _doc("h1", _HUMAN_BODY)]
    result, deranked, dropped = func(docs, Preferences(ai_drop_threshold=0.6))
    assert dropped == 1
    assert deranked == 0
    assert [d.source_id for d in result] == ["h1"]


def test_ai_derank_never_penalizes_short_docs(ai_classifier_env):
    """Unreliable (short) samples keep their rank — metrics attach, no verdict."""
    func = getattr(triage_process_b5_ai_derank, "fn", triage_process_b5_ai_derank)
    doc = _doc("tiny", "A tapestry of game-changers in a fast-paced world.")
    result, deranked, dropped = func([doc], Preferences(ai_drop_threshold=0.1))
    assert (deranked, dropped) == (0, 0)
    assert result[0].heuristic_score == 3
    assert result[0].ai_metrics is not None
    assert result[0].ai_metrics.reliable is False


def test_ai_derank_does_not_mutate_inputs(ai_classifier_env):
    func = getattr(triage_process_b5_ai_derank, "fn", triage_process_b5_ai_derank)
    doc = _doc("slop", _SLOP_BODY)
    result, _, _ = func([doc], Preferences())
    assert result[0].heuristic_score == 5
    assert doc.heuristic_score == 3  # original untouched
    assert doc.ai_metrics is None


def test_ai_derank_configurable_threshold_and_penalty(ai_classifier_env):
    func = getattr(triage_process_b5_ai_derank, "fn", triage_process_b5_ai_derank)
    docs = [_doc("h1", _HUMAN_BODY)]
    # A hair-trigger threshold flags even clean human prose…
    result, deranked, _ = func(
        docs, Preferences(ai_derank_threshold=0.0, ai_derank_penalty=7)
    )
    assert deranked == 1
    assert result[0].heuristic_score == 10
    # …while penalty 0 makes the screen observe-only.
    result, deranked, _ = func(
        docs, Preferences(ai_derank_threshold=0.0, ai_derank_penalty=0)
    )
    assert deranked == 1
    assert result[0].heuristic_score == 3


def test_budget_enforces_category_limits():
    func = getattr(triage_process_c_budget, "fn", triage_process_c_budget)
    limits = {"Tech": 2, "Science": 1}
    prefs = Preferences(default_category_limit=1)
    docs = [
        RawDocument(source_id="1", content_type="rss", raw_text="A", category="Tech"),
        RawDocument(source_id="2", content_type="rss", raw_text="A", category="Tech"),
        RawDocument(source_id="3", content_type="rss", raw_text="A", category="Tech"),
        RawDocument(
            source_id="4", content_type="rss", raw_text="A", category="Science"
        ),
        RawDocument(
            source_id="5", content_type="rss", raw_text="A", category="Science"
        ),
    ]
    result = func(docs, limits, prefs)
    assert len(result) == 3
    assert [d.source_id for d in result] == ["1", "2", "4"]


def test_budget_uses_default_limit():
    func = getattr(triage_process_c_budget, "fn", triage_process_c_budget)
    limits = {"Tech": 2}
    prefs = Preferences(default_category_limit=1)
    docs = [
        RawDocument(source_id="1", content_type="rss", raw_text="A", category="Misc"),
        RawDocument(source_id="2", content_type="rss", raw_text="A", category="Misc"),
    ]
    result = func(docs, limits, prefs)
    assert len(result) == 1
    assert result[0].source_id == "1"


def test_filter_drops_overly_long():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="A" * 50000,
        title="Too Long Document",
    )
    result = func([doc], Preferences(max_char_length=20000))
    assert len(result) == 0


def test_filter_keeps_overly_long_academic():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="academic_pdf",
        raw_text="A" * 50000,
        title="Long Academic Document",
    )
    result = func([doc], Preferences(max_char_length=20000))
    assert len(result) == 1


def test_filter_keeps_malformed():
    """
    Simulates sending a malformed dummy document (e.g. empty title, weird unicode).
    Ensures that missing metadata or weird characters don't crash the deterministic filter.
    """
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="A" * 900 + "\x00\x01\x02 \U0001f600 malformed data",
        # Malformed: no title
    )
    result = func([doc], Preferences(blacklist_words=["badword"]))
    assert len(result) == 1
