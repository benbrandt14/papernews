import os

os.environ["GEMINI_API_KEY"] = "dummy"
from papernews.core.main import (
    triage_process_a_filter,
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
        metadata={"title": "Short Document"},
    )
    result = func([doc], {})
    assert len(result) == 0


def test_filter_keeps_short_non_rss():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="wiki_event",
        raw_text="A" * 100,
        metadata={"title": "Short Document"},
    )
    result = func([doc], {})
    assert len(result) == 1


def test_filter_drops_blacklisted():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    prefs = {"blacklist_words": ["badword", "terrible"]}
    doc1 = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="This has a badword in it." + ("A" * 800),
        metadata={"title": "Doc 1"},
    )
    doc2 = RawDocument(
        source_id="2",
        content_type="rss",
        raw_text="This is fine." + ("A" * 800),
        metadata={"title": "This is terrible!"},
    )
    doc3 = RawDocument(
        source_id="3",
        content_type="rss",
        raw_text="This is perfectly fine." + ("A" * 800),
        metadata={"title": "Good title"},
    )
    result = func([doc1, doc2, doc3], prefs)
    assert len(result) == 1
    assert result[0].source_id == "3"


def test_filter_keeps_clean():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    prefs = {"blacklist_words": ["badword", "terrible"]}
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="This is clean." + ("A" * 800),
        metadata={"title": "Clean"},
    )
    result = func([doc], prefs)
    assert len(result) == 1


def test_rank_prioritizes_interests():
    func = getattr(triage_process_b_rank, "fn", triage_process_b_rank)
    prefs = {"interest": ["Science", "AI tech"]}
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="Text",
        metadata={"title": "A new science discovery"},
    )
    result = func([doc], prefs)
    assert len(result) == 1
    assert result[0].metadata["heuristic_score"] == 1


def test_rank_assigns_default():
    func = getattr(triage_process_b_rank, "fn", triage_process_b_rank)
    prefs = {"interest": ["Science", "AI"]}
    doc = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="Text",
        metadata={"title": "Some random topic"},
    )
    result = func([doc], prefs)
    assert len(result) == 1
    assert result[0].metadata["heuristic_score"] == 3


def test_rank_sorts_correctly():
    func = getattr(triage_process_b_rank, "fn", triage_process_b_rank)
    prefs = {"interest": ["AI"]}
    doc1 = RawDocument(
        source_id="1",
        content_type="rss",
        raw_text="Text",
        metadata={"title": "Random topic"},
    )
    doc2 = RawDocument(
        source_id="2",
        content_type="rss",
        raw_text="Text",
        metadata={"title": "AI is cool"},
    )
    doc3 = RawDocument(
        source_id="3",
        content_type="rss",
        raw_text="Text",
        metadata={"title": "Another random topic"},
    )
    result = func([doc1, doc2, doc3], prefs)
    assert len(result) == 3
    assert result[0].source_id == "2"
    assert result[0].metadata["heuristic_score"] == 1
    assert result[1].metadata["heuristic_score"] == 3
    assert result[2].metadata["heuristic_score"] == 3


def test_budget_enforces_category_limits():
    func = getattr(triage_process_c_budget, "fn", triage_process_c_budget)
    limits = {"Tech": 2, "Science": 1}
    prefs = {"default_category_limit": 1}
    docs = [
        RawDocument(
            source_id="1",
            content_type="rss",
            raw_text="A",
            metadata={"category": "Tech"},
        ),
        RawDocument(
            source_id="2",
            content_type="rss",
            raw_text="A",
            metadata={"category": "Tech"},
        ),
        RawDocument(
            source_id="3",
            content_type="rss",
            raw_text="A",
            metadata={"category": "Tech"},
        ),
        RawDocument(
            source_id="4",
            content_type="rss",
            raw_text="A",
            metadata={"category": "Science"},
        ),
        RawDocument(
            source_id="5",
            content_type="rss",
            raw_text="A",
            metadata={"category": "Science"},
        ),
    ]
    result = func(docs, limits, prefs)
    assert len(result) == 3
    assert [d.source_id for d in result] == ["1", "2", "4"]


def test_budget_uses_default_limit():
    func = getattr(triage_process_c_budget, "fn", triage_process_c_budget)
    limits = {"Tech": 2}
    prefs = {"default_category_limit": 1}
    docs = [
        RawDocument(
            source_id="1",
            content_type="rss",
            raw_text="A",
            metadata={"category": "Misc"},
        ),
        RawDocument(
            source_id="2",
            content_type="rss",
            raw_text="A",
            metadata={"category": "Misc"},
        ),
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
        metadata={"title": "Too Long Document"},
    )
    result = func([doc], {"max_char_length": 20000})
    assert len(result) == 0


def test_filter_keeps_overly_long_academic():
    func = getattr(triage_process_a_filter, "fn", triage_process_a_filter)
    doc = RawDocument(
        source_id="1",
        content_type="academic_pdf",
        raw_text="A" * 50000,
        metadata={"title": "Long Academic Document"},
    )
    result = func([doc], {"max_char_length": 20000})
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
        metadata={},  # Malformed, no title
    )
    result = func([doc], {"blacklist_words": ["badword"]})
    assert len(result) == 1
