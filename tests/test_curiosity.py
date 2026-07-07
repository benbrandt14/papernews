"""Tests for the curiosity-queue enrichment plugin."""

from __future__ import annotations

from unittest.mock import Mock

from papernews.config import AppConfig
from papernews.models import ArticleChunk, LLMOpenQuestions, Telemetry
from papernews.plugins import curiosity_plugin
from papernews.store import SimpleStore


def _article(url: str = "https://example.com/a") -> ArticleChunk:
    return ArticleChunk(
        category="Science",
        source="example.com",
        title="A curious discovery",
        summary="Researchers observed something unexpected.",
        body_markdown="Body.",
        url=url,
    )


class _FakeBackend:
    """Records calls; returns a fixed structured payload."""

    def __init__(self, questions: list[str]):
        self._questions = questions
        self.calls = 0

    def structured(self, contents, system_instruction, temperature, schema):
        self.calls += 1
        return LLMOpenQuestions(
            questions=self._questions
        ).model_dump_json(), Telemetry()

    def text(self, contents, system_instruction, temperature):  # pragma: no cover
        return None, Telemetry()


def test_generate_questions_caches_by_url(tmp_path, mocker):
    store = SimpleStore(str(tmp_path / "state.db"))
    backend = _FakeBackend(["Q1?", "Q2?", "Q3?", "Q4?"])
    mocker.patch("papernews.plugins.curiosity_plugin.get_backend", return_value=backend)

    art = _article()
    first = curiosity_plugin._generate_questions(art, store)
    # Capped at MAX_QUESTIONS_PER_ARTICLE.
    assert first == ["Q1?", "Q2?", "Q3?"]

    # Second call for the same URL is served from cache — no new backend hit.
    second = curiosity_plugin._generate_questions(art, store)
    assert second == ["Q1?", "Q2?", "Q3?"]
    assert backend.calls == 1


def _openalex_response(results):
    resp = Mock()
    resp.json.return_value = {"results": results}
    resp.raise_for_status.return_value = None
    return resp


def test_lookup_openalex_prefers_doi_and_gates_on_relevance(mocker):
    # Strong hit with a DOI wins.
    mocker.patch(
        "papernews.plugins.curiosity_plugin.requests.get",
        return_value=_openalex_response(
            [
                {
                    "title": "The Answer",
                    "doi": "https://doi.org/x",
                    "relevance_score": 42.0,
                }
            ]
        ),
    )
    assert curiosity_plugin._lookup_openalex("q?") == (
        "The Answer",
        "https://doi.org/x",
    )

    # Weak relevance is rejected — the question stays open.
    mocker.patch(
        "papernews.plugins.curiosity_plugin.requests.get",
        return_value=_openalex_response(
            [
                {
                    "title": "Barely related",
                    "id": "https://openalex.org/W1",
                    "relevance_score": 0.3,
                }
            ]
        ),
    )
    assert curiosity_plugin._lookup_openalex("q?") is None

    # No results at all → None.
    mocker.patch(
        "papernews.plugins.curiosity_plugin.requests.get",
        return_value=_openalex_response([]),
    )
    assert curiosity_plugin._lookup_openalex("q?") is None


def test_lookup_openalex_falls_back_to_openalex_id(mocker):
    mocker.patch(
        "papernews.plugins.curiosity_plugin.requests.get",
        return_value=_openalex_response(
            [
                {
                    "title": "No DOI here",
                    "id": "https://openalex.org/W9",
                    "relevance_score": 20.0,
                }
            ]
        ),
    )
    assert curiosity_plugin._lookup_openalex("q?") == (
        "No DOI here",
        "https://openalex.org/W9",
    )


def test_enrich_articles_disabled_llm_raises_no_questions(tmp_path, mocker):
    store = SimpleStore(str(tmp_path / "state.db"))
    mocker.patch(
        "papernews.plugins.curiosity_plugin.get_settings",
        return_value=mocker.Mock(llm_enabled=False),
    )
    # No open questions to chase, so requests.get must never be called.
    getter = mocker.patch("papernews.plugins.curiosity_plugin.requests.get")

    arts = [_article()]
    curiosity_plugin.enrich_articles(arts, AppConfig(), store)

    assert arts[0].enrichment.open_questions == []
    assert store.open_questions() == []
    getter.assert_not_called()


def test_enrich_articles_generates_then_resolves(tmp_path, mocker):
    store = SimpleStore(str(tmp_path / "state.db"))

    # A question parked on an earlier run, waiting to be answered.
    store.add_question("What resolves this?", "https://old", "2026-07-01")

    mocker.patch(
        "papernews.plugins.curiosity_plugin.get_settings",
        return_value=mocker.Mock(llm_enabled=True),
    )
    mocker.patch(
        "papernews.plugins.curiosity_plugin.get_backend",
        return_value=_FakeBackend(["Fresh question?"]),
    )
    mocker.patch(
        "papernews.plugins.curiosity_plugin.requests.get",
        return_value=_openalex_response(
            [
                {
                    "title": "Resolving Work",
                    "doi": "https://doi.org/z",
                    "relevance_score": 30.0,
                }
            ]
        ),
    )

    arts = [_article()]
    curiosity_plugin.enrich_articles(arts, AppConfig(), store)

    # New question attached to the article and parked in the queue.
    assert arts[0].enrichment.open_questions == ["Fresh question?"]

    # The pre-existing question got resolved; the fresh one is still open.
    answered = store.recently_answered()
    assert ("What resolves this?", "Resolving Work", "https://doi.org/z") in answered
    assert [q for _id, q in store.open_questions()] == ["Fresh question?"]


def test_fetch_decorations_surfaces_answered(tmp_path, mocker):
    path = str(tmp_path / "state.db")
    store = SimpleStore(path)
    store.add_question("Answered one?", "https://x", "2026-07-01")
    qid = store.open_questions()[0][0]
    store.mark_answered(qid, "2026-07-02", "The Paper", "https://doi.org/a")

    # fetch_decorations builds its own SimpleStore(); point it at our db.
    mocker.patch.dict("os.environ", {"PAPERNEWS_STATE": path})

    deco = curiosity_plugin.fetch_decorations(AppConfig())
    assert len(deco.curiosities) == 1
    assert deco.curiosities[0].question == "Answered one?"
    assert deco.curiosities[0].answer_title == "The Paper"
    assert deco.curiosities[0].answer_url == "https://doi.org/a"
