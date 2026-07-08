import os

import pytest

# We need to disable Prefect's background processes and network connections for tests
os.environ["PREFECT_API_URL"] = ""
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "false"
os.environ["PREFECT_TEST_MODE"] = "true"
os.environ["DEEPSEEK_API_KEY"] = "fake-key-for-tests"

from papernews.config import Preferences
from papernews.core.router import (
    llm_select_article,
    llm_summarize_article,
)
from papernews.models import RawDocument
from papernews.store import SimpleStore


@pytest.fixture
def test_db(tmp_path):
    # Setup an on-disk SQLite database in tmp_path to avoid the Path(":memory:") issues in store.py
    # SimpleStore passes db_path to Path(), and then connects multiple times across functions.
    # In-memory connections are per-thread unless we pass the connection directly, so using a temp file is safer.
    db_path = tmp_path / "test_state.db"
    store = SimpleStore(str(db_path))
    yield store


@pytest.fixture
def llm_enabled(monkeypatch, test_db, mocker):
    """Enable the LLM via env (as production would) and wire the temp store."""
    monkeypatch.setenv("PAPERNEWS_LLM_ENABLED", "1")
    mocker.patch("papernews.core.router._db", return_value=test_db)
    mocker.patch(
        "papernews.core.router.get_run_logger", return_value=mocker.MagicMock()
    )


def _mock_gemini(mocker, text: str, prompt_tokens: int, output_tokens: int):
    """Mock the HTTP transport so the real backend + router code runs.

    Named for history; it now stubs the OpenAI-compatible chat endpoint.
    Returns the patched ``requests.post`` mock.
    """
    resp = mocker.MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
        },
    }
    return mocker.patch("papernews.core.backends.requests.post", return_value=resp)


def test_llm_select_article(test_db, llm_enabled, mocker):
    doc = RawDocument(
        source_id="test_rss",
        content_type="rss",
        raw_text="This is a test article about Python and AI.",
        title="Python AI",
    )

    _mock_gemini(mocker, '{"is_selected": true}', 50, 10)

    selected, telemetry = llm_select_article.fn(doc, Preferences())

    assert selected is True
    assert telemetry.prompt_tokens == 50
    assert test_db.get_cache("select_test_rss") == '{"is_selected": true}'


def test_llm_select_article_cache_hit_skips_api(test_db, llm_enabled, mocker):
    doc = RawDocument(
        source_id="test_rss",
        content_type="rss",
        raw_text="Body",
        title="Python AI",
    )
    test_db.set_cache("select_test_rss", '{"is_selected": false}')
    post = _mock_gemini(mocker, '{"is_selected": true}', 50, 10)

    selected, telemetry = llm_select_article.fn(doc, Preferences())

    assert selected is False  # from cache, not the mock response
    assert telemetry.prompt_tokens == 0  # cache hits cost nothing
    post.assert_not_called()


def test_llm_summarize_article(test_db, llm_enabled, mocker):
    doc = RawDocument(
        source_id="test_rss",
        content_type="rss",
        raw_text="This is a test article about Python and AI. It is very long.",
        title="Python AI",
    )

    _mock_gemini(mocker, '{"summary": "A Python and AI test."}', 100, 20)

    summary, telemetry = llm_summarize_article.fn(doc)

    assert summary == "A Python and AI test."
    assert telemetry.prompt_tokens == 100
    assert (
        test_db.get_cache("summary_test_rss") == '{"summary": "A Python and AI test."}'
    )


def test_llm_disabled_short_circuits(monkeypatch, mocker):
    monkeypatch.delenv("PAPERNEWS_LLM_ENABLED", raising=False)
    mocker.patch(
        "papernews.core.router.get_run_logger", return_value=mocker.MagicMock()
    )
    post = _mock_gemini(mocker, '{"is_selected": true}', 1, 1)

    doc = RawDocument(source_id="x", content_type="rss", raw_text="Body", title="Title")
    selected, telemetry = llm_select_article.fn(doc, Preferences())

    assert selected is True  # auto-accept
    assert telemetry.prompt_tokens == 0
    post.assert_not_called()
