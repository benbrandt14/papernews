import os

import pytest

# We need to disable Prefect's background processes and network connections for tests
os.environ["PREFECT_API_URL"] = ""
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "false"
os.environ["PREFECT_TEST_MODE"] = "true"
os.environ["GEMINI_API_KEY"] = "fake-key-for-tests"

from papernews.core.router import (
    llm_select_article,
    llm_summarize_article,
)
from papernews.models import (
    RawDocument,
)
from papernews.store import SimpleStore


@pytest.fixture
def test_db(tmp_path):
    # Setup an on-disk SQLite database in tmp_path to avoid the Path(":memory:") issues in store.py
    # SimpleStore passes db_path to Path(), and then connects multiple times across functions.
    # In-memory connections are per-thread unless we pass the connection directly, so using a temp file is safer.
    db_path = tmp_path / "test_state.db"
    store = SimpleStore(str(db_path))
    yield store


def test_llm_select_article(test_db, mocker):
    # Enable LLM for the test
    mocker.patch("papernews.core.router.LLM_ENABLE", True)
    mocker.patch("papernews.core.router.db", test_db)
    mocker.patch(
        "papernews.core.router.get_run_logger", return_value=mocker.MagicMock()
    )

    doc = RawDocument(
        source_id="test_rss",
        content_type="rss",
        raw_text="This is a test article about Python and AI.",
        metadata={"title": "Python AI", "url": "http://example.com/python-ai"},
    )

    mock_response = mocker.MagicMock()
    mock_response.text = '{"is_selected": true}'
    mock_response.usage_metadata.prompt_token_count = 50
    mock_response.usage_metadata.candidates_token_count = 10
    mocker.patch(
        "papernews.core.router.client.models.generate_content",
        return_value=mock_response,
    )

    selected, telemetry = llm_select_article.fn(doc, {})

    assert selected is True
    assert telemetry.prompt_tokens == 50
    assert test_db.get_cache("select_test_rss") == '{"is_selected": true}'


def test_llm_summarize_article(test_db, mocker):
    mocker.patch("papernews.core.router.LLM_ENABLE", True)
    mocker.patch("papernews.core.router.db", test_db)
    mocker.patch(
        "papernews.core.router.get_run_logger", return_value=mocker.MagicMock()
    )

    doc = RawDocument(
        source_id="test_rss",
        content_type="rss",
        raw_text="This is a test article about Python and AI. It is very long.",
        metadata={"title": "Python AI", "url": "http://example.com/python-ai"},
    )

    mock_response = mocker.MagicMock()
    mock_response.text = '{"summary": "A Python and AI test."}'
    mock_response.usage_metadata.prompt_token_count = 100
    mock_response.usage_metadata.candidates_token_count = 20
    mocker.patch(
        "papernews.core.router.client.models.generate_content",
        return_value=mock_response,
    )

    summary, telemetry = llm_summarize_article.fn(doc)

    assert summary == "A Python and AI test."
    assert telemetry.prompt_tokens == 100
    assert (
        test_db.get_cache("summary_test_rss") == '{"summary": "A Python and AI test."}'
    )
