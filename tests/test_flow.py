"""Flow-level tests: real Prefect orchestration, not `.fn` bypasses.

These are the tests the audit flagged as missing — retry policies were
declared on every LLM task but nothing ever exercised them.
"""

from __future__ import annotations

import os

import pytest
import requests

os.environ.setdefault("DEEPSEEK_API_KEY", "fake-key-for-tests")

from prefect.testing.utilities import prefect_test_harness  # noqa: E402

from papernews.config import AppConfig, Preferences  # noqa: E402
from papernews.core.router import _is_transient, llm_select_article  # noqa: E402
from papernews.models import RawDocument, Telemetry  # noqa: E402
from papernews.store import SimpleStore  # noqa: E402


@pytest.fixture(scope="module")
def prefect_harness():
    with prefect_test_harness():
        yield


@pytest.fixture
def llm_env(monkeypatch, tmp_path, mocker):
    monkeypatch.setenv("PAPERNEWS_LLM_ENABLED", "1")
    mocker.patch(
        "papernews.core.router._db",
        return_value=SimpleStore(str(tmp_path / "state.db")),
    )


DOC = RawDocument(
    source_id="retry_doc",
    content_type="rss",
    raw_text="Body text for the retry test.",
    title="Retry Test",
)


class FlakyBackend:
    """Raises transient errors for the first `failures` calls, then succeeds."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def structured(self, contents, system_instruction, temperature, schema):
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.exceptions.ConnectionError("503 Service Unavailable")
        return '{"is_selected": true}', Telemetry(prompt_tokens=5, output_tokens=2)

    def text(self, contents, system_instruction, temperature):
        raise AssertionError("not used in this test")


def test_llm_task_retries_transient_then_succeeds(prefect_harness, llm_env, mocker):
    """Two 503s, then success: Prefect's retry policy must carry the task
    to the third attempt and return the real result."""
    backend = FlakyBackend(failures=2)
    mocker.patch("papernews.core.router.get_backend", return_value=backend)

    fast = llm_select_article.with_options(retry_delay_seconds=0)
    selected, telemetry = fast(DOC, Preferences())

    assert selected is True
    assert telemetry.prompt_tokens == 5
    assert backend.calls == 3


def test_llm_task_falls_back_after_exhausted_retries(prefect_harness, llm_env, mocker):
    """Persistent transient failure: after the initial run + 3 retries the
    task degrades to its fallback instead of failing the flow."""
    backend = FlakyBackend(failures=99)
    mocker.patch("papernews.core.router.get_backend", return_value=backend)

    fast = llm_select_article.with_options(retry_delay_seconds=0)
    selected, telemetry = fast(DOC, Preferences())

    assert selected is False  # degraded fallback, not an exception
    assert backend.calls == 4  # initial attempt + retries=3


def test_non_transient_error_falls_back_immediately(prefect_harness, llm_env, mocker):
    """A schema/logic error must NOT burn retries — immediate fallback."""

    class BrokenBackend:
        calls = 0

        def structured(self, *a, **k):
            BrokenBackend.calls += 1
            raise ValueError("model returned garbage")

        def text(self, *a, **k):
            raise AssertionError("not used")

    mocker.patch("papernews.core.router.get_backend", return_value=BrokenBackend())

    fast = llm_select_article.with_options(retry_delay_seconds=0)
    selected, _ = fast(DOC, Preferences())

    assert selected is False
    assert BrokenBackend.calls == 1


def test_flow_end_to_end_produces_pdf(prefect_harness, tmp_path, monkeypatch, mocker):
    """The real run_papernews flow (real orchestration, real pluggy relay,
    real Typst compile) with the network stubbed out."""
    monkeypatch.setenv("PAPERNEWS_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setenv("PAPERNEWS_STATE", str(tmp_path / "state.db"))
    monkeypatch.delenv("PAPERNEWS_LLM_ENABLED", raising=False)

    class E:
        link = "https://example.com/a"

        def get(self, k, d=""):
            return {"title": "Flow Test", "published": ""}.get(k, d)

    class F:
        entries = [E()]

    class WikiResp:
        text = '<div class="current-events-content"><li>Event. [1]</li></div>'

        def raise_for_status(self):
            pass

    mocker.patch("feedparser.parse", return_value=F())
    mocker.patch("trafilatura.fetch_url", return_value="<html></html>")
    mocker.patch(
        "trafilatura.extract",
        return_value="A paragraph with **bold** text and $x^2$ math. " * 30,
    )
    mocker.patch("papernews.plugins.wiki_plugin.requests.get", return_value=WikiResp())

    from papernews.core.main import run_papernews

    config = AppConfig(
        sources=[{"name": "S", "kind": "rss", "url": "http://x", "category": "Sci"}],
        category_limits={"Sci": 3},
    )
    pdf = run_papernews(config=config)

    assert pdf.exists()
    assert pdf.read_bytes()[:4] == b"%PDF"
    # Decorations survived to the page.
    typ = (tmp_path / "out" / ".build" / f"{pdf.stem}.typ").read_text()
    assert "Event." in typ


# --- _is_transient unit coverage ---------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (requests.exceptions.ConnectionError("boom"), True),
        (TimeoutError(), True),
        (ValueError("bad json"), False),
        (KeyError("x"), False),
    ],
)
def test_is_transient_classification(exc, expected):
    assert _is_transient(exc) is expected


def test_is_transient_honors_status_code_attribute():
    class FakeAPIError(Exception):
        code = 503

    assert _is_transient(FakeAPIError()) is True

    class FakeClientError(Exception):
        code = 400

    assert _is_transient(FakeClientError()) is False
