"""Tests for the FastAPI serving layer (papernews/serve.py).

These tests avoid the network, the LLM, Prefect, and Typst. They cover:

    1. INGEST_SCHEDULE cron-style scheduling
    2. INGEST_INTERVAL_SECONDS fallback when no schedule is set
    3. GET /ingest returns a 405 with a helpful JSON hint
    4. POST_INGEST_HOOK fires after a successful ingest
    5. /healthz (shallow + deep), /digest.pdf, /sources behavior
"""

from __future__ import annotations

import os
import stat

import pytest

# Make sure the background scheduler does not actually start during import.
os.environ.setdefault("PAPERNEWS_NO_SCHED", "1")

from fastapi.testclient import TestClient  # noqa: E402

import papernews.serve as serve  # noqa: E402


@pytest.fixture
def client():
    return TestClient(serve.app)


# --- 1 & 2: scheduler modes -----------------------------------------------


@pytest.fixture
def clean_scheduler_env(monkeypatch):
    monkeypatch.delenv("INGEST_SCHEDULE", raising=False)
    monkeypatch.delenv("INGEST_TIMEZONE", raising=False)
    monkeypatch.delenv("INGEST_INTERVAL_SECONDS", raising=False)


def test_cron_schedule_creates_one_job_per_time(clean_scheduler_env, monkeypatch):
    monkeypatch.setenv("INGEST_SCHEDULE", "07:00,18:30")
    monkeypatch.setenv("INGEST_TIMEZONE", "Europe/London")
    sched = serve.start_scheduler()
    try:
        jobs = sched.get_jobs()
        assert len(jobs) == 2
        triggers = [str(j.trigger) for j in jobs]
        assert any("hour='7'" in t and "minute='0'" in t for t in triggers), (
            f"no 07:00 trigger in {triggers}"
        )
        assert any("hour='18'" in t and "minute='30'" in t for t in triggers), (
            f"no 18:30 trigger in {triggers}"
        )
        tzs = {str(j.trigger.timezone) for j in jobs}
        assert any("Europe/London" in z for z in tzs), (
            f"timezone not propagated; got {tzs}"
        )
    finally:
        sched.shutdown(wait=False)


def test_cron_ignores_malformed_entries_but_keeps_valid_ones(
    clean_scheduler_env, monkeypatch
):
    monkeypatch.setenv("INGEST_SCHEDULE", "07:00,not-a-time,18:00")
    sched = serve.start_scheduler()
    try:
        jobs = sched.get_jobs()
        assert len(jobs) == 2, "malformed entry must be skipped"
    finally:
        sched.shutdown(wait=False)


def test_interval_fallback_when_no_schedule(clean_scheduler_env, monkeypatch):
    monkeypatch.setenv("INGEST_INTERVAL_SECONDS", "60")
    sched = serve.start_scheduler()
    try:
        jobs = sched.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == "ingest"
        assert "interval[" in str(jobs[0].trigger)
    finally:
        sched.shutdown(wait=False)


# --- 3: GET /ingest helper -------------------------------------------------


def test_get_ingest_returns_helpful_405(client):
    r = client.get("/ingest")
    assert r.status_code == 405
    body = r.json()
    assert "error" in body
    assert "POST" in body["error"]
    assert "hint" in body
    assert "curl" in body["hint"].lower()


def test_post_ingest_starts_background_run(client, mocker):
    ran = mocker.patch.object(serve, "_do_ingest")
    r = client.post("/ingest")
    assert r.status_code == 202
    assert r.json()["status"] == "started"
    # The daemon thread targets the (mocked) ingest function.
    import time

    for _ in range(50):
        if ran.called:
            break
        time.sleep(0.01)
    ran.assert_called_once()


# --- 4: POST_INGEST_HOOK ---------------------------------------------------


@pytest.fixture
def hook_env(tmp_path, monkeypatch):
    monkeypatch.delenv("POST_INGEST_HOOK", raising=False)
    monkeypatch.delenv("POST_INGEST_HOOK_TIMEOUT", raising=False)

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-stub")

    hook_log = tmp_path / "hook.log"
    hook = tmp_path / "hook.sh"
    hook.write_text(f'#!/usr/bin/env bash\necho "$1" > "{hook_log}"\n')
    hook.chmod(hook.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return tmp_path, fake_pdf, hook_log, hook


def test_hook_runs_with_pdf_path_after_successful_ingest(hook_env, monkeypatch, mocker):
    tmp_path, fake_pdf, hook_log, hook = hook_env
    monkeypatch.setenv("POST_INGEST_HOOK", str(hook))
    mocker.patch.object(serve, "_run_edition", return_value=fake_pdf)

    serve._do_ingest()

    assert hook_log.exists(), "hook script did not run"
    assert hook_log.read_text().strip() == str(fake_pdf)
    assert serve._last_build["status"] == "ok"
    assert serve._last_build["pdf"] == str(fake_pdf)


def test_hook_failure_does_not_propagate(hook_env, monkeypatch, mocker):
    tmp_path, fake_pdf, _, _ = hook_env
    bad_hook = tmp_path / "bad.sh"
    bad_hook.write_text("#!/usr/bin/env bash\nexit 1\n")
    bad_hook.chmod(bad_hook.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("POST_INGEST_HOOK", str(bad_hook))
    mocker.patch.object(serve, "_run_edition", return_value=fake_pdf)

    # Must not raise.
    serve._do_ingest()
    assert serve._last_build["status"] == "ok"


def test_no_hook_means_no_subprocess(hook_env, mocker):
    _, fake_pdf, _, _ = hook_env
    mocker.patch.object(serve, "_run_edition", return_value=fake_pdf)
    fake_sub = mocker.patch.object(serve, "subprocess")

    serve._do_ingest()
    fake_sub.run.assert_not_called()


def test_failed_ingest_reports_error_status(hook_env, mocker):
    mocker.patch.object(serve, "_run_edition", side_effect=RuntimeError("boom"))

    # Must not raise.
    serve._do_ingest()
    assert serve._last_build["status"] == "error"
    assert "boom" in serve._last_build["error"]


# --- 5: routes --------------------------------------------------------------


def test_healthz_shallow_and_deep(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r = client.get("/healthz", params={"deep": 1})
    assert r.status_code == 200
    assert "last_build" in r.json()


def test_digest_pdf_404_when_no_edition(client, tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERNEWS_OUTPUT", str(tmp_path / "empty"))
    r = client.get("/digest.pdf")
    assert r.status_code == 404
    assert "hint" in r.json()


def test_digest_pdf_serves_newest_edition(client, tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    older = out / "2026-01-01.pdf"
    older.write_bytes(b"%PDF-old")
    newer = out / "2026-01-02.pdf"
    newer.write_bytes(b"%PDF-new")
    os.utime(older, (1, 1))
    monkeypatch.setenv("PAPERNEWS_OUTPUT", str(out))

    r = client.get("/digest.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content == b"%PDF-new"


def test_sources_lists_configured_sources(client, tmp_path, monkeypatch):
    cfg = tmp_path / "sources.toml"
    cfg.write_text(
        '[[source]]\nname = "Example"\nkind = "rss"\nurl = "https://example.com/feed"\n'
        'category = "Tech"\nlimit = 3\n'
    )
    monkeypatch.setenv("PAPERNEWS_CONFIG", str(cfg))

    r = client.get("/sources")
    assert r.status_code == 200
    assert r.json()["sources"] == [{"name": "Example", "kind": "rss", "limit": 3}]
