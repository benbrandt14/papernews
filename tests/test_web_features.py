"""Tests for the changes that addressed issue #1.

These tests intentionally avoid touching the network, the Anthropic SDK, or
Typst. They cover the four user-visible features added in that issue:

    1. INGEST_SCHEDULE cron-style scheduling
    2. INGEST_INTERVAL_SECONDS fallback when no schedule is set
    3. GET /ingest returns a 405 with a helpful JSON hint
    4. POST_INGEST_HOOK fires after a successful ingest
"""

from __future__ import annotations

import os
import stat

import pytest

# Make sure the background scheduler does not actually start during import.
os.environ.setdefault("PAPERNEWS_NO_SCHED", "1")
os.environ.setdefault("PAPERNEWS_STATE", "/tmp/papernews-tests-state.db")
os.environ.setdefault("PAPERNEWS_CACHE", "/tmp/papernews-tests-cache")


def _fresh_scheduler():
    """Re-import start_scheduler so it picks up the current env vars."""
    from importlib import reload

    import papernews.web as web

    reload(web)
    return web.start_scheduler()


# --- 1 & 2: scheduler modes -----------------------------------------------


@pytest.fixture
def clean_scheduler_env(monkeypatch):
    monkeypatch.delenv("INGEST_SCHEDULE", raising=False)
    monkeypatch.delenv("INGEST_TIMEZONE", raising=False)
    monkeypatch.delenv("INGEST_INTERVAL_SECONDS", raising=False)


def test_cron_schedule_creates_one_job_per_time(clean_scheduler_env, monkeypatch):
    monkeypatch.setenv("INGEST_SCHEDULE", "07:00,18:30")
    monkeypatch.setenv("INGEST_TIMEZONE", "Europe/London")
    sched = _fresh_scheduler()
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
    sched = _fresh_scheduler()
    try:
        jobs = sched.get_jobs()
        assert len(jobs) == 2, "malformed entry must be skipped"
    finally:
        sched.shutdown(wait=False)


def test_interval_fallback_when_no_schedule(clean_scheduler_env, monkeypatch):
    monkeypatch.setenv("INGEST_INTERVAL_SECONDS", "60")
    sched = _fresh_scheduler()
    try:
        jobs = sched.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == "ingest"
        assert "interval[" in str(jobs[0].trigger)
    finally:
        sched.shutdown(wait=False)


# --- 3: GET /ingest helper -------------------------------------------------


def test_get_ingest_returns_helpful_405():
    from papernews.web import app

    client = app.test_client()
    r = client.get("/ingest")
    assert r.status_code == 405
    body = r.get_json()
    assert "error" in body
    assert "POST" in body["error"]
    assert "hint" in body
    assert "curl" in body["hint"].lower()


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

    from importlib import reload

    import papernews.web as web

    reload(web)

    mocker.patch.object(web, "cmd_ingest", return_value=0)
    mocker.patch.object(web, "_build_pdf_for_key", return_value=fake_pdf)
    mocker.patch.object(web, "_load_sources", return_value=[])
    mocker.patch.object(web, "Store", return_value=mocker.MagicMock())
    mocker.patch.object(web, "_current_key", return_value="testkey")

    web._do_ingest()

    assert hook_log.exists(), "hook script did not run"
    assert hook_log.read_text().strip() == str(fake_pdf)


def test_hook_failure_does_not_propagate(hook_env, monkeypatch, mocker):
    tmp_path, fake_pdf, _, _ = hook_env
    bad_hook = tmp_path / "bad.sh"
    bad_hook.write_text("#!/usr/bin/env bash\nexit 1\n")
    bad_hook.chmod(bad_hook.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("POST_INGEST_HOOK", str(bad_hook))

    from importlib import reload

    import papernews.web as web

    reload(web)

    mocker.patch.object(web, "cmd_ingest", return_value=0)
    mocker.patch.object(web, "_build_pdf_for_key", return_value=fake_pdf)
    mocker.patch.object(web, "_load_sources", return_value=[])
    mocker.patch.object(web, "Store", return_value=mocker.MagicMock())
    mocker.patch.object(web, "_current_key", return_value="testkey")

    # Must not raise.
    web._do_ingest()


def test_no_hook_means_no_subprocess(hook_env, mocker):
    from importlib import reload

    import papernews.web as web

    reload(web)

    mocker.patch.object(web, "cmd_ingest", return_value=0)
    mocker.patch.object(web, "_load_sources", return_value=[])
    mocker.patch.object(web, "Store", return_value=mocker.MagicMock())
    fake_sub = mocker.patch.object(web, "subprocess")

    web._do_ingest()
    fake_sub.run.assert_not_called()
