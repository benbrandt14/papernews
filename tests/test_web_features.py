"""Tests for the changes that addressed issue #1.

These tests intentionally avoid touching the network, the Anthropic SDK, or
Typst. They cover the four user-visible features added in that issue:

    1. INGEST_SCHEDULE cron-style scheduling
    2. INGEST_INTERVAL_SECONDS fallback when no schedule is set
    3. GET /ingest returns a 405 with a helpful JSON hint
    4. POST_INGEST_HOOK fires after a successful ingest

Run them with:

    python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


# Make sure the background scheduler does not actually start during import.
os.environ.setdefault("PAPERNEWS_NO_SCHED", "1")
os.environ.setdefault("PAPERNEWS_STATE", "/tmp/papernews-tests-state.db")
os.environ.setdefault("PAPERNEWS_CACHE", "/tmp/papernews-tests-cache")


def _fresh_scheduler():
    """Re-import start_scheduler so it picks up the current env vars.

    APScheduler keeps its own list of jobs once a scheduler is created, so we
    create a fresh one per test and shut it down afterwards.
    """
    from importlib import reload
    import papernews.web as web
    reload(web)
    return web.start_scheduler()


# --- 1 & 2: scheduler modes -----------------------------------------------

class SchedulerModeTests(unittest.TestCase):
    def setUp(self):
        # Clean env for each test
        for k in ("INGEST_SCHEDULE", "INGEST_TIMEZONE", "INGEST_INTERVAL_SECONDS"):
            os.environ.pop(k, None)

    def test_cron_schedule_creates_one_job_per_time(self):
        os.environ["INGEST_SCHEDULE"] = "07:00,18:30"
        os.environ["INGEST_TIMEZONE"] = "Europe/London"
        sched = _fresh_scheduler()
        try:
            jobs = sched.get_jobs()
            self.assertEqual(len(jobs), 2)
            triggers = [str(j.trigger) for j in jobs]
            self.assertTrue(
                any("hour='7'" in t and "minute='0'" in t for t in triggers),
                f"no 07:00 trigger in {triggers}",
            )
            self.assertTrue(
                any("hour='18'" in t and "minute='30'" in t for t in triggers),
                f"no 18:30 trigger in {triggers}",
            )
            tzs = {str(j.trigger.timezone) for j in jobs}
            self.assertTrue(
                any("Europe/London" in z for z in tzs),
                f"timezone not propagated; got {tzs}",
            )
        finally:
            sched.shutdown(wait=False)

    def test_cron_ignores_malformed_entries_but_keeps_valid_ones(self):
        os.environ["INGEST_SCHEDULE"] = "07:00,not-a-time,18:00"
        sched = _fresh_scheduler()
        try:
            jobs = sched.get_jobs()
            self.assertEqual(len(jobs), 2, "malformed entry must be skipped")
        finally:
            sched.shutdown(wait=False)

    def test_interval_fallback_when_no_schedule(self):
        os.environ["INGEST_INTERVAL_SECONDS"] = "60"
        sched = _fresh_scheduler()
        try:
            jobs = sched.get_jobs()
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].id, "ingest")
            self.assertIn("interval[", str(jobs[0].trigger))
        finally:
            sched.shutdown(wait=False)


# --- 3: GET /ingest helper -------------------------------------------------

class GetIngestHintTests(unittest.TestCase):
    def test_get_ingest_returns_helpful_405(self):
        from papernews.web import app
        client = app.test_client()
        r = client.get("/ingest")
        self.assertEqual(r.status_code, 405)
        body = r.get_json()
        self.assertIn("error", body)
        self.assertIn("POST", body["error"])
        self.assertIn("hint", body)
        self.assertIn("curl", body["hint"].lower())


# --- 4: POST_INGEST_HOOK ---------------------------------------------------

class PostIngestHookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        # Stub PDF the hook would receive
        self.fake_pdf = self.tmp_path / "fake.pdf"
        self.fake_pdf.write_bytes(b"%PDF-stub")
        # Hook script that just records its argv
        self.hook_log = self.tmp_path / "hook.log"
        self.hook = self.tmp_path / "hook.sh"
        self.hook.write_text(f'#!/usr/bin/env bash\necho "$1" > "{self.hook_log}"\n')
        self.hook.chmod(self.hook.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def tearDown(self):
        for k in ("POST_INGEST_HOOK", "POST_INGEST_HOOK_TIMEOUT"):
            os.environ.pop(k, None)
        self.tmp.cleanup()

    def test_hook_runs_with_pdf_path_after_successful_ingest(self):
        os.environ["POST_INGEST_HOOK"] = str(self.hook)

        from importlib import reload
        import papernews.web as web
        reload(web)

        with (
            # Bypass actual ingest work (no network, no Anthropic).
            mock.patch.object(web, "cmd_ingest", return_value=0),
            # Pretend the PDF was already built.
            mock.patch.object(web, "_build_pdf_for_key", return_value=self.fake_pdf),
            # Avoid hitting sources.toml.
            mock.patch.object(web, "_load_sources", return_value=[]),
            # Avoid creating a real Store.
            mock.patch.object(web, "Store", return_value=mock.MagicMock()),
            mock.patch.object(web, "_current_key", return_value="testkey"),
        ):
            web._do_ingest()

        self.assertTrue(self.hook_log.exists(), "hook script did not run")
        self.assertEqual(self.hook_log.read_text().strip(), str(self.fake_pdf))

    def test_hook_failure_does_not_propagate(self):
        # Hook that exits non-zero — ingest should still complete cleanly.
        bad_hook = self.tmp_path / "bad.sh"
        bad_hook.write_text("#!/usr/bin/env bash\nexit 1\n")
        bad_hook.chmod(bad_hook.stat().st_mode | stat.S_IEXEC)
        os.environ["POST_INGEST_HOOK"] = str(bad_hook)

        from importlib import reload
        import papernews.web as web
        reload(web)

        with (
            mock.patch.object(web, "cmd_ingest", return_value=0),
            mock.patch.object(web, "_build_pdf_for_key", return_value=self.fake_pdf),
            mock.patch.object(web, "_load_sources", return_value=[]),
            mock.patch.object(web, "Store", return_value=mock.MagicMock()),
            mock.patch.object(web, "_current_key", return_value="testkey"),
        ):
            # Must not raise.
            web._do_ingest()

    def test_no_hook_means_no_subprocess(self):
        os.environ.pop("POST_INGEST_HOOK", None)

        from importlib import reload
        import papernews.web as web
        reload(web)

        with (
            mock.patch.object(web, "cmd_ingest", return_value=0),
            mock.patch.object(web, "_load_sources", return_value=[]),
            mock.patch.object(web, "Store", return_value=mock.MagicMock()),
            mock.patch.object(web, "subprocess") as fake_sub,
        ):
            web._do_ingest()
            fake_sub.run.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
