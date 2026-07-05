"""FastAPI web service for papernews.

Routes:
  GET  /              landing page (cover preview + 'Read today' link)
  GET  /digest.pdf    newest edition PDF from the output directory
  GET  /preview.png   page-1 PNG of the newest edition
  GET  /sources       JSON list of configured sources
  GET  /healthz       liveness probe (?deep=1 adds last-build status)
  POST /ingest        trigger a pipeline run in the background

Background:
  APScheduler runs the Prefect flow on a schedule.

Environment:
  PAPERNEWS_CONFIG           path to sources.toml      (default: sources.toml)
  PAPERNEWS_OUTPUT           path to the PDF output dir (default: output)
  PAPERNEWS_NO_SCHED=1       disable the background scheduler

  Scheduling — pick one:
    INGEST_INTERVAL_SECONDS  every N seconds           (default: 14400 = 4h)
    INGEST_SCHEDULE          "HH:MM,HH:MM,..." cron-style fixed times
    INGEST_TIMEZONE          IANA tz, used with INGEST_SCHEDULE (default: UTC)

  Post-ingest delivery hook:
    POST_INGEST_HOOK           executable on disk; receives the PDF path as $1
    POST_INGEST_HOOK_TIMEOUT   seconds (default: 300)
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from papernews.config import AppConfig, load_config
from papernews.preview import render_cover_png

# --- Config helpers -------------------------------------------------------


def _config_path() -> Path:
    return Path(os.environ.get("PAPERNEWS_CONFIG", "sources.toml"))


def _output_dir() -> Path:
    return Path(os.environ.get("PAPERNEWS_OUTPUT", "output"))


def _load_config() -> AppConfig:
    return load_config(_config_path())


def _latest_pdf() -> Path | None:
    out = _output_dir()
    if not out.is_dir():
        return None
    pdfs = sorted(out.glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    return pdfs[-1] if pdfs else None


# --- Build pipeline -------------------------------------------------------

_ingest_lock = threading.Lock()

# Last-build status, reported by /healthz?deep=1.
_last_build: dict = {"status": "never", "time": None, "pdf": None, "error": None}


def _run_edition() -> Path:
    """Run the Prefect pipeline once and return the produced PDF path."""
    # Imported lazily so serving/scheduling works without Prefect fully
    # initialized at module import (and so tests can stub it cheaply).
    from papernews.core.main import run_papernews

    config = _load_config()
    return Path(run_papernews(config=config))


def _do_ingest() -> None:
    if not _ingest_lock.acquire(blocking=False):
        return  # one ingest at a time
    try:
        pdf = _run_edition()
        _last_build.update(
            status="ok",
            time=datetime.now(UTC).isoformat(),
            pdf=str(pdf),
            error=None,
        )
        _run_post_ingest_hook(pdf)
    except Exception as e:
        _last_build.update(
            status="error",
            time=datetime.now(UTC).isoformat(),
            error=f"{type(e).__name__}: {e}",
        )
        sys.stderr.write(f"[ingest] failed: {e}\n")
        sys.stderr.flush()
    finally:
        _ingest_lock.release()


def _run_post_ingest_hook(pdf: Path) -> None:
    """Fire the optional delivery hook with the fresh PDF path.

    Hook failures are logged but never propagate — a broken hook must not
    crash the ingest loop.
    """
    hook = os.environ.get("POST_INGEST_HOOK", "").strip()
    if not hook:
        return
    try:
        subprocess.run(
            [hook, str(pdf)],
            timeout=int(os.environ.get("POST_INGEST_HOOK_TIMEOUT", "300")),
            check=False,
        )
    except Exception as e:
        sys.stderr.write(f"[post-ingest hook] {e}\n")
        sys.stderr.flush()


# --- FastAPI app -----------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="papernews", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz(deep: int = 0):
        if deep:
            return JSONResponse({"status": "ok", "last_build": _last_build})
        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _LANDING_HTML

    @app.get("/sources")
    def sources_endpoint():
        cfg = _load_config()
        return JSONResponse(
            {
                "sources": [
                    {"name": s.name, "kind": s.kind, "limit": s.limit}
                    for s in cfg.sources
                ],
            }
        )

    @app.get("/digest.pdf")
    def digest_pdf():
        pdf = _latest_pdf()
        if pdf is None:
            return JSONResponse(
                {
                    "error": "no edition built yet",
                    "hint": "curl -X POST http://localhost:8000/ingest",
                },
                status_code=404,
            )
        return FileResponse(
            pdf,
            media_type="application/pdf",
            filename=f"papernews-{pdf.stem}.pdf",
            content_disposition_type="inline",
            headers={"Cache-Control": "max-age=300"},
        )

    @app.get("/preview.png")
    def preview_png():
        pdf = _latest_pdf()
        if pdf is None:
            return JSONResponse({"error": "no edition built yet"}, status_code=404)
        png = pdf.with_suffix(".preview.png")
        if not png.exists() or png.stat().st_mtime < pdf.stat().st_mtime:
            render_cover_png(pdf, png, dpi=180)
        return FileResponse(
            png, media_type="image/png", headers={"Cache-Control": "max-age=300"}
        )

    @app.post("/ingest")
    def trigger_ingest():
        # Optional manual kick; for cron-style external triggers.
        if _ingest_lock.locked():
            return JSONResponse({"status": "already running"}, status_code=202)
        threading.Thread(target=_do_ingest, daemon=True).start()
        return JSONResponse({"status": "started"}, status_code=202)

    @app.get("/ingest")
    def ingest_get_hint():
        # Friendly 405 — easier than rediscovering you wanted POST.
        return JSONResponse(
            {
                "error": "POST required to trigger ingest",
                "hint": "curl -X POST http://localhost:8000/ingest",
                "note": "the background scheduler also runs ingest automatically",
            },
            status_code=405,
        )

    return app


def start_scheduler() -> BackgroundScheduler:
    """Start the background ingest scheduler.

    Two modes (in priority order):
      INGEST_SCHEDULE=07:00,18:00   → cron-style at the listed HH:MM times
      INGEST_INTERVAL_SECONDS=14400 → every N seconds (default 4h)

    The cron mode also honours INGEST_TIMEZONE (an IANA tz, default UTC).
    """
    sched = BackgroundScheduler(daemon=True)
    schedule = os.environ.get("INGEST_SCHEDULE", "").strip()
    if schedule:
        tz = os.environ.get("INGEST_TIMEZONE", "UTC")
        times = [s.strip() for s in schedule.split(",") if s.strip()]
        for i, hm in enumerate(times):
            try:
                h, m = hm.split(":")
                sched.add_job(
                    _do_ingest,
                    "cron",
                    hour=int(h),
                    minute=int(m),
                    id=f"ingest_cron_{i}",
                    timezone=tz,
                )
            except (ValueError, KeyError):
                sys.stderr.write(f"[scheduler] ignoring invalid time: {hm!r}\n")
                sys.stderr.flush()
    else:
        interval = int(os.environ.get("INGEST_INTERVAL_SECONDS", str(4 * 3600)))
        sched.add_job(
            _do_ingest,
            "interval",
            seconds=interval,
            id="ingest",
            next_run_time=None,
        )
    sched.start()
    return sched


_LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>papernews</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { font-family: Georgia, "Times New Roman", serif; max-width: 720px;
           margin: 4rem auto; padding: 0 1.25rem; color: #222; }
    h1   { font-size: 2.4rem; margin: 0 0 0.2rem; }
    .sub { color: #777; margin: 0 0 2rem; font-size: 1rem; }
    a.cta { display: inline-block; padding: 0.7rem 1.4rem; border: 1px solid #222;
            text-decoration: none; color: #222; font-weight: bold; margin-top: 1rem;}
    a.cta:hover { background: #222; color: #fff; }
    img.cover { width: 100%; height: auto; border: 1px solid #eee;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
    .meta { color: #999; font-size: 0.85rem; margin-top: 3rem; }
  </style>
</head>
<body>
  <h1>papernews</h1>
  <p class="sub">A curated PDF you read on your Boox Note, not in a browser.</p>
  <img class="cover" src="/preview.png" alt="Cover preview">
  <p><a class="cta" href="/digest.pdf">Read today (PDF)</a></p>
  <p class="meta">Updated automatically every few hours. <a href="/sources">Sources</a>.</p>
</body>
</html>
"""


# ASGI entry point
app = create_app()
_scheduler = start_scheduler() if os.environ.get("PAPERNEWS_NO_SCHED") != "1" else None
