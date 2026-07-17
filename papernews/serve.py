"""FastAPI web service for papernews.

Routes:
  GET  /              landing page (cover preview, rebuild button, edit link)
  GET  /digest.pdf    newest edition PDF from the output directory
  GET  /preview.png   page-1 PNG of the newest edition
  GET  /sources       JSON list of configured sources
  GET  /edit          sources.toml editor page
  GET  /config        raw sources.toml text (JSON-wrapped)
  POST /config        validate + write sources.toml, optionally rebuild
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
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from papernews.config import AppConfig, load_config, parse_config
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


class ConfigUpdate(BaseModel):
    """POST /config payload: new sources.toml text, optionally rebuild after."""

    content: str
    rebuild: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="papernews", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    def healthz(deep: int = 0, llm: int = 0) -> JSONResponse:
        body: dict = {"status": "ok"}
        if deep:
            body["last_build"] = _last_build
        if llm:
            # Opt-in: probing makes a real provider call, so it is never part
            # of a plain liveness check.
            from papernews.config import get_settings
            from papernews.core.backends import get_backend

            try:
                ok, detail = get_backend(get_settings()).check()
            except ValueError as e:
                ok, detail = False, str(e)
            body["llm"] = {"ok": ok, "detail": detail}
        return JSONResponse(body)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _LANDING_HTML

    @app.get("/sources")
    def sources_endpoint() -> JSONResponse:
        cfg = _load_config()
        return JSONResponse(
            {
                "sources": [
                    {"name": s.name, "kind": s.kind, "limit": s.limit}
                    for s in cfg.sources
                ],
            }
        )

    @app.get("/edit", response_class=HTMLResponse)
    def edit_page() -> str:
        return _EDITOR_HTML

    @app.get("/config")
    def get_config() -> JSONResponse:
        path = _config_path()
        if not path.exists():
            return JSONResponse(
                {"error": f"config not found at {path}"}, status_code=404
            )
        return JSONResponse({"path": str(path), "content": path.read_text("utf-8")})

    @app.post("/config")
    def post_config(update: ConfigUpdate) -> JSONResponse:
        # Validate before anything touches disk — a broken config must never
        # replace a working one.
        try:
            cfg = parse_config(update.content)
        except tomllib.TOMLDecodeError as e:
            return JSONResponse({"error": f"invalid TOML: {e}"}, status_code=422)
        except ValidationError as e:
            return JSONResponse({"error": str(e)}, status_code=422)

        # Write in place (no atomic tmp+rename): sources.toml is typically a
        # single-file docker bind mount, and replacing the inode would detach
        # the container's view of the file from the host's.
        path = _config_path()
        with open(path, "w", encoding="utf-8") as f:
            f.write(update.content)

        body: dict = {"status": "saved", "sources": len(cfg.sources)}
        if update.rebuild:
            if _ingest_lock.locked():
                body["rebuild"] = "already running"
            else:
                threading.Thread(target=_do_ingest, daemon=True).start()
                body["rebuild"] = "started"
        return JSONResponse(body)

    @app.get("/digest.pdf")
    def digest_pdf() -> Response:
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
    def preview_png() -> Response:
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
    def trigger_ingest() -> JSONResponse:
        # Optional manual kick; for cron-style external triggers.
        if _ingest_lock.locked():
            return JSONResponse({"status": "already running"}, status_code=202)
        threading.Thread(target=_do_ingest, daemon=True).start()
        return JSONResponse({"status": "started"}, status_code=202)

    @app.get("/ingest")
    def ingest_get_hint() -> JSONResponse:
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


def start_initial_ingest_if_empty() -> bool:
    """On boot, build a first edition when none exists yet.

    A fresh deploy would otherwise show /digest.pdf 404 until the first
    scheduled run (up to INGEST_INTERVAL_SECONDS later). Restarts that already
    have an edition on the mounted volume skip this. Returns True if an ingest
    was kicked off.
    """
    if _latest_pdf() is not None:
        return False
    threading.Thread(target=_do_ingest, daemon=True).start()
    return True


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
    .row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1rem; }
    a.cta, button.cta {
            display: inline-block; padding: 0.7rem 1.4rem; border: 1px solid #222;
            text-decoration: none; color: #222; font-weight: bold;
            background: #fff; font-family: inherit; font-size: 1rem; cursor: pointer; }
    a.cta:hover, button.cta:hover { background: #222; color: #fff; }
    button.cta:disabled { opacity: 0.5; cursor: wait; }
    img.cover { width: 100%; height: auto; border: 1px solid #eee;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
    .meta { color: #999; font-size: 0.85rem; margin-top: 3rem; }
    #status { color: #777; font-size: 0.9rem; margin-top: 0.8rem; min-height: 1.2em; }
  </style>
</head>
<body>
  <h1>papernews</h1>
  <p class="sub">A curated PDF you read on your Boox Note, not in a browser.</p>
  <img class="cover" src="/preview.png" alt="Cover preview">
  <div class="row">
    <a class="cta" href="/digest.pdf">Read today (PDF)</a>
    <button class="cta" id="rebuild">Rebuild now</button>
    <a class="cta" href="/edit">Edit sources</a>
  </div>
  <p id="status"></p>
  <p class="meta">Updated automatically every few hours. <a href="/sources">Sources</a>.</p>
  <script>
    const btn = document.getElementById('rebuild');
    const status = document.getElementById('status');
    let poll = null;
    async function refreshStatus() {
      const r = await fetch('/healthz?deep=1');
      const b = (await r.json()).last_build || {};
      if (b.status === 'ok') {
        status.textContent = 'Last build: ok (' + (b.time || '') + ')';
        clearInterval(poll); poll = null; btn.disabled = false;
        document.querySelector('img.cover').src = '/preview.png?t=' + Date.now();
      } else if (b.status === 'error') {
        status.textContent = 'Last build failed: ' + (b.error || 'unknown error');
        clearInterval(poll); poll = null; btn.disabled = false;
      } else {
        status.textContent = 'Building…';
      }
    }
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      status.textContent = 'Starting…';
      await fetch('/ingest', { method: 'POST' });
      poll = setInterval(refreshStatus, 3000);
    });
  </script>
</body>
</html>
"""

_EDITOR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>papernews — edit sources</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { font-family: Georgia, "Times New Roman", serif; max-width: 860px;
           margin: 3rem auto; padding: 0 1.25rem; color: #222; }
    h1   { font-size: 1.8rem; margin: 0 0 0.2rem; }
    .sub { color: #777; margin: 0 0 1.5rem; font-size: 0.95rem; }
    textarea { width: 100%; height: 60vh; font-family: ui-monospace, Menlo, Consolas,
               monospace; font-size: 0.85rem; border: 1px solid #ccc;
               padding: 0.8rem; box-sizing: border-box; }
    .row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1rem; }
    button.cta, a.cta {
            display: inline-block; padding: 0.6rem 1.2rem; border: 1px solid #222;
            color: #222; font-weight: bold; background: #fff; text-decoration: none;
            font-family: inherit; font-size: 0.95rem; cursor: pointer; }
    button.cta:hover, a.cta:hover { background: #222; color: #fff; }
    button.cta:disabled { opacity: 0.5; cursor: wait; }
    #status { margin-top: 1rem; font-size: 0.9rem; white-space: pre-wrap;
              font-family: ui-monospace, Menlo, Consolas, monospace; }
    #status.ok { color: #2a6f2a; }
    #status.err { color: #a33; }
  </style>
</head>
<body>
  <h1>Edit sources</h1>
  <p class="sub">This is <code>sources.toml</code>. Saves are validated first —
     a broken config is rejected and the current one stays in place.</p>
  <textarea id="toml" spellcheck="false">loading…</textarea>
  <div class="row">
    <button class="cta" id="save">Save</button>
    <button class="cta" id="saveRebuild">Save &amp; rebuild</button>
    <a class="cta" href="/">← Back</a>
  </div>
  <p id="status"></p>
  <script>
    const ta = document.getElementById('toml');
    const status = document.getElementById('status');
    fetch('/config').then(r => r.json()).then(b => { ta.value = b.content || ''; });
    async function save(rebuild) {
      const btns = document.querySelectorAll('button');
      btns.forEach(b => b.disabled = true);
      status.className = ''; status.textContent = 'Saving…';
      try {
        const r = await fetch('/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: ta.value, rebuild }),
        });
        const b = await r.json();
        if (r.ok) {
          status.className = 'ok';
          status.textContent = 'Saved (' + b.sources + ' sources)'
            + (b.rebuild ? ' — rebuild ' + b.rebuild : '');
        } else {
          status.className = 'err';
          status.textContent = b.error || 'Save failed';
        }
      } catch (e) {
        status.className = 'err';
        status.textContent = 'Save failed: ' + e;
      } finally {
        btns.forEach(b => b.disabled = false);
      }
    }
    document.getElementById('save').addEventListener('click', () => save(false));
    document.getElementById('saveRebuild').addEventListener('click', () => save(true));
  </script>
</body>
</html>
"""


# ASGI entry point
app = create_app()
_scheduler = None
if os.environ.get("PAPERNEWS_NO_SCHED") != "1":
    _scheduler = start_scheduler()
    # Catch up immediately on a fresh deploy so the first paper isn't an
    # interval away. Fires in a background thread; the server stays responsive.
    start_initial_ingest_if_empty()
