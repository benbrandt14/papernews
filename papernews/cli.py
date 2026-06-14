from __future__ import annotations

import argparse
import sys
import tomllib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_cls, datetime
from pathlib import Path

from .extract import extract
from .fetch import fetch_hn, fetch_rss, fetch_wikipedia_events
from .render import build_pdf
from .store import Store
from .wiki import (
    fetch_did_you_know,
    fetch_quote_of_day,
    fetch_world_news,
    summarize_world_news,
)


def _log(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


def _load_config(path: Path) -> tuple[list[dict], dict, dict]:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
        
    sources = cfg.get("source", [])
    prefs = cfg.get("preferences", {})
    cat_limits = cfg.get("category_limits", {})
    
    if isinstance(prefs, list):
        prefs = prefs[0] if prefs else {}
        
    return sources, prefs, cat_limits


# --- stages -----------------------------------------------------------------

def cmd_clean(state_path: Path, out_dir: Path, reset_db: bool) -> int:
    """Clears out local temporary build artifacts and optionally resets database state."""
    build_dir = out_dir / ".build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        _log(f"[clean] Removed temporary layout directory: {build_dir}")
    else:
        _log("[clean] Layout cache directory already clean")

    if reset_db and state_path.exists():
        state_path.unlink()
        _log(f"[clean] Erased localized tracking database: {state_path}")
    elif reset_db:
        _log("[clean] Database file does not exist; skipping erase")
        
    return 0


def cmd_gather(store: Store, sources: list[dict], force: bool = False) -> int:
    new_count = 0
    failed_count = 0
    for src in sources:
        name = src["name"]
        kind = src.get("kind", "rss")
        category = src.get("category", "Uncategorized")
        fetch_limit = src.get("fetch_limit", 40)
        _log(f"[gather] {name} ({category})")
        
        try:
            if kind == "hn":
                items = fetch_hn(
                    source_name=name,
                    limit=fetch_limit,
                    since_hours=src.get("since_hours", 48),
                    min_points=src.get("min_points", 50),
                )
            elif kind == "rss":
                items = fetch_rss(name, src["url"], limit=fetch_limit)
            elif kind == "wikipedia_events":
                items = fetch_wikipedia_events(
                    source_name=name,
                    days_back=src.get("days_back", 1),
                )
            else:
                _log(f"  [warn] unknown source kind '{kind}'")
                continue
        except Exception as e:
            _log(f"  [error] fetch failed: {e}")
            continue

        for it in items:
            import hashlib
            url_hash = hashlib.sha256(it.url.encode("utf-8")).hexdigest()[:16]
            
            if store.exists(it.url, it.title):
                if force:
                    # Decoupled eviction to force clean re-extraction during prompt/parser iteration
                    store.con.execute("DELETE FROM article WHERE url_hash = ?", (url_hash,))
                    store.con.commit()
                else:
                    store.insert_raw(
                        source=it.source, category=category, url=it.url, title=it.title,
                        text=None, surfaced=it.surfaced,
                    )
                    continue
            try:
                art = extract(it.url, it.title, it.source)
            except Exception as e:
                _log(f"  [error] extract: {it.title[:60]}: {e}")
                art = None
            
            text = art.text if art else None
            
            if not text or len(text.strip()) < 200:
                store.insert_raw(
                    source=it.source, category=category, url=it.url, title=it.title,
                    text=None, surfaced=it.surfaced,
                )
                failed_count += 1
                _log(f"  - {it.title[:70]}  (no readable content)")
            else:
                pub = art.published if art else None
                pub = pub or it.surfaced
                store.insert_raw(
                    source=it.source, category=category, url=it.url, title=it.title,
                    text=text, surfaced=it.surfaced, published=pub,
                )
                new_count += 1
                _log(f"  + {it.title[:70]}  ({len(text)} chars)")
                
    _log(f"[gather] +{new_count} new, {failed_count} unreadable")
    return 0


def cmd_select(store: Store, sources: list[dict], prefs: dict, cat_limits: dict) -> int:
    from .select import select_articles
    
    total_sel = 0
    total_rej = 0
    
    # Extract unique categories dynamically based on source ordering
    categories = []
    for src in sources:
        cat = src.get("category", "Uncategorized")
        if cat not in categories:
            categories.append(cat)
            
    default_limit = prefs.get("default_category_limit", 3)
    
    for cat in categories:
        limit = cat_limits.get(cat, default_limit)
        
        pending = store.pending_selection_by_category(cat)
        if not pending:
            continue
            
        _log(f"[select] Category '{cat}': evaluating {len(pending)} pending (limit {limit})")
        selected, rejected = select_articles(cat, pending, limit, prefs)
        
        if selected:
            store.set_selection_status(selected, 1)
        if rejected:
            store.set_selection_status(rejected, -1)
            
        total_sel += len(selected)
        total_rej += len(rejected)
        _log(f"  > selected {len(selected)}, rejected {len(rejected)}")
        
    if total_sel > 0 or total_rej > 0:
        _log(f"[select] Total: selected {total_sel}, rejected {total_rej}")
    else:
        _log(f"[select] nothing pending")
        
    return 0


_BATCH_SIZE = 8

def _chunks(seq: list, n: int) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def cmd_summarize(store: Store, workers: int) -> int:
    from .summarize import summarize_batch
    pending = store.pending_summary()
    if not pending:
        _log("[summarize] nothing pending")
        return 0
        
    batches = _chunks(pending, _BATCH_SIZE)
    _log(f"[summarize] {len(pending)} pending in {len(batches)} batch(es) "
         f"of {_BATCH_SIZE} (workers={workers})")

    def _run_batch(rows: list) -> list[tuple[str, str]]:
        items = [(r["title"], r["text"]) for r in rows]
        out = summarize_batch(items)
        if len(out) != len(rows):
            raise ValueError(f"LLM returned {len(out)} items, expected {len(rows)}. Aborting batch.")
        return [(rows[i]["url_hash"], out[i]) for i in range(len(rows))]

    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_batch, b): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                results = fut.result()
            except Exception as e:
                errors += len(batch)
                _log(f"  [error] batch ({len(batch)} articles): {e}")
                continue
            for h, s in results:
                if s:
                    store.set_summary(h, s)
                    done += 1
                else:
                    errors += 1
            _log(f"  ✓ batch of {len(batch)}")
            
    _log(f"[summarize] done {done}/{len(pending)}, {errors} errors")
    return 0


def cmd_rewrite(store: Store, workers: int) -> int:
    from .rewrite import rewrite_batch
    pending = store.pending_rewrite()
    if not pending:
        _log("[rewrite] nothing pending")
        return 0
        
    batches = _chunks(pending, _BATCH_SIZE)
    _log(f"[rewrite] {len(pending)} pending in {len(batches)} batch(es) "
         f"of {_BATCH_SIZE} (workers={workers})")

    def _run_batch(rows: list) -> list[tuple[str, str]]:
        items = [(r["title"], r["text"]) for r in rows]
        out = rewrite_batch(items)
        if len(out) != len(rows):
            raise ValueError(f"LLM returned {len(out)} items, expected {len(rows)}. Aborting batch.")
        return [(rows[i]["url_hash"], out[i]) for i in range(len(rows))]

    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_batch, b): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                results = fut.result()
            except Exception as e:
                errors += len(batch)
                _log(f"  [error] batch ({len(batch)} articles): {e}")
                continue
            for h, body in results:
                if body:
                    store.set_body(h, body)
                    done += 1
                else:
                    errors += 1
            _log(f"  ✓ batch of {len(batch)}")
            
    _log(f"[rewrite] done {done}/{len(pending)}, {errors} errors")
    return 0


def _format_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.strptime(iso[:10], "%Y-%m-%d").strftime("%b %d, %Y")
    except ValueError:
        return iso


def _gather_decorations() -> dict:
    decorations: dict = {}
    try:
        wn = fetch_world_news()
        if wn:
            wn = summarize_world_news(wn)
            decorations["world_news"] = wn
            from datetime import date as _d
            decorations["world_news_date"] = _d.today().strftime("%B %-d, %Y")
    except Exception as e:
        _log(f"  [warn] world news: {e}")
    try:
        qotd = fetch_quote_of_day()
        if qotd:
            decorations["quote"] = {"text": qotd[0], "author": qotd[1]}
    except Exception as e:
        _log(f"  [warn] qotd: {e}")
    try:
        dyk = fetch_did_you_know(limit=4)
        if dyk:
            decorations["dyk"] = dyk
    except Exception as e:
        _log(f"  [warn] dyk: {e}")
    return decorations


def _collect_current_edition(store: Store, sources: list[dict], prefs: dict, cat_limits: dict) -> list[dict]:
    out: list[dict] = []
    
    categories = []
    for src in sources:
        cat = src.get("category", "Uncategorized")
        if cat not in categories:
            categories.append(cat)
            
    default_limit = prefs.get("default_category_limit", 3)
    
    for cat in categories:
        limit = cat_limits.get(cat, default_limit)
        rows = store.latest_per_category(cat, limit)
        for r in rows:
            out.append({
                "url_hash": r["url_hash"],
                "source": r["source"],
                "category": r["category"],
                "url": r["url"],
                "title": r["title"],
                "text": r["body"] if r["body"] else r["text"],
                "summary": r["summary"],
                "date": _format_date(r["published"] or r["surfaced"]),
            })
    return out


def cmd_render(store: Store, date: str, out_dir: Path, sources: list[dict], prefs: dict, cat_limits: dict) -> int:
    articles = _collect_current_edition(store, sources, prefs, cat_limits)
    if not articles:
        _log("[render] no ready articles in store yet")
        return 0
        
    _log("[render] fetching cover decorations (Wikipedia world news + QOTD + DYK)")
    decorations = _gather_decorations()
    _log(f"[render] {len(articles)} articles → PDF")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = build_pdf(date, articles, out_dir, decorations=decorations)
    
    hashes = [a["url_hash"] for a in articles]
    store.mark_rendered(hashes, date)
    
    print(str(pdf))
    return 0


def cmd_ingest(store: Store, sources: list[dict], prefs: dict, cat_limits: dict, workers: int, force_gather: bool = False, skip_rewrite: bool = False) -> int:
    rc = cmd_gather(store, sources, force=force_gather)
    if rc: return rc
    rc = cmd_select(store, sources, prefs, cat_limits)
    if rc: return rc
    rc = cmd_summarize(store, workers)
    if rc: return rc
    
    if skip_rewrite:
        _log("[ingest] Skipping optional long-form paragraph body optimization step")
        return 0
    return cmd_rewrite(store, workers)


def cmd_status(store: Store) -> int:
    c = store.counts()
    print(f"total articles         : {c['total']}")
    print(f"  unreadable           : {c['unreadable']}")
    print(f"  rejected by filter   : {c['rejected']}")
    print(f"  awaiting selection   : {c['pending_select']}")
    print(f"  awaiting summary     : {c['pending_summary']}")
    print(f"  awaiting rewrite     : {c['pending_rewrite']}")
    print(f"  awaiting render      : {c['pending_render']}")
    print(f"  already rendered     : {c['rendered']}")
    return 0


# --- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="papernews")
    p.add_argument("--config", type=Path, default=Path("sources.toml"))
    p.add_argument("--out",    type=Path, default=Path("archive"))
    p.add_argument("--state",  type=Path, default=Path("state.db"))

    sub = p.add_subparsers(dest="cmd")
    
    # Dev utilities
    sp_cl = sub.add_parser("clean", help="clean localized compile caches and storage indices")
    sp_cl.add_argument("--db", action="store_true", help="wipe database clean to force full structural refetching")

    sp_gat = sub.add_parser("gather", help="fetch + extract new articles into the store")
    sp_gat.add_argument("--force", action="store_true", help="evict and re-extract existing matching records")
    
    sub.add_parser("select", help="downselect pending articles via LLM/heuristics")
    
    sp_sum = sub.add_parser("summarize", help="summarize articles missing a summary")
    sp_sum.add_argument("--workers", type=int, default=6)
    
    sp_rw = sub.add_parser("rewrite", help="reformat article bodies into clean paragraphs")
    sp_rw.add_argument("--workers", type=int, default=6)
    
    sp_ing = sub.add_parser("ingest", help="gather + select + summarize + rewrite (no PDF)")
    sp_ing.add_argument("--workers", type=int, default=6)
    sp_ing.add_argument("--force", action="store_true", help="force raw collection overrides")
    sp_ing.add_argument("--skip-rewrite", action="store_true", help="bypass expensive context text structural rewrites")
    
    sp_ren = sub.add_parser("render", help="render the current edition PDF")
    sp_ren.add_argument("--date", default=date_cls.today().isoformat())
    
    sub.add_parser("status", help="print store counts")
    
    sp_b = sub.add_parser("build", help="ingest + render (default)")
    sp_b.add_argument("--workers", type=int, default=6)
    sp_b.add_argument("--date",    default=date_cls.today().isoformat())
    sp_b.add_argument("--force", action="store_true", help="force raw collection overrides")
    sp_b.add_argument("--skip-rewrite", action="store_true", help="bypass expensive context text structural rewrites")

    args = p.parse_args(argv)
    cmd = args.cmd or "build"

    if cmd == "clean":
        return cmd_clean(args.state, args.out, args.db)

    if not args.config.exists():
        _log(f"[fatal] config not found: {args.config}")
        return 2

    sources, prefs, cat_limits = _load_config(args.config)
    if cmd in ("gather", "select", "ingest", "build") and not sources:
        _log("[fatal] no sources configured")
        return 2

    store = Store(args.state)
    
    source_category_map = {s["name"]: s.get("category", "Uncategorized") for s in sources}
    store.sync_categories(source_category_map)

    if cmd == "gather":
        return cmd_gather(store, sources, force=args.force)
    if cmd == "select":
        return cmd_select(store, sources, prefs, cat_limits)
    if cmd == "summarize":
        return cmd_summarize(store, args.workers)
    if cmd == "rewrite":
        return cmd_rewrite(store, args.workers)
    if cmd == "ingest":
        return cmd_ingest(store, sources, prefs, cat_limits, args.workers, force_gather=args.force, skip_rewrite=args.skip_rewrite)
    if cmd == "render":
        return cmd_render(store, args.date, args.out, sources, prefs, cat_limits)
    if cmd == "status":
        return cmd_status(store)
    if cmd == "build":
        rc = cmd_ingest(store, sources, prefs, cat_limits, args.workers, force_gather=args.force, skip_rewrite=args.skip_rewrite)
        if rc:
            return rc
        return cmd_render(store, args.date, args.out, sources, prefs, cat_limits)

    _log(f"[fatal] unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())