# System Directives for AI Agents (`AGENTS.md`)

## 1. Objective and Scope
This document defines the architectural invariants, technology stack guidelines, and operational boundaries for the Papernews repository. All code generation, refactoring, and architectural suggestions must strictly adhere to these constraints.

## 2. Architectural Invariants: Separation of Concerns
Papernews enforces a strict boundary between data processing and presentation.

* **The Backend (Python/Prefect):** Must remain generic, deterministic, and strictly typed via Pydantic. It handles ingestion, routing, filtering, and enrichment.
* **The Frontend (Jinja/Typst):** Must remain bespoke and deeply opinionated. Page layout, font selection, and column generation live exclusively in `papernews/template.typ.j2`.
* **The Bridge:** Backend and frontend communicate exclusively through `RenderContext` (models.py) flattened by `papernews/adapter.py` (`render_context_to_template_vars`) — the single serialization point. The template never reads Pydantic models directly.
* **Inline body emission** is the one sanctioned exception to "no Typst in Python": `papernews/typst_emit.py` emits article bodies from the typed markdown IR (`Block`/`Span`). Layout still belongs to the template; the emitter only produces inline/body markup.

## 3. Technology Stack & Best Practices

### Configuration
* `sources.toml` loads through `papernews/config.py` into strict Pydantic models (`extra="forbid"`); unknown keys and `[category_limits]` entries matching no source category fail at load time. Never bypass `load_config`.
* Process-level switches are `PAPERNEWS_*` env vars via `Settings` (llm_enabled, llm_backend, paths).

### Python & Prefect (Orchestration & Data Flow)
* **Idempotency & Resiliency:** Wrap external network requests and LLM invocations in Prefect `@task` decorators with explicit retry policies. Transient errors must propagate so retries actually fire; degrade to fallbacks only on the final attempt.
* **State Management:** Persist state incrementally to SQLite (`SimpleStore`). Schema changes are append-only entries in `store.MIGRATIONS` (tracked via `PRAGMA user_version`); never edit or reorder a shipped migration.
* **Strict Typing:** All data flowing through the pipeline is validated by Pydantic (`RawDocument`, `ArticleChunk`, `Telemetry`, `RenderContext`). Tasks must not mutate their inputs; return updated copies (`model_copy`).

### LLM Integration (Backends)
* All model calls go through the `LLMBackend` protocol (`papernews/core/backends.py`) — a single OpenAI-compatible transport (`requests`) over any provider. Providers are presets (base URL + key env + default model) chosen by `PAPERNEWS_LLM_PROVIDER` (default `deepseek`; also `openrouter`, `openai`, `groq`, `together`, `fireworks`, `local`), or point `PAPERNEWS_LLM_BASE_URL`/`_API_KEY`/`_MODEL` at any other OpenAI-compatible endpoint. `resolve_provider` fails fast when a preset's key env is missing; structured output uses JSON mode + schema-in-prompt run through `_extract_json` (fence/prose tolerant), validated by the router. Verify a provider with `papernews check-llm` / `papernews providers`, or `GET /healthz?llm=1`. The SQLite response cache sits above the backend in `router._cached_structured_call`; backends stay dumb transport wrappers.
* **Function-as-a-Service:** The LLM's scope is limited to gatekeeping (boolean classification), summarization, and markdown formatting.
* **Data Determinism:** Do not pass deterministic metadata (URLs, timestamps, authors) to the LLM to echo back. Manage deterministic data natively in Python.
* **Immutable Telemetry:** Every LLM invocation returns a populated `Telemetry`; aggregate cumulatively, attributing rejected documents' costs to the run total.

### Ingestion & Plugins (pluggy)
* The plugin contract is explicit: `papernews/plugins/hookspecs.py` declares `fetch_sources`, `enrich_articles`, and `fetch_decorations`; managers are built only via `plugins/registry.get_plugin_manager()`.
* New data sources are `fetch_sources` plugins yielding `RawDocument`s (typed fields `title`/`category`/`published` populated; `metadata` only for genuine extras).
* Cross-article features (salience, entities, marginalia, curiosity queue) are `enrich_articles` plugins: they see the whole day's `ArticleChunk`s at Stage 3.5 and attach sidecar data in place (`blocks` spans, `enrichment`, `annotations`).
* When scraping HTML, use `trafilatura` with `include_images=True` and `include_links=True` strictly maintained.
* **Cost Control (The Triage Funnel):** Never bypass Stage 2. All documents pass deterministic local filters (noise regexes, blacklists, ranking, the AI-likeness screen, category limits) before the LLM.
* **AI-Likeness Screen (Stage 2B.5):** `papernews/ai_detect.py` (adapted from lyc8503/AITextDetector) scores each document's source text with pure-Python stylometrics and deranks formulaic articles below the category-budget cut; optional hard drop via `ai_drop_threshold`. It is a configurable noise dial, not a detector — unreliable (short) samples are never penalized, and the scorer must stay deterministic with zero dependencies. Metrics ride `RawDocument.ai_metrics` → `ArticleChunk.ai_metrics` and surface as a per-article stylometrics footer plus funnel counts in the finished paper.

### Rendering (IR + Typst)
* Article bodies flow through the markdown IR: `markdown_ir.parse_markdown` (pure; plain text + char-offset `Span`s) → enrichers → `typst_emit.emit_blocks` (escapes exactly once, no sentinel tokens). Enrichers operate on `Block.text` offsets and must never see markdown or Typst.
* There is no other render path: new inline constructs are new `Span`/`Block` kinds plus an emitter case. Never reintroduce string-level markup munging.
* `build_pdf` raises `RenderError` on Typst compile failure; never swallow it.

## 4. The Pipeline Reference
Any new feature must map cleanly to one of the following stages:
1. **Ingestion** (pluggy `fetch_sources`) → `RawDocument`.
2. **Triage Funnel** — deterministic filtering/ranking/budget, zero API cost.
3. **LLM Gateway** — backend-routed select/summarize/format → validated `ArticleChunk` & `Telemetry`.
3.5. **Enrichment** (pluggy `enrich_articles`) — whole-day cross-article sidecar data.
4. **Decorations** (pluggy `fetch_decorations`) → `FrontpageDecorations`.
5. **Adapter → Renderer** — `RenderContext` → template vars → Typst PDF.

## 5. Testing Bar
* Every change lands with tests; CI runs pytest (with Hypothesis + the regression corpus and real Typst compilation), ruff format/check, mypy, an import smoke, and a Docker build/boot smoke.
* New render-path behavior must be pinned by golden fragments and survive the hostile-string gauntlet and property fuzz.
* Grow the regression corpus (`tests/fixtures/test_db.json`) whenever a real-world input breaks rendering.
