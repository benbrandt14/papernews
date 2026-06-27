# System Directives for AI Agents (`AGENTS.md`)

## 1. Objective and Scope
This document defines the architectural invariants, technology stack guidelines, and operational boundaries for the Papernews repository. All code generation, refactoring, and architectural suggestions must strictly adhere to these constraints.

## 2. Architectural Invariants: Separation of Concerns
Papernews enforces a strict boundary between data processing and presentation. 

* **The Backend (Python/Prefect):** Must remain generic, deterministic, and strictly typed via Pydantic. It handles ingestion, routing, and filtering.
* **The Frontend (Jinja/Typst):** Must remain bespoke and deeply opinionated. It handles typography, layout, and visual rendering.
* **The Bridge (Stage 4 Adapter):** The backend and frontend communicate exclusively through a data adapter that converts modern Pydantic models into the legacy dictionary structures expected by the Jinja template. 
* **Constraint:** Do not attempt to unify the frontend and backend. Do not propose replacing Jinja templates with programmatic Python-Typst generation loops.

## 3. Technology Stack & Best Practices

### Python & Prefect (Orchestration & Data Flow)
* **Idempotency & Resiliency:** Wrap all external network requests (RSS fetching, API calls) and LLM invocations in Prefect `@task` decorators with explicit retry policies (e.g., `@task(retries=3)` for `503` errors).
* **State Management:** Assume pipeline interruptions. Persist state changes incrementally to the SQLite database (`state.db`) so the pipeline can resume without duplicating API costs.
* **Strict Typing:** All data flowing through the pipeline must be validated by Pydantic schemas (`RawDocument`, `ArticleChunk`, `Telemetry`).

### LLM Integration (Google GenAI)
* **Function-as-a-Service:** Treat the LLM strictly as an isolated text-processing function. Its scope is limited to:
  1. Gatekeeping (Boolean classification).
  2. Summarization (Text reduction).
  3. Formatting (Markdown sanitization).
* **Data Determinism:** Do not pass deterministic metadata (URLs, timestamps, author names) to the LLM with instructions to return them in a JSON schema. Manage all deterministic data natively in Python and map it directly to the output models.
* **Immutable Telemetry:** Token usage and cost tracking are first-class requirements. Every LLM invocation must return a populated `Telemetry` object. Aggregate telemetry data cumulatively, ensuring rejected documents still attribute their token costs to the run total.

### RSS & Data Ingestion (Pluggy & Trafilatura)
* **Plugin Architecture:** Implement new data sources (e.g., Hacker News, Academic feeds, Reddit) as decoupled `pluggy` modules. 
* **Standardized Output:** All ingestion plugins must yield `RawDocument` models.
* **Extraction:** When scraping HTML, use `trafilatura`. Ensure parameters `include_images=True` and `include_links=True` are strictly maintained to preserve source context.
* **Cost Control (The Triage Funnel):** Never bypass the Python-native Triage Funnel (Stage 2). All ingested documents must pass through deterministic local filters (regex blacklists, source limits, ranking heuristics) before being sent to the LLM API. 

### Typst & Jinja (Presentation Layer)
* **Template Immutability:** Layout logic, font selection, and column generation reside exclusively in `papernews/template.typ.j2`. Do not migrate layout logic into Python strings.
* **Compilation Safety:** Typst compilation is fragile when exposed to raw scraped web text. The custom regex pipeline (`_stash_typography`, `_stash_math`, `_stash_images`) in `render.py` acts as a critical safety net. 
* **Constraint:** Do not modify the stashing regex pipeline unless specifically directed to fix a targeted rendering bug (e.g., unbalanced brackets, malformed LaTeX). Fallback to raw text (`doc.raw_text`) gracefully if a Markdown formatting exception occurs.

## 4. The 5-Stage Pipeline Reference
Any new feature must map cleanly to one of the following stages:
1. **Ingestion:** (`pluggy`, `trafilatura`) -> Outputs `RawDocument`.
2. **Triage Funnel:** Python-native deterministic filtering -> Zero API cost.
3. **LLM Gateway:** Gemini API routing -> Outputs validated `ArticleChunk` & `Telemetry`.
4. **Legacy Adapter:** Converts Pydantic models to Jinja-compatible `dict`.
5. **Renderer:** Jinja templating -> Typst PDF compilation.