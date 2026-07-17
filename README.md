# papernews

An offline-first, highly customized E-Ink daily digest. A Prefect pipeline pulls feeds via plugins, uses an LLM filter, summarize, and format full text, and passes it through an adapter to render one consistently typeset Typst PDF using jinja.

## Pipeline Stages

1. **Stage 1: Ingestion** — Dynamically loads plugins (RSS, Hacker News, Wiki). Outputs strict Pydantic `RawDocument` models.
2. **Stage 2: Filtering** — Enforces deterministic category limits, local ranking heuristics, and regex blacklists natively in Python (zero API cost). An article registry in SQLite records every processed article (URL, computed heuristic score, first-seen timestamp) and drops anything already typeset into a previous edition, so no story ever repeats.
3. **Stage 3: LLM Handling** — Routes surviving documents to the configured provider (any OpenAI-compatible API; DeepSeek by default, OpenRouter or a local Ollama/vLLM server by env switch) for gatekeeper selection, summarization, and strict markdown formatting. Enforces Pydantic schema validation and tracks token `Telemetry`. Article bodies are parsed into the markdown IR (`Block`/`Span`) here.
4. **Stage 3.5: Enrichment** — A whole-day, cross-article pass (`enrich_articles` plugins) that attaches sidecar data in place. The **curiosity queue** lives here: it asks the LLM for a few researchable questions per lead story, parks them in SQLite, and resolves *earlier* runs' questions against the OpenAlex corpus — answered pairs surface on the front matter.
5. **Stage 4: Templating** — The adapter (`adapter.py`) flattens the typed `RenderContext` into the plain dictionaries the template consumes, keeping layout logic strictly decoupled.
6. **Stage 5: Render** — Jinja injects the adapted data into a Typst template (`template.typ.j2`); the typed emitter (`typst_emit.emit_blocks`) turns the markdown IR into Typst, escaping exactly once with no sentinel tokens.
7. **Stage 6: Record Edition** — Only after the PDF compiles successfully, every article in it is stamped `typeset_at` (+ edition date) in the registry. A failed run leaves its articles eligible for the next edition; a successful one retires them for good.

## Configuration (`sources.toml`)

Manage your feeds and categories in `sources.toml` — either by hand, or in the
browser at `http://localhost:8000/edit` (a validated editor: broken TOML or a
schema violation is rejected before anything is written, and "Save & rebuild"
kicks off a fresh ingest immediately). Order matters: sources appear in this
sequence in the generated PDF.

```toml
[[source]]
name = "Hacker News Top"
kind = "hn" # URL handled internally by the plugin
category = "Technology"
limit = 5

[[source]]
name = "Quanta Magazine"
url = "https://api.quantamagazine.org/feed/"
kind = "rss"
category = "Science"

```

The file is validated at load time: unknown keys and `[category_limits]`
entries that don't match any source category are rejected loudly.

## Plugins & Hacker News (HN) Fetching

Ingestion is decoupled via `pluggy`.

* **RSS Plugin:** Uses `feedparser` and `trafilatura` to extract full-text bodies and images.
* **HN Plugin:** Bypasses standard web scraping and queries the official Algolia Search API (`kind = "hn"`). It pulls the top articles by points within a specific time window to guarantee high-quality curation.
* **Extensibility:** To add a new source (e.g., Academic PDFs), write a new `pluggy` module yielding `RawDocument` objects. As long as it respects the extraction parameters, the pipeline handles the rest.

## Docker & API Structure

**Prerequisites:** Docker and a DeepSeek API key (or any OpenAI-compatible provider).

```bash
git clone [https://github.com/benbrandt14/papernews](https://github.com/benbrandt14/papernews)
cd papernews
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY (or switch PAPERNEWS_LLM_PROVIDER)
docker compose up --build -d

```

On first boot the server builds an initial edition automatically (no need to wait for the schedule), so `http://localhost:8000` shows a paper within a minute or two. `docker compose` reads `.env` if present; without one it runs LLM-off with sensible defaults and still produces a digest.

**API Interaction:**
The application exposes a web server at `http://localhost:8000`.

* To trigger a new compilation pipeline run, send a request to the `/ingest` endpoint (or use the "Rebuild now" button on the landing page).
* Edit sources in the browser at `/edit`; the file is validated before every save, and `sources.toml` is bind-mounted read-write so changes persist on the host.
* State (SQLite) and PDF outputs live in `./data/state.db` and `./data/output/` (bind-mounted from the host).
* The state DB includes the **article registry** — which URLs were processed when (with their computed heuristic scores) and which edition typeset them. Deleting it makes previously published articles eligible again.
* **Resetting State:** To perform a hard reset, simply delete `data/state.db` and restart the container. The database schema will automatically rebuild on boot.
* **Verify the LLM provider:** `docker compose exec papernews papernews check-llm` (and `papernews providers` to list presets).

### Deploy on a Synology NAS

`git clone` (or copy) the repo into a shared folder, then from that folder:

```bash
docker compose up --build -d      # or use Container Manager → Project, pointed at docker-compose.yml
```

* `sources.toml` is committed, so the bind mount resolves on a fresh checkout; `./data/` is created on first run and holds `state.db` + the PDFs (owned by root — the container runs as root, which is fine for a bind mount).
* Everything works without `.env` (LLM off). To enable the LLM, create `.env` from `.env.example` and set a provider key.
* The container makes no unexpected outbound calls (Prefect telemetry is disabled); only your configured feeds and LLM provider are contacted.
* Turn features off without editing code via `PAPERNEWS_DISABLE_PLUGINS` (e.g. `wiki_plugin,curiosity_plugin,salience_plugin`).
* **Note:** if an article contains LaTeX math, Typst fetches the `mitex` package from `packages.typst.org` on first compile — the NAS needs outbound internet for that (cached afterwards).


## Scheduling ingests

Two modes; pick whichever fits your routine. Set the env var in `.env`.

### Every N hours (default)

```bash
# .env
INGEST_INTERVAL_SECONDS=14400   # 4 hours (the default)
```

### Cron-style fixed times — "morning and evening edition"

```bash
# .env
INGEST_SCHEDULE=07:00,18:00     # comma-separated HH:MM
INGEST_TIMEZONE=Europe/London   # any IANA tz; default UTC
```

If both are set, `INGEST_SCHEDULE` wins. The render is still on-demand —
hitting `/digest.pdf` between scheduled runs gives you the cached PDF
instantly.

You can also kick a manual ingest any time:

```bash
curl -X POST http://localhost:8000/ingest
```

## Delivery — push the PDF wherever you want

A built-in hook fires after every successful ingest. Point
`POST_INGEST_HOOK` at any executable on the container's filesystem (drop
the script into your `./data/hooks/` directory so it survives rebuilds via
the bind mount). The hook receives the freshly-built PDF path as its first
argument.

```bash
# .env
POST_INGEST_HOOK=/data/hooks/push-to-boox.sh
POST_INGEST_HOOK_TIMEOUT=300    # optional; default 300s
```

Hook failures are non-fatal — a broken hook logs an error but doesn't
crash the ingest loop.

### Sample: push to a Boox Note over WiFi

Drop this in `./data/hooks/push-to-boox.sh` and `chmod +x` it:

```bash
#!/usr/bin/env bash
# Push the latest issue to a Boox Note via SSH.
# Usage: push-to-boox.sh <pdf-path>
set -euo pipefail

PDF="$1"
BOOX="root@10.11.99.1"            # adjust to your device's IP
SSH_KEY=/data/hooks/boox_id_ed25519

scp -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
    "$PDF" "$BOOX:/sdcard/papernews.pdf"

# Refresh the UI so the file appears immediately.
ssh -i "$SSH_KEY" "$BOOX" 'am start -a android.intent.action.VIEW -d file:///sdcard/papernews.pdf -t application/pdf'
```

Generate a passwordless key (`ssh-keygen -t ed25519 -f
data/hooks/boox_id_ed25519 -N ""`), add the `.pub` to the
Boox Note's `/data/data/com.termux/files/home/.ssh/authorized_keys` once, and from then on
every ingest pushes the new paper to your device.

The same pattern works for Kindle (`scp` over USB networking), a network
printer (`lp -d papernews "$PDF"`), an email (`mutt -a "$PDF"`), or
anything else you can script.

## Local development

TODO add more content here.

## Customizing the typography

Everything visual lives in one file: [`papernews/template.typ.j2`](papernews/template.typ.j2).

- Page size: `width: 203mm, height: 270mm` (tuned for Boox Note Max)
- Page 1 is a broadsheet cover: blackletter nameplate (Chomsky, OFL,
  vendored in `papernews/fonts/`), dateline, lead story above a fold
  rule, a snippet grid of secondary stories below it, and a bottom
  strip with the quote of the day, world news, and "Did you know…"
- Kickers/labels set in Libre Franklin (OFL, vendored); body in
  New Computer Modern (bundled with the Typst compiler)
- Two-column body for any article over 2000 characters; single-column
  otherwise
- First-line paragraph indent instead of vertical parskip (classic
  magazine convention)
- All vertical spacing derives from one rhythm unit (`u`)

Customize whatever you like — the Jinja delimiters are Typst-safe
(`((* ... *))` for blocks, `((( ... )))` for variables) so your `#`, `{`, and `}` don't fight each other.

## Contributing

Go contribute to the real thing (https://github.com/marcj/papernews), this is a fork so I'm not polluting the original with vibe-coded nonesense.

## License

MIT — see [LICENSE](LICENSE).
