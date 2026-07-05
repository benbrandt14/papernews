# papernews

An offline-first, highly customized E-Ink daily digest. A Prefect pipeline pulls feeds via plugins, uses an LLM filter, summarize, and format full text, and passes it through an adapter to render one consistently typeset Typst PDF using jinja.

## Pipeline Stages

1. **Stage 1: Ingestion** — Dynamically loads plugins (RSS, Hacker News, Wiki). Outputs strict Pydantic `RawDocument` models.
2. **Stage 2: Filtering** — Enforces deterministic category limits, local ranking heuristics, and regex blacklists natively in Python (zero API cost).
3. **Stage 3: LLM Handling** — Routes surviving documents to the Gemini API for gatekeeper selection, summarization, and strict markdown formatting. Enforces Pydantic schema validation and tracks token `Telemetry`.
4. **Stage 4: Templating** — Converts Pydantic objects into the dictionaries required by the legacy templating engine, keeping layout logic strictly decoupled.
5. **Stage 5: Render** — Jinja injects the adapted data into a Typst template (`template.typ.j2`), utilizing a regex pipeline (`_stash_typography`) to safely compile LaTeX math, markdown headers, and remote images.

## Configuration (`sources.toml`)

Manage your feeds and categories in `sources.toml`. Order matters: sources appear in this sequence in the generated PDF.

```toml
[[sources]]
name = "Hacker News Top"
url = "" # Handled internally by the plugin
kind = "hn"
category = "Technology"
limit = 5

[[sources]]
name = "Quanta Magazine"
url = "[https://api.quantamagazine.org/feed/](https://api.quantamagazine.org/feed/)"
kind = "rss"
category = "Science"

```

## Plugins & Hacker News (HN) Fetching

Ingestion is decoupled via `pluggy`.

* **RSS Plugin:** Uses `feedparser` and `trafilatura` to extract full-text bodies and images.
* **HN Plugin:** Bypasses standard web scraping and queries the official Algolia Search API (`kind = "hn"`). It pulls the top articles by points within a specific time window to guarantee high-quality curation.
* **Extensibility:** To add a new source (e.g., Academic PDFs), write a new `pluggy` module yielding `RawDocument` objects. As long as it respects the extraction parameters, the pipeline handles the rest.

## Docker & API Structure

**Prerequisites:** Docker and a Gemini API key.

```bash
git clone [https://github.com/benbrandt14/papernews](https://github.com/benbrandt14/papernews)
cd papernews
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
docker compose up --build -d

```

**API Interaction:**
The application exposes a web server at `http://localhost:8000`.

* To trigger a new compilation pipeline run, send a request to the `/ingest` endpoint.
* State (SQLite) and PDF outputs live in `./data/state.db` and `./data/output/` (bind-mounted from the host).
* **Resetting State:** To perform a hard reset, simply delete `data/state.db` and restart the container. The database schema will automatically rebuild on boot.


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
- Body font: New Computer Modern 11pt
- Two-column body for any article over 2000 characters; single-column
  otherwise
- First-line paragraph indent instead of vertical parskip (classic
  magazine convention)
- Letter-spacing on small-caps source labels via `tracking`

Customize whatever you like — the Jinja delimiters are Typst-safe
(`((* ... *))` for blocks, `((( ... )))` for variables) so your `#`, `{`, and `}` don't fight each other.

## Contributing

Go contribute to the real thing (https://github.com/marcj/papernews), this is a fork so I'm not polluting the original with vibe-coded nonesense.

## License

MIT — see [LICENSE](LICENSE).
