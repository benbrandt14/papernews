# papernews container: TeX Live + Python + rmapi + the project
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages: typst, poppler for previews, Python 3, and a few build-essentials trafilatura wants.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils \
        poppler-utils \
        python3 python3-pip python3-venv \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://github.com/typst/typst/releases/download/v0.11.1/typst-x86_64-unknown-linux-musl.tar.xz \
    | tar -xJ --strip-components=1 -C /usr/local/bin typst-x86_64-unknown-linux-musl/typst

WORKDIR /app

# Install the project into a venv so we don't fight with Debian's PEP 668 lock.
COPY pyproject.toml ./
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir \
        requests feedparser trafilatura jinja2 \
        flask apscheduler gunicorn \
        google-genai httpx

COPY papernews ./papernews
COPY sources.toml ./
RUN /opt/venv/bin/pip install --no-cache-dir -e .

# State + cache live on a mounted volume.
RUN mkdir -p /data/archive/cache
ENV PAPERNEWS_STATE=/data/state.db
ENV PAPERNEWS_CONFIG=/app/sources.toml
ENV PAPERNEWS_CACHE=/data/archive/cache

EXPOSE 8000

# Use gunicorn with one worker; APScheduler runs in-process, multiple workers
# would multiply ingest runs.
CMD ["/opt/venv/bin/gunicorn", \
     "--workers", "1", \
     "--threads", "8", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "900", \
     "--graceful-timeout", "30", \
     "papernews.web:app"]
