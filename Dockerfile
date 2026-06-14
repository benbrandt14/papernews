# papernews container: TeX Live + Python + rmapi + the project
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages: typst, poppler for previews, Python 3, and a few build-essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils \
        poppler-utils \
        python3 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Typst
RUN curl -sL https://github.com/typst/typst/releases/download/v0.11.1/typst-x86_64-unknown-linux-musl.tar.xz \
    | tar -xJ --strip-components=1 -C /usr/local/bin typst-x86_64-unknown-linux-musl/typst

# Install uv directly from astral-sh
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies using uv (caches dependencies layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy the rest of the project and install it
COPY papernews ./papernews
COPY sources.toml ./
RUN uv sync --frozen

# State + cache live on a mounted volume.
RUN mkdir -p /data/archive/cache
ENV PAPERNEWS_STATE=/data/state.db
ENV PAPERNEWS_CONFIG=/app/sources.toml
ENV PAPERNEWS_CACHE=/data/archive/cache

EXPOSE 8000

# Use uv run to execute gunicorn within the environment.
# --reload added so local code modifications restart the worker automatically.
CMD ["uv", "run", "gunicorn", \
     "--reload", \
     "--workers", "1", \
     "--threads", "8", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "900", \
     "--graceful-timeout", "30", \
     "papernews.web:app"]