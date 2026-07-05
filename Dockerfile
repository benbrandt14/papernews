# papernews container: Python + uv + the project.
# Typst compilation uses the `typst` Python wheel; poppler renders previews.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        poppler-utils \
        python3 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly from astral-sh
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies using uv (caches dependencies layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the project and install it
COPY papernews ./papernews
COPY sources.toml ./
RUN uv sync --frozen --no-dev

# State + output live on a mounted volume.
RUN mkdir -p /data/output
ENV PAPERNEWS_CONFIG=/app/sources.toml
ENV PAPERNEWS_OUTPUT=/data/output
ENV PAPERNEWS_STATE=/data/state.db

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=15s \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uv", "run", "--no-dev", "uvicorn", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "papernews.serve:app"]
