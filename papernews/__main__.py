"""Command-line entry point: `papernews run` / `papernews serve`."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="papernews")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the pipeline once and build the PDF")
    run_p.add_argument(
        "--config",
        default=os.environ.get("PAPERNEWS_CONFIG", "sources.toml"),
        help="path to sources.toml",
    )

    serve_p = sub.add_parser("serve", help="start the web server + scheduler")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "run":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: {config_path.absolute()} not found.", file=sys.stderr)
            return 1
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        print(f"Loaded config with {len(config.get('source', []))} sources.")

        from papernews.core.main import run_papernews

        pdf_path = run_papernews(source_config=config)
        print(f"Built {pdf_path}")
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("papernews.serve:app", host=args.host, port=args.port)
        return 0

    return 2  # unreachable; argparse enforces the subcommand


if __name__ == "__main__":
    raise SystemExit(main())
