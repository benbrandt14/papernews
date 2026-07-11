"""Command-line entry point: `papernews run` / `papernews serve`."""

from __future__ import annotations

import argparse
import os
import sys
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

    sub.add_parser("providers", help="list the built-in LLM provider presets")
    sub.add_parser(
        "check-llm", help="probe the configured LLM provider for connectivity"
    )

    args = parser.parse_args(argv)

    if args.command == "providers":
        from papernews.config import get_settings
        from papernews.core.backends import PROVIDERS

        active = get_settings().llm_provider
        for name, p in PROVIDERS.items():
            key = p.key_env or "(no key)"
            mark = " *" if name == active else ""
            print(f"{name:<11} {p.base_url:<40} {key:<20} {p.default_model}{mark}")
        print(
            "\n* = active (PAPERNEWS_LLM_PROVIDER). Override with "
            "PAPERNEWS_LLM_BASE_URL/_API_KEY/_MODEL."
        )
        return 0

    if args.command == "check-llm":
        from papernews.config import get_settings
        from papernews.core.backends import get_backend

        try:
            backend = get_backend(get_settings())
        except ValueError as e:
            print(f"Misconfigured: {e}", file=sys.stderr)
            return 1
        ok, detail = backend.check()
        print(("OK: " if ok else "FAILED: ") + detail)
        return 0 if ok else 1

    if args.command == "run":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: {config_path.absolute()} not found.", file=sys.stderr)
            return 1

        from papernews.config import load_config

        config = load_config(config_path)
        print(f"Loaded config with {len(config.sources)} sources.")

        from papernews.core.main import run_papernews

        pdf_path = run_papernews(config=config)
        print(f"Built {pdf_path}")
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("papernews.serve:app", host=args.host, port=args.port)
        return 0

    return 2  # unreachable; argparse enforces the subcommand


if __name__ == "__main__":
    raise SystemExit(main())
