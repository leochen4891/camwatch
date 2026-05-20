"""Run the enricher service:

    python -m camwatch_enricher serve [--host 127.0.0.1] [--port 8765]
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .config import load_config
from .server import create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="camwatch_enricher")
    sub = ap.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="run the FastAPI enrichment service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--log-level", default="info")

    args = ap.parse_args(argv)
    if args.cmd != "serve":
        ap.print_help()
        return 1

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    host = args.host or cfg.service.host
    port = args.port or cfg.service.port
    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
