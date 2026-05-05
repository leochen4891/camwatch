"""camwatch entrypoint.

Usage:
  python -m camwatch                   # headless live mode (writes events.jsonl)
  python -m camwatch serve [--host H]  # FastAPI web UI + always-on capture
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        parser = argparse.ArgumentParser(prog="python -m camwatch serve")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args(sys.argv[2:])
        from camwatch.server import serve
        serve(host=args.host, port=args.port)
        return 0

    from camwatch.main import main as run_headless
    run_headless()
    return 0


if __name__ == "__main__":
    sys.exit(main())
