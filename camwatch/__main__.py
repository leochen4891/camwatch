"""camwatch entrypoint.

Usage:
  python -m camwatch serve [--host H]  # FastAPI web UI + always-on capture
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m camwatch")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument(
        "--profile",
        action="store_true",
        help="Log per-stage capture-loop timings every 30s (yolo, roi, "
             "recorder, preview, crossing). Use during traffic to find "
             "what's eating per-frame budget.",
    )
    args = parser.parse_args()

    if args.cmd == "serve":
        from camwatch.server import serve
        serve(host=args.host, port=args.port, profile=args.profile)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
