"""StreetTracker CLI entry point.

Subcommands:
- `streettracker run` — live RTSP (Orin only)
- `streettracker batch <video>` — file input
- `streettracker pull` — rsync session from device
- `streettracker recolor <session>` — rerun color heuristic
- `streettracker export-engine` — `.pt` → `.engine` via Ultralytics
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streettracker")
    parser.add_argument(
        "--version", action="store_true", help="print version and exit"
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")
    sub.add_parser("run", help="live RTSP tracker (Orin only)")
    sub.add_parser("batch", help="batch process a video file")
    sub.add_parser("pull", help="rsync a session from a remote device")
    sub.add_parser("recolor", help="rerun color heuristic on a closed session")
    sub.add_parser("export-engine", help="export .pt to .engine via Ultralytics")

    args = parser.parse_args(argv)
    if args.version:
        from streettracker import __version__
        print(__version__)
        return 0
    if not args.cmd:
        parser.print_help()
        return 0
    print(f"[streettracker] subcommand '{args.cmd}' not yet implemented", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
