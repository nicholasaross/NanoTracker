"""StreetTracker CLI entry point.

Subcommands:
- ``streettracker run`` — live RTSP (Orin only)         [phase 3]
- ``streettracker batch <video>`` — file input          [phase 5]
- ``streettracker pull`` — rsync session from device    [phase 5]
- ``streettracker recolor <session>`` — rerun color heuristic
- ``streettracker debug-color <crop.jpg>`` — inspect a single crop
- ``streettracker export-engine`` — ``.pt`` → ``.engine``  [phase 3]

Each subcommand owns its own ``argparse.ArgumentParser`` in its module's
``main()`` — we dispatch on the first positional rather than using
``add_subparsers`` so subcommand parsers receive ``--help`` cleanly
instead of having the top-level parser intercept it.
"""

from __future__ import annotations

import sys

_HELP = """\
usage: streettracker [--version] <command> [<args>]

commands:
  run             live RTSP tracker (Orin only)               [phase 3]
  batch           batch process a video file                  [phase 5]
  pull            rsync a session from a remote device        [phase 5]
  recolor         rerun color heuristic on a closed session
  debug-color     inspect HSV vote on one or more crop JPEGs
  export-engine   export .pt to .engine via Ultralytics       [phase 3]

Run ``streettracker <command> --help`` for per-command options.
"""


def _print_help() -> None:
    print(_HELP, end="")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _print_help()
        return 0

    head, rest = argv[0], argv[1:]

    if head in ("-h", "--help"):
        _print_help()
        return 0
    if head == "--version":
        from streettracker import __version__
        print(__version__)
        return 0

    if head == "recolor":
        from streettracker.analysis.recolor import main as recolor_main
        return recolor_main(rest)
    if head == "debug-color":
        from streettracker.analysis.debug_color import main as debug_color_main
        return debug_color_main(rest)

    if head in ("run", "batch", "pull", "export-engine"):
        print(
            f"[streettracker] subcommand '{head}' not yet implemented",
            file=sys.stderr,
        )
        return 2

    print(f"[streettracker] unknown subcommand: {head}", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
