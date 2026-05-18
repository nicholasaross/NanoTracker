"""Render <session>_alpr_report.html from a session directory's ALPR sidecar files.

  python -m alpr.cli.report <session_dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from alpr.eval.report import render_html
from alpr.pipelines.base import atomic_write_text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    args = ap.parse_args(argv)

    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2

    label = session_dir.name
    alpr_path = session_dir / f"{label}_alpr.json"
    if not alpr_path.exists():
        # Fall back to glob in case session_dir name doesn't match the embedded label.
        candidates = list(session_dir.glob("*_alpr.json"))
        if not candidates:
            print(f"no *_alpr.json in {session_dir}; run `python -m alpr.cli.run {session_dir}` first",
                  file=sys.stderr)
            return 1
        alpr_path = candidates[0]
    records = json.loads(alpr_path.read_text(encoding="utf-8"))

    labels_path = session_dir / f"{label}_alpr_labels.json"
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        # fallback glob
        cands = list(session_dir.glob("*_alpr_labels.json"))
        if cands:
            labels = json.loads(cands[0].read_text(encoding="utf-8"))

    html = render_html(
        session_dir=session_dir,
        records=records,
        labels=labels,
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
    )
    out_path = session_dir / f"{label}_alpr_report.html"
    atomic_write_text(out_path, html)
    print(f"[alpr-report] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
