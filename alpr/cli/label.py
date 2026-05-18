"""Interactive labeling CLI.  Resumable; saves after every entry.

  python -m alpr.cli.label <session_dir> [--filter unread|disagreement|all]

Default filter is `disagreement` — labeling images where pipelines disagree gives the
most score-discriminating data per minute of human time.

Special label entries:
  - empty / [s]kip: skip this image, do not write any record
  - [u]nreadable: write {"plate": "__UNREADABLE__"} so it's excluded from accuracy
  - [n]o-plate: write {"plate": "__NO_PLATE__"} (no plate visible in image)
  - [q]uit: stop labeling, exit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from alpr.eval.metrics import NO_PLATE, SKIP_VALUES, UNREADABLE
from alpr.pipelines.base import atomic_write_text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--filter", choices=("unread", "disagreement", "all"), default="disagreement")
    args = ap.parse_args(argv)

    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2

    label = session_dir.name
    alpr_path = session_dir / f"{label}_alpr.json"
    if not alpr_path.exists():
        cands = list(session_dir.glob("*_alpr.json"))
        if not cands:
            print(f"no *_alpr.json in {session_dir}; run `python -m alpr.cli.run` first",
                  file=sys.stderr)
            return 1
        alpr_path = cands[0]
    records = json.loads(alpr_path.read_text(encoding="utf-8"))

    labels_path = session_dir / f"{label}_alpr_labels.json"
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

    queue = _select_queue(records, labels, args.filter)
    if not queue:
        print(f"[alpr-label] nothing to label under filter={args.filter!r}.")
        return 0

    print(f"[alpr-label] {len(queue)} images to review (filter={args.filter})")
    print(f"[alpr-label] hints shown per image: pipeline reads.  Type the true plate or:")
    print(f"  [u]nreadable   [n]o-plate   [s]kip   [q]uit")

    for i, (image_name, hints) in enumerate(queue, 1):
        image_path = session_dir / image_name
        if not image_path.exists():
            print(f"[{i}/{len(queue)}] missing file {image_path}, skipping")
            continue

        _open_in_viewer(image_path)
        print(f"\n[{i}/{len(queue)}] {image_name}")
        for p, text in hints.items():
            print(f"   {p:>10}: {text or '—'}")
        try:
            ans = input("plate> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[alpr-label] interrupted; saved progress.")
            return 0

        cmd = ans.lower()
        if cmd in ("q", "quit"):
            print("[alpr-label] quitting.")
            return 0
        if cmd in ("", "s", "skip"):
            continue
        if cmd in ("u", "unreadable"):
            value = UNREADABLE
        elif cmd in ("n", "no-plate", "noplate"):
            value = NO_PLATE
        else:
            value = ans.upper()

        labels[image_name] = {
            "plate": value,
            "labeled_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_text(labels_path, json.dumps(labels, indent=2, sort_keys=True))

    print(f"[alpr-label] done; labels saved to {labels_path}")
    return 0


def _select_queue(records: list[dict], labels: dict, mode: str) -> list[tuple[str, dict]]:
    """Return [(image_name, {pipeline: ocr_text})] ordered by track_id then snap_index."""
    by_image: dict[str, dict[str, str | None]] = defaultdict(dict)
    image_order: dict[str, tuple[int, int]] = {}
    for r in records:
        img = r["image"]
        by_image[img][r["pipeline"]] = r.get("ocr_text")
        image_order.setdefault(img, (r["track_id"], r["snap_index"]))

    queue = []
    for img, hints in by_image.items():
        # Skip already-labeled (any non-empty entry, including sentinels — they're a decision)
        if img in labels:
            existing = labels[img].get("plate")
            if existing and existing not in (None, ""):
                continue

        reads = [t for t in hints.values() if t]
        if mode == "unread":
            if reads:
                continue
        elif mode == "disagreement":
            if not reads or len(set(reads)) <= 1:
                continue
        # mode == "all": include all unlabeled

        queue.append((img, hints))

    queue.sort(key=lambda kv: image_order[kv[0]])
    return queue


def _open_in_viewer(path: Path) -> None:
    """Cross-platform: open image in OS default viewer.

    The viewer pops up; the user reads the plate; types it into our prompt.
    """
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"  (could not open viewer: {e}; path: {path})")


if __name__ == "__main__":
    sys.exit(main())
