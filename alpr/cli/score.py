"""Score ALPR pipelines against user-provided labels.

  python -m alpr.cli.score <session_dir>

Writes <session>_alpr_scores.json and prints a human-readable summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alpr.eval.metrics import (
    char_confusion_table,
    cross_pipeline_agreement,
    label_based_accuracy,
    per_pipeline_detection_rate,
    per_pipeline_read_rate,
    per_track_best_of_n,
)
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
        cands = list(session_dir.glob("*_alpr.json"))
        if not cands:
            print(f"no *_alpr.json in {session_dir}; run alpr.cli.run first", file=sys.stderr)
            return 1
        alpr_path = cands[0]
    records = json.loads(alpr_path.read_text(encoding="utf-8"))

    labels_path = session_dir / f"{label}_alpr_labels.json"
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

    scores = {
        "session": session_dir.name,
        "n_records": len(records),
        "n_labels": len(labels),
        "detection_rate": per_pipeline_detection_rate(records),
        "read_rate": per_pipeline_read_rate(records),
        "agreement": cross_pipeline_agreement(records),
        "label_based": label_based_accuracy(records, labels) if labels else {},
        "per_track_best_of_n": per_track_best_of_n(records, labels) if labels else {},
        "char_confusions": char_confusion_table(records, labels) if labels else {},
    }

    out_path = session_dir / f"{label}_alpr_scores.json"
    atomic_write_text(out_path, json.dumps(scores, indent=2))
    print(f"[alpr-score] wrote {out_path}")

    _print_summary(scores)
    return 0


def _print_summary(scores: dict) -> None:
    print()
    print("=" * 64)
    print(f"ALPR comparison — {scores['session']}")
    print("=" * 64)
    print(f"records: {scores['n_records']}  labels: {scores['n_labels']}")

    print("\nDetection rate (per pipeline):")
    for p, v in sorted(scores["detection_rate"].items()):
        print(f"  {p:>34}: {v * 100:5.1f}%")

    print("\nRead rate (per pipeline):")
    for p, v in sorted(scores["read_rate"].items()):
        print(f"  {p:>34}: {v * 100:5.1f}%")

    if scores.get("label_based"):
        print("\nAccuracy vs labels:")
        print(f"  {'pipeline':>34}  {'n':>4}  {'exact':>7}  {'canon':>7}  {'mean ED':>8}")
        for p, s in sorted(scores["label_based"].items()):
            print(f"  {p:>34}  {s['labeled_n']:>4}  "
                  f"{s['exact_accuracy']*100:>6.1f}%  "
                  f"{s['canonical_accuracy']*100:>6.1f}%  "
                  f"{s['mean_edit_distance']:>8.2f}")
    else:
        print("\n(no labels — run `python -m alpr.cli.label` to enable accuracy metrics)")

    if scores.get("per_track_best_of_n"):
        print("\nPer-track best-of-N accuracy:")
        for p, s in sorted(scores["per_track_best_of_n"].items()):
            print(f"  {p:>34}: {s['best_of_n_accuracy']*100:5.1f}% over {s['labeled_tracks']} tracks")

    agreement = scores.get("agreement", {})
    if agreement:
        print("\nCross-pipeline agreement:")
        for k, v in sorted(agreement.items()):
            if k.endswith("_exact"):
                pair = k[:-6]
                n = agreement.get(f"{pair}_overlap_n", 0)
                canon = agreement.get(f"{pair}_canon", 0)
                print(f"  {pair:>34}: exact={v*100:5.1f}%  canon={canon*100:5.1f}%  (n={n})")

    if scores.get("char_confusions"):
        print("\nTop character confusions (truth -> ocr) per pipeline:")
        for p, items in sorted(scores["char_confusions"].items()):
            print(f"  {p}:")
            for truth, ocr, n in items:
                print(f"    {truth} -> {ocr}  x{n}")
    print()


if __name__ == "__main__":
    sys.exit(main())
