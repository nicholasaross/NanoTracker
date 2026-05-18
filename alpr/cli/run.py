"""Run ALPR pipelines over a pulled session's main snaps.

  python -m alpr.cli.run <session_dir> [--pipeline both|bespoke|preferred]
                                       [--ablation]
                                       [--bespoke-model PATH]
                                       [--ocr-model NAME]
                                       [--detector-model NAME]
                                       [--gpu]
                                       [--limit N]

Writes <session>_alpr.json and <session>_alpr_by_track.json into session_dir.
Plate crops land under session_dir/alpr_crops/<pipeline>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from alpr.pipelines.base import (
    atomic_write_text,
    parse_snap_filename,
)
from alpr.pipelines.runner import PipelineRunner


DEFAULT_BESPOKE_MODEL = Path("alpr/models/license_plate_detector.pt")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--pipeline", choices=("both", "bespoke", "preferred"), default="both")
    ap.add_argument("--ablation", action="store_true",
                    help="Also run bespoke detector + fast-plate-ocr OCR.")
    ap.add_argument("--bespoke-model", type=Path, default=DEFAULT_BESPOKE_MODEL)
    ap.add_argument("--detector-model", default="yolo-v9-t-384-license-plate-end2end",
                    help="open-image-models alias for the preferred pipeline detector.")
    ap.add_argument("--ocr-model", default="global-plates-mobile-vit-v2-model",
                    help="fast-plate-ocr model alias for the preferred pipeline OCR.")
    ap.add_argument("--gpu", action="store_true", help="Pass gpu=True to EasyOCR.")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N images (0 = all).")
    args = ap.parse_args(argv)

    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2

    snaps = _discover_snaps(session_dir)
    if not snaps:
        print(f"no vehicle_*_main_*.jpg snaps in {session_dir}", file=sys.stderr)
        return 1
    if args.limit > 0:
        snaps = snaps[: args.limit]
    print(f"[alpr] {len(snaps)} vehicle snaps")

    pipelines = _build_pipelines(args)
    if not pipelines:
        print("no pipelines selected", file=sys.stderr)
        return 2

    all_records: list[dict] = []
    crops_root = session_dir / "alpr_crops"
    for runner in pipelines:
        print(f"[alpr] running pipeline: {runner.name}")
        crop_dir = crops_root / runner.name
        for i, (image_path, tid, snap_index, cls) in enumerate(snaps, 1):
            result = runner.run(image_path, tid, snap_index, cls, crop_dir)
            all_records.append(result.to_json())
            if i % 10 == 0 or i == len(snaps):
                print(f"  [{runner.name}] {i}/{len(snaps)} done")
            if result.error:
                print(f"  [{runner.name}] {image_path.name}: ERROR {result.error}")

    session_label = _session_label(session_dir)
    out_path = session_dir / f"{session_label}_alpr.json"
    atomic_write_text(out_path, json.dumps(all_records, indent=2))
    print(f"[alpr] wrote {out_path}")

    rollup = _rollup_by_track(all_records)
    rollup_path = session_dir / f"{session_label}_alpr_by_track.json"
    atomic_write_text(rollup_path, json.dumps(rollup, indent=2))
    print(f"[alpr] wrote {rollup_path}")
    return 0


def _discover_snaps(session_dir: Path) -> list[tuple[Path, int, int, str]]:
    """Return [(path, track_id, snap_index, class_name)] for vehicle main snaps only."""
    out = []
    for p in sorted(session_dir.glob("*_main_*.jpg")):
        parsed = parse_snap_filename(p.name)
        if not parsed:
            continue
        cls, tid, n = parsed
        if cls != "vehicle":
            continue
        out.append((p, tid, n, cls))
    return out


def _build_pipelines(args) -> list[PipelineRunner]:
    pipelines: list[PipelineRunner] = []
    bespoke_model = args.bespoke_model
    if not bespoke_model.is_absolute():
        # Resolve relative to CWD so users running from repo root get the expected path.
        bespoke_model = Path.cwd() / bespoke_model

    if args.pipeline in ("both", "bespoke") or args.ablation:
        # Bespoke detector is shared between the main bespoke pipeline and the ablation.
        from alpr.pipelines.bespoke import BespokeDetector, EasyOcrRecognizer
        bespoke_det = BespokeDetector(bespoke_model)
        if args.pipeline in ("both", "bespoke"):
            pipelines.append(PipelineRunner(
                name="bespoke",
                detector=bespoke_det,
                recognizer=EasyOcrRecognizer(use_gpu=args.gpu),
            ))

    if args.pipeline in ("both", "preferred") or args.ablation:
        from alpr.pipelines.preferred import OpenImageModelsDetector, FastPlateOcrRecognizer
        oim_det = OpenImageModelsDetector(args.detector_model)
        if args.pipeline in ("both", "preferred"):
            pipelines.append(PipelineRunner(
                name="preferred",
                detector=oim_det,
                recognizer=FastPlateOcrRecognizer(args.ocr_model),
            ))

    if args.ablation:
        # Bespoke detector + fast-plate-ocr OCR.  Isolates whether the bespoke detector
        # or the EasyOCR head is the weak link in the bespoke pipeline.
        from alpr.pipelines.preferred import FastPlateOcrRecognizer
        pipelines.append(PipelineRunner(
            name="ablation_bespokedet_fastocr",
            detector=bespoke_det,
            recognizer=FastPlateOcrRecognizer(args.ocr_model),
        ))

    return pipelines


def _session_label(session_dir: Path) -> str:
    """Mirror the session-naming convention used by nano_tracker: dir name is session_YYYYMMDD_HHMMSS."""
    return session_dir.name


def _rollup_by_track(records: list[dict]) -> dict:
    """Per-track best-of-N: for each (pipeline, track_id), pick the read with highest ocr_conf."""
    by_pipe_track: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in records:
        p = r["pipeline"]
        tid = r["track_id"]
        if not r.get("ocr_text"):
            continue
        cur_best = by_pipe_track[p].get(tid)
        if cur_best is None or (r.get("ocr_conf") or 0) > (cur_best.get("ocr_conf") or 0):
            by_pipe_track[p][tid] = {
                "track_id": tid,
                "snap_index": r["snap_index"],
                "image": r["image"],
                "ocr_text": r["ocr_text"],
                "ocr_conf": r.get("ocr_conf"),
                "det_conf": r.get("det_conf"),
            }

    # Reshape: one record per track_id, with best_<pipeline> = {...}
    tracks: dict[int, dict] = {}
    for pipe, by_tid in by_pipe_track.items():
        for tid, best in by_tid.items():
            tracks.setdefault(tid, {"track_id": tid})[f"best_{pipe}"] = best

    return {"tracks": sorted(tracks.values(), key=lambda r: r["track_id"])}


if __name__ == "__main__":
    sys.exit(main())
