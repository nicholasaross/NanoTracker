#!/usr/bin/env python3
"""Export YOLOv8n .pt -> .onnx on the DEV BOX (not the Nano).

Why: the Jetson Nano can't run ultralytics (Python 3.6, PyTorch 1.10), so
we do the export on a machine with the modern stack and ship the ONNX file
to the Nano.  On the Nano, scripts/build_engine.sh turns that ONNX into a
TensorRT .engine.

Install on dev box (Python 3.9+ recommended):

    pip install "ultralytics>=8.0,<9.0" onnx onnxsim

Run:

    python scripts/export_onnx.py                       # YOLOv8n, 640px
    python scripts/export_onnx.py --model yolov8s.pt    # other variant
    python scripts/export_onnx.py --imgsz 416           # smaller for more FPS

The exported file lands in the current directory as ``<stem>.onnx``.
Transfer it to the Nano alongside the build_engine.sh script.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 PyTorch weights to ONNX for TRT engine builds")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 weights file (downloads from Ultralytics if not local). Default: yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (square). Default: 640")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset. 13 is the sweet spot for TRT 8.2 on JetPack 4.6. Default: 13")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Run onnx-simplifier (default on)")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        sys.exit("ultralytics not installed.  Run: pip install 'ultralytics>=8.0,<9.0' onnx onnxsim")

    print("[export] Loading model: {}".format(args.model))
    model = YOLO(args.model)
    print("[export] Exporting to ONNX (imgsz={}, opset={}, simplify={}) ...".format(
        args.imgsz, args.opset, args.simplify,
    ))
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
    )
    out_path = Path(out)
    print("[export] Done: {}  ({:.1f} MB)".format(out_path, out_path.stat().st_size / 1e6))
    print("\nNext: scp this file to the Nano and run scripts/build_engine.sh on the Nano.")


if __name__ == "__main__":
    main()
