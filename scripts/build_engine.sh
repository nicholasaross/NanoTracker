#!/usr/bin/env bash
# Build a TensorRT FP16 engine from an ONNX file ON THE NANO.
#
# TRT engines are NOT portable across GPU architectures.  An engine built
# on an RTX 3080 (Ampere sm_86) won't load on a Jetson Nano (Maxwell sm_53),
# so this script must run on the Nano itself.
#
# Usage:
#   ./scripts/build_engine.sh yolov8n.onnx                 # FP16, 1024 MB workspace
#   ./scripts/build_engine.sh yolov8n.onnx yolov8n.engine  # custom output name
#   WORKSPACE_MB=512 ./scripts/build_engine.sh yolov8n.onnx
#
# FP16 is the right precision for Maxwell:
#   - FP32 is correct but slower
#   - INT8 would be faster but Maxwell's INT8 throughput is unimpressive
#     and needs a calibration dataset (see Nvidia's calibration cache docs)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <input.onnx> [output.engine]" >&2
    exit 2
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.onnx}_fp16.engine}"
WORKSPACE_MB="${WORKSPACE_MB:-1024}"

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [[ ! -x "$TRTEXEC" ]]; then
    # Fall back to PATH lookup
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC="$(command -v trtexec)"
    else
        echo "[build] trtexec not found.  Expected at /usr/src/tensorrt/bin/trtexec." >&2
        echo "        Install with: sudo apt install tensorrt" >&2
        exit 1
    fi
fi

if [[ ! -f "$INPUT" ]]; then
    echo "[build] Input ONNX not found: $INPUT" >&2
    exit 1
fi

echo "[build] trtexec:   $TRTEXEC"
echo "[build] input:     $INPUT"
echo "[build] output:    $OUTPUT"
echo "[build] workspace: ${WORKSPACE_MB} MB"
echo "[build] precision: FP16"

# Note: --workspace is in megabytes on TRT 8.2 (older API).  Newer TRT (8.5+)
# uses --memPoolSize=workspace:NMiB.  JetPack 4.6.1 ships 8.2 so we use the
# older flag.
#
# stdbuf -oL forces line buffering so progress is visible during the
# multi-minute kernel search.  Without it, output is block-buffered through
# the pipeline and we don't see anything until trtexec exits.
echo "[build] Starting trtexec.  First build of YOLOv8n on Maxwell typically takes 8-12 min."
echo "[build] Watch GPU activity in another shell with: sudo jtop"
stdbuf -oL "$TRTEXEC" \
    --onnx="$INPUT" \
    --saveEngine="$OUTPUT" \
    --fp16 \
    --workspace="$WORKSPACE_MB" 2>&1 | tail -n 60

if [[ -f "$OUTPUT" ]]; then
    SIZE_MB="$(du -m "$OUTPUT" | cut -f1)"
    echo ""
    echo "[build] OK: $OUTPUT  (${SIZE_MB} MB)"
else
    echo "[build] FAILED -- no engine produced.  Scroll up for trtexec errors." >&2
    exit 1
fi
