#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEGMENT="segment-14081240615915270380_4399_000_4419_000_with_camera_labels"
EXPECTED_TIMESTAMPS="1518657495713061,1518657495913095,1518657496113101"
SOURCE_TFRECORD=""
ASSET_ROOT=""
EXAMPLE_INPUTS=""
BLENDER="${BLENDER_PATH:-}"
GPU_ID="0"
MODE="prepare"
RETRY="false"
SKIP_BLENDER_RENDER="false"
OUTPUT_ROOT="${ROOT}/run_outputs/val_140812_clip00"
PYTHON_BIN="${WAYPOP_RENDER_PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: bash reconstruction/run_example.sh \
  --source-tfrecord /path/to/segment-14081240615915270380_4399_000_4419_000_with_camera_labels.tfrecord \
  --asset-root /path/to/objaverse/cache \
  --example-inputs /path/to/val_140812_clip00 [--mode prepare|render] \
  [--blender /path/to/blender] [--gpu-id 0] [--output-root /path/to/output] \
  [--retry] [--skip-blender-render]

The default mode prepares and checks the three-frame validation example.
Use --mode render to run Blender after preparation.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-tfrecord) SOURCE_TFRECORD="${2:-}"; shift 2 ;;
    --asset-root) ASSET_ROOT="${2:-}"; shift 2 ;;
    --example-inputs) EXAMPLE_INPUTS="${2:-}"; shift 2 ;;
    --blender) BLENDER="${2:-}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --retry) RETRY="true"; shift ;;
    --skip-blender-render) SKIP_BLENDER_RENDER="true"; shift ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$SOURCE_TFRECORD" || "$(basename -- "$SOURCE_TFRECORD")" != "${SEGMENT}.tfrecord" ]]; then
  echo "Provide the original Waymo validation TFRecord for ${SEGMENT}." >&2
  exit 2
fi
if [[ ! -d "$ASSET_ROOT" ]]; then
  echo "Provide the Objaverse cache root with --asset-root." >&2
  exit 2
fi
if [[ ! -f "${EXAMPLE_INPUTS}/objects.json" || ! -f "${EXAMPLE_INPUTS}/placement.json" ]]; then
  echo "Provide the separately distributed validation example files with --example-inputs." >&2
  exit 2
fi
if [[ "$MODE" != "prepare" && "$MODE" != "render" ]]; then
  echo "--mode must be prepare or render." >&2
  exit 2
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "--gpu-id must be a nonnegative integer." >&2
  exit 2
fi
if [[ "$SKIP_BLENDER_RENDER" == "true" && "$RETRY" != "true" ]]; then
  echo "--skip-blender-render requires --retry with saved render outputs." >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ "$MODE" == "render" && "$SKIP_BLENDER_RENDER" != "true" ]]; then
  if [[ -z "$BLENDER" ]]; then
    BLENDER="$(command -v blender || true)"
  fi
  if [[ ! -x "$BLENDER" ]]; then
    echo "Provide a Blender executable with --blender or BLENDER_PATH." >&2
    exit 2
  fi
fi

export WAYPOP_OBJAVERSE_ROOT="$(cd -- "$ASSET_ROOT" && pwd)"
OBJECTS_JSON="${EXAMPLE_INPUTS}/objects.json"
PLACEMENT_JSON="${EXAMPLE_INPUTS}/placement.json"
"$PYTHON_BIN" "${ROOT}/scripts/check_example_inputs.py" \
  --objects "$OBJECTS_JSON" --placement "$PLACEMENT_JSON" \
  --segment "$SEGMENT" --count 8

if [[ "$SKIP_BLENDER_RENDER" == "true" ]]; then
  "$PYTHON_BIN" "${ROOT}/scripts/check_retry_state.py" \
    --output-root "$OUTPUT_ROOT" --objects "$OBJECTS_JSON" \
    --segment "$SEGMENT"
fi

INPUT_DIR="${OUTPUT_ROOT}/inputs/waymo"
MINI_TFRECORD="${INPUT_DIR}/${SEGMENT}.tfrecord"
mkdir -p -- "$INPUT_DIR"
"$PYTHON_BIN" "${ROOT}/scripts/create_waymo_mini_tfrecord.py" \
  --source_tfrecord "$SOURCE_TFRECORD" \
  --output_tfrecord "$MINI_TFRECORD" \
  --frame_tokens 2,4,6 \
  --expected_timestamps "$EXPECTED_TIMESTAMPS"

echo "Prepared validation source tokens 2,4,6 at ${MINI_TFRECORD}."
if [[ "$MODE" == "prepare" ]]; then
  exit 0
fi

"$PYTHON_BIN" "${ROOT}/object_sampling.py" \
  --segment "$SEGMENT" \
  --waymo_dir "$INPUT_DIR" \
  --objects_json "$OBJECTS_JSON" \
  --placement_json "$PLACEMENT_JSON" \
  --frame_token_list 2,4,6 \
  --fixed_ood_objects 8 \
  --retry "$RETRY" \
  --skip_blender_render "$SKIP_BLENDER_RENDER" \
  --blender_path "$BLENDER" \
  --gpu_id "$GPU_ID" \
  --save_path "${OUTPUT_ROOT}/outputs" \
  --image_path "${OUTPUT_ROOT}/images" \
  --blender_logs_dir "${OUTPUT_ROOT}/blender_logs"

"$PYTHON_BIN" "${ROOT}/scripts/check_example_outputs.py" \
  --output-root "$OUTPUT_ROOT" \
  --objects "$OBJECTS_JSON" \
  --segment "$SEGMENT"
