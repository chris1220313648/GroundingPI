#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OUTPUT_PATH="${OUTPUT_PATH:-$PWD/outputs/groundingpi_robotwin_clean50/$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_PATH"
echo "GroundingPI pipeline output: $OUTPUT_PATH"
printf '%s\n' "$$" > "$OUTPUT_PATH/pipeline.pid"
printf '%s\n' training > "$OUTPUT_PATH/status"
trap 'printf "%s\n" failed > "$OUTPUT_PATH/status"' ERR
bash scripts/train_groundingpi_robotwin.sh "$@" > "$OUTPUT_PATH/train.log" 2>&1
run="$(find "$OUTPUT_PATH" -mindepth 2 -maxdepth 2 -name config.yaml -printf '%h\n' | sort | tail -1)"
[[ -n "$run" ]]
printf '%s\n' "$run" > "$OUTPUT_PATH/run_path.txt"
"${OPENWAM_ENV:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/openwam-groundingpi}/bin/python" scripts/plot_training_loss.py "$run"
printf '%s\n' evaluating > "$OUTPUT_PATH/status"
bash scripts/eval_groundingpi_robotwin.sh "$run"
printf '%s\n' complete > "$OUTPUT_PATH/status"
