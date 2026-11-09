#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export IMAGE_INPUT_MODE=separate
export PREPARED_DIR="${PREPARED_DIR:-$PWD/outputs/groundingpi_robotwin_cot_eef_prepared_20261029}"
export OUTPUT_PATH="${OUTPUT_PATH:-$PWD/outputs/groundingpi_robotwin_cot_separate/$(date -u +%Y%m%d_%H%M%S)}"
exec bash scripts/train_groundingpi_robotwin_cot.sh \
    dataloader.image_input_mode=separate dataloader.height=240 dataloader.width=320 \
    'dataloader.color_jitter={brightness:0.2,contrast:0.2,saturation:0.2,hue:0.0}' \
    dataloader.filter_static_segments=true dataloader.static_segment_threshold=1e-5 \
    dataloader.max_static_retry=3 "$@"
