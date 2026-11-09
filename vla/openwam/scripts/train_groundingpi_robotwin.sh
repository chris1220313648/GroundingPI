#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENWAM_ENV="${OPENWAM_ENV:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/openwam-groundingpi}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$OPENWAM_ENV/bin:$CUDA_HOME/bin:$PATH"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
export NNODES="${NNODES:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}" GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
export EPOCHS="${EPOCHS:-10}" SAVE_STEPS="${SAVE_STEPS:-2000}"
export DATASET_DIR="${DATASET_DIR:-/share/project/robot-rfm/robot-raw-data/RoboTwin2.0/dataset}"
export OUTPUT_PATH="${OUTPUT_PATH:-$PWD/outputs/groundingpi_robotwin_clean50}"
export WANDB_MODE="${WANDB_MODE:-offline}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_PATH}"
mkdir -p "$OUTPUT_PATH"
python scripts/preflight_groundingpi_data.py "$DATASET_DIR" "$OUTPUT_PATH/data_manifest.json"
python -m pip freeze > "$OUTPUT_PATH/environment.txt"
git rev-parse HEAD > "$OUTPUT_PATH/git_revision.txt"
git diff -- . > "$OUTPUT_PATH/source_changes.patch"
tar --exclude=__pycache__ -czf "$OUTPUT_PATH/source_snapshot.tar.gz" configs openwam scripts benchmarks LICENSE README.md GROUNDINGPI_ROBOTWIN.md
exec bash scripts/train_fixed16.sh groundingpi \
  training.log_metrics_csv=true training.retain_resume_state=true \
  training.keep_last_k_ckpts=1 training.dataset_num_workers=4 "$@"
