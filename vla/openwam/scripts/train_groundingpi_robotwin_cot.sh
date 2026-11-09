#!/usr/bin/env bash
# Full preflight -> real-model checks -> production-size smoke/resume -> fresh training.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENWAM_ENV="${OPENWAM_ENV:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/openwam-groundingpi}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$OPENWAM_ENV/bin:$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}" NNODES=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29639}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_PATH="${OUTPUT_PATH:-$PWD/outputs/groundingpi_robotwin_cot_eef/$(date -u +%Y%m%d_%H%M%S)}"
export WANDB_DIR="$OUTPUT_PATH"
export HF_HOME="${HF_HOME:-/share/project/robot-rfm/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/share/project/robot-rfm/cache/modelscope}"
DATASET_DIR="${DATASET_DIR:-/share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_Lerobotv3_cot/clean/20261029_v1/datasets/groundingpi_target_only}"
RAW_EEF_ROOT="${RAW_EEF_ROOT:-/share/project/robot-rfm/robot-raw-data/RoboTwin2.0/dataset}"
PREPARED_DIR="${PREPARED_DIR:-$OUTPUT_PATH/prepared}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
EPOCHS="${EPOCHS:-10}"
PER_STEP=$((BATCH_SIZE * NPROC_PER_NODE))
(( GLOBAL_BATCH % PER_STEP == 0 )) || { echo "Global batch must divide microbatch x GPUs" >&2; exit 1; }
GRAD_ACCUM=$((GLOBAL_BATCH / PER_STEP))
mkdir -p "$OUTPUT_PATH"
printf '%s\n' "$$" > "$OUTPUT_PATH/pipeline.pid"
trap 'printf "failed\n" > "$OUTPUT_PATH/status"' ERR
. envs/vlm.env
printf 'preflight\n' > "$OUTPUT_PATH/status"
if [[ ! -f "$PREPARED_DIR/manifest.json" ]]; then
    python scripts/prepare_groundingpi_cot.py "$DATASET_DIR" "$RAW_EEF_ROOT" "$PREPARED_DIR" > "$OUTPUT_PATH/preflight.log" 2>&1
fi
cp "$PREPARED_DIR/manifest.json" "$OUTPUT_PATH/data_manifest.json"
python -m pip freeze > "$OUTPUT_PATH/environment.txt"
git rev-parse HEAD > "$OUTPUT_PATH/git_revision.txt"
git diff -- . > "$OUTPUT_PATH/source_changes.patch"
tar --exclude=__pycache__ -czf "$OUTPUT_PATH/source_snapshot.tar.gz" configs openwam scripts tests GROUNDINGPI_ROBOTWIN_COT.md
python -m pytest -q tests/test_groundingpi_backbone.py tests/test_robotwin_cot.py tests/test_robotwin_separate.py tests/test_openwam_trainer.py tests/test_vlm_system_fixed16_pi.py > "$OUTPUT_PATH/tests.log" 2>&1
printf 'model_checks\n' > "$OUTPUT_PATH/status"
python scripts/check_groundingpi_cot.py "$DATASET_DIR" "$PREPARED_DIR" "$OUTPUT_PATH/model_checks.json" --image-input-mode "${IMAGE_INPUT_MODE:-composite}" > "$OUTPUT_PATH/model_checks.log" 2>&1

common=(+experiment=fixed16_comparison model=vlm_system model/vlm_backbone=groundingpi
    +model.vlm_backbone.grounding_generation_prompt=true
    dataloader=robotwin_cot "dataloader.dataset_dir=$DATASET_DIR" "dataloader.prepared_dir=$PREPARED_DIR"
    model.architecture.action_dim=20 model.architecture.state_dim=20
    training.lambda_grounding=1.0 training.log_metrics_csv=true training.retain_resume_state=true
    "training.batch_size=$BATCH_SIZE" "training.gradient_accumulation_steps=$GRAD_ACCUM"
    "training.num_epochs=$EPOCHS" training.keep_last_k_ckpts=1 training.dataset_num_workers=4)
printf 'smoke\n' > "$OUTPUT_PATH/status"
# max_steps is measured in microsteps; four complete optimizer updates.
SMOKE_STEPS=$((4 * GRAD_ACCUM))
bash scripts/train.sh "${common[@]}" "training.max_steps=$SMOKE_STEPS" "training.save_steps=$SMOKE_STEPS" \
    "training.output_path=$OUTPUT_PATH/smoke" "$@" > "$OUTPUT_PATH/smoke.log" 2>&1
smoke_run="$(find "$OUTPUT_PATH/smoke" -name config.yaml -printf '%h\n' | sort | tail -1)"
[[ -n "$smoke_run" ]]
printf 'smoke_resume\n' > "$OUTPUT_PATH/status"
bash scripts/train.sh "${common[@]}" "training.resume_ckpt_path=$smoke_run" \
    "training.max_steps=$((SMOKE_STEPS + GRAD_ACCUM))" "training.save_steps=$GRAD_ACCUM" \
    "training.output_path=$OUTPUT_PATH/smoke" "$@" > "$OUTPUT_PATH/smoke_resume.log" 2>&1
python - "$smoke_run" "$OUTPUT_PATH" <<'PY'
import csv, json, math, shutil, sys
from pathlib import Path
run, output = map(Path, sys.argv[1:])
rows = list(csv.DictReader((run / 'debug_loss_history.csv').open()))
assert int(rows[-1]['opt_step']) >= 5
for row in rows:
    assert all(math.isfinite(float(row[k])) for k in ['loss', 'loss_action', 'loss_grounding', 'grad_norm'])
    assert float(row['loss_grounding']) > 0 and float(row['grounding_tokens']) > 0
assert any(float(r['grad_norm']) > 0 for r in rows)
assert any(run.glob('checkpoint_step_*.safetensors'))
assert any(run.glob('accel_state_step_*'))
shutil.copy2(run / 'debug_loss_history.csv', output / 'smoke_loss_history.csv')
(output / 'smoke_report.json').write_text(json.dumps({'passed': True, 'optimizer_updates': int(rows[-1]['opt_step']),
    'checkpoint_resume_passed': True, 'last_metrics': rows[-1]}, indent=2) + '\n')
# Only remove the smoke directory created by this pipeline; keep its reports and logs.
shutil.rmtree(output / 'smoke')
PY
printf 'training\n' > "$OUTPUT_PATH/status"
bash scripts/train.sh "${common[@]}" "training.save_steps=${SAVE_STEPS:-2000}" \
    "training.output_path=$OUTPUT_PATH/train" "$@" > "$OUTPUT_PATH/train.log" 2>&1 &
train_pid=$!
printf '%s\n' "$train_pid" > "$OUTPUT_PATH/train.pid"
wait "$train_pid"
printf 'complete\n' > "$OUTPUT_PATH/status"
