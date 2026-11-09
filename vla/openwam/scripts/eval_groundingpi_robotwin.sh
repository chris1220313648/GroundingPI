#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run="${1:?usage: eval_groundingpi_robotwin.sh RUN_DIR [TASK ...]}"
shift
run="$(realpath "$run")"
export OPENWAM_ENV="${OPENWAM_ENV:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/openwam-groundingpi}"
export PATH="$OPENWAM_ENV/bin:$PATH"
export HF_HOME="${HF_HOME:-/share/project/robot-rfm/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/share/project/robot-rfm/cache/modelscope}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# SSH X11 forwarding can hang SAPIEN initialization on unattended jobs.
unset DISPLAY
if [[ -n "${ROBOTWIN_DISPLAY:-}" ]]; then export DISPLAY="$ROBOTWIN_DISPLAY"; fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/share/project/robot-rfm/robot-exp/usr/cjy/simulation/RoboTwin}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/robotwin/bin/python}"
export DISPATCHER_PYTHON="$OPENWAM_ENV/bin/python"
export ROBOTWIN_TEST_NUM="${TEST_NUM:-20}"
export NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
export PORT_BASE="${PORT_BASE:-18848}"
ckpt="$(python - "$run" <<'PY'
import sys
from pathlib import Path
files=list(Path(sys.argv[1]).glob('checkpoint_step_*.safetensors'))
if not files: raise SystemExit('No checkpoint found')
print(max(files,key=lambda p:int(p.stem.split('_')[-1])).name)
PY
)"
export ROBOTWIN_LOG_ROOT="${ROBOTWIN_LOG_ROOT:-$run/eval/${ckpt%.safetensors}/$(date -u +%Y%m%d_%H%M%S)}"
export LOG_DIR="$ROBOTWIN_LOG_ROOT/servers"
export SIM_SAVE_VIDEO="${SIM_SAVE_VIDEO:-0}"
export ROBOTWIN_EVAL_RESULT_ROOT="$ROBOTWIN_LOG_ROOT/simulator"
mkdir -p "$LOG_DIR"
printf 'POLICY_NAME=groundingpi\nTASK_CONFIG=%s\nTEST_NUM=%s\nBASE_SEED=0\nNUM_WORKERS=%s\nCHECKPOINT=%s\n' "${EVAL_MODE:-all}" "$ROBOTWIN_TEST_NUM" "$NUM_GPUS" "$ckpt" > "$ROBOTWIN_LOG_ROOT/checkpoint.env"
cp "$run/config.yaml" "$ROBOTWIN_LOG_ROOT/training_config.yaml"
unset ROBOTWIN_COT_PROMPT_FILE
python - "$run/config.yaml" "$ROBOTWIN_LOG_ROOT/prompt.json" <<'PYPROMPT'
import json, sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
if cfg['dataloader']['type'] == 'robotwin_cot':
    from openwam.dataloader.robotwin_cot import coordinate_setup
    manifest = json.loads((Path(cfg['dataloader']['prepared_dir']) / 'manifest.json').read_text())
    Path(sys.argv[2]).write_text(json.dumps({'coordinate_setup': coordinate_setup(cfg['dataloader'].get('image_input_mode', 'composite')), 'request': manifest['request']}, indent=2) + '\n')
PYPROMPT
if [[ -f "$ROBOTWIN_LOG_ROOT/prompt.json" ]]; then
    export ROBOTWIN_COT_PROMPT_FILE="$ROBOTWIN_LOG_ROOT/prompt.json"
fi
python - "$PORT_BASE" "$NUM_GPUS" "${DISPATCH_PORT:-18790}" <<'PYPORT'
import socket,sys
ports=list(range(int(sys.argv[1]),int(sys.argv[1])+int(sys.argv[2])))+[int(sys.argv[3])]
for port in ports:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: s.bind(('127.0.0.1',port))
        except OSError: raise SystemExit(f'Port {port} is occupied; choose another PORT_BASE/DISPATCH_PORT')
PYPORT
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -- -"$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
# Servers see the parent's visible devices. Simulator workers receive the same physical mapping.
export ROBOTWIN_VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
setsid bash scripts/deploy_multi.sh "$run" --ckpt-name "$ckpt" > "$LOG_DIR/launcher.log" 2>&1 &
pids+=("$!")
python - "$PORT_BASE" "$NUM_GPUS" <<'PY'
import socket,sys,time
for port in range(int(sys.argv[1]),int(sys.argv[1])+int(sys.argv[2])):
    deadline=time.monotonic()+900
    while True:
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=2): break
        except OSError:
            if time.monotonic()>deadline: raise SystemExit(f'Server {port} did not start; inspect server logs')
            time.sleep(2)
PY
if (( $# == 0 )); then set -- all; fi
setsid bash benchmarks/robotwin/parallel_eval.sh -m "${EVAL_MODE:-all}" -n groundingpi \
  -w "$NUM_GPUS" --port "$PORT_BASE" --dispatch-port "${DISPATCH_PORT:-18790}" \
  --test-num "$ROBOTWIN_TEST_NUM" --seed 0 --no-dup "$@" > "$ROBOTWIN_LOG_ROOT/evaluation.log" 2>&1 &
pids+=("$!")
rc=0
wait "${pids[-1]}" || rc=$?
python benchmarks/robotwin/export_results_csv.py "$ROBOTWIN_LOG_ROOT" --strict || rc=$?
python scripts/summarize_robotwin_eval.py "$ROBOTWIN_LOG_ROOT" || rc=$?
exit "$rc"
