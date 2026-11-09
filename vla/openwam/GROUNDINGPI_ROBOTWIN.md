# GroundingPI on RoboTwin

The `groundingpi` adapter uses the released GroundingPI checkpoint with OpenWAM's
Fixed-16 action expert. It is an added backbone experiment under this repository's
comparison protocol, not a claim that the release already contains trained RoboTwin
policy weights.

## Environment

```bash
bash scripts/setup_groundingpi_env.sh
```

The dedicated `openwam-groundingpi` environment reuses the existing StarVLA Torch
and Transformers 5.7.0 source installation read-only. Extra packages are installed
only in the new environment. These shared paths must remain available. The original
OpenWAM `<5` Transformers requirement is not used for this GroundingPI runtime.

## Train and automatically evaluate

```bash
bash scripts/run_groundingpi_robotwin.sh
```

For a detached server job:

```bash
nohup bash scripts/run_groundingpi_robotwin.sh > pipeline.log 2>&1 &
```

All visible GPUs are used by default. On 8 GPUs, batch 4 per GPU with accumulation
8 gives global batch 256. Set `CUDA_VISIBLE_DEVICES` to restrict devices. Set
`BATCH_SIZE`, `GLOBAL_BATCH`, `OUTPUT_PATH`, `OPENWAM_ENV` if needed. The global
batch must divide exactly by per-GPU batch times world size. Training uses clean_50,
50 tasks/2500 demonstrations, 10 epochs, seed 42, 20D EEF actions, 32-step horizon,
384x320 composed camera image, end-to-end backbone LR 1e-5 and action/bridge LR 1e-4.
The backbone emits eight normalized-depth language hidden states without LM logits.

Train only: `bash scripts/train_groundingpi_robotwin.sh`.
Resume: pass `training.resume_ckpt_path=/absolute/path/to/run` to that script with
the same batch/topology/protocol. `max_steps` and `save_steps` count **microsteps**;
optimizer steps are recorded separately. The periodic save interval defaults to
2000 microsteps. Latest periodic weights/full resume state and final weights are
retained. Only a completed training process triggers automatic evaluation.

## Evaluate

```bash
bash scripts/eval_groundingpi_robotwin.sh /absolute/path/to/run
```

Default: final/latest weights, clean and randomized, 50 tasks each, **20 valid
rollouts per task and mode (2000 total)**, base scene seed 0, upstream step limits,
one environment per task (`--no-dup`). Set `EVAL_MODE`, `TEST_NUM`, `NUM_GPUS`,
`PORT_BASE`, `DISPATCH_PORT` to override. Optional positional task names restrict
the evaluation. `ROBOTWIN_PATH` and `ROBOTWIN_PYTHON` select the simulator checkout
and its separate environment. SSH `DISPLAY` is cleared for unattended headless
rendering; set `ROBOTWIN_DISPLAY` explicitly if a display is required.

A rollout infrastructure exception retries the same committed scene up to three
times; failures are logged and never reported as policy failures. Exhausted workers
stop, and the dispatcher's completion check remains authoritative. Results with
fewer than the target rollouts are incomplete. Expert-invalid scenes are skipped.
Step-limit exhaustion during a normal policy rollout counts as failure.

## Artifacts

`outputs/groundingpi_robotwin_clean50/<launch>/<run>/` holds config, normalization
stats, deploy assets, policy weights and resume state. The launch folder holds
`train.log`, `pipeline.pid`, `status`, `environment.txt`, `data_manifest.json`, and
`run_path.txt`. W&B runs offline by default. `debug_loss_history.csv` is also written
for formal runs and `training_loss.png` is generated after training; no held-out
validation split is introduced by this protocol.

Evaluation logs, server logs, per-episode JSONL, summary TSV and strict CSV export
live under `<run>/eval/<checkpoint>/<timestamp>/`. Raw simulator diagnostics remain
in its log files. The wrapper sets `SIM_SAVE_VIDEO=0` by default and redirects simulator outputs into the evaluation directory.
