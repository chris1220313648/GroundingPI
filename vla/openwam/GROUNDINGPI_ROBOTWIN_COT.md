# GroundingPI: EEF actions with auxiliary target grounding CE

Run from `vla/openwam`:

```bash
bash scripts/train_groundingpi_robotwin_cot.sh
```

The script validates the complete clean delivery, runs tests and real-model
gradient/causality checks, trains four smoke optimizer updates on eight GPUs,
restores that checkpoint for one more update, and starts fresh production
training only after these checks pass. Default scale is per-GPU batch 16,
accumulation 2, global batch 256, ten epochs. VLM LR is 1e-5; action and bridge
LRs are 1e-4. Output contains the pipeline PID, status, logs, configuration,
environment, data manifest, source snapshot, smoke reports, and production run.
Smoke model/optimizer artifacts are removed after successful validation.

`DATASET_DIR` defaults to the `20261029_v1/datasets/groundingpi_target_only`
delivery. `RAW_EEF_ROOT` defaults to the original RoboTwin2.0 dataset. The
preflight checks every parquet row against the original HDF5 joint sequence:
state[t] equals raw q[t], action[t] equals raw q[t+1]. It then uses raw EEF[t]
as proprioception and EEF[t+1:t+33] as action. The 20D order is left xyz, left
rot6d, left gripper, right xyz, right rot6d, right gripper. End-of-episode
padding is excluded from action loss. Min-max normalization preserves rot6d
dimensions and saves the existing deploy-compatible `eef` stats format.

Images come from the delivery's AV1 videos, using per-episode video offsets.
They retain the old 384x320 three-camera layout. **Grounding answers retain
their original 0–999 head-camera coordinates.** Prompts explicitly describe
coordinates relative to the top head-camera subimage. Boxes are never painted
on the RGB image, and geometric image augmentation is disabled. Video decoding
can differ in pixels from the original HDF5 images.

The model sees the annotation-bound `cot_primary_instruction`, coordinate
setup and target-location request. Only available answers with complete target
coverage receive assistant-token CE (including the end token). Other frames
still train actions. These are unapproved candidate annotations, recorded as
such in the manifest; `train_export_ready=false` is not an automatic training
gate. QC metadata and future frames never become policy inputs.

## Three separate camera inputs

`bash scripts/train_groundingpi_robotwin_separate.sh` starts a fresh 10-epoch
experiment using the same prepared data. It sets `image_input_mode=separate`,
resizes each RGB image to 320x240 (BILINEAR), and passes head, left wrist, right
wrist as three images in one VLM message. The sample field `observation_images`
contains the ordered views; `video` retains only the head frame for legacy
shape plumbing. No composed image is fed to the VLM in this mode.

The prompt describes the first image as the head camera; existing 0-999 boxes
remain relative to that image. All three views share training-only brightness,
contrast and saturation factors in [0.8, 1.2], with no geometric augmentation.
Static windows use the original normalized first-action minus state threshold
1e-5: try up to three random replacement indices, retain the original if all
are static. Replacement moves the entire sample, including its instruction and
grounding labels. This does not delete all static frames.

Deployment reads the saved view mode and image dimensions, requires all three
cameras, uses the same ordering and resizing, and applies no color jitter.
The evaluation launcher selects the matching head-camera coordinate prompt.
Legacy checkpoints default to `composite` and retain the old image layout.
Global batch 256, both loss weights 1, EEF targets and padding masks are unchanged.

Total loss is action flow matching plus grounding CE, both weight 1. The VLM
runs once with causal attention. Action conditioning masks all assistant answer
tokens; inference uses just the observation/prompt prefix, including the
assistant generation header. Deployment callers must use the same task and
subimage-coordinate prompt. The saved backbone option
`grounding_generation_prompt=true` preserves the matching chat header.

`training.lambda_grounding` defaults to zero for existing experiments. This
script sets it to one and starts from the original pretrained VLM, with fresh
action/bridge weights. It does not resume the previous clean50 model.

Optional environment overrides: `OUTPUT_PATH`, `PREPARED_DIR`, `EPOCHS`,
`BATCH_SIZE`, `GLOBAL_BATCH`, `SAVE_STEPS`, `MASTER_PORT`. A prepared directory
may be reused only for the same immutable delivery; its manifest is written
only after the full validation passes. No simulation evaluation or external
experiment publication is performed by this script.
