#!/usr/bin/env python
"""Check real-model grounding gradients and answer-independent action conditions."""
import argparse
import json
import re
from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.dataloader.robotwin_cot import RoboTwinCoTDataset
from openwam.model import build_architecture, resolve_architecture_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    parser.add_argument('prepared_dir')
    parser.add_argument('report', type=Path)
    parser.add_argument('--image-input-mode', choices=['composite', 'separate'], default='composite')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = RoboTwinCoTDataset(args.prepared_dir, args.dataset_dir,
                                height=240 if args.image_input_mode == 'separate' else 384,
                                image_input_mode=args.image_input_mode)
    index = next(i for i, flag in enumerate(dataset.rows['ce_eligible'].to_pylist()) if flag)
    sample = dataset[index]
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=[
            '+experiment=fixed16_comparison', 'model=vlm_system', 'model/vlm_backbone=groundingpi',
            'model.architecture.action_dim=20', 'model.architecture.state_dim=20',
            '+model.vlm_backbone.grounding_generation_prompt=true'])
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    model.set_dtype_device(model.dtype, torch.device('cuda:0'))
    model.init_training_schedulers(1000)
    model.set_training_runtime(use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
                               max_timestep_boundary=1., min_timestep_boundary=0.)
    model.grounding_supervision = True
    model.eval()
    with torch.no_grad():
        inputs = model.prepare_inputs([sample])
        box = {}
        first = model.encode_conditions(**inputs, grounding_out=box)
        changed_answer = re.sub(r'<(\d+)>', lambda m: f'<{(int(m[1]) + 17) % 1000}>', sample['grounding_answer'])
        changed = dict(sample, grounding_answer=changed_answer)
        changed_inputs = model.prepare_inputs([changed])
        assert torch.equal(inputs['vlm_inputs']['action_prefix_mask'], changed_inputs['vlm_inputs']['action_prefix_mask'])
        assert inputs['vlm_inputs']['input_ids'].shape == changed_inputs['vlm_inputs']['input_ids'].shape
        second = model.encode_conditions(**changed_inputs, grounding_out={})
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        model.grounding_supervision = False
        observation = model.encode_conditions(**model.prepare_inputs([sample]))
        prefix_max_error = max((a.float() - b.float()).abs().max().item() for a, b in zip(first, observation))
        prefix_relative_error = max(((a.float() - b.float()).norm() / a.float().norm()).item()
                                    for a, b in zip(first, observation))
        # Different sequence lengths select different BF16 GEMM/SDPA kernels.
        # The same-length answer substitution above must remain bitwise identical.
        assert prefix_relative_error < .01, prefix_relative_error
        for a, b in zip(first, observation):
            torch.testing.assert_close(a, b, atol=.04, rtol=.02)
    del first, second, observation, inputs, changed_inputs, box
    model.train()
    model.grounding_supervision = True
    box = {}
    model.encode_conditions(**model.prepare_inputs([sample]), grounding_out=box)
    ce = box['ce']
    assert torch.isfinite(ce) and ce > 0
    ce.backward()
    lm_grad = model.vlm_backbone.vlm_model.lm_head.weight.grad.float().norm().item()
    assert lm_grad > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.vlm_backbone.vlm_model.model.parameters())
    model.zero_grad(set_to_none=True)
    del box, ce
    model.grounding_supervision = False
    losses = model.compute_loss(**model.prepare_inputs([sample]), lambda_video=0, lambda_action=1)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    for module in (model.action_backbone, model.backbone_conditioner, model.vlm_backbone):
        assert any(p.grad is not None and p.grad.norm() > 0 for p in module.parameters())
    model.zero_grad(set_to_none=True)
    separate_report = {}
    if args.image_input_mode == 'separate':
        from PIL import ImageOps
        from openwam.deploy.obs_preprocess import ObsPreprocessor
        from openwam.deploy.policy import WAMPolicy
        from openwam.deploy.engine import JointInferenceEngine
        from omegaconf import OmegaConf
        model.eval()
        assert len(sample['observation_images']) == 3
        prepared = model.prepare_inputs([sample])['vlm_inputs']
        assert len(prepared['image_grid_thw']) == 3
        # Bound every packed sequence without truncation; BPE boundary slack is conservative.
        tokenizer = model.vlm_backbone.processor.tokenizer
        token_counts = []
        for column in ('instruction', 'answer'):
            values = dataset.rows[column].unique().to_pylist()
            maximum = 0
            for start in range(0, len(values), 4096):
                encoded = tokenizer([x or '' for x in values[start:start+4096]], add_special_tokens=False)['input_ids']
                maximum = max(maximum, max(map(len, encoded), default=0))
            token_counts.append(maximum)
        empty = dict(sample, prompt=dataset.coordinate_setup + '\nTask instruction: \n' + dataset.manifest['request'])
        overhead = model.prepare_inputs([empty])['vlm_inputs']['input_ids'].shape[1]
        bound = overhead + sum(token_counts) + 16
        assert bound <= model.vlm_backbone._max_length, (bound, model.vlm_backbone._max_length)
        pre = ObsPreprocessor(multiview=True, camera_layout=['head_camera','left_camera','right_camera'],
                              img_height=240, img_width=320, requires_proprio=True, image_input_mode='separate')
        obs = pre.preprocess({'images': dict(zip(['head_camera','left_wrist_camera','right_wrist_camera'], sample['observation_images'])),
                              'prompt': sample['prompt'], 'state': [0.] * 20})
        policy = WAMPolicy.__new__(WAMPolicy)
        conditions = policy._build_conditions(obs)
        # Exercise the real deploy engine and generate adapter, not only processor packing.
        engine = JointInferenceEngine(OmegaConf.create({'inference': {'denoise_steps': 10}}), architecture=model)
        with torch.no_grad():
            base = model.encode_conditions(**model.prepare_inputs([sample]))
            for camera in range(3):
                views = list(sample['observation_images']); views[camera] = ImageOps.invert(views[camera])
                altered = model.encode_conditions(**model.prepare_inputs([dict(sample, observation_images=views)]))
                assert any(not torch.equal(a,b) for a,b in zip(base, altered)), f'Camera {camera} ignored'
            result = engine.generate(conditions)
            assert torch.isfinite(result['actions']).all() if torch.is_tensor(result['actions']) else __import__('numpy').isfinite(result['actions']).all()
        separate_report = {'three_camera_sensitivity_passed': True, 'three_image_generate_passed': True,
                           'dataset_token_length_upper_bound': bound}
    report = {'passed': True, 'lm_head_ce_grad_norm': lm_grad,
              'answer_independent_action_conditions': True, 'observation_only_prefix_matches': True,
              'same_length_answer_substitution_bitwise_equal': True,
              'bf16_observation_only_max_abs_error': prefix_max_error,
              'bf16_observation_only_relative_l2_error': prefix_relative_error,
              'action_gradient_reaches_vlm_bridge_and_head': True,
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated()}
    report.update(separate_report)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
