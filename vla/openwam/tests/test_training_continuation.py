"""Continuation preserves Adam moments and keeps the final resumable checkpoint."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from openwam.train.utils.training_utils import build_cosine_scheduler, restart_resume_scheduler


def test_restart_preserves_moments_and_later_resume_progress():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    cfg = OmegaConf.create({'training': {'learning_rate': 1e-4, 'warmup_ratio': .1, 'lr_min_ratio': .01}})
    scheduler = build_cosine_scheduler(optimizer, total_opt_steps=100, cfg=cfg)
    initial_state = copy.deepcopy(scheduler.state_dict())
    initial_lrs = [g['lr'] for g in optimizer.param_groups]
    for _ in range(15):
        parameter.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    moments = copy.deepcopy(optimizer.state_dict()['state'])
    assert restart_resume_scheduler(scheduler, optimizer, opt_step=21000, restart_opt_step=21000,
                                    initial_state=initial_state, initial_lrs=initial_lrs)
    assert optimizer.param_groups[0]['lr'] == initial_lrs[0]
    for key, value in moments[0].items():
        torch.testing.assert_close(optimizer.state_dict()['state'][0][key], value)
    optimizer.step()
    scheduler.step()
    resumed_state = copy.deepcopy(scheduler.state_dict())
    resumed_lr = optimizer.param_groups[0]['lr']
    assert not restart_resume_scheduler(scheduler, optimizer, opt_step=21001, restart_opt_step=21000,
                                        initial_state=initial_state, initial_lrs=initial_lrs)
    assert scheduler.state_dict() == resumed_state
    assert optimizer.param_groups[0]['lr'] == resumed_lr
    with pytest.raises(ValueError, match='precedes'):
        restart_resume_scheduler(scheduler, optimizer, opt_step=20999, restart_opt_step=21000,
                                 initial_state=initial_state, initial_lrs=initial_lrs)


def test_final_checkpoint_has_state_and_prunes_to_three(tmp_path):
    from openwam.train.openwam_trainer import OpenWAMTrainer

    trainer = object.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({'training': {'retain_resume_state': True, 'keep_last_k_ckpts': 3}})
    trainer.accelerator = SimpleNamespace(wait_for_everyone=lambda: None)
    trainer.architecture = object()
    for step in [60000, 70000, 80000]:
        (tmp_path / f'checkpoint_step_{step}.safetensors').touch()
        (tmp_path / f'accel_state_step_{step}').mkdir()

    def weights(accelerator, architecture, directory, step, final):
        assert final
        (Path(directory) / f'checkpoint_step_{step}.safetensors').touch()

    def state(accelerator, directory, step, opt_step, epoch):
        assert (step, opt_step, epoch) == (84960, 42480, 19)
        (Path(directory) / f'accel_state_step_{step}').mkdir()

    module = 'openwam.train.openwam_trainer'
    with patch(module + '.save_weights', side_effect=weights), patch(module + '.save_full_state', side_effect=state):
        trainer.finish_training(str(tmp_path), 84960, 10000, True, None, opt_step=42480, epoch=19)
    assert sorted(p.name for p in tmp_path.glob('checkpoint_step_*')) == [
        'checkpoint_step_70000.safetensors', 'checkpoint_step_80000.safetensors', 'checkpoint_step_84960.safetensors']
    assert sorted(p.name for p in tmp_path.glob('accel_state_step_*')) == [
        'accel_state_step_70000', 'accel_state_step_80000', 'accel_state_step_84960']
