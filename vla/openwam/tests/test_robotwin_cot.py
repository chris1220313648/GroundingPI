from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.dataloader.robotwin_cot import COORDINATE_SETUP, eef_window
from openwam.model.vlm_backbone.groundingpi_backbone import GroundingPIBackbone


def test_eef_time_alignment_and_episode_padding():
    eef = np.arange(12 * 20, dtype=np.float32).reshape(12, 20)
    state, action, valid = eef_window(eef, 7, 10, horizon=5)
    np.testing.assert_array_equal(state, eef[7:8])
    np.testing.assert_array_equal(action, eef[[8, 9, 9, 9, 9]])
    assert valid.tolist() == [True, True, False, False, False]
    assert "subimage, not the full composite" in COORDINATE_SETUP


class CausalModel(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, 4)
        self.lm_head = torch.nn.Linear(4, vocab, bias=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4, num_hidden_layers=8))


@pytest.fixture
def grounding():
    from transformers import AutoProcessor
    path = Path('/share/project/robot-rfm/robot-exp/usr/cjy/project/starVLA/playground/Pretrained_models/GroundingPI')
    if not path.is_dir():
        pytest.skip("Local GroundingPI processor is unavailable")
    b = GroundingPIBackbone.__new__(GroundingPIBackbone)
    torch.nn.Module.__init__(b)
    b.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    b.processor.tokenizer.padding_side = 'right'
    b.vlm_model = CausalModel(len(b.processor.tokenizer))
    b.dtype = torch.float32
    b._max_length = 2048
    b.grounding_generation_prompt = True
    def run(inputs, **kwargs):
        h = b.vlm_model.embedding(inputs['input_ids']).cumsum(dim=1) / 100
        return SimpleNamespace(last_hidden_state=h, hidden_states=(h,) * 9)
    b.run_backbone = run
    return b


ANSWER = '<|object_ref_start|>red block<|object_ref_end|><|box_start|><92><355><203><519><|box_end|>'


def test_native_answer_labels_padding_and_causal_prefix(grounding):
    b = grounding
    images = [Image.new('RGB', (320, 384))] * 2
    prompts = [COORDINATE_SETUP + '\nPick red block', COORDINATE_SETUP + '\nMove bottle']
    inputs = b.prepare_grounding_inputs(prompts, images, [ANSWER, None])
    labels = inputs['labels']
    assert (labels[inputs['action_prefix_mask']] == -100).all()
    assert (labels[~inputs['attention_mask'].bool()] == -100).all()
    assert (labels[1] == -100).all()
    decoded = b.processor.tokenizer.decode(labels[0][labels[0] != -100])
    assert decoded == ANSWER + '<|im_end|>'
    prompt_only = b.prepare_vlm_inputs(prompts, images)
    for i in range(2):
        n = int(inputs['action_prefix_mask'][i].sum())
        assert torch.equal(inputs['input_ids'][i, :n], prompt_only['input_ids'][i, :n])
    taps, mask, ce, metrics = b.extract_grounding_features(inputs, [0, 7])
    changed = b.prepare_grounding_inputs(prompts, images, [ANSWER.replace('<92>', '<100>'), None])
    other, other_mask, _, _ = b.extract_grounding_features(changed, [0, 7])
    assert torch.equal(mask, other_mask)
    for k in taps:
        torch.testing.assert_close(taps[k][mask], other[k][other_mask], rtol=0, atol=0)
    h = b.run_backbone(inputs).last_hidden_state
    valid = labels[:, 1:] != -100
    expected = torch.nn.functional.cross_entropy(b.vlm_model.lm_head(h[:, :-1][valid]), labels[:, 1:][valid])
    torch.testing.assert_close(ce, expected)
    assert metrics['grounding_tokens'] == valid.sum()
    assert metrics['grounding_sample_fraction'] == .5
    ce.backward()
    assert b.vlm_model.lm_head.weight.grad.norm() > 0
    assert b.vlm_model.embedding.weight.grad.norm() > 0


def test_all_missing_supervision_is_finite_zero(grounding):
    b = grounding
    inputs = b.prepare_grounding_inputs(['Move bottle'], [Image.new('RGB', (320, 384))], [None])
    _, _, ce, metrics = b.extract_grounding_features(inputs, [0])
    assert torch.isfinite(ce) and ce.item() == 0
    assert metrics['grounding_tokens'] == 0
    ce.backward()
    assert b.vlm_model.lm_head.weight.grad is not None
    assert b.vlm_model.lm_head.weight.grad.norm() == 0


def test_uneven_rank_token_normalization(grounding, monkeypatch):
    b = grounding
    inputs = b.prepare_grounding_inputs(['Pick block'], [Image.new('RGB', (320, 384))], [ANSWER])
    _, _, reference, metrics = b.extract_grounding_features(inputs, [0])
    local_count = metrics['grounding_tokens'].item()
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    def reduce(totals):
        totals += totals.new_tensor([3, 1, 1])
    monkeypatch.setattr(torch.distributed, 'all_reduce', reduce)
    _, _, loss, metrics = b.extract_grounding_features(inputs, [0])
    torch.testing.assert_close(loss, reference * 2 * local_count / (local_count + 3))
    assert metrics['grounding_tokens'].item() == local_count + 3


def test_eef_normalizer_preserves_rotations_and_roundtrips():
    from openwam.dataloader.transforms.normalize import Normalizer
    from openwam.dataloader.utils.normalization import pin_rot6d_identity, ROT6D_DIMS_EEF20
    stats = {k: np.full(20, v, dtype=np.float32) for k, v in
             [('min', -2), ('max', 2), ('mean', 0), ('std', 1), ('q01', -2), ('q99', 2)]}
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    n = Normalizer('min_max', stats)
    pose = np.linspace(-.9, .9, 20, dtype=np.float32)[None]
    norm = n.normalize(pose)
    np.testing.assert_allclose(norm[:, ROT6D_DIMS_EEF20], pose[:, ROT6D_DIMS_EEF20], atol=1e-6)
    np.testing.assert_allclose(n.unnormalize(norm), pose, atol=1e-6)


def test_trainer_enters_train_mode_after_hf_eval_initialization():
    from openwam.train.openwam_trainer import OpenWAMTrainer
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    arch = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.1))
    arch.eval()
    arch.dtype = torch.float32
    arch.set_dtype_device = lambda *args: None
    arch.move_frozen_to_device = lambda *args: None
    trainer.architecture = arch
    trainer.accelerator = SimpleNamespace(prepare=lambda *args: args,
                                         unwrap_model=lambda model: model,
                                         device=torch.device('cpu'))
    trainer.prepare_accelerate(None, None, None)
    assert all(module.training for module in arch.modules())
