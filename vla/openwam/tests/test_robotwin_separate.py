import json
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from openwam.dataloader.robotwin_cot import RoboTwinCoTDataset, CAMERAS, coordinate_setup
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError
from openwam.deploy.policy import WAMPolicy
from openwam.deploy.engine import JointInferenceEngine
from openwam.model.vlm_backbone.groundingpi_backbone import GroundingPIBackbone
from omegaconf import OmegaConf


@pytest.fixture
def data():
    d = RoboTwinCoTDataset.__new__(RoboTwinCoTDataset)
    d.root = __import__('pathlib').Path('/unused')
    d.manifest = {'request': 'Locate the target.'}
    d.episodes = [{'eef_offset': 0, 'length': 3, 'videos': {k: {'path': k, 'from_timestamp': 0} for k in CAMERAS}}]
    d.rows = pa.Table.from_pylist([{'episode_index': 0, 'frame_index': i, 'timestamp': i / 50,
                                 'instruction': str(i), 'answer': str(i), 'ce_eligible': True} for i in range(3)])
    d.eef = np.zeros((4,20), dtype=np.float32);d.eef[2:] = 1
    d._normalizer = SimpleNamespace(normalize=lambda x: x)
    d.height = 240;d.width = 320;d.image_input_mode = 'separate'
    d.coordinate_setup = coordinate_setup('separate');d.filter_static_segments = True
    d.static_segment_threshold = 1e-5;d.max_static_retry = 3;d._color_jitter = None
    colors = {k: c for k,c in zip(CAMERAS, ['red','green','blue'])}
    d._read_frame = lambda path,t: Image.new('RGB',(640,480),colors[path.name])
    return d


def test_static_resampling_keeps_whole_sample_aligned(data, monkeypatch):
    candidates = iter([2,1])
    monkeypatch.setattr('openwam.dataloader.robotwin_cot.random.randint',lambda a,b: next(candidates))
    sample = data[0]
    assert sample['start_frame'] == 1 and sample['grounding_answer'] == '1'
    assert 'Task instruction: 1\n' in sample['prompt']
    assert sample['action'][0].eq(1).all() and sample['proprio'].eq(0).all()
    assert sample['action_mask'].sum() == 40
    views = sample['observation_images']
    assert [im.size for im in views] == [(320,240)] * 3
    assert [im.getpixel((0,0)) for im in views] == [(255,0,0),(0,128,0),(0,0,255)]


def test_static_retry_limit_keeps_original(data, monkeypatch):
    calls=[]
    monkeypatch.setattr('openwam.dataloader.robotwin_cot.random.randint',lambda a,b: calls.append(1) or 2)
    assert data[0]['start_frame'] == 0
    assert len(calls) == 3


def test_color_factors_shared_and_eval_has_no_augmentation(data):
    data.filter_static_segments = False
    data._read_frame = lambda *args: Image.new('RGB',(320,240),(100,110,120))
    data._color_jitter = VideoColorJitter(.2,.2,.2,0)
    views = data[1]['observation_images']
    assert all(np.array_equal(np.asarray(views[0]),np.asarray(im)) for im in views)
    data._color_jitter.eval()
    assert data[1]['observation_images'][0].getpixel((0,0)) == (100,110,120)


def test_preprocessor_requires_three_cameras_and_engine_passes_them(data):
    data.filter_static_segments = False
    sample=data[1]
    cfg=OmegaConf.create({'dataloader':{'image_input_mode':'separate','height':240,'width':320,'multiview':True},
                         'inference':{'denoise_steps':10,'height':384,'width':320}})
    pre=ObsPreprocessor.from_cfg(cfg)
    payload={'images':dict(zip(['head_camera','left_wrist_camera','right_wrist_camera'],sample['observation_images'])),
             'state':[0.]*20,'prompt':sample['prompt']}
    obs=pre.preprocess(payload)
    assert [im.tobytes() for im in obs['observation_images']] == [im.tobytes() for im in sample['observation_images']]
    conditions=WAMPolicy.__new__(WAMPolicy)._build_conditions(obs)
    calls=[]
    arch=SimpleNamespace(normalize_deploy_proprio=lambda x:x,generate=lambda **kw:calls.append(kw) or {})
    JointInferenceEngine(cfg,architecture=arch).generate(conditions)
    assert calls[0]['observation_images'] == obs['observation_images']
    del payload['images']['left_wrist_camera']
    with pytest.raises(ObsValidationError,match='left_wrist'):
        pre.preprocess(payload)


def test_real_processor_three_images_batched_order_and_answer_mask():
    processor=AutoProcessor.from_pretrained('/share/project/robot-rfm/robot-exp/usr/cjy/project/starVLA/playground/Pretrained_models/GroundingPI',trust_remote_code=True)
    processor.tokenizer.padding_side='right'
    b=GroundingPIBackbone.__new__(GroundingPIBackbone);torch.nn.Module.__init__(b)
    b.processor=processor;b._max_length=2048;b.grounding_generation_prompt=True
    views=[[Image.new('RGB',(320,240),color) for color in colors] for colors in [('red','green','blue'),('white','black','yellow')]]
    answer='<|object_ref_start|>bottle<|object_ref_end|><|box_start|><1><2><3><4><|box_end|>'
    packed=b.prepare_grounding_inputs(['Pick bottle','Move bottle'],views,[answer,None])
    assert packed['image_grid_thw'].shape == (6,3)
    image_token=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    assert (packed['input_ids']==image_token).sum(1).tolist()==[324,324]
    pixels=processor.image_processor(images=views[0]+views[1],return_tensors='pt')
    torch.testing.assert_close(packed['pixel_values'],pixels['pixel_values'])
    assert (packed['labels'][packed['action_prefix_mask']]==-100).all()
    assert (packed['labels'][1]==-100).all()
    assert processor.tokenizer.decode(packed['labels'][0][packed['labels'][0]!=-100]) == answer+'<|im_end|>'
    singles=[b.prepare_vlm_inputs([p],[v]) for p,v in zip(['Pick bottle','Move bottle'],views)]
    batch=b.batch_vlm_inputs(singles)
    assert len(batch['image_grid_thw'])==6
    torch.testing.assert_close(batch['pixel_values'],pixels['pixel_values'])
    with pytest.raises(ValueError,match='three'):
        b.prepare_vlm_inputs(['bad'],[views[0][:2]])
