from types import SimpleNamespace
import pytest
import torch
from openwam.model.vlm_backbone.groundingpi_backbone import GroundingPIBackbone

class Tokenizer:
    pad_token_id = 0
    def __call__(self, texts, **kwargs):
        self.texts = texts
        rows = [[1] * (s.count('<|image_pad|>') + len(s.replace('<|image_pad|>', ''))) for s in texts]
        n = max(map(len, rows))
        return {'input_ids': torch.tensor([r + [0]*(n-len(r)) for r in rows]),
                'attention_mask': torch.tensor([[1]*len(r)+[0]*(n-len(r)) for r in rows])}

class ImageProcessor:
    merge_size = 2
    def __call__(self, images, **kwargs):
        return {'image_grid_thw': torch.tensor([[1,4,4]]*len(images)),
                'pixel_values': torch.zeros(len(images)*16, 12)}

class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=3,num_hidden_layers=8))

@pytest.fixture
def backbone():
    b = GroundingPIBackbone.__new__(GroundingPIBackbone)
    torch.nn.Module.__init__(b)
    b.dtype = torch.float32
    b._max_length = 100
    b.vlm_model = FakeModel()
    b.processor = SimpleNamespace(tokenizer=Tokenizer(), image_processor=ImageProcessor(),
        apply_chat_template=lambda m, **kw: '<|image_pad|>'+m[0]['content'][1]['text'])
    return b

def test_packing_and_collation(backbone):
    a = backbone.prepare_vlm_inputs(['a'], [object()])
    b = backbone.prepare_vlm_inputs(['longer'], [object()])
    batch = backbone.batch_vlm_inputs([a,b])
    assert batch['pixel_values'].shape == (32,12)
    assert batch['image_grid_thw'].shape == (2,3)
    assert batch['attention_mask'].sum(1).tolist() == [5,10]
    assert backbone.processor.tokenizer.texts[0].count('<|image_pad|>') == 4

def test_no_image_token_truncation(backbone):
    backbone._max_length=2
    with pytest.raises(ValueError,match='exceeds'):
        backbone.prepare_vlm_inputs(['a'],[object()])

def test_layer_indices_mask_and_gradient(backbone):
    def forward(inputs, **kwargs):
        shape=(*inputs['input_ids'].shape,3)
        return SimpleNamespace(hidden_states=tuple(backbone.vlm_model.weight * torch.ones(shape)*i for i in range(9)))
    backbone.run_backbone=forward
    inputs=backbone.prepare_vlm_inputs(['a','longer'],[object(),object()])
    taps,mask=backbone.extract_layerwise_features(inputs,range(8))
    assert taps[0][0,0,0].item()==1
    assert taps[7][0,0,0].item()==8
    assert mask.sum(1).tolist()==[5,10]
    sum(x.sum() for x in taps.values()).backward()
    assert backbone.vlm_model.weight.grad.item()>0
