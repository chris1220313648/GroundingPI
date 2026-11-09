from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import load_file
from openwam.model.architectures.base import BaseWAMArchitecture


def test_failed_save_preserves_previous_weights(tmp_path, monkeypatch):
    path = tmp_path/'checkpoint_step_1.safetensors'
    model = SimpleNamespace(vlm_is_trainable=lambda:True)
    BaseWAMArchitecture.save_checkpoint(model,str(path),state_dict={'weight':torch.ones(2)})
    before=path.read_bytes()
    def fail_save(state, temporary):
        with open(temporary,'wb') as f: f.write(b'incomplete')
        raise OSError('simulated interrupted write')
    monkeypatch.setattr('safetensors.torch.save_file',fail_save)
    with pytest.raises(OSError):
        BaseWAMArchitecture.save_checkpoint(model,str(path),state_dict={'weight':torch.zeros(2)})
    assert path.read_bytes()==before
    assert load_file(path)['weight'].tolist()==[1,1]
    assert not list(tmp_path.glob('.checkpoint-*'))
