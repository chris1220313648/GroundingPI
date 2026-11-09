import json

import pytest

from benchmarks.robotwin.prompt_template import format_prompt_for_inference
from openwam.dataloader.robotwin_cot import COORDINATE_SETUP


def test_cot_prompt_matches_training(tmp_path, monkeypatch):
    path = tmp_path / 'prompt.json'
    request = 'Locate targets.'
    path.write_text(json.dumps({'coordinate_setup': COORDINATE_SETUP, 'request': request}))
    monkeypatch.setenv('ROBOTWIN_COT_PROMPT_FILE', str(path))
    assert format_prompt_for_inference('Pick bottle') == COORDINATE_SETUP + '\nTask instruction: Pick bottle\n' + request
    monkeypatch.delenv('ROBOTWIN_COT_PROMPT_FILE')
    assert format_prompt_for_inference('Pick bottle') == 'A video recorded from a robot\'s point of view executing the following instruction: Pick bottle'


def test_cot_prompt_rejects_missing_request(tmp_path, monkeypatch):
    path = tmp_path / 'prompt.json'
    path.write_text(json.dumps({'coordinate_setup': COORDINATE_SETUP}))
    monkeypatch.setenv('ROBOTWIN_COT_PROMPT_FILE', str(path))
    with pytest.raises(ValueError, match='request'):
        format_prompt_for_inference('Pick bottle')
