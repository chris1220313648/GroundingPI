"""Verify the complete clean_50 data contract before a formal run."""
import json
import sys
from pathlib import Path
import h5py
from openwam.dataloader.robotwin import ROBOTWIN_ALL_TASKS
root, out = map(Path, sys.argv[1:3])
counts = {}
for task in ROBOTWIN_ALL_TASKS:
    folder = root/task/'aloha-agilex_clean_50'
    files = sorted((folder/'data').glob('*.hdf5'))
    if len(files) != 50:
        raise ValueError(f'{task}: expected 50 demonstrations, found {len(files)}')
    counts[task] = len(files)
    for path in files:
        instruction = folder/'instructions'/(path.stem+'.json')
        with instruction.open() as f:
            if not json.load(f): raise ValueError(f'Empty instruction: {instruction}')
        with h5py.File(path) as f:
            for key in ['observation/head_camera/rgb','observation/left_camera/rgb',
                        'observation/right_camera/rgb','endpose/left_endpose',
                        'endpose/right_endpose','endpose/left_gripper','endpose/right_gripper']:
                if key not in f or not len(f[key]):
                    raise ValueError(f'{path}: missing/empty {key}')
out.write_text(json.dumps({'root':str(root),'tasks':counts,'episodes':sum(counts.values())},indent=2))
print(f'Validated {len(counts)} tasks, {sum(counts.values())} demonstrations')
