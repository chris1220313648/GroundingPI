"""LeRobot v3 observations/grounding with frame-aligned RoboTwin EEF targets."""
from collections import OrderedDict
import json
import random
from pathlib import Path

import numpy as np
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.transforms.video import VideoColorJitter
from PIL import Image

CAMERAS = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
COORDINATE_SETUP = (
    "The image contains three camera views: the head camera at the top, "
    "the left wrist at bottom left, and the right wrist at bottom right. "
    "Locate targets only in the top head-camera view. Return xyxy coordinates "
    "on the 0-999 grid relative to that head-camera subimage, not the full composite."
)
SEPARATE_COORDINATE_SETUP = (
    "Three separate images are provided in order: head camera, left wrist camera, right wrist camera. "
    "Locate targets only in the first image (head camera). Return xyxy coordinates "
    "on the 0-999 grid relative to that first image."
)


def coordinate_setup(image_input_mode):
    if image_input_mode not in ("composite", "separate"):
        raise ValueError(f"Unknown image_input_mode: {image_input_mode}")
    return SEPARATE_COORDINATE_SETUP if image_input_mode == "separate" else COORDINATE_SETUP


def eef_window(eef, start, stop, horizon=32):
    """EEF[t] is state; EEF[t+1:] is action. stop includes the final raw pose."""
    indices = np.arange(start + 1, start + horizon + 1)
    valid = indices < stop
    return eef[start:start + 1].copy(), eef[np.minimum(indices, stop - 1)].copy(), valid


class RoboTwinCoTDataset(BaseDataset):
    action_dim = 20

    def __init__(self, prepared_dir, dataset_dir, height=384, width=320, image_input_mode="composite",
                 color_jitter=None, filter_static_segments=False, static_segment_threshold=1e-5, max_static_retry=3):
        import pyarrow.parquet as pq

        self.root = Path(dataset_dir)
        prepared = Path(prepared_dir)
        self.manifest = json.loads((prepared / "manifest.json").read_text())
        if str(self.root.resolve()) != self.manifest["dataset_dir"]:
            raise ValueError("Prepared data belongs to another dataset")
        self.episodes = self.manifest["episodes"]
        self.rows = pq.read_table(prepared / "samples.parquet", memory_map=True)
        self.eef = np.load(prepared / "eef.npy", mmap_mode="r")
        self.normalization_stats_path = str(prepared / "normalization_stats.npy")
        self._normalizer = Normalizer("min_max", np.load(self.normalization_stats_path, allow_pickle=True).item()["eef"])
        self.height, self.width = height, width
        self.image_input_mode = image_input_mode
        self.coordinate_setup = coordinate_setup(image_input_mode)
        self._color_jitter = VideoColorJitter(**dict(color_jitter)) if color_jitter else None
        self.filter_static_segments = filter_static_segments
        self.static_segment_threshold = float(static_segment_threshold)
        self.max_static_retry = int(max_static_retry)
        if self.static_segment_threshold < 0 or self.max_static_retry < 0:
            raise ValueError("Static threshold and retry count must be nonnegative")
        self._videos = OrderedDict()

    @classmethod
    def from_config(cls, config, split="train"):
        if split != "train":
            raise ValueError("This delivery only defines a training split")
        return cls(config.prepared_dir, config.dataset_dir, config.height, config.width,
                   **{k: config[k] for k in ("image_input_mode", "color_jitter", "filter_static_segments",
                                             "static_segment_threshold", "max_static_retry") if k in config})

    def __len__(self):
        return len(self.rows)

    def _read_frame(self, path, timestamp):
        import av

        path = str(path)
        if path not in self._videos:
            if len(self._videos) >= 12:
                _, old = self._videos.popitem(last=False)
                old.close()
            container = av.open(path)
            container.streams.video[0].codec_context.thread_count = 1
            self._videos[path] = container
        self._videos.move_to_end(path)
        container = self._videos[path]
        stream = container.streams.video[0]
        container.seek(max(0, int(timestamp / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time >= timestamp - 0.005:
                if abs(frame.time - timestamp) > 0.011:
                    raise ValueError(f"Video timestamp mismatch: {path}: {frame.time} vs {timestamp}")
                return frame.to_image()
        raise ValueError(f"No RGB frame at {timestamp}: {path}")

    def _is_static(self, index):
        row = self.rows.slice(int(index), 1).to_pylist()[0]
        ep = self.episodes[row["episode_index"]]
        pos = ep["eef_offset"] + row["frame_index"]
        poses = self._normalizer.normalize(self.eef[pos:pos + 2].copy())
        return float(np.abs(poses[1] - poses[0]).max()) < self.static_segment_threshold

    def __getitem__(self, index):
        if self.filter_static_segments and self._is_static(index) and len(self) > 1:
            for _ in range(self.max_static_retry):
                candidate = random.randint(0, len(self) - 1)
                if not self._is_static(candidate):
                    index = candidate
                    break
        row = self.rows.slice(int(index), 1).to_pylist()[0]
        ep = self.episodes[row["episode_index"]]
        frame = row["frame_index"]
        proprio, action, valid = eef_window(self.eef, ep["eef_offset"] + frame,
                                           ep["eef_offset"] + ep["length"] + 1)
        proprio = self._normalizer.normalize(proprio)
        action = self._normalizer.normalize(action)
        images = {}
        for key in CAMERAS:
            video = ep["videos"][key]
            images[key] = self._read_frame(self.root / video["path"], video["from_timestamp"] + row["timestamp"])
        if self.image_input_mode == "separate":
            views = [images[k].convert("RGB").resize((self.width, self.height), Image.Resampling.BILINEAR) for k in CAMERAS]
        else:
            views = [assemble_multiview_layout(images, list(CAMERAS), self.height, self.width)]
        if self._color_jitter is not None:
            views = self._color_jitter.apply({"video": views})["video"]
        image = views[0]
        prompt = (self.coordinate_setup + "\nTask instruction: " + row["instruction"] + "\n"
                  + self.manifest["request"])
        return {
            "video": [image], "first_frame_image": [image], "prompt": prompt,
            **({"observation_images": views} if self.image_input_mode == "separate" else {}),
            "action": torch.from_numpy(action), "proprio": torch.from_numpy(proprio),
            "action_mask": torch.from_numpy(np.broadcast_to(valid[:, None], (32, 20)).copy()),
            "proprio_mask": torch.ones(1, 20, dtype=torch.bool),
            "grounding_answer": row["answer"] if row["ce_eligible"] else None,
            "episode_index": row["episode_index"], "start_frame": frame,
        }
