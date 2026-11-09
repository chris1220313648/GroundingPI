"""Video backbone package for OpenWAM.

This package is the public facade. The pieces live in dedicated modules:
  - ``base`` — :class:`VideoBackbone` ABC + :class:`BlockLoopState`
  - ``registry``          — the registry dict + ``register_video_backbone`` /
                            ``build_video_backbone``
This file re-exports them and registers the built-in Wan backbone at the bottom::

    from openwam.model.video_backbone import build_video_backbone
    backbone = build_video_backbone("wan22_ti2v_5b", cfg)

Adding a new backbone:
1. Subclass :class:`VideoBackbone` (implement prepare/run_block/finalize).
2. Register it at the bottom of this file via ``register_video_backbone("name")(YourClass)``.
3. Set ``video_backbone.name: your_name`` in the model config yaml.
"""

from __future__ import annotations

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.registry import (
    _VIDEO_BACKBONE_REGISTRY,
    build_video_backbone,
    register_video_backbone,
)

__all__ = [
    "BlockLoopState",
    "VideoBackbone",
    "Wan22Ti2v",
    "CosmosPredict25VideoBackbone",
    "build_video_backbone",
    "register_video_backbone",
    "_VIDEO_BACKBONE_REGISTRY",
]

# Load optional implementations only when requested. Base interfaces remain
# importable in VLM-only installations without the vendored Wan runtime.
def __getattr__(name):
    if name == "Wan22Ti2v":
        from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
        return Wan22Ti2v
    if name == "CosmosPredict25VideoBackbone":
        from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone
        return CosmosPredict25VideoBackbone
    raise AttributeError(name)
