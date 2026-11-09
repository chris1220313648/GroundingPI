#!/usr/bin/env python
"""Validate all identities and materialize EEF/grounding training metadata once."""
import argparse
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.dataloader.robotwin import RoboTwinDataset
from openwam.dataloader.utils.normalization import pin_rot6d_identity, ROT6D_DIMS_EEF20

ENTRY = r"<\|object_ref_start\|>[^<>]+<\|object_ref_end\|><\|box_start\|>(?:<\d{1,3}>){4}(?:,(?:<\d{1,3}>){4})*<\|box_end\|>"
BBOX = re.compile(ENTRY + r"(?:,?\s*" + ENTRY + r")*")


def prepare(root, raw_root, out):
    root, raw_root, out = root.resolve(), raw_root.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Use a fresh prepared directory: {out}")
    info = json.loads((root / "meta/info.json").read_text())
    contract = json.loads((root / "meta/cot_contract.json").read_text())
    assert contract["profile"] == "groundingpi_target_only"
    mapping = pq.read_table(root / "annotations/episode_mapping.parquet").to_pylist()
    episode_rows = sorted([r for f in sorted((root / "meta/episodes").rglob("*.parquet"))
                           for r in pq.read_table(f).to_pylist()], key=lambda r: r["episode_index"])
    mapping = {r["episode_index"]: r for r in mapping}
    episodes, eef_arrays, joints = [], [], []
    offset = 0
    for ep in episode_rows:
        i = ep["episode_index"]
        assert i == len(episodes)
        source = mapping[i]
        length = ep["length"]
        assert source["frames"] == length
        path = raw_root / source["source_task"] / "aloha-agilex_clean_50/data" / f"episode{source['source_episode']}.hdf5"
        with h5py.File(path, "r") as f:
            q = f["joint_action/vector"][:].astype(np.float32)
            eef = RoboTwinDataset._read_eef_actions(None, f, 0, len(q))
        assert len(q) == length + 1, (path, len(q), length)
        assert eef.shape == (length + 1, 20) and np.isfinite(eef).all()
        videos = {}
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            prefix = "videos/" + key
            rel = info["video_path"].format(video_key=key, chunk_index=ep[prefix + "/chunk_index"],
                                           file_index=ep[prefix + "/file_index"])
            assert (root / rel).is_file(), rel
            videos[key] = {"path": rel, "from_timestamp": ep[prefix + "/from_timestamp"]}
        episodes.append({"length": length, "eef_offset": offset, "videos": videos,
                         "source": str(path), "data_from": ep["dataset_from_index"]})
        offset += len(eef)
        eef_arrays.append(eef)
        joints.append(q)
        if (i + 1) % 500 == 0:
            print(f"Validated EEF sources: {i + 1}/{len(episode_rows)}", flush=True)
    eef = np.concatenate(eef_arrays)
    np.save(out / "eef.npy", eef)
    stats = {"min": eef.min(0), "max": eef.max(0), "mean": eef.mean(0), "std": eef.std(0),
             "q01": np.quantile(eef, .01, axis=0), "q99": np.quantile(eef, .99, axis=0)}
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    np.save(out / "normalization_stats.npy", {"eef": stats})
    writer = None
    count = eligible_count = available_count = 0
    try:
        for f in sorted((root / "data").rglob("*.parquet")):
            side = pq.ParquetFile(root / "annotations" / f.relative_to(root))
            data = pq.ParquetFile(f)
            assert data.metadata.num_rows == side.metadata.num_rows
            # Identical batching permits identity checks without loading the large QC table at once.
            for batch, qc_batch in zip(data.iter_batches(batch_size=4096), side.iter_batches(batch_size=4096)):
                rows, qc_rows = batch.to_pylist(), qc_batch.to_pylist()
                assert len(rows) == len(qc_rows)
                samples = []
                for row, qc in zip(rows, qc_rows):
                    ep, frame, index = (row[k] for k in ("episode_index", "frame_index", "index"))
                    assert (ep, frame, index) == tuple(qc[k] for k in ("episode_index", "frame_index", "index"))
                    assert index == count == episodes[ep]["data_from"] + frame
                    np.testing.assert_allclose(row["observation.state"], joints[ep][frame], rtol=1e-5, atol=1e-5)
                    np.testing.assert_allclose(row["action"], joints[ep][frame + 1], rtol=1e-5, atol=1e-5)
                    assert abs(row["timestamp"] - frame / info["fps"]) < 1e-4
                    metadata = json.loads(qc["metadata_json"])
                    target = json.loads(qc["target_only_metadata_json"])
                    answer = row["annotation.cot"]
                    available = bool(row["annotation.cot_available"])
                    eligible = available and bool(target["target_coverage_complete"])
                    if available:
                        assert answer and BBOX.fullmatch(answer), (index, answer)
                        for box in re.findall(r"(?:<\d{1,3}>){4}", answer):
                            x1,y1,x2,y2 = map(int, re.findall(r"\d+", box))
                            assert 0 <= x1 <= x2 <= 999 and 0 <= y1 <= y2 <= 999
                    assert target["camera"] == "observation.images.cam_high"
                    instruction = metadata["cot_primary_instruction"]
                    assert instruction
                    samples.append({"episode_index": ep, "frame_index": frame, "timestamp": row["timestamp"],
                                    "instruction": instruction, "answer": answer, "ce_eligible": eligible})
                    count += 1
                    eligible_count += eligible
                    available_count += available
                table = pa.Table.from_pylist(samples)
                if writer is None:
                    writer = pq.ParquetWriter(out / "samples.parquet", table.schema)
                writer.write_table(table)
            print(f"Validated rows: {count}; CE eligible: {eligible_count}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    assert count == info["total_frames"] and len(episodes) == info["total_episodes"]
    manifest = {"dataset_dir": str(root), "raw_eef_root": str(raw_root), "episodes": episodes,
                "total_frames": count, "ce_available": available_count, "ce_eligible": eligible_count,
                "request": contract["request"], "candidate_only": True, "train_export_ready": False,
                "eef_alignment": "raw EEF[t] state, EEF[t+1:t+33] action; all joint rows verified",
                "bbox_coordinates": "unchanged 0-999 relative to head-camera subimage"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Preflight passed: {len(episodes)} episodes, {count} frames, {eligible_count} CE samples", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("raw_eef_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    prepare(args.dataset_dir, args.raw_eef_root, args.output_dir)
