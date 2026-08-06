"""Materialize compact four-scale Canonical DA V2 teacher feature targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_dav2_model_variant_gate_r0 import sha256_file  # noqa: E402
from produce_external_rgb_metric_depth_observations import DepthAnythingV2MetricSource  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--dav2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "clearance-student-mobile-s1":
        raise ValueError("S1 protocol mismatch")
    if manifest.get("truth_inputs_opened") is not False:
        raise ValueError("teacher truth firewall failed")
    declared_checkpoint_sha = str(manifest.get("checkpoint_sha256", "")).upper()
    if len(declared_checkpoint_sha) != 64:
        raise ValueError("teacher manifest checkpoint SHA missing or malformed")
    actual_checkpoint_sha = sha256_file(args.checkpoint).upper()
    if actual_checkpoint_sha != declared_checkpoint_sha:
        raise ValueError(
            "teacher checkpoint binding mismatch: "
            f"manifest={declared_checkpoint_sha} actual={actual_checkpoint_sha}"
        )
    records = manifest["records"]
    count = min(len(records), args.limit) if args.limit else len(records)
    args.output_root.mkdir(parents=True)
    # Four compact teacher feature tensors: [N, 384, 8, 8], float16.
    feature_path = args.output_root / "teacher_features_f16.npy"
    partial = args.output_root / "teacher_features_f16.partial.npy"
    cache = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(count, 4, 384, 8, 8))
    source = DepthAnythingV2MetricSource(args.dav2_repo.resolve(), args.checkpoint.resolve(), "cuda" if torch.cuda.is_available() else "cpu", 518, "fp16" if torch.cuda.is_available() else "fp32")
    model = source.model
    with torch.inference_mode():
        for index, record in enumerate(records[:count]):
            bgr = cv2.imread(str(record["rgb_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"cannot decode teacher RGB: {record['frame_id']}")
            image, _ = model.image2tensor(bgr, input_size=518)
            features = model.pretrained.get_intermediate_layers(image, model.intermediate_layer_idx[model.encoder], return_class_token=True)
            compact = []
            for feature in features:
                tokens = feature[0]
                patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14
                spatial = tokens[:, : patch_h * patch_w].permute(0, 2, 1).reshape(1, -1, patch_h, patch_w)
                compact.append(F.adaptive_avg_pool2d(spatial, (8, 8))[0].float().cpu().numpy())
            cache[index] = np.stack(compact).astype(np.float16)
    cache.flush()
    mmap = getattr(cache, "_mmap", None)
    if mmap is not None:
        mmap.close()
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, feature_path)
    receipt = {
        "schema": "blindassist_clearance_student_mobile_s1_teacher_features",
        "protocol_sha256": sha256_file(args.protocol),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "teacher_checkpoint_sha256": sha256_file(args.checkpoint),
        "truth_inputs_opened": False,
        "shape": [count, 4, 384, 8, 8],
        "dtype": "float16",
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": sha256_file(feature_path),
        "pooling": "adaptive_avg_pool_8x8",
        "terminal": "S1_TEACHER_FEATURE_CACHE_COMPLETE_DEVELOPMENT_ONLY",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
