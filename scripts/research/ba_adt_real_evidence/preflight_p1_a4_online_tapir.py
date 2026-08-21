#!/usr/bin/env python3
"""Outcome-blind local mechanics preflight for the frozen P1-A4 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


CHECKPOINT_URL = (
    "https://storage.googleapis.com/dm-tapnet/bootstap/"
    "causal_bootstapir_checkpoint.pt"
)
EXPECTED_REPOSITORY = "https://github.com/google-deepmind/tapnet.git"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(repository: Path) -> tuple[list[dict[str, object]], str]:
    names = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "-z"]
    ).decode("utf-8").split("\0")
    rows: list[dict[str, object]] = []
    for name in sorted(filter(None, names)):
        path = repository / name
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return rows, hashlib.sha256(canonical).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    remote = subprocess.check_output(
        ["git", "-C", str(repository), "remote", "get-url", "origin"], text=True
    ).strip()
    if remote.removesuffix("/").removesuffix(".git") != EXPECTED_REPOSITORY.removesuffix(".git"):
        raise RuntimeError(f"unexpected repository remote: {remote}")
    if subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("official source checkout is dirty")

    manifest, manifest_hash = source_manifest(repository)
    model_source = repository / "tapnet" / "torch" / "tapir_model.py"
    postprocess_source = repository / "tapnet" / "pytorch_live_demo.py"
    license_path = repository / "LICENSE"
    sys.path.insert(0, str(repository))

    import cv2  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as torch_functional  # pylint: disable=import-outside-toplevel
    import tree  # pylint: disable=import-outside-toplevel
    from tapnet.torch import tapir_model  # pylint: disable=import-outside-toplevel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen selection preflight")
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # A deterministic 256x256 RGB mechanics-only frame; no ADT asset or truth is read.
    yy, xx = np.mgrid[0:256, 0:256]
    frame_np = np.stack((xx, yy, (xx + yy) % 256), axis=-1).astype(np.uint8)
    frame = torch.from_numpy(frame_np).to(device)
    offsets = np.asarray((0.10, 0.30, 0.50, 0.70, 0.90), dtype=np.float32)
    points_yx = np.asarray(
        [(64 + 128 * oy, 64 + 128 * ox) for oy in offsets for ox in offsets],
        dtype=np.float32,
    )
    query_points = torch.from_numpy(
        np.column_stack((np.zeros(25, dtype=np.float32), points_yx))
    ).to(device)[None]
    frames = (frame[None, None].float() / 255 * 2) - 1
    feature_grids = model.get_feature_grids(frames, is_training=False)
    query_features = model.get_query_features(
        frames,
        is_training=False,
        query_points=query_points,
        feature_grids=feature_grids,
    )
    causal_state = model.construct_initial_causal_state(
        25, len(query_features.resolutions) - 1
    )
    causal_state = tree.map_structure(lambda value: value.to(device), causal_state)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=query_features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_state,
        get_causal_context=True,
    )
    tracks = trajectories["tracks"][-1]
    occlusion = trajectories["occlusion"][-1]
    expected_distance = trajectories["expected_dist"][-1]
    visibility_probability = (
        (1 - torch_functional.sigmoid(occlusion))
        * (1 - torch_functional.sigmoid(expected_distance))
    )
    visible = visibility_probability > 0.5
    expected_shape = (1, 25, 1, 2)
    if tuple(tracks.shape) != expected_shape:
        raise RuntimeError(f"unexpected track shape: {tuple(tracks.shape)}")
    tensors = (tracks, occlusion, expected_distance, visibility_probability)
    if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
        raise RuntimeError("non-finite output in mechanics canary")
    torch.cuda.synchronize(device)

    receipt = {
        "schema_version": "p1-a4-selection-preflight-v1",
        "status": "PYTORCH_ONLINE_BOOTSTAPIR_MECHANICS_PASS",
        "private_truth_reads": 0,
        "performance_arms_run": 0,
        "repository": {
            "url": EXPECTED_REPOSITORY,
            "commit": commit,
            "source_manifest_sha256": manifest_hash,
            "file_count": len(manifest),
        },
        "checkpoint": {
            "url": CHECKPOINT_URL,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "license": {
            "spdx": "Apache-2.0",
            "sha256": sha256_file(license_path),
        },
        "constructor": "tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)",
        "postprocessing": (
            "visibility_probability=(1-sigmoid(occlusion))*"
            "(1-sigmoid(expected_dist)); visible=probability>0.5"
        ),
        "source_hashes": {
            "tapir_model.py": sha256_file(model_source),
            "pytorch_live_demo.py": sha256_file(postprocess_source),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "canary": {
            "input_shape": list(frames.shape),
            "query_shape": list(query_points.shape),
            "track_shape": list(tracks.shape),
            "occlusion_shape": list(occlusion.shape),
            "expected_distance_shape": list(expected_distance.shape),
            "visible_shape": list(visible.shape),
            "finite": True,
            "visible_count": int(visible.sum().item()),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "wall_seconds": time.perf_counter() - started,
            "max_source_frame_read": 0,
            "output_frame": 0,
        },
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
