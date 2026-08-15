#!/usr/bin/env python3
"""Materialize the one frozen Bonn + DepthART SATOM-R0 Real E0 manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import torch

from .bonn import BONN_INTRINSICS, DEPTH_SCALE, estimate_camera_height_m, frozen_frame_rows, sha256_file
from .freeze_bonn_real_e0 import build as build_roster


MANIFEST_SCHEMA = "blindassist.satom_r0.dataset_manifest.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def _load_model(source: Path, checkpoint: Path, contract: dict[str, Any]):
    deployment = source / "deploy" / "shared"
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(deployment))
    sys.path.insert(0, str(deployment / "selective_scan"))
    from scripts.research.hftf.deployment.depthart.export_depthart_camera_external import install_timm_compat

    install_timm_compat()
    from common import preprocess  # type: ignore
    from depthart_selective_scan import install_depthart  # type: ignore
    from model import load_model  # type: ignore
    from network import tvimblock  # type: ignore

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    model = load_model(checkpoint, contract["encoder"], contract["domain"], contract["device"]).eval()
    install_depthart(tvimblock)
    return model, preprocess


def _payload_sequence(payload_root: Path, sequence_id: str) -> Path:
    candidates = (payload_root / sequence_id, payload_root / "rgbd_bonn_dataset" / sequence_id)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"missing frozen Bonn payload: {sequence_id}")


def _verify_parent_metadata(sequence_root: Path, frozen: dict[str, Any]) -> None:
    for name, identity in frozen["metadata"].items():
        path = sequence_root / name
        require(path.is_file(), f"missing metadata: {path}")
        require(path.stat().st_size == int(identity["bytes"]), f"metadata byte drift: {path}")
        require(sha256_file(path) == str(identity["sha256"]), f"metadata SHA drift: {path}")


def _resize_depth(depth: torch.Tensor, height: int, width: int) -> np.ndarray:
    if tuple(depth.shape) != (1, height, width):
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=True
        ).squeeze(1)
    output = depth.detach().cpu().numpy()[0].astype(np.float32, copy=False)
    require(output.shape == (height, width) and np.all(np.isfinite(output)), "DepthART output invalid")
    require(float(output.min()) > 0.0, "DepthART output contains non-positive metric depth")
    return output


def _materialize_parent(
    sequence_root: Path,
    frozen_parent: dict[str, Any],
    source_contract: dict[str, Any],
    height_amendment: dict[str, Any],
    parent_height_contract: dict[str, Any],
    prior_contract: dict[str, Any],
    model: torch.nn.Module,
    preprocess: Any,
    output_root: Path,
) -> dict[str, Any]:
    sequence_id = str(frozen_parent["sequence_id"])
    bundle = output_root / f"{sequence_id}.npz"
    receipt_path = output_root / f"{sequence_id}.receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(bundle.is_file(), f"resume bundle missing: {bundle}")
        require(bundle.stat().st_size == receipt["bytes"], f"resume bundle bytes drift: {bundle}")
        require(sha256_file(bundle) == receipt["sha256"], f"resume bundle SHA drift: {bundle}")
        return receipt
    require(not bundle.exists(), f"bundle exists without receipt: {bundle}")
    _verify_parent_metadata(sequence_root, frozen_parent)
    frames = frozen_frame_rows(sequence_root, source_contract)
    require(len(frames) == int(frozen_parent["selected_frame_count"]), "selected frame count drift")

    timestamps: list[float] = []
    truths: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    intrinsics_rows: list[list[float]] = []
    poses: list[np.ndarray] = []
    candidate_height_estimates: list[float | None] = []
    truth_height_estimates: list[float | None] = []
    gravities: list[np.ndarray] = []
    source_frames: list[dict[str, Any]] = []
    for row in frames:
        rgb_path = sequence_root / row["rgb_relative_path"]
        depth_path = sequence_root / row["depth_relative_path"]
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        require(bgr is not None and raw_depth is not None, f"failed to decode frozen frame: {rgb_path}")
        require(bgr.shape[:2] == (480, 640), f"unexpected Bonn RGB shape: {rgb_path}/{bgr.shape}")
        require(raw_depth.shape == (480, 640) and raw_depth.dtype == np.uint16, f"unexpected Bonn depth: {depth_path}")
        image, model_k = preprocess(bgr, BONN_INTRINSICS.astype(np.float32), 640, 480)
        require(tuple(image.shape) == (1, 3, 480, 640), f"DepthART tensor shape drift: {image.shape}")
        image = image.to(prior_contract["device"])
        model_k = model_k.to(prior_contract["device"])
        with torch.inference_mode():
            depth = model(image, model_k)
        prior = _resize_depth(depth, 480, 640)
        truth = raw_depth.astype(np.float32) / np.float32(DEPTH_SCALE)
        truth[raw_depth == 0] = np.nan
        gravity = np.asarray(row["gravity_down_camera"], dtype=np.float64)
        height_args = (
            BONN_INTRINSICS,
            gravity,
            float(height_amendment["minimum_m"]),
            float(height_amendment["maximum_m"]),
            float(height_amendment["histogram_bin_m"]),
            float(height_amendment["support_tolerance_m"]),
            int(height_amendment["minimum_support_points"]),
            float(height_amendment["minimum_support_fraction"]),
        )
        try:
            candidate_height = estimate_camera_height_m(prior, *height_args)
        except ValueError:
            candidate_height = None
        try:
            truth_height = estimate_camera_height_m(truth, *height_args)
        except ValueError:
            truth_height = None
        timestamps.append(float(row["rgb_timestamp_s"]))
        truths.append(truth)
        priors.append(prior)
        intrinsics_rows.append(
            [float(BONN_INTRINSICS[0, 0]), float(BONN_INTRINSICS[1, 1]),
             float(BONN_INTRINSICS[0, 2]), float(BONN_INTRINSICS[1, 2])]
        )
        poses.append(np.asarray(row["world_from_camera"], dtype=np.float64))
        candidate_height_estimates.append(candidate_height)
        truth_height_estimates.append(truth_height)
        gravities.append(gravity)
        source_frames.append(
            {
                "frame_index": int(row["frame_index"]),
                "rgb_timestamp_s": float(row["rgb_timestamp_s"]),
                "rgb_relative_path": str(row["rgb_relative_path"]),
                "depth_timestamp_s": float(row["depth_timestamp_s"]),
                "depth_relative_path": str(row["depth_relative_path"]),
            }
        )
        print(json.dumps({"parent": sequence_id, "frame": len(timestamps), "of": len(frames)}), flush=True)

    def parent_height(values: list[float | None], source_name: str) -> tuple[float, float, int]:
        valid_values = np.asarray([value for value in values if value is not None], dtype=np.float64)
        minimum = int(parent_height_contract["minimum_valid_height_frames_each_source"])
        require(len(valid_values) >= minimum, f"{sequence_id}: insufficient {source_name} parent height support")
        median = float(np.median(valid_values))
        mad = float(np.median(np.abs(valid_values - median)))
        require(mad <= float(parent_height_contract["maximum_height_mad_m"]), f"{sequence_id}: unstable {source_name} parent height")
        return median, mad, len(valid_values)

    candidate_height, candidate_height_mad, candidate_height_count = parent_height(
        candidate_height_estimates, "candidate"
    )
    truth_height, truth_height_mad, truth_height_count = parent_height(truth_height_estimates, "truth")
    atomic_npz(
        bundle,
        timestamp_s=np.asarray(timestamps, dtype=np.float64),
        truth_depth_m=np.stack(truths).astype(np.float32),
        prior_depth_m=np.stack(priors).astype(np.float32),
        intrinsics=np.asarray(intrinsics_rows, dtype=np.float64),
        world_from_camera=np.stack(poses).astype(np.float64),
        candidate_camera_height_m=np.full(len(frames), candidate_height, dtype=np.float64),
        truth_camera_height_m=np.full(len(frames), truth_height, dtype=np.float64),
        gravity_down_camera=np.stack(gravities).astype(np.float64),
    )
    receipt = {
        "parent_id": sequence_id,
        "bundle": bundle.name,
        "bytes": bundle.stat().st_size,
        "sha256": sha256_file(bundle),
        "frames": len(frames),
        "candidate_camera_height_m": {
            "parent_median": candidate_height, "valid_frames": candidate_height_count,
            "mad_m": candidate_height_mad,
        },
        "truth_camera_height_m": {
            "parent_median": truth_height, "valid_frames": truth_height_count, "mad_m": truth_height_mad,
        },
        "source_frames": source_frames,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--validity-amendment", type=Path, required=True)
    parser.add_argument("--parent-height-amendment", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--depthart-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    amendment_path = args.validity_amendment.resolve()
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    parent_amendment_path = args.parent_height_amendment.resolve()
    parent_amendment = json.loads(parent_amendment_path.read_text(encoding="utf-8"))
    require(lock["status"] == "FROZEN_BEFORE_BONN_PIXEL_OR_DEPTHART_OUTPUT_ACCESS", "execution lock is not frozen")
    require(amendment["original_lock"]["sha256"] == sha256_file(lock_path), "validity amendment/lock SHA drift")
    require(amendment["status"] == "FROZEN_AFTER_INVALID_ONE_FRAME_MATERIALIZATION_PREFLIGHT_BEFORE_ANY_ARM_METRIC", "validity amendment is not frozen")
    require(parent_amendment["predecessor_amendment"]["sha256"] == sha256_file(amendment_path), "parent-height amendment/predecessor SHA drift")
    require(parent_amendment["status"] == "FROZEN_AFTER_INVALID_18_FRAME_MATERIALIZATION_BEFORE_ANY_OUTPUT_FILE_OR_ARM_METRIC", "parent-height amendment is not frozen")
    source_contract = lock["source_contract"]
    expected_roster = source_contract["roster"]
    require(build_roster(args.metadata_root.resolve(), source_contract) == expected_roster, "metadata-only roster drift")
    prior = lock["prior_contract"]
    source = args.depthart_source.resolve()
    checkpoint = args.checkpoint.resolve()
    require(_git(source, "rev-parse", "HEAD") == prior["source_git_commit"], "DepthART source commit drift")
    require(not _git(source, "status", "--short"), "DepthART source tree is dirty")
    require(checkpoint.stat().st_size == prior["checkpoint_bytes"], "DepthART checkpoint byte drift")
    require(sha256_file(checkpoint) == prior["checkpoint_sha256"], "DepthART checkpoint SHA drift")
    runtime = prior["host_runtime"]
    require(torch.__version__ == runtime["torch"] and torch.version.cuda == runtime["cuda"], "torch runtime drift")
    require(cv2.__version__ == runtime["opencv"] and np.__version__ == runtime["numpy"], "host runtime drift")
    require(torch.cuda.is_available(), "frozen CUDA device unavailable")

    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.json"
    require(not manifest_path.exists(), f"Real E0 manifest already exists: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    model, preprocess = _load_model(source, checkpoint, prior)
    parent_receipts = []
    for frozen_parent in expected_roster["parents"]:
        sequence_root = _payload_sequence(args.payload_root.resolve(), frozen_parent["sequence_id"])
        parent_receipts.append(
            _materialize_parent(
                sequence_root, frozen_parent, source_contract, amendment["camera_height_contract_override"],
                parent_amendment["parent_height_contract"],
                prior, model, preprocess, output_root,
            )
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset": "Bonn RGB-D Dynamic Dataset",
        "evidence_role": lock["evidence_role"],
        "execution_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "execution_validity_amendment": {
            "path": str(amendment_path), "sha256": sha256_file(amendment_path),
        },
        "parent_height_amendment": {
            "path": str(parent_amendment_path), "sha256": sha256_file(parent_amendment_path),
        },
        "prior_provenance": {
            "family": "DepthART", "frozen": True, "truth_derived": False,
            "model_id": prior["model_id"], "source_git_commit": prior["source_git_commit"],
            "checkpoint_sha256": prior["checkpoint_sha256"], "confidence": prior["confidence"],
            "truth_scale_calibration": False,
        },
        "tof_source": "registered Bonn truth used only inside frozen deterministic sensor simulator",
        "parents": [
            {"parent_id": row["parent_id"], "bundle": row["bundle"], "sha256": row["sha256"]}
            for row in parent_receipts
        ],
        "frames": sum(int(row["frames"]) for row in parent_receipts),
        "claim_ceiling": lock["claim_ceiling"],
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "MATERIALIZED", "manifest": str(manifest_path), "frames": manifest["frames"]}, indent=2))


if __name__ == "__main__":
    main()
