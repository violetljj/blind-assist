#!/usr/bin/env python3
"""Outcome-blind ARKitScenes pose frontdoor for TARO task observability.

Frame and reference identities are selected from the frozen source manifest,
timestamps, and trajectories before depth, confidence, or RGB payloads are read.
Only landscape-preserving 0/180 degree canonical rotations are admitted so the
frozen 256x192 TARO observation contract is not resampled or distorted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.research.assistive_geometry import arkitscenes_truth_reader as arkit
from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn


SCHEMA = "blindassist.taro.task_observability_arkitscenes_balanced_pose_source_frontdoor.v1"
MANIFEST_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_media_manifest_v1"
EXPECTED_PARENT_COUNT = 32
EXPECTED_FRAMES_PER_PARENT = 300
MAX_POSE_BRACKET_S = 0.25
MAX_REFERENCES_PER_PARENT = 4
SOURCE_SIZE_WH = (256, 192)
ALLOWED_ORIENTATION_INDICES = (0, 2)


class FrontdoorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FrontdoorError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _stem(entry: Mapping[str, Any]) -> str:
    return Path(str(entry["path"])).stem


@dataclass(frozen=True)
class FrameAssets:
    depth: Mapping[str, Any]
    confidence: Mapping[str, Any]
    intrinsics: Mapping[str, Any]
    orientation_index: int


def canonical_landscape_pose(camera_to_world: np.ndarray) -> tuple[np.ndarray, int] | None:
    index = arkit.orientation_index(camera_to_world)
    if index not in ALLOWED_ORIENTATION_INDICES:
        return None
    output = np.asarray(camera_to_world, dtype=np.float64).copy()
    output[:3, :3] = output[:3, :3] @ arkit.upright_to_source_basis(index)
    return np.ascontiguousarray(output), index


def load_outcome_blind_roster(
    dataset_root: Path,
) -> tuple[list[bonn.Frame], dict[str, FrameAssets], dict[str, Any]]:
    manifest_path = dataset_root / "manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == MANIFEST_SCHEMA, "manifest schema drift")
    require(manifest.get("task_outcome_opened") is False, "manifest outcome boundary drift")
    require(manifest.get("model_outputs_read") is False, "manifest model-read boundary drift")
    videos = list(manifest.get("videos", []))
    require(manifest.get("video_count") == len(videos) == EXPECTED_PARENT_COUNT, "parent count drift")
    frames: list[bonn.Frame] = []
    assets: dict[str, FrameAssets] = {}
    parents: list[dict[str, Any]] = []
    orientation_counts = {str(index): 0 for index in range(4)}
    for video in sorted(videos, key=lambda row: str(row["video_id"])):
        parent_id = str(video["video_id"])
        selected = [str(value) for value in video["selected_frame_stems"]]
        require(len(selected) == EXPECTED_FRAMES_PER_PARENT, f"frame count drift: {parent_id}")
        extracted = video["extracted"]
        for modality in ("lowres_depth", "confidence", "lowres_wide_intrinsics"):
            require(len(extracted[modality]) == len(selected), f"{modality} count drift: {parent_id}")
        trajectory_path = Path(str(video["trajectory"]["path"]))
        require(trajectory_path.is_file(), f"missing trajectory: {parent_id}")
        require(sha256_file(trajectory_path) == str(video["trajectory"]["sha256"]), f"trajectory hash drift: {parent_id}")
        trajectory = arkit.parse_trajectory(trajectory_path)
        admitted = 0
        for frame_index, stem in enumerate(selected):
            depth_entry = extracted["lowres_depth"][frame_index]
            confidence_entry = extracted["confidence"][frame_index]
            intrinsics_entry = extracted["lowres_wide_intrinsics"][frame_index]
            require(_stem(depth_entry) == _stem(confidence_entry) == _stem(intrinsics_entry) == stem, f"stem drift: {stem}")
            timestamp = float(stem.rsplit("_", 1)[1])
            pose, _interpolation = arkit.interpolate_camera_to_world(trajectory, timestamp, MAX_POSE_BRACKET_S)
            index = arkit.orientation_index(pose)
            orientation_counts[str(index)] += 1
            canonical = canonical_landscape_pose(pose)
            if canonical is None:
                continue
            canonical_pose, index = canonical
            depth_path = Path(str(depth_entry["path"]))
            require(depth_path.is_file(), f"missing selected-window depth identity: {stem}")
            frame = bonn.Frame(parent_id, timestamp, depth_path, depth_path, canonical_pose)
            require(frame.frame_id not in assets, f"duplicate frame identity: {frame.frame_id}")
            assets[frame.frame_id] = FrameAssets(depth_entry, confidence_entry, intrinsics_entry, index)
            frames.append(frame)
            admitted += 1
        parents.append(
            {
                "parent_id": parent_id,
                "role": str(video["role"]),
                "visit_id": str(video["visit_id"]),
                "source_frame_count": len(selected),
                "landscape_contract_frame_count": admitted,
                "trajectory_sha256": str(video["trajectory"]["sha256"]),
            }
        )
    require(frames, "empty landscape-preserving roster")
    return frames, assets, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_schema": MANIFEST_SCHEMA,
        "analysis_role": "PROJECT_CONSUMED_DEVELOPMENT",
        "parent_count": len(parents),
        "source_frame_count": sum(row["source_frame_count"] for row in parents),
        "landscape_contract_frame_count": len(frames),
        "orientation_counts": orientation_counts,
        "orientation_policy": "admit indices 0 and 2 only; canonicalize without changing 256x192 shape",
        "selection_inputs": ["frozen manifest identities", "frame timestamps", "official trajectory payloads"],
        "source_selection_reads_task_outcome": False,
        "depth_confidence_intrinsics_payload_reads_during_selection": 0,
        "rgb_payload_reads": 0,
        "parents": parents,
    }


def _verify_entry(entry: Mapping[str, Any]) -> Path:
    path = Path(str(entry["path"]))
    require(path.is_file(), f"missing payload: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"payload byte drift: {path}")
    require(sha256_file(path) == str(entry["sha256"]), f"payload hash drift: {path}")
    return path


def load_observation(asset: FrameAssets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_path = _verify_entry(asset.depth)
    confidence_path = _verify_entry(asset.confidence)
    intrinsics_path = _verify_entry(asset.intrinsics)
    with Image.open(depth_path) as image:
        depth_raw = np.asarray(image).copy()
    with Image.open(confidence_path) as image:
        confidence = np.asarray(image).copy()
    intrinsics, size = arkit.parse_pincam(intrinsics_path)
    require(size == SOURCE_SIZE_WH, f"source size drift: {depth_path}")
    require(depth_raw.shape == confidence.shape == (SOURCE_SIZE_WH[1], SOURCE_SIZE_WH[0]), f"registered shape drift: {depth_path}")
    if asset.orientation_index == 2:
        depth_raw = arkit.rotate_array_upright(depth_raw, 2)
        confidence = arkit.rotate_array_upright(confidence, 2)
        intrinsics, rotated_size = arkit.rotate_intrinsics_upright(intrinsics, *SOURCE_SIZE_WH, 2)
        require(rotated_size == SOURCE_SIZE_WH, "180-degree rotation changed source shape")
    depth_m = np.asarray(arkit.depth_mm_to_metres(depth_raw), dtype=np.float64)
    return depth_m, np.asarray(confidence), np.asarray(intrinsics, dtype=np.float64)


def observation_geometry(
    depth_m: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence_valid = confidence >= 1
    range_valid = (depth_m >= adapter.DEPTH_RANGE_M[0]) & (depth_m <= adapter.DEPTH_RANGE_M[1])
    truth_valid = confidence_valid & range_valid
    truth_depth = np.where(truth_valid, depth_m, 0.0)
    maximum = ndimage.maximum_filter(np.where(truth_valid, depth_m, -np.inf), size=3, mode="constant", cval=-np.inf)
    minimum = ndimage.minimum_filter(np.where(truth_valid, depth_m, np.inf), size=3, mode="constant", cval=np.inf)
    static_valid = truth_valid & np.isfinite(maximum) & np.isfinite(minimum) & ((maximum - minimum) <= bonn.LOCAL_STABILITY_RANGE_M)
    rows, columns = np.indices(depth_m.shape, dtype=np.float64)
    points = np.stack(
        ((columns - intrinsics[0, 2]) * depth_m / intrinsics[0, 0],
         (rows - intrinsics[1, 2]) * depth_m / intrinsics[1, 1], depth_m), axis=-1
    )
    return np.ascontiguousarray(truth_depth), np.ascontiguousarray(points), np.ascontiguousarray(static_valid)


def _query_and_labels(
    frame: bonn.Frame,
    asset: FrameAssets,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[bool, ...]] | None:
    depth, confidence, intrinsics = load_observation(asset)
    truth_depth, points, static_valid = observation_geometry(depth, confidence, intrinsics)
    up_camera = adapter._normalize_vector(frame.camera_to_world[:3, :3].T @ arkit.WORLD_UP, "ARKIT_GRAVITY_INVALID")
    plane = prospective._fit_depth_plane(truth_depth, intrinsics, up_camera)
    if not plane["evaluable"]:
        return None
    queries = prospective._build_queries(
        frame.frame_id,
        hashlib.sha256(frame.frame_id.encode("utf-8")).hexdigest().upper(),
        round(frame.timestamp_s * 1_000_000_000),
        plane,
    )
    geometry = prospective._build_geometry(truth_depth, adapter.canonical_sha256(truth_depth), intrinsics)
    labels = tuple(str(r7_canary._truth_query_label(geometry, plane, intrinsics, query)["state"]) for query in queries)
    static = tuple(bool(r7_canary._occupied_grid(points, static_valid, query)[0][0][2]) for query in queries)
    return queries, labels, static


def evaluate(dataset_root: Path, limit: int = MAX_REFERENCES_PER_PARENT) -> dict[str, Any]:
    frames, assets, source = load_outcome_blind_roster(dataset_root)
    selected, capability = shared.select_pose_capable_references(frames, limit)
    if capability["eligible_parent_count"] < shared.MIN_RECOVERY_OPPORTUNITY_PARENTS or capability["selected_reference_count"] < shared.MIN_EVALUATED_REFERENCES:
        terminal = "NOT_EVALUABLE_DATA_OBSERVABILITY_PAIR_SUPPORT"
        census = None
        evaluated = 0
        abstained = 0
        payload_reads = 0
        checks = {"minimum_evaluated_references": False, "minimum_recovery_opportunity_parents": False, "minimum_clear_denominator_parents": False}
    else:
        counts = {"truth_occupied": 0, "truth_clear": 0, "truth_unknown": 0, "static_unknown_occupied_opportunity": 0}
        per_parent: dict[str, dict[str, int]] = defaultdict(lambda: {key: 0 for key in counts})
        evaluated = abstained = payload_reads = 0
        receipts: list[dict[str, Any]] = []
        for row in selected:
            outcome = _query_and_labels(row.reference, assets[row.reference.frame_id])
            payload_reads += 3
            if outcome is None:
                abstained += 1
                continue
            queries, labels, static = outcome
            parent = per_parent[row.reference.parent_id]
            local = {
                "truth_occupied": sum(label == "OCCUPIED_OBSERVED" for label in labels),
                "truth_clear": sum(label == "CLEAR_OBSERVED" for label in labels),
                "truth_unknown": sum(label == "UNKNOWN" for label in labels),
                "static_unknown_occupied_opportunity": sum((not state) and label == "OCCUPIED_OBSERVED" for state, label in zip(static, labels, strict=True)),
            }
            for key, value in local.items():
                counts[key] += int(value)
                parent[key] += int(value)
            receipts.append({"reference_frame_id": row.reference.frame_id, **local, "query_count": len(queries)})
            evaluated += 1
        checks, terminal = shared.decide_frontdoor(evaluated, per_parent)
        census = {
            "evaluated_reference_count": evaluated,
            "geometry_abstention_count": abstained,
            "query_count": sum(counts[key] for key in ("truth_occupied", "truth_clear", "truth_unknown")),
            "counts": counts,
            "recovery_opportunity_parent_count": sum(row["static_unknown_occupied_opportunity"] > 0 for row in per_parent.values()),
            "clear_denominator_parent_count": sum(row["truth_clear"] > 0 for row in per_parent.values()),
            "per_parent": dict(sorted(per_parent.items())),
            "reference_receipt_sha256": hashlib.sha256(canonical_json_bytes(receipts)).hexdigest().upper(),
        }
    result = {
        "schema": SCHEMA,
        "mode": "REVERSIBLE_EXPLORATION_PROJECT_CONSUMED_DEVELOPMENT",
        "question": "Does a pose-rich ARKitScenes source provide enough task-visible OCCUPIED recovery opportunities and CLEAR denominators for the frozen TARO five-arm R2?",
        "source": source,
        "pose_pair_capability": capability,
        "label_support_census": census,
        "evaluability": {
            "minimum_evaluated_reference_count": shared.MIN_EVALUATED_REFERENCES,
            "minimum_recovery_opportunity_parent_count": shared.MIN_RECOVERY_OPPORTUNITY_PARENTS,
            "minimum_clear_denominator_parent_count": shared.MIN_CLEAR_DENOMINATOR_PARENTS,
            "checks": checks,
        },
        "read_boundary": {"rgb_payload_decodes": 0, "depth_confidence_intrinsics_payload_reads_after_selection": payload_reads, "model_runs": 0, "training_steps": 0, "network_requests": 0, "r11_reads": 0},
        "terminal": terminal,
        "r2_five_arm_authorized": terminal == "TARO_TASK_OBSERVABILITY_BALANCED_POSE_SOURCE_FRONTDOOR_PASS",
        "claim_ceiling": "Consumed ARKitScenes source-derived Development frontdoor evidence only; not a sensing-arm result, fresh Confirmation, Android, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
    return result


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-references-per-parent", type=int, default=MAX_REFERENCES_PER_PARENT)
    args = parser.parse_args()
    require(args.max_references_per_parent > 0, "max references must be positive")
    result = evaluate(args.dataset_root.resolve(), args.max_references_per_parent)
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
