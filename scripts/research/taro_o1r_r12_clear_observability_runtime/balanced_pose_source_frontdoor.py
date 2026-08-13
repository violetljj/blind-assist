#!/usr/bin/env python3
"""Outcome-blind TartanGround source selection plus TARO label-support census.

The source roster and reference identities are frozen from tracked corpus manifests,
local media presence, timestamps, and poses before any depth payload is decoded.
Depth is opened only after selection to decide whether the frozen TARO query has
enough OCCUPIED-recovery and CLEAR parent support for a five-arm R2 comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy import ndimage

from scripts.research.hftf import run_stage_c_d5_tartanground_development_pilot as tartan
from scripts.research.hftf import materialize_stage_c_d5_tartanground_development_corpus as materializer
from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn


SCHEMA = "blindassist.taro.task_observability_balanced_pose_source_frontdoor.v1"
PROVIDER = {"repo_id": tartan.REPO_ID, "revision": tartan.REVISION}
NATIVE_SIZE_WH = (640, 640)
TARTANGROUND_INTRINSICS = np.asarray(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
LOW_INTRINSICS = bonn._scaled_intrinsics(
    TARTANGROUND_INTRINSICS,
    NATIVE_SIZE_WH,
    bonn.LOW_SIZE_WH,
)
WORLD_UP = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
# TartanGround pose rotations map local NED [forward,right,down] into world NED.
# TARO geometry uses the pinhole convention [right,down,forward].
STANDARD_CAMERA_TO_LOCAL_NED = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
MAX_REFERENCES_PER_PARENT = 4
MIN_EVALUATED_REFERENCES = 48
MIN_RECOVERY_OPPORTUNITY_PARENTS = 4
MIN_CLEAR_DENOMINATOR_PARENTS = 4
ALLOWED_SCHEMAS = {
    "blindassist_hftf_stage_c_d5_tartanground_development_corpus_v0",
    "blindassist_hftf_stage_c_d5_tartanground_development_expansion_v1",
}


class FrontdoorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FrontdoorError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tartanground_pose_to_standard_camera(row: np.ndarray) -> np.ndarray:
    translation, local_to_world = tartan.pose_matrix(np.asarray(row, dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = local_to_world @ STANDARD_CAMERA_TO_LOCAL_NED
    transform[:3, 3] = translation
    return np.ascontiguousarray(transform)


def _source_depth_root(corpus_root: Path, source: Mapping[str, Any]) -> Path:
    return corpus_root / "media" / str(source["role"]) / str(source["parent_id"]) / "depth"


def load_outcome_blind_roster(
    corpus_roots: Sequence[Path],
    metadata_root: Path,
) -> tuple[list[bonn.Frame], dict[str, Any]]:
    frames: list[bonn.Frame] = []
    source_rows: list[dict[str, Any]] = []
    seen_parents: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []
    for corpus_root in sorted(path.resolve() for path in corpus_roots):
        manifest_path = corpus_root / "manifest.json"
        require(manifest_path.is_file(), f"missing corpus manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("schema") in ALLOWED_SCHEMAS, f"unsupported corpus schema: {manifest_path}")
        require(manifest.get("provider") == PROVIDER, f"provider drift: {manifest_path}")
        require(bool(manifest.get("policy", {}).get("outcome_open")), f"source is not outcome-open: {manifest_path}")
        manifest_rows.append(
            {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "schema": manifest["schema"],
            }
        )
        for source in manifest.get("sources", []):
            parent_id = str(source["parent_id"])
            require(parent_id not in seen_parents, f"duplicate parent: {parent_id}")
            seen_parents.add(parent_id)
            metadata_path = metadata_root / parent_id / "metadata.zip"
            require(metadata_path.is_file(), f"missing metadata: {metadata_path}")
            metadata, poses = tartan.load_metadata(metadata_root, parent_id)
            require(int(metadata["num_poses"]) == int(source["num_poses"]), f"pose count drift: {parent_id}")
            require(
                abs(float(metadata["robot_height"]) - float(source["robot_height_m"])) <= 1e-9,
                f"robot height drift: {parent_id}",
            )
            depth_root = _source_depth_root(corpus_root, source)
            local_depth_paths = sorted(depth_root.glob("*.png"))
            require(len(local_depth_paths) == int(source["depth_frame_count"]), f"depth count drift: {parent_id}")
            local_frame_ids = [int(path.stem) for path in local_depth_paths]
            require(local_frame_ids == sorted(set(local_frame_ids)), f"depth identity drift: {parent_id}")
            block_start = int(source["block_start_frame_id"])
            block_end = int(source["block_end_frame_id"])
            frame_ids = list(range(block_start, block_end + 1))
            require(frame_ids and max(frame_ids) < len(poses), f"pose coverage drift: {parent_id}")
            for frame_id in frame_ids:
                depth_path = depth_root / f"{frame_id:06d}.png"
                frames.append(
                    bonn.Frame(
                        parent_id=parent_id,
                        timestamp_s=frame_id / 10.0,
                        rgb_path=depth_path,
                        depth_path=depth_path,
                        camera_to_world=tartanground_pose_to_standard_camera(poses[frame_id]),
                    )
                )
            source_rows.append(
                {
                    "parent_id": parent_id,
                    "role": str(source["role"]),
                    "environment": str(source["environment"]),
                    "frame_count": len(frame_ids),
                    "local_depth_frame_count_before_selection": len(local_frame_ids),
                    "minimum_frame_id": min(frame_ids),
                    "maximum_frame_id": max(frame_ids),
                    "metadata_sha256": sha256_file(metadata_path),
                    "robot_height_m": float(source["robot_height_m"]),
                }
            )
    require(frames, "empty source roster")
    return frames, {
        "provider": PROVIDER,
        "analysis_role": "PROJECT_CONSUMED_DEVELOPMENT",
        "source_selection_reads_task_outcome": False,
        "selection_inputs": [
            "tracked corpus manifests",
            "local depth path identities without payload decode",
            "metadata.zip pose and camera metadata",
        ],
        "depth_payload_reads_during_selection": 0,
        "manifest_receipts": manifest_rows,
        "parents": sorted(source_rows, key=lambda row: row["parent_id"]),
        "parent_count": len(source_rows),
        "frame_count": len(frames),
        "camera_model": {
            "resolution_wh": list(NATIVE_SIZE_WH),
            "intrinsics": TARTANGROUND_INTRINSICS.tolist(),
            "source_axes": "local NED [forward,right,down] to world NED",
            "taro_axes": "standard pinhole [right,down,forward]",
        },
    }


def select_pose_capable_references(
    frames: Sequence[bonn.Frame],
    limit: int = MAX_REFERENCES_PER_PARENT,
) -> tuple[list[bonn.ReferenceSupport], dict[str, Any]]:
    by_parent: dict[str, list[bonn.Frame]] = defaultdict(list)
    for frame in frames:
        by_parent[frame.parent_id].append(frame)
    selected: list[bonn.ReferenceSupport] = []
    parent_rows: list[dict[str, Any]] = []
    for parent_id in sorted(by_parent):
        ordered = sorted(by_parent[parent_id], key=lambda frame: frame.timestamp_s)
        supports = bonn.build_reference_support(ordered)
        parent_selected = bonn.select_references(supports, limit)
        selected.extend(parent_selected)
        parent_rows.append(
            {
                "parent_id": parent_id,
                "frame_count": len(ordered),
                "legal_reference_count": len(supports),
                "legal_pair_count": sum(len(row.candidates) for row in supports),
                "micro_pair_count": sum(len(row.micro_candidates) for row in supports),
                "selected_reference_count": len(parent_selected),
            }
        )
    identities = [
        {
            "reference": row.reference.frame_id,
            "candidates": [pair.neighbor.frame_id for pair in row.candidates],
            "micro_candidates": [pair.neighbor.frame_id for pair in row.micro_candidates],
        }
        for row in selected
    ]
    return selected, {
        "parent_count": len(parent_rows),
        "eligible_parent_count": sum(row["selected_reference_count"] > 0 for row in parent_rows),
        "selected_reference_count": len(selected),
        "maximum_references_per_parent": limit,
        "passive_window_s": [bonn.PASSIVE_MIN_GAP_S, bonn.PASSIVE_MAX_GAP_S],
        "micro_translation_range_m": list(bonn.MICRO_TRANSLATION_RANGE_M),
        "micro_max_rotation_deg": bonn.MICRO_MAX_ROTATION_DEG,
        "selection_identity_sha256": hashlib.sha256(canonical_json_bytes(identities)).hexdigest().upper(),
        "parents": parent_rows,
    }


def _load_depth(path: Path) -> np.ndarray:
    depth = tartan.decode_depth(path.read_bytes())
    require(depth.shape == (NATIVE_SIZE_WH[1], NATIVE_SIZE_WH[0]), f"depth shape drift: {path}")
    require(np.all(np.isfinite(depth)), f"non-finite depth: {path}")
    return np.ascontiguousarray(depth, dtype=np.float64)


def _low_observation(depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = cv2.resize(depth_m, bonn.LOW_SIZE_WH, interpolation=cv2.INTER_NEAREST)
    valid = (low >= adapter.DEPTH_RANGE_M[0]) & (low <= adapter.DEPTH_RANGE_M[1])
    maximum = ndimage.maximum_filter(np.where(valid, low, -np.inf), size=3, mode="constant", cval=-np.inf)
    minimum = ndimage.minimum_filter(np.where(valid, low, np.inf), size=3, mode="constant", cval=np.inf)
    stable = valid & np.isfinite(maximum) & np.isfinite(minimum) & ((maximum - minimum) <= bonn.LOCAL_STABILITY_RANGE_M)
    rows, columns = np.indices(low.shape, dtype=np.float64)
    points = np.stack(
        (
            (columns - LOW_INTRINSICS[0, 2]) * low / LOW_INTRINSICS[0, 0],
            (rows - LOW_INTRINSICS[1, 2]) * low / LOW_INTRINSICS[1, 1],
            low,
        ),
        axis=-1,
    )
    return np.ascontiguousarray(low), np.ascontiguousarray(points), np.ascontiguousarray(stable)


def _gravity_up_camera(frame: bonn.Frame) -> np.ndarray:
    return adapter._normalize_vector(frame.camera_to_world[:3, :3].T @ WORLD_UP, "TARTANGROUND_GRAVITY_INVALID")


def _query_and_labels(
    frame: bonn.Frame,
    depth_m: np.ndarray,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[bool, ...]] | None:
    low_depth, points, stable = _low_observation(depth_m)
    low_plane = prospective._fit_depth_plane(low_depth, LOW_INTRINSICS, _gravity_up_camera(frame))
    if not low_plane["evaluable"]:
        return None
    queries = prospective._build_queries(
        frame.frame_id,
        hashlib.sha256(frame.frame_id.encode("utf-8")).hexdigest().upper(),
        round(frame.timestamp_s * 1_000_000_000),
        low_plane,
    )
    full_plane = prospective._fit_depth_plane(depth_m, TARTANGROUND_INTRINSICS, _gravity_up_camera(frame))
    if not full_plane["evaluable"]:
        return None
    geometry = prospective._build_geometry(
        depth_m,
        adapter.canonical_sha256(depth_m),
        TARTANGROUND_INTRINSICS,
    )
    labels = tuple(
        str(r7_canary._truth_query_label(geometry, full_plane, TARTANGROUND_INTRINSICS, query)["state"])
        for query in queries
    )
    static = tuple(bool(r7_canary._occupied_grid(points, stable, query)[0][0][2]) for query in queries)
    return queries, labels, static


def decide_frontdoor(
    evaluated_reference_count: int,
    per_parent: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, bool], str]:
    recovery_parents = sum(row.get("static_unknown_occupied_opportunity", 0) > 0 for row in per_parent.values())
    clear_parents = sum(row.get("truth_clear", 0) > 0 for row in per_parent.values())
    checks = {
        "minimum_evaluated_references": evaluated_reference_count >= MIN_EVALUATED_REFERENCES,
        "minimum_recovery_opportunity_parents": recovery_parents >= MIN_RECOVERY_OPPORTUNITY_PARENTS,
        "minimum_clear_denominator_parents": clear_parents >= MIN_CLEAR_DENOMINATOR_PARENTS,
    }
    terminal = (
        "TARO_TASK_OBSERVABILITY_BALANCED_POSE_SOURCE_FRONTDOOR_PASS"
        if all(checks.values())
        else "NOT_EVALUABLE_DATA_OBSERVABILITY_DENOMINATOR"
    )
    return checks, terminal


def evaluate_frontdoor(
    corpus_roots: Sequence[Path],
    metadata_root: Path,
    limit: int = MAX_REFERENCES_PER_PARENT,
    materialize_missing: bool = False,
) -> dict[str, Any]:
    frames, source = load_outcome_blind_roster(corpus_roots, metadata_root.resolve())
    selected, capability = select_pose_capable_references(frames, limit)
    if (
        capability["eligible_parent_count"] < MIN_RECOVERY_OPPORTUNITY_PARENTS
        or capability["selected_reference_count"] < MIN_EVALUATED_REFERENCES
    ):
        result = {
            "schema": SCHEMA,
            "mode": "REVERSIBLE_EXPLORATION_PROJECT_CONSUMED_DEVELOPMENT",
            "source": source,
            "pose_pair_capability": capability,
            "label_support_census": None,
            "read_boundary": {
                "rgb_payload_decodes": 0,
                "depth_payload_reads_before_selection": 0,
                "depth_payload_reads_after_selection": 0,
                "network_archive_open_count": 0,
                "model_runs": 0,
                "training_steps": 0,
                "r11_reads": 0,
            },
            "terminal": "NOT_EVALUABLE_DATA_OBSERVABILITY_PAIR_SUPPORT",
            "r2_five_arm_authorized": False,
            "claim_ceiling": "Consumed TartanGround source-derived Development source-frontdoor evidence only; not a sensing-arm result, fresh Confirmation, Android, product, default-App, or safety evidence.",
        }
        result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
        return result
    missing_by_parent: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in selected:
        if not row.reference.depth_path.is_file():
            missing_by_parent[(row.reference.parent_id, str(row.reference.depth_path.parent))].append(
                int(round(row.reference.timestamp_s * 10.0))
            )
    missing_before = sum(len(frame_ids) for frame_ids in missing_by_parent.values())
    acquisition_rows: list[dict[str, Any]] = []
    if missing_by_parent and materialize_missing:
        for (parent_id, output_directory), frame_ids in sorted(missing_by_parent.items()):
            unique_ids = sorted(set(frame_ids))
            materializer.fetch_frames(parent_id, "depth", unique_ids, Path(output_directory))
            acquisition_rows.append(
                {
                    "parent_id": parent_id,
                    "frame_ids": unique_ids,
                    "file_count": len(unique_ids),
                    "bytes": sum((Path(output_directory) / f"{frame_id:06d}.png").stat().st_size for frame_id in unique_ids),
                }
            )
    missing_after = [
        row.reference.frame_id
        for row in selected
        if not row.reference.depth_path.is_file()
    ]
    if missing_after:
        result = {
            "schema": SCHEMA,
            "mode": "REVERSIBLE_EXPLORATION_PROJECT_CONSUMED_DEVELOPMENT",
            "source": source,
            "pose_pair_capability": capability,
            "selected_depth_materialization": {
                "materialize_missing_requested": materialize_missing,
                "missing_before_count": missing_before,
                "missing_after_count": len(missing_after),
                "missing_after_frame_ids": missing_after,
                "archive_open_count": len(acquisition_rows),
                "acquisitions": acquisition_rows,
            },
            "label_support_census": None,
            "read_boundary": {
                "rgb_payload_decodes": 0,
                "depth_payload_reads_before_selection": 0,
                "depth_payload_reads_after_selection": 0,
                "network_archive_open_count": len(acquisition_rows),
                "model_runs": 0,
                "training_steps": 0,
                "r11_reads": 0,
            },
            "terminal": "ENV_BLOCKED_SELECTED_DEPTH_PAYLOAD_ABSENT",
            "r2_five_arm_authorized": False,
            "claim_ceiling": "Consumed TartanGround source-derived Development source-frontdoor evidence only; not a sensing-arm result, fresh Confirmation, Android, product, default-App, or safety evidence.",
        }
        result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
        return result
    counts = {"truth_occupied": 0, "truth_clear": 0, "truth_unknown": 0, "static_unknown_occupied_opportunity": 0}
    per_parent: dict[str, dict[str, int]] = defaultdict(
        lambda: {"truth_occupied": 0, "truth_clear": 0, "truth_unknown": 0, "static_unknown_occupied_opportunity": 0}
    )
    evaluated = 0
    abstained = 0
    query_count = 0
    receipts: list[dict[str, Any]] = []
    for row in selected:
        outcome = _query_and_labels(row.reference, _load_depth(row.reference.depth_path))
        if outcome is None:
            abstained += 1
            continue
        queries, labels, static = outcome
        parent = per_parent[row.reference.parent_id]
        label_counts = {
            "OCCUPIED_OBSERVED": sum(label == "OCCUPIED_OBSERVED" for label in labels),
            "CLEAR_OBSERVED": sum(label == "CLEAR_OBSERVED" for label in labels),
            "UNKNOWN": sum(label == "UNKNOWN" for label in labels),
        }
        opportunity = sum((not state) and label == "OCCUPIED_OBSERVED" for state, label in zip(static, labels, strict=True))
        for target, label_name in (
            ("truth_occupied", "OCCUPIED_OBSERVED"),
            ("truth_clear", "CLEAR_OBSERVED"),
            ("truth_unknown", "UNKNOWN"),
        ):
            value = int(label_counts[label_name])
            counts[target] += value
            parent[target] += value
        counts["static_unknown_occupied_opportunity"] += opportunity
        parent["static_unknown_occupied_opportunity"] += opportunity
        query_count += len(queries)
        evaluated += 1
        receipts.append(
            {
                "reference_frame_id": row.reference.frame_id,
                "label_counts": label_counts,
                "static_unknown_occupied_opportunity": opportunity,
            }
        )
    checks, terminal = decide_frontdoor(evaluated, per_parent)
    recovery_parents = sum(row["static_unknown_occupied_opportunity"] > 0 for row in per_parent.values())
    clear_parents = sum(row["truth_clear"] > 0 for row in per_parent.values())
    result = {
        "schema": SCHEMA,
        "mode": "REVERSIBLE_EXPLORATION_PROJECT_CONSUMED_DEVELOPMENT",
        "source": source,
        "pose_pair_capability": capability,
        "selected_depth_materialization": {
            "materialize_missing_requested": materialize_missing,
            "missing_before_count": missing_before,
            "missing_after_count": 0,
            "archive_open_count": len(acquisition_rows),
            "acquisitions": acquisition_rows,
        },
        "label_support_census": {
            "evaluated_reference_count": evaluated,
            "geometry_abstention_count": abstained,
            "query_count": query_count,
            **counts,
            "recovery_opportunity_parent_count": recovery_parents,
            "clear_denominator_parent_count": clear_parents,
            "per_parent": {key: per_parent[key] for key in sorted(per_parent)},
        },
        "frozen_gates": {
            "minimum_evaluated_reference_count": MIN_EVALUATED_REFERENCES,
            "minimum_recovery_opportunity_parent_count": MIN_RECOVERY_OPPORTUNITY_PARENTS,
            "minimum_clear_denominator_parent_count": MIN_CLEAR_DENOMINATOR_PARENTS,
            "checks": checks,
        },
        "selection_receipt_sha256": capability["selection_identity_sha256"],
        "label_receipt_sha256": hashlib.sha256(canonical_json_bytes(receipts)).hexdigest().upper(),
        "read_boundary": {
            "rgb_payload_decodes": 0,
            "depth_payload_reads_before_selection": 0,
            "depth_payload_reads_after_selection": evaluated + abstained,
            "model_runs": 0,
            "training_steps": 0,
            "network_archive_open_count": len(acquisition_rows),
            "r11_reads": 0,
        },
        "terminal": terminal,
        "r2_five_arm_authorized": terminal.endswith("_PASS"),
        "claim_ceiling": "Consumed TartanGround source-derived Development source-frontdoor evidence only; not a sensing-arm result, fresh Confirmation, Android, product, default-App, or safety evidence.",
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
    parser.add_argument("--corpus-root", action="append", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-references-per-parent", type=int, default=MAX_REFERENCES_PER_PARENT)
    parser.add_argument("--materialize-missing", action="store_true")
    args = parser.parse_args()
    require(args.max_references_per_parent > 0, "max references must be positive")
    result = evaluate_frontdoor(
        args.corpus_root,
        args.metadata_root,
        args.max_references_per_parent,
        args.materialize_missing,
    )
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
