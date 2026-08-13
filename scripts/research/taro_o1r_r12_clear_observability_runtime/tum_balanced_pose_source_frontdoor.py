#!/usr/bin/env python3
"""Outcome-blind TUM RGB-D pose/depth frontdoor for TARO R2.

Two tracked, already-consumed TUM cohort manifests define the source parents.
Reference identities are selected from indexes and ground-truth poses before any
image payload is decoded.  Native 640x480 registered depth supplies the
source-derived Development label; a fixed 256x192 observation supplies R7 state.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[3]
AG_HELPER_ROOT = REPO_ROOT / "scripts/research/assistive_geometry"
sys.path.insert(0, str(AG_HELPER_ROOT))
import ag_st_tum_rgbd as tum  # noqa: E402

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective  # noqa: E402
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter  # noqa: E402
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary  # noqa: E402
from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as shared  # noqa: E402
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn  # noqa: E402


SCHEMA = "blindassist.taro.task_observability_tum_balanced_pose_source_frontdoor.v1"
COHORT_SCHEMA = "blindassist_ag_st_tum_rgbd_third_domain_cohort_v1"
DEFAULT_MANIFESTS = (
    REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_RGBD_THIRD_DOMAIN_COHORT_R0_2026-08-10.json",
    REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_THIRD_TEACHER_COHORT_R2_2026-08-10.json",
)
NATIVE_SIZE_WH = (640, 480)
LOW_SIZE_WH = bonn.LOW_SIZE_WH
DEPTH_SCALE = 5000.0
MAX_REFERENCES_PER_PARENT = 12
WORLD_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


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


def _manifest_rows(manifest_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in manifest_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        require(value.get("schema") == COHORT_SCHEMA, f"cohort schema drift: {path}")
        require(str(value.get("status", "")).startswith("FROZEN_BEFORE_"), f"cohort not pre-outcome frozen: {path}")
        receipts.append({"path": str(path), "sha256": sha256_file(path), "status": value["status"], "token": value["token"]})
        for role_key, role in (("fit_parents", "FIT"), ("evaluation_parents", "EVALUATION")):
            for row in value[role_key]:
                parent_id = str(row["parent_id"])
                require(parent_id not in seen, f"duplicate parent across manifests: {parent_id}")
                seen.add(parent_id)
                sources.append(dict(row) | {"cohort_role": role, "cohort_token": value["token"]})
    require(len(sources) >= 4, "source parent count below frontdoor minimum")
    return sources, receipts


def _tar_controls(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    with tarfile.open(path, "r:*") as bundle:
        members = [member for member in bundle.getmembers() if member.isfile()]
        controls: dict[str, str] = {}
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            matches = [member for member in members if member.name == name or member.name.endswith("/" + name)]
            require(len(matches) == 1, f"{name} member ambiguity: {path}")
            stream = bundle.extractfile(matches[0])
            require(stream is not None, f"cannot read {name}: {path}")
            controls[name] = stream.read().decode("utf-8")
        member_lookup = {member.name.replace("\\", "/"): member.name for member in members}
    return controls, member_lookup


def _directory_controls(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    controls = {}
    for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
        control = path / name
        require(control.is_file(), f"missing {name}: {path}")
        controls[name] = control.read_text(encoding="utf-8")
    return controls, {}


@dataclass(frozen=True)
class DepthAsset:
    parent_id: str
    storage_kind: str
    source_path: Path
    depth_identity: str
    intrinsics: np.ndarray


def load_outcome_blind_roster(
    manifest_paths: Sequence[Path],
    verify_archive_hashes: bool = True,
) -> tuple[list[bonn.Frame], dict[str, DepthAsset], dict[str, Any]]:
    source_rows, manifest_receipts = _manifest_rows(manifest_paths)
    frames: list[bonn.Frame] = []
    assets: dict[str, DepthAsset] = {}
    parents: list[dict[str, Any]] = []
    for source in sorted(source_rows, key=lambda row: str(row["parent_id"])):
        parent_id = str(source["parent_id"])
        storage_kind = str(source["storage_kind"])
        source_path = (REPO_ROOT / str(source["source_path"])).resolve()
        require(source_path.exists(), f"source absent: {source_path}")
        if storage_kind == "tgz":
            require(source_path.is_file(), f"archive source invalid: {source_path}")
            require(source_path.stat().st_size == int(source["source_bytes"]), f"archive byte drift: {parent_id}")
            if verify_archive_hashes:
                require(sha256_file(source_path) == str(source["source_sha256"]), f"archive hash drift: {parent_id}")
            controls, members = _tar_controls(source_path)
        elif storage_kind == "directory":
            require(source_path.is_dir(), f"directory source invalid: {source_path}")
            controls, members = _directory_controls(source_path)
            for name, key in (("rgb.txt", "rgb_index_sha256"), ("depth.txt", "depth_index_sha256"), ("groundtruth.txt", "groundtruth_index_sha256")):
                require(sha256_file(source_path / name) == str(source[key]), f"control hash drift: {parent_id}/{name}")
        else:
            raise FrontdoorError(f"unsupported storage kind: {storage_kind}")
        rgb_rows = tum.parse_tum_index(controls["rgb.txt"])
        depth_rows = tum.parse_tum_index(controls["depth.txt"])
        pose_rows = tum.parse_tum_poses(controls["groundtruth.txt"])
        pairs = tum.pair_rgb_depth_unique(rgb_rows, depth_rows)
        intrinsics_values = [float(value) for value in source["intrinsics_fx_fy_cx_cy"]]
        intrinsics = np.asarray([[intrinsics_values[0], 0.0, intrinsics_values[2]], [0.0, intrinsics_values[1], intrinsics_values[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
        pose_abstentions = 0
        admitted = 0
        for rgb in rgb_rows:
            depth = pairs.get(rgb.row_index)
            if depth is None:
                continue
            try:
                camera_to_world, _gap = tum.interpolate_camera_to_world(pose_rows, rgb.timestamp_seconds)
            except Exception:
                pose_abstentions += 1
                continue
            relative = depth.relative_path.replace("\\", "/")
            if storage_kind == "tgz":
                candidates = [name for name in members if name == relative or name.endswith("/" + relative)]
                require(len(candidates) == 1, f"depth member ambiguity: {parent_id}/{relative}")
                depth_identity = candidates[0]
                placeholder = Path(relative)
            else:
                local = source_path / relative
                require(local.is_file(), f"depth payload absent: {local}")
                depth_identity = str(local)
                placeholder = local
            frame = bonn.Frame(parent_id, rgb.timestamp_seconds, placeholder, placeholder, camera_to_world)
            require(frame.frame_id not in assets, f"duplicate frame identity: {frame.frame_id}")
            assets[frame.frame_id] = DepthAsset(parent_id, storage_kind, source_path, depth_identity, intrinsics)
            frames.append(frame)
            admitted += 1
        parents.append(
            {
                "parent_id": parent_id,
                "cohort_role": source["cohort_role"],
                "storage_kind": storage_kind,
                "source_path": str(source_path),
                "expected_source_sha256": source.get("source_sha256"),
                "archive_hash_recomputed": storage_kind == "tgz" and verify_archive_hashes,
                "paired_pose_valid_frame_count": admitted,
                "pose_abstention_count": pose_abstentions,
                "intrinsics": intrinsics.tolist(),
            }
        )
    return frames, assets, {
        "family": "TUM_RGBD_BENCHMARK",
        "analysis_role": "PROJECT_CONSUMED_DEVELOPMENT",
        "manifest_receipts": manifest_receipts,
        "parent_count": len(parents),
        "frame_count": len(frames),
        "selection_inputs": ["tracked pre-outcome cohort manifests", "rgb.txt/depth.txt identities", "groundtruth.txt camera poses"],
        "selection_reads_task_outcome": False,
        "image_payload_reads_during_selection": 0,
        "native_resolution_wh": list(NATIVE_SIZE_WH),
        "depth_scale_divisor": DEPTH_SCALE,
        "world_up_xyz": WORLD_UP.tolist(),
        "parents": parents,
    }


def _decode_depth(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        raw = np.asarray(image).copy()
    require(raw.shape == (NATIVE_SIZE_WH[1], NATIVE_SIZE_WH[0]) and raw.dtype == np.uint16, "TUM depth shape/dtype drift")
    return np.ascontiguousarray(raw.astype(np.float64) / DEPTH_SCALE)


def load_selected_depths(
    selected: Sequence[bonn.ReferenceSupport],
    assets: Mapping[str, DepthAsset],
) -> dict[str, np.ndarray]:
    return load_depth_frame_ids([row.reference.frame_id for row in selected], assets)


def load_depth_frame_ids(
    frame_ids: Sequence[str],
    assets: Mapping[str, DepthAsset],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    archive_groups: dict[Path, list[tuple[str, DepthAsset]]] = defaultdict(list)
    for frame_id in frame_ids:
        asset = assets[frame_id]
        if asset.storage_kind == "directory":
            output[frame_id] = _decode_depth(Path(asset.depth_identity).read_bytes())
        else:
            archive_groups[asset.source_path].append((frame_id, asset))
    for archive, rows in sorted(archive_groups.items(), key=lambda item: str(item[0])):
        with tarfile.open(archive, "r:*") as bundle:
            members = {member.name: member for member in bundle.getmembers() if member.isfile()}
            ordered = sorted(rows, key=lambda item: members[item[1].depth_identity].offset_data)
            for frame_id, asset in ordered:
                stream = bundle.extractfile(members[asset.depth_identity])
                require(stream is not None, f"cannot read selected depth: {frame_id}")
                output[frame_id] = _decode_depth(stream.read())
    require(len(output) == len(set(frame_ids)), "selected depth cache incomplete")
    return output


def _low_observation(depth_m: np.ndarray, native_intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = cv2.resize(depth_m, LOW_SIZE_WH, interpolation=cv2.INTER_NEAREST)
    intrinsics = bonn._scaled_intrinsics(native_intrinsics, NATIVE_SIZE_WH, LOW_SIZE_WH)
    valid = (low >= adapter.DEPTH_RANGE_M[0]) & (low <= adapter.DEPTH_RANGE_M[1])
    maximum = ndimage.maximum_filter(np.where(valid, low, -np.inf), size=3, mode="constant", cval=-np.inf)
    minimum = ndimage.minimum_filter(np.where(valid, low, np.inf), size=3, mode="constant", cval=np.inf)
    stable = valid & np.isfinite(maximum) & np.isfinite(minimum) & ((maximum - minimum) <= bonn.LOCAL_STABILITY_RANGE_M)
    rows, columns = np.indices(low.shape, dtype=np.float64)
    points = np.stack(((columns - intrinsics[0, 2]) * low / intrinsics[0, 0], (rows - intrinsics[1, 2]) * low / intrinsics[1, 1], low), axis=-1)
    return np.ascontiguousarray(low), np.ascontiguousarray(points), np.ascontiguousarray(stable)


def query_and_labels(
    frame: bonn.Frame,
    depth_m: np.ndarray,
    native_intrinsics: np.ndarray,
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[bool, ...]] | None:
    low_depth, points, stable = _low_observation(depth_m, native_intrinsics)
    low_intrinsics = bonn._scaled_intrinsics(native_intrinsics, NATIVE_SIZE_WH, LOW_SIZE_WH)
    up_camera = adapter._normalize_vector(frame.camera_to_world[:3, :3].T @ WORLD_UP, "TUM_GRAVITY_INVALID")
    low_plane = prospective._fit_depth_plane(low_depth, low_intrinsics, up_camera)
    if not low_plane["evaluable"]:
        return None
    queries = prospective._build_queries(frame.frame_id, hashlib.sha256(frame.frame_id.encode("utf-8")).hexdigest().upper(), round(frame.timestamp_s * 1_000_000_000), low_plane)
    full_plane = prospective._fit_depth_plane(depth_m, native_intrinsics, up_camera)
    if not full_plane["evaluable"]:
        return None
    geometry = prospective._build_geometry(depth_m, adapter.canonical_sha256(depth_m), native_intrinsics)
    labels = tuple(str(r7_canary._truth_query_label(geometry, full_plane, native_intrinsics, query)["state"]) for query in queries)
    static = tuple(bool(r7_canary._occupied_grid(points, stable, query)[0][0][2]) for query in queries)
    return queries, labels, static


def evaluate(
    manifest_paths: Sequence[Path],
    limit: int = MAX_REFERENCES_PER_PARENT,
    verify_archive_hashes: bool = True,
) -> dict[str, Any]:
    frames, assets, source = load_outcome_blind_roster(manifest_paths, verify_archive_hashes)
    selected, capability = shared.select_pose_capable_references(frames, limit)
    if capability["eligible_parent_count"] < shared.MIN_RECOVERY_OPPORTUNITY_PARENTS or capability["selected_reference_count"] < shared.MIN_EVALUATED_REFERENCES:
        census = None
        checks = {"minimum_evaluated_references": False, "minimum_recovery_opportunity_parents": False, "minimum_clear_denominator_parents": False}
        terminal = "NOT_EVALUABLE_DATA_OBSERVABILITY_PAIR_SUPPORT"
        payload_reads = 0
    else:
        depth_cache = load_selected_depths(selected, assets)
        counts = {"truth_occupied": 0, "truth_clear": 0, "truth_unknown": 0, "static_unknown_occupied_opportunity": 0}
        per_parent: dict[str, dict[str, int]] = defaultdict(lambda: {key: 0 for key in counts})
        evaluated = abstained = 0
        receipts: list[dict[str, Any]] = []
        for row in selected:
            asset = assets[row.reference.frame_id]
            outcome = query_and_labels(row.reference, depth_cache[row.reference.frame_id], asset.intrinsics)
            if outcome is None:
                abstained += 1
                continue
            queries, labels, static = outcome
            local = {
                "truth_occupied": sum(label == "OCCUPIED_OBSERVED" for label in labels),
                "truth_clear": sum(label == "CLEAR_OBSERVED" for label in labels),
                "truth_unknown": sum(label == "UNKNOWN" for label in labels),
                "static_unknown_occupied_opportunity": sum((not state) and label == "OCCUPIED_OBSERVED" for state, label in zip(static, labels, strict=True)),
            }
            for key, value in local.items():
                counts[key] += int(value)
                per_parent[row.reference.parent_id][key] += int(value)
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
        payload_reads = len(selected)
    result = {
        "schema": SCHEMA,
        "mode": "REVERSIBLE_EXPLORATION_PROJECT_CONSUMED_DEVELOPMENT",
        "question": "Does a pre-outcome frozen multi-sequence TUM source provide enough task-visible OCCUPIED recovery and CLEAR parent support for TARO five-arm R2?",
        "source": source,
        "pose_pair_capability": capability,
        "label_support_census": census,
        "evaluability": {"minimum_evaluated_reference_count": shared.MIN_EVALUATED_REFERENCES, "minimum_recovery_opportunity_parent_count": shared.MIN_RECOVERY_OPPORTUNITY_PARENTS, "minimum_clear_denominator_parent_count": shared.MIN_CLEAR_DENOMINATOR_PARENTS, "checks": checks},
        "read_boundary": {"rgb_payload_decodes": 0, "depth_payload_reads_before_selection": 0, "depth_payload_reads_after_selection": payload_reads, "model_runs": 0, "training_steps": 0, "network_requests": 0, "r11_reads": 0},
        "terminal": terminal,
        "r2_five_arm_authorized": terminal == "TARO_TASK_OBSERVABILITY_BALANCED_POSE_SOURCE_FRONTDOOR_PASS",
        "claim_ceiling": "Consumed TUM RGB-D source-derived Development frontdoor evidence only; not a sensing-arm result, fresh Confirmation, Android, product, default-App, or safety evidence.",
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
    parser.add_argument("--manifest", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-references-per-parent", type=int, default=MAX_REFERENCES_PER_PARENT)
    parser.add_argument("--skip-archive-hash-verification", action="store_true")
    args = parser.parse_args()
    manifests = tuple(path.resolve() for path in (args.manifest or DEFAULT_MANIFESTS))
    result = evaluate(manifests, args.max_references_per_parent, not args.skip_archive_hash_verification)
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
