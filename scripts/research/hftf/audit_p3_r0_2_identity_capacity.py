#!/usr/bin/env python3
"""Label-blind identity-capacity audit for the finite P3 R0.2 source universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_p3_r0_2_identity_capacity_audit_result"
PROTOCOL_SCHEMA = "blindassist_p3_r0_2_identity_capacity_audit_protocol"
READY = "P3_R0_2_IDENTITY_CAPACITY_SUFFICIENT_DATA_ROLE_FREEZE_PERMITTED"
NOT_READY = "P3_TEMPORAL_ROUTE_DATA_NOT_READY"
FORBIDDEN_KEYS = {
    "clearance", "clearance_m", "geometry_state", "geometry_target_valid",
    "transition", "transition_label", "prediction", "predictions", "score",
    "scores", "candidate_performance", "model_output", "model_outputs",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        collided = {str(key).lower() for key in value} & FORBIDDEN_KEYS
        require(not collided, f"label/performance fields forbidden: {sorted(collided)}")
        for nested in value.values():
            reject_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_forbidden_keys(nested)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    reject_forbidden_keys(value)
    return value


def resolve_inside(root: Path, relative: str) -> Path:
    path = (root.resolve() / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path leaves repository: {relative}") from error
    return path


def bound_path(root: Path, binding: dict[str, Any]) -> Path:
    require(set(binding) == {"path", "sha256"}, "binding fields drift")
    path = resolve_inside(root, str(binding["path"]))
    require(path.is_file(), f"bound file missing: {path}")
    require(sha256_file(path) == str(binding["sha256"]).upper(), f"bound SHA mismatch: {path}")
    return path


def parse_timestamp(stem: str) -> int:
    seconds = float(stem.rsplit("_", 1)[-1])
    return int(round(seconds * 1_000_000_000))


def clip_capacity(timestamps: list[int]) -> int:
    ordered = sorted(set(timestamps))
    count = 0
    for start in range(max(0, len(ordered) - 3)):
        window = ordered[start : start + 4]
        if len(window) == 4 and all(0 < right - left <= 500_000_000 for left, right in zip(window, window[1:])):
            count += 1
    return count


def file_list(path: Path) -> list[tuple[float, str]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative = line.split(maxsplit=1)
        rows.append((float(timestamp), relative.replace("/", os.sep)))
    return rows


def asset_digest(root: Path, rows: list[tuple[float, str]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    for timestamp, relative in rows:
        path = (root / relative).resolve()
        require(path.is_file(), f"listed source asset missing: {path}")
        digest.update(f"{timestamp:.9f}|{relative}|{sha256_file(path)}\n".encode("utf-8"))
    return len(rows), digest.hexdigest().upper()


def arkit_inventory(root: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    source = protocol["source_universe"]["arkitscenes"]
    manifest = load_json(bound_path(root, source["scoped_manifest"]))
    require(manifest.get("schema") == "blindassist_spatial_calibration_head_r1_scoped_media_manifest", "ARKit manifest schema drift")
    attempted = set(protocol["exclusions"]["r0_1_attempted_holdout_parent_ids"])
    parents = []
    for video in manifest["videos"]:
        video_id = str(video["video_id"])
        role = str(video["role"])
        stems = [str(value) for value in video["selected_frame_stems"]]
        extracted = set(video["extracted"])
        truth_assets = {"lowres_depth", "confidence"}.issubset(extracted)
        existing_development = role in {"train", "validation"}
        excluded = video_id in attempted
        parents.append({
            "source": "arkitscenes", "parent_id": video_id, "existing_role": role,
            "rgb_identity_count": len(stems), "four_frame_clip_capacity": clip_capacity([parse_timestamp(stem) for stem in stems]),
            "raw_truth_assets_present": truth_assets, "r0_1_attempted_holdout": excluded,
            "eligible_train": role == "train" and truth_assets and not excluded,
            "eligible_validation": role == "validation" and truth_assets and not excluded,
            "eligible_new_holdout": False,
        })
    return parents


def bonn_inventory(root: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    parents = []
    for source in protocol["source_universe"]["bonn_rgbd"]:
        rgb_path = bound_path(root, source["rgb_index"])
        depth_path = bound_path(root, source["depth_index"])
        rgb_rows, depth_rows = file_list(rgb_path), file_list(depth_path)
        rgb_count, rgb_digest = asset_digest(rgb_path.parent, rgb_rows)
        depth_count, depth_digest = asset_digest(depth_path.parent, depth_rows)
        depth_times = [value for value, _ in depth_rows]
        paired = sum(1 for value, _ in rgb_rows if depth_times and min(abs(value - depth) for depth in depth_times) <= 0.05)
        capacity = clip_capacity([int(round(value * 1_000_000_000)) for value, _ in rgb_rows])
        truth = paired >= 4 and depth_count > 0
        parents.append({
            "source": "bonn_rgbd", "parent_id": source["parent_id"], "existing_role": "unassigned",
            "rgb_identity_count": rgb_count, "four_frame_clip_capacity": capacity,
            "raw_truth_assets_present": truth, "paired_depth_identity_count": paired,
            "rgb_asset_aggregate_sha256": rgb_digest, "depth_asset_aggregate_sha256": depth_digest,
            "eligible_train": False, "eligible_validation": False,
            "eligible_new_holdout": truth and capacity > 0,
        })
    return parents


def build_result(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol_sha = sha256_file(protocol_path)
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "audit protocol schema drift")
    require(protocol["auditor"]["sha256"] == sha256_file(Path(__file__)), "auditor SHA drift")
    bound_path(root, protocol["exclusions"]["r0_1_closure"])
    bound_path(root, protocol["exclusions"]["legacy_p1_ledger"])
    bound_path(root, protocol["existing_development_ancestry"])
    parents = arkit_inventory(root, protocol) + bonn_inventory(root, protocol)
    train = {row["parent_id"] for row in parents if row["eligible_train"]}
    validation = {row["parent_id"] for row in parents if row["eligible_validation"]}
    holdout = {row["parent_id"] for row in parents if row["eligible_new_holdout"]}
    minimum = protocol["capacity_gate"]
    capacity = {
        "eligible_train_parent_count": len(train),
        "eligible_validation_parent_count": len(validation),
        "eligible_new_holdout_parent_count": len(holdout),
        "minimum_train_parent_count": minimum["minimum_train_parent_count"],
        "minimum_validation_parent_count": minimum["minimum_validation_parent_count"],
        "minimum_new_holdout_parent_count": minimum["minimum_new_holdout_parent_count"],
        "preferred_new_holdout_parent_count": minimum["preferred_new_holdout_parent_count"],
    }
    sufficient = (
        len(train) >= minimum["minimum_train_parent_count"]
        and len(validation) >= minimum["minimum_validation_parent_count"]
        and len(holdout) >= minimum["minimum_new_holdout_parent_count"]
    )
    return {
        "schema": SCHEMA,
        "protocol_sha256": protocol_sha,
        "source_universe_closed": True,
        "label_blind": True,
        "forbidden_fields_read": False,
        "model_outputs_read": False,
        "candidate_performance_read": False,
        "parent_inventory": parents,
        "capacity": capacity,
        "sufficient_for_train_validation_and_new_sealed_holdout": sufficient,
        "r0_2_data_role_freeze_permitted": sufficient,
        "terminal": READY if sufficient else NOT_READY,
    }


def exclusive_write(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    result = build_result(root, args.protocol.resolve())
    exclusive_write(args.output.resolve(), result)
    reproduced = build_result(root, args.protocol.resolve())
    validation = {
        "schema": "blindassist_p3_r0_2_identity_capacity_audit_validation",
        "result_sha256": sha256_file(args.output.resolve()),
        "exact_reproduction": result == reproduced,
        "valid": result == reproduced,
        "terminal": "P3_R0_2_IDENTITY_CAPACITY_AUDIT_VALID" if result == reproduced else "P3_R0_2_IDENTITY_CAPACITY_AUDIT_INVALID",
    }
    exclusive_write(args.validation_output.resolve(), validation)
    print(json.dumps({"terminal": result["terminal"], "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
