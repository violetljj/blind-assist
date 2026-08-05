#!/usr/bin/env python3
"""Freeze P3 role identities, then finalize train/validation manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from p3_r0_1_asset_common import (
    ROLE_MANIFEST_SCHEMA,
    STATES,
    TRANSITIONS,
    commit_outputs,
    exact_fields,
    load_json,
    output_receipt,
    pretty_bytes,
    request_sha256,
    require,
    resolve_inside,
    sha256_file,
    sha256_bytes,
    validate_protocol,
    valid_sha,
    verify_bound_file,
    verify_producer_sha,
)


REQUEST_SCHEMA = "blindassist_p3_r0_1_role_manifest_materialization_request"
CATALOG_SCHEMA = "blindassist_p3_r0_1_role_source_catalog"
ASSIGNMENT_SCHEMA = "blindassist_p3_r0_1_role_assignment"
LOCK_SCHEMA = "blindassist_p3_r0_1_role_identity_lock"
DISAGREEMENT_SCHEMA = "blindassist_p3_r0_1_frozen_a2_disagreement_cache_jsonl"
WEIGHT_SCHEMA = "blindassist_p3_r0_1_transition_class_weight_derivation_receipt"
RECEIPT_SCHEMA = "blindassist_p3_r0_1_role_manifest_materialization_receipt"

PUBLIC_FIELDS = {
    "frame_id", "video_id", "parent_id", "timestamp_ns", "rgb_identity", "rgb_path", "rgb_sha256"
}
TRAIN_FIELDS = PUBLIC_FIELDS | {
    "teacher_depth_ref", "teacher_depth_path", "teacher_depth_sha256",
    "teacher_timestamp_ns", "teacher_valid", "tof_valid", "clearance_m",
    "geometry_state", "geometry_target_valid",
}


def _load_catalog(repo_root: Path, path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    value = load_json(path)
    exact_fields(value, {"schema", "train_validation_frames", "public_holdout_frames"}, "catalog")
    require(value["schema"] == CATALOG_SCHEMA, "catalog schema drift")
    result = []
    for role, rows, fields in (
        ("train_validation", value["train_validation_frames"], TRAIN_FIELDS),
        ("public_holdout", value["public_holdout_frames"], PUBLIC_FIELDS),
    ):
        require(isinstance(rows, list) and rows, f"{role} catalog is empty")
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            exact_fields(row, fields, f"{role} row")
            frame_id = str(row["frame_id"])
            require(frame_id and frame_id not in indexed, f"duplicate {role} frame")
            require(valid_sha(row["rgb_sha256"]), "RGB SHA invalid")
            rgb_path = resolve_inside(repo_root, row["rgb_path"])
            require(rgb_path.is_file() and sha256_file(rgb_path) == row["rgb_sha256"], "RGB file binding mismatch")
            require(isinstance(row["timestamp_ns"], int) and row["timestamp_ns"] > 0, "timestamp invalid")
            if role == "train_validation":
                require(valid_sha(row["teacher_depth_sha256"]), "teacher SHA invalid")
                teacher_path = resolve_inside(repo_root, row["teacher_depth_path"])
                require(teacher_path.is_file() and sha256_file(teacher_path) == row["teacher_depth_sha256"], "teacher file binding mismatch")
                require(isinstance(row["teacher_valid"], bool) and isinstance(row["tof_valid"], bool), "validity must be bool")
                require(isinstance(row["teacher_timestamp_ns"], int), "teacher timestamp invalid")
                require(isinstance(row["clearance_m"], list) and len(row["clearance_m"]) == 3, "clearance must have three bands")
                require(all(value is None or isinstance(value, (int, float)) and math.isfinite(float(value)) for value in row["clearance_m"]), "clearance invalid")
                require(isinstance(row["geometry_state"], list) and len(row["geometry_state"]) == 3 and all(item in STATES for item in row["geometry_state"]), "geometry state invalid")
                require(isinstance(row["geometry_target_valid"], list) and len(row["geometry_target_valid"]) == 3 and all(isinstance(item, bool) for item in row["geometry_target_valid"]), "geometry validity invalid")
            indexed[frame_id] = row
        result.append(indexed)
    return result[0], result[1]


def _load_assignment(path: Path) -> list[dict[str, Any]]:
    value = load_json(path)
    exact_fields(value, {"schema", "clip_length", "clips"}, "assignment")
    require(value["schema"] == ASSIGNMENT_SCHEMA and value["clip_length"] == 4, "assignment contract drift")
    require(isinstance(value["clips"], list) and value["clips"], "assignment clips missing")
    return value["clips"]


def _validate_assignments(
    clips: list[dict[str, Any]],
    training: dict[str, dict[str, Any]],
    holdout: dict[str, dict[str, Any]],
    excluded: set[str],
) -> dict[str, set[str]]:
    parents = {role: set() for role in ("train", "validation", "public_holdout")}
    frames: set[str] = set()
    for clip in clips:
        exact_fields(clip, {"clip_id", "role", "video_id", "parent_id", "frame_ids"}, "assignment clip")
        role = str(clip["role"])
        require(role in parents, "unknown role")
        ids = clip["frame_ids"]
        require(isinstance(ids, list) and len(ids) == 4, "clip must contain four frames")
        require(not (set(ids) & frames), "frame reused across clips")
        frames.update(ids)
        source = holdout if role == "public_holdout" else training
        rows = [source.get(str(frame_id)) for frame_id in ids]
        require(all(row is not None for row in rows), "assignment frame missing from role catalog")
        timestamps = []
        for row in rows:
            assert row is not None
            require(row["video_id"] == clip["video_id"] and row["parent_id"] == clip["parent_id"], "clip identity drift")
            timestamps.append(row["timestamp_ns"])
        require(all(0 < right - left <= 500_000_000 for left, right in zip(timestamps, timestamps[1:])), "non-native or over-gap clip timestamps")
        parent = str(clip["parent_id"])
        require(parent not in excluded, "legacy P1 parent overlap")
        parents[role].add(parent)
    require(parents["train"].isdisjoint(parents["validation"]), "train/validation parent overlap")
    require(parents["train"].isdisjoint(parents["public_holdout"]), "train/holdout parent overlap")
    require(parents["validation"].isdisjoint(parents["public_holdout"]), "validation/holdout parent overlap")
    return parents


def _public_manifest(clips: list[dict[str, Any]], holdout: dict[str, dict[str, Any]], protocol_sha: str) -> dict[str, Any]:
    output = []
    for clip in clips:
        if clip["role"] != "public_holdout":
            continue
        frames = []
        for frame_id in clip["frame_ids"]:
            row = holdout[frame_id]
            frames.append({
                "frame_id": row["frame_id"], "video_id": row["video_id"], "parent_id": row["parent_id"],
                "timestamp_ns": row["timestamp_ns"], "sealed_target_id": f"P3R01:{row['frame_id']}",
                "rgb_identity": row["rgb_identity"], "rgb_sha256": row["rgb_sha256"],
            })
        output.append({"clip_id": clip["clip_id"], "video_id": clip["video_id"], "parent_id": clip["parent_id"], "frames": frames})
    require(output, "public holdout assignment missing")
    return {"schema": ROLE_MANIFEST_SCHEMA, "protocol_sha256": protocol_sha, "role": "public_holdout", "outcomes_opened": False, "clips": output}


def _training_manifest(role: str, clips: list[dict[str, Any]], training: dict[str, dict[str, Any]], disagreements: dict[str, float], protocol_sha: str) -> tuple[dict[str, Any], dict[str, int]]:
    output = []
    counts = {name: 0 for name in TRANSITIONS}
    for clip in clips:
        if clip["role"] != role:
            continue
        frames = []
        for frame_id in clip["frame_ids"]:
            row = training[frame_id]
            require(frame_id in disagreements, "frozen A2 disagreement missing")
            frames.append({key: row[key] for key in (
                "frame_id", "video_id", "parent_id", "timestamp_ns", "rgb_identity", "rgb_sha256",
                "teacher_depth_ref", "teacher_depth_sha256", "teacher_timestamp_ns", "teacher_valid", "tof_valid",
                "clearance_m", "geometry_state", "geometry_target_valid",
            )} | {"frozen_a2_mean_abs_log_depth_disagreement": disagreements[frame_id]})
        for left, right in zip(frames, frames[1:]):
            for band in range(3):
                if left["geometry_target_valid"][band] and right["geometry_target_valid"][band]:
                    counts[f"{left['geometry_state'][band]}_TO_{right['geometry_state'][band]}"] += 1
        output.append({"clip_id": clip["clip_id"], "video_id": clip["video_id"], "parent_id": clip["parent_id"], "frames": frames})
    require(output, f"{role} assignment missing")
    require(all(value > 0 for value in counts.values()), f"{role} lacks nine-class transition support")
    return {"schema": ROLE_MANIFEST_SCHEMA, "protocol_sha256": protocol_sha, "role": role, "clips": output}, counts


def _weights(counts: dict[str, int], protocol_sha: str, train_manifest_sha: str) -> dict[str, Any]:
    beta = 0.999
    raw = [(1.0 - beta) / (1.0 - beta ** counts[name]) for name in TRANSITIONS]
    mean = sum(raw) / 9
    return {
        "schema": WEIGHT_SCHEMA, "protocol_sha256": protocol_sha, "train_manifest_sha256": train_manifest_sha,
        "transition_order": list(TRANSITIONS), "transition_counts": counts, "beta": beta,
        "formula": "(1-beta)/(1-beta^count), normalized to mean 1",
        "weights": [value / mean for value in raw], "holdout_used": False,
    }


def build(repo_root: Path, request: dict[str, Any], source_path: Path) -> None:
    exact_fields(request, {"schema", "operation", "protocol", "source_catalog", "assignment", "exclusion_ledger", "identity_lock", "disagreement_cache", "producer_sha256", "outputs"}, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    require(request["operation"] in {"freeze_identity", "finalize_training"}, "operation invalid")
    producer_sha = verify_producer_sha(request["producer_sha256"], source_path)
    _, protocol_sha = validate_protocol(repo_root, request["protocol"])
    catalog_path = verify_bound_file(repo_root, request["source_catalog"], "source catalog")
    assignment_path = verify_bound_file(repo_root, request["assignment"], "assignment")
    exclusion_path = verify_bound_file(repo_root, request["exclusion_ledger"], "exclusion ledger")
    exclusion = load_json(exclusion_path)
    require(exclusion.get("schema") == "blindassist_p3_r0_1_legacy_p1_ancestry_exclusion_ledger", "exclusion schema drift")
    require(exclusion.get("outcomes_read") is False, "legacy outcomes boundary violated")
    training, holdout = _load_catalog(repo_root, catalog_path)
    clips = _load_assignment(assignment_path)
    parents = _validate_assignments(clips, training, holdout, set(exclusion["parent_ids"]))
    input_sha = {"protocol": protocol_sha, "source_catalog": request["source_catalog"]["sha256"].upper(), "assignment": request["assignment"]["sha256"].upper(), "exclusion_ledger": request["exclusion_ledger"]["sha256"].upper()}
    if request["operation"] == "freeze_identity":
        require(request["identity_lock"] is None and request["disagreement_cache"] is None, "freeze_identity accepts no downstream assets")
        exact_fields(request["outputs"], {"identity_lock", "public_holdout_manifest", "receipt"}, "outputs")
        lock = {
            "schema": LOCK_SCHEMA, "protocol_sha256": protocol_sha,
            "source_catalog_sha256": input_sha["source_catalog"], "assignment_sha256": input_sha["assignment"],
            "exclusion_ledger_sha256": input_sha["exclusion_ledger"], "clip_length": 4,
            "parents_by_role": {role: sorted(values) for role, values in parents.items()},
            "clips": clips, "holdout_outcomes_opened": False,
        }
        public = _public_manifest(clips, holdout, protocol_sha)
        outputs = {
            "identity_lock": (str(request["outputs"]["identity_lock"]), pretty_bytes(lock)),
            "public_holdout_manifest": (str(request["outputs"]["public_holdout_manifest"]), pretty_bytes(public)),
        }
    else:
        identity_path = verify_bound_file(repo_root, request["identity_lock"], "identity lock")
        disagreement_path = verify_bound_file(repo_root, request["disagreement_cache"], "disagreement cache")
        lock = load_json(identity_path)
        require(lock.get("schema") == LOCK_SCHEMA and lock.get("clips") == clips, "identity lock drift")
        disagreements = {}
        for line in disagreement_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            exact_fields(row, {"schema", "frame_id", "mean_abs_log_depth_disagreement"}, "disagreement row")
            require(row["schema"] == DISAGREEMENT_SCHEMA, "disagreement row schema drift")
            value = float(row["mean_abs_log_depth_disagreement"])
            require(math.isfinite(value) and value >= 0 and row["frame_id"] not in disagreements, "invalid disagreement row")
            disagreements[row["frame_id"]] = value
        train_manifest, train_counts = _training_manifest("train", clips, training, disagreements, protocol_sha)
        validation_manifest, _ = _training_manifest("validation", clips, training, disagreements, protocol_sha)
        train_bytes = pretty_bytes(train_manifest)
        weight_receipt = _weights(train_counts, protocol_sha, sha256_bytes(train_bytes))
        exact_fields(request["outputs"], {"train_manifest", "validation_manifest", "class_weight_receipt", "receipt"}, "outputs")
        outputs = {
            "train_manifest": (str(request["outputs"]["train_manifest"]), train_bytes),
            "validation_manifest": (str(request["outputs"]["validation_manifest"]), pretty_bytes(validation_manifest)),
            "class_weight_receipt": (str(request["outputs"]["class_weight_receipt"]), pretty_bytes(weight_receipt)),
        }
        input_sha.update({"identity_lock": request["identity_lock"]["sha256"].upper(), "disagreement_cache": request["disagreement_cache"]["sha256"].upper()})
    receipt = output_receipt(schema=RECEIPT_SCHEMA, producer_sha256=producer_sha, request_sha256=request_sha256(request), input_sha256=input_sha, outputs=outputs)
    commit_outputs(repo_root, outputs=outputs, receipt_relative=str(request["outputs"]["receipt"]), receipt=receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), Path(__file__).resolve())


if __name__ == "__main__":
    main()
