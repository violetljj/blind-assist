#!/usr/bin/env python3
"""Attest aggregate P3 holdout coverage without opening the sealed bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from p3_r0_1_asset_common import (
    STATES,
    TRANSITIONS,
    commit_outputs,
    exact_fields,
    load_json,
    output_receipt,
    pretty_bytes,
    request_sha256,
    require,
    sha256_file,
    validate_protocol,
    verify_bound_file,
    verify_producer_sha,
)


REQUEST_SCHEMA = "blindassist_p3_r0_1_sealed_coverage_request"
PRIVATE_SCHEMA = "blindassist_p3_r0_1_private_holdout_targets"
COVERAGE_SCHEMA = "blindassist_dav2_temporal_392_student_p3_r0_1_sealed_coverage_receipt"
OUTPUT_RECEIPT_SCHEMA = "blindassist_p3_r0_1_sealed_coverage_materialization_receipt"


def _coverage(private: dict[str, Any]) -> tuple[dict[str, int], dict[str, int], int, int]:
    require(private.get("schema") == PRIVATE_SCHEMA, "private target schema drift")
    geometry = {name: 0 for name in TRANSITIONS}
    key = {
        "CLEAR_TO_OCCUPIED": 0,
        "OCCUPIED_TO_CLEAR": 0,
        "KNOWN_TO_UNKNOWN_GROUND": 0,
        "UNKNOWN_GROUND_TO_KNOWN": 0,
    }
    parents = set()
    evaluable = 0
    for clip in private.get("clips", []):
        require(isinstance(clip, dict) and len(clip.get("frames", [])) == 4, "private clip invalid")
        parents.add(str(clip["parent_id"]))
        frames = clip["frames"]
        clip_evaluable = True
        for left, right in zip(frames, frames[1:]):
            for band in range(3):
                valid = left["geometry_target_valid"][band] and right["geometry_target_valid"][band]
                if not valid:
                    clip_evaluable = False
                    continue
                left_state = left["geometry_state"][band]
                right_state = right["geometry_state"][band]
                require(left_state in STATES and right_state in STATES, "private geometry state invalid")
                geometry[f"{left_state}_TO_{right_state}"] += 1
                if left_state == "CLEAR" and right_state == "OCCUPIED":
                    key["CLEAR_TO_OCCUPIED"] += 1
                if left_state == "OCCUPIED" and right_state == "CLEAR":
                    key["OCCUPIED_TO_CLEAR"] += 1
                if left_state != "UNKNOWN_GROUND" and right_state == "UNKNOWN_GROUND":
                    key["KNOWN_TO_UNKNOWN_GROUND"] += 1
                if left_state == "UNKNOWN_GROUND" and right_state != "UNKNOWN_GROUND":
                    key["UNKNOWN_GROUND_TO_KNOWN"] += 1
        if clip_evaluable:
            evaluable += 1
    return key, geometry, evaluable, len(parents)


def build(repo_root: Path, request: dict[str, Any], source_path: Path) -> None:
    exact_fields(request, {"schema", "protocol", "public_holdout_manifest", "private_targets", "sealed_target_bundle", "bundle_production_receipt", "producer_sha256", "outputs"}, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    producer_sha = verify_producer_sha(request["producer_sha256"], source_path)
    _, protocol_sha = validate_protocol(repo_root, request["protocol"])
    public_path = verify_bound_file(repo_root, request["public_holdout_manifest"], "public holdout manifest")
    private_path = verify_bound_file(repo_root, request["private_targets"], "private targets")
    bundle_path = verify_bound_file(repo_root, request["sealed_target_bundle"], "sealed target bundle")
    bundle_receipt_path = verify_bound_file(repo_root, request["bundle_production_receipt"], "bundle production receipt")
    public = load_json(public_path)
    require(public.get("role") == "public_holdout" and public.get("outcomes_opened") is False, "public holdout opened")
    bundle_receipt = load_json(bundle_receipt_path)
    require(bundle_receipt.get("outputs", {}).get("sealed_target_bundle", {}).get("sha256") == sha256_file(bundle_path), "bundle production receipt mismatch")
    private = load_json(private_path)
    public_ids = {frame["sealed_target_id"] for clip in public["clips"] for frame in clip["frames"]}
    private_ids = {frame["sealed_target_id"] for clip in private["clips"] for frame in clip["frames"]}
    require(public_ids == private_ids, "coverage identity mismatch")
    key, geometry, evaluable, parent_count = _coverage(private)
    require(evaluable >= 32, "fewer than 32 evaluable clips")
    require(parent_count >= 8, "fewer than 8 video parents")
    require(all(value >= 8 for value in key.values()), "key transition coverage insufficient")
    receipt_value = {
        "schema": COVERAGE_SCHEMA,
        "status": "SEALED_COVERAGE_VERIFIED",
        "protocol_sha256": protocol_sha,
        "identity_manifest_sha256": request["public_holdout_manifest"]["sha256"].upper(),
        "sealed_bundle_sha256": request["sealed_target_bundle"]["sha256"].upper(),
        "coverage_producer_sha256": producer_sha,
        "created_before_training_activation": True,
        "label_rows_disclosed": False,
        "evaluable_clip_count": evaluable,
        "video_parent_count": parent_count,
        "key_transition_counts": key,
        "geometry_transition_counts": geometry,
    }
    exact_fields(request["outputs"], {"coverage_receipt", "materialization_receipt"}, "outputs")
    outputs = {"sealed_coverage_receipt": (str(request["outputs"]["coverage_receipt"]), pretty_bytes(receipt_value))}
    materialization = output_receipt(
        schema=OUTPUT_RECEIPT_SCHEMA,
        producer_sha256=producer_sha,
        request_sha256=request_sha256(request),
        input_sha256={
            "protocol": protocol_sha,
            "public_holdout_manifest": request["public_holdout_manifest"]["sha256"].upper(),
            "private_targets": request["private_targets"]["sha256"].upper(),
            "sealed_target_bundle": request["sealed_target_bundle"]["sha256"].upper(),
            "bundle_production_receipt": request["bundle_production_receipt"]["sha256"].upper(),
        },
        outputs=outputs,
    )
    commit_outputs(repo_root, outputs=outputs, receipt_relative=str(request["outputs"]["materialization_receipt"]), receipt=materialization)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), Path(__file__).resolve())


if __name__ == "__main__":
    main()
