#!/usr/bin/env python3
"""Produce aggregate-only P3 R0.2.1 sealed holdout coverage evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from p3_r0_2_1_sealing_common import STATES, TRANSITIONS, exact_fields, exclusive_write, load_json, materialization_receipt, pretty_bytes, require, sha256_file, verify_bound_file


REQUEST_SCHEMA = "blindassist_p3_r0_2_1_aggregate_coverage_request"
COVERAGE_SCHEMA = "blindassist_p3_r0_2_1_sealed_coverage_receipt"
RECEIPT_SCHEMA = "blindassist_p3_r0_2_1_coverage_materialization_receipt"


def coverage(private: dict[str, Any]) -> tuple[dict[str, int], dict[str, int], int, int]:
    geometry = {name: 0 for name in TRANSITIONS}
    key = {"CLEAR_TO_OCCUPIED": 0, "OCCUPIED_TO_CLEAR": 0, "KNOWN_TO_UNKNOWN": 0, "UNKNOWN_TO_KNOWN": 0}
    parents, evaluable = set(), 0
    for clip in private["clips"]:
        parents.add(clip["parent_id"])
        clip_evaluable = True
        for left, right in zip(clip["frames"], clip["frames"][1:]):
            for band_index in range(3):
                left_band, right_band = left["bands"][band_index], right["bands"][band_index]
                if not left_band["geometry_target_valid"] or not right_band["geometry_target_valid"]:
                    clip_evaluable = False
                    continue
                a, b = left_band["geometry_state"], right_band["geometry_state"]
                require(a in STATES and b in STATES, "geometry state drift")
                geometry[f"{a}_TO_{b}"] += 1
                if a == "CLEAR" and b == "OCCUPIED": key["CLEAR_TO_OCCUPIED"] += 1
                if a == "OCCUPIED" and b == "CLEAR": key["OCCUPIED_TO_CLEAR"] += 1
                if a != "UNKNOWN_GROUND" and b == "UNKNOWN_GROUND": key["KNOWN_TO_UNKNOWN"] += 1
                if a == "UNKNOWN_GROUND" and b != "UNKNOWN_GROUND": key["UNKNOWN_TO_KNOWN"] += 1
        evaluable += int(clip_evaluable)
    return key, geometry, evaluable, len(parents)


def build(repo_root: Path, request: dict[str, Any], source_path: Path) -> None:
    exact_fields(request, {"schema", "protocol", "public_holdout_manifest", "private_targets", "sealed_target_bundle", "bundle_receipt", "producer_sha256", "outputs"}, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    require(sha256_file(source_path) == request["producer_sha256"], "producer SHA mismatch")
    protocol = verify_bound_file(repo_root, request["protocol"], "protocol")
    public_path = verify_bound_file(repo_root, request["public_holdout_manifest"], "public manifest")
    private_path = verify_bound_file(repo_root, request["private_targets"], "private targets")
    bundle_path = verify_bound_file(repo_root, request["sealed_target_bundle"], "sealed bundle")
    bundle_receipt_path = verify_bound_file(repo_root, request["bundle_receipt"], "bundle receipt")
    public, private, bundle_receipt = load_json(public_path), load_json(private_path), load_json(bundle_receipt_path)
    require(public.get("outcomes_opened") is False, "public holdout opened")
    require(bundle_receipt["outputs"]["sealed_target_bundle"]["sha256"] == sha256_file(bundle_path), "bundle receipt mismatch")
    public_ids = {f["sealed_target_id"] for c in public["clips"] for f in c["frames"]}
    private_ids = {f["sealed_target_id"] for c in private["clips"] for f in c["frames"]}
    require(public_ids == private_ids, "coverage identity mismatch")
    key, geometry, evaluable, parent_count = coverage(private)
    terminal = "P3_R0_2_1_SEALED_COVERAGE_VERIFIED" if evaluable >= 32 and parent_count >= 8 and all(value >= 8 for value in key.values()) else "P3_R0_2_SEALED_HOLDOUT_NOT_EVALUABLE_NO_COHORT_SUBSTITUTION"
    value = {"schema": COVERAGE_SCHEMA, "terminal": terminal, "protocol_sha256": sha256_file(protocol), "identity_manifest_sha256": sha256_file(public_path), "sealed_bundle_sha256": sha256_file(bundle_path), "coverage_producer_sha256": request["producer_sha256"], "label_rows_disclosed": False, "evaluable_clip_count": evaluable, "video_parent_count": parent_count, "key_transition_counts": key, "geometry_transition_counts": geometry, "holdout_outcomes_opened": False, "model_loaded": False, "training_started": False}
    exact_fields(request["outputs"], {"coverage_receipt", "materialization_receipt"}, "outputs")
    payload = pretty_bytes(value)
    outputs = {"sealed_coverage_receipt": (request["outputs"]["coverage_receipt"], payload)}
    receipt = materialization_receipt(RECEIPT_SCHEMA, request["producer_sha256"], {"protocol": sha256_file(protocol), "public_holdout_manifest": sha256_file(public_path), "private_targets": sha256_file(private_path), "sealed_target_bundle": sha256_file(bundle_path), "bundle_receipt": sha256_file(bundle_receipt_path)}, outputs)
    exclusive_write(repo_root / request["outputs"]["coverage_receipt"], payload)
    exclusive_write(repo_root / request["outputs"]["materialization_receipt"], pretty_bytes(receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), Path(__file__).resolve())


if __name__ == "__main__":
    main()
