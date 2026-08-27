#!/usr/bin/env python3
"""Freeze a fresh ProcTHOR Development roster for the R1C-G0 pose probe."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_grail_r1c_g0_manifest_v1"
ROSTER_KEY = "grail-r1cg-g0-development-v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def used_train_indices(r1cl_manifest: dict[str, Any]) -> set[int]:
    return {
        int(row["house_index"])
        for role in ("train", "validation")
        for row in r1cl_manifest["rosters"][role]
    }


def build_manifest(dataset: Path, r1cl_manifest_path: Path, houses: int) -> dict[str, Any]:
    if houses < 1:
        raise ValueError("R1C-G0 requires at least one Development house")
    r1cl = json.loads(r1cl_manifest_path.read_text(encoding="utf-8"))
    dataset_hash = sha256_file(dataset)
    if dataset_hash != r1cl["source"]["train_sha256"]:
        raise ValueError("R1C-G0 source dataset does not match the frozen R1C-L source revision")
    excluded = used_train_indices(r1cl)
    candidates: list[dict[str, Any]] = []
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in excluded:
                continue
            house_hash = canonical_sha256(json.loads(line))
            rank = hashlib.sha256(f"{ROSTER_KEY}:{index}:{house_hash}".encode("utf-8")).hexdigest()
            candidates.append({"rank": rank, "house_index": index, "house_sha256": house_hash})
    roster = sorted(candidates, key=lambda row: row["rank"])[:houses]
    if len(roster) != houses:
        raise ValueError(f"R1C-G0 source contains only {len(roster)} eligible houses")
    return {
        "schema": SCHEMA,
        "mode": "EXPLORE_DEVELOPMENT",
        "question": (
            "Does source-native reference-to-query camera yaw provide the missing information "
            "needed to identify canonical PRESERVE versus FLIP permutations?"
        ),
        "frozen_before_collection_or_outcome": True,
        "source": {
            "dataset": r1cl["source"]["dataset"],
            "dataset_revision": r1cl["source"]["dataset_revision"],
            "license": r1cl["source"]["license"],
            "train_sha256": dataset_hash,
            "excluded_r1cl_train_validation_houses": len(excluded),
        },
        "rosters": {"validation": roster},
        "collection": {
            "views_per_owner_group_maximum": 8,
            "visible_siblings_minimum": 2,
            "visible_siblings_maximum": 4,
            "native_owner_subgroup": r1cl["collection"]["native_owner_subgroup"],
            "pair_requires_shared_physical_siblings_minimum": 2,
            "ordered_reference_query_pairs": True,
            "train_pair_range": [0, 1000000],
            "validation_pair_range": [1, 1000000],
            "sampling": r1cl["collection"]["sampling"],
        },
        "arms": {
            "preserve_prior": "Always predict PRESERVE.",
            "g0_pose_transport": (
                "Predict PRESERVE when cos(source-native query camera yaw minus reference "
                "camera yaw) is non-negative; otherwise predict FLIP. The 90-degree boundary "
                "is fixed geometrically and is not tuned on Development outcomes."
            ),
        },
        "primary_metrics": [
            "discriminative_slot_accuracy",
            "flip_only_accuracy",
            "preserve_only_accuracy",
            "Doorway_accuracy",
            "Drawer_accuracy",
        ],
        "advance_if": {
            "overall_uplift_over_preserve_percentage_points_minimum": 8.0,
            "flip_only_accuracy_minimum": 0.65,
            "both_object_types_positive_uplift": True,
        },
        "boundaries": [
            "FRESH_HOUSE_DISJOINT_FROM_R1CL_TRAIN_AND_VALIDATION",
            "SOURCE_NATIVE_RELATIVE_CAMERA_YAW_IS_MECHANISM_ORACLE_NOT_PRODUCT_INPUT",
            "NO_OWNER_YAW_OWNER_POSITION_OBJECT_COORDINATES_DEPTH_OR_FINAL_TEST",
            "NO_THRESHOLD_SWEEP_OR_MODEL_TRAINING",
            "SYNTHETIC_PROCTHOR_DEVELOPMENT_ONLY",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--r1cl-manifest", type=Path, required=True)
    parser.add_argument("--houses", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset, args.r1cl_manifest, args.houses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": manifest["schema"],
        "houses": len(manifest["rosters"]["validation"]),
        "output": str(args.output),
        "sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
