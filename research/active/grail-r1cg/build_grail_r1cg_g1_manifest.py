#!/usr/bin/env python3
"""Freeze fresh train/Development rosters for R1C-G1 before pixel collection."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_grail_r1c_g1_manifest_v1"
ROSTER_KEY = "grail-r1cg-g1-active-multiview-appearance-v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _used_r1cl(manifest: dict[str, Any]) -> set[int]:
    return {
        int(row["house_index"])
        for role in ("train", "validation")
        for row in manifest["rosters"][role]
    }


def _used_g0(manifest: dict[str, Any]) -> set[int]:
    return {int(row["house_index"]) for row in manifest["rosters"]["validation"]}


def build_manifest(dataset: Path, r1cl_path: Path, g0_path: Path,
                   train_houses: int, validation_houses: int) -> dict[str, Any]:
    if train_houses < 1 or validation_houses < 1:
        raise ValueError("R1C-G1 requires non-empty train and Development rosters")
    r1cl = json.loads(r1cl_path.read_text(encoding="utf-8"))
    g0 = json.loads(g0_path.read_text(encoding="utf-8"))
    dataset_hash = sha256_file(dataset)
    if dataset_hash != r1cl["source"]["train_sha256"] or dataset_hash != g0["source"]["train_sha256"]:
        raise ValueError("R1C-G1 source does not match the frozen R1C-L/G0 source revision")
    excluded_r1cl = _used_r1cl(r1cl)
    excluded_g0 = _used_g0(g0)
    if excluded_r1cl & excluded_g0:
        raise ValueError("G0 roster unexpectedly overlaps R1C-L")
    excluded = excluded_r1cl | excluded_g0
    candidates: list[dict[str, Any]] = []
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in excluded:
                continue
            house_hash = canonical_sha256(json.loads(line))
            rank = hashlib.sha256(f"{ROSTER_KEY}:{index}:{house_hash}".encode("utf-8")).hexdigest()
            candidates.append({"rank": rank, "house_index": index, "house_sha256": house_hash})
    ranked = sorted(candidates, key=lambda row: row["rank"])
    needed = train_houses + validation_houses
    if len(ranked) < needed:
        raise ValueError(f"R1C-G1 source contains only {len(ranked)} eligible houses")
    train = ranked[:train_houses]
    validation = ranked[train_houses:needed]
    return {
        "schema": SCHEMA,
        "mode": "EXPLORE_DEVELOPMENT",
        "question": (
            "Can a fixed anchor-plus-left-plus-right reference scan reveal symmetry-breaking "
            "RGB/mask appearance that materially improves direct owner-canonical PRESERVE/FLIP "
            "prediction over a matched single-anchor arm?"
        ),
        "frozen_before_collection_pixels_or_model_outcome": True,
        "source": {
            "dataset": r1cl["source"]["dataset"],
            "dataset_revision": r1cl["source"]["dataset_revision"],
            "license": r1cl["source"]["license"],
            "train_sha256": dataset_hash,
            "ai2thor_release": r1cl["source"]["ai2thor_release"],
            "excluded_r1cl_train_validation_houses": len(excluded_r1cl),
            "excluded_g0_development_houses": len(excluded_g0),
        },
        "rosters": {"train": train, "validation": validation},
        "collection": {
            "reference_views": ["anchor", "left", "right"],
            "scan_frame": "anchor_camera_lateral_axis",
            "target_lateral_baseline_m": 0.30,
            "accepted_absolute_lateral_baseline_m": [0.20, 0.45],
            "maximum_absolute_longitudinal_drift_m": 0.20,
            "scans_per_owner_group_maximum": 6,
            "visible_siblings_minimum": 2,
            "visible_siblings_maximum": 4,
            "pair_requires_shared_physical_siblings_minimum": 2,
            "query_is_an_anchor_from_a_different_independently_built_scan": True,
            "triplet_order_is_not_a_model_input": True,
            "train_sample_range": [4000, 6000],
            "validation_sample_range": [800, 2000],
            "not_evaluable": "A scan missing either valid side view is omitted; anchor duplication is forbidden.",
        },
        "architecture": {
            "backbone": "facebook/dinov2-small pinned local weights",
            "shared_pair_encoder": (
                "DINOv2-S RGB tokens plus owner-union/sibling-centroid mask adapter and two "
                "bidirectional cross-attention blocks"
            ),
            "aggregation": "permutation-invariant concatenate(mean(evidence), max(evidence))",
            "head": "shared-form MLP with two PRESERVE/FLIP logits",
            "baseline_reference_count": 1,
            "challenger_reference_count": 3,
            "train_discriminative_pairs_only": True,
            "balanced_mode_and_object_type_sampler": True,
            "seeds": [1701, 2701],
            "epochs": 8,
            "checkpoint_selection": "maximum Development balanced accuracy independently within each arm/seed",
        },
        "primary_metrics": [
            "discriminative_balanced_accuracy",
            "flip_accuracy",
            "preserve_accuracy",
            "rescue",
            "collateral",
            "Drawer_balanced_accuracy",
            "Doorway_balanced_accuracy",
            "owner_group_macro_balanced_accuracy",
        ],
        "advance_if": {
            "balanced_accuracy_uplift_percentage_points_minimum_each_seed": 8.0,
            "both_object_types_mean_balanced_accuracy_uplift_positive": True,
            "rescue_exceeds_collateral_each_seed": True,
            "preserve_accuracy_drop_percentage_points_maximum_each_seed": 5.0,
        },
        "boundaries": [
            "FRESH_TRAIN_AND_DEVELOPMENT_HOUSES_DISJOINT_FROM_R1CL_AND_G0",
            "VIEW_SELECTION_USES_ANCHOR_CAMERA_FRAME_NOT_OWNER_YAW_OR_CANONICAL_SIGN",
            "MODEL_INPUT_IS_RGB_AND_MASKS_ONLY",
            "NO_CAMERA_YAW_OWNER_YAW_DEPTH_OBJECT_COORDINATES_OR_POSE_AT_INFERENCE",
            "NO_NEXT_BEST_VIEW_POLICY_OR_G0_FUSION",
            "NO_FINAL_TEST",
            "SYNTHETIC_PROCTHOR_DEVELOPMENT_ONLY",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--r1cl-manifest", type=Path, required=True)
    parser.add_argument("--g0-manifest", type=Path, required=True)
    parser.add_argument("--train-houses", type=int, default=96)
    parser.add_argument("--validation-houses", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset, args.r1cl_manifest, args.g0_manifest,
                              args.train_houses, args.validation_houses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": manifest["schema"],
        "train_houses": len(manifest["rosters"]["train"]),
        "validation_houses": len(manifest["rosters"]["validation"]),
        "output": str(args.output),
        "sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
