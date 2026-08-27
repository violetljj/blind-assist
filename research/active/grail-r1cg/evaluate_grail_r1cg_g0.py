#!/usr/bin/env python3
"""Evaluate the fixed source-native relative-yaw transport rule for R1C-G0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_transport_mode(relative_yaw_degrees: float) -> str:
    return "PRESERVE" if math.cos(math.radians(relative_yaw_degrees)) >= 0.0 else "FLIP"


def _metric_row(rows: list[dict[str, Any]], predictor: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    correct = sum(predictor(row) in row["valid_slot_modes"] for row in rows)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows) if rows else None}


def evaluate(collection: dict[str, Any]) -> dict[str, Any]:
    pairs = collection["pairs"]
    discriminative = [row for row in pairs if len(row["valid_slot_modes"]) == 1]
    flip_only = [row for row in discriminative if row["valid_slot_modes"] == ["FLIP"]]
    preserve_only = [row for row in discriminative if row["valid_slot_modes"] == ["PRESERVE"]]
    ambiguous = [row for row in pairs if len(row["valid_slot_modes"]) > 1]
    predictors = {
        "preserve_prior": lambda row: "PRESERVE",
        "g0_pose_transport": lambda row: pose_transport_mode(float(row["relative_yaw_label_degrees"])),
    }
    arms: dict[str, Any] = {}
    for name, predictor in predictors.items():
        row = {
            "all_pairs": _metric_row(pairs, predictor),
            "discriminative": _metric_row(discriminative, predictor),
            "flip_only": _metric_row(flip_only, predictor),
            "preserve_only": _metric_row(preserve_only, predictor),
            "ambiguous": _metric_row(ambiguous, predictor),
            "by_type": {},
        }
        for object_type in sorted({pair["object_type"] for pair in pairs}):
            type_rows = [pair for pair in discriminative if pair["object_type"] == object_type]
            row["by_type"][object_type] = _metric_row(type_rows, predictor)
        arms[name] = row
    pose = arms["g0_pose_transport"]["discriminative"]
    prior = arms["preserve_prior"]["discriminative"]
    uplift_pp = 100.0 * (pose["accuracy"] - prior["accuracy"])
    by_type_uplift = {
        object_type: 100.0 * (
            arms["g0_pose_transport"]["by_type"][object_type]["accuracy"]
            - arms["preserve_prior"]["by_type"][object_type]["accuracy"]
        )
        for object_type in arms["g0_pose_transport"]["by_type"]
        if arms["g0_pose_transport"]["by_type"][object_type]["total"] > 0
    }
    advance = (
        uplift_pp >= 8.0
        and arms["g0_pose_transport"]["flip_only"]["accuracy"] is not None
        and arms["g0_pose_transport"]["flip_only"]["accuracy"] >= 0.65
        and bool(by_type_uplift)
        and all(value > 0.0 for value in by_type_uplift.values())
    )
    return {
        "schema": "blindassist_grail_r1c_g0_development_result_v1",
        "state": "ADVANCE_TO_ACTIVE_MULTIVIEW" if advance else "STOP_G0_POSE_TRANSPORT",
        "cohort": {
            "houses": int(collection["houses"]),
            "pairs": len(pairs),
            "discriminative_pairs": len(discriminative),
            "flip_only_pairs": len(flip_only),
            "preserve_only_pairs": len(preserve_only),
            "ambiguous_pairs": len(ambiguous),
        },
        "arms": arms,
        "decision": {
            "discriminative_uplift_over_preserve_percentage_points": uplift_pp,
            "by_type_uplift_percentage_points": by_type_uplift,
            "minimum_uplift_percentage_points": 8.0,
            "minimum_flip_only_accuracy": 0.65,
            "both_object_types_positive_uplift": bool(by_type_uplift) and all(
                value > 0.0 for value in by_type_uplift.values()
            ),
            "passed": advance,
        },
        "boundaries": {
            "source_native_relative_camera_yaw": True,
            "owner_pose_or_coordinates_used": False,
            "model_training": False,
            "threshold_sweep": False,
            "final_test_accessed": False,
            "claim_ceiling": (
                "Synthetic ProcTHOR fresh house-disjoint Development mechanism evidence only; "
                "source-native cross-session relative camera yaw is an oracle, not yet a deployable input."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blindassist_grail_r1c_g0_manifest_v1":
        raise ValueError("R1C-G0 manifest schema mismatch")
    collection = json.loads(args.collection.read_text(encoding="utf-8"))
    if collection.get("schema") != "blindassist_grail_r1c_g0_collection_v1":
        raise ValueError("R1C-G0 collection schema mismatch")
    result = evaluate(collection)
    result["manifest_sha256"] = sha256_file(args.manifest)
    result["collection_sha256"] = sha256_file(args.collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
