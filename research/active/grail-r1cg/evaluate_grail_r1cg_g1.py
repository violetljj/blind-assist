#!/usr/bin/env python3
"""Apply the frozen R1C-G1 balanced-accuracy Development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(root: Path, arm: str, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_root = root / arm / f"seed-{seed}"
    result = json.loads((run_root / "result.json").read_text(encoding="utf-8"))
    predictions = json.loads((run_root / "predictions.json").read_text(encoding="utf-8"))
    if result["arm"] != arm or int(result["seed"]) != seed:
        raise ValueError("R1C-G1 training result identity mismatch")
    if sha256_file(run_root / "predictions.json") != result["predictions_sha256"]:
        raise ValueError("R1C-G1 prediction hash mismatch")
    return result, predictions


def evaluate(manifest: dict[str, Any], validation: dict[str, Any], training_root: Path) -> dict[str, Any]:
    seeds = [int(value) for value in manifest["architecture"]["seeds"]]
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        baseline_result, baseline_predictions = _load_run(training_root, "b1_single", seed)
        challenger_result, challenger_predictions = _load_run(training_root, "g1_triplet", seed)
        baseline_by_id = {row["sample_id"]: row for row in baseline_predictions}
        challenger_by_id = {row["sample_id"]: row for row in challenger_predictions}
        if set(baseline_by_id) != set(challenger_by_id):
            raise ValueError(f"R1C-G1 arm sample mismatch for seed {seed}")
        if any(baseline_by_id[key]["truth"] != challenger_by_id[key]["truth"] for key in baseline_by_id):
            raise ValueError(f"R1C-G1 arm truth mismatch for seed {seed}")
        rescue = collateral = 0
        for sample_id, baseline in baseline_by_id.items():
            challenger = challenger_by_id[sample_id]
            baseline_correct = baseline["prediction"] == baseline["truth"]
            challenger_correct = challenger["prediction"] == challenger["truth"]
            rescue += int(not baseline_correct and challenger_correct)
            collateral += int(baseline_correct and not challenger_correct)
        baseline_metrics = baseline_result["validation"]
        challenger_metrics = challenger_result["validation"]
        uplift_pp = 100.0 * (
            challenger_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
        )
        preserve_drop_pp = 100.0 * (
            baseline_metrics["by_mode"]["PRESERVE"]["accuracy"]
            - challenger_metrics["by_mode"]["PRESERVE"]["accuracy"]
        )
        by_type_uplift = {
            object_type: 100.0 * (
                challenger_metrics["by_type"][object_type]["balanced_accuracy"]
                - baseline_metrics["by_type"][object_type]["balanced_accuracy"]
            )
            for object_type in ("Doorway", "Drawer")
        }
        seed_rows.append({
            "seed": seed,
            "baseline": baseline_metrics,
            "challenger": challenger_metrics,
            "balanced_accuracy_uplift_percentage_points": uplift_pp,
            "preserve_accuracy_drop_percentage_points": preserve_drop_pp,
            "rescue": rescue,
            "collateral": collateral,
            "net_rescue": rescue - collateral,
            "by_type_balanced_accuracy_uplift_percentage_points": by_type_uplift,
            "baseline_best_epoch": baseline_result["best_epoch"],
            "challenger_best_epoch": challenger_result["best_epoch"],
            "baseline_checkpoint_sha256": baseline_result["checkpoint_sha256"],
            "challenger_checkpoint_sha256": challenger_result["checkpoint_sha256"],
        })
    threshold = float(manifest["advance_if"]["balanced_accuracy_uplift_percentage_points_minimum_each_seed"])
    preserve_limit = float(manifest["advance_if"]["preserve_accuracy_drop_percentage_points_maximum_each_seed"])
    each_seed_uplift = all(row["balanced_accuracy_uplift_percentage_points"] >= threshold for row in seed_rows)
    each_seed_net_rescue = all(row["rescue"] > row["collateral"] for row in seed_rows)
    each_seed_preserve = all(row["preserve_accuracy_drop_percentage_points"] <= preserve_limit for row in seed_rows)
    mean_by_type_uplift = {
        object_type: sum(row["by_type_balanced_accuracy_uplift_percentage_points"][object_type]
                         for row in seed_rows) / len(seed_rows)
        for object_type in ("Doorway", "Drawer")
    }
    both_types_positive = all(value > 0.0 for value in mean_by_type_uplift.values())
    passed = each_seed_uplift and each_seed_net_rescue and each_seed_preserve and both_types_positive
    samples = validation["samples"]
    discriminative = [row for row in samples if len(row["valid_slot_modes"]) == 1]
    return {
        "schema": "blindassist_grail_r1c_g1_development_result_v1",
        "state": "ADVANCE_G1_ACTIVE_MULTIVIEW_APPEARANCE" if passed else "STOP_G1_ACTIVE_MULTIVIEW_APPEARANCE",
        "gate": "DEVELOPMENT_GATE_MET" if passed else "DEVELOPMENT_GATE_NOT_MET",
        "final": "NO_FINAL_TEST",
        "cohort": {
            "houses": int(validation["houses"]),
            "samples": len(samples),
            "discriminative_samples": len(discriminative),
            "flip_only_samples": sum(row["valid_slot_modes"] == ["FLIP"] for row in discriminative),
            "preserve_only_samples": sum(row["valid_slot_modes"] == ["PRESERVE"] for row in discriminative),
            "ambiguous_samples": sum(len(row["valid_slot_modes"]) > 1 for row in samples),
            "Drawer_samples": sum(row["object_type"] == "Drawer" for row in samples),
            "Doorway_samples": sum(row["object_type"] == "Doorway" for row in samples),
        },
        "seeds": seed_rows,
        "decision": {
            "minimum_balanced_accuracy_uplift_percentage_points_each_seed": threshold,
            "each_seed_uplift_passed": each_seed_uplift,
            "rescue_exceeds_collateral_each_seed": each_seed_net_rescue,
            "maximum_preserve_accuracy_drop_percentage_points_each_seed": preserve_limit,
            "preserve_drop_passed": each_seed_preserve,
            "mean_by_type_balanced_accuracy_uplift_percentage_points": mean_by_type_uplift,
            "both_object_types_mean_positive_uplift": both_types_positive,
            "passed": passed,
        },
        "boundaries": {
            "fresh_from_r1cl_and_g0": True,
            "view_selection_used_owner_yaw_or_canonical_sign": False,
            "model_inputs": ["RGB", "owner_union_mask", "sibling_centroid_mask"],
            "camera_or_owner_pose_used_at_inference": False,
            "next_best_view_policy": False,
            "g0_fusion": False,
            "final_test_accessed": False,
            "claim_ceiling": (
                "Fresh house-disjoint synthetic ProcTHOR Development evidence for the fixed "
                "three-view RGB/mask appearance acquisition and direct permutation formulation only."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-collection", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_collection.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blindassist_grail_r1c_g1_manifest_v1":
        raise ValueError("R1C-G1 manifest schema mismatch")
    if validation.get("schema") != "blindassist_grail_r1c_g1_collection_v1":
        raise ValueError("R1C-G1 validation collection schema mismatch")
    result = evaluate(manifest, validation, args.training_root)
    result["manifest_sha256"] = sha256_file(args.manifest)
    result["validation_collection_sha256"] = sha256_file(args.validation_collection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
