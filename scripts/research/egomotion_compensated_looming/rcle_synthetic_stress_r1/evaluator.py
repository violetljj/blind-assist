from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_low_reference_false_trigger_r1.temporal_confirmation import (
    apply_confirmation,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.evaluator import (
    _summarize,
    evaluate_algorithm_ledger,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.io_utils import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.truth import (
    evaluate_truth_pair,
)


def _normalize_algorithm_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(row))
    normalized["sequence_id"] = "<MATCHED>"
    normalized["window_id"] = "<MATCHED>"
    return normalized


def evaluate_stress_dataset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    scientific_outputs = (
        "algorithm_pair_ledger.jsonl",
        "latent_truth_pair_ledger.jsonl",
        "old_r1_comparison.jsonl",
        "observed_depth_truth_pair_ledger.jsonl",
        "scientific_summary.json",
    )
    if any((root / name).exists() for name in scientific_outputs):
        raise FileExistsError("STRESS_SCIENTIFIC_OUTPUT_ALREADY_EXISTS")
    from scripts.research.egomotion_compensated_looming.rcle_synthetic_stress_r1.validator import (
        validate_stress_dataset,
    )

    preflight = validate_stress_dataset(root)
    if preflight["status"] != "PASS":
        raise ValueError(
            f"STRESS_DATASET_PREFLIGHT_FAILED:{preflight['errors']}"
        )
    spec = read_json(root / "dataset_spec.json")
    base_spec = read_json(root / "base" / "dataset_spec.json")
    sequences = read_jsonl(root / "stress_sequence_manifest.jsonl")
    metadata = {row["sequence_id"]: row for row in sequences}
    algorithm_rows = evaluate_algorithm_ledger(root)
    algorithm_sha = write_jsonl(
        root / "algorithm_pair_ledger.jsonl", algorithm_rows
    )

    base_root = root / "base"
    base_frames = read_jsonl(base_root / "frame_manifest.jsonl")
    base_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_frames:
        base_by_sequence[row["sequence_id"]].append(row)
    base_truth: dict[str, list[dict[str, Any]]] = {}
    for sequence_id, rows in sorted(base_by_sequence.items()):
        selected = sorted(rows, key=lambda row: row["frame_index"])
        truth_rows: list[dict[str, Any]] = []
        for pair_index, (previous, current) in enumerate(
            zip(selected, selected[1:])
        ):
            truth_rows.append(
                {
                    "window_id": sequence_id,
                    "sequence_id": sequence_id,
                    "scene_id": previous["scene_id"],
                    "motion_id": previous["motion_id"],
                    "role": previous["role"],
                    "pair_index": pair_index,
                    "previous_timestamp_s": previous["timestamp"]["seconds"],
                    "current_timestamp_s": current["timestamp"]["seconds"],
                    "dt_s": (
                        current["timestamp"]["seconds"]
                        - previous["timestamp"]["seconds"]
                    ),
                    **evaluate_truth_pair(
                        base_root, previous, current
                    ),
                }
            )
        base_truth[sequence_id] = truth_rows

    truth_rows_all: list[dict[str, Any]] = []
    for sequence in sequences:
        for row in base_truth[sequence["base_sequence_id"]]:
            truth_rows_all.append(
                {
                    **row,
                    "window_id": sequence["sequence_id"],
                    "sequence_id": sequence["sequence_id"],
                }
            )
    truth_sha = write_jsonl(
        root / "latent_truth_pair_ledger.jsonl", truth_rows_all
    )

    joined = [
        {
            **row,
            "role": metadata[row["sequence_id"]]["role"],
            "scene_id": metadata[row["sequence_id"]]["scene_id"],
            "motion_id": metadata[row["sequence_id"]]["motion_id"],
        }
        for row in algorithm_rows
    ]
    comparison_rows = apply_confirmation(joined)
    comparison_sha = write_jsonl(
        root / "old_r1_comparison.jsonl", comparison_rows
    )

    condition_summaries: list[dict[str, Any]] = []
    for condition in spec["conditions"]:
        condition_id = condition["condition_id"]
        sequence_ids = {
            row["sequence_id"]
            for row in sequences
            if row["condition_id"] == condition_id
        }
        condition_truth = [
            row
            for row in truth_rows_all
            if row["sequence_id"] in sequence_ids
        ]
        condition_algorithm = [
            row
            for row in algorithm_rows
            if row["sequence_id"] in sequence_ids
        ]
        condition_comparison = [
            row
            for row in comparison_rows
            if row["window_id"] in sequence_ids
        ]
        summary = _summarize(
            base_spec,
            "development",
            condition_truth,
            condition_algorithm,
            condition_comparison,
        )
        condition_summaries.append(
            {
                "condition_id": condition_id,
                "family": condition["family"],
                "all_local_gates_pass": summary[
                    "all_local_gates_pass"
                ],
                "failed_sequences": [
                    row
                    for row in summary["sequences"]
                    if not row["pass"]
                ],
                "sequences": summary["sequences"],
                "rotation_approach_comparisons": summary[
                    "positive_rotation_approach_comparisons"
                ],
            }
        )

    algorithm_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in algorithm_rows:
        algorithm_by_sequence[row["sequence_id"]].append(row)
    depth_identity_checks: list[dict[str, Any]] = []
    for sequence in sequences:
        if sequence["condition_id"] != "depth_noise_gaussian_2pct":
            continue
        control_id = (
            sequence["base_sequence_id"] + "__matched_clean_control"
        )
        depth_rows = sorted(
            algorithm_by_sequence[sequence["sequence_id"]],
            key=lambda row: row["pair_index"],
        )
        control_rows = sorted(
            algorithm_by_sequence[control_id],
            key=lambda row: row["pair_index"],
        )
        identical = [
            _normalize_algorithm_row(row) for row in depth_rows
        ] == [_normalize_algorithm_row(row) for row in control_rows]
        depth_identity_checks.append(
            {
                "base_sequence_id": sequence["base_sequence_id"],
                "algorithm_ledger_identical_to_control": identical,
            }
        )
    if not all(
        row["algorithm_ledger_identical_to_control"]
        for row in depth_identity_checks
    ):
        raise RuntimeError("DEPTH_NOISE_ALGORITHM_FIREWALL_FAILED")

    observed_rows = read_jsonl(root / "observed_depth_manifest.jsonl")
    observed_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observed_rows:
        observed_by_base[row["base_sequence_id"]].append(row)
    observed_truth_diagnostics: list[dict[str, Any]] = []
    observed_truth_rows: list[dict[str, Any]] = []
    for base_sequence_id, depth_rows in sorted(observed_by_base.items()):
        depth_by_frame = {row["frame_index"]: row for row in depth_rows}
        base_selected = sorted(
            base_by_sequence[base_sequence_id],
            key=lambda row: row["frame_index"],
        )
        sequence_truth: list[dict[str, Any]] = []
        for pair_index, (previous, current) in enumerate(
            zip(base_selected, base_selected[1:])
        ):
            previous_observed = {
                **previous,
                "depth_npy_path": depth_by_frame[
                    previous["frame_index"]
                ]["depth_npy_path"],
            }
            current_observed = {
                **current,
                "depth_npy_path": depth_by_frame[
                    current["frame_index"]
                ]["depth_npy_path"],
            }
            result = {
                "base_sequence_id": base_sequence_id,
                "pair_index": pair_index,
                **evaluate_truth_pair(
                    root, previous_observed, current_observed
                ),
            }
            sequence_truth.append(result)
            observed_truth_rows.append(result)
        latent_values = [
            row["compensated_expansion_median_per_s"]
            for row in base_truth[base_sequence_id]
            if row["evaluable"]
        ]
        observed_values = [
            row["compensated_expansion_median_per_s"]
            for row in sequence_truth
            if row["evaluable"]
        ]
        latent_median = float(np.median(latent_values))
        observed_median = float(np.median(observed_values))
        observed_truth_diagnostics.append(
            {
                "base_sequence_id": base_sequence_id,
                "latent_truth_median_per_s": latent_median,
                "observed_noisy_depth_truth_median_per_s": observed_median,
                "absolute_difference_per_s": abs(
                    observed_median - latent_median
                ),
                "observed_evaluable_pair_count": len(observed_values),
            }
        )
    observed_truth_sha = write_jsonl(
        root / "observed_depth_truth_pair_ledger.jsonl",
        observed_truth_rows,
    )

    summary = {
        "schema": "rcle.synthetic_stress.scientific_summary.v1",
        "protocol_id": spec["protocol_id"],
        "status": "STRESS_R1_CHARACTERIZATION_COMPLETE",
        "parent_r0_terminal": spec["decision_source"],
        "condition_count": len(condition_summaries),
        "sequence_count": len(sequences),
        "pair_count": len(algorithm_rows),
        "condition_summaries": condition_summaries,
        "all_conditions_pass": all(
            row["all_local_gates_pass"]
            for row in condition_summaries
        ),
        "depth_noise_algorithm_firewall": depth_identity_checks,
        "observed_depth_truth_diagnostics": observed_truth_diagnostics,
        "pooled_rescue_used": False,
        "r0_rescue_or_mutation_used": False,
        "authority_ceiling": spec["authority"]["maximum_claim"],
        "dataset_receipt_sha256": sha256_file(root / "receipt.json"),
        "preflight_validation_status": preflight["status"],
        "ledger_sha256": {
            "algorithm": algorithm_sha,
            "latent_truth": truth_sha,
            "old_r1_comparison": comparison_sha,
            "observed_depth_truth": observed_truth_sha,
        },
    }
    write_json(root / "scientific_summary.json", summary)
    return summary
