#!/usr/bin/env python3
"""Test budgeted UNKNOWN deferral using the frozen D3R5 risk score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.research.hftf.deployment.depthart.analyze_depthart_d3r5_parent_relative_veto import (
    load_dataset,
    parent_grid_relative_features,
    predict,
)
from scripts.research.hftf.deployment.depthart.run_depthart_d3r4_selective_router_canary import (
    CertificateHead,
    STATE_CLEAR,
    STATE_OCCUPIED,
    STATE_UNKNOWN,
    atomic_json,
    metrics,
    require,
    sha256_file,
)


BUDGET_CANDIDATES = (0.005, 0.01, 0.015, 0.02)
CELLS_PER_PARENT = 2700


def load_frozen_score(
    checkpoint: dict[str, Any]
) -> tuple[CertificateHead, np.ndarray, np.ndarray]:
    require(
        checkpoint.get("schema")
        == "blindassist_depthart_d3r5_parent_relative_veto_checkpoint_v1",
        "D3R5 checkpoint schema drift",
    )
    head = CertificateHead().to(dtype=torch.float64)
    head.load_state_dict({
        name: torch.as_tensor(value, dtype=torch.float64)
        for name, value in checkpoint["head"].items()
    })
    return (
        head,
        np.asarray(checkpoint["mean"], dtype=np.float64),
        np.asarray(checkpoint["std"], dtype=np.float64),
    )


def budgeted_deferral(
    dataset: dict[str, np.ndarray], probabilities: np.ndarray, budget_fraction: float
) -> tuple[np.ndarray, dict[str, Any]]:
    require(0.0 < budget_fraction <= 0.02, "invalid deferral budget")
    action = np.zeros(len(probabilities), dtype=bool)
    per_parent_budget = int(np.floor(CELLS_PER_PARENT * budget_fraction))
    require(per_parent_budget > 0, "empty parent deferral budget")
    for parent in sorted(np.unique(dataset["parent_index"])):
        domain = np.flatnonzero(
            (dataset["parent_index"] == parent)
            & dataset["hard_evidence"]
            & dataset["source_available"]
            & (dataset["baseline_state"] == STATE_CLEAR)
        )
        count = min(per_parent_budget, len(domain))
        if count:
            order = np.argsort(-probabilities[domain], kind="stable")
            action[domain[order[:count]]] = True
    result = dataset["baseline_state"].copy()
    result[action] = STATE_UNKNOWN
    truth = dataset["truth_state"]
    return result, {
        "budget_fraction_of_all_cells_per_parent": budget_fraction,
        "maximum_cells_per_parent": per_parent_budget,
        "deferred_cell_count": int(action.sum()),
        "deferred_truth_occupied_count": int(
            np.sum(action & (truth == STATE_OCCUPIED))
        ),
        "deferred_truth_clear_count": int(np.sum(action & (truth == STATE_CLEAR))),
        "deferred_truth_unknown_count": int(
            np.sum(action & (truth == STATE_UNKNOWN))
        ),
        "occupied_action_count": 0,
        "clear_action_count": 0,
    }


def evaluate(
    dataset: dict[str, np.ndarray], probabilities: np.ndarray, budget_fraction: float
) -> dict[str, Any]:
    baseline = metrics(dataset, dataset["baseline_state"])
    states, actions = budgeted_deferral(dataset, probabilities, budget_fraction)
    candidate = metrics(dataset, states)
    base = baseline["pooled"]
    cand = candidate["pooled"]
    return {
        "budget_fraction": budget_fraction,
        "actions": actions,
        "baseline": baseline,
        "candidate": candidate,
        "false_clear_all_known_improvement": (
            base["false_clear_all_known"] - cand["false_clear_all_known"]
        ),
        "false_block_given_clear_improvement": (
            base["false_block_given_clear"] - cand["false_block_given_clear"]
        ),
        "known_coverage_decrease": (
            base["known_coverage_all_cells"] - cand["known_coverage_all_cells"]
        ),
    }


def select_budget(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["false_clear_all_known_improvement"] >= 0.01
        and row["false_block_given_clear_improvement"] >= 0.0
        and row["known_coverage_decrease"] <= 0.0200000001
    ]
    require(eligible, "TRAIN has no useful deferral budget")
    return max(
        eligible,
        key=lambda row: (
            row["false_clear_all_known_improvement"],
            -row["known_coverage_decrease"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3r4-root", type=Path, required=True)
    parser.add_argument("--d3r5-discovery-root", type=Path, required=True)
    parser.add_argument("--d3r5-confirmation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_root.exists(), f"fresh output root exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    d3r5_confirmation_path = args.d3r5_confirmation_root / "result.json"
    d3r5_confirmation = json.loads(d3r5_confirmation_path.read_text(encoding="utf-8"))
    require(
        d3r5_confirmation.get("status")
        == "D3R5_PARENT_RELATIVE_ZERO_FALSE_BLOCK_VETO_FRESH_CONFIRMATION_FAIL",
        "D3R5 fresh failure binding drift",
    )
    checkpoint_path = args.d3r5_discovery_root / "veto-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    head, mean, std = load_frozen_score(checkpoint)
    paths = {
        "TRAIN": args.d3r4_root / "train-dataset.npz",
        "REUSED_DEVELOPMENT": args.d3r4_root / "development-dataset.npz",
        "REUSED_D3R5_CONFIRMATION": args.d3r5_confirmation_root
        / "confirmation-dataset.npz",
    }
    datasets = {role: load_dataset(path) for role, path in paths.items()}
    probabilities = {
        role: predict(parent_grid_relative_features(dataset), head, mean, std)
        for role, dataset in datasets.items()
    }
    search = [
        evaluate(datasets["TRAIN"], probabilities["TRAIN"], budget)
        for budget in BUDGET_CANDIDATES
    ]
    selected = select_budget(search)
    budget = float(selected["budget_fraction"])
    cohorts = {
        role: evaluate(dataset, probabilities[role], budget)
        for role, dataset in datasets.items()
        if role != "TRAIN"
    }
    supported = all(
        row["false_clear_all_known_improvement"] >= 0.01
        and row["false_block_given_clear_improvement"] >= 0.0
        and row["known_coverage_decrease"] <= 0.0200000001
        for row in cohorts.values()
    )
    checkpoint_out = {
        "schema": "blindassist_depthart_d3r6_budgeted_deferral_checkpoint_v1",
        "risk_score": "frozen D3R5 parent-grid relative veto head",
        "action": "baseline CLEAR to UNKNOWN only; never emit CLEAR or OCCUPIED",
        "budget_fraction_of_all_cells_per_parent": budget,
        "maximum_cells_per_parent": int(np.floor(CELLS_PER_PARENT * budget)),
        "ranking": "descending risk score within each parent over source-available hard-evidence baseline-CLEAR cells; stable row-order tie-break",
        "budget_selection": "TRAIN_ONLY maximize false-clear improvement subject to >=0.01 improvement, no false-block increase, and <=0.02 coverage decrease",
        "source_score_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
    }
    checkpoint_out_path = args.output_root / "deferral-checkpoint.json"
    atomic_json(checkpoint_out_path, checkpoint_out)
    result = {
        "schema": "blindassist_depthart_d3r6_budgeted_deferral_discovery_result_v1",
        "status": (
            "D3R6_BUDGETED_UNKNOWN_DEFERRAL_DISCOVERY_SUPPORTED"
            if supported
            else "D3R6_BUDGETED_UNKNOWN_DEFERRAL_DISCOVERY_NOT_SUPPORTED"
        ),
        "problem": "D3R5 risk ranking transferred partially, but converting high-risk CLEAR directly to OCCUPIED was not calibrated across parents.",
        "hypothesis": "A fixed per-parent UNKNOWN budget converts transferable ranking into bounded risk reduction without false-block amplification.",
        "train_budget_search": search,
        "selected_budget_fraction": budget,
        "reused_cohorts": cohorts,
        "decision_rule": {
            "false_clear_improvement_min": 0.01,
            "false_block_improvement_min": 0.0,
            "known_coverage_decrease_max": 0.02,
        },
        "mechanism_supported_on_reused_cohorts": supported,
        "confirmation_outputs_used_for_training_or_budget_selection": False,
        "evidence_role": "DISCOVERY_ONLY_D3R5_CONFIRMATION_WAS_ALREADY_OPENED_FOR_FAILURE_DIAGNOSIS",
        "fresh_parent_confirmation_required": True,
        "source_unavailable_as_negative": False,
        "unknown_as_negative": False,
        "r2_access": "NONE",
        "performance_claim": False,
        "checkpoint": {
            "path": str(checkpoint_out_path.resolve()),
            "bytes": checkpoint_out_path.stat().st_size,
            "sha256": sha256_file(checkpoint_out_path),
        },
        "input_bindings": {
            role: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for role, path in paths.items()
        }
        | {
            "d3r5_confirmation_result": {
                "path": str(d3r5_confirmation_path.resolve()),
                "bytes": d3r5_confirmation_path.stat().st_size,
                "sha256": sha256_file(d3r5_confirmation_path),
            }
        },
        "next_action": (
            "FRESH_PARENT_BUDGETED_DEFERRAL_CONFIRMATION"
            if supported
            else "RETHINK_SELECTIVE_ACTION"
        ),
    }
    atomic_json(args.output_root / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "selected_budget_fraction": budget,
        "train_false_clear_improvement": selected[
            "false_clear_all_known_improvement"
        ],
        "reused_development_false_clear_improvement": cohorts[
            "REUSED_DEVELOPMENT"
        ]["false_clear_all_known_improvement"],
        "reused_confirmation_false_clear_improvement": cohorts[
            "REUSED_D3R5_CONFIRMATION"
        ]["false_clear_all_known_improvement"],
        "false_block_improvement": cohorts["REUSED_D3R5_CONFIRMATION"][
            "false_block_given_clear_improvement"
        ],
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
