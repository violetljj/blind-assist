#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_SEED = 20_260_725
BOOTSTRAP_REPLICATES = 2000
MIN_ORACLE_SPEARMAN = 0.30
MIN_ORACLE_MINUS_UNCOMPENSATED_SPEARMAN = -0.05
MIN_COMMON_SUPPORT_FRACTION = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        raise ValueError("at least three observations required")
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def robust_standardize(values: np.ndarray) -> np.ndarray:
    center = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - center))
    if scale <= 1e-12:
        scale = float(values.std())
    if scale <= 1e-12:
        return np.zeros_like(values)
    return (values - center) / scale


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    slopes: list[np.ndarray] = []
    for index in range(len(x) - 1):
        dx = x[index + 1 :] - x[index]
        keep = np.abs(dx) > 1e-12
        if keep.any():
            slopes.append((y[index + 1 :][keep] - y[index]) / dx[keep])
    if not slopes:
        return 0.0
    return float(np.median(np.concatenate(slopes)))


def percentile_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def session_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray(
        [item["truth_closing_rate_mps"] for item in rows],
        dtype=np.float64,
    )
    arms = {
        "RAW_FLOW_ENERGY": np.asarray(
            [item["raw_flow_energy"] for item in rows],
            dtype=np.float64,
        ),
        "UNCOMPENSATED_LOCAL_RADIAL_EXPANSION": np.asarray(
            [item["uncompensated_radial"] for item in rows],
            dtype=np.float64,
        ),
        "ORACLE_ROTATION_COMPENSATION": np.asarray(
            [item["oracle_radial"] for item in rows],
            dtype=np.float64,
        ),
        "FULL_6DOF_RESIDUAL_DIAGNOSTIC": np.asarray(
            [item["full_6dof_radial"] for item in rows],
            dtype=np.float64,
        ),
    }
    metrics: dict[str, Any] = {}
    truth_standardized = robust_standardize(truth)
    for arm_id, signal in arms.items():
        metrics[arm_id] = {
            "signed_spearman_with_static_surface_closing_rate": spearman(
                signal, truth
            ),
            "theil_sen_signal_per_mps_truth": theil_sen_slope(
                truth, signal
            ),
            "standardized_theil_sen_slope": theil_sen_slope(
                truth_standardized, robust_standardize(signal)
            ),
            "signal_median": float(np.median(signal)),
            "signal_q10": float(np.quantile(signal, 0.10)),
            "signal_q90": float(np.quantile(signal, 0.90)),
        }
    return {
        "common_support_pair_count": len(rows),
        "truth_closing_rate_mps_median": float(np.median(truth)),
        "truth_closing_rate_mps_q10": float(np.quantile(truth, 0.10)),
        "truth_closing_rate_mps_q90": float(np.quantile(truth, 0.90)),
        "arms": metrics,
    }


def equal_session_bootstrap(
    session_results: list[dict[str, Any]], arm_id: str
) -> dict[str, Any]:
    values = np.asarray(
        [
            item["metrics"]["arms"][arm_id][
                "signed_spearman_with_static_surface_closing_rate"
            ]
            for item in session_results
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = generator.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    replicates = values[indexes].mean(axis=1)
    return {
        "equal_session_mean_spearman": float(values.mean()),
        "session_block_bootstrap_95_percent_interval": percentile_interval(
            replicates
        ),
        "session_values": values.tolist(),
    }


def evaluate(
    contract: dict[str, Any],
    pair_manifest: dict[str, Any],
    base_receipt: dict[str, Any],
    oracle_receipt: dict[str, Any],
    truth_receipt: dict[str, Any],
) -> dict[str, Any]:
    base_by_id = {
        item["unit_id"]: item for item in base_receipt["traces"]
    }
    oracle_by_id = {
        item["unit_id"]: item for item in oracle_receipt["traces"]
    }
    truth_by_key = {
        (sequence["sequence_id"], round(unit["rgb_timestamp"], 6)): unit
        for sequence in truth_receipt["sequences"]
        for unit in sequence["units"]
    }
    sessions: list[dict[str, Any]] = []
    joined_rows: list[dict[str, Any]] = []
    for sequence in pair_manifest["sequences"]:
        candidate_count = len(sequence["pairs"])
        rows: list[dict[str, Any]] = []
        abstentions: dict[str, int] = {}
        for pair in sequence["pairs"]:
            base_trace = base_by_id[pair["unit_id"]]
            oracle_trace = oracle_by_id[pair["unit_id"]]
            truth = truth_by_key.get(
                (
                    sequence["session_id"],
                    round(pair["current_timestamp"], 6),
                )
            )
            reason: str | None = None
            if not base_trace.get("evaluated"):
                reason = "BASE_ARM_ABSTAINED"
            elif not oracle_trace.get("evaluated"):
                reason = "ORACLE_ARM_ABSTAINED"
            elif truth is None or not truth.get("evaluated"):
                reason = "STATIC_SURFACE_TRUTH_ABSTAINED"
            elif (
                truth.get(
                    "static_surface_closing_rate_meters_per_second"
                )
                is None
            ):
                reason = "CONTIGUOUS_CLOSING_RATE_TRUTH_ABSTAINED"
            elif not oracle_trace["FULL_6DOF_RESIDUAL_DIAGNOSTIC"][
                "evaluated"
            ]:
                reason = "FULL_6DOF_DIAGNOSTIC_ABSTAINED"
            if reason is not None:
                abstentions[reason] = abstentions.get(reason, 0) + 1
                continue
            row = {
                "unit_id": pair["unit_id"],
                "source_family": "BONN_RGBD_DYNAMIC",
                "session_id": sequence["session_id"],
                "current_timestamp": pair["current_timestamp"],
                "truth_closing_rate_mps": truth[
                    "static_surface_closing_rate_meters_per_second"
                ],
                "raw_flow_energy": base_trace["RAW_FLOW_ENERGY"][
                    "q90_flow_magnitude_pixels_per_second"
                ],
                "uncompensated_radial": base_trace[
                    "UNCOMPENSATED_LOCAL_RADIAL_EXPANSION"
                ]["q90_positive_radial_rate_per_second"],
                "oracle_radial": oracle_trace[
                    "ORACLE_ROTATION_COMPENSATION"
                ]["q90_positive_radial_rate_per_second"],
                "full_6dof_radial": oracle_trace[
                    "FULL_6DOF_RESIDUAL_DIAGNOSTIC"
                ]["q90_positive_radial_rate_per_second"],
            }
            rows.append(row)
            joined_rows.append(row)
        support = len(rows) / candidate_count
        sessions.append(
            {
                "session_id": sequence["session_id"],
                "candidate_pair_count": candidate_count,
                "common_support_pair_count": len(rows),
                "common_support_fraction": support,
                "abstention_counts": abstentions,
                "metrics": session_metrics(rows),
            }
        )

    arm_ids = [
        "RAW_FLOW_ENERGY",
        "UNCOMPENSATED_LOCAL_RADIAL_EXPANSION",
        "ORACLE_ROTATION_COMPENSATION",
        "FULL_6DOF_RESIDUAL_DIAGNOSTIC",
    ]
    source_metrics = {
        arm_id: equal_session_bootstrap(sessions, arm_id)
        for arm_id in arm_ids
    }
    oracle_metric = source_metrics["ORACLE_ROTATION_COMPENSATION"]
    uncompensated_metric = source_metrics[
        "UNCOMPENSATED_LOCAL_RADIAL_EXPANSION"
    ]
    oracle_rho = oracle_metric["equal_session_mean_spearman"]
    oracle_lower = oracle_metric[
        "session_block_bootstrap_95_percent_interval"
    ][0]
    rho_delta = (
        oracle_rho
        - uncompensated_metric["equal_session_mean_spearman"]
    )
    support_pass = all(
        item["common_support_fraction"] >= MIN_COMMON_SUPPORT_FRACTION
        for item in sessions
    )
    oracle_association_pass = (
        oracle_rho >= MIN_ORACLE_SPEARMAN and oracle_lower > 0.0
    )
    retention_pass = (
        rho_delta >= MIN_ORACLE_MINUS_UNCOMPENSATED_SPEARMAN
    )
    bonn_gate_pass = (
        support_pass and oracle_association_pass and retention_pass
    )
    return {
        "schema_version": "bonn_r1a_continuous_signal_evaluation_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "claim_id": "C2_STATIC_SURFACE_CLOSING_RETENTION",
        "source_family": "BONN_RGBD_DYNAMIC",
        "frozen_input_receipts": {
            "signal_contract_sha256": None,
            "pair_manifest_sha256": None,
            "base_trace_sha256": None,
            "oracle_trace_sha256": None,
            "truth_ledger_sha256": None,
        },
        "analysis_contract": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "minimum_oracle_spearman": MIN_ORACLE_SPEARMAN,
            "minimum_oracle_minus_uncompensated_spearman": (
                MIN_ORACLE_MINUS_UNCOMPENSATED_SPEARMAN
            ),
            "minimum_common_support_fraction": MIN_COMMON_SUPPORT_FRACTION,
            "source_session_weighting": "EQUAL_SESSION",
            "missing_value_imputation": False,
            "alarm_threshold_selected": False,
        },
        "sessions": sessions,
        "source_metrics": source_metrics,
        "comparisons": {
            "oracle_minus_uncompensated_equal_session_mean_spearman": (
                rho_delta
            ),
            "bbox_growth": (
                "SOURCE_CLAIM_ABSTAIN_NO_FROZEN_TARGET_BBOX"
            ),
            "full_6dof_acceptance_authority": False,
        },
        "gates": {
            "common_support_gate_passed": support_pass,
            "oracle_truth_association_gate_passed": (
                oracle_association_pass
            ),
            "oracle_retention_gate_passed": retention_pass,
            "Bonn_C2_discovery_family_gate_passed": bonn_gate_pass,
            "overall_R1A_claim_passed": False,
            "overall_hold_reason": (
                "CONTROLLED_SECOND_FAMILY_ABSENT_AND_BONN_C1_ABSTAINS"
            ),
        },
        "counts": {
            "candidate_pair_count": sum(
                item["candidate_pair_count"] for item in sessions
            ),
            "common_support_pair_count": len(joined_rows),
            "session_count": len(sessions),
        },
        "read_firewall": {
            "base_and_oracle_traces_hash_frozen_before_truth_join": True,
            "old_window_selection_tuning_acceptance_reads": 0,
            "validation_or_holdout_read_count": 0,
            "alarm_threshold_selected": False,
            "route_or_event_truth_used": False,
            "app_or_lifecycle_connected": False,
        },
        "terminal": (
            "BONN_C2_DISCOVERY_ORACLE_SIGNAL_SUPPORTED_"
            "CONTROLLED_FAMILY_PENDING"
            if bonn_gate_pass
            else "STOP_R1A_BONN_C2_ORACLE_GATE_FAILED"
        ),
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-contract", required=True, type=Path)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--base-traces", required=True, type=Path)
    parser.add_argument("--oracle-traces", required=True, type=Path)
    parser.add_argument("--truth-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "contract": json.loads(
            args.signal_contract.read_text(encoding="utf-8")
        ),
        "pair_manifest": json.loads(
            args.pair_manifest.read_text(encoding="utf-8")
        ),
        "base_receipt": json.loads(
            args.base_traces.read_text(encoding="utf-8")
        ),
        "oracle_receipt": json.loads(
            args.oracle_traces.read_text(encoding="utf-8")
        ),
        "truth_receipt": json.loads(
            args.truth_ledger.read_text(encoding="utf-8")
        ),
    }
    receipt = evaluate(**inputs)
    receipt["frozen_input_receipts"].update(
        {
            "signal_contract_sha256": sha256(args.signal_contract),
            "pair_manifest_sha256": sha256(args.pair_manifest),
            "base_trace_sha256": sha256(args.base_traces),
            "oracle_trace_sha256": sha256(args.oracle_traces),
            "truth_ledger_sha256": sha256(args.truth_ledger),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                **receipt["counts"],
                **receipt["gates"],
            },
            sort_keys=True,
        )
    )
    return 0 if "SUPPORTED" in receipt["terminal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
