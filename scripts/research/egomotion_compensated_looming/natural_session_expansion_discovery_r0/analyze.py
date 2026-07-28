from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL_ID = "RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0"
DISCOVERY_SESSIONS = (13, 14, 15, 17)
SEALED_SESSION = 16


def load_session(path: Path, expected_session: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (path / "pair_ledger.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    context = summary.get("evidence_context", {})
    execution = summary.get("execution", {})
    if (
        summary.get("protocol_id") != PROTOCOL_ID
        or context.get("session_number") != expected_session
        or context.get("implementation_version")
        != "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
        or context.get("algorithm_adjustment") is not False
        or context.get("threshold_changed") is not False
        or context.get("three_pair_rule_changed") is not False
        or context.get("sealed_session_accessed") is not False
        or execution.get("candidate_pair_count") != 601
        or execution.get("threshold_per_s") != 0.01
        or execution.get("required_consecutive_pairs") != 3
        or execution.get("single_process_pair_state_continuous") is not True
        or execution.get("support_manager_baseline_pair_count") != 1
        or len(rows) != 601
        or [row.get("pair_index") for row in rows] != list(range(601))
    ):
        raise ValueError(f"SESSION_EXECUTION_IDENTITY_MISMATCH:{expected_session}")
    if any(
        key.lower() in {"auroc", "f1", "roc_auc"}
        for key in summary
    ):
        raise ValueError("FORBIDDEN_CLASSIFICATION_METRIC_PRESENT")
    return summary, rows


def high_angular_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    angular = np.asarray(
        [float(row["angular_speed_deg_per_s"]) for row in rows],
        dtype=np.float64,
    )
    cutoff = float(np.quantile(angular, 0.8))
    selected = [
        row for row in rows
        if float(row["angular_speed_deg_per_s"]) >= cutoff
    ]
    evaluable = [
        row for row in selected
        if row.get("raw_expansion_median_per_s") is not None
        and row.get("compensated_expansion_median_per_s") is not None
    ]
    if not evaluable:
        return {
            "status": "NOT_EVALUABLE",
            "angular_speed_cutoff_deg_per_s": cutoff,
            "fixed_denominator_pair_count": len(selected),
            "reason": "NO_COMMON_EVALUABLE_PAIR",
        }
    denominator = len(selected)
    raw_density = sum(
        bool(row.get("raw_three_pair_trigger")) for row in selected
    ) / denominator
    compensated_density = sum(
        bool(row.get("compensated_three_pair_trigger"))
        for row in selected
    ) / denominator
    raw_abs = float(
        np.median(
            [
                abs(float(row["raw_expansion_median_per_s"]))
                for row in evaluable
            ]
        )
    )
    compensated_abs = float(
        np.median(
            [
                abs(float(row["compensated_expansion_median_per_s"]))
                for row in evaluable
            ]
        )
    )
    return {
        "status": "EVALUABLE",
        "angular_speed_cutoff_deg_per_s": cutoff,
        "fixed_denominator_pair_count": denominator,
        "common_evaluable_pair_count": len(evaluable),
        "raw_trigger_density": raw_density,
        "compensated_trigger_density": compensated_density,
        "raw_median_abs_response_per_s": raw_abs,
        "compensated_median_abs_response_per_s": compensated_abs,
        "compensation_deteriorated": (
            compensated_density > raw_density
            and compensated_abs > raw_abs
        ),
    }


def session_result(
    session: int, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    raw = summary["methods"]["raw_local_expansion"]
    compensated = summary["methods"][
        "source_pose_rotation_compensated_local_expansion"
    ]
    failures = Counter(
        str(row.get("reason"))
        for row in rows
        if row.get("evaluable") is not True
    )
    return {
        "session_number": session,
        "source_id": summary["source"]["source_id"],
        "role": summary["research_track"],
        "duration_s": summary["execution"]["duration_s"],
        "candidate_pair_count_longitudinal_only": 601,
        "support_rate": summary["execution"]["evaluable_pair_fraction"],
        "response": {
            "raw": {
                "median_per_s": raw["median_per_s"],
                "median_abs_per_s": raw["median_abs_per_s"],
                "p10_per_s": raw["p10_per_s"],
                "p90_per_s": raw["p90_per_s"],
            },
            "compensated": {
                "median_per_s": compensated["median_per_s"],
                "median_abs_per_s": compensated["median_abs_per_s"],
                "p10_per_s": compensated["p10_per_s"],
                "p90_per_s": compensated["p90_per_s"],
            },
        },
        "trigger_density_fixed_pair_denominator": {
            "raw": raw["three_pair_trigger_fraction_fixed_denominator"],
            "compensated": compensated[
                "three_pair_trigger_fraction_fixed_denominator"
            ],
        },
        "angular_speed_association_spearman": {
            "raw_abs_response": summary["diagnostics"][
                "angular_speed_correlation"
            ]["raw_abs_expansion"]["spearman"],
            "compensated_abs_response": summary["diagnostics"][
                "angular_speed_correlation"
            ]["compensated_abs_expansion"]["spearman"],
        },
        "failure_types": dict(sorted(failures.items())),
        "high_angular_speed": high_angular_result(rows),
    }


def analyze(session_dirs: dict[int, Path]) -> dict[str, Any]:
    if set(session_dirs) != set(DISCOVERY_SESSIONS):
        raise ValueError("FROZEN_DISCOVERY_SESSION_SET_MISMATCH")
    sessions = []
    for session in DISCOVERY_SESSIONS:
        summary, rows = load_session(session_dirs[session], session)
        sessions.append(session_result(session, summary, rows))
    deteriorated = [
        item["session_number"]
        for item in sessions
        if item["high_angular_speed"].get("compensation_deteriorated")
        is True
    ]
    route_stopped = len(deteriorated) >= 2
    return {
        "schema": "rcle.natural_session_expansion.discovery.result.v1",
        "protocol_id": PROTOCOL_ID,
        "analysis_unit": "CAPTURE_SESSION",
        "session_count": len(sessions),
        "pair_records_are_longitudinal_not_independent_samples": True,
        "sessions": sessions,
        "standalone_rotation_stop": {
            "sessions_with_high_angular_compensation_deterioration": (
                deteriorated
            ),
            "required_session_count": 2,
            "route_stopped": route_stopped,
            "next_mechanism_diagnostics": (
                [
                    "gait oscillation",
                    "motion blur",
                    "low texture",
                    "flow-quality gate",
                ]
                if route_stopped
                else []
            ),
        },
        "sealed_session": {
            "session_number": SEALED_SESSION,
            "state": "SEALED_UNSEEN",
            "accessed": False,
            "included_in_analysis": False,
        },
        "coverage_categories": {
            "normal_walking": "DESCRIPTIVE_ONLY",
            "head_turn": "SOURCE_POSE_HIGH_ANGULAR_STRATUM",
            "static_approach": "NOT_EVALUABLE_NO_FROZEN_LABEL",
            "crossing": "NOT_EVALUABLE_NO_FROZEN_EVENT_LABEL",
            "motion_blur": "NOT_EVALUABLE_NO_FROZEN_BLUR_LABEL",
            "gait_oscillation": "DESCRIPTIVE_FAILURE_HYPOTHESIS_ONLY",
        },
        "forbidden_metrics_computed": [],
        "claim_ceiling": (
            "MULTI_SESSION_DESCRIPTIVE_DISCOVERY_AND_MECHANISM_ROUTE_STOP"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for session in DISCOVERY_SESSIONS:
        parser.add_argument(f"--session-{session}-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        session: getattr(args, f"session_{session}_dir").resolve()
        for session in DISCOVERY_SESSIONS
    }
    result = analyze(paths)
    args.output.resolve().write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["standalone_rotation_stop"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
