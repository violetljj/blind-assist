#!/usr/bin/env python3
"""Run the frozen R4 obstacle-only swept-envelope arm comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pilot_swept_envelope_reference_metrics import (
    LAYERS,
    _cohort_metrics,
    _pixel_lattices_disjoint,
    _session_pilot,
)
from run_geometry_teacher_canary import _sha256
from run_stage_b_reference_comparison import _metric_delta
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


PROTOCOL_SCHEMA = "blindassist_hftf_stage_b_split_source_validation_r4"
LEDGER_SCHEMA = "blindassist_hftf_r4_source_pool_burn_ledger"
PLAN_SCHEMA = "blindassist_hftf_r4_obstacle_inventory_candidate_plan"
LOCK_SCHEMA = "blindassist_hftf_r4_obstacle_opportunity_cohort_lock"
SCHEMA = "blindassist_hftf_r4_obstacle_reference_comparison_result"
SOURCE_NOT_EVALUABLE = "R4_OBSTACLE_OPPORTUNITY_COHORT_NOT_EVALUABLE"
GAIN_STOP = "R4_OBSTACLE_ENVELOPE_GAIN_NOT_SUPPORTED_STOP"
GAIN_SUPPORTED = "R4_OBSTACLE_ENVELOPE_GAIN_SUPPORTED"
EXPECTED_MECHANICS_SHA256 = (
    "a69d25d77f1e2b72f407980f005c758b965517fd032562a009f91746ea1e0e6a"
)


def _terminal(source_ready: bool, gain_supported: bool) -> str:
    if not source_ready:
        return SOURCE_NOT_EVALUABLE
    return GAIN_SUPPORTED if gain_supported else GAIN_STOP


def _validate_lock(
    protocol_path: Path,
    ledger_path: Path,
    plan_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_json(protocol_path)
    ledger = _load_json(ledger_path)
    plan = _load_json(plan_path)
    lock = _load_json(lock_path)
    protocol_sha = _sha256(protocol_path)
    ledger_sha = _sha256(ledger_path)
    plan_sha = _sha256(plan_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
    ):
        raise ValueError("R4 protocol is not frozen")
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
    ):
        raise ValueError("R4 burn ledger is not frozen")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("terminal")
        != "R4_OBSTACLE_INVENTORY_CANDIDATE_PLAN_READY"
        or plan.get("protocol_sha256") != protocol_sha
        or plan.get("burn_ledger_sha256") != ledger_sha
    ):
        raise ValueError("R4 inventory plan binding mismatch")
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("terminal")
        != "R4_OBSTACLE_OPPORTUNITY_COHORT_QUALIFIED"
        or lock.get("protocol_sha256") != protocol_sha
        or lock.get("burn_ledger_sha256") != ledger_sha
        or lock.get("inventory_plan_sha256") != plan_sha
        or lock.get("required_session_count") != 4
        or lock.get("formal_arm_outcome_authorized") is not True
        or lock.get("joint_terminal_decided") is not False
    ):
        raise ValueError("R4 cohort lock binding mismatch")
    expected_firewall = {
        "obstacle_reference_used": True,
        "ground_reference_used": False,
        "candidate_used": False,
        "angular_baseline_used": False,
        "arm_metric_or_delta_used": False,
    }
    if lock.get("selection_firewall") != expected_firewall:
        raise ValueError("R4 cohort lock selection firewall mismatch")
    sessions = lock.get("required_sessions", [])
    ids = [str(item["source_session_id"]) for item in sessions]
    ranks = [int(item["inventory_eligible_rank"]) for item in sessions]
    if len(sessions) != 4 or len(set(ids)) != 4 or ranks != sorted(ranks):
        raise ValueError("R4 cohort lock is not an exact four-session set")
    for item in sessions:
        qualification = Path(str(item["qualification_report_path"]))
        authority = Path(str(item["authority_report_path"]))
        if (
            _sha256(qualification)
            != item["qualification_report_sha256"]
            or _sha256(authority) != item["authority_report_sha256"]
        ):
            raise ValueError("R4 cohort-lock input hash drift")
    return protocol, plan, lock


def _source_paths_ready(
    session_inputs: list[tuple[Path, Path]],
    lock: dict[str, Any],
) -> bool:
    expected = {
        str(item["source_session_id"]): item
        for item in lock["required_sessions"]
    }
    seen: set[str] = set()
    for replay_root, authority_path in session_inputs:
        rows = _load_jsonl(replay_root / "manifest.replay.jsonl")
        ids = {str(row.get("session_id")) for row in rows}
        if len(ids) != 1:
            return False
        session_id = next(iter(ids))
        if session_id in seen or session_id not in expected:
            return False
        item = expected[session_id]
        if (
            _sha256(authority_path) != item["authority_report_sha256"]
            or _sha256(replay_root / "manifest.replay.jsonl")
            != item["manifest_sha256"]
            or _sha256(replay_root / "dataset_spec.json")
            != item["dataset_spec_sha256"]
            or _sha256(
                replay_root / "source_metadata/camera_poses.csv"
            )
            != item["camera_poses_sha256"]
        ):
            return False
        seen.add(session_id)
    return seen == set(expected)


def _evaluate_obstacle(
    sessions: list[dict[str, Any]],
    cohort: dict[str, Any],
    thresholds: list[int],
    primary: str,
    gates: dict[str, Any],
    *,
    exact: bool,
    lattice_disjoint: bool,
    source_paths_ready: bool,
) -> dict[str, Any]:
    source_authority_ready = (
        exact
        and lattice_disjoint
        and source_paths_ready
        and all(item.get("ok") for item in sessions)
    )
    known_ready = all(
        all(
            coverage
            >= float(
                gates[
                    "minimum_known_coverage_each_height_each_session"
                ]
            )
            for coverage in item["known_coverage_by_height"].values()
        )
        for item in sessions
    )
    opportunity_ready = all(
        all(
            item["reference_opportunity_by_threshold"][str(threshold)][
                "positive"
            ]
            > 0
            and item["reference_opportunity_by_threshold"][
                str(threshold)
            ]["negative"]
            > 0
            for threshold in thresholds
        )
        for item in sessions
    )
    height_evaluable = all(
        cohort[primary]["candidate"][layer][
            "positive_reference_cells"
        ]
        > 0
        and cohort[primary]["candidate"][layer][
            "negative_reference_cells"
        ]
        > 0
        for layer in LAYERS
    )
    source_ready = (
        source_authority_ready
        and known_ready
        and opportunity_ready
        and height_evaluable
    )
    candidate = cohort[primary]["candidate"]["micro_all_layers"]
    baseline = cohort[primary]["baseline"]["micro_all_layers"]
    f1_delta = _metric_delta(candidate, baseline, "f1")
    precision_delta = _metric_delta(candidate, baseline, "precision")
    recall_delta = _metric_delta(candidate, baseline, "recall")
    session_f1_deltas = [
        _metric_delta(
            item["metrics_by_reference_count_threshold"][primary][
                "candidate"
            ]["micro_all_layers"],
            item["metrics_by_reference_count_threshold"][primary][
                "baseline"
            ]["micro_all_layers"],
            "f1",
        )
        for item in sessions
    ]
    height_deltas = {
        layer: _metric_delta(
            cohort[primary]["candidate"][layer],
            cohort[primary]["baseline"][layer],
            "f1",
        )
        for layer in LAYERS
    }
    height_supported = all(
        value is not None and value > 0.0
        for value in height_deltas.values()
    )
    sensitivity_f1 = all(
        (
            _metric_delta(
                cohort[str(threshold)]["candidate"][
                    "micro_all_layers"
                ],
                cohort[str(threshold)]["baseline"][
                    "micro_all_layers"
                ],
                "f1",
            )
            or 0.0
        )
        > 0.0
        for threshold in thresholds
    )
    sensitivity_paired = all(
        cohort[str(threshold)]["candidate"]["micro_all_layers"][
            "candidate_only_correct"
        ]
        > cohort[str(threshold)]["candidate"]["micro_all_layers"][
            "baseline_only_correct"
        ]
        for threshold in thresholds
    )
    supported = (
        source_ready
        and f1_delta is not None
        and f1_delta
        >= float(gates["primary_minimum_cohort_micro_f1_delta"])
        and precision_delta is not None
        and precision_delta
        >= float(gates["primary_minimum_cohort_precision_delta"])
        and recall_delta is not None
        and recall_delta
        >= float(gates["primary_minimum_cohort_recall_delta"])
        and all(
            value is not None
            and value
            >= float(
                gates["primary_minimum_session_micro_f1_delta"]
            )
            for value in session_f1_deltas
        )
        and height_supported
        and sensitivity_f1
        and sensitivity_paired
    )
    return {
        "source_authority_and_exact_set": source_authority_ready,
        "obstacle_known_coverage": known_ready,
        "reference_opportunity": opportunity_ready,
        "primary_height_reference_opportunity": height_evaluable,
        "source_and_reference_ready": source_ready,
        "primary_cohort_micro_f1_delta": f1_delta,
        "primary_cohort_precision_delta": precision_delta,
        "primary_cohort_recall_delta": recall_delta,
        "primary_session_micro_f1_deltas": session_f1_deltas,
        "primary_height_f1_deltas": height_deltas,
        "primary_height_f1_direction_supported": height_supported,
        "all_sensitivity_f1_directions_supported": sensitivity_f1,
        "all_sensitivity_paired_directions_supported": (
            sensitivity_paired
        ),
        "obstacle_envelope_gain_supported": supported,
    }


def run(
    protocol_path: Path,
    ledger_path: Path,
    inventory_plan_path: Path,
    cohort_lock_path: Path,
    mechanics_path: Path,
    session_inputs: list[tuple[Path, Path]],
) -> dict[str, Any]:
    protocol, _, lock = _validate_lock(
        protocol_path, ledger_path, inventory_plan_path, cohort_lock_path
    )
    if _sha256(mechanics_path) != EXPECTED_MECHANICS_SHA256:
        raise ValueError("R4 mechanics protocol hash mismatch")
    if len(session_inputs) != 4:
        raise ValueError("Expected exactly four R4 session inputs")
    mechanics = _load_json(mechanics_path)
    formal = protocol["obstacle_source_role"]["formal_comparison"]
    candidate = formal["candidate"]
    reference = formal["reference"]
    lattice_disjoint = _pixel_lattices_disjoint(
        int(candidate["point_sample_stride_xy"]),
        int(candidate["point_sample_offset_xy"]),
        int(reference["point_sample_stride_xy"]),
        int(reference["point_sample_offset_xy"]),
    )
    required_ids = [
        str(item["source_session_id"])
        for item in lock["required_sessions"]
    ]
    pseudo_pilot = {
        "parent_sessions": required_ids,
        "candidate": candidate,
        "reference": reference,
    }
    pseudo_expected = {"required_sessions": lock["required_sessions"]}
    sessions = [
        _session_pilot(
            replay.resolve(),
            authority.resolve(),
            pseudo_pilot,
            mechanics,
            pseudo_expected,
        )
        for replay, authority in session_inputs
    ]
    ids = [str(item["source_session_id"]) for item in sessions]
    exact = set(ids) == set(required_ids) and len(ids) == len(set(ids))
    thresholds = [
        int(value)
        for value in reference["positive_count_threshold_sensitivity"]
    ]
    cohort = _cohort_metrics(sessions, thresholds)
    primary = str(reference["primary_positive_count_threshold"])
    checks = _evaluate_obstacle(
        sessions,
        cohort,
        thresholds,
        primary,
        formal["gates"],
        exact=exact,
        lattice_disjoint=lattice_disjoint,
        source_paths_ready=_source_paths_ready(session_inputs, lock),
    )
    terminal = _terminal(
        bool(checks["source_and_reference_ready"]),
        bool(checks["obstacle_envelope_gain_supported"]),
    )
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": protocol["workflow_profile"],
        "evidence_role": (
            "OPPORTUNITY_QUALIFIED_CHALLENGE_COHORT_OBSTACLE_COMPARISON"
        ),
        "claim_population": (
            "reference_obstacle_opportunity_qualified_challenge_cohort_only"
        ),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": _sha256(ledger_path),
        "inventory_plan_path": str(inventory_plan_path.resolve()),
        "inventory_plan_sha256": _sha256(inventory_plan_path),
        "cohort_lock_path": str(cohort_lock_path.resolve()),
        "cohort_lock_sha256": _sha256(cohort_lock_path),
        "mechanics_protocol_path": str(mechanics_path.resolve()),
        "mechanics_protocol_sha256": _sha256(mechanics_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "dependency_implementation_sha256": {
            "pilot_swept_envelope_reference_metrics.py": _sha256(
                Path(__file__).with_name(
                    "pilot_swept_envelope_reference_metrics.py"
                )
            ),
            "audit_swept_envelope_label_mechanics.py": _sha256(
                Path(__file__).with_name(
                    "audit_swept_envelope_label_mechanics.py"
                )
            ),
        },
        "exact_locked_session_set": exact,
        "candidate_reference_pixel_lattices_disjoint": (
            lattice_disjoint
        ),
        "obstacle_sessions": sessions,
        "obstacle_cohort_metrics_by_reference_count_threshold": cohort,
        "ordered_checks": checks,
        "joint_terminal_decided": False,
        "stage_c_source_feasibility_contract_freeze_authorized": False,
        "stage_c_execution_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
        "production_authorized": False,
        "safety_claim_authorized": False,
    }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--burn-ledger", type=Path, required=True)
    parser.add_argument("--inventory-plan", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--mechanics-protocol", type=Path, required=True)
    parser.add_argument(
        "--session", type=Path, nargs=2, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        result = run(
            args.protocol.resolve(),
            args.burn_ledger.resolve(),
            args.inventory_plan.resolve(),
            args.cohort_lock.resolve(),
            args.mechanics_protocol.resolve(),
            [(pair[0].resolve(), pair[1].resolve()) for pair in args.session],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": result["terminal"],
                    "output": str(output),
                }
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
