#!/usr/bin/env python3
"""Aggregate independent SANPO frozen-proxy replication reports for HFTF H0.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_hftf_sanpo_proxy_replication_cohort_r0"
EXPECTED_REPORT_SCHEMA = "blindassist_hftf_sanpo_pose_geometry_authority_r0"
EXPECTED_TERMINAL = "HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED"
EXPECTED_TRANSFORM = (
    "p_world = R_xyzw @ p_opencv_camera + camera_translation_m"
)
MINIMUM_INDEPENDENT_SESSIONS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def aggregate(report_paths: list[Path]) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("At least one report is required")
    sessions: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    for path in report_paths:
        report = _load_json(path)
        session_ids = report.get("source_session_ids", [])
        session_id = (
            session_ids[0]
            if isinstance(session_ids, list)
            and len(session_ids) == 1
            and isinstance(session_ids[0], str)
            and session_ids[0]
            else None
        )
        transform = report.get("transform_direction_canary", {})
        ground = report.get("ground_and_body_proxy_canary", {})
        decisions = report.get("capability_decisions", {})
        checks = {
            "schema": report.get("schema") == EXPECTED_REPORT_SCHEMA,
            "evaluation_mode": (
                report.get("evaluation_mode")
                == "frozen_canonical_replication"
            ),
            "terminal": report.get("terminal") == EXPECTED_TERMINAL,
            "single_session": session_id is not None,
            "canonical_rank_one": (
                transform.get("frozen_canonical_rank") == 1
            ),
            "canonical_replication_admitted": (
                transform.get("frozen_canonical_replication_admitted")
                is True
            ),
            "transform_semantics": (
                transform.get("admitted_semantics") == EXPECTED_TRANSFORM
            ),
            "vertical_axis_plus_z": ground.get("vertical_axis") == "+Z",
            "ground_proxy_admitted": (
                ground.get("standard_body_proxy_frame_admitted_for_h1")
                is True
            ),
            "physical_calibration_not_claimed": (
                decisions.get("physical_camera_to_person_calibration")
                == "NOT_EVALUABLE"
                and ground.get(
                    "physical_camera_to_body_calibration_admitted"
                )
                is False
            ),
        }
        ok = all(checks.values())
        if session_id:
            observed_ids.append(session_id)
        canonical = transform.get("frozen_canonical_hypothesis", {})
        chosen_axis = ground.get("chosen_axis", {})
        sessions.append(
            {
                "report_path": str(path),
                "report_sha256": _sha256(path),
                "source_session_id": session_id,
                "manifest_frame_count": report.get("manifest_frame_count"),
                "checks": checks,
                "canonical_median_relative_depth_error": canonical.get(
                    "median_relative_depth_error"
                ),
                "canonical_p75_relative_depth_error": canonical.get(
                    "p75_relative_depth_error"
                ),
                "canonical_coverage": canonical.get("coverage"),
                "ground_median_mad_m": chosen_axis.get(
                    "median_ground_mad_m"
                ),
                "camera_clearance_m": chosen_axis.get(
                    "median_camera_clearance_m"
                ),
                "ok": ok,
            }
        )

    independent = len(observed_ids) == len(set(observed_ids))
    enough_sessions = len(sessions) >= MINIMUM_INDEPENDENT_SESSIONS
    all_reports_pass = all(session["ok"] for session in sessions)
    admitted = independent and enough_sessions and all_reports_pass
    return {
        "schema": SCHEMA,
        "terminal": (
            "HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED"
            if admitted
            else "HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_NOT_EVALUABLE"
        ),
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "MULTI_SESSION_SYNTHETIC_GEOMETRY_PROXY_ONLY",
        "selection_rule": (
            "lexicographically smallest eligible official SANPO-Synthetic "
            "train sessions excluding the H0.1 discovery session"
        ),
        "minimum_independent_sessions": MINIMUM_INDEPENDENT_SESSIONS,
        "session_count": len(sessions),
        "unique_session_count": len(set(observed_ids)),
        "independent_session_ids": independent,
        "enough_sessions": enough_sessions,
        "all_reports_pass": all_reports_pass,
        "sessions": sessions,
        "mainline_changed": False,
        "physical_camera_to_person_calibration": "NOT_EVALUABLE",
        "student_or_event_effect": "NOT_EVALUABLE",
        "allowed_next_step": (
            "H1_GEOMETRY_TEACHER_CANARY"
            if admitted
            else "REPAIR_OR_EXPAND_H0_2_BEFORE_H1"
        ),
        "prohibited_inferences": [
            "multi-session proxy replication is physical body calibration",
            "synthetic geometry proxy is human collision truth",
            "H0.2 admits a student or effect claim",
            "research mainline promotion",
            "Android, alert, safety, or production authorization",
        ],
    }


def _require_artifacts_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = (repo_root / "artifacts.local").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = aggregate([path.resolve() for path in args.report])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "allowed_next_step": report["allowed_next_step"],
                    "output": str(output),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
