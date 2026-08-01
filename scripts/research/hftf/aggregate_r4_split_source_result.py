#!/usr/bin/env python3
"""Aggregate the two frozen HFTF R4 source-role results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_hftf_stage_b_split_source_validation_r4"
TERRAIN_SCHEMA = "blindassist_hftf_r4_analytic_terrain_result_r0"
OBSTACLE_SCHEMA = (
    "blindassist_hftf_r4_obstacle_reference_comparison_result"
)
SCHEMA = "blindassist_hftf_stage_b_split_source_result_r4"
OBSTACLE_SOURCE_FAIL = "R4_OBSTACLE_OPPORTUNITY_COHORT_NOT_EVALUABLE"
OBSTACLE_GAIN_FAIL = "R4_OBSTACLE_ENVELOPE_GAIN_NOT_SUPPORTED_STOP"
OBSTACLE_SUPPORTED = "R4_OBSTACLE_ENVELOPE_GAIN_SUPPORTED"
TERRAIN_FAIL = "R4_ANALYTIC_TERRAIN_MECHANICS_NOT_SUPPORTED_STOP"
TERRAIN_SUPPORTED = "R4_ANALYTIC_TERRAIN_MECHANICS_SUPPORTED"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal(
    obstacle_terminal: str,
    terrain_terminal: str,
    ordered: list[str],
) -> str:
    if obstacle_terminal == OBSTACLE_SOURCE_FAIL:
        return ordered[0]
    if obstacle_terminal != OBSTACLE_SUPPORTED:
        if obstacle_terminal != OBSTACLE_GAIN_FAIL:
            raise ValueError("Unexpected R4 obstacle terminal")
        return ordered[1]
    if terrain_terminal != TERRAIN_SUPPORTED:
        if terrain_terminal != TERRAIN_FAIL:
            raise ValueError("Unexpected R4 terrain terminal")
        return ordered[2]
    return ordered[3]


def aggregate(
    protocol_path: Path,
    terrain_report_path: Path,
    obstacle_report_path: Path,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    terrain = _load(terrain_report_path)
    obstacle = _load(obstacle_report_path)
    protocol_sha = _sha256(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
    ):
        raise ValueError("R4 protocol is not frozen")
    if (
        terrain.get("schema") != TERRAIN_SCHEMA
        or terrain.get("protocol_sha256") != protocol_sha
        or terrain.get("joint_terminal_decided") is not False
        or terrain.get("stage_c_execution_authorized") is not False
    ):
        raise ValueError("R4 terrain report binding mismatch")
    if (
        obstacle.get("schema") != OBSTACLE_SCHEMA
        or obstacle.get("protocol_sha256") != protocol_sha
        or obstacle.get("joint_terminal_decided") is not False
        or obstacle.get("stage_c_execution_authorized") is not False
    ):
        raise ValueError("R4 obstacle report binding mismatch")
    terminal = _terminal(
        str(obstacle["terminal"]),
        str(terrain["terminal"]),
        [str(value) for value in protocol["joint_ordered_terminals"]],
    )
    success = terminal == protocol["joint_ordered_terminals"][-1]
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": protocol["workflow_profile"],
        "estimand": protocol["estimand"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "terrain_report_path": str(terrain_report_path.resolve()),
        "terrain_report_sha256": _sha256(terrain_report_path),
        "terrain_terminal": terrain["terminal"],
        "obstacle_report_path": str(obstacle_report_path.resolve()),
        "obstacle_report_sha256": _sha256(obstacle_report_path),
        "obstacle_terminal": obstacle["terminal"],
        "joint_checks": {
            "obstacle_envelope_gain_supported": (
                obstacle["terminal"] == OBSTACLE_SUPPORTED
            ),
            "analytic_terrain_mechanics_supported": (
                terrain["terminal"] == TERRAIN_SUPPORTED
            ),
            "both_source_roles_supported": success,
        },
        "claim_ceiling": (
            "SPLIT_SOURCE_DEVELOPMENT_TEACHER_MECHANICS_ONLY"
        ),
        "stage_c_source_feasibility_contract_freeze_authorized": success,
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
    parser.add_argument("--terrain-report", type=Path, required=True)
    parser.add_argument("--obstacle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        result = aggregate(
            args.protocol.resolve(),
            args.terrain_report.resolve(),
            args.obstacle_report.resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {"terminal": result["terminal"], "output": str(output)}
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
