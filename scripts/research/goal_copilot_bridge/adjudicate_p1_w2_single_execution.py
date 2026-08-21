#!/usr/bin/env python3
"""Adjudicate a sealed P1-W2 provider output against private ADT truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


VIEWPOINT_STRATA = (
    "ROTATION_DOMINANT",
    "SMALL_TRANSLATION",
    "LARGE_TRANSLATION",
    "REAPPEARANCE",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parent_macro(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    by_parent: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_parent[row["parent_id"]].append(bool(predicate(row)))
    if not by_parent:
        return 0.0
    return float(sum(sum(values) / len(values) for values in by_parent.values()) / len(by_parent))


def validate_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite provider value at {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--roster-receipt", type=Path, required=True)
    parser.add_argument("--public-roster", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--provider-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    roster_receipt = json.loads(args.roster_receipt.read_text(encoding="utf-8"))
    public = json.loads(args.public_roster.read_text(encoding="utf-8"))
    private = json.loads(args.private_truth.read_text(encoding="utf-8"))
    provider = json.loads(args.provider_output.read_text(encoding="utf-8"))
    provider_receipt = json.loads(args.provider_receipt.read_text(encoding="utf-8"))
    if roster_receipt.get("terminal") != "P1_W2_FRESH_PRIVATE_ROSTER_FROZEN":
        raise ValueError("roster is not frozen/evaluable")
    if digest(args.public_roster) != roster_receipt["output_sha256"]["public_roster.json"]:
        raise ValueError("public roster hash drift")
    if digest(args.private_truth) != roster_receipt["output_sha256"]["evaluator_private_truth_map.json"]:
        raise ValueError("private truth hash drift")
    if provider_receipt.get("terminal") != "P1_W2_PROVIDER_OUTPUT_SEALED":
        raise ValueError("provider output is not sealed")
    if digest(args.provider_output) != provider_receipt["provider_output_sha256"]:
        raise ValueError("provider output hash drift")
    if provider.get("private_truth_loaded") is not False or provider_receipt.get("private_truth_loaded") is not False:
        raise ValueError("provider truth firewall failed")
    if len(provider["cases"]) != public["pair_count"]:
        raise ValueError("provider/public case denominator mismatch")
    validate_finite(provider)

    truth_by_case = {case["case_id"]: case for case in private["cases"]}
    if set(truth_by_case) != {case["case_id"] for case in provider["cases"]}:
        raise ValueError("provider/private case identity mismatch")
    rows = []
    for case in provider["cases"]:
        truth = truth_by_case[case["case_id"]]
        candidate_by_id = {candidate["candidate_id"]: candidate for candidate in case["candidates"]}
        truth_candidate_ids = {candidate["candidate_id"] for candidate in truth["candidates"]}
        if set(candidate_by_id) != truth_candidate_ids:
            raise ValueError(f"candidate identity mismatch: {case['case_id']}")
        true_id = truth["true_candidate_id"]
        if true_id not in candidate_by_id:
            raise ValueError(f"true candidate absent: {case['case_id']}")
        identity_selected = case["identity_selected_candidate_id"]
        joint_selected = case["joint_selected_candidate_id"]
        row = {
            "case_id": case["case_id"],
            "parent_id": case["parent_id"],
            "support_buckets": truth["support_buckets"],
            "candidate_count": len(case["candidates"]),
            "true_geometry_supported": bool(candidate_by_id[true_id]["geometry"]["geometry_supported"]),
            "identity_state": case["identity_state"],
            "identity_correct": identity_selected == true_id,
            "identity_false_bind": identity_selected is not None and identity_selected != true_id,
            "joint_state": case["joint_state"],
            "joint_correct": joint_selected == true_id,
            "joint_false_bind": joint_selected is not None and joint_selected != true_id,
        }
        rows.append(row)

    confuser_rows = [row for row in rows if "SAME_SCENE_CONFUSER" in row["support_buckets"]]
    geometry_overall = parent_macro(rows, lambda row: row["true_geometry_supported"])
    geometry_by_stratum = {
        stratum: parent_macro(
            [row for row in rows if stratum in row["support_buckets"]],
            lambda row: row["true_geometry_supported"],
        )
        for stratum in VIEWPOINT_STRATA
    }
    identity_overall = parent_macro(confuser_rows, lambda row: row["identity_correct"])
    joint_overall = parent_macro(rows, lambda row: row["joint_correct"])
    identity_false_binds = sum(row["identity_false_bind"] for row in confuser_rows)
    joint_false_binds = sum(row["joint_false_bind"] for row in rows)
    adjudication = freeze["future_adjudication"]
    geometry_sufficient = bool(
        geometry_overall >= 0.70
        and all(geometry_by_stratum[stratum] >= 0.50 for stratum in VIEWPOINT_STRATA)
    )
    identity_sufficient = bool(identity_overall >= 0.70 and identity_false_binds == 0)
    joint_sufficient = bool(joint_overall >= 0.60 and joint_false_binds == 0)
    if geometry_sufficient and identity_sufficient and joint_sufficient:
        terminal = "P1_W2_ANCHOR_INTERFACE_SIGNAL_ESTABLISHED"
    elif geometry_sufficient and not identity_sufficient:
        terminal = "P1_W2_IDENTITY_SEPARABILITY_LIMITED"
    elif not geometry_sufficient and identity_sufficient:
        terminal = "P1_W2_GEOMETRY_SUPPORT_LIMITED"
    else:
        terminal = "P1_W2_RGB_REFERENT_INTERFACE_NOT_SUPPORTED"

    result = {
        "schema_version": "p1_w2_single_execution_result_v1",
        "terminal": terminal,
        "claim_role": "FRESH_ADT_INDOOR_OBJECT_PROXY_SINGLE_EXECUTION",
        "denominators": {
            "fixed_parents": public["parent_denominator"],
            "eligible_parents": public["eligible_parent_count"],
            "pairs": len(rows),
            "same_scene_confuser_pairs": len(confuser_rows),
        },
        "geometry": {
            "parent_macro_true_candidate_support": geometry_overall,
            "parent_macro_by_viewpoint_stratum": geometry_by_stratum,
            "supported_true_pairs": sum(row["true_geometry_supported"] for row in rows),
            "pair_count": len(rows),
            "sufficient": geometry_sufficient,
        },
        "identity": {
            "parent_macro_unique_true_on_confuser_pairs": identity_overall,
            "unique_true_count": sum(row["identity_correct"] for row in confuser_rows),
            "false_bind_count": identity_false_binds,
            "state_counts": dict(sorted(Counter(row["identity_state"] for row in confuser_rows).items())),
            "sufficient": identity_sufficient,
        },
        "joint": {
            "parent_macro_correct_eligible": joint_overall,
            "correct_eligible_count": sum(row["joint_correct"] for row in rows),
            "false_bind_count": joint_false_binds,
            "state_counts": dict(sorted(Counter(row["joint_state"] for row in rows).items())),
            "sufficient": joint_sufficient,
        },
        "frozen_gates": {
            "geometry_overall_min": 0.70,
            "geometry_each_viewpoint_min": 0.50,
            "identity_confuser_parent_macro_min": 0.70,
            "joint_parent_macro_min": 0.60,
            "identity_false_bind_max": 0,
            "joint_false_bind_max": 0,
        },
        "case_diagnostics": rows,
        "no_threshold_or_roster_change": True,
        "claim_ceiling": "ADT_INDOOR_OBJECT_PROXY_ONLY_NO_BUILDING_ENTRANCE_PRODUCT_OR_SAFETY_AUTHORITY",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise ValueError("result already exists; refusing readjudication")
    atomic_json(result_path, result)
    receipt = {
        "schema_version": "p1_w2_single_execution_result_receipt_v1",
        "freeze_sha256": digest(args.freeze),
        "roster_receipt_sha256": digest(args.roster_receipt),
        "public_roster_sha256": digest(args.public_roster),
        "private_truth_sha256": digest(args.private_truth),
        "provider_output_sha256": digest(args.provider_output),
        "provider_receipt_sha256": digest(args.provider_receipt),
        "result_sha256": digest(result_path),
        "terminal": terminal,
    }
    atomic_json(args.output_dir / "result_receipt.json", receipt)
    print(json.dumps({"terminal": terminal, "geometry": geometry_sufficient, "identity": identity_sufficient, "joint": joint_sufficient}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
