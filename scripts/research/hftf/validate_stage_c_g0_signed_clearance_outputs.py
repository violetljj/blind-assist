#!/usr/bin/env python3
"""Independently validate frozen HFTF G0 source-plan and D0 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_signed_clearance_current_bridge_g0"
)
SOURCE_SCHEMA = "blindassist_hftf_stage_c_g0_signed_clearance_source_plan"
MECHANICS_SCHEMA = (
    "blindassist_hftf_stage_c_g0_signed_clearance_mechanics_result"
)
VALIDATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_signed_clearance_output_validation"
)
SOURCE_READY = "G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY"
MECHANICS_SUPPORTED = (
    "G0_SIGNED_CLEARANCE_MECHANICS_SUPPORTED_FOR_FRESH_LEARNABILITY_CANARY"
)
VALIDATED = "G0_SIGNED_CLEARANCE_SOURCE_AND_MECHANICS_TERMINAL_VALIDATED"
NOT_EVALUABLE = "G0_SIGNED_CLEARANCE_OUTPUT_VALIDATION_NOT_EVALUABLE"
EXPECTED_PROTOCOL_SHA256 = (
    "0aa8e5828665a869837a1aa9027601d45610c0f66696da737351c9ec361da383"
)
EXPECTED_SOURCE_PLAN_SHA256 = (
    "886271cd1546e2f3f4cd91991f39725ed39b12907e0d4294b980404d132648a4"
)
EXPECTED_MECHANICS_SHA256 = (
    "050670764e15a8b9059dc893edb71534d6112ab8931a4fb118668653f8b577bf"
)
HEIGHTS = ("body", "head")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _session_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["session_id"]) for record in records]


def _validate_role_ids(
    development: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
) -> None:
    if [len(development), len(fresh), len(heldout)] != [9, 3, 3]:
        raise ValueError("G0 validation role count mismatch")
    groups = [
        set(_session_ids(records))
        for records in (development, fresh, heldout)
    ]
    if [len(group) for group in groups] != [9, 3, 3]:
        raise ValueError("G0 validation duplicate session within role")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("G0 validation session appears in multiple roles")


def _metric_passes(
    metric: dict[str, Any],
    gates: dict[str, Any],
) -> bool:
    return (
        int(metric["positive_known"]) > 0
        and int(metric["negative_known"]) > 0
        and int(metric["binary_equivalence_violations"]) == 0
        and int(metric["known_nonfinite_clipped_target"]) == 0
        and int(metric["unknown_nonnull_target_violations"]) == 0
        and int(metric["unknown_to_safe_violations"]) == 0
        and int(metric["distinct_clipped_target_millimeter_bins"])
        >= int(
            gates[
                "each_source_height_distinct_clipped_"
                "millimeter_bins_minimum"
            ]
        )
        and int(metric["known_near_boundary"])
        >= int(
            gates[
                "each_source_height_known_near_boundary_count_minimum"
            ]
        )
        and int(metric["risk_not_clip_min"]) > 0
        and int(metric["safe_not_clip_max"]) > 0
    )


def validate(
    protocol_path: Path,
    source_plan_path: Path,
    mechanics_path: Path,
) -> dict[str, Any]:
    hashes = {
        "protocol": _sha256(protocol_path),
        "source_plan": _sha256(source_plan_path),
        "mechanics": _sha256(mechanics_path),
    }
    if hashes != {
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "source_plan": EXPECTED_SOURCE_PLAN_SHA256,
        "mechanics": EXPECTED_MECHANICS_SHA256,
    }:
        raise ValueError("G0 validation frozen input hash mismatch")
    protocol = _load(protocol_path)
    source_plan = _load(source_plan_path)
    mechanics = _load(mechanics_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or source_plan.get("schema") != SOURCE_SCHEMA
        or mechanics.get("schema") != MECHANICS_SCHEMA
        or source_plan.get("terminal") != SOURCE_READY
        or mechanics.get("terminal") != MECHANICS_SUPPORTED
        or source_plan.get("protocol_sha256") != hashes["protocol"]
        or mechanics.get("protocol_sha256") != hashes["protocol"]
    ):
        raise ValueError("G0 validation identity or terminal mismatch")
    repository_root = Path(__file__).resolve().parents[3]
    for implementation in protocol["implementations"].values():
        implementation_path = (
            repository_root / str(implementation["path"])
        ).resolve()
        if (
            _sha256(implementation_path) != implementation["sha256"]
            or implementation.get("execution_authorized") is not True
        ):
            raise ValueError("G0 validation implementation receipt mismatch")
    roles = source_plan["roles"]
    development = roles["development_reuse"]
    fresh = roles["one_shot_fresh_evaluation"]
    heldout = roles["reserved_fresh_heldout"]
    _validate_role_ids(development, fresh, heldout)
    if (
        _session_ids(fresh)
        != protocol["source_role_contract"]["one_shot_fresh_evaluation"][
            "session_ids"
        ]
        or [str(item["role"]) for item in development[:6]]
        != ["train"] * 6
        or [str(item["role"]) for item in development[6:]]
        != ["dev"] * 3
        or any(item.get("fresh_evidence_credit") is not False for item in development)
        or any(item.get("fresh_evidence_obtained") is not False for item in fresh)
        or any(item.get("fresh_evidence_obtained") is not False for item in heldout)
        or any(source_plan["firewall"].values())
    ):
        raise ValueError("G0 validation source-role or firewall mismatch")
    sources = mechanics["sources"]
    gates = protocol["g0_d0_consumed_mechanics_canary"]["data_gates"]
    if (
        len(sources) != 12
        or len(set(_session_ids(sources))) != 12
        or int(mechanics["source_count"]) != 12
        or int(mechanics["frame_count"]) != 300
        or not all(mechanics["structural_canaries"].values())
        or not all(mechanics["checks"].values())
    ):
        raise ValueError("G0 validation mechanics coverage mismatch")
    metrics = [
        source["height_metrics"][height]
        for source in sources
        for height in HEIGHTS
    ]
    if any(
        set(source["height_metrics"]) != set(HEIGHTS)
        or source.get("passed")
        is not all(
            _metric_passes(source["height_metrics"][height], gates)
            for height in HEIGHTS
        )
        for source in sources
    ):
        raise ValueError("G0 validation source pass derivation mismatch")
    if not all(_metric_passes(metric, gates) for metric in metrics):
        raise ValueError("G0 validation data gate did not pass")
    summary = {
        "minimum_positive_known": min(
            int(metric["positive_known"]) for metric in metrics
        ),
        "minimum_negative_known": min(
            int(metric["negative_known"]) for metric in metrics
        ),
        "minimum_distinct_clipped_millimeter_bins": min(
            int(metric["distinct_clipped_target_millimeter_bins"])
            for metric in metrics
        ),
        "minimum_known_near_boundary": min(
            int(metric["known_near_boundary"]) for metric in metrics
        ),
        "maximum_risk_clip_min_fraction": max(
            float(metric["risk_clip_min_fraction"]) for metric in metrics
        ),
        "maximum_safe_clip_max_fraction": max(
            float(metric["safe_clip_max_fraction"]) for metric in metrics
        ),
        "total_binary_equivalence_violations": sum(
            int(metric["binary_equivalence_violations"])
            for metric in metrics
        ),
        "total_unknown_nonnull_target_violations": sum(
            int(metric["unknown_nonnull_target_violations"])
            for metric in metrics
        ),
        "total_unknown_to_safe_violations": sum(
            int(metric["unknown_to_safe_violations"]) for metric in metrics
        ),
    }
    return {
        "schema": VALIDATION_SCHEMA,
        "terminal": VALIDATED,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "evidence_role": "INDEPENDENT_OUTPUT_TERMINAL_VALIDATION",
        "claim_ceiling": "CONSUMED_SYNTHETIC_PROXY_MECHANICS_ONLY",
        "input_sha256": hashes,
        "role_counts": {
            "development_reuse": len(development),
            "one_shot_fresh_evaluation": len(fresh),
            "reserved_fresh_heldout": len(heldout),
        },
        "fresh_evaluation_session_ids": _session_ids(fresh),
        "reserved_heldout_session_ids": _session_ids(heldout),
        "mechanics_summary": summary,
        "checks": {
            "source_roles_disjoint_and_frozen": True,
            "fresh_outcomes_remain_unopened": True,
            "mechanics_terminal_rederived": True,
            "all_g0_d0_gates_recomputed": True,
        },
        "authorization": {
            "freeze_d1_contract": True,
            "fresh_evaluation_acquisition_executed": False,
            "student_training_executed": False,
            "reserved_heldout_acquisition_authorized": False,
            "future_or_temporal_experiment_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }


def _canonical_output(path: Path) -> Path:
    expected = (
        Path(__file__).resolve().parents[3]
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-signed-clearance-validation-20260801/"
        "validation.json"
    ).resolve()
    if path.resolve() != expected:
        raise ValueError("G0 validation output path is not canonical")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--mechanics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _canonical_output(args.output)
        if output.exists():
            raise FileExistsError("Refusing to overwrite G0 validation")
        report = validate(
            args.protocol.resolve(),
            args.source_plan.resolve(),
            args.mechanics.resolve(),
        )
        output.parent.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(
            tempfile.mkdtemp(
                prefix=f"{output.parent.name}.partial-",
                dir=output.parent.parent,
            )
        )
        with (partial / output.name).open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        if output.parent.exists():
            raise FileExistsError("G0 validation output root appeared")
        partial.replace(output.parent)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "validation_sha256": _sha256(output),
                }
            )
        )
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
