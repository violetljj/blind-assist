from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner as discovery,
)


PROTOCOL_ID = "RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1"
IMPLEMENTATION_VERSION = "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
EXPECTED_ARM_IDENTITIES = {
    "raw": {
        "implementation_version": "ADVIO_WXYZ_TCAMIMU_CONTINUOUS_R2",
        "implementation_hashes": {
            "mechanism_runner": (
                "00bf66a229e1eae408420e5a320d3de0"
                "b981c6a1dfa0a1b00db20151dad54715"
            ),
            "discovery_engine": (
                "47cdea83741d5227fd37a384cf574336"
                "6c6734b6a9488d6e1363d0b6d4fa0c0f"
            ),
        },
    },
    "undistorted": {
        "implementation_version": (
            "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
        ),
        "implementation_hashes": {
            "mechanism_runner": (
                "6631562a9f30063979ebde94fca22418"
                "d1281ffcc98571a024e654436f1f083b"
            ),
            "discovery_engine": (
                "0290fc4763e1a0e196851334fc4838fb"
                "5a03c8b96e8c26a9f4d11746f0334645"
            ),
        },
    },
}
EXPECTED_POSE_TO_CAMERA_ROTATION = [
    [0.9999763379093255, -0.004079205042965442, -0.005539287650170447],
    [-0.004066386342107199, -0.9999890330121858, 0.0023234365646622014],
    [-0.00554870467502187, -0.0023008567036498766, -0.9999819588046867],
]
STRATA = {
    "high": (343, 462),
    "low": (2, 121),
}
CHUNK_BOUNDARIES = (150, 300, 400, 450, 500, 550)


def load_arm(
    path: Path, expected_arm: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (path / "pair_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if summary["protocol_id"] != PROTOCOL_ID or len(rows) != 600:
        raise ValueError("ARM_IDENTITY_OR_DENOMINATOR_MISMATCH")
    if [row["pair_index"] for row in rows] != list(range(600)):
        raise ValueError("PAIR_INDEX_CONTINUITY_MISMATCH")
    context = summary.get("evidence_context", {})
    expected_identity = EXPECTED_ARM_IDENTITIES[expected_arm]
    if (
        context.get("implementation_version")
        != expected_identity["implementation_version"]
        or context.get("ab_arm") != expected_arm
        or context.get("runtime_pilot") is not False
        or context.get("threshold_changed") is not False
        or context.get("three_pair_rule_changed") is not False
        or context.get("implementation_hashes")
        != expected_identity["implementation_hashes"]
    ):
        raise ValueError("R3_IMPLEMENTATION_OR_ARM_IDENTITY_MISMATCH")
    source = summary.get("source", {})
    if (
        source.get("pose_quaternion_component_order") != "wxyz"
        or source.get("pose_to_camera_rotation")
        != EXPECTED_POSE_TO_CAMERA_ROTATION
        or source.get("distortion_correction_applied")
        is not (expected_arm == "undistorted")
    ):
        raise ValueError("R3_SOURCE_COORDINATE_OR_DISTORTION_MISMATCH")
    execution = summary.get("execution", {})
    if (
        execution.get("candidate_pair_count") != 600
        or execution.get("threshold_per_s") != 0.01
        or execution.get("required_consecutive_pairs") != 3
        or execution.get("support_manager_baseline_pair_count") != 1
        or execution.get("single_process_pair_state_continuous") is not True
    ):
        raise ValueError("R3_EXECUTION_RULE_MISMATCH")
    baseline = [
        row["pair_index"]
        for row in rows
        if row.get("support_manager", {}).get("baseline_only") is True
    ]
    if baseline != [0]:
        raise ValueError("R3_CONTINUOUS_STATE_BASELINE_MISMATCH")
    return summary, rows


def method(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    field = f"{prefix}_expansion_median_per_s"
    trigger = f"{prefix}_three_pair_trigger"
    values = np.asarray(
        [row[field] for row in rows if row.get(field) is not None],
        dtype=np.float64,
    )
    result = discovery.method_summary(rows, field, trigger)
    result["median_abs_response_per_s"] = (
        float(np.median(np.abs(values))) if values.size else None
    )
    result["angular_speed_spearman"] = discovery.correlation(rows, field)[
        "spearman"
    ]
    return result


def arm_result(
    summary: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = [
        int(row["pair_index"])
        for row in rows
        if row.get("support_manager", {}).get("baseline_only") is True
    ]
    return {
        "distortion_correction_applied": summary["source"][
            "distortion_correction_applied"
        ],
        "full": {
            "evaluable_pair_fraction": summary["execution"][
                "evaluable_pair_fraction"
            ],
            "raw": method(rows, "raw"),
            "compensated": method(rows, "compensated"),
            "raw_abs_angular_speed_spearman": summary["diagnostics"][
                "angular_speed_correlation"
            ]["raw_abs_expansion"]["spearman"],
            "compensated_abs_angular_speed_spearman": summary[
                "diagnostics"
            ]["angular_speed_correlation"]["compensated_abs_expansion"][
                "spearman"
            ],
            "median_raw_track_count": float(
                np.median([row.get("raw_track_count", 0) for row in rows])
            ),
            "median_compensated_track_count": float(
                np.median(
                    [row.get("compensated_track_count", 0) for row in rows]
                )
            ),
        },
        "strata": {
            name: {
                "pair_start": start,
                "pair_end_inclusive": end,
                "raw": method(rows[start : end + 1], "raw"),
                "compensated": method(
                    rows[start : end + 1], "compensated"
                ),
            }
            for name, (start, end) in STRATA.items()
        },
        "continuous_state": {
            "support_manager_baseline_pair_indices": baseline,
            "expected_only_first_pair": baseline == [0],
            "former_chunk_boundaries_are_not_baseline": all(
                not rows[index]
                .get("support_manager", {})
                .get("baseline_only", False)
                for index in CHUNK_BOUNDARIES
            ),
        },
    }


def run_analysis(
    raw_dir: Path, undistorted_dir: Path
) -> dict[str, Any]:
    raw_summary, raw_rows = load_arm(raw_dir, "raw")
    und_summary, und_rows = load_arm(undistorted_dir, "undistorted")
    raw_identity = [
        (
            row["frame_index_previous_zero_based"],
            row["frame_index_current_zero_based"],
            row["previous_timestamp_s"],
            row["current_timestamp_s"],
        )
        for row in raw_rows
    ]
    und_identity = [
        (
            row["frame_index_previous_zero_based"],
            row["frame_index_current_zero_based"],
            row["previous_timestamp_s"],
            row["current_timestamp_s"],
        )
        for row in und_rows
    ]
    if raw_identity != und_identity:
        raise ValueError("AB_PAIR_IDENTITY_MISMATCH")
    return {
        "schema": "rcle.rotation_compensation.mechanism_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "pair_identity_equal": True,
        "pair_count_per_arm": 600,
        "threshold_per_s": 0.01,
        "required_consecutive_pairs": 3,
        "implementation_version": IMPLEMENTATION_VERSION,
        "arm_implementation_identities": EXPECTED_ARM_IDENTITIES,
        "raw_r2_reuse_basis": (
            "R2 and R3 50-pair raw ledgers are byte-identical: "
            "ab6529aceebce7a3813876a83f59b155"
            "ab3a3a2425c0a0aa20cfa074c0926b79"
        ),
        "strict_r3_identity_validation": True,
        "raw_image_arm": arm_result(raw_summary, raw_rows),
        "undistorted_image_arm": arm_result(und_summary, und_rows),
        "forbidden_metrics_computed": [],
        "claim_ceiling": (
            "IMPLEMENTATION_AND_SINGLE_SESSION_MECHANISM_DIAGNOSTIC"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--undistorted-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(
        args.raw_dir.resolve(), args.undistorted_dir.resolve()
    )
    discovery.write_json(args.output.resolve(), result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
