"""Re-evaluate cached R0 predictions with the posthoc causal scale R1 operator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import core as scale_core

REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from evaluate_consumed_tartanground import (  # noqa: E402
    INTRINSICS,
    sha256,
    strict_band_values,
    summarize_arm,
    write_json_new,
)
from evaluate_metric3d_clearance_field_a0 import clearance_field  # noqa: E402


def effect_gates(summary: dict, jointly_better: list[dict]) -> dict[str, bool]:
    macro = summary["parent_macro"]
    return {
        "known_coverage": macro["known_coverage"] is not None and macro["known_coverage"] >= 0.60,
        "clearance_mae": macro["clearance_mae_m"] is not None and macro["clearance_mae_m"] <= 0.25,
        "envelope_agreement": macro["envelope_agreement"] is not None and macro["envelope_agreement"] >= 0.90,
        "false_clear": macro["false_clear_rate"] is not None and macro["false_clear_rate"] <= 0.05,
        "temporal_delta_mae": macro["temporal_delta_mae_m"] is not None and macro["temporal_delta_mae_m"] <= 0.15,
        "jointly_better_parents": sum(row["jointly_better"] for row in jointly_better) >= 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0-result", required=True, type=Path)
    parser.add_argument("--optimization-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    r0 = json.loads(arguments.r0_result.read_text(encoding="utf-8"))
    receipt = json.loads(arguments.optimization_receipt.read_text(encoding="utf-8"))
    if r0.get("data_role") != "CONSUMED_DEVELOPMENT":
        raise ValueError("R0 input is not consumed Development evidence")
    if receipt.get("status") != "POSTHOC_CONSUMED_DEVELOPMENT_OPTIMIZATION_FROZEN":
        raise ValueError("optimization receipt is not frozen")
    window = int(receipt["selected_causal_window_valid_scales"])
    if window != scale_core.TEMPORAL_SCALE_WINDOW:
        raise ValueError("receipt/implementation window mismatch")

    output_records = []
    for parent_id in sorted({row["parent_id"] for row in r0["records"]}):
        scale_history: list[float] = []
        parent_rows = sorted(
            (row for row in r0["records"] if row["parent_id"] == parent_id),
            key=lambda row: row["anchor_frame_id"],
        )
        for row in parent_rows:
            with np.load(row["prediction_path"]) as payload:
                relative_depth = payload["da_depth"].astype(np.float64)
            recovery = scale_core.recover_metric_scale(
                relative_depth,
                INTRINSICS,
                scale_core.CameraHeightReceipt(parent_id, parent_id, row["height_m"], 0.0),
                parent_id,
                parent_id,
            )
            candidate = None
            reason = None
            causal_scale = None
            if recovery["status"] == "VALID":
                scale_history.append(float(recovery["scale"]))
                causal_scale = scale_core.causal_median_scale(scale_history, window)
                plane = recovery["ground"]
                candidate = strict_band_values(
                    clearance_field(
                        relative_depth * causal_scale,
                        INTRINSICS,
                        plane_override=(
                            plane.normal,
                            row["height_m"],
                            plane.normalized_median_residual * row["height_m"],
                        ),
                    )
                )
                if candidate is None:
                    reason = "STRICT_CLEARANCE_BAND_UNKNOWN"
            else:
                reason = str(recovery["reason"])
            output_records.append(
                {
                    **row,
                    "r1_candidate": candidate,
                    "r1_unknown_reason": reason,
                    "r1_causal_scale": causal_scale,
                    "r1_valid_scale_history_count": len(scale_history),
                }
            )

    summary = summarize_arm(output_records, "r1_candidate")
    raw_by_parent = {row["parent_id"]: row for row in r0["raw_da"]["parents"]}
    jointly_better = []
    for row in summary["parents"]:
        raw = raw_by_parent[row["parent_id"]]
        better = (
            row["clearance_mae_m"] is not None
            and raw["clearance_mae_m"] is not None
            and row["clearance_mae_m"] < raw["clearance_mae_m"]
            and row["false_clear_rate"] is not None
            and raw["false_clear_rate"] is not None
            and row["false_clear_rate"] <= raw["false_clear_rate"]
        )
        jointly_better.append({"parent_id": row["parent_id"], "jointly_better": better})
    gates = effect_gates(summary, jointly_better)
    unknown_reasons: dict[str, int] = {}
    for row in output_records:
        if row["r1_candidate"] is None:
            reason = row["r1_unknown_reason"] or "UNKNOWN"
            unknown_reasons[reason] = unknown_reasons.get(reason, 0) + 1

    result = {
        "schema": "blindassist_known_camera_height_ground_scale_consumed_temporal_r1_result_v1",
        "data_role": "CONSUMED_DEVELOPMENT_POSTHOC_OPTIMIZATION",
        "claim_ceiling": "SAME_CONSUMED_SYNTHETIC_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "r0_result_sha256": sha256(arguments.r0_result),
        "optimization_receipt_sha256": sha256(arguments.optimization_receipt),
        "causal_window_valid_scales": window,
        "records": output_records,
        "raw_da": r0["raw_da"],
        "r0_known_height_candidate": r0["known_height_candidate"],
        "r1_temporal_candidate": summary,
        "jointly_better_parents": jointly_better,
        "candidate_unknown_reason_counts": unknown_reasons,
        "gates": gates,
        "terminal": (
            "POSTHOC_CONSUMED_R1_ALL_ABSOLUTE_GATES_PASS_NOT_GENERALIZATION_EVIDENCE"
            if all(gates.values())
            else "POSTHOC_CONSUMED_R1_ABSOLUTE_GATES_FAIL_STOP_OPTIMIZATION"
        ),
    }
    write_json_new(arguments.output, result)
    print(json.dumps({key: result[key] for key in ("r1_temporal_candidate", "gates", "terminal")}, indent=2))


if __name__ == "__main__":
    main()
