from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from scripts.research.riskseg_r0_pidnet_preflight.modeling import sha256_file


FULL_DELEGATION_MARKER = (
    "TfLiteQnnDelegate delegate: 163 nodes delegated out of 163 nodes with 1 partitions."
)
RESTORE_MARKER = "caching in RESTORE MODE."
QNN_ERROR_PATTERN = re.compile(
    r"(?:\sE\s+tflite\s+:.*\[(?:Qnn|QNN)\])|"
    r"(?:\[(?:Qnn|QNN)[^\]]*\].*(?:ERROR|Failed|failure))"
)


def validate(receipt_path: Path, logcat_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    logcat = logcat_path.read_text(encoding="utf-8", errors="replace")
    gates = receipt["gates"]
    expected_gates = {
        "failure_count_zero",
        "total_p95_at_most_100_ms",
        "degradation_at_most_1_20x",
        "no_severe_thermal",
        "qnn_cached_context_created",
        "strict_int8_tensor_contract",
        "argmax_in_0_to_3",
    }
    checks = {
        "formal_status": (
            receipt["status"] == "QNN_HTP_FORMAL_SUSTAINED_PREFLIGHT_PASS"
            and receipt["formal_sustained_run"] is True
        ),
        "duration_at_least_600000_ms": receipt["duration_observed_ms"] >= 600_000,
        "failure_count_zero": receipt["failure_count"] == 0,
        "all_receipt_gates_true": expected_gates.issubset(gates)
        and all(gates[name] is True for name in expected_gates),
        "device_is_sm_s9280": receipt["device"]["model"] == "SM-S9280",
        "soc_is_sm8650": receipt["device"]["soc_model"] == "SM8650",
        "qnn_quantized_contract": (
            receipt["delegate"]["backend"] == "QNN_HTP"
            and receipt["delegate"]["precision"] == "HTP_PRECISION_QUANTIZED"
            and receipt["delegate"]["capability"] == "HTP_RUNTIME_QUANTIZED"
        ),
        "full_graph_delegated_twice": logcat.count(FULL_DELEGATION_MARKER) >= 2,
        "cached_context_restored_twice": logcat.count(RESTORE_MARKER) >= 2,
        "no_qnn_error_log": not QNN_ERROR_PATTERN.search(logcat),
        "model_sha_bound": (
            receipt["model"]["sha256"]
            == "d492d05012dd750cd6e5e642ea7d56682fa2fd1fba2bb3052b10ab1ebb0c2ddb"
        ),
        "class_order_exact": receipt["model"]["class_order"]
        == [
            "walkable",
            "blocking_obstacle",
            "boundary_level_change",
            "unknown_nonwalkable",
        ],
    }
    status = (
        "PIDNET_S_TECHNICAL_PREFLIGHT_PASS"
        if all(checks.values())
        else "PIDNET_S_TECHNICAL_PREFLIGHT_INVALID"
    )
    return {
        "schema_version": "blindassist.riskseg_r0.device_preflight_validation.v1",
        "protocol_id": "RISKSEG-R0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "logcat_path": str(logcat_path),
        "logcat_sha256": sha256_file(logcat_path),
        "full_delegation_marker_count": logcat.count(FULL_DELEGATION_MARKER),
        "restore_mode_marker_count": logcat.count(RESTORE_MARKER),
        "checks": checks,
        "metrics": {
            "sample_count": receipt["sample_count"],
            "total_p95_ms": receipt["timing_ms"]["total"]["p95"],
            "initial_window_p95_ms": receipt["timing_ms"]["initial_window_p95"],
            "final_window_p95_ms": receipt["timing_ms"]["final_window_p95"],
            "final_over_initial_p95_ratio": receipt["timing_ms"][
                "final_over_initial_p95_ratio"
            ],
            "inference_p95_ms": receipt["timing_ms"]["inference"]["p95"],
            "maximum_thermal_status": receipt["thermal"]["maximum_status"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--logcat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.receipt.resolve(), args.logcat.resolve())
    args.output.resolve().write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PIDNET_S_TECHNICAL_PREFLIGHT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
