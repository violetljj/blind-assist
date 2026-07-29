#!/usr/bin/env python3
"""Validate causal F-1B0 semantic and geometry timing traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class TimingReceiptError(ValueError):
    pass


SEMANTIC_FIELDS = (
    "capturedAt",
    "receivedAt",
    "queuedAt",
    "startedAt",
    "completedAt",
    "publishedAt",
    "availableAt",
    "consumedAt",
)
GEOMETRY_FIELDS = (
    "previousObservationAt",
    "currentObservationAt",
    "geometryQueuedAt",
    "geometryStartedAt",
    "geometryCompletedAt",
    "geometryPublishedAt",
    "geometryAvailableAt",
    "geometryConsumedAt",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer_trace(row: dict[str, Any], fields: tuple[str, ...], where: str) -> list[int]:
    values = []
    for field in fields:
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TimingReceiptError(f"{where}.{field} must be a non-negative integer")
        values.append(value)
    return values


def validate_receipt(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != "blindassist_dual_loop_f1b0_timing_baseline_v1":
        raise TimingReceiptError("schema mismatch")
    if value.get("effect_outputs_accessed") is not False:
        raise TimingReceiptError("timing baseline accessed effect outputs")
    if value.get("alerts_invoked") is not False:
        raise TimingReceiptError("timing baseline invoked alerts")
    semantic = value.get("semantic_results")
    geometry = value.get("geometry_results")
    if not isinstance(semantic, list) or len(semantic) < 20:
        raise TimingReceiptError("semantic trace has fewer than 20 results")
    if not isinstance(geometry, list) or len(geometry) < 20:
        raise TimingReceiptError("geometry trace has fewer than 20 results")

    for index, row in enumerate(semantic):
        if not isinstance(row, dict):
            raise TimingReceiptError(f"semantic_results[{index}] is not an object")
        values = integer_trace(row, SEMANTIC_FIELDS, f"semantic_results[{index}]")
        if values != sorted(values):
            raise TimingReceiptError(f"semantic_results[{index}] is not causal")
        if row.get("clockDomain") != "ANDROID_ELAPSED_REALTIME_NANOS":
            raise TimingReceiptError(f"semantic_results[{index}] clock domain is unverified")
        if row.get("detectorBackend") != "qualcomm_qnn_htp":
            raise TimingReceiptError(f"semantic_results[{index}] is not production QNN")
        if not isinstance(row.get("backendRouteReason"), str) or not row[
            "backendRouteReason"
        ]:
            raise TimingReceiptError(f"semantic_results[{index}] lacks route reason")
        if row.get("dropReason") not in (None, "", "NONE"):
            raise TimingReceiptError(f"semantic_results[{index}] is marked dropped")

    for index, row in enumerate(geometry):
        if not isinstance(row, dict):
            raise TimingReceiptError(f"geometry_results[{index}] is not an object")
        values = integer_trace(row, GEOMETRY_FIELDS, f"geometry_results[{index}]")
        if values != sorted(values):
            raise TimingReceiptError(f"geometry_results[{index}] is not causal")
        if row.get("clockDomain") != "ANDROID_ELAPSED_REALTIME_NANOS":
            raise TimingReceiptError(f"geometry_results[{index}] clock domain is unverified")
        if row.get("dropReason") not in (None, "", "NONE"):
            raise TimingReceiptError(f"geometry_results[{index}] is marked dropped")
        if not isinstance(row.get("abstainReason"), str):
            raise TimingReceiptError(f"geometry_results[{index}] lacks abstain reason")

    return {
        "semantic_result_count": len(semantic),
        "geometry_result_count": len(geometry),
        "semantic_backend": "qualcomm_qnn_htp",
        "clock_domain": "ANDROID_ELAPSED_REALTIME_NANOS",
        "terminal": "READY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    result = validate_receipt(receipt)
    output = {
        "schema": "blindassist_dual_loop_f1b0_timing_validation_v1",
        "receipt_sha256": sha256_file(args.receipt),
        "timing_protocol_status": "VALID",
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
