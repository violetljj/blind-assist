#!/usr/bin/env python3
"""Join all frozen D1 chunks, derive candidate geometry, and evaluate once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.assistive_geometry.arkitscenes_truth_reader import TruthReaderPolicy, derive_assistive_truth
from scripts.research.hftf.deployment.depthart.evaluate_depthart_task_preserving_d1_quality import (
    BANDS,
    HORIZONS,
    PROTOCOL_ID,
    SCHEMA,
    evaluate,
)


TASK_HORIZON_M = 2.0


def chunk_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    chunk_size = int(protocol["execution"]["chunk_size_frames"])
    require(chunk_size == 50 and 300 % chunk_size == 0, "chunk size drift")
    return [
        {"chunk_index": session_index * (300 // chunk_size) + start // chunk_size,
         "session_index": session_index, "visit_id": identity["visit_id"],
         "video_id": identity["video_id"], "frame_start": start, "frame_stop": start + chunk_size}
        for session_index, identity in enumerate(protocol["cohort"]["ordered_sessions"])
        for start in range(0, 300, chunk_size)
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clearance_payload(band: dict[str, Any] | None) -> dict[str, Any]:
    band = band or {}
    value = band.get("clearance_m")
    occupied = band.get("occupied_by_horizon", {})
    if value is not None and np.isfinite(value):
        return {"clearance_valid": True, "clearance_m": min(float(value), TASK_HORIZON_M)}
    if all(occupied.get(str(horizon)) is False for horizon in HORIZONS):
        return {"clearance_valid": True, "clearance_m": TASK_HORIZON_M}
    return {"clearance_valid": False, "clearance_m": None}


def state(value: bool | None) -> str:
    return "UNKNOWN_GROUND" if value is None else "OCCUPIED" if value else "CLEAR"


def add_candidate(record: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    require(output["frame_id"] == record["frame_id"], "candidate/reference frame mapping drift")
    path = Path(output["path"])
    require(path.is_file() and path.stat().st_size == output["bytes"] and sha256(path) == output["sha256"],
            f"candidate output drift: {path}")
    depth = np.fromfile(path, dtype=np.float32).reshape(608, 448)
    require(np.all(np.isfinite(depth)), "candidate depth contains nonfinite values")
    geometry = derive_assistive_truth(
        depth, np.full(depth.shape, 2, dtype=np.uint8),
        np.asarray(record["intrinsics_tensor"], dtype=np.float64),
        np.asarray(record["up_camera"], dtype=np.float64), TruthReaderPolicy(),
    )
    row = {key: record[key] for key in (
        "parent_id", "session_id", "frame_id", "frame_index", "timestamp_ns",
        "orientation", "orientation_index",
    )}
    row["bands"] = []
    for source in record["bands"]:
        name = source["band"]
        candidate_band = geometry.get("bands", {}).get(name)
        cells = []
        for source_cell in source["cells"]:
            horizon = float(source_cell["horizon_m"])
            cells.append({
                "horizon_m": horizon,
                "truth": source_cell["truth"], "reference": source_cell["reference"],
                "candidate": {"state": state((candidate_band or {}).get("occupied_by_horizon", {}).get(str(horizon)))},
            })
        row["bands"].append({
            "band": name, "truth": source["truth"], "reference": source["reference"],
            "candidate": clearance_payload(candidate_band), "cells": cells,
        })
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--payload-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path, activation_path = args.protocol.resolve(), args.activation_receipt.resolve()
    protocol, activation = load_json(protocol_path), load_json(activation_path)
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol drift")
    require(activation.get("status") == "OUTCOME_ACCESS_ACTIVATED" and activation.get("execution_authorized") is True,
            "D1 outcome access is not activated")
    require(activation["protocol_sha256"] == sha256(protocol_path), "activation/protocol drift")
    require(protocol["bindings"]["finalizer"]["sha256"] == sha256(Path(__file__)), "finalizer SHA drift")
    frozen_policy = json.loads(json.dumps(asdict(TruthReaderPolicy())))
    require(protocol["task_postprocess"]["truth_reader_policy"] == frozen_policy,
            "truth reader policy drift")
    rows = []
    chunk_receipts = []
    for chunk_index, chunk in enumerate(chunk_schedule(protocol)):
        chunk_root = args.quality_root.resolve() / f"chunk-{chunk_index:02d}"
        materialization_path = chunk_root / "materialization-receipt.json"
        device_path = chunk_root / "device-run-receipt.json"
        materialization, device = load_json(materialization_path), load_json(device_path)
        require(materialization["chunk"] == device["chunk"] == chunk, "chunk receipt schedule drift")
        require(device["materialization_receipt_sha256"] == sha256(materialization_path), "device/materialization drift")
        require(len(materialization["records"]) == len(device["outputs"]) == 50, "chunk output count drift")
        rows.extend(add_candidate(record, output)
                    for record, output in zip(materialization["records"], device["outputs"], strict=True))
        chunk_receipts.append({"chunk_index": chunk_index,
                               "materialization_sha256": sha256(materialization_path),
                               "device_run_sha256": sha256(device_path)})
    payload = {"schema": SCHEMA, "protocol_id": PROTOCOL_ID, "rows": rows,
               "chunk_receipts": chunk_receipts, "development_outcomes_opened": True,
               "r2_cohort_accessed": False}
    atomic_json(args.payload_output.resolve(), payload)
    result = evaluate(protocol, payload)
    result["identities"] = {
        "protocol_sha256": sha256(protocol_path), "activation_receipt_sha256": sha256(activation_path),
        "payload_sha256": sha256(args.payload_output.resolve()), "chunk_receipts": chunk_receipts,
    }
    atomic_json(args.result_output.resolve(), result)
    print(json.dumps({"status": result["status"], "terminal": result["terminal"],
                      "gates": result["gates"], "result_sha256": sha256(args.result_output.resolve())}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
