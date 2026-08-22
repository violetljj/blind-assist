#!/usr/bin/env python3
"""Development-only candidate-wise depth fusion on a sealed completion run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import (
    DAV2_ONNX_SHA256,
    _atomic_json,
    _confusion,
    _read,
    _require,
    preprocess_depth,
    region_depth_median,
)
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_CANDIDATE_WISE_DEPTH_FUSION_DEV_V1"
RUN_SCHEMA = "blindassist_candidate_wise_depth_fusion_dev_run_v1"
EVAL_SCHEMA = "blindassist_candidate_wise_depth_fusion_dev_evaluation_v1"


def _select(candidates: Sequence[Mapping[str, Any]], center_x: float, interaction_range_m: float) -> Mapping[str, Any] | None:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["bbox_xyxy"][0] <= center_x <= candidate["bbox_xyxy"][2]
        and candidate.get("predicted_region_depth_m") is not None
        and float(candidate["predicted_region_depth_m"]) <= interaction_range_m
    ]
    return max(eligible, key=lambda candidate: (float(candidate["proposal_score"]), -int(candidate["provider_rank"]))) if eligible else None


def run(public_path: Path, sealed_run_path: Path, sealed_journal_path: Path, depth_onnx_path: Path, output_path: Path, journal_path: Path) -> dict[str, Any]:
    _require(not output_path.exists() and not journal_path.exists(), "development output already exists")
    public, sealed, sealed_journal = _read(public_path), _read(sealed_run_path), _read(sealed_journal_path)
    _require(sealed.get("private_truth_access") is False, "sealed run crossed private boundary")
    _require(sealed.get("public_sha256") == sha256(public_path), "sealed run/public binding mismatch")
    _require(sealed_journal.get("status") == "COMPLETED" and sealed_journal.get("run_sha256") == sha256(sealed_run_path), "sealed run is incomplete")
    _require(depth_onnx_path.is_file() and sha256(depth_onnx_path) == DAV2_ONNX_SHA256, "metric depth ONNX drift")

    import onnxruntime as ort

    session = ort.InferenceSession(str(depth_onnx_path.resolve()), providers=["CPUExecutionProvider"])
    public_cases = {case["case_id"]: case for case in public["cases"]}
    sealed_cases = {case["case_id"]: case for case in sealed["cases"]}
    _require(set(public_cases) == set(sealed_cases), "development roster mismatch")
    journal = {
        "schema_version": "blindassist_candidate_wise_depth_fusion_dev_journal_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "STARTED",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases_completed": 0,
        "depth_calls_completed": 0,
        "development_only": True,
        "threshold_or_rule_sweep": False,
    }
    _atomic_json(journal_path, journal)
    rows = []
    for case_id in sorted(public_cases):
        image_path = Path(public_cases[case_id]["query"]["image_path"])
        _require(image_path.is_file() and sha256(image_path) == public_cases[case_id]["query"]["image_sha256"], "public image drift")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            started = time.perf_counter()
            depth = np.asarray(session.run(["depth_m"], {"image": preprocess_depth(image)})[0][0], dtype=np.float32)
            latency_ms = (time.perf_counter() - started) * 1000.0
        candidates = []
        for candidate in sealed_cases[case_id]["candidates"]:
            candidates.append(dict(candidate) | {"predicted_region_depth_m": region_depth_median(depth, candidate["bbox_xyxy"], width, height)})
        selected = _select(candidates, width / 2.0, 2.0)
        rows.append({
            "case_id": case_id,
            "image_width": width,
            "image_height": height,
            "candidates": candidates,
            "selected_candidate": selected,
            "completion": selected is not None,
            "depth_latency_ms": latency_ms,
        })
        journal.update({"status": "ACTIVE", "cases_completed": journal["cases_completed"] + 1, "depth_calls_completed": journal["depth_calls_completed"] + 1})
        _atomic_json(journal_path, journal)
    payload = {
        "schema_version": RUN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_sha256": sha256(public_path),
        "sealed_run_sha256": sha256(sealed_run_path),
        "sealed_journal_sha256": sha256(sealed_journal_path),
        "depth_onnx_sha256": sha256(depth_onnx_path),
        "selection_rule": "highest provider score among candidates covering center ray with predicted inset-median depth <= 2.0m",
        "development_only": True,
        "private_truth_access": False,
        "threshold_or_rule_sweep": False,
        "cases": rows,
    }
    _atomic_json(output_path, payload)
    journal.update({"status": "COMPLETED", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_sha256": sha256(output_path)})
    _atomic_json(journal_path, journal)
    return payload


def evaluate(private_path: Path, run_path: Path, journal_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "development evaluation already exists")
    private, run_payload, journal = _read(private_path), _read(run_path), _read(journal_path)
    _require(run_payload.get("schema_version") == RUN_SCHEMA and run_payload.get("private_truth_access") is False, "development run boundary mismatch")
    _require(journal.get("status") == "COMPLETED" and journal.get("run_sha256") == sha256(run_path), "development journal incomplete")
    truths = {case["case_id"]: case for case in private["cases"]}
    runs = {case["case_id"]: case for case in run_payload["cases"]}
    _require(set(truths) == set(runs), "development evaluation roster mismatch")
    rows = []
    for case_id in sorted(truths):
        selected = runs[case_id]["selected_candidate"]
        legal = truths[case_id]["legal_targets"]
        matched_index = None
        if selected is not None:
            selected_box = validated_box(selected["bbox_xyxy"], f"{case_id} selected")
            overlaps = [iou(selected_box, validated_box(target["target_bbox_xyxy"], f"{case_id} target")) for target in legal]
            matched_index = int(np.argmax(overlaps)) if overlaps and max(overlaps) >= 0.30 else None
        matched_depth = float(legal[matched_index]["target_depth_median_m"]) if matched_index is not None else None
        rows.append({
            "case_id": case_id,
            "decision": bool(runs[case_id]["completion"]),
            "target_selected": matched_index is not None,
            "truth_positive": matched_depth is not None and matched_depth <= 2.0,
            "matched_target_depth_m": matched_depth,
            "predicted_selected_depth_m": selected.get("predicted_region_depth_m") if selected is not None else None,
            "selected_provider_rank": selected.get("provider_rank") if selected is not None else None,
        })
    confusion = _confusion(rows, "decision")
    payload = {
        "schema_version": EVAL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "case_count": len(rows),
        "completion_confusion": confusion,
        "target_selection_count": sum(row["target_selected"] for row in rows),
        "selected_provider_rank_histogram": {str(rank): sum(row["selected_provider_rank"] == rank for row in rows) for rank in range(1, 11)},
        "rows": rows,
        "development_only": True,
        "confirmation_claim_authorized": False,
        "terminal": "DEV_CANDIDATE_DEPTH_FUSION_PROMISING" if confusion["fp"] <= 1 and confusion["tp"] >= 2 else "DEV_CANDIDATE_DEPTH_FUSION_NOT_PROMISING",
    }
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("public", "sealed-run", "sealed-journal", "depth-onnx", "output", "journal"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    evaluation = sub.add_parser("evaluate")
    for name in ("private", "run", "journal", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.public, args.sealed_run, args.sealed_journal, args.depth_onnx, args.output, args.journal)
    else:
        result = evaluate(args.private, args.run, args.journal, args.output)
        print(json.dumps({key: result[key] for key in ("completion_confusion", "target_selection_count", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
