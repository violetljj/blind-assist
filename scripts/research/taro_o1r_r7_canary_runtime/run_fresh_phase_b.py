#!/usr/bin/env python3
"""Open FARO only after sealed R7 fresh Phase A and score frozen gates."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation_io as r6io
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as cohort
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_inventory
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as phase_a
from scripts.research.taro_o1r_r7_canary_runtime import validate_fresh_confirmation_protocol


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r7_fresh_phase_b_execution_lock.v1"
LOCK_ID = "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_B_FARO_ONE_SHOT_EXECUTION_LOCK"
PHASE_A_ROOT = "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-phase-a-r1"
INVENTORY_PATH = phase_a.INVENTORY_PATH
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-phase-b-r0"
PASS_TERMINAL = "TARO_O1R_R7_POSITIVE_OCCUPANCY_FRESH_CONFIRMATION_PASS"
FAIL_TERMINAL = "TARO_O1R_R7_POSITIVE_OCCUPANCY_FRESH_CONFIRMATION_FAIL"
NOT_EVALUABLE_TERMINAL = "TARO_O1R_R7_FRESH_CONFIRMATION_NOT_EVALUABLE_DUAL_CLASS_COVERAGE"
INVALID_TERMINAL = "TARO_O1R_R7_FRESH_CONFIRMATION_EXECUTION_INVALID"
FRAME_COUNT = 170
QUERY_COUNT = 1530

EXPECTED_BINDINGS = {
    "R7_FRESH_PROTOCOL": "docs/research/taro/TARO_O1R_R7_FRESH_PARENT_DISJOINT_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R7_DATA_LOCK": "docs/research/taro/TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK_2026-08-12.json",
    "R7_INVENTORY_PLAN": INVENTORY_PATH,
    "R7_PHASE_A_COMPLETION": f"{PHASE_A_ROOT}/phase-a-completion.json",
    "R7_PHASE_A_RESULT": f"{PHASE_A_ROOT}/result.json",
    "R7_PHASE_A_MANIFEST": f"{PHASE_A_ROOT}/manifest.json",
    "R7_PHASE_A_RUNNER": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_phase_a.py",
    "R7_PHASE_A_R1_RUNNER": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_phase_a_r1.py",
    "R7_CANARY_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R7_PHASE_B_RUNNER": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_phase_b.py",
}
EXPECTED_AUTHORITY = {
    "phase_a_reload": True,
    "faro_payload_read": True,
    "faro_frame_count": FRAME_COUNT,
    "truth_label_construction": True,
    "fixed_gate_evaluation": True,
    "source_reselection": False,
    "threshold_fit": False,
    "training": False,
    "network": False,
    "device": False,
    "product": False,
    "safety": False,
}


class FreshPhaseBError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FreshPhaseBError(code, message, **context)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R7_PHASE_B_SEAL_COLLISION", "caller supplied content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _label_relative(frame: r6io.R6FrameRef) -> str:
    return f"labels/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _verify_phase_a_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    require(
        manifest.get("schema") == "blindassist.taro.o1r.r7_fresh_phase_a_manifest.v1"
        and manifest.get("terminal") == "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_A_SEALED_PASS_R1",
        "R7_PHASE_B_PHASE_A_MANIFEST_INVALID",
        "Phase-A manifest terminal drift",
    )
    files = manifest.get("files")
    require(isinstance(files, dict) and len(files) == manifest.get("file_count_before_manifest") == 854, "R7_PHASE_B_PHASE_A_MANIFEST_INVALID", "Phase-A manifest cardinality drift")
    total = 0
    for relative, receipt in files.items():
        target = materializer.safe_join(root, relative)
        require(
            target.is_file() and target.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(target) == receipt.get("sha256") and receipt.get("path") == relative,
            "R7_PHASE_B_PHASE_A_FILE_DRIFT",
            f"Phase-A artifact drift: {relative}",
        )
        total += target.stat().st_size
    require(total == manifest.get("bytes_before_manifest"), "R7_PHASE_B_PHASE_A_BYTE_DRIFT", "Phase-A evidence byte total drift")
    return manifest


def _load_phase_a_records(frames: Sequence[r6io.R6FrameRef], root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    completion = phase_a._validate_seal(_read_json(root / "phase-a-completion.json"), "blindassist.taro.o1r.r7_fresh_phase_a_completion.v1")
    require(completion["frame_count"] == FRAME_COUNT and completion["query_count"] == QUERY_COUNT and completion["faro_reads"] == completion["truth_reads"] == 0 and completion["clear_output_allowed"] is False, "R7_PHASE_B_PHASE_A_COMPLETION_INVALID", "Phase-A completion is not admitted")
    sources: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for frame in frames:
        lineage = _read_gzip_json(root / phase_a._lineage_relative(frame))
        source = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        require(source["physical_frame_id"] == frame.physical_frame_id, "R7_PHASE_B_SOURCE_IDENTITY_DRIFT", "Phase-A source identity drift")
        receipt = phase_a._validate_seal(_read_json(root / phase_a._source_receipt_relative(frame)), "blindassist.taro.o1r.r7_fresh_source_frame_receipt.v1")
        require(source["source_frame_receipt_sha256"] == receipt["content_sha256"], "R7_PHASE_B_SOURCE_LINEAGE_DRIFT", "Phase-A source receipt lineage drift")
        sources.append(source)
        source_receipts.append(receipt)
    require(adapter.canonical_sha256([row["content_sha256"] for row in sources]) == completion["source_frame_hash_sequence_sha256"], "R7_PHASE_B_SOURCE_SEQUENCE_DRIFT", "Phase-A source hash sequence drift")
    return completion, sources, source_receipts


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = _read_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R7_PHASE_B_LOCK_IDENTITY", "Phase-B lock identity drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv, "R7_PHASE_B_ARGV_DRIFT", "Phase-B argv drift")
    require(lock.get("phase_a_root") == PHASE_A_ROOT and lock.get("inventory_path") == INVENTORY_PATH and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R7_PHASE_B_ROOT_DRIFT", "Phase-B root policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R7_PHASE_B_BINDINGS", "Phase-B binding count drift")
    seen = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in seen and EXPECTED_BINDINGS.get(role) == relative, "R7_PHASE_B_BINDING_ROW", "Phase-B binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R7_PHASE_B_BINDING_HASH", f"Phase-B binding drift: {relative}")
        seen.add(role)
    require(validate_fresh_confirmation_protocol.validate(_repo_path(EXPECTED_BINDINGS["R7_FRESH_PROTOCOL"]))["passed"], "R7_PHASE_B_PROTOCOL_INVALID", "fresh confirmation protocol invalid")
    phase_a_result = _read_json(_repo_path(EXPECTED_BINDINGS["R7_PHASE_A_RESULT"]))
    require(phase_a_result.get("execution_valid") is True and phase_a_result.get("passed") is True and phase_a_result.get("terminal") == "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_A_SEALED_PASS_R1" and phase_a_result.get("faro_reads") == 0, "R7_PHASE_B_PHASE_A_NOT_ADMITTED", "Phase-A result not admitted")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R7_PHASE_B_AUTHORITY_DRIFT", "Phase-B authority drift")
    require(lock.get("resource_budget") == {"maximum_wall_seconds": 14400, "maximum_peak_rss_bytes": 17179869184, "maximum_evidence_bytes": 536870912}, "R7_PHASE_B_BUDGET_DRIFT", "Phase-B resource budget drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R7_PHASE_B_ROOT_COLLISION", "Phase-B output root exists")
    lock["_lock_path"] = lock_path
    return lock


def _summarize(sources: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    parent_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    parent_occ: dict[str, list[tuple[str, str]]] = defaultdict(list)
    occupied_tp = occupied_fp = occupied_fn = predicted_on_unknown = clear_outputs = 0
    for source, label_record in zip(sources, labels, strict=True):
        parent = str(source["parent_id"])
        for feature, label in zip(source["query_features"], label_record["query_labels"], strict=True):
            prediction = phase_a._positive_state(feature)
            truth = str(label["state"])
            label_counts[truth] += 1
            prediction_counts[prediction] += 1
            parent_label_counts[parent][truth] += 1
            clear_outputs += prediction == "CLEAR_OBSERVED"
            if truth == "OCCUPIED_OBSERVED":
                parent_occ[parent].append((str(feature["r6_state"]), prediction))
                occupied_tp += prediction == "OCCUPIED_OBSERVED"
                occupied_fn += prediction != "OCCUPIED_OBSERVED"
            elif truth == "CLEAR_OBSERVED":
                occupied_fp += prediction == "OCCUPIED_OBSERVED"
            elif truth == "UNKNOWN":
                predicted_on_unknown += prediction == "OCCUPIED_OBSERVED"
    definite_occupied = int(label_counts["OCCUPIED_OBSERVED"])
    definite_clear = int(label_counts["CLEAR_OBSERVED"])
    precision_denominator = occupied_tp + occupied_fp
    precision = occupied_tp / precision_denominator if precision_denominator else 0.0
    recall = occupied_tp / definite_occupied if definite_occupied else 0.0
    wilson = r7_canary._wilson_lower(occupied_tp, precision_denominator)
    parent_improvements = []
    per_parent = {}
    for parent, _, _ in cohort.EXPECTED_ROSTER:
        occupied_rows = parent_occ[parent]
        denominator = len(occupied_rows)
        baseline = sum(base == "OCCUPIED_OBSERVED" for base, _ in occupied_rows) / denominator if denominator else None
        candidate = sum(pred == "OCCUPIED_OBSERVED" for _, pred in occupied_rows) / denominator if denominator else None
        improvement = None if denominator == 0 else float(candidate - baseline)
        if improvement is not None:
            parent_improvements.append(improvement)
        per_parent[parent] = {
            "label_state_counts": {state: int(parent_label_counts[parent][state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
            "definite_occupied_query_count": denominator,
            "baseline_definite_occupancy_coverage": baseline,
            "candidate_definite_occupancy_coverage": candidate,
            "coverage_increase_absolute": improvement,
        }
    macro_increase = sum(parent_improvements) / len(parent_improvements) if parent_improvements else 0.0
    evaluability = {
        "evaluable_parent_count": sum(sum(parent_label_counts[parent].values()) - parent_label_counts[parent]["UNKNOWN"] > 0 for parent, _, _ in cohort.EXPECTED_ROSTER),
        "parents_with_definite_occupied_label": sum(parent_label_counts[parent]["OCCUPIED_OBSERVED"] > 0 for parent, _, _ in cohort.EXPECTED_ROSTER),
        "parents_with_definite_clear_label": sum(parent_label_counts[parent]["CLEAR_OBSERVED"] > 0 for parent, _, _ in cohort.EXPECTED_ROSTER),
        "definite_occupied_query_count": definite_occupied,
        "definite_clear_query_count": definite_clear,
    }
    evaluable = evaluability["evaluable_parent_count"] >= 8 and evaluability["parents_with_definite_occupied_label"] >= 6 and evaluability["parents_with_definite_clear_label"] >= 4 and definite_occupied >= 100 and definite_clear >= 50
    gates = {
        "occupied_precision_on_definite_labels": {"value": precision, "minimum": 0.9, "passed": precision >= 0.9},
        "one_sided_95_wilson_occupied_precision_lower_bound": {"value": wilson, "minimum": 0.8, "passed": wilson >= 0.8},
        "occupied_recall": {"value": recall, "minimum": 0.9, "passed": recall >= 0.9},
        "parent_macro_definite_occupancy_coverage_increase_absolute": {"value": macro_increase, "minimum": 0.05, "parent_denominator": len(parent_improvements), "passed": macro_increase >= 0.05},
        "maximum_clear_outputs": {"value": clear_outputs, "maximum": 0, "passed": clear_outputs == 0},
    }
    all_gates = all(row["passed"] for row in gates.values())
    terminal = NOT_EVALUABLE_TERMINAL if not evaluable else PASS_TERMINAL if all_gates else FAIL_TERMINAL
    return {
        "terminal": terminal,
        "passed": bool(evaluable and all_gates),
        "scientifically_evaluable": evaluable,
        "evaluability": evaluability,
        "label_state_counts": {state: int(label_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
        "prediction_state_counts": {state: int(prediction_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
        "occupied_true_positive": int(occupied_tp),
        "occupied_false_positive_against_definite_clear": int(occupied_fp),
        "occupied_false_negative": int(occupied_fn),
        "occupied_predictions_on_truth_unknown": int(predicted_on_unknown),
        "unknown_is_negative": False,
        "gates": gates,
        "all_confirmation_gates_passed": bool(all_gates),
        "per_parent": per_parent,
    }


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> None:
    try:
        writer.write_json("failure.json", {"schema": "blindassist.taro.o1r.r7_fresh_phase_b_failure.v1", "terminal": INVALID_TERMINAL, "execution_valid": False, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "one_shot_consumed": True})
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r7_fresh_phase_b_manifest.v1", "terminal": INVALID_TERMINAL, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
    except Exception:
        pass


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    budget = lock["resource_budget"]
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(budget["maximum_evidence_bytes"]))
    started = time.monotonic()
    process = psutil.Process()

    def guard() -> None:
        require(time.monotonic() - started <= budget["maximum_wall_seconds"], "R7_PHASE_B_TIMEOUT", "Phase-B wall budget exceeded")
        require(process.memory_info().rss <= budget["maximum_peak_rss_bytes"], "R7_PHASE_B_RSS_EXCEEDED", "Phase-B RSS budget exceeded")

    writer.activate({"schema": "blindassist.taro.o1r.r7_fresh_phase_b_execution_receipt.v1", "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "expected_frame_count": FRAME_COUNT, "expected_query_count": QUERY_COUNT, "phase_a_reloaded_before_faro": True, "source_reselection": False, "threshold_reselection": False, "training_steps": 0, "network_requests": 0, "one_shot_consumed_on_root_creation": True})
    try:
        frames = phase_a._load_frames(_repo_path(INVENTORY_PATH))
        phase_root = _repo_path(PHASE_A_ROOT)
        _verify_phase_a_manifest(phase_root)
        completion, sources, source_receipts = _load_phase_a_records(frames, phase_root)
        require(completion["content_sha256"] == _read_json(phase_root / "result.json")["phase_a_completion_sha256"], "R7_PHASE_B_COMPLETION_RESULT_DRIFT", "Phase-A result/completion lineage drift")
        faro_reads: Counter[str] = Counter()
        labels: list[dict[str, Any]] = []
        label_hashes: list[str] = []

        def observed(role: str, _: str) -> None:
            require(role == "highres_depth", "R7_PHASE_B_PAYLOAD_FIREWALL", "Phase-B attempted non-FARO payload read", role=role)
            faro_reads[role] += 1

        completed = 0
        for _, parent_frames_iter in groupby(list(zip(frames, sources, source_receipts, strict=True)), key=lambda row: (row[0].parent_id, row[0].video_id)):
            parent_rows = list(parent_frames_iter)
            with zipfile.ZipFile(parent_rows[0][0].upsampling_archive) as up_bundle:
                for frame, source, receipt in parent_rows:
                    faro_payload, _binding = r6io._read_member(up_bundle, frame.members["highres_depth"], observer=observed)
                    faro = materializer._decode_png(faro_payload, "highres_depth")
                    label = r7_canary.build_label_frame_record(source, faro, receipt["intrinsics_highres"]["matrix_3x3"], receipt["gravity_up_camera_xyz"])
                    writer.write_json_gzip(_label_relative(frame), label)
                    labels.append(label)
                    label_hashes.append(label["content_sha256"])
                    completed += 1
                    guard()
                    if completed % 10 == 0 or completed == FRAME_COUNT:
                        print(json.dumps({"phase": "R7_FRESH_FARO_LABEL", "completed": completed, "total": FRAME_COUNT, "physical_frame_id": frame.physical_frame_id}, sort_keys=True), flush=True)
        require(faro_reads == Counter({"highres_depth": FRAME_COUNT}), "R7_PHASE_B_FARO_READ_COUNT_DRIFT", "Phase-B FARO read count drift", reads=dict(faro_reads))
        summary = _summarize(sources, labels)
        label_completion = _seal({"schema": "blindassist.taro.o1r.r7_fresh_phase_b_label_completion.v1", "frame_count": FRAME_COUNT, "query_count": QUERY_COUNT, "label_hash_sequence_sha256": adapter.canonical_sha256(label_hashes), "phase_a_completion_sha256": completion["content_sha256"], "faro_payload_reads": dict(faro_reads), "source_reselection": False, "threshold_reselection": False, "unknown_is_negative": False})
        writer.write_json("label-completion.json", label_completion)
        result = {"schema": "blindassist.taro.o1r.r7_fresh_confirmation_result.v1", **summary, "execution_valid": True, "frame_count": FRAME_COUNT, "query_count": QUERY_COUNT, "phase_a_completion_sha256": completion["content_sha256"], "label_completion_sha256": label_completion["content_sha256"], "faro_frame_count": FRAME_COUNT, "source_reselection": False, "threshold_reselection": False, "training_steps": 0, "network_requests": 0, "elapsed_seconds": round(time.monotonic() - started, 6), "one_shot_consumed": True, "promotion_scope": "RESEARCH_ROUTE_POSITIVE_OCCUPANCY_FACTOR_ONLY" if summary["passed"] else None, "clear_branch_promotion": False, "claim_ceiling": "Fresh ARKitScenes research confirmation of the positive-occupancy factor only; no clear, deployment, device, product, or safety claim."}
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r7_fresh_phase_b_manifest.v1", "terminal": result["terminal"], "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
        return result
    except Exception as error:
        _write_failure(writer, error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"terminal": INVALID_TERMINAL, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "scientifically_evaluable": result["scientifically_evaluable"], "label_state_counts": result["label_state_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
