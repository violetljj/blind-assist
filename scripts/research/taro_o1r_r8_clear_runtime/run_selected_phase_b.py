#!/usr/bin/env python3
"""Open FARO only for the sealed R8 top eight and evaluate unchanged R7 gates."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
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
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as shared_phase_a
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_phase_a
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_phase_a_recovery
from scripts.research.taro_o1r_r8_clear_runtime import run_top8_selection


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r8_clear_selected_phase_b_execution_lock.v1"
LOCK_ID = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_SELECTED_TOP8_PHASE_B_FARO_ONE_SHOT_EXECUTION_LOCK"
PHASE_A_ROOT = run_pool_phase_a_recovery.OUTPUT_ROOT
SELECTION_ROOT = run_top8_selection.OUTPUT_ROOT
INVENTORY_PATH = run_pool_phase_a.INVENTORY_PATH
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r8-clear-selected-phase-b-r0"
PASS_TERMINAL = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_FRESH_CONFIRMATION_PASS"
FAIL_TERMINAL = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_FRESH_CONFIRMATION_FAIL"
NOT_EVALUABLE_TERMINAL = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_NOT_EVALUABLE_DUAL_CLASS_COVERAGE"
INVALID_TERMINAL = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_EXECUTION_INVALID"
SELECTED_PARENT_COUNT = 8

EXPECTED_BINDINGS = {
    "R8_PROTOCOL": "docs/research/taro/TARO_O1R_R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_COHORT_ENRICHMENT_PROTOCOL_LOCK_2026-08-12.json",
    "R8_INVENTORY_PLAN": INVENTORY_PATH,
    "R8_PHASE_A_COMPLETION": f"{PHASE_A_ROOT}/phase-a-completion.json",
    "R8_PHASE_A_RESULT": f"{PHASE_A_ROOT}/result.json",
    "R8_PHASE_A_MANIFEST": f"{PHASE_A_ROOT}/manifest.json",
    "R8_SELECTION": f"{SELECTION_ROOT}/selection.json",
    "R8_SELECTION_RESULT": f"{SELECTION_ROOT}/result.json",
    "R8_SELECTION_MANIFEST": f"{SELECTION_ROOT}/manifest.json",
    "R8_PHASE_A_RECOVERY_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_pool_phase_a_recovery.py",
    "R8_SELECTION_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_top8_selection.py",
    "R7_CANARY_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R8_PHASE_B_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_selected_phase_b.py",
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "授权",
    "scope": "After the exact R8 top eight are irreversibly sealed from source-only evidence, read highres_depth FARO for exactly those 133 selected frames, construct fixed labels, and evaluate the unchanged dual-class and positive-occupancy gates; no source reselection, threshold fit, or training.",
}
EXPECTED_GATES = {"minimum_evaluable_parents": 8, "minimum_parents_with_definite_occupied": 6, "minimum_parents_with_definite_clear": 4, "minimum_definite_occupied_queries": 100, "minimum_definite_clear_queries": 50, "minimum_occupied_precision": 0.9, "minimum_one_sided_95_wilson_precision_lower_bound": 0.8, "minimum_occupied_recall": 0.9, "minimum_parent_macro_occupancy_coverage_increase_absolute": 0.05, "maximum_clear_outputs": 0, "unknown_is_negative": False}


class SelectedPhaseBError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise SelectedPhaseBError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R8_PHASE_B_SEAL_COLLISION", "Phase-B caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _label_relative(frame: r6io.R6FrameRef) -> str:
    return f"labels/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def load_selected_rows() -> tuple[dict[str, Any], list[r6io.R6FrameRef], list[dict[str, Any]], list[dict[str, Any]]]:
    selection = run_top8_selection.validate_selection(_read_json(_repo_path(f"{SELECTION_ROOT}/selection.json")))
    selected = {(row["parent_id"], row["video_id"]) for row in selection["selected_parents"]}
    require(len(selected) == SELECTED_PARENT_COUNT, "R8_PHASE_B_SELECTION_COUNT", "selected identity count drift")
    all_frames = run_pool_phase_a.load_frames(_repo_path(INVENTORY_PATH))
    completion = run_pool_phase_a_recovery.validate_completion(_read_json(_repo_path(f"{PHASE_A_ROOT}/phase-a-completion.json")))
    frames: list[r6io.R6FrameRef] = []
    sources: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for frame in all_frames:
        if (frame.parent_id, frame.video_id) not in selected:
            continue
        lineage = _read_gzip_json(_repo_path(PHASE_A_ROOT) / shared_phase_a._lineage_relative(frame))
        source = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        receipt = shared_phase_a._validate_seal(_read_json(_repo_path(PHASE_A_ROOT) / shared_phase_a._source_receipt_relative(frame)), "blindassist.taro.o1r.r7_fresh_source_frame_receipt.v1")
        require(source["physical_frame_id"] == frame.physical_frame_id and source["source_frame_receipt_sha256"] == receipt["content_sha256"], "R8_PHASE_B_SOURCE_LINEAGE", "selected source lineage drift")
        frames.append(frame)
        sources.append(source)
        receipts.append(receipt)
    require({(frame.parent_id, frame.video_id) for frame in frames} == selected and len(frames) == sum(row["frame_count"] for row in selection["selected_parents"]), "R8_PHASE_B_SELECTED_FRAME_COUNT", "selected frame set drift")
    require(selection["phase_a_completion_sha256"] == completion["content_sha256"], "R8_PHASE_B_SELECTION_LINEAGE", "selection does not bind admitted Phase A")
    return selection, frames, sources, receipts


def summarize(selected_identities: Sequence[tuple[str, str]], sources: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    parent_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    parent_occ: dict[str, list[tuple[str, str]]] = defaultdict(list)
    occupied_tp = occupied_fp = occupied_fn = predicted_on_unknown = clear_outputs = 0
    for source, label_record in zip(sources, labels, strict=True):
        parent = str(source["parent_id"])
        for feature, label in zip(source["query_features"], label_record["query_labels"], strict=True):
            prediction = shared_phase_a._positive_state(feature)
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
    for parent, _video in selected_identities:
        occupied_rows = parent_occ[parent]
        denominator = len(occupied_rows)
        baseline = sum(base == "OCCUPIED_OBSERVED" for base, _ in occupied_rows) / denominator if denominator else None
        candidate = sum(pred == "OCCUPIED_OBSERVED" for _, pred in occupied_rows) / denominator if denominator else None
        improvement = None if denominator == 0 else float(candidate - baseline)
        if improvement is not None:
            parent_improvements.append(improvement)
        per_parent[parent] = {"label_state_counts": {state: int(parent_label_counts[parent][state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}, "definite_occupied_query_count": denominator, "baseline_definite_occupancy_coverage": baseline, "candidate_definite_occupancy_coverage": candidate, "coverage_increase_absolute": improvement}
    macro_increase = sum(parent_improvements) / len(parent_improvements) if parent_improvements else 0.0
    evaluability = {"evaluable_parent_count": sum(sum(parent_label_counts[parent].values()) - parent_label_counts[parent]["UNKNOWN"] > 0 for parent, _ in selected_identities), "parents_with_definite_occupied_label": sum(parent_label_counts[parent]["OCCUPIED_OBSERVED"] > 0 for parent, _ in selected_identities), "parents_with_definite_clear_label": sum(parent_label_counts[parent]["CLEAR_OBSERVED"] > 0 for parent, _ in selected_identities), "definite_occupied_query_count": definite_occupied, "definite_clear_query_count": definite_clear}
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
    return {"terminal": terminal, "passed": bool(evaluable and all_gates), "scientifically_evaluable": evaluable, "evaluability": evaluability, "label_state_counts": {state: int(label_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}, "prediction_state_counts": {state: int(prediction_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")}, "occupied_true_positive": int(occupied_tp), "occupied_false_positive_against_definite_clear": int(occupied_fp), "occupied_false_negative": int(occupied_fn), "occupied_predictions_on_truth_unknown": int(predicted_on_unknown), "unknown_is_negative": False, "gates": gates, "all_confirmation_gates_passed": bool(all_gates), "per_parent": per_parent}


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = _read_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R8_PHASE_B_LOCK_IDENTITY", "Phase-B lock identity drift")
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R8_PHASE_B_USER_AUTHORITY", "Phase-B user authority drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv and lock.get("phase_a_root") == PHASE_A_ROOT and lock.get("selection_root") == SELECTION_ROOT and lock.get("inventory_path") == INVENTORY_PATH and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R8_PHASE_B_LOCK_POLICY", "Phase-B root/argv policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R8_PHASE_B_BINDINGS", "Phase-B binding count drift")
    seen = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in seen and EXPECTED_BINDINGS.get(role) == relative, "R8_PHASE_B_BINDING_ROW", "Phase-B binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R8_PHASE_B_BINDING_HASH", f"Phase-B binding drift: {relative}")
        seen.add(role)
    selection = run_top8_selection.validate_selection(_read_json(_repo_path(EXPECTED_BINDINGS["R8_SELECTION"])))
    selection_result = _read_json(_repo_path(EXPECTED_BINDINGS["R8_SELECTION_RESULT"]))
    require(selection_result.get("execution_valid") is True and selection_result.get("passed") is True and selection_result.get("terminal") == run_top8_selection.PASS_TERMINAL and selection_result.get("selection_sha256") == selection["content_sha256"] and selection_result.get("faro_reads") == selection_result.get("truth_reads") == selection_result.get("label_reads") == selection_result.get("outcome_reads") == 0, "R8_PHASE_B_SELECTION_RESULT", "selection result not admitted")
    selection_manifest = _read_json(_repo_path(EXPECTED_BINDINGS["R8_SELECTION_MANIFEST"]))
    require(selection_manifest.get("schema") == "blindassist.taro.o1r.r8_clear_top8_selection_manifest.v1" and selection_manifest.get("terminal") == run_top8_selection.PASS_TERMINAL and selection_manifest.get("file_count_before_manifest") == len(selection_manifest.get("files", {})) == 3, "R8_PHASE_B_SELECTION_MANIFEST", "selection manifest not admitted")
    for relative, receipt in selection_manifest["files"].items():
        target = _repo_path(SELECTION_ROOT) / relative
        require(target.is_file() and target.stat().st_size == receipt.get("bytes") and materializer.sha256_file(target) == receipt.get("sha256") and receipt.get("path") == relative, "R8_PHASE_B_SELECTION_FILE", f"selection artifact drift: {relative}")
    selected_frames = sum(row["frame_count"] for row in selection["selected_parents"])
    require(lock.get("execution_authority") == {"phase_a_reload": True, "sealed_top8_reload": True, "faro_payload_read": True, "faro_frame_count": selected_frames, "truth_label_construction": True, "fixed_gate_evaluation": True, "source_reselection": False, "threshold_fit": False, "training": False, "network": False, "device": False, "product": False, "safety": False}, "R8_PHASE_B_AUTHORITY", "Phase-B authority drift")
    require(lock.get("selected_cohort") == {"parent_count": SELECTED_PARENT_COUNT, "physical_frame_count": selected_frames, "query_count": selected_frames * 9, "selected_parent_identities": [[row["parent_id"], row["video_id"]] for row in selection["selected_parents"]], "selection_sha256": selection["content_sha256"]}, "R8_PHASE_B_SELECTED_COHORT", "selected cohort drift")
    require(lock.get("unchanged_gates") == EXPECTED_GATES, "R8_PHASE_B_GATE_DRIFT", "Phase-B gate drift")
    require(lock.get("phase_firewall") == {"selection_sha256": selection["content_sha256"], "source_reselection": False, "threshold_reselection": False, "only_payload_role_read": "highres_depth", "read_unselected_parent_faro": False}, "R8_PHASE_B_FIREWALL", "Phase-B firewall drift")
    require(lock.get("resource_budget") == {"maximum_wall_seconds": 14400, "maximum_peak_rss_bytes": 17179869184, "maximum_evidence_bytes": 536870912}, "R8_PHASE_B_BUDGET", "Phase-B budget drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R8_PHASE_B_ROOT_COLLISION", "Phase-B output root exists")
    lock["_lock_path"] = lock_path
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    budget = lock["resource_budget"]
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(budget["maximum_evidence_bytes"]))
    started = time.monotonic()
    process = psutil.Process()
    writer.activate({"schema": "blindassist.taro.o1r.r8_clear_selected_phase_b_execution_receipt.v1", "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "selected_parent_count": SELECTED_PARENT_COUNT, "expected_frame_count": lock["selected_cohort"]["physical_frame_count"], "phase_a_and_top8_reloaded_before_faro": True, "source_reselection": False, "threshold_reselection": False, "training_steps": 0, "network_requests": 0, "one_shot_consumed_on_root_creation": True})
    try:
        selection, frames, sources, receipts = load_selected_rows()
        faro_reads: Counter[str] = Counter()
        labels: list[dict[str, Any]] = []
        label_hashes: list[str] = []

        def observed(role: str, _: str) -> None:
            require(role == "highres_depth", "R8_PHASE_B_PAYLOAD_FIREWALL", "Phase-B attempted non-FARO payload read")
            faro_reads[role] += 1

        for index, (_key, parent_rows_iter) in enumerate(groupby(list(zip(frames, sources, receipts, strict=True)), key=lambda row: (row[0].parent_id, row[0].video_id))):
            parent_rows = list(parent_rows_iter)
            with zipfile.ZipFile(parent_rows[0][0].upsampling_archive) as bundle:
                for frame, source, receipt in parent_rows:
                    faro_payload, _binding = r6io._read_member(bundle, frame.members["highres_depth"], observer=observed)
                    faro = materializer._decode_png(faro_payload, "highres_depth")
                    label = r7_canary.build_label_frame_record(source, faro, receipt["intrinsics_highres"]["matrix_3x3"], receipt["gravity_up_camera_xyz"])
                    writer.write_json_gzip(_label_relative(frame), label)
                    labels.append(label)
                    label_hashes.append(label["content_sha256"])
                    require(time.monotonic() - started <= budget["maximum_wall_seconds"] and process.memory_info().rss <= budget["maximum_peak_rss_bytes"], "R8_PHASE_B_RESOURCE", "Phase-B resource budget exceeded")
                    if len(labels) % 10 == 0 or len(labels) == len(frames):
                        print(json.dumps({"phase": "R8_SELECTED_FARO_LABEL", "completed": len(labels), "total": len(frames), "physical_frame_id": frame.physical_frame_id}, sort_keys=True), flush=True)
        require(faro_reads == Counter({"highres_depth": len(frames)}), "R8_PHASE_B_FARO_COUNT", "Phase-B FARO read count drift")
        identities = [(row["parent_id"], row["video_id"]) for row in selection["selected_parents"]]
        summary = summarize(identities, sources, labels)
        completion = _seal({"schema": "blindassist.taro.o1r.r8_clear_selected_phase_b_label_completion.v1", "frame_count": len(frames), "query_count": len(frames) * 9, "label_hash_sequence_sha256": adapter.canonical_sha256(label_hashes), "phase_a_completion_sha256": selection["phase_a_completion_sha256"], "selection_sha256": selection["content_sha256"], "faro_payload_reads": dict(faro_reads), "source_reselection": False, "threshold_reselection": False, "unknown_is_negative": False})
        writer.write_json("label-completion.json", completion)
        result = {"schema": "blindassist.taro.o1r.r8_clear_negative_control_confirmation_result.v1", **summary, "execution_valid": True, "selected_parent_count": SELECTED_PARENT_COUNT, "frame_count": len(frames), "query_count": len(frames) * 9, "selection_sha256": selection["content_sha256"], "phase_a_completion_sha256": selection["phase_a_completion_sha256"], "label_completion_sha256": completion["content_sha256"], "faro_frame_count": len(frames), "source_reselection": False, "threshold_reselection": False, "training_steps": 0, "network_requests": 0, "elapsed_seconds": round(time.monotonic() - started, 6), "one_shot_consumed": True, "promotion_scope": "RESEARCH_ROUTE_POSITIVE_OCCUPANCY_FACTOR_ONLY" if summary["passed"] else None, "clear_branch_promotion": False, "claim_ceiling": "Fresh ARKitScenes research confirmation of the fail-safe positive-occupancy factor and definite-clear negative control only; no clear-output, deployment, device, product, or safety claim."}
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r8_clear_selected_phase_b_manifest.v1", "terminal": result["terminal"], "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
        return result
    except Exception as error:
        try:
            writer.write_json("failure.json", {"schema": "blindassist.taro.o1r.r8_clear_selected_phase_b_failure.v1", "terminal": INVALID_TERMINAL, "execution_valid": False, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "one_shot_consumed": True})
        except Exception:
            pass
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
