#!/usr/bin/env python3
"""Materialize remaining R8 FARO development labels and fit the R9 selector."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import json
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
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
from scripts.research.taro_o1r_r8_clear_runtime import run_ray_space_canary as shared_verify
from scripts.research.taro_o1r_r8_clear_runtime import run_selected_phase_b as phase_b
from scripts.research.taro_o1r_r8_clear_runtime import run_top8_selection
from scripts.research.taro_o1r_r9_clear_runtime import clear_enrichment_fit as fit


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r9_clear_enrichment_development_truth_execution_lock.v1"
LOCK_ID = "TARO_O1R_R9_CLEAR_ENRICHMENT_REMAINING_R8_POOL_DEVELOPMENT_TRUTH_ONE_SHOT_EXECUTION_LOCK"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R9_CLEAR_ENRICHMENT_REMAINING_R8_POOL_DEVELOPMENT_TRUTH_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r9-clear-enrichment-development-r0"
PASS_TERMINAL = "TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_SELECTOR_FROZEN_PASS"
FAIL_TERMINAL = "TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_CLEAR_COVERAGE_INSUFFICIENT"
INVALID_TERMINAL = "TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_EXECUTION_INVALID"
DEVELOPMENT_PARENT_COUNT = 16
DEVELOPMENT_FRAME_COUNT = 269
DEVELOPMENT_QUERY_COUNT = 2421
MAX_WORKERS = 4
EXPECTED_DEVELOPMENT_ROSTER = (
    ("421264", "42444885", 17),
    ("422155", "42445684", 12),
    ("423611", "42898089", 15),
    ("423770", "42898065", 21),
    ("435353", "42899542", 15),
    ("435729", "42899650", 25),
    ("435730", "42899817", 16),
    ("467304", "47333988", 22),
    ("467344", "47333964", 11),
    ("467345", "47333963", 14),
    ("468296", "47431071", 6),
    ("469837", "47334046", 15),
    ("470945", "47115624", 16),
    ("482095", "47895719", 13),
    ("482115", "47895520", 14),
    ("482761", "47670197", 37),
)
EXPECTED_BINDINGS = {
    "R8_PHASE_A_COMPLETION": f"{run_pool_phase_a_recovery.OUTPUT_ROOT}/phase-a-completion.json",
    "R8_PHASE_A_MANIFEST": f"{run_pool_phase_a_recovery.OUTPUT_ROOT}/manifest.json",
    "R8_SELECTION": f"{run_top8_selection.OUTPUT_ROOT}/selection.json",
    "R8_FINAL_RESULT": "docs/research/taro/TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_AND_TRUTH_INTERFACE_RESULT_2026-08-12.json",
    "R8_INVENTORY_PLAN": run_pool_phase_a.INVENTORY_PATH,
    "R7_DENSE_LABEL_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R9_SELECTOR_RUNTIME": "scripts/research/taro_o1r_r9_clear_runtime/clear_enrichment_fit.py",
    "R9_SELECTOR_TEST": "scripts/research/taro_o1r_r9_clear_runtime/test_clear_enrichment_fit.py",
    "R9_DEVELOPMENT_RUNNER": "scripts/research/taro_o1r_r9_clear_runtime/run_development_truth.py",
    "R9_DEVELOPMENT_RUNNER_TEST": "scripts/research/taro_o1r_r9_clear_runtime/test_run_development_truth.py",
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "我授权 TARO O0R 使用锁定的 24 个 ARKitScenes Training 视频及每个视频的 upsampling.zip、lowres_wide_intrinsics.zip、lowres_wide.traj，用于 HEAD 预检和 source/truth-only WILD_LAB 物化与校验",
    "scope": "Open highres_depth FARO for the remaining 16 already-downloaded R8 parents and 269 frames as development-only truth, then fit and seal a source-only clear-enrichment rule; all 24 R8 parents remain consumed and no R8 outcome has confirmation authority.",
}
EXPECTED_BUDGET = {"maximum_wall_seconds": 7200, "maximum_peak_rss_bytes": 8589934592, "maximum_evidence_bytes": 536870912, "maximum_workers": MAX_WORKERS}


class DevelopmentTruthError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise DevelopmentTruthError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R9_DEV_SEAL_COLLISION", "development caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _label_relative(frame: r6io.R6FrameRef) -> str:
    return f"development-labels/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def load_development_rows() -> tuple[list[r6io.R6FrameRef], list[dict[str, Any]], list[dict[str, Any]]]:
    selection = run_top8_selection.validate_selection(_read_json(_repo_path(f"{run_top8_selection.OUTPUT_ROOT}/selection.json")))
    selected = {(row["parent_id"], row["video_id"]) for row in selection["selected_parents"]}
    frames = []
    sources = []
    receipts = []
    for frame in run_pool_phase_a.load_frames(_repo_path(run_pool_phase_a.INVENTORY_PATH)):
        if (frame.parent_id, frame.video_id) in selected:
            continue
        lineage = _read_gzip_json(_repo_path(run_pool_phase_a_recovery.OUTPUT_ROOT) / shared_phase_a._lineage_relative(frame))
        source = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        receipt = shared_phase_a._validate_seal(_read_json(_repo_path(run_pool_phase_a_recovery.OUTPUT_ROOT) / shared_phase_a._source_receipt_relative(frame)), "blindassist.taro.o1r.r7_fresh_source_frame_receipt.v1")
        require(source["physical_frame_id"] == frame.physical_frame_id and source["source_frame_receipt_sha256"] == receipt["content_sha256"], "R9_DEV_SOURCE_LINEAGE", "development source lineage drift")
        frames.append(frame)
        sources.append(source)
        receipts.append(receipt)
    counts = Counter((frame.parent_id, frame.video_id) for frame in frames)
    roster = tuple((parent, video, counts[(parent, video)]) for parent, video in sorted(counts))
    require(roster == EXPECTED_DEVELOPMENT_ROSTER and len(frames) == DEVELOPMENT_FRAME_COUNT, "R9_DEV_ROSTER", "remaining development roster drift")
    require(not selected.intersection(counts), "R9_DEV_SELECTED_LEAK", "selected R8 parent entered development remainder")
    return frames, sources, receipts


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R9_DEV_LOCK_PATH", "development lock path drift")
    lock = _read_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R9_DEV_LOCK_IDENTITY", "development lock identity drift")
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R9_DEV_USER_AUTHORITY", "development user authority drift")
    expected_argv = ["scripts/research/taro_o1r_r9_clear_runtime/run_development_truth.py", "--execution-lock", LOCK_RELATIVE]
    require(lock.get("argv") == expected_argv and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R9_DEV_LOCK_POLICY", "development root/argv policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R9_DEV_BINDINGS", "development binding count drift")
    seen = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in seen and EXPECTED_BINDINGS.get(role) == relative, "R9_DEV_BINDING_ROW", "development binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R9_DEV_BINDING_HASH", f"development binding drift: {relative}")
        seen.add(role)
    phase_a_manifest = _read_json(_repo_path(EXPECTED_BINDINGS["R8_PHASE_A_MANIFEST"]))
    shared_verify._verify_manifest(run_pool_phase_a_recovery.OUTPUT_ROOT, phase_a_manifest, "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_manifest.v1", "TARO_O1R_R8_CLEAR_POOL_PHASE_A_SOURCE_ONLY_RECOVERY_SEALED_PASS_R1")
    r8_result = _read_json(_repo_path(EXPECTED_BINDINGS["R8_FINAL_RESULT"]))
    require(r8_result.get("terminal") == "TARO_O1R_R8_CLOSED_NOT_EVALUABLE_CLEAR_COVERAGE_TRUTH_INTERFACE_EXHAUSTED" and r8_result.get("r8_consumed") is True and r8_result.get("unique_successor", {}).get("id") == "TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_ON_REMAINING_AUTHORIZED_R8_POOL", "R9_DEV_R8_TERMINAL", "R8 terminal does not admit development successor")
    frames, _sources, _receipts = load_development_rows()
    require(lock.get("development_roster") == {"parent_count": DEVELOPMENT_PARENT_COUNT, "physical_frame_count": DEVELOPMENT_FRAME_COUNT, "query_count": DEVELOPMENT_QUERY_COUNT, "parents": [list(row) for row in EXPECTED_DEVELOPMENT_ROSTER]}, "R9_DEV_LOCK_ROSTER", "development lock roster drift")
    require(lock.get("execution_authority") == {"phase_a_source_reload": True, "remaining_parent_faro_read": True, "faro_frame_count": DEVELOPMENT_FRAME_COUNT, "dense_label_construction": True, "source_only_selector_fit": True, "read_selected_r8_parent_faro": False, "fresh_confirmation": False, "training": False, "network": False, "device": False, "deployment": False, "product": False, "safety": False}, "R9_DEV_AUTHORITY", "development authority drift")
    require(lock.get("selector_search") == {"selector_id": fit.SELECTOR_ID, "candidate_rule_count": len(fit.candidate_rules()), "selected_parent_count": fit.SELECTED_PARENT_COUNT, "minimum_nonzero_selected_parents": fit.MINIMUM_NONZERO_SELECTED_PARENTS, "minimum_clear_queries": fit.MINIMUM_CLEAR_QUERIES, "minimum_clear_parents": fit.MINIMUM_CLEAR_PARENTS, "unknown_is_negative": False}, "R9_DEV_SEARCH", "development selector search drift")
    require(lock.get("resource_budget") == EXPECTED_BUDGET and lock.get("claim_ceiling") == "Development-only FARO labels and a frozen source-only selector from consumed R8 parents; no fresh confirmation, effectiveness promotion, deployment, product, or safety claim.", "R9_DEV_BUDGET_CLAIM", "development budget/claim drift")
    require(len(frames) == DEVELOPMENT_FRAME_COUNT and not _repo_path(OUTPUT_ROOT).exists(), "R9_DEV_ROOT_COLLISION", "development output root exists or frame count drift")
    lock["_lock_path"] = lock_path
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), *sys.argv[1:]]
    require(actual_argv == lock["argv"], "R9_DEV_ACTUAL_ARGV", "development run must use the unique locked argv")
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(lock["resource_budget"]["maximum_evidence_bytes"]))
    started = time.monotonic()
    process = psutil.Process()
    writer.activate({"schema": "blindassist.taro.o1r.r9_clear_enrichment_development_execution_receipt.v1", "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "development_parent_count": DEVELOPMENT_PARENT_COUNT, "expected_frame_count": DEVELOPMENT_FRAME_COUNT, "selected_r8_faro_read": False, "training_steps": 0, "network_requests": 0, "one_shot_consumed_on_root_creation": True})
    try:
        frames, sources, receipts = load_development_rows()
        index_by_frame = {frame.physical_frame_id: index for index, frame in enumerate(frames)}
        groups: dict[tuple[str, str], list[tuple[r6io.R6FrameRef, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for row in zip(frames, sources, receipts, strict=True):
            groups[(row[0].parent_id, row[0].video_id)].append(row)
        faro_reads: Counter[str] = Counter()
        counter_lock = threading.Lock()

        def observed(role: str, _: str) -> None:
            require(role == "highres_depth", "R9_DEV_PAYLOAD_FIREWALL", "development attempted non-FARO payload read")
            with counter_lock:
                faro_reads[role] += 1

        def label_parent(rows: list[tuple[r6io.R6FrameRef, dict[str, Any], dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
            output = []
            with zipfile.ZipFile(rows[0][0].upsampling_archive) as bundle:
                for frame, source, receipt in rows:
                    payload, _binding = r6io._read_member(bundle, frame.members["highres_depth"], observer=observed)
                    faro = materializer._decode_png(payload, "highres_depth")
                    label = r7_canary.build_label_frame_record(source, faro, receipt["intrinsics_highres"]["matrix_3x3"], receipt["gravity_up_camera_xyz"])
                    output.append((index_by_frame[frame.physical_frame_id], label))
            return output

        indexed_labels = []
        completed_parents = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="taro-r9-faro") as executor:
            futures = [executor.submit(label_parent, rows) for _identity, rows in sorted(groups.items())]
            for future in concurrent.futures.as_completed(futures):
                indexed_labels.extend(future.result())
                completed_parents += 1
                require(time.monotonic() - started <= lock["resource_budget"]["maximum_wall_seconds"] and process.memory_info().rss <= lock["resource_budget"]["maximum_peak_rss_bytes"], "R9_DEV_RESOURCE", "development resource budget exceeded")
                print(json.dumps({"phase": "R9_DEVELOPMENT_FARO_LABEL", "completed_parents": completed_parents, "total_parents": DEVELOPMENT_PARENT_COUNT, "completed_frames": len(indexed_labels), "total_frames": DEVELOPMENT_FRAME_COUNT}, sort_keys=True), flush=True)
        indexed_labels.sort(key=lambda row: row[0])
        require([index for index, _label in indexed_labels] == list(range(DEVELOPMENT_FRAME_COUNT)) and faro_reads == Counter({"highres_depth": DEVELOPMENT_FRAME_COUNT}), "R9_DEV_LABEL_COUNT", "development label/FARO count drift")
        labels = [label for _index, label in indexed_labels]
        label_hashes = []
        for frame, label in zip(frames, labels, strict=True):
            writer.write_json_gzip(_label_relative(frame), label)
            label_hashes.append(label["content_sha256"])
        selector = fit.validate_selector(fit.fit_selector(sources, labels))
        selector = _seal(selector)
        writer.write_json("selector.json", selector)
        completion = _seal({"schema": "blindassist.taro.o1r.r9_clear_enrichment_development_label_completion.v1", "parent_count": DEVELOPMENT_PARENT_COUNT, "frame_count": DEVELOPMENT_FRAME_COUNT, "query_count": DEVELOPMENT_QUERY_COUNT, "label_hash_sequence_sha256": adapter.canonical_sha256(label_hashes), "faro_payload_reads": dict(faro_reads), "selector_sha256": selector["content_sha256"], "training_steps": 0, "network_requests": 0, "unknown_is_negative": False})
        writer.write_json("label-completion.json", completion)
        target_passed = bool(selector["development_target"]["passed"])
        result = {"schema": "blindassist.taro.o1r.r9_clear_enrichment_development_result.v1", "terminal": PASS_TERMINAL if target_passed else FAIL_TERMINAL, "execution_valid": True, "passed": target_passed, "development_parent_count": DEVELOPMENT_PARENT_COUNT, "frame_count": DEVELOPMENT_FRAME_COUNT, "query_count": DEVELOPMENT_QUERY_COUNT, "faro_frame_count": DEVELOPMENT_FRAME_COUNT, "label_state_counts": selector["development_label_state_counts"], "selector_sha256": selector["content_sha256"], "chosen_rule": selector["chosen_rule"], "selected_parent_identities": selector["selected_parent_identities"], "selected_parent_scores": selector["selected_parent_scores"], "selected_label_state_counts": selector["selected_label_state_counts"], "selected_parents_with_clear": selector["selected_parents_with_clear"], "selected_nonzero_score_parent_count": selector["selected_nonzero_score_parent_count"], "matched_rule_label_state_counts": selector["matched_rule_label_state_counts"], "matched_rule_clear_precision_on_definite_labels": selector["matched_rule_clear_precision_on_definite_labels"], "development_target": selector["development_target"], "label_completion_sha256": completion["content_sha256"], "all_r8_parents_consumed": True, "confirmation_authority": False, "training_steps": 0, "network_requests": 0, "elapsed_seconds": round(time.monotonic() - started, 6), "one_shot_consumed": True, "unknown_is_negative": False, "claim_ceiling": "Development-only FARO labels and a frozen source-only selector from consumed R8 parents; no fresh confirmation, effectiveness promotion, deployment, product, or safety claim."}
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r9_clear_enrichment_development_manifest.v1", "terminal": result["terminal"], "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
        return result
    except Exception as error:
        try:
            writer.write_json("failure.json", {"schema": "blindassist.taro.o1r.r9_clear_enrichment_development_failure.v1", "terminal": INVALID_TERMINAL, "execution_valid": False, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "one_shot_consumed": True})
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            lock = validate_execution_lock(args.execution_lock)
            print(json.dumps({"lock_id": lock["lock_id"], "valid": True, "output_root_absent": True}, sort_keys=True))
            return 0
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"terminal": INVALID_TERMINAL, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "label_state_counts": result["label_state_counts"], "selected_label_state_counts": result["selected_label_state_counts"], "selected_parents_with_clear": result["selected_parents_with_clear"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
