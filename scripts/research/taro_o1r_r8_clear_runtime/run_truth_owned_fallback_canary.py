#!/usr/bin/env python3
"""Run the bounded R8 dense FARO truth-owned fallback canary."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import sys
import time
import zipfile
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation_io as r6io
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r8_clear_runtime import run_ray_space_canary as ray_v1_runner
from scripts.research.taro_o1r_r8_clear_runtime import run_selected_phase_b as phase_b
from scripts.research.taro_o1r_r8_clear_runtime import truth_owned_fallback as fallback


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r8_dense_truth_owned_fallback_canary_execution_lock.v1"
LOCK_ID = "TARO_O1R_R8_DENSE_FARO_TRUTH_OWNED_FALLBACK_CANARY_ONE_SHOT_EXECUTION_LOCK"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R8_DENSE_FARO_TRUTH_OWNED_FALLBACK_CANARY_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r8-dense-truth-owned-fallback-canary-r0"
V1_ROOT = ray_v1_runner.OUTPUT_ROOT
PASS_TERMINAL = "TARO_O1R_R8_DENSE_FARO_TRUTH_OWNED_FALLBACK_CANARY_PASS"
FAIL_TERMINAL = "TARO_O1R_R8_DENSE_FARO_TRUTH_OWNED_FALLBACK_CANARY_FAIL"
INVALID_TERMINAL = "TARO_O1R_R8_DENSE_FARO_TRUTH_OWNED_FALLBACK_CANARY_EXECUTION_INVALID"
SELECTED_PARENT_COUNT = ray_v1_runner.SELECTED_PARENT_COUNT
SELECTED_FRAME_COUNT = ray_v1_runner.SELECTED_FRAME_COUNT
SELECTED_QUERY_COUNT = ray_v1_runner.SELECTED_QUERY_COUNT

EXPECTED_BINDINGS = {
    "R8_PHASE_A_COMPLETION": ray_v1_runner.EXPECTED_BINDINGS["R8_PHASE_A_COMPLETION"],
    "R8_PHASE_A_MANIFEST": ray_v1_runner.EXPECTED_BINDINGS["R8_PHASE_A_MANIFEST"],
    "R8_SELECTION": ray_v1_runner.EXPECTED_BINDINGS["R8_SELECTION"],
    "R8_PHASE_B_RESULT": ray_v1_runner.EXPECTED_BINDINGS["R8_PHASE_B_RESULT"],
    "R8_PHASE_B_LABEL_COMPLETION": ray_v1_runner.EXPECTED_BINDINGS["R8_PHASE_B_LABEL_COMPLETION"],
    "R8_PHASE_B_MANIFEST": ray_v1_runner.EXPECTED_BINDINGS["R8_PHASE_B_MANIFEST"],
    "R7_DENSE_LABEL_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R8_RAY_V1_RESULT": f"{V1_ROOT}/result.json",
    "R8_RAY_V1_COMPLETION": f"{V1_ROOT}/ray-label-completion.json",
    "R8_RAY_V1_MANIFEST": f"{V1_ROOT}/manifest.json",
    "R8_RAY_V1_RUNTIME": "scripts/research/taro_o1r_r8_clear_runtime/ray_space_clear.py",
    "R8_FALLBACK_RUNTIME": "scripts/research/taro_o1r_r8_clear_runtime/truth_owned_fallback.py",
    "R8_FALLBACK_TEST": "scripts/research/taro_o1r_r8_clear_runtime/test_truth_owned_fallback.py",
    "R8_FALLBACK_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_truth_owned_fallback_canary.py",
    "R8_FALLBACK_RUNNER_TEST": "scripts/research/taro_o1r_r8_clear_runtime/test_run_truth_owned_fallback_canary.py",
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "继续推进",
    "scope": "Continue TARO after the consumed sparse ray-space V1 failure by running the dense FARO truth-owned fallback on exactly the same 8 parents and 133 frames; preserve all existing dense labels and use FARO-owned queries only for missing source-query slots; no unselected FARO, fitting, training, deployment, product, or safety authority.",
}
EXPECTED_BUDGET = ray_v1_runner.EXPECTED_BUDGET


class FallbackCanaryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise FallbackCanaryError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R8_FALLBACK_RUN_SEAL_COLLISION", "fallback canary caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _label_relative(frame: r6io.R6FrameRef) -> str:
    return f"fallback-labels/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _verify_v1_failure() -> tuple[dict[str, Any], dict[str, Any]]:
    result = _read_json(_repo_path(EXPECTED_BINDINGS["R8_RAY_V1_RESULT"]))
    require(result.get("schema") == "blindassist.taro.o1r.r8_faro_ray_space_truth_interface_canary_result.v1" and result.get("execution_valid") is True and result.get("terminal") == ray_v1_runner.FAIL_TERMINAL and result.get("passed") is False and result.get("coverage_gate", {}).get("passed") is True and result.get("guardrails_passed") is False and result.get("old_occupied_reclassified_clear") == 36 and result.get("positive_occupancy_predictions_on_ray_clear") == 33, "R8_FALLBACK_V1_FAILURE", "sparse ray-space V1 failure not admitted")
    manifest = _read_json(_repo_path(EXPECTED_BINDINGS["R8_RAY_V1_MANIFEST"]))
    ray_v1_runner._verify_manifest(V1_ROOT, manifest, "blindassist.taro.o1r.r8_faro_ray_space_canary_manifest.v1", ray_v1_runner.FAIL_TERMINAL)
    completion = _read_json(_repo_path(EXPECTED_BINDINGS["R8_RAY_V1_COMPLETION"]))
    require(completion.get("content_sha256") == result.get("ray_label_completion_sha256") and completion.get("frame_count") == SELECTED_FRAME_COUNT and completion.get("query_count") == SELECTED_QUERY_COUNT and completion.get("faro_payload_reads") == {"highres_depth": SELECTED_FRAME_COUNT}, "R8_FALLBACK_V1_COMPLETION", "sparse ray-space V1 completion drift")
    return result, completion


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R8_FALLBACK_LOCK_PATH", "fallback canary lock path drift")
    lock = _read_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R8_FALLBACK_LOCK_IDENTITY", "fallback canary lock identity drift")
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R8_FALLBACK_USER_AUTHORITY", "fallback canary user authority drift")
    expected_argv = ["scripts/research/taro_o1r_r8_clear_runtime/run_truth_owned_fallback_canary.py", "--execution-lock", LOCK_RELATIVE]
    require(lock.get("argv") == expected_argv and lock.get("output_root") == OUTPUT_ROOT and lock.get("prior_phase_b_root") == phase_b.OUTPUT_ROOT and lock.get("sparse_v1_root") == V1_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R8_FALLBACK_LOCK_POLICY", "fallback canary root/argv policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R8_FALLBACK_BINDINGS", "fallback canary binding count drift")
    seen = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in seen and EXPECTED_BINDINGS.get(role) == relative, "R8_FALLBACK_BINDING_ROW", "fallback canary binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R8_FALLBACK_BINDING_HASH", f"fallback canary binding drift: {relative}")
        seen.add(role)
    ray_v1_runner._verify_prior_evidence()
    v1_result, _ = _verify_v1_failure()
    selection, frames, _sources, _receipts = phase_b.load_selected_rows()
    identities = [[row["parent_id"], row["video_id"]] for row in selection["selected_parents"]]
    require(len(identities) == SELECTED_PARENT_COUNT and len(frames) == SELECTED_FRAME_COUNT and lock.get("selected_cohort") == {"parent_count": SELECTED_PARENT_COUNT, "physical_frame_count": SELECTED_FRAME_COUNT, "query_count": SELECTED_QUERY_COUNT, "selected_parent_identities": identities, "selection_sha256": selection["content_sha256"]}, "R8_FALLBACK_COHORT", "fallback canary selected cohort drift")
    require(lock.get("execution_authority") == {"prior_phase_b_reload": True, "sparse_v1_failure_reload": True, "phase_a_source_lineage_reload": True, "faro_payload_reread": True, "faro_frame_count": SELECTED_FRAME_COUNT, "dense_truth_label_construction": True, "faro_owned_query_only_for_missing_source_query": True, "read_unselected_parent_faro": False, "source_reselection": False, "selector_fit": False, "threshold_fit": False, "training": False, "network": False, "device": False, "deployment": False, "product": False, "safety": False}, "R8_FALLBACK_AUTHORITY", "fallback canary authority drift")
    fixed = {"labeler_id": fallback.LABELER_ID, "existing_query_policy": "PRESERVE_SOURCE_QUERY_AND_R7_DENSE_FARO_LABEL_EXACTLY", "missing_query_policy": "FARO_TRUTH_PLANE_3X3_QUERY_WITH_R7_DENSE_FARO_LABEL", "minimum_truth_obstacle_pixels": r7_canary.MINIMUM_TRUTH_OBSTACLE_PIXELS, "minimum_query_support_points": adapter.MINIMUM_QUERY_SUPPORT_POINTS, "minimum_observed_forward_m": adapter.MINIMUM_QUERY_OBSERVED_FORWARD_M, "minimum_local_valid_fraction": adapter.MINIMUM_BOUNDARY_LOCAL_VALID_FRACTION, "minimum_clear_query_count": fallback.MINIMUM_CLEAR_QUERY_COUNT, "minimum_clear_parent_count": fallback.MINIMUM_CLEAR_PARENT_COUNT, "unknown_is_negative": False}
    require(lock.get("fixed_interface") == fixed and lock.get("resource_budget") == EXPECTED_BUDGET and lock.get("sparse_v1_result_sha256") == materializer.sha256_file(_repo_path(EXPECTED_BINDINGS["R8_RAY_V1_RESULT"])) and v1_result.get("route_promotion_authorized") is False, "R8_FALLBACK_INTERFACE", "fallback interface/budget/prior binding drift")
    require(lock.get("claim_ceiling") == "Post-hoc dense FARO truth-owned fallback interface evidence on consumed R8 selected frames only; no effectiveness, route promotion, deployment, product, or safety claim.", "R8_FALLBACK_CLAIM", "fallback claim ceiling drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R8_FALLBACK_ROOT_COLLISION", "fallback canary output root exists")
    lock["_lock_path"] = lock_path
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), *sys.argv[1:]]
    require(actual_argv == lock["argv"], "R8_FALLBACK_ACTUAL_ARGV", "fallback canary must use the unique locked argv")
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(lock["resource_budget"]["maximum_evidence_bytes"]))
    started = time.monotonic()
    process = psutil.Process()
    writer.activate({"schema": "blindassist.taro.o1r.r8_dense_truth_owned_fallback_canary_execution_receipt.v1", "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "prior_phase_b_and_sparse_v1_verified_before_faro": True, "selected_parent_count": SELECTED_PARENT_COUNT, "expected_frame_count": SELECTED_FRAME_COUNT, "source_reselection": False, "selector_fit": False, "threshold_fit": False, "training_steps": 0, "network_requests": 0, "one_shot_consumed_on_root_creation": True})
    try:
        selection, frames, sources, receipts = phase_b.load_selected_rows()
        old_labels = []
        old_hashes = []
        for frame, source in zip(frames, sources, strict=True):
            old = r7_canary.validate_label_frame_record(_read_gzip_json(_repo_path(phase_b.OUTPUT_ROOT) / phase_b._label_relative(frame)), source)
            old_labels.append(old)
            old_hashes.append(old["content_sha256"])
        faro_reads: Counter[str] = Counter()
        labels = []
        label_hashes = []

        def observed(role: str, _: str) -> None:
            require(role == "highres_depth", "R8_FALLBACK_PAYLOAD_FIREWALL", "fallback canary attempted non-FARO payload read")
            faro_reads[role] += 1

        for _key, parent_rows_iter in groupby(list(zip(frames, sources, receipts, strict=True)), key=lambda row: (row[0].parent_id, row[0].video_id)):
            parent_rows = list(parent_rows_iter)
            with zipfile.ZipFile(parent_rows[0][0].upsampling_archive) as bundle:
                for frame, source, receipt in parent_rows:
                    payload, _binding = r6io._read_member(bundle, frame.members["highres_depth"], observer=observed)
                    faro = materializer._decode_png(payload, "highres_depth")
                    label = fallback.build_label_frame(source, faro, receipt["intrinsics_highres"]["matrix_3x3"], receipt["gravity_up_camera_xyz"])
                    writer.write_json_gzip(_label_relative(frame), label)
                    labels.append(label)
                    label_hashes.append(label["content_sha256"])
                    require(time.monotonic() - started <= lock["resource_budget"]["maximum_wall_seconds"] and process.memory_info().rss <= lock["resource_budget"]["maximum_peak_rss_bytes"], "R8_FALLBACK_RESOURCE", "fallback canary resource budget exceeded")
                    if len(labels) % 10 == 0 or len(labels) == len(frames):
                        print(json.dumps({"phase": "R8_DENSE_TRUTH_OWNED_FALLBACK_LABEL", "completed": len(labels), "total": len(frames), "physical_frame_id": frame.physical_frame_id}, sort_keys=True), flush=True)
        require(faro_reads == Counter({"highres_depth": SELECTED_FRAME_COUNT}), "R8_FALLBACK_FARO_COUNT", "fallback canary FARO read count drift")
        summary = fallback.summarize(sources, old_labels, labels)
        completion = _seal({"schema": "blindassist.taro.o1r.r8_dense_truth_owned_fallback_canary_completion.v1", "frame_count": SELECTED_FRAME_COUNT, "query_count": SELECTED_QUERY_COUNT, "selection_sha256": selection["content_sha256"], "prior_label_hash_sequence_sha256": adapter.canonical_sha256(old_hashes), "fallback_label_hash_sequence_sha256": adapter.canonical_sha256(label_hashes), "faro_payload_reads": dict(faro_reads), "source_reselection": False, "selector_fit": False, "threshold_fit": False, "training_steps": 0, "network_requests": 0, "unknown_is_negative": False})
        writer.write_json("fallback-label-completion.json", completion)
        result = {"schema": "blindassist.taro.o1r.r8_dense_truth_owned_fallback_canary_result.v1", **summary, "terminal": PASS_TERMINAL if summary["passed"] else FAIL_TERMINAL, "execution_valid": True, "passed": bool(summary["passed"]), "selected_parent_count": SELECTED_PARENT_COUNT, "frame_count": SELECTED_FRAME_COUNT, "query_count": SELECTED_QUERY_COUNT, "faro_frame_count": SELECTED_FRAME_COUNT, "selection_sha256": selection["content_sha256"], "sparse_v1_terminal": ray_v1_runner.FAIL_TERMINAL, "fallback_label_completion_sha256": completion["content_sha256"], "source_reselection": False, "selector_fit": False, "threshold_fit": False, "training_steps": 0, "network_requests": 0, "route_promotion_authorized": False, "elapsed_seconds": round(time.monotonic() - started, 6), "one_shot_consumed": True}
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r8_dense_truth_owned_fallback_canary_manifest.v1", "terminal": result["terminal"], "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
        return result
    except Exception as error:
        try:
            writer.write_json("failure.json", {"schema": "blindassist.taro.o1r.r8_dense_truth_owned_fallback_canary_failure.v1", "terminal": INVALID_TERMINAL, "execution_valid": False, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "one_shot_consumed": True})
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
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "label_state_counts": result["label_state_counts"], "fallback_label_state_counts": result["fallback_label_state_counts"], "parents_with_clear": result["parents_with_clear"], "guardrails_passed": result["guardrails_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
