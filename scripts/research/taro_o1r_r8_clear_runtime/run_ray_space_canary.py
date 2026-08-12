#!/usr/bin/env python3
"""Run the bounded R8 FARO-owned ray-space truth-interface canary."""

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
from scripts.research.taro_o1r_r8_clear_runtime import ray_space_clear as ray
from scripts.research.taro_o1r_r8_clear_runtime import run_selected_phase_b as phase_b


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r8_faro_ray_space_canary_execution_lock.v1"
LOCK_ID = "TARO_O1R_R8_FARO_RAY_SPACE_TRUTH_INTERFACE_CANARY_ONE_SHOT_EXECUTION_LOCK"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R8_FARO_RAY_SPACE_TRUTH_INTERFACE_CANARY_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r8-faro-ray-space-canary-r0"
PRIOR_PHASE_B_ROOT = phase_b.OUTPUT_ROOT
PASS_TERMINAL = "TARO_O1R_R8_FARO_RAY_SPACE_TRUTH_INTERFACE_CANARY_PASS"
FAIL_TERMINAL = "TARO_O1R_R8_FARO_RAY_SPACE_TRUTH_INTERFACE_CANARY_FAIL"
INVALID_TERMINAL = "TARO_O1R_R8_FARO_RAY_SPACE_TRUTH_INTERFACE_CANARY_EXECUTION_INVALID"
SELECTED_PARENT_COUNT = 8
SELECTED_FRAME_COUNT = 133
SELECTED_QUERY_COUNT = 1197

EXPECTED_BINDINGS = {
    "R8_PHASE_A_COMPLETION": f"{phase_b.PHASE_A_ROOT}/phase-a-completion.json",
    "R8_PHASE_A_MANIFEST": f"{phase_b.PHASE_A_ROOT}/manifest.json",
    "R8_SELECTION": f"{phase_b.SELECTION_ROOT}/selection.json",
    "R8_PHASE_B_RESULT": f"{PRIOR_PHASE_B_ROOT}/result.json",
    "R8_PHASE_B_LABEL_COMPLETION": f"{PRIOR_PHASE_B_ROOT}/label-completion.json",
    "R8_PHASE_B_MANIFEST": f"{PRIOR_PHASE_B_ROOT}/manifest.json",
    "R8_PHASE_B_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_selected_phase_b.py",
    "R8_RAY_SPACE_RUNTIME": "scripts/research/taro_o1r_r8_clear_runtime/ray_space_clear.py",
    "R8_RAY_SPACE_TEST": "scripts/research/taro_o1r_r8_clear_runtime/test_ray_space_clear.py",
    "R8_RAY_SPACE_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_ray_space_canary.py",
    "R8_RAY_SPACE_RUNNER_TEST": "scripts/research/taro_o1r_r8_clear_runtime/test_run_ray_space_canary.py",
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "继续推进",
    "scope": "Resume TARO after restart and run the truth-owned FARO ray-space interface canary on exactly the same already-selected 8 parents and 133 frames; no unselected FARO, source reselection, selector or threshold fit, training, deployment, product, or safety authority.",
}
EXPECTED_BUDGET = {
    "maximum_wall_seconds": 3600,
    "maximum_peak_rss_bytes": 8589934592,
    "maximum_evidence_bytes": 268435456,
}


class RaySpaceCanaryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise RaySpaceCanaryError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R8_RAY_RUN_SEAL_COLLISION", "ray canary caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _ray_label_relative(frame: r6io.R6FrameRef) -> str:
    return f"ray-labels/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _verify_manifest(root_relative: str, manifest: Mapping[str, Any], schema: str, terminal: str | None = None) -> None:
    require(manifest.get("schema") == schema, "R8_RAY_MANIFEST_SCHEMA", f"manifest schema drift: {root_relative}")
    if terminal is not None:
        require(manifest.get("terminal") == terminal, "R8_RAY_MANIFEST_TERMINAL", f"manifest terminal drift: {root_relative}")
    files = manifest.get("files")
    require(isinstance(files, dict) and manifest.get("file_count_before_manifest") == len(files) and "manifest.json" not in files, "R8_RAY_MANIFEST_COUNT", f"manifest count drift: {root_relative}")
    root = _repo_path(root_relative)
    for relative, receipt in files.items():
        require(isinstance(receipt, dict) and receipt.get("path") == relative, "R8_RAY_MANIFEST_ROW", f"manifest row drift: {relative}")
        target = materializer.safe_join(root, relative)
        require(target.is_file() and target.stat().st_size == receipt.get("bytes") and materializer.sha256_file(target) == receipt.get("sha256"), "R8_RAY_MANIFEST_FILE", f"manifest file drift: {root_relative}/{relative}")


def _verify_prior_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    phase_a_manifest = _read_json(_repo_path(EXPECTED_BINDINGS["R8_PHASE_A_MANIFEST"]))
    _verify_manifest(phase_b.PHASE_A_ROOT, phase_a_manifest, "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_manifest.v1", "TARO_O1R_R8_CLEAR_POOL_PHASE_A_SOURCE_ONLY_RECOVERY_SEALED_PASS_R1")
    prior_result = _read_json(_repo_path(EXPECTED_BINDINGS["R8_PHASE_B_RESULT"]))
    require(prior_result.get("schema") == "blindassist.taro.o1r.r8_clear_negative_control_confirmation_result.v1" and prior_result.get("execution_valid") is True and prior_result.get("terminal") == phase_b.NOT_EVALUABLE_TERMINAL and prior_result.get("frame_count") == SELECTED_FRAME_COUNT and prior_result.get("query_count") == SELECTED_QUERY_COUNT and prior_result.get("faro_frame_count") == SELECTED_FRAME_COUNT, "R8_RAY_PRIOR_RESULT", "prior Phase-B result not admitted")
    prior_manifest = _read_json(_repo_path(EXPECTED_BINDINGS["R8_PHASE_B_MANIFEST"]))
    _verify_manifest(PRIOR_PHASE_B_ROOT, prior_manifest, "blindassist.taro.o1r.r8_clear_selected_phase_b_manifest.v1", phase_b.NOT_EVALUABLE_TERMINAL)
    completion = _read_json(_repo_path(EXPECTED_BINDINGS["R8_PHASE_B_LABEL_COMPLETION"]))
    require(completion.get("content_sha256") == prior_result.get("label_completion_sha256") and completion.get("frame_count") == SELECTED_FRAME_COUNT and completion.get("query_count") == SELECTED_QUERY_COUNT and completion.get("faro_payload_reads") == {"highres_depth": SELECTED_FRAME_COUNT}, "R8_RAY_PRIOR_COMPLETION", "prior Phase-B label completion drift")
    return prior_result, completion


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R8_RAY_LOCK_PATH", "ray canary lock path drift")
    lock = _read_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R8_RAY_LOCK_IDENTITY", "ray canary lock identity drift")
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R8_RAY_USER_AUTHORITY", "ray canary user authority drift")
    expected_argv = ["scripts/research/taro_o1r_r8_clear_runtime/run_ray_space_canary.py", "--execution-lock", LOCK_RELATIVE]
    require(lock.get("argv") == expected_argv and lock.get("output_root") == OUTPUT_ROOT and lock.get("prior_phase_b_root") == PRIOR_PHASE_B_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R8_RAY_LOCK_POLICY", "ray canary root/argv policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R8_RAY_BINDINGS", "ray canary binding count drift")
    seen = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in seen and EXPECTED_BINDINGS.get(role) == relative, "R8_RAY_BINDING_ROW", "ray canary binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R8_RAY_BINDING_HASH", f"ray canary binding drift: {relative}")
        seen.add(role)
    prior_result, _ = _verify_prior_evidence()
    selection, frames, _sources, _receipts = phase_b.load_selected_rows()
    identities = [[row["parent_id"], row["video_id"]] for row in selection["selected_parents"]]
    require(len(identities) == SELECTED_PARENT_COUNT and len(frames) == SELECTED_FRAME_COUNT, "R8_RAY_COHORT", "ray canary selected cohort drift")
    require(lock.get("selected_cohort") == {"parent_count": SELECTED_PARENT_COUNT, "physical_frame_count": SELECTED_FRAME_COUNT, "query_count": SELECTED_QUERY_COUNT, "selected_parent_identities": identities, "selection_sha256": selection["content_sha256"]}, "R8_RAY_COHORT_LOCK", "ray canary lock cohort drift")
    require(lock.get("execution_authority") == {"prior_phase_b_reload": True, "phase_a_source_lineage_reload": True, "faro_payload_reread": True, "faro_frame_count": SELECTED_FRAME_COUNT, "truth_owned_ray_label_construction": True, "post_hoc_interface_guardrail_evaluation": True, "read_unselected_parent_faro": False, "source_reselection": False, "selector_fit": False, "threshold_fit": False, "training": False, "network": False, "device": False, "deployment": False, "product": False, "safety": False}, "R8_RAY_AUTHORITY", "ray canary authority drift")
    require(lock.get("fixed_interface") == {"labeler_id": ray.LABELER_ID, "forward_slices_m": list(ray.FORWARD_SLICES_M), "lateral_anchors_m": list(ray.LATERAL_ANCHORS_M), "height_anchors_m": list(ray.HEIGHT_ANCHORS_M), "patch_radius_px": ray.PATCH_RADIUS_PX, "depth_endpoint_tolerance_m": ray.DEPTH_ENDPOINT_TOLERANCE_M, "minimum_projected_anchors_per_slice": ray.MINIMUM_PROJECTED_ANCHORS_PER_SLICE, "minimum_valid_anchor_fraction_per_slice": ray.MINIMUM_VALID_ANCHOR_FRACTION_PER_SLICE, "minimum_blocked_anchors_for_occupied": ray.MINIMUM_BLOCKED_ANCHORS_FOR_OCCUPIED, "minimum_clear_query_count": ray.MINIMUM_CLEAR_QUERY_COUNT, "minimum_clear_parent_count": ray.MINIMUM_CLEAR_PARENT_COUNT, "unknown_is_negative": False}, "R8_RAY_INTERFACE", "ray canary interface drift")
    require(lock.get("resource_budget") == EXPECTED_BUDGET and lock.get("prior_phase_b_result_sha256") == materializer.sha256_file(_repo_path(EXPECTED_BINDINGS["R8_PHASE_B_RESULT"])) and prior_result.get("clear_branch_promotion") is False, "R8_RAY_BUDGET_OR_PRIOR", "ray canary budget/prior binding drift")
    require(lock.get("claim_ceiling") == "Post-hoc FARO truth-interface observability and compatibility evidence on consumed R8 selected frames only; no effectiveness, route promotion, deployment, product, or safety claim.", "R8_RAY_CLAIM", "ray canary claim ceiling drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R8_RAY_ROOT_COLLISION", "ray canary output root exists")
    lock["_lock_path"] = lock_path
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), *sys.argv[1:]]
    require(actual_argv == lock["argv"], "R8_RAY_ACTUAL_ARGV", "ray canary must use the unique locked argv")
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(lock["resource_budget"]["maximum_evidence_bytes"]))
    started = time.monotonic()
    process = psutil.Process()
    writer.activate({"schema": "blindassist.taro.o1r.r8_faro_ray_space_canary_execution_receipt.v1", "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "prior_phase_b_reloaded_and_verified_before_faro": True, "selected_parent_count": SELECTED_PARENT_COUNT, "expected_frame_count": SELECTED_FRAME_COUNT, "source_reselection": False, "selector_fit": False, "threshold_fit": False, "training_steps": 0, "network_requests": 0, "one_shot_consumed_on_root_creation": True})
    try:
        selection, frames, sources, receipts = phase_b.load_selected_rows()
        old_labels = []
        old_hashes = []
        for frame, source in zip(frames, sources, strict=True):
            old = r7_canary.validate_label_frame_record(_read_gzip_json(_repo_path(PRIOR_PHASE_B_ROOT) / phase_b._label_relative(frame)), source)
            old_labels.append(old)
            old_hashes.append(old["content_sha256"])
        faro_reads: Counter[str] = Counter()
        ray_labels = []
        ray_hashes = []

        def observed(role: str, _: str) -> None:
            require(role == "highres_depth", "R8_RAY_PAYLOAD_FIREWALL", "ray canary attempted non-FARO payload read")
            faro_reads[role] += 1

        for _key, parent_rows_iter in groupby(list(zip(frames, sources, receipts, strict=True)), key=lambda row: (row[0].parent_id, row[0].video_id)):
            parent_rows = list(parent_rows_iter)
            with zipfile.ZipFile(parent_rows[0][0].upsampling_archive) as bundle:
                for frame, source, receipt in parent_rows:
                    payload, _binding = r6io._read_member(bundle, frame.members["highres_depth"], observer=observed)
                    faro = materializer._decode_png(payload, "highres_depth")
                    label = ray.build_label_frame(source, faro, receipt["intrinsics_highres"]["matrix_3x3"], receipt["gravity_up_camera_xyz"])
                    writer.write_json_gzip(_ray_label_relative(frame), label)
                    ray_labels.append(label)
                    ray_hashes.append(label["content_sha256"])
                    require(time.monotonic() - started <= lock["resource_budget"]["maximum_wall_seconds"] and process.memory_info().rss <= lock["resource_budget"]["maximum_peak_rss_bytes"], "R8_RAY_RESOURCE", "ray canary resource budget exceeded")
                    if len(ray_labels) % 10 == 0 or len(ray_labels) == len(frames):
                        print(json.dumps({"phase": "R8_FARO_RAY_SPACE_LABEL", "completed": len(ray_labels), "total": len(frames), "physical_frame_id": frame.physical_frame_id}, sort_keys=True), flush=True)
        require(faro_reads == Counter({"highres_depth": SELECTED_FRAME_COUNT}), "R8_RAY_FARO_COUNT", "ray canary FARO read count drift")
        summary = ray.summarize(sources, old_labels, ray_labels)
        completion = _seal({"schema": "blindassist.taro.o1r.r8_faro_ray_space_canary_completion.v1", "frame_count": SELECTED_FRAME_COUNT, "query_count": SELECTED_QUERY_COUNT, "selection_sha256": selection["content_sha256"], "prior_label_hash_sequence_sha256": adapter.canonical_sha256(old_hashes), "ray_label_hash_sequence_sha256": adapter.canonical_sha256(ray_hashes), "faro_payload_reads": dict(faro_reads), "source_reselection": False, "selector_fit": False, "threshold_fit": False, "training_steps": 0, "network_requests": 0, "unknown_is_negative": False})
        writer.write_json("ray-label-completion.json", completion)
        result = {"schema": "blindassist.taro.o1r.r8_faro_ray_space_truth_interface_canary_result.v1", **summary, "terminal": PASS_TERMINAL if summary["passed"] else FAIL_TERMINAL, "execution_valid": True, "passed": bool(summary["passed"]), "selected_parent_count": SELECTED_PARENT_COUNT, "frame_count": SELECTED_FRAME_COUNT, "query_count": SELECTED_QUERY_COUNT, "faro_frame_count": SELECTED_FRAME_COUNT, "selection_sha256": selection["content_sha256"], "prior_phase_b_terminal": phase_b.NOT_EVALUABLE_TERMINAL, "ray_label_completion_sha256": completion["content_sha256"], "source_reselection": False, "selector_fit": False, "threshold_fit": False, "training_steps": 0, "network_requests": 0, "route_promotion_authorized": False, "elapsed_seconds": round(time.monotonic() - started, 6), "one_shot_consumed": True}
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r8_faro_ray_space_canary_manifest.v1", "terminal": result["terminal"], "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
        return result
    except Exception as error:
        try:
            writer.write_json("failure.json", {"schema": "blindassist.taro.o1r.r8_faro_ray_space_canary_failure.v1", "terminal": INVALID_TERMINAL, "execution_valid": False, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "one_shot_consumed": True})
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
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "label_state_counts": result["label_state_counts"], "parents_with_clear": result["parents_with_clear"], "guardrails_passed": result["guardrails_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
