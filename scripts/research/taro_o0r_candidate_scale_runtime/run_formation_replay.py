#!/usr/bin/env python3
"""Execute the sealed 24-parent TARO R6 non-promotable formation replay."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import json
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import formation_replay
from scripts.research.taro_o0r_candidate_scale_runtime import formation_replay_io as replay_io
from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as runtime
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


RESULT_SCHEMA = "blindassist.taro.o0r.r6_formation_replay_result.v1"
MANIFEST_SCHEMA = "blindassist.taro.o0r.r6_formation_replay_manifest.v1"
PHASE_A_SCHEMA = "blindassist.taro.o0r.r6_formation_replay_phase_a_completion.v1"
FAIL_TERMINAL = "TARO_O0R_R6_FORMATION_REPLAY_EXECUTION_INVALID"


class FormationReplayRunError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FormationReplayRunError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "FORMATION_RUN_RECORD_INVALID", "formation replay record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(record.get("schema") == schema and observed == adapter.canonical_sha256(record), "FORMATION_RUN_RECORD_HASH_DRIFT", "formation replay record seal drift", schema=schema)
    record["content_sha256"] = observed
    return record


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def bundle_relative(frame: replay_io.FormationFrameRef) -> str:
    return f"phase-a-bundles/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def score_relative(frame: replay_io.FormationFrameRef) -> str:
    return f"phase-b-query-scores/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _lowres_matrix(source: Mapping[str, Any]) -> list[list[float]]:
    value = source["lowres_intrinsics_source"]
    return [[float(value["fx"]), 0.0, float(value["cx"])], [0.0, float(value["fy"]), float(value["cy"])], [0.0, 0.0, 1.0]]


def _load_bundle(root: Path, frame: replay_io.FormationFrameRef, highres: np.ndarray) -> dict[str, Any]:
    path = materializer.safe_join(root, bundle_relative(frame))
    require(path.is_file(), "FORMATION_PHASE_A_BUNDLE_MISSING", "formation Phase-A bundle is missing", path=bundle_relative(frame))
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise FormationReplayRunError("FORMATION_PHASE_A_BUNDLE_INVALID", "formation Phase-A bundle cannot be read") from error
    return runtime.validate_prospective_factor_bundle(value, candidate_highres_depth_m=highres)


def _phase_a_worker(
    frame: replay_io.FormationFrameRef,
    fit_candidate_root: Path,
    eval_candidate_root: Path,
) -> tuple[replay_io.FormationFrameRef, dict[str, Any]]:
    source = frame.source_frame_receipt
    with zipfile.ZipFile(frame.upsampling_archive) as bundle:
        apple = replay_io.read_bound_payload(frame, bundle, "lowres_depth")
        confidence = replay_io.read_bound_payload(frame, bundle, "confidence")
    candidate, native = replay_io.load_candidate_frame(fit_candidate_root, eval_candidate_root, frame)
    highres = depthart_runner.upsample_native_depth(native)
    factor_bundle = runtime.build_prospective_factor_bundle(
        parent_id=frame.parent_id,
        video_id=frame.video_id,
        timestamp_token=frame.timestamp_token,
        source_frame_receipt_sha256=source["content_sha256"],
        candidate_frame_record_sha256=candidate["content_sha256"],
        max_source_timestamp_ns=int(source["max_source_timestamp_ns"]),
        candidate_highres_depth_m=highres,
        apple_depth_mm=apple,
        confidence=confidence,
        intrinsics_highres_3x3=source["intrinsics_highres"]["matrix_3x3"],
        intrinsics_apple_3x3=_lowres_matrix(source),
        gravity_up_camera_xyz=source["gravity_up_camera_xyz"],
    )
    return frame, factor_bundle


def _phase_b_worker(
    frame: replay_io.FormationFrameRef,
    root: Path,
    fit_candidate_root: Path,
    eval_candidate_root: Path,
) -> tuple[replay_io.FormationFrameRef, list[dict[str, Any]]]:
    _, native = replay_io.load_candidate_frame(fit_candidate_root, eval_candidate_root, frame)
    highres = depthart_runner.upsample_native_depth(native)
    bundle = _load_bundle(root, frame, highres)
    with zipfile.ZipFile(frame.upsampling_archive) as source_bundle:
        apple = replay_io.read_bound_payload(frame, source_bundle, "lowres_depth")
        confidence = replay_io.read_bound_payload(frame, source_bundle, "confidence")
        faro = replay_io.read_bound_payload(frame, source_bundle, "highres_depth")
    records = formation_replay.score_frame(
        source_role=frame.source_role,
        source_frame_receipt=frame.source_frame_receipt,
        candidate_highres_depth_m=highres,
        apple_depth_mm=apple,
        confidence=confidence,
        prospective_bundle=bundle,
        highres_faro_depth_mm=faro,
    )
    return frame, records


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> dict[str, Any]:
    failure = {
        "schema": RESULT_SCHEMA,
        "terminal": FAIL_TERMINAL,
        "execution_valid": False,
        "scientific_pass_fail_assigned": False,
        "failure_code": str(getattr(error, "code", type(error).__name__)),
        "message": str(error),
        "one_shot_consumed": True,
    }
    try:
        writer.write_json("failure.json", failure)
        writer.write_json("manifest.json", {"schema": MANIFEST_SCHEMA, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written, "terminal": FAIL_TERMINAL})
    except Exception:
        pass
    return failure


def execute(execution_lock_path: Path) -> dict[str, Any]:
    from scripts.research.taro_o0r_candidate_scale_runtime.validate_formation_replay_execution_lock import validate_execution_lock

    lock = validate_execution_lock(execution_lock_path.resolve())
    roots = lock["roots"]
    repo_root = Path(roots["repo_root"]).resolve()
    source_root = Path(roots["source_root"]).resolve()
    source_evidence_root = Path(roots["source_evidence_root"]).resolve()
    fit_candidate_root = Path(roots["fit_candidate_root"]).resolve()
    eval_candidate_root = Path(roots["eval_candidate_root"]).resolve()
    output_root = Path(roots["output_root"]).resolve()
    plan_path = Path(roots["frame_plan_path"]).resolve()
    require(repo_root.is_dir() and source_root.is_dir() and source_evidence_root.is_dir() and fit_candidate_root.is_dir() and eval_candidate_root.is_dir() and plan_path.is_file() and not output_root.exists(), "FORMATION_ROOT_PREFLIGHT_INVALID", "formation replay roots do not satisfy preflight")
    frames = replay_io.load_exact_cohort(plan_path, source_evidence_root, source_root, verify_containers=True)
    preflight = replay_io.preflight_inventory(frames, fit_candidate_root, eval_candidate_root)
    frozen = lock["cohort_bindings"]
    for field in ("frame_key_sequence_sha256", "source_receipt_hash_sequence_sha256", "candidate_record_hash_sequence_sha256", "candidate_native_hash_sequence_sha256"):
        require(preflight[field] == frozen[field], "FORMATION_COHORT_BINDING_DRIFT", "formation replay preflight differs from execution lock", field=field)
    budget = lock["resource_budget"]
    started = time.monotonic()
    process = psutil.Process()
    writer = FactorEvidenceWriter(output_root, int(budget["maximum_evidence_bytes"]))

    def guard() -> None:
        require(time.monotonic() - started <= float(budget["maximum_wall_seconds"]), "FORMATION_EXECUTION_TIMEOUT", "formation replay exceeded wall-time budget")
        require(process.memory_info().rss <= int(budget["maximum_peak_rss_bytes"]), "FORMATION_EXECUTION_RSS_EXCEEDED", "formation replay exceeded RSS budget")

    try:
        lock_sha = materializer.sha256_file(execution_lock_path.resolve())
        writer.activate(
            {
                "schema": "blindassist.taro.o0r.r6_formation_replay_start.v1",
                "execution_lock_sha256": lock_sha,
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "one_shot_consumed_on_root_creation": True,
                "prior_outcome_roots_enumerated": False,
                "r6_untouched_parents_enumerated": False,
            }
        )
        bundle_hashes: list[str] = []
        selected_counts: Counter[str] = Counter()
        with ThreadPoolExecutor(max_workers=int(lock["worker_count"])) as executor:
            iterator = executor.map(lambda frame: _phase_a_worker(frame, fit_candidate_root, eval_candidate_root), frames)
            for index, (frame, bundle) in enumerate(iterator, start=1):
                writer.write_json_gzip(bundle_relative(frame), bundle)
                bundle_hashes.append(bundle["content_sha256"])
                selected_counts[bundle["selected_support_boundary_owner"]] += 1
                guard()
                if index % 10 == 0 or index == len(frames):
                    _emit({"phase": "FORMATION_PHASE_A", "completed_frames": index, "total_frames": len(frames), "selected_owner_counts": dict(sorted(selected_counts.items()))})
        phase_a = _seal(
            {
                "schema": PHASE_A_SCHEMA,
                "parent_count": 24,
                "frame_count": 450,
                "query_slot_count": 4050,
                "bundle_hash_sequence_sha256": adapter.canonical_sha256(bundle_hashes),
                "selected_owner_counts": dict(sorted(selected_counts.items())),
                "source_receipt_hash_sequence_sha256": preflight["source_receipt_hash_sequence_sha256"],
                "candidate_record_hash_sequence_sha256": preflight["candidate_record_hash_sequence_sha256"],
                "apple_depth_reads": 450,
                "confidence_reads": 450,
                "faro_reads": 0,
                "task_metric_reads": 0,
                "prior_outcome_reads": 0,
                "r6_untouched_parent_reads": 0,
                "all_bundles_sealed_before_faro": True,
                "training_steps": 0,
                "network_requests": 0,
            }
        )
        writer.write_json("phase-a-completion.json", phase_a)
        reloaded_phase_a = _validate_seal(json.loads((output_root / "phase-a-completion.json").read_text(encoding="utf-8")), PHASE_A_SCHEMA)
        require(reloaded_phase_a["content_sha256"] == phase_a["content_sha256"] and reloaded_phase_a["faro_reads"] == 0, "FORMATION_PHASE_A_RELOAD_DRIFT", "formation Phase-A completion changed before FARO")
        all_records: list[dict[str, Any]] = []
        faro_unobservable_frames: set[str] = set()
        with ThreadPoolExecutor(max_workers=int(lock["worker_count"])) as executor:
            iterator = executor.map(lambda frame: _phase_b_worker(frame, output_root, fit_candidate_root, eval_candidate_root), frames)
            for index, (frame, records) in enumerate(iterator, start=1):
                writer.write_json_gzip(score_relative(frame), records)
                all_records.extend(records)
                if all(record["truth_status"]["evaluable"] is not True for record in records):
                    faro_unobservable_frames.add(frame.physical_frame_id)
                guard()
                if index % 10 == 0 or index == len(frames):
                    _emit({"phase": "FORMATION_PHASE_B", "completed_frames": index, "total_frames": len(frames), "query_records": len(all_records), "faro_unobservable_frames": len(faro_unobservable_frames)})
        summary = formation_replay.summarize(all_records)
        writer.write_json("summary.json", summary)
        result = {
            "schema": RESULT_SCHEMA,
            "terminal": summary["terminal"],
            "execution_valid": True,
            "scientific_pass_fail_assigned": False,
            "execution_lock_sha256": lock_sha,
            "phase_a_completion_sha256": phase_a["content_sha256"],
            "summary_sha256": summary["content_sha256"],
            "parent_count": 24,
            "frame_count": 450,
            "query_record_count": 4050,
            "faro_support_unobservable_frame_count": len(faro_unobservable_frames),
            "selected_owner_counts": dict(sorted(selected_counts.items())),
            "candidate_inference_count": 0,
            "training_steps": 0,
            "network_requests": 0,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "promotion_authorized": False,
            "claim_ceiling": lock["claim_ceiling"],
        }
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": MANIFEST_SCHEMA, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written, "terminal": result["terminal"]})
        return result
    except Exception as error:
        if writer.activated:
            return _write_failure(writer, error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        _emit({"status": "FORMATION_REPLAY_NOT_STARTED", "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)})
        return 2
    _emit({"terminal": result["terminal"], "execution_valid": result.get("execution_valid", False), "scientific_pass_fail_assigned": result.get("scientific_pass_fail_assigned")})
    return 0 if result.get("execution_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
