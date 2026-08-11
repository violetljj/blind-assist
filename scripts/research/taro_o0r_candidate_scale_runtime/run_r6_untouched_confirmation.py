#!/usr/bin/env python3
"""Run the one-shot TARO R6 untouched-parent factor-split confirmation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import zipfile
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation as r6
from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation_io as r6io
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


RESULT_SCHEMA = "blindassist.taro.o0r.r6_untouched_confirmation_result.v1"
MANIFEST_SCHEMA = "blindassist.taro.o0r.r6_untouched_confirmation_manifest.v1"
FAIL_TERMINAL = "TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_CONFIRMATION_EXECUTION_INVALID"


class R6ConfirmationRunError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R6ConfirmationRunError(code, message, **context)


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def _groups(frames: Sequence[r6io.R6FrameRef]):
    return groupby(frames, key=lambda row: (row.parent_id, row.video_id, row.upsampling_archive, row.intrinsics_archive))


def _phase_a_frame(frame: r6io.R6FrameRef, up_bundle: zipfile.ZipFile, intr_bundle: zipfile.ZipFile, read_counts: Counter[str]) -> dict[str, Any]:
    def observed(role: str, _: str) -> None:
        read_counts[role] += 1

    return r6io.read_phase_a_frame(frame, up_bundle, intr_bundle, observer=observed)


def _candidate_phase(
    frames: Sequence[r6io.R6FrameRef],
    *,
    writer: FactorEvidenceWriter,
    model: Any,
    runtime_identity: Mapping[str, Any],
    guard: Callable[[], None],
    read_counts: Counter[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for _, grouped in _groups(frames):
        parent_frames = list(grouped)
        with zipfile.ZipFile(parent_frames[0].upsampling_archive) as up_bundle, zipfile.ZipFile(parent_frames[0].intrinsics_archive) as intr_bundle:
            for frame in parent_frames:
                loaded = _phase_a_frame(frame, up_bundle, intr_bundle, read_counts)
                source = loaded["source_receipt"]
                writer.write_json(r6io.source_receipt_relative(frame), source)
                candidate_input = r6.build_candidate_input(source, loaded["color_rgb_u8"])
                inference = r6.infer_candidate(model, candidate_input, loaded["color_rgb_u8"], runtime_identity, device="cuda")
                native = inference["native_depth_m"]
                payload = depthart_runner.deterministic_npy_gzip_bytes(native)
                blob_receipt = writer.write_bytes(r6io.candidate_blob_relative(frame), payload)
                blob = {**blob_receipt, "array_sha256": adapter.canonical_sha256(native), "shape_hw": list(depthart_runner.NATIVE_SHAPE_HW), "dtype": "float32", "encoding": "DETERMINISTIC_GZIP_NPY_MTIME_0"}
                record = r6.build_candidate_frame(candidate_input, inference["inference_receipt"], blob)
                writer.write_json(r6io.candidate_record_relative(frame), record)
                records.append(record)
                guard()
                _emit({"phase": "R6_CANDIDATE", "completed": len(records), "total": len(frames), "physical_frame_id": frame.physical_frame_id})
    completion = r6.build_candidate_completion(records, r6io.expected_keys(frames))
    writer.write_json("candidate-completion.json", completion)
    return completion


def _decision_phase(
    frames: Sequence[r6io.R6FrameRef],
    *,
    root: Path,
    writer: FactorEvidenceWriter,
    candidate_completion: Mapping[str, Any],
    guard: Callable[[], None],
    read_counts: Counter[str],
) -> dict[str, Any]:
    r6.validate_candidate_completion(dict(candidate_completion))
    decisions: list[dict[str, Any]] = []
    for _, grouped in _groups(frames):
        parent_frames = list(grouped)
        with zipfile.ZipFile(parent_frames[0].upsampling_archive) as up_bundle, zipfile.ZipFile(parent_frames[0].intrinsics_archive) as intr_bundle:
            for frame in parent_frames:
                loaded = _phase_a_frame(frame, up_bundle, intr_bundle, read_counts)
                source = r6io.load_source_receipt(root, frame)
                require(loaded["source_receipt"]["content_sha256"] == source["content_sha256"], "R6_PHASE_A_SOURCE_REPLAY_DRIFT", "R6 Phase-A source replay changed")
                candidate, native = r6io.load_candidate_frame(root, frame)
                decision = r6.build_source_decision(source, candidate, native, loaded["apple_depth_mm"], loaded["confidence"])
                writer.write_json(r6io.source_decision_relative(frame), decision)
                decisions.append(decision)
                guard()
                _emit({"phase": "R6_SOURCE_DECISION", "completed": len(decisions), "total": len(frames), "physical_frame_id": frame.physical_frame_id, "selected_branch": decision["selected_branch"]})
    phase_reads = {
        "COLOR": int(read_counts.get("color", 0)), "APPLE_DEPTH": int(read_counts.get("lowres_depth", 0)),
        "CONFIDENCE": int(read_counts.get("confidence", 0)), "INTRINSICS": int(read_counts.get("intrinsics", 0)),
        "TRAJECTORY": int(read_counts.get("trajectory", 0)), "FARO": int(read_counts.get("highres_depth", 0)),
        "QUERY_TRUTH": 0, "TASK_METRIC": 0, "PRIOR_OUTCOME": 0,
    }
    completion = r6.build_phase_a_completion(candidate_completion, decisions, r6io.expected_keys(frames), read_counts=phase_reads)
    writer.write_json("phase-a-completion.json", completion)
    return completion


def _verify_phase_a(root: Path, frames: Sequence[r6io.R6FrameRef], expected_candidate: Mapping[str, Any], expected_phase_a: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = r6.validate_candidate_completion(json.loads(materializer.safe_join(root, "candidate-completion.json").read_text(encoding="utf-8")))
    phase_a = r6.validate_phase_a_completion(json.loads(materializer.safe_join(root, "phase-a-completion.json").read_text(encoding="utf-8")))
    require(candidate["content_sha256"] == expected_candidate["content_sha256"] and phase_a["content_sha256"] == expected_phase_a["content_sha256"] and phase_a["candidate_completion_sha256"] == candidate["content_sha256"], "R6_PHASE_A_COMPLETION_RELOAD_DRIFT", "R6 Phase-A completion changed before FARO")
    decisions = [r6io.load_source_decision(root, frame) for frame in frames]
    require(adapter.canonical_sha256([row["content_sha256"] for row in decisions]) == phase_a["decision_hash_sequence_sha256"], "R6_PHASE_A_DECISION_RELOAD_DRIFT", "R6 decision sequence changed before FARO")
    return phase_a, decisions


def _phase_b(
    frames: Sequence[r6io.R6FrameRef],
    *,
    root: Path,
    writer: FactorEvidenceWriter,
    phase_a: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    guard: Callable[[], None],
    read_counts: Counter[str],
) -> tuple[dict[str, Any], Counter[str]]:
    completion = r6.validate_phase_a_completion(dict(phase_a))
    decision_by_frame = {row["physical_frame_id"]: r6.validate_source_decision(dict(row)) for row in decisions}
    records: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    unobservable: Counter[str] = Counter()

    def observed(role: str, _: str) -> None:
        require(role == "highres_depth", "R6_PHASE_B_SOURCE_ROLE_INVALID", "R6 Phase B attempted non-FARO source read", role=role)
        read_counts[role] += 1

    for _, grouped in _groups(frames):
        parent_frames = list(grouped)
        with zipfile.ZipFile(parent_frames[0].upsampling_archive) as up_bundle:
            for frame in parent_frames:
                source = r6io.load_source_receipt(root, frame)
                candidate, native = r6io.load_candidate_frame(root, frame)
                decision = decision_by_frame[frame.physical_frame_id]
                faro, member_binding = r6io.read_faro_frame(frame, up_bundle, observer=observed)
                truth = r6.build_truth_binding(source, completion, member_binding=member_binding, highres_depth_mm=faro)
                writer.write_json(r6io.truth_binding_relative(frame), truth)
                try:
                    geometry = r6.derive_faro_geometry(faro, source, decision, completion, truth)
                except r6.R6ConfirmationError as error:
                    if error.code not in adapter._SUPPORT_UNOBSERVABLE_CODES:
                        raise
                    frame_records = r6.evaluate_unobservable_faro_frame(source, candidate, native, decision, completion, truth, faro, error.code)
                    unobservable[error.code] += 1
                else:
                    frame_records = r6.evaluate_frame(source, candidate, native, decision, geometry)
                require(len(frame_records) == 9, "R6_PHASE_B_QUERY_COUNT_DRIFT", "R6 Phase B did not retain nine query slots")
                writer.write_json_gzip(r6io.query_pairs_relative(frame), [{"truth_scoring_record": truth_row, "factor_components": components, "composite_query": composite} for truth_row, components, composite in frame_records])
                records.extend(frame_records)
                guard()
                _emit({"phase": "R6_PHASE_B", "completed_frames": len(records) // 9, "total_frames": len(frames), "query_records": len(records), "physical_frame_id": frame.physical_frame_id, "support_unobservable_frames": sum(unobservable.values())})
    require(read_counts.get("highres_depth", 0) == len(frames), "R6_PHASE_B_FARO_READ_COUNT_DRIFT", "R6 Phase B FARO read count drift")
    summary = r6.summarize(records)
    writer.write_json("summary.json", summary)
    return summary, unobservable


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> dict[str, Any]:
    failure = {"schema": RESULT_SCHEMA, "terminal": FAIL_TERMINAL, "passed": False, "execution_valid": False, "scientific_status": "EXECUTION_INVALID_NO_SCIENTIFIC_INTERPRETATION", "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "one_shot_consumed": True}
    try:
        writer.write_json("failure.json", failure)
        writer.write_json("manifest.json", {"schema": MANIFEST_SCHEMA, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written, "evidence_root_consumed": True})
    except Exception:
        pass
    return failure


def execute_confirmation(
    execution_lock_path: Path,
    *,
    lock_validator: Callable[[Path], dict[str, Any]] | None = None,
    model_loader: Callable[..., tuple[Any, dict[str, Any]]] = depthart_runner.load_official_depthart,
) -> dict[str, Any]:
    if lock_validator is None:
        from scripts.research.taro_o0r_candidate_scale_runtime.validate_r6_confirmation_execution_lock import validate_execution_lock
        lock_validator = validate_execution_lock
    lock = lock_validator(execution_lock_path.resolve())
    repo_root = Path(lock["roots"]["repo_root"]).resolve()
    inventory_path = Path(lock["roots"]["inventory_path"]).resolve()
    evidence_root = Path(lock["roots"]["evidence_root"]).resolve()
    require(repo_root.is_dir() and inventory_path.is_file() and not evidence_root.exists(), "R6_ROOT_PREFLIGHT_INVALID", "R6 roots do not satisfy one-shot preflight")
    frames = r6io.load_exact_cohort(inventory_path, repo_root)
    require(len(frames) == r6.EXPECTED_FRAME_COUNT, "R6_COHORT_PREFLIGHT_INVALID", "R6 exact cohort differs from frozen 120 frames")
    budget = lock["resource_budget"]
    started = time.monotonic()
    process = psutil.Process()
    writer = FactorEvidenceWriter(evidence_root, int(budget["maximum_evidence_bytes"]))

    def guard() -> None:
        require(time.monotonic() - started <= float(budget["maximum_wall_seconds"]), "R6_EXECUTION_TIMEOUT", "R6 execution exceeded wall-time budget")
        require(process.memory_info().rss <= int(budget["maximum_peak_rss_bytes"]), "R6_EXECUTION_RSS_EXCEEDED", "R6 execution exceeded RSS budget")

    try:
        model, runtime_identity = model_loader(Path(lock["candidate_identity"]["source_root"]), Path(lock["candidate_identity"]["checkpoint_path"]), device="cuda", seed=0)
        guard()
        lock_sha = materializer.sha256_file(execution_lock_path.resolve())
        writer.activate({"schema": "blindassist.taro.o0r.r6_untouched_confirmation_start.v1", "execution_lock_sha256": lock_sha, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "argv": lock["unique_argv"], "one_shot_consumed_on_root_creation": True, "prior_outcome_roots_enumerated": False})
        read_counts: Counter[str] = Counter()
        candidate = _candidate_phase(frames, writer=writer, model=model, runtime_identity=runtime_identity, guard=guard, read_counts=read_counts)
        try:
            import torch
            cuda_peak = int(torch.cuda.max_memory_allocated())
            del model
            torch.cuda.empty_cache()
        except Exception:
            cuda_peak = 0
        require(cuda_peak <= int(budget["maximum_cuda_allocated_bytes"]), "R6_CUDA_BUDGET_EXCEEDED", "R6 candidate phase exceeded CUDA budget")
        phase_a = _decision_phase(frames, root=evidence_root, writer=writer, candidate_completion=candidate, guard=guard, read_counts=read_counts)
        phase_a, decisions = _verify_phase_a(evidence_root, frames, candidate, phase_a)
        require(read_counts.get("highres_depth", 0) == 0, "R6_PHASE_A_FARO_READ_FIREWALL_BREACH", "FARO was read before Phase-A completion reload")
        summary, unobservable = _phase_b(frames, root=evidence_root, writer=writer, phase_a=phase_a, decisions=decisions, guard=guard, read_counts=read_counts)
        result = {
            "schema": RESULT_SCHEMA, "terminal": summary["terminal"], "passed": bool(summary["passed"]), "execution_valid": True, "scientific_status": summary["terminal"],
            "execution_lock_sha256": lock_sha, "candidate_completion_sha256": candidate["content_sha256"], "phase_a_completion_sha256": phase_a["content_sha256"], "summary_sha256": summary["content_sha256"],
            "runtime_identity_sha256": adapter.canonical_sha256(runtime_identity), "candidate_inference_count": r6.EXPECTED_FRAME_COUNT, "phase_b_faro_frame_count": r6.EXPECTED_FRAME_COUNT,
            "query_record_count": r6.EXPECTED_QUERY_COUNT, "faro_support_unobservable_frame_count": sum(unobservable.values()), "faro_support_unobservable_reason_counts": dict(sorted(unobservable.items())),
            "training_steps": 0, "network_requests": 0, "cuda_peak_allocated_bytes": cuda_peak, "elapsed_seconds": round(time.monotonic() - started, 6),
            "claim_ceiling": "WILD_LAB untouched-parent confirmation of the frozen R6 factor compositor; not deployment, product, or safety evidence.",
        }
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": MANIFEST_SCHEMA, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written, "evidence_root_consumed": True, "terminal": result["terminal"]})
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
        result = execute_confirmation(args.execution_lock)
    except Exception as error:
        print(json.dumps({"status": "EXECUTION_NOT_STARTED", "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    _emit({"terminal": result["terminal"], "passed": result["passed"], "execution_valid": result.get("execution_valid", False)})
    return 0 if result.get("execution_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
