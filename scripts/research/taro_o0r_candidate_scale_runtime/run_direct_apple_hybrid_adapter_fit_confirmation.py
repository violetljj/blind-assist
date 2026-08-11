#!/usr/bin/env python3
"""Run the one-shot TARO R5 parent-disjoint task-metric confirmation."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import time
import zipfile
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation_io as r5io
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


RESULT_SCHEMA = "blindassist.taro.o0r.r5_confirmation_result.v1"
MANIFEST_SCHEMA = "blindassist.taro.o0r.r5_confirmation_manifest.v1"
FAIL_TERMINAL = "TARO_O0R_DIRECT_APPLE_HYBRID_R5_EXECUTION_INVALID"


class R5ConfirmationRunError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R5ConfirmationRunError(code, message, **context)


def _failure_code(error: Exception) -> str:
    return str(getattr(error, "code", type(error).__name__))


def _emit_progress(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def _group_frames(frames: Sequence[r5io.R5FrameRef]):
    return groupby(frames, key=lambda frame: (frame.parent_id, frame.video_id, frame.upsampling_archive))


def _candidate_phase(
    frames: Sequence[r5io.R5FrameRef],
    *,
    writer: FactorEvidenceWriter,
    model: Any,
    runtime_identity: Mapping[str, Any],
    guard: Callable[[], None],
    read_counts: Counter[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []

    def observed(role: str, _: str) -> None:
        read_counts[role] += 1

    for _, grouped in _group_frames(frames):
        parent_frames = list(grouped)
        with zipfile.ZipFile(parent_frames[0].upsampling_archive) as bundle:
            for frame in parent_frames:
                color = r5io.read_bound_payload(frame, bundle, "color", read_observer=observed)
                candidate_input = r5.build_candidate_input(frame.source_frame_receipt, color)
                inference = r5.infer_candidate(
                    model,
                    candidate_input,
                    color,
                    runtime_identity,
                    device="cuda",
                )
                native = inference["native_depth_m"]
                blob_payload = depthart_runner.deterministic_npy_gzip_bytes(native)
                blob_receipt = writer.write_bytes(r5io.candidate_blob_relative(frame), blob_payload)
                blob = {
                    **blob_receipt,
                    "array_sha256": adapter.canonical_sha256(native),
                    "shape_hw": list(depthart_runner.NATIVE_SHAPE_HW),
                    "dtype": "float32",
                    "encoding": "DETERMINISTIC_GZIP_NPY_MTIME_0",
                }
                record = r5.build_candidate_frame_record(candidate_input, inference["inference_receipt"], blob)
                writer.write_json(r5io.candidate_record_relative(frame), record)
                records.append(record)
                guard()
                _emit_progress({"phase": "R5_CANDIDATE", "completed": len(records), "total": len(frames), "physical_frame_id": frame.physical_frame_id})
    completion = r5.build_candidate_phase_completion(records, r5io.expected_keys(frames))
    writer.write_json("candidate-phase-completion.json", completion)
    return completion, records


def _source_decision_phase(
    frames: Sequence[r5io.R5FrameRef],
    *,
    root: Path,
    writer: FactorEvidenceWriter,
    candidate_completion: Mapping[str, Any],
    guard: Callable[[], None],
    read_counts: Counter[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    r5.validate_candidate_phase_completion(candidate_completion)
    decisions: list[dict[str, Any]] = []

    def observed(role: str, _: str) -> None:
        read_counts[role] += 1

    for _, grouped in _group_frames(frames):
        parent_frames = list(grouped)
        with zipfile.ZipFile(parent_frames[0].upsampling_archive) as bundle:
            for frame in parent_frames:
                candidate, native = r5io.load_candidate_frame(root, frame)
                apple = r5io.read_bound_payload(frame, bundle, "lowres_depth", read_observer=observed)
                confidence = r5io.read_bound_payload(frame, bundle, "confidence", read_observer=observed)
                decision = r5.build_source_decision(frame.source_frame_receipt, candidate, native, apple, confidence)
                writer.write_json(r5io.source_decision_relative(frame), decision)
                decisions.append(decision)
                guard()
                _emit_progress({"phase": "R5_SOURCE_DECISION", "completed": len(decisions), "total": len(frames), "physical_frame_id": frame.physical_frame_id, "selected_branch": decision["selected_branch"]})
    phase_reads = {
        "COLOR": int(read_counts.get("color", 0)),
        "APPLE_DEPTH": int(read_counts.get("lowres_depth", 0)),
        "CONFIDENCE": int(read_counts.get("confidence", 0)),
        "FARO": int(read_counts.get("highres_depth", 0)),
        "QUERY_TRUTH": 0,
        "COMPACT_TRUTH": 0,
        "TASK_METRIC": 0,
        "PRIOR_EVAL_OUTCOME": 0,
    }
    completion = r5.build_phase_a_completion(
        candidate_completion,
        decisions,
        r5io.expected_keys(frames),
        read_counts=phase_reads,
    )
    writer.write_json("phase-a-completion.json", completion)
    return completion, decisions


def _verify_phase_a_from_disk(
    frames: Sequence[r5io.R5FrameRef],
    root: Path,
    expected_candidate_completion: Mapping[str, Any],
    expected_phase_a_completion: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_path = materializer.safe_join(root, "candidate-phase-completion.json")
    phase_a_path = materializer.safe_join(root, "phase-a-completion.json")
    candidate = r5.validate_candidate_phase_completion(json.loads(candidate_path.read_text(encoding="utf-8")))
    phase_a = r5.validate_phase_a_completion(json.loads(phase_a_path.read_text(encoding="utf-8")))
    require(candidate["content_sha256"] == expected_candidate_completion["content_sha256"], "R5_CANDIDATE_COMPLETION_RELOAD_DRIFT", "candidate completion changed before Phase B")
    require(phase_a["content_sha256"] == expected_phase_a_completion["content_sha256"] and phase_a["candidate_phase_completion_sha256"] == candidate["content_sha256"], "R5_PHASE_A_COMPLETION_RELOAD_DRIFT", "Phase-A completion changed before Phase B")
    decisions = [r5io.load_source_decision(root, frame) for frame in frames]
    keys = [(row["parent_id"], row["video_id"], row["timestamp_token"]) for row in decisions]
    require(keys == r5io.expected_keys(frames), "R5_PHASE_A_DECISION_RELOAD_DRIFT", "reloaded Phase-A decisions changed order/identity")
    require(adapter.canonical_sha256([row["content_sha256"] for row in decisions]) == phase_a["source_decision_hash_sequence_sha256"], "R5_PHASE_A_DECISION_RELOAD_DRIFT", "reloaded Phase-A decision hash sequence drift")
    return phase_a, decisions


def _phase_b(
    frames: Sequence[r5io.R5FrameRef],
    *,
    root: Path,
    writer: FactorEvidenceWriter,
    phase_a_completion: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    guard: Callable[[], None],
    read_counts: Counter[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    phase_a = r5.validate_phase_a_completion(phase_a_completion)
    require(len(decisions) == len(frames), "R5_PHASE_B_DECISION_COUNT_DRIFT", "Phase B decision count drift")
    all_records: list[dict[str, Any]] = []
    decision_by_frame = {str(row["physical_frame_id"]): r5.validate_source_decision(dict(row)) for row in decisions}

    def observed(role: str, _: str) -> None:
        read_counts[role] += 1

    for _, grouped in _group_frames(frames):
        parent_frames = list(grouped)
        with zipfile.ZipFile(parent_frames[0].upsampling_archive) as bundle:
            for frame in parent_frames:
                decision = decision_by_frame[frame.physical_frame_id]
                candidate, native = r5io.load_candidate_frame(root, frame)
                faro_depth = r5io.read_bound_payload(frame, bundle, "highres_depth", read_observer=observed)
                geometry = r5.derive_faro_geometry(faro_depth, frame.source_frame_receipt, decision, phase_a)
                records = r5.evaluate_frame(frame.source_frame_receipt, candidate, native, decision, geometry)
                require(len(records) == 9, "R5_PHASE_B_QUERY_COUNT_DRIFT", "Phase B did not retain nine query slots")
                writer.write_json_gzip(r5io.query_records_relative(frame), records)
                all_records.extend(records)
                guard()
                _emit_progress({"phase": "R5_PHASE_B", "completed_frames": len(all_records) // 9, "total_frames": len(frames), "query_records": len(all_records), "physical_frame_id": frame.physical_frame_id})
    require(read_counts.get("highres_depth", 0) == len(frames), "R5_PHASE_B_FARO_READ_COUNT_DRIFT", "Phase B FARO payload count drift")
    summary = r5.summarize(all_records)
    writer.write_json("summary.json", summary)
    return summary, all_records


def _result(summary: Mapping[str, Any], *, lock_sha256: str, phase_a: Mapping[str, Any], candidate_completion: Mapping[str, Any], runtime_identity: Mapping[str, Any], cuda_peak_bytes: int, elapsed_seconds: float) -> dict[str, Any]:
    terminal = str(summary["terminal"])
    return {
        "schema": RESULT_SCHEMA,
        "terminal": terminal,
        "passed": terminal == "TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_PASS",
        "execution_valid": True,
        "scientific_status": terminal,
        "execution_lock_sha256": lock_sha256,
        "candidate_phase_completion_sha256": candidate_completion["content_sha256"],
        "phase_a_completion_sha256": phase_a["content_sha256"],
        "summary_sha256": summary["content_sha256"],
        "runtime_identity_sha256": adapter.canonical_sha256(runtime_identity),
        "candidate_inference_count": r5.EXPECTED_FRAME_COUNT,
        "phase_b_faro_frame_count": r5.EXPECTED_FRAME_COUNT,
        "query_record_count": r5.EXPECTED_QUERY_COUNT,
        "training_steps": 0,
        "network_requests": 0,
        "cuda_peak_allocated_bytes": int(cuda_peak_bytes),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "claim_ceiling": r5.CLAIM_CEILING,
    }


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> dict[str, Any]:
    failure = {
        "schema": RESULT_SCHEMA,
        "terminal": FAIL_TERMINAL,
        "passed": False,
        "execution_valid": False,
        "scientific_status": "EXECUTION_INVALID_NO_SCIENTIFIC_INTERPRETATION",
        "failure_code": _failure_code(error),
        "message": str(error),
        "one_shot_consumed": True,
    }
    try:
        writer.write_json("failure.json", failure)
        writer.write_json(
            "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "files": {key: value for key, value in sorted(writer.file_receipts.items())},
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
                "evidence_root_consumed": True,
            },
        )
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
        from scripts.research.taro_o0r_candidate_scale_runtime.validate_r5_execution_lock import validate_execution_lock

        lock_validator = validate_execution_lock
    lock = lock_validator(execution_lock_path.resolve())
    roots = lock["roots"]
    source_root = Path(roots["source_root"]).resolve()
    r3_root = Path(roots["r3_evidence_root"]).resolve()
    evidence_root = Path(roots["r5_evidence_root"]).resolve()
    require(source_root.is_dir() and r3_root.is_dir() and not evidence_root.exists(), "R5_ROOT_PREFLIGHT_INVALID", "R5 source/R3 roots must exist and R5 root must be absent")
    frames = r5io.load_exact_cohort(
        Path(lock["frame_plan_path"]).resolve(),
        r3_root,
        source_root,
        verify_containers=True,
    )
    require(len(frames) == r5.EXPECTED_FRAME_COUNT, "R5_COHORT_PREFLIGHT_INVALID", "R5 preflight frame count drift")
    budget = lock["resource_budget"]
    started = time.monotonic()
    process = psutil.Process()
    writer = FactorEvidenceWriter(evidence_root, int(budget["maximum_evidence_bytes"]))

    def guard() -> None:
        require(time.monotonic() - started <= float(budget["maximum_wall_seconds"]), "R5_EXECUTION_TIMEOUT", "R5 execution exceeded wall-time budget")
        require(process.memory_info().rss <= int(budget["maximum_peak_rss_bytes"]), "R5_EXECUTION_RSS_EXCEEDED", "R5 execution exceeded RSS budget")

    try:
        model, runtime_identity = model_loader(Path(lock["depthart_source_root"]), Path(lock["checkpoint_path"]), device="cuda", seed=0)
        guard()
        lock_sha = materializer.sha256_file(execution_lock_path.resolve())
        writer.activate(
            {
                "schema": "blindassist.taro.o0r.r5_confirmation_execution_start.v1",
                "execution_lock_sha256": lock_sha,
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "argv": lock["unique_argv"],
                "verified_binding_sha256s": {role: value["sha256"] for role, value in sorted(lock["_verified_bindings"].items())},
                "one_shot_consumed_on_root_creation": True,
                "prior_eval_truth_or_outcome_roots_enumerated": False,
            }
        )
        read_counts: Counter[str] = Counter()
        candidate_completion, _ = _candidate_phase(
            frames,
            writer=writer,
            model=model,
            runtime_identity=runtime_identity,
            guard=guard,
            read_counts=read_counts,
        )
        try:
            import torch

            cuda_peak = int(torch.cuda.max_memory_allocated())
            del model
            torch.cuda.empty_cache()
        except Exception:
            cuda_peak = 0
        require(cuda_peak <= int(budget["maximum_cuda_allocated_bytes"]), "R5_EXECUTION_CUDA_BUDGET_EXCEEDED", "R5 candidate inference exceeded CUDA budget")
        phase_a, _ = _source_decision_phase(
            frames,
            root=evidence_root,
            writer=writer,
            candidate_completion=candidate_completion,
            guard=guard,
            read_counts=read_counts,
        )
        phase_a, decisions = _verify_phase_a_from_disk(frames, evidence_root, candidate_completion, phase_a)
        require(read_counts.get("highres_depth", 0) == 0, "R5_PHASE_A_FARO_READ_FIREWALL_BREACH", "FARO was read before Phase-A completion reload")
        summary, _ = _phase_b(
            frames,
            root=evidence_root,
            writer=writer,
            phase_a_completion=phase_a,
            decisions=decisions,
            guard=guard,
            read_counts=read_counts,
        )
        result = _result(
            summary,
            lock_sha256=lock_sha,
            phase_a=phase_a,
            candidate_completion=candidate_completion,
            runtime_identity=runtime_identity,
            cuda_peak_bytes=cuda_peak,
            elapsed_seconds=time.monotonic() - started,
        )
        writer.write_json("result.json", result)
        writer.write_json(
            "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "files": {key: value for key, value in sorted(writer.file_receipts.items())},
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
                "evidence_root_consumed": True,
                "terminal": result["terminal"],
            },
        )
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
        print(json.dumps({"status": "EXECUTION_NOT_STARTED", "error_code": _failure_code(error), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "execution_valid": result.get("execution_valid", False)}, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("execution_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
