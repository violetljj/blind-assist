#!/usr/bin/env python3
"""Resume TARO R5 Phase B from the sealed R5 Phase-A evidence.

R2 is a narrow execution repair: frozen support-plane unobservability becomes
nine retained UNKNOWN query slots.  It performs no inference, source decision,
training, branch reselection, threshold change, or gate change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation_io as r5io
from scripts.research.taro_o0r_candidate_scale_runtime import run_direct_apple_hybrid_adapter_fit_confirmation as r5run
from scripts.research.taro_o0r_candidate_scale_runtime.validate_r5_phase_b_repair_execution_lock import validate_execution_lock
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


RESULT_SCHEMA = "blindassist.taro.o0r.r5_phase_b_repair_result.v1"
MANIFEST_SCHEMA = "blindassist.taro.o0r.r5_phase_b_repair_manifest.v1"
FAIL_TERMINAL = "TARO_O0R_DIRECT_APPLE_HYBRID_R5_R2_EXECUTION_INVALID"


class R5PhaseBRepairError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R5PhaseBRepairError(code, message, **context)


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def _verify_predecessor(root: Path, binding: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = materializer.safe_join(root, "manifest.json")
    failure_path = materializer.safe_join(root, "failure.json")
    candidate_path = materializer.safe_join(root, "candidate-phase-completion.json")
    phase_a_path = materializer.safe_join(root, "phase-a-completion.json")
    for path, key in (
        (manifest_path, "manifest_sha256"),
        (failure_path, "failure_sha256"),
        (candidate_path, "candidate_phase_completion_sha256"),
        (phase_a_path, "phase_a_completion_sha256"),
    ):
        require(path.is_file() and materializer.sha256_file(path) == binding[key], "R5_R2_PREDECESSOR_BINDING_DRIFT", "R5 predecessor binding differs", path=str(path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    require(
        failure.get("terminal") == "TARO_O0R_DIRECT_APPLE_HYBRID_R5_EXECUTION_INVALID"
        and failure.get("failure_code") == "SUPPORT_PLAUSIBLE_INSUFFICIENT"
        and failure.get("one_shot_consumed") is True,
        "R5_R2_PREDECESSOR_TERMINAL_DRIFT",
        "R5 predecessor is not the exact consumed support-unobservable failure",
    )
    files = manifest.get("files")
    require(
        isinstance(files, dict)
        and manifest.get("file_count_before_manifest") == len(files)
        and manifest.get("bytes_before_manifest") == sum(int(row["bytes"]) for row in files.values()),
        "R5_R2_PREDECESSOR_MANIFEST_DRIFT",
        "R5 predecessor manifest accounting differs",
    )
    for relative, receipt in files.items():
        path = materializer.safe_join(root, relative)
        require(
            path.is_file()
            and path.stat().st_size == int(receipt["bytes"])
            and materializer.sha256_file(path) == receipt["sha256"],
            "R5_R2_PREDECESSOR_FILE_DRIFT",
            "R5 predecessor file differs from its manifest",
            path=relative,
        )
    candidate = r5.validate_candidate_phase_completion(json.loads(candidate_path.read_text(encoding="utf-8")))
    phase_a = r5.validate_phase_a_completion(json.loads(phase_a_path.read_text(encoding="utf-8")))
    require(phase_a["candidate_phase_completion_sha256"] == candidate["content_sha256"], "R5_R2_PREDECESSOR_LINEAGE_DRIFT", "R5 predecessor Phase-A/candidate lineage differs")
    return candidate, phase_a, manifest


def _load_decisions(frames: Sequence[r5io.R5FrameRef], root: Path, phase_a: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = [r5io.load_source_decision(root, frame) for frame in frames]
    require(
        [(row["parent_id"], row["video_id"], row["timestamp_token"]) for row in decisions] == r5io.expected_keys(frames)
        and adapter.canonical_sha256([row["content_sha256"] for row in decisions]) == phase_a["source_decision_hash_sequence_sha256"],
        "R5_R2_DECISION_LINEAGE_DRIFT",
        "sealed Phase-A decisions differ from the completion receipt",
    )
    return decisions


def _phase_b(
    frames: Sequence[r5io.R5FrameRef],
    *,
    predecessor_root: Path,
    writer: FactorEvidenceWriter,
    phase_a: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    guard: Any,
) -> tuple[dict[str, Any], Counter[str]]:
    all_records: list[dict[str, Any]] = []
    unobservable: Counter[str] = Counter()
    decision_by_frame = {str(row["physical_frame_id"]): r5.validate_source_decision(dict(row)) for row in decisions}
    faro_reads = 0

    def observed(role: str, _: str) -> None:
        nonlocal faro_reads
        require(role == "highres_depth", "R5_R2_FORBIDDEN_SOURCE_READ", "R2 Phase B attempted a non-FARO source read", role=role)
        faro_reads += 1

    for _, grouped in r5run._group_frames(frames):
        parent_frames = list(grouped)
        with zipfile.ZipFile(parent_frames[0].upsampling_archive) as bundle:
            for frame in parent_frames:
                decision = decision_by_frame[frame.physical_frame_id]
                candidate, native = r5io.load_candidate_frame(predecessor_root, frame)
                faro_depth = r5io.read_bound_payload(frame, bundle, "highres_depth", read_observer=observed)
                try:
                    geometry = r5.derive_faro_geometry(faro_depth, frame.source_frame_receipt, decision, phase_a)
                except r5.R5ConfirmationError as error:
                    if error.code not in adapter._SUPPORT_UNOBSERVABLE_CODES:
                        raise
                    records = r5.evaluate_unobservable_faro_frame(
                        frame.source_frame_receipt,
                        candidate,
                        native,
                        decision,
                        phase_a,
                        faro_depth,
                        error.code,
                    )
                    unobservable[error.code] += 1
                else:
                    records = r5.evaluate_frame(frame.source_frame_receipt, candidate, native, decision, geometry)
                require(len(records) == 9, "R5_R2_QUERY_COUNT_DRIFT", "R2 Phase B did not retain nine query slots")
                writer.write_json_gzip(r5io.query_records_relative(frame), records)
                all_records.extend(records)
                guard()
                _emit(
                    {
                        "phase": "R5_R2_PHASE_B",
                        "completed_frames": len(all_records) // 9,
                        "total_frames": len(frames),
                        "query_records": len(all_records),
                        "physical_frame_id": frame.physical_frame_id,
                        "support_unobservable_frames": sum(unobservable.values()),
                    }
                )
    require(faro_reads == len(frames), "R5_R2_FARO_READ_COUNT_DRIFT", "R2 Phase B FARO read count differs")
    summary = r5.summarize(all_records)
    writer.write_json("summary.json", summary)
    return summary, unobservable


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> dict[str, Any]:
    failure = {
        "schema": RESULT_SCHEMA,
        "terminal": FAIL_TERMINAL,
        "passed": False,
        "execution_valid": False,
        "scientific_status": "EXECUTION_INVALID_NO_SCIENTIFIC_INTERPRETATION",
        "failure_code": str(getattr(error, "code", type(error).__name__)),
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


def execute(
    execution_lock_path: Path,
    *,
    lock_validator: Callable[[Path], dict[str, Any]] = validate_execution_lock,
) -> dict[str, Any]:
    lock = lock_validator(execution_lock_path.resolve())
    roots = lock["roots"]
    source_root = Path(roots["source_root"]).resolve()
    r3_root = Path(roots["r3_evidence_root"]).resolve()
    predecessor_root = Path(roots["r5_predecessor_root"]).resolve()
    output_root_key = "r5_r3_evidence_root" if "r5_r3_evidence_root" in roots else "r5_r2_evidence_root"
    evidence_root = Path(roots[output_root_key]).resolve()
    require(source_root.is_dir() and r3_root.is_dir() and predecessor_root.is_dir() and not evidence_root.exists(), "R5_R2_ROOT_PREFLIGHT_INVALID", "R2 roots do not satisfy one-shot preflight")
    frames = r5io.load_exact_cohort(Path(lock["frame_plan_path"]).resolve(), r3_root, source_root, verify_containers=True)
    require(len(frames) == r5.EXPECTED_FRAME_COUNT, "R5_R2_COHORT_PREFLIGHT_INVALID", "R2 cohort differs from frozen 211 frames")
    candidate, phase_a, predecessor_manifest = _verify_predecessor(predecessor_root, lock["predecessor_evidence_binding"])
    decisions = _load_decisions(frames, predecessor_root, phase_a)
    budget = lock["resource_budget"]
    started = time.monotonic()
    process = psutil.Process()
    writer = FactorEvidenceWriter(evidence_root, int(budget["maximum_evidence_bytes"]))

    def guard() -> None:
        require(time.monotonic() - started <= float(budget["maximum_wall_seconds"]), "R5_R2_EXECUTION_TIMEOUT", "R2 exceeded wall-time budget")
        require(process.memory_info().rss <= int(budget["maximum_peak_rss_bytes"]), "R5_R2_EXECUTION_RSS_EXCEEDED", "R2 exceeded RSS budget")

    try:
        writer.activate(
            {
                "schema": "blindassist.taro.o0r.r5_phase_b_repair_execution_start.v1",
                "execution_lock_sha256": materializer.sha256_file(execution_lock_path.resolve()),
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "argv": lock["unique_argv"],
                "predecessor_manifest_sha256": lock["predecessor_evidence_binding"]["manifest_sha256"],
                "candidate_phase_completion_sha256": candidate["content_sha256"],
                "phase_a_completion_sha256": phase_a["content_sha256"],
                "predecessor_file_count": predecessor_manifest["file_count_before_manifest"],
                "model_inference_performed": False,
                "source_decision_recomputed": False,
                "one_shot_consumed_on_root_creation": True,
            }
        )
        summary, unobservable = _phase_b(
            frames,
            predecessor_root=predecessor_root,
            writer=writer,
            phase_a=phase_a,
            decisions=decisions,
            guard=guard,
        )
        terminal = str(summary["terminal"])
        result = {
            "schema": RESULT_SCHEMA,
            "terminal": terminal,
            "passed": terminal == "TARO_O0R_DIRECT_APPLE_HYBRID_R5_TASK_METRIC_CONFIRMATION_PASS",
            "execution_valid": True,
            "scientific_status": terminal,
            "execution_lock_sha256": materializer.sha256_file(execution_lock_path.resolve()),
            "predecessor_manifest_sha256": lock["predecessor_evidence_binding"]["manifest_sha256"],
            "candidate_phase_completion_sha256": candidate["content_sha256"],
            "phase_a_completion_sha256": phase_a["content_sha256"],
            "summary_sha256": summary["content_sha256"],
            "candidate_inference_count_carried_forward": r5.EXPECTED_FRAME_COUNT,
            "candidate_inference_count_new": 0,
            "source_decision_count_carried_forward": r5.EXPECTED_FRAME_COUNT,
            "source_decision_count_new": 0,
            "phase_b_faro_frame_count": r5.EXPECTED_FRAME_COUNT,
            "query_record_count": r5.EXPECTED_QUERY_COUNT,
            "faro_support_unobservable_frame_count": sum(unobservable.values()),
            "faro_support_unobservable_reason_counts": dict(sorted(unobservable.items())),
            "training_steps": 0,
            "network_requests": 0,
            "elapsed_seconds": round(float(time.monotonic() - started), 6),
            "claim_ceiling": r5.CLAIM_CEILING,
        }
        writer.write_json("result.json", result)
        writer.write_json(
            "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "files": {key: value for key, value in sorted(writer.file_receipts.items())},
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
                "evidence_root_consumed": True,
                "terminal": terminal,
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
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"status": "EXECUTION_NOT_STARTED", "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    _emit({"terminal": result["terminal"], "passed": result["passed"], "execution_valid": result.get("execution_valid", False)})
    return 0 if result.get("execution_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
