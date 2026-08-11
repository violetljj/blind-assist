#!/usr/bin/env python3
"""One-shot source-only O1R reducer replay on the frozen 239 eval frames."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import formation_replay_io as replay_io
from scripts.research.taro_o0r_candidate_scale_runtime import run_formation_replay as r6_run
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as runtime
from scripts.research.taro_o1r_reducer_integration_runtime.locked_uncertainty import load_locked_uncertainty_model


RESULT_SCHEMA = "blindassist.taro.o1r.r6_reducer_integration_eval_replay_result.v1"
SUMMARY_SCHEMA = "blindassist.taro.o1r.r6_reducer_integration_eval_replay_summary.v1"
MANIFEST_SCHEMA = "blindassist.taro.o1r.r6_reducer_integration_eval_replay_manifest.v1"
TERMINAL = "TARO_O1R_R6_REDUCER_INTEGRATION_NOT_EVALUABLE_ALL_UNKNOWN"
FAIL_TERMINAL = "TARO_O1R_R6_REDUCER_INTEGRATION_EXECUTION_INVALID"
SUCCESSOR = "TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_TASK_LOCK"


class ReplayError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise ReplayError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "O1R_REPLAY_SEAL_COLLISION", "caller supplied a content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, separators=(",", ":")), flush=True)


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.quantile(array, 0.5, method="linear")),
        "q95": float(np.quantile(array, 0.95, method="linear")),
        "maximum": float(np.max(array)),
    }


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> dict[str, Any]:
    failure = _seal(
        {
            "schema": RESULT_SCHEMA,
            "terminal": FAIL_TERMINAL,
            "execution_valid": False,
            "scientific_status": "INVALID",
            "failure_code": str(getattr(error, "code", type(error).__name__)),
            "message": str(error),
            "one_shot_consumed": True,
            "promotion_authorized": False,
        }
    )
    try:
        writer.write_json("failure.json", failure)
        writer.write_json(
            "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "terminal": FAIL_TERMINAL,
                "files": dict(sorted(writer.file_receipts.items())),
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
            },
        )
    except Exception:
        pass
    return failure


def execute(execution_lock_path: Path) -> dict[str, Any]:
    from scripts.research.taro_o1r_reducer_integration_runtime.validate_execution_lock import validate

    lock = validate(execution_lock_path.resolve(), require_output_absent=True)
    if not lock["passed"]:
        raise ReplayError("O1R_EXECUTION_LOCK_INVALID", "one-shot execution lock validation failed", errors=lock["errors"])
    frozen = lock["lock"]
    roots = frozen["roots"]
    output_root = Path(roots["output_root"]).resolve()
    writer = FactorEvidenceWriter(output_root, int(frozen["resource_budget"]["maximum_evidence_bytes"]))
    start = time.monotonic()
    execution_receipt = _seal(
        {
            "schema": "blindassist.taro.o1r.r6_reducer_integration_execution_receipt.v1",
            "execution_lock_path": execution_lock_path.resolve().as_posix(),
            "execution_lock_sha256": hashlib.sha256(execution_lock_path.read_bytes()).hexdigest().upper(),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "expected_eval_frames": 239,
            "expected_query_slots": 2151,
            "allowed_source_payload_roles": ["confidence"],
            "forbidden_payload_roles": ["highres_depth", "lowres_depth", "color"],
            "training_steps": 0,
            "network_requests": 0,
            "one_shot_consumed_on_root_creation": True,
        }
    )
    writer.activate(execution_receipt)
    try:
        repo_root = Path(roots["repo_root"]).resolve()
        frames = replay_io.load_exact_cohort(
            Path(roots["frame_plan_path"]),
            Path(roots["source_evidence_root"]),
            Path(roots["source_root"]),
            verify_containers=True,
        )
        eval_frames = [frame for frame in frames if frame.source_role == "O0R_EVAL_CANDIDATE"]
        require(len(eval_frames) == 239 and len({frame.parent_id for frame in eval_frames}) == 16, "O1R_EVAL_COHORT_DRIFT", "O1R eval cohort is not exact 16/239")
        model = load_locked_uncertainty_model()
        source_reads: Counter[str] = Counter()
        state_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        parent_states: dict[str, Counter[str]] = defaultdict(Counter)
        parent_frame_counts: Counter[str] = Counter()
        parent_definite_frames: Counter[str] = Counter()
        numeric_uncertainties: list[float] = []
        numeric_values: list[float] = []
        component_values: dict[str, list[float]] = defaultdict(list)
        definite_frame_count = 0

        def observe(role: str, _member: str) -> None:
            require(role == "confidence", "O1R_FORBIDDEN_SOURCE_PAYLOAD_READ", "O1R replay attempted a forbidden source payload", role=role)
            source_reads[role] += 1

        for index, frame in enumerate(eval_frames, 1):
            candidate, native = replay_io.load_candidate_frame(Path(roots["fit_candidate_root"]), Path(roots["eval_candidate_root"]), frame)
            highres = depthart_runner.upsample_native_depth(native)
            bundle = r6_run._load_bundle(Path(roots["phase_a_root"]), frame, highres)
            require(candidate["content_sha256"] == bundle["candidate_frame_record_sha256"], "O1R_CANDIDATE_BUNDLE_DRIFT", "candidate and R6 bundle identity differ")
            with zipfile.ZipFile(frame.upsampling_archive) as archive:
                confidence = replay_io.read_bound_payload(frame, archive, "confidence", read_observer=observe)
            result = runtime._integrate_with_validated_model(
                prospective_bundle=bundle,
                candidate_highres_depth_m=highres,
                confidence=confidence,
                intrinsics_apple_3x3=r6_run._lowres_matrix(frame.source_frame_receipt),
                uncertainty_model=model,
            )
            writer.write_json_gzip(f"frame-results/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz", result)
            state_counts.update(result["state_counts"])
            parent_states[frame.parent_id].update(result["state_counts"])
            parent_frame_counts[frame.parent_id] += 1
            definite = result["state_counts"]["CLEAR_OBSERVED"] + result["state_counts"]["OCCUPIED_OBSERVED"]
            if definite:
                definite_frame_count += 1
                parent_definite_frames[frame.parent_id] += 1
            for query in result["query_results"]:
                reason_counts.update(query["reason_codes"])
                if query["value_m"] is not None:
                    numeric_values.append(float(query["value_m"]))
                    numeric_uncertainties.append(float(query["uncertainty_m"]))
                    components = query["uncertainty"]["components_m"]
                    for name in ("scale_m", "support_m", "boundary_m"):
                        component_values[name].append(float(components[name]))
            if index % 40 == 0:
                _emit({"phase": "O1R_EVAL_REPLAY", "completed_frames": index, "total_frames": len(eval_frames), "state_counts": dict(state_counts)})
        require(source_reads == Counter({"confidence": 239}), "O1R_SOURCE_READ_RECEIPT_DRIFT", "O1R replay source-read receipt drift", reads=dict(source_reads))
        require(sum(state_counts.values()) == 2151, "O1R_QUERY_CARDINALITY_DRIFT", "O1R replay did not retain 2151 queries")
        summary = _seal(
            {
                "schema": SUMMARY_SCHEMA,
                "terminal": TERMINAL,
                "execution_valid": True,
                "frame_count": 239,
                "parent_count": 16,
                "query_count": 2151,
                "state_counts": {state: int(state_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
                "definite_frame_count": definite_frame_count,
                "definite_parent_count": len(parent_definite_frames),
                "parent_frame_counts": dict(sorted(parent_frame_counts.items())),
                "parent_definite_frame_counts": dict(sorted(parent_definite_frames.items())),
                "parent_state_counts": {parent: {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")} for parent, counts in sorted(parent_states.items())},
                "reason_counts": dict(sorted(reason_counts.items())),
                "numeric_interval_query_count": len(numeric_uncertainties),
                "numeric_value_m_distribution": _percentiles(numeric_values),
                "numeric_uncertainty_m_distribution": _percentiles(numeric_uncertainties),
                "uncertainty_component_m_distributions": {name: _percentiles(values) for name, values in sorted(component_values.items())},
                "source_payload_reads": dict(source_reads),
                "faro_payload_reads": 0,
                "training_steps": 0,
                "network_requests": 0,
                "unknown_is_negative": False,
                "promotion_authorized": False,
            }
        )
        writer.write_json("summary.json", summary)
        elapsed = time.monotonic() - start
        result = _seal(
            {
                "schema": RESULT_SCHEMA,
                "terminal": TERMINAL,
                "execution_valid": True,
                "scientific_status": "NOT_EVALUABLE_FINAL_STATE_COVERAGE",
                "passed": False,
                "promotion_authorized": False,
                "frame_count": 239,
                "query_count": 2151,
                "state_counts": summary["state_counts"],
                "definite_frame_count": definite_frame_count,
                "definite_parent_count": len(parent_definite_frames),
                "summary_sha256": summary["content_sha256"],
                "uncertainty_model_sha256": model.content_sha256,
                "source_payload_reads": dict(source_reads),
                "faro_payload_reads": 0,
                "training_steps": 0,
                "network_requests": 0,
                "resource_receipt": {
                    "elapsed_seconds": elapsed,
                    "peak_rss_bytes_observed_at_completion": int(psutil.Process().memory_info().rss),
                    "evidence_bytes_before_result": writer.bytes_written,
                },
                "claim_ceiling": "Locked ARKitScenes source-only O1R reducer replay; no final effectiveness, deployment, device, product, or safety claim.",
                "unique_successor": SUCCESSOR,
            }
        )
        writer.write_json("result.json", result)
        writer.write_json(
            "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "terminal": TERMINAL,
                "files": dict(sorted(writer.file_receipts.items())),
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
            },
        )
        return result
    except Exception as error:
        _write_failure(writer, error)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        _emit({"terminal": FAIL_TERMINAL, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)})
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
