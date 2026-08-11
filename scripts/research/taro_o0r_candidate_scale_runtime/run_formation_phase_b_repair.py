#!/usr/bin/env python3
"""Resume TARO formation scoring from a fully sealed predecessor Phase-A completion."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import formation_replay
from scripts.research.taro_o0r_candidate_scale_runtime import formation_replay_io as replay_io
from scripts.research.taro_o0r_candidate_scale_runtime import run_formation_replay as base_runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


RESULT_SCHEMA = "blindassist.taro.o0r.r6_formation_phase_b_repair_result.v1"
MANIFEST_SCHEMA = "blindassist.taro.o0r.r6_formation_phase_b_repair_manifest.v1"
FAIL_TERMINAL = "TARO_O0R_R6_FORMATION_PHASE_B_REPAIR_EXECUTION_INVALID"


class FormationPhaseBRepairError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FormationPhaseBRepairError(code, message, **context)


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def _verify_phase_a_files(root: Path, frames: Sequence[replay_io.FormationFrameRef], expected_completion_sha256: str) -> dict[str, Any]:
    completion_path = root / "phase-a-completion.json"
    manifest_path = root / "manifest.json"
    require(completion_path.is_file() and manifest_path.is_file(), "FORMATION_PHASE_A_PREDECESSOR_MISSING", "sealed Phase-A predecessor completion/manifest is missing")
    completion = base_runner._validate_seal(json.loads(completion_path.read_text(encoding="utf-8")), base_runner.PHASE_A_SCHEMA)
    require(
        completion["content_sha256"] == expected_completion_sha256
        and completion["frame_count"] == 450
        and completion["query_slot_count"] == 4050
        and completion["faro_reads"] == 0
        and completion["all_bundles_sealed_before_faro"] is True,
        "FORMATION_PHASE_A_PREDECESSOR_INVALID",
        "predecessor Phase-A completion is not the admitted zero-FARO 450-frame seal",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    require(isinstance(files, dict), "FORMATION_PHASE_A_MANIFEST_INVALID", "predecessor manifest file ledger is missing")
    hashes: list[str] = []
    for frame in frames:
        relative = base_runner.bundle_relative(frame)
        binding = files.get(relative)
        path = materializer.safe_join(root, relative)
        require(
            isinstance(binding, dict)
            and path.is_file()
            and path.stat().st_size == binding.get("bytes")
            and materializer.sha256_file(path) == binding.get("sha256"),
            "FORMATION_PHASE_A_BUNDLE_LEDGER_DRIFT",
            "sealed predecessor Phase-A bundle differs from its manifest",
            path=relative,
        )
        hashes.append(binding["sha256"])
    require(len(hashes) == 450, "FORMATION_PHASE_A_BUNDLE_COUNT_DRIFT", "predecessor Phase-A manifest does not cover 450 bundles")
    return completion


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
    from scripts.research.taro_o0r_candidate_scale_runtime.validate_formation_phase_b_repair_lock import validate_execution_lock

    lock = validate_execution_lock(execution_lock_path.resolve())
    roots = lock["roots"]
    source_root = Path(roots["source_root"]).resolve()
    source_evidence_root = Path(roots["source_evidence_root"]).resolve()
    fit_candidate_root = Path(roots["fit_candidate_root"]).resolve()
    eval_candidate_root = Path(roots["eval_candidate_root"]).resolve()
    phase_a_root = Path(roots["phase_a_root"]).resolve()
    output_root = Path(roots["output_root"]).resolve()
    plan_path = Path(roots["frame_plan_path"]).resolve()
    require(not output_root.exists(), "FORMATION_PHASE_B_OUTPUT_COLLISION", "Phase-B repair output root already exists")
    frames = replay_io.load_exact_cohort(plan_path, source_evidence_root, source_root, verify_containers=True)
    preflight = replay_io.preflight_inventory(frames, fit_candidate_root, eval_candidate_root)
    for field, expected in lock["cohort_bindings"].items():
        require(preflight[field] == expected, "FORMATION_PHASE_B_COHORT_DRIFT", "Phase-B repair cohort binding drift", field=field)
    phase_a = _verify_phase_a_files(phase_a_root, frames, lock["phase_a_completion_sha256"])
    budget = lock["resource_budget"]
    started = time.monotonic()
    process = psutil.Process()
    writer = FactorEvidenceWriter(output_root, int(budget["maximum_evidence_bytes"]))

    def guard() -> None:
        require(time.monotonic() - started <= float(budget["maximum_wall_seconds"]), "FORMATION_PHASE_B_TIMEOUT", "Phase-B repair exceeded wall-time budget")
        require(process.memory_info().rss <= int(budget["maximum_peak_rss_bytes"]), "FORMATION_PHASE_B_RSS_EXCEEDED", "Phase-B repair exceeded RSS budget")

    try:
        lock_sha = materializer.sha256_file(execution_lock_path.resolve())
        writer.activate(
            {
                "schema": "blindassist.taro.o0r.r6_formation_phase_b_repair_start.v1",
                "execution_lock_sha256": lock_sha,
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "predecessor_phase_a_completion_sha256": phase_a["content_sha256"],
                "predecessor_phase_a_root": str(phase_a_root),
                "phase_a_reexecution": False,
                "one_shot_consumed_on_root_creation": True,
            }
        )
        records: list[dict[str, Any]] = []
        unobservable: set[str] = set()
        with ThreadPoolExecutor(max_workers=int(lock["worker_count"])) as executor:
            iterator = executor.map(lambda frame: base_runner._phase_b_worker(frame, phase_a_root, fit_candidate_root, eval_candidate_root), frames)
            for index, (frame, rows) in enumerate(iterator, start=1):
                writer.write_json_gzip(base_runner.score_relative(frame), rows)
                records.extend(rows)
                if all(row["truth_status"]["evaluable"] is not True for row in rows):
                    unobservable.add(frame.physical_frame_id)
                guard()
                if index % 10 == 0 or index == 450:
                    _emit({"phase": "FORMATION_PHASE_B_REPAIR", "completed_frames": index, "total_frames": 450, "query_records": len(records), "faro_unobservable_frames": len(unobservable)})
        summary = formation_replay.summarize(records)
        writer.write_json("summary.json", summary)
        result = {
            "schema": RESULT_SCHEMA,
            "terminal": summary["terminal"],
            "execution_valid": True,
            "scientific_pass_fail_assigned": False,
            "execution_lock_sha256": lock_sha,
            "predecessor_phase_a_completion_sha256": phase_a["content_sha256"],
            "summary_sha256": summary["content_sha256"],
            "parent_count": 24,
            "frame_count": 450,
            "query_record_count": 4050,
            "faro_support_unobservable_frame_count": len(unobservable),
            "candidate_inference_count": 0,
            "phase_a_reexecution": False,
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
        _emit({"status": "FORMATION_PHASE_B_REPAIR_NOT_STARTED", "error_code": getattr(error, "code", type(error).__name__), "message": str(error)})
        return 2
    _emit({"terminal": result["terminal"], "execution_valid": result.get("execution_valid", False), "scientific_pass_fail_assigned": result.get("scientific_pass_fail_assigned")})
    return 0 if result.get("execution_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
