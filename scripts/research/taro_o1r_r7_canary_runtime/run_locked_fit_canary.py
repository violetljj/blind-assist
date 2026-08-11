#!/usr/bin/env python3
"""Execute the one-shot TARO O1R R7 ADAPTER_FIT LOPO canary."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import formation_replay_io as replay_io
from scripts.research.taro_o0r_candidate_scale_runtime import run_formation_replay as r6_run
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as r6_reducer
from scripts.research.taro_o1r_reducer_integration_runtime.locked_uncertainty import load_locked_uncertainty_model


MANIFEST_SCHEMA = "blindassist.taro.o1r.r7_fit_lopo_canary_manifest.v1"
SOURCE_COMPLETION_SCHEMA = "blindassist.taro.o1r.r7_source_phase_completion.v1"
EXECUTION_RECEIPT_SCHEMA = "blindassist.taro.o1r.r7_fit_lopo_execution_receipt.v1"
FAILURE_SCHEMA = "blindassist.taro.o1r.r7_fit_lopo_execution_failure.v1"
FAIL_TERMINAL = "TARO_O1R_R7_FIT_LOPO_CANARY_EXECUTION_INVALID"


class R7RunError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R7RunError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in output, "R7_RUN_SEAL_COLLISION", "caller supplied a content seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, separators=(",", ":")), flush=True)


def _source_relative(frame: replay_io.FormationFrameRef) -> str:
    return f"source-features/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _label_relative(frame: replay_io.FormationFrameRef) -> str:
    return f"label-frames/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _load_gzip_json(root: Path, relative: str) -> dict[str, Any]:
    path = (root / Path(relative)).resolve()
    require(root.resolve() in path.parents and path.is_file(), "R7_EVIDENCE_RECORD_MISSING", "R7 evidence record is missing", path=relative)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise R7RunError("R7_EVIDENCE_RECORD_INVALID", "R7 evidence record cannot be decoded", path=relative) from error
    require(isinstance(value, dict), "R7_EVIDENCE_RECORD_INVALID", "R7 evidence record must be an object", path=relative)
    return value


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> None:
    failure = _seal(
        {
            "schema": FAILURE_SCHEMA,
            "terminal": FAIL_TERMINAL,
            "execution_valid": False,
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


def execute(execution_lock_path: Path) -> dict[str, Any]:
    from scripts.research.taro_o1r_r7_canary_runtime.validate_execution_lock import validate

    validation = validate(execution_lock_path.resolve(), require_output_absent=True)
    if not validation["passed"]:
        raise R7RunError("R7_EXECUTION_LOCK_INVALID", "R7 execution lock validation failed", errors=validation["errors"])
    frozen = validation["lock"]
    roots = frozen["roots"]
    output_root = Path(roots["output_root"]).resolve()
    writer = FactorEvidenceWriter(output_root, int(frozen["resource_budget"]["maximum_evidence_bytes"]))
    start = time.monotonic()
    receipt = _seal(
        {
            "schema": EXECUTION_RECEIPT_SCHEMA,
            "execution_lock_path": execution_lock_path.resolve().as_posix(),
            "execution_lock_sha256": hashlib.sha256(execution_lock_path.read_bytes()).hexdigest().upper(),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "role": "ADAPTER_FIT",
            "expected_parent_count": 8,
            "expected_frame_count": 211,
            "expected_query_count": 1899,
            "source_phase_allowed_payload_roles": ["lowres_depth", "confidence"],
            "label_phase_allowed_payload_roles": ["highres_depth"],
            "source_phase_sealed_before_label_join": True,
            "training_steps": 0,
            "network_requests": 0,
            "one_shot_consumed_on_root_creation": True,
        }
    )
    writer.activate(receipt)
    try:
        frames = replay_io.load_exact_cohort(
            Path(roots["frame_plan_path"]),
            Path(roots["source_evidence_root"]),
            Path(roots["source_root"]),
            verify_containers=True,
        )
        fit_frames = [frame for frame in frames if frame.source_role == "ADAPTER_FIT"]
        require(len(fit_frames) == 211 and len({frame.parent_id for frame in fit_frames}) == 8, "R7_FIT_COHORT_DRIFT", "R7 fit cohort is not exact 8/211")
        model = load_locked_uncertainty_model()
        source_reads: Counter[str] = Counter()
        source_hashes: list[str] = []

        def observe_source(role: str, _member: str) -> None:
            require(role in {"lowres_depth", "confidence"}, "R7_SOURCE_PHASE_PAYLOAD_FIREWALL", "source phase attempted a forbidden payload read", role=role)
            source_reads[role] += 1

        for index, frame in enumerate(fit_frames, 1):
            _, native = replay_io.load_candidate_frame(Path(roots["fit_candidate_root"]), Path(roots["eval_candidate_root"]), frame)
            highres = depthart_runner.upsample_native_depth(native)
            bundle = r6_run._load_bundle(Path(roots["phase_a_root"]), frame, highres)
            with zipfile.ZipFile(frame.upsampling_archive) as archive:
                apple = replay_io.read_bound_payload(frame, archive, "lowres_depth", read_observer=observe_source)
                confidence = replay_io.read_bound_payload(frame, archive, "confidence", read_observer=observe_source)
            low_matrix = r6_run._lowres_matrix(frame.source_frame_receipt)
            prior = r6_reducer._integrate_with_validated_model(
                prospective_bundle=bundle,
                candidate_highres_depth_m=highres,
                confidence=confidence,
                intrinsics_apple_3x3=low_matrix,
                uncertainty_model=model,
            )
            source = r7_canary.build_source_frame_record(
                bundle,
                highres,
                apple,
                confidence,
                low_matrix,
                frame.source_frame_receipt["intrinsics_highres"]["matrix_3x3"],
                prior,
            )
            writer.write_json_gzip(_source_relative(frame), source)
            source_hashes.append(source["content_sha256"])
            if index % 25 == 0:
                _emit({"phase": "R7_SOURCE", "completed_frames": index, "total_frames": len(fit_frames)})

        require(source_reads == Counter({"lowres_depth": 211, "confidence": 211}), "R7_SOURCE_READ_RECEIPT_DRIFT", "R7 source read counts drift", reads=dict(source_reads))
        source_completion = _seal(
            {
                "schema": SOURCE_COMPLETION_SCHEMA,
                "parent_count": 8,
                "frame_count": 211,
                "query_count": 1899,
                "source_frame_hash_sequence_sha256": adapter.canonical_sha256(source_hashes),
                "source_payload_reads": dict(sorted(source_reads.items())),
                "faro_reads": 0,
                "label_reads": 0,
                "source_phase_has_label_input": False,
                "source_phase_complete": True,
                "training_steps": 0,
                "network_requests": 0,
            }
        )
        writer.write_json("source-phase-completion.json", source_completion)
        reloaded_completion = json.loads((output_root / "source-phase-completion.json").read_text(encoding="utf-8"))
        require(reloaded_completion == source_completion and reloaded_completion["faro_reads"] == 0, "R7_SOURCE_COMPLETION_RELOAD_DRIFT", "R7 source completion changed before label phase")

        source_records: list[dict[str, Any]] = []
        for frame, expected_hash in zip(fit_frames, source_hashes, strict=True):
            source = r7_canary.validate_source_frame_record(_load_gzip_json(output_root, _source_relative(frame)))
            require(source["content_sha256"] == expected_hash, "R7_SOURCE_RELOAD_DRIFT", "R7 source record changed before label join")
            source_records.append(source)

        label_reads: Counter[str] = Counter()
        label_records: list[dict[str, Any]] = []

        def observe_label(role: str, _member: str) -> None:
            require(role == "highres_depth", "R7_LABEL_PHASE_PAYLOAD_FIREWALL", "label phase attempted a forbidden payload read", role=role)
            label_reads[role] += 1

        for index, (frame, source) in enumerate(zip(fit_frames, source_records, strict=True), 1):
            with zipfile.ZipFile(frame.upsampling_archive) as archive:
                faro = replay_io.read_bound_payload(frame, archive, "highres_depth", read_observer=observe_label)
            label = r7_canary.build_label_frame_record(
                source,
                faro,
                frame.source_frame_receipt["intrinsics_highres"]["matrix_3x3"],
                frame.source_frame_receipt["gravity_up_camera_xyz"],
            )
            writer.write_json_gzip(_label_relative(frame), label)
            label_records.append(label)
            if index % 25 == 0:
                _emit({"phase": "R7_LABEL", "completed_frames": index, "total_frames": len(fit_frames)})
        require(label_reads == Counter({"highres_depth": 211}), "R7_LABEL_READ_RECEIPT_DRIFT", "R7 label read counts drift", reads=dict(label_reads))

        canary = r7_canary.run_lopo_canary(source_records, label_records)
        result = dict(canary)
        result["execution_receipts"] = {
            "source_payload_reads": dict(sorted(source_reads.items())),
            "label_payload_reads": dict(sorted(label_reads.items())),
            "source_phase_completion_sha256": source_completion["content_sha256"],
            "source_phase_reloaded_before_label_join": True,
            "source_phase_reselection_after_labels": False,
            "training_steps": 0,
            "network_requests": 0,
            "elapsed_seconds": time.monotonic() - start,
            "peak_rss_bytes_observed_at_completion": int(psutil.Process().memory_info().rss),
        }
        canary_without_seal = dict(result)
        canary_without_seal.pop("content_sha256")
        result = _seal(canary_without_seal)
        writer.write_json("result.json", result)
        writer.write_json(
            "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "terminal": result["terminal"],
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
    _emit({"terminal": result["terminal"], "passed": result["passed"], "promotion_authorized": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
