#!/usr/bin/env python3
"""Run the two-phase, CPU-only TARO AppleDepth scale replay canary."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import PIL
import torch

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale
from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import (
    expected_candidate_keys,
    load_sealed_candidate_frame,
    validate_candidate_phase_completion,
)
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


SOURCE_ROOT = REPO_ROOT / "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r3"
TRUTH_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3"
FACTOR_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-factor-headroom-r3"
EVIDENCE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-candidate-scale-r0"
FRAME_PLAN_PATH = TRUTH_ROOT / "exact-frame-plan.json.gz"
CANDIDATE_COMPLETION_PATH = FACTOR_ROOT / "candidate-phase-completion.json"
FACTOR_MANIFEST_PATH = FACTOR_ROOT / "manifest.json"
ORACLE_RECORDS_PATH = FACTOR_ROOT / "descriptive-factor-canary-records.json.gz"
MAXIMUM_EVIDENCE_BYTES = 64 * 1024 * 1024


class CandidateScaleReplayError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise CandidateScaleReplayError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    require("content_sha256" not in result, "SEAL_COLLISION", "payload already contains a content seal")
    result["content_sha256"] = adapter.canonical_sha256(result)
    return result


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _code_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "APPLE_SCALE": Path(apple_scale.__file__).resolve(),
        "RUNNER": Path(__file__).resolve(),
        "CANDIDATE_PHASE": REPO_ROOT / "scripts/research/taro_o0r_factor_headroom_runtime/candidate_phase.py",
        "DEPTHART_RUNNER": REPO_ROOT / "scripts/research/taro_o0r_factor_headroom_runtime/depthart_runner.py",
        "FACTOR_CANARY": REPO_ROOT / "scripts/research/taro_o0r_factor_headroom_runtime/factor_canary.py",
    }
    return {
        role: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": materializer.sha256_file(path),
        }
        for role, path in sorted(paths.items())
    }


def _validate_phase_a_inputs() -> tuple[list[dict[str, Any]], list[tuple[str, str, str]], dict[str, Any]]:
    require(SOURCE_ROOT.is_dir() and FACTOR_ROOT.is_dir() and TRUTH_ROOT.is_dir(), "INPUT_ROOT_MISSING", "source/truth/factor roots must exist")
    require(not EVIDENCE_ROOT.exists(), "ONE_SHOT_ROOT_COLLISION", "candidate-scale evidence root already exists", root=str(EVIDENCE_ROOT))
    frame_plan = _load_json_gzip(FRAME_PLAN_PATH)
    require(isinstance(frame_plan, list) and len(frame_plan) == 24, "FRAME_PLAN_INVALID", "bound frame plan must contain 24 parents")
    expected = expected_candidate_keys(frame_plan)
    completion = validate_candidate_phase_completion(_load_json(CANDIDATE_COMPLETION_PATH))
    require(completion["candidate_frame_count"] == len(expected) == 239, "CANDIDATE_COUNT_DRIFT", "sealed candidate count differs from frozen eval plan")
    require(completion["candidate_frame_sequence_sha256"] == adapter.canonical_sha256([list(key) for key in expected]), "CANDIDATE_SEQUENCE_DRIFT", "sealed candidate sequence differs from frozen eval plan")
    return frame_plan, expected, completion


def _verify_oracle_binding_after_phase_a() -> dict[str, Any]:
    """This function is the sole Phase-B gate that opens oracle metadata."""

    manifest = _load_json(FACTOR_MANIFEST_PATH)
    require(manifest.get("schema") == "blindassist.taro.o0r.factor_headroom_manifest.v1" and isinstance(manifest.get("files"), dict), "FACTOR_MANIFEST_INVALID", "factor manifest schema/files drift")
    relative = "descriptive-factor-canary-records.json.gz"
    receipt = manifest["files"].get(relative)
    require(isinstance(receipt, dict) and receipt.get("path") == relative, "ORACLE_BINDING_MISSING", "oracle aggregate is absent from factor manifest")
    require(ORACLE_RECORDS_PATH.is_file() and ORACLE_RECORDS_PATH.stat().st_size == receipt.get("bytes") and materializer.sha256_file(ORACLE_RECORDS_PATH) == receipt.get("sha256"), "ORACLE_BINDING_DRIFT", "oracle aggregate differs from factor manifest")
    return {"factor_manifest_sha256": materializer.sha256_file(FACTOR_MANIFEST_PATH), "oracle_records": dict(receipt)}


def _source_record_relative(parent_id: str, video_id: str, token: str) -> str:
    return f"source-scale/{parent_id}/{video_id}/{token}.json.gz"


def execute() -> dict[str, Any]:
    frame_plan, expected, candidate_completion = _validate_phase_a_inputs()
    parent_lookup = {
        (str(row["parent"]["visit_id"]), str(row["parent"]["video_id"])): row
        for row in frame_plan
        if row["parent"]["role"] == "O0R_EVAL_CANDIDATE"
    }
    require(len(parent_lookup) == 16, "EVAL_PARENT_COUNT_DRIFT", "candidate-scale replay requires 16 frozen eval parents")
    code_bindings = _code_bindings()
    started = time.monotonic()
    writer = FactorEvidenceWriter(EVIDENCE_ROOT, MAXIMUM_EVIDENCE_BYTES)
    writer.activate(
        {
            "schema": "blindassist.taro.o0r.candidate_scale_execution_start.v1",
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "claim_ceiling": apple_scale.CLAIM_CEILING,
            "method": {
                "estimator_id": apple_scale.ESTIMATOR_ID,
                "registration_id": apple_scale.REGISTRATION_ID,
                "depth_range_m": list(apple_scale.DEPTH_RANGE_M),
                "minimum_pair_count": apple_scale.MINIMUM_PAIR_COUNT,
                "confidence_selection": "CONFIDENCE_EQ_2",
            },
            "phase_order": "SEAL_ALL_239_SOURCE_ESTIMATES_BEFORE_OPENING_ORACLE_RECORDS",
            "code_bindings": code_bindings,
            "input_bindings": {
                "frame_plan": {"path": FRAME_PLAN_PATH.relative_to(REPO_ROOT).as_posix(), "bytes": FRAME_PLAN_PATH.stat().st_size, "sha256": materializer.sha256_file(FRAME_PLAN_PATH)},
                "candidate_phase_completion": {"path": CANDIDATE_COMPLETION_PATH.relative_to(REPO_ROOT).as_posix(), "bytes": CANDIDATE_COMPLETION_PATH.stat().st_size, "sha256": materializer.sha256_file(CANDIDATE_COMPLETION_PATH), "content_sha256": candidate_completion["content_sha256"]},
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_cuda": str(torch.version.cuda),
                "pillow": PIL.__version__,
                "device": "cpu",
            },
            "training": False,
            "network": False,
            "gpu_inference": False,
            "one_shot_consumed_on_root_creation": True,
        }
    )

    # Phase A: do not open factor manifest, oracle aggregate, truth commitments,
    # FARO members, or RGB members.  Only sealed candidate blobs plus the two
    # explicitly authorized source members are accessed.
    source_records: list[dict[str, Any]] = []
    source_record_hashes: list[str] = []
    for index, (parent_id, video_id, token) in enumerate(expected, start=1):
        row = parent_lookup[(parent_id, video_id)]
        container_receipt = row["container_receipts"]["upsampling.zip"]
        archive_path = materializer.safe_join(SOURCE_ROOT, str(container_receipt["relative_path"]))
        sealed = load_sealed_candidate_frame(FACTOR_ROOT, parent_id, video_id, token)
        replay = apple_scale.build_candidate_replay_binding(sealed["candidate_frame_record"], sealed["native_depth_m"])
        source = apple_scale.decode_apple_scale_source(
            archive_path,
            container_receipt,
            parent_id=parent_id,
            video_id=video_id,
            timestamp_token=token,
            physical_frame_id=f"{video_id}:{token}",
            frame_plan_sha256=adapter.canonical_sha256(row),
            candidate_phase_completion_sha256=candidate_completion["content_sha256"],
        )
        record = apple_scale.build_source_scale_record(
            source["apple_depth_mm"],
            source["confidence"],
            replay["candidate_highres_depth_m"],
            source["source_receipt"],
            replay["candidate_binding"],
        )
        writer.write_json_gzip(_source_record_relative(parent_id, video_id, token), record)
        source_records.append(record)
        source_record_hashes.append(record["content_sha256"])
        if index == 1 or index % 20 == 0 or index == len(expected):
            print(json.dumps({"phase": "SOURCE_ONLY_SCALE", "completed": index, "total": len(expected), "evaluable": sum(bool(item["evaluable"]) for item in source_records)}, sort_keys=True), flush=True)

    phase_a_completion = _seal(
        {
            "schema": "blindassist.taro.o0r.candidate_scale_source_phase_completion.v1",
            "source_frame_count": len(source_records),
            "source_frame_sequence_sha256": adapter.canonical_sha256([list(key) for key in expected]),
            "source_scale_record_hashes_sha256": adapter.canonical_sha256(source_record_hashes),
            "source_evaluable_frame_count": sum(bool(record["evaluable"]) for record in source_records),
            "source_unknown_frame_count": sum(not bool(record["evaluable"]) for record in source_records),
            "candidate_phase_completion_sha256": candidate_completion["content_sha256"],
            "all_source_records_sealed_before_oracle_open": True,
            "oracle_files_opened_before_completion": 0,
            "truth_commitments_opened_before_completion": 0,
            "faro_members_opened_before_completion": 0,
            "rgb_members_opened_before_completion": 0,
            "gpu_inference_count": 0,
            "training_steps": 0,
        }
    )
    writer.write_json("source-phase-completion.json", phase_a_completion)

    # Phase B: the source estimator is now immutable evidence.  Open and score
    # the existing FARO oracle records without feeding any value back to it.
    oracle_binding = _verify_oracle_binding_after_phase_a()
    raw_oracles = _load_json_gzip(ORACLE_RECORDS_PATH)
    require(isinstance(raw_oracles, list) and len(raw_oracles) == 1494, "ORACLE_RECORD_COUNT_DRIFT", "expected the sealed 1494-query descriptive oracle set")
    source_lookup = {record["physical_frame_id"]: record for record in source_records}
    comparisons: list[dict[str, Any]] = []
    for raw in raw_oracles:
        validated = apple_scale.validate_factor_canary_record(dict(raw))
        source_record = source_lookup.get(validated["physical_frame_id"])
        require(source_record is not None, "ORACLE_SOURCE_FRAME_MISSING", "oracle record lacks a sealed source-scale estimate", physical_frame_id=validated["physical_frame_id"])
        comparisons.append(apple_scale.build_oracle_comparison(source_record, validated))
    summary = apple_scale.summarize_source_scale_canary(source_records, comparisons)
    writer.write_json_gzip("oracle-comparisons.json.gz", comparisons)
    writer.write_json("summary.json", summary)
    result = {
        "schema": "blindassist.taro.o0r.candidate_scale_canary_result.v1",
        "terminal": "TARO_O0R_APPLE_SCALE_SOURCE_CANARY_COMPLETE",
        "execution_valid": True,
        "scientific_status": "POST_HOC_DESCRIPTIVE_TRUTH_BLIND_SCALE_CANARY_ONLY",
        "claim_ceiling": apple_scale.CLAIM_CEILING,
        "source_phase_completion_sha256": phase_a_completion["content_sha256"],
        "summary_sha256": summary["content_sha256"],
        "factor_manifest_sha256": oracle_binding["factor_manifest_sha256"],
        "source_frame_count": summary["source_frame_count"],
        "source_evaluable_frame_count": summary["source_evaluable_frame_count"],
        "oracle_query_count": summary["oracle_query_count"],
        "oracle_paired_frame_count": summary["oracle_paired_frame_count"],
        "oracle_paired_parent_count": summary["oracle_paired_parent_count"],
        "formal_o0r_pass_authorized": False,
        "threshold_or_pass_fail_decision_applied": False,
        "gpu_inference_count": 0,
        "training_steps": 0,
        "network_requests": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    writer.write_json("result.json", result)
    files_before_manifest = dict(sorted(writer.file_receipts.items()))
    writer.write_json(
        "manifest.json",
        {
            "schema": "blindassist.taro.o0r.candidate_scale_manifest.v1",
            "files": files_before_manifest,
            "file_count_before_manifest": len(files_before_manifest),
            "bytes_before_manifest": sum(int(value["bytes"]) for value in files_before_manifest.values()),
            "source_phase_sealed_before_oracle_join": True,
            "one_shot_root_consumed": True,
        },
    )
    print(json.dumps({"terminal": result["terminal"], "execution_valid": True, "source_frames": result["source_frame_count"], "paired_frames": result["oracle_paired_frame_count"], "summary_sha256": result["summary_sha256"]}, sort_keys=True), flush=True)
    return result


def main() -> int:
    try:
        execute()
    except Exception as error:
        payload = {"terminal": "TARO_O0R_APPLE_SCALE_SOURCE_CANARY_FAILED", "error_code": str(getattr(error, "code", type(error).__name__)), "error": str(error)}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
