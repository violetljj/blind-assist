#!/usr/bin/env python3
"""Replay the TARO R1 source-anchored factor/query point canary on 171 frames."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import platform
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import PIL
import scipy

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale, source_factor
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
SCALE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-candidate-scale-r0"
EVIDENCE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-factor-r1"
FRAME_PLAN_PATH = TRUTH_ROOT / "exact-frame-plan.json.gz"
CANDIDATE_COMPLETION_PATH = FACTOR_ROOT / "candidate-phase-completion.json"
SCALE_ORACLE_PATH = SCALE_ROOT / "oracle-comparisons.json.gz"
MAXIMUM_EVIDENCE_BYTES = 256 * 1024 * 1024


class SourceFactorRunError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise SourceFactorRunError(code, message, **context)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise SourceFactorRunError("BOUND_JSON_INVALID", "bound JSON cannot be decoded", path=str(path)) from error


def _load_json_gzip(path: Path) -> Any:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as error:
        raise SourceFactorRunError("BOUND_GZIP_JSON_INVALID", "bound gzip JSON cannot be decoded", path=str(path)) from error


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    require("content_sha256" not in output, "RUNNER_SEAL_COLLISION", "payload already carries a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _code_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "APPLE_SCALE": Path(apple_scale.__file__).resolve(),
        "SOURCE_FACTOR": Path(source_factor.__file__).resolve(),
        "RUNNER": Path(__file__).resolve(),
        "SOURCE_ADAPTER": Path(adapter.__file__).resolve(),
        "MATERIALIZER": Path(materializer.__file__).resolve(),
    }
    return {
        role: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": materializer.sha256_file(path),
        }
        for role, path in sorted(paths.items())
    }


def _verify_manifest_file(root: Path, manifest: Mapping[str, Any], relative: str) -> Path:
    receipt = manifest.get("files", {}).get(relative)
    require(isinstance(receipt, dict) and receipt.get("path") == relative, "MANIFEST_FILE_MISSING", "bound file is absent from manifest", relative=relative)
    path = materializer.safe_join(root, relative)
    require(
        path.is_file()
        and path.stat().st_size == receipt.get("bytes")
        and materializer.sha256_file(path) == receipt.get("sha256"),
        "MANIFEST_FILE_HASH_DRIFT",
        "bound file differs from manifest",
        relative=relative,
    )
    return path


def _hydrate_truth(path: Path, truth_manifest: Mapping[str, Any]) -> dict[str, Any]:
    relative = path.relative_to(TRUTH_ROOT).as_posix()
    _verify_manifest_file(TRUTH_ROOT, truth_manifest, relative)
    package = _load_json_gzip(path)
    hydrated = materializer.hydrate_content_addressed_artifact(
        package,
        lambda child: materializer.safe_join(TRUTH_ROOT, child).read_bytes(),
    )
    return materializer.validate_eval_truth_commitment_record(hydrated)


def _decode_bound_faro(
    compact_truth: Mapping[str, Any],
    frame_plan_row: Mapping[str, Any],
) -> np.ndarray:
    source = compact_truth["source_frame_receipt"]
    envelope = materializer.validate_bound_source_frame_envelope(compact_truth["bound_source_frame_envelope"], source)
    container = frame_plan_row["container_receipts"]["upsampling.zip"]
    archive = materializer.safe_join(SOURCE_ROOT, str(container["relative_path"]))
    materializer.verify_bound_container(archive, container)
    member = envelope["members"]["highres_depth"]
    require(member["source_container_sha256"] == container["sha256"], "FARO_CONTAINER_BINDING_DRIFT", "FARO member container differs from frame plan")
    with zipfile.ZipFile(archive) as bundle:
        try:
            info = bundle.getinfo(member["source_member_path"])
        except KeyError as error:
            raise SourceFactorRunError("FARO_MEMBER_MISSING", "bound FARO member is absent") from error
        materializer._validate_zip_info(info)
        payload = bundle.read(info)
    require(
        len(payload) == member["source_member_bytes"]
        and materializer.sha256_bytes(payload) == member["source_member_sha256"]
        and materializer.crc32_bytes(payload) == member["source_member_crc32"],
        "FARO_MEMBER_HASH_DRIFT",
        "bound FARO member bytes drift",
    )
    faro = materializer._decode_png(payload, "highres_depth")
    faro_hash = adapter.canonical_sha256(faro)
    require(
        faro_hash == member["decoded_content_sha256"]
        == source["decoded_payload_bindings"]["highres_depth"]["decoded_content_sha256"]
        == compact_truth["highres_depth_array_sha256"],
        "FARO_DECODED_HASH_DRIFT",
        "decoded FARO differs from compact truth/source receipt",
    )
    return faro


def _source_scale_relative(parent_id: str, video_id: str, token: str) -> str:
    return f"source-scale/{parent_id}/{video_id}/{token}.json.gz"


def _frame_output_relative(parent_id: str, video_id: str, token: str) -> str:
    return f"frame-canary/{parent_id}/{video_id}/{token}.json.gz"


def _load_source_scale(
    parent_id: str,
    video_id: str,
    token: str,
    scale_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = _verify_manifest_file(SCALE_ROOT, scale_manifest, _source_scale_relative(parent_id, video_id, token))
    return apple_scale.validate_source_scale_record(_load_json_gzip(path))


def _oracle_frame_errors() -> dict[str, float]:
    raw = _load_json_gzip(SCALE_ORACLE_PATH)
    require(isinstance(raw, list) and len(raw) == 1494, "SCALE_ORACLE_COUNT_DRIFT", "R0 oracle comparison count drift")
    grouped: dict[str, list[float]] = {}
    for item in raw:
        row = apple_scale.validate_oracle_comparison(item)
        if row["evaluable"]:
            grouped.setdefault(row["physical_frame_id"], []).append(float(row["source_abs_log_error"]))
    return {frame_id: float(np.median(np.asarray(values, dtype=np.float64))) for frame_id, values in grouped.items()}


def _frame_keys(frame_plan: list[dict[str, Any]]) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], dict[str, Any]]]:
    parent_lookup = {
        (str(row["parent"]["visit_id"]), str(row["parent"]["video_id"])): row
        for row in frame_plan
        if row["parent"]["role"] == "O0R_EVAL_CANDIDATE"
    }
    keys: list[tuple[str, str, str]] = []
    for parent_id, video_id, token in expected_candidate_keys(frame_plan):
        if (TRUTH_ROOT / "truth-frames" / parent_id / video_id / f"{token}.json.gz").is_file():
            keys.append((parent_id, video_id, token))
    require(len(parent_lookup) == 16 and len(keys) == 171, "R1_FRAME_COHORT_DRIFT", "R1 requires the 171 compact-truth frames across 16 eval parents", parent_count=len(parent_lookup), frame_count=len(keys))
    return keys, parent_lookup


def _process_frame(
    parent_id: str,
    video_id: str,
    token: str,
    row: Mapping[str, Any],
    candidate_completion: Mapping[str, Any],
    truth_manifest: Mapping[str, Any],
    scale_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    truth_path = TRUTH_ROOT / "truth-frames" / parent_id / video_id / f"{token}.json.gz"
    compact = _hydrate_truth(truth_path, truth_manifest)
    source = adapter._validate_base_receipt(compact["source_frame_receipt"])
    require(
        (source["parent_id"], source["session_id"], source["sensor_timestamp"]["decimal_token"])
        == (parent_id, video_id, token),
        "R1_FRAME_IDENTITY_DRIFT",
        "compact truth identity differs from frame plan",
    )
    sealed = load_sealed_candidate_frame(FACTOR_ROOT, parent_id, video_id, token)
    replay = apple_scale.build_candidate_replay_binding(sealed["candidate_frame_record"], sealed["native_depth_m"])
    container = row["container_receipts"]["upsampling.zip"]
    archive = materializer.safe_join(SOURCE_ROOT, str(container["relative_path"]))
    apple_source = apple_scale.decode_apple_scale_source(
        archive,
        container,
        parent_id=parent_id,
        video_id=video_id,
        timestamp_token=token,
        physical_frame_id=source["physical_frame_id"],
        frame_plan_sha256=adapter.canonical_sha256(row),
        candidate_phase_completion_sha256=candidate_completion["content_sha256"],
    )
    scale_record = _load_source_scale(parent_id, video_id, token, scale_manifest)
    prepared = source_factor.prepare_source_anchored_candidate(
        replay["candidate_highres_depth_m"],
        apple_source["apple_depth_mm"],
        apple_source["confidence"],
        source,
        apple_source["source_receipt"],
        replay["candidate_binding"],
        scale_record,
    )
    faro = _decode_bound_faro(compact, row)
    matrix = np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
    geometry = adapter.derive_faro_geometry(faro, matrix, source["gravity_up_camera_xyz"], source)
    generated_queries = adapter.build_query_receipts(source, geometry)
    compact_queries = compact["query_receipts"]
    require(
        [item["content_sha256"] for item in generated_queries] == [item["content_sha256"] for item in compact_queries],
        "R1_QUERY_RECEIPT_RECOMPUTATION_DRIFT",
        "current query receipts differ from compact R3 commitments",
    )
    commitment_by_query = {item["query_id"]: item for item in compact["factor_frame_commitments"]}
    result_by_query = {item["query_id"]: item for item in compact["query_bundle"]["results"]}
    records: list[dict[str, Any]] = []
    for query in compact_queries:
        base = source_factor.build_query_truth_base(geometry, query)
        commitment = commitment_by_query[base.query_id]
        records.append(
            source_factor.evaluate_source_anchored_query(
                prepared,
                matrix,
                source["gravity_up_camera_xyz"],
                base,
                current_faro_geometry_sha256=geometry.content_sha256,
                compact_truth_record_sha256=compact["content_sha256"],
                committed_faro_geometry_sha256=compact["faro_geometry_sha256"],
                committed_factor_frame_sha256=commitment["factor_frame_sha256"],
                committed_base_geometry_sha256=commitment["base_geometry_sha256"],
                compact_query_result=result_by_query[base.query_id],
            )
        )
    return {
        "schema": "blindassist.taro.o0r.source_anchored_factor_frame_canary.v1",
        "parent_id": parent_id,
        "video_id": video_id,
        "timestamp_token": token,
        "physical_frame_id": source["physical_frame_id"],
        "compact_truth_record_sha256": compact["content_sha256"],
        "current_faro_geometry_sha256": geometry.content_sha256,
        "committed_faro_geometry_sha256": compact["faro_geometry_sha256"],
        "runtime_geometry_matches_r3_commitment": geometry.content_sha256 == compact["faro_geometry_sha256"],
        "source_scale_record_sha256": scale_record["content_sha256"],
        "reliability": prepared.reliability,
        "query_records": records,
        "query_count": len(records),
    }


def _preflight() -> tuple[list[dict[str, Any]], list[tuple[str, str, str]], dict[tuple[str, str], dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(all(root.is_dir() for root in (SOURCE_ROOT, TRUTH_ROOT, FACTOR_ROOT, SCALE_ROOT)), "R1_INPUT_ROOT_MISSING", "source/truth/factor/scale roots must exist")
    frame_plan = _load_json_gzip(FRAME_PLAN_PATH)
    require(isinstance(frame_plan, list) and len(frame_plan) == 24, "R1_FRAME_PLAN_INVALID", "bound frame plan must contain 24 parents")
    keys, parent_lookup = _frame_keys(frame_plan)
    completion = validate_candidate_phase_completion(_load_json(CANDIDATE_COMPLETION_PATH))
    require(completion["candidate_frame_count"] == 239, "R1_CANDIDATE_COUNT_DRIFT", "sealed candidate cohort drift")
    truth_manifest = _load_json(TRUTH_ROOT / "manifest.json")
    scale_manifest = _load_json(SCALE_ROOT / "manifest.json")
    require(isinstance(truth_manifest.get("files"), dict) and isinstance(scale_manifest.get("files"), dict), "R1_MANIFEST_INVALID", "truth/scale manifests are malformed")
    return frame_plan, keys, parent_lookup, completion, truth_manifest, scale_manifest


def execute(*, smoke_only: bool = False) -> dict[str, Any]:
    _, keys, parent_lookup, completion, truth_manifest, scale_manifest = _preflight()
    selected = keys[:1] if smoke_only else keys
    if not smoke_only:
        require(not EVIDENCE_ROOT.exists(), "ONE_SHOT_ROOT_COLLISION", "source-factor R1 evidence root already exists", root=str(EVIDENCE_ROOT))
    started = time.monotonic()
    writer = None if smoke_only else FactorEvidenceWriter(EVIDENCE_ROOT, MAXIMUM_EVIDENCE_BYTES)
    if writer is not None:
        writer.activate(
            {
                "schema": "blindassist.taro.o0r.source_anchored_factor_execution_start.v1",
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "claim_ceiling": source_factor.CLAIM_CEILING,
                "anchor_id": source_factor.ANCHOR_ID,
                "cohort": {"physical_frames": len(keys), "queries_per_frame": 9, "parents": 16},
                "code_bindings": _code_bindings(),
                "input_bindings": {
                    "frame_plan_sha256": materializer.sha256_file(FRAME_PLAN_PATH),
                    "candidate_phase_completion_sha256": completion["content_sha256"],
                    "truth_manifest_sha256": materializer.sha256_file(TRUTH_ROOT / "manifest.json"),
                    "scale_manifest_sha256": materializer.sha256_file(SCALE_ROOT / "manifest.json"),
                },
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__, "device": "cpu"},
                "training": False,
                "network": False,
                "gpu_inference": False,
                "formal_reducer": False,
                "uncertainty_state": False,
                "one_shot_consumed_on_root_creation": True,
            }
        )
    all_records: list[dict[str, Any]] = []
    reliability_records: list[dict[str, Any]] = []
    try:
        for index, (parent_id, video_id, token) in enumerate(selected, start=1):
            frame = _process_frame(parent_id, video_id, token, parent_lookup[(parent_id, video_id)], completion, truth_manifest, scale_manifest)
            all_records.extend(frame["query_records"])
            reliability_records.append(frame["reliability"])
            if writer is not None:
                writer.write_json_gzip(_frame_output_relative(parent_id, video_id, token), frame)
            progress = {
                "phase": "SOURCE_ANCHORED_FACTOR_QUERY",
                "completed_frames": index,
                "total_frames": len(selected),
                "query_records": len(all_records),
                "baseline_extraction_failures": sum(not row["baseline"]["extraction_evaluable"] for row in all_records),
                "anchored_extraction_failures": sum(not row["source_anchored"]["extraction_evaluable"] for row in all_records),
                "recovered_queries": sum(bool(row["effects"]["extraction_recovered"]) for row in all_records),
                "physical_frame_id": frame["physical_frame_id"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
        if smoke_only:
            return {"smoke_only": True, "frame": selected[0], "query_records": len(all_records), "elapsed_seconds": time.monotonic() - started}
        require(len(all_records) == 171 * 9 and len(reliability_records) == 171, "R1_OUTPUT_CARDINALITY_DRIFT", "R1 did not account for every compact-truth query/frame")
        summary = source_factor.summarize_source_anchored_canary(all_records, reliability_records)
        reliability_association = source_factor.summarize_reliability_association(reliability_records, _oracle_frame_errors())
        writer.write_json_gzip("query-records.json.gz", all_records)
        writer.write_json_gzip("reliability-records.json.gz", reliability_records)
        writer.write_json("summary.json", summary)
        writer.write_json("reliability-association.json", reliability_association)
        result = {
            "schema": "blindassist.taro.o0r.source_anchored_factor_canary_r1_result.v1",
            "terminal": "TARO_O0R_SOURCE_ANCHORED_FACTOR_CANARY_R1_COMPLETE",
            "execution_valid": True,
            "scientific_status": "POST_HOC_DESCRIPTIVE_SOURCE_ANCHORED_FACTOR_AND_POINT_CLEARANCE_CANARY_ONLY",
            "claim_ceiling": source_factor.CLAIM_CEILING,
            "summary_sha256": summary["content_sha256"],
            "reliability_association_sha256": reliability_association["content_sha256"],
            "physical_frame_count": 171,
            "query_record_count": 1539,
            "parent_count": 16,
            "training_steps": 0,
            "gpu_inference_count": 0,
            "network_requests": 0,
            "formal_reducer_executed": False,
            "formal_o0r_pass_authorized": False,
            "threshold_or_pass_fail_decision_applied": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        writer.write_json("result.json", result)
        files_before_manifest = dict(sorted(writer.file_receipts.items()))
        writer.write_json(
            "manifest.json",
            {
                "schema": "blindassist.taro.o0r.source_anchored_factor_canary_manifest.v1",
                "files": files_before_manifest,
                "file_count_before_manifest": len(files_before_manifest),
                "bytes_before_manifest": sum(int(item["bytes"]) for item in files_before_manifest.values()),
                "one_shot_root_consumed": True,
            },
        )
        print(json.dumps({"terminal": result["terminal"], "frames": 171, "queries": 1539, "summary_sha256": result["summary_sha256"]}, sort_keys=True), flush=True)
        return result
    except Exception as error:
        if writer is not None:
            failure = {
                "schema": "blindassist.taro.o0r.source_anchored_factor_canary_r1_failure.v1",
                "terminal": "TARO_O0R_SOURCE_ANCHORED_FACTOR_CANARY_R1_EXECUTION_INVALID",
                "execution_valid": False,
                "error_code": str(getattr(error, "code", type(error).__name__)),
                "message": str(error),
                "one_shot_consumed": True,
            }
            try:
                writer.write_json("failure.json", failure)
                writer.write_json("manifest.json", {"schema": "blindassist.taro.o0r.source_anchored_factor_canary_manifest.v1", "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written, "one_shot_root_consumed": True})
            except Exception:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true", help="process one frame without creating the evidence root")
    args = parser.parse_args()
    try:
        result = execute(smoke_only=args.smoke_only)
    except Exception as error:
        print(json.dumps({"terminal": "SOURCE_FACTOR_R1_FAILED", "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
        return 1
    if args.smoke_only:
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
