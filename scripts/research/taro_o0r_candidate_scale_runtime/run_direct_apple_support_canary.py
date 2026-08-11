#!/usr/bin/env python3
"""Run the two-phase TARO R3 direct-Apple SUPPORT canary."""

from __future__ import annotations

import argparse
import datetime as dt
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

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale, direct_apple_support, source_factor
from scripts.research.taro_o0r_candidate_scale_runtime import run_apple_support_seed_canary as r2runner
from scripts.research.taro_o0r_candidate_scale_runtime import run_source_factor_canary as r1runner
from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import load_sealed_candidate_frame
from scripts.research.taro_o0r_factor_headroom_runtime.depthart_runner import validate_candidate_input_receipt
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


EVIDENCE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r3"
R2_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-apple-support-seed-r2"
MAXIMUM_EVIDENCE_BYTES = 64 * 1024 * 1024
EXPECTED_LOST_FRAMES = 14
EXPECTED_LOST_QUERIES = 112


class DirectAppleSupportRunError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise DirectAppleSupportRunError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in output, "R3_SEAL_COLLISION", "payload already contains a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _code_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "APPLE_SCALE": Path(apple_scale.__file__).resolve(),
        "DIRECT_APPLE_SUPPORT": Path(direct_apple_support.__file__).resolve(),
        "SOURCE_FACTOR": Path(source_factor.__file__).resolve(),
        "RUNNER": Path(__file__).resolve(),
        "R1_RUNNER": Path(r1runner.__file__).resolve(),
        "R2_RUNNER": Path(r2runner.__file__).resolve(),
        "SOURCE_ADAPTER": Path(adapter.__file__).resolve(),
        "MATERIALIZER": Path(materializer.__file__).resolve(),
    }
    return {
        name: {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": materializer.sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _read_intrinsics_and_trajectory(
    row: Mapping[str, Any],
    candidate_input: Mapping[str, Any],
    video_id: str,
    token: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    containers = row["container_receipts"]
    intrinsics_receipt = containers["lowres_wide_intrinsics.zip"]
    intrinsics_path = materializer.safe_join(r1runner.SOURCE_ROOT, str(intrinsics_receipt["relative_path"]))
    materializer.verify_bound_container(intrinsics_path, intrinsics_receipt)
    member_path = f"lowres_wide_intrinsics/{video_id}_{token}.pincam"
    with zipfile.ZipFile(intrinsics_path) as bundle:
        try:
            info = bundle.getinfo(member_path)
        except KeyError as error:
            raise DirectAppleSupportRunError("R3_INTRINSICS_MEMBER_MISSING", "exact bound intrinsics member is absent", member=member_path) from error
        materializer._validate_zip_info(info)
        require(not info.is_dir() and info.file_size > 0, "R3_INTRINSICS_MEMBER_INVALID", "intrinsics member must be a non-empty file")
        payload = bundle.read(info)
    intrinsics_hash = materializer.sha256_bytes(payload)
    intrinsics_crc = materializer.crc32_bytes(payload)
    bound = candidate_input["intrinsics_member_binding"]
    require(
        len(payload) == info.file_size == bound["bytes"]
        and intrinsics_hash == bound["sha256"]
        and intrinsics_crc == f"{info.CRC:08X}" == bound["crc32"]
        and bound["container_id"] == f"sha256:{str(intrinsics_receipt['sha256']).upper()}",
        "R3_INTRINSICS_MEMBER_DRIFT",
        "raw intrinsics member differs from the sealed candidate input",
    )
    lowres = materializer.parse_pincam_payload(payload)

    trajectory_receipt = containers["lowres_wide.traj"]
    trajectory_path = materializer.safe_join(r1runner.SOURCE_ROOT, str(trajectory_receipt["relative_path"]))
    materializer.verify_bound_container(trajectory_path, trajectory_receipt)
    trajectory_payload = trajectory_path.read_bytes()
    require(
        len(trajectory_payload) == trajectory_receipt["bytes"]
        and materializer.sha256_bytes(trajectory_payload) == str(trajectory_receipt["sha256"]).upper(),
        "R3_TRAJECTORY_PAYLOAD_DRIFT",
        "raw trajectory differs from its bound container receipt",
    )
    trajectory_rows = materializer.parse_trajectory_payload(trajectory_payload)
    return lowres, trajectory_rows, {
        "intrinsics_member_sha256": intrinsics_hash,
        "intrinsics_member_crc32": intrinsics_crc,
        "trajectory_container_sha256": str(trajectory_receipt["sha256"]).upper(),
        "trajectory_payload_sha256": materializer.sha256_bytes(trajectory_payload),
    }


def _load_direct_source_inputs(
    parent_id: str,
    video_id: str,
    token: str,
    row: Mapping[str, Any],
    candidate_completion: Mapping[str, Any],
    scale_manifest: Mapping[str, Any],
) -> tuple[source_factor.PreparedSourceCandidate, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    sealed = load_sealed_candidate_frame(r1runner.FACTOR_ROOT, parent_id, video_id, token)
    candidate_record = sealed["candidate_frame_record"]
    candidate_input = validate_candidate_input_receipt(candidate_record["candidate_input_receipt"])
    replay = apple_scale.build_candidate_replay_binding(candidate_record, sealed["native_depth_m"])
    require(
        (candidate_input["parent_id"], candidate_input["video_id"], candidate_input["timestamp_token"])
        == (parent_id, video_id, token),
        "R3_CANDIDATE_IDENTITY_DRIFT",
        "sealed candidate input identity differs from the lost cohort",
    )

    container = row["container_receipts"]["upsampling.zip"]
    archive = materializer.safe_join(r1runner.SOURCE_ROOT, str(container["relative_path"]))
    apple_source = apple_scale.decode_apple_scale_source(
        archive,
        container,
        parent_id=parent_id,
        video_id=video_id,
        timestamp_token=token,
        physical_frame_id=f"{video_id}:{token}",
        frame_plan_sha256=adapter.canonical_sha256(row),
        candidate_phase_completion_sha256=candidate_completion["content_sha256"],
    )
    lowres, trajectory_rows, bindings = _read_intrinsics_and_trajectory(row, candidate_input, video_id, token)
    direct_source = direct_apple_support.build_direct_apple_source_receipt(
        candidate_input,
        apple_source["source_receipt"],
        apple_source["apple_depth_mm"],
        apple_source["confidence"],
        lowres,
        trajectory_rows,
        **bindings,
    )
    scale = r1runner._load_source_scale(parent_id, video_id, token, scale_manifest)
    prepared = direct_apple_support.prepare_direct_source_candidate(
        replay["candidate_highres_depth_m"],
        apple_source["apple_depth_mm"],
        apple_source["confidence"],
        direct_source,
        candidate_input,
        apple_source["source_receipt"],
        replay["candidate_binding"],
        scale,
    )
    return prepared, direct_source, apple_source, scale, candidate_input


def _source_failure_record(
    parent_id: str,
    source: Mapping[str, Any],
    prepared: source_factor.PreparedSourceCandidate,
    error: Exception,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": "blindassist.taro.o0r.direct_apple_support_source_failure.v1",
            "analysis_kind": direct_apple_support.ANALYSIS_KIND,
            "claim_ceiling": direct_apple_support.CLAIM_CEILING,
            "method_id": direct_apple_support.METHOD_ID,
            "parent_id": parent_id,
            "physical_frame_id": source["physical_frame_id"],
            "direct_source_receipt_sha256": source["content_sha256"],
            "source_scale_record_sha256": prepared.source_scale_record_sha256,
            "candidate_binding_sha256": prepared.candidate_binding_sha256,
            "error_code": str(getattr(error, "code", type(error).__name__)),
            "message": str(error),
            "faro_payload_read": False,
            "compact_truth_read": False,
            "query_receipt_read": False,
            "computed_before_truth_join": True,
            "unknown_preserved": True,
        }
    )


def _input_frame_cohort() -> tuple[
    list[tuple[str, str, str]],
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Select retrospective frames without decoding R1 query records."""

    require(r2runner.R1_ROOT.is_dir() and r2runner.R1A_ROOT.is_dir(), "R3_R1_INPUT_ROOT_MISSING", "R1 and R1A evidence roots must exist")
    r1_manifest = r1runner._load_json(r2runner.R1_ROOT / "manifest.json")
    r1a_manifest = r1runner._load_json(r2runner.R1A_ROOT / "manifest.json")
    diagnostics_path = r2runner._verify_external_manifest_file(r2runner.R1A_ROOT, r1a_manifest, "diagnostics.json")
    diagnostics = r1runner._load_json(diagnostics_path)
    lost_frames = diagnostics.get("extraction_lost_frames")
    require(
        isinstance(lost_frames, list)
        and len(lost_frames) == EXPECTED_LOST_FRAMES
        and sum(int(row["lost_query_count"]) for row in lost_frames) == EXPECTED_LOST_QUERIES,
        "R3_R1_LOST_COHORT_DRIFT",
        "R1A lost cohort drift",
    )
    _, _, parent_lookup, completion, truth_manifest, scale_manifest = r1runner._preflight()
    keys: list[tuple[str, str, str]] = []
    for row in lost_frames:
        parent_id = str(row["parent_id"])
        video_id, token = str(row["physical_frame_id"]).split(":", 1)
        require((parent_id, video_id) in parent_lookup, "R3_R1_LOST_COHORT_DRIFT", "lost frame is outside eval roster")
        keys.append((parent_id, video_id, token))
    require(len(set(keys)) == EXPECTED_LOST_FRAMES, "R3_R1_LOST_COHORT_DRIFT", "lost frame key duplication")
    return keys, parent_lookup, list(lost_frames), completion, truth_manifest, scale_manifest, r1_manifest


def _load_r1_lost_queries(
    lost_frames: list[dict[str, Any]],
    r1_manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Decode query diagnostics only after the source phase is sealed."""

    query_path = r2runner._verify_external_manifest_file(r2runner.R1_ROOT, r1_manifest, "query-records.json.gz")
    all_r1 = [source_factor.validate_query_record(row) for row in r1runner._load_json_gzip(query_path)]
    lost = [row for row in all_r1 if row["effects"]["extraction_lost"]]
    require(
        len(lost) == EXPECTED_LOST_QUERIES and len({row["physical_frame_id"] for row in lost}) == EXPECTED_LOST_FRAMES,
        "R3_R1_LOST_COHORT_DRIFT",
        "R1 query records disagree with R1A diagnostics",
    )
    expected_counts = {str(row["physical_frame_id"]): int(row["lost_query_count"]) for row in lost_frames}
    by_frame: dict[str, list[dict[str, Any]]] = {}
    for frame_id in sorted(expected_counts):
        rows = [row for row in lost if row["physical_frame_id"] == frame_id]
        require(len(rows) == expected_counts[frame_id], "R3_R1_LOST_COHORT_DRIFT", "per-frame lost-query count drift", frame_id=frame_id)
        by_frame[frame_id] = rows
    return by_frame


def _validate_truth_join_camera(direct_source: Mapping[str, Any], compact_source: Mapping[str, Any]) -> None:
    require(
        (direct_source["parent_id"], direct_source["video_id"], direct_source["timestamp_token"], direct_source["physical_frame_id"])
        == (compact_source["parent_id"], compact_source["session_id"], compact_source["sensor_timestamp"]["decimal_token"], compact_source["physical_frame_id"]),
        "R3_TRUTH_JOIN_IDENTITY_DRIFT",
        "compact truth source identity differs from the sealed narrow source receipt",
    )
    require(
        adapter.canonical_sha256(direct_source["intrinsics_highres"]) == adapter.canonical_sha256(compact_source["intrinsics_highres"])
        and adapter.canonical_sha256(direct_source["camera_to_world_4x4"]) == adapter.canonical_sha256(compact_source["camera_to_world_4x4"])
        and adapter.canonical_sha256(direct_source["gravity_up_camera_xyz"]) == adapter.canonical_sha256(compact_source["gravity_up_camera_xyz"]),
        "R3_TRUTH_JOIN_CAMERA_DRIFT",
        "truth-join K/pose/gravity differs from the source-only reconstruction",
    )


def _write_consumed_failure(writer: FactorEvidenceWriter, error: Exception) -> None:
    """Best-effort terminalization after the one-shot root has been created."""

    if not writer.activated or not writer.root.exists():
        return
    failure = {
        "schema": "blindassist.taro.o0r.direct_apple_support_r3_failure.v1",
        "terminal": "TARO_O0R_DIRECT_APPLE_SUPPORT_R3_EXECUTION_INVALID",
        "execution_valid": False,
        "error_code": str(getattr(error, "code", type(error).__name__)),
        "message": str(error),
        "one_shot_consumed": True,
    }
    try:
        if "failure.json" not in writer.file_receipts:
            writer.write_json("failure.json", failure)
        files_before_manifest = dict(sorted(writer.file_receipts.items()))
        if "manifest.json" not in writer.file_receipts:
            writer.write_json(
                "manifest.json",
                {
                    "schema": "blindassist.taro.o0r.direct_apple_support_r3_manifest.v1",
                    "files": files_before_manifest,
                    "file_count_before_manifest": len(files_before_manifest),
                    "bytes_before_manifest": sum(int(item["bytes"]) for item in files_before_manifest.values()),
                    "one_shot_root_consumed": True,
                },
            )
    except Exception:
        pass


def _activate_execution_writer(writer: FactorEvidenceWriter, execution_receipt: Mapping[str, Any]) -> None:
    """Activate one-shot evidence and terminalize any post-mkdir write failure."""

    try:
        writer.activate(execution_receipt)
    except Exception as error:
        _write_consumed_failure(writer, error)
        raise


def execute(*, smoke_only: bool = False, smoke_index: int = 0) -> dict[str, Any]:
    keys, parent_lookup, lost_frames, completion, truth_manifest, scale_manifest, r1_manifest = _input_frame_cohort()
    require(isinstance(smoke_index, int) and not isinstance(smoke_index, bool) and smoke_index >= 0, "R3_SMOKE_INDEX_INVALID", "smoke index must be a non-negative integer")
    require(not smoke_only or smoke_index < len(keys), "R3_SMOKE_INDEX_INVALID", "smoke index exceeds the lost-frame cohort")
    require(smoke_only or smoke_index == 0, "R3_SMOKE_INDEX_INVALID", "smoke index is only valid with --smoke")
    selected = [keys[smoke_index]] if smoke_only else keys
    if not smoke_only:
        require(not EVIDENCE_ROOT.exists(), "R3_ONE_SHOT_ROOT_COLLISION", "R3 evidence root already exists", root=str(EVIDENCE_ROOT))
    writer = None if smoke_only else FactorEvidenceWriter(EVIDENCE_ROOT, MAXIMUM_EVIDENCE_BYTES)
    started = time.monotonic()
    if writer is not None:
        _activate_execution_writer(
            writer,
            {
                "schema": "blindassist.taro.o0r.direct_apple_support_execution_start.v1",
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "claim_ceiling": direct_apple_support.CLAIM_CEILING,
                "method_id": direct_apple_support.METHOD_ID,
                "cohort": {"r1_lost_frames": EXPECTED_LOST_FRAMES, "r1_lost_queries": EXPECTED_LOST_QUERIES},
                "two_phase_truth_firewall": True,
                "phase_a_opened_source_roles": ["LOWRES_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"],
                "phase_a_compact_truth_read": False,
                "r1a_diagnostics_read_for_retrospective_cohort_selection": True,
                "r1_query_records_loaded_before_source_phase_completion": False,
                "code_bindings": _code_bindings(),
                "input_bindings": {
                    "r1_manifest_sha256": materializer.sha256_file(r2runner.R1_ROOT / "manifest.json"),
                    "r1a_manifest_sha256": materializer.sha256_file(r2runner.R1A_ROOT / "manifest.json"),
                    "r2_manifest_sha256": materializer.sha256_file(R2_ROOT / "manifest.json"),
                    "candidate_phase_completion_sha256": completion["content_sha256"],
                    "truth_manifest_sha256": materializer.sha256_file(r1runner.TRUTH_ROOT / "manifest.json"),
                    "scale_manifest_sha256": materializer.sha256_file(r1runner.SCALE_ROOT / "manifest.json"),
                },
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__, "device": "cpu"},
                "training": False,
                "network": False,
                "gpu_inference": False,
                "formal_reducer": False,
                "one_shot_consumed_on_root_creation": True,
            },
        )

    plane_records: dict[str, dict[str, Any]] = {}
    failure_records: dict[str, dict[str, Any]] = {}
    source_inputs: dict[str, tuple[source_factor.PreparedSourceCandidate, dict[str, Any], dict[str, Any]]] = {}
    try:
        for index, (parent_id, video_id, token) in enumerate(selected, start=1):
            prepared, source, apple_source, scale, _ = _load_direct_source_inputs(
                parent_id, video_id, token, parent_lookup[(parent_id, video_id)], completion, scale_manifest
            )
            frame_id = source["physical_frame_id"]
            source_inputs[frame_id] = (prepared, source, apple_source)
            try:
                plane = direct_apple_support.derive_direct_apple_support_plane(
                    prepared, apple_source["apple_depth_mm"], apple_source["confidence"], source, scale
                )
            except direct_apple_support.DirectAppleSupportError as error:
                failure_records[frame_id] = _source_failure_record(parent_id, source, prepared, error)
            else:
                plane_records[frame_id] = plane.record
            if writer is not None:
                writer.write_json(f"source-receipts/{parent_id}/{video_id}/{token}.json", source)
                writer.write_json(f"source-phase/{parent_id}/{video_id}/{token}.json", plane_records.get(frame_id, failure_records.get(frame_id)))
            print(
                json.dumps(
                    {
                        "phase": "DIRECT_APPLE_SOURCE_ONLY_SUPPORT",
                        "completed_frames": index,
                        "total_frames": len(selected),
                        "support_frames": len(plane_records),
                        "unknown_frames": len(failure_records),
                        "physical_frame_id": frame_id,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        phase_a = _seal(
            {
                "schema": "blindassist.taro.o0r.direct_apple_support_source_phase_completion.v1",
                "method_id": direct_apple_support.METHOD_ID,
                "selected_frame_count": len(selected),
                "source_support_frame_count": len(plane_records),
                "source_unknown_frame_count": len(failure_records),
                "direct_source_receipt_sha256s": [source_inputs[key][1]["content_sha256"] for key in sorted(source_inputs)],
                "plane_record_sha256s": [plane_records[key]["content_sha256"] for key in sorted(plane_records)],
                "failure_record_sha256s": [failure_records[key]["content_sha256"] for key in sorted(failure_records)],
                "all_source_records_sealed_before_truth_join": True,
                "opened_source_roles": ["LOWRES_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"],
                "faro_payload_read": False,
                "compact_truth_read": False,
                "query_receipt_read": False,
            }
        )
        if writer is not None:
            writer.write_json("source-phase-completion.json", phase_a)

        r1_by_frame = _load_r1_lost_queries(lost_frames, r1_manifest)
        all_records: list[dict[str, Any]] = []
        for index, (parent_id, video_id, token) in enumerate(selected, start=1):
            frame_id = f"{video_id}:{token}"
            prepared, direct_source, _ = source_inputs[frame_id]
            row = parent_lookup[(parent_id, video_id)]
            compact = r1runner._hydrate_truth(r1runner.TRUTH_ROOT / "truth-frames" / parent_id / video_id / f"{token}.json.gz", truth_manifest)
            compact_source = adapter._validate_base_receipt(dict(compact["source_frame_receipt"]))
            _validate_truth_join_camera(direct_source, compact_source)
            faro = r1runner._decode_bound_faro(compact, row)
            matrix = np.asarray(compact_source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
            geometry = adapter.derive_faro_geometry(faro, matrix, compact_source["gravity_up_camera_xyz"], compact_source)
            generated_queries = adapter.build_query_receipts(compact_source, geometry)
            require(
                [query["content_sha256"] for query in generated_queries] == [query["content_sha256"] for query in compact["query_receipts"]],
                "R3_QUERY_RECEIPT_RECOMPUTATION_DRIFT",
                "current query receipts differ from compact commitments",
            )
            r1_lookup = {row["query_id"]: row for row in r1_by_frame[frame_id]}
            plane = direct_apple_support.load_direct_apple_support_plane(plane_records[frame_id]) if frame_id in plane_records else None
            source_failure_code = failure_records[frame_id]["error_code"] if frame_id in failure_records else None
            frame_records: list[dict[str, Any]] = []
            for query in generated_queries:
                if query["query_id"] not in r1_lookup:
                    continue
                base = source_factor.build_query_truth_base(geometry, query)
                frame_records.append(
                    direct_apple_support.evaluate_direct_apple_query(
                        prepared,
                        matrix,
                        compact_source["gravity_up_camera_xyz"],
                        base,
                        plane,
                        r1_lookup[query["query_id"]],
                        source_failure_code=source_failure_code,
                    )
                )
            require(len(frame_records) == len(r1_lookup), "R3_QUERY_CARDINALITY_DRIFT", "lost queries were not fully reproduced", frame_id=frame_id)
            all_records.extend(frame_records)
            if writer is not None:
                writer.write_json_gzip(
                    f"truth-join/{parent_id}/{video_id}/{token}.json.gz",
                    {
                        "schema": "blindassist.taro.o0r.direct_apple_support_truth_join_frame.v1",
                        "parent_id": parent_id,
                        "physical_frame_id": frame_id,
                        "source_phase_completion_sha256": phase_a["content_sha256"],
                        "direct_source_receipt_sha256": direct_source["content_sha256"],
                        "source_record_sha256": (plane_records.get(frame_id) or failure_records[frame_id])["content_sha256"],
                        "current_faro_geometry_sha256": geometry.content_sha256,
                        "query_records": frame_records,
                    },
                )
            print(
                json.dumps(
                    {
                        "phase": "POSTHOC_TRUTH_JOIN",
                        "completed_frames": index,
                        "total_frames": len(selected),
                        "query_records": len(all_records),
                        "posthoc_evaluable_queries": sum(row["posthoc_query_comparison_evaluable"] for row in all_records),
                        "no_regret_queries": sum(row["support_no_regret_vs_r1_baseline"] for row in all_records),
                        "physical_frame_id": frame_id,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if smoke_only:
            return {
                "smoke_only": True,
                "frames": len(selected),
                "queries": len(all_records),
                "source_support_frames": len(plane_records),
                "posthoc_evaluable_queries": sum(row["posthoc_query_comparison_evaluable"] for row in all_records),
                "elapsed_seconds": time.monotonic() - started,
            }
        require(len(all_records) == EXPECTED_LOST_QUERIES, "R3_QUERY_CARDINALITY_DRIFT", "R3 did not account for every lost query")
        summary = direct_apple_support.summarize_direct_apple(all_records, [failure_records[key] for key in sorted(failure_records)])
        writer.write_json_gzip("direct-apple-query-records.json.gz", all_records)
        writer.write_json("summary.json", summary)
        result = {
            "schema": "blindassist.taro.o0r.direct_apple_support_r3_result.v1",
            "terminal": "TARO_O0R_DIRECT_APPLE_SUPPORT_R3_COMPLETE",
            "execution_valid": True,
            "scientific_status": "POST_HOC_RETROSPECTIVE_DIRECT_APPLE_SUPPORT_CANARY_ONLY",
            "claim_ceiling": direct_apple_support.CLAIM_CEILING,
            "source_phase_completion_sha256": phase_a["content_sha256"],
            "summary_sha256": summary["content_sha256"],
            "physical_frame_count": EXPECTED_LOST_FRAMES,
            "query_record_count": EXPECTED_LOST_QUERIES,
            "training_steps": 0,
            "gpu_inference_count": 0,
            "network_requests": 0,
            "formal_reducer_executed": False,
            "threshold_or_pass_fail_decision_applied": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        writer.write_json("result.json", result)
        files_before_manifest = dict(sorted(writer.file_receipts.items()))
        writer.write_json(
            "manifest.json",
            {
                "schema": "blindassist.taro.o0r.direct_apple_support_r3_manifest.v1",
                "files": files_before_manifest,
                "file_count_before_manifest": len(files_before_manifest),
                "bytes_before_manifest": sum(int(item["bytes"]) for item in files_before_manifest.values()),
                "one_shot_root_consumed": True,
            },
        )
        print(
            json.dumps(
                {
                    "terminal": result["terminal"],
                    "frames": EXPECTED_LOST_FRAMES,
                    "queries": EXPECTED_LOST_QUERIES,
                    "summary_sha256": summary["content_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result
    except Exception as error:
        if writer is not None:
            _write_consumed_failure(writer, error)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one lost frame without writing evidence")
    parser.add_argument("--smoke-index", type=int, default=0, help="zero-based lost-frame index for a smoke run")
    args = parser.parse_args()
    execute(smoke_only=args.smoke, smoke_index=args.smoke_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
