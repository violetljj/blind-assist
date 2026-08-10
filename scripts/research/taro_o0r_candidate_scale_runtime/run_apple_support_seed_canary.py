#!/usr/bin/env python3
"""Run the two-phase TARO R2 Apple-seeded support recovery canary."""

from __future__ import annotations

import argparse
import datetime as dt
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
import scipy

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale, apple_support_seed, source_factor
from scripts.research.taro_o0r_candidate_scale_runtime import run_source_factor_canary as r1runner
from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import load_sealed_candidate_frame
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


R1_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-factor-r1"
R1A_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-factor-r1a-reconciliation"
EVIDENCE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-apple-support-seed-r2"
MAXIMUM_EVIDENCE_BYTES = 64 * 1024 * 1024
EXPECTED_LOST_FRAMES = 14
EXPECTED_LOST_QUERIES = 112


class AppleSupportSeedRunError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise AppleSupportSeedRunError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in output, "R2_SEAL_COLLISION", "payload already contains a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _code_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "APPLE_SCALE": Path(apple_scale.__file__).resolve(),
        "APPLE_SUPPORT_SEED": Path(apple_support_seed.__file__).resolve(),
        "SOURCE_FACTOR": Path(source_factor.__file__).resolve(),
        "RUNNER": Path(__file__).resolve(),
        "R1_RUNNER": Path(r1runner.__file__).resolve(),
        "SOURCE_ADAPTER": Path(adapter.__file__).resolve(),
        "MATERIALIZER": Path(materializer.__file__).resolve(),
    }
    return {
        name: {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": materializer.sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _verify_external_manifest_file(root: Path, manifest: Mapping[str, Any], relative: str) -> Path:
    receipt = manifest.get("files", {}).get(relative)
    require(isinstance(receipt, dict) and receipt.get("path") == relative, "R2_INPUT_MANIFEST_FILE_MISSING", "input manifest lacks a file", relative=relative)
    path = materializer.safe_join(root, relative)
    require(path.is_file() and path.stat().st_size == receipt.get("bytes") and materializer.sha256_file(path) == receipt.get("sha256"), "R2_INPUT_MANIFEST_FILE_DRIFT", "input file differs from manifest", relative=relative)
    return path


def _load_source_metadata(parent_id: str, video_id: str, token: str, truth_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the inline source receipt; do not hydrate FARO/query arrays."""

    relative = f"truth-frames/{parent_id}/{video_id}/{token}.json.gz"
    path = r1runner._verify_manifest_file(r1runner.TRUTH_ROOT, truth_manifest, relative)
    package = materializer.validate_sealed_record(r1runner._load_json_gzip(path), materializer.ARRAY_ARTIFACT_SCHEMA, "R2_SOURCE_METADATA_PACKAGE_DRIFT")
    require(package["array_count"] == 0 and package["array_blob_reference_count"] == 0 and package["blob_file_count"] == 0, "R2_SOURCE_METADATA_ARRAY_REFERENCE_FORBIDDEN", "phase A source metadata unexpectedly requires truth arrays")
    payload = package["payload_with_array_refs"]
    require(adapter.canonical_sha256(payload) == package["artifact_canonical_sha256"], "R2_SOURCE_METADATA_ARTIFACT_DRIFT", "inline source package canonical hash drift")
    source = adapter._validate_base_receipt(dict(payload["source_frame_receipt"]))
    require((source["parent_id"], source["session_id"], source["sensor_timestamp"]["decimal_token"]) == (parent_id, video_id, token), "R2_SOURCE_METADATA_IDENTITY_DRIFT", "source metadata identity drift")
    return source


def _load_source_inputs(
    parent_id: str,
    video_id: str,
    token: str,
    row: Mapping[str, Any],
    candidate_completion: Mapping[str, Any],
    truth_manifest: Mapping[str, Any],
    scale_manifest: Mapping[str, Any],
) -> tuple[source_factor.PreparedSourceCandidate, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _load_source_metadata(parent_id, video_id, token, truth_manifest)
    sealed = load_sealed_candidate_frame(r1runner.FACTOR_ROOT, parent_id, video_id, token)
    replay = apple_scale.build_candidate_replay_binding(sealed["candidate_frame_record"], sealed["native_depth_m"])
    container = row["container_receipts"]["upsampling.zip"]
    archive = materializer.safe_join(r1runner.SOURCE_ROOT, str(container["relative_path"]))
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
    scale = r1runner._load_source_scale(parent_id, video_id, token, scale_manifest)
    prepared = source_factor.prepare_source_anchored_candidate(
        replay["candidate_highres_depth_m"],
        apple_source["apple_depth_mm"],
        apple_source["confidence"],
        source,
        apple_source["source_receipt"],
        replay["candidate_binding"],
        scale,
    )
    return prepared, source, apple_source, scale


def _input_cohort() -> tuple[
    list[tuple[str, str, str]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    require(R1_ROOT.is_dir() and R1A_ROOT.is_dir(), "R2_R1_INPUT_ROOT_MISSING", "R1 and R1A evidence roots must exist")
    r1_manifest = r1runner._load_json(R1_ROOT / "manifest.json")
    r1a_manifest = r1runner._load_json(R1A_ROOT / "manifest.json")
    query_path = _verify_external_manifest_file(R1_ROOT, r1_manifest, "query-records.json.gz")
    diagnostics_path = _verify_external_manifest_file(R1A_ROOT, r1a_manifest, "diagnostics.json")
    diagnostics = r1runner._load_json(diagnostics_path)
    lost_frames = diagnostics.get("extraction_lost_frames")
    require(isinstance(lost_frames, list) and len(lost_frames) == EXPECTED_LOST_FRAMES and sum(int(row["lost_query_count"]) for row in lost_frames) == EXPECTED_LOST_QUERIES, "R2_R1_LOST_COHORT_DRIFT", "R1A lost cohort drift")
    all_r1 = [source_factor.validate_query_record(row) for row in r1runner._load_json_gzip(query_path)]
    lost = [row for row in all_r1 if row["effects"]["extraction_lost"]]
    require(len(lost) == EXPECTED_LOST_QUERIES and len({row["physical_frame_id"] for row in lost}) == EXPECTED_LOST_FRAMES, "R2_R1_LOST_COHORT_DRIFT", "R1 query records disagree with R1A diagnostics")
    expected_counts = {str(row["physical_frame_id"]): int(row["lost_query_count"]) for row in lost_frames}
    by_frame: dict[str, list[dict[str, Any]]] = {}
    for frame_id in sorted(expected_counts):
        rows = [row for row in lost if row["physical_frame_id"] == frame_id]
        require(len(rows) == expected_counts[frame_id], "R2_R1_LOST_COHORT_DRIFT", "per-frame lost-query count drift", frame_id=frame_id)
        by_frame[frame_id] = rows
    _, _, parent_lookup, completion, truth_manifest, scale_manifest = r1runner._preflight()
    keys: list[tuple[str, str, str]] = []
    for row in lost_frames:
        parent_id = str(row["parent_id"])
        video_id, token = str(row["physical_frame_id"]).split(":", 1)
        require((parent_id, video_id) in parent_lookup, "R2_R1_LOST_COHORT_DRIFT", "lost frame is outside eval roster")
        keys.append((parent_id, video_id, token))
    require(len(set(keys)) == EXPECTED_LOST_FRAMES, "R2_R1_LOST_COHORT_DRIFT", "lost frame key duplication")
    return keys, parent_lookup, by_frame, completion, truth_manifest, scale_manifest


def _source_failure_record(
    parent_id: str,
    source: Mapping[str, Any],
    prepared: source_factor.PreparedSourceCandidate,
    error: Exception,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": "blindassist.taro.o0r.apple_seeded_support_source_failure.v1",
            "analysis_kind": apple_support_seed.ANALYSIS_KIND,
            "claim_ceiling": apple_support_seed.CLAIM_CEILING,
            "seed_id": apple_support_seed.SEED_ID,
            "parent_id": parent_id,
            "physical_frame_id": source["physical_frame_id"],
            "source_frame_receipt_sha256": source["content_sha256"],
            "source_scale_record_sha256": prepared.source_scale_record_sha256,
            "candidate_binding_sha256": prepared.candidate_binding_sha256,
            "error_code": str(getattr(error, "code", type(error).__name__)),
            "message": str(error),
            "faro_payload_read": False,
            "query_receipt_read": False,
            "computed_before_truth_join": True,
            "unknown_preserved": True,
        }
    )


def execute(*, smoke_only: bool = False) -> dict[str, Any]:
    keys, parent_lookup, r1_by_frame, completion, truth_manifest, scale_manifest = _input_cohort()
    selected = keys[:1] if smoke_only else keys
    if not smoke_only:
        require(not EVIDENCE_ROOT.exists(), "R2_ONE_SHOT_ROOT_COLLISION", "R2 evidence root already exists", root=str(EVIDENCE_ROOT))
    writer = None if smoke_only else FactorEvidenceWriter(EVIDENCE_ROOT, MAXIMUM_EVIDENCE_BYTES)
    started = time.monotonic()
    if writer is not None:
        writer.activate(
            {
                "schema": "blindassist.taro.o0r.apple_seeded_support_execution_start.v1",
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "claim_ceiling": apple_support_seed.CLAIM_CEILING,
                "seed_id": apple_support_seed.SEED_ID,
                "cohort": {"r1_lost_frames": EXPECTED_LOST_FRAMES, "r1_lost_queries": EXPECTED_LOST_QUERIES},
                "two_phase_truth_firewall": True,
                "code_bindings": _code_bindings(),
                "input_bindings": {
                    "r1_manifest_sha256": materializer.sha256_file(R1_ROOT / "manifest.json"),
                    "r1a_manifest_sha256": materializer.sha256_file(R1A_ROOT / "manifest.json"),
                    "r1_query_records_sha256": materializer.sha256_file(R1_ROOT / "query-records.json.gz"),
                    "r1a_diagnostics_sha256": materializer.sha256_file(R1A_ROOT / "diagnostics.json"),
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
            }
        )

    # Phase A is completed for every selected frame before Phase B opens FARO
    # or accesses query receipts.
    plane_records: dict[str, dict[str, Any]] = {}
    failure_records: dict[str, dict[str, Any]] = {}
    try:
        for index, (parent_id, video_id, token) in enumerate(selected, start=1):
            row = parent_lookup[(parent_id, video_id)]
            prepared, source, apple_source, scale = _load_source_inputs(parent_id, video_id, token, row, completion, truth_manifest, scale_manifest)
            try:
                plane = apple_support_seed.derive_apple_seeded_candidate_plane(prepared, apple_source["apple_depth_mm"], apple_source["confidence"], source, scale)
            except apple_support_seed.AppleSupportSeedError as error:
                failure_records[source["physical_frame_id"]] = _source_failure_record(parent_id, source, prepared, error)
            else:
                plane_records[source["physical_frame_id"]] = plane.record
            if writer is not None:
                relative = f"source-phase/{parent_id}/{video_id}/{token}.json"
                writer.write_json(relative, plane_records.get(source["physical_frame_id"], failure_records.get(source["physical_frame_id"])))
            print(json.dumps({"phase": "SOURCE_ONLY_PLANE", "completed_frames": index, "total_frames": len(selected), "recovered_frames": len(plane_records), "unknown_frames": len(failure_records), "physical_frame_id": source["physical_frame_id"], "elapsed_seconds": round(time.monotonic() - started, 3)}, sort_keys=True), flush=True)
        phase_a = _seal(
            {
                "schema": "blindassist.taro.o0r.apple_seeded_support_source_phase_completion.v1",
                "seed_id": apple_support_seed.SEED_ID,
                "selected_frame_count": len(selected),
                "source_recovered_frame_count": len(plane_records),
                "source_unknown_frame_count": len(failure_records),
                "plane_record_sha256s": [plane_records[key]["content_sha256"] for key in sorted(plane_records)],
                "failure_record_sha256s": [failure_records[key]["content_sha256"] for key in sorted(failure_records)],
                "all_source_records_sealed_before_truth_join": True,
                "faro_payload_read": False,
                "query_receipt_read": False,
            }
        )
        if writer is not None:
            writer.write_json("source-phase-completion.json", phase_a)

        all_records: list[dict[str, Any]] = []
        for index, (parent_id, video_id, token) in enumerate(selected, start=1):
            frame_id = f"{video_id}:{token}"
            row = parent_lookup[(parent_id, video_id)]
            prepared, source, _, _ = _load_source_inputs(parent_id, video_id, token, row, completion, truth_manifest, scale_manifest)
            compact = r1runner._hydrate_truth(r1runner.TRUTH_ROOT / "truth-frames" / parent_id / video_id / f"{token}.json.gz", truth_manifest)
            faro = r1runner._decode_bound_faro(compact, row)
            matrix = np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
            geometry = adapter.derive_faro_geometry(faro, matrix, source["gravity_up_camera_xyz"], source)
            generated_queries = adapter.build_query_receipts(source, geometry)
            require([query["content_sha256"] for query in generated_queries] == [query["content_sha256"] for query in compact["query_receipts"]], "R2_QUERY_RECEIPT_RECOMPUTATION_DRIFT", "current query receipts differ from compact commitments")
            r1_lookup = {row["query_id"]: row for row in r1_by_frame[frame_id]}
            plane = apple_support_seed.load_apple_seeded_candidate_plane(plane_records[frame_id]) if frame_id in plane_records else None
            source_failure_code = failure_records[frame_id]["error_code"] if frame_id in failure_records else None
            frame_records: list[dict[str, Any]] = []
            for query in generated_queries:
                if query["query_id"] not in r1_lookup:
                    continue
                base = source_factor.build_query_truth_base(geometry, query)
                frame_records.append(
                    apple_support_seed.evaluate_recovery_query(
                        prepared,
                        matrix,
                        source["gravity_up_camera_xyz"],
                        base,
                        plane,
                        r1_lookup[query["query_id"]],
                        source_failure_code=source_failure_code,
                    )
                )
            require(len(frame_records) == len(r1_lookup), "R2_QUERY_CARDINALITY_DRIFT", "lost queries were not fully reproduced", frame_id=frame_id)
            all_records.extend(frame_records)
            if writer is not None:
                writer.write_json_gzip(
                    f"truth-join/{parent_id}/{video_id}/{token}.json.gz",
                    {
                        "schema": "blindassist.taro.o0r.apple_seeded_support_truth_join_frame.v1",
                        "parent_id": parent_id,
                        "physical_frame_id": frame_id,
                        "source_phase_completion_sha256": phase_a["content_sha256"],
                        "source_record_sha256": (plane_records.get(frame_id) or failure_records[frame_id])["content_sha256"],
                        "current_faro_geometry_sha256": geometry.content_sha256,
                        "query_records": frame_records,
                    },
                )
            print(json.dumps({"phase": "POSTHOC_TRUTH_JOIN", "completed_frames": index, "total_frames": len(selected), "query_records": len(all_records), "posthoc_evaluable_queries": sum(row["posthoc_query_comparison_evaluable"] for row in all_records), "no_regret_queries": sum(row["support_no_regret_vs_r1_baseline"] for row in all_records), "physical_frame_id": frame_id, "elapsed_seconds": round(time.monotonic() - started, 3)}, sort_keys=True), flush=True)

        if smoke_only:
            return {"smoke_only": True, "frames": len(selected), "queries": len(all_records), "source_recovered_frames": len(plane_records), "posthoc_evaluable_queries": sum(row["posthoc_query_comparison_evaluable"] for row in all_records), "elapsed_seconds": time.monotonic() - started}
        require(len(all_records) == EXPECTED_LOST_QUERIES, "R2_QUERY_CARDINALITY_DRIFT", "R2 did not account for every lost query")
        summary = apple_support_seed.summarize_recovery(all_records, [failure_records[key] for key in sorted(failure_records)])
        writer.write_json_gzip("recovery-query-records.json.gz", all_records)
        writer.write_json("summary.json", summary)
        result = {
            "schema": "blindassist.taro.o0r.apple_seeded_support_recovery_r2_result.v1",
            "terminal": "TARO_O0R_APPLE_SEEDED_SUPPORT_RECOVERY_R2_COMPLETE",
            "execution_valid": True,
            "scientific_status": "POST_HOC_RETROSPECTIVE_SOURCE_ONLY_SUPPORT_RECOVERY_CANARY_ONLY",
            "claim_ceiling": apple_support_seed.CLAIM_CEILING,
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
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o0r.apple_seeded_support_recovery_r2_manifest.v1", "files": files_before_manifest, "file_count_before_manifest": len(files_before_manifest), "bytes_before_manifest": sum(int(item["bytes"]) for item in files_before_manifest.values()), "one_shot_root_consumed": True})
        print(json.dumps({"terminal": result["terminal"], "frames": EXPECTED_LOST_FRAMES, "queries": EXPECTED_LOST_QUERIES, "summary_sha256": summary["content_sha256"]}, sort_keys=True), flush=True)
        return result
    except Exception as error:
        if writer is not None:
            failure = {"schema": "blindassist.taro.o0r.apple_seeded_support_recovery_r2_failure.v1", "terminal": "TARO_O0R_APPLE_SEEDED_SUPPORT_RECOVERY_R2_EXECUTION_INVALID", "execution_valid": False, "error_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "one_shot_consumed": True}
            try:
                writer.write_json("failure.json", failure)
                writer.write_json("manifest.json", {"schema": "blindassist.taro.o0r.apple_seeded_support_recovery_r2_manifest.v1", "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written, "one_shot_root_consumed": True})
            except Exception:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one lost frame without writing evidence")
    args = parser.parse_args()
    execute(smoke_only=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
