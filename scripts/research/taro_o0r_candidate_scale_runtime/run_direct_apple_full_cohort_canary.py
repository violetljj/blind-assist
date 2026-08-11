#!/usr/bin/env python3
"""Run the TARO R4 full-cohort direct Apple SUPPORT replay."""

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

from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_full_cohort as r4
from scripts.research.taro_o0r_candidate_scale_runtime import direct_apple_support as r3
from scripts.research.taro_o0r_candidate_scale_runtime import run_direct_apple_support_canary as r3runner
from scripts.research.taro_o0r_candidate_scale_runtime import run_source_factor_canary as r1runner
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


EVIDENCE_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r4-full-cohort"
R3_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r3"
MAXIMUM_EVIDENCE_BYTES = 256 * 1024 * 1024
EXPECTED_FRAMES = 171
EXPECTED_QUERIES = 1539
EXPECTED_PARENTS = 16


class FullCohortRunError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FullCohortRunError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in output, "R4_RUNNER_SEAL_COLLISION", "payload already contains a seal")
    output["content_sha256"] = adapter.canonical_sha256(output)
    return output


def _code_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "R4_FULL_COHORT": Path(r4.__file__).resolve(),
        "R3_DIRECT_SUPPORT": Path(r3.__file__).resolve(),
        "RUNNER": Path(__file__).resolve(),
        "R3_RUNNER": Path(r3runner.__file__).resolve(),
        "R1_RUNNER": Path(r1runner.__file__).resolve(),
        "SOURCE_FACTOR": Path(source_factor.__file__).resolve(),
        "SOURCE_ADAPTER": Path(adapter.__file__).resolve(),
        "MATERIALIZER": Path(materializer.__file__).resolve(),
    }
    return {
        name: {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": materializer.sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _load_all_r1_queries() -> dict[str, dict[str, dict[str, Any]]]:
    manifest = r1runner._load_json(r1runner.EVIDENCE_ROOT / "manifest.json")
    path = r1runner._verify_manifest_file(r1runner.EVIDENCE_ROOT, manifest, "query-records.json.gz")
    rows = [source_factor.validate_query_record(row) for row in r1runner._load_json_gzip(path)]
    require(
        len(rows) == EXPECTED_QUERIES
        and len({(row["physical_frame_id"], row["query_id"]) for row in rows}) == EXPECTED_QUERIES
        and len({row["physical_frame_id"] for row in rows}) == EXPECTED_FRAMES
        and len({row["parent_id"] for row in rows}) == EXPECTED_PARENTS,
        "R4_R1_QUERY_COHORT_DRIFT",
        "R1 query records do not cover the full R4 cohort",
    )
    by_frame: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(row["physical_frame_id"], {})[row["query_id"]] = row
    require(all(len(items) == 9 for items in by_frame.values()), "R4_R1_QUERY_COHORT_DRIFT", "R1 frame lacks nine queries")
    return by_frame


def _write_consumed_failure(writer: FactorEvidenceWriter, error: Exception) -> None:
    if not writer.activated or not writer.root.exists():
        return
    if "manifest.json" in writer.file_receipts:
        return
    failure = {
        "schema": "blindassist.taro.o0r.direct_apple_support_r4_full_cohort_failure.v1",
        "terminal": "TARO_O0R_DIRECT_APPLE_SUPPORT_R4_FULL_COHORT_EXECUTION_INVALID",
        "execution_valid": False,
        "error_code": str(getattr(error, "code", type(error).__name__)),
        "message": str(error),
        "one_shot_consumed": True,
    }
    try:
        if "failure.json" not in writer.file_receipts:
            writer.write_json("failure.json", failure)
        files = dict(sorted(writer.file_receipts.items()))
        if "manifest.json" not in writer.file_receipts:
            writer.write_json(
                "manifest.json",
                {
                    "schema": "blindassist.taro.o0r.direct_apple_support_r4_full_cohort_manifest.v1",
                    "files": files,
                    "file_count_before_manifest": len(files),
                    "bytes_before_manifest": sum(int(item["bytes"]) for item in files.values()),
                    "one_shot_root_consumed": True,
                },
            )
    except Exception:
        pass


def _activate_execution_writer(writer: FactorEvidenceWriter, receipt: Mapping[str, Any]) -> None:
    try:
        writer.activate(receipt)
    except Exception as error:
        _write_consumed_failure(writer, error)
        raise


def _best_effort_terminal_print(payload: Mapping[str, Any]) -> None:
    """Do not mutate committed evidence because terminal console output failed."""

    try:
        print(json.dumps(dict(payload), sort_keys=True), flush=True)
    except Exception:
        pass


def execute(*, smoke_only: bool = False, smoke_index: int = 0) -> dict[str, Any]:
    _, keys, parent_lookup, completion, truth_manifest, scale_manifest = r1runner._preflight()
    require(len(keys) == EXPECTED_FRAMES, "R4_FRAME_COHORT_DRIFT", "R4 requires all 171 R1 frames")
    require(isinstance(smoke_index, int) and not isinstance(smoke_index, bool) and smoke_index >= 0, "R4_SMOKE_INDEX_INVALID", "smoke index must be non-negative")
    require(not smoke_only or smoke_index < len(keys), "R4_SMOKE_INDEX_INVALID", "smoke index exceeds frame cohort")
    require(smoke_only or smoke_index == 0, "R4_SMOKE_INDEX_INVALID", "smoke index is only valid with --smoke")
    selected = [keys[smoke_index]] if smoke_only else keys
    if not smoke_only:
        require(not EVIDENCE_ROOT.exists(), "R4_ONE_SHOT_ROOT_COLLISION", "R4 evidence root already exists", root=str(EVIDENCE_ROOT))
    writer = None if smoke_only else FactorEvidenceWriter(EVIDENCE_ROOT, MAXIMUM_EVIDENCE_BYTES)
    started = time.monotonic()
    if writer is not None:
        _activate_execution_writer(
            writer,
            {
                "schema": "blindassist.taro.o0r.direct_apple_support_r4_full_cohort_execution_start.v1",
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "claim_ceiling": r4.CLAIM_CEILING,
                "method_id": r4.METHOD_ID,
                "cohort": {"physical_frames": EXPECTED_FRAMES, "queries": EXPECTED_QUERIES, "parents": EXPECTED_PARENTS},
                "two_phase_truth_firewall": True,
                "phase_a_opened_source_roles": ["LOWRES_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"],
                "phase_a_compact_truth_read": False,
                "r1_query_records_loaded_before_source_phase_completion": False,
                "code_bindings": _code_bindings(),
                "input_bindings": {
                    "r1_manifest_sha256": materializer.sha256_file(r1runner.EVIDENCE_ROOT / "manifest.json"),
                    "r3_direct_support_manifest_sha256": materializer.sha256_file(R3_ROOT / "manifest.json"),
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
    source_receipt_hashes: dict[str, str] = {}
    try:
        for index, (parent_id, video_id, token) in enumerate(selected, start=1):
            prepared, source, apple_source, scale, _ = r3runner._load_direct_source_inputs(
                parent_id, video_id, token, parent_lookup[(parent_id, video_id)], completion, scale_manifest
            )
            frame_id = source["physical_frame_id"]
            source_receipt_hashes[frame_id] = source["content_sha256"]
            try:
                plane = r4.derive_full_cohort_plane(
                    prepared, apple_source["apple_depth_mm"], apple_source["confidence"], source, scale
                )
            except r3.DirectAppleSupportError as error:
                failure_records[frame_id] = r4.build_source_failure_record(parent_id, source, prepared, error)
            else:
                plane_records[frame_id] = plane.record
            if writer is not None:
                writer.write_json(f"source-receipts/{parent_id}/{video_id}/{token}.json", source)
                writer.write_json(f"source-phase/{parent_id}/{video_id}/{token}.json", plane_records.get(frame_id, failure_records.get(frame_id)))
            print(
                json.dumps(
                    {
                        "phase": "FULL_COHORT_SOURCE_ONLY_SUPPORT",
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
                "schema": "blindassist.taro.o0r.direct_apple_support_r4_source_phase_completion.v1",
                "method_id": r4.METHOD_ID,
                "selected_frame_count": len(selected),
                "source_support_frame_count": len(plane_records),
                "source_unknown_frame_count": len(failure_records),
                "direct_source_receipt_sha256s": [source_receipt_hashes[key] for key in sorted(source_receipt_hashes)],
                "plane_record_sha256s": [plane_records[key]["content_sha256"] for key in sorted(plane_records)],
                "failure_record_sha256s": [failure_records[key]["content_sha256"] for key in sorted(failure_records)],
                "all_source_records_sealed_before_truth_join": True,
                "opened_source_roles": ["LOWRES_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"],
                "faro_payload_read": False,
                "compact_truth_read": False,
                "r1_query_records_read": False,
                "query_receipt_read": False,
            }
        )
        if writer is not None:
            writer.write_json("source-phase-completion.json", phase_a)

        r1_by_frame = _load_all_r1_queries()
        all_records: list[dict[str, Any]] = []
        for index, (parent_id, video_id, token) in enumerate(selected, start=1):
            frame_id = f"{video_id}:{token}"
            row = parent_lookup[(parent_id, video_id)]
            prepared, direct_source, _, _, _ = r3runner._load_direct_source_inputs(
                parent_id, video_id, token, row, completion, scale_manifest
            )
            require(direct_source["content_sha256"] == source_receipt_hashes[frame_id], "R4_SOURCE_REPLAY_DRIFT", "Phase B narrow source differs from sealed Phase A")
            compact = r1runner._hydrate_truth(r1runner.TRUTH_ROOT / "truth-frames" / parent_id / video_id / f"{token}.json.gz", truth_manifest)
            compact_source = adapter._validate_base_receipt(dict(compact["source_frame_receipt"]))
            r3runner._validate_truth_join_camera(direct_source, compact_source)
            faro = r1runner._decode_bound_faro(compact, row)
            matrix = np.asarray(compact_source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
            geometry = adapter.derive_faro_geometry(faro, matrix, compact_source["gravity_up_camera_xyz"], compact_source)
            queries = adapter.build_query_receipts(compact_source, geometry)
            require(
                [query["content_sha256"] for query in queries] == [query["content_sha256"] for query in compact["query_receipts"]],
                "R4_QUERY_RECEIPT_RECOMPUTATION_DRIFT",
                "current query receipts differ from compact commitments",
            )
            lookup = r1_by_frame[frame_id]
            plane = r4.load_full_cohort_plane(plane_records[frame_id]) if frame_id in plane_records else None
            failure_code = failure_records[frame_id]["error_code"] if frame_id in failure_records else None
            frame_records = []
            for query in queries:
                base = source_factor.build_query_truth_base(geometry, query)
                frame_records.append(
                    r4.evaluate_full_cohort_query(
                        prepared,
                        matrix,
                        compact_source["gravity_up_camera_xyz"],
                        base,
                        plane,
                        lookup[query["query_id"]],
                        current_faro_geometry_sha256=geometry.content_sha256,
                        compact_faro_geometry_sha256=compact["faro_geometry_sha256"],
                        source_failure_code=failure_code,
                    )
                )
            require(len(frame_records) == 9, "R4_QUERY_CARDINALITY_DRIFT", "full-cohort frame did not produce nine query records")
            all_records.extend(frame_records)
            if writer is not None:
                writer.write_json_gzip(
                    f"truth-join/{parent_id}/{video_id}/{token}.json.gz",
                    {
                        "schema": "blindassist.taro.o0r.direct_apple_support_r4_truth_join_frame.v1",
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
                        "phase": "FULL_COHORT_POSTHOC_TRUTH_JOIN",
                        "completed_frames": index,
                        "total_frames": len(selected),
                        "query_records": len(all_records),
                        "direct_evaluable_queries": sum(item["direct_apple_support"]["extraction_evaluable"] for item in all_records),
                        "no_regret_vs_baseline": sum(item["effects"]["support_no_regret_vs_baseline"] for item in all_records),
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
                "direct_evaluable_queries": sum(item["direct_apple_support"]["extraction_evaluable"] for item in all_records),
                "elapsed_seconds": time.monotonic() - started,
            }
        require(len(all_records) == EXPECTED_QUERIES, "R4_QUERY_CARDINALITY_DRIFT", "R4 did not account for all 1,539 queries")
        summary = r4.summarize_full_cohort(all_records, [failure_records[key] for key in sorted(failure_records)])
        writer.write_json_gzip("full-cohort-query-records.json.gz", all_records)
        writer.write_json("summary.json", summary)
        result = {
            "schema": "blindassist.taro.o0r.direct_apple_support_r4_full_cohort_result.v1",
            "terminal": "TARO_O0R_DIRECT_APPLE_SUPPORT_R4_FULL_COHORT_COMPLETE",
            "execution_valid": True,
            "scientific_status": "POST_HOC_RETROSPECTIVE_DIRECT_APPLE_SUPPORT_FULL_COHORT_MAP_ONLY",
            "claim_ceiling": r4.CLAIM_CEILING,
            "source_phase_completion_sha256": phase_a["content_sha256"],
            "summary_sha256": summary["content_sha256"],
            "physical_frame_count": EXPECTED_FRAMES,
            "query_record_count": EXPECTED_QUERIES,
            "parent_count": EXPECTED_PARENTS,
            "training_steps": 0,
            "gpu_inference_count": 0,
            "network_requests": 0,
            "formal_reducer_executed": False,
            "threshold_or_pass_fail_decision_applied": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        writer.write_json("result.json", result)
        files = dict(sorted(writer.file_receipts.items()))
        writer.write_json(
            "manifest.json",
            {
                "schema": "blindassist.taro.o0r.direct_apple_support_r4_full_cohort_manifest.v1",
                "files": files,
                "file_count_before_manifest": len(files),
                "bytes_before_manifest": sum(int(item["bytes"]) for item in files.values()),
                "one_shot_root_consumed": True,
            },
        )
        _best_effort_terminal_print(
            {"terminal": result["terminal"], "frames": EXPECTED_FRAMES, "queries": EXPECTED_QUERIES, "summary_sha256": summary["content_sha256"]}
        )
        return result
    except Exception as error:
        if writer is not None:
            _write_consumed_failure(writer, error)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one frame without writing evidence")
    parser.add_argument("--smoke-index", type=int, default=0, help="zero-based full-cohort frame index for smoke")
    args = parser.parse_args()
    execute(smoke_only=args.smoke, smoke_index=args.smoke_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
