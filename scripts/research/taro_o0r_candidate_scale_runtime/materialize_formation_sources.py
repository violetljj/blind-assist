#!/usr/bin/env python3
"""Materialize a complete 450-frame source-receipt layer for TARO formation replay.

Existing R3 source receipts are reused exactly.  Only plan frames lacking an R3
record are decoded from the already bound local containers.  No model output,
candidate, factor, metric, prior outcome, or R6 untouched parent is read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import formation_replay_io as formation_io
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


RESULT_SCHEMA = "blindassist.taro.o0r.r6_formation_source_materialization_result.v1"
ROOT_NAME = "o0r-arkitscenes-formation-source-r0"


class FormationSourceError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FormationSourceError(code, message, **context)


def _load_plan(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise FormationSourceError("FORMATION_PLAN_READ_FAILED", "formation exact plan cannot be read") from error
    require(isinstance(value, list) and len(value) == 24, "FORMATION_PLAN_INVALID", "formation exact plan must contain 24 parents")
    return value


def _member(bundle: zipfile.ZipFile, role: str, source_path: str, token: str) -> materializer.MemberBinding:
    try:
        info = bundle.getinfo(source_path)
    except KeyError as error:
        raise FormationSourceError("FORMATION_SOURCE_MEMBER_MISSING", "formation exact source member is missing", role=role, path=source_path) from error
    require(not info.is_dir() and info.file_size > 0 and "\\" not in info.filename, "FORMATION_SOURCE_MEMBER_INVALID", "formation source member is invalid", role=role)
    suffix = ".pincam" if role == "intrinsics" else ".png"
    return materializer.MemberBinding(
        role=role,
        timestamp_token=token,
        source_member_path=info.filename,
        canonical_member_path=f"{role}/{token}{suffix}",
        bytes=int(info.file_size),
        crc32=f"{info.CRC:08X}",
    )


def _decode_missing_source(
    row: Mapping[str, Any],
    token: str,
    source_root: Path,
    trajectory_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    parent = {key: str(row["parent"][key]) for key in ("role", "visit_id", "video_id", "official_fold")}
    video_id = parent["video_id"]
    receipts = row["container_receipts"]
    up_path = materializer.safe_join(source_root, receipts["upsampling.zip"]["relative_path"])
    intr_path = materializer.safe_join(source_root, receipts["lowres_wide_intrinsics.zip"]["relative_path"])
    traj_path = materializer.safe_join(source_root, receipts["lowres_wide.traj"]["relative_path"])
    with zipfile.ZipFile(up_path) as up_bundle, zipfile.ZipFile(intr_path) as intr_bundle:
        upsampling = {
            role: {
                token: _member(
                    up_bundle,
                    role,
                    f"{video_id}/{directory}/{video_id}_{token}.png",
                    token,
                )
            }
            for role, directory in (
                ("color", "wide"),
                ("highres_depth", "highres_depth"),
                ("lowres_depth", "lowres_depth"),
                ("confidence", "confidence"),
            )
        }
        intrinsics = {
            token: _member(
                intr_bundle,
                "intrinsics",
                f"lowres_wide_intrinsics/{video_id}_{token}.pincam",
                token,
            )
        }
    try:
        frame = materializer.decode_source_frame(
            parent=parent,
            timestamp_token=token,
            upsampling_archive=up_path,
            intrinsics_archive=intr_path,
            trajectory_path=traj_path,
            upsampling_inventory=upsampling,
            intrinsics_inventory=intrinsics,
            trajectory_rows=trajectory_rows,
            container_receipts=receipts,
        )
    except (materializer.MaterializerError, adapter.AdapterError) as error:
        raise FormationSourceError(error.code, str(error), **error.context) from error
    return {
        "source_role": parent["role"],
        "source_frame_receipt": frame["source_frame_receipt"],
        "bound_source_frame_envelope": frame["bound_source_frame_envelope"],
        "source_origin": "DETERMINISTIC_BOUND_CONTAINER_RECONSTRUCTION",
        "model_outputs_absent": True,
    }


def _existing_source(predecessor_root: Path, role: str, parent_id: str, video_id: str, token: str) -> dict[str, Any] | None:
    try:
        _, _, record = formation_io._load_predecessor_record(predecessor_root, role, parent_id, video_id, token)
    except formation_io.FormationReplayIOError as error:
        if error.code == "FORMATION_PREDECESSOR_MISSING":
            return None
        raise
    return {
        "source_role": role,
        "source_frame_receipt": record["source_frame_receipt"],
        "bound_source_frame_envelope": record["bound_source_frame_envelope"],
        "source_origin": "VALIDATED_R3_PREDECESSOR_REUSE",
        "model_outputs_absent": True,
    }


def execute(
    *,
    frame_plan_path: Path,
    predecessor_root: Path,
    source_root: Path,
    output_root: Path,
    maximum_evidence_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    started = time.monotonic()
    plan_path = frame_plan_path.resolve()
    predecessor = predecessor_root.resolve()
    source = source_root.resolve()
    output = output_root.resolve()
    require(output.name == ROOT_NAME and not output.exists(), "FORMATION_SOURCE_ROOT_INVALID", "formation source output root must be the absent frozen root")
    plan = _load_plan(plan_path)
    writer = FactorEvidenceWriter(output, maximum_evidence_bytes)
    writer.activate(
        {
            "schema": "blindassist.taro.o0r.r6_formation_source_materialization_start.v1",
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "frame_plan_sha256": materializer.sha256_file(plan_path),
            "user_source_truth_only_authorization_applied": True,
            "candidate_reads": 0,
            "model_outputs": 0,
            "r6_untouched_parent_reads": 0,
        }
    )
    origins: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    source_hashes: list[str] = []
    observed_roster: list[tuple[str, str]] = []
    for row in plan:
        parent = row["parent"]
        role = str(parent["role"])
        parent_id = str(parent["visit_id"])
        video_id = str(parent["video_id"])
        adapter._validate_roster_identity(role, parent_id, video_id)
        observed_roster.append((parent_id, video_id))
        receipts = row["container_receipts"]
        trajectory_path = materializer.safe_join(source, receipts["lowres_wide.traj"]["relative_path"])
        materializer.verify_bound_container(trajectory_path, receipts["lowres_wide.traj"])
        trajectory_rows = materializer.parse_trajectory_payload(trajectory_path.read_bytes())
        for token in row["frame_plan"]["exact_timestamp_tokens"]:
            record = _existing_source(predecessor, role, parent_id, video_id, token)
            if record is None:
                record = _decode_missing_source(row, token, source, trajectory_rows)
            validated_source = adapter._validate_base_receipt(record["source_frame_receipt"])
            validated_envelope = materializer.validate_bound_source_frame_envelope(record["bound_source_frame_envelope"], validated_source)
            require(
                validated_source["source_role"] == role
                and validated_source["physical_frame_id"] == f"{video_id}:{token}"
                and validated_envelope["source_frame_receipt_sha256"] == validated_source["content_sha256"],
                "FORMATION_SOURCE_IDENTITY_DRIFT",
                "formation materialized source identity drift",
            )
            writer.write_json_gzip(f"source-frames/all/{parent_id}/{video_id}/{token}.json.gz", record)
            origins[record["source_origin"]] += 1
            parent_counts[parent_id] += 1
            role_counts[role] += 1
            source_hashes.append(validated_source["content_sha256"])
    require(observed_roster == list(formation_io.FORMATION_ROSTER), "FORMATION_SOURCE_ROSTER_DRIFT", "formation source roster/order drift")
    require(sum(parent_counts.values()) == 450 and len(parent_counts) == 24 and dict(role_counts) == formation_io.EXPECTED_ROLE_FRAME_COUNTS, "FORMATION_SOURCE_COUNT_DRIFT", "formation source output is not exact 24/450")
    result = {
        "schema": RESULT_SCHEMA,
        "terminal": "TARO_O0R_R6_FORMATION_SOURCE_450_MATERIALIZATION_PASS",
        "passed": True,
        "parent_count": len(parent_counts),
        "frame_count": sum(parent_counts.values()),
        "role_frame_counts": dict(sorted(role_counts.items())),
        "parent_frame_counts": dict(sorted(parent_counts.items())),
        "source_origin_counts": dict(sorted(origins.items())),
        "source_receipt_hash_sequence_sha256": adapter.canonical_sha256(source_hashes),
        "candidate_reads": 0,
        "model_outputs": 0,
        "factor_outputs": 0,
        "task_metric_reads": 0,
        "prior_outcome_reads": 0,
        "r6_untouched_parent_reads": 0,
        "training_steps": 0,
        "network_requests": 0,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "claim_ceiling": "Complete hash-bound source receipt layer for non-promotable WILD_LAB formation replay only.",
    }
    writer.write_json("result.json", result)
    writer.write_json(
        "manifest.json",
        {
            "schema": "blindassist.taro.o0r.r6_formation_source_materialization_manifest.v1",
            "files": dict(sorted(writer.file_receipts.items())),
            "file_count_before_manifest": len(writer.file_receipts),
            "bytes_before_manifest": writer.bytes_written,
            "terminal": result["terminal"],
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-plan", type=Path, required=True)
    parser.add_argument("--predecessor-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(
            frame_plan_path=args.frame_plan,
            predecessor_root=args.predecessor_root,
            source_root=args.source_root,
            output_root=args.output_root,
        )
    except Exception as error:
        print(json.dumps({"status": "FORMATION_SOURCE_EXECUTION_INVALID", "error_code": getattr(error, "code", type(error).__name__), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
