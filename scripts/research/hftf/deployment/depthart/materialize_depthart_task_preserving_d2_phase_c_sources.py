#!/usr/bin/env python3
"""Materialize exact D2 Phase-C source members without image decode or model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
SPATIAL_HELPER_ROOT = REPO_ROOT / "scripts" / "research" / "spatial_calibration_head_r1"
sys.path.insert(0, str(SPATIAL_HELPER_ROOT))

from scripts.research.spatial_calibration_head_r1.download_locked_assets import (  # noqa: E402
    download_file,
    extract_named_members,
)


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_body_execution_protocol_v1"
SCOPE_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_body_materialization_scope_protocol_v1"
MANIFEST_INPUT_SCHEMA = "blindassist_depthart_task_preserving_d2r1_manifest_v1"
RGB_HEAD_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_rgb_head_result_v1"
SUPPORT_HEAD_SCHEMA = "blindassist_depthart_task_preserving_d2r1_asset_header_preflight_v1"
PHASE_A_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_a_manifest_v1"
RECEIPT_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_body_scope_receipt_v1"
CHECKPOINT_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_identity_source_checkpoint_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_source_manifest_v1"
ASSET_SPECS = {
    "lowres_wide.zip": ((".png", ".jpg", ".jpeg"), "lowres_wide"),
    "lowres_wide_intrinsics.zip": ((".pincam",), "lowres_wide_intrinsics"),
    "lowres_depth.zip": ((".png",), "lowres_depth"),
    "confidence.zip": ((".png",), "confidence"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def archive_member_map(archive: Path, suffixes: tuple[str, ...]) -> tuple[dict[str, str], int]:
    mapping: dict[str, str] = {}
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        require(bad is None, f"ZIP CRC failure: {archive}: {bad}")
        infos = bundle.infolist()
        for info in infos:
            pure = Path(info.filename)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe ZIP member: {info.filename}")
            if pure.suffix.lower() not in suffixes:
                continue
            require(pure.stem not in mapping, f"duplicate source stem: {pure.stem}")
            mapping[pure.stem] = info.filename
    require(mapping, f"no matching source members: {archive}")
    return mapping, len(infos)


def role_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "role": str(row["role"]),
            "role_order": int(row["role_order"]),
            "phase_a_order": int(row["phase_a_order"]),
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "selected_frame_stems": [str(value) for value in row["selected_frame_stems"]],
        }
        for row in manifest["role_assignments"]
    ]
    require(len(rows) == 8, "role count drift")
    require([row["role"] for row in rows] == ["D2_TRAIN"] * 4 + ["D2_DEVELOPMENT_SEALED"] * 4, "role order drift")
    require(all(len(row["selected_frame_stems"]) == 300 and len(set(row["selected_frame_stems"])) == 300 for row in rows), "selected stem drift")
    return rows


def asset_lookup(rgb_head: dict[str, Any], support_head: dict[str, Any], video_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = list(rgb_head["assets"]) + [
        row
        for row in support_head["assets"]
        if str(row["video_id"]) in video_ids and str(row["asset"]) != "lowres_wide.traj"
    ]
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = str(row["video_id"]), str(row["asset"])
        require(key not in lookup, f"duplicate asset row: {key}")
        require(row["http_status"] == 200 and int(row["content_length_bytes"]) > 0, f"unavailable asset: {key}")
        lookup[key] = row
    require(len(lookup) == 32, "asset lookup count drift")
    return lookup


def trajectory_lookup(phase_a: dict[str, Any], video_ids: set[str]) -> dict[str, dict[str, Any]]:
    result = {
        str(row["video_id"]): row["trajectory"]
        for row in phase_a["processed"]
        if str(row["video_id"]) in video_ids
    }
    require(len(result) == 8, "trajectory lookup count drift")
    for video_id, row in result.items():
        path = Path(row["path"])
        require(path.is_file(), f"trajectory missing: {video_id}")
        require(path.stat().st_size == int(row["bytes"]), f"trajectory bytes drift: {video_id}")
        require(sha256_file(path) == row["sha256"], f"trajectory SHA drift: {video_id}")
    return result


def safe_remove_tree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    require(resolved_root in resolved.parents and resolved != resolved_root, f"unsafe scoped cleanup: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def write_checkpoint(path: Path, value: dict[str, Any]) -> None:
    payload = json_bytes(value)
    write_json_exclusive(path, value)
    write_json_exclusive(
        path.with_suffix(".sha256.json"),
        {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()},
    )


def read_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    seal = load_json(path.with_suffix(".sha256.json"))
    require(path.stat().st_size == int(seal["bytes"]) and sha256_file(path) == seal["sha256"], f"checkpoint seal drift: {path}")
    value = load_json(path)
    require(value.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema drift")
    for key in ("role", "role_order", "phase_a_order", "pool_order", "visit_id", "video_id"):
        require(str(value[key]) == str(expected[key]), f"checkpoint identity drift: {key}")
    require(value["selected_frame_stems"] == expected["selected_frame_stems"], "checkpoint stem drift")
    for entries in value["extracted"].values():
        require(len(entries) == 300, "checkpoint extracted count drift")
        for entry in entries:
            extracted = Path(entry["path"])
            require(extracted.stat().st_size == int(entry["bytes"]), f"extracted bytes drift: {extracted}")
            require(sha256_file(extracted) == entry["sha256"], f"extracted SHA drift: {extracted}")
    require(value["image_decode"] is False and value["model_output_read"] is False, "checkpoint authority drift")
    return value


def expected_attempt(
    protocol_path: Path,
    scope_path: Path,
    manifest_path: Path,
    rgb_head_path: Path,
    support_head_path: Path,
    phase_a_path: Path,
    receipt_path: Path,
    roles: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "blindassist_depthart_task_preserving_d2_phase_c_source_attempt_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "scope_protocol_sha256": sha256_file(scope_path),
        "d2r1_manifest_sha256": sha256_file(manifest_path),
        "rgb_head_result_sha256": sha256_file(rgb_head_path),
        "support_head_result_sha256": sha256_file(support_head_path),
        "phase_a_manifest_sha256": sha256_file(phase_a_path),
        "license_receipt_sha256": sha256_file(receipt_path),
        "roles": roles,
        "assets_per_identity": list(ASSET_SPECS),
        "image_decode": False,
        "truth_derivation": False,
        "model_output_read": False,
        "training_executed": False,
        "r2_cohort_access": "NONE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scope-protocol", type=Path, required=True)
    parser.add_argument("--d2r1-manifest", type=Path, required=True)
    parser.add_argument("--rgb-head-result", type=Path, required=True)
    parser.add_argument("--support-head-result", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    scope = load_json(args.scope_protocol)
    manifest_input = load_json(args.d2r1_manifest)
    rgb_head = load_json(args.rgb_head_result)
    support_head = load_json(args.support_head_result)
    phase_a = load_json(args.phase_a_manifest)
    receipt = load_json(args.license_receipt)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(scope.get("schema") == SCOPE_SCHEMA, "scope schema drift")
    require(manifest_input.get("schema") == MANIFEST_INPUT_SCHEMA, "D2R1 manifest schema drift")
    require(rgb_head.get("schema") == RGB_HEAD_SCHEMA, "RGB HEAD schema drift")
    require(support_head.get("schema") == SUPPORT_HEAD_SCHEMA, "support HEAD schema drift")
    require(phase_a.get("schema") == PHASE_A_SCHEMA, "Phase-A schema drift")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "receipt schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for dependency in protocol["dependencies"]:
        path = Path(dependency["path"])
        require(path.stat().st_size == int(dependency["bytes"]), f"dependency bytes drift: {path}")
        require(sha256_file(path) == dependency["sha256"], f"dependency SHA drift: {path}")
    for name, path in (
        ("scope_protocol", args.scope_protocol),
        ("d2r1_manifest", args.d2r1_manifest),
        ("rgb_head_result", args.rgb_head_result),
        ("support_head_result", args.support_head_result),
        ("phase_a_manifest", args.phase_a_manifest),
        ("license_receipt", args.license_receipt),
    ):
        require(protocol[name]["sha256"] == sha256_file(path), f"{name} SHA drift")
    require(receipt["authority"]["body_download"] is True and receipt["authority"]["exact_member_extraction"] is True, "body not authorized")
    require(receipt["authority"]["image_decode"] is False and receipt["authority"]["model_output"] is False, "authority drift")
    roles = role_rows(manifest_input)
    video_ids = {row["video_id"] for row in roles}
    lookup = asset_lookup(rgb_head, support_head, video_ids)
    trajectories = trajectory_lookup(phase_a, video_ids)
    total_body_bytes = sum(int(row["content_length_bytes"]) for row in lookup.values())
    require(total_body_bytes == int(scope["body_request"]["exact_total_body_bytes"]), "exact body total drift")
    require(total_body_bytes <= int(receipt["source_scope"]["maximum_body_bytes"]), "authorized body bound exceeded")
    require(shutil.disk_usage(args.output_root.parent).free >= total_body_bytes + 10_000_000_000, "insufficient bounded storage")
    attempt = expected_attempt(
        args.protocol,
        args.scope_protocol,
        args.d2r1_manifest,
        args.rgb_head_result,
        args.support_head_result,
        args.phase_a_manifest,
        args.license_receipt,
        roles,
    )
    if args.output_root.exists():
        require(args.resume, f"output exists; --resume required: {args.output_root}")
        require(load_json(args.output_root / "attempt.json") == attempt, "resume attempt binding drift")
        require(not (args.output_root / "manifest.json").exists(), "completed manifest already exists")
    else:
        require(not args.resume, "--resume requires existing output root")
        args.output_root.mkdir(parents=True)
        write_json_exclusive(args.output_root / "attempt.json", attempt)
    temp_root = args.output_root / "_temporary_archives"
    temp_root.mkdir(exist_ok=True)
    run_temp = temp_root / uuid.uuid4().hex
    run_temp.mkdir()
    completed: list[dict[str, Any]] = []
    try:
        for index, role in enumerate(roles, start=1):
            checkpoint = args.output_root / "receipts" / f"{index:02d}-{role['video_id']}.json"
            if checkpoint.exists():
                value = read_checkpoint(checkpoint, role)
                completed.append(value)
                print(json.dumps({"resumed": index, "video_id": role["video_id"]}), flush=True)
                continue
            role_directory = "train" if role["role"] == "D2_TRAIN" else "development_sealed"
            video_output = args.output_root / "source" / role_directory / role["video_id"]
            if video_output.exists():
                safe_remove_tree(video_output, args.output_root / "source" / role_directory)
            video_temp = run_temp / f"{index:02d}-{role['video_id']}"
            video_temp.mkdir()
            source_assets: list[dict[str, Any]] = []
            extracted: dict[str, list[dict[str, Any]]] = {}
            try:
                for asset, (suffixes, destination_name) in ASSET_SPECS.items():
                    head_row = lookup[(role["video_id"], asset)]
                    archive = video_temp / asset
                    digest, attempts = download_file(head_row["url"], archive, int(head_row["content_length_bytes"]))
                    mapping, archive_member_count = archive_member_map(archive, suffixes)
                    missing = [stem for stem in role["selected_frame_stems"] if stem not in mapping]
                    require(not missing, f"missing exact stems for {asset}: {missing[:3]}")
                    entries = extract_named_members(
                        archive,
                        [mapping[stem] for stem in role["selected_frame_stems"]],
                        video_output / destination_name,
                    )
                    require(len(entries) == 300, f"extracted count drift: {asset}")
                    extracted[destination_name] = entries
                    source_assets.append(
                        {
                            "asset": asset,
                            "url": head_row["url"],
                            "bytes": int(head_row["content_length_bytes"]),
                            "sha256": digest,
                            "attempts": attempts,
                            "archive_member_count": archive_member_count,
                            "archive_crc_all_members_verified": True,
                        }
                    )
                    archive.unlink()
                trajectory = trajectories[role["video_id"]]
                value = {
                    "schema": CHECKPOINT_SCHEMA,
                    **role,
                    "source_assets": source_assets,
                    "trajectory": trajectory,
                    "extracted": extracted,
                    "extracted_file_count": sum(len(values) for values in extracted.values()),
                    "image_decode": False,
                    "truth_derivation": False,
                    "model_output_read": False,
                    "training_executed": False,
                    "development_outcome_opened": False,
                    "r2_cohort_access": "NONE",
                }
                write_checkpoint(checkpoint, value)
                completed.append(value)
                print(json.dumps({"completed": index, "total": 8, "video_id": role["video_id"], "role": role["role"], "files": value["extracted_file_count"]}), flush=True)
            finally:
                safe_remove_tree(video_temp, run_temp)
    finally:
        safe_remove_tree(run_temp, temp_root)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "scope_protocol_sha256": sha256_file(args.scope_protocol),
        "d2r1_manifest_sha256": sha256_file(args.d2r1_manifest),
        "rgb_head_result_sha256": sha256_file(args.rgb_head_result),
        "support_head_result_sha256": sha256_file(args.support_head_result),
        "phase_a_manifest_sha256": sha256_file(args.phase_a_manifest),
        "license_receipt_sha256": sha256_file(args.license_receipt),
        "identity_count": len(completed),
        "train_identity_count": sum(row["role"] == "D2_TRAIN" for row in completed),
        "development_sealed_identity_count": sum(row["role"] == "D2_DEVELOPMENT_SEALED" for row in completed),
        "source_asset_count": sum(len(row["source_assets"]) for row in completed),
        "extracted_file_count": sum(int(row["extracted_file_count"]) for row in completed),
        "exact_total_body_bytes": total_body_bytes,
        "roles": [
            {
                **{key: row[key] for key in ("role", "role_order", "phase_a_order", "pool_order", "visit_id", "video_id", "selected_frame_stems")},
                "checkpoint_path": str((args.output_root / "receipts" / f"{index:02d}-{row['video_id']}.json").resolve()),
                "checkpoint_sha256": sha256_file(args.output_root / "receipts" / f"{index:02d}-{row['video_id']}.json"),
            }
            for index, row in enumerate(completed, start=1)
        ],
        "image_decode": False,
        "truth_derivation": False,
        "model_output_read": False,
        "training_executed": False,
        "development_outcome_opened": False,
        "r2_cohort_access": "NONE",
        "terminal": "D2_PHASE_C_SOURCE_MATERIALIZATION_PASS_EXACT_EIGHT_SEALED",
    }
    write_json_exclusive(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "roles"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
