#!/usr/bin/env python3
"""Materialize D3 Phase-A intrinsics/trajectory and freeze portrait continuity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    interpolate_camera_to_world,
    orientation_index,
    parse_trajectory,
)
PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_body_protocol_v1"
ROSTER_SCHEMA = "blindassist_depthart_task_preserving_d3_fresh_metadata_roster_lock_v1"
HEAD_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_asset_header_preflight_v1"
SCOPE_SCHEMA = "blindassist_depthart_task_preserving_d3_source_scope_and_metadata_roster_receipt_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_body_activation_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_manifest_v1"
ATTEMPT_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_attempt_v1"
CHECKPOINT_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_identity_checkpoint_v1"
ASSETS = ("lowres_wide_intrinsics.zip", "lowres_wide.traj")


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


def write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        require(written > 0, "exclusive file write made no progress")
        remaining = remaining[written:]


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        write_all(
            descriptor,
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def timestamp_from_stem(stem: str) -> float:
    try:
        value = float(stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"cannot parse timestamp from {stem}") from error
    require(math.isfinite(value), f"non-finite timestamp in {stem}")
    return value


def download_file(
    row: dict[str, Any], output: Path, retries: int = 3
) -> dict[str, Any]:
    url = str(row["url"])
    expected_length = int(row["content_length_bytes"])
    expected_etag = row.get("etag")
    expected_last_modified = row.get("last_modified")
    require(not output.exists(), f"download overwrite forbidden: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        partial = output.with_name(f"{output.name}.attempt-{attempt}.partial")
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "BlindAssist-DepthART-D3-phase-a-body"}
            )
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
                status = int(response.status)
                response_length_text = response.headers.get("Content-Length")
                response_length = int(response_length_text) if response_length_text else None
                response_etag = response.headers.get("ETag")
                response_last_modified = response.headers.get("Last-Modified")
                require(status == 200, f"GET status drift: {status}")
                require(response_length == expected_length, "GET Content-Length differs from frozen HEAD")
                require(response_etag == expected_etag, "GET ETag differs from frozen HEAD")
                require(
                    response_last_modified == expected_last_modified,
                    "GET Last-Modified differs from frozen HEAD",
                )
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            require(size == expected_length, f"content length mismatch: {size} != {expected_length}")
            os.replace(partial, output)
            return {
                "url": url,
                "bytes": size,
                "sha256": digest.hexdigest().upper(),
                "attempts": attempt,
                "response_http_status": status,
                "response_content_length_bytes": response_length,
                "response_etag": response_etag,
                "response_last_modified": response_last_modified,
                "frozen_head_etag": expected_etag,
                "frozen_head_last_modified": expected_last_modified,
            }
        except Exception as error:  # pragma: no cover - live transport only
            errors.append(f"{type(error).__name__}: {error}")
            if partial.exists():
                partial.unlink()
    raise OSError(f"download failed after {retries} attempts: {errors}")


def pincam_members(archive: Path) -> list[tuple[float, str]]:
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        require(bad is None, f"ZIP CRC failure: {bad}")
        rows: list[tuple[float, str]] = []
        for name in bundle.namelist():
            pure = Path(name)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe ZIP member: {name}")
            if pure.suffix.lower() == ".pincam":
                rows.append((timestamp_from_stem(pure.stem), name))
        return sorted(rows)


def continuous_window(stems: list[str], count: int, maximum_gap: float) -> list[str]:
    ordered = sorted(stems, key=lambda stem: (timestamp_from_stem(stem), stem))
    run: list[str] = []
    previous: float | None = None
    for stem in ordered:
        timestamp = timestamp_from_stem(stem)
        if previous is None or 0 < timestamp - previous <= maximum_gap:
            run.append(stem)
        else:
            run = [stem]
        previous = timestamp
        if len(run) == count:
            return run
    raise ValueError(f"fewer than {count} continuous eligible frames")


def split_continuous_portrait_runs(
    classified: list[dict[str, Any]], maximum_adjacent_gap_seconds: float
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in classified:
        if not row["portrait"]:
            if current:
                runs.append(current)
                current = []
            continue
        timestamp = float(row["timestamp"])
        if current:
            gap = timestamp - float(current[-1]["timestamp"])
            if not 0 < gap <= maximum_adjacent_gap_seconds:
                runs.append(current)
                current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def portrait_runs(
    stems: list[str],
    trajectory: Any,
    maximum_pose_gap: float,
    maximum_adjacent_gap: float,
    portrait_indices: set[int],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    orientation_counts = {str(index): 0 for index in range(4)}
    pose_rejected = 0
    maximum_observed_pose_gap = 0.0
    for stem in sorted(stems, key=lambda item: (timestamp_from_stem(item), item)):
        try:
            pose, metadata = interpolate_camera_to_world(
                trajectory, timestamp_from_stem(stem), maximum_pose_gap
            )
        except ValueError:
            pose_rejected += 1
            classified.append(
                {
                    "stem": stem,
                    "timestamp": timestamp_from_stem(stem),
                    "portrait": False,
                }
            )
            continue
        index = orientation_index(pose)
        orientation_counts[str(index)] += 1
        maximum_observed_pose_gap = max(
            maximum_observed_pose_gap,
            float(metadata["bracketing_gap_seconds"]),
        )
        classified.append(
            {
                "stem": stem,
                "timestamp": timestamp_from_stem(stem),
                "portrait": index in portrait_indices,
            }
        )
    runs = split_continuous_portrait_runs(classified, maximum_adjacent_gap)
    return runs, {
        "intrinsics_frame_count": len(stems),
        "pose_rejected_frame_count": pose_rejected,
        "pose_covered_orientation_counts": orientation_counts,
        "portrait_pose_covered_frame_count": sum(len(run) for run in runs),
        "portrait_run_lengths": [len(run) for run in runs],
        "maximum_observed_pose_bracketing_gap_seconds": maximum_observed_pose_gap,
    }


def first_continuous_window(runs: list[list[dict[str, Any]]], count: int) -> list[str]:
    for run in runs:
        if len(run) >= count:
            return [str(row["stem"]) for row in run[:count]]
    raise ValueError(f"fewer than {count} continuous eligible frames")


def parse_intrinsics_payload(
    payload: bytes, label: str
) -> tuple[int, int, float, float, float, float]:
    fields = payload.decode("utf-8").split()
    require(len(fields) == 6, f"intrinsics must have six fields: {label}")
    values = [float(field) for field in fields]
    require(all(math.isfinite(value) for value in values), f"non-finite intrinsics: {label}")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height, f"non-integral dimensions: {label}")
    fx, fy, cx, cy = values[2:]
    require(width > 0 and height > 0 and fx > 0 and fy > 0, f"invalid intrinsics: {label}")
    require(0 <= cx < width and 0 <= cy < height, f"principal point outside image: {label}")
    return width, height, fx, fy, cx, cy


def validate_intrinsics_archive(archive: Path) -> tuple[dict[str, str], dict[str, int]]:
    result: dict[str, str] = {}
    dimension_counts: dict[str, int] = {}
    members = pincam_members(archive)
    with zipfile.ZipFile(archive) as bundle:
        for _, member in members:
            stem = Path(member).stem
            require(stem not in result, f"duplicate intrinsics stem: {stem}")
            width, height, *_ = parse_intrinsics_payload(bundle.read(member), member)
            key = f"{width}x{height}"
            dimension_counts[key] = dimension_counts.get(key, 0) + 1
            result[stem] = member
    require(result, "intrinsics ZIP contains no .pincam members")
    return result, dimension_counts


def parse_intrinsics(path: Path) -> tuple[int, int, float, float, float, float]:
    return parse_intrinsics_payload(path.read_bytes(), str(path))


def lookup_assets(preflight: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in preflight["assets"]:
        key = str(row["video_id"]), str(row["asset"])
        require(key not in lookup, f"duplicate HEAD row: {key}")
        require(
            row["http_status"] == 200 and int(row["content_length_bytes"]) > 0,
            f"unavailable: {key}",
        )
        lookup[key] = row
    require(len(lookup) == 96, "HEAD row count drift")
    return lookup


def first_qualified(
    processed: list[dict[str, Any]], selected_count: int
) -> list[dict[str, Any]]:
    selected = [row for row in processed if row["eligible"]][:selected_count]
    require(
        all(
            int(left["pool_order"]) < int(right["pool_order"])
            for left, right in zip(selected, selected[1:])
        ),
        "selected pool order is not increasing",
    )
    return selected


def finalize_phase_a_selection(
    processed: list[dict[str, Any]], pool_count: int, selected_count: int
) -> tuple[bool, list[dict[str, Any]]]:
    candidates = first_qualified(processed, selected_count)
    passed = len(processed) == pool_count and len(candidates) == selected_count
    return passed, candidates if passed else []


def expected_attempt(
    protocol_path: Path,
    roster_path: Path,
    head_path: Path,
    scope_path: Path,
    activation_path: Path,
    pool: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "bindings": {
            "protocol_sha256": sha256_file(protocol_path),
            "roster_sha256": sha256_file(roster_path),
            "head_result_sha256": sha256_file(head_path),
            "source_scope_sha256": sha256_file(scope_path),
            "activation_sha256": sha256_file(activation_path),
            "producer_sha256": sha256_file(Path(__file__)),
        },
        "identity_plan": [
            {
                "pool_order": int(row["pool_order"]),
                "visit_id": str(row["visit_id"]),
                "video_id": str(row["video_id"]),
                "fold": str(row["fold"]),
            }
            for row in pool
        ],
        "policy": {
            "assets": list(protocol["assets"]),
            "selected_identity_count": int(protocol["selected_identity_count"]),
            "continuous_portrait_frame_count": int(protocol["continuous_portrait_frame_count"]),
            "maximum_adjacent_frame_gap_seconds": float(
                protocol["maximum_adjacent_frame_gap_seconds"]
            ),
            "maximum_pose_bracketing_gap_seconds": float(
                protocol["maximum_pose_bracketing_gap_seconds"]
            ),
            "portrait_orientation_indices": [
                int(value) for value in protocol["portrait_orientation_indices"]
            ],
            "selection_rule": protocol["selection_rule"],
        },
        "authority": "D3_PHASE_A_BODY_AND_LABEL_BLIND_CONTINUITY_ONLY",
    }


def write_checkpoint(path: Path, value: dict[str, Any]) -> None:
    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    write_json_exclusive(
        path.with_suffix(".sha256.json"),
        {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()},
    )


def _verify_file_entry(entry: dict[str, Any]) -> None:
    path = Path(entry["path"])
    require(path.is_file(), f"retained file missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"retained file bytes drift: {path}")
    require(sha256_file(path) == entry["sha256"], f"retained file SHA drift: {path}")


def verify_frozen_file(entry: dict[str, Any], label: str) -> Path:
    path = Path(entry["path"])
    require(path.is_file(), f"{label} missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"{label} bytes drift: {path}")
    require(sha256_file(path) == entry["sha256"], f"{label} SHA drift: {path}")
    return path


def read_checkpoint(
    path: Path, expected_identity: dict[str, Any], attempt_sha256: str
) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256.json")
    require(sidecar.is_file(), f"checkpoint sidecar missing: {sidecar}")
    seal = load_json(sidecar)
    require(path.stat().st_size == int(seal["bytes"]), f"checkpoint bytes drift: {path}")
    require(sha256_file(path) == seal["sha256"], f"checkpoint SHA drift: {path}")
    value = load_json(path)
    require(value.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema drift")
    require(value["attempt_sha256"] == attempt_sha256, "checkpoint attempt drift")
    for key in ("pool_order", "visit_id", "video_id", "fold"):
        require(str(value[key]) == str(expected_identity[key]), f"checkpoint identity drift: {key}")
    require(
        value["rgb_depth_confidence_read"] is False
        and value["truth_or_model_output_read"] is False,
        "checkpoint authority drift",
    )
    for entry in value.get("source_assets", []):
        _verify_file_entry(entry)
    _verify_file_entry(value["trajectory"])
    return value


def validate_bindings(
    protocol_path: Path,
    roster_path: Path,
    head_path: Path,
    scope_path: Path,
    activation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    roster = load_json(roster_path)
    head_result = load_json(head_path)
    source_scope = load_json(scope_path)
    activation = load_json(activation_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(roster.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(head_result.get("schema") == HEAD_SCHEMA, "HEAD schema drift")
    require(source_scope.get("schema") == SCOPE_SCHEMA, "source scope schema drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    frozen_files = [
        (protocol["producer"], "producer"),
        (protocol["validator"], "validator"),
        *[(entry, "test") for entry in protocol["tests"]],
        *[(entry, "dependency") for entry in protocol["dependencies"]],
        (protocol["runtime"]["launcher"], "runtime launcher"),
        (protocol["runtime"]["python_executable"], "Python executable"),
    ]
    for entry, label in frozen_files:
        verify_frozen_file(entry, label)
    require(
        Path(sys.executable).resolve()
        == Path(protocol["runtime"]["python_executable"]["path"]).resolve(),
        "Python executable path drift",
    )
    require(
        ".".join(str(value) for value in sys.version_info[:3])
        == protocol["runtime"]["python_version"],
        "Python version drift",
    )
    require(
        importlib.metadata.version("numpy") == protocol["runtime"]["numpy_version"],
        "NumPy version drift",
    )
    require(
        importlib.metadata.version("Pillow") == protocol["runtime"]["pillow_version"],
        "Pillow version drift",
    )
    for name, path in (
        ("roster", roster_path),
        ("head_result", head_path),
        ("source_scope", scope_path),
    ):
        require(protocol[name]["sha256"] == sha256_file(path), f"{name} SHA drift")
    require(
        activation["bindings"]["body_protocol"]["sha256"] == sha256_file(protocol_path),
        "activation protocol mismatch",
    )
    for name in ("roster", "head_result", "source_scope"):
        require(
            activation["bindings"][name]["sha256"] == protocol[name]["sha256"],
            f"activation {name} mismatch",
        )
    require(
        head_result["terminal"]
        == "D3_PHASE_A_ASSET_HEADERS_96_OF_96_AVAILABLE_MEDIA_BODY_UNOPENED",
        "HEAD terminal drift",
    )
    require(activation["authority"]["phase_a_body"] is True, "Phase-A body not authorized")
    require(
        activation["authority"]["label_blind_continuity_selection"] is True,
        "Phase-A continuity selection not authorized",
    )
    require(activation["authority"]["rgb_depth_confidence"] is False, "modality scope drift")
    require(activation["authority"]["truth_or_model"] is False, "outcome scope drift")
    require(
        activation["authority"]["train_development_role_assignment"] is False
        and activation["authority"]["phase_b"] is False
        and activation["authority"]["r2"] is False,
        "successor authority drift",
    )
    require(tuple(protocol["assets"]) == ASSETS, "asset scope drift")
    return protocol, roster, head_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--head-result", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol, roster, head_result = validate_bindings(
        args.protocol,
        args.roster,
        args.head_result,
        args.source_scope,
        args.activation,
    )
    pool = roster["pool"]
    require(
        len(pool) == 48
        and [int(row["pool_order"]) for row in pool] == list(range(1, 49)),
        "pool drift",
    )
    lookup = lookup_assets(head_result)
    expected_total = sum(int(row["content_length_bytes"]) for row in lookup.values())
    require(expected_total == int(protocol["expected_total_content_length_bytes"]), "size sum drift")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    maximum_video_bytes = max(
        sum(
            int(lookup[(str(row["video_id"]), asset)]["content_length_bytes"])
            for asset in ASSETS
        )
        for row in pool
    )
    require(
        shutil.disk_usage(args.output_root.parent).free
        >= maximum_video_bytes * 3 + int(protocol["minimum_free_space_margin_bytes"]),
        "insufficient bounded working space",
    )
    attempt = expected_attempt(
        args.protocol,
        args.roster,
        args.head_result,
        args.source_scope,
        args.activation,
        pool,
        protocol,
    )
    attempt_path = args.output_root / "attempt.json"
    if args.output_root.exists():
        require(args.resume, f"output exists; --resume required: {args.output_root}")
        require(attempt_path.is_file(), "resume attempt receipt missing")
        require(load_json(attempt_path) == attempt, "resume attempt binding drift")
        require(not (args.output_root / "manifest.json").exists(), "completed manifest already exists")
        require(
            not (args.output_root / "_temporary_downloads").exists(),
            "partial temporary downloads remain; preserve this failed attempt and use a fresh root",
        )
    else:
        require(not args.resume, "--resume requires an existing attempt root")
        args.output_root.mkdir()
        write_json_exclusive(attempt_path, attempt)
    attempt_sha256 = sha256_file(attempt_path)

    temporary_root = args.output_root / "_temporary_downloads"
    temporary_root.mkdir()
    run_temporary_root = temporary_root / uuid.uuid4().hex
    run_temporary_root.mkdir()
    receipts_root = args.output_root / "receipts"
    selected_count = int(protocol["selected_identity_count"])
    frame_count = int(protocol["continuous_portrait_frame_count"])
    maximum_frame_gap = float(protocol["maximum_adjacent_frame_gap_seconds"])
    maximum_pose_gap = float(protocol["maximum_pose_bracketing_gap_seconds"])
    portrait_indices = {int(value) for value in protocol["portrait_orientation_indices"]}
    processed: list[dict[str, Any]] = []
    try:
        for parent in pool:
            video_id = str(parent["video_id"])
            checkpoint_path = receipts_root / f"{int(parent['pool_order']):02d}-{video_id}.json"
            if checkpoint_path.exists():
                value = read_checkpoint(checkpoint_path, parent, attempt_sha256)
                processed.append(value)
                print(
                    json.dumps(
                        {
                            "resumed": int(parent["pool_order"]),
                            "video_id": video_id,
                            "eligible": value["eligible"],
                        }
                    ),
                    flush=True,
                )
                continue

            source_video_root = args.output_root / "source" / "Training" / video_id
            require(
                not source_video_root.exists(),
                f"orphan retained identity without checkpoint: {video_id}",
            )
            video_temp = run_temporary_root / f"{int(parent['pool_order']):02d}-{video_id}"
            video_temp.mkdir()
            staged_source = video_temp / "source"
            staged_source.mkdir()
            trajectory_row = lookup[(video_id, "lowres_wide.traj")]
            trajectory_temp = staged_source / "lowres_wide.traj"
            trajectory_receipt = download_file(trajectory_row, trajectory_temp)
            trajectory = parse_trajectory(trajectory_temp)
            intrinsics_row = lookup[(video_id, "lowres_wide_intrinsics.zip")]
            archive = staged_source / "lowres_wide_intrinsics.zip"
            intrinsics_receipt = download_file(intrinsics_row, archive)
            intrinsics_receipt["zip_crc_and_member_safety_checked"] = True
            members, source_dimension_counts = validate_intrinsics_archive(archive)
            runs, coverage = portrait_runs(
                list(members),
                trajectory,
                maximum_pose_gap,
                maximum_frame_gap,
                portrait_indices,
            )
            try:
                selected_stems = first_continuous_window(runs, frame_count)
                reason = "PASS"
            except ValueError as error:
                selected_stems = []
                reason = str(error)

            trajectory_entry: dict[str, Any]
            source_video_root.parent.mkdir(parents=True, exist_ok=True)
            require(not source_video_root.exists(), "source destination already exists")
            os.rename(staged_source, source_video_root)
            retained_trajectory = source_video_root / "lowres_wide.traj"
            retained_archive = source_video_root / "lowres_wide_intrinsics.zip"
            trajectory_entry = {
                "path": str(retained_trajectory.resolve()),
                "bytes": retained_trajectory.stat().st_size,
                "sha256": sha256_file(retained_trajectory),
                "row_count": int(trajectory.shape[0]),
            }
            trajectory_receipt["path"] = str(retained_trajectory.resolve())
            intrinsics_receipt["path"] = str(retained_archive.resolve())
            times = [timestamp_from_stem(stem) for stem in selected_stems]
            value = {
                "schema": CHECKPOINT_SCHEMA,
                "attempt_sha256": attempt_sha256,
                **parent,
                "eligible": bool(selected_stems),
                "eligibility_reason": reason,
                "selected_frame_stems": selected_stems,
                "selected_frame_count": len(selected_stems),
                "selected_start_timestamp": times[0] if times else None,
                "selected_end_timestamp": times[-1] if times else None,
                "maximum_selected_adjacent_gap_seconds": (
                    max(right - left for left, right in zip(times, times[1:]))
                    if len(times) > 1
                    else None
                ),
                "coverage": coverage,
                "source_assets": [
                    {"asset": "lowres_wide.traj", **trajectory_receipt},
                    {"asset": "lowres_wide_intrinsics.zip", **intrinsics_receipt},
                ],
                "intrinsics_payload_validated_count": len(members),
                "source_intrinsics_dimension_counts": source_dimension_counts,
                "trajectory": trajectory_entry,
                "rgb_depth_confidence_read": False,
                "truth_or_model_output_read": False,
            }
            write_checkpoint(checkpoint_path, value)
            processed.append(value)
            print(
                json.dumps(
                    {
                        "processed": len(processed),
                        "pool": len(pool),
                        "eligible": sum(bool(row["eligible"]) for row in processed),
                        "target": selected_count,
                        "video_id": video_id,
                        "status": reason,
                    }
                ),
                flush=True,
            )
            if video_temp.exists():
                shutil.rmtree(video_temp)
    finally:
        if run_temporary_root.exists():
            shutil.rmtree(run_temporary_root)
        if temporary_root.exists() and not any(temporary_root.iterdir()):
            temporary_root.rmdir()

    passed, selected_rows = finalize_phase_a_selection(
        processed, len(pool), selected_count
    )
    selected_phase_a = [
        {
            "selection_order": index + 1,
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
            "role": "D3_PHASE_A_SELECTED_IDENTITY_ONLY",
            "selected_frame_stems": row["selected_frame_stems"],
        }
        for index, row in enumerate(selected_rows)
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": sha256_file(args.roster),
        "head_result_sha256": sha256_file(args.head_result),
        "source_scope_sha256": sha256_file(args.source_scope),
        "activation_sha256": sha256_file(args.activation),
        "attempt_sha256": attempt_sha256,
        "pool_count": len(pool),
        "processed_identity_count": len(processed),
        "eligible_identity_count": sum(bool(row["eligible"]) for row in processed),
        "eligible_candidates": [
            {
                "pool_order": int(row["pool_order"]),
                "visit_id": str(row["visit_id"]),
                "video_id": str(row["video_id"]),
                "fold": str(row["fold"]),
            }
            for row in processed
            if row["eligible"]
        ],
        "phase_a_selection_locked": passed,
        "selected_identity_count": len(selected_phase_a),
        "selected_phase_a": selected_phase_a,
        "processed": processed,
        "media_body_scope": list(ASSETS),
        "declared_download_bytes": expected_total,
        "downloaded_body_bytes": sum(
            int(asset["bytes"])
            for row in processed
            for asset in row["source_assets"]
        ),
        "rgb_depth_confidence_read": False,
        "truth_or_model_output_read": False,
        "train_development_roles_assigned": False,
        "r2_cohort_access": "NONE",
        "temporary_archives_retained": False,
        "source_archives_retained_for_offline_validation": True,
        "terminal": (
            "D3_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED"
            if passed
            else "D3_PHASE_A_FAIL_FEWER_THAN_32_ELIGIBLE_IDENTITIES"
        ),
    }
    write_checkpoint(args.output_root / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "processed"}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
