#!/usr/bin/env python3
"""Materialize D3R1 Phase-B depth/confidence and audit fixed source support.

This entrypoint is deliberately separate from the consumed D2/D3 producers. It
only opens the exact 64 bodies and the exact 9,600 Phase-A-selected frame stems
authorized by the D3R1 Phase-B body receipt. It never reads RGB or model output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np
from PIL import Image

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (
    WORLD_UP,
    TruthReaderPolicy,
    canonicalize_frame,
    depth_mm_to_metres,
    derive_assistive_truth,
    interpolate_camera_to_world,
    parse_trajectory,
)
from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_b_assets import (
    request_plan_sha256,
    selection_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_body_protocol_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_body_activation_v1"
SCOPE_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r1_phase_b_depth_confidence_source_scope_receipt_v1"
)
PHASE_A_RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_governed_result_v1"
PHASE_A_MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_manifest_v1"
HEAD_GOVERNED_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_head_governed_result_v1"
HEAD_MACHINE_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_asset_header_preflight_v1"
ATTEMPT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_attempt_v1"
CHECKPOINT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_identity_checkpoint_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_manifest_v1"

ASSETS = ("lowres_depth.zip", "confidence.zip")
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
PASS_TERMINAL = "D3R1_PHASE_B_SOURCE_TRUTH_SUPPORT_PASS_16_IDENTITIES_LOCKED"
FAIL_TERMINAL = "D3_DATA_SUPPORT_NOT_EVALUABLE"
PASS_NEXT_GATE = "EXPLICIT_D3R1_PHASE_C_RGB_SOURCE_SCOPE_REGISTRATION_FOR_EXACT_16_PHASE_B_SELECTION"


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


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
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def truth_policy_dict(policy: TruthReaderPolicy) -> dict[str, Any]:
    value = asdict(policy)
    value["horizons_m"] = list(value["horizons_m"])
    return value


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "exclusive write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    write_bytes_exclusive(path, json_bytes(value))


def write_sealed_json(path: Path, value: dict[str, Any]) -> None:
    payload = json_bytes(value)
    write_bytes_exclusive(path, payload)
    write_json_exclusive(
        path.with_suffix(".sha256.json"),
        {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()},
    )


def verify_file_entry(entry: dict[str, Any], label: str) -> Path:
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = REPO_ROOT / path
    require(path.is_file(), f"{label} missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"{label} byte drift")
    require(sha256_file(path) == str(entry["sha256"]), f"{label} SHA drift")
    return path


def same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def require_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    require(resolved.is_relative_to(root.resolve()), f"{label} escapes root: {path}")
    return resolved


def timestamp_from_stem(stem: str) -> float:
    try:
        value = Decimal(stem.rsplit("_", 1)[-1])
    except (InvalidOperation, IndexError) as error:
        raise ValueError(f"invalid frame timestamp: {stem}") from error
    require(value.is_finite(), f"non-finite frame timestamp: {stem}")
    result = float(value)
    require(math.isfinite(result), f"non-finite frame timestamp: {stem}")
    return result


def parse_pincam_payload(payload: bytes, label: str) -> tuple[np.ndarray, tuple[int, int]]:
    try:
        values = [float(value) for value in payload.decode("utf-8").split()]
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid pincam payload: {label}") from error
    require(len(values) == 6 and all(math.isfinite(value) for value in values), f"invalid pincam: {label}")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height and width > 0 and height > 0, f"invalid pincam dimensions: {label}")
    fx, fy, cx, cy = values[2:]
    require(fx > 0 and fy > 0 and 0 <= cx < width and 0 <= cy < height, f"invalid pincam values: {label}")
    matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return matrix, (width, height)


def safe_member_map(archive: zipfile.ZipFile, suffix: str) -> tuple[dict[str, str], int]:
    require(archive.testzip() is None, f"ZIP CRC failure: {archive.filename}")
    mapping: dict[str, str] = {}
    file_count = 0
    seen_names: set[str] = set()
    for info in archive.infolist():
        path = PurePosixPath(info.filename.replace("\\", "/"))
        require(not path.is_absolute() and ".." not in path.parts, f"unsafe ZIP member: {info.filename}")
        require(info.filename not in seen_names, f"duplicate ZIP member name: {info.filename}")
        seen_names.add(info.filename)
        if info.is_dir():
            continue
        file_count += 1
        if path.suffix.lower() != suffix:
            continue
        stem = path.stem
        require(stem not in mapping, f"duplicate ZIP frame stem: {stem}")
        mapping[stem] = info.filename
    require(mapping, f"no {suffix} members in {archive.filename}")
    return mapping, file_count


def frame_plan_sha256(selected: list[dict[str, Any]]) -> str:
    lines = [
        f"{row['selection_order']}:{row['pool_order']}:{row['visit_id']}:{row['video_id']}:{stem}"
        for row in selected
        for stem in row["selected_frame_stems"]
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest().upper()


def phase_a_selection(
    source_scope: dict[str, Any], phase_a_result: dict[str, Any], phase_a: dict[str, Any]
) -> list[dict[str, Any]]:
    base = selection_rows(source_scope, phase_a_result)
    require(phase_a.get("schema") == PHASE_A_MANIFEST_SCHEMA, "Phase-A manifest schema drift")
    require(phase_a.get("terminal") == "D3R1_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED", "Phase-A manifest terminal drift")
    require(phase_a.get("phase_a_selection_locked") is True, "Phase-A selection is not locked")
    selected = phase_a.get("selected_phase_a")
    require(isinstance(selected, list) and len(selected) == 32, "Phase-A manifest selection drift")
    processed = {
        (int(row["pool_order"]), str(row["visit_id"]), str(row["video_id"])): row
        for row in phase_a["processed"]
    }
    require(len(processed) == 127, "Phase-A processed inventory drift")
    result: list[dict[str, Any]] = []
    for expected, (identity, selected_row) in enumerate(zip(base, selected, strict=True), start=1):
        for key in ("selection_order", "pool_order", "visit_id", "video_id"):
            require(identity[key] == selected_row[key], f"Phase-A selected mismatch: {key}")
        key = (int(identity["pool_order"]), str(identity["visit_id"]), str(identity["video_id"]))
        require(key in processed, f"Phase-A checkpoint missing: {key}")
        checkpoint = processed[key]
        stems = list(selected_row["selected_frame_stems"])
        require(expected == identity["selection_order"], "selection order drift")
        require(checkpoint["eligible"] is True, "selected Phase-A identity is not eligible")
        require(checkpoint["selected_frame_count"] == 300, "Phase-A frame count drift")
        require(stems == checkpoint["selected_frame_stems"] and len(stems) == 300, "Phase-A frame plan drift")
        require(len(stems) == len(set(stems)), "duplicate Phase-A frame stem")
        require(all(stem.startswith(f"{identity['video_id']}_") for stem in stems), "frame/video mismatch")
        result.append(identity | {"selected_frame_stems": stems, "phase_a_checkpoint": checkpoint})
    registered = source_scope["registered_future_asset_scope"]
    require(frame_plan_sha256(result) == registered["selected_frame_plan_sha256"], "selected frame plan SHA drift")
    require(sum(len(row["selected_frame_stems"]) for row in result) == 9600, "selected frame total drift")
    return result


def head_lookup(head_result: dict[str, Any], selected: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    require(head_result.get("schema") == HEAD_MACHINE_SCHEMA, "HEAD machine schema drift")
    require(head_result.get("terminal") == "D3R1_PHASE_B_ASSET_HEADERS_64_OF_64_AVAILABLE_MEDIA_BODY_UNOPENED", "HEAD machine terminal drift")
    rows = head_result.get("assets")
    require(isinstance(rows, list) and len(rows) == 64, "HEAD row count drift")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    expected_plan: list[dict[str, Any]] = []
    for identity in selected:
        for asset in ASSETS:
            expected_plan.append(identity | {"asset": asset, "url": f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{identity['video_id']}/{asset}"})
    require(request_plan_sha256(expected_plan) == head_result["request_plan_sha256"], "HEAD request plan SHA drift")
    for expected, row in zip(expected_plan, rows, strict=True):
        for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold", "asset", "url"):
            require(row[key] == expected[key], f"HEAD plan mismatch: {key}")
        require(row["role"] == "D3R1_PHASE_A_SELECTED_IDENTITY_ONLY", "HEAD role drift")
        require(row["http_status"] == 200 and row["final_url"] == row["url"], "HEAD availability drift")
        require(int(row["content_length_bytes"]) > 0, "HEAD Content-Length drift")
        require(isinstance(row["etag"], str) and row["etag"], "HEAD ETag missing")
        require(isinstance(row["last_modified"], str) and row["last_modified"], "HEAD Last-Modified missing")
        require(row["redirect_count"] == 0 and row["response_body_bytes_read"] == 0, "HEAD boundary drift")
        require(row["unresolved_error"] is False, "HEAD unresolved error")
        key = str(row["video_id"]), str(row["asset"])
        require(key not in lookup, f"duplicate HEAD row: {key}")
        lookup[key] = row
    require(len(lookup) == 64, "HEAD lookup count drift")
    return lookup


def empty_counts() -> dict[str, Any]:
    grid = {f"{band}@{horizon:.1f}m": 0 for band in BANDS for horizon in HORIZONS}
    return {
        "known_cells": 0,
        "clear_cells": 0,
        "occupied_cells": 0,
        "valid_band_clearances": 0,
        "known_by_grid": dict(grid),
        "clear_by_grid": dict(grid),
        "occupied_by_grid": dict(grid),
    }


def summarize_truth(truth: dict[str, Any]) -> dict[str, Any]:
    result = empty_counts()
    for band in BANDS:
        band_result = truth.get("bands", {}).get(band)
        if not band_result:
            continue
        if band_result.get("clearance_m") is not None:
            result["valid_band_clearances"] += 1
        occupied = band_result.get("occupied_by_horizon", {})
        for horizon in HORIZONS:
            value = occupied.get(str(horizon))
            if value is None:
                continue
            key = f"{band}@{horizon:.1f}m"
            result["known_cells"] += 1
            result["known_by_grid"][key] += 1
            if bool(value):
                result["occupied_cells"] += 1
                result["occupied_by_grid"][key] += 1
            else:
                result["clear_cells"] += 1
                result["clear_by_grid"][key] += 1
    return result


def add_counts(total: dict[str, Any], frame: dict[str, Any]) -> None:
    for key in ("known_cells", "clear_cells", "occupied_cells", "valid_band_clearances"):
        total[key] += int(frame[key])
    for group in ("known_by_grid", "clear_by_grid", "occupied_by_grid"):
        for key, value in frame[group].items():
            total[group][key] += int(value)


def validate_count_identities(counts: dict[str, Any]) -> None:
    require(counts["known_cells"] == counts["clear_cells"] + counts["occupied_cells"], "known total identity drift")
    require(sum(counts["known_by_grid"].values()) == counts["known_cells"], "known grid total drift")
    require(sum(counts["clear_by_grid"].values()) == counts["clear_cells"], "clear grid total drift")
    require(sum(counts["occupied_by_grid"].values()) == counts["occupied_cells"], "occupied grid total drift")
    for key in counts["known_by_grid"]:
        require(counts["known_by_grid"][key] == counts["clear_by_grid"][key] + counts["occupied_by_grid"][key], f"grid identity drift: {key}")


def qualifies(counts: dict[str, Any], thresholds: dict[str, Any]) -> tuple[bool, list[str]]:
    validate_count_identities(counts)
    failures: list[str] = []
    for count_key, threshold_key in (
        ("known_cells", "minimum_truth_known_cells"),
        ("clear_cells", "minimum_truth_clear_cells"),
        ("occupied_cells", "minimum_truth_occupied_cells"),
        ("valid_band_clearances", "minimum_valid_band_clearances"),
    ):
        if int(counts[count_key]) < int(thresholds[threshold_key]):
            failures.append(f"{count_key}={counts[count_key]}<{thresholds[threshold_key]}")
    minimum_clear = int(thresholds["minimum_truth_clear_cells_per_band_horizon"])
    minimum_occupied = int(thresholds["minimum_truth_occupied_cells_per_band_horizon"])
    for key in sorted(counts["known_by_grid"]):
        if int(counts["clear_by_grid"][key]) < minimum_clear:
            failures.append(f"{key}_clear={counts['clear_by_grid'][key]}<{minimum_clear}")
        if int(counts["occupied_by_grid"][key]) < minimum_occupied:
            failures.append(f"{key}_occupied={counts['occupied_by_grid'][key]}<{minimum_occupied}")
    return not failures, failures


def _response_headers(response: Any) -> tuple[int | None, str | None, str | None]:
    length_text = response.headers.get("Content-Length")
    return (
        int(length_text) if length_text is not None else None,
        response.headers.get("ETag"),
        response.headers.get("Last-Modified"),
    )


def download_file(
    row: dict[str, Any],
    output: Path,
    temporary_root: Path,
    max_attempts: int = 3,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    require(max_attempts == 3, "download retry policy drift")
    require(not output.exists(), f"download overwrite forbidden: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=False)
    open_request = opener or urllib.request.build_opener(NoRedirectHandler()).open
    history: list[dict[str, Any]] = []
    url = str(row["url"])
    expected_length = int(row["content_length_bytes"])
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        partial = temporary_root / f"{output.name}.attempt-{attempt}.partial"
        status: int | None = None
        final_url: str | None = None
        response_length: int | None = None
        response_etag: str | None = None
        response_last_modified: str | None = None
        try:
            request = urllib.request.Request(
                url,
                method="GET",
                headers={
                    "User-Agent": "BlindAssist-DepthART-D3R1-phase-b-body",
                    "Accept-Encoding": "identity",
                },
            )
            digest = hashlib.sha256()
            size = 0
            with open_request(request, timeout=60) as response, partial.open("xb") as stream:
                status = int(response.status)
                final_url = str(response.geturl())
                response_length, response_etag, response_last_modified = _response_headers(response)
                require(status == 200, f"GET status drift: {status}")
                require(final_url == url, "GET redirect/final URL drift")
                require(response_length == expected_length, "GET Content-Length differs from frozen HEAD")
                require(response_etag == row["etag"], "GET ETag differs from frozen HEAD")
                require(response_last_modified == row["last_modified"], "GET Last-Modified differs from frozen HEAD")
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            require(size == expected_length, f"download length mismatch: {size} != {expected_length}")
            os.rename(partial, output)
            history.append(
                {
                    "attempt": attempt,
                    "method": "GET",
                    "http_status": status,
                    "error": None,
                    "error_type": None,
                    "retry_class": None,
                }
            )
            return {
                "asset": row["asset"],
                "url": url,
                "bytes": size,
                "sha256": digest.hexdigest().upper(),
                "path": str(output.resolve()),
                "attempts": attempt,
                "attempt_history": history,
                "response_http_status": status,
                "response_final_url": final_url,
                "response_content_length_bytes": response_length,
                "response_etag": response_etag,
                "response_last_modified": response_last_modified,
                "frozen_head_content_length_bytes": expected_length,
                "frozen_head_etag": row["etag"],
                "frozen_head_last_modified": row["last_modified"],
                "range_request_used": False,
                "redirect_followed": False,
            }
        except Exception as error:  # live transport paths are integration-tested with fakes
            last_error = error
            status = int(error.code) if isinstance(error, urllib.error.HTTPError) else status
            transient_http = status in {408, 429} or (status is not None and 500 <= status <= 599)
            transient_transport = isinstance(error, (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError)) and not isinstance(error, urllib.error.HTTPError)
            retry_class = (
                "TRANSIENT_HTTP"
                if transient_http
                else "TRANSIENT_TRANSPORT"
                if transient_transport
                else "TERMINAL"
            )
            history.append(
                {
                    "attempt": attempt,
                    "method": "GET",
                    "http_status": status,
                    "error": f"{type(error).__name__}: {error}",
                    "error_type": type(error).__name__,
                    "retry_class": retry_class,
                }
            )
            if partial.exists():
                partial.unlink()
            if not (transient_http or transient_transport) or attempt == max_attempts:
                break
    raise OSError(f"download failed under frozen retry policy: {last_error}; history={history}")


def _phase_a_sources(identity: dict[str, Any], phase_a_root: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    checkpoint = identity["phase_a_checkpoint"]
    assets = {str(row["asset"]): row for row in checkpoint["source_assets"]}
    require(set(assets) == {"lowres_wide_intrinsics.zip", "lowres_wide.traj"}, "Phase-A source asset drift")
    expected_dir = phase_a_root / "source" / "Training" / str(identity["video_id"])
    intrinsics = expected_dir / "lowres_wide_intrinsics.zip"
    trajectory = expected_dir / "lowres_wide.traj"
    result: list[dict[str, Any]] = []
    for asset, expected in (("lowres_wide_intrinsics.zip", intrinsics), ("lowres_wide.traj", trajectory)):
        entry = assets[asset]
        actual = Path(str(entry["path"]))
        require(same_path(actual, expected), f"Phase-A source path drift: {asset}")
        require(actual.is_file() and actual.stat().st_size == int(entry["bytes"]), f"Phase-A source bytes drift: {asset}")
        require(sha256_file(actual) == entry["sha256"], f"Phase-A source SHA drift: {asset}")
        result.append({"asset": asset, "path": str(actual.resolve()), "bytes": int(entry["bytes"]), "sha256": entry["sha256"]})
    return intrinsics, trajectory, result


def audit_identity(
    *,
    identity: dict[str, Any],
    depth_path: Path,
    confidence_path: Path,
    phase_a_root: Path,
    policy: TruthReaderPolicy,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    intrinsics_path, trajectory_path, phase_a_sources = _phase_a_sources(identity, phase_a_root)
    trajectory = parse_trajectory(trajectory_path)
    selected_stems = identity["selected_frame_stems"]
    counts = empty_counts()
    orientation_counts = {str(index): 0 for index in range(4)}
    depth_sizes: set[tuple[int, int]] = set()
    confidence_values: set[int] = set()
    maximum_pose_gap = 0.0
    with (
        zipfile.ZipFile(intrinsics_path) as intrinsics_zip,
        zipfile.ZipFile(depth_path) as depth_zip,
        zipfile.ZipFile(confidence_path) as confidence_zip,
    ):
        intrinsics_map, intrinsics_files = safe_member_map(intrinsics_zip, ".pincam")
        depth_map, depth_files = safe_member_map(depth_zip, ".png")
        confidence_map, confidence_files = safe_member_map(confidence_zip, ".png")
        for label, mapping in (("intrinsics", intrinsics_map), ("depth", depth_map), ("confidence", confidence_map)):
            missing = [stem for stem in selected_stems if stem not in mapping]
            require(not missing, f"missing selected {label} frames: {missing[:3]}")
        for stem in selected_stems:
            intrinsics, source_size = parse_pincam_payload(
                intrinsics_zip.read(intrinsics_map[stem]), intrinsics_map[stem]
            )
            with depth_zip.open(depth_map[stem]) as stream, Image.open(stream) as image:
                depth_raw = np.asarray(image).copy()
            with confidence_zip.open(confidence_map[stem]) as stream, Image.open(stream) as image:
                confidence = np.asarray(image).copy()
            require(depth_raw.ndim == 2 and np.issubdtype(depth_raw.dtype, np.integer), "invalid depth raster")
            require(confidence.ndim == 2 and np.issubdtype(confidence.dtype, np.integer), "invalid confidence raster")
            require(confidence.shape == depth_raw.shape, "depth/confidence shape drift")
            require(source_size == (depth_raw.shape[1], depth_raw.shape[0]), "intrinsics/depth shape drift")
            depth_sizes.add((int(depth_raw.shape[1]), int(depth_raw.shape[0])))
            confidence_values.update(int(value) for value in np.unique(confidence))
            pose, pose_meta = interpolate_camera_to_world(
                trajectory, timestamp_from_stem(stem), policy.maximum_pose_bracketing_gap_seconds
            )
            maximum_pose_gap = max(maximum_pose_gap, float(pose_meta["bracketing_gap_seconds"]))
            dummy_rgb = np.zeros((*depth_raw.shape, 3), dtype=np.uint8)
            canonical = canonicalize_frame(dummy_rgb, depth_raw, confidence, intrinsics, pose)
            orientation_counts[str(canonical["rotation_index"])] += 1
            require(canonical["rotation_index"] in (1, 3), "Phase-A portrait orientation drift")
            up_camera = canonical["camera_to_world"][:3, :3].T @ WORLD_UP
            truth = derive_assistive_truth(
                depth_mm_to_metres(canonical["depth_raw_mm"]),
                canonical["confidence"],
                canonical["intrinsics"],
                up_camera,
                policy,
            )
            add_counts(counts, summarize_truth(truth))
        archive_validation = {
            "intrinsics_total_file_members": intrinsics_files,
            "intrinsics_frame_members": len(intrinsics_map),
            "depth_total_file_members": depth_files,
            "depth_frame_members": len(depth_map),
            "confidence_total_file_members": confidence_files,
            "confidence_frame_members": len(confidence_map),
            "selected_intrinsics_coverage": len(selected_stems),
            "selected_depth_coverage": len(selected_stems),
            "selected_confidence_coverage": len(selected_stems),
            "zip_crc_and_member_safety_checked": True,
        }
    require(confidence_values.issubset({0, 1, 2}), f"confidence values drift: {sorted(confidence_values)}")
    validate_count_identities(counts)
    qualified, failures = qualifies(counts, thresholds)
    return {
        "frame_count": len(selected_stems),
        "selected_frame_plan_sha256": hashlib.sha256(("\n".join(selected_stems) + "\n").encode("ascii")).hexdigest().upper(),
        "archive_validation": archive_validation,
        "trajectory_row_count": int(trajectory.shape[0]),
        "maximum_pose_bracketing_gap_seconds": maximum_pose_gap,
        "depth_sizes_wh": [list(value) for value in sorted(depth_sizes)],
        "confidence_values": sorted(confidence_values),
        "orientation_counts": orientation_counts,
        "truth_support": counts,
        "source_truth_support_qualified": qualified,
        "qualification_failures": failures,
        "phase_a_sources": phase_a_sources,
        "rgb_read": False,
        "model_output_read": False,
        "per_frame_truth_retained": False,
    }


def finalize_selection(processed: list[dict[str, Any]], target: int = 16) -> tuple[bool, list[dict[str, Any]]]:
    require(len(processed) == 32, "all exact 32 identities must be processed before selection")
    qualified = [row for row in processed if row["source_truth_support_qualified"]]
    if len(qualified) < target:
        return False, []
    selected: list[dict[str, Any]] = []
    for order, row in enumerate(qualified[:target], start=1):
        selected.append(
            {
                "phase_b_selection_order": order,
                "phase_a_selection_order": row["selection_order"],
                "pool_order": row["pool_order"],
                "visit_id": row["visit_id"],
                "video_id": row["video_id"],
                "fold": "Training",
                "selected_frame_count": 300,
                "selected_frame_plan_sha256": row["selected_frame_plan_sha256"],
                "role_assigned": False,
            }
        )
    return True, selected


def checkpoint_path(receipts_root: Path, identity: dict[str, Any]) -> Path:
    return receipts_root / f"{int(identity['selection_order']):03d}-{identity['video_id']}.json"


def read_sealed_json(path: Path, schema: str) -> dict[str, Any]:
    require(path.is_file(), f"sealed JSON missing: {path}")
    seal_path = path.with_suffix(".sha256.json")
    require(seal_path.is_file(), f"seal missing: {seal_path}")
    payload = path.read_bytes()
    seal = load_json(seal_path)
    require(seal == {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()}, f"seal mismatch: {path}")
    value = json.loads(payload.decode("utf-8"))
    require(value.get("schema") == schema, f"schema drift: {path}")
    return value


def expected_attempt(
    *,
    protocol_path: Path,
    activation_path: Path,
    source_scope_path: Path,
    phase_a_result_path: Path,
    phase_a_manifest_path: Path,
    head_governed_path: Path,
    head_machine_path: Path,
    output_root: Path,
    selected: list[dict[str, Any]],
    protocol: dict[str, Any],
    declared_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "bindings": {
            "protocol_sha256": sha256_file(protocol_path),
            "activation_sha256": sha256_file(activation_path),
            "source_scope_sha256": sha256_file(source_scope_path),
            "phase_a_result_sha256": sha256_file(phase_a_result_path),
            "phase_a_manifest_sha256": sha256_file(phase_a_manifest_path),
            "head_governed_result_sha256": sha256_file(head_governed_path),
            "head_machine_result_sha256": sha256_file(head_machine_path),
            "producer_sha256": sha256_file(Path(__file__)),
        },
        "output_root": str(output_root.resolve()),
        "identity_plan": [
            {key: row[key] for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold")}
            | {"selected_frame_plan_sha256": hashlib.sha256(("\n".join(row["selected_frame_stems"]) + "\n").encode("ascii")).hexdigest().upper()}
            for row in selected
        ],
        "policy": {
            "assets": list(ASSETS),
            "identity_count": 32,
            "frames_per_identity": 300,
            "total_frame_count": 9600,
            "truth_reader_policy": truth_policy_dict(TruthReaderPolicy()),
            "truth_support_thresholds": protocol["truth_support_thresholds"],
            "selection_rule": "process all exact 32 then take first 16 qualified in Phase-A selection order",
            "max_attempts": 3,
            "range_get": False,
            "redirect_following": False,
            "source_archives_retained": True,
        },
        "declared_download_bytes": declared_bytes,
        "authority": "D3R1_PHASE_B_BODY_SOURCE_INTEGRITY_AND_SOURCE_TRUTH_SUPPORT_ONLY",
        "rgb_access": False,
        "model_output_access": False,
        "role_assignment": False,
        "training": False,
        "development_outcome": False,
        "r2_access": "NONE",
    }


def _validate_activation(activation: dict[str, Any]) -> None:
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    require(activation.get("status") == "D3R1_PHASE_B_BODY_AND_SOURCE_TRUTH_SUPPORT_ACTIVATED", "activation status drift")
    require(activation.get("authorization_verbatim") == "授权", "authorization verbatim drift")
    authority = activation["authority"]
    for key in ("phase_b_body_download", "archive_member_read", "depth_confidence_decode", "phase_a_intrinsics_trajectory_reread", "source_truth_support", "phase_b_conditional_selection"):
        require(authority.get(key) is True, f"activation missing authority: {key}")
    for key in ("range_get", "redirect_following", "phase_c_rgb", "model_output", "role_assignment", "training", "development_outcome", "r2_access", "performance", "android_default", "production", "safety"):
        expected: Any = "NONE" if key == "r2_access" else False
        require(authority.get(key) == expected, f"activation authority widened: {key}")


def validate_bindings(
    *,
    protocol_path: Path,
    activation_path: Path,
    source_scope_path: Path,
    phase_a_result_path: Path,
    phase_a_manifest_path: Path,
    head_governed_path: Path,
    head_machine_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    activation = load_json(activation_path)
    source_scope = load_json(source_scope_path)
    phase_a_result = load_json(phase_a_result_path)
    phase_a = load_json(phase_a_manifest_path)
    head_governed = load_json(head_governed_path)
    head_machine = load_json(head_machine_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(source_scope.get("schema") == SCOPE_SCHEMA, "source scope schema drift")
    require(phase_a_result.get("schema") == PHASE_A_RESULT_SCHEMA, "Phase-A result schema drift")
    require(head_governed.get("schema") == HEAD_GOVERNED_SCHEMA, "HEAD governed schema drift")
    require(head_governed.get("terminal") == "D3R1_PHASE_B_ASSET_HEADERS_64_OF_64_AVAILABLE_MEDIA_BODY_UNOPENED", "HEAD governed terminal drift")
    require(head_governed.get("next_gate") == "EXPLICIT_D3R1_PHASE_B_DEPTH_CONFIDENCE_BODY_AND_SOURCE_TRUTH_SUPPORT_ACTIVATION", "HEAD successor drift")
    for label, entry in protocol["frozen_files"].items():
        verify_file_entry(entry, label)
    verify_file_entry(protocol["runtime"]["launcher"], "runtime launcher")
    python_path = verify_file_entry(protocol["runtime"]["python_executable"], "Python executable")
    require(same_path(Path(sys.executable), python_path), "Python executable drift")
    require(sys.version.split()[0] == protocol["runtime"]["python_version"], "Python version drift")
    require(importlib.metadata.version("numpy") == protocol["runtime"]["numpy_version"], "numpy version drift")
    require(importlib.metadata.version("Pillow") == protocol["runtime"]["pillow_version"], "Pillow version drift")
    passed = {
        "source_scope": source_scope_path,
        "phase_a_result": phase_a_result_path,
        "phase_a_manifest": phase_a_manifest_path,
        "head_governed_result": head_governed_path,
        "head_machine_result": head_machine_path,
    }
    for label, path in passed.items():
        require(same_path(path, verify_file_entry(protocol["frozen_files"][label], label)), f"passed {label} path drift")
    _validate_activation(activation)
    require(activation["protocol_sha256"] == sha256_file(protocol_path), "activation protocol SHA drift")
    for label in ("source_scope", "phase_a_result", "phase_a_manifest", "head_governed_result", "head_machine_result"):
        require(
            activation["bindings"][label]["sha256"] == protocol["frozen_files"][label]["sha256"],
            f"activation binding drift: {label}",
        )
    require(activation["request_scope"] == {"identity_count": 32, "asset_count": 64, "assets": list(ASSETS), "selected_frame_count": 9600}, "activation request scope drift")
    require(protocol["truth_support_thresholds"] == source_scope["unchanged_future_support_gates_per_identity"], "truth support threshold drift")
    require(protocol["truth_reader_policy"] == truth_policy_dict(TruthReaderPolicy()), "truth reader policy drift")
    require(source_scope["future_selection_rule"]["full_exact_32_processing_required"] is True, "full-pool rule drift")
    require(source_scope["future_selection_rule"]["target_count"] == 16, "target count drift")
    return protocol, activation, source_scope, phase_a_result, phase_a, head_machine


def validate_resume_inventory(
    *, output_root: Path, selected: list[dict[str, Any]], attempt: dict[str, Any]
) -> list[dict[str, Any]]:
    require(not (output_root / "manifest.json").exists(), "completed manifest forbids resume")
    temporary = output_root / "_temporary_downloads"
    require(not temporary.exists(), "temporary/partial inventory makes attempt non-resumable")
    receipts_root = output_root / "receipts"
    source_root = output_root / "source" / "Training"
    receipts = sorted(receipts_root.glob("[0-9][0-9][0-9]-*.json")) if receipts_root.exists() else []
    receipts = [path for path in receipts if not path.name.endswith(".sha256.json")]
    completed: list[dict[str, Any]] = []
    for index, path in enumerate(receipts, start=1):
        require(index <= len(selected), "extra checkpoint")
        expected = checkpoint_path(receipts_root, selected[index - 1])
        require(path.name == expected.name, "checkpoint inventory is not a continuous prefix")
        value = read_sealed_json(path, CHECKPOINT_SCHEMA)
        identity = selected[index - 1]
        for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold"):
            require(value[key] == identity[key], f"checkpoint identity drift: {key}")
        require(value["attempt_sha256"] == hashlib.sha256(json_bytes(attempt)).hexdigest().upper(), "checkpoint attempt drift")
        expected_stem_digest = hashlib.sha256(
            ("\n".join(identity["selected_frame_stems"]) + "\n").encode("ascii")
        ).hexdigest().upper()
        require(value["selected_frame_count"] == 300, "checkpoint selected frame count drift")
        require(value["selected_frame_plan_sha256"] == expected_stem_digest, "checkpoint frame plan drift")
        require(value["truth_reader_policy"] == attempt["policy"]["truth_reader_policy"], "checkpoint truth policy drift")
        require(value["truth_support_thresholds"] == attempt["policy"]["truth_support_thresholds"], "checkpoint threshold drift")
        checkpoint_pass, checkpoint_failures = qualifies(
            value["truth_support"], value["truth_support_thresholds"]
        )
        require(checkpoint_pass is value["source_truth_support_qualified"], "checkpoint qualification drift")
        require(checkpoint_failures == value["qualification_failures"], "checkpoint qualification failures drift")
        require(value["range_get_used"] is False and value["redirect_followed"] is False, "checkpoint transport boundary drift")
        require(value["role_assigned"] is False and value["training"] is False, "checkpoint role/training drift")
        require(value["development_outcome_read"] is False and value["r2_access"] == "NONE", "checkpoint outcome boundary drift")
        expected_dir = source_root / str(identity["video_id"])
        require(expected_dir.is_dir(), "checkpoint source directory missing")
        require(all(child.is_file() for child in expected_dir.iterdir()), "checkpoint source subdirectory forbidden")
        files = sorted(path.name for path in expected_dir.iterdir())
        require(files == sorted(ASSETS), "checkpoint source inventory drift")
        require(
            [entry["asset"] for entry in value["source_assets"]] == list(ASSETS),
            "checkpoint source receipt family/order drift",
        )
        for entry in value["source_assets"]:
            expected_path = expected_dir / str(entry["asset"])
            actual = Path(str(entry["path"]))
            require(same_path(actual, expected_path), "checkpoint source path drift")
            require(actual.stat().st_size == int(entry["bytes"]) and sha256_file(actual) == entry["sha256"], "checkpoint source seal drift")
        completed.append(value)
    expected_dirs = {str(row["video_id"]) for row in selected[: len(completed)]}
    actual_dirs: set[str] = set()
    if source_root.exists():
        for child in source_root.iterdir():
            require(child.is_dir(), f"extra source-root file: {child}")
            actual_dirs.add(child.name)
    require(actual_dirs == expected_dirs, "orphan or sparse source inventory")
    sidecars = sorted(receipts_root.glob("*.sha256.json")) if receipts_root.exists() else []
    require(len(sidecars) == len(receipts), "checkpoint sidecar inventory drift")
    if receipts_root.exists():
        expected_receipt_names = {path.name for path in receipts} | {
            path.with_suffix(".sha256.json").name for path in receipts
        }
        require(
            {path.name for path in receipts_root.iterdir()} == expected_receipt_names,
            "extra receipt inventory",
        )
    allowed_root_names = {"attempt.json", "receipts", "source"}
    require(
        {path.name for path in output_root.iterdir()}.issubset(allowed_root_names),
        "extra attempt-root inventory",
    )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--phase-a-result", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--head-governed-result", type=Path, required=True)
    parser.add_argument("--head-machine-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol, _, source_scope, phase_a_result, phase_a, head_machine = validate_bindings(
        protocol_path=args.protocol,
        activation_path=args.activation,
        source_scope_path=args.source_scope,
        phase_a_result_path=args.phase_a_result,
        phase_a_manifest_path=args.phase_a_manifest,
        head_governed_path=args.head_governed_result,
        head_machine_path=args.head_machine_result,
    )
    expected_output = REPO_ROOT / protocol["output_root"]
    require(same_path(args.output_root, expected_output), "output root differs from frozen protocol")
    require_under(args.output_root, REPO_ROOT / "artifacts.local", "output root")
    selected = phase_a_selection(source_scope, phase_a_result, phase_a)
    heads = head_lookup(head_machine, selected)
    declared_bytes = sum(int(row["content_length_bytes"]) for row in heads.values())
    require(declared_bytes == int(protocol["declared_download_bytes"]), "declared byte total drift")
    maximum_identity_bytes = max(
        sum(int(heads[(row["video_id"], asset)]["content_length_bytes"]) for asset in ASSETS)
        for row in selected
    )
    required_free = declared_bytes + 2 * maximum_identity_bytes + int(protocol["free_space_margin_bytes"])
    free = shutil.disk_usage(args.output_root.parent if args.output_root.parent.exists() else args.output_root.parent.parent).free
    require(free >= required_free, f"insufficient free space: {free} < {required_free}")
    attempt = expected_attempt(
        protocol_path=args.protocol,
        activation_path=args.activation,
        source_scope_path=args.source_scope,
        phase_a_result_path=args.phase_a_result,
        phase_a_manifest_path=args.phase_a_manifest,
        head_governed_path=args.head_governed_result,
        head_machine_path=args.head_machine_result,
        output_root=args.output_root,
        selected=selected,
        protocol=protocol,
        declared_bytes=declared_bytes,
    )
    attempt_path = args.output_root / "attempt.json"
    if args.resume:
        require(args.output_root.is_dir(), "resume root missing")
        require(attempt_path.is_file() and load_json(attempt_path) == attempt, "attempt receipt drift")
    else:
        require(not args.output_root.exists(), f"fresh output root already exists: {args.output_root}")
        args.output_root.mkdir(parents=True, exist_ok=False)
        write_json_exclusive(attempt_path, attempt)
    completed = validate_resume_inventory(output_root=args.output_root, selected=selected, attempt=attempt)
    receipts_root = args.output_root / "receipts"
    source_root = args.output_root / "source" / "Training"
    phase_a_root = args.phase_a_manifest.parent
    policy = TruthReaderPolicy()
    policy.validate()
    thresholds = protocol["truth_support_thresholds"]
    attempt_sha = hashlib.sha256(json_bytes(attempt)).hexdigest().upper()
    for identity in selected[len(completed) :]:
        video_id = str(identity["video_id"])
        temporary = args.output_root / "_temporary_downloads" / str(uuid.uuid4())
        identity_source = source_root / video_id
        require(not identity_source.exists(), f"orphan source directory: {identity_source}")
        source_assets: list[dict[str, Any]] = []
        try:
            for asset in ASSETS:
                row = heads[(video_id, asset)]
                output = identity_source / asset
                receipt = download_file(row, output, temporary / asset.replace(".zip", ""))
                source_assets.append(receipt)
            audit = audit_identity(
                identity=identity,
                depth_path=identity_source / "lowres_depth.zip",
                confidence_path=identity_source / "confidence.zip",
                phase_a_root=phase_a_root,
                policy=policy,
                thresholds=thresholds,
            )
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "attempt_sha256": attempt_sha,
                **{key: identity[key] for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold")},
                "selected_frame_count": 300,
                "selected_frame_plan_sha256": audit["selected_frame_plan_sha256"],
                "source_assets": source_assets,
                **audit,
                "truth_reader_policy": truth_policy_dict(policy),
                "truth_support_thresholds": thresholds,
                "range_get_used": False,
                "redirect_followed": False,
                "role_assigned": False,
                "training": False,
                "development_outcome_read": False,
                "r2_access": "NONE",
            }
            write_sealed_json(checkpoint_path(receipts_root, identity), checkpoint)
            completed.append(checkpoint)
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "total": 32,
                        "video_id": video_id,
                        "qualified": checkpoint["source_truth_support_qualified"],
                        "qualified_so_far": sum(bool(row["source_truth_support_qualified"]) for row in completed),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary_parent = args.output_root / "_temporary_downloads"
            if temporary_parent.exists() and not any(temporary_parent.iterdir()):
                temporary_parent.rmdir()

    passed, selected_phase_b = finalize_selection(completed)
    terminal = PASS_TERMINAL if passed else FAIL_TERMINAL
    qualified = [row for row in completed if row["source_truth_support_qualified"]]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "terminal": terminal,
        "bindings": attempt["bindings"],
        "attempt_sha256": attempt_sha,
        "declared_download_bytes": declared_bytes,
        "downloaded_body_bytes": sum(int(asset["bytes"]) for row in completed for asset in row["source_assets"]),
        "identity_count": 32,
        "processed_identity_count": len(completed),
        "selected_frame_count": 9600,
        "source_truth_support_qualified_identity_count": len(qualified),
        "source_truth_support_qualified_identities": [
            {key: row[key] for key in ("selection_order", "pool_order", "visit_id", "video_id")}
            for row in qualified
        ],
        "phase_b_selection_locked": passed,
        "selected_identity_count": len(selected_phase_b),
        "selected_phase_b": selected_phase_b,
        "processed": completed,
        "truth_reader_policy": truth_policy_dict(policy),
        "truth_support_thresholds": thresholds,
        "source_archives_retained_for_offline_validation": True,
        "per_frame_truth_retained": False,
        "rgb_read": False,
        "model_output_read": False,
        "role_assignment_made": False,
        "training": False,
        "development_outcome_read": False,
        "r2_access": "NONE",
        "performance_claim": False,
        "android_default_authority": False,
        "production_authority": False,
        "safety_authority": False,
        "next_gate": PASS_NEXT_GATE if passed else None,
    }
    write_sealed_json(args.output_root / "manifest.json", manifest)
    print(json.dumps({"terminal": terminal, "qualified": len(qualified), "selected": len(selected_phase_b)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
