#!/usr/bin/env python3
"""Materialize the D3R3 Phase-B source-member coverage census.

This recovery stage downloads only the already scoped exact 64 depth/confidence
archives, validates their transport/container integrity, and inventories member
names against the unchanged exact-32 x exact-300 Phase-A frame plan.  It never
decodes pixels, derives source truth, evaluates a support gate, or makes a
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[5]
PROTOCOL_PATH = REPO_ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_CENSUS_PROTOCOL_2026-08-13.json"
ASSETS = ("lowres_depth.zip", "confidence.zip")
COVERAGE_KEYS = (
    "archive_file_count",
    "png_frame_member_count",
    "selected_present_count",
    "selected_missing_count",
    "selected_missing_stems",
    "selected_extra_member_count",
    "selected_present_stems_sha256",
    "archive_member_payload_bytes_read",
    "zip_crc_verified",
)
PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_census_protocol_v1"
SOURCE_SCOPE_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_scope_receipt_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_census_activation_v1"
ATTEMPT_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_attempt_v1"
CHECKPOINT_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_asset_checkpoint_v1"
FAILURE_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_asset_failure_v1"
MANIFEST_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_manifest_v1"
PASS_TERMINAL = "D3R3_PHASE_B_EXACT64_MEMBER_COVERAGE_CENSUS_COMPLETE_NO_MEMBER_PAYLOAD_OR_TRUTH_READ"
PASS_NEXT_GATE = "EXPLICIT_D3R3_PHASE_B_MISSING_SOURCE_POLICY_REGISTRATION"

ACTIVATION_TRUE_AUTHORITY = (
    "exact64_get",
    "archive_container_read",
    "archive_member_name_read",
    "zip_directory_parse",
)
ACTIVATION_FALSE_AUTHORITY = (
    "range_get",
    "redirect_following",
    "archive_member_payload_read",
    "zip_crc_verification",
    "pixel_decode",
    "source_truth",
    "phase_b_selection",
    "phase_c_rgb",
    "model_output",
    "role_assignment",
    "training",
    "development_outcome",
    "performance",
    "android_default",
    "production",
    "safety",
)
ACTIVATION_TOP_LEVEL_FIELDS = (
    "schema", "activation_id", "activated_at", "activated_by",
    "authorization_verbatim", "authorization_context", "status", "protocol",
    "protocol_sha256", "bindings", "request_scope", "execution_policy",
    "authority", "forbidden", "next_action",
)
ACTIVATION_EXECUTION_POLICY = {
    "one_shot_census_terminal": True,
    "fresh_output_root": True,
    "overwrite": False,
    "resume_only_from_exact_continuous_checkpoint_prefix": True,
    "all_64_before_terminal": True,
    "archive_member_payload_read": False,
    "pixel_decode": False,
    "source_truth": False,
    "selection_evaluated": False,
    "stop_after_census_and_offline_validation": True,
}
ACTIVATION_FORBIDDEN = (
    "reuse any D3R2 r0 body, checkpoint, failure, temp marker or partial coverage",
    "use Range or follow redirects",
    "change, extend, reorder or replace the exact 32 identities or 9,600 selected stems",
    "open, read or decompress any ZIP member payload, including testzip",
    "present ZipInfo CRC metadata as verified payload integrity",
    "decode depth/confidence pixels or derive source truth/support",
    "remove, replace, nearest-join or reinterpret a missing exact stem",
    "make a Phase-B selection or open RGB/model/role/training/Development/R2 authority",
    "claim performance, Android-default, production or safety authority",
)
ACTIVATION_NEXT_ACTION = "RUN_FROZEN_D3R3_PHASE_B_SOURCE_COVERAGE_PRODUCER_THEN_INDEPENDENT_OFFLINE_VALIDATOR"
SOURCE_SCOPE_AUTHORITY_FIELDS = (
    "source_scope_registration", "protocol_design", "synthetic_tests", "media_head", "media_get",
    "range_get", "archive_container_read", "archive_member_name_read",
    "archive_member_payload_read", "pixel_decode", "source_truth",
    "phase_b_selection", "phase_c_rgb", "model_output", "role_assignment",
    "training", "development_outcome", "r2_access", "performance",
    "android_default", "production", "safety",
)
TRANSIENT_TRANSPORT_ERROR_TYPES = {
    "URLError", "TimeoutError", "ConnectionError", "BrokenPipeError",
    "ConnectionAbortedError", "ConnectionRefusedError", "ConnectionResetError",
}


class DownloadFailure(OSError):
    def __init__(self, error: Exception | None, history: list[dict[str, Any]]) -> None:
        super().__init__(f"download failed under frozen retry policy: {error}; history={history}")
        self.history = history


class BodyShortRead(OSError):
    def __init__(self, expected: int, received: int) -> None:
        super().__init__(f"premature EOF: expected {expected} bytes, received {received}")
        self.expected = expected
        self.received = received


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


def read_sealed_json(path: Path, schema: str) -> dict[str, Any]:
    require(path.is_file(), f"sealed JSON missing: {path}")
    payload = path.read_bytes()
    seal = load_json(path.with_suffix(".sha256.json"))
    require(
        seal == {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()},
        f"seal mismatch: {path}",
    )
    value = json.loads(payload.decode("utf-8"))
    require(value.get("schema") == schema, f"schema drift: {path}")
    return value


def selection_rows(
    source_scope: dict[str, Any], phase_a_result: dict[str, Any], phase_a_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    result_rows = phase_a_result["selected_phase_a"]
    identities = source_scope["exact_phase_a_selection"].get("identities", result_rows)
    manifest_rows = phase_a_manifest["selected_phase_a"]
    require(len(identities) == len(result_rows) == len(manifest_rows) == 32, "Phase-A selection count drift")
    processed = {
        (int(row["pool_order"]), str(row["visit_id"]), str(row["video_id"])): row
        for row in phase_a_manifest["processed"]
    }
    require(len(processed) == 127, "Phase-A processed inventory drift")
    selected: list[dict[str, Any]] = []
    for order, (scope_row, result_row, manifest_row) in enumerate(
        zip(identities, result_rows, manifest_rows, strict=True), start=1
    ):
        for key in ("selection_order", "pool_order", "visit_id", "video_id"):
            require(scope_row[key] == result_row[key] == manifest_row[key], f"Phase-A selection drift: {key}")
        require(scope_row["selection_order"] == order, "Phase-A selection order drift")
        key = (int(scope_row["pool_order"]), str(scope_row["visit_id"]), str(scope_row["video_id"]))
        require(key in processed and processed[key]["eligible"] is True, "Phase-A processed selection drift")
        stems = [str(value) for value in manifest_row["selected_frame_stems"]]
        require(stems == processed[key]["selected_frame_stems"], "Phase-A frame plan drift")
        require(len(stems) == len(set(stems)) == 300, "Phase-A exact-300 drift")
        require(all(stem.startswith(f"{scope_row['video_id']}_") for stem in stems), "frame/video drift")
        selected.append(dict(scope_row) | {"fold": "Training", "selected_frame_stems": stems})
    selection_lines = [
        f"{row['selection_order']}/{row['pool_order']}/{row['visit_id']}/{row['video_id']}"
        for row in selected
    ]
    selection_sha = hashlib.sha256(("\n".join(selection_lines) + "\n").encode("ascii")).hexdigest().upper()
    require(selection_sha == source_scope["exact_phase_a_selection"]["selection_sha256"], "selection digest drift")
    require(frame_plan_sha256(selected) == source_scope["registered_asset_scope"]["selected_frame_plan_sha256"], "frame plan digest drift")
    return selected


def frame_plan_sha256(selected: list[dict[str, Any]]) -> str:
    lines = [
        f"{row['selection_order']}:{row['pool_order']}:{row['visit_id']}:{row['video_id']}:{stem}"
        for row in selected for stem in row["selected_frame_stems"]
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest().upper()


def request_plan_sha256(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"{row['selection_order']}/{row['pool_order']}/{row['visit_id']}/{row['video_id']}/{row['asset']}/{row['url']}"
        for row in rows
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest().upper()


def head_lookup(head_result: dict[str, Any], selected: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    require(head_result.get("schema") == "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_head_preflight_v1", "HEAD schema drift")
    require(head_result.get("head_terminal") == "D3R3_PHASE_B_EXACT64_FRESH_HEAD_PASS_MEDIA_BODY_UNOPENED", "HEAD terminal drift")
    require(head_result.get("scientific_terminal") is None, "HEAD scientific terminal drift")
    expected: list[dict[str, Any]] = []
    for identity in selected:
        for asset in ASSETS:
            expected.append(identity | {
                "asset": asset,
                "url": f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{identity['video_id']}/{asset}",
            })
    require(request_plan_sha256(expected) == head_result["request_plan_sha256"], "HEAD request plan drift")
    rows = head_result.get("assets")
    require(isinstance(rows, list) and len(rows) == 64, "HEAD row count drift")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for planned, row in zip(expected, rows, strict=True):
        for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold", "asset", "url"):
            require(row[key] == planned[key], f"HEAD plan mismatch: {key}")
        require(row["http_status"] == 200 and row["final_url"] == row["url"], "HEAD availability drift")
        require(int(row["content_length_bytes"]) > 0 and row["etag"] and row["last_modified"], "HEAD header drift")
        require(row["redirect_count"] == 0 and row["response_body_bytes_read"] == 0, "HEAD boundary drift")
        key = str(row["video_id"]), str(row["asset"])
        require(key not in result, f"duplicate HEAD row: {key}")
        result[key] = row
    return result


def download_file(
    row: dict[str, Any], output: Path, temporary_root: Path, max_attempts: int = 3,
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
        size = 0
        try:
            request = urllib.request.Request(url, method="GET", headers={
                "User-Agent": "BlindAssist-DepthART-D3R3-phase-b-coverage",
                "Accept-Encoding": "identity",
            })
            digest = hashlib.sha256()
            with open_request(request, timeout=60) as response, partial.open("xb") as stream:
                status = int(response.status)
                final_url = str(response.geturl())
                response_length = int(response.headers.get("Content-Length"))
                response_etag = response.headers.get("ETag")
                response_last_modified = response.headers.get("Last-Modified")
                require(status == 200 and final_url == url, "GET status/final URL drift")
                require(response_length == expected_length, "GET Content-Length differs from HEAD")
                require(response_etag == row["etag"], "GET ETag differs from HEAD")
                require(response_last_modified == row["last_modified"], "GET Last-Modified differs from HEAD")
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            if size < expected_length:
                raise BodyShortRead(expected_length, size)
            require(size == expected_length, "download body exceeded frozen Content-Length")
            os.rename(partial, output)
            history.append({
                "attempt": attempt, "method": "GET", "http_status": status,
                "error": None, "error_type": None, "retry_class": None,
                "expected_body_bytes": expected_length,
                "response_body_bytes_read": size,
            })
            return {
                "asset": row["asset"], "url": url, "bytes": size,
                "sha256": digest.hexdigest().upper(), "path": str(output.resolve()),
                "attempts": attempt, "attempt_history": history,
                "response_http_status": status, "response_final_url": final_url,
                "response_content_length_bytes": response_length,
                "response_etag": response_etag, "response_last_modified": response_last_modified,
                "frozen_head_content_length_bytes": expected_length,
                "frozen_head_etag": row["etag"], "frozen_head_last_modified": row["last_modified"],
                "range_request_used": False, "redirect_followed": False,
                "transport_response_body_bytes_read_total": sum(
                    int(event["response_body_bytes_read"]) for event in history
                ),
            }
        except Exception as error:
            last_error = error
            status = int(error.code) if isinstance(error, urllib.error.HTTPError) else status
            transient_http = status in {408, 429} or (status is not None and 500 <= status <= 599)
            transient_transport = isinstance(error, (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError)) and not isinstance(error, urllib.error.HTTPError)
            transient_short_read = isinstance(error, BodyShortRead)
            retry_class = (
                "TRANSIENT_HTTP" if transient_http else
                "TRANSIENT_TRANSPORT" if transient_transport else
                "TRANSIENT_BODY_SHORT_READ" if transient_short_read else
                "TERMINAL"
            )
            history.append({
                "attempt": attempt, "method": "GET", "http_status": status,
                "error": f"{type(error).__name__}: {error}",
                "error_type": type(error).__name__, "retry_class": retry_class,
                "expected_body_bytes": expected_length,
                "response_body_bytes_read": size,
            })
            if partial.exists():
                partial.unlink()
            if not (transient_http or transient_transport or transient_short_read) or attempt == max_attempts:
                break
    raise DownloadFailure(last_error, history)


def selected_stem_sha256(stems: list[str]) -> str:
    return hashlib.sha256(("\n".join(stems) + "\n").encode("ascii")).hexdigest().upper()


def archive_coverage(path: Path, selected_stems: list[str]) -> dict[str, Any]:
    """Read only the ZIP directory and return exact member-name coverage."""

    mapping: dict[str, str] = {}
    seen_names: set[str] = set()
    file_count = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            pure = PurePosixPath(info.filename.replace("\\", "/"))
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe ZIP member: {info.filename}")
            require(info.filename not in seen_names, f"duplicate ZIP member name: {info.filename}")
            seen_names.add(info.filename)
            if info.is_dir():
                continue
            file_count += 1
            if pure.suffix.lower() != ".png":
                continue
            require(pure.stem not in mapping, f"duplicate ZIP frame stem: {pure.stem}")
            mapping[pure.stem] = info.filename
    require(mapping, f"no PNG members: {path}")
    selected_set = set(selected_stems)
    present = [stem for stem in selected_stems if stem in mapping]
    missing = [stem for stem in selected_stems if stem not in mapping]
    return {
        "archive_file_count": file_count,
        "png_frame_member_count": len(mapping),
        "selected_present_count": len(present),
        "selected_missing_count": len(missing),
        "selected_missing_stems": missing,
        "selected_extra_member_count": len(set(mapping) - selected_set),
        "selected_present_stems_sha256": selected_stem_sha256(present),
        "archive_member_payload_bytes_read": 0,
        "zip_crc_verified": False,
    }


def combine_identity_coverage(
    identity: dict[str, Any], depth: dict[str, Any], confidence: dict[str, Any]
) -> dict[str, Any]:
    stems = [str(value) for value in identity["selected_frame_stems"]]
    require(len(stems) == 300 and len(set(stems)) == 300, "selected frame plan drift")
    depth_summary = {key: depth[key] for key in COVERAGE_KEYS}
    confidence_summary = {key: confidence[key] for key in COVERAGE_KEYS}
    depth_missing = set(depth_summary["selected_missing_stems"])
    confidence_missing = set(confidence_summary["selected_missing_stems"])
    paired_present = [
        stem for stem in stems if stem not in depth_missing and stem not in confidence_missing
    ]
    paired_missing = [stem for stem in stems if stem not in paired_present]
    return {
        "selected_frame_count": 300,
        "selected_frame_plan_sha256": selected_stem_sha256(stems),
        "lowres_depth": depth_summary,
        "confidence": confidence_summary,
        "paired_exact_present_count": len(paired_present),
        "paired_exact_missing_count": len(paired_missing),
        "paired_exact_missing_stems": paired_missing,
        "paired_exact_present_stems_sha256": selected_stem_sha256(paired_present),
        "neighbor_substitution_used": False,
        "source_truth_derived": False,
    }


def request_plan(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in selected:
        for asset in ASSETS:
            rows.append(identity | {"request_order": len(rows) + 1, "asset": asset})
    require(len(rows) == 64, "request plan count drift")
    return rows


def checkpoint_path(receipts_root: Path, row: dict[str, Any]) -> Path:
    asset_token = str(row["asset"]).replace(".zip", "")
    return receipts_root / f"{int(row['request_order']):03d}-{row['video_id']}-{asset_token}.json"


def failure_path(receipts_root: Path, row: dict[str, Any]) -> Path:
    asset_token = str(row["asset"]).replace(".zip", "")
    return receipts_root / "failures" / f"{int(row['request_order']):03d}-{row['video_id']}-{asset_token}.json"


def expected_attempt(
    *,
    protocol_path: Path,
    activation_path: Path,
    source_scope_path: Path,
    phase_a_result_path: Path,
    phase_a_manifest_path: Path,
    head_validation_path: Path,
    head_machine_path: Path,
    output_root: Path,
    selected: list[dict[str, Any]],
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
            "head_validation_sha256": sha256_file(head_validation_path),
            "head_machine_result_sha256": sha256_file(head_machine_path),
            "producer_sha256": sha256_file(Path(__file__)),
        },
        "output_root": str(output_root.resolve()),
        "identity_plan": [
            {key: row[key] for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold")}
            | {"selected_frame_plan_sha256": selected_stem_sha256(row["selected_frame_stems"])}
            for row in selected
        ],
        "asset_plan": [
            {key: row[key] for key in ("request_order", "selection_order", "pool_order", "visit_id", "video_id", "fold", "asset")}
            for row in request_plan(selected)
        ],
        "policy": {
            "assets": list(ASSETS),
            "identity_count": 32,
            "frames_per_identity": 300,
            "total_frame_count": 9600,
            "all_32_processed": True,
            "pixel_decode": False,
            "source_truth": False,
            "selection": False,
            "neighbor_substitution": False,
            "max_attempts": 3,
            "range_get": False,
            "redirect_following": False,
            "source_archives_retained": True,
        },
        "declared_download_bytes": declared_bytes,
        "scientific_terminal": None,
        "selection_evaluated": False,
        "authority": "D3R3_PHASE_B_SOURCE_TRANSPORT_CONTAINER_AND_MEMBER_NAME_COVERAGE_ONLY",
        "rgb_access": False,
        "model_output_access": False,
        "role_assignment": False,
        "training": False,
        "development_outcome": False,
        "r2_access": "NONE",
    }


def validate_activation(activation: dict[str, Any], protocol_path: Path) -> None:
    require(set(activation) == set(ACTIVATION_TOP_LEVEL_FIELDS), "activation top-level field set drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    require(
        activation.get("status") == "D3R3_PHASE_B_SOURCE_MEMBER_COVERAGE_CENSUS_ACTIVATED",
        "activation status drift",
    )
    require(
        isinstance(activation.get("authorization_verbatim"), str)
        and bool(activation["authorization_verbatim"].strip()),
        "activation authorization missing",
    )
    for key in ("activation_id", "activated_at", "authorization_context"):
        require(isinstance(activation.get(key), str) and bool(activation[key].strip()), f"activation {key} missing")
    require(activation.get("activated_by") == "user", "activation author drift")
    require(activation.get("protocol_sha256") == sha256_file(protocol_path), "activation protocol SHA drift")
    protocol_entry = activation["protocol"]
    require(set(protocol_entry) == {"path", "bytes", "sha256"}, "activation protocol binding field set drift")
    bound_protocol_path = Path(str(protocol_entry["path"]))
    if not bound_protocol_path.is_absolute():
        bound_protocol_path = REPO_ROOT / bound_protocol_path
    require(same_path(bound_protocol_path, protocol_path), "activation protocol path drift")
    require(int(protocol_entry["bytes"]) == protocol_path.stat().st_size, "activation protocol byte drift")
    require(protocol_entry["sha256"] == sha256_file(protocol_path), "activation protocol binding SHA drift")
    require(activation["execution_policy"] == ACTIVATION_EXECUTION_POLICY, "activation execution policy drift")
    require(activation["forbidden"] == list(ACTIVATION_FORBIDDEN), "activation forbidden list drift")
    require(activation["next_action"] == ACTIVATION_NEXT_ACTION, "activation next action drift")
    authority = activation["authority"]
    require(
        set(authority) == set(ACTIVATION_TRUE_AUTHORITY + ACTIVATION_FALSE_AUTHORITY + ("r2_access",)),
        "activation authority field set drift",
    )
    for key in ACTIVATION_TRUE_AUTHORITY:
        require(authority.get(key) is True, f"activation missing authority: {key}")
    for key in ACTIVATION_FALSE_AUTHORITY:
        require(authority.get(key) is False, f"activation authority widened: {key}")
    require(authority.get("r2_access") == "NONE", "activation R2 authority widened")


def validate_protocol_activation_contract(protocol: dict[str, Any]) -> None:
    contract = protocol["activation_contract"]
    require(contract["current_activation_exists"] is False, "protocol activation state drift")
    require(contract["required_activation_schema"] == ACTIVATION_SCHEMA, "protocol activation schema drift")
    require(contract["required_status"] == "D3R3_PHASE_B_SOURCE_MEMBER_COVERAGE_CENSUS_ACTIVATED", "protocol activation status drift")
    require(contract["required_top_level_fields"] == list(ACTIVATION_TOP_LEVEL_FIELDS), "protocol activation field contract drift")
    require(contract["required_execution_policy"] == ACTIVATION_EXECUTION_POLICY, "protocol execution policy drift")
    require(contract["must_authorize_true"] == list(ACTIVATION_TRUE_AUTHORITY), "protocol true-authority contract drift")
    require(contract["must_keep_false"] == list(ACTIVATION_FALSE_AUTHORITY), "protocol false-authority contract drift")
    require(contract["r2_access"] == "NONE", "protocol R2 activation contract drift")
    require(contract["required_forbidden"] == list(ACTIVATION_FORBIDDEN), "protocol forbidden contract drift")
    require(contract["required_next_action"] == ACTIVATION_NEXT_ACTION, "protocol activation next-action drift")


def validate_bindings(
    *,
    protocol_path: Path,
    activation_path: Path,
    source_scope_path: Path,
    phase_a_result_path: Path,
    phase_a_manifest_path: Path,
    head_validation_path: Path,
    head_machine_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(same_path(protocol_path, PROTOCOL_PATH), "protocol path drift")
    protocol = load_json(protocol_path)
    activation = load_json(activation_path)
    source_scope = load_json(source_scope_path)
    phase_a_result = load_json(phase_a_result_path)
    phase_a_manifest = load_json(phase_a_manifest_path)
    head_machine = load_json(head_machine_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol.get("status") == "D3R3_PHASE_B_COVERAGE_CENSUS_PROTOCOL_FROZEN_FRESH_HEAD_BOUND", "protocol status drift")
    validate_protocol_activation_contract(protocol)
    require(source_scope.get("schema") == SOURCE_SCOPE_SCHEMA, "source scope schema drift")
    require(source_scope.get("status") == "D3R3_PHASE_B_SOURCE_COVERAGE_TRANSPORT_RECOVERY_SCOPE_REGISTERED_MEDIA_UNOPENED", "source scope status drift")
    current_authority = source_scope["current_authority"]
    require(set(current_authority) == set(SOURCE_SCOPE_AUTHORITY_FIELDS), "source scope authority field set drift")
    require(current_authority.get("source_scope_registration") is True, "source scope registration authority missing")
    require(current_authority.get("protocol_design") is True, "source scope protocol-design authority missing")
    require(current_authority.get("synthetic_tests") is True, "source scope synthetic-test authority missing")
    for key in ("media_head", "media_get", "range_get", "archive_container_read", "archive_member_name_read", "archive_member_payload_read", "pixel_decode", "source_truth", "phase_b_selection", "phase_c_rgb", "model_output", "role_assignment", "training", "development_outcome", "performance", "android_default", "production", "safety"):
        require(current_authority.get(key) is False, f"source scope authority widened: {key}")
    require(current_authority.get("r2_access") == "NONE", "source scope R2 authority widened")
    registered = source_scope["registered_asset_scope"]
    require(registered["assets"] == list(ASSETS), "source scope asset family/order drift")
    require(registered["identity_count"] == 32 and registered["future_body_asset_count"] == 64, "source scope count drift")
    require(registered["total_selected_frame_count"] == 9600, "source scope frame count drift")
    require(registered["request_plan_sha256"] == "F957FDDE8423D262BF7652B51AD42B7C877F26BB1A776C32B8920C25A62CFEB6", "source scope request digest drift")
    for label, entry in protocol["frozen_files"].items():
        verify_file_entry(entry, label)
    verify_file_entry(protocol["runtime"]["launcher"], "runtime launcher")
    python_path = verify_file_entry(protocol["runtime"]["python_executable"], "Python executable")
    require(same_path(Path(sys.executable), python_path), "Python executable drift")
    require(sys.version.split()[0] == protocol["runtime"]["python_version"], "Python version drift")
    passed = {
        "source_scope": source_scope_path,
        "phase_a_result": phase_a_result_path,
        "phase_a_manifest": phase_a_manifest_path,
        "head_validation": head_validation_path,
        "head_machine_result": head_machine_path,
    }
    for label, path in passed.items():
        require(same_path(path, verify_file_entry(protocol["frozen_files"][label], label)), f"passed {label} path drift")
    validate_activation(activation, protocol_path)
    activation_labels = (
        "source_scope",
        "phase_a_result",
        "phase_a_manifest",
        "head_validation",
        "head_machine_result",
        "d3r2_execution_stop",
    )
    require(set(activation["bindings"]) == set(activation_labels), "activation binding field set drift")
    for label in activation_labels:
        require(
            activation["bindings"][label] == protocol["frozen_files"][label],
            f"activation binding drift: {label}",
        )
    require(activation["request_scope"] == {
        "identity_count": 32,
        "asset_count": 64,
        "assets": list(ASSETS),
        "selected_frame_count": 9600,
    }, "activation request scope drift")
    require(protocol["output_root_existed_at_freeze"] is False, "protocol root-freeze fact drift")
    return protocol, source_scope, phase_a_result, phase_a_manifest, head_machine


def validate_transport_receipt(
    receipt: dict[str, Any], planned: dict[str, Any], head: dict[str, Any]
) -> None:
    expected_keys = {
        "asset", "url", "bytes", "sha256", "path", "attempts", "attempt_history",
        "response_http_status", "response_final_url", "response_content_length_bytes",
        "response_etag", "response_last_modified", "frozen_head_content_length_bytes",
        "frozen_head_etag", "frozen_head_last_modified", "range_request_used",
        "redirect_followed", "transport_response_body_bytes_read_total",
    }
    require(set(receipt) == expected_keys, "source receipt field set drift")
    require(receipt["asset"] == planned["asset"] and receipt["url"] == head["url"], "source receipt plan drift")
    attempts = int(receipt["attempts"])
    history = receipt["attempt_history"]
    require(1 <= attempts <= 3 and isinstance(history, list) and len(history) == attempts, "source retry count drift")
    for index, event in enumerate(history, start=1):
        require(set(event) == {
            "attempt", "method", "http_status", "error", "error_type", "retry_class",
            "expected_body_bytes", "response_body_bytes_read",
        }, "source attempt field set drift")
        require(event["attempt"] == index and event["method"] == "GET", "source attempt order/method drift")
        require(
            event["expected_body_bytes"] == int(head["content_length_bytes"])
            and isinstance(event["response_body_bytes_read"], int)
            and 0 <= event["response_body_bytes_read"] <= event["expected_body_bytes"],
            "source attempt byte accounting drift",
        )
        if index < attempts:
            require(event["error"] and event["error_type"], "successful attempt cannot be retried")
            require(event["retry_class"] in {
                "TRANSIENT_HTTP", "TRANSIENT_TRANSPORT", "TRANSIENT_BODY_SHORT_READ"
            }, "terminal attempt was retried")
            status = event["http_status"]
            if event["retry_class"] == "TRANSIENT_HTTP":
                require(status in {408, 429} or (isinstance(status, int) and 500 <= status <= 599), "HTTP retry class drift")
                require(event["error_type"] == "HTTPError", "HTTP retry error type drift")
            elif event["retry_class"] == "TRANSIENT_TRANSPORT":
                require(status in {None, 200}, "transport retry status drift")
                require(event["error_type"] in TRANSIENT_TRANSPORT_ERROR_TYPES, "transport retry error type drift")
            else:
                require(status == 200, "short-read status drift")
                require(event["error_type"] == "BodyShortRead", "short-read error type drift")
                require(
                    event["response_body_bytes_read"] < event["expected_body_bytes"],
                    "short-read byte evidence drift",
                )
    final = history[-1]
    require(
        final["http_status"] == 200
        and final["error"] is None
        and final["error_type"] is None
        and final["retry_class"] is None,
        "source final attempt is not successful",
    )
    require(
        final["response_body_bytes_read"] == final["expected_body_bytes"]
        == int(head["content_length_bytes"]),
        "source final-attempt body byte drift",
    )
    require(receipt["response_http_status"] == 200, "source response status drift")
    require(receipt["response_final_url"] == receipt["url"] == head["url"], "source final URL drift")
    require(int(receipt["bytes"]) == int(receipt["response_content_length_bytes"]) == int(head["content_length_bytes"]), "source response length drift")
    require(receipt["response_etag"] == receipt["frozen_head_etag"] == head["etag"], "source response ETag drift")
    require(receipt["response_last_modified"] == receipt["frozen_head_last_modified"] == head["last_modified"], "source response Last-Modified drift")
    require(int(receipt["frozen_head_content_length_bytes"]) == int(head["content_length_bytes"]), "source frozen HEAD length drift")
    require(receipt["range_request_used"] is False and receipt["redirect_followed"] is False, "source transport boundary drift")
    require(
        receipt["transport_response_body_bytes_read_total"]
        == sum(int(event["response_body_bytes_read"]) for event in history),
        "source transport total byte drift",
    )
    require(isinstance(receipt["sha256"], str) and len(receipt["sha256"]) == 64, "source SHA format drift")


def verify_asset_checkpoint(
    checkpoint: dict[str, Any], planned: dict[str, Any], source_root: Path,
    heads: dict[tuple[str, str], dict[str, Any]],
) -> None:
    require(set(checkpoint) == {
        "schema", "attempt_sha256", "request_order", "selection_order", "pool_order",
        "visit_id", "video_id", "fold", "asset", "selected_frame_plan_sha256",
        "source_asset", *COVERAGE_KEYS, "transport_response_body_bytes_read",
        "scientific_terminal", "pixel_decode", "source_truth", "selection_evaluated",
        "range_get_used", "redirect_followed", "role_assigned", "training",
        "development_outcome_read", "r2_access",
    }, "asset checkpoint field set drift")
    for key in ("request_order", "selection_order", "pool_order", "visit_id", "video_id", "fold", "asset"):
        require(checkpoint[key] == planned[key], f"asset checkpoint drift: {key}")
    receipt = checkpoint["source_asset"]
    head = heads[(str(planned["video_id"]), str(planned["asset"]))]
    validate_transport_receipt(receipt, planned, head)
    expected = source_root / str(planned["video_id"]) / str(planned["asset"])
    require(same_path(Path(receipt["path"]), expected), "source receipt path drift")
    require(expected.is_file(), f"source body missing: {expected}")
    require(expected.stat().st_size == int(receipt["bytes"]), "source body byte drift")
    require(sha256_file(expected) == receipt["sha256"], "source body SHA drift")
    replay = archive_coverage(expected, planned["selected_frame_stems"])
    require({key: checkpoint[key] for key in COVERAGE_KEYS} == replay, "checkpoint coverage replay drift")
    require(
        checkpoint["transport_response_body_bytes_read"]
        == int(receipt["transport_response_body_bytes_read_total"]),
        "transport-body count drift",
    )
    require(checkpoint["archive_member_payload_bytes_read"] == 0, "member payload boundary drift")
    require(checkpoint["zip_crc_verified"] is False, "CRC authority drift")
    require(checkpoint["scientific_terminal"] is None, "checkpoint scientific terminal drift")
    for key in ("pixel_decode", "source_truth", "selection_evaluated", "range_get_used", "redirect_followed", "role_assigned", "training", "development_outcome_read"):
        require(checkpoint[key] is False, f"checkpoint authority widened: {key}")
    require(checkpoint["r2_access"] == "NONE", "checkpoint R2 authority widened")


def validate_resume_inventory(
    *,
    output_root: Path,
    selected: list[dict[str, Any]],
    attempt: dict[str, Any],
    heads: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    require(not (output_root / "manifest.json").exists(), "completed manifest forbids resume")
    require(not (output_root / "validation.json").exists(), "completed validation forbids resume")
    require(not (output_root / "_temporary_downloads").exists(), "partial downloads make attempt non-resumable")
    receipts_root = output_root / "receipts"
    source_parent = output_root / "source"
    source_root = source_parent / "Training"
    if source_parent.exists():
        require({path.name for path in source_parent.iterdir()} <= {"Training"}, "extra source fold")
    require(not (receipts_root / "failures").exists(), "failed asset makes attempt non-resumable")
    plan = request_plan(selected)
    paths = sorted(
        path for path in receipts_root.glob("[0-9][0-9][0-9]-*.json")
        if not path.name.endswith(".sha256.json")
    ) if receipts_root.exists() else []
    completed: list[dict[str, Any]] = []
    attempt_sha = hashlib.sha256(json_bytes(attempt)).hexdigest().upper()
    for index, path in enumerate(paths, start=1):
        require(index <= len(plan), "extra checkpoint")
        planned = plan[index - 1]
        require(path.name == checkpoint_path(receipts_root, planned).name, "checkpoint prefix drift")
        value = read_sealed_json(path, CHECKPOINT_SCHEMA)
        require(value["attempt_sha256"] == attempt_sha, "checkpoint attempt drift")
        require(value["selected_frame_plan_sha256"] == selected_stem_sha256(planned["selected_frame_stems"]), "checkpoint frame plan drift")
        verify_asset_checkpoint(value, planned, source_root, heads)
        completed.append(value)
    expected_assets: dict[str, set[str]] = {}
    for row in plan[:len(completed)]:
        expected_assets.setdefault(str(row["video_id"]), set()).add(str(row["asset"]))
    actual_dirs = {path.name for path in source_root.iterdir()} if source_root.exists() else set()
    require(actual_dirs == set(expected_assets), "orphan or sparse source inventory")
    for video_id, assets in expected_assets.items():
        directory = source_root / video_id
        require(directory.is_dir(), "source identity directory missing")
        require({path.name for path in directory.iterdir()} == assets, "source asset prefix inventory drift")
    if receipts_root.exists():
        expected_names = {path.name for path in paths} | {path.with_suffix(".sha256.json").name for path in paths}
        require({path.name for path in receipts_root.iterdir()} == expected_names, "extra receipt inventory")
    require({path.name for path in output_root.iterdir()} <= {"attempt.json", "receipts", "source"}, "extra attempt-root inventory")
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--phase-a-result", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--head-validation", type=Path, required=True)
    parser.add_argument("--head-machine-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol, source_scope, phase_a_result, phase_a_manifest, head_machine = validate_bindings(
        protocol_path=args.protocol,
        activation_path=args.activation,
        source_scope_path=args.source_scope,
        phase_a_result_path=args.phase_a_result,
        phase_a_manifest_path=args.phase_a_manifest,
        head_validation_path=args.head_validation,
        head_machine_path=args.head_machine_result,
    )
    expected_output = REPO_ROOT / protocol["output_root"]
    require(same_path(args.output_root, expected_output), "output root differs from frozen protocol")
    require_under(args.output_root, REPO_ROOT / "artifacts.local", "output root")
    selected = selection_rows(source_scope, phase_a_result, phase_a_manifest)
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
        head_validation_path=args.head_validation,
        head_machine_path=args.head_machine_result,
        output_root=args.output_root,
        selected=selected,
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
    completed = validate_resume_inventory(
        output_root=args.output_root, selected=selected, attempt=attempt, heads=heads
    )
    receipts_root = args.output_root / "receipts"
    source_root = args.output_root / "source" / "Training"
    attempt_sha = hashlib.sha256(json_bytes(attempt)).hexdigest().upper()
    plan = request_plan(selected)
    for planned in plan[len(completed):]:
        video_id = str(planned["video_id"])
        asset = str(planned["asset"])
        temporary = args.output_root / "_temporary_downloads" / str(uuid.uuid4())
        source_dir = source_root / video_id
        output = source_dir / asset
        require(not output.exists(), f"orphan source asset: {output}")
        receipt: dict[str, Any] | None = None
        stage = "DOWNLOAD"
        checkpoint_committed = False
        try:
            receipt = download_file(
                heads[(video_id, asset)], output, temporary / asset.replace(".zip", "")
            )
            stage = "ZIP_DIRECTORY_PARSE"
            coverage = archive_coverage(output, planned["selected_frame_stems"])
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA,
                "attempt_sha256": attempt_sha,
                **{key: planned[key] for key in ("request_order", "selection_order", "pool_order", "visit_id", "video_id", "fold", "asset")},
                "selected_frame_plan_sha256": selected_stem_sha256(planned["selected_frame_stems"]),
                "source_asset": receipt,
                **coverage,
                "transport_response_body_bytes_read": int(
                    receipt["transport_response_body_bytes_read_total"]
                ),
                "scientific_terminal": None,
                "pixel_decode": False,
                "source_truth": False,
                "selection_evaluated": False,
                "range_get_used": False,
                "redirect_followed": False,
                "role_assigned": False,
                "training": False,
                "development_outcome_read": False,
                "r2_access": "NONE",
            }
            write_sealed_json(checkpoint_path(receipts_root, planned), checkpoint)
            checkpoint_committed = True
            completed.append(checkpoint)
            print(json.dumps({
                "completed": len(completed),
                "total": 64,
                "video_id": video_id,
                "asset": asset,
                "selected_missing": coverage["selected_missing_count"],
            }, sort_keys=True), flush=True)
        except Exception as error:
            failure = {
                "schema": FAILURE_SCHEMA,
                "attempt_sha256": attempt_sha,
                **{key: planned[key] for key in ("request_order", "selection_order", "pool_order", "visit_id", "video_id", "fold", "asset")},
                "selected_frame_plan_sha256": selected_stem_sha256(planned["selected_frame_stems"]),
                "failure_stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "attempt_history": error.history if isinstance(error, DownloadFailure) else (receipt or {}).get("attempt_history", []),
                "source_body_retained": output.is_file(),
                "scientific_terminal": None,
                "selection_evaluated": False,
                "selected_phase_b": None,
                "next_gate": None,
                "archive_member_payload_bytes_read": 0,
                "pixel_decode": False,
                "source_truth": False,
                "role_assigned": False,
                "training": False,
                "development_outcome_read": False,
                "r2_access": "NONE",
            }
            write_sealed_json(failure_path(receipts_root, planned), failure)
            raise
        finally:
            if checkpoint_committed:
                if temporary.exists():
                    shutil.rmtree(temporary)
                parent = args.output_root / "_temporary_downloads"
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()

    require(len(completed) == 64, "full exact-64 census incomplete")
    checkpoint_lookup = {
        (str(row["video_id"]), str(row["asset"])): row for row in completed
    }
    processed: list[dict[str, Any]] = []
    for identity in selected:
        video_id = str(identity["video_id"])
        depth = checkpoint_lookup[(video_id, "lowres_depth.zip")]
        confidence = checkpoint_lookup[(video_id, "confidence.zip")]
        processed.append(
            {key: identity[key] for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold")}
            | combine_identity_coverage(identity, depth, confidence)
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "terminal": PASS_TERMINAL,
        "scientific_terminal": None,
        "bindings": attempt["bindings"],
        "attempt_sha256": attempt_sha,
        "declared_download_bytes": declared_bytes,
        "downloaded_body_bytes": sum(int(row["source_asset"]["bytes"]) for row in completed),
        "transport_response_body_bytes_read": sum(
            int(row["source_asset"]["transport_response_body_bytes_read_total"])
            for row in completed
        ),
        "archive_container_bytes_sha256_hashed": sum(int(row["source_asset"]["bytes"]) for row in completed),
        "asset_checkpoint_count": 64,
        "identity_count": 32,
        "processed_identity_count": 32,
        "selected_frame_count": 9600,
        "paired_exact_present_frame_count": sum(int(row["paired_exact_present_count"]) for row in processed),
        "paired_exact_missing_frame_count": sum(int(row["paired_exact_missing_count"]) for row in processed),
        "identities_with_any_paired_missing": sum(int(row["paired_exact_missing_count"] > 0) for row in processed),
        "processed": processed,
        "source_archives_retained_for_offline_validation": True,
        "archive_member_payload_bytes_read": 0,
        "zip_directory_parsed_count": 64,
        "zip_crc_verified": False,
        "pixel_decode": False,
        "source_truth_derived": False,
        "truth_support_gate_evaluated": False,
        "selection_evaluated": False,
        "phase_b_selection_locked": False,
        "selected_phase_b": None,
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
        "next_gate": PASS_NEXT_GATE,
    }
    write_sealed_json(args.output_root / "manifest.json", manifest)
    print(json.dumps({
        "terminal": PASS_TERMINAL,
        "paired_exact_missing": manifest["paired_exact_missing_frame_count"],
        "next_gate": PASS_NEXT_GATE,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
