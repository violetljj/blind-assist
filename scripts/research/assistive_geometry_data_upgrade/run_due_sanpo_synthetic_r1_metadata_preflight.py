"""Run the exact AG-DUE R1 SANPO-Synthetic metadata preflight.

This runner may read four metadata objects and list three exact frame-object
prefixes.  It must never request an RGB, panoptic-mask, or metric-depth body.
The result is inventory evidence only; it cannot establish task truth or source
support.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from scripts.research.assistive_geometry_data_upgrade import (
    validate_due_sanpo_synthetic_r1_protocol as r1_validator,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / (
    "docs/research/assistive-geometry-data-upgrade/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_"
    "OBJECT_INVENTORY_PREFLIGHT_EXECUTION_LOCK_2026-08-10.json"
)
OUTPUT_ROOT = REPO_ROOT / (
    "artifacts.local/evidence/assistive-geometry-data-upgrade/"
    "sanpo-synthetic-r1-metadata-preflight"
)
EXECUTION_ID = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_"
    "OBJECT_INVENTORY_PREFLIGHT_EXECUTION_2026-08-10"
)
SOURCE_ID = "sanpo_synthetic_v0_train_discovery"
SESSION_ID = "17c7d6bc6d4d4573afecc730cabf4db65f66b04ced504396a71d1185920179cb"
CAMERA = "camera_chest"
LENS = "left"
BUCKET = "gresearch"
API_HOST = "storage.googleapis.com"
API_ROOT = f"https://{API_HOST}/storage/v1/b/{BUCKET}/o"
MEDIA_ROOT = f"https://{API_HOST}/{BUCKET}"
INDEX_PATTERN = re.compile(r"^[0-9]{6}$")
BODY_CANARY_SUCCESSOR = (
    "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_25_FRAME_DEPTH_"
    "PANOPTIC_BODY_CANARY_PROTOCOL_LOCK"
)
STOP_SUCCESSOR = "NONE_STOP_AT_PREFLIGHT_TERMINAL"


class PreflightError(ValueError):
    """A frozen preflight contract, source receipt, or budget was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_metadata_objects() -> list[dict[str, str]]:
    prefix = f"sanpo_dataset/v0/sanpo-synthetic/{SESSION_ID}"
    return [
        {"role": "session_description", "path": f"{prefix}/description.json"},
        {"role": "global_labelmap", "path": "sanpo_dataset/v0/labelmap.json"},
        {
            "role": "frame_annotation_type",
            "path": f"{prefix}/{CAMERA}/{LENS}/frame_segmentation_annotation_type.json",
        },
        {"role": "camera_pose_table", "path": f"{prefix}/{CAMERA}/camera_poses.csv"},
    ]


def _expected_frame_prefixes() -> list[dict[str, str]]:
    prefix = f"sanpo_dataset/v0/sanpo-synthetic/{SESSION_ID}/{CAMERA}/{LENS}"
    return [
        {"role": "rgb", "prefix": f"{prefix}/video_frames/", "suffix": ".png"},
        {
            "role": "panoptic_mask",
            "prefix": f"{prefix}/segmentation_masks/",
            "suffix": ".png",
        },
        {
            "role": "metric_depth",
            "prefix": f"{prefix}/depth_maps/",
            "suffix": ".float16.gz",
        },
    ]


def validate_execution_lock(lock: dict[str, Any], repo_root: Path = REPO_ROOT) -> None:
    require(
        set(lock)
        == {
            "schema",
            "execution_id",
            "status",
            "predecessor_protocol",
            "locked_source",
            "provider_contract",
            "object_contract",
            "parsing_contract",
            "resource_budget",
            "recovery_contract",
            "failure_finalization_contract",
            "output_contract",
            "decision_contract",
            "execution_authority",
            "implementation",
            "claim_ceiling",
        },
        "execution lock field set drift",
    )
    require(
        lock["schema"]
        == "blindassist.assistive_geometry_due.sanpo_synthetic_r1_metadata_preflight_execution.v1",
        "execution lock schema drift",
    )
    require(lock["execution_id"] == EXECUTION_ID, "execution identity drift")
    require(
        lock["status"] == "METADATA_PREFLIGHT_EXECUTION_AUTHORIZED_FRAME_BODY_FORBIDDEN",
        "execution status drift",
    )

    predecessor = lock["predecessor_protocol"]
    require(
        predecessor
        == {
            "path": r1_validator.PROTOCOL_PATH.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(r1_validator.PROTOCOL_PATH),
            "required_successor": r1_validator.SUCCESSOR,
        },
        "predecessor protocol binding drift",
    )
    r1_validator.validate_protocol(r1_validator.load_json(r1_validator.PROTOCOL_PATH), repo_root)

    require(
        lock["locked_source"]
        == {
            "source_id": SOURCE_ID,
            "source_family": "SANPO_SYNTHETIC",
            "official_split": "train",
            "official_split_sha256": "F9C5DC4C289FA87342ABC0D2CC49F112FCC78C7E02E0B6B081E296A99344173C",
            "split_network_refresh": False,
            "session_id": SESSION_ID,
            "camera": CAMERA,
            "lens": LENS,
            "parent_count": 1,
            "fallback_authorized": False,
        },
        "locked source drift",
    )
    require(
        lock["provider_contract"]
        == {
            "provider": "GOOGLE_CLOUD_STORAGE_JSON_API",
            "bucket": BUCKET,
            "api_scheme": "https",
            "api_host": API_HOST,
            "redirect_to_other_host_authorized": False,
            "anonymous_read_only": True,
            "split_object_network_access": False,
        },
        "provider contract drift",
    )
    objects = lock["object_contract"]
    require(objects["metadata_objects"] == _expected_metadata_objects(), "metadata object scope drift")
    require(objects["frame_prefixes"] == _expected_frame_prefixes(), "frame prefix scope drift")
    require(
        objects["allowed_operations"]
        == {
            "metadata_objects": ["GET_OBJECT_METADATA", "GET_OBJECT_BODY_GENERATION_PINNED"],
            "frame_prefixes": ["LIST_OBJECT_METADATA"],
        },
        "allowed operation drift",
    )
    require(
        objects["forbidden_operations"]
        == [
            "GET_RGB_BODY",
            "GET_PANOPTIC_MASK_BODY",
            "GET_METRIC_DEPTH_BODY",
            "OPEN_LOCAL_EXISTING_PAYLOAD",
            "DECODE_FRAME_BODY",
        ],
        "forbidden operation drift",
    )
    require(
        lock["parsing_contract"]
        == {
            "numeric_filename_regex": "^[0-9]{6}$",
            "suffix_removed_as_a_whole": True,
            "numeric_aliases_or_duplicates_rejected": True,
            "eligible_receipt_fields": [
                "name",
                "generation",
                "metageneration",
                "size",
                "md5Hash",
                "crc32c",
            ],
            "numeric_index_intersection": ["rgb", "panoptic_mask", "metric_depth"],
            "selection": "sorted(numeric_index_intersection)[:25]",
            "selection_before_any_frame_body_request": True,
            "minimum_intersection_count": 25,
            "pose_numeric_values_parsed": False,
            "index_divided_by_fps_is_explicit_timestamp": False,
        },
        "parsing contract drift",
    )
    require(
        lock["resource_budget"]
        == {
            "max_network_requests": 80,
            "max_retries_per_request": 2,
            "connect_read_timeout_seconds": 30,
            "max_wall_time_seconds": 180,
            "max_list_pages_per_prefix": 20,
            "max_objects_per_prefix": 10000,
            "max_total_listed_objects": 30000,
            "max_metadata_object_bytes_each": 8388608,
            "max_metadata_object_bytes_total": 16777216,
            "max_network_response_bytes_total": 67108864,
            "max_receipt_output_bytes": 33554432,
            "max_local_disk_bytes": 33554432,
            "max_frame_body_bytes": 0,
            "max_local_existing_payload_bytes_read": 0,
        },
        "resource budget drift",
    )
    require(
        lock["recovery_contract"]
        == {
            "required": True,
            "prior_attempt_lock_sha256": "CC5846BE38A1EC8B9A2A6B0B95F685531F0819400F8A8DF3F8FAE55AE1B7FBD4",
            "prior_attempt_request_count": 2,
            "prior_attempt_failure": "FIRST_METADATA_OBJECT_HEAD_HTTP_ERROR_AFTER_TWO_RETRIES",
            "prior_attempt_frame_body_request_count": 0,
            "prior_attempt_preserved": True,
            "allowed_existing_files_before_retry": ["attempt_receipt.json"],
            "retry_receipt_written_before_network": True,
        },
        "recovery contract drift",
    )
    missing_annotation = _expected_metadata_objects()[2]["path"]
    require(
        lock["failure_finalization_contract"]
        == {
            "required": True,
            "network_authorized": False,
            "allowed_existing_files_before_finalization": [
                "attempt_receipt.json",
                "retry_attempt_receipt.json",
            ],
            "first_attempt_lock_sha256": "CC5846BE38A1EC8B9A2A6B0B95F685531F0819400F8A8DF3F8FAE55AE1B7FBD4",
            "retry_attempt_lock_sha256": "8084A20252125CE25EA87499ACE77B4B231DA5B30C370B29C6AE4800AD62574C",
            "observed_missing_metadata_object": missing_annotation,
            "observed_http_status": 404,
            "requests_per_failed_attempt_derived_from_frozen_control_flow": 6,
            "total_actual_requests_across_two_attempts": 12,
            "retry_receipt_prior_request_count_field_correct": False,
            "frame_prefix_listing_started": False,
            "frame_body_request_count": 0,
            "terminal": "NOT_EVALUABLE",
            "unique_successor": STOP_SUCCESSOR,
        },
        "failure finalization contract drift",
    )
    require(
        lock["output_contract"]
        == {
            "owned_root": r1_validator.OUTPUT_ROOT,
            "overwrite": False,
            "attempt_receipt_written_before_network": True,
            "required_receipts": [
                "attempt_receipt.json",
                "retry_attempt_receipt.json",
                "source_object_inventory.json",
                "metadata_schema_receipt.json",
                "preflight_result.json",
            ],
            "raw_metadata_bytes_persisted": False,
            "frame_body_bytes_persisted": False,
        },
        "output contract drift",
    )
    require(
        lock["decision_contract"]
        == {
            "PASS": "METADATA_INVENTORY_VALID_BODY_CANARY_PROTOCOL_LOCK_ELIGIBLE",
            "PARTIAL_POSE_UNBOUND": "DEPTH_AND_PANOPTIC_BODY_CANARY_PROTOCOL_LOCK_ELIGIBLE_POSE_TEMPORAL_HELD",
            "NOT_EVALUABLE": "STOP_SOURCE_OBJECT_INVENTORY_OR_SCHEMA_INCOMPLETE",
            "REJECT": "STOP_IDENTITY_SPLIT_LICENSE_ANCESTRY_OR_INTEGRITY_CONFLICT",
            "eligible_successor": BODY_CANARY_SUCCESSOR,
            "stop_successor": STOP_SUCCESSOR,
        },
        "decision contract drift",
    )
    require(
        lock["execution_authority"]
        == {
            "metadata_preflight": True,
            "network_exact_gcs_objects": True,
            "metadata_object_body_read": True,
            "frame_object_listing": True,
            "frame_body_request_or_read": False,
            "rgb_visual_access": False,
            "mask_or_depth_decode": False,
            "local_existing_payload_open": False,
            "capability_truth_audit": False,
            "body_canary_execution": False,
            "derivation_teacher_or_training": False,
            "source_data_support": False,
            "dca_or_f1_pass": False,
            "development_confirmation_android_product_safety": False,
        },
        "execution authority drift",
    )
    expected_impl = {
        "scripts/research/assistive_geometry_data_upgrade/run_due_sanpo_synthetic_r1_metadata_preflight.py",
        "scripts/research/assistive_geometry_data_upgrade/test_run_due_sanpo_synthetic_r1_metadata_preflight.py",
    }
    require(set(lock["implementation"]) == expected_impl, "implementation set drift")
    for logical_path, expected_sha in lock["implementation"].items():
        require(sha256_file(repo_root / logical_path) == expected_sha, f"implementation SHA drift: {logical_path}")
    require(
        lock["claim_ceiling"]
        == "One exact SANPO-Synthetic TRAIN-session metadata/object inventory preflight only. Inventory is not capability truth; frame bodies, pose/timestamp admission, source support, DCA/F1 pass, derivation, Teacher, training, Development, Confirmation, Android, product and safety authority remain false.",
        "claim ceiling drift",
    )


class MetadataProvider(Protocol):
    request_count: int
    network_response_bytes: int
    list_pages: dict[str, int]

    def get_object_metadata(self, object_name: str) -> dict[str, Any]: ...

    def read_metadata_object(self, object_name: str, generation: str) -> bytes: ...

    def list_frame_objects(self, prefix: str) -> list[dict[str, Any]]: ...


class GcsMetadataProvider:
    """Exact-host GCS provider with a metadata-only body allowlist."""

    def __init__(self, lock: dict[str, Any], *, initial_request_count: int = 0) -> None:
        self.lock = lock
        self.budget = lock["resource_budget"]
        self.metadata_names = {item["path"] for item in lock["object_contract"]["metadata_objects"]}
        self.frame_prefixes = {item["prefix"] for item in lock["object_contract"]["frame_prefixes"]}
        self.request_count = initial_request_count
        self.network_response_bytes = 0
        self.list_pages: dict[str, int] = {}
        self.started = time.monotonic()

    def _check_budget(self) -> None:
        require(self.request_count < self.budget["max_network_requests"], "network request budget exceeded")
        require(
            time.monotonic() - self.started <= self.budget["max_wall_time_seconds"],
            "wall-time budget exceeded",
        )

    def _read_url(self, url: str, *, max_bytes: int) -> bytes:
        parsed = urlparse(url)
        require(parsed.scheme == "https" and parsed.hostname == API_HOST, "provider host escape")
        last_error: Exception | None = None
        for attempt in range(self.budget["max_retries_per_request"]):
            self._check_budget()
            self.request_count += 1
            try:
                request = Request(url, headers={"User-Agent": "BlindAssist-AG-DUE-R1/1"})
                with urlopen(request, timeout=self.budget["connect_read_timeout_seconds"]) as response:
                    final = urlparse(response.geturl())
                    require(
                        final.scheme == "https" and final.hostname == API_HOST,
                        "redirected provider host escape",
                    )
                    payload = response.read(max_bytes + 1)
                require(len(payload) <= max_bytes, "network response byte cap exceeded")
                self.network_response_bytes += len(payload)
                require(
                    self.network_response_bytes <= self.budget["max_network_response_bytes_total"],
                    "total network response budget exceeded",
                )
                return payload
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt + 1 < self.budget["max_retries_per_request"]:
                    time.sleep(1)
        detail = f" status={last_error.code}" if isinstance(last_error, HTTPError) else ""
        raise PreflightError(f"network request failed: {type(last_error).__name__}{detail}")

    def _json(self, url: str, *, max_bytes: int = 8 * 1024 * 1024) -> dict[str, Any]:
        payload = self._read_url(url, max_bytes=max_bytes)
        value = json.loads(payload.decode("utf-8"))
        require(isinstance(value, dict), "GCS JSON response is not an object")
        return value

    def get_object_metadata(self, object_name: str) -> dict[str, Any]:
        require(object_name in self.metadata_names, "metadata object path escape")
        return self._json(f"{API_ROOT}/{quote(object_name, safe='')}")

    def read_metadata_object(self, object_name: str, generation: str) -> bytes:
        require(object_name in self.metadata_names, "frame or unknown object body read attempted")
        url = f"{MEDIA_ROOT}/{quote(object_name, safe='/')}?{urlencode({'generation': generation})}"
        return self._read_url(url, max_bytes=self.budget["max_metadata_object_bytes_each"])

    def list_frame_objects(self, prefix: str) -> list[dict[str, Any]]:
        require(prefix in self.frame_prefixes, "frame prefix escape")
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        pages = 0
        while True:
            require(pages < self.budget["max_list_pages_per_prefix"], "listing pagination cap reached")
            query: dict[str, Any] = {"prefix": prefix, "maxResults": 1000}
            if page_token:
                query["pageToken"] = page_token
            payload = self._json(f"{API_ROOT}?{urlencode(query)}")
            pages += 1
            page_items = payload.get("items", [])
            require(isinstance(page_items, list), "listing items schema invalid")
            items.extend(page_items)
            require(len(items) <= self.budget["max_objects_per_prefix"], "objects-per-prefix cap exceeded")
            next_token = payload.get("nextPageToken")
            if not next_token:
                break
            require(isinstance(next_token, str), "listing page token schema invalid")
            page_token = next_token
        self.list_pages[prefix] = pages
        return items


def _provider_receipt(item: dict[str, Any], *, expected_name: str, body: bytes | None) -> dict[str, Any]:
    require(str(item.get("name", "")) == expected_name, "object name drift")
    generation = str(item.get("generation", ""))
    metageneration = str(item.get("metageneration", ""))
    require(generation.isdigit() and metageneration.isdigit(), "object generation receipt incomplete")
    try:
        size = int(item["size"])
    except (KeyError, TypeError, ValueError) as error:
        raise PreflightError("object size receipt incomplete") from error
    require(size > 0, "object size must be positive")
    md5_hash = str(item.get("md5Hash", ""))
    crc32c = str(item.get("crc32c", ""))
    require(bool(md5_hash) and bool(crc32c), "object provider hash receipt incomplete")
    receipt = {
        "provider": "GOOGLE_CLOUD_STORAGE_JSON_API",
        "bucket": BUCKET,
        "name": expected_name,
        "generation": generation,
        "metageneration": metageneration,
        "size": size,
        "md5_hash": md5_hash,
        "crc32c": crc32c,
        "content_type": item.get("contentType"),
        "content_encoding": item.get("contentEncoding"),
        "body_read": body is not None,
        "sha256_after_read": None,
    }
    if body is not None:
        require(len(body) == size, "metadata object size mismatch")
        actual_md5 = base64.b64encode(hashlib.md5(body, usedforsecurity=False).digest()).decode("ascii")
        require(actual_md5 == md5_hash, "metadata object MD5 mismatch")
        receipt["sha256_after_read"] = sha256_bytes(body)
        receipt["bytes_read"] = len(body)
        receipt["provider_md5_verified"] = True
        receipt["body_class"] = "METADATA_OBJECT_BYTES"
    else:
        receipt["provider_receipt_only"] = True
        receipt["body_class"] = "FRAME_OBJECT_METADATA_ONLY"
    return receipt


def _json_object(body: bytes, role: str) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"{role} JSON invalid") from error
    require(isinstance(value, dict), f"{role} must be a JSON object")
    return value


def _analyze_description(body: bytes) -> dict[str, Any]:
    value = _json_object(body, "description")
    require(value.get("session_type") == "synthetic", "description session_type drift")
    locations = value.get("session_camera_location")
    details = value.get("session_camera_details")
    require(isinstance(locations, list) and CAMERA in locations, "description camera missing")
    require(isinstance(details, list), "description camera details missing")
    camera_index = locations.index(CAMERA)
    require(camera_index < len(details) and isinstance(details[camera_index], dict), "camera detail alignment invalid")
    detail = details[camera_index]
    params = detail.get(f"{LENS}_camera_params")
    require(isinstance(params, dict), "description lens parameters missing")
    try:
        fps = float(detail["fps"])
        numbers = {name: float(params[name]) for name in ("image_width", "image_height", "fx", "fy", "cx", "cy")}
    except (KeyError, TypeError, ValueError) as error:
        raise PreflightError("description camera numeric schema invalid") from error
    require(math.isfinite(fps) and fps > 0, "description fps invalid")
    require(all(math.isfinite(number) for number in numbers.values()), "description camera values non-finite")
    require(numbers["image_width"] > 0 and numbers["image_height"] > 0, "description dimensions invalid")
    require(numbers["fx"] > 0 and numbers["fy"] > 0, "description focal length invalid")
    return {
        "top_level_key_types": {key: type(value[key]).__name__ for key in sorted(value)},
        "session_type": "synthetic",
        "camera_locations": locations,
        "camera": CAMERA,
        "lens": LENS,
        "fps": fps,
        "camera_parameters": numbers,
        "orientation_from_dimensions": "PORTRAIT" if numbers["image_height"] > numbers["image_width"] else "LANDSCAPE",
        "upright_mapping_admitted": False,
        "capability_orientation_count": 0,
    }


def _analyze_labelmap(body: bytes) -> dict[str, Any]:
    value = _json_object(body, "labelmap")
    require(bool(value), "labelmap empty")
    pairs: list[tuple[str, int]] = []
    for name, raw_id in value.items():
        require(isinstance(name, str) and bool(name.strip()), "labelmap class name invalid")
        require(isinstance(raw_id, int) and not isinstance(raw_id, bool) and raw_id >= 0, "labelmap class id invalid")
        pairs.append((name, raw_id))
    ids = [item[1] for item in pairs]
    require(len(ids) == len(set(ids)), "labelmap class ids are not unique")
    normalized = [{"name": name, "id": class_id} for name, class_id in sorted(pairs)]
    lowered = {name.strip().lower() for name, _ in pairs}
    return {
        "schema": "class_name_to_nonnegative_integer_id",
        "class_count": len(pairs),
        "mapping_sha256": sha256_bytes(canonical_json_bytes(normalized)),
        "unknown_or_void_names_present": sorted(lowered & {"unknown", "void", "unlabeled", "unlabelled"}),
        "obstacle_taxonomy_mapping_executed": False,
        "boundary_truth_materialized": False,
    }


def _analyze_annotation_types(body: bytes) -> dict[str, Any]:
    value = _json_object(body, "annotation types")
    counts: dict[str, int] = {}
    numeric_keys: list[int] = []
    invalid_keys: list[str] = []
    for key, annotation_type in value.items():
        if isinstance(key, str) and key.isdigit():
            numeric_keys.append(int(key))
        else:
            invalid_keys.append(str(key))
        require(isinstance(annotation_type, str) and bool(annotation_type), "annotation type value invalid")
        counts[annotation_type] = counts.get(annotation_type, 0) + 1
    require(not invalid_keys, "annotation type index key invalid")
    return {
        "entry_count": len(value),
        "numeric_key_count": len(numeric_keys),
        "numeric_index_min": min(numeric_keys) if numeric_keys else None,
        "numeric_index_max": max(numeric_keys) if numeric_keys else None,
        "annotation_type_distribution": dict(sorted(counts.items())),
        "mask_body_verified": False,
        "boundary_derivation_executed": False,
    }


def _analyze_pose_table(body: bytes) -> dict[str, Any]:
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise PreflightError("pose table encoding invalid") from error
    rows = list(csv.reader(io.StringIO(text)))
    require(bool(rows) and bool(rows[0]), "pose table header missing")
    header = [item.strip() for item in rows[0]]
    require(all(header), "pose table header contains empty column")
    data_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    uniform = all(len(row) == len(header) for row in data_rows)
    require(uniform, "pose table row width drift")
    normalized = {item.strip().lower() for item in header}
    frame_columns = sorted(normalized & {"frame", "frame_id", "frame_index", "source_frame_index", "image_index"})
    timestamp_columns = sorted(normalized & {"timestamp", "timestamp_ms", "timestamp_us", "timestamp_ns", "time_s"})
    return {
        "header": header,
        "header_sha256": sha256_bytes(",".join(header).encode("utf-8")),
        "data_row_count": len(data_rows),
        "uniform_column_count": True,
        "explicit_frame_columns": frame_columns,
        "explicit_timestamp_columns": timestamp_columns,
        "numeric_pose_values_parsed": False,
        "row_count_coverage_is_frame_binding": False,
        "pose_binding_admitted": False,
        "coordinate_receipt_present": False,
        "coordinate_axis": "UNKNOWN",
        "quaternion_order": "UNRESOLVED",
        "handedness": "UNRESOLVED",
        "transform_direction": "UNRESOLVED",
    }


def _frame_inventory(
    items: list[dict[str, Any]], spec: dict[str, str], max_objects: int
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    require(len(items) <= max_objects, "frame object count budget exceeded")
    indexed: dict[int, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for item in items:
        name = str(item.get("name", ""))
        require(name.startswith(spec["prefix"]), f"{spec['role']} object prefix drift")
        require(name.endswith(spec["suffix"]), f"{spec['role']} object suffix drift")
        stem = name[len(spec["prefix"]) : -len(spec["suffix"])]
        require(bool(INDEX_PATTERN.fullmatch(stem)), f"{spec['role']} filename is not canonical six-digit numeric")
        index = int(stem)
        require(index not in indexed, f"{spec['role']} duplicate numeric index")
        receipt = _provider_receipt(item, expected_name=name, body=None)
        receipt["role"] = spec["role"]
        receipt["numeric_index"] = index
        indexed[index] = receipt
        receipts.append(receipt)
    receipts.sort(key=lambda receipt: receipt["numeric_index"])
    return indexed, receipts


def evaluate_preflight(provider: MetadataProvider, lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    budget = lock["resource_budget"]
    metadata_receipts: dict[str, dict[str, Any]] = {}
    bodies: dict[str, bytes] = {}
    total_metadata_bytes = 0
    for item in lock["object_contract"]["metadata_objects"]:
        name = item["path"]
        metadata = provider.get_object_metadata(name)
        generation = str(metadata.get("generation", ""))
        body = provider.read_metadata_object(name, generation)
        require(len(body) <= budget["max_metadata_object_bytes_each"], "metadata object byte cap exceeded")
        total_metadata_bytes += len(body)
        require(total_metadata_bytes <= budget["max_metadata_object_bytes_total"], "metadata byte budget exceeded")
        metadata_receipts[item["role"]] = _provider_receipt(metadata, expected_name=name, body=body)
        bodies[item["role"]] = body

    schema_receipt = {
        "schema": "blindassist.assistive_geometry_due.sanpo_synthetic_r1_metadata_schema_receipt.v1",
        "execution_id": EXECUTION_ID,
        "metadata_objects": metadata_receipts,
        "description": _analyze_description(bodies["session_description"]),
        "labelmap": _analyze_labelmap(bodies["global_labelmap"]),
        "annotation_types": _analyze_annotation_types(bodies["frame_annotation_type"]),
        "pose_table": _analyze_pose_table(bodies["camera_pose_table"]),
        "raw_metadata_bytes_persisted": False,
    }

    index_by_role: dict[str, dict[int, dict[str, Any]]] = {}
    receipts_by_role: dict[str, list[dict[str, Any]]] = {}
    for spec in lock["object_contract"]["frame_prefixes"]:
        listed = provider.list_frame_objects(spec["prefix"])
        indexed, receipts = _frame_inventory(listed, spec, budget["max_objects_per_prefix"])
        index_by_role[spec["role"]] = indexed
        receipts_by_role[spec["role"]] = receipts
    total_listed = sum(len(items) for items in receipts_by_role.values())
    require(total_listed <= budget["max_total_listed_objects"], "total listed-object budget exceeded")

    intersection = sorted(
        set(index_by_role["rgb"])
        & set(index_by_role["panoptic_mask"])
        & set(index_by_role["metric_depth"])
    )
    selected = intersection[:25]
    missing = {
        role: sorted(set().union(*(set(values) for values in index_by_role.values())) - set(index_by_role[role]))
        for role in ("rgb", "panoptic_mask", "metric_depth")
    }
    selected_depth_bytes = sum(index_by_role["metric_depth"][index]["size"] for index in selected)
    selected_mask_bytes = sum(index_by_role["panoptic_mask"][index]["size"] for index in selected)
    inventory = {
        "schema": "blindassist.assistive_geometry_due.sanpo_synthetic_r1_source_object_inventory.v1",
        "execution_id": EXECUTION_ID,
        "source": {
            "source_id": SOURCE_ID,
            "official_split": "train",
            "session_id": SESSION_ID,
            "camera": CAMERA,
            "lens": LENS,
            "bucket": BUCKET,
        },
        "metadata_object_receipts": metadata_receipts,
        "frame_inventory_receipts": receipts_by_role,
        "list_pages": dict(provider.list_pages),
        "pagination_exhausted": True,
        "inventory_counts": {role: len(receipts_by_role[role]) for role in receipts_by_role},
        "numeric_indices": {role: sorted(index_by_role[role]) for role in index_by_role},
        "numeric_index_intersection": intersection,
        "numeric_index_intersection_count": len(intersection),
        "missing_in_role": missing,
        "selected_lowest_25": selected,
        "selected_lowest_25_sha256": sha256_bytes(canonical_json_bytes(selected)),
        "selection_rule": "sorted(numeric_index_intersection)[:25]",
        "selection_frozen_before_any_frame_body_request": True,
        "frame_body_request_count": 0,
        "frame_body_bytes_read": 0,
        "future_body_canary_provider_size_estimate": {
            "metric_depth_object_count": len(selected),
            "metric_depth_bytes": selected_depth_bytes,
            "panoptic_mask_object_count": len(selected),
            "panoptic_mask_bytes": selected_mask_bytes,
            "total_objects": len(selected) * 2,
            "total_bytes": selected_depth_bytes + selected_mask_bytes,
        },
    }

    if len(intersection) < 25:
        decision = "NOT_EVALUABLE"
        terminal = lock["decision_contract"][decision]
        successor = STOP_SUCCESSOR
    else:
        pose = schema_receipt["pose_table"]
        if pose["explicit_frame_columns"] and pose["explicit_timestamp_columns"]:
            decision = "PASS"
        else:
            decision = "PARTIAL_POSE_UNBOUND"
        terminal = lock["decision_contract"][decision]
        successor = BODY_CANARY_SUCCESSOR

    result = {
        "schema": "blindassist.assistive_geometry_due.sanpo_synthetic_r1_metadata_preflight_result.v1",
        "execution_id": EXECUTION_ID,
        "decision": decision,
        "terminal": terminal,
        "source": inventory["source"],
        "inventory_counts": {
            "rgb_object_inventory_count": len(receipts_by_role["rgb"]),
            "panoptic_object_inventory_count": len(receipts_by_role["panoptic_mask"]),
            "metric_depth_object_inventory_count": len(receipts_by_role["metric_depth"]),
            "numeric_index_intersection_count": len(intersection),
            "pose_table_row_count": schema_receipt["pose_table"]["data_row_count"],
        },
        "capability_counts": {
            "oracle_depth_factor_frames": 0,
            "oracle_support_factor_frames": 0,
            "boundary_truth_frames": 0,
            "explicit_timestamp_frames": 0,
            "pose_transform_frames": 0,
            "portrait_frames": 0,
            "landscape_frames": 0,
        },
        "selected_lowest_25": selected,
        "future_body_canary_provider_size_estimate": inventory["future_body_canary_provider_size_estimate"],
        "pose_and_time": {
            "pose_header_explicit_frame_column_present": bool(schema_receipt["pose_table"]["explicit_frame_columns"]),
            "pose_header_explicit_timestamp_column_present": bool(schema_receipt["pose_table"]["explicit_timestamp_columns"]),
            "row_count_coverage_is_frame_binding": False,
            "pose_binding_admitted": False,
            "pose_transform_materialized": False,
            "explicit_timestamp_materialized": False,
            "index_divided_by_fps_is_explicit_timestamp": False,
            "coordinate_axis": "UNKNOWN",
            "quaternion_order": "UNRESOLVED",
            "handedness": "UNRESOLVED",
            "transform_direction": "UNRESOLVED",
        },
        "execution_disclosure": {
            "network_used": True,
            "exact_metadata_object_head_performed": True,
            "exact_frame_prefix_list_performed": True,
            "metadata_object_bytes_read": True,
            "metadata_bytes_read": total_metadata_bytes,
            "network_request_count": provider.request_count,
            "network_response_bytes": provider.network_response_bytes,
            "frame_body_requested_or_read": False,
            "frame_body_bytes_read": 0,
            "rgb_visual_access": False,
            "mask_or_depth_decoded": False,
            "local_existing_payload_opened": False,
            "teacher_model_derivation_or_training": False,
        },
        "authority": {
            "inventory_is_capability_truth": False,
            "source_data_support_established": False,
            "dca_pass": False,
            "r2_f1_parent_gate_pass": False,
            "body_canary_execution_authorized": False,
            "pose_or_timestamp_admitted": False,
            "support_truth_established": False,
            "boundary_truth_materialized": False,
            "derivation_label_materialization_or_training": False,
            "development_confirmation_android_product_safety": False,
        },
        "unique_successor": successor,
        "claim_ceiling": lock["claim_ceiling"],
    }
    return inventory, schema_receipt, result


def _write_exclusive(path: Path, payload: Any) -> None:
    require(not path.exists(), f"refusing to overwrite receipt: {path.name}")
    path.write_bytes(canonical_json_bytes(payload))


def build_observed_failure_terminal(
    lock: dict[str, Any], *, attempt_sha256: str, retry_attempt_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    failure = lock["failure_finalization_contract"]
    source = {
        "source_id": SOURCE_ID,
        "official_split": "train",
        "session_id": SESSION_ID,
        "camera": CAMERA,
        "lens": LENS,
        "bucket": BUCKET,
    }
    metadata_status = {
        "session_description": "BODY_READ_IN_MEMORY_RECEIPT_NOT_PERSISTED_BEFORE_FAIL_CLOSE",
        "global_labelmap": "BODY_READ_IN_MEMORY_RECEIPT_NOT_PERSISTED_BEFORE_FAIL_CLOSE",
        "frame_annotation_type": "OBJECT_METADATA_HEAD_404_BODY_NOT_READ",
        "camera_pose_table": "NOT_REQUESTED",
    }
    inventory = {
        "schema": "blindassist.assistive_geometry_due.sanpo_synthetic_r1_source_object_inventory.v1",
        "execution_id": EXECUTION_ID,
        "source": source,
        "terminal": "NOT_EVALUABLE",
        "metadata_access_status": metadata_status,
        "metadata_object_receipts": {},
        "frame_inventory_receipts": {"rgb": [], "panoptic_mask": [], "metric_depth": []},
        "list_pages": {},
        "pagination_exhausted": False,
        "inventory_counts": {"rgb": 0, "panoptic_mask": 0, "metric_depth": 0},
        "numeric_indices": {"rgb": [], "panoptic_mask": [], "metric_depth": []},
        "numeric_index_intersection": [],
        "numeric_index_intersection_count": 0,
        "selected_lowest_25": [],
        "selection_not_run_reason": "EXACT_REQUIRED_METADATA_OBJECT_HEAD_404",
        "frame_body_request_count": 0,
        "frame_body_bytes_read": 0,
    }
    schema_receipt = {
        "schema": "blindassist.assistive_geometry_due.sanpo_synthetic_r1_metadata_schema_receipt.v1",
        "execution_id": EXECUTION_ID,
        "terminal": "NOT_EVALUABLE",
        "metadata_access_status": metadata_status,
        "missing_exact_object": failure["observed_missing_metadata_object"],
        "observed_http_status": 404,
        "description_schema_finalized": False,
        "labelmap_schema_finalized": False,
        "annotation_type_schema_finalized": False,
        "pose_header_or_row_count_inspected": False,
        "pose_numeric_values_parsed": False,
        "raw_metadata_bytes_persisted": False,
    }
    result = {
        "schema": "blindassist.assistive_geometry_due.sanpo_synthetic_r1_metadata_preflight_result.v1",
        "execution_id": EXECUTION_ID,
        "decision": "NOT_EVALUABLE",
        "terminal": lock["decision_contract"]["NOT_EVALUABLE"],
        "source": source,
        "failure": {
            "kind": "EXACT_REQUIRED_METADATA_OBJECT_MISSING",
            "object_name": failure["observed_missing_metadata_object"],
            "http_status": 404,
            "fallback_or_path_substitution_attempted": False,
            "request_count_basis": "DERIVED_FROM_HASH_LOCKED_SEQUENTIAL_CONTROL_FLOW",
            "requests_per_attempt": 6,
            "attempt_count": 2,
            "total_actual_network_requests": 12,
            "retry_attempt_receipt_prior_request_count_field_correct": False,
            "retry_attempt_receipt_field_value": 2,
            "corrected_first_attempt_request_count": 6,
        },
        "inventory_counts": {
            "rgb_object_inventory_count": 0,
            "panoptic_object_inventory_count": 0,
            "metric_depth_object_inventory_count": 0,
            "numeric_index_intersection_count": 0,
            "pose_table_row_count": 0,
        },
        "capability_counts": {
            "oracle_depth_factor_frames": 0,
            "oracle_support_factor_frames": 0,
            "boundary_truth_frames": 0,
            "explicit_timestamp_frames": 0,
            "pose_transform_frames": 0,
            "portrait_frames": 0,
            "landscape_frames": 0,
        },
        "selected_lowest_25": [],
        "execution_disclosure": {
            "network_used": True,
            "description_and_labelmap_bodies_read_in_memory": True,
            "raw_metadata_bytes_persisted": False,
            "annotation_type_body_read": False,
            "pose_table_body_read": False,
            "frame_prefix_listing_performed": False,
            "frame_body_requested_or_read": False,
            "frame_body_bytes_read": 0,
            "rgb_visual_access": False,
            "mask_or_depth_decoded": False,
            "local_existing_payload_opened": False,
            "teacher_model_derivation_or_training": False,
            "failure_finalization_network_used": False,
        },
        "authority": {
            "inventory_is_capability_truth": False,
            "source_data_support_established": False,
            "dca_pass": False,
            "r2_f1_parent_gate_pass": False,
            "body_canary_execution_authorized": False,
            "pose_or_timestamp_admitted": False,
            "support_truth_established": False,
            "boundary_truth_materialized": False,
            "derivation_label_materialization_or_training": False,
            "development_confirmation_android_product_safety": False,
        },
        "artifact_receipts": {
            "attempt_receipt_sha256": attempt_sha256,
            "retry_attempt_receipt_sha256": retry_attempt_sha256,
        },
        "unique_successor": STOP_SUCCESSOR,
        "claim_ceiling": lock["claim_ceiling"],
    }
    return inventory, schema_receipt, result


def finalize_observed_failure(lock: dict[str, Any], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    require(output_root.resolve() == OUTPUT_ROOT.resolve(), "output root escape")
    failure = lock["failure_finalization_contract"]
    require(output_root.is_dir(), "failed-attempt evidence root missing")
    existing = sorted(path.name for path in output_root.iterdir())
    require(existing == failure["allowed_existing_files_before_finalization"], "failure-finalization file set drift")
    attempt = load_json(output_root / "attempt_receipt.json")
    retry_attempt = load_json(output_root / "retry_attempt_receipt.json")
    require(attempt.get("lock_sha256") == failure["first_attempt_lock_sha256"], "first attempt lock drift")
    require(retry_attempt.get("lock_sha256") == failure["retry_attempt_lock_sha256"], "retry attempt lock drift")
    require(attempt.get("frame_body_request_budget_bytes") == 0, "first attempt frame-body budget drift")
    require(retry_attempt.get("frame_body_request_budget_bytes") == 0, "retry frame-body budget drift")
    inventory, schema_receipt, result = build_observed_failure_terminal(
        lock,
        attempt_sha256=sha256_file(output_root / "attempt_receipt.json"),
        retry_attempt_sha256=sha256_file(output_root / "retry_attempt_receipt.json"),
    )
    _write_exclusive(output_root / "source_object_inventory.json", inventory)
    _write_exclusive(output_root / "metadata_schema_receipt.json", schema_receipt)
    result["artifact_receipts"]["source_object_inventory_sha256"] = sha256_file(
        output_root / "source_object_inventory.json"
    )
    result["artifact_receipts"]["metadata_schema_receipt_sha256"] = sha256_file(
        output_root / "metadata_schema_receipt.json"
    )
    _write_exclusive(output_root / "preflight_result.json", result)
    return result


def execute(lock: dict[str, Any], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    require(output_root.resolve() == OUTPUT_ROOT.resolve(), "output root escape")
    recovery = lock["recovery_contract"]
    require(output_root.is_dir(), "required failed-attempt evidence root missing")
    existing = sorted(path.name for path in output_root.iterdir())
    require(existing == recovery["allowed_existing_files_before_retry"], "failed-attempt file set drift")
    prior_attempt = load_json(output_root / "attempt_receipt.json")
    require(prior_attempt.get("execution_id") == EXECUTION_ID, "prior attempt identity drift")
    require(
        prior_attempt.get("lock_sha256") == recovery["prior_attempt_lock_sha256"],
        "prior attempt lock SHA drift",
    )
    require(prior_attempt.get("written_before_network") is True, "prior attempt timing receipt drift")
    require(prior_attempt.get("frame_body_request_budget_bytes") == 0, "prior frame-body budget drift")
    retry_attempt = {
        "schema": "blindassist.assistive_geometry_due.sanpo_synthetic_r1_metadata_preflight_retry_attempt.v1",
        "execution_id": EXECUTION_ID,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": sha256_file(LOCK_PATH),
        "prior_attempt_receipt_sha256": sha256_file(output_root / "attempt_receipt.json"),
        "prior_attempt_failure": recovery["prior_attempt_failure"],
        "prior_attempt_request_count": recovery["prior_attempt_request_count"],
        "prior_attempt_frame_body_request_count": 0,
        "session_id": SESSION_ID,
        "camera": CAMERA,
        "lens": LENS,
        "bucket": BUCKET,
        "resource_budget": lock["resource_budget"],
        "frame_body_request_budget_bytes": 0,
        "source_data_support_authority": False,
        "written_before_network": True,
    }
    _write_exclusive(output_root / "retry_attempt_receipt.json", retry_attempt)
    provider = GcsMetadataProvider(lock, initial_request_count=recovery["prior_attempt_request_count"])
    inventory, schema_receipt, result = evaluate_preflight(provider, lock)
    inventory_bytes = canonical_json_bytes(inventory)
    schema_bytes = canonical_json_bytes(schema_receipt)
    require(
        len(inventory_bytes) + len(schema_bytes) <= lock["resource_budget"]["max_receipt_output_bytes"],
        "receipt output byte budget exceeded",
    )
    _write_exclusive(output_root / "source_object_inventory.json", inventory)
    _write_exclusive(output_root / "metadata_schema_receipt.json", schema_receipt)
    result["artifact_receipts"] = {
        "attempt_receipt_sha256": sha256_file(output_root / "attempt_receipt.json"),
        "retry_attempt_receipt_sha256": sha256_file(output_root / "retry_attempt_receipt.json"),
        "source_object_inventory_sha256": sha256_file(output_root / "source_object_inventory.json"),
        "metadata_schema_receipt_sha256": sha256_file(output_root / "metadata_schema_receipt.json"),
    }
    _write_exclusive(output_root / "preflight_result.json", result)
    total_disk = sum(path.stat().st_size for path in output_root.iterdir() if path.is_file())
    require(total_disk <= lock["resource_budget"]["max_local_disk_bytes"], "local disk budget exceeded")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="perform the one exact metadata-only preflight")
    parser.add_argument(
        "--finalize-observed-not-evaluable",
        action="store_true",
        help="write the no-network terminal for the exact observed 404",
    )
    args = parser.parse_args()
    lock = load_json(LOCK_PATH)
    validate_execution_lock(lock)
    require(not (args.execute and args.finalize_observed_not_evaluable), "choose one execution mode")
    if args.finalize_observed_not_evaluable:
        result = finalize_observed_failure(lock)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps({"execution_id": EXECUTION_ID, "status": "VALID_NOT_EXECUTED"}, indent=2))
        return 0
    result = execute(lock)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
