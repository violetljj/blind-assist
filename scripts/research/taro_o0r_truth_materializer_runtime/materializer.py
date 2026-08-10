#!/usr/bin/env python3
"""Fail-closed TARO O0R ARKitScenes HEAD/source/truth materializer mechanics."""

from __future__ import annotations

import binascii
import copy
import concurrent.futures
import gzip
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Sequence as RuntimeSequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


PREFLIGHT_SCHEMA = "blindassist.taro.o0r.truth_only_one_shot_preflight_lock.v1"
AUTHORIZATION_SCHEMA = "blindassist.taro.o0r.data_use_authorization_receipt.v1"
HEAD_RECEIPT_SCHEMA = "blindassist.taro.o0r.asset_header_preflight_receipt.v1"
EXECUTION_LOCK_SCHEMA = "blindassist.taro.o0r.truth_only_execution_lock.v1"
TRUTH_RESULT_SCHEMA = "blindassist.taro.o0r.truth_only_result.v1"
MATERIALIZER_MANIFEST_SCHEMA = "blindassist.taro.o0r.truth_materializer_manifest.v1"
BOUND_SOURCE_FRAME_SCHEMA = "blindassist.taro.o0r.bound_source_frame_envelope.v1"
ARRAY_ARTIFACT_SCHEMA = "blindassist.taro.o0r.content_addressed_array_artifact.v1"
ARRAY_BLOB_REF_SCHEMA = "blindassist.taro.ndarray_blob_ref.v1"
MINIMUM_EXACT_FRAMES_PER_EVAL_PARENT = 8
MINIMUM_STATE_FRAMES_PER_EVAL_PARENT = 6
MINIMUM_EVALUABLE_EVAL_PARENTS = 12
OFFICIAL_HOST = "docs-assets.developer.apple.com"
USER_AGENT = "BlindAssist-TARO-O0R-truth-only/1"
CHUNK_BYTES = 1024 * 1024
ASSET_NAMES = ("upsampling.zip", "lowres_wide_intrinsics.zip", "lowres_wide.traj")
DECODED_ROLES = ("color", "highres_depth", "lowres_depth", "confidence", "intrinsics", "trajectory")
UPSAMPLING_DIRECTORY_TO_ROLE = {
    "wide": "color",
    "highres_depth": "highres_depth",
    "lowres_depth": "lowres_depth",
    "confidence": "confidence",
}
SUPPORTED_ZIP_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_VERIFIED_CONTAINER_CACHE: dict[tuple[str, int, int, int, str], bool] = {}
EXPECTED_AUTHORIZATION_VERBATIM = (
    "我授权 TARO O0R 使用锁定的 24 个 ARKitScenes Training 视频及每个视频的 "
    "upsampling.zip、lowres_wide_intrinsics.zip、lowres_wide.traj，用于 HEAD 预检和 "
    "source/truth-only WILD_LAB 物化与校验"
)
EXPECTED_SUCCESSOR_AMENDMENT_VERBATIM = (
    "不管怎么样都行，赶快推进算法前进，你做了这么久浪费这么多token告诉我失败"
)


class MaterializerError(RuntimeError):
    """Stable fail-closed error for future signed receipts."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise MaterializerError(code, message, **context)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_bound_container(path: Path, receipt: Mapping[str, Any]) -> None:
    require(path.is_file(), "SOURCE_CONTAINER_MISSING", "bound source container is missing", path=str(path))
    stat = path.stat()
    expected_bytes = receipt.get("bytes")
    expected_sha256 = receipt.get("sha256")
    require(stat.st_size == expected_bytes and isinstance(expected_sha256, str), "SOURCE_CONTAINER_SIZE_DRIFT", "source container size differs from its receipt", path=str(path))
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected_sha256)
    if key not in _VERIFIED_CONTAINER_CACHE:
        require(sha256_file(path) == expected_sha256, "SOURCE_CONTAINER_HASH_DRIFT", "source container differs from its download receipt", path=str(path))
        _VERIFIED_CONTAINER_CACHE[key] = True


def crc32_bytes(value: bytes) -> str:
    return f"{binascii.crc32(value) & 0xFFFFFFFF:08X}"


def canonical_json_bytes(value: Any) -> bytes:
    return adapter.canonical_json_bytes(value)


def canonical_sha256(value: Any) -> str:
    return adapter.canonical_sha256(value)


def seal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    require("content_sha256" not in output, "SEAL_FIELD_COLLISION", "record already contains content_sha256")
    output["content_sha256"] = canonical_sha256(output)
    return output


def validate_sealed_record(value: Any, schema: str, code: str) -> dict[str, Any]:
    require(isinstance(value, dict), code, "sealed record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(record.get("schema") == schema and observed == canonical_sha256(record), code, "sealed record hash/schema drift")
    return copy.deepcopy(value)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON_OBJECT_REQUIRED", "JSON document must be an object", path=str(path))
    return value


def _safe_relative_path(value: str) -> Path:
    require(isinstance(value, str) and value and "\\" not in value, "PATH_INVALID", "relative path must use POSIX separators")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and all(part not in ("", ".", "..") for part in pure.parts), "PATH_INVALID", "relative path is unsafe", path=value)
    return Path(*pure.parts)


def safe_join(root: Path, relative: str) -> Path:
    root_lexical = root.absolute()
    relative_path = _safe_relative_path(relative)
    output = root_lexical / relative_path
    root_resolved = root_lexical.resolve()
    parent_resolved = output.parent.resolve(strict=False)
    if parent_resolved == root_resolved or root_resolved in parent_resolved.parents:
        return output
    # The repository deliberately exposes artifacts.local as a trusted, F-backed
    # junction. Preserve its lexical repository path for receipts while proving
    # every nested parent still resolves inside that one explicit namespace.
    require(
        relative_path.parts and relative_path.parts[0] == "artifacts.local",
        "PATH_ESCAPE",
        "path escapes bound root",
        path=relative,
    )
    artifacts_lexical = root_lexical / "artifacts.local"
    require(artifacts_lexical.is_dir(), "ARTIFACTS_NAMESPACE_MISSING", "artifacts.local namespace is unavailable")
    artifacts_resolved = artifacts_lexical.resolve()
    require(
        parent_resolved == artifacts_resolved or artifacts_resolved in parent_resolved.parents,
        "PATH_ESCAPE",
        "path escapes artifacts.local namespace",
        path=relative,
    )
    return output


def expanded_asset_plan(preflight: Mapping[str, Any]) -> list[dict[str, str]]:
    require(preflight.get("schema") == PREFLIGHT_SCHEMA, "PREFLIGHT_SCHEMA_DRIFT", "preflight schema drift")
    require(preflight.get("passed") is True and preflight.get("head_requests_executed") is False, "PREFLIGHT_STATE_INVALID", "preflight lock is not the frozen HEAD_NOT_RUN lock")
    plan = preflight.get("asset_plan")
    require(isinstance(plan, dict), "ASSET_PLAN_MISSING", "asset plan missing")
    require(plan.get("request_method") == "HEAD" and plan.get("response_body_bytes_allowed") == 0, "HEAD_PLAN_INVALID", "asset plan must be HEAD-only with zero body bytes")
    require(plan.get("off_host_redirect_allowed") is False, "REDIRECT_POLICY_DRIFT", "off-host redirect must remain forbidden")
    parents = plan.get("selected_parents")
    templates = plan.get("asset_templates")
    require(isinstance(parents, list) and len(parents) == 24, "ROSTER_COUNT_DRIFT", "exactly 24 parents are required")
    require(isinstance(templates, list) and len(templates) == 3, "ASSET_TEMPLATE_DRIFT", "exactly three asset templates are required")
    rows: list[dict[str, str]] = []
    identities: set[tuple[str, str, str]] = set()
    for parent in parents:
        require(isinstance(parent, dict), "ROSTER_ROW_INVALID", "parent row must be an object")
        role = str(parent.get("role"))
        visit_id = str(parent.get("visit_id"))
        video_id = str(parent.get("video_id"))
        fold = str(parent.get("official_fold"))
        require(role in adapter.FROZEN_ROSTER, "ROLE_INVALID", "unknown source role", role=role)
        require((visit_id, video_id) in adapter.FROZEN_ROSTER[role], "ROSTER_IDENTITY_DRIFT", "parent is outside frozen adapter roster", role=role, visit_id=visit_id, video_id=video_id)
        require(fold == "Training", "FOLD_DRIFT", "only official Training is allowed")
        identity = (role, visit_id, video_id)
        require(identity not in identities, "ROSTER_DUPLICATE", "duplicate parent identity")
        identities.add(identity)
        fields = {"official_fold": fold, "video_id": video_id}
        for template in templates:
            asset = str(template.get("asset"))
            require(asset in ASSET_NAMES, "ASSET_NAME_INVALID", "unexpected source asset", asset=asset)
            relative = str(template.get("relative_path_template", "")).format(**fields)
            url = str(template.get("url_template", "")).format(**fields)
            parsed = urllib.parse.urlparse(url)
            require(parsed.scheme == "https" and parsed.hostname == OFFICIAL_HOST and not parsed.query and not parsed.fragment, "URL_NOT_ALLOWED", "asset URL is outside exact official HTTPS host", url=url)
            rows.append(
                {
                    "role": role,
                    "visit_id": visit_id,
                    "video_id": video_id,
                    "official_fold": fold,
                    "asset": asset,
                    "relative_path": relative,
                    "url": url,
                }
            )
    require(len(rows) == 72 and len({row["url"] for row in rows}) == 72, "ASSET_PLAN_CARDINALITY", "asset plan must expand to 72 unique URLs")
    require(canonical_sha256(rows) == plan.get("expanded_requests_sha256"), "ASSET_PLAN_DIGEST_DRIFT", "expanded asset plan digest drift")
    return rows


def validate_authorization(
    preflight: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    preflight_sha256: str,
) -> list[dict[str, str]]:
    rows = expanded_asset_plan(preflight)
    require(authorization.get("schema") == AUTHORIZATION_SCHEMA, "AUTHORIZATION_SCHEMA_DRIFT", "authorization schema drift")
    require(authorization.get("confirmed_by") == "user", "AUTHORIZATION_SIGNER_INVALID", "authorization must be user-confirmed")
    require(authorization.get("confirmation_verbatim") == EXPECTED_AUTHORIZATION_VERBATIM, "AUTHORIZATION_VERBATIM_DRIFT", "authorization verbatim drift")
    binding = authorization.get("scope_binding")
    require(isinstance(binding, dict), "AUTHORIZATION_BINDING_MISSING", "authorization binding missing")
    require(binding.get("preflight_lock_sha256") == preflight_sha256, "AUTHORIZATION_PREFLIGHT_MISMATCH", "authorization is not bound to this preflight lock")
    require(binding.get("request_plan_sha256") == preflight["asset_plan"]["expanded_requests_sha256"], "AUTHORIZATION_PLAN_MISMATCH", "authorization request-plan digest drift")
    scope = authorization.get("interpreted_scope")
    require(isinstance(scope, dict), "AUTHORIZATION_SCOPE_MISSING", "authorization scope missing")
    expected_ids = [row["video_id"] for row in preflight["asset_plan"]["selected_parents"]]
    require(
        authorization.get("scope_amendment_verbatim") == EXPECTED_SUCCESSOR_AMENDMENT_VERBATIM,
        "AUTHORIZATION_AMENDMENT_DRIFT",
        "availability-successor authorization amendment drift",
    )
    require(scope.get("research_route") == "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R1", "AUTHORIZATION_ROUTE_DRIFT", "authorization route drift")
    require(scope.get("official_fold") == "Training", "AUTHORIZATION_FOLD_DRIFT", "authorization fold drift")
    require(scope.get("selected_parent_count") == 24 and scope.get("selected_video_ids") == expected_ids, "AUTHORIZATION_ROSTER_DRIFT", "authorization roster is not the exact 24-parent order")
    expected_patterns = [template["url_template"].replace("{official_fold}", "Training") for template in preflight["asset_plan"]["asset_templates"]]
    require(scope.get("authorized_asset_patterns") == expected_patterns, "AUTHORIZATION_ASSET_DRIFT", "authorization asset patterns drift")
    require(scope.get("authorization_does_not_itself_activate_execution") is True, "AUTHORIZATION_OVERCLAIM", "authorization must not self-activate execution")
    require(scope.get("separate_implementation_and_execution_locks_required") is True, "AUTHORIZATION_OVERCLAIM", "separate locks must remain required")
    require(scope.get("availability_replacement_authorized") is True, "AUTHORIZATION_REPLACEMENT_MISSING", "availability successor is not authorized")
    require(scope.get("retired_unavailable_video_id") == "47333152", "AUTHORIZATION_RETIRED_PARENT_DRIFT", "retired unavailable parent drift")
    require(scope.get("replacement_video_id") == "47204786", "AUTHORIZATION_REPLACEMENT_DRIFT", "replacement parent drift")
    return rows


def validate_head_receipt(
    preflight: Mapping[str, Any],
    authorization_sha256: str,
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    plan = expanded_asset_plan(preflight)
    require(receipt.get("schema") == HEAD_RECEIPT_SCHEMA, "HEAD_RECEIPT_SCHEMA_DRIFT", "HEAD receipt schema drift")
    require(receipt.get("request_method") == "HEAD", "HEAD_METHOD_DRIFT", "HEAD receipt method drift")
    require(receipt.get("response_body_bytes_read") == 0 and receipt.get("media_body_bytes_read") is False, "HEAD_BODY_READ", "HEAD receipt must prove zero body bytes")
    require(receipt.get("authorization_receipt_sha256") == authorization_sha256, "HEAD_AUTHORIZATION_MISMATCH", "HEAD receipt authorization binding drift")
    assets = receipt.get("assets")
    require(isinstance(assets, list) and len(assets) == 72, "HEAD_RECEIPT_CARDINALITY", "HEAD receipt must contain 72 assets")
    lookup: dict[str, dict[str, Any]] = {}
    total = 0
    for expected, observed in zip(plan, assets, strict=True):
        require(isinstance(observed, dict), "HEAD_ROW_INVALID", "HEAD asset row must be an object")
        for key, value in expected.items():
            require(observed.get(key) == value, "HEAD_IDENTITY_DRIFT", "HEAD asset identity/order drift", key=key, url=expected["url"])
        require(observed.get("http_status") == 200, "ASSET_HEAD_UNAVAILABLE", "HEAD did not return 200", url=expected["url"])
        length = observed.get("content_length_bytes")
        require(isinstance(length, int) and not isinstance(length, bool) and length > 0, "CONTENT_LENGTH_MISSING", "positive Content-Length required", url=expected["url"])
        require(observed.get("redirect_chain") == [], "REDIRECT_FORBIDDEN", "redirect chain must be empty", url=expected["url"])
        require(observed.get("transport_errors") == [], "HEAD_TRANSPORT_ERROR", "HEAD transport error present", url=expected["url"])
        attempt_count = observed.get("attempt_count")
        attempts = observed.get("attempts")
        require(isinstance(attempt_count, int) and 1 <= attempt_count <= int(preflight["resource_budget"]["head_retries"]), "HEAD_ATTEMPT_RECEIPT_INVALID", "HEAD attempt count is invalid", url=expected["url"])
        require(isinstance(attempts, list) and len(attempts) == attempt_count, "HEAD_ATTEMPT_RECEIPT_INVALID", "HEAD attempt receipt cardinality drift", url=expected["url"])
        total += length
        lookup[expected["url"]] = dict(observed)
    maximum = int(preflight["resource_budget"]["maximum_compressed_source_bytes"])
    require(total <= maximum, "CONTENT_LENGTH_OVER_BUDGET", "compressed source exceeds frozen budget", total=total, maximum=maximum)
    require(receipt.get("asset_count") == 72 and receipt.get("available_asset_count") == 72, "HEAD_RECEIPT_SUMMARY_DRIFT", "HEAD receipt count summary drift")
    require(receipt.get("total_content_length_bytes") == total, "HEAD_RECEIPT_SUMMARY_DRIFT", "HEAD receipt byte total drift")
    require(receipt.get("terminal") == "TARO_O0R_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED", "HEAD_RECEIPT_TERMINAL", "HEAD receipt terminal is not availability PASS")
    return lookup


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise MaterializerError("REDIRECT_FORBIDDEN", "redirect is forbidden", status=code, url=newurl)


def production_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_RejectRedirect())


def production_head(row: Mapping[str, str], timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(row["url"], method="HEAD", headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    try:
        with production_opener().open(request, timeout=timeout_seconds) as response:
            final_url = response.geturl()
            require(final_url == row["url"], "REDIRECT_FORBIDDEN", "final HEAD URL drift", final_url=final_url)
            content_length = response.headers.get("Content-Length")
            require(response.headers.get("Content-Encoding") in (None, "identity"), "CONTENT_ENCODING_FORBIDDEN", "encoded HEAD response forbidden")
            return {
                "http_status": int(response.status),
                "content_length_bytes": int(content_length) if content_length is not None else None,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "redirect_chain": [],
                "transport_errors": [],
            }
    except MaterializerError:
        raise
    except Exception as error:
        raise MaterializerError("HEAD_TRANSPORT_ERROR", f"HEAD failed: {type(error).__name__}: {error}", url=row["url"]) from error


def build_head_receipt(
    preflight: Mapping[str, Any],
    *,
    preflight_sha256: str,
    authorization_sha256: str,
    head_fn: Callable[[Mapping[str, str], float], Mapping[str, Any]],
) -> dict[str, Any]:
    plan = expanded_asset_plan(preflight)
    budget = preflight["resource_budget"]
    timeout = float(budget["head_timeout_seconds"])
    workers = int(budget["head_workers"])
    maximum_attempts = int(budget["head_retries"])
    require(maximum_attempts >= 1, "HEAD_RETRY_POLICY_INVALID", "HEAD attempts must be positive")

    def invoke(row: dict[str, str]) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        observed: dict[str, Any] | None = None
        for attempt in range(1, maximum_attempts + 1):
            try:
                current = dict(head_fn(row, timeout))
            except MaterializerError as error:
                current = {
                    "http_status": None,
                    "content_length_bytes": None,
                    "etag": None,
                    "last_modified": None,
                    "redirect_chain": [],
                    "transport_errors": [f"{error.code}:{error}"],
                }
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": current.get("http_status"),
                    "content_length_bytes": current.get("content_length_bytes"),
                    "redirect_chain": list(current.get("redirect_chain") or []),
                    "transport_errors": list(current.get("transport_errors") or []),
                }
            )
            observed = current
            if (
                current.get("http_status") == 200
                and isinstance(current.get("content_length_bytes"), int)
                and current.get("content_length_bytes", 0) > 0
                and current.get("redirect_chain") == []
                and current.get("transport_errors") == []
            ):
                break
        require(observed is not None, "HEAD_RETRY_POLICY_INVALID", "HEAD attempt loop produced no observation")
        return dict(row) | observed | {"attempt_count": len(attempts), "attempts": attempts}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        assets = list(executor.map(invoke, plan))
    available = sum(row.get("http_status") == 200 and isinstance(row.get("content_length_bytes"), int) and row.get("content_length_bytes", 0) > 0 and row.get("redirect_chain") == [] and row.get("transport_errors") == [] for row in assets)
    total = sum(int(row.get("content_length_bytes") or 0) for row in assets)
    terminal = "TARO_O0R_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED" if available == 72 and total <= int(budget["maximum_compressed_source_bytes"]) else "TARO_O0R_ASSET_HEADERS_NOT_AVAILABLE_NO_REPLACEMENT"
    return {
        "schema": HEAD_RECEIPT_SCHEMA,
        "preflight_lock_sha256": preflight_sha256,
        "authorization_receipt_sha256": authorization_sha256,
        "request_method": "HEAD",
        "response_body_bytes_read": 0,
        "media_body_bytes_read": False,
        "assets": assets,
        "asset_count": len(assets),
        "available_asset_count": available,
        "total_content_length_bytes": total,
        "replacement_allowed": False,
        "terminal": terminal,
    }


def download_bound_asset(
    row: Mapping[str, Any],
    head_row: Mapping[str, Any],
    *,
    source_root: Path,
    opener: urllib.request.OpenerDirector | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    require(row["url"] == head_row.get("url"), "DOWNLOAD_HEAD_IDENTITY", "download/HEAD URL mismatch")
    expected = head_row.get("content_length_bytes")
    require(head_row.get("http_status") == 200 and isinstance(expected, int) and expected > 0, "DOWNLOAD_HEAD_NOT_ADMITTED", "download requires successful bound HEAD")
    destination = safe_join(source_root, str(row["relative_path"]))
    require(not destination.exists(), "DOWNLOAD_OVERWRITE_FORBIDDEN", "download destination already exists", path=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    require(not partial.exists(), "DOWNLOAD_PARTIAL_COLLISION", "partial download already exists", path=str(partial))
    request = urllib.request.Request(str(row["url"]), method="GET", headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    selected_opener = opener or production_opener()
    digest = hashlib.sha256()
    crc32 = 0
    size = 0
    try:
        with selected_opener.open(request, timeout=timeout_seconds) as response:
            require(int(response.status) == 200 and response.geturl() == row["url"], "DOWNLOAD_RESPONSE_INVALID", "GET status/final URL drift")
            header_length = response.headers.get("Content-Length")
            require(header_length is not None and int(header_length) == expected, "DOWNLOAD_CONTENT_LENGTH_DRIFT", "GET Content-Length differs from HEAD")
            require(response.headers.get("Content-Encoding") in (None, "identity"), "CONTENT_ENCODING_FORBIDDEN", "encoded GET response forbidden")
            require(response.headers.get("Content-Range") is None, "RANGE_RESPONSE_FORBIDDEN", "partial/range response is forbidden")
            for header_name, head_key in (("ETag", "etag"), ("Last-Modified", "last_modified")):
                bound_value = head_row.get(head_key)
                if bound_value is not None:
                    require(response.headers.get(header_name) == bound_value, "DOWNLOAD_VALIDATOR_DRIFT", "GET validator differs from HEAD", header=header_name)
            with partial.open("xb") as stream:
                while True:
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    size += len(block)
                    require(size <= expected, "DOWNLOAD_BODY_OVERRUN", "GET body exceeds bound Content-Length")
                    digest.update(block)
                    crc32 = binascii.crc32(block, crc32)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
        require(size == expected, "DOWNLOAD_BODY_TRUNCATED", "GET body length differs from bound Content-Length", expected=expected, actual=size)
        os.replace(partial, destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return {
        "asset": row["asset"],
        "url": row["url"],
        "relative_path": row["relative_path"],
        "bytes": size,
        "sha256": digest.hexdigest().upper(),
        "crc32": f"{crc32 & 0xFFFFFFFF:08X}",
        "head_content_length_bytes": expected,
        "head_etag": head_row.get("etag"),
        "head_last_modified": head_row.get("last_modified"),
        "redirect_chain": [],
        "attempt_count": 1,
    }


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    name = info.filename
    require("\\" not in name, "ZIP_MEMBER_PATH_INVALID", "ZIP member uses backslash", member=name)
    pure = PurePosixPath(name)
    require(not pure.is_absolute() and all(part not in ("", ".", "..") for part in pure.parts), "ZIP_MEMBER_PATH_INVALID", "unsafe ZIP member path", member=name)
    require(not (info.external_attr >> 16) & 0o170000 == 0o120000, "ZIP_SYMLINK_FORBIDDEN", "ZIP symlink forbidden", member=name)
    require(info.flag_bits & 0x1 == 0, "ZIP_ENCRYPTION_FORBIDDEN", "encrypted ZIP member forbidden", member=name)
    require(info.compress_type in SUPPORTED_ZIP_COMPRESSION, "ZIP_COMPRESSION_UNSUPPORTED", "ZIP compression method is not frozen", member=name, compress_type=info.compress_type)
    require(info.file_size >= 0 and info.compress_size >= 0, "ZIP_MEMBER_SIZE_INVALID", "ZIP member size is invalid", member=name)


def _validate_zip_inventory(bundle: zipfile.ZipFile, maximum_uncompressed_bytes: int | None) -> int:
    seen_names: set[str] = set()
    total = 0
    for info in bundle.infolist():
        _validate_zip_info(info)
        require(info.filename not in seen_names, "ZIP_MEMBER_DUPLICATE", "duplicate ZIP member path", member=info.filename)
        seen_names.add(info.filename)
        total += int(info.file_size)
        if maximum_uncompressed_bytes is not None:
            require(total <= maximum_uncompressed_bytes, "ZIP_MATERIALIZED_BUDGET_EXCEEDED", "ZIP uncompressed bytes exceed the frozen materialized-source budget", total=total, maximum=maximum_uncompressed_bytes)
    return total


def zip_uncompressed_bytes(path: Path, maximum_uncompressed_bytes: int | None = None) -> int:
    try:
        with zipfile.ZipFile(path) as bundle:
            return _validate_zip_inventory(bundle, maximum_uncompressed_bytes)
    except zipfile.BadZipFile as error:
        raise MaterializerError("ZIP_CONTAINER_INVALID", "source archive is not a valid ZIP", path=str(path)) from error


@dataclass(frozen=True)
class MemberBinding:
    role: str
    timestamp_token: str
    source_member_path: str
    canonical_member_path: str
    bytes: int
    crc32: str


def _timestamp_token_from_member(video_id: str, name: str, suffix: str) -> str:
    pure = PurePosixPath(name)
    require(pure.suffix.lower() == suffix, "MEMBER_SUFFIX_INVALID", "member suffix drift", member=name)
    prefix = f"{video_id}_"
    require(pure.stem.startswith(prefix), "MEMBER_VIDEO_PREFIX_MISMATCH", "member is not bound to video identity", member=name, video_id=video_id)
    token = pure.stem[len(prefix) :]
    adapter.decimal_timestamp_ns(token)
    return token


def index_upsampling_archive(
    path: Path,
    video_id: str,
    *,
    maximum_uncompressed_bytes: int | None = None,
) -> dict[str, dict[str, MemberBinding]]:
    result: dict[str, dict[str, MemberBinding]] = {role: {} for role in UPSAMPLING_DIRECTORY_TO_ROLE.values()}
    with zipfile.ZipFile(path) as bundle:
        _validate_zip_inventory(bundle, maximum_uncompressed_bytes)
        bad = bundle.testzip()
        require(bad is None, "ZIP_CRC_FAILURE", "ZIP CRC failure", member=bad, path=str(path))
        for info in bundle.infolist():
            if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".png":
                continue
            pure = PurePosixPath(info.filename)
            if len(pure.parts) < 2 or pure.parts[-2] not in UPSAMPLING_DIRECTORY_TO_ROLE:
                continue
            role = UPSAMPLING_DIRECTORY_TO_ROLE[pure.parts[-2]]
            token = _timestamp_token_from_member(video_id, info.filename, ".png")
            require(token not in result[role], "FRAME_MEMBER_DUPLICATE", "duplicate modality timestamp", role=role, token=token)
            require(info.file_size > 0, "ZIP_MEMBER_SIZE_INVALID", "bound source member must be non-empty", member=info.filename)
            result[role][token] = MemberBinding(
                role=role,
                timestamp_token=token,
                source_member_path=info.filename,
                canonical_member_path=f"{role}/{token}.png",
                bytes=int(info.file_size),
                crc32=f"{info.CRC:08X}",
            )
    require(all(result.values()), "UPSAMPLING_MODALITY_MISSING", "one or more upsampling modalities are empty")
    return result


def index_intrinsics_archive(
    path: Path,
    video_id: str,
    *,
    maximum_uncompressed_bytes: int | None = None,
) -> dict[str, MemberBinding]:
    result: dict[str, MemberBinding] = {}
    with zipfile.ZipFile(path) as bundle:
        _validate_zip_inventory(bundle, maximum_uncompressed_bytes)
        bad = bundle.testzip()
        require(bad is None, "ZIP_CRC_FAILURE", "intrinsics ZIP CRC failure", member=bad, path=str(path))
        for info in bundle.infolist():
            if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".pincam":
                continue
            token = _timestamp_token_from_member(video_id, info.filename, ".pincam")
            require(token not in result, "INTRINSICS_MEMBER_DUPLICATE", "duplicate intrinsics timestamp", token=token)
            require(info.file_size > 0, "ZIP_MEMBER_SIZE_INVALID", "bound intrinsics member must be non-empty", member=info.filename)
            result[token] = MemberBinding(
                role="intrinsics",
                timestamp_token=token,
                source_member_path=info.filename,
                canonical_member_path=f"intrinsics/{token}.pincam",
                bytes=int(info.file_size),
                crc32=f"{info.CRC:08X}",
            )
    require(bool(result), "INTRINSICS_MODALITY_MISSING", "intrinsics archive has no pincam members")
    return result


def parse_trajectory_payload(payload: bytes) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MaterializerError("TRAJECTORY_UTF8_INVALID", "trajectory is not UTF-8") from error
    rows: list[dict[str, Any]] = []
    previous_ns: int | None = None
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split()
        require(len(fields) == 7, "TRAJECTORY_ROW_INVALID", "trajectory row must have seven fields", line=line_number)
        token = fields[0]
        timestamp_ns = adapter.decimal_timestamp_ns(token)
        require(previous_ns is None or timestamp_ns > previous_ns, "TRAJECTORY_ORDER_INVALID", "trajectory timestamps must increase", line=line_number)
        previous_ns = timestamp_ns
        values = [float(value) for value in fields[1:]]
        require(all(math.isfinite(value) for value in values), "TRAJECTORY_NONFINITE", "trajectory values must be finite", line=line_number)
        rows.append(
            {
                "timestamp_token": token,
                "rotation_vector": values[:3],
                "translation": values[3:],
            }
        )
    require(len(rows) >= 2, "TRAJECTORY_TOO_SHORT", "trajectory requires at least two rows")
    return rows


def parse_pincam_payload(payload: bytes) -> dict[str, Any]:
    try:
        fields = payload.decode("utf-8").split()
    except UnicodeDecodeError as error:
        raise MaterializerError("INTRINSICS_UTF8_INVALID", "pincam is not UTF-8") from error
    require(len(fields) == 6, "INTRINSICS_ROW_INVALID", "pincam must contain six fields")
    values = [float(value) for value in fields]
    require(all(math.isfinite(value) for value in values), "INTRINSICS_NONFINITE", "pincam values must be finite")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height, "INTRINSICS_DIMENSION_INVALID", "pincam dimensions must be integers")
    result = {"width": width, "height": height, "fx": values[2], "fy": values[3], "cx": values[4], "cy": values[5]}
    adapter.scale_lowres_intrinsics(result)
    return result


def exact_frame_plan(
    video_id: str,
    upsampling: Mapping[str, Mapping[str, MemberBinding]],
    intrinsics: Mapping[str, MemberBinding],
    trajectory_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    nanosecond_tokens: dict[int, str] = {}
    for token in set().union(*(set(upsampling[role]) for role in ("color", "highres_depth", "lowres_depth", "confidence")), set(intrinsics)):
        timestamp_ns = adapter.decimal_timestamp_ns(token)
        prior = nanosecond_tokens.get(timestamp_ns)
        require(prior is None or prior == token, "SAME_NS_TIMESTAMP_ALIAS", "distinct exact timestamp tokens map to the same nanosecond", first=prior, second=token, timestamp_ns=timestamp_ns)
        nanosecond_tokens[timestamp_ns] = token
    role_sets = [set(upsampling[role]) for role in ("color", "highres_depth", "lowres_depth", "confidence")]
    common_source = set.intersection(*role_sets)
    exact_common = common_source & set(intrinsics)
    ordered = sorted(exact_common, key=lambda token: (adapter.decimal_timestamp_ns(token), token))
    admitted: list[str] = []
    rejected: list[dict[str, str]] = []
    for token in ordered:
        try:
            adapter.interpolate_camera_to_world_exact(trajectory_rows, token)
        except adapter.AdapterError as error:
            rejected.append({"timestamp_token": token, "reason_code": error.code})
        else:
            admitted.append(token)
    return {
        "video_id": video_id,
        "source_common_frame_count": len(common_source),
        "exact_intrinsics_common_frame_count": len(exact_common),
        "exact_pose_bounded_frame_count": len(admitted),
        "exact_timestamp_tokens": admitted,
        "pose_rejected": rejected,
        "selection_rule": "ALL_SAME_STEM_MODALITIES_EXACT_INTRINSICS_WITH_BOUNDED_POSE_SORTED_BY_EXACT_NS_THEN_TOKEN",
        "truth_or_model_fields_read": False,
    }


def _read_zip_member(bundle: zipfile.ZipFile, binding: MemberBinding) -> bytes:
    info = bundle.getinfo(binding.source_member_path)
    _validate_zip_info(info)
    payload = bundle.read(info)
    require(len(payload) == binding.bytes, "ZIP_MEMBER_LENGTH_DRIFT", "ZIP member length drift", member=info.filename)
    require(crc32_bytes(payload) == binding.crc32, "ZIP_MEMBER_CRC_DRIFT", "ZIP member CRC drift", member=info.filename)
    return payload


def _decode_png(payload: bytes, role: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if role == "color":
                value = np.asarray(image.convert("RGB")).copy()
            else:
                value = np.asarray(image).copy()
    except Exception as error:
        raise MaterializerError("PNG_DECODE_FAILED", f"PNG decode failed: {error}", role=role) from error
    if role == "color":
        require(value.shape == (1440, 1920, 3) and value.dtype == np.uint8, "COLOR_DECODE_INVALID", "color must decode to uint8 1440x1920x3")
    elif role == "highres_depth":
        require(value.shape == adapter.HIGHRES_SHAPE_HW and value.dtype == np.uint16, "FARO_DECODE_INVALID", "FARO depth must decode to uint16 1440x1920")
    elif role == "lowres_depth":
        require(value.shape == adapter.APPLE_SHAPE_HW and value.dtype == np.uint16, "APPLE_DECODE_INVALID", "AppleDepth must decode to uint16 192x256")
    elif role == "confidence":
        require(value.shape == adapter.APPLE_SHAPE_HW and value.dtype == np.uint8 and bool(np.all(value <= 2)), "CONFIDENCE_DECODE_INVALID", "confidence must decode to uint8 0..2 at 192x256")
    return np.ascontiguousarray(value)


def decode_source_frame(
    *,
    parent: Mapping[str, str],
    timestamp_token: str,
    upsampling_archive: Path,
    intrinsics_archive: Path,
    trajectory_path: Path,
    upsampling_inventory: Mapping[str, Mapping[str, MemberBinding]],
    intrinsics_inventory: Mapping[str, MemberBinding],
    trajectory_rows: Sequence[dict[str, Any]],
    container_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    verify_bound_container(upsampling_archive, container_receipts["upsampling.zip"])
    verify_bound_container(intrinsics_archive, container_receipts["lowres_wide_intrinsics.zip"])
    role_payloads: dict[str, bytes] = {}
    original_provenance: dict[str, Any] = {}
    with zipfile.ZipFile(upsampling_archive) as up_bundle, zipfile.ZipFile(intrinsics_archive) as intr_bundle:
        for role in ("color", "highres_depth", "lowres_depth", "confidence"):
            binding = upsampling_inventory[role][timestamp_token]
            payload = _read_zip_member(up_bundle, binding)
            role_payloads[role] = payload
            original_provenance[role] = {
                "container_sha256": container_receipts["upsampling.zip"]["sha256"],
                "source_member_path": binding.source_member_path,
                "canonical_member_path": binding.canonical_member_path,
                "member_bytes": binding.bytes,
                "member_sha256": sha256_bytes(payload),
                "member_crc32": binding.crc32,
            }
        intr_binding = intrinsics_inventory[timestamp_token]
        intr_payload = _read_zip_member(intr_bundle, intr_binding)
        role_payloads["intrinsics"] = intr_payload
        original_provenance["intrinsics"] = {
            "container_sha256": container_receipts["lowres_wide_intrinsics.zip"]["sha256"],
            "source_member_path": intr_binding.source_member_path,
            "canonical_member_path": intr_binding.canonical_member_path,
            "member_bytes": intr_binding.bytes,
            "member_sha256": sha256_bytes(intr_payload),
            "member_crc32": intr_binding.crc32,
        }
    verify_bound_container(trajectory_path, container_receipts["lowres_wide.traj"])
    trajectory_payload = trajectory_path.read_bytes()
    original_provenance["trajectory"] = {
        "container_sha256": container_receipts["lowres_wide.traj"]["sha256"],
        "source_member_path": "lowres_wide.traj",
        "canonical_member_path": "trajectory/lowres_wide.traj",
        "member_bytes": len(trajectory_payload),
        "member_sha256": sha256_bytes(trajectory_payload),
        "member_crc32": crc32_bytes(trajectory_payload),
    }

    decoded = {
        "color": _decode_png(role_payloads["color"], "color"),
        "highres_depth": _decode_png(role_payloads["highres_depth"], "highres_depth"),
        "lowres_depth": _decode_png(role_payloads["lowres_depth"], "lowres_depth"),
        "confidence": _decode_png(role_payloads["confidence"], "confidence"),
        "intrinsics": parse_pincam_payload(role_payloads["intrinsics"]),
        "trajectory": list(trajectory_rows),
    }
    asset_bindings: dict[str, dict[str, Any]] = {}
    for role in DECODED_ROLES:
        provenance = original_provenance[role]
        canonical_path = provenance["canonical_member_path"]
        asset_bindings[role] = {
            "container_id": f"sha256:{provenance['container_sha256']}",
            "member_path": canonical_path,
            "exact_timestamp_stem": None if role == "trajectory" else timestamp_token,
            "bytes": int(provenance["member_bytes"]),
            "sha256": provenance["member_sha256"],
            "crc32": provenance["member_crc32"],
        }
    decoded_bindings = {
        role: {
            "asset_role": role,
            "member_sha256": asset_bindings[role]["sha256"],
            "member_crc32": asset_bindings[role]["crc32"],
            "decoded_kind": adapter.DECODED_PAYLOAD_KINDS[role],
            "decoded_content_sha256": canonical_sha256(decoded[role]),
        }
        for role in DECODED_ROLES
    }
    receipt = adapter.build_source_frame_receipt(
        source_role=str(parent["role"]),
        visit_id=str(parent["visit_id"]),
        video_id=str(parent["video_id"]),
        frame_timestamp_token=timestamp_token,
        lowres_intrinsics=decoded["intrinsics"],
        trajectory_rows=decoded["trajectory"],
        asset_bindings=asset_bindings,
        decoded_payload_bindings=decoded_bindings,
    )
    envelope_members = {
        role: {
            "source_container_sha256": original_provenance[role]["container_sha256"],
            "source_member_path": original_provenance[role]["source_member_path"],
            "source_member_bytes": int(original_provenance[role]["member_bytes"]),
            "source_member_sha256": original_provenance[role]["member_sha256"],
            "source_member_crc32": original_provenance[role]["member_crc32"],
            "canonical_member_path": original_provenance[role]["canonical_member_path"],
            "decoded_content_sha256": decoded_bindings[role]["decoded_content_sha256"],
        }
        for role in DECODED_ROLES
    }
    envelope = seal_record(
        {
            "schema": BOUND_SOURCE_FRAME_SCHEMA,
            "source_role": receipt["source_role"],
            "visit_id": receipt["site_id"],
            "video_id": receipt["session_id"],
            "timestamp_token": receipt["sensor_timestamp"]["decimal_token"],
            "physical_frame_id": receipt["physical_frame_id"],
            "source_frame_receipt_sha256": receipt["content_sha256"],
            "members": envelope_members,
            "mapping_rule": "OFFICIAL_VIDEO_ID_TIMESTAMP_TO_EXPLICIT_TIMESTAMP_ONLY_CANONICAL_MEMBER",
            "silent_rename_allowed": False,
        }
    )
    validate_bound_source_frame_envelope(envelope, receipt)
    return {
        "source_frame_receipt": receipt,
        "highres_faro_depth_mm": decoded["highres_depth"],
        "apple_depth_mm": decoded["lowres_depth"],
        "confidence": decoded["confidence"],
        "color_array_sha256": canonical_sha256(decoded["color"]),
        "bound_source_frame_envelope": envelope,
    }


def validate_bound_source_frame_envelope(
    value: Any,
    source_frame_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = validate_sealed_record(value, BOUND_SOURCE_FRAME_SCHEMA, "SOURCE_ENVELOPE_HASH_MISMATCH")
    receipt = dict(source_frame_receipt)
    require(
        envelope.get("source_role") == receipt.get("source_role")
        and envelope.get("visit_id") == receipt.get("site_id")
        and envelope.get("video_id") == receipt.get("session_id")
        and envelope.get("timestamp_token") == receipt.get("sensor_timestamp", {}).get("decimal_token")
        and envelope.get("physical_frame_id") == receipt.get("physical_frame_id")
        and envelope.get("source_frame_receipt_sha256") == receipt.get("content_sha256"),
        "SOURCE_ENVELOPE_RECEIPT_MISMATCH",
        "source envelope identity is not bound to the adapter receipt",
    )
    require(envelope.get("mapping_rule") == "OFFICIAL_VIDEO_ID_TIMESTAMP_TO_EXPLICIT_TIMESTAMP_ONLY_CANONICAL_MEMBER" and envelope.get("silent_rename_allowed") is False, "SOURCE_ENVELOPE_MAPPING_DRIFT", "source/canonical member mapping policy drift")
    members = envelope.get("members")
    require(isinstance(members, dict) and set(members) == set(DECODED_ROLES), "SOURCE_ENVELOPE_ROLE_SET", "source envelope role set drift")
    required = {
        "source_container_sha256",
        "source_member_path",
        "source_member_bytes",
        "source_member_sha256",
        "source_member_crc32",
        "canonical_member_path",
        "decoded_content_sha256",
    }
    source_paths: set[str] = set()
    canonical_paths: set[str] = set()
    for role in DECODED_ROLES:
        member = members[role]
        require(isinstance(member, dict) and set(member) == required, "SOURCE_ENVELOPE_MEMBER_FIELDS", "source envelope member fields drift", role=role)
        source_path = str(member["source_member_path"])
        canonical_path = str(member["canonical_member_path"])
        _safe_relative_path(source_path)
        _safe_relative_path(canonical_path)
        require(source_path not in source_paths and canonical_path not in canonical_paths, "SOURCE_ENVELOPE_MEMBER_DUPLICATE", "source/canonical member path duplicated", role=role)
        source_paths.add(source_path)
        canonical_paths.add(canonical_path)
        asset = receipt["asset_bindings"][role]
        decoded = receipt["decoded_payload_bindings"][role]
        require(
            asset["container_id"] == f"sha256:{member['source_container_sha256']}"
            and asset["member_path"] == canonical_path
            and asset["bytes"] == member["source_member_bytes"]
            and asset["sha256"] == member["source_member_sha256"]
            and asset["crc32"] == member["source_member_crc32"]
            and decoded["decoded_content_sha256"] == member["decoded_content_sha256"],
            "SOURCE_ENVELOPE_MEMBER_BINDING_MISMATCH",
            "source envelope does not reproduce the adapter binding",
            role=role,
        )
        if role == "trajectory":
            require(source_path == "lowres_wide.traj", "SOURCE_ENVELOPE_ORIGINAL_PATH", "trajectory original path drift")
        else:
            source_name = PurePosixPath(source_path).name
            canonical_name = PurePosixPath(canonical_path).name
            token = str(envelope["timestamp_token"])
            require(PurePosixPath(source_name).stem == f"{envelope['video_id']}_{token}", "SOURCE_ENVELOPE_ORIGINAL_PATH", "official source member basename drift", role=role)
            require(PurePosixPath(canonical_name).stem == token, "SOURCE_ENVELOPE_CANONICAL_PATH", "adapter canonical member basename drift", role=role)
    return envelope


def adapter_fit_source_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_frame_receipt": frame["source_frame_receipt"],
        "highres_faro_depth_mm": frame["highres_faro_depth_mm"],
        "apple_depth_mm": frame["apple_depth_mm"],
        "confidence": frame["confidence"],
    }


def uncertainty_model_receipt(model: Any) -> dict[str, Any]:
    model._assert_integrity()
    cells = [
        {
            "target": cell.target,
            "parent_id": cell.parent_id,
            "confidence": cell.confidence,
            "range_bin": cell.range_bin,
            "values": cell.values,
        }
        for cell in model.cells
    ]
    return {
        "schema": adapter.UNCERTAINTY_MODEL_SCHEMA,
        "content_sha256": model.content_sha256,
        "fit_parent_ids": list(model.fit_parent_ids),
        "source_receipt_sha256s": list(model.source_receipt_sha256s),
        "source_evidence_sha256": model.source_evidence_sha256,
        "source_frame_count": len(model.source_receipt_sha256s),
        "support_frame_observations": model.support_frame_observations,
        "observation_counts": model.observation_counts,
        "cells_sha256": canonical_sha256(cells),
        "cell_count": len(cells),
    }


def uncertainty_model_artifact(model: Any) -> dict[str, Any]:
    receipt = uncertainty_model_receipt(model)
    cells = [
        {
            "target": cell.target,
            "parent_id": cell.parent_id,
            "confidence": cell.confidence,
            "range_bin": cell.range_bin,
            "values": cell.values,
        }
        for cell in model.cells
    ]
    require(canonical_sha256(cells) == receipt["cells_sha256"], "UNCERTAINTY_MODEL_CELL_HASH_DRIFT", "uncertainty model cell payload drift")
    return seal_record(
        {
            "schema": "blindassist.taro.o0r.uncertainty_model_array_artifact.v1",
            "uncertainty_model_receipt": receipt,
            "factory_bound_model_sha256": model.content_sha256,
            "cells": cells,
        }
    )


def _normalized_vector(value: Any, code: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    require(vector.shape == (3,) and bool(np.all(np.isfinite(vector))), code, "query vector must be finite length three")
    norm = float(np.linalg.norm(vector))
    require(math.isfinite(norm) and norm > 1e-12, code, "query vector norm is invalid")
    return vector / norm


def derive_query_uncertainty_lookup(
    faro_depth_mm: np.ndarray,
    confidence: np.ndarray,
    source_frame_receipt: Mapping[str, Any],
    query_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one query's uncertainty lookup only from bound FARO/confidence evidence."""

    receipt = dict(source_frame_receipt)
    query = dict(query_receipt)
    receipt_payload = copy.deepcopy(receipt)
    receipt_hash = receipt_payload.pop("content_sha256", None)
    query_payload = copy.deepcopy(query)
    query_hash = query_payload.pop("content_sha256", None)
    require(receipt_hash == canonical_sha256(receipt_payload), "BASE_RECEIPT_HASH_MISMATCH", "source receipt hash drift")
    require(query_hash == canonical_sha256(query_payload), "QUERY_RECEIPT_HASH_MISMATCH", "query receipt hash drift")
    require(
        query.get("source_frame_receipt_sha256") == receipt_hash
        and query.get("physical_frame_id") == receipt.get("physical_frame_id")
        and query.get("max_source_timestamp_ns") == receipt.get("max_source_timestamp_ns"),
        "QUERY_BASE_RECEIPT_MISMATCH",
        "query uncertainty lookup is not bound to the source receipt",
    )
    faro_raw = np.asarray(faro_depth_mm)
    conf = np.asarray(confidence)
    require(faro_raw.shape == adapter.HIGHRES_SHAPE_HW and np.issubdtype(faro_raw.dtype, np.integer), "FARO_SHAPE_INVALID", "FARO lookup input must be integer 1440x1920")
    require(conf.shape == adapter.APPLE_SHAPE_HW and np.issubdtype(conf.dtype, np.integer), "CONFIDENCE_DECODE_INVALID", "confidence lookup input must be integer 192x256")
    require(bool(np.all((conf >= 0) & (conf <= 2))), "CONFIDENCE_DECODE_INVALID", "confidence array drift")
    require(receipt["decoded_payload_bindings"]["highres_depth"]["decoded_content_sha256"] == canonical_sha256(faro_raw), "DECODED_PAYLOAD_CONTENT_MISMATCH", "FARO lookup input is not bound to the source receipt")
    require(receipt["decoded_payload_bindings"]["confidence"]["decoded_content_sha256"] == canonical_sha256(conf), "DECODED_PAYLOAD_CONTENT_MISMATCH", "confidence lookup input is not bound to the source receipt")

    faro_m = adapter.sample_faro_at_apple_centers(faro_raw).astype(np.float64) / 1000.0
    lowres = receipt["lowres_intrinsics_source"]
    fx, fy, cx, cy = (float(lowres[key]) for key in ("fx", "fy", "cx", "cy"))
    require(fx > 0.0 and fy > 0.0 and all(math.isfinite(value) for value in (fx, fy, cx, cy)), "INTRINSICS_RANGE", "lowres intrinsics are invalid")
    rows, columns = np.mgrid[0 : adapter.APPLE_SHAPE_HW[0], 0 : adapter.APPLE_SHAPE_HW[1]]
    points = np.stack(
        (
            (columns.astype(np.float64) - cx) * faro_m / fx,
            (rows.astype(np.float64) - cy) * faro_m / fy,
            faro_m,
        ),
        axis=-1,
    )
    frame = query["virtual_query_frame"]
    origin = np.asarray(frame["origin_camera_xyz"], dtype=np.float64)
    lateral = _normalized_vector(frame["lateral_camera_xyz"], "QUERY_FRAME_INVALID")
    heading = _normalized_vector(frame["path_heading_camera_xyz"], "QUERY_FRAME_INVALID")
    query_up = _normalized_vector(frame["gravity_up_camera_xyz"], "QUERY_FRAME_INVALID")
    require(origin.shape == (3,) and bool(np.all(np.isfinite(origin))), "QUERY_FRAME_INVALID", "query origin is invalid")
    path_origin = origin + float(query["path_lateral_offset_m"]) * lateral
    rel = points - path_origin
    along = rel @ heading
    perpendicular_vector = rel - along[..., None] * heading
    perpendicular_ground = perpendicular_vector - (perpendicular_vector @ query_up)[..., None] * query_up
    perpendicular = np.linalg.norm(perpendicular_ground, axis=-1)
    valid_depth = np.isfinite(faro_m) & (faro_m >= adapter.DEPTH_RANGE_M[0]) & (faro_m <= adapter.DEPTH_RANGE_M[1])
    support = (
        valid_depth
        & (along >= float(query["minimum_forward_m"]))
        & (along <= float(query["horizon_m"]))
        & (perpendicular <= float(query["capsule_radius_m"]))
    )
    support_count = int(np.sum(support))
    require(
        support_count >= adapter.MINIMUM_QUERY_SUPPORT_POINTS,
        "QUERY_UNCERTAINTY_SUPPORT_INVALID_FRAME_NOT_ADMITTED",
        "query has insufficient FARO/confidence common support",
        query_id=query.get("query_id"),
        support_count=support_count,
        minimum=adapter.MINIMUM_QUERY_SUPPORT_POINTS,
    )
    selected_confidence = conf[support].astype(np.int64)
    counts = np.bincount(selected_confidence, minlength=3)
    confidence_value = int(np.flatnonzero(counts == np.max(counts))[-1])
    range_m = float(np.median(faro_m[support]))
    support_ids_uv = np.ascontiguousarray(np.column_stack((columns[support], rows[support])), dtype=np.int32)
    lookup = seal_record(
        {
            "schema": "blindassist.taro.o0r.query_uncertainty_lookup.v1",
            "physical_frame_id": receipt["physical_frame_id"],
            "source_frame_receipt_sha256": receipt_hash,
            "query_id": query["query_id"],
            "query_receipt_sha256": query_hash,
            "max_source_timestamp_ns": receipt["max_source_timestamp_ns"],
            "faro_depth_array_sha256": canonical_sha256(faro_raw),
            "confidence_array_sha256": canonical_sha256(conf),
            "support_cell_ids_uv_sha256": canonical_sha256(support_ids_uv),
            "support_count": support_count,
            "confidence_counts": [int(value) for value in counts],
            "confidence_value": confidence_value,
            "range_m": range_m,
            "range_kind": "FARO_OPTICAL_AXIS_Z_METERS_MEDIAN",
            "tie_rule": "HIGHEST_CONFIDENCE_INDEX",
            "corridor_rule": "MINIMUM_FORWARD_THROUGH_HORIZON_INCLUSIVE_CAPSULE_GROUND_PROJECTION_NO_HEIGHT_GATE",
            "caller_scalar_allowed": False,
        }
    )
    return lookup


def build_eval_truth_record(
    frame: Mapping[str, Any],
    uncertainty_model: Any,
    *,
    geometry: Any | None = None,
) -> dict[str, Any]:
    receipt = frame["source_frame_receipt"]
    envelope = validate_bound_source_frame_envelope(frame["bound_source_frame_envelope"], receipt)
    highres = np.asarray(frame["highres_faro_depth_mm"])
    matrix = np.asarray(receipt["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
    if geometry is None:
        geometry = adapter.derive_faro_geometry(highres, matrix, receipt["gravity_up_camera_xyz"], receipt)
    query_receipts = adapter.build_query_receipts(receipt, geometry)
    uncertainty_lookups = [
        derive_query_uncertainty_lookup(highres, np.asarray(frame["confidence"]), receipt, query)
        for query in query_receipts
    ]
    factor_frames = [
        adapter.build_truth_query_factor_frame(
            geometry,
            query,
            uncertainty_model,
            confidence_value=lookup["confidence_value"],
            range_m=lookup["range_m"],
        )
        for query, lookup in zip(query_receipts, uncertainty_lookups, strict=True)
    ]
    bundle = adapter.reduce_complete_query_bundle(factor_frames, query_receipts, receipt)
    return seal_record(
        {
            "schema": "blindassist.taro.o0r.eval_truth_record.v1",
            "source_frame_receipt": receipt,
            "bound_source_frame_envelope": envelope,
            "bound_source_frame_envelope_sha256": envelope["content_sha256"],
            "faro_geometry_sha256": geometry.content_sha256,
            "highres_depth_array_sha256": geometry.highres_depth_array_sha256,
            "uncertainty_lookups": uncertainty_lookups,
            "query_receipts": query_receipts,
            "factor_frames": factor_frames,
            "query_bundle": bundle,
            "model_outputs_absent": True,
        }
    )


def _canonical_array_storage(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(value)
    require(array.dtype.kind in "biuf", "ARRAY_DTYPE_UNSUPPORTED", "only numeric and boolean arrays are canonical")
    if array.dtype.kind == "f":
        require(bool(np.all(np.isfinite(array))), "NONFINITE_CANONICAL_VALUE", "NaN and infinity are forbidden")
        normalized = np.round(array.astype(np.float64), 12)
        normalized[normalized == 0.0] = 0.0
        normalized = np.ascontiguousarray(normalized.astype("<f8", copy=False))
        dtype = "float64_round12_le"
    elif array.dtype.kind == "b":
        normalized = np.ascontiguousarray(array.astype(np.uint8, copy=False))
        dtype = "bool_u8"
    elif array.dtype.kind == "u":
        normalized = np.ascontiguousarray(array.astype("<u8", copy=False))
        dtype = "uint64_le"
    else:
        normalized = np.ascontiguousarray(array.astype("<i8", copy=False))
        dtype = "int64_le"
    raw = normalized.tobytes(order="C")
    receipt = {
        "schema": "blindassist.canonical.ndarray_receipt.v1",
        "dtype": dtype,
        "shape": [int(item) for item in array.shape],
        "sha256": sha256_bytes(raw),
    }
    require(canonical_json_bytes(array) == canonical_json_bytes(receipt), "ARRAY_CANONICALIZATION_DRIFT", "array storage receipt differs from source-adapter canonicalization")
    return normalized, receipt


def _deterministic_gzip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=9, mtime=0) as stream:
        stream.write(payload)
    return output.getvalue()


def package_content_addressed_artifact(value: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
    blobs: dict[str, bytes] = {}
    references: dict[str, dict[str, Any]] = {}
    array_count = 0

    def encode(child: Any) -> Any:
        nonlocal array_count
        if isinstance(child, np.ndarray):
            array_count += 1
            normalized, array_receipt = _canonical_array_storage(child)
            raw = normalized.tobytes(order="C")
            compressed = _deterministic_gzip(raw)
            path = f"arrays/{array_receipt['sha256']}.bin.gz"
            prior = blobs.get(path)
            require(prior is None or prior == compressed, "ARRAY_BLOB_HASH_COLLISION", "array blob path collision")
            blobs[path] = compressed
            reference = {
                "schema": ARRAY_BLOB_REF_SCHEMA,
                "path": path,
                "storage_dtype": array_receipt["dtype"],
                "shape": list(array_receipt["shape"]),
                "raw_bytes": len(raw),
                "raw_sha256": array_receipt["sha256"],
                "gzip_bytes": len(compressed),
                "gzip_sha256": sha256_bytes(compressed),
                "canonical_array_receipt": array_receipt,
            }
            references[canonical_sha256(reference)] = reference
            return reference
        if isinstance(child, dict):
            require(child.get("schema") != ARRAY_BLOB_REF_SCHEMA, "ARRAY_BLOB_REF_SCHEMA_COLLISION", "source artifact may not inject an array-blob reference")
            return {str(key): encode(item) for key, item in child.items()}
        if isinstance(child, (list, tuple)):
            return [encode(item) for item in child]
        if isinstance(child, (np.floating, float)):
            number = float(child)
            require(math.isfinite(number), "NONFINITE_CANONICAL_VALUE", "NaN and infinity are forbidden")
            return number
        if isinstance(child, (np.integer, int)) and not isinstance(child, (np.bool_, bool)):
            return int(child)
        if isinstance(child, (np.bool_, bool)):
            return bool(child)
        if child is None or isinstance(child, str):
            return child
        raise MaterializerError("CANONICAL_TYPE_UNSUPPORTED", f"unsupported artifact type: {type(child).__name__}")

    encoded_payload = encode(value)
    package = seal_record(
        {
            "schema": ARRAY_ARTIFACT_SCHEMA,
            "artifact_canonical_sha256": canonical_sha256(value),
            "payload_with_array_refs": encoded_payload,
            "array_count": array_count,
            "array_blob_references": [references[key] for key in sorted(references)],
            "array_blob_reference_count": len(references),
            "blob_file_count": len(blobs),
        }
    )
    return package, blobs


def _hydrate_array_reference(reference: Mapping[str, Any], compressed: bytes) -> np.ndarray:
    required = {
        "schema",
        "path",
        "storage_dtype",
        "shape",
        "raw_bytes",
        "raw_sha256",
        "gzip_bytes",
        "gzip_sha256",
        "canonical_array_receipt",
    }
    require(isinstance(reference, Mapping) and set(reference) == required and reference.get("schema") == ARRAY_BLOB_REF_SCHEMA, "ARRAY_BLOB_REF_INVALID", "array blob reference fields/schema drift")
    path = str(reference["path"])
    require(path == f"arrays/{reference['raw_sha256']}.bin.gz", "ARRAY_BLOB_PATH_INVALID", "array blob path is not content-addressed")
    require(len(compressed) == reference["gzip_bytes"] and sha256_bytes(compressed) == reference["gzip_sha256"], "ARRAY_BLOB_GZIP_HASH_MISMATCH", "compressed array blob hash/length drift", path=path)
    try:
        raw = gzip.decompress(compressed)
    except (OSError, EOFError) as error:
        raise MaterializerError("ARRAY_BLOB_GZIP_INVALID", "array blob cannot be decompressed", path=path) from error
    require(len(raw) == reference["raw_bytes"] and sha256_bytes(raw) == reference["raw_sha256"], "ARRAY_BLOB_RAW_HASH_MISMATCH", "raw canonical array bytes drift", path=path)
    shape = reference["shape"]
    require(isinstance(shape, list) and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in shape), "ARRAY_BLOB_SHAPE_INVALID", "array blob shape is invalid")
    dtype_name = reference["storage_dtype"]
    storage_dtypes = {
        "float64_round12_le": np.dtype("<f8"),
        "bool_u8": np.dtype("u1"),
        "uint64_le": np.dtype("<u8"),
        "int64_le": np.dtype("<i8"),
    }
    require(dtype_name in storage_dtypes, "ARRAY_BLOB_DTYPE_INVALID", "array blob dtype is invalid")
    dtype = storage_dtypes[dtype_name]
    element_count = math.prod(shape)
    require(len(raw) == element_count * dtype.itemsize, "ARRAY_BLOB_SHAPE_INVALID", "array blob byte length does not match shape/dtype")
    array = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape), order="C").copy()
    if dtype_name == "bool_u8":
        require(bool(np.all((array == 0) | (array == 1))), "ARRAY_BLOB_BOOL_INVALID", "boolean blob contains values other than zero/one")
        array = np.ascontiguousarray(array.astype(np.bool_))
    _, recomputed = _canonical_array_storage(array)
    require(recomputed == reference["canonical_array_receipt"], "ARRAY_BLOB_CANONICAL_RECEIPT_MISMATCH", "hydrated array receipt differs from reference")
    return array


def hydrate_content_addressed_artifact(
    package: Mapping[str, Any],
    blob_loader: Callable[[str], bytes],
) -> Any:
    validated = validate_sealed_record(package, ARRAY_ARTIFACT_SCHEMA, "ARRAY_ARTIFACT_HASH_MISMATCH")
    observed_references: dict[str, dict[str, Any]] = {}
    observed_count = 0

    def decode(child: Any) -> Any:
        nonlocal observed_count
        if isinstance(child, dict) and child.get("schema") == ARRAY_BLOB_REF_SCHEMA:
            observed_count += 1
            reference = copy.deepcopy(child)
            key = canonical_sha256(reference)
            observed_references[key] = reference
            return _hydrate_array_reference(reference, blob_loader(str(reference["path"])))
        if isinstance(child, dict):
            return {str(key): decode(item) for key, item in child.items()}
        if isinstance(child, list):
            return [decode(item) for item in child]
        return child

    hydrated = decode(validated["payload_with_array_refs"])
    expected_references = validated["array_blob_references"]
    require(
        isinstance(expected_references, list)
        and expected_references == [observed_references[key] for key in sorted(observed_references)]
        and validated["array_count"] == observed_count
        and validated["array_blob_reference_count"] == len(observed_references)
        and validated["blob_file_count"] == len({reference["path"] for reference in observed_references.values()}),
        "ARRAY_ARTIFACT_REFERENCE_SET_MISMATCH",
        "artifact array reference summary drift",
    )
    require(canonical_sha256(hydrated) == validated["artifact_canonical_sha256"], "ARRAY_ARTIFACT_ROUND_TRIP_MISMATCH", "hydrated artifact canonical hash differs from in-memory source artifact")
    return hydrated


def evaluate_truth_only_gates(
    fit_parent_ids: Sequence[str],
    parent_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_fit = [visit_id for visit_id, _ in adapter.ADAPTER_FIT_ROSTER]
    expected_eval = [visit_id for visit_id, _ in adapter.O0R_EVAL_CANDIDATE_ROSTER]
    failures: list[str] = []
    if list(fit_parent_ids) != expected_fit:
        failures.append("ALL_ADAPTER_FIT_PARENTS_PRESENT")
    observed_eval = [str(row.get("parent_id")) for row in parent_summaries]
    if observed_eval != expected_eval or len(set(observed_eval)) != len(expected_eval):
        failures.append("EXACT_O0R_EVAL_PARENT_ROSTER")
    evaluable = 0
    truth_clear = 0
    truth_occupied = 0
    normalized: list[dict[str, Any]] = []
    for row in parent_summaries:
        exact = int(row.get("exact_timestamp_frames", 0))
        eligible = int(row.get("source_eligible_frames", 0))
        admitted = int(row.get("admitted_frames", 0))
        clear_frames = int(row.get("clear_frames", 0))
        occupied_frames = int(row.get("occupied_frames", 0))
        if not (0 <= eligible <= exact and 0 <= admitted <= eligible and 0 <= clear_frames <= admitted and 0 <= occupied_frames <= admitted):
            failures.append(f"PARENT_COUNT_INVARIANT:{row.get('parent_id')}")
        eligible_fraction = eligible / exact if exact else None
        complete_fraction = admitted / eligible if eligible else None
        parent_evaluable = (
            exact >= MINIMUM_EXACT_FRAMES_PER_EVAL_PARENT
            and eligible_fraction is not None
            and eligible_fraction >= 0.5
            and complete_fraction == 1.0
        )
        evaluable += int(parent_evaluable)
        truth_clear += int(parent_evaluable and clear_frames >= MINIMUM_STATE_FRAMES_PER_EVAL_PARENT)
        truth_occupied += int(parent_evaluable and occupied_frames >= MINIMUM_STATE_FRAMES_PER_EVAL_PARENT)
        normalized.append(
            {
                **dict(row),
                "source_eligible_fraction": eligible_fraction,
                "complete_factor_query_fraction": complete_fraction,
                "evaluable": parent_evaluable,
            }
        )
        # A source-opportunity-short parent remains in the fixed 16-parent
        # roster and is explicitly NOT_EVALUABLE. It does not veto the cohort
        # when the frozen global minimum of independent parents is evaluable.
    if evaluable < MINIMUM_EVALUABLE_EVAL_PARENTS:
        failures.append("MINIMUM_EVALUABLE_O0R_PARENTS")
    if truth_clear < 6:
        failures.append("MINIMUM_TRUTH_CLEAR_PARENTS")
    if truth_occupied < 6:
        failures.append("MINIMUM_TRUTH_OCCUPIED_PARENTS")
    failures = list(dict.fromkeys(failures))
    return {
        "passed": not failures,
        "terminal": "TARO_O0R_TRUTH_ONLY_ADMISSION_PASS" if not failures else "TARO_O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE",
        "fit_parent_count": len(fit_parent_ids),
        "evaluable_o0r_parent_count": evaluable,
        "truth_clear_parent_count": truth_clear,
        "truth_occupied_parent_count": truth_occupied,
        "source_opportunity_thresholds": {
            "minimum_exact_frames_per_evaluable_parent": MINIMUM_EXACT_FRAMES_PER_EVAL_PARENT,
            "minimum_state_frames_per_evaluable_parent": MINIMUM_STATE_FRAMES_PER_EVAL_PARENT,
            "minimum_evaluable_parent_count": MINIMUM_EVALUABLE_EVAL_PARENTS,
        },
        "parent_summaries": normalized,
        "failure_codes": failures,
        "undefined_denominator_policy": "RETAIN_PARENT_NOT_EVALUABLE_GLOBAL_DENOMINATOR_FIXED_16",
        "replacement_allowed": False,
        "model_outputs_absent": True,
    }


class AtomicEvidenceWriter:
    """Exclusive one-shot writer; root creation is the irreversible consumption event."""

    def __init__(self, root: Path, maximum_bytes: int) -> None:
        self.root = root.resolve()
        self.maximum_bytes = int(maximum_bytes)
        require(self.maximum_bytes > 0, "EVIDENCE_BUDGET_INVALID", "evidence byte budget must be positive")
        self.bytes_written = 0
        self.activated = False
        self.file_receipts: dict[str, dict[str, Any]] = {}

    def activate(self, execution_receipt: Mapping[str, Any]) -> None:
        require(not self.activated and not self.root.exists(), "ROOT_COLLISION", "truth evidence root already exists", root=str(self.root))
        self.root.mkdir(parents=True, exist_ok=False)
        self.activated = True
        self.write_json("execution-receipt.json", dict(execution_receipt))

    def _destination(self, relative: str) -> Path:
        require(self.activated, "WRITER_NOT_ACTIVATED", "evidence root is not activated")
        return safe_join(self.root, relative)

    def _write_bytes(self, relative: str, payload: bytes) -> dict[str, Any]:
        destination = self._destination(relative)
        require(not destination.exists(), "EVIDENCE_OVERWRITE_FORBIDDEN", "evidence path already exists", path=relative)
        require(self.bytes_written + len(payload) <= self.maximum_bytes, "EVIDENCE_BUDGET_EXCEEDED", "evidence byte budget exceeded")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        require(not temporary.exists(), "EVIDENCE_PARTIAL_COLLISION", "evidence partial exists", path=str(temporary))
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        self.bytes_written += len(payload)
        receipt = {"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        self.file_receipts[relative] = receipt
        return receipt

    def _write_blob_once(self, relative: str, payload: bytes) -> dict[str, Any]:
        destination = self._destination(relative)
        if destination.exists():
            prior = self.file_receipts.get(relative)
            require(
                prior is not None
                and destination.is_file()
                and destination.stat().st_size == len(payload)
                and sha256_file(destination) == sha256_bytes(payload),
                "ARRAY_BLOB_DEDUP_MISMATCH",
                "existing content-addressed blob differs from requested payload",
                path=relative,
            )
            return dict(prior) | {"deduplicated": True}
        return self._write_bytes(relative, payload) | {"deduplicated": False}

    def write_json(self, relative: str, value: Any) -> dict[str, Any]:
        return self._write_bytes(relative, canonical_json_bytes(value) + b"\n")

    def write_json_gzip(self, relative: str, value: Any) -> dict[str, Any]:
        raw = canonical_json_bytes(value) + b"\n"
        return self._write_bytes(relative, _deterministic_gzip(raw))

    def write_content_addressed_artifact(self, relative: str, value: Any) -> dict[str, Any]:
        package, blobs = package_content_addressed_artifact(value)
        blob_receipts = [self._write_blob_once(path, blobs[path]) for path in sorted(blobs)]
        package_receipt = self.write_json_gzip(relative, package)
        hydrated = hydrate_content_addressed_artifact(
            package,
            lambda path: self._destination(path).read_bytes(),
        )
        require(canonical_sha256(hydrated) == canonical_sha256(value), "ARRAY_ARTIFACT_ROUND_TRIP_MISMATCH", "writer reload gate differs from in-memory artifact")
        return {
            "artifact_canonical_sha256": package["artifact_canonical_sha256"],
            "package_content_sha256": package["content_sha256"],
            "package_file": package_receipt,
            "blob_files": blob_receipts,
            "reload_verified": True,
        }


def execution_deadline_exceeded(started: float, wall_seconds: float) -> bool:
    return time.monotonic() - started > wall_seconds
