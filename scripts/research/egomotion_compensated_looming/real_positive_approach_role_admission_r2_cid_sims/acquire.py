"""One-shot ScienceDB acquisition after the exclusive CID-SIMS R2 claim.

This module never performs a HEAD, range request, redirect, retry, mirror,
fallback, or RGB member read.  Archive inspection is limited to ZIP metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable


PROTOCOL_ID = "RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R2_CID_SIMS"
CANDIDATE_ID = "CID_SIMS_V6_FLOOR3_1"
SEQUENCE_ID = "floor3_1"
OFFICIAL_FILE_ID = "c595882daafe788a29d687872cc1fc2a"
OFFICIAL_PAYLOAD_URL = (
    "https://china.scidb.cn/download?fileId="
    "c595882daafe788a29d687872cc1fc2a"
)
OFFICIAL_PAYLOAD_BYTES = 2_211_008_069
OFFICIAL_PAYLOAD_MD5 = "585d38855ad7d04817991cdbbb72016b"
ARCHIVE_NAME = "floor3_1.zip"
USER_AGENT = "BlindAssist-RCLE-RoleAdmission-R2-CID-SIMS/1.0"
HOLD = "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID"


class SourceAccessHold(RuntimeError):
    """A consumed one-shot source attempt with a valid HOLD terminal."""

    def __init__(self, code: str, receipt: dict[str, Any]):
        super().__init__(code)
        self.code = code
        self.receipt = receipt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("R2_JSON_OBJECT_REQUIRED")
    return value


def _candidate(authority: dict[str, Any]) -> dict[str, Any]:
    candidate = authority.get("candidate")
    if not isinstance(candidate, dict):
        candidate = authority.get("source_selection")
    if not isinstance(candidate, dict):
        raise ValueError("R2_AUTHORITY_CANDIDATE")
    return candidate


def _candidate_value(candidate: dict[str, Any], key: str) -> Any:
    if key in candidate:
        return candidate[key]
    archive = candidate.get("archive")
    if isinstance(archive, dict) and key in archive:
        return archive[key]
    raise ValueError(f"R2_AUTHORITY_CANDIDATE_FIELD:{key}")


def _validate_exact_url(value: str) -> None:
    if value != OFFICIAL_PAYLOAD_URL:
        raise ValueError("R2_OFFICIAL_GET_RESPONSE")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect; the frozen canonical entry must be the response."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        raise ValueError("R2_REDIRECT_FORBIDDEN")


def _official_urlopen(request: urllib.request.Request, timeout: int) -> Any:
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _safe_archive_inventory(path: Path) -> list[dict[str, object]]:
    """Read only ZIP central-directory metadata; never open archive members."""

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        normalized: list[str] = []
        for info in infos:
            name = info.filename
            parsed = PurePosixPath(name)
            normalized_name = parsed.as_posix()
            if (
                parsed.is_absolute()
                or ".." in parsed.parts
                or "\\" in name
                or bool(re.match(r"^[A-Za-z]:", name))
                or info.flag_bits & 0x1
            ):
                raise ValueError("R2_ARCHIVE_UNSAFE_MEMBER")
            normalized.append(normalized_name)
        if len(normalized) != len(set(normalized)):
            raise ValueError("R2_ARCHIVE_DUPLICATE_MEMBER")
        return [
            {
                "name": normalized_name,
                "crc32": f"{info.CRC:08x}",
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "is_directory": info.is_dir(),
            }
            for info, normalized_name in zip(infos, normalized)
        ]


def _validate_preaccess_bindings(
    claim_path: Path,
    contract_path: Path,
    source_authority_path: Path,
    burned_manifest_path: Path,
    implementation_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    claim = _load_object(claim_path)
    if (
        claim.get("protocol_id") != PROTOCOL_ID
        or claim.get("candidate_id") != CANDIDATE_ID
        or claim.get("sequence_id") != SEQUENCE_ID
    ):
        raise ValueError("R2_CLAIM_IDENTITY")
    expected_claim_identity = {
        "official_file_id": OFFICIAL_FILE_ID,
        "official_payload_url": OFFICIAL_PAYLOAD_URL,
        "official_payload_bytes": OFFICIAL_PAYLOAD_BYTES,
        "official_payload_md5": OFFICIAL_PAYLOAD_MD5,
    }
    for key, expected in expected_claim_identity.items():
        if claim.get(key) != expected:
            raise ValueError(f"R2_CLAIM_SOURCE_IDENTITY:{key}")
    if (
        claim.get("source_access_started_before_claim") is not False
        or claim.get("candidate_path_probe_started_before_claim") is not False
        or claim.get("request_count_before_claim") != 0
        or claim.get("payload_bytes_read_before_claim") != 0
        or claim.get("algorithm_outcome_access_started") is not False
    ):
        raise ValueError("R2_PRECLAIM_ACCESS")

    paths = {
        "contract": contract_path,
        "source_authority": source_authority_path,
        "burned_manifest": burned_manifest_path,
        "implementation_lock": implementation_lock_path,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    for name, digest in observed.items():
        if claim.get("bindings", {}).get(name, {}).get("sha256") != digest:
            raise ValueError(f"R2_CLAIM_BINDING_MISMATCH:{name}")

    authority_candidate = _candidate(_load_object(source_authority_path))
    expected_authority_identity = {
        "candidate_id": CANDIDATE_ID,
        "sequence_id": SEQUENCE_ID,
        "file_id": OFFICIAL_FILE_ID,
        "official_payload_url": OFFICIAL_PAYLOAD_URL,
        "bytes": OFFICIAL_PAYLOAD_BYTES,
        "official_md5": OFFICIAL_PAYLOAD_MD5,
    }
    for key, expected in expected_authority_identity.items():
        if _candidate_value(authority_candidate, key) != expected:
            raise ValueError(f"R2_AUTHORITY_SOURCE_IDENTITY:{key}")
    return claim, observed


def acquire_once(
    claim_path: Path,
    contract_path: Path,
    source_authority_path: Path,
    burned_manifest_path: Path,
    implementation_lock_path: Path,
    archive_path: Path,
    receipt_path: Path,
    *,
    opener: Callable[..., Any] = _official_urlopen,
    progress_callback: Callable[[int], None] | None = None,
    expected_payload_bytes: int = OFFICIAL_PAYLOAD_BYTES,
    expected_payload_md5: str = OFFICIAL_PAYLOAD_MD5,
) -> dict[str, Any]:
    """Consume exactly one GET and freeze either a receipt or valid HOLD."""

    claim, document_hashes = _validate_preaccess_bindings(
        claim_path,
        contract_path,
        source_authority_path,
        burned_manifest_path,
        implementation_lock_path,
    )
    if archive_path.name != ARCHIVE_NAME:
        raise ValueError("R2_ARCHIVE_FILENAME")
    if expected_payload_bytes <= 0 or not re.fullmatch(
        r"[0-9a-f]{32}", expected_payload_md5
    ):
        raise ValueError("R2_EXPECTED_PAYLOAD_IDENTITY")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_fd = os.open(
        os.fspath(archive_path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    sha256_digest = hashlib.sha256()
    md5_digest = hashlib.md5(usedforsecurity=False)
    byte_count = 0
    request = urllib.request.Request(
        OFFICIAL_PAYLOAD_URL,
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    request_started_at = datetime.now(timezone.utc).isoformat()
    final_url: str | None = None
    source_error: BaseException | None = None

    def freeze_source_hold(error: BaseException, code: str) -> None:
        receipt = {
            "schema_version": "rcle.real_positive_approach_role_source_hold.v2",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "sequence_id": SEQUENCE_ID,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "terminal": HOLD,
            "validation_scope": "ONE_SHOT_SOURCE_ACCESS_PROCEDURE_ONLY",
            "source_access_complete": False,
            "request_method": "GET",
            "request_count": 1,
            "request_started_at_utc": request_started_at,
            "requested_url": OFFICIAL_PAYLOAD_URL,
            "final_url": final_url,
            "redirect_chain": [],
            "redirect_count": 0,
            "retry_count": 0,
            "fallback_count": 0,
            "mirror_count": 0,
            "head_request_count": 0,
            "range_request_count": 0,
            "replacement_source_count": 0,
            "response_artifact_bytes": byte_count,
            "response_artifact_md5": md5_digest.hexdigest(),
            "response_artifact_sha256": sha256_digest.hexdigest(),
            "expected_archive_bytes": expected_payload_bytes,
            "expected_archive_md5": expected_payload_md5,
            "claim_sha256": sha256_file(claim_path),
            "claim_created_at_utc": claim["created_at_utc"],
            "preaccess_document_sha256": document_hashes,
            "source_error_type": type(error).__name__,
            "source_error_code": code,
            "retry_authorized": False,
            "replacement_source_authorized": False,
            "rgb_members_read": 0,
        }
        encoded = (
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_fd = os.open(
            os.fspath(receipt_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(receipt_fd, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        raise SourceAccessHold(code, receipt)

    # Exactly one opener invocation. No HEAD, Range, redirect, retry, mirror,
    # fallback, alternate Floor3 run, CoRBS, or KITTI-360 request exists here.
    try:
        response_context = opener(request, timeout=120)
    except (TimeoutError, urllib.error.URLError) as error:
        response_context = None
        source_error = error

    with os.fdopen(archive_fd, "wb", closefd=True) as output:
        if response_context is not None:
            response = None
            try:
                response = response_context.__enter__()
                status = getattr(response, "status", 200)
                final_url = response.geturl()
                if status != 200:
                    source_error = ValueError("R2_OFFICIAL_GET_RESPONSE")
                else:
                    try:
                        _validate_exact_url(final_url)
                    except ValueError as error:
                        source_error = error
                while source_error is None:
                    try:
                        chunk = response.read(1024 * 1024)
                    except (
                        OSError,
                        TimeoutError,
                        urllib.error.URLError,
                    ) as error:
                        source_error = error
                        break
                    if not chunk:
                        break
                    output.write(chunk)
                    sha256_digest.update(chunk)
                    md5_digest.update(chunk)
                    byte_count += len(chunk)
                    if progress_callback is not None:
                        progress_callback(byte_count)
            except (TimeoutError, urllib.error.URLError) as error:
                source_error = error
            finally:
                if response is not None:
                    try:
                        response_context.__exit__(None, None, None)
                    except (
                        OSError,
                        TimeoutError,
                        urllib.error.URLError,
                    ) as error:
                        if source_error is None:
                            source_error = error
        output.flush()
        os.fsync(output.fileno())

    if source_error is not None:
        freeze_source_hold(source_error, "R2_OFFICIAL_SOURCE_ACCESS_FAILED")
    if (
        byte_count != expected_payload_bytes
        or md5_digest.hexdigest() != expected_payload_md5
    ):
        freeze_source_hold(
            ValueError("R2_OFFICIAL_PAYLOAD_IDENTITY_MISMATCH"),
            "R2_OFFICIAL_PAYLOAD_IDENTITY_MISMATCH",
        )

    try:
        inventory = _safe_archive_inventory(archive_path)
    except zipfile.BadZipFile as error:
        freeze_source_hold(error, "R2_OFFICIAL_CONTAINER_UNREADABLE")
        raise AssertionError("unreachable")

    file_names = {
        str(row["name"]) for row in inventory if row["is_directory"] is False
    }
    roots = {PurePosixPath(name).parts[0] for name in file_names if name}
    if len(roots) != 1:
        raise ValueError("R2_ARCHIVE_MULTI_SEQUENCE_ROOT")
    if roots != {SEQUENCE_ID}:
        freeze_source_hold(
            ValueError("R2_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE"),
            "R2_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE",
        )
    if (
        f"{SEQUENCE_ID}/pose.txt" not in file_names
        or not any(name.startswith(f"{SEQUENCE_ID}/color/") for name in file_names)
        or not any(name.startswith(f"{SEQUENCE_ID}/depth/") for name in file_names)
    ):
        freeze_source_hold(
            ValueError("R2_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE"),
            "R2_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE",
        )

    receipt: dict[str, Any] = {
        "schema_version": "rcle.real_positive_approach_role_acquisition.v2",
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "sequence_id": SEQUENCE_ID,
        "official_file_id": OFFICIAL_FILE_ID,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_method": "GET",
        "request_count": 1,
        "request_started_at_utc": request_started_at,
        "requested_url": OFFICIAL_PAYLOAD_URL,
        "final_url": final_url,
        "redirect_chain": [],
        "redirect_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "mirror_count": 0,
        "head_request_count": 0,
        "range_request_count": 0,
        "replacement_source_count": 0,
        "archive_filename": ARCHIVE_NAME,
        "archive_bytes": byte_count,
        "archive_md5": md5_digest.hexdigest(),
        "archive_sha256": sha256_digest.hexdigest(),
        "archive_member_inventory": inventory,
        "archive_member_inventory_sha256": hashlib.sha256(
            json.dumps(
                inventory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "claim_sha256": sha256_file(claim_path),
        "claim_created_at_utc": claim["created_at_utc"],
        "preaccess_document_sha256": document_hashes,
        "rgb_members_read": 0,
    }
    encoded = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_fd = os.open(
        os.fspath(receipt_path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(receipt_fd, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-authority", required=True, type=Path)
    parser.add_argument("--burned-manifest", required=True, type=Path)
    parser.add_argument("--implementation-lock", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = acquire_once(
        args.claim,
        args.contract,
        args.source_authority,
        args.burned_manifest,
        args.implementation_lock,
        args.archive,
        args.receipt,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
