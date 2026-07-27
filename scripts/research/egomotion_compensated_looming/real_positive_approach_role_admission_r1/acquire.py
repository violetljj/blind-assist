"""One-shot exact-official acquisition after the exclusive R1 claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import urllib.parse
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable


PROTOCOL_ID = "RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R1"
CANDIDATE_ID = "ETH3D_SLAM_SOFA_3_RGBD"
OFFICIAL_PAYLOAD_URL = "https://www.eth3d.net/data/slam/datasets/sofa_3_rgbd.zip"
ARCHIVE_NAME = "sofa_3_rgbd.zip"
USER_AGENT = "BlindAssist-RCLE-PhaseB-RoleAdmission-R1/1.0"
APPROVED_OFFICIAL_HOSTS = {"www.eth3d.net", "eth3d.ethz.ch"}
OFFICIAL_PAYLOAD_PATH = "/data/slam/datasets/sofa_3_rgbd.zip"
HOLD = "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID"


class SourceAccessHold(RuntimeError):
    """A one-shot official-source failure whose frozen scientific result is HOLD."""

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
        raise ValueError("R1_JSON_OBJECT_REQUIRED")
    return value


def _validate_final_url(value: str) -> None:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in APPROVED_OFFICIAL_HOSTS
        or parsed.path != OFFICIAL_PAYLOAD_PATH
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("R1_OFFICIAL_GET_RESPONSE")


class _StrictOfficialRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        super().__init__()
        self.redirect_chain: list[str] = []

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        _validate_final_url(new_url)
        self.redirect_chain.append(new_url)
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def _official_urlopen(request: urllib.request.Request, timeout: int) -> Any:
    redirect_handler = _StrictOfficialRedirectHandler()
    opener = urllib.request.build_opener(redirect_handler)
    response = opener.open(request, timeout=timeout)
    response.r1_redirect_chain = list(redirect_handler.redirect_chain)
    return response


def _safe_archive_inventory(path: Path) -> list[dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        normalized = [PurePosixPath(name).as_posix() for name in names]
        if len(normalized) != len(set(normalized)):
            raise ValueError("R1_ARCHIVE_DUPLICATE_MEMBER")
        for info, name in zip(infos, names):
            parsed = PurePosixPath(name)
            if (
                parsed.is_absolute()
                or ".." in parsed.parts
                or "\\" in name
                or bool(re.match(r"^[A-Za-z]:", name))
                or info.flag_bits & 0x1
            ):
                raise ValueError("R1_ARCHIVE_UNSAFE_MEMBER")
        return [
            {
                "name": info.filename,
                "crc32": f"{info.CRC:08x}",
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "is_directory": info.is_dir(),
            }
            for info in infos
        ]


def _validate_preaccess_bindings(
    claim_path: Path,
    contract_path: Path,
    source_authority_path: Path,
    burned_manifest_path: Path,
    implementation_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    claim = _load_object(claim_path)
    if claim.get("protocol_id") != PROTOCOL_ID or claim.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("R1_CLAIM_IDENTITY")
    if claim.get("official_payload_url") != OFFICIAL_PAYLOAD_URL:
        raise ValueError("R1_CLAIM_URL")
    if (
        claim.get("source_access_started_before_claim") is not False
        or claim.get("candidate_path_probe_started_before_claim") is not False
        or claim.get("request_count_before_claim") != 0
        or claim.get("payload_bytes_read_before_claim") != 0
    ):
        raise ValueError("R1_PRECLAIM_ACCESS")
    paths = {
        "contract": contract_path,
        "source_authority": source_authority_path,
        "burned_manifest": burned_manifest_path,
        "implementation_lock": implementation_lock_path,
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    for name, digest in observed.items():
        if claim.get("bindings", {}).get(name, {}).get("sha256") != digest:
            raise ValueError(f"R1_CLAIM_BINDING_MISMATCH:{name}")
    authority = _load_object(source_authority_path)
    if authority["candidate"]["official_payload_url"] != OFFICIAL_PAYLOAD_URL:
        raise ValueError("R1_AUTHORITY_URL")
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
) -> dict[str, Any]:
    claim, document_hashes = _validate_preaccess_bindings(
        claim_path,
        contract_path,
        source_authority_path,
        burned_manifest_path,
        implementation_lock_path,
    )
    if archive_path.name != ARCHIVE_NAME:
        raise ValueError("R1_ARCHIVE_FILENAME")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_fd = os.open(
        os.fspath(archive_path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    digest = hashlib.sha256()
    byte_count = 0
    request = urllib.request.Request(
        OFFICIAL_PAYLOAD_URL,
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    request_started_at = datetime.now(timezone.utc).isoformat()
    final_url: str | None = None
    redirect_chain: list[str] = []
    source_error: BaseException | None = None

    def freeze_source_hold(error: BaseException, code: str) -> None:
        receipt = {
            "schema_version": "rcle.real_positive_approach_role_source_hold.v1",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "terminal": HOLD,
            "validation_scope": "ONE_SHOT_SOURCE_ACCESS_PROCEDURE_ONLY",
            "source_access_complete": False,
            "request_method": "GET",
            "request_count": 1,
            "request_started_at_utc": request_started_at,
            "requested_url": OFFICIAL_PAYLOAD_URL,
            "final_url": final_url,
            "redirect_chain": redirect_chain,
            "retry_count": 0,
            "fallback_count": 0,
            "mirror_count": 0,
            "head_request_count": 0,
            "replacement_source_count": 0,
            "partial_archive_bytes": byte_count,
            "partial_archive_sha256": digest.hexdigest(),
            "claim_sha256": sha256_file(claim_path),
            "claim_created_at_utc": claim["created_at_utc"],
            "preaccess_document_sha256": document_hashes,
            "source_error_type": type(error).__name__,
            "source_error_code": code,
            "retry_authorized": False,
            "replacement_source_authorized": False,
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
        raise SourceAccessHold(code, receipt)
    # Exactly one opener invocation. There is intentionally no retry,
    # fallback, mirror, index request, range request, or HEAD probe.
    try:
        response_context = opener(request, timeout=120)
    except (TimeoutError, urllib.error.URLError) as error:
        response_context = None
        source_error = error
    with os.fdopen(descriptor_fd, "wb", closefd=True) as output:
        if response_context is not None:
            response = None
            try:
                response = response_context.__enter__()
                status = getattr(response, "status", 200)
                final_url = response.geturl()
                redirect_chain = list(
                    getattr(response, "r1_redirect_chain", [])
                )
                if status != 200:
                    source_error = ValueError("R1_OFFICIAL_GET_RESPONSE")
                else:
                    try:
                        _validate_final_url(final_url)
                    except ValueError as error:
                        source_error = error
                while source_error is None:
                    try:
                        chunk = response.read(1024 * 1024)
                    except (OSError, TimeoutError, urllib.error.URLError) as error:
                        source_error = error
                        break
                    if not chunk:
                        break
                    # Local evidence writes and progress callbacks are outside
                    # any source-error catch and therefore remain INVALID.
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if progress_callback is not None:
                        progress_callback(byte_count)
            except (TimeoutError, urllib.error.URLError) as error:
                source_error = error
            finally:
                if response is not None:
                    try:
                        response_context.__exit__(None, None, None)
                    except (OSError, TimeoutError, urllib.error.URLError) as error:
                        if source_error is None:
                            source_error = error
        output.flush()
        os.fsync(output.fileno())

    if source_error is not None:
        freeze_source_hold(source_error, "R1_OFFICIAL_SOURCE_ACCESS_FAILED")

    try:
        inventory = _safe_archive_inventory(archive_path)
    except zipfile.BadZipFile as error:
        freeze_source_hold(error, "R1_OFFICIAL_CONTAINER_UNREADABLE")
        raise AssertionError("unreachable")
    file_names = {
        row["name"] for row in inventory if row["is_directory"] is False
    }
    required = {
        f"sofa_3/{name}"
        for name in (
            "calibration.txt",
            "groundtruth.txt",
            "associated.txt",
            "rgb.txt",
            "depth.txt",
        )
    }
    roots = {PurePosixPath(name).parts[0] for name in file_names if name}
    if len(roots) > 1:
        raise ValueError("R1_ARCHIVE_MULTI_SEQUENCE_ROOT")
    if (
        not required.issubset(file_names)
        or roots != {"sofa_3"}
        or not any(name.startswith("sofa_3/rgb/") for name in file_names)
        or not any(name.startswith("sofa_3/depth/") for name in file_names)
    ):
        freeze_source_hold(
            ValueError("R1_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE"),
            "R1_OFFICIAL_CONTAINER_IDENTITY_INCOMPLETE",
        )

    receipt: dict[str, Any] = {
        "schema_version": "rcle.real_positive_approach_role_acquisition.v1",
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_method": "GET",
        "request_count": 1,
        "request_started_at_utc": request_started_at,
        "requested_url": OFFICIAL_PAYLOAD_URL,
        "final_url": final_url,
        "redirect_chain": redirect_chain,
        "http_status": 200,
        "retry_count": 0,
        "fallback_count": 0,
        "mirror_count": 0,
        "head_request_count": 0,
        "archive_filename": ARCHIVE_NAME,
        "archive_bytes": byte_count,
        "archive_sha256": digest.hexdigest(),
        "archive_member_inventory": inventory,
        "archive_member_inventory_sha256": hashlib.sha256(
            json.dumps(
                inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "claim_sha256": sha256_file(claim_path),
        "claim_created_at_utc": claim["created_at_utc"],
        "preaccess_document_sha256": document_hashes,
        "replacement_source_count": 0,
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
