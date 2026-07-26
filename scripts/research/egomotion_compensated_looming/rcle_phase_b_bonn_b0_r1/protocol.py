from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence
import urllib.request
import zipfile


SEQUENCE_IDS = (
    "rgbd_bonn_crowd2",
    "rgbd_bonn_balloon_tracking",
    "rgbd_bonn_balloon_tracking2",
    "rgbd_bonn_moving_obstructing_box2",
    "rgbd_bonn_balloon2",
    "rgbd_bonn_moving_nonobstructing_box2",
)
URL_TEMPLATE = (
    "https://www.ipb.uni-bonn.de/html/projects/"
    "rgbd_dynamic2019/{sequence_id}.zip"
)
DESIGN_LOCK_SHA256 = (
    "396444305bae01eb5a8e95a92044cbea9aa7084c605993c789ecb4f47e234e74"
)
PREREGISTRATION_SHA256 = (
    "f50cf66c46fe33aa3c1e60fa3c25cb120389eafdbad92f4d0a9df22d7cc68da2"
)
R0_CONTRACT_RESULT_SHA256 = (
    "7c5aa8c66b6d99803b4ae2945dfcf95fe7c7bffc7919423df9b68a03fdf1f734"
)
R3_RECEIPT_SHA256 = (
    "05a283b84f62bee000447bb567eadd63b424afaa9d81f5f0d83d36a9ed02489b"
)
COHORT_IDENTITY_SHA256 = (
    "513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e"
)
PASS_TERMINAL = (
    "PHASE_B_B0_R1_INVENTORY_PASS_B1_METRIC_PROTOCOL_MAY_BE_FROZEN"
)
FAIL_TERMINAL = (
    "HOLD_PHASE_B_B0_R1_NOT_EVALUABLE_NO_REPLACEMENT_NO_RERUN"
)
RECEIPT_SCHEMA_VERSION = "rcle.phase_b.bonn_b0_r1.receipt.v1"
VALIDATION_SCHEMA_VERSION = "rcle.phase_b.bonn_b0_r1.validation.v1"
CHUNK_BYTES = 1024 * 1024
MAX_ATTEMPTS = 3
WINDOW_SECONDS = Decimal("10")
TIMESTAMP_PATTERN = re.compile(
    rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


class DurableLedgerError(BaseException):
    """Stop the claimed run if attempt evidence cannot be persisted."""


class RetryableTransportError(Exception):
    """A frozen transport-stage failure that consumes one bounded attempt."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def now_hong_kong() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def canonical_paths(repo_root: Path) -> dict[str, Path]:
    module = (
        repo_root / "scripts" / "research" / "egomotion_compensated_looming"
    )
    docs = repo_root / "docs" / "research" / "rcle"
    archive_root = (
        repo_root
        / "artifacts.local"
        / "datasets"
        / "rcle_phase_b_bonn_b0_r1"
        / "archives"
    )
    output = (
        repo_root
        / "artifacts.local"
        / "evidence"
        / "rcle_phase_b_bonn_b0_r1"
        / "formal_entry_b0_r1"
    )
    return {
        "design_lock": docs
        / "RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1_DESIGN_LOCK_2026-07-26.json",
        "preregistration": docs
        / "RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1_PREREGISTRATION_2026-07-26.md",
        "r0_contract_result": docs
        / "RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R0_EXECUTION_CONTRACT_RESULT_2026-07-26.md",
        "r3_receipt": (
            repo_root
            / "artifacts.local"
            / "evidence"
            / "rcle_phase_b_bonn_entry_r3"
            / "authority_gate_r3"
            / "receipt.json"
        ),
        "implementation_lock": (
            module
            / "rcle_phase_b_bonn_b0_r1"
            / "RCLE_PHASE_B_BONN_B0_R1_IMPLEMENTATION_LOCK.json"
        ),
        "archive_root": archive_root,
        "output": output,
        "run_claim": output / "run_claim.json",
        "receipt": output / "receipt.json",
        "validation": output / "receipt_validation.json",
        "attempt_ledger": output / "transport_attempt_ledger.json",
    }


def environment_manifest() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
    }


def _git(args: Sequence[str], repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def validate_implementation_lock(
    repo_root: Path, paths: dict[str, Path]
) -> dict[str, Any]:
    lock = json.loads(paths["implementation_lock"].read_text(encoding="utf-8"))
    controls = {
        relative: sha256_file(repo_root / relative)
        for relative in sorted(lock["control_source_manifest"])
    }
    expected_paths = {
        "archive_root": str(paths["archive_root"]),
        "output": str(paths["output"]),
        "run_claim": str(paths["run_claim"]),
        "receipt": str(paths["receipt"]),
        "validation": str(paths["validation"]),
        "attempt_ledger": str(paths["attempt_ledger"]),
    }
    checks = {
        "protocol": lock.get("protocol_id")
        == "RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1",
        "design": (
            lock.get("design_lock_sha256") == DESIGN_LOCK_SHA256
            and sha256_file(paths["design_lock"]) == DESIGN_LOCK_SHA256
        ),
        "preregistration": (
            lock.get("preregistration_sha256") == PREREGISTRATION_SHA256
            and sha256_file(paths["preregistration"])
            == PREREGISTRATION_SHA256
        ),
        "r0_contract": (
            lock.get("r0_contract_result_sha256")
            == R0_CONTRACT_RESULT_SHA256
            and sha256_file(paths["r0_contract_result"])
            == R0_CONTRACT_RESULT_SHA256
        ),
        "r3": (
            lock.get("metadata_authority_r3_receipt_sha256")
            == R3_RECEIPT_SHA256
            and sha256_file(paths["r3_receipt"]) == R3_RECEIPT_SHA256
        ),
        "cohort": lock.get("cohort_identity_sha256")
        == COHORT_IDENTITY_SHA256,
        "sequences": lock.get("sequence_ids_in_rank_order")
        == list(SEQUENCE_IDS),
        "paths": lock.get("canonical_paths") == expected_paths,
        "controls": lock.get("control_source_manifest") == controls,
        "claim": (
            lock.get("preclaim_network_rule")
            == (
                "BEFORE_CLAIM_FSYNC_NO_NETWORK_OPERATION_TO_ANY_DESTINATION_"
                "INCLUDING_DNS_SEARCH_API_MIRROR_CDN_URL_HOST_METHOD_OR_"
                "TRANSPORT_LIBRARY"
            )
            and lock.get("maximum_run_claims") == 1
        ),
        "attempts": (
            lock.get("maximum_get_attempts_per_url") == MAX_ATTEMPTS
            and lock.get("range_resume_authorized") is False
        ),
        "firewall": (
            lock.get("rgb_depth_decode_authorized") is False
            and lock.get("pose_numeric_authorized") is False
            and lock.get("phase_b_metrics_authorized") is False
        ),
        "terminals": (
            lock.get("pass_terminal") == PASS_TERMINAL
            and lock.get("fail_terminal") == FAIL_TERMINAL
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("B0_R1_IMPLEMENTATION_LOCK_MISMATCH:" + ",".join(failed))
    return lock


def preflight_not_started(paths: dict[str, Path]) -> None:
    if paths["run_claim"].exists():
        raise FileExistsError("B0_R1_RUN_CLAIM_ALREADY_EXISTS_NO_RERUN")
    for key in ("archive_root", "output"):
        root = paths[key]
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"B0_R1_CANONICAL_{key.upper()}_NOT_EMPTY")


def create_run_claim(repo_root: Path, command: Sequence[str]) -> dict[str, Any]:
    paths = canonical_paths(repo_root)
    preflight_not_started(paths)
    paths["archive_root"].mkdir(parents=True, exist_ok=True)
    paths["output"].mkdir(parents=True, exist_ok=True)
    claim = {
        "schema_version": "rcle.phase_b.bonn_b0_r1.run_claim.v1",
        "protocol_id": "RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1",
        "claimed_at": now_hong_kong(),
        "canonical_output": str(paths["output"]),
        "command": list(command),
        "maximum_run_claims": 1,
        "exclusive_create": True,
        "survives_failure_interrupt_and_success": True,
        "network_operations_before_claim": 0,
        "pre_r1_head_disclosure": {
            "request_count": 6,
            "response_body_bytes": 0,
            "reported_content_length_total_bytes": 2262988443,
            "authority": (
                "NON_AUTHORITATIVE_TRANSPORT_DISCOVERY_"
                "EXCLUDED_FROM_ALL_GATES"
            ),
        },
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "metadata_authority_r3_receipt_sha256": R3_RECEIPT_SHA256,
        "cohort_identity_sha256": COHORT_IDENTITY_SHA256,
        "implementation_lock_sha256": sha256_file(
            paths["implementation_lock"]
        ),
        "canonical_archive_root": str(paths["archive_root"]),
        "canonical_output": str(paths["output"]),
        "canonical_run_claim": str(paths["run_claim"]),
    }
    descriptor = os.open(
        paths["run_claim"],
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        payload = (
            json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return claim


def _safe_header_subset(headers: Any) -> dict[str, str]:
    allowed = (
        "Content-Length",
        "Content-Type",
        "ETag",
        "Last-Modified",
        "Accept-Ranges",
        "Content-Encoding",
    )
    return {
        name.lower().replace("-", "_"): str(headers.get(name))
        for name in allowed
        if headers.get(name) is not None
    }


def _default_open(url: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": "BlindAssist-RCLE-PhaseB-B0-R1/1.0",
            "Accept": "application/zip",
            "Accept-Encoding": "identity",
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def acquire_archive(
    sequence_id: str,
    destination: Path,
    *,
    opener: Callable[[str, float], Any] = _default_open,
    timeout: float = 120.0,
    maximum_attempts: int = MAX_ATTEMPTS,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sequence_id not in SEQUENCE_IDS:
        raise ValueError("SEQUENCE_OUTSIDE_FROZEN_COHORT")
    if maximum_attempts != MAX_ATTEMPTS:
        raise ValueError("ATTEMPT_COUNT_OVERRIDE_FORBIDDEN")
    url = URL_TEMPLATE.format(sequence_id=sequence_id)
    part = destination.with_suffix(destination.suffix + ".part")
    attempts: list[dict[str, Any]] = []
    if destination.exists():
        raise FileExistsError("CANONICAL_ARCHIVE_ALREADY_EXISTS_NO_RERUN")
    for attempt_number in range(1, maximum_attempts + 1):
        started = now_hong_kong()
        digest = hashlib.sha256()
        bytes_written = 0
        response_status: int | None = None
        final_url: str | None = None
        headers: dict[str, str] = {}
        try:
            target = part.open("wb")
            target.flush()
            os.fsync(target.fileno())
        except Exception:
            raise
        try:
            try:
                response_context = opener(url, timeout)
            except Exception as error:
                raise RetryableTransportError(
                    "TRANSPORT_OPEN_FAILED"
                ) from error
            with response_context as response:
                response_status = int(getattr(response, "status", 200))
                final_url = str(response.geturl())
                headers = _safe_header_subset(response.headers)
                if response_status != 200:
                    raise RetryableTransportError(
                        f"HTTP_STATUS_{response_status}"
                    )
                if final_url != url:
                    raise RetryableTransportError(
                        "REDIRECT_OR_URL_IDENTITY_MISMATCH"
                    )
                content_encoding = headers.get(
                    "content_encoding", "identity"
                ).lower()
                if content_encoding != "identity":
                    raise RetryableTransportError(
                        "CONTENT_ENCODING_NOT_IDENTITY"
                    )
                content_type = (
                    headers.get("content_type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                if content_type != "application/zip":
                    raise RetryableTransportError("NON_ZIP_CONTENT_TYPE")
                expected_length = headers.get("content_length")
                try:
                    parsed_length = int(expected_length or "")
                except ValueError as error:
                    raise RetryableTransportError(
                        "CONTENT_LENGTH_INVALID"
                    ) from error
                if parsed_length <= 0:
                    raise RetryableTransportError("CONTENT_LENGTH_INVALID")
                while True:
                    try:
                        chunk = response.read(CHUNK_BYTES)
                    except Exception as error:
                        raise RetryableTransportError(
                            "TRANSPORT_BODY_READ_FAILED"
                        ) from error
                    if not chunk:
                        break
                    if bytes_written == 0 and not chunk.startswith(b"PK"):
                        raise RetryableTransportError(
                            "NON_ZIP_MAGIC_OR_ERROR_BODY"
                        )
                    target.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)
                target.flush()
                os.fsync(target.fileno())
            if bytes_written != int(expected_length):
                raise RetryableTransportError("CONTENT_LENGTH_MISMATCH")
            if bytes_written <= 0:
                raise RetryableTransportError("EMPTY_ARCHIVE")
            target.close()
            if destination.exists():
                raise FileExistsError(
                    "CANONICAL_ARCHIVE_DESTINATION_EXISTS_NO_OVERWRITE"
                )
            os.rename(part, destination)
            attempt = {
                "sequence_id": sequence_id,
                "attempt": attempt_number,
                "started_at": started,
                "completed_at": now_hong_kong(),
                "requested_url": url,
                "final_url": final_url,
                "status": response_status,
                "headers": headers,
                "bytes_written": bytes_written,
                "sha256": digest.hexdigest(),
                "outcome": "COMPLETE",
                "range_resume_used": False,
                "part_truncated_before_network": True,
            }
            attempts.append(attempt)
            if on_attempt is not None:
                try:
                    on_attempt(attempt)
                except Exception as error:
                    raise DurableLedgerError(
                        "ATTEMPT_LEDGER_PERSISTENCE_FAILED"
                    ) from error
            return (
                {
                    "requested_url": url,
                    "final_url": final_url,
                    "archive_path": str(destination),
                    "archive_bytes": bytes_written,
                    "archive_sha256": digest.hexdigest(),
                    "transport_attempts": attempt_number,
                    "response_headers": headers,
                },
                attempts,
            )
        except RetryableTransportError as error:
            target.close()
            attempt = {
                "sequence_id": sequence_id,
                "attempt": attempt_number,
                "started_at": started,
                "completed_at": now_hong_kong(),
                "requested_url": url,
                "final_url": final_url,
                "status": response_status,
                "headers": headers,
                "bytes_written": bytes_written,
                "outcome": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "range_resume_used": False,
                "part_truncated_before_network": True,
            }
            attempts.append(attempt)
            if on_attempt is not None:
                try:
                    on_attempt(attempt)
                except Exception as persistence_error:
                    raise DurableLedgerError(
                        "ATTEMPT_LEDGER_PERSISTENCE_FAILED"
                    ) from persistence_error
        except BaseException:
            target.close()
            raise
    raise RuntimeError(
        "TRANSPORT_ATTEMPTS_EXHAUSTED:" + canonical_json(attempts)
    )


def _normalized_member_name(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError("UNSAFE_ZIP_MEMBER_NAME")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("ZIP_PATH_TRAVERSAL_OR_NONCANONICAL_MEMBER")
    if any(":" in part for part in path.parts):
        raise ValueError("ZIP_DRIVE_MEMBER_FORBIDDEN")
    return str(path)


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    fixed = format(value, "f")
    if "." in fixed:
        fixed = fixed.rstrip("0").rstrip(".")
    return fixed


def _parse_timestamp_line(
    raw_line: bytes,
    values: list[Decimal],
    tokens: list[str],
) -> None:
    stripped = raw_line.lstrip(b" \t\r\n\v\f")
    if not stripped or stripped.startswith(b"#"):
        return
    token_bytes = stripped.split(None, 1)[0]
    if TIMESTAMP_PATTERN.fullmatch(token_bytes) is None:
        raise ValueError("INVALID_TIMESTAMP_TOKEN_GRAMMAR")
    try:
        token = token_bytes.decode("ascii")
        value = Decimal(token)
    except (UnicodeDecodeError, InvalidOperation) as error:
        raise ValueError("INVALID_TIMESTAMP_TOKEN") from error
    if not value.is_finite():
        raise ValueError("NONFINITE_TIMESTAMP")
    if values and value <= values[-1]:
        raise ValueError("TIMESTAMPS_NOT_STRICTLY_INCREASING")
    values.append(value)
    tokens.append(_decimal_text(value))


def stream_member_once(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    *,
    parse_timestamp_first_token: bool,
) -> dict[str, Any]:
    values: list[Decimal] = []
    tokens: list[str] = []
    raw_digest = hashlib.sha256()
    streamed = 0
    with archive.open(member, "r") as handle:
        if parse_timestamp_first_token:
            for raw_line in handle:
                streamed += len(raw_line)
                raw_digest.update(raw_line)
                _parse_timestamp_line(raw_line, values, tokens)
        else:
            while chunk := handle.read(CHUNK_BYTES):
                streamed += len(chunk)
                raw_digest.update(chunk)
    if streamed != member.file_size:
        raise ValueError("CRC_STREAM_SIZE_MISMATCH")
    result: dict[str, Any] = {
        "streamed_bytes": streamed,
        "raw_member_sha256": raw_digest.hexdigest(),
    }
    if parse_timestamp_first_token:
        if not values:
            raise ValueError("EMPTY_TIMESTAMP_SERIES")
        result["timestamps"] = {
            "count": len(values),
            "first": tokens[0],
            "last": tokens[-1],
            "canonical_token_ledger_sha256": hashlib.sha256(
                "".join(f"{token}\n" for token in tokens).encode("ascii")
            ).hexdigest(),
            "_values": values,
        }
    return result


def build_windows(
    rgb: dict[str, Any],
    depth: dict[str, Any],
    pose: dict[str, Any],
) -> list[dict[str, Any]]:
    start = max(
        rgb["_values"][0],
        depth["_values"][0],
        pose["_values"][0],
    )
    end = min(
        rgb["_values"][-1],
        depth["_values"][-1],
        pose["_values"][-1],
    )
    duration = end - start
    count = 0
    if duration > 0:
        count = max(
            0,
            int(
                (duration / WINDOW_SECONDS).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            ),
        )
    windows: list[dict[str, Any]] = []
    for rank in range(count):
        cursor = start + WINDOW_SECONDS * rank
        windows.append(
            {
                "window_rank": rank,
                "start": _decimal_text(cursor),
                "end": _decimal_text(cursor + WINDOW_SECONDS),
                "interval": "HALF_OPEN",
            }
        )
    return windows


def _relative_member_names(
    normalized_file_names: list[str],
) -> dict[str, str]:
    paths = [PurePosixPath(name) for name in normalized_file_names]
    strip_common = (
        bool(paths)
        and all(len(path.parts) >= 2 for path in paths)
        and len({path.parts[0] for path in paths}) == 1
    )
    relative_names: dict[str, str] = {}
    for path in paths:
        relative = (
            str(PurePosixPath(*path.parts[1:])) if strip_common else str(path)
        )
        _normalized_member_name(relative)
        relative_names[str(path)] = relative
    return relative_names


def inspect_archive(sequence_id: str, archive_path: Path) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    normalized_seen: set[str] = set()
    casefold_seen: set[str] = set()
    file_members: list[tuple[str, zipfile.ZipInfo]] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            normalized = _normalized_member_name(member.filename)
            folded = normalized.casefold()
            if normalized in normalized_seen or folded in casefold_seen:
                raise ValueError("DUPLICATE_NORMALIZED_ZIP_MEMBER")
            normalized_seen.add(normalized)
            casefold_seen.add(folded)
            is_dir = member.is_dir()
            inventory.append(
                {
                    "name": normalized,
                    "is_directory": is_dir,
                    "uncompressed_bytes": member.file_size,
                    "compressed_bytes": member.compress_size,
                    "crc32": f"{member.CRC:08x}",
                }
            )
            if not is_dir:
                file_members.append((normalized, member))
        relative_names = _relative_member_names(
            [name for name, _member in file_members]
        )
        required: dict[str, list[zipfile.ZipInfo]] = {
            "rgb.txt": [],
            "depth.txt": [],
            "groundtruth.txt": [],
        }
        rgb_files = 0
        depth_files = 0
        for normalized, member in file_members:
            relative = relative_names[normalized]
            if relative in required:
                required[relative].append(member)
            relative_parts = PurePosixPath(relative).parts
            if len(relative_parts) >= 2 and relative_parts[0] == "rgb":
                rgb_files += 1
            if len(relative_parts) >= 2 and relative_parts[0] == "depth":
                depth_files += 1
        for name, matches in required.items():
            if len(matches) != 1:
                raise ValueError(
                    f"REQUIRED_MEMBER_CARDINALITY_{name}:{len(matches)}"
                )
        if rgb_files < 1 or depth_files < 1:
            raise ValueError("RGB_OR_DEPTH_MEMBER_PREFIX_EMPTY")
        text_by_member = {
            required["rgb.txt"][0].filename: "rgb",
            required["depth.txt"][0].filename: "depth",
            required["groundtruth.txt"][0].filename: "pose",
        }
        parsed: dict[str, dict[str, Any]] = {}
        stream_receipts: dict[str, dict[str, Any]] = {}
        crc_uncompressed_bytes = 0
        for normalized, member in file_members:
            stream = stream_member_once(
                archive,
                member,
                parse_timestamp_first_token=(
                    member.filename in text_by_member
                ),
            )
            stream_receipts[normalized] = {
                key: value
                for key, value in stream.items()
                if key != "timestamps"
            }
            crc_uncompressed_bytes += int(stream["streamed_bytes"])
            if "timestamps" in stream:
                parsed[text_by_member[member.filename]] = stream["timestamps"]
    windows = build_windows(parsed["rgb"], parsed["depth"], parsed["pose"])
    timestamp_summary = {
        name: {key: value for key, value in detail.items() if key != "_values"}
        for name, detail in parsed.items()
    }
    inventory_hash = hashlib.sha256(
        canonical_json(inventory).encode("utf-8")
    ).hexdigest()
    return {
        "sequence_id": sequence_id,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "member_count": len(inventory),
        "file_member_count": len(file_members),
        "member_inventory": inventory,
        "member_inventory_sha256": inventory_hash,
        "member_stream_receipts": stream_receipts,
        "member_stream_receipts_sha256": hashlib.sha256(
            canonical_json(stream_receipts).encode("utf-8")
        ).hexdigest(),
        "crc_only_stream": {
            "status": "PASS",
            "file_members_streamed": len(file_members),
            "uncompressed_bytes_streamed": crc_uncompressed_bytes,
            "decode_operations": 0,
            "persisted_extracted_bytes": 0,
            "sample_or_inspection_operations": 0,
        },
        "required_member_paths": {
            name: _normalized_member_name(matches[0].filename)
            for name, matches in required.items()
        },
        "relative_root_rule": (
            "STRIPPED_ONE_COMMON_TOP_LEVEL_COMPONENT"
            if any(
                relative_names[name] != name
                for name, _member in file_members
            )
            else "NO_TOP_LEVEL_COMPONENT_STRIPPED"
        ),
        "rgb_file_members": rgb_files,
        "depth_file_members": depth_files,
        "timestamps": timestamp_summary,
        "pose_tokens_parsed_after_first": 0,
        "windows": windows,
        "window_count": len(windows),
        "window_denominator_sha256": hashlib.sha256(
            canonical_json(windows).encode("utf-8")
        ).hexdigest(),
        "timestamp_firewall": "PASS_FIRST_TOKEN_ONLY",
        "status": "EVALUABLE_ARCHIVE_AUTHORITY",
    }


def _sequence_inventory_path(output: Path, sequence_id: str) -> Path:
    return output / "sequences" / f"{sequence_id}.json"


def build_receipt(
    repo_root: Path,
    claim: dict[str, Any],
    sequence_results: list[dict[str, Any]],
    attempt_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = canonical_paths(repo_root)
    evaluable = [
        result
        for result in sequence_results
        if result["status"] == "EVALUABLE_ARCHIVE_AUTHORITY"
    ]
    with_windows = [
        result for result in evaluable if result.get("window_count", 0) >= 1
    ]
    all_six = len(evaluable) == len(SEQUENCE_IDS)
    gate_pass = all_six and len(with_windows) >= 2
    status_short = _git(["status", "--short"], repo_root).splitlines()
    denominator = [
        {
            "rank": rank,
            "sequence_id": sequence_id,
            "status": next(
                result["status"]
                for result in sequence_results
                if result["sequence_id"] == sequence_id
            ),
            "window_count": next(
                int(result.get("window_count", 0))
                for result in sequence_results
                if result["sequence_id"] == sequence_id
            ),
        }
        for rank, sequence_id in enumerate(SEQUENCE_IDS, start=1)
    ]
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_id": "RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1",
        "created_at": now_hong_kong(),
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "r0_contract_result_sha256": R0_CONTRACT_RESULT_SHA256,
        "metadata_authority_r3_receipt_sha256": R3_RECEIPT_SHA256,
        "cohort_identity_sha256": COHORT_IDENTITY_SHA256,
        "implementation_lock_sha256": sha256_file(
            paths["implementation_lock"]
        ),
        "run_claim_sha256": sha256_file(paths["run_claim"]),
        "repo": {
            "head": _git(["rev-parse", "HEAD"], repo_root),
            "branch": _git(["branch", "--show-current"], repo_root),
            "dirty": bool(status_short),
            "status_short": status_short,
        },
        "environment": environment_manifest(),
        "disclosed_pre_r1_head_observations": {
            "request_count": 6,
            "response_body_bytes": 0,
            "reported_content_length_total_bytes": 2262988443,
            "authority": (
                "NON_AUTHORITATIVE_TRANSPORT_DISCOVERY_"
                "EXCLUDED_FROM_ALL_GATES"
            ),
            "used_for_selection_or_gate": False,
        },
        "claim": claim,
        "sequence_ids_in_rank_order": list(SEQUENCE_IDS),
        "sequence_results": sequence_results,
        "transport_attempt_count": len(attempt_ledger),
        "transport_attempt_ledger_sha256": sha256_file(
            paths["attempt_ledger"]
        ),
        "window_denominator": denominator,
        "window_denominator_sha256": hashlib.sha256(
            canonical_json(denominator).encode("utf-8")
        ).hexdigest(),
        "evaluable_sequence_count": len(evaluable),
        "sequences_with_windows_count": len(with_windows),
        "failed_or_zero_window_units_retained": len(denominator) == 6,
        "read_firewall": {
            "rgb_depth_decode_operations": 0,
            "pose_tokens_parsed_after_first": 0,
            "static_map_reads": 0,
            "legacy_outcome_reads": 0,
            "phase_b_metric_reads_or_computations": 0,
        },
        "gate_pass": gate_pass,
        "terminal_state": PASS_TERMINAL if gate_pass else FAIL_TERMINAL,
        "b1_metric_protocol_may_be_frozen": gate_pass,
        "phase_b_metrics_authorized": False,
        "replay_android_human_safety_production_authorized": False,
    }


def run_b0(
    repo_root: Path,
    claim: dict[str, Any],
    *,
    opener: Callable[[str, float], Any] = _default_open,
) -> dict[str, Any]:
    paths = canonical_paths(repo_root)
    results: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []

    def persist_attempt(attempt: dict[str, Any]) -> None:
        all_attempts.append(attempt)
        write_json(paths["attempt_ledger"], all_attempts)

    for rank, sequence_id in enumerate(SEQUENCE_IDS, start=1):
        archive_path = paths["archive_root"] / f"{sequence_id}.zip"
        try:
            transport, _attempts = acquire_archive(
                sequence_id,
                archive_path,
                opener=opener,
                on_attempt=persist_attempt,
            )
            inspected = inspect_archive(sequence_id, archive_path)
            inspected["rank"] = rank
            inspected["transport"] = transport
            result = inspected
        except Exception as error:
            result = {
                "rank": rank,
                "sequence_id": sequence_id,
                "status": "NOT_EVALUABLE_ARCHIVE_AUTHORITY",
                "window_count": 0,
                "error_type": type(error).__name__,
                "error": str(error),
                "replacement_used": False,
            }
        results.append(result)
        write_json(
            _sequence_inventory_path(paths["output"], sequence_id),
            result,
        )
        write_json(paths["attempt_ledger"], all_attempts)
    receipt = build_receipt(repo_root, claim, results, all_attempts)
    write_json(paths["receipt"], receipt)
    return receipt
