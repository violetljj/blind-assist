from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
import zipfile


SEQUENCES = (
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
DESIGN_SHA = (
    "396444305bae01eb5a8e95a92044cbea9aa7084c605993c789ecb4f47e234e74"
)
PREREG_SHA = (
    "f50cf66c46fe33aa3c1e60fa3c25cb120389eafdbad92f4d0a9df22d7cc68da2"
)
R0_SHA = "7c5aa8c66b6d99803b4ae2945dfcf95fe7c7bffc7919423df9b68a03fdf1f734"
R3_SHA = "05a283b84f62bee000447bb567eadd63b424afaa9d81f5f0d83d36a9ed02489b"
COHORT_SHA = (
    "513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e"
)
PASS_TERMINAL = (
    "PHASE_B_B0_R1_INVENTORY_PASS_B1_METRIC_PROTOCOL_MAY_BE_FROZEN"
)
FAIL_TERMINAL = (
    "HOLD_PHASE_B_B0_R1_NOT_EVALUABLE_NO_REPLACEMENT_NO_RERUN"
)
TIMESTAMP_RE = re.compile(
    rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
    rb"(?:[eE][+-]?[0-9]+)?"
)
CHUNK = 1024 * 1024
TEN = Decimal("10")
RECEIPT_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "created_at",
    "design_lock_sha256",
    "preregistration_sha256",
    "r0_contract_result_sha256",
    "metadata_authority_r3_receipt_sha256",
    "cohort_identity_sha256",
    "implementation_lock_sha256",
    "run_claim_sha256",
    "repo",
    "environment",
    "disclosed_pre_r1_head_observations",
    "claim",
    "sequence_ids_in_rank_order",
    "sequence_results",
    "transport_attempt_count",
    "transport_attempt_ledger_sha256",
    "window_denominator",
    "window_denominator_sha256",
    "evaluable_sequence_count",
    "sequences_with_windows_count",
    "failed_or_zero_window_units_retained",
    "read_firewall",
    "gate_pass",
    "terminal_state",
    "b1_metric_protocol_may_be_frozen",
    "phase_b_metrics_authorized",
    "replay_android_human_safety_production_authorized",
}
CLAIM_KEYS = {
    "schema_version",
    "protocol_id",
    "claimed_at",
    "canonical_archive_root",
    "canonical_output",
    "canonical_run_claim",
    "command",
    "maximum_run_claims",
    "exclusive_create",
    "survives_failure_interrupt_and_success",
    "network_operations_before_claim",
    "pre_r1_head_disclosure",
    "design_lock_sha256",
    "preregistration_sha256",
    "metadata_authority_r3_receipt_sha256",
    "cohort_identity_sha256",
    "implementation_lock_sha256",
}
RETRYABLE_ERROR_CODES = {
    "TRANSPORT_OPEN_FAILED",
    "REDIRECT_OR_URL_IDENTITY_MISMATCH",
    "CONTENT_ENCODING_NOT_IDENTITY",
    "NON_ZIP_CONTENT_TYPE",
    "CONTENT_LENGTH_INVALID",
    "TRANSPORT_BODY_READ_FAILED",
    "NON_ZIP_MAGIC_OR_ERROR_BODY",
    "CONTENT_LENGTH_MISMATCH",
    "EMPTY_ARCHIVE",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(root: Path) -> dict[str, Path]:
    module = root / "scripts" / "research" / "egomotion_compensated_looming"
    output = (
        root
        / "artifacts.local"
        / "evidence"
        / "rcle_phase_b_bonn_b0_r1"
        / "formal_entry_b0_r1"
    )
    archive_root = (
        root
        / "artifacts.local"
        / "datasets"
        / "rcle_phase_b_bonn_b0_r1"
        / "archives"
    )
    return {
        "module": module,
        "archive_root": archive_root,
        "output": output,
        "claim": output / "run_claim.json",
        "receipt": output / "receipt.json",
        "ledger": output / "transport_attempt_ledger.json",
        "lock": module
        / "rcle_phase_b_bonn_b0_r1"
        / "RCLE_PHASE_B_BONN_B0_R1_IMPLEMENTATION_LOCK.json",
    }


def _normalize(name: str) -> str:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError("VALIDATOR_UNSAFE_MEMBER")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("VALIDATOR_TRAVERSAL")
    if any(":" in part for part in path.parts):
        raise ValueError("VALIDATOR_DRIVE")
    return str(path)


def _relative(names: list[str]) -> dict[str, str]:
    paths = [PurePosixPath(name) for name in names]
    strip = (
        bool(paths)
        and all(len(path.parts) >= 2 for path in paths)
        and len({path.parts[0] for path in paths}) == 1
    )
    result: dict[str, str] = {}
    for path in paths:
        value = (
            str(PurePosixPath(*path.parts[1:])) if strip else str(path)
        )
        _normalize(value)
        result[str(path)] = value
    return result


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    fixed = format(value, "f")
    return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed


def _stream_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    parse_timestamp: bool,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    values: list[Decimal] = []
    tokens: list[str] = []
    with archive.open(member, "r") as handle:
        if parse_timestamp:
            for raw_line in handle:
                digest.update(raw_line)
                byte_count += len(raw_line)
                stripped = raw_line.lstrip(b" \t\r\n\v\f")
                if not stripped or stripped.startswith(b"#"):
                    continue
                token_bytes = stripped.split(None, 1)[0]
                if TIMESTAMP_RE.fullmatch(token_bytes) is None:
                    raise ValueError("VALIDATOR_TIMESTAMP_GRAMMAR")
                try:
                    value = Decimal(token_bytes.decode("ascii"))
                except (UnicodeDecodeError, InvalidOperation) as error:
                    raise ValueError("VALIDATOR_TIMESTAMP_PARSE") from error
                if not value.is_finite():
                    raise ValueError("VALIDATOR_TIMESTAMP_NONFINITE")
                if values and value <= values[-1]:
                    raise ValueError("VALIDATOR_TIMESTAMP_ORDER")
                values.append(value)
                tokens.append(_decimal_text(value))
        else:
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
                byte_count += len(chunk)
    if byte_count != member.file_size:
        raise ValueError("VALIDATOR_STREAM_SIZE")
    result: dict[str, Any] = {
        "streamed_bytes": byte_count,
        "raw_member_sha256": digest.hexdigest(),
    }
    if parse_timestamp:
        if not values:
            raise ValueError("VALIDATOR_EMPTY_TIMESTAMPS")
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


def _windows(series: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    start = max(
        series["rgb"]["_values"][0],
        series["depth"]["_values"][0],
        series["pose"]["_values"][0],
    )
    end = min(
        series["rgb"]["_values"][-1],
        series["depth"]["_values"][-1],
        series["pose"]["_values"][-1],
    )
    duration = end - start
    count = (
        max(
            0,
            int(
                (duration / TEN).to_integral_value(rounding=ROUND_FLOOR)
            ),
        )
        if duration > 0
        else 0
    )
    return [
        {
            "window_rank": rank,
            "start": _decimal_text(start + TEN * rank),
            "end": _decimal_text(start + TEN * (rank + 1)),
            "interval": "HALF_OPEN",
        }
        for rank in range(count)
    ]


def _inspect(sequence: str, path: Path) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    members: list[tuple[str, zipfile.ZipInfo]] = []
    seen: set[str] = set()
    folded: set[str] = set()
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.infolist():
            name = _normalize(member.filename)
            if name in seen or name.casefold() in folded:
                raise ValueError("VALIDATOR_DUPLICATE_MEMBER")
            seen.add(name)
            folded.add(name.casefold())
            inventory.append(
                {
                    "name": name,
                    "is_directory": member.is_dir(),
                    "uncompressed_bytes": member.file_size,
                    "compressed_bytes": member.compress_size,
                    "crc32": f"{member.CRC:08x}",
                }
            )
            if not member.is_dir():
                members.append((name, member))
        relatives = _relative([name for name, _member in members])
        required: dict[str, list[zipfile.ZipInfo]] = {
            "rgb.txt": [],
            "depth.txt": [],
            "groundtruth.txt": [],
        }
        rgb_files = 0
        depth_files = 0
        for name, member in members:
            relative = relatives[name]
            if relative in required:
                required[relative].append(member)
            parts = PurePosixPath(relative).parts
            rgb_files += int(len(parts) >= 2 and parts[0] == "rgb")
            depth_files += int(len(parts) >= 2 and parts[0] == "depth")
        if any(len(matches) != 1 for matches in required.values()):
            raise ValueError("VALIDATOR_REQUIRED_MEMBER_CARDINALITY")
        if rgb_files < 1 or depth_files < 1:
            raise ValueError("VALIDATOR_PREFIX_EMPTY")
        text_roles = {
            required["rgb.txt"][0].filename: "rgb",
            required["depth.txt"][0].filename: "depth",
            required["groundtruth.txt"][0].filename: "pose",
        }
        streams: dict[str, dict[str, Any]] = {}
        series: dict[str, dict[str, Any]] = {}
        total = 0
        for name, member in members:
            stream = _stream_member(
                archive, member, member.filename in text_roles
            )
            streams[name] = {
                key: value
                for key, value in stream.items()
                if key != "timestamps"
            }
            total += int(stream["streamed_bytes"])
            if "timestamps" in stream:
                series[text_roles[member.filename]] = stream["timestamps"]
    windows = _windows(series)
    timestamp_summary = {
        role: {
            key: value
            for key, value in detail.items()
            if key != "_values"
        }
        for role, detail in series.items()
    }
    return {
        "sequence_id": sequence,
        "archive_path": str(path),
        "archive_sha256": _sha(path),
        "archive_bytes": path.stat().st_size,
        "member_count": len(inventory),
        "file_member_count": len(members),
        "member_inventory": inventory,
        "member_inventory_sha256": hashlib.sha256(
            _canonical_json(inventory).encode("utf-8")
        ).hexdigest(),
        "member_stream_receipts": streams,
        "member_stream_receipts_sha256": hashlib.sha256(
            _canonical_json(streams).encode("utf-8")
        ).hexdigest(),
        "crc_only_stream": {
            "status": "PASS",
            "file_members_streamed": len(members),
            "uncompressed_bytes_streamed": total,
            "decode_operations": 0,
            "persisted_extracted_bytes": 0,
            "sample_or_inspection_operations": 0,
        },
        "required_member_paths": {
            key: _normalize(value[0].filename)
            for key, value in required.items()
        },
        "relative_root_rule": (
            "STRIPPED_ONE_COMMON_TOP_LEVEL_COMPONENT"
            if any(relatives[name] != name for name, _member in members)
            else "NO_TOP_LEVEL_COMPONENT_STRIPPED"
        ),
        "rgb_file_members": rgb_files,
        "depth_file_members": depth_files,
        "timestamps": timestamp_summary,
        "pose_tokens_parsed_after_first": 0,
        "windows": windows,
        "window_count": len(windows),
        "window_denominator_sha256": hashlib.sha256(
            _canonical_json(windows).encode("utf-8")
        ).hexdigest(),
        "timestamp_firewall": "PASS_FIRST_TOKEN_ONLY",
        "status": "EVALUABLE_ARCHIVE_AUTHORITY",
    }


def _comparison(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "sequence_id",
        "archive_path",
        "archive_sha256",
        "archive_bytes",
        "member_count",
        "file_member_count",
        "member_inventory",
        "member_inventory_sha256",
        "member_stream_receipts",
        "member_stream_receipts_sha256",
        "crc_only_stream",
        "required_member_paths",
        "relative_root_rule",
        "rgb_file_members",
        "depth_file_members",
        "timestamps",
        "pose_tokens_parsed_after_first",
        "windows",
        "window_count",
        "window_denominator_sha256",
        "timestamp_firewall",
        "status",
    )
    return {key: value.get(key) for key in keys}


def _failed_attempt_contract(
    attempt: dict[str, Any], expected_url: str
) -> bool:
    error = attempt.get("error")
    status = attempt.get("status")
    final_url = attempt.get("final_url")
    headers = attempt.get("headers")
    bytes_written = attempt.get("bytes_written")
    if (
        attempt.get("error_type") != "RetryableTransportError"
        or not isinstance(error, str)
        or not error
        or not isinstance(bytes_written, int)
        or bytes_written < 0
        or not isinstance(headers, dict)
    ):
        return False
    if error.startswith("HTTP_STATUS_"):
        suffix = error.removeprefix("HTTP_STATUS_")
        return (
            suffix.isdigit()
            and status == int(suffix)
            and status != 200
            and final_url == expected_url
        )
    if error not in RETRYABLE_ERROR_CODES:
        return False
    if error == "TRANSPORT_OPEN_FAILED":
        return status is None and final_url is None and headers == {}
    if status != 200 or final_url is None:
        return False
    if error == "REDIRECT_OR_URL_IDENTITY_MISMATCH":
        return final_url != expected_url
    return final_url == expected_url


def validate(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    controls = {
        relative: _sha(root / relative)
        for relative in sorted(lock["control_source_manifest"])
    }
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    claim = json.loads(paths["claim"].read_text(encoding="utf-8"))
    ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
    results = receipt.get("sequence_results")
    denominator = receipt.get("window_denominator")
    if not isinstance(results, list) or len(results) != 6:
        raise ValueError("VALIDATOR_SEQUENCE_RESULT_CARDINALITY")
    if not isinstance(denominator, list) or len(denominator) != 6:
        raise ValueError("VALIDATOR_DENOMINATOR_CARDINALITY")
    if not isinstance(ledger, list) or not ledger:
        raise ValueError("VALIDATOR_ATTEMPT_LEDGER_EMPTY")
    expected_disclosure = {
        "request_count": 6,
        "response_body_bytes": 0,
        "reported_content_length_total_bytes": 2262988443,
        "authority": (
            "NON_AUTHORITATIVE_TRANSPORT_DISCOVERY_EXCLUDED_FROM_ALL_GATES"
        ),
    }
    expected_receipt_disclosure = {
        **expected_disclosure,
        "used_for_selection_or_gate": False,
    }
    expected_claim_paths = {
        "canonical_archive_root": str(paths["archive_root"]),
        "canonical_output": str(paths["output"]),
        "canonical_run_claim": str(paths["claim"]),
    }
    checks: dict[str, bool] = {
        "receipt_exact_top_level_keys": set(receipt)
        == RECEIPT_TOP_LEVEL_KEYS,
        "lock_controls": lock.get("control_source_manifest") == controls,
        "lock_authority": lock.get("canonical_execution_authorized") is True,
        "receipt_schema": receipt.get("schema_version")
        == "rcle.phase_b.bonn_b0_r1.receipt.v1",
        "receipt_protocol": receipt.get("protocol_id")
        == "RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1",
        "receipt_hashes": (
            receipt.get("design_lock_sha256") == DESIGN_SHA
            and receipt.get("preregistration_sha256") == PREREG_SHA
            and receipt.get("r0_contract_result_sha256") == R0_SHA
            and receipt.get("metadata_authority_r3_receipt_sha256")
            == R3_SHA
            and receipt.get("cohort_identity_sha256") == COHORT_SHA
            and receipt.get("implementation_lock_sha256")
            == _sha(paths["lock"])
        ),
        "claim_embedded": receipt.get("claim") == claim,
        "claim_exact_keys": set(claim) == CLAIM_KEYS,
        "claim_schema": claim.get("schema_version")
        == "rcle.phase_b.bonn_b0_r1.run_claim.v1",
        "claim_protocol": claim.get("protocol_id")
        == "RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1",
        "claim_hashes": (
            claim.get("design_lock_sha256") == DESIGN_SHA
            and claim.get("preregistration_sha256") == PREREG_SHA
            and claim.get("metadata_authority_r3_receipt_sha256") == R3_SHA
            and claim.get("cohort_identity_sha256") == COHORT_SHA
            and claim.get("implementation_lock_sha256")
            == _sha(paths["lock"])
        ),
        "claim_paths": all(
            claim.get(key) == value
            for key, value in expected_claim_paths.items()
        ),
        "claim_command_time": (
            isinstance(claim.get("command"), list)
            and bool(claim["command"])
            and isinstance(claim.get("claimed_at"), str)
            and bool(claim["claimed_at"])
        ),
        "claim_disclosure": claim.get("pre_r1_head_disclosure")
        == expected_disclosure,
        "receipt_disclosure": receipt.get(
            "disclosed_pre_r1_head_observations"
        )
        == expected_receipt_disclosure,
        "claim_one_run": (
            claim.get("maximum_run_claims") == 1
            and claim.get("network_operations_before_claim") == 0
            and claim.get("exclusive_create") is True
            and claim.get("survives_failure_interrupt_and_success") is True
        ),
        "claim_sha": receipt.get("run_claim_sha256") == _sha(paths["claim"]),
        "sequence_order": receipt.get("sequence_ids_in_rank_order")
        == list(SEQUENCES),
        "ledger_sha": receipt.get("transport_attempt_ledger_sha256")
        == _sha(paths["ledger"]),
        "ledger_count": receipt.get("transport_attempt_count") == len(ledger),
        "firewall": receipt.get("read_firewall")
        == {
            "rgb_depth_decode_operations": 0,
            "pose_tokens_parsed_after_first": 0,
            "static_map_reads": 0,
            "legacy_outcome_reads": 0,
            "phase_b_metric_reads_or_computations": 0,
        },
        "metrics_closed": (
            receipt.get("phase_b_metrics_authorized") is False
            and receipt.get(
                "replay_android_human_safety_production_authorized"
            )
            is False
        ),
        "failed_or_zero_retained": receipt.get(
            "failed_or_zero_window_units_retained"
        )
        is True,
    }
    attempts_by_sequence: dict[str, list[dict[str, Any]]] = {
        sequence: [] for sequence in SEQUENCES
    }
    last_sequence_rank = 0
    for attempt in ledger:
        sequence = attempt.get("sequence_id")
        if sequence not in attempts_by_sequence:
            checks["ledger_unknown_sequence"] = False
            continue
        sequence_rank = SEQUENCES.index(sequence) + 1
        checks[f"ledger_global_order_{len(attempts_by_sequence[sequence])}"] = (
            sequence_rank >= last_sequence_rank
        )
        last_sequence_rank = sequence_rank
        checks[f"ledger_no_attempt_after_complete_{sequence}"] = not any(
            prior.get("outcome") == "COMPLETE"
            for prior in attempts_by_sequence[sequence]
        )
        attempts_by_sequence[sequence].append(attempt)
        expected_url = URL_TEMPLATE.format(sequence_id=sequence)
        checks[f"ledger_contract_{sequence}_{len(attempts_by_sequence[sequence])}"] = (
            attempt.get("requested_url") == expected_url
            and attempt.get("attempt")
            == len(attempts_by_sequence[sequence])
            and 1 <= int(attempt.get("attempt", 0)) <= 3
            and attempt.get("range_resume_used") is False
            and attempt.get("part_truncated_before_network") is True
            and attempt.get("outcome") in ("COMPLETE", "FAILED")
            and (
                attempt.get("final_url") == expected_url
                if attempt.get("outcome") == "COMPLETE"
                else True
            )
            and (
                _failed_attempt_contract(attempt, expected_url)
                if attempt.get("outcome") == "FAILED"
                else True
            )
        )
    recomputed_evaluable = 0
    recomputed_windows = 0
    expected_denominator: list[dict[str, Any]] = []
    for rank, (sequence, result) in enumerate(
        zip(SEQUENCES, results, strict=True), start=1
    ):
        checks[f"result_identity_{sequence}"] = (
            result.get("rank") == rank
            and result.get("sequence_id") == sequence
            and result.get("status")
            in (
                "EVALUABLE_ARCHIVE_AUTHORITY",
                "NOT_EVALUABLE_ARCHIVE_AUTHORITY",
            )
        )
        stored_path = paths["output"] / "sequences" / f"{sequence}.json"
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        checks[f"stored_result_{sequence}"] = stored == result
        archive = paths["archive_root"] / f"{sequence}.zip"
        sequence_attempts = attempts_by_sequence[sequence]
        complete = [
            attempt
            for attempt in sequence_attempts
            if attempt.get("outcome") == "COMPLETE"
        ]
        failed = [
            attempt
            for attempt in sequence_attempts
            if attempt.get("outcome") == "FAILED"
        ]
        checks[f"attempt_total_bound_{sequence}"] = (
            1 <= len(sequence_attempts) <= 3
        )
        if archive.exists():
            checks[f"archive_attempt_shape_{sequence}"] = (
                len(complete) == 1
                and sequence_attempts[-1].get("outcome") == "COMPLETE"
                and len(failed) == len(sequence_attempts) - 1
            )
            completed = complete[0]
            headers = completed.get("headers", {})
            expected_url = URL_TEMPLATE.format(sequence_id=sequence)
            content_type = (
                str(headers.get("content_type", ""))
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            content_encoding = str(
                headers.get("content_encoding", "identity")
            ).lower()
            try:
                content_length = int(headers.get("content_length", ""))
            except (TypeError, ValueError):
                content_length = -1
            actual_bytes = archive.stat().st_size
            actual_sha = _sha(archive)
            checks[f"complete_transport_{sequence}"] = (
                completed.get("status") == 200
                and completed.get("requested_url") == expected_url
                and completed.get("final_url") == expected_url
                and content_type == "application/zip"
                and content_encoding == "identity"
                and content_length == actual_bytes
                and completed.get("bytes_written") == actual_bytes
                and completed.get("sha256") == actual_sha
            )
        else:
            checks[f"no_archive_attempt_shape_{sequence}"] = (
                len(complete) == 0
                and len(failed) == 3
                and len(sequence_attempts) == 3
            )
        if result.get("status") == "EVALUABLE_ARCHIVE_AUTHORITY":
            checks[f"transport_complete_{sequence}"] = len(complete) == 1
            recomputed = _inspect(sequence, archive)
            checks[f"recorded_inventory_hash_{sequence}"] = (
                result.get("member_inventory_sha256")
                == hashlib.sha256(
                    _canonical_json(
                        result.get("member_inventory")
                    ).encode("utf-8")
                ).hexdigest()
            )
            checks[f"archive_recompute_{sequence}"] = (
                _comparison(recomputed) == _comparison(result)
            )
            completed = complete[0]
            checks[f"recorded_transport_{sequence}"] = result.get(
                "transport"
            ) == {
                "requested_url": completed["requested_url"],
                "final_url": completed["final_url"],
                "archive_path": str(archive),
                "archive_bytes": completed["bytes_written"],
                "archive_sha256": completed["sha256"],
                "transport_attempts": completed["attempt"],
                "response_headers": completed["headers"],
            }
            recomputed_evaluable += 1
            recomputed_windows += int(recomputed["window_count"] >= 1)
        else:
            checks[f"failed_result_{sequence}"] = (
                result.get("window_count") == 0
                and result.get("replacement_used") is False
                and isinstance(result.get("error"), str)
                and bool(result["error"])
            )
            if archive.exists():
                try:
                    _inspect(sequence, archive)
                except Exception:
                    pass
                else:
                    checks[f"failed_archive_must_remain_invalid_{sequence}"] = (
                        False
                    )
        expected_denominator.append(
            {
                "rank": rank,
                "sequence_id": sequence,
                "status": result["status"],
                "window_count": int(result.get("window_count", 0)),
            }
        )
    checks["denominator_content"] = denominator == expected_denominator
    checks["denominator_sha"] = receipt.get(
        "window_denominator_sha256"
    ) == hashlib.sha256(
        _canonical_json(expected_denominator).encode("utf-8")
    ).hexdigest()
    gate = recomputed_evaluable == 6 and recomputed_windows >= 2
    checks["evaluable_count"] = receipt.get(
        "evaluable_sequence_count"
    ) == recomputed_evaluable
    checks["window_sequence_count"] = receipt.get(
        "sequences_with_windows_count"
    ) == recomputed_windows
    checks["gate"] = receipt.get("gate_pass") is gate
    checks["terminal"] = receipt.get("terminal_state") == (
        PASS_TERMINAL if gate else FAIL_TERMINAL
    )
    checks["b1"] = receipt.get("b1_metric_protocol_may_be_frozen") is gate
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            "B0_R1_INDEPENDENT_VALIDATION_MISMATCH:" + ",".join(failed)
        )
    return {
        "schema_version": "rcle.phase_b.bonn_b0_r1.validation.v1",
        "status": "VALID",
        "receipt_sha256": _sha(paths["receipt"]),
        "run_claim_sha256": _sha(paths["claim"]),
        "implementation_lock_sha256": _sha(paths["lock"]),
        "sequence_count": 6,
        "evaluable_sequence_count": recomputed_evaluable,
        "sequences_with_windows_count": recomputed_windows,
        "gate_pass": gate,
        "terminal_state": PASS_TERMINAL if gate else FAIL_TERMINAL,
        "b1_metric_protocol_may_be_frozen": gate,
        "phase_b_metrics_authorized": False,
    }
