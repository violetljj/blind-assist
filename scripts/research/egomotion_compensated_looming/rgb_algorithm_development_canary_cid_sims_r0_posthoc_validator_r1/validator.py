from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Iterable, Sequence
import zipfile
import zlib


PROTOCOL_ID = (
    "RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_POSTHOC_VALIDATOR_R1"
)
R0_PROTOCOL_ID = (
    "RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_CID_SIMS_FLOOR3_1"
)


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digests(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xFFFFFFFF


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        value = json.loads(line, parse_float=Decimal)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL_OBJECT_REQUIRED:{index}")
        rows.append(value)
    return rows


def decimal_value(value: Any) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise InvalidOperation
    return Decimal(str(value))


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, Decimal))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def float_equal(actual: Any, expected: float) -> bool:
    return finite_number(actual) and float(actual).hex() == float(expected).hex()


def expected_window_timestamps(
    names: Iterable[str],
    windows: Sequence[dict[str, Any]],
    sequence_id: str,
    maximum_pair_dt: Decimal = Decimal("0.100"),
) -> tuple[dict[int, list[Decimal]], list[str]]:
    errors: list[str] = []
    color: dict[Decimal, str] = {}
    depth: set[Decimal] = set()
    for raw in names:
        name = PurePosixPath(raw)
        if len(name.parts) != 3 or name.parts[0] != sequence_id:
            continue
        if name.suffix.lower() != ".png":
            continue
        try:
            timestamp = Decimal(name.stem)
        except InvalidOperation:
            continue
        if name.parts[1] == "color":
            if timestamp in color:
                errors.append(f"ZIP_COLOR_TIMESTAMP_DUPLICATE:{timestamp}")
            color[timestamp] = name.as_posix()
        elif name.parts[1] == "depth":
            if timestamp in depth:
                errors.append(f"ZIP_DEPTH_TIMESTAMP_DUPLICATE:{timestamp}")
            depth.add(timestamp)
    shared = sorted(set(color) & depth)
    result: dict[int, list[Decimal]] = {}
    assigned: set[Decimal] = set()
    for window in windows:
        index = int(window["window_index"])
        start = Decimal(window["start_timestamp_s"])
        end = Decimal(window["end_timestamp_s"])
        rows = [timestamp for timestamp in shared if start <= timestamp < end]
        if len(rows) != int(window["expected_frame_count"]):
            errors.append(f"ZIP_WINDOW_FRAME_COUNT:{index}:{len(rows)}")
        if len(rows) != len(set(rows)):
            errors.append(f"ZIP_WINDOW_DUPLICATE:{index}")
        if any(timestamp in assigned for timestamp in rows):
            errors.append(f"ZIP_CROSS_WINDOW_DUPLICATE:{index}")
        assigned.update(rows)
        pairs = list(zip(rows, rows[1:]))
        if len(pairs) != int(window["expected_pair_count"]):
            errors.append(f"ZIP_WINDOW_PAIR_COUNT:{index}:{len(pairs)}")
        for pair_index, (previous, current) in enumerate(pairs):
            dt = current - previous
            if not (Decimal("0") < dt <= maximum_pair_dt):
                errors.append(f"ZIP_PAIR_DT:{index}:{pair_index}:{dt}")
        result[index] = rows
    expected_total = sum(int(window["expected_frame_count"]) for window in windows)
    if sum(len(rows) for rows in result.values()) != expected_total:
        errors.append("ZIP_SELECTED_MEMBER_TOTAL")
    return result, errors


def validate_manifest_inventory(
    manifest: dict[str, Any],
    windows: Sequence[dict[str, Any]],
    expected: dict[int, list[Decimal]],
    zip_infos: dict[str, zipfile.ZipInfo],
    cache_root: Path,
    sequence_id: str,
    control_member: str,
    expected_total_file_count: int,
    expected_archive_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "rcle.rgb_algorithm_cache.v1":
        errors.append("CACHE_MANIFEST_SCHEMA")
    if manifest.get("protocol_id") != R0_PROTOCOL_ID:
        errors.append("CACHE_MANIFEST_PROTOCOL")
    if manifest.get("archive_sha256") != expected_archive_sha256:
        errors.append("CACHE_MANIFEST_ARCHIVE_SHA256")
    payload = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if manifest.get("payload_sha256") != canonical_sha(payload):
        errors.append("CACHE_MANIFEST_SELF_HASH")
    members = manifest.get("members")
    if not isinstance(members, list):
        return ["CACHE_MEMBERS_NOT_LIST"]
    expected_flat = sorted(
        (timestamp, index)
        for index, timestamps in expected.items()
        for timestamp in timestamps
    )
    if len(members) != len(expected_flat):
        errors.append(f"CACHE_MEMBER_COUNT:{len(members)}")
    seen_paths: set[str] = set()
    seen_timestamps: set[Decimal] = set()
    for ordinal, expected_item in enumerate(expected_flat):
        if ordinal >= len(members):
            break
        timestamp, window_index = expected_item
        item = members[ordinal]
        if not isinstance(item, dict):
            errors.append(f"CACHE_MEMBER_OBJECT:{ordinal}")
            continue
        expected_member = f"{sequence_id}/color/{timestamp}.png"
        expected_relative = f"color/{timestamp}.png"
        comparisons = {
            "archive_ordinal": ordinal,
            "timestamp_s": str(timestamp),
            "member_path": expected_member,
            "cache_relative_path": expected_relative,
            "window_indices": [window_index],
        }
        for field, value in comparisons.items():
            if item.get(field) != value:
                errors.append(f"CACHE_MEMBER_FIELD:{ordinal}:{field}")
        path_value = str(item.get("member_path", ""))
        try:
            timestamp_value = Decimal(str(item.get("timestamp_s", "")))
        except InvalidOperation:
            errors.append(f"CACHE_TIMESTAMP_PARSE:{ordinal}")
            timestamp_value = Decimal("-1")
        if path_value in seen_paths:
            errors.append(f"CACHE_MEMBER_PATH_DUPLICATE:{ordinal}")
        if timestamp_value in seen_timestamps:
            errors.append(f"CACHE_TIMESTAMP_DUPLICATE:{ordinal}")
        seen_paths.add(path_value)
        seen_timestamps.add(timestamp_value)
        info = zip_infos.get(expected_member)
        cache_path = cache_root / expected_relative
        if info is None:
            errors.append(f"ZIP_EXPECTED_MEMBER_MISSING:{ordinal}")
            continue
        if not cache_path.is_file():
            errors.append(f"CACHE_FILE_MISSING:{ordinal}")
            continue
        size = cache_path.stat().st_size
        if item.get("size_bytes") != size or info.file_size != size:
            errors.append(f"CACHE_SIZE:{ordinal}")
        if item.get("sha256") != digest_file(cache_path):
            errors.append(f"CACHE_SHA256:{ordinal}")
        if crc32_file(cache_path) != info.CRC:
            errors.append(f"CACHE_CRC:{ordinal}")
    control = manifest.get("control")
    if not isinstance(control, dict):
        errors.append("CACHE_CONTROL_OBJECT")
    else:
        info = zip_infos.get(control_member)
        relative = str(control.get("cache_relative_path", ""))
        path = cache_root / relative
        if control.get("member_path") != control_member:
            errors.append("CACHE_CONTROL_MEMBER")
        if info is None or not path.is_file():
            errors.append("CACHE_CONTROL_MISSING")
        else:
            if control.get("size_bytes") != path.stat().st_size:
                errors.append("CACHE_CONTROL_SIZE")
            if control.get("sha256") != digest_file(path):
                errors.append("CACHE_CONTROL_SHA256")
            if info.file_size != path.stat().st_size or crc32_file(path) != info.CRC:
                errors.append("CACHE_CONTROL_CRC")
    if manifest.get("member_count") != len(members):
        errors.append("CACHE_MEMBER_COUNT_FIELD")
    actual_file_count = sum(
        1 for path in cache_root.rglob("*") if path.is_file()
    )
    if actual_file_count != expected_total_file_count:
        errors.append(f"CACHE_TOTAL_FILE_COUNT:{actual_file_count}")
    if manifest.get("network_request_count") != 0:
        errors.append("CACHE_NETWORK_REQUEST")
    if manifest.get("downloaded_bytes") != 0:
        errors.append("CACHE_DOWNLOADED_BYTES")
    return errors


def longest_trigger_run(rows: Sequence[dict[str, Any]]) -> tuple[int, float]:
    longest_count = 0
    longest_duration = 0.0
    current_count = 0
    current_start: float | None = None
    for row in rows:
        if row.get("evaluable") is True and row.get("trigger") is True:
            if current_count == 0:
                current_start = float(row["previous_timestamp_s"])
            current_count += 1
            duration = float(row["current_timestamp_s"]) - current_start
            if current_count > longest_count or (
                current_count == longest_count and duration > longest_duration
            ):
                longest_count = current_count
                longest_duration = duration
        else:
            current_count = 0
            current_start = None
    return longest_count, longest_duration


def exact_trigger_timing_diagnostics(
    window: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    triggered = [
        row
        for row in rows
        if row.get("evaluable") is True and row.get("trigger") is True
    ]
    first_delay = (
        decimal_value(triggered[0]["current_timestamp_s"])
        - Decimal(window["start_timestamp_s"])
        if triggered
        else None
    )
    longest_count = 0
    longest_duration = Decimal("0")
    current_count = 0
    current_start: Decimal | None = None
    for row in rows:
        if row.get("evaluable") is True and row.get("trigger") is True:
            if current_count == 0:
                current_start = decimal_value(row["previous_timestamp_s"])
            current_count += 1
            duration = decimal_value(row["current_timestamp_s"]) - current_start
            if current_count > longest_count or (
                current_count == longest_count and duration > longest_duration
            ):
                longest_count = current_count
                longest_duration = duration
        else:
            current_count = 0
            current_start = None
    return {
        "first_trigger_delay_decimal_s": (
            str(first_delay) if first_delay is not None else None
        ),
        "longest_consecutive_trigger_pair_count": longest_count,
        "longest_consecutive_trigger_duration_decimal_s": str(longest_duration),
    }


def recompute_window(
    window: dict[str, Any],
    timestamps: Sequence[Decimal],
    rows: Sequence[dict[str, Any]],
    trigger_threshold: Decimal,
) -> tuple[dict[str, Any], list[str]]:
    index = int(window["window_index"])
    errors: list[str] = []
    expected_pairs = int(window["expected_pair_count"])
    if len(rows) != expected_pairs:
        errors.append(f"LEDGER_WINDOW_COUNT:{index}:{len(rows)}")
    for pair_index, row in enumerate(rows):
        if row.get("window_index") != index:
            errors.append(f"LEDGER_WINDOW_INDEX:{index}:{pair_index}")
        if row.get("pair_index") != pair_index:
            errors.append(f"LEDGER_PAIR_INDEX:{index}:{pair_index}")
        if pair_index + 1 >= len(timestamps):
            continue
        previous = timestamps[pair_index]
        current = timestamps[pair_index + 1]
        expected_dt = current - previous
        try:
            actual_previous = decimal_value(row.get("previous_timestamp_s"))
            actual_current = decimal_value(row.get("current_timestamp_s"))
            actual_dt = decimal_value(row.get("dt_s"))
        except InvalidOperation:
            errors.append(f"LEDGER_DECIMAL_PARSE:{index}:{pair_index}")
            continue
        if actual_previous != previous:
            errors.append(f"LEDGER_PREVIOUS_TIMESTAMP:{index}:{pair_index}")
        if actual_current != current:
            errors.append(f"LEDGER_CURRENT_TIMESTAMP:{index}:{pair_index}")
        if actual_dt != expected_dt:
            errors.append(f"LEDGER_DT:{index}:{pair_index}")
        evaluable = row.get("evaluable")
        if evaluable is True:
            value = row.get("compensated_expansion_median_per_s")
            if not finite_number(value):
                errors.append(f"LEDGER_COMPENSATED_NUMERIC:{index}:{pair_index}")
            else:
                expected_trigger = decimal_value(value) > trigger_threshold
                if row.get("trigger") is not expected_trigger:
                    errors.append(f"LEDGER_TRIGGER:{index}:{pair_index}")
            if row.get("reason") is not None:
                errors.append(f"LEDGER_EVALUABLE_REASON:{index}:{pair_index}")
            if (
                not finite_number(row.get("trigger_threshold_per_s"))
                or decimal_value(row["trigger_threshold_per_s"])
                != trigger_threshold
            ):
                errors.append(f"LEDGER_TRIGGER_THRESHOLD:{index}:{pair_index}")
        elif evaluable is False:
            if not isinstance(row.get("reason"), str) or not row["reason"]:
                errors.append(f"LEDGER_ABSTENTION_REASON:{index}:{pair_index}")
            if row.get("trigger") is not False:
                errors.append(f"LEDGER_ABSTENTION_TRIGGER:{index}:{pair_index}")
            if "compensated_expansion_median_per_s" in row:
                errors.append(f"LEDGER_ABSTENTION_NUMERIC:{index}:{pair_index}")
        else:
            errors.append(f"LEDGER_EVALUABLE_TYPE:{index}:{pair_index}")
    evaluable_rows = [row for row in rows if row.get("evaluable") is True]
    triggered_rows = [row for row in evaluable_rows if row.get("trigger") is True]
    abstentions = Counter(
        str(row.get("reason"))
        for row in rows
        if row.get("evaluable") is not True
    )
    values = [
        float(row["compensated_expansion_median_per_s"])
        for row in evaluable_rows
        if finite_number(row.get("compensated_expansion_median_per_s"))
    ]
    longest_count, longest_duration = longest_trigger_run(rows)
    start = Decimal(window["start_timestamp_s"])
    summary = {
        "candidate_pair_count": len(rows),
        "evaluable_pair_count": len(evaluable_rows),
        "pair_coverage": len(evaluable_rows) / len(rows) if rows else 0.0,
        "abstention_count": len(rows) - len(evaluable_rows),
        "abstention_reasons": dict(sorted(abstentions.items())),
        "median_compensated_expansion_per_s": (
            float(statistics.median(values)) if values else None
        ),
        "trigger_count": len(triggered_rows),
        "trigger_coverage_fixed_denominator": (
            len(triggered_rows) / len(rows) if rows else 0.0
        ),
        "trigger_coverage_evaluable": (
            len(triggered_rows) / len(evaluable_rows)
            if evaluable_rows
            else None
        ),
        "first_trigger_delay_s": (
            float(triggered_rows[0]["current_timestamp_s"]) - float(start)
            if triggered_rows
            else None
        ),
        "longest_consecutive_trigger_pair_count": longest_count,
        "longest_consecutive_trigger_duration_s": longest_duration,
    }
    return summary, errors


def compare_summary(
    index: int, observed: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    for field, value in expected.items():
        actual = observed.get(field)
        if isinstance(value, float):
            if not float_equal(actual, value):
                errors.append(f"RESULT_WINDOW_FIELD:{index}:{field}")
        elif actual != value:
            errors.append(f"RESULT_WINDOW_FIELD:{index}:{field}")
    return errors


def validate_result_and_ledger(
    contract: dict[str, Any],
    r0_contract: dict[str, Any],
    result: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    expected: dict[int, list[Decimal]],
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    errors: list[str] = []
    if result.get("protocol_id") != R0_PROTOCOL_ID:
        errors.append("RESULT_PROTOCOL")
    if result.get("archive_sha256") != r0_contract["source"]["archive_sha256"]:
        errors.append("RESULT_ARCHIVE_SHA256")
    if result.get("authority") != "REAL_RGB_ALGORITHM_DEVELOPMENT_CANARY_ONLY":
        errors.append("RESULT_AUTHORITY")
    if result.get("threshold_tuned") is not False:
        errors.append("RESULT_THRESHOLD_TUNED")
    if result.get("network_request_count") != 0:
        errors.append("RESULT_NETWORK_REQUEST")
    if result.get("downloaded_bytes") != 0:
        errors.append("RESULT_DOWNLOADED_BYTES")
    trigger_threshold = Decimal(
        contract["ledger_recomputation"]["trigger_threshold_per_s"]
    )
    if Decimal(
        str(r0_contract["algorithm"]["trigger_rule"]["threshold"])
    ) != trigger_threshold:
        errors.append("R0_TRIGGER_BINDING")
    ordered_identities = [
        (int(row.get("window_index", -1)), int(row.get("pair_index", -1)))
        for row in rows
    ]
    expected_identities = [
        (int(window["window_index"]), pair_index)
        for window in contract["windows"]
        for pair_index in range(int(window["expected_pair_count"]))
    ]
    if ordered_identities != expected_identities:
        errors.append("LEDGER_GLOBAL_ORDER")
    observed_windows = result.get("windows")
    if not isinstance(observed_windows, list):
        return {}, errors + ["RESULT_WINDOWS_NOT_LIST"]
    observed_by_index = {
        int(window.get("window_index", -1)): window
        for window in observed_windows
        if isinstance(window, dict)
    }
    r0_window_by_index = {
        int(window["window_index"]): window
        for window in r0_contract.get("windows", [])
        if isinstance(window, dict)
    }
    summaries: dict[int, dict[str, Any]] = {}
    for window in contract["windows"]:
        index = int(window["window_index"])
        window_rows = [row for row in rows if row.get("window_index") == index]
        summary, window_errors = recompute_window(
            window, expected[index], window_rows, trigger_threshold
        )
        summaries[index] = summary
        errors.extend(window_errors)
        observed = observed_by_index.get(index)
        if observed is None:
            errors.append(f"RESULT_WINDOW_MISSING:{index}")
        else:
            errors.extend(compare_summary(index, observed, summary))
            if Decimal(str(observed.get("trigger_threshold_per_s"))) != trigger_threshold:
                errors.append(f"RESULT_WINDOW_THRESHOLD:{index}")
            for field, expected_value in (
                ("role", window["role"]),
                ("window_start_s", float(Decimal(window["start_timestamp_s"]))),
                ("window_end_s", float(Decimal(window["end_timestamp_s"]))),
                (
                    "geometry_median_signed_radial_expansion_per_s",
                    float(
                        r0_window_by_index[index][
                            "geometry_median_signed_radial_expansion_per_s"
                        ]
                    ),
                ),
            ):
                actual_value = observed.get(field)
                if isinstance(expected_value, float):
                    if not float_equal(actual_value, expected_value):
                        errors.append(f"RESULT_WINDOW_IDENTITY:{index}:{field}")
                elif actual_value != expected_value:
                    errors.append(f"RESULT_WINDOW_IDENTITY:{index}:{field}")
    if set(observed_by_index) != set(summaries):
        errors.append("RESULT_WINDOW_INVENTORY")
    if set(summaries) == {0, 1}:
        control = summaries[0]
        positive = summaries[1]
        expansion_gap = (
            positive["median_compensated_expansion_per_s"]
            - control["median_compensated_expansion_per_s"]
        )
        coverage_gap = (
            positive["trigger_coverage_fixed_denominator"]
            - control["trigger_coverage_fixed_denominator"]
        )
        direction = bool(
            control["pair_coverage"] >= 0.8
            and positive["pair_coverage"] >= 0.8
            and expansion_gap > 0.0
            and coverage_gap > 0.0
        )
        separation = result.get("separation")
        if not isinstance(separation, dict):
            errors.append("RESULT_SEPARATION_OBJECT")
        else:
            if not float_equal(
                separation.get(
                    "positive_minus_control_median_compensated_expansion_per_s"
                ),
                expansion_gap,
            ):
                errors.append("RESULT_SEPARATION_EXPANSION")
            if not float_equal(
                separation.get(
                    "positive_minus_control_trigger_coverage_fixed_denominator"
                ),
                coverage_gap,
            ):
                errors.append("RESULT_SEPARATION_COVERAGE")
            if separation.get("direction_supported") is not direction:
                errors.append("RESULT_SEPARATION_DIRECTION")
            if (
                separation.get("performance_qualification") is not False
                or separation.get("independent_confirmation") is not False
            ):
                errors.append("RESULT_SEPARATION_AUTHORITY")
        expected_terminal = (
            "DEVELOPMENT_SIGNAL_DIRECTION_SUPPORTED / VALID"
            if direction
            else (
                "DEVELOPMENT_SIGNAL_DIRECTION_NOT_SUPPORTED / VALID"
                if control["pair_coverage"] >= 0.8
                and positive["pair_coverage"] >= 0.8
                else "NOT_EVALUABLE / VALID"
            )
        )
        if result.get("terminal") != expected_terminal:
            errors.append("RESULT_PRODUCER_TERMINAL")
    without_hash = {
        key: value for key, value in result.items() if key != "result_payload_sha256"
    }
    if result.get("result_payload_sha256") != canonical_sha(without_hash):
        errors.append("RESULT_SELF_HASH")
    return summaries, errors


def validate_invalid_r0_validation(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if value.get("status") != "INVALID":
        errors.append("R0_VALIDATION_STATUS")
    if value.get("algorithm_reexecution_performed") is not False:
        errors.append("R0_VALIDATION_REEXECUTION")
    found = value.get("errors")
    if not isinstance(found, list) or len(found) != 598:
        errors.append("R0_VALIDATION_ERROR_COUNT")
        return errors
    expected = {
        f"PAIR_TIMESTAMP:{window}:{pair}:dt_s"
        for window in (0, 1)
        for pair in range(299)
    }
    if set(found) != expected:
        errors.append("R0_VALIDATION_ERROR_IDENTITY")
    return errors


def validate(
    repo_root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    contract = load_object(contract_path)
    if contract.get("protocol_id") != PROTOCOL_ID:
        errors.append("POSTHOC_PROTOCOL")
    if contract.get("status") != "POSTHOC_PREREGISTERED_AFTER_R0_OUTCOME_ACCESS":
        errors.append("POSTHOC_STATUS")
    expected_authority = {
        "maximum_claim": "POSTHOC_R0_IDENTITY_CACHE_LEDGER_AGGREGATE_AUDIT_ONLY",
        "r0_evidence_revalidation_authorized": False,
        "algorithm_reexecution_authorized": False,
        "outcome_blind_claim_authorized": False,
        "independent_confirmation_authorized": False,
        "performance_qualification_authorized": False,
        "threshold_tuning_authorized": False,
        "product_or_safety_claim_authorized": False,
    }
    if contract.get("authority") != expected_authority:
        errors.append("POSTHOC_AUTHORITY_ESCALATION")
    binding_by_id: dict[str, Path] = {}
    verified_bindings: list[dict[str, str]] = []
    bindings = contract.get("immutable_r0_bindings", [])
    expected_binding_ids = {
        "r0-contract",
        "r0-implementation-lock",
        "r0-activation-lock",
        "r0-result",
        "r0-pair-ledger",
        "r0-invalid-validation",
        "r0-cache-manifest",
    }
    actual_binding_ids = [
        str(binding.get("id", "")) for binding in bindings
        if isinstance(binding, dict)
    ]
    if (
        len(actual_binding_ids) != len(set(actual_binding_ids))
        or set(actual_binding_ids) != expected_binding_ids
    ):
        errors.append("IMMUTABLE_BINDING_INVENTORY")
    for binding in bindings:
        path = repo_root / binding["path"]
        binding_by_id[str(binding["id"])] = path
        actual = digest_file(path) if path.is_file() else "MISSING"
        verified_bindings.append(
            {
                "id": str(binding["id"]),
                "expected_sha256": str(binding["sha256"]),
                "actual_sha256": actual,
            }
        )
        if actual != binding["sha256"]:
            errors.append(f"IMMUTABLE_BINDING:{binding['id']}")
    source = contract["source"]
    archive_path = repo_root / source["archive_path"]
    archive_sha256 = "MISSING"
    archive_md5 = "MISSING"
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != source["archive_bytes"]
    ):
        errors.append("ARCHIVE_SIZE")
    else:
        archive_sha256, archive_md5 = source_digests(archive_path)
        if archive_sha256 != source["archive_sha256"]:
            errors.append("ARCHIVE_SHA256")
        if archive_md5 != source["archive_md5"]:
            errors.append("ARCHIVE_MD5")
    if errors:
        payload = _payload(errors)
        payload["verified_bindings"] = verified_bindings
        return payload

    manifest = load_object(binding_by_id["r0-cache-manifest"])
    r0_contract = load_object(binding_by_id["r0-contract"])
    result = load_object(binding_by_id["r0-result"])
    r0_validation = load_object(binding_by_id["r0-invalid-validation"])
    rows = load_jsonl(binding_by_id["r0-pair-ledger"])
    if result.get("contract_sha256") != digest_file(binding_by_id["r0-contract"]):
        errors.append("RESULT_CONTRACT_SHA")
    if result.get("pair_ledger_sha256") != digest_file(
        binding_by_id["r0-pair-ledger"]
    ):
        errors.append("RESULT_PAIR_LEDGER_SHA")
    if result.get("cache_manifest_sha256") != digest_file(
        binding_by_id["r0-cache-manifest"]
    ):
        errors.append("RESULT_CACHE_MANIFEST_SHA")
    with zipfile.ZipFile(archive_path) as archive:
        info_list = archive.infolist()
        names = [item.filename for item in info_list]
        if len(names) != len(set(names)):
            errors.append("ZIP_MEMBER_NAME_DUPLICATE")
        infos = {item.filename: item for item in info_list}
        expected, inventory_errors = expected_window_timestamps(
            infos,
            contract["windows"],
            source["sequence_id"],
            Decimal(contract["ledger_recomputation"]["maximum_pair_dt_s"]),
        )
        errors.extend(inventory_errors)
        errors.extend(
            validate_manifest_inventory(
                manifest,
                contract["windows"],
                expected,
                infos,
                repo_root / contract["cache"]["root"],
                source["sequence_id"],
                source["control_member"],
                int(contract["cache"]["expected_total_file_count"]),
                source["archive_sha256"],
            )
        )
    summaries, result_errors = validate_result_and_ledger(
        contract, r0_contract, result, rows, expected
    )
    errors.extend(result_errors)
    errors.extend(validate_invalid_r0_validation(r0_validation))
    payload = _payload(errors)
    payload["verified_bindings"] = verified_bindings
    payload["archive_identity"] = {
        "bytes": archive_path.stat().st_size,
        "sha256": archive_sha256,
        "md5": archive_md5,
        "zip_member_count": len(infos),
    }
    payload["recomputed_windows"] = [
        {"window_index": index, **summaries[index]}
        for index in sorted(summaries)
    ]
    if set(summaries) == {0, 1}:
        payload["recomputed_separation"] = {
            "positive_minus_control_median_compensated_expansion_per_s": (
                summaries[1]["median_compensated_expansion_per_s"]
                - summaries[0]["median_compensated_expansion_per_s"]
            ),
            "positive_minus_control_trigger_coverage_fixed_denominator": (
                summaries[1]["trigger_coverage_fixed_denominator"]
                - summaries[0]["trigger_coverage_fixed_denominator"]
            ),
        }
    payload["immutable_r0_evidence_status"] = "INVALID_R0_EVIDENCE / INVALID"
    payload["decimal_trigger_timing_diagnostics"] = [
        {
            "window_index": int(window["window_index"]),
            **exact_trigger_timing_diagnostics(
                window,
                [
                    row
                    for row in rows
                    if row.get("window_index") == int(window["window_index"])
                ],
            ),
        }
        for window in contract["windows"]
    ]
    return payload


def _payload(errors: Sequence[str]) -> dict[str, Any]:
    unique = sorted(set(errors))
    return {
        "schema_version": "rcle.rgb_algorithm_development_canary.posthoc_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": (
            "POSTHOC_OUTPUT_AUDIT_VALID / VALID"
            if not unique
            else "POSTHOC_OUTPUT_AUDIT_INVALID / INVALID"
        ),
        "status": "VALID" if not unique else "INVALID",
        "errors": unique,
        "algorithm_reexecution_performed": False,
        "r0_evidence_revalidated": False,
        "outcome_blind": False,
        "independent_confirmation": False,
        "performance_qualification": False,
        "threshold_tuned": False,
        "network_request_count": 0,
        "downloaded_bytes": 0,
        "authority": "POSTHOC_R0_IDENTITY_CACHE_LEDGER_AGGREGATE_AUDIT_ONLY",
    }
