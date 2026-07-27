"""Independent identity, ledger and aggregate validator for holdout R0.

This module intentionally does not import the formal runner or RGB producer.
It does not claim an independent RGB algorithm implementation replay.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Mapping, Sequence
import zipfile


PROTOCOL_ID = "RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_2_CROSS_SEQUENCE_HOLDOUT_R0"
AUTHORITY = "CROSS_SEQUENCE_SAME_SOURCE_DEVELOPMENT_HOLDOUT_ONLY"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line, parse_float=Decimal)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL_OBJECT_REQUIRED")
    return rows


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _payload_hash_valid(value: Mapping[str, Any], field: str) -> bool:
    expected = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    return expected == _canonical_sha(_plain(payload))


def _band(value: Decimal) -> str:
    if value < Decimal("0.01"):
        return "BELOW_TRIGGER_REFERENCE"
    if value < Decimal("0.05"):
        return "WEAK_POSITIVE_RADIAL"
    return "POSITIVE_APPROACH_GEOMETRY"


def _longest_run(
    rows: Sequence[dict[str, Any]], band: str
) -> tuple[int, Decimal]:
    longest_count = 0
    longest_duration = Decimal("0")
    count = 0
    start: Decimal | None = None
    previous_pair: int | None = None
    for row in rows:
        pair_index = int(row["pair_index"])
        contiguous = previous_pair is not None and pair_index == previous_pair + 1
        if row.get("geometry_band") == band:
            if count == 0 or not contiguous:
                count = 0
                start = Decimal(str(row["previous_timestamp_s"]))
            count += 1
            duration = Decimal(str(row["current_timestamp_s"])) - start
            if count > longest_count or (
                count == longest_count and duration > longest_duration
            ):
                longest_count = count
                longest_duration = duration
        else:
            count = 0
            start = None
        previous_pair = pair_index
    return longest_count, longest_duration


def _depth_timestamps(
    infos: Sequence[zipfile.ZipInfo], sequence_id: str
) -> list[Decimal]:
    prefix = f"{sequence_id}/depth/"
    return sorted(
        Decimal(PurePosixPath(info.filename).stem)
        for info in infos
        if info.filename.startswith(prefix) and info.filename.endswith(".png")
    )


def _expected_windows(
    timestamps: Sequence[Decimal], contract: Mapping[str, Any]
) -> dict[int, dict[str, Any]]:
    rules = contract["geometry_only_selection"]
    if not timestamps:
        return {}
    anchor = timestamps[0]
    duration = Decimal(rules["window_duration_s"])
    maximum_dt = Decimal(rules["maximum_pair_dt_s"])
    minimum_frames = int(rules["minimum_frames_per_window"])
    maximum_frames = int(rules["maximum_frames_per_window"])
    result: dict[int, dict[str, Any]] = {}
    complete_window_count = int((timestamps[-1] - anchor) // duration)
    for index in range(complete_window_count):
        start = anchor + index * duration
        end = start + duration
        rows = [value for value in timestamps if start <= value < end]
        pairs = list(zip(rows, rows[1:]))
        eligible = minimum_frames <= len(rows) <= maximum_frames and all(
            Decimal("0") < right - left <= maximum_dt
            for left, right in pairs
        )
        result[index] = {
            "start": start,
            "end": end,
            "timestamps": rows,
            "pairs": pairs,
            "eligible": eligible,
        }
    return result


def _role_from_rows(
    window: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    rules = contract["geometry_only_selection"]
    denominator = len(window["pairs"])
    evaluable = [row for row in rows if row.get("geometry_evaluable") is True]
    counts = Counter(str(row.get("geometry_band")) for row in evaluable)
    positive_count, positive_duration = _longest_run(
        rows, "POSITIVE_APPROACH_GEOMETRY"
    )
    below_count, below_duration = _longest_run(
        rows, "BELOW_TRIGGER_REFERENCE"
    )
    coverage = (
        Decimal(len(evaluable)) / Decimal(denominator)
        if denominator
        else Decimal(0)
    )
    positive_fraction = (
        Decimal(counts["POSITIVE_APPROACH_GEOMETRY"]) / Decimal(denominator)
        if denominator
        else Decimal(0)
    )
    below_fraction = (
        Decimal(counts["BELOW_TRIGGER_REFERENCE"]) / Decimal(denominator)
        if denominator
        else Decimal(0)
    )
    positive_rule = rules["role_eligibility"]["positive_approach_window"]
    below_rule = rules["role_eligibility"]["below_trigger_reference_window"]
    coverage_ok = (
        window["eligible"]
        and coverage >= Decimal(str(rules["minimum_pair_geometry_coverage"]))
    )
    positive = bool(
        coverage_ok
        and positive_fraction
        >= Decimal(
            str(
                positive_rule[
                    "minimum_fixed_denominator_positive_fraction"
                ]
            )
        )
        and positive_duration
        >= Decimal(
            str(positive_rule["minimum_longest_positive_run_duration_s"])
        )
    )
    below = bool(
        coverage_ok
        and below_fraction
        >= Decimal(
            str(below_rule["minimum_fixed_denominator_below_fraction"])
        )
        and below_duration
        >= Decimal(str(below_rule["minimum_longest_below_run_duration_s"]))
    )
    role = (
        "POSITIVE_APPROACH_WINDOW"
        if positive
        else (
            "BELOW_TRIGGER_REFERENCE_WINDOW"
            if below
            else "AMBIGUOUS_OR_INELIGIBLE"
        )
    )
    values = [
        Decimal(str(row["geometry_signed_radial_expansion_per_s"]))
        for row in evaluable
    ]
    first_positive = next(
        (
            Decimal(str(row["current_timestamp_s"]))
            for row in rows
            if row.get("geometry_band") == "POSITIVE_APPROACH_GEOMETRY"
        ),
        None,
    )
    return role, {
        "coverage": coverage,
        "positive_fraction": positive_fraction,
        "below_fraction": below_fraction,
        "positive_count": positive_count,
        "positive_duration": positive_duration,
        "below_count": below_count,
        "below_duration": below_duration,
        "median": statistics.median(values) if values else None,
        "first_positive": first_positive,
        "evaluable_count": len(evaluable),
        "band_counts": counts,
    }


def _select(
    summaries: Sequence[dict[str, Any]], contract: Mapping[str, Any]
) -> list[int]:
    rules = contract["geometry_only_selection"]["selection"]
    positive = [
        item for item in summaries if item["role"] == "POSITIVE_APPROACH_WINDOW"
    ]
    below = [
        item
        for item in summaries
        if item["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"
    ]
    minimum = Decimal(rules["minimum_selected_start_separation_s"])
    feasible: list[tuple[int, ...]] = []
    for p_rows in itertools.combinations(
        positive, int(rules["required_positive_windows"])
    ):
        for b_rows in itertools.combinations(
            below, int(rules["required_below_reference_windows"])
        ):
            rows = sorted(
                (*p_rows, *b_rows), key=lambda item: int(item["window_index"])
            )
            starts = [Decimal(str(item["start_timestamp_s"])) for item in rows]
            if all(
                right - left >= minimum
                for left, right in zip(starts, starts[1:])
            ):
                feasible.append(tuple(int(item["window_index"]) for item in rows))
    return list(sorted(feasible)[0]) if feasible else []


def _same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return float(left).hex() == float(right).hex()


def _validate_geometry(
    archive: zipfile.ZipFile,
    contract: Mapping[str, Any],
    selection: Mapping[str, Any],
    ledger_rows: Sequence[dict[str, Any]],
    errors: list[str],
) -> None:
    expected = _expected_windows(
        _depth_timestamps(archive.infolist(), contract["source"]["sequence_id"]),
        contract,
    )
    rows_by_window: dict[int, list[dict[str, Any]]] = {
        index: [] for index in expected
    }
    for row in ledger_rows:
        index = int(row.get("window_index", -1))
        if index not in rows_by_window:
            errors.append(f"GEOMETRY_LEDGER_WINDOW:{index}")
            continue
        rows_by_window[index].append(row)
    summaries = selection.get("candidate_windows", [])
    if [int(item["window_index"]) for item in summaries] != list(expected):
        errors.append("CANDIDATE_WINDOW_ORDER")
        return
    for summary in summaries:
        index = int(summary["window_index"])
        window = expected[index]
        rows = rows_by_window[index]
        expected_pairs = window["pairs"] if window["eligible"] else []
        if len(rows) != len(expected_pairs):
            errors.append(f"GEOMETRY_LEDGER_COUNT:{index}")
            continue
        for pair_index, (row, pair) in enumerate(zip(rows, expected_pairs)):
            previous, current = pair
            if (
                int(row.get("pair_index", -1)) != pair_index
                or Decimal(str(row.get("previous_timestamp_s"))) != previous
                or Decimal(str(row.get("current_timestamp_s"))) != current
                or Decimal(str(row.get("dt_s"))) != current - previous
            ):
                errors.append(f"GEOMETRY_LEDGER_IDENTITY:{index}:{pair_index}")
            if row.get("geometry_evaluable") is True:
                signed = Decimal(
                    str(row.get("geometry_signed_radial_expansion_per_s"))
                )
                if row.get("geometry_band") != _band(signed):
                    errors.append(f"GEOMETRY_BAND:{index}:{pair_index}")
        role, aggregate = _role_from_rows(window, rows, contract)
        checks = {
            "role": role,
            "identity_eligible": window["eligible"],
            "frame_count": len(window["timestamps"]),
            "candidate_pair_count": len(window["pairs"]),
            "geometry_evaluable_pair_count": aggregate["evaluable_count"],
            "longest_positive_run_pair_count": aggregate["positive_count"],
            "longest_below_run_pair_count": aggregate["below_count"],
        }
        for key, value in checks.items():
            if summary.get(key) != value:
                errors.append(f"GEOMETRY_SUMMARY:{index}:{key}")
        numeric_checks = {
            "geometry_pair_coverage_fixed_denominator": aggregate["coverage"],
            "positive_fraction_fixed_denominator": aggregate[
                "positive_fraction"
            ],
            "below_fraction_fixed_denominator": aggregate["below_fraction"],
            "longest_positive_run_duration_s": aggregate["positive_duration"],
            "longest_below_run_duration_s": aggregate["below_duration"],
            "median_signed_radial_expansion_per_s": aggregate["median"],
            "first_positive_geometry_timestamp_s": aggregate["first_positive"],
        }
        for key, value in numeric_checks.items():
            if not _same_number(summary.get(key), value):
                errors.append(f"GEOMETRY_SUMMARY_NUMERIC:{index}:{key}")
    expected_selected = _select(summaries, contract)
    actual_selected = [
        int(item["window_index"]) for item in selection.get("selected_windows", [])
    ]
    if actual_selected != expected_selected:
        errors.append("SELECTION_FORGERY")
    if bool(selection.get("selection_evaluable")) != bool(expected_selected):
        errors.append("SELECTION_EVALUABLE")


def _validate_rgb_identity_and_cache(
    archive: zipfile.ZipFile,
    contract: Mapping[str, Any],
    selection: Mapping[str, Any],
    identity: Mapping[str, Any],
    cache_dir: Path,
    errors: list[str],
) -> None:
    selected = [
        int(item["window_index"]) for item in selection["selected_windows"]
    ]
    if identity.get("selected_window_indices") != selected:
        errors.append("RGB_IDENTITY_WINDOW_ORDER")
    if identity.get("selection_sha256") != _sha256_file(
        cache_dir.parent / "geometry_selection.json"
    ):
        errors.append("RGB_IDENTITY_SELECTION_BINDING")
    infos = archive.infolist()
    info = {item.filename: item for item in infos}
    expected_windows = _expected_windows(
        _depth_timestamps(infos, contract["source"]["sequence_id"]), contract
    )
    expected_members: list[dict[str, Any]] = []
    for index in selected:
        for timestamp in expected_windows[index]["timestamps"]:
            timestamp_text = str(timestamp)
            name = (
                f"{contract['source']['sequence_id']}/color/"
                f"{timestamp_text}.png"
            )
            source = info.get(name)
            if source is None:
                errors.append(f"RGB_EXPECTED_MEMBER_MISSING:{name}")
                continue
            expected_members.append(
                {
                    "window_index": index,
                    "timestamp_s": timestamp_text,
                    "archive_member": name,
                    "crc32": f"{source.CRC:08x}",
                    "uncompressed_bytes": source.file_size,
                    "compressed_bytes": source.compress_size,
                }
            )
    if identity.get("members") != expected_members:
        errors.append("RGB_IDENTITY_EXACT_ORDERED_SET")
    if not _payload_hash_valid(identity, "identity_payload_sha256"):
        errors.append("RGB_IDENTITY_PAYLOAD_SHA")
    manifest = _load_object(cache_dir / "manifest.json")
    if manifest.get("identity_payload_sha256") != identity.get(
        "identity_payload_sha256"
    ):
        errors.append("CACHE_IDENTITY_BINDING")
    manifest_members = manifest.get("members", [])
    if len(manifest_members) != len(expected_members):
        errors.append("CACHE_MANIFEST_MEMBER_COUNT")
    for expected, item in zip(expected_members, manifest_members):
        expected_mapping = {
            **expected,
            "cache_relative_path": f"color/{expected['timestamp_s']}.png",
            "window_indices": [expected["window_index"]],
        }
        for key, value in expected_mapping.items():
            if item.get(key) != value:
                errors.append(
                    f"CACHE_MANIFEST_MAPPING:{expected['archive_member']}:{key}"
                )
        path = cache_dir / str(item["cache_relative_path"])
        if not path.is_file() or _sha256_file(path) != item["sha256"]:
            errors.append(f"CACHE_MEMBER_SHA:{path.name}")
    control = manifest.get("control", {})
    control_path = cache_dir / str(control.get("cache_relative_path", ""))
    if not control_path.is_file() or _sha256_file(control_path) != control.get(
        "sha256"
    ):
        errors.append("CACHE_CONTROL_SHA")
    if control.get("archive_member") != contract["source"]["pose_member"]:
        errors.append("CACHE_CONTROL_MEMBER")


def _longest_trigger(
    rows: Sequence[dict[str, Any]],
) -> tuple[int, Decimal]:
    longest_count = 0
    longest_duration = Decimal("0")
    count = 0
    start: Decimal | None = None
    for row in rows:
        if row.get("evaluable") is True and row.get("trigger") is True:
            if count == 0:
                start = Decimal(str(row["previous_timestamp_s"]))
            count += 1
            duration = Decimal(str(row["current_timestamp_s"])) - start
            if count > longest_count or (
                count == longest_count and duration > longest_duration
            ):
                longest_count = count
                longest_duration = duration
        else:
            count = 0
            start = None
    return longest_count, longest_duration


def _validate_rgb_ledger(
    selection: Mapping[str, Any],
    result: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    errors: list[str],
) -> None:
    selected = {
        int(item["window_index"]): item for item in selection["selected_windows"]
    }
    result_windows = {
        int(item["window_index"]): item
        for item in result.get("aggregate", {}).get("windows", [])
    }
    rows_by_window = {index: [] for index in selected}
    for row in rows:
        index = int(row.get("window_index", -1))
        if index not in rows_by_window:
            errors.append(f"RGB_LEDGER_WINDOW:{index}")
            continue
        rows_by_window[index].append(row)
    for index, geometry_summary in selected.items():
        window_rows = rows_by_window[index]
        if len(window_rows) != int(geometry_summary["candidate_pair_count"]):
            errors.append(f"RGB_LEDGER_COUNT:{index}")
            continue
        for pair_index, row in enumerate(window_rows):
            if int(row.get("pair_index", -1)) != pair_index:
                errors.append(f"RGB_LEDGER_ORDER:{index}:{pair_index}")
            if row.get("evaluable") is True:
                expected_trigger = (
                    Decimal(str(row["compensated_expansion_median_per_s"]))
                    > Decimal("0.01")
                )
                if row.get("trigger") is not expected_trigger:
                    errors.append(f"RGB_TRIGGER:{index}:{pair_index}")
            elif row.get("trigger") is not False:
                errors.append(f"RGB_ABSTENTION_TRIGGER:{index}:{pair_index}")
        evaluable = [row for row in window_rows if row.get("evaluable") is True]
        triggered = [row for row in evaluable if row.get("trigger") is True]
        values = [
            Decimal(str(row["compensated_expansion_median_per_s"]))
            for row in evaluable
        ]
        longest_count, longest_duration = _longest_trigger(window_rows)
        start = Decimal(str(geometry_summary["start_timestamp_s"]))
        expected_values = {
            "candidate_pair_count": len(window_rows),
            "evaluable_pair_count": len(evaluable),
            "abstention_count": len(window_rows) - len(evaluable),
            "trigger_count": len(triggered),
            "longest_consecutive_trigger_pair_count": longest_count,
        }
        actual = result_windows.get(index, {})
        for key, value in expected_values.items():
            if actual.get(key) != value:
                errors.append(f"RGB_AGGREGATE:{index}:{key}")
        numeric = {
            "pair_coverage": Decimal(len(evaluable)) / Decimal(len(window_rows)),
            "median_compensated_expansion_per_s": (
                statistics.median(values) if values else None
            ),
            "trigger_coverage_fixed_denominator": (
                Decimal(len(triggered)) / Decimal(len(window_rows))
            ),
            "trigger_coverage_evaluable": (
                Decimal(len(triggered)) / Decimal(len(evaluable))
                if evaluable
                else None
            ),
            "first_trigger_delay_s": (
                Decimal(str(triggered[0]["current_timestamp_s"])) - start
                if triggered
                else None
            ),
            "longest_consecutive_trigger_duration_s": longest_duration,
        }
        for key, value in numeric.items():
            if not _same_number(actual.get(key), value):
                errors.append(f"RGB_AGGREGATE_NUMERIC:{index}:{key}")
    positive = [
        result_windows[index]
        for index, item in selected.items()
        if item["role"] == "POSITIVE_APPROACH_WINDOW"
    ]
    below = [
        result_windows[index]
        for index, item in selected.items()
        if item["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"
    ]
    direction = all(
        p_item["median_compensated_expansion_per_s"]
        > b_item["median_compensated_expansion_per_s"]
        and p_item["trigger_coverage_fixed_denominator"]
        > b_item["trigger_coverage_fixed_denominator"]
        for p_item in positive
        for b_item in below
    )
    minimum = Decimal("0.8")
    evaluable = all(
        Decimal(str(item["pair_coverage"])) >= minimum
        for item in result_windows.values()
    )
    expected_terminal = (
        "RGB_HOLDOUT_NOT_EVALUABLE / VALID"
        if not evaluable
        else (
            "CROSS_SEQUENCE_DIRECTION_REPLICATED / VALID"
            if direction
            else "CROSS_SEQUENCE_DIRECTION_NOT_REPLICATED / VALID"
        )
    )
    if result.get("terminal") != expected_terminal:
        errors.append("RGB_TERMINAL")
    if result.get("aggregate", {}).get("direction_replicated") is not direction:
        errors.append("RGB_DIRECTION_AGGREGATE")


def validate(
    repo_root: Path,
    contract_path: Path,
    implementation_lock_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    contract = _load_object(contract_path)
    lock = _load_object(implementation_lock_path)
    selection_path = run_dir / "geometry_selection.json"
    ledger_path = run_dir / "geometry_pair_ledger.jsonl"
    result_path = run_dir / "result.json"
    selection = _load_object(selection_path)
    result = _load_object(result_path)
    geometry_rows = _load_jsonl(ledger_path)
    if contract.get("protocol_id") != PROTOCOL_ID:
        errors.append("CONTRACT_PROTOCOL")
    if lock.get("protocol_id") != PROTOCOL_ID:
        errors.append("LOCK_PROTOCOL")
    for entry in lock.get("files", []):
        path = repo_root / str(entry["path"])
        if not path.is_file() or _sha256_file(path) != entry["sha256"]:
            errors.append(f"LOCK_DRIFT:{entry['path']}")
    for binding in contract.get("immutable_algorithm_bindings", []):
        path = repo_root / str(binding["path"])
        if not path.is_file() or _sha256_file(path) != binding["sha256"]:
            errors.append(f"ALGORITHM_BINDING:{binding['path']}")
    if selection.get("contract_sha256") != _sha256_file(contract_path):
        errors.append("SELECTION_CONTRACT_BINDING")
    if selection.get("geometry_pair_ledger_sha256") != _sha256_file(
        ledger_path
    ):
        errors.append("GEOMETRY_LEDGER_SHA")
    if not _payload_hash_valid(selection, "selection_payload_sha256"):
        errors.append("SELECTION_PAYLOAD_SHA")
    if result.get("authority") != AUTHORITY:
        errors.append("RESULT_AUTHORITY")
    if not _payload_hash_valid(result, "result_payload_sha256"):
        errors.append("RESULT_PAYLOAD_SHA")
    archive_path = repo_root / str(contract["source"]["archive_path"])
    if archive_path.stat().st_size != int(contract["source"]["archive_bytes"]):
        errors.append("ARCHIVE_SIZE")
    if _sha256_file(archive_path) != contract["source"]["archive_sha256"]:
        errors.append("ARCHIVE_SHA256")
    if _md5_file(archive_path) != contract["source"]["archive_md5"]:
        errors.append("ARCHIVE_MD5")
    for binding_name in ("transport_lock", "transport_receipt"):
        path = repo_root / str(contract["source"][f"{binding_name}_path"])
        if (
            not path.is_file()
            or _sha256_file(path)
            != contract["source"][f"{binding_name}_sha256"]
        ):
            errors.append(f"{binding_name.upper()}_DRIFT")
    with zipfile.ZipFile(archive_path) as archive:
        _validate_geometry(archive, contract, selection, geometry_rows, errors)
        selected = selection.get("selected_windows", [])
        if not selected:
            forbidden = (
                run_dir / "selected_rgb_identity.json",
                run_dir / "rgb_cache",
                run_dir / "rgb_pair_ledger.jsonl",
            )
            if any(path.exists() for path in forbidden):
                errors.append("RGB_ARTIFACT_EXISTS_AFTER_NOT_EVALUABLE")
            if (
                result.get("terminal")
                != "GEOMETRY_STRATIFIED_WINDOWS_NOT_EVALUABLE / VALID"
                or result.get("rgb_member_bytes_read") != 0
                or result.get("rgb_algorithm_executed") is not False
            ):
                errors.append("NOT_EVALUABLE_RESULT")
        else:
            identity_path = run_dir / "selected_rgb_identity.json"
            cache_dir = run_dir / "rgb_cache"
            rgb_ledger_path = run_dir / "rgb_pair_ledger.jsonl"
            identity = _load_object(identity_path)
            _validate_rgb_identity_and_cache(
                archive, contract, selection, identity, cache_dir, errors
            )
            if result.get("rgb_pair_ledger_sha256") != _sha256_file(
                rgb_ledger_path
            ):
                errors.append("RGB_LEDGER_SHA")
            _validate_rgb_ledger(
                selection, result, _load_jsonl(rgb_ledger_path), errors
            )
    return {
        "schema_version": "rcle.disjoint_holdout.validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "validator_imports_formal_runner": False,
        "validator_reexecutes_rgb_algorithm": False,
        "independent_geometry_ledger_identity_and_aggregate_recomputation": True,
        "independent_rgb_cache_ledger_and_aggregate_recomputation": bool(
            selection.get("selected_windows")
        ),
        "contract_sha256": _sha256_file(contract_path),
        "implementation_lock_sha256": _sha256_file(
            implementation_lock_path
        ),
        "selection_sha256": _sha256_file(selection_path),
        "result_sha256": _sha256_file(result_path),
    }
