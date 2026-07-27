from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_ID = "RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_CID_SIMS_FLOOR3_1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def longest_trigger_run(rows: list[dict[str, Any]]) -> tuple[int, float]:
    longest_count = 0
    longest_duration = 0.0
    current_count = 0
    current_start: float | None = None
    for row in rows:
        if row.get("evaluable") is True and row.get("trigger") is True:
            if current_count == 0:
                current_start = float(row["previous_timestamp_s"])
            current_count += 1
            duration = float(row["current_timestamp_s"]) - float(current_start)
            if current_count > longest_count or (
                current_count == longest_count and duration > longest_duration
            ):
                longest_count = current_count
                longest_duration = duration
        else:
            current_count = 0
            current_start = None
    return longest_count, longest_duration


def validate(
    repo_root: Path,
    contract_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    contract = load_object(contract_path)
    result = load_object(output_dir / "result.json")
    if contract.get("protocol_id") != PROTOCOL_ID:
        errors.append("CONTRACT_PROTOCOL")
    if result.get("protocol_id") != PROTOCOL_ID:
        errors.append("RESULT_PROTOCOL")
    if result.get("contract_sha256") != sha256_file(contract_path):
        errors.append("CONTRACT_SHA")
    if result.get("archive_sha256") != contract["source"]["archive_sha256"]:
        errors.append("ARCHIVE_SHA_BINDING")
    archive_path = repo_root / contract["source"]["archive_path"]
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != contract["source"]["archive_bytes"]
        or sha256_file(archive_path) != contract["source"]["archive_sha256"]
    ):
        errors.append("LIVE_ARCHIVE_BINDING")
    for binding in contract["upstream_bindings"] + contract["algorithm_bindings"]:
        path = repo_root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            errors.append(f"LIVE_BOUND_FILE:{binding['path']}")
    if result.get("threshold_tuned") is not False:
        errors.append("THRESHOLD_TUNING")
    if result.get("network_request_count") != 0 or result.get("downloaded_bytes") != 0:
        errors.append("NETWORK_OR_DOWNLOAD")
    if result.get("cache_manifest_sha256") != sha256_file(
        cache_dir / "manifest.json"
    ):
        errors.append("CACHE_MANIFEST_SHA")

    manifest = load_object(cache_dir / "manifest.json")
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "payload_sha256"
    }
    if manifest.get("payload_sha256") != canonical_sha(manifest_payload):
        errors.append("CACHE_MANIFEST_SELF_HASH")
    for item in manifest.get("members", []):
        path = cache_dir / item["cache_relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            errors.append(f"CACHE_MEMBER:{item.get('member_path')}")
            break
    control = manifest.get("control", {})
    control_path = cache_dir / str(control.get("cache_relative_path", ""))
    if (
        not control_path.is_file()
        or sha256_file(control_path) != control.get("sha256")
    ):
        errors.append("CACHE_CONTROL")

    ledger_path = output_dir / "pair_ledger.jsonl"
    if result.get("pair_ledger_sha256") != sha256_file(ledger_path):
        errors.append("LEDGER_SHA")
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        errors.append("LEDGER_OBJECT")
    result_by_window = {
        int(window["window_index"]): window for window in result.get("windows", [])
    }
    contract_by_window = {
        int(window["window_index"]): window for window in contract["windows"]
    }
    recomputed: dict[int, dict[str, Any]] = {}
    for index in sorted(contract_by_window):
        window_rows = [row for row in rows if int(row["window_index"]) == index]
        expected_pairs = int(contract_by_window[index]["candidate_pair_count"])
        identities = [int(row["pair_index"]) for row in window_rows]
        if identities != list(range(expected_pairs)):
            errors.append(f"PAIR_IDENTITY:{index}")
        cached_timestamps = sorted(
            float(item["timestamp_s"])
            for item in manifest["members"]
            if index in item["window_indices"]
            and float(contract_by_window[index]["start_timestamp_s"])
            <= float(item["timestamp_s"])
            < float(contract_by_window[index]["end_timestamp_s"])
        )
        if len(cached_timestamps) != expected_pairs + 1:
            errors.append(f"CACHE_WINDOW_TIMESTAMPS:{index}")
        else:
            for pair_index, row in enumerate(window_rows):
                expected_previous = cached_timestamps[pair_index]
                expected_current = cached_timestamps[pair_index + 1]
                expected_dt = expected_current - expected_previous
                for field, expected in (
                    ("previous_timestamp_s", expected_previous),
                    ("current_timestamp_s", expected_current),
                    ("dt_s", expected_dt),
                ):
                    actual = row.get(field)
                    if (
                        not isinstance(actual, (int, float))
                        or not math.isclose(
                            float(actual), expected, rel_tol=0.0, abs_tol=1e-9
                        )
                    ):
                        errors.append(
                            f"PAIR_TIMESTAMP:{index}:{pair_index}:{field}"
                        )
        evaluable = [row for row in window_rows if row.get("evaluable") is True]
        triggered = [row for row in evaluable if row.get("trigger") is True]
        for row in evaluable:
            value = row.get("compensated_expansion_median_per_s")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"PAIR_NUMERIC:{index}:{row.get('pair_index')}")
            if row.get("trigger") != (float(value) > 0.01):
                errors.append(f"PAIR_TRIGGER:{index}:{row.get('pair_index')}")
        abstentions = Counter(
            str(row.get("reason"))
            for row in window_rows
            if row.get("evaluable") is not True
        )
        longest_count, longest_duration = longest_trigger_run(window_rows)
        median = (
            sorted(
                float(row["compensated_expansion_median_per_s"])
                for row in evaluable
            )
            if evaluable
            else []
        )
        if median:
            middle = len(median) // 2
            median_value = (
                median[middle]
                if len(median) % 2
                else 0.5 * (median[middle - 1] + median[middle])
            )
        else:
            median_value = None
        recomputed[index] = {
            "candidate_pair_count": len(window_rows),
            "evaluable_pair_count": len(evaluable),
            "pair_coverage": len(evaluable) / len(window_rows),
            "abstention_count": len(window_rows) - len(evaluable),
            "abstention_reasons": dict(sorted(abstentions.items())),
            "median_compensated_expansion_per_s": median_value,
            "trigger_count": len(triggered),
            "trigger_coverage_fixed_denominator": len(triggered) / len(window_rows),
            "trigger_coverage_evaluable": (
                len(triggered) / len(evaluable) if evaluable else None
            ),
            "first_trigger_delay_s": (
                float(triggered[0]["current_timestamp_s"])
                - float(contract_by_window[index]["start_timestamp_s"])
                if triggered
                else None
            ),
            "longest_consecutive_trigger_pair_count": longest_count,
            "longest_consecutive_trigger_duration_s": longest_duration,
        }
        observed = result_by_window.get(index)
        if observed is None:
            errors.append(f"RESULT_WINDOW_MISSING:{index}")
            continue
        for field, expected in recomputed[index].items():
            actual = observed.get(field)
            if isinstance(expected, float):
                if (
                    not isinstance(actual, (int, float))
                    or float(actual).hex() != expected.hex()
                ):
                    errors.append(f"WINDOW_FIELD:{index}:{field}")
            elif actual != expected:
                errors.append(f"WINDOW_FIELD:{index}:{field}")

    if set(result_by_window) != set(contract_by_window):
        errors.append("WINDOW_INVENTORY")
    if set(recomputed) == {0, 1}:
        control = recomputed[0]
        positive = recomputed[1]
        direction = bool(
            control["pair_coverage"] >= 0.8
            and positive["pair_coverage"] >= 0.8
            and positive["median_compensated_expansion_per_s"]
            > control["median_compensated_expansion_per_s"]
            and positive["trigger_coverage_fixed_denominator"]
            > control["trigger_coverage_fixed_denominator"]
        )
        separation = result.get("separation", {})
        expected_expansion_gap = (
            positive["median_compensated_expansion_per_s"]
            - control["median_compensated_expansion_per_s"]
        )
        expected_coverage_gap = (
            positive["trigger_coverage_fixed_denominator"]
            - control["trigger_coverage_fixed_denominator"]
        )
        if (
            float(
                separation.get(
                    "positive_minus_control_median_compensated_expansion_per_s",
                    math.nan,
                )
            ).hex()
            != expected_expansion_gap.hex()
        ):
            errors.append("SEPARATION_EXPANSION")
        if (
            float(
                separation.get(
                    "positive_minus_control_trigger_coverage_fixed_denominator",
                    math.nan,
                )
            ).hex()
            != expected_coverage_gap.hex()
        ):
            errors.append("SEPARATION_COVERAGE")
        if separation.get("direction_supported") is not direction:
            errors.append("SEPARATION_DIRECTION")
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
            errors.append("RESULT_TERMINAL")
    if result.get("authority") != contract["authority"]["maximum_claim"]:
        errors.append("RESULT_AUTHORITY")
    if any(
        window.get("trigger_threshold_per_s")
        != contract["algorithm"]["trigger_rule"]["threshold"]
        for window in result_by_window.values()
    ):
        errors.append("RESULT_TRIGGER_THRESHOLD")
    separation = result.get("separation", {})
    if (
        separation.get("performance_qualification") is not False
        or separation.get("independent_confirmation") is not False
    ):
        errors.append("RESULT_AUTHORITY_ESCALATION")
    payload_without_hash = {
        key: value for key, value in result.items() if key != "result_payload_sha256"
    }
    if result.get("result_payload_sha256") != canonical_sha(payload_without_hash):
        errors.append("RESULT_SELF_HASH")
    return {
        "schema_version": "rcle.rgb_algorithm_development_canary.validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "algorithm_reexecution_performed": False,
        "independent_identity_cache_ledger_aggregate_recomputation": True,
        "independent_confirmation": False,
        "result_sha256": sha256_file(output_dir / "result.json"),
        "pair_ledger_sha256": sha256_file(ledger_path),
    }
