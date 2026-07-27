"""Independently recompute RGB ledger aggregates without importing the producer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any


EXPECTED = {
    "desk_changing_1@4065.364250422": ("POSITIVE_APPROACH_WINDOW", 270),
    "japanesealley/Hard/P002@000260": ("POSITIVE_APPROACH_WINDOW", 99),
    "TUM_RGBD_FR2_RPY@2": ("BELOW_TRIGGER_REFERENCE_WINDOW", 299),
    "TUM_RGBD_FR2_RPY@7": ("BELOW_TRIGGER_REFERENCE_WINDOW", 299),
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def longest(rows: list[dict[str, Any]]) -> tuple[int, float]:
    best_count = count = 0
    best_duration = 0.0
    start = None
    for row in rows:
        if row["evaluable"] is True and row["trigger"] is True:
            if count == 0:
                start = float(row["previous_timestamp_s"])
            count += 1
            duration = float(row["current_timestamp_s"]) - float(start)
            if count > best_count or (count == best_count and duration > best_duration):
                best_count, best_duration = count, duration
        else:
            count = 0
            start = None
    return best_count, best_duration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--adapter-amendment", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--rgb-manifest", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--failure-receipt", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    canary_path = args.canary.resolve()
    amendment_path = args.adapter_amendment.resolve()
    lock_path = args.implementation_lock.resolve()
    manifest_path = args.rgb_manifest.resolve()
    result_dir = args.result_dir.resolve()
    canary = load(canary_path)
    amendment = load(amendment_path)
    lock = load(lock_path)
    manifest = load(manifest_path)
    result_path = result_dir / "result.json"
    ledger_path = result_dir / "rgb_pair_ledger.jsonl"
    result = load(result_path)
    if lock["local_inputs"] != {
        "adapter_amendment_sha256": sha(amendment_path),
        "canary_sha256": sha(canary_path),
        "rgb_manifest_sha256": sha(manifest_path),
    }:
        raise ValueError("VALIDATOR_LOCK_INPUT")
    for entry in lock["files"]:
        if sha(repo / entry["path"]) != entry["sha256"]:
            raise ValueError(f"VALIDATOR_IMPLEMENTATION_DRIFT:{entry['path']}")
    if result["adapter_amendment_sha256"] != sha(amendment_path):
        raise ValueError("VALIDATOR_AMENDMENT_RESULT_IDENTITY")
    if result["rgb_pair_ledger_sha256"] != sha(ledger_path):
        raise ValueError("VALIDATOR_LEDGER_HASH")
    if (
        result["algorithm_changed"] is not False
        or result["threshold_tuned"] is not False
        or result["window_substitution"] is not False
        or result["workers"] != 8
    ):
        raise ValueError("VALIDATOR_EXECUTION_BOUNDARY")
    if result["authority"] != {
        "all_real_cross_source_holdout": False,
        "android": False,
        "development_cohort": True,
        "performance": False,
        "product": False,
    }:
        raise ValueError("VALIDATOR_AUTHORITY")
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["window_id"]].append(row)
    if set(grouped) != set(EXPECTED):
        raise ValueError("VALIDATOR_WINDOW_SET")
    recomputed = {}
    for window_id, (role, denominator) in EXPECTED.items():
        selected = sorted(grouped[window_id], key=lambda row: int(row["pair_index"]))
        if len(selected) != denominator or [row["pair_index"] for row in selected] != list(range(denominator)):
            raise ValueError(f"VALIDATOR_PAIR_SET:{window_id}")
        if any(row["role"] != role for row in selected):
            raise ValueError(f"VALIDATOR_PAIR_ROLE:{window_id}")
        evaluable = [row for row in selected if row["evaluable"] is True]
        triggered = [row for row in evaluable if row["trigger"] is True]
        trigger_threshold = float(canary["unchanged_algorithm"]["trigger_threshold_per_s"])
        if any(
            bool(float(row["compensated_expansion_median_per_s"]) > trigger_threshold)
            != bool(row["trigger"])
            for row in evaluable
        ):
            raise ValueError(f"VALIDATOR_TRIGGER_RULE:{window_id}")
        longest_count, longest_duration = longest(selected)
        recomputed[window_id] = {
            "role": role,
            "candidate_pair_count": denominator,
            "evaluable_pair_count": len(evaluable),
            "pair_coverage": len(evaluable) / denominator,
            "abstention_count": denominator - len(evaluable),
            "abstention_reasons": dict(
                sorted(Counter(str(row["reason"]) for row in selected if row["evaluable"] is not True).items())
            ),
            "median_compensated_expansion_per_s": (
                statistics.median(float(row["compensated_expansion_median_per_s"]) for row in evaluable)
                if evaluable
                else None
            ),
            "trigger_count": len(triggered),
            "trigger_coverage_fixed_denominator": len(triggered) / denominator,
            "longest_consecutive_trigger_pair_count": longest_count,
            "longest_consecutive_trigger_duration_s": longest_duration,
        }
    summaries = {row["window_id"]: row for row in result["window_summaries"]}
    if set(summaries) != set(EXPECTED):
        raise ValueError("VALIDATOR_SUMMARY_SET")
    exact_fields = (
        "role",
        "candidate_pair_count",
        "evaluable_pair_count",
        "abstention_count",
        "abstention_reasons",
        "trigger_count",
        "longest_consecutive_trigger_pair_count",
    )
    numeric_fields = (
        "pair_coverage",
        "median_compensated_expansion_per_s",
        "trigger_coverage_fixed_denominator",
        "longest_consecutive_trigger_duration_s",
    )
    for window_id, expected in recomputed.items():
        actual = summaries[window_id]
        if any(actual[field] != expected[field] for field in exact_fields):
            raise ValueError(f"VALIDATOR_SUMMARY_EXACT:{window_id}")
        if any(not close(actual[field], expected[field]) for field in numeric_fields):
            raise ValueError(f"VALIDATOR_SUMMARY_NUMERIC:{window_id}")
    coverage_ok = all(row["pair_coverage"] >= 0.8 for row in recomputed.values())
    positive = [row for row in recomputed.values() if row["role"] == "POSITIVE_APPROACH_WINDOW"]
    below = [row for row in recomputed.values() if row["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"]
    aggregates = {}
    for name, selected in (("positive", positive), ("below_reference", below)):
        aggregates[name] = {
            "median_compensated_expansion_per_s": statistics.median(
                row["median_compensated_expansion_per_s"] for row in selected
            ),
            "median_trigger_coverage_fixed_denominator": statistics.median(
                row["trigger_coverage_fixed_denominator"] for row in selected
            ),
        }
    for role, expected in aggregates.items():
        if any(not close(result["role_aggregates"][role][key], value) for key, value in expected.items()):
            raise ValueError(f"VALIDATOR_ROLE_AGGREGATE:{role}")
    direction = bool(
        coverage_ok
        and aggregates["positive"]["median_compensated_expansion_per_s"]
        > aggregates["below_reference"]["median_compensated_expansion_per_s"]
        and aggregates["positive"]["median_trigger_coverage_fixed_denominator"]
        > aggregates["below_reference"]["median_trigger_coverage_fixed_denominator"]
    )
    expected_terminal = (
        "DEVELOPMENT_SIGNAL_DIRECTION_SUPPORTED / VALID"
        if direction
        else "DEVELOPMENT_SIGNAL_DIRECTION_NOT_SUPPORTED / VALID"
        if coverage_ok
        else "NOT_EVALUABLE / VALID"
    )
    if (
        result["direction_supported"] is not direction
        or result["all_window_coverage_pass"] is not coverage_ok
        or result["terminal"] != expected_terminal
    ):
        raise ValueError("VALIDATOR_TERMINAL")
    member_counts = []
    for window in manifest["windows"]:
        root = Path(window["rgb_root"])
        for record in window["members"]:
            relative = record.get("path", record.get("relative_path"))
            path = root.joinpath(*Path(relative).parts) if "path" in record else root / relative
            if path.stat().st_size != int(record["bytes"]) or sha(path) != record["sha256"]:
                raise ValueError(f"VALIDATOR_RGB_MEMBER:{window['window_id']}:{relative}")
        member_counts.append(len(window["members"]))
    if member_counts != [271, 100, 300, 300]:
        raise ValueError("VALIDATOR_RGB_MEMBER_COUNTS")
    failures = []
    for path in args.failure_receipt:
        receipt = load(path.resolve())
        if (
            receipt["terminal"] != "IMPLEMENTATION_ADAPTER_FAILURE_NO_RESULT"
            or receipt["result_written"] is not False
            or receipt["aggregate_metric_written"] is not False
            or receipt["window_metric_written"] is not False
        ):
            raise ValueError("VALIDATOR_FAILURE_RECEIPT")
        failures.append({"path": path.resolve().relative_to(repo).as_posix(), "sha256": sha(path.resolve())})
    validation = {
        "schema": "rcle.motion_diverse_rgbd.source_search.rgb_independent_validation.v1",
        "protocol_id": result["protocol_id"],
        "validated_result_sha256": sha(result_path),
        "validated_ledger_sha256": sha(ledger_path),
        "validated_rgb_manifest_sha256": sha(manifest_path),
        "validated_implementation_lock_sha256": sha(lock_path),
        "window_count": len(recomputed),
        "pair_count": len(rows),
        "rgb_member_counts": member_counts,
        "role_aggregates_recomputed": aggregates,
        "terminal_recomputed": expected_terminal,
        "failure_history": failures,
        "producer_imported": False,
        "validation": "PASS",
        "authority": result["authority"],
    }
    payload = json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(os.fspath(args.output.resolve()), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "validation": "PASS",
                "pair_count": len(rows),
                "terminal_recomputed": expected_terminal,
                "validation_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
