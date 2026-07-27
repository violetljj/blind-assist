"""Independently recompute the frozen low-reference attribution aggregates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(
        row["attribution"] for row in rows if row["old_trigger"] is True
    )
    false_categories = Counter(
        row["attribution"]
        for row in rows
        if row["old_trigger"] is True and row["geometry_below_threshold"] is True
    )
    transitions = Counter(
        (bool(row["old_trigger"]), bool(row["baseline_trigger"])) for row in rows
    )
    return {
        "candidate_pair_count": len(rows),
        "old_trigger_count": sum(row["old_trigger"] is True for row in rows),
        "baseline_trigger_count": sum(row["baseline_trigger"] is True for row in rows),
        "geometry_below_old_trigger_count": sum(
            row["old_trigger"] is True and row["geometry_below_threshold"] is True
            for row in rows
        ),
        "old_trigger_attribution_counts": dict(sorted(categories.items())),
        "geometry_below_attribution_counts": dict(sorted(false_categories.items())),
        "managed_baseline_trigger_transitions": {
            f"old_{str(old).lower()}_baseline_{str(base).lower()}": count
            for (old, base), count in sorted(transitions.items())
        },
        "manager_active_old_trigger_count": sum(
            row["old_trigger"] is True and row["support_manager_active"] is True
            for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    contract = load_object(args.contract.resolve())
    evidence = args.evidence_dir.resolve()
    result = load_object(evidence / "result.json")
    baseline_path = evidence / "baseline_pair_ledger.jsonl"
    attribution_path = evidence / "attribution_ledger.jsonl"
    baseline = read_jsonl(baseline_path)
    rows = read_jsonl(attribution_path)
    errors: list[str] = []
    for entry in contract["inputs"].values():
        path = repo / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            errors.append(f"BOUND_INPUT_IDENTITY:{entry['path']}")
    if len(baseline) != 598 or len(rows) != 598:
        errors.append("PAIR_DENOMINATOR")
    if result.get("baseline_pair_ledger_sha256") != sha256(baseline_path):
        errors.append("BASELINE_LEDGER_SHA256")
    if result.get("attribution_ledger_sha256") != sha256(attribution_path):
        errors.append("ATTRIBUTION_LEDGER_SHA256")
    recomputed = counts(rows)
    for key, value in recomputed.items():
        if result.get("aggregate", {}).get(key) != value:
            errors.append(f"AGGREGATE_MISMATCH:{key}")
    for window_id in ("TUM_RGBD_FR2_RPY@2", "TUM_RGBD_FR2_RPY@7"):
        selected = [row for row in rows if row["window_id"] == window_id]
        recomputed_window = counts(selected)
        for key, value in recomputed_window.items():
            if result.get("per_window", {}).get(window_id, {}).get(key) != value:
                errors.append(f"WINDOW_MISMATCH:{window_id}:{key}")
    allowed = set(contract["attribution_order"]) | {"OLD_NOT_TRIGGERED"}
    if any(row.get("attribution") not in allowed for row in rows):
        errors.append("ATTRIBUTION_ENUM")
    receipt = {
        "schema": "rcle.low_reference_false_trigger.attribution_validation.v1",
        "protocol_id": contract["protocol_id"],
        "status": "VALID" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "baseline_pair_count": len(baseline),
        "attribution_pair_count": len(rows),
        "recomputed_aggregate": recomputed,
    }
    if args.output.exists():
        raise FileExistsError("VALIDATION_OUTPUT_EXISTS")
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"], "errors": receipt["errors"]}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
