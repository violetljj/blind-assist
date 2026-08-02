from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import EXCLUSION_SCHEMA, PROTOCOL_ID, read_json, read_jsonl, sha256_file


def legacy_session_id(value: str) -> str:
    """RISKSEG's historical train/dev keys are `source:session`; keep native IDs."""
    return value.split(":", 1)[1] if ":" in value else value


def load_riskseg_train_dev_and_regression(ledger_path: Path) -> set[str]:
    ledger = read_json(ledger_path)
    roles = ledger.get("roles")
    if not isinstance(roles, dict):
        raise ValueError(f"{ledger_path}: missing roles")
    sessions: set[str] = set()
    for role in ("train", "dev"):
        role_data = roles.get(role)
        if not isinstance(role_data, dict) or not isinstance(role_data.get("sessions"), dict):
            raise ValueError(f"{ledger_path}: missing {role} sessions")
        sessions.update(legacy_session_id(str(value)) for value in role_data["sessions"])
    fixed = roles.get("fixed_regression")
    if not isinstance(fixed, dict) or not isinstance(fixed.get("events"), list):
        raise ValueError(f"{ledger_path}: missing fixed_regression events")
    for item in fixed["events"]:
        if not isinstance(item, dict) or not isinstance(item.get("source_session_id"), str):
            raise ValueError(f"{ledger_path}: invalid fixed_regression event")
        sessions.add(item["source_session_id"])
    return sessions


def load_consumed_event_sessions(truth_ledger_path: Path) -> set[str]:
    rows = read_jsonl(truth_ledger_path)
    sessions: set[str] = set()
    for index, row in enumerate(rows):
        session = row.get("source_session_id")
        if not isinstance(session, str) or not session:
            raise ValueError(f"{truth_ledger_path}:{index + 1}: missing source_session_id")
        sessions.add(session)
    return sessions


def build_registry(ledger_path: Path, truth_ledger_path: Path) -> dict[str, Any]:
    groups = {
        "riskseg_r0_train_dev_and_fixed_regression": sorted(load_riskseg_train_dev_and_regression(ledger_path)),
        "riskseg_r0_consumed_event_eval": sorted(load_consumed_event_sessions(truth_ledger_path)),
    }
    combined = sorted({session for group in groups.values() for session in group})
    return {
        "schema_version": EXCLUSION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_EXCLUSION_REGISTRY",
        "source_ledgers": [
            {"path": str(ledger_path).replace("\\", "/"), "sha256": sha256_file(ledger_path)},
            {"path": str(truth_ledger_path).replace("\\", "/"), "sha256": sha256_file(truth_ledger_path)},
        ],
        "groups": groups,
        "excluded_source_sessions": combined,
        "excluded_source_session_count": len(combined),
        "rule": "A candidate EVAL-VALIDITY event must have a native source session absent from every listed group. Camera view, crop, adjacent frame and parent-event renaming do not create independence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--riskseg-ledger", type=Path, required=True)
    parser.add_argument("--consumed-truth-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = build_registry(args.riskseg_ledger, args.consumed_truth_ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(__import__("json").dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"excluded_sessions={result['excluded_source_session_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
