"""Audit Codex JSONL events for model-only Silver-B Brain execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


class BrainAuditError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "brain-baseline-report.json"
    if not report_path.is_file():
        raise BrainAuditError("completed brain-baseline-report.json missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    event_types: Counter[str] = Counter()
    item_types: Counter[str] = Counter()
    receipts = []
    invalid_lines = 0
    for path in sorted((run_dir / "batches").glob("batch-*/attempt-*-stdout.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            event_type = str(event.get("type"))
            event_types[event_type] += 1
            if event_type.startswith("item."):
                item_types[str(event.get("item", {}).get("type"))] += 1
        receipts.append({
            "path": str(path.relative_to(run_dir)).replace("\\", "/"),
            "sha256": _sha256_file(path),
        })
    forbidden_items = {key: value for key, value in item_types.items() if key != "agent_message"}
    complete_turns = event_types["turn.completed"]
    expected_batches = len(report.get("batch_receipts", []))
    verdict = (
        "NO_TOOL_OR_EXTERNAL_CALL_EVENTS"
        if not forbidden_items and invalid_lines == 0 and complete_turns == expected_batches
        else "MODEL_EXECUTION_EVENT_AUDIT_FAILED"
    )
    result = {
        "schema_version": 1,
        "verdict": verdict,
        "brain_report_sha256": str(report["report_sha256"]),
        "expected_batch_count": expected_batches,
        "event_types": dict(sorted(event_types.items())),
        "item_types": dict(sorted(item_types.items())),
        "forbidden_item_types": forbidden_items,
        "invalid_jsonl_line_count": invalid_lines,
        "stdout_receipts": receipts,
    }
    result["audit_sha256"] = materializer.content_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve())
    materializer.write_json(args.run_dir.resolve() / "model-execution-audit.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "NO_TOOL_OR_EXTERNAL_CALL_EVENTS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
