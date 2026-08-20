"""Read-only progress summary for a B4-A run directory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


TOTAL = 144


def summarize(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "execution_manifest.json"
    events_path = run_dir / "events.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = [] if not events_path.exists() else [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    dispatches = [row for row in events if row.get("kind") == "dispatch"]
    completions = [row for row in events if row.get("kind") == "completion"]
    completed_ids = {row["request_id"] for row in completions}
    active = next((row for row in reversed(dispatches) if row["request_id"] not in completed_ids), None)
    failures = [
        row
        for row in completions
        if row.get("transport_runtime_failure")
        or row.get("in_doubt")
        or row.get("returncode") not in (0,)
    ]
    timestamps = [
        row.get("completed_at") or row.get("started_at")
        for row in events
        if row.get("completed_at") or row.get("started_at")
    ]
    last_activity = timestamps[-1] if timestamps else manifest.get("started_at")
    age_seconds = None
    if last_activity:
        age_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - datetime.fromisoformat(str(last_activity))).total_seconds(),
        )
    return {
        "run_id": run_dir.name,
        "status": manifest.get("status"),
        "terminal": manifest.get("terminal"),
        "completed": len(completions),
        "total": TOTAL,
        "percent": 100.0 * len(completions) / TOTAL,
        "active": None
        if active is None
        else {
            "instance_id": active["instance_id"],
            "replicate": active["replicate"],
            "paired_identity": active["paired_identity"],
            "arm": active["arm"],
            "generation": active["generation"],
        },
        "failure_count": len(failures),
        "failure_classes": sorted(
            {
                str(row.get("semantic_error") or "provider_or_in_doubt")
                for row in failures
            }
        ),
        "last_activity": last_activity,
        "last_activity_age_seconds": age_seconds,
        "eta": "complete" if len(completions) == TOTAL else "unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
