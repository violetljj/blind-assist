"""Read-only progress summary for a B3-A run directory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def summarize(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    events_path = run_dir / "events.jsonl"
    events = [] if not events_path.exists() else [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    dispatches = [event for event in events if event.get("kind") == "dispatch"]
    completions = [event for event in events if event.get("kind") == "completion"]
    failures = [event for event in completions if event.get("transport_runtime_failure") or event.get("semantic_error")]
    total = int(manifest["planned_model_calls"])
    last = progress.get("last_activity")
    age_seconds = None
    if isinstance(last, str):
        age_seconds = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds())
    return {
        "run_id": run_dir.name,
        "status": manifest["status"],
        "completed": len(completions),
        "started": len(dispatches),
        "total": total,
        "percent": 100.0 * len(completions) / total,
        "active_seed": progress.get("seed"),
        "active_arm": progress.get("arm"),
        "active_generation": progress.get("generation"),
        "last_activity": last,
        "last_activity_age_seconds": age_seconds,
        "failure_count": len(failures),
        "failure_classes": sorted({str(event.get("semantic_error")) for event in failures}),
        "eta": progress.get("eta", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
