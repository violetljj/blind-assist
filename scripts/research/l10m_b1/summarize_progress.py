"""Read-only progress summarizer for an L10M-B1 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(run_dir: Path) -> dict[str, object]:
    events_path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()] if events_path.exists() else []
    completions = [event for event in events if event.get("kind") == "completion"]
    total = 48
    by_arm = {}
    for arm in ("raw", "structured"):
        rows = [row for row in completions if row.get("arm") == arm]
        scores = [row["behavioral_score"] for row in rows if row.get("behavioral_score") is not None]
        by_arm[arm] = {
            "completed": len(rows),
            "target": 24,
            "valid": sum(bool(row.get("semantic_valid")) for row in rows),
            "unsafe": sum(bool(row.get("unsafe_candidate")) for row in rows),
            "best_score": max(scores) if scores else None,
        }
    last = completions[-1] if completions else None
    return {
        "run_id": run_dir.name,
        "completed_evaluations": len(completions),
        "total_evaluations": total,
        "percent": round(100 * len(completions) / total, 2),
        "active": None if last is None else {"seed": last.get("seed"), "arm": last.get("arm"), "generation": last.get("generation")},
        "last_activity": None if last is None else last.get("completed_at"),
        "failure_classes": {
            "semantic_invalid": sum(not bool(row.get("semantic_valid")) for row in completions),
            "unsafe": sum(bool(row.get("unsafe_candidate")) for row in completions),
            "provider_nonzero": sum(row.get("returncode") not in (0, None) for row in completions),
        },
        "eta": "unknown" if len(completions) < total else "0s",
        "by_arm": by_arm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
