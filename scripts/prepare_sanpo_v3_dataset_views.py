#!/usr/bin/env python3
"""Split reviewed v3 rows into trainer-visible and benchmark-only manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True, help="Reviewed source metadata; do not pass this path to training.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_manifest.resolve()
    root = args.dataset_root.resolve()
    training_path = root / "training_manifest.jsonl"
    blind_path = root / "blind_holdout" / "manifest.jsonl"
    if training_path.exists() or blind_path.exists():
        raise SystemExit("refusing to overwrite existing v3 manifest views")
    rows = load_jsonl(source)
    training = [row for row in rows if row.get("split") in {"train", "dev"}]
    blind = [row for row in rows if row.get("split") == "blind"]
    unexpected = [row.get("id") for row in rows if row.get("split") not in {"train", "dev", "blind"}]
    if unexpected or not training or not blind:
        raise SystemExit("source manifest must contain non-empty train/dev and blind rows only")
    blind_sessions = sorted({str(row.get("session_id") or row.get("source", {}).get("session_id") or "").strip() for row in blind})
    if len(blind_sessions) != 2 or not all(blind_sessions):
        raise SystemExit("v3 requires exactly two non-empty blind source sessions before creating views")
    write_jsonl(training_path, training)
    write_jsonl(blind_path, blind)
    policy = {
        "format": "blindassist_sanpo_v3_access_policy_v2",
        "training_manifest": "training_manifest.jsonl",
        "blind_manifest": "blind_holdout/manifest.jsonl",
        "blind_label_access": "benchmark_only",
        "forbidden_training_paths": ["blind_holdout"],
        "forbidden_threshold_selection_paths": ["blind_holdout"],
        "benchmark_only_sessions": blind_sessions,
        "forbidden_training_sessions": blind_sessions,
        "forbidden_threshold_selection_sessions": blind_sessions,
        "trainer_contract": "Training accepts only --dataset-root, runs the SHA256 total gate, and opens only training_manifest.jsonl.",
        "threshold_selection_contract": "Threshold selection receives train/dev only and must never inspect blind_holdout paths, labels, or sessions.",
    }
    (root / "access_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"v3_views_ok=true training_rows={len(training)} blind_rows={len(blind)} root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
