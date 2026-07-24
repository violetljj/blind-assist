#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metric_profiles_r2_l1 import (
    atomic_write_json,
    build_terminal_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    receipt, profiles = build_terminal_receipt(repo, config_path)
    output_root = (
        repo
        / "artifacts.local/evidence/"
        "ustrf-route-target-r2-l1-metric-profile-r1"
    )
    for candidate_id, profile in profiles.items():
        atomic_write_json(output_root / "profiles" / f"{candidate_id}.json", profile)
    terminal_path = output_root / "terminal-receipt-r1.json"
    atomic_write_json(terminal_path, receipt)
    print(
        json.dumps(
            {
                "terminal_state": receipt["terminal_state"],
                "profiles": len(profiles),
                "candidate_reruns": 0,
                "receipt": str(terminal_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
