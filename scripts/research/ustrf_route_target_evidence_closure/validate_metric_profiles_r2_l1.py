#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metric_profiles_r2_l1 import (
    atomic_write_json,
    validate_terminal_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt = args.receipt or (
        repo
        / "artifacts.local/evidence/"
        "ustrf-route-target-r2-l1-metric-profile-r1/terminal-receipt-r1.json"
    )
    result = validate_terminal_receipt(
        repo, args.config.resolve(), receipt.resolve()
    )
    validation_path = receipt.with_name("validation-receipt-r1.json")
    atomic_write_json(validation_path, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
