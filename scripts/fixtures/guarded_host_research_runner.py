#!/usr/bin/env python3
"""Stable fixture runner for the guarded host launcher integration test."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--success", type=Path, required=True)
    parser.add_argument("--failure", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--progress-status", default="complete")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--require-isolated", action="store_true")
    args = parser.parse_args()
    if args.require_isolated and (sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1):
        raise SystemExit("isolated no-bytecode interpreter required")

    args.progress.parent.mkdir(parents=True, exist_ok=True)
    completed_units = 1 if args.fail else 2
    progress_status = "failed" if args.fail else args.progress_status
    args.progress.write_text(
        json.dumps(
            {
                "phase": "producer",
                "completed_units": completed_units,
                "total_units": 2,
                "throughput": 10.0,
                "eta_seconds": 0,
                "last_progress_at": datetime.now(timezone.utc).isoformat(),
                "status": progress_status,
            }
        ),
        encoding="utf-8",
    )
    if args.fail:
        if args.failure is None:
            raise SystemExit("failure path required")
        args.failure.write_text(
            json.dumps({"status": "failed", "workers": args.workers}),
            encoding="utf-8",
        )
        return 1
    args.success.write_text(
        json.dumps({"status": "complete", "workers": args.workers}),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
