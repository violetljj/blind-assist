#!/usr/bin/env python3
"""Stable adapter for D0 ego-motion error attribution R2 runtime recovery."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


MODULE_DIR = (
    Path(__file__).resolve().parent
    / "research"
    / "dual_loop_d0_egomotion_error_attribution_r2"
)
COMMANDS = {
    "activate": "create_activation",
    "create-implementation-lock": "create_implementation_lock",
    "produce": "runner",
    "runtime": "runtime_environment",
    "test": "run_tests",
    "validate-execution": "validate_execution_independent",
    "validate-implementation": "validate_implementation_lock",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{choices}> [args...]")
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    sys.path.insert(0, str(MODULE_DIR))
    sys.path.insert(1, str(MODULE_DIR.parents[2]))
    return int(importlib.import_module(COMMANDS[command]).main())


if __name__ == "__main__":
    raise SystemExit(main())
