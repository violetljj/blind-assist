#!/usr/bin/env python3
"""Stable adapter for production temporal geometry factorial A/B R0."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


MODULE_DIR = Path(__file__).resolve().parent / "research" / "dual_loop_production_temporal_ab"
COMMANDS = {
    "activate": "activate",
    "create-implementation-lock": "create_implementation_lock",
    "validate-producer": "validate_producer",
    "evaluate": "evaluate_trace",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{choices}> [args...]")
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    sys.path.insert(0, str(MODULE_DIR))
    return int(importlib.import_module(COMMANDS[command]).main())


if __name__ == "__main__":
    raise SystemExit(main())
