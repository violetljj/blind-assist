#!/usr/bin/env python3
"""Stable root adapter for the frozen dual-loop radial-geometry R1 tools."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


MODULE_DIR = (
    Path(__file__).resolve().parent
    / "research"
    / "dual_loop_radial_geometry_lite_r1"
)
COMMANDS = {
    "audit-shapes": "audit_source_shapes",
    "produce": "run_replay",
    "evaluate": "evaluate_replay",
    "validate-implementation": "validate_implementation_lock",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{choices}> [args...]")
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    sys.path.insert(0, str(MODULE_DIR))
    module = importlib.import_module(COMMANDS[command])
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
