#!/usr/bin/env python3
"""Stable root adapter for dual-loop radial-geometry LITE R2."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


MODULE_DIR = (
    Path(__file__).resolve().parent
    / "research"
    / "dual_loop_radial_geometry_lite_r2"
)
COMMANDS = {
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
    return int(importlib.import_module(COMMANDS[command]).main())


if __name__ == "__main__":
    raise SystemExit(main())
