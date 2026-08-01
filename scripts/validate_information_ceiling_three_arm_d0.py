#!/usr/bin/env python3
"""Stable CLI adapter for the information-ceiling three-arm validator."""

from __future__ import annotations

import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent / "research" / "information_ceiling_three_arm_d0"
sys.path.insert(0, str(MODULE_DIR))

from validate_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
