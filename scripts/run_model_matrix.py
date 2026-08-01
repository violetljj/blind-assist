#!/usr/bin/env python3
"""Stable root entrypoint for the manifest-driven model matrix runner."""

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.research.model_matrix.run_model_matrix import main


if __name__ == "__main__":
    raise SystemExit(main())
