"""Stable unittest entry point for the archived public-video research module."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
CAMPAIGN_DIR = SCRIPTS_DIR / "research" / "public_video"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args()

    sys.path[:0] = [str(CAMPAIGN_DIR), str(SCRIPTS_DIR), str(REPO_ROOT)]
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CAMPAIGN_DIR),
        pattern=args.pattern,
        top_level_dir=str(CAMPAIGN_DIR),
    )
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
