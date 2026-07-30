from __future__ import annotations

import argparse
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbosity", type=int, default=2)
    args = parser.parse_args(argv)
    suite = unittest.defaultTestLoader.discover(
        str(MODULE_DIR),
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
