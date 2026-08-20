"""Read-only progress summary for a B5-A run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_b4a import summarize as _summarize


def summarize(run_dir: Path) -> dict[str, object]:
    return _summarize(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
