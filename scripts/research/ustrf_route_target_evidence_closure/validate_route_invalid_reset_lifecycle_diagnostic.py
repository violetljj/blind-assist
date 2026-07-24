#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from route_invalid_reset_lifecycle_diagnostic import (
    atomic_write_json,
    validate_written,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    validation = validate_written(
        args.repo.resolve(),
        args.config.resolve(),
        args.terminal.resolve(),
    )
    atomic_write_json(args.output.resolve(), validation)
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
