#!/usr/bin/env python3
"""Validate a hash-bound fresh VITG G0 RGB/IMU capture without opening truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capture_contract import validate_capture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_capture(args.manifest.resolve(), args.protocol.resolve()), indent=2))


if __name__ == "__main__":
    main()
