#!/usr/bin/env python3
"""Run USTRF G0-A in an isolated process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import canonical_observation_authority_inventory_r0 as core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = core.build_inventory(args.repo.resolve(), args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
