#!/usr/bin/env python3
"""Run USTRF G0-B after verifying the frozen A inventory first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import canonical_observation_denominator_availability_r0 as core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    args = parser.parse_args()
    result = core.run_availability(
        args.repo.resolve(),
        args.config.resolve(),
        args.inventory.resolve(),
        args.inventory_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
