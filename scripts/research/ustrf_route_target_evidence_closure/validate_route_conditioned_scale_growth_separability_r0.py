#!/usr/bin/env python3
"""Independently validate the fail-closed scale-growth R0 terminal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import route_conditioned_scale_growth_separability_r0 as core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = core.validate_blocked_outputs(
        args.repo.resolve(), args.config.resolve()
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
