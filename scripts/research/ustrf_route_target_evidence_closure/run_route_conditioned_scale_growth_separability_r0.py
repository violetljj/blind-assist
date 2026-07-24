#!/usr/bin/env python3
"""Run one isolated phase of route-conditioned scale growth R0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import route_conditioned_scale_growth_separability_r0 as core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("producer", "audit"), required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config.resolve()
    result = (
        core.run_blocking_preflight(repo, config)
        if args.phase == "producer"
        else core.run_blocked_audit(repo, config)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
