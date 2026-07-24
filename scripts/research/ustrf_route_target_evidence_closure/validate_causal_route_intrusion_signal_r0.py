#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import causal_route_intrusion_signal_r0 as core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ustrf_causal_route_intrusion_signal_r0.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = args.config if args.config.is_absolute() else repo / args.config
    result = core.validate_outputs(repo, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
