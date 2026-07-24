#!/usr/bin/env python3
"""Independently recompute the frozen delivery-failure attribution."""

from __future__ import annotations

import argparse
from pathlib import Path

from known_route_eligible_delivery_failure_attribution import (
    atomic_write_json,
    load_json,
    validate_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/"
            "ustrf_route_target_known_route_eligible_delivery_"
            "failure_attribution_r1.json"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (
        args.config
        if args.config.is_absolute()
        else repo / args.config
    )
    config = load_json(config_path)
    output_root = repo / config["outputs"]["root"]
    receipt = validate_outputs(
        repo,
        config_path,
        output_root / config["outputs"]["terminal_receipt"],
    )
    validation_path = output_root / config["outputs"]["validation_receipt"]
    atomic_write_json(validation_path, receipt)
    print(validation_path)
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
