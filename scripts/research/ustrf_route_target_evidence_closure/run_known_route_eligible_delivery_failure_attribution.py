#!/usr/bin/env python3
"""Run the frozen known-route eligible-delivery failure attribution."""

from __future__ import annotations

import argparse
from pathlib import Path

from known_route_eligible_delivery_failure_attribution import (
    atomic_write_json,
    build_and_write_blind_inventory,
    build_and_write_event_scope_blind_pack,
    build_terminal_from_persisted_blind,
    load_json,
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
    build_and_write_blind_inventory(repo, config_path)
    build_and_write_event_scope_blind_pack(repo, config_path)
    config = load_json(config_path)
    terminal = build_terminal_from_persisted_blind(repo, config_path)
    terminal_path = (
        repo
        / config["outputs"]["root"]
        / config["outputs"]["terminal_receipt"]
    )
    atomic_write_json(terminal_path, terminal)
    print(terminal_path)
    print(terminal["aggregate_label_counts"])
    print(terminal["attribution_gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
