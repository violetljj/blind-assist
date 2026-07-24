#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from causal_per_track_attribution_token_audit_r0 import (
    build_and_freeze_blind_inventory,
    build_terminal_from_frozen_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/ustrf_truth_blind_causal_per_track_"
            "attribution_token_producer_audit_r0.json"
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("producer", "audit"),
        required=True,
        help=(
            "Run producer first in one process; only a later audit process may "
            "decode oracle/negative-exposure evidence."
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = (args.repo / args.config).resolve()
    if args.phase == "producer":
        inventory = build_and_freeze_blind_inventory(repo, config)
        print(inventory["status"])
    else:
        terminal = build_terminal_from_frozen_inventory(repo, config)
        print(terminal["terminal_state"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
