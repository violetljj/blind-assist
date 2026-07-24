#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from causal_token_policy_risk_gate_r1 import (
    build_and_freeze_policy_inventory,
    build_terminal_from_frozen_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/ustrf_candidate_independent_causal_token_"
            "policy_risk_gate_r1.json"
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("producer", "audit"),
        required=True,
        help=(
            "Run producer first; only a later process may decode oracle or "
            "negative-exposure evidence."
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = (repo / args.config).resolve()
    if args.phase == "producer":
        result = build_and_freeze_policy_inventory(repo, config)
        print(result["status"])
    else:
        result = build_terminal_from_frozen_inventory(repo, config)
        print(result["terminal_state"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
