#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from candidate_independent_policy_failure_attribution_r1 import validate_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/ustrf_candidate_independent_policy_failure_"
            "attribution_r1.json"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    receipt = validate_outputs(repo, (repo / args.config).resolve())
    print(receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
