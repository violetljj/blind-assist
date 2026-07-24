#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from causal_per_track_attribution_token_audit_r0 import validate_outputs


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
    args = parser.parse_args()
    receipt = validate_outputs(
        args.repo.resolve(), (args.repo / args.config).resolve()
    )
    print(receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
