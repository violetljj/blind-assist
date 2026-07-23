"""Write the candidate-blind R2-L1 eligibility mask and denominator receipt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from contract import ContractError, load_json
from metric_eligibility import json_bytes, materialize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    try:
        config = load_json(config_path)
        mask, receipt = materialize(
            config, repo=repo, config_path=config_path
        )
        output = repo / str(config["outputs"]["root"])
        output.mkdir(parents=True, exist_ok=True)
        mask_path = output / str(config["outputs"]["event_mask"])
        receipt_path = output / str(config["outputs"]["denominator_receipt"])
        mask_path.write_bytes(json_bytes(mask))
        receipt_path.write_bytes(json_bytes(receipt))
    except (ContractError, KeyError, OSError, ValueError) as exc:
        print(f"INVALID_METRIC_ELIGIBILITY_R2_L1: {exc}", file=sys.stderr)
        return 2
    print(
        "MATERIALIZED_METRIC_ELIGIBILITY_R2_L1 "
        f"events={mask['event_count']} "
        f"event_metric_cells={mask['event_metric_classification_count']} "
        f"mask={mask_path} receipt={receipt_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
