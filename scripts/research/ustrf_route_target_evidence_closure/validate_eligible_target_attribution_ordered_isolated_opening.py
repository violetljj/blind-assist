"""Validate the eligible-attribution-first isolated one-shot diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eligible_target_attribution_ordered_isolated_opening import (
    validate_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "ustrf_eligible_target_attribution_ordered_isolated_opening_r1.json"
        ),
    )
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    result = validate_outputs(
        Path(args.repo).resolve(), Path(args.config).resolve()
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
