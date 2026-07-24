"""Run the frozen eligible-attribution-first isolated one-shot diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eligible_target_attribution_ordered_isolated_opening import run


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
    result = run(Path(args.repo).resolve(), Path(args.config).resolve())
    print(
        json.dumps(
            {
                "terminal_state": result["terminal_state"],
                "mechanism_gate": result["mechanism_gate"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
