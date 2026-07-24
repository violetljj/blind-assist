#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from route_invalid_reset_lifecycle_diagnostic import (
    build_expected,
    write_expected,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    terminal, outputs = build_expected(repo, config_path)
    terminal_path = write_expected(repo, config_path, terminal, outputs)
    print(
        json.dumps(
            {
                "terminal_state": terminal["terminal_state"],
                "overall_mechanism_gate_passed": terminal[
                    "overall_mechanism_gate_passed"
                ],
                "candidate_reruns": 0,
                "detector_reruns": 0,
                "terminal_receipt": str(terminal_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
