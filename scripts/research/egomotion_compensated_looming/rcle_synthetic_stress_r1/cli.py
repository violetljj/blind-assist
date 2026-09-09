from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .builder import DEFAULT_SPEC, build_stress_dataset
from .evaluator import evaluate_stress_dataset
from .qa import build_condition_montages
from .validator import validate_stress_dataset


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    qa = subparsers.add_parser("qa")
    qa.add_argument("--dataset", type=Path, required=True)
    qa.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        _print(build_stress_dataset(args.output, args.spec))
    elif args.command == "evaluate":
        _print(evaluate_stress_dataset(args.dataset))
    elif args.command == "validate":
        result = validate_stress_dataset(args.dataset)
        _print(result)
        return 0 if result["status"] == "PASS" else 1
    elif args.command == "qa":
        _print(build_condition_montages(args.dataset, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
