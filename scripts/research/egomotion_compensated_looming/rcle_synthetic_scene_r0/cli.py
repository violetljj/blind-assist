from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .dataset import build_dataset_governed, validate_sealed_activation
from .evaluator import evaluate_dataset
from .io_utils import write_json
from .sealed_executor import execute_sealed_once
from .validator import compare_development_generations, validate_dataset


DEFAULT_SPEC = Path(__file__).with_name("dataset_spec.json")


def _print(value: object) -> None:
    print(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--split", choices=("development", "sealed"), default="development"
    )
    generate.add_argument("--activation", type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--check-algorithm", action="store_true")

    compare = subparsers.add_parser("compare-determinism")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path)

    sealed_preflight = subparsers.add_parser("preflight-sealed")
    sealed_preflight.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    sealed_preflight.add_argument("--output", type=Path, required=True)
    sealed_preflight.add_argument("--activation", type=Path, required=True)

    execute_sealed = subparsers.add_parser("execute-sealed")
    execute_sealed.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    execute_sealed.add_argument("--output", type=Path, required=True)
    execute_sealed.add_argument("--activation", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "generate":
        if args.split == "sealed":
            raise PermissionError(
                "SEALED_GENERATION_REQUIRES_EXECUTE_SEALED"
            )
        _print(
            build_dataset_governed(
                args.spec, args.output, args.split, args.activation
            )
        )
    elif args.command == "evaluate":
        _print(evaluate_dataset(args.dataset))
    elif args.command == "validate":
        _print(
            validate_dataset(
                args.dataset,
                check_algorithm=args.check_algorithm,
            )
        )
    elif args.command == "compare-determinism":
        result = compare_development_generations(args.first, args.second)
        if args.output is not None:
            write_json(args.output, result)
        _print(result)
    elif args.command == "preflight-sealed":
        result = validate_sealed_activation(
            args.spec, args.output, args.activation
        )
        _print(result)
    elif args.command == "execute-sealed":
        _print(
            execute_sealed_once(
                args.spec, args.output, args.activation
            )
        )
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
