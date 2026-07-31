"""Stable root adapter for target-local warp residual R0 Development."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.research.dual_loop_target_local_background_warp_residual_r0 import evaluate, produce, test_evaluate, test_pipeline, test_prepare_burned_revel, test_produce
from scripts.research.dual_loop_target_local_background_warp_residual_r0.create_implementation_lock import main as create_lock
from scripts.research.dual_loop_target_local_background_warp_residual_r0.prepare_burned_revel import main as prepare_burned
from scripts.research.dual_loop_target_local_background_warp_residual_r0.validate_implementation_lock import main as validate_lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce_parser = subparsers.add_parser("produce")
    produce_parser.add_argument("--input", required=True)
    produce_parser.add_argument("--output", required=True)
    produce_parser.add_argument("--receipt", required=True)
    produce_parser.add_argument("--repo-root")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--producer-output", required=True)
    evaluate_parser.add_argument("--producer-receipt", required=True)
    evaluate_parser.add_argument("--truth", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--repo-root")
    lock_parser = subparsers.add_parser("create-lock")
    lock_parser.add_argument("--output", required=True)
    lock_parser.add_argument("--repo-root")
    validate_parser = subparsers.add_parser("validate-implementation")
    validate_parser.add_argument("--lock", required=True)
    validate_parser.add_argument("--repo-root")
    prepare_parser = subparsers.add_parser("prepare-burned")
    prepare_parser.add_argument("prepare_command", choices=["input", "truth"])
    prepare_parser.add_argument("args", nargs=argparse.REMAINDER)
    subparsers.add_parser("test")
    args = parser.parse_args(argv)
    if args.command == "produce":
        return produce.main(["--input", args.input, "--output", args.output, "--receipt", args.receipt] + (["--repo-root", args.repo_root] if args.repo_root else []))
    if args.command == "evaluate":
        return evaluate.main(["--producer-output", args.producer_output, "--producer-receipt", args.producer_receipt, "--truth", args.truth, "--output", args.output] + (["--repo-root", args.repo_root] if args.repo_root else []))
    if args.command == "create-lock":
        return create_lock(["--output", args.output] + (["--repo-root", args.repo_root] if args.repo_root else []))
    if args.command == "validate-implementation":
        return validate_lock(["--lock", args.lock] + (["--repo-root", args.repo_root] if args.repo_root else []))
    if args.command == "prepare-burned":
        return prepare_burned([args.prepare_command] + args.args)
    suite = unittest.TestSuite(
        [
            unittest.defaultTestLoader.loadTestsFromModule(test_produce),
            unittest.defaultTestLoader.loadTestsFromModule(test_prepare_burned_revel),
            unittest.defaultTestLoader.loadTestsFromModule(test_evaluate),
            unittest.defaultTestLoader.loadTestsFromModule(test_pipeline),
        ]
    )
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
