#!/usr/bin/env python3
"""Run exactly one missing R3 detector-input shard, then exit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import exploratory_profiles_r2_l1 as r1
import l1e_materialization_recovery_r3_remaining as continuation
import run_l1e_materialization_recovery_r3_one_shard as frozen_materializer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    config = continuation.verify_config(repo, config_path)
    materializer_config_path = continuation.materializer_config_path(repo, config)
    materializer_config = frozen_materializer.verify_overlay(
        repo, materializer_config_path
    )
    root = continuation.output_root(repo, materializer_config)
    maximum_attempts = int(config["execution"]["initial_attempts_per_ledger"]) + int(
        config["execution"]["bounded_retries_per_ledger"]
    )

    with continuation.exclusive_shard_lock(root):
        before, _ = continuation.coverage(repo, materializer_config)
        if before["complete"]:
            print(json.dumps({"status": "ALL_41_INPUT_LEDGERS_ALREADY_COMPLETE"}))
            return 0
        selected = before["next_missing"]
        attempt_number, attempt_dir = continuation.create_control_attempt(
            root, selected, maximum_attempts
        )
        receipt_path = attempt_dir / "control-receipt.json"
        receipt = {
            "schema": (
                "blindassist_ustrf_l1e_materialization_recovery_r3_"
                "continuation_child_receipt_a1"
            ),
            "stage": continuation.STAGE,
            "status": "STARTED",
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "pid": os.getpid(),
            "attempt_number": attempt_number,
            "config_sha256": r1.sha256_file(config_path),
            "selected": selected,
            "coverage_before": before,
            "coverage_after": None,
            "error": None,
            "authority": config["authority"],
        }
        r1.atomic_write_json(receipt_path, receipt)

        def select_exactly_one(
            selected_repo: Path, _base_prereg_path: Path
        ):
            if selected_repo.resolve() != repo:
                raise continuation.ContinuationError(
                    "materializer_repo_changed_after_selection"
                )
            return continuation.selected_crowdbot_input(
                repo, materializer_config, selected
            )

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                str(materializer_config_path),
                "--config",
                str(materializer_config_path),
                "--repo",
                str(repo),
            ]
            with mock.patch.object(
                frozen_materializer.recovery,
                "first_missing_crowdbot",
                side_effect=select_exactly_one,
            ):
                result = frozen_materializer.main()
            if result != 0:
                raise continuation.ContinuationError(
                    f"frozen_materializer_nonzero:{result}"
                )
            after, _ = continuation.coverage(repo, materializer_config)
            if (
                after["verified_ledgers"] != before["verified_ledgers"] + 1
                or after["verified_frames"]
                != before["verified_frames"] + int(selected["frame_count"])
            ):
                raise continuation.ContinuationError(
                    "one_process_did_not_add_exactly_one_validated_shard"
                )
            receipt.update(
                {
                    "status": "ONE_SHARD_MATERIALIZED_AND_PROCESS_EXITING",
                    "finished_at_utc": utc_now(),
                    "coverage_after": after,
                }
            )
            r1.atomic_write_json(receipt_path, receipt)
            print(json.dumps(receipt, sort_keys=True))
            return 0
        except BaseException as error:
            after, _ = continuation.coverage(repo, materializer_config)
            receipt.update(
                {
                    "status": "FAIL_CLOSED_ONE_SHARD",
                    "finished_at_utc": utc_now(),
                    "coverage_after": after,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            r1.atomic_write_json(receipt_path, receipt)
            print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
            return 1
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

