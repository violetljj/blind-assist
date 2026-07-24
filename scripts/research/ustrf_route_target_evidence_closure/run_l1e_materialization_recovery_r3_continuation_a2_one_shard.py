#!/usr/bin/env python3
"""Run exactly one A2 R3 shard using short controller receipt paths."""

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
import l1e_materialization_recovery_r3_continuation_a2 as a2
import run_l1e_materialization_recovery_r3_one_shard as frozen_materializer


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    config = a2.verify_config(repo, config_path)
    materializer_path = a2.materializer_config_path(repo, config)
    parent_materializer_config = frozen_materializer.verify_overlay(
        repo, materializer_path
    )
    materializer_config = parent_materializer_config
    parent_canary_path = (
        repo / parent_materializer_config["canary_config"]["path"]
    ).resolve()
    parent_canary_config = frozen_materializer.recovery.verify_config(
        repo, parent_canary_path
    )
    amended_overlay = a2.amended_materializer_overlay(parent_materializer_config)
    amended_canary = a2.amended_canary_config(parent_canary_config)
    root = a2.output_root(repo, materializer_config)
    with a2.exclusive_shard_lock(root):
        before, _ = a2.coverage(repo, materializer_config)
        if before["complete"]:
            print(json.dumps({"status": "ALL_41_INPUT_LEDGERS_ALREADY_COMPLETE"}))
            return 0
        selected = before["next_missing"]
        attempt_number, _, receipt_path = a2.create_control_attempt(root, selected)
        receipt = {
            "schema": (
                "blindassist_ustrf_l1e_materialization_recovery_r3_"
                "continuation_child_receipt_a2"
            ),
            "stage": a2.STAGE,
            "status": "STARTED",
            "started_at_utc": now(),
            "finished_at_utc": None,
            "pid": os.getpid(),
            "cumulative_attempt_number": attempt_number,
            "config_sha256": r1.sha256_file(config_path),
            "host_memory_guard_amendment": {
                "parent_required_bytes": a2.PARENT_MEMORY_GUARD_BYTES,
                "a2_required_bytes": a2.A2_MEMORY_GUARD_BYTES,
                "authority": "explicit_user_direction_2026-07-24",
            },
            "selected": selected,
            "coverage_before": before,
            "coverage_after": None,
            "error": None,
            "authority": config["authority"],
        }
        a2.atomic_write(receipt_path, receipt)

        def select_exactly_one(selected_repo: Path, _base_prereg_path: Path):
            if selected_repo.resolve() != repo:
                raise a2.ContinuationA2Error(
                    "materializer_repo_changed_after_a2_selection"
                )
            return a2.selected_crowdbot_input(
                repo, materializer_config, selected
            )

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                str(materializer_path),
                "--config",
                str(materializer_path),
                "--repo",
                str(repo),
            ]
            def verify_amended_overlay(
                selected_repo: Path, selected_path: Path
            ):
                if (
                    selected_repo.resolve() != repo
                    or selected_path.resolve() != materializer_path
                ):
                    raise a2.ContinuationA2Error(
                        "unexpected_materializer_overlay_during_a2"
                    )
                return amended_overlay

            def verify_amended_canary(
                selected_repo: Path, selected_path: Path
            ):
                if (
                    selected_repo.resolve() != repo
                    or selected_path.resolve() != parent_canary_path
                ):
                    raise a2.ContinuationA2Error(
                        "unexpected_canary_config_during_a2"
                    )
                return amended_canary

            with (
                mock.patch.object(
                    frozen_materializer,
                    "verify_overlay",
                    side_effect=verify_amended_overlay,
                ),
                mock.patch.object(
                    frozen_materializer.recovery,
                    "verify_config",
                    side_effect=verify_amended_canary,
                ),
                mock.patch.object(
                    frozen_materializer.recovery,
                    "first_missing_crowdbot",
                    side_effect=select_exactly_one,
                ),
            ):
                result = frozen_materializer.main()
            if result != 0:
                raise a2.ContinuationA2Error(
                    f"frozen_materializer_nonzero:{result}"
                )
            after, _ = a2.coverage(repo, materializer_config)
            if (
                after["verified_ledgers"] != before["verified_ledgers"] + 1
                or after["verified_frames"]
                != before["verified_frames"] + int(selected["frame_count"])
            ):
                raise a2.ContinuationA2Error(
                    "a2_process_did_not_add_exactly_one_validated_shard"
                )
            receipt.update(
                {
                    "status": "ONE_SHARD_MATERIALIZED_AND_PROCESS_EXITING",
                    "finished_at_utc": now(),
                    "coverage_after": after,
                }
            )
            a2.atomic_write(receipt_path, receipt)
            print(json.dumps(receipt, sort_keys=True))
            return 0
        except BaseException as error:
            after, _ = a2.coverage(repo, materializer_config)
            receipt.update(
                {
                    "status": "FAIL_CLOSED_ONE_SHARD",
                    "finished_at_utc": now(),
                    "coverage_after": after,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            a2.atomic_write(receipt_path, receipt)
            print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
            return 1
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
