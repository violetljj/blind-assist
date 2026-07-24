#!/usr/bin/env python3
"""Run one A3 shard with 4 GiB host guard and extended-path atomic writes."""

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
import l1e_materialization_recovery_r3_continuation_a3 as a3
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
    config = a3.verify_config(repo, config_path)
    materializer_path = a3.materializer_config_path(repo, config)
    parent_overlay = frozen_materializer.verify_overlay(repo, materializer_path)
    parent_canary_path = (repo / parent_overlay["canary_config"]["path"]).resolve()
    parent_canary = frozen_materializer.recovery.verify_config(
        repo, parent_canary_path
    )
    amended_overlay = a3.a2.amended_materializer_overlay(parent_overlay)
    amended_canary = a3.a2.amended_canary_config(parent_canary)
    root = a3.output_root(repo, parent_overlay)
    with a3.exclusive_shard_lock(root):
        before, _ = a3.coverage(repo, parent_overlay)
        if before["complete"]:
            print(json.dumps({"status": "ALL_41_INPUT_LEDGERS_ALREADY_COMPLETE"}))
            return 0
        selected = before["next_missing"]
        attempt_number, receipt_path = a3.create_control_attempt(root, selected)
        receipt = {
            "schema": (
                "blindassist_ustrf_l1e_materialization_recovery_r3_"
                "continuation_child_receipt_a3"
            ),
            "stage": a3.STAGE,
            "status": "STARTED",
            "started_at_utc": now(),
            "finished_at_utc": None,
            "pid": os.getpid(),
            "cumulative_attempt_number": attempt_number,
            "config_sha256": r1.sha256_file(config_path),
            "selected": selected,
            "coverage_before": before,
            "coverage_after": None,
            "host_memory_guard_bytes": a3.A3_MEMORY_GUARD_BYTES,
            "windows_extended_path_atomic_writes": True,
            "error": None,
            "authority": config["authority"],
        }
        a3.atomic_write(receipt_path, receipt)

        def select_exactly_one(selected_repo: Path, _base: Path):
            if selected_repo.resolve() != repo:
                raise a3.ContinuationA3Error("a3_repo_changed_after_selection")
            return a3.selected_crowdbot_input(repo, parent_overlay, selected)

        def verify_overlay(selected_repo: Path, selected_path: Path):
            if (
                selected_repo.resolve() != repo
                or selected_path.resolve() != materializer_path
            ):
                raise a3.ContinuationA3Error("unexpected_a3_materializer_overlay")
            return amended_overlay

        def verify_canary(selected_repo: Path, selected_path: Path):
            if (
                selected_repo.resolve() != repo
                or selected_path.resolve() != parent_canary_path
            ):
                raise a3.ContinuationA3Error("unexpected_a3_canary_config")
            return amended_canary

        original_atomic = frozen_materializer.r1.atomic_write_json

        def extended_atomic(path: Path, payload):
            a3.atomic_write_with_extended_path(original_atomic, path, payload)

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                str(materializer_path),
                "--config",
                str(materializer_path),
                "--repo",
                str(repo),
            ]
            with (
                mock.patch.object(
                    frozen_materializer, "verify_overlay", side_effect=verify_overlay
                ),
                mock.patch.object(
                    frozen_materializer.recovery,
                    "verify_config",
                    side_effect=verify_canary,
                ),
                mock.patch.object(
                    frozen_materializer.recovery,
                    "first_missing_crowdbot",
                    side_effect=select_exactly_one,
                ),
                mock.patch.object(
                    frozen_materializer.r1,
                    "atomic_write_json",
                    side_effect=extended_atomic,
                ),
            ):
                result = frozen_materializer.main()
            if result != 0:
                raise a3.ContinuationA3Error(f"materializer_nonzero:{result}")
            after, _ = a3.coverage(repo, parent_overlay)
            if (
                after["verified_ledgers"] != before["verified_ledgers"] + 1
                or after["verified_frames"]
                != before["verified_frames"] + int(selected["frame_count"])
            ):
                raise a3.ContinuationA3Error(
                    "a3_process_did_not_add_exactly_one_shard"
                )
            receipt.update(
                {
                    "status": "ONE_SHARD_MATERIALIZED_AND_PROCESS_EXITING",
                    "finished_at_utc": now(),
                    "coverage_after": after,
                }
            )
            a3.atomic_write(receipt_path, receipt)
            print(json.dumps(receipt, sort_keys=True))
            return 0
        except BaseException as error:
            after, _ = a3.coverage(repo, parent_overlay)
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
            a3.atomic_write(receipt_path, receipt)
            print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
            return 1
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

