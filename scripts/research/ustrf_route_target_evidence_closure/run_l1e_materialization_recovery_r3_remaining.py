#!/usr/bin/env python3
"""Serially spawn one fresh host process per remaining R3 input shard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import exploratory_profiles_r2_l1 as r1
import l1e_materialization_recovery_r3_remaining as continuation
import run_l1e_materialization_recovery_r3_one_shard as frozen_materializer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress(
    path: Path,
    config_path: Path,
    config: dict,
    starting: dict,
    current: dict,
    successful_children: int,
    failed_children: int,
    status: str,
) -> dict:
    payload = {
        "schema": (
            "blindassist_ustrf_l1e_materialization_recovery_r3_"
            "continuation_progress_a1"
        ),
        "stage": continuation.STAGE,
        "status": status,
        "updated_at_utc": utc_now(),
        "config_sha256": r1.sha256_file(config_path),
        "starting_coverage": starting,
        "current_coverage": current,
        "successful_child_processes": successful_children,
        "failed_child_processes": failed_children,
        "one_process_per_successful_shard": (
            successful_children
            == current["verified_ledgers"] - starting["verified_ledgers"]
        ),
        "candidate_execution_started": False,
        "authority": config["authority"],
    }
    r1.atomic_write_json(path, payload)
    return payload


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
    progress_path = root / "continuation-progress-a1.json"
    terminal_path = root / "continuation-terminal-a1.json"
    child_runner = (
        repo
        / config["implementation_bindings"]["remaining_one_shard_runner"]["path"]
    ).resolve()
    starting, _ = continuation.coverage(repo, materializer_config)
    if starting["verified_ledgers"] < config["expected_coverage"]["starting_ledgers"]:
        raise continuation.ContinuationError(
            "continuation_starting_ledger_coverage_regressed"
        )
    if starting["verified_frames"] < config["expected_coverage"]["starting_frames"]:
        raise continuation.ContinuationError(
            "continuation_starting_frame_coverage_regressed"
        )
    if starting["missing_ledgers"] > int(
        config["execution"]["maximum_total_remaining_crowdbot_shards"]
    ):
        raise continuation.ContinuationError(
            "continuation_missing_shards_exceed_frozen_maximum"
        )

    successful_children = 0
    failed_children = 0
    current = starting
    write_progress(
        progress_path,
        config_path,
        config,
        starting,
        current,
        successful_children,
        failed_children,
        "RUNNING" if not current["complete"] else "COMPLETE",
    )
    while not current["complete"]:
        selected = current["next_missing"]
        maximum_attempts = int(config["execution"]["initial_attempts_per_ledger"]) + int(
            config["execution"]["bounded_retries_per_ledger"]
        )
        attempts_before = continuation.count_control_attempts(
            root, selected["source_id"], selected["sequence_id"]
        )
        if attempts_before >= maximum_attempts:
            raise continuation.ContinuationError(
                "continuation_selected_ledger_retry_limit_already_exhausted"
            )
        completed = subprocess.run(
            [
                sys.executable,
                str(child_runner),
                "--config",
                str(config_path),
                "--repo",
                str(repo),
            ],
            cwd=repo,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        log_root = root / "continuation-child-logs"
        log_root.mkdir(parents=True, exist_ok=True)
        slug = r1.stable_slug(selected["source_id"], selected["sequence_id"])
        attempts_after = continuation.count_control_attempts(
            root, selected["source_id"], selected["sequence_id"]
        )
        log_path = log_root / f"{slug}.attempt-{attempts_after:03d}.log"
        log_path.write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
        )
        updated, _ = continuation.coverage(repo, materializer_config)
        if completed.returncode == 0:
            if (
                updated["verified_ledgers"] != current["verified_ledgers"] + 1
                or updated["verified_frames"]
                != current["verified_frames"] + int(selected["frame_count"])
            ):
                raise continuation.ContinuationError(
                    "successful_child_did_not_add_exactly_one_shard"
                )
            successful_children += 1
            current = updated
            progress = write_progress(
                progress_path,
                config_path,
                config,
                starting,
                current,
                successful_children,
                failed_children,
                "RUNNING" if not current["complete"] else "COMPLETE",
            )
            print(json.dumps(progress, sort_keys=True), flush=True)
            continue

        failed_children += 1
        current = updated
        progress = write_progress(
            progress_path,
            config_path,
            config,
            starting,
            current,
            successful_children,
            failed_children,
            "RETRYING_ONE_LEDGER",
        )
        print(json.dumps(progress, sort_keys=True), flush=True)
        if current["next_missing"] != selected:
            raise continuation.ContinuationError(
                "failed_child_changed_next_missing_ledger"
            )
        if attempts_after >= maximum_attempts:
            failure = {
                **progress,
                "schema": (
                    "blindassist_ustrf_l1e_materialization_recovery_r3_"
                    "continuation_terminal_a1"
                ),
                "status": "FAIL_CLOSED_LEDGER_ATTEMPTS_EXHAUSTED",
                "failed_ledger": selected,
                "last_child_log": str(log_path.relative_to(repo)).replace("\\", "/"),
            }
            r1.atomic_write_json(terminal_path, failure)
            print(json.dumps(failure, sort_keys=True), file=sys.stderr)
            return 1

    if (
        current["verified_ledgers"] != continuation.EXPECTED_LEDGERS
        or current["verified_frames"] != continuation.EXPECTED_FRAMES
        or current["discontinuity_resets"] != continuation.EXPECTED_RESETS
        or current["missing_ledgers"] != 0
    ):
        raise continuation.ContinuationError(
            "full_coverage_gate_not_satisfied_after_materialization"
        )
    terminal = {
        "schema": (
            "blindassist_ustrf_l1e_materialization_recovery_r3_"
            "continuation_terminal_a1"
        ),
        "stage": continuation.STAGE,
        "status": "CANONICAL_INPUT_41_OF_41_COMPLETE",
        "finished_at_utc": utc_now(),
        "config_sha256": r1.sha256_file(config_path),
        "starting_coverage": starting,
        "final_coverage": current,
        "successful_child_processes": successful_children,
        "failed_child_processes": failed_children,
        "one_process_per_successful_shard": (
            successful_children
            == current["verified_ledgers"] - starting["verified_ledgers"]
        ),
        "c1_c2_c3_executed": False,
        "candidate_trace_count": 0,
        "candidate_profile_count": 0,
        "authority": config["authority"],
    }
    r1.atomic_write_json(terminal_path, terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

