#!/usr/bin/env python3
"""Serial A2 orchestrator for the remaining R3 input shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import exploratory_profiles_r2_l1 as r1
import l1e_materialization_recovery_r3_continuation_a2 as a2
import run_l1e_materialization_recovery_r3_one_shard as frozen_materializer


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress(
    path: Path,
    config_path: Path,
    config: dict,
    starting: dict,
    current: dict,
    successful: int,
    failed: int,
    status: str,
) -> dict:
    payload = {
        "schema": (
            "blindassist_ustrf_l1e_materialization_recovery_r3_"
            "continuation_progress_a2"
        ),
        "stage": a2.STAGE,
        "status": status,
        "updated_at_utc": now(),
        "config_sha256": r1.sha256_file(config_path),
        "starting_coverage": starting,
        "current_coverage": current,
        "successful_child_processes": successful,
        "failed_child_processes": failed,
        "one_process_per_successful_shard": (
            successful
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
    config = a2.verify_config(repo, config_path)
    materializer_path = a2.materializer_config_path(repo, config)
    materializer_config = frozen_materializer.verify_overlay(repo, materializer_path)
    root = a2.output_root(repo, materializer_config)
    child_runner = (
        repo / config["implementation_bindings"]["a2_one_shard_runner"]["path"]
    ).resolve()
    progress_path = root / "continuation-progress-a2.json"
    terminal_path = root / "continuation-terminal-a2.json"
    starting, _ = a2.coverage(repo, materializer_config)
    if (
        starting["verified_ledgers"]
        < config["expected_coverage"]["starting_ledgers"]
        or starting["verified_frames"]
        < config["expected_coverage"]["starting_frames"]
    ):
        raise a2.ContinuationA2Error("a2_starting_coverage_regressed")
    if (
        starting["missing_ledgers"]
        > config["execution"]["maximum_remaining_crowdbot_shards"]
    ):
        raise a2.ContinuationA2Error("a2_missing_shards_exceed_frozen_maximum")

    current = starting
    successful = 0
    failed = 0
    write_progress(
        progress_path, config_path, config, starting, current, 0, 0,
        "RUNNING" if not current["complete"] else "COMPLETE",
    )
    while not current["complete"]:
        selected = current["next_missing"]
        count_before = a2.cumulative_attempt_count(
            root, selected["source_id"], selected["sequence_id"]
        )
        if count_before >= a2.MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS:
            raise a2.ContinuationA2Error(
                "a2_selected_ledger_cumulative_retry_limit_exhausted"
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
        count_after = a2.cumulative_attempt_count(
            root, selected["source_id"], selected["sequence_id"]
        )
        log_root = root / "logs-a2"
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = (
            log_root
            / f"{a2.short_id(selected['source_id'], selected['sequence_id'])}"
            f".a{count_after:03d}.log"
        )
        log_path.write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
        )
        updated, _ = a2.coverage(repo, materializer_config)
        if completed.returncode == 0:
            if (
                updated["verified_ledgers"] != current["verified_ledgers"] + 1
                or updated["verified_frames"]
                != current["verified_frames"] + int(selected["frame_count"])
            ):
                raise a2.ContinuationA2Error(
                    "a2_successful_child_did_not_add_exactly_one_shard"
                )
            successful += 1
            current = updated
            progress = write_progress(
                progress_path,
                config_path,
                config,
                starting,
                current,
                successful,
                failed,
                "RUNNING" if not current["complete"] else "COMPLETE",
            )
            print(json.dumps(progress, sort_keys=True), flush=True)
            continue

        failed += 1
        current = updated
        progress = write_progress(
            progress_path,
            config_path,
            config,
            starting,
            current,
            successful,
            failed,
            "RETRYING_ONE_LEDGER",
        )
        print(json.dumps(progress, sort_keys=True), flush=True)
        if current["next_missing"] != selected:
            raise a2.ContinuationA2Error(
                "a2_failed_child_changed_next_missing_ledger"
            )
        if count_after >= a2.MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS:
            terminal = {
                **progress,
                "schema": (
                    "blindassist_ustrf_l1e_materialization_recovery_r3_"
                    "continuation_terminal_a2"
                ),
                "status": "FAIL_CLOSED_LEDGER_ATTEMPTS_EXHAUSTED",
                "failed_ledger": selected,
                "last_child_log": str(log_path.relative_to(repo)).replace("\\", "/"),
            }
            r1.atomic_write_json(terminal_path, terminal)
            print(json.dumps(terminal, sort_keys=True), file=sys.stderr)
            return 1

    if (
        current["verified_ledgers"] != a2.a1.EXPECTED_LEDGERS
        or current["verified_frames"] != a2.a1.EXPECTED_FRAMES
        or current["discontinuity_resets"] != a2.a1.EXPECTED_RESETS
        or current["missing_ledgers"] != 0
    ):
        raise a2.ContinuationA2Error("a2_full_coverage_gate_not_satisfied")
    terminal = {
        "schema": (
            "blindassist_ustrf_l1e_materialization_recovery_r3_"
            "continuation_terminal_a2"
        ),
        "stage": a2.STAGE,
        "status": "CANONICAL_INPUT_41_OF_41_COMPLETE",
        "finished_at_utc": now(),
        "config_sha256": r1.sha256_file(config_path),
        "starting_coverage": starting,
        "final_coverage": current,
        "successful_child_processes": successful,
        "failed_child_processes": failed,
        "one_process_per_successful_shard": (
            successful
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

