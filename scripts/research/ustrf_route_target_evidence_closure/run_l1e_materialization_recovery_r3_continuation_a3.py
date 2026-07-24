#!/usr/bin/env python3
"""Serial A3 orchestrator for the final R3 input shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import exploratory_profiles_r2_l1 as r1
import l1e_materialization_recovery_r3_continuation_a3 as a3
import run_l1e_materialization_recovery_r3_one_shard as frozen_materializer


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_payload(config_path, config, starting, current, success, failed, status):
    return {
        "schema": (
            "blindassist_ustrf_l1e_materialization_recovery_r3_"
            "continuation_progress_a3"
        ),
        "stage": a3.STAGE,
        "status": status,
        "updated_at_utc": now(),
        "config_sha256": r1.sha256_file(config_path),
        "starting_coverage": starting,
        "current_coverage": current,
        "successful_child_processes": success,
        "failed_child_processes": failed,
        "one_process_per_successful_shard": (
            success == current["verified_ledgers"] - starting["verified_ledgers"]
        ),
        "candidate_execution_started": False,
        "authority": config["authority"],
    }


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
    root = a3.output_root(repo, parent_overlay)
    child = (
        repo / config["implementation_bindings"]["a3_one_shard_runner"]["path"]
    ).resolve()
    progress_path = root / "continuation-progress-a3.json"
    terminal_path = root / "continuation-terminal-a3.json"
    starting, _ = a3.coverage(repo, parent_overlay)
    if (
        starting["verified_ledgers"] < a3.EXPECTED_STARTING_LEDGERS
        or starting["verified_frames"] < a3.EXPECTED_STARTING_FRAMES
        or starting["missing_ledgers"] > a3.EXPECTED_REMAINING_SHARDS
    ):
        raise a3.ContinuationA3Error("a3_starting_coverage_drift")
    current = starting
    success = 0
    failed = 0
    r1.atomic_write_json(
        progress_path,
        progress_payload(
            config_path, config, starting, current, success, failed,
            "RUNNING" if not current["complete"] else "COMPLETE",
        ),
    )
    while not current["complete"]:
        selected = current["next_missing"]
        before_count = a3.cumulative_attempt_count(
            root, selected["source_id"], selected["sequence_id"]
        )
        if before_count >= a3.MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS:
            raise a3.ContinuationA3Error("a3_retry_limit_already_exhausted")
        completed = subprocess.run(
            [
                sys.executable,
                str(child),
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
        count = a3.cumulative_attempt_count(
            root, selected["source_id"], selected["sequence_id"]
        )
        log_root = root / "logs-a3"
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = (
            log_root
            / f"{a3.short_id(selected['source_id'], selected['sequence_id'])}"
            f".a{count:03d}.log"
        )
        log_path.write_text(
            completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
        )
        updated, _ = a3.coverage(repo, parent_overlay)
        if completed.returncode == 0:
            if (
                updated["verified_ledgers"] != current["verified_ledgers"] + 1
                or updated["verified_frames"]
                != current["verified_frames"] + int(selected["frame_count"])
            ):
                raise a3.ContinuationA3Error(
                    "a3_success_did_not_add_exactly_one_shard"
                )
            success += 1
            current = updated
            payload = progress_payload(
                config_path, config, starting, current, success, failed,
                "RUNNING" if not current["complete"] else "COMPLETE",
            )
            r1.atomic_write_json(progress_path, payload)
            print(json.dumps(payload, sort_keys=True), flush=True)
            continue
        failed += 1
        current = updated
        payload = progress_payload(
            config_path, config, starting, current, success, failed,
            "RETRYING_ONE_LEDGER",
        )
        r1.atomic_write_json(progress_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if current["next_missing"] != selected:
            raise a3.ContinuationA3Error("a3_failure_changed_next_missing")
        if count >= a3.MAXIMUM_CUMULATIVE_CHILD_ATTEMPTS:
            terminal = {
                **payload,
                "schema": (
                    "blindassist_ustrf_l1e_materialization_recovery_r3_"
                    "continuation_terminal_a3"
                ),
                "status": "FAIL_CLOSED_LEDGER_ATTEMPTS_EXHAUSTED",
                "failed_ledger": selected,
                "last_child_log": str(log_path.relative_to(repo)).replace("\\", "/"),
            }
            r1.atomic_write_json(terminal_path, terminal)
            print(json.dumps(terminal, sort_keys=True), file=sys.stderr)
            return 1
    if (
        current["verified_ledgers"] != a3.a2.a1.EXPECTED_LEDGERS
        or current["verified_frames"] != a3.a2.a1.EXPECTED_FRAMES
        or current["discontinuity_resets"] != a3.a2.a1.EXPECTED_RESETS
        or current["missing_ledgers"] != 0
    ):
        raise a3.ContinuationA3Error("a3_full_coverage_gate_not_satisfied")
    terminal = {
        "schema": (
            "blindassist_ustrf_l1e_materialization_recovery_r3_"
            "continuation_terminal_a3"
        ),
        "stage": a3.STAGE,
        "status": "CANONICAL_INPUT_41_OF_41_COMPLETE",
        "finished_at_utc": now(),
        "config_sha256": r1.sha256_file(config_path),
        "starting_coverage": starting,
        "final_coverage": current,
        "successful_child_processes": success,
        "failed_child_processes": failed,
        "one_process_per_successful_shard": (
            success == current["verified_ledgers"] - starting["verified_ledgers"]
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

