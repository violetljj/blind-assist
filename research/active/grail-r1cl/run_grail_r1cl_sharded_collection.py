#!/usr/bin/env python3
"""Run resumable R1C-L collectors concurrently and merge them deterministically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


XVFB_SCREEN = "-screen 0 1024x768x24 -ac +extension GLX +render -noreset"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    script_root = Path(__file__).resolve().parent
    args.shard_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[str], Any]] = []
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "LP_NUM_THREADS": "1",
    })
    for shard_index in range(args.shard_count):
        shard_output = args.shard_root / f"shard-{shard_index:02d}"
        log_path = args.shard_root / f"shard-{shard_index:02d}.log"
        log = log_path.open("a", encoding="utf-8")
        command = [
            "xvfb-run", "-a", "-s", XVFB_SCREEN, sys.executable,
            str(script_root / "collect_grail_pairwise_owner_coordinate_r1cl.py"),
            "--dataset", str(args.dataset), "--manifest", str(args.manifest),
            "--role", args.role, "--output", str(shard_output),
            "--shard-index", str(shard_index), "--shard-count", str(args.shard_count),
        ]
        if args.house_limit is not None:
            command.extend(("--house-limit", str(args.house_limit)))
        if args.allow_under_minimum:
            command.append("--allow-under-minimum")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True,
                                   env=environment, start_new_session=True)
        processes.append((shard_index, process, log))

    def stop_all(*_: Any) -> None:
        for _, process, _ in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    previous_term = signal.signal(signal.SIGTERM, stop_all)
    previous_int = signal.signal(signal.SIGINT, stop_all)
    try:
        failures: list[dict[str, int]] = []
        pending = {shard_index: process for shard_index, process, _ in processes}
        while pending:
            for shard_index, process in list(pending.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                pending.pop(shard_index)
                if returncode:
                    failures.append({"shard_index": shard_index, "exit_code": returncode})
            if failures and pending:
                stop_all()
            if pending:
                time.sleep(1.0)
        if failures:
            _atomic_json(args.shard_root / "failure.json", {
                "schema": "blindassist_grail_r1c_l_sharded_failure_v1", "role": args.role,
                "shard_count": args.shard_count, "failures": failures,
            })
            return 1
        if not args.skip_merge:
            merge_command = [
                sys.executable, str(script_root / "merge_grail_r1cl_collection_shards.py"),
                "--manifest", str(args.manifest), "--dataset", str(args.dataset),
                "--role", args.role, "--shard-root", str(args.shard_root),
                "--shard-count", str(args.shard_count), "--output", str(args.output),
            ]
            if args.allow_under_minimum:
                merge_command.append("--allow-under-minimum")
            completed = subprocess.run(merge_command, check=False, env=environment)
            if completed.returncode:
                return completed.returncode
        _atomic_json(args.shard_root / "run.json", {
            "schema": "blindassist_grail_r1c_l_sharded_run_v1", "role": args.role,
            "shard_count": args.shard_count, "house_limit": args.house_limit,
            "elapsed_seconds": time.monotonic() - started, "status": "complete",
            "merged": not args.skip_merge,
        })
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        for _, _, log in processes:
            log.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--house-limit", type=int)
    parser.add_argument("--allow-under-minimum", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
