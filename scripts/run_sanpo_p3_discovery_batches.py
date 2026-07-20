#!/usr/bin/env python3
"""Run resumable, fail-closed batches of official SANPO candidate discovery.

Each completed batch keeps the discoverer's normal final JSON.  A checkpoint
binds the requested official-session order and all selection parameters, so a
timeout can repeat at most one small batch instead of discarding a long scan.
No aggregate report is written while any batch is unresolved or has failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import discover_sanpo_sequence_candidates as discover


CHECKPOINT_FORMAT = "blindassist_sanpo_p3_discovery_batches_v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def batch_specs(ids: list[str], start_index: int, batch_size: int) -> list[dict[str, Any]]:
    return [
        {"start_session_index": start_index + offset, "session_ids": ids[offset:offset + batch_size]}
        for offset in range(0, len(ids), batch_size)
    ]


def contract_from_args(args: argparse.Namespace, ids: list[str]) -> dict[str, Any]:
    return {
        "split": args.split,
        "start_session_index": args.start_session_index,
        "max_sessions": args.max_sessions,
        "batch_size": args.batch_size,
        "sample_count": args.sample_count,
        "minimum_hits": args.minimum_hits,
        "camera": args.camera,
        "profiles": list(args.profiles),
        "labels": list(args.labels),
        "local_lateral_frame_count": args.local_lateral_frame_count,
        "local_lateral_min_target_frames": args.local_lateral_min_target_frames,
        "local_lateral_min_target_run": args.local_lateral_min_target_run,
        "local_lateral_min_path_frames": args.local_lateral_min_path_frames,
        "target_fps": args.target_fps,
        "retries": args.retries,
        "selected_session_ids": ids,
        "selected_session_ids_sha256": canonical_sha256(ids),
    }


def new_checkpoint(contract: dict[str, Any]) -> dict[str, Any]:
    return {"format": CHECKPOINT_FORMAT, "complete": False, "contract": contract, "completed_batches": {}}


def load_checkpoint(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read checkpoint: {error}") from error
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("complete") is not False:
        raise ValueError("checkpoint format is invalid or already finalized")
    if checkpoint.get("contract") != contract:
        raise ValueError("checkpoint contract or official session order differs; refusing resume")
    if not isinstance(checkpoint.get("completed_batches"), dict):
        raise ValueError("checkpoint completed_batches is invalid")
    return checkpoint


def discover_command(args: argparse.Namespace, spec: dict[str, Any], output: Path) -> list[str]:
    script = Path(__file__).with_name("discover_sanpo_sequence_candidates.py")
    command = [
        sys.executable, str(script), "--output", str(output), "--split", args.split,
        "--start-session-index", str(spec["start_session_index"]),
        "--max-sessions", str(len(spec["session_ids"])),
        "--sample-count", str(args.sample_count), "--minimum-hits", str(args.minimum_hits),
        "--camera", args.camera, "--target-fps", str(args.target_fps), "--retries", str(args.retries),
        "--local-lateral-frame-count", str(args.local_lateral_frame_count),
        "--local-lateral-min-target-frames", str(args.local_lateral_min_target_frames),
        "--local-lateral-min-target-run", str(args.local_lateral_min_target_run),
        "--local-lateral-min-path-frames", str(args.local_lateral_min_path_frames),
        "--labels", *args.labels, "--profiles", *args.profiles,
    ]
    return command


def validate_batch_report(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"batch report is unreadable: {error}") from error
    coverage = report.get("scan_coverage")
    if not isinstance(coverage, list) or len(coverage) != 1:
        raise ValueError("batch report has invalid scan coverage")
    row = coverage[0]
    expected = len(spec["session_ids"])
    if row.get("requested_start_session_index") != spec["start_session_index"] or row.get("selected_session_count") != expected:
        raise ValueError("batch report does not bind the expected index range")
    if row.get("attempted_session_count") != expected or row.get("network_or_data_failure_count") != 0:
        raise ValueError("batch has unresolved session failures")
    return report


def run_batches(
    args: argparse.Namespace,
    ids: list[str],
    checkpoint_path: Path,
    output_dir: Path,
    invoke: Callable[[list[str]], int] | None = None,
    max_batches_per_invocation: int = 0,
) -> dict[str, Any] | None:
    contract = contract_from_args(args, ids)
    checkpoint = load_checkpoint(checkpoint_path, contract) if args.resume else new_checkpoint(contract)
    if not args.resume:
        atomic_write_json(checkpoint_path, checkpoint)
    invoke = invoke or (lambda command: subprocess.run(command, check=False).returncode)
    completed = checkpoint["completed_batches"]
    invoked_batch_count = 0
    for spec in batch_specs(ids, args.start_session_index, args.batch_size):
        key = str(spec["start_session_index"])
        batch_path = output_dir / f"batch_{spec['start_session_index']:03d}_{len(spec['session_ids']):03d}.json"
        if key in completed:
            if completed[key].get("sha256") != sha256_file(batch_path):
                raise ValueError(f"completed batch {key} has changed or is missing")
            validate_batch_report(batch_path, spec)
            continue
        if max_batches_per_invocation and invoked_batch_count >= max_batches_per_invocation:
            return None
        code = invoke(discover_command(args, spec, batch_path))
        invoked_batch_count += 1
        if code != 0:
            raise RuntimeError(f"batch {key} discovery exited {code}; checkpoint remains resumable")
        report = validate_batch_report(batch_path, spec)
        completed[key] = {
            "session_ids": spec["session_ids"], "sha256": sha256_file(batch_path),
            "candidate_count": len(report.get("candidates", [])),
        }
        atomic_write_json(checkpoint_path, checkpoint)
    reports = [json.loads((output_dir / f"batch_{spec['start_session_index']:03d}_{len(spec['session_ids']):03d}.json").read_text(encoding="utf-8")) for spec in batch_specs(ids, args.start_session_index, args.batch_size)]
    aggregate = {
        "format": "blindassist_sanpo_p3_discovery_aggregate_v1", "complete": True,
        "contract": contract, "batches": completed,
        "candidates": [item for report in reports for item in report.get("candidates", [])],
        "local_lateral_prefilter_rejections": [item for report in reports for item in report.get("local_lateral_prefilter_rejections", [])],
        "failures": [],
    }
    atomic_write_json(args.aggregate_output, aggregate)
    checkpoint["complete"] = True
    checkpoint["aggregate_sha256"] = sha256_file(args.aggregate_output)
    atomic_write_json(checkpoint_path, checkpoint)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--start-session-index", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-batches-per-invocation", type=int, default=0,
                        help="stop cleanly after this many newly run batches; 0 means finish the range")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--minimum-hits", type=int, default=2)
    parser.add_argument("--camera", choices=(discover.AUTO_CAMERA, *discover.CAMERAS), default=discover.AUTO_CAMERA)
    parser.add_argument("--profiles", nargs="+", choices=sorted(discover.PROFILE_TARGETS), default=("center_obstacle", "lateral_pedestrian_or_ebike", "step_curb"))
    parser.add_argument("--labels", nargs="+", choices=sorted(discover.LABELS), default=sorted(discover.LABELS))
    parser.add_argument("--local-lateral-frame-count", type=int, default=16)
    parser.add_argument("--local-lateral-min-target-frames", type=int, default=8)
    parser.add_argument("--local-lateral-min-target-run", type=int, default=8)
    parser.add_argument("--local-lateral-min-path-frames", type=int, default=13)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.start_session_index < 0 or args.max_sessions < 0 or args.batch_size <= 0 or args.max_batches_per_invocation < 0:
        raise SystemExit("start/max values must be non-negative and batch values must be positive")
    all_ids = discover.session_ids(args.split)
    ids = all_ids[args.start_session_index:]
    if args.max_sessions:
        ids = ids[:args.max_sessions]
    if not ids:
        raise SystemExit("selected official session range is empty")
    try:
        result = run_batches(args, ids, args.checkpoint, args.output_dir, max_batches_per_invocation=args.max_batches_per_invocation)
    except (RuntimeError, ValueError) as error:
        print(f"checkpointed_scan_failed: {error}", file=sys.stderr)
        return 1
    if result is None:
        print(f"checkpointed_incomplete=true checkpoint={args.checkpoint}")
        return 0
    print(f"complete_batches={len(result['batches'])} candidates={len(result['candidates'])} aggregate={args.aggregate_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
