#!/usr/bin/env python3
"""Run the R1 outcome-blind producer with fail-closed publication."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import cv2

from radial_geometry import ARMS, FrameObservation, evaluate_pair


FORBIDDEN_PATH_TOKENS = ("truth", "event", "vicon", "decision")
FORMAL_INPUT_ROWS = 13_014
FORMAL_OUTPUT_ROWS = FORMAL_INPUT_ROWS * len(ARMS)
FORMAL_SHAPE_CHANGE_OPPORTUNITIES = 32
FORMAL_SHAPE_CHANGE_ARM_ROWS = FORMAL_SHAPE_CHANGE_OPPORTUNITIES * len(ARMS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _assert_producer_path(path: Path, label: str) -> None:
    lowered = path.as_posix().lower()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{label} path contains a forbidden truth/decision token")


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _assert_paths_isolated(
    replay_input: Path,
    image_root: Path,
    owned_paths: tuple[Path, ...],
) -> None:
    resolved_owned: set[Path] = set()
    for candidate in owned_paths:
        _assert_producer_path(candidate, "producer-owned")
        resolved = candidate.resolve()
        if resolved == replay_input.resolve():
            raise ValueError("producer-owned path collides with replay input")
        input_directory = replay_input.resolve().parent
        if input_directory.name.lower() == "input-freeze" and _is_within(
            resolved, input_directory
        ):
            raise ValueError("producer-owned path must not be inside input-freeze")
        if _is_within(resolved, image_root):
            raise ValueError("producer-owned path must not be inside image root")
        if resolved in resolved_owned:
            raise ValueError("producer-owned paths must be distinct")
        resolved_owned.add(resolved)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_progress(
    path: Path,
    *,
    phase: str,
    completed: int,
    total: int,
    started: float,
    status: str,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    throughput = completed / elapsed
    eta = max(total - completed, 0) / throughput if throughput > 0 else None
    payload = {
        "phase": phase,
        "completed_units": completed,
        "total_units": total,
        "throughput": throughput,
        "eta_seconds": eta,
        "last_progress_at": _utc_now(),
        "status": status,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _count_rows(path: Path, maximum: int | None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
                if maximum is not None and count >= maximum:
                    break
    return count


def _load_gray(path: Path) -> Any:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot decode {path}")
    return gray


def run(
    replay_input: Path,
    image_root: Path,
    output_path: Path,
    expected_replay_input_sha256: str,
    *,
    receipt_path: Path | None = None,
    progress_path: Path | None = None,
    failure_receipt_path: Path | None = None,
    mode: str = "pilot",
    max_rows: int | None = None,
    opencv_threads: int = 1,
    implementation_lock_sha256: str | None = None,
) -> dict[str, Any]:
    if mode not in {"pilot", "formal"}:
        raise ValueError("mode must be pilot or formal")
    if mode == "formal" and max_rows is not None:
        raise ValueError("formal replay cannot be truncated")
    if mode == "formal" and (
        implementation_lock_sha256 is None
        or len(implementation_lock_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in implementation_lock_sha256
        )
    ):
        raise ValueError("formal replay requires implementation-lock SHA-256")
    if opencv_threads < 1:
        raise ValueError("opencv_threads must be positive")
    receipt_path = receipt_path or output_path.with_suffix(
        output_path.suffix + ".receipt.json"
    )
    progress_path = progress_path or output_path.with_suffix(
        output_path.suffix + ".progress.json"
    )
    failure_receipt_path = failure_receipt_path or output_path.with_suffix(
        output_path.suffix + ".failure.json"
    )
    owned = (output_path, receipt_path, progress_path, failure_receipt_path)
    _assert_producer_path(replay_input, "replay input")
    _assert_producer_path(image_root, "image root")
    _assert_paths_isolated(replay_input, image_root, owned)
    if any(path.exists() for path in owned):
        raise FileExistsError("producer-owned terminal/progress path already exists")

    replay_sha = _sha256(replay_input)
    if replay_sha != expected_replay_input_sha256:
        raise ValueError("replay input SHA-256 does not match the activation identity")
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root missing: {image_root}")

    total = _count_rows(replay_input, max_rows)
    if mode == "formal" and total != FORMAL_INPUT_ROWS:
        raise ValueError(f"formal replay row count drift: {total}")
    started = time.monotonic()
    started_at = _utc_now()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        output_path.name + f".tmp-{os.getpid()}"
    )
    previous_by_target: dict[str, FrameObservation] = {}
    last_image_relative: str | None = None
    last_gray: Any = None
    input_rows = 0
    output_rows = 0
    shape_change_arm_rows = 0
    shape_change_opportunities = 0
    cv2.setNumThreads(opencv_threads)
    _write_progress(
        progress_path,
        phase="producer_replay",
        completed=0,
        total=total,
        started=started,
        status="running",
    )
    try:
        with (
            replay_input.open("r", encoding="utf-8") as source,
            temporary_output.open("x", encoding="utf-8", newline="\n") as sink,
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if max_rows is not None and input_rows >= max_rows:
                    break
                row = json.loads(line)
                relative = Path(str(row["image_relative_path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"line {line_number}: image path escapes root")
                image_path = (image_root / relative).resolve()
                image_path.relative_to(image_root)
                if str(relative) != last_image_relative:
                    last_gray = _load_gray(image_path)
                    last_image_relative = str(relative)
                observation = FrameObservation(
                    source_frame_id=str(row["source_frame_id"]),
                    captured_at_ns=int(row["captured_at_ns"]),
                    target_id=str(row["target_id"]),
                    track_epoch=str(row["track_epoch"]),
                    region=str(row["region"]),
                    roi_xywh_normalized=tuple(
                        float(value) for value in row["roi_xywh_normalized"]
                    ),
                    gray=last_gray,
                    history_reset=bool(row["history_reset"]),
                )
                previous = previous_by_target.get(observation.target_id)
                arm_rows = evaluate_pair(previous, observation)
                if [item["arm_id"] for item in arm_rows] != list(ARMS):
                    raise AssertionError("arm order drift")
                reasons = [item["abstention_reason"] for item in arm_rows]
                if "FRAME_SHAPE_CHANGE" in reasons:
                    if reasons != ["FRAME_SHAPE_CHANGE"] * len(ARMS):
                        raise AssertionError("shape guard must abstain both arms")
                    shape_change_opportunities += 1
                    shape_change_arm_rows += len(arm_rows)
                for arm_row in arm_rows:
                    sink.write(
                        json.dumps(
                            arm_row,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    sink.write("\n")
                    output_rows += 1
                previous_by_target[observation.target_id] = observation
                input_rows += 1
                if input_rows % 250 == 0 or input_rows == total:
                    _write_progress(
                        progress_path,
                        phase="producer_replay",
                        completed=input_rows,
                        total=total,
                        started=started,
                        status="running",
                    )
        if input_rows != total or output_rows != input_rows * len(ARMS):
            raise AssertionError("producer denominator drift")
        if mode == "formal" and (
            output_rows != FORMAL_OUTPUT_ROWS
            or shape_change_opportunities != FORMAL_SHAPE_CHANGE_OPPORTUNITIES
            or shape_change_arm_rows != FORMAL_SHAPE_CHANGE_ARM_ROWS
        ):
            raise AssertionError("formal R1 shape-change invariant drift")
        temporary_output.replace(output_path)
        elapsed = time.monotonic() - started
        receipt = {
            "status": "PRODUCER_COMPLETE",
            "mode": mode,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": elapsed,
            "opencv_threads": opencv_threads,
            "implementation_lock_sha256": implementation_lock_sha256,
            "replay_input_sha256": replay_sha,
            "input_rows": input_rows,
            "output_path": output_path.as_posix(),
            "output_rows": output_rows,
            "output_sha256": _sha256(output_path),
            "shape_change_opportunities": shape_change_opportunities,
            "shape_change_arm_rows": shape_change_arm_rows,
            "arm_ids": list(ARMS),
            "truth_joined": False,
            "offline_runtime_measured": True,
            "device_latency_measured": False,
        }
        _write_json_exclusive(receipt_path, receipt)
        _write_progress(
            progress_path,
            phase="producer_replay",
            completed=input_rows,
            total=total,
            started=started,
            status="complete",
        )
        return receipt
    except Exception as exc:
        temporary_output.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        failure = {
            "status": "PRODUCER_FAILED",
            "mode": mode,
            "started_at": started_at,
            "failed_at": _utc_now(),
            "completed_input_rows": input_rows,
            "completed_output_rows": output_rows,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rerun_authorized": False if mode == "formal" else None,
        }
        _write_json_exclusive(failure_receipt_path, failure)
        _write_progress(
            progress_path,
            phase="producer_replay",
            completed=input_rows,
            total=total,
            started=started,
            status="failed",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--failure-receipt", type=Path)
    parser.add_argument("--expected-replay-input-sha256", required=True)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--implementation-lock-sha256")
    args = parser.parse_args()
    receipt = run(
        args.replay_input,
        args.image_root,
        args.output,
        args.expected_replay_input_sha256,
        receipt_path=args.receipt,
        progress_path=args.progress,
        failure_receipt_path=args.failure_receipt,
        mode=args.mode,
        max_rows=args.max_rows,
        opencv_threads=args.opencv_threads,
        implementation_lock_sha256=args.implementation_lock_sha256,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
