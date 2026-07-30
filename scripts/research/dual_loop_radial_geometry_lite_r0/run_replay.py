#!/usr/bin/env python3
"""Run the frozen producer allowlist without importing evaluator truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

from radial_geometry import ARMS, FrameObservation, evaluate_pair


FORBIDDEN_PATH_TOKENS = ("truth", "event", "vicon", "decision")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _assert_output_isolated(
    replay_input: Path,
    image_root: Path,
    output_path: Path,
    receipt_path: Path,
) -> None:
    _assert_producer_path(output_path, "output")
    _assert_producer_path(receipt_path, "receipt")
    protected_files = {replay_input.resolve()}
    for candidate, label in ((output_path, "output"), (receipt_path, "receipt")):
        resolved = candidate.resolve()
        if resolved in protected_files:
            raise ValueError(f"{label} collides with replay input")
        input_directory = replay_input.resolve().parent
        if input_directory.name.lower() == "input-freeze" and _is_within(resolved, input_directory):
            raise ValueError(f"{label} must not be written inside input-freeze")
        if _is_within(resolved, image_root):
            raise ValueError(f"{label} must not be written inside image root")


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
) -> dict[str, Any]:
    _assert_producer_path(replay_input, "replay input")
    _assert_producer_path(image_root, "image root")
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    _assert_output_isolated(replay_input, image_root, output_path, receipt_path)
    replay_input_sha256 = _sha256(replay_input)
    if replay_input_sha256 != expected_replay_input_sha256:
        raise ValueError("replay input SHA-256 does not match the activation identity")
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root missing: {image_root}")
    previous_by_target: dict[str, FrameObservation] = {}
    outputs: list[dict[str, Any]] = []
    last_image_relative: str | None = None
    last_gray: Any = None
    with replay_input.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
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
                roi_xywh_normalized=tuple(float(value) for value in row["roi_xywh_normalized"]),
                gray=last_gray,
                history_reset=bool(row["history_reset"]),
            )
            previous = previous_by_target.get(observation.target_id)
            arm_rows = evaluate_pair(previous, observation)
            if [item["arm_id"] for item in arm_rows] != list(ARMS):
                raise AssertionError("arm order drift")
            outputs.extend(arm_rows)
            previous_by_target[observation.target_id] = observation
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in outputs:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    receipt = {
        "status": "PRODUCER_COMPLETE",
        "replay_input_sha256": replay_input_sha256,
        "output_path": output_path.as_posix(),
        "output_rows": len(outputs),
        "output_sha256": _sha256(output_path),
        "arm_ids": list(ARMS),
        "truth_joined": False,
        "runtime_latency_measured": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-replay-input-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(
        run(
            args.replay_input,
            args.image_root,
            args.output,
            args.expected_replay_input_sha256,
        ),
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
