"""Prepare burned REveL input and a separate truth-late join for B Development."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
from typing import Any

import cv2

from .common import manifest_hash, sha256_file


INPUT_SCHEMA = "blindassist.target_local_warp_residual_input.v1"
INPUT_RECEIPT_SCHEMA = "blindassist.target_local_warp_residual_burned_input_receipt.v1"
TRUTH_SCHEMA = "blindassist.target_local_warp_residual_truth.v1"
TRUTH_RECEIPT_SCHEMA = "blindassist.target_local_warp_residual_truth_late_receipt.v1"


def _write_exclusive(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_exclusive(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_exclusive(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def _image_info(path: Path, cache: dict[str, tuple[str, list[int]]]) -> tuple[str, list[int]]:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3):
        raise ValueError(f"unable to decode image or unsupported shape: {path}")
    value = (sha256_file(path), [int(image.shape[0]), int(image.shape[1])])
    cache[key] = value
    return value


def _resolve_image(image_root: Path, relative: str) -> Path:
    root = image_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"image path escapes image root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _bbox_from_normalized(roi: list[float], shape: list[int]) -> list[float]:
    if len(roi) != 4:
        raise ValueError("roi_xywh_normalized must contain four values")
    cx, cy, width, height = (float(value) for value in roi)
    if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)) or width <= 0 or height <= 0:
        raise ValueError("invalid normalized ROI")
    image_height, image_width = shape
    return [(cx - width / 2.0) * image_width, (cy - height / 2.0) * image_height,
            (cx + width / 2.0) * image_width, (cy + height / 2.0) * image_height]


def _input_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "source_id", "session_id", "sequence_id", "previous_source_frame_id", "current_source_frame_id",
        "previous_frame_index", "current_frame_index", "captured_at_ns_previous", "captured_at_ns_current",
        "target_id", "track_epoch", "previous_bbox", "current_bbox", "previous_frame_shape",
        "current_frame_shape", "previous_image_sha256", "current_image_sha256", "parent_event_id",
    )}


def _detection_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {"previous_dynamic_bboxes": row["previous_dynamic_bboxes"], "current_dynamic_bboxes": row["current_dynamic_bboxes"]}


def prepare_input(replay_path: Path, image_root: Path, output_path: Path, receipt_path: Path, *, source_id: str, session_id: str, sequence_id: str) -> dict[str, Any]:
    replay_rows = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not replay_rows:
        raise ValueError("replay input is empty")
    by_target_index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in replay_rows:
        key = (str(row["target_id"]), int(row["source_frame_index"]))
        if key in by_target_index:
            raise ValueError(f"duplicate replay identity: {key}")
        by_target_index[key] = row
    image_cache: dict[str, tuple[str, list[int]]] = {}
    input_rows: list[dict[str, Any]] = []
    for target_id, frame_index in sorted(by_target_index):
        current = by_target_index[(target_id, frame_index)]
        previous = by_target_index.get((target_id, frame_index - 1))
        if previous is None or str(previous["track_epoch"]) != str(current["track_epoch"]):
            continue
        previous_path = _resolve_image(image_root, str(previous["image_relative_path"]))
        current_path = _resolve_image(image_root, str(current["image_relative_path"]))
        previous_sha, previous_shape = _image_info(previous_path, image_cache)
        current_sha, current_shape = _image_info(current_path, image_cache)
        input_rows.append({
            "schema": INPUT_SCHEMA,
            "source_id": source_id, "session_id": session_id, "sequence_id": sequence_id,
            "previous_source_frame_id": str(previous["source_frame_id"]), "current_source_frame_id": str(current["source_frame_id"]),
            "previous_frame_index": int(previous["source_frame_index"]), "current_frame_index": int(current["source_frame_index"]),
            "previous_image": previous_path.as_posix(), "current_image": current_path.as_posix(),
            "previous_image_sha256": previous_sha, "current_image_sha256": current_sha,
            "previous_frame_shape": previous_shape, "current_frame_shape": current_shape,
            "captured_at_ns_previous": int(previous["captured_at_ns"]), "captured_at_ns_current": int(current["captured_at_ns"]),
            "target_id": target_id, "track_epoch": str(current["track_epoch"]),
            "track_reset": bool(previous.get("history_reset") or current.get("history_reset")),
            "previous_bbox": _bbox_from_normalized(previous["roi_xywh_normalized"], previous_shape),
            "current_bbox": _bbox_from_normalized(current["roi_xywh_normalized"], current_shape),
            "previous_dynamic_bboxes": [], "current_dynamic_bboxes": [],
        })
    if not input_rows:
        raise ValueError("no adjacent same-track input pairs")
    _write_jsonl(output_path, input_rows)
    receipt = {
        "schema": INPUT_RECEIPT_SCHEMA, "status": "FROZEN", "stage": "B_DEVELOPMENT_INPUT_FREEZE",
        "source_role": "BURNED_REPLAY_INPUT", "source_id": source_id, "session_id": session_id, "sequence_id": sequence_id,
        "replay_input_sha256": sha256_file(replay_path), "image_root": image_root.resolve().as_posix(),
        "image_count": len(image_cache), "input_row_count": len(input_rows), "input_sha256": sha256_file(output_path),
        "shape_mismatch_pair_count": sum(row["previous_frame_shape"] != row["current_frame_shape"] for row in input_rows),
        "input_manifest_sha256": manifest_hash(input_rows, _input_identity),
        "detection_manifest_sha256": manifest_hash(input_rows, _detection_identity),
        "dynamic_boxes_policy": "EMPTY_NO_SEMANTIC_DETECTOR_DUMP_IN_FROZEN_REPLAY_INPUT",
        "truth_read": False, "candidate_output_read": False, "oracle_or_pose_read": False,
        "claim_ceiling": "DEVELOPMENT_SIGNAL_DIAGNOSTIC_ONLY",
    }
    _write_json(receipt_path, receipt)
    return receipt


def _canonical_truth_state(value: Any) -> str | None:
    return {"approaching": "approach", "receding": "receding", "quasi_static": "quasi-static"}.get(value)


def prepare_truth(input_path: Path, source_truth_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    input_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    input_index = {(str(row["target_id"]), int(row["current_frame_index"])): row for row in input_rows}
    source_rows = [json.loads(line) for line in source_truth_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_target_index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in source_rows:
        key = (str(row["target_id"]), int(row["source_frame_index"]))
        if key in by_target_index:
            raise ValueError(f"duplicate truth identity: {key}")
        by_target_index[key] = row
    output_rows: list[dict[str, Any]] = []
    truth_sha = sha256_file(source_truth_path)
    for target_id, frame_index in sorted(by_target_index):
        current = by_target_index[(target_id, frame_index)]
        previous = by_target_index.get((target_id, frame_index - 1))
        input_row = input_index.get((target_id, frame_index))
        if previous is None or input_row is None:
            continue
        if not (previous.get("truth_available") and current.get("truth_available") and previous.get("unique_roi_available") and current.get("unique_roi_available")):
            continue
        event_id = current.get("event_id")
        if not event_id or event_id != previous.get("event_id"):
            continue
        truth_state = _canonical_truth_state(current.get("truth_state"))
        if truth_state is None:
            continue
        output_rows.append({
            "schema": TRUTH_SCHEMA, "source_id": input_row["source_id"], "session_id": input_row["session_id"],
            "sequence_id": input_row["sequence_id"], "previous_source_frame_id": input_row["previous_source_frame_id"],
            "current_source_frame_id": input_row["current_source_frame_id"], "target_id": input_row["target_id"],
            "track_epoch": input_row["track_epoch"], "parent_event_id": str(event_id), "truth_eligible": True,
            "truth_state": truth_state, "source_truth_row_sha256": truth_sha,
        })
    if not output_rows:
        raise ValueError("no truth-eligible adjacent pairs")
    _write_jsonl(output_path, output_rows)
    receipt = {
        "schema": TRUTH_RECEIPT_SCHEMA, "status": "FROZEN_TRUTH_LATE", "stage": "B_DEVELOPMENT_TRUTH_LATE_JOIN",
        "input_sha256": sha256_file(input_path), "source_truth_sha256": truth_sha, "truth_late_sha256": sha256_file(output_path),
        "truth_late_row_count": len(output_rows), "parent_event_count": len({row["parent_event_id"] for row in output_rows}),
        "target_ids": sorted({row["target_id"] for row in output_rows}), "truth_states": dict(Counter(row["truth_state"] for row in output_rows)),
        "candidate_output_read": False, "producer_output_read": False, "claim_ceiling": "DEVELOPMENT_SIGNAL_DIAGNOSTIC_ONLY",
    }
    _write_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    input_parser = subparsers.add_parser("input")
    input_parser.add_argument("--replay-input", type=Path, required=True)
    input_parser.add_argument("--image-root", type=Path, required=True)
    input_parser.add_argument("--output", type=Path, required=True)
    input_parser.add_argument("--receipt", type=Path, required=True)
    input_parser.add_argument("--source-id", default="REVEL_DYNAMIC_V1")
    input_parser.add_argument("--session-id", default="REVEL_DYNAMIC_SINGLE_CAPTURE")
    input_parser.add_argument("--sequence-id", default="REVEL_DYNAMIC_V1")
    truth_parser = subparsers.add_parser("truth")
    truth_parser.add_argument("--input", type=Path, required=True)
    truth_parser.add_argument("--source-truth", type=Path, required=True)
    truth_parser.add_argument("--output", type=Path, required=True)
    truth_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "input":
        receipt = prepare_input(args.replay_input, args.image_root, args.output, args.receipt, source_id=args.source_id, session_id=args.session_id, sequence_id=args.sequence_id)
    else:
        receipt = prepare_truth(args.input, args.source_truth, args.output, args.receipt)
    print(json.dumps({"status": receipt["status"], "rows": receipt.get("input_row_count", receipt.get("truth_late_row_count"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
