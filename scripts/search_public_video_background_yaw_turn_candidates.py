#!/usr/bin/env python3
"""Search marker-bearing rows for robust background-yaw LEFT/RIGHT turn candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_background_yaw_turn_candidates_v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify_flow(flow: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    height, width = flow.shape[:2]
    x1, y1, x2, y2 = map(float, spec["background_roi_xyxy_norm"])
    roi = flow[int(y1 * height):max(int(y1 * height) + 1, int(y2 * height)),
               int(x1 * width):max(int(x1 * width) + 1, int(x2 * width))]
    dx = roi[..., 0].astype(np.float64) / float(width)
    dy = roi[..., 1].astype(np.float64) / float(height)
    median_dx = float(np.median(dx))
    median_dy = float(np.median(dy))
    same_sign = float(np.mean(dx > 0.0)) if median_dx > 0 else float(np.mean(dx < 0.0))
    horizontal_ratio = abs(median_dx) / max(abs(median_dy), 1e-6)
    active = (abs(median_dx) >= float(spec["minimum_absolute_median_dx_norm"])
              and horizontal_ratio >= float(spec["minimum_horizontal_to_vertical_median_ratio"])
              and same_sign >= float(spec["minimum_same_sign_dx_fraction"]))
    direction = ("LEFT" if median_dx > 0 else "RIGHT") if active else "NONE"
    return {"direction": direction, "median_dx_norm": median_dx, "median_dy_norm": median_dy,
            "horizontal_to_vertical_ratio": horizontal_ratio, "same_sign_dx_fraction": same_sign}


def overlaps(source_id: str, start: int, end: int,
             excluded: dict[str, list[tuple[int, int]]], padding: int) -> bool:
    return any(start < b + padding and end > a - padding for a, b in excluded.get(source_id, []))


def candidate_runs(scored: list[dict[str, Any]], search: dict[str, Any],
                   excluded: dict[str, list[tuple[int, int]]]) -> list[dict[str, Any]]:
    required = int(search["required_consecutive_samples"])
    step = int(search["expected_sample_step_ms"])
    padding = int(search["exclude_r789_window_padding_ms"])
    found: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_source.setdefault(row["source_id"], []).append(row)
    for source_id, rows in sorted(by_source.items()):
        run: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda value: int(value["timestamp_ms"])) + [None]:
            contiguous = (row is not None and run
                          and int(row["timestamp_ms"]) - int(run[-1]["timestamp_ms"]) == step
                          and row["direction"] == run[-1]["direction"] != "NONE")
            if row is not None and row["direction"] != "NONE" and (not run or contiguous):
                run.append(row)
                continue
            if len(run) >= required:
                start, end = int(run[0]["timestamp_ms"]), int(run[-1]["timestamp_ms"]) + step
                if not overlaps(source_id, start, end, excluded, padding):
                    found.append({
                        "candidate_id": f"{source_id}:{run[0]['direction'].lower()}:{start}",
                        "parent_source_id": source_id,
                        "direction": run[0]["direction"],
                        "window_ms": [start, end],
                        "run_length": len(run),
                        "mean_absolute_median_dx_norm": float(np.mean([abs(x["median_dx_norm"]) for x in run])),
                        "minimum_same_sign_dx_fraction": float(min(x["same_sign_dx_fraction"] for x in run)),
                        "timestamps_ms": [int(x["timestamp_ms"]) for x in run],
                        "route_aux_item_ids": [x["item_id"] for x in run],
                        "local_video_path": run[0]["local_video_path"],
                        "source_video_sha256": run[0]["source_video_sha256"],
                    })
            run = [row] if row is not None and row["direction"] != "NONE" else []
    found.sort(key=lambda row: (-row["run_length"], -row["mean_absolute_median_dx_norm"], row["candidate_id"]))
    retained: list[dict[str, Any]] = []
    limit = int(search["maximum_candidates_per_direction"])
    for name in ("LEFT", "RIGHT"):
        retained.extend([row for row in found if row["direction"] == name][:limit])
    return sorted(retained, key=lambda row: (row["direction"], -row["run_length"], -row["mean_absolute_median_dx_norm"], row["candidate_id"]))


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or Path(str(output_path) + ".sha256").exists():
        raise ValueError("refusing to overwrite background-yaw candidate report")
    contract = common.load_json(contract_path)
    bound = contract["bound_inputs"]
    paths = {key[:-5]: Path(value) for key, value in bound.items() if key.endswith("_path")}
    for stem, path in paths.items():
        if common.sha256_file(path) != bound[f"{stem}_sha256"]:
            raise ValueError(f"bound input hash mismatch: {stem}")
    review = common.load_json(paths["r802_review"])
    if review["semantic_audit"]["r800_mean_anchor_x_as_turn_direction_supported"] is not False:
        raise ValueError("r802 did not establish the search-semantic failure")
    rows = load_jsonl(paths["route_aux_manifest"])
    if any(row.get("event_label") is not None for row in rows):
        raise ValueError("route auxiliary rows contain event labels")
    flow_spec = contract["flow"]
    width, height = int(flow_spec["resize_width"]), int(flow_spec["resize_height"])
    step = int(flow_spec["frame_step_ms"])
    scored: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for source_id, source_rows in sorted(by_source.items()):
        video = Path(source_rows[0]["local_video_path"])
        if common.sha256_file(video) != source_rows[0]["source_video_sha256"]:
            raise ValueError(f"source video hash mismatch: {source_id}")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        try:
            for row in sorted(source_rows, key=lambda value: int(value["timestamp_ms"])):
                frames = []
                for timestamp in (int(row["timestamp_ms"]), int(row["timestamp_ms"]) + step):
                    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp))
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        frames = []
                        break
                    gray = cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                    frames.append(gray)
                if len(frames) != 2:
                    continue
                flow = dis.calc(frames[0], frames[1], None)
                score = classify_flow(flow, flow_spec)
                scored.append({**row, **score})
        finally:
            capture.release()
    actionability = common.load_json(paths["actionability_manifest"])
    excluded: dict[str, list[tuple[int, int]]] = {}
    for item in actionability["items"]:
        excluded.setdefault(item["parent_source_id"], []).append(tuple(map(int, item["window_ms"])))
    candidates = candidate_runs(scored, contract["search"], excluded)
    by_direction = {name: [row for row in candidates if row["direction"] == name] for name in ("LEFT", "RIGHT")}
    gate = contract["gate"]
    checks = {
        "minimum_left_candidate_count": len(by_direction["LEFT"]) >= int(gate["minimum_left_candidate_count"]),
        "minimum_right_candidate_count": len(by_direction["RIGHT"]) >= int(gate["minimum_right_candidate_count"]),
        "minimum_left_source_count": len({row["parent_source_id"] for row in by_direction["LEFT"]}) >= int(gate["minimum_left_source_count"]),
        "minimum_right_source_count": len({row["parent_source_id"] for row in by_direction["RIGHT"]}) >= int(gate["minimum_right_source_count"]),
        "all_event_labels_null": True,
        "direction_independent_of_future_anchor_x": True,
        "all_parent_video_hashes_verified": True,
    }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(contract_path),
                   **{f"{stem}_sha256": common.sha256_file(path) for stem, path in paths.items()}},
        "summary": {"input_row_count": len(rows), "scored_row_count": len(scored),
                    "active_turn_row_count": sum(row["direction"] != "NONE" for row in scored),
                    "left_candidate_count": len(by_direction["LEFT"]), "right_candidate_count": len(by_direction["RIGHT"]),
                    "left_source_count": len({row["parent_source_id"] for row in by_direction["LEFT"]}),
                    "right_source_count": len({row["parent_source_id"] for row in by_direction["RIGHT"]})},
        "candidates": candidates,
        "checks": checks,
        "review_queue_ready": bool(all(checks.values())),
        "authorization": contract["authorization"],
        "evidence_limit": "Background-yaw runs are candidate proposals only and require continuous model/VLM review before any provisional direction coverage credit."
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output_path) + ".sha256").write_text(common.sha256_file(output_path) + "\n", encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.contract, args.output)
    print(json.dumps({"summary": report["summary"], "review_queue_ready": report["review_queue_ready"]}))


if __name__ == "__main__":
    main()
