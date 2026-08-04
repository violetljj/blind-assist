"""Evaluate the frozen R0 operator against consumed public Bonn RGB-D truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from core import CAUSAL_WINDOW, decide_relative_open, score_relative_intrusion


REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path.insert(0, str(HFTF_DIR))

from prepare_bonn_rgbd_metric_depth_manifest import read_tum_index  # noqa: E402
from produce_external_rgb_metric_depth_observations import (  # noqa: E402
    DepthAnythingV2MetricSource,
)


EXPECTED_CHECKPOINT_SHA256 = (
    "B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/research/hftf/"
    "SCALE_FREE_TRAVERSABILITY_R1_BONN_RGBD_PROTOCOL_2026-08-04.json"
)
DIRECTIONS = {
    "RELATIVELY_OPEN_LEFT": "left",
    "RELATIVELY_OPEN_CENTER": "center",
    "RELATIVELY_OPEN_RIGHT": "right",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def associate_unique_nearest(
    rgb_rows: list[tuple[float, Path]],
    depth_rows: list[tuple[float, Path]],
    max_delta_s: float,
) -> list[tuple[float, Path, float, Path]]:
    """Greedily bind each RGB row to a nearest unused depth row."""

    if max_delta_s < 0:
        raise ValueError("max_delta_s must be non-negative")
    pairs: list[tuple[float, Path, float, Path]] = []
    cursor = 0
    for rgb_time, rgb_path in rgb_rows:
        while (
            cursor + 1 < len(depth_rows)
            and abs(depth_rows[cursor + 1][0] - rgb_time)
            <= abs(depth_rows[cursor][0] - rgb_time)
        ):
            cursor += 1
        depth_time, depth_path = depth_rows[cursor]
        if abs(depth_time - rgb_time) <= max_delta_s:
            pairs.append((rgb_time, rgb_path, depth_time, depth_path))
            cursor += 1
            if cursor >= len(depth_rows):
                break
    return pairs


def sample_causal(
    pairs: list[tuple[float, Path, float, Path]], target_fps: float
) -> list[tuple[float, Path, float, Path]]:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if not pairs:
        return []
    period = 1.0 / target_fps
    selected = [pairs[0]]
    next_time = pairs[0][0] + period
    for pair in pairs[1:]:
        if pair[0] + 1e-12 >= next_time:
            selected.append(pair)
            next_time = pair[0] + period
    return selected


def advance(
    score: dict[str, Any], history: list[dict[str, float]]
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    if score["status"] != "VALID":
        return {"status": "UNKNOWN", "reason": score["reason"]}, []
    updated = (history + [score["scores"]])[-CAUSAL_WINDOW:]
    return decide_relative_open(updated), updated


def summarize_sequence(sequence_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    truth_valid = sum(row["truth_score"]["status"] == "VALID" for row in rows)
    candidate_valid = sum(row["candidate_score"]["status"] == "VALID" for row in rows)
    truth_directional = [
        row for row in rows if row["truth_decision"].get("label") in DIRECTIONS
    ]
    recommended = [
        row
        for row in truth_directional
        if row["candidate_decision"].get("label") in DIRECTIONS
    ]
    correct = sum(
        row["candidate_decision"]["label"] == row["truth_decision"]["label"]
        for row in recommended
    )
    opposite = sum(
        {
            DIRECTIONS[row["candidate_decision"]["label"]],
            DIRECTIONS[row["truth_decision"]["label"]],
        }
        == {"left", "right"}
        for row in recommended
    )
    common = [
        row
        for row in rows
        if row["truth_decision"].get("status") == "VALID"
        and row["candidate_decision"].get("status") == "VALID"
    ]
    exact = sum(
        row["candidate_decision"].get("label")
        == row["truth_decision"].get("label")
        for row in common
    )
    return {
        "sequence_id": sequence_id,
        "frame_count": count,
        "truth_score_valid_count": truth_valid,
        "truth_score_coverage": truth_valid / count,
        "candidate_score_valid_count": candidate_valid,
        "candidate_execution_coverage": candidate_valid / count,
        "truth_directional_support": len(truth_directional),
        "candidate_recommendation_count": len(recommended),
        "recommendation_coverage": len(recommended) / max(1, len(truth_directional)),
        "directional_correct_count": correct,
        "directional_accuracy": correct / max(1, len(recommended)),
        "opposite_direction_count": opposite,
        "opposite_direction_rate": opposite / max(1, len(recommended)),
        "common_decision_count": len(common),
        "exact_decision_agreement": exact / max(1, len(common)),
    }


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bonn-root", required=True, type=Path)
    parser.add_argument("--dav2-repo", required=True, type=Path)
    parser.add_argument("--dav2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_CANDIDATE_OUTPUT_EXECUTION":
        raise ValueError("protocol is not frozen")
    if sha256(args.dav2_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("unexpected DA V2 checkpoint")

    sequence_roots = {
        "bonn_person_tracking": args.bonn_root
        / "clean-extracted/rgbd_bonn_person_tracking",
        "bonn_person_tracking2": args.bonn_root
        / "clean-extracted-2/rgbd_bonn_person_tracking2",
    }
    archive_names = {
        "bonn_person_tracking": "rgbd_bonn_person_tracking.clean.zip",
        "bonn_person_tracking2": "rgbd_bonn_person_tracking2.clean.zip",
    }
    expected_archives = {
        row["id"]: row["archive_sha256"] for row in protocol["data"]["sequences"]
    }
    for sequence_id, archive_name in archive_names.items():
        if sha256(args.bonn_root / archive_name) != expected_archives[sequence_id]:
            raise ValueError(f"{sequence_id}: archive identity failure")

    source = DepthAnythingV2MetricSource(
        args.dav2_repo,
        args.dav2_checkpoint,
        args.device,
        input_size=518,
        precision="fp16" if args.device.startswith("cuda") else "fp32",
    )
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for sequence_id, root in sequence_roots.items():
        pairs = associate_unique_nearest(
            read_tum_index(root / "rgb.txt"),
            read_tum_index(root / "depth.txt"),
            0.02,
        )
        sampled = sample_causal(pairs, float(protocol["data"]["target_fps"]))
        if len(sampled) < 10:
            raise ValueError(f"{sequence_id}: insufficient associated sample")
        candidate_history: list[dict[str, float]] = []
        truth_history: list[dict[str, float]] = []
        sequence_rows: list[dict[str, Any]] = []
        for frame_index, (rgb_time, rgb_rel, depth_time, depth_rel) in enumerate(sampled):
            rgb_path = root / rgb_rel
            depth_path = root / depth_rel
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"failed to decode RGB: {rgb_path}")
            candidate_depth, _ = source.infer(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {}
            )
            candidate_score = score_relative_intrusion(candidate_depth)
            candidate_decision, candidate_history = advance(
                candidate_score, candidate_history
            )

            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is None or depth_raw.ndim != 2:
                raise OSError(f"failed to decode registered depth: {depth_path}")
            if depth_raw.shape != bgr.shape[:2]:
                raise ValueError(f"{sequence_id}: registered depth shape mismatch")
            truth_score = score_relative_intrusion(depth_raw.astype(np.float64) / 5000.0)
            truth_decision, truth_history = advance(truth_score, truth_history)
            row = {
                "schema": "blindassist_scale_free_traversability_r1_bonn_frame_v1",
                "sequence_id": sequence_id,
                "frame_index": frame_index,
                "rgb_timestamp_s": rgb_time,
                "depth_timestamp_s": depth_time,
                "association_delta_s": abs(rgb_time - depth_time),
                "rgb_relative_path": rgb_rel.as_posix(),
                "depth_relative_path": depth_rel.as_posix(),
                "candidate_score": candidate_score,
                "candidate_decision": candidate_decision,
                "truth_score": truth_score,
                "truth_decision": truth_decision,
            }
            sequence_rows.append(row)
            all_rows.append(row)
        summaries.append(summarize_sequence(sequence_id, sequence_rows))

    gates = protocol["gates"]
    source_evaluable = all(
        row["truth_score_coverage"]
        >= gates["minimum_truth_score_coverage_each_sequence"]
        and row["truth_directional_support"]
        >= gates["minimum_directional_truth_support_each_sequence"]
        for row in summaries
    )
    accuracy_macro = sum(row["directional_accuracy"] for row in summaries) / len(summaries)
    opposite_macro = sum(row["opposite_direction_rate"] for row in summaries) / len(summaries)
    supported = source_evaluable and all(
        row["candidate_execution_coverage"]
        >= gates["minimum_candidate_execution_coverage_each_sequence"]
        and row["recommendation_coverage"]
        >= gates["minimum_recommendation_coverage_each_sequence"]
        and row["directional_accuracy"]
        >= gates["minimum_directional_accuracy_worst_sequence"]
        for row in summaries
    ) and accuracy_macro >= gates["minimum_directional_accuracy_macro"] \
        and opposite_macro <= gates["maximum_macro_opposite_direction_rate"]
    terminal = (
        "SCALE_FREE_TRAVERSABILITY_R1_NOT_EVALUABLE_SOURCE_SUPPORT"
        if not source_evaluable
        else (
            "SCALE_FREE_TRAVERSABILITY_R1_EXTERNAL_RGBD_REPLICATION_SUPPORTED_DEVELOPMENT_ONLY"
            if supported
            else "SCALE_FREE_TRAVERSABILITY_R1_EXTERNAL_RGBD_REPLICATION_NOT_SUPPORTED"
        )
    )

    args.output_root.mkdir(parents=True)
    frames_path = args.output_root / "frames.jsonl"
    write_new(
        frames_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows).encode(),
    )
    result = {
        "schema": "blindassist_scale_free_traversability_r1_bonn_result_v1",
        "status": "DEVELOPMENT_EXTERNAL_RGBD_EVALUATION_COMPLETE",
        "terminal": terminal,
        "authority": protocol["authority"],
        "data_role": protocol["data"]["role"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "sequence_count": len(summaries),
        "frame_count": len(all_rows),
        "sequence_results": summaries,
        "directional_accuracy_sequence_macro": accuracy_macro,
        "opposite_direction_rate_sequence_macro": opposite_macro,
        "gates": gates,
        "frames_jsonl": frames_path.name,
        "frames_jsonl_sha256": sha256(frames_path),
        "limitations": [
            "Both public sequences were consumed by older project experiments.",
            "The independent unit count is two source sequences.",
            "Person-tracking scenes do not represent general blind navigation.",
            "Relative-direction agreement does not establish clearance, distance, alerts, safety, or production fitness.",
        ],
    }
    write_new(
        args.output_root / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
