"""Evaluate frozen scale-free R0 on consumed parent-disjoint ARKitScenes RGB-D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from evaluate_bonn_rgbd_consumed import (
    DIRECTIONS,
    EXPECTED_CHECKPOINT_SHA256,
    advance,
    sha256,
    summarize_sequence,
    write_new,
)
from produce_external_rgb_metric_depth_observations import DepthAnythingV2MetricSource
from core import score_relative_intrusion


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/research/hftf/"
    "SCALE_FREE_TRAVERSABILITY_R2_ARKITSCENES_RGBD_PROTOCOL_2026-08-04.json"
)


def timestamp_from_stem(stem: str) -> float:
    return float(stem.rsplit("_", 1)[-1])


def matched_stems(root: Path) -> list[str]:
    sets = [
        {path.stem for path in (root / folder).glob("*.png")}
        for folder in ("lowres_wide", "lowres_depth", "confidence")
    ]
    common = set.intersection(*sets)
    return sorted(common, key=lambda stem: (timestamp_from_stem(stem), stem))


def dense_truth(
    depth_raw: np.ndarray,
    confidence: np.ndarray,
    truth_contract: dict[str, Any],
) -> tuple[np.ndarray | None, float]:
    depth = depth_raw.astype(np.float64) / 1000.0
    valid = (
        (confidence == int(truth_contract["confidence_value"]))
        & np.isfinite(depth)
        & (depth >= float(truth_contract["minimum_depth_m"]))
        & (depth <= float(truth_contract["maximum_depth_m"]))
    )
    fraction = float(np.mean(valid))
    if fraction < float(truth_contract["minimum_source_valid_fraction_per_frame"]):
        return None, fraction
    nearest = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    return depth[tuple(nearest)], fraction


def load_roster(path: Path) -> list[dict[str, Any]]:
    roster = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for role in ("train", "validation"):
        rows.extend({**row, "role": role} for row in roster["roles"][role])
    if len(rows) != 20 or len({str(row["visit_id"]) for row in rows}) != 20:
        raise ValueError("expected 20 unique ARKitScenes visits")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--dav2-repo", required=True, type=Path)
    parser.add_argument("--dav2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_ARKITSCENES_CANDIDATE_OUTPUT_EXECUTION":
        raise ValueError("protocol is not frozen")
    if sha256(args.roster) != protocol["data"]["roster_sha256"]:
        raise ValueError("roster identity failure")
    if sha256(args.dav2_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint identity failure")
    roster = load_roster(args.roster)
    source = DepthAnythingV2MetricSource(
        args.dav2_repo,
        args.dav2_checkpoint,
        args.device,
        input_size=518,
        precision="fp16" if args.device.startswith("cuda") else "fp32",
    )
    args.output_root.mkdir(parents=True)
    progress_path = args.output_root / "progress.json"
    progress_path.write_text(
        json.dumps({"status": "RUNNING", "completed_visits": 0, "total_visits": 20}),
        encoding="utf-8",
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for visit_index, roster_row in enumerate(roster, 1):
        visit_id = str(roster_row["visit_id"])
        video_id = str(roster_row["video_id"])
        root = args.dataset_root / video_id
        stems = matched_stems(root)
        if len(stems) != int(protocol["data"]["frame_count_each_video"]):
            raise ValueError(f"{video_id}: expected exactly 150 matched frames")
        candidate_history: list[dict[str, float]] = []
        truth_history: list[dict[str, float]] = []
        rows = []
        for frame_index, stem in enumerate(stems):
            rgb_path = root / "lowres_wide" / f"{stem}.png"
            depth_path = root / "lowres_depth" / f"{stem}.png"
            confidence_path = root / "confidence" / f"{stem}.png"
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
            confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is None or confidence is None:
                raise OSError(f"failed to decode truth: {stem}")
            if bgr.shape[:2] != depth_raw.shape or depth_raw.shape != confidence.shape:
                raise ValueError(f"{video_id}: RGB/depth/confidence shape mismatch")
            truth_depth, source_fraction = dense_truth(
                depth_raw, confidence, protocol["truth"]
            )
            truth_score = (
                score_relative_intrusion(truth_depth)
                if truth_depth is not None
                else {
                    "status": "UNKNOWN",
                    "reason": "INSUFFICIENT_CONFIDENCE2_SOURCE_FRACTION",
                }
            )
            truth_decision, truth_history = advance(truth_score, truth_history)
            row = {
                "schema": "blindassist_scale_free_traversability_r2_arkitscenes_frame_v1",
                "sequence_id": visit_id,
                "video_id": video_id,
                "role": roster_row["role"],
                "frame_index": frame_index,
                "timestamp_s": timestamp_from_stem(stem),
                "rgb_relative_path": f"{video_id}/lowres_wide/{stem}.png",
                "depth_relative_path": f"{video_id}/lowres_depth/{stem}.png",
                "truth_source_valid_fraction": source_fraction,
                "candidate_score": candidate_score,
                "candidate_decision": candidate_decision,
                "truth_score": truth_score,
                "truth_decision": truth_decision,
            }
            rows.append(row)
            all_rows.append(row)
        summary = summarize_sequence(visit_id, rows)
        summary.update({"video_id": video_id, "role": roster_row["role"]})
        summaries.append(summary)
        progress_path.write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "completed_visits": visit_index,
                    "total_visits": 20,
                }
            ),
            encoding="utf-8",
        )

    gates = protocol["gates"]
    source_evaluable = all(
        row["truth_score_coverage"]
        >= gates["minimum_truth_score_coverage_each_sequence"]
        and row["truth_directional_support"]
        >= gates["minimum_directional_truth_support_each_sequence"]
        for row in summaries
    )
    accuracy_macro = float(np.mean([row["directional_accuracy"] for row in summaries]))
    opposite_macro = float(np.mean([row["opposite_direction_rate"] for row in summaries]))
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
    terminals = protocol["terminals"]
    terminal = (
        terminals["not_evaluable"]
        if not source_evaluable
        else terminals["supported"] if supported else terminals["not_supported"]
    )

    frames_path = args.output_root / "frames.jsonl"
    write_new(
        frames_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows).encode(),
    )
    result = {
        "schema": "blindassist_scale_free_traversability_r2_arkitscenes_result_v1",
        "status": "DEVELOPMENT_EXTERNAL_RGBD_EVALUATION_COMPLETE",
        "terminal": terminal,
        "authority": protocol["authority"],
        "data_role": protocol["data"]["role"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "roster_sha256": sha256(args.roster),
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
            "All 20 visits were consumed by an older spatial-calibration development round.",
            "Dense reference depth uses nearest confidence-2 sensor-return reconstruction.",
            "ARKitScenes indoor scans do not represent fixed eyeglass navigation capture.",
            "Relative-direction agreement does not establish clearance, distance, alerts, safety, or production fitness.",
        ],
    }
    result_path = args.output_root / "result.json"
    write_new(
        result_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    progress_path.write_text(
        json.dumps({"status": "COMPLETE", "completed_visits": 20, "total_visits": 20}),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
