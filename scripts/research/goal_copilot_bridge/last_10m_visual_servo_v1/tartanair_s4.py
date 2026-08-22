#!/usr/bin/env python3
"""S4 wrapper adding a frozen apparent physical-height gate to S3 mechanics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1 import tartanair_s3 as engine
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, validated_box


PROTOCOL_ID = "BLINDASSIST_TARTANAIR_CURRENT_FRAME_COMPLETION_S4_V1"
APPARENT_HEIGHT_PROXY_MIN_M = 0.35


def select_s4_candidate(candidates: Sequence[Mapping[str, Any]], dino_candidates: Sequence[Mapping[str, Any]], width: int, height: int) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        box = validated_box(candidate["bbox_xyxy"], "YOLOE candidate")
        height_fraction = (box[3] - box[1]) / height
        sensor_depth = float(candidate["sensor_region_depth_m"])
        apparent_height_proxy = sensor_depth * height_fraction
        if not box[0] <= width / 2.0 <= box[2]:
            continue
        if height_fraction < engine.HEIGHT_FRACTION_MIN or sensor_depth > engine.SENSOR_DEPTH_MAX_M or apparent_height_proxy < APPARENT_HEIGHT_PROXY_MIN_M:
            continue
        overlaps = [iou(box, validated_box(row["bbox_xyxy"], "DINO candidate")) for row in dino_candidates]
        best_index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else None
        best_iou = overlaps[best_index] if best_index is not None else 0.0
        if best_iou >= engine.DINO_IOU_MIN:
            eligible.append(dict(candidate) | {"height_fraction": height_fraction, "apparent_height_proxy_m": apparent_height_proxy, "dino_consensus_iou": best_iou, "dino_candidate": dict(dino_candidates[best_index])})
    return max(eligible, key=lambda row: (float(row["dino_consensus_iou"]), float(row["proposal_score"]), -int(row["provider_rank"]))) if eligible else None


def main() -> int:
    engine.PROTOCOL_ID = PROTOCOL_ID
    engine.AUTH_SCHEMA = "blindassist_tartanair_current_frame_completion_s4_authorization_v1"
    engine.RUN_SCHEMA = "blindassist_tartanair_current_frame_completion_s4_run_v1"
    engine.EVAL_SCHEMA = "blindassist_tartanair_current_frame_completion_s4_evaluation_v1"
    engine.select_geometric_candidate = select_s4_candidate
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
