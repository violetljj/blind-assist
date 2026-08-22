#!/usr/bin/env python3
"""S5 wrapper freezing current-frame depth-aperture selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1 import tartanair_s3 as engine
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, validated_box


PROTOCOL_ID = "BLINDASSIST_TARTANAIR_CURRENT_FRAME_COMPLETION_S5_V1"
INTERACTION_BOUNDARY_M = 2.0
MIN_APPARENT_HEIGHT_M = 0.35


def select_s5_candidate(candidates: Sequence[Mapping[str, Any]], dino_candidates: Sequence[Mapping[str, Any]], width: int, height: int) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        box = validated_box(candidate["bbox_xyxy"], "YOLOE candidate")
        height_fraction = (box[3] - box[1]) / height
        near_surface = candidate.get("sensor_region_depth_p20_m")
        interior = candidate.get("sensor_region_depth_m")
        if near_surface is None or interior is None or not box[0] <= width / 2.0 <= box[2]:
            continue
        overlaps = [iou(box, validated_box(row["bbox_xyxy"], "DINO candidate")) for row in dino_candidates]
        best_index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else None
        consensus = overlaps[best_index] if best_index is not None else 0.0
        if height_fraction >= engine.HEIGHT_FRACTION_MIN and float(near_surface) <= INTERACTION_BOUNDARY_M < float(interior) and float(near_surface) * height_fraction >= MIN_APPARENT_HEIGHT_M and consensus >= engine.DINO_IOU_MIN:
            eligible.append(dict(candidate) | {"height_fraction": height_fraction, "apparent_height_proxy_m": float(near_surface) * height_fraction, "depth_aperture_span_m": float(interior) - float(near_surface), "dino_consensus_iou": consensus, "dino_candidate": dict(dino_candidates[best_index])})
    return max(eligible, key=lambda row: (float(row["depth_aperture_span_m"]), float(row["dino_consensus_iou"]), float(row["proposal_score"]))) if eligible else None


def main() -> int:
    engine.PROTOCOL_ID = PROTOCOL_ID
    engine.AUTH_SCHEMA = "blindassist_tartanair_current_frame_completion_s5_authorization_v1"
    engine.RUN_SCHEMA = "blindassist_tartanair_current_frame_completion_s5_run_v1"
    engine.EVAL_SCHEMA = "blindassist_tartanair_current_frame_completion_s5_evaluation_v1"
    engine.SENSOR_DEPTH_MAX_M = INTERACTION_BOUNDARY_M
    engine.PROVIDER_EXTRA = {"selection_rule": "p20 <=2.0m < median; rank by median-minus-p20", "sensor_depth_p20_max_m": INTERACTION_BOUNDARY_M, "sensor_depth_median_min_exclusive_m": INTERACTION_BOUNDARY_M, "minimum_apparent_height_m": MIN_APPARENT_HEIGHT_M}
    engine.select_geometric_candidate = select_s5_candidate
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
