#!/usr/bin/env python3
"""Produce bounded door guidance actions from public goal, RGB-D, and proposals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.evaluate_future_approach_proposals import bearing_action
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


STOP_DEPTH_M = 1.60
MAX_GUIDANCE_CANDIDATES = 3


def candidate_depth(depth: np.ndarray, box: list[float]) -> float | None:
    height, width = depth.shape
    x1, y1, x2, y2 = box
    # Use the interior of a proposal to suppress frame/background edges and
    # the lowest strip where floor pixels commonly enter a door box.
    left = max(0, min(width, int(round(x1 + 0.20 * (x2 - x1)))))
    right = max(0, min(width, int(round(x1 + 0.80 * (x2 - x1)))))
    top = max(0, min(height, int(round(y1 + 0.10 * (y2 - y1)))))
    bottom = max(0, min(height, int(round(y1 + 0.75 * (y2 - y1)))))
    values = depth[top:bottom, left:right]
    valid = values[np.isfinite(values) & (values >= 0.4) & (values <= 10.0)]
    return float(np.median(valid)) if valid.size >= 16 else None


def guidance_action(box: list[float], width: int, range_m: float | None) -> str:
    if range_m is not None and range_m <= STOP_DEPTH_M:
        return "STOP"
    return bearing_action(box, width)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"), required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "goal RGB-D servo output already exists")
    public, proposals = _read(args.public), _read(args.proposals)
    public_hash = sha256(args.public)
    _require(public.get("provider_truth_access") is False and proposals.get("private_truth_access") is False and proposals.get("public_sha256") == public_hash, "goal RGB-D servo boundary mismatch")
    proposal_cases = {row["case_id"]: row for row in proposals["cases"]}
    rows = []
    for case in public["cases"]:
        observed = proposal_cases[case["case_id"]]
        depth_path = Path(case["range_sensor"]["depth_path"])
        _require(sha256(depth_path) == case["range_sensor"]["depth_sha256"], "goal RGB-D depth drift")
        depth = decode_depth(depth_path)
        ranked = []
        for proposal in sorted(observed["candidates"], key=lambda row: row["provider_rank"]):
            range_m = float(proposal["source_mask_depth_p20_m"]) if proposal.get("source_mask_depth_p20_m") is not None else candidate_depth(depth, proposal["bbox_xyxy"])
            ranked.append({
                "proposal_rank": proposal["provider_rank"],
                "source_provider": proposal["source_provider"],
                "bbox_xyxy": proposal["bbox_xyxy"],
                "range_m": range_m,
                "action": guidance_action(proposal["bbox_xyxy"], observed["image_width"], range_m),
            })
        ranked.sort(key=lambda row: (row["action"] != "STOP", row["proposal_rank"]))
        candidates = [row | {"guidance_rank": index} for index, row in enumerate(ranked[:MAX_GUIDANCE_CANDIDATES], start=1)]
        rows.append({"case_id": case["case_id"], "image_width": observed["image_width"], "image_height": observed["image_height"], "candidates": candidates})
    payload = {
        "schema_version": "blindassist_goal_rgbd_servo_prediction_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "private_truth_access": False,
        "public_sha256": public_hash,
        "proposal_sha256": sha256(args.proposals),
        "contract": {"stop_depth_m": STOP_DEPTH_M, "maximum_guidance_candidates": MAX_GUIDANCE_CANDIDATES, "range_statistic": "SAM3_mask_p20_else_interior_median", "near_candidates_rank_before_far_candidates": True},
        "cases": rows,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"case_count": len(rows), "action_counts": {action: sum(candidate["action"] == action for row in rows for candidate in row["candidates"]) for action in ("TURN_LEFT", "ADVANCE", "TURN_RIGHT", "STOP")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
