#!/usr/bin/env python3
"""Round-robin merge heterogeneous door proposals without score calibration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256


MAX_CANDIDATES = 10
NMS_IOU = 0.70


def normalized_candidates(case: dict) -> list[dict]:
    raw = case.get("candidates", case.get("dino_candidates", []))
    return [
        {"bbox_xyxy": row["bbox_xyxy"], "source_score": float(row.get("proposal_score", row.get("score", 0.0))), "source_rank": index}
        for index, row in enumerate(raw, start=1)
    ]


def round_robin_merge(provider_rows: list[tuple[str, list[dict]]]) -> list[dict]:
    merged = []
    max_rank = max((len(rows) for _, rows in provider_rows), default=0)
    for rank in range(max_rank):
        for provider, rows in provider_rows:
            if rank >= len(rows):
                continue
            candidate = rows[rank] | {"source_provider": provider}
            if any(iou(candidate["bbox_xyxy"], existing["bbox_xyxy"]) >= NMS_IOU for existing in merged):
                continue
            merged.append(candidate)
            if len(merged) >= MAX_CANDIDATES:
                return [row | {"provider_rank": index} for index, row in enumerate(merged, start=1)]
    return [row | {"provider_rank": index} for index, row in enumerate(merged, start=1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--provider", action="append", required=True, help="NAME=prediction.json in frozen priority order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "merged future proposal output already exists")
    providers = []
    public_sha = sha256(args.public)
    for specification in args.provider:
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        payload = _read(path)
        _require(payload.get("private_truth_access") is False and payload.get("public_sha256") == public_sha, "proposal provider boundary mismatch")
        providers.append((name, path, payload))
    case_maps = [(name, {case["case_id"]: case for case in payload["cases"]}) for name, _, payload in providers]
    case_ids = list(case_maps[0][1])
    _require(all(set(mapping) == set(case_ids) for _, mapping in case_maps), "proposal provider roster mismatch")
    rows = []
    for case_id in case_ids:
        first = case_maps[0][1][case_id]
        candidates = round_robin_merge([(name, normalized_candidates(mapping[case_id])) for name, mapping in case_maps])
        rows.append({"case_id": case_id, "image_width": first["image_width"], "image_height": first["image_height"], "candidates": candidates})
    payload = {
        "schema_version": "blindassist_future_approach_merged_proposals_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "private_truth_access": False,
        "public_sha256": public_sha,
        "merge": {"provider_priority": [name for name, _, _ in providers], "round_robin": True, "nms_iou": NMS_IOU, "max_candidates": MAX_CANDIDATES},
        "provider_receipts": [{"name": name, "path": str(path.resolve()), "sha256": sha256(path)} for name, path, _ in providers],
        "cases": rows,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"case_count": len(rows), "cases_with_candidates": sum(bool(row["candidates"]) for row in rows), "provider_priority": payload["merge"]["provider_priority"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
