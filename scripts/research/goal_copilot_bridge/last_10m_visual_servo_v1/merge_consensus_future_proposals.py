#!/usr/bin/env python3
"""Merge heterogeneous door proposals by cross-provider geometric consensus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.merge_future_approach_proposals_dev import MAX_CANDIDATES, normalized_candidates
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256


CONSENSUS_IOU = 0.30


def consensus_merge(provider_rows: list[tuple[str, list[dict]]]) -> list[dict]:
    provider_order = {name: index for index, (name, _) in enumerate(provider_rows)}
    clusters: list[list[dict]] = []
    for provider, rows in provider_rows:
        for row in rows:
            candidate = row | {"source_provider": provider}
            compatible = [cluster for cluster in clusters if any(iou(candidate["bbox_xyxy"], member["bbox_xyxy"]) >= CONSENSUS_IOU for member in cluster)]
            if compatible:
                max(compatible, key=lambda cluster: max(iou(candidate["bbox_xyxy"], member["bbox_xyxy"]) for member in cluster)).append(candidate)
            else:
                clusters.append([candidate])
    ranked = []
    for cluster in clusters:
        support = len({row["source_provider"] for row in cluster})
        representative = max(cluster, key=lambda row: (
            sum(iou(row["bbox_xyxy"], other["bbox_xyxy"]) for other in cluster),
            -provider_order[row["source_provider"]],
            -row["source_rank"],
        ))
        ranked.append(representative | {"provider_support": support, "supporting_providers": sorted({row["source_provider"] for row in cluster}, key=provider_order.get)})
    ranked.sort(key=lambda row: (-row["provider_support"], row["source_rank"], provider_order[row["source_provider"]], row["bbox_xyxy"]))
    return [row | {"provider_rank": index} for index, row in enumerate(ranked[:MAX_CANDIDATES], start=1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--provider", action="append", required=True, help="NAME=prediction.json in frozen priority order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "consensus proposal output already exists")
    public_hash = sha256(args.public)
    providers = []
    for specification in args.provider:
        name, raw_path = specification.split("=", 1)
        path, payload = Path(raw_path), _read(Path(raw_path))
        _require(payload.get("private_truth_access") is False and payload.get("public_sha256") == public_hash, "consensus provider boundary mismatch")
        providers.append((name, path, payload))
    maps = [(name, {row["case_id"]: row for row in payload["cases"]}) for name, _, payload in providers]
    case_ids = list(maps[0][1])
    _require(all(set(mapping) == set(case_ids) for _, mapping in maps), "consensus provider roster mismatch")
    rows = []
    for case_id in case_ids:
        first = maps[0][1][case_id]
        candidates = consensus_merge([(name, normalized_candidates(mapping[case_id])) for name, mapping in maps])
        rows.append({"case_id": case_id, "image_width": first["image_width"], "image_height": first["image_height"], "candidates": candidates})
    payload = {
        "schema_version": "blindassist_future_consensus_proposals_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "private_truth_access": False,
        "public_sha256": public_hash,
        "merge": {"provider_priority": [name for name, _, _ in providers], "consensus_iou": CONSENSUS_IOU, "max_candidates": MAX_CANDIDATES},
        "provider_receipts": [{"name": name, "sha256": sha256(path)} for name, path, _ in providers],
        "cases": rows,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"case_count": len(rows), "cases_with_candidates": sum(bool(row["candidates"]) for row in rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
