#!/usr/bin/env python3
"""Run the pre-existing frozen Grounding DINO functional-region proposal arm."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import content_sha256, sha256, validate_public


PROTOCOL_ID = "P1-FRG1-FROZEN-FUNCTIONAL-REGION-PROPOSAL-V1"
PREDICTION_SCHEMA = "blindassist_p1_frg1_prediction_v1"
BOUNDED_POOL_SIZE = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--prompt-map", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("FRG1 prediction already exists; refusing replay")
    public = json.loads(args.public.read_text(encoding="utf-8"))
    prompt_map = json.loads(args.prompt_map.read_text(encoding="utf-8"))
    cases = validate_public(public, prompt_map, args.public.resolve().parent)
    metadata = [{
        "id": case["case_id"], "path": str(case["image_path"]), "image_sha256": sha256(case["image_path"]),
    } for case in cases]
    inference, runtime = dino.run_inference(args.model_dir.resolve(), metadata)
    by_id = {row["image_id"]: row for row in inference}
    outputs = []
    for case in cases:
        proposals = by_id[case["case_id"]]["proposals"][:BOUNDED_POOL_SIZE]
        outputs.append({
            "case_id": case["case_id"],
            "candidates": [{
                "rank": rank, "bbox_xyxy": row["bbox_xyxy"], "proposal_score": row["score"],
                "functional_label": row["label"], "source": "grounding_dino_frozen_functional_prompt",
            } for rank, row in enumerate(proposals, start=1)],
            "provider_postprocessed_candidate_count": len(by_id[case["case_id"]]["proposals"]),
        })
    payload = {
        "schema_version": PREDICTION_SCHEMA, "protocol_id": PROTOCOL_ID,
        "public_input_sha256": sha256(args.public), "prompt_map_sha256": content_sha256(prompt_map),
        "private_truth_access": False,
        "provider": {
            "model_repository": dino.MODEL_REPOSITORY, "model_revision": dino.MODEL_REVISION,
            "weights_sha256": dino.WEIGHTS_SHA256, "prompt": dino.PROMPT,
            "box_threshold": dino.BOX_THRESHOLD, "text_threshold": dino.TEXT_THRESHOLD,
            "nms_iou_threshold": dino.NMS_IOU_THRESHOLD,
            "provider_max_proposals": dino.MAX_PROPOSALS_PER_IMAGE,
            "bounded_pool_size": BOUNDED_POOL_SIZE, "identity_selection": "FORBIDDEN",
            "threshold_or_configuration_sweep": False,
        },
        "runtime": runtime, "cases": outputs,
        "claim_role": "POST_PA3_OUTCOME_CONSUMED_DEVELOPMENT_MECHANISM_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
