#!/usr/bin/env python3
"""Freeze the source-disjoint GRAIL-R1C-P paired-orientation probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freeze_grail_m1 import VAL_SHA256, ranked_houses
from grail_procthor_native_m0 import sha256_file


PREVIOUS_VAL_HOUSES = {
    529, 407, 296, 206, 768, 327, 857, 482, 372, 825, 628, 477,
    631, 708, 908, 9, 696, 469, 512, 368, 320, 485, 486, 555,
    663, 513, 636, 403, 860, 910,
}
OA_V2_CODE_COMMIT = "73b11c9dc83e84daeb563d0c766831f2c66b0a18"
OA_V2_CHECKPOINT_SHA256 = "7b6b7f258d32b95123b9d023005ecca357d8ab944fb83476f532d3cf7a2295eb"


def freeze(dataset: Path) -> dict:
    if sha256_file(dataset) != VAL_SHA256:
        raise ValueError("ProcTHOR val split hash mismatch")
    roster = ranked_houses(
        dataset,
        "BLINDASSIST_GRAIL_R1C_P_FRESH_VAL_V1",
        PREVIOUS_VAL_HOUSES,
    )[:12]
    return {
        "schema": "blindassist_grail_r1c_p_manifest_v1",
        "frozen_before_collection_pixels_or_model_outcome": True,
        "source": {
            "dataset_revision": "439193522244720b86d8c81cde2e51e3a4d150cf",
            "val_sha256": VAL_SHA256,
            "excluded_prior_train_dev_houses": sorted(PREVIOUS_VAL_HOUSES),
            "fresh_house_roster": roster,
        },
        "collection": {
            "profile": "V2B_SINGLE_HASH_RANKED_YAW_PER_POSITION_WITH_BILATERAL_FULL_SCENE_MASKS",
            "positive_denominator": 78,
            "wrong_target_denominator": 43,
            "admission": (
                "hash-rank same-type-distractor rows first to exactly 43, then hash-rank "
                "remaining rows to 78; fail NOT_EVALUABLE if either quota is unavailable"
            ),
            "admission_salt": "BLINDASSIST_GRAIL_R1C_P_ADMISSION_V1",
            "reads_model_output": False,
        },
        "orientation_model": {
            "name": "Orient Anything V2",
            "code_commit": OA_V2_CODE_COMMIT,
            "checkpoint": "demo_ckpts/rotmod_realrotaug_best.pt",
            "checkpoint_sha256": OA_V2_CHECKPOINT_SHA256,
            "checkpoint_bytes": 5048116892,
            "inference": "fixed zero-shot bfloat16 CUDA; no prompt, fit, calibration, or retry selection",
        },
        "frozen_downstream": {
            "checkpoint_sha256": "d838e8c1f648a771a41a32df7cbc0146b6bcebe98715fcd7f7c6c24ed7988b18",
            "grail_threshold": 0.9353410602,
            "selector_fields": ["semantic_type", "sibling_ordinal", "nearby_type"],
            "slot": "unchanged 3x3 rank_bin",
            "appearance_tiebreak": "unchanged frozen DINO/M1 local match",
            "pose_head": "unchanged frozen K=3 head",
        },
        "arms": {
            "OA_V2_INDEPENDENT_ABSOLUTE_DIAGNOSTIC": "diagnostic only",
            "OA_V2_PAIRED_RELATIVE_FINAL": "only adjudicated arm",
        },
        "gates": {
            "cross_view_slot_agreement_minimum": 70,
            "referent_top1_minimum": 70,
            "complete_pose_minimum": 50,
            "wrong_target_maximum": 1,
            "absence_false_commit_maximum": 1,
            "permutation_consistent": 156,
            "selector_collateral": 0,
            "complete_collateral": 0,
        },
        "claim_ceiling": (
            "one fixed zero-shot model on one fresh house-disjoint synthetic ProcTHOR cohort; "
            "no formal, natural-scene, device, product, or safety claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "houses": [row["house_index"] for row in manifest["source"]["fresh_house_roster"]],
        "orientation_checkpoint_sha256": manifest["orientation_model"]["checkpoint_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
