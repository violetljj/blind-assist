#!/usr/bin/env python3
"""Freeze house-disjoint data roles and the sole GRAIL-R1C-L architecture."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from grail_procthor_native_m0 import canonical_sha256, sha256_file


DATASET_REVISION = "439193522244720b86d8c81cde2e51e3a4d150cf"
TRAIN_SHA256 = "ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587"
TEST_SHA256 = "9a9fa6f134e76fe87f3fd92c00883651cf9fadf4e9ad4072d6d73be229f001dc"
EXCLUDED_TEST_HOUSES = {
    0, 38, 41, 144, 184, 187, 278, 298, 310, 325, 343, 344, 394, 439,
    498, 500, 516, 518, 652, 676, 791, 906,
}


def _roster(path: Path, salt: str, count: int, excluded: set[int] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    ranked: list[tuple[str, int, dict[str, Any]]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in excluded:
                continue
            rank = hashlib.sha256(f"{salt}:{index}".encode("utf-8")).hexdigest()
            ranked.append((rank, index, json.loads(line)))
    selected = sorted(ranked)[:count]
    return [
        {"rank": rank, "house_index": index, "house_sha256": canonical_sha256(house)}
        for rank, index, house in selected
    ]


def freeze(train_dataset: Path, test_dataset: Path, runtime_image_id: str) -> dict[str, Any]:
    if sha256_file(train_dataset) != TRAIN_SHA256:
        raise ValueError("R1C-L ProcTHOR train identity mismatch")
    if sha256_file(test_dataset) != TEST_SHA256:
        raise ValueError("R1C-L ProcTHOR test identity mismatch")
    train_and_validation = _roster(
        train_dataset, "BLINDASSIST_GRAIL_R1C_L_TRAIN_VALIDATION_V1", 180
    )
    return {
        "schema": "blindassist_grail_r1c_l_manifest_v3",
        "authorization": "AUTHORIZE_GRAIL_R1C_L_TASK_TRAINED_SYMMETRY_MARGINALIZED_PAIRWISE_OWNER_COORDINATE",
        "frozen_before_collection_pixels_or_model_outcome": True,
        "source": {
            "dataset": "allenai/procthor-10k",
            "dataset_revision": DATASET_REVISION,
            "license": "Apache-2.0",
            "train_sha256": TRAIN_SHA256,
            "test_sha256": TEST_SHA256,
            "runtime_image_id": runtime_image_id,
            "ai2thor_release": "f0825767cd50d69f666c7f282e54abfe58f1e917",
        },
        "rosters": {
            "train": train_and_validation[:160],
            "validation": train_and_validation[160:],
            "final_test": _roster(
                test_dataset,
                "BLINDASSIST_GRAIL_R1C_L_FINAL_TEST_V1",
                12,
                EXCLUDED_TEST_HOUSES,
            ),
        },
        "collection": {
            "views_per_owner_group_maximum": 8,
            "visible_siblings_minimum": 2,
            "visible_siblings_maximum": 4,
            "native_owner_subgroup": "native owner/type group; groups above 4 become deduplicated target-local 4-nearest sibling neighborhoods",
            "pair_requires_shared_physical_siblings_minimum": 2,
            "ordered_reference_query_pairs": True,
            "train_pair_range": [20000, 50000],
            "validation_pair_range": [2000, 5000],
            "sampling": "hash-ranked quadrant and near/far balanced reachable views; Drawer/Doorway balanced at loader",
            "inference_inputs": [
                "reference owner-group RGB crop", "reference sibling-union mask",
                "reference sibling-centroid mask", "query owner-group RGB crop",
                "query sibling-union mask", "query sibling-centroid mask",
            ],
            "forbidden_inputs": [
                "native owner yaw", "camera pose", "metric depth", "AI2-THOR object coordinates",
                "OA-V2 prediction", "target slot truth",
            ],
        },
        "architecture": {
            "name": "GRAIL-R1C-L Task-Trained Symmetry-Marginalized Pairwise Owner Coordinate",
            "backbone": "DINOv2-S shared RGB encoder; last 2/12 blocks trainable",
            "mask_adapter": "2-channel 14x14 stride-14 convolution added to patch tokens",
            "pair_module": "2 bidirectional cross-attention blocks, 6 heads, hidden size 384",
            "head": "36 bins x 10 degrees",
            "loss": "negative log total probability of all bins inducing the native-correct slot permutation",
            "exchange_consistency_weight": 0.05,
            "seeds": [1701, 2701],
            "model_selection": "best validation slot uplift over frozen OA-V2; no architecture selection",
        },
        "development_stop": {
            "minimum_validation_slot_uplift_over_oa_v2": 8,
            "if_failed": "STOP_R1C_L_WITHOUT_FINAL_TEST",
        },
        "final_gates": {
            "slot_uplift_over_same_cohort_oa_v2_minimum": 12,
            "slot_minimum": 55,
            "referent_uplift_minimum": 12,
            "referent_minimum": 55,
            "complete_uplift_minimum": 8,
            "complete_minimum": 40,
            "wrong_target_maximum": 2,
            "absence_false_commit_maximum": 1,
            "candidate_permutation": 156,
            "drawer_and_doorway_positive_uplift": True,
        },
        "boundaries": [
            "FRESH_HOUSE_DISJOINT", "RGB_MASK_ONLY_AT_INFERENCE",
            "ONE_ARCHITECTURE_TWO_SEEDS_MAX", "STOP_BEFORE_DEPTH_GEOMETRY_AND_M2",
        ],
        "claim_ceiling": "synthetic ProcTHOR Development mechanism evidence only; no natural, device, product, safety, or M2 authority",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.train_dataset, args.test_dataset, args.runtime_image_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({role: [row["house_index"] for row in rows]
                      for role, rows in manifest["rosters"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
