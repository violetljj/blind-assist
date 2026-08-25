#!/usr/bin/env python3
"""Freeze a test-house roster without reading AI2-THOR teacher outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from grail_procthor_native_m0 import canonical_sha256, sha256_file


EXPECTED_TEST_SHA256 = "9a9fa6f134e76fe87f3fd92c00883651cf9fadf4e9ad4072d6d73be229f001dc"
DATASET_REVISION = "439193522244720b86d8c81cde2e51e3a4d150cf"
AI2THOR_COMMIT = "f0825767cd50d69f666c7f282e54abfe58f1e917"
SELECTION_SALT = "BLINDASSIST_GRAIL_PROCTHOR_NATIVE_M0_V1"


def freeze(
    dataset: Path,
    docker_image_id: str,
    roster_size: int = 12,
    selection_salt: str = SELECTION_SALT,
    excluded_indices: tuple[int, ...] = (0,),
    manifest_version: int = 1,
) -> dict:
    dataset_hash = sha256_file(dataset)
    if dataset_hash != EXPECTED_TEST_SHA256:
        raise ValueError(f"unexpected ProcTHOR test SHA-256: {dataset_hash}")
    candidates = []
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in excluded_indices:
                continue
            house = json.loads(line)
            house_hash = canonical_sha256(house)
            rank = hashlib.sha256(
                f"{selection_salt}:{dataset_hash}:{index}:{house_hash}".encode("ascii")
            ).hexdigest()
            candidates.append((rank, index, house_hash))
    if len(candidates) < roster_size:
        raise ValueError("dataset is smaller than requested roster")
    selected = sorted(candidates)[:roster_size]
    return {
        "schema": f"blindassist_grail_procthor_native_m0_manifest_v{manifest_version}",
        "profile": "FRESH_SOURCE_DISJOINT_ONE_SHOT_TEST",
        "frozen_before_ai2thor_test_outcome": True,
        "source": {
            "dataset": "ProcTHOR-10K test",
            "official_repository": "https://github.com/allenai/procthor-10k",
            "license": "Apache-2.0",
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256": dataset_hash,
            "development_split": "ProcTHOR-10K val",
            "held_out_split": "ProcTHOR-10K test",
        },
        "selection": {
            "algorithm": "lowest SHA256 rank over immutable dataset hash, house index, and canonical house hash",
            "salt": selection_salt,
            "excluded_indices": list(excluded_indices),
            "excluded_reason": "historical or earlier formal runtime work consumed these indices",
            "reads_runtime_or_teacher_outcome": False,
        },
        "runtime": {
            "ai2thor_version": "5.0.0",
            "ai2thor_commit": AI2THOR_COMMIT,
            "platform": "Linux64/Xvfb/Mesa software GL/FIFO",
            "docker_image_id": docker_image_id,
        },
        "controller": {
            "grid_size_m": 0.25,
            "snap_to_grid": False,
            "yaw_step_deg": 30,
            "horizons_deg": [0],
            "standing": [True],
            "visibility_distance_m": 1.5,
            "nearby_query_radius_m": 1.75,
        },
        "roster": [
            {"rank": rank, "house_index": index, "house_sha256": house_hash}
            for rank, index, house_hash in selected
        ],
        "gates": {
            "minimum_scenes": 8,
            "minimum_targets": 128,
            "minimum_target_types": 6,
            "minimum_action_canaries": 8,
            "minimum_none_cases": 8,
            "minimum_valid_pose_coverage": 0.80,
            "minimum_local_stability": 0.90,
        },
        "one_shot_semantics": {
            "on_gate_fail": "STOP_BEFORE_M1",
            "replay_after_outcome": False,
            "threshold_or_target_filter_tuning_after_outcome": False,
            "m1_authorized_only_if_all_gates_pass": True,
        },
        "claim_ceiling": (
            "synthetic ProcTHOR 3D and AI2-THOR native reachability/interactable-pose/action mechanics only; "
            "no RGB, natural-scene, real-device, user, product, or safety claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roster-size", type=int, default=12)
    parser.add_argument("--selection-salt", default=SELECTION_SALT)
    parser.add_argument("--exclude-index", type=int, action="append", default=[])
    parser.add_argument("--manifest-version", type=int, default=1)
    args = parser.parse_args()
    excluded = tuple(sorted(set(args.exclude_index or [0])))
    manifest = freeze(
        args.dataset,
        args.docker_image_id,
        args.roster_size,
        args.selection_salt,
        excluded,
        args.manifest_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"roster": manifest["roster"], "gates": manifest["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
