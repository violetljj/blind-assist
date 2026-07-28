"""R1 guard-target geometry repair producer.

R0 artifacts remain immutable. This producer reuses all MAIN records, all
numeric seeds, all trajectories, fixtures, projective samples, and replay
evidence. It deterministically replaces only the eight GUARD scene records
with a rendered persistent middle-depth target layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry as r0,
)


IMPLEMENTATION_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_R1"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
R0_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r0"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r1"
)
AMENDMENT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_SPEC_REPAIR_R1_2026-07-28.json"
)
EXPECTED_R0_RECEIPT_SHA256 = (
    "72e0b8e042be9eb6208389eb8d83e9e9e4ad28e54ec82f7064b5387cc1abd279"
)
COPIED_EVIDENCE = (
    "runtime_manifest.json",
    "trajectory_manifest.json",
    "analytic_fixture_ledger.json",
    "deterministic_replay_ledger.json",
    "projective_sample_ledger.json",
)


def _load_r0_records() -> list[dict[str, Any]]:
    path = R0_EVIDENCE / "all_seed_geometry_manifest.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _reference_bounds(
    u0: float, u1: float, v0: float, v1: float, z: float
) -> tuple[float, float, float, float]:
    return (
        (u0 - r0.K[0, 2]) / r0.K[0, 0] * z,
        (u1 - r0.K[0, 2]) / r0.K[0, 0] * z,
        (v0 - r0.K[1, 2]) / r0.K[1, 1] * z,
        (v1 - r0.K[1, 2]) / r0.K[1, 1] * z,
    )


def build_guard_scene_r1(block: str, ordinal: int) -> dict[str, Any]:
    seed = r0.derive_seed("GUARD", block, ordinal)
    objects: list[dict[str, Any]] = []
    object_id = 1
    for u0, u1 in ((0.0, 80.0), (280.0, 360.0)):
        for v0, v1 in ((0.0, 213.0), (213.0, 426.0), (426.0, 640.0)):
            z = 1.5 + 0.25 * r0.unit_hash(seed, object_id, "depth")
            objects.append(
                r0._surface(
                    object_id,
                    z,
                    *_reference_bounds(u0, u1, v0, v1, z),
                    seed,
                )
            )
            object_id += 1
    target_edges = np.linspace(220.0, 450.0, 6, dtype=np.float64)
    for v0, v1 in zip(target_edges[:-1], target_edges[1:]):
        z = 4.0
        objects.append(
            r0._surface(
                object_id,
                z,
                *_reference_bounds(80.0, 280.0, float(v0), float(v1), z),
                seed,
            )
        )
        object_id += 1
    objects.append(
        r0._surface(object_id, 18.0, -12.0, 12.0, -16.0, 16.0, seed)
    )
    scene = {
        "schema": "rcle.periodic_self_motion_counterfactual.p1_geometry_manifest.v2",
        "protocol_id": r0.PROTOCOL_ID,
        "namespace": "GUARD",
        "block": block,
        "ordinal": ordinal,
        "numeric_seed_uint64": seed,
        "camera": {
            "projection": "pinhole",
            "width_px": r0.WIDTH,
            "height_px": r0.HEIGHT,
            "intrinsic": r0.K.tolist(),
            "near_clip_m": 0.5,
            "far_clip_m": 25.0,
            "distortion": "none",
            "camera_axes": "+x right, +y down, +z optical forward",
        },
        "world": {
            "static": True,
            "moving_objects": False,
            "renderer": "deterministic analytic ray/rectangle z-buffer v1",
            "objects": objects,
        },
        "designated_target": {
            "object_id": 9,
            "world_point_m": [0.2, 0.1, 4.0],
            "role": "PERSISTENT_RENDERED_MIDDLE_DEPTH_GUARDRAIL_TARGET",
        },
        "repair_identity": (
            "R1_FIXED_CENTRAL_TARGET_LAYOUT_ALL_BLOCKS_ALL_GUARD_SEEDS"
        ),
    }
    scene["scene_geometry_sha256"] = r0.sha256_bytes(r0.canonical_bytes(scene))
    return scene


def build_guard_record_r1(record: dict[str, Any]) -> dict[str, Any]:
    block = str(record["block"])
    ordinal = int(record["ordinal"])
    expected_seed = r0.derive_seed("GUARD", block, ordinal)
    if int(record["numeric_seed_uint64"]) != expected_seed:
        raise ValueError(f"R0_GUARD_SEED_MISMATCH:{block}:{ordinal}")
    scene = build_guard_scene_r1(block, ordinal)
    arms = []
    for old_arm in record["arms"]:
        arm = dict(old_arm)
        arm["scene_geometry_sha256"] = scene["scene_geometry_sha256"]
        arms.append(arm)
    return {
        "record_type": "guardrail_cluster",
        "cluster_id": record["cluster_id"],
        "block": block,
        "ordinal": ordinal,
        "numeric_seed_uint64": expected_seed,
        "scene": scene,
        "reference_metrics": r0.reference_metrics(scene),
        "designated_middle_target_depth_m": 4.0,
        "approach_translation_m": 0.8,
        "arms": arms,
    }


def produce(output: Path) -> dict[str, Any]:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if amendment.get("formal_execution_authorized") is not False:
        raise ValueError("R1_FORMAL_EXECUTION_MUST_REMAIN_FALSE")
    r0_receipt_path = R0_EVIDENCE / "independent_geometry_validation_receipt.json"
    if r0.sha256_file(r0_receipt_path) != EXPECTED_R0_RECEIPT_SHA256:
        raise ValueError("R0_RECEIPT_HASH_MISMATCH")
    records = _load_r0_records()
    main = [item for item in records if item["record_type"] == "main_cluster"]
    guards = [
        item for item in records if item["record_type"] == "guardrail_cluster"
    ]
    if len(main) != 80 or len(guards) != 8:
        raise ValueError("R0_RECORD_COUNT_MISMATCH")
    output.mkdir(parents=True, exist_ok=True)
    for name in COPIED_EVIDENCE:
        shutil.copyfile(R0_EVIDENCE / name, output / name)
        if r0.sha256_file(R0_EVIDENCE / name) != r0.sha256_file(output / name):
            raise ValueError(f"COPIED_EVIDENCE_HASH_MISMATCH:{name}")
    repaired_guards = [build_guard_record_r1(item) for item in guards]
    manifest_path = output / "all_seed_geometry_manifest.jsonl"
    with manifest_path.open("wb") as stream:
        for record in [*main, *repaired_guards]:
            stream.write(r0.canonical_bytes(record))
    main_identity = {
        item["cluster_id"]: r0.sha256_bytes(r0.canonical_bytes(item))
        for item in main
    }
    receipt = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p1_generator_geometry_r1_receipt.v1"
        ),
        "protocol_id": r0.PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": "R1_GUARD_TARGET_GEOMETRY_MATERIALIZED",
        "r0_receipt_sha256": EXPECTED_R0_RECEIPT_SHA256,
        "main_cluster_count": 80,
        "main_record_sha256": main_identity,
        "guardrail_cluster_count": 8,
        "guardrail_numeric_seed_replacement_count": 0,
        "guardrail_trajectory_change_count": 0,
        "designated_rendered_target_count": 8,
        "artifact_sha256": {
            path.name: r0.sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "generator_r1_receipt.json"
        },
        "rcle_output_accessed_or_executed": False,
        "quality_strength_calibrated": False,
        "performance_preflight_run": False,
        "formal_sequences_run": False,
        "formal_execution_authorized": False,
    }
    r0.write_json(output / "generator_r1_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = produce(args.output.resolve())
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "INTERVENTION_NOT_EVALUABLE",
                    "state": "HOLD_P1",
                    "error": f"{type(error).__name__}:{error}",
                    "formal_execution_authorized": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
