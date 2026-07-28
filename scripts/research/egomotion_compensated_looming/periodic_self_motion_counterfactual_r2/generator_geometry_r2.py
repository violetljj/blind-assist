"""R2 evidence materializer for the fail-closed R1 validator repair.

R2 does not change a numeric seed, scene, camera, trajectory, threshold, or the
all-seed manifest schema. It copies the hash-pinned R1 geometry evidence and
adds a producer-side two-build replay ledger for all eight repaired GUARD
scenes. R0 and the failed R1 validation receipt remain immutable predecessors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry as r0,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry_r1 as r1,
)


IMPLEMENTATION_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_R2"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
R1_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r2"
)
AMENDMENT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_VALIDATOR_REPAIR_R2_2026-07-28.json"
)
EXPECTED_R1_FAILED_RECEIPT_SHA256 = (
    "af00df05c115036ea31bb3d05addbebfcebad73122d2b354f7e52170c2277e9a"
)
COPIED_EVIDENCE_SHA256 = {
    "all_seed_geometry_manifest.jsonl": (
        "3dcf37496997a1edb2e47871c0dfc5185fd207016a26a86e29514412484e7ac6"
    ),
    "analytic_fixture_ledger.json": (
        "6e9f8afb80ac8cb647d137daf3c8238cfa449dbff528c49d14422aa8c987bb25"
    ),
    "deterministic_replay_ledger.json": (
        "663971f546ba3f28bf29aca8b59e02c43cab944fb737938c050fc6110763a80d"
    ),
    "projective_sample_ledger.json": (
        "2576a26dcd5ee81ae5058c94c129ca8dcd72a68b1ade4629b5fe8dc53d0ebcaa"
    ),
    "runtime_manifest.json": (
        "06020e73f46b5da0c7219b398940274539c566956877bdbe75ce3097aa3375e2"
    ),
    "trajectory_manifest.json": (
        "c394641b6419c7a58a58c1bc485e2783cd710e8ba2893415dd6687c20b7d2652"
    ),
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _build_guard_replay(records: list[dict[str, Any]]) -> dict[str, Any]:
    guards = [
        item for item in records if item.get("record_type") == "guardrail_cluster"
    ]
    items = []
    for record in guards:
        block = str(record["block"])
        ordinal = int(record["ordinal"])
        first = r1.build_guard_scene_r1(block, ordinal)
        second = r1.build_guard_scene_r1(block, ordinal)
        first_hash = r0.sha256_bytes(r0.canonical_bytes(first))
        second_hash = r0.sha256_bytes(r0.canonical_bytes(second))
        manifest_hash = r0.sha256_bytes(r0.canonical_bytes(record["scene"]))
        items.append(
            {
                "kind": "r1_guard_scene",
                "block": block,
                "ordinal": ordinal,
                "numeric_seed_uint64": int(record["numeric_seed_uint64"]),
                "first_scene_sha256": first_hash,
                "second_scene_sha256": second_hash,
                "manifest_scene_sha256": manifest_hash,
                "match": (
                    first_hash == second_hash == manifest_hash
                    and first == second == record["scene"]
                ),
            }
        )
    mismatch_count = sum(item["match"] is not True for item in items)
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p1_guard_scene_replay_r2.v1"
        ),
        "implementation_id": IMPLEMENTATION_ID,
        "items": items,
        "mismatch_count": mismatch_count,
    }


def produce(output: Path) -> dict[str, Any]:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if amendment.get("formal_execution_authorized") is not False:
        raise ValueError("R2_FORMAL_EXECUTION_MUST_REMAIN_FALSE")
    failed_receipt = R1_EVIDENCE / "independent_geometry_validation_receipt.json"
    if r0.sha256_file(failed_receipt) != EXPECTED_R1_FAILED_RECEIPT_SHA256:
        raise ValueError("R1_FAILED_RECEIPT_HASH_MISMATCH")
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in COPIED_EVIDENCE_SHA256.items():
        source = R1_EVIDENCE / name
        if r0.sha256_file(source) != expected:
            raise ValueError(f"R1_EVIDENCE_HASH_MISMATCH:{name}")
        shutil.copyfile(source, output / name)
        if r0.sha256_file(output / name) != expected:
            raise ValueError(f"R2_COPIED_EVIDENCE_HASH_MISMATCH:{name}")
    records = _load_records(output / "all_seed_geometry_manifest.jsonl")
    replay = _build_guard_replay(records)
    r0.write_json(output / "guard_scene_replay_ledger.json", replay)
    if replay["mismatch_count"] != 0 or len(replay["items"]) != 8:
        raise ValueError("R2_GUARD_REPLAY_MISMATCH")
    package_manifest = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p1_geometry_package_r2.v1"
        ),
        "protocol_id": r0.PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runtime_manifest_role": (
            "byte-identical inherited R0 renderer/environment evidence; "
            "its embedded R0 implementation_id is intentionally historical"
        ),
        "outer_jsonl_record_contract": (
            "unchanged main_cluster or guardrail_cluster record envelope"
        ),
        "scene_schema_union": {
            "main_cluster": (
                "rcle.periodic_self_motion_counterfactual."
                "p1_geometry_manifest.v1"
            ),
            "guardrail_cluster": (
                "rcle.periodic_self_motion_counterfactual."
                "p1_geometry_manifest.v2"
            ),
        },
        "record_counts": {"main_cluster": 80, "guardrail_cluster": 8},
        "all_seed_manifest_sha256": COPIED_EVIDENCE_SHA256[
            "all_seed_geometry_manifest.jsonl"
        ],
        "guard_scene_replay_sha256": r0.sha256_file(
            output / "guard_scene_replay_ledger.json"
        ),
        "formal_execution_authorized": False,
        "automatic_p2_authority": False,
    }
    r0.write_json(output / "package_manifest.json", package_manifest)
    receipt = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p1_generator_geometry_r2_receipt.v1"
        ),
        "protocol_id": r0.PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": "R2_VALIDATOR_REPAIR_EVIDENCE_MATERIALIZED",
        "r1_failed_receipt_sha256": EXPECTED_R1_FAILED_RECEIPT_SHA256,
        "main_cluster_count": 80,
        "guardrail_cluster_count": 8,
        "main_record_change_count_from_r1": 0,
        "guardrail_record_change_count_from_r1": 0,
        "numeric_seed_replacement_count": 0,
        "trajectory_change_count": 0,
        "guard_scene_replay_count": 8,
        "scene_schema_union": package_manifest["scene_schema_union"],
        "artifact_sha256": {
            path.name: r0.sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "generator_r2_receipt.json"
        },
        "rcle_output_accessed_or_executed": False,
        "quality_strength_calibrated": False,
        "performance_preflight_run": False,
        "formal_sequences_run": False,
        "formal_execution_authorized": False,
    }
    r0.write_json(output / "generator_r2_receipt.json", receipt)
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
