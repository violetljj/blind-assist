"""Validate the frozen DTR Final Reckoning Roster contract and byte locks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist-dtr-final-reckoning-roster-v1"
EXPECTED_STRATA = {
    "S01_CLEAN_CONSTANT_MOTION",
    "S02_SINGLE_FRAME_DROPOUT",
    "S03_MULTI_FRAME_DROPOUT",
    "S04_LATERAL_CROSSING",
    "S05_RECEDING_NEAR_MISS",
    "S06_FRAGMENTATION_ID_INSTABILITY",
    "S07_CURVED_WEARER_ROUTE",
    "S08_STATIC_PSEUDO_MOTION_EGO_ROTATION",
    "S09_PARTIAL_VISIBILITY_SURFACE_FRAGMENTATION",
    "S10_DISAPPEAR_REAPPEAR_CLEAR_REONSET",
}
EXPECTED_ARMS = {
    "RADIAL_TTC",
    "FINITE_DIFFERENCE_CV_ROUTE_TUBE",
    "KALMAN_CV_ROUTE_TUBE",
    "KALMAN_CV_ROUTE_TUBE_HYSTERESIS_0P60S",
    "CAUSAL_CTRV_ROUTE_TUBE",
    "TINY_LEARNED_PREDICTOR",
    "X24_CORE",
    "X73_STRUCTURAL_GEOMETRY",
    "X94_EVIDENCE_MODEL",
    "X94_EVIDENCE_PLUS_SIMPLE_HYSTERESIS_0P60S",
    "X95_EVENT_CHALLENGER",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(label)


def validate(protocol: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    require(protocol.get("schema") == SCHEMA, "schema")
    require(protocol.get("terminal_decision_contract", {}).get("x97_forbidden") is True, "x97")
    source = protocol["source_design"]
    groups = source["seed_groups"]
    require(len(groups) == 3, "seed_group_count")
    require(len({int(row["capture_seed"]) for row in groups}) == 3, "seed_uniqueness")
    require(sum(row["role"] == "LEARNED_ARM_FIT_ONLY" for row in groups) == 1, "fit_group")
    require(sum(row["role"] == "FINAL_ADJUDICATION" for row in groups) == 2, "final_groups")
    strata = {row["stratum_id"] for row in source["strata"]}
    require(strata == EXPECTED_STRATA, "strata")
    require(source["total_episode_count"] == len(groups) * len(strata) == 30, "episode_count")
    arms = {row["arm_id"] for row in protocol["arms"]}
    require(arms == EXPECTED_ARMS, "arms")
    require(len(protocol["arms"]) == len(arms), "duplicate_arms")

    evaluator = protocol["evaluator_contract"]
    require(evaluator["matching"].startswith("maximum_one_to_one"), "event_matching")
    require(evaluator["report_per_stratum"] is True, "stratum_reporting")
    require(evaluator["uncertainty"]["cluster_unit"] == "episode", "cluster_unit")
    retention = protocol["retention_contract"]
    require(retention["asset_catalog_authority_token"] == "sealed_final", "retention_authority")
    require("raw_sensor_shards_rgb_depth_instance_witness" in retention["must_retain"], "raw_retention")
    require(protocol["materialization_authority"]["authorized_now"] is False, "premature_materialization")
    require(
        protocol["materialization_authority"]["blocking_locks"]
        == ["SOURCE_CELL_MATERIALIZER_AND_SOURCE_ONLY_GEOMETRY_GATES"],
        "blocking_locks",
    )

    locks: dict[str, str] = {}
    for row in protocol["implementation_locks"]:
        path = (repo_root / row["path"]).resolve(strict=True)
        path.relative_to(repo_root.resolve(strict=True))
        actual = sha256_file(path)
        require(actual == row["sha256"], f"implementation_hash:{row['lock_id']}")
        locks[row["lock_id"]] = actual
    return {
        "status": "FINAL_RECKONING_ROSTER_DESIGN_AND_ARMS_VALID_PENDING_SOURCE_MATERIALIZER",
        "roster_id": protocol["roster_id"],
        "seed_groups": len(groups),
        "strata": len(strata),
        "episodes": source["total_episode_count"],
        "arms": len(arms),
        "verified_implementation_locks": locks,
        "materialization_authorized": False,
        "blocking_locks": protocol["materialization_authority"]["blocking_locks"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = validate(protocol, repo_root=args.repo_root.resolve(strict=True))
    result["protocol_sha256"] = sha256_file(protocol_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
