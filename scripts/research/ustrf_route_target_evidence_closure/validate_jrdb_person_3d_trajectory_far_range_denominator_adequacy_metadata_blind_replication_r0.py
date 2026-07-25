"""Validate the frozen JRDB far-range replication without rewriting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import freeze_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0 as freezer
import run_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0 as runner


SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_far_range_denominator_"
    "adequacy_metadata_blind_replication_r0_validation"
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conserved(value: dict[str, Any]) -> bool:
    return value["expected"] == sum(
        value.get(name, 0)
        for name in ("sensor-supported", "annotation-only", "abstained", "invalid")
    )


def validate(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze_path = repo / config["outputs"]["sequence_freeze"]
    initial_path = repo / config["outputs"]["initial_metadata_freeze"]
    denominator_path = repo / config["outputs"]["denominator_ledger"]
    denominator_receipt_path = repo / config["outputs"]["denominator_receipt"]
    support_path = repo / config["outputs"]["support_ledger"]
    support_receipt_path = repo / config["outputs"]["support_receipt"]
    actual_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    initial_freeze = json.loads(initial_path.read_text(encoding="utf-8"))
    denominator = json.loads(denominator_path.read_text(encoding="utf-8"))
    denominator_receipt = json.loads(
        denominator_receipt_path.read_text(encoding="utf-8")
    )
    support = json.loads(support_path.read_text(encoding="utf-8"))
    support_receipt = json.loads(support_receipt_path.read_text(encoding="utf-8"))
    rebuilt_freeze = freezer.build_freeze(repo, config_path)
    rebuilt_support, rebuilt_receipt = runner.aggregate(
        repo, config_path, config, actual_freeze, denominator
    )
    identity_fields = (
        "sequence",
        "window_first_position",
        "window_last_position",
        "frame_first_stem",
        "frame_last_stem",
        "frame_count",
    )
    initial_identity = [
        {key: row[key] for key in identity_fields} for row in initial_freeze["selected"]
    ]
    amended_identity = [
        {key: row[key] for key in identity_fields} for row in actual_freeze["selected"]
    ]
    adequate = denominator["adequate_sequences"]
    denominator_rows = {row["sequence"]: row for row in denominator["per_sequences"]}
    support_rows = {row["sequence"]: row for row in support["per_sequences"]}
    kernel_hashes = {
        name: sha256_file(repo / value["path"])
        for name, value in config["frozen_kernel"].items()
        if isinstance(value, dict) and "path" in value and "sha256" in value
    }
    implementation_paths = [
        "scripts/research/ustrf_route_target_evidence_closure/freeze_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0.py",
        "scripts/research/ustrf_route_target_evidence_closure/audit_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0.py",
        "scripts/research/ustrf_route_target_evidence_closure/run_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0.py",
        "scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0.py",
        "scripts/research/ustrf_route_target_evidence_closure/test_jrdb_person_3d_trajectory_far_range_denominator_adequacy_metadata_blind_replication_r0.py",
    ]
    implementation_hashes = {
        path: sha256_file(repo / path) for path in implementation_paths
    }
    per_sequence_internal = True
    minimum = int(config["frozen_kernel"]["invariants"]["minimum_fused_in_box_points"])
    for sequence in adequate:
        source_binding = support_rows[sequence]["source_ledger"]
        source = json.loads((repo / source_binding["path"]).read_text(encoding="utf-8"))
        per_sequence_internal &= sha256_file(repo / source_binding["path"]) == source_binding[
            "sha256"
        ]
        per_sequence_internal &= all(
            conserved(value) for value in source["denominators"].values()
        )
        for row in source["object_frames"]:
            if row["cross_modal_presence"] == "2d-only":
                per_sequence_internal &= row["classification"] == "abstained"
                continue
            if row["classification"] == "invalid":
                continue
            count = int(row["fused_in_box_points"])
            expected = (
                "sensor-supported"
                if count >= minimum
                else ("annotation-only" if count == 0 else "abstained")
            )
            per_sequence_internal &= row["classification"] == expected
    nonadequate = sorted(
        set(row["sequence"] for row in actual_freeze["selected"]) - set(adequate)
    )
    checks = {
        "config_identity": config["stage"] == runner.STAGE,
        "metadata_freeze_exact_rebuild": freezer.canonical_bytes(rebuilt_freeze)
        == freeze_path.read_bytes(),
        "prepayload_amendment_identity_unchanged": initial_identity == amended_identity,
        "eight_unseen_frozen_windows": len(actual_freeze["selected"]) == 8
        and not (
            set(config["freeze_boundary"]["previously_seen_sequences"])
            & {row["sequence"] for row in actual_freeze["selected"]}
        ),
        "all_windows_360_continuous": all(
            row["frame_count"] == 360
            and int(row["frame_last_stem"]) - int(row["frame_first_stem"]) + 1 == 360
            for row in actual_freeze["selected"]
        ),
        "denominator_gate_exact": (
            config["denominator_adequacy"]["minimum_40_plus_object_frames_per_sequence"]
            == 100
            and config["denominator_adequacy"]["minimum_adequate_sequences"] == 3
        ),
        "denominator_receipt_bound": denominator_receipt[
            "denominator_ledger_sha256"
        ]
        == sha256_file(denominator_path),
        "four_sequences_adequate": len(adequate) == 4
        and all(
            denominator_rows[sequence]["far_40_plus_object_frames"] >= 100
            and denominator_rows[sequence]["near_0_20_object_frames"] >= 100
            for sequence in adequate
        ),
        "support_only_after_gate": denominator_receipt["pcd_support_authorized"] is True,
        "no_nonadequate_support_execution": all(
            not (runner.root(repo, sequence) / "support-ledger.json").exists()
            for sequence in nonadequate
        ),
        "frozen_kernel_hashes": all(
            kernel_hashes[name] == value["sha256"]
            for name, value in config["frozen_kernel"].items()
            if isinstance(value, dict) and "path" in value and "sha256" in value
        ),
        "execution_implementations_final_hash_bound": len(implementation_hashes) == 5
        and all(len(value) == 64 for value in implementation_hashes.values()),
        "per_sequence_ledger_and_point_gate_recomputed": per_sequence_internal,
        "far_denominators_match_label_only_gate": all(
            support_rows[sequence]["far_range"]["far_40_plus_denominator"]
            == denominator_rows[sequence]["far_40_plus_object_frames"]
            for sequence in adequate
        ),
        "support_aggregate_exact_rebuild": runner.canonical_bytes(rebuilt_support)
        == support_path.read_bytes(),
        "support_receipt_exact_rebuild": runner.canonical_bytes(rebuilt_receipt)
        == support_receipt_path.read_bytes(),
        "pooled_denominators_conserved": all(
            conserved(value) for value in support["pooled_denominators"].values()
        ),
        "required_parallel_denominator_reports": all(
            all(name in row for name in ("three_d_only", "occlusion", "sparse_pointcloud"))
            for row in support["per_sequences"]
        ),
        "four_sequence_direction_replicated": support["far_range_replication"][
            "adequate_sequence_count"
        ]
        == 4
        and support["far_range_replication"]["status"] == "DIRECTION_REPLICATED"
        and all(row["far_range"]["support_decline"] for row in support["per_sequences"]),
        "terminal_legal": support_receipt["terminal_state"]
        == "FAR_RANGE_SUPPORT_DECLINE_REPLICATED",
        "diagnostic_authority_only": support_receipt["authority"]["ceiling"]
        == "DIAGNOSTIC"
        and not any(
            support_receipt["authority"][key]
            for key in (
                "candidate_selection",
                "route_risk",
                "event_lifecycle",
                "alert_logic",
                "android",
                "human_safety",
                "production",
            )
        ),
        "temporary_bags_cleaned": not any(
            (repo / "artifacts.local/tmp/jrdb-far-range-denominator-r0").glob("*.bag")
        ),
    }
    return {
        "schema": SCHEMA,
        "stage": config["stage"],
        "status": "VALID" if all(checks.values()) else "INVALID",
        "checks": checks,
        "config_sha256": sha256_file(config_path),
        "sequence_freeze_sha256": sha256_file(freeze_path),
        "denominator_ledger_sha256": sha256_file(denominator_path),
        "support_ledger_sha256": sha256_file(support_path),
        "support_receipt_sha256": sha256_file(support_receipt_path),
        "recomputed_terminal_state": rebuilt_receipt["terminal_state"],
        "adequate_sequences": adequate,
        "implementation_hashes": implementation_hashes,
        "authority": config["authority"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    result = validate(repo, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = repo / config["outputs"]["validation"]
    require(not output.exists(), "validation already exists; never overwrite")
    runner.write_canonical(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": sum(result["checks"].values()),
                "total": len(result["checks"]),
            }
        )
    )
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
