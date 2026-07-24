#!/usr/bin/env python3
"""Audit whether the observability program is blocked by real-world authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_ustrf_observability_program_authority_terminal_r0"
CONFIG_SCHEMA = f"{SCHEMA}_config"
STAGE = "USTRF_OBSERVABILITY_PROGRAM_REAL_WORLD_AUTHORITY_TERMINAL_R0"
TERMINALS = (
    "FAIL_CLOSED_PROGRAM_AUDIT_INCOMPLETE",
    "NEXT_STAGE_REAL_WORLD_AUTHORITY_PRESENT",
    "EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY",
)
IMPLEMENTATIONS = {
    "producer": "scripts/research/ustrf_route_target_evidence_closure/audit_ustrf_observability_program_authority_terminal_r0.py",
    "validator": "scripts/research/ustrf_route_target_evidence_closure/validate_ustrf_observability_program_authority_terminal_r0.py",
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_root_not_object:{path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_config(repo: Path, path: Path) -> dict[str, Any]:
    config = load_json(path)
    require(config["schema"] == CONFIG_SCHEMA, "config_schema_drift")
    require(config["stage"] == STAGE, "stage_drift")
    require(config["status"] == "frozen_before_execution", "config_not_frozen")
    require(tuple(config["terminal_states"]) == TERMINALS, "terminal_order_drift")
    for label, binding in config["bindings"].items():
        source = repo / binding["path"]
        require(source.is_file(), f"{label}_missing")
        require(sha256_file(source) == binding["sha256"], f"{label}_sha256_drift")
    digests = config["research_implementation_digests"]
    require(set(digests) == set(IMPLEMENTATIONS), "implementation_digest_keys_drift")
    for label, relative in IMPLEMENTATIONS.items():
        require(sha256_file(repo / relative) == digests[label], f"{label}_implementation_drift")
    return config


def audit(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(repo, config_path)
    bound = {
        key: repo / value["path"]
        for key, value in config["bindings"].items()
        if key != "source_goal" and key != "stable_research_adapter"
    }
    g0 = load_json(bound["g0_validation"])
    jrdb_source = load_json(bound["jrdb_single_frame_validation"])
    jrdb_motion = load_json(bound["jrdb_egomotion_validation"])
    arcore = load_json(bound["arcore_freshness"])
    replay_r2 = load_json(bound["sensor_replay_r2"])
    replay_r3 = load_json(bound["sensor_replay_r3"])
    status_text = bound["sanpo_current_status"].read_text(encoding="utf-8")

    checks = {
        "g0_source_authority_absent": (
            g0.get("status") == "VALID" and g0.get("terminal_state") == "SOURCE_AUTHORITY_ABSENT"
        ),
        "new_source_transport_present": (
            jrdb_source.get("status") == "VALID"
            and jrdb_source.get("terminal_state") == "RGB_TIME_TRANSFORM_CANARY_PRESENT"
        ),
        "global_affine_availability_insufficient": (
            jrdb_motion.get("status") == "VALID"
            and jrdb_motion.get("terminal_state") == "EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT"
        ),
        "arcore_fresh_depth_insufficient": (
            arcore.get("fresh_source_aligned_raw_depth_count") == 1
            and arcore.get("raw_depth_candidate_count") == 861
            and arcore["raw_pose_world_frame_stability"] == "EPHEMERAL_PER_FRAME"
            and not arcore["authorization"]["vio_gate_open"]
            and not arcore["authorization"]["raw_depth_metric_geometry_gate_open"]
        ),
        "rgbd_transport_present_but_route_event_absent": (
            replay_r2.get("geometry_transport_source_pass_count") == 3
            and replay_r2["route_event_review_consensus"]["event_truth_authority"] is False
            and replay_r2.get("algorithm_closed_loop_proven") is False
            and replay_r2.get("verdict") == "DO_NOT_SELECT_HARDWARE"
        ),
        "dynamic_sources_route_event_rejected": (
            replay_r3["route_event_review_consensus"]["all_sources_admitted"] is False
            and replay_r3["route_event_review_consensus"]["event_truth_authority"] is False
            and all(
                source["route_event_admitted"] is False
                for source in replay_r3["route_event_review_consensus"]["sources"]
            )
            and replay_r3.get("verdict") == "DO_NOT_SELECT_HARDWARE"
        ),
        "current_status_real_authority_blocked": (
            "BLOCKED_ON_SOURCE_ALIGNED_METRIC_DEPTH_AND_INTER_FRAME_STABLE_POSE" in status_text
        ),
        "higher_authority_closed": (
            replay_r2.get("u0_authorized") is False
            and replay_r2.get("production_authority") is False
            and replay_r3.get("u0_authorized") is False
            and replay_r3.get("production_authority") is False
        ),
    }
    terminal = (
        "EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY"
        if all(checks.values())
        else "NEXT_STAGE_REAL_WORLD_AUTHORITY_PRESENT"
    )
    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": terminal,
        "process_id": os.getpid(),
        "config_sha256": sha256_file(config_path),
        "checks": checks,
        "blockers": {
            "canonical_source_authority": "absent_in_current_41_sequence_pack",
            "fresh_metric_geometry": "arcore_1_of_861_and_world_pose_ephemeral",
            "route_truth": "no_hash_bound_intended_route_authority",
            "event_truth": "no_admitted_real_world_route_event_lifecycle",
            "consent_or_collection_authority": "not_authorized_by_continuous_goal",
        },
        "non_claims": {
            "core_hypothesis_rejected": False,
            "task_unobservable_with_authoritative_inputs": False,
            "all_metric_depth_or_vio_infeasible": False,
            "production_ready": False,
        },
        "resume_requirements": [
            "fresh_frame_bound_metric_geometry_with_inter_frame_stable_pose",
            "hash_bound_intended_route_provider_truth",
            "independent_route_event_lifecycle_truth",
            "explicit_authority_for_new_participant_or_real_world_collection_if_required",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    result = audit(repo, config_path)
    output = repo / load_json(config_path)["outputs"]["receipt"]
    atomic_write(output, canonical_bytes(result))
    print(json.dumps({"terminal_state": result["terminal_state"], "process_id": result["process_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

