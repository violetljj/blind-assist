"""Independent validator for the user-authorized W8 scheduler successor.

The predecessor W4 profile is retained only as an independent numeric
equivalence comparator.  It is not eligible for scheduler selection because it
did not observe its effective OpenBLAS thread count.  The successor W8 profile
must carry the observed legacy OpenBLAS=18/OpenCV=1 thread evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

from . import validate_p3_independent_r0 as base


class InvalidSchedulerSuccessor(ValueError):
    pass


def validate_predecessor_w4(
    directory: Path,
    identity_lock_path: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    value = base.load(directory / "success.json")
    if value.get("workers") != 4 or value.get("profile") != "W4":
        raise InvalidSchedulerSuccessor("PREDECESSOR_PROFILE")
    if value.get("terminal") != "PROFILE_COMPLETE / PREFLIGHT_ONLY":
        raise InvalidSchedulerSuccessor("PREDECESSOR_TERMINAL")
    if value.get("identity_lock_sha256") != base.sha256_file(identity_lock_path):
        raise InvalidSchedulerSuccessor("PREDECESSOR_IDENTITY_LOCK")
    if value.get("identity_set_sha256") != identity["identity_set_sha256"]:
        raise InvalidSchedulerSuccessor("PREDECESSOR_IDENTITY_SET")
    if (
        value.get("sequence_count") != 8
        or value.get("frame_count") != 8 * base.FRAME_COUNT
        or value.get("pair_count") != 8 * base.PAIR_COUNT
    ):
        raise InvalidSchedulerSuccessor("PREDECESSOR_COUNTS")
    receipts = value.get("sequence_receipts")
    expected_ids = [item["sequence_id"] for item in identity["identities"]]
    if (
        not isinstance(receipts, list)
        or [item.get("sequence_id") for item in receipts] != expected_ids
    ):
        raise InvalidSchedulerSuccessor("PREDECESSOR_RECEIPTS")
    for expected, receipt in zip(identity["identities"], receipts):
        for key in ("sequence_id", "cluster_kind", "arm", "numeric_seed_uint64"):
            if receipt.get(key) != expected[key]:
                raise InvalidSchedulerSuccessor(
                    f"PREDECESSOR_SEQUENCE_IDENTITY:{key}"
                )
        firewall = receipt.get("outcome_firewall", {})
        if any(
            firewall.get(key) is not False
            for key in (
                "response_values_emitted",
                "trigger_values_emitted",
                "scientific_interpretation",
            )
        ):
            raise InvalidSchedulerSuccessor("PREDECESSOR_OUTCOME_FIREWALL")
    resource = value.get("resource", {})
    if (
        resource.get("available_ram_at_launch_bytes", 0) < 8 * base.GIB
        or resource.get("minimum_available_ram_bytes", 0) < 4 * base.GIB
        or resource.get("sustained_paging") is not False
        or resource.get("heartbeat_max_interval_seconds", 31.0) > 30.0
    ):
        raise InvalidSchedulerSuccessor("PREDECESSOR_RESOURCE")
    if value.get("residual_worker_pids") != []:
        raise InvalidSchedulerSuccessor("PREDECESSOR_RESIDUAL")
    if (
        value.get("formal_execution_authorized") is not False
        or value.get("p4_activated") is not False
        or value.get("scientific_outcome_interpreted") is not False
    ):
        raise InvalidSchedulerSuccessor("PREDECESSOR_AUTHORITY")
    base._validate_telemetry(directory, 4)
    return value


def validate_successor(
    root: Path,
    identity_lock_path: Path,
    transport_lock_path: Path,
    analysis_lock_path: Path,
    predecessor_w4_directory: Path,
    successor_w8_directory: Path,
) -> dict[str, Any]:
    identity = base.validate_identity_lock(identity_lock_path)
    base.validate_transport_lock(root, transport_lock_path)
    base.validate_analysis_lock(root, analysis_lock_path)
    predecessor = validate_predecessor_w4(
        predecessor_w4_directory,
        identity_lock_path,
        identity,
    )
    successor = base.validate_profile(
        successor_w8_directory,
        8,
        identity_lock_path,
        identity,
    )
    predecessor_by_id = {
        item["sequence_id"]: item for item in predecessor["sequence_receipts"]
    }
    for item in successor["sequence_receipts"]:
        previous = predecessor_by_id[item["sequence_id"]]
        for field in (
            "scene_geometry_sha256",
            "frame_manifest_sha256",
            "ordered_pair_numeric_sha256",
            "transport_identity_sha256",
        ):
            if item[field] != previous[field]:
                raise InvalidSchedulerSuccessor(
                    f"PROFILE_EQUIVALENCE:{item['sequence_id']}:{field}"
                )
    projection = base.profile_projection(successor)
    if (
        not math.isfinite(projection["total_seconds"])
        or projection["total_seconds"] > base.WALL_CEILING_SECONDS
    ):
        raise InvalidSchedulerSuccessor("SUCCESSOR_W8_EXCEEDS_WALL_CEILING")
    formal_paths = [
        root
        / "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/p4_formal",
        root
        / "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/formal_480_plus_16",
    ]
    if any(path.exists() for path in formal_paths):
        raise InvalidSchedulerSuccessor("FORMAL_PATH_PRESENT")
    binding_paths = (
        identity_lock_path,
        transport_lock_path,
        analysis_lock_path,
        root
        / "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "p3_runtime_preflight_r0.py",
        root
        / "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "validate_p3_independent_r0.py",
        root
        / "scripts/research/egomotion_compensated_looming/"
        "tests_periodic_self_motion_counterfactual_r2/"
        "test_p3_runtime_preflight.py",
        root
        / "scripts/research/egomotion_compensated_looming/"
        "tests_periodic_self_motion_counterfactual_r2/"
        "test_p3_independent_validator.py",
        predecessor_w4_directory / "success.json",
        predecessor_w4_directory / "telemetry.json",
        successor_w8_directory / "success.json",
        successor_w8_directory / "telemetry.json",
        Path(__file__).resolve(),
    )
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p3_scheduler_successor_independent_receipt.v1"
        ),
        "protocol_id": base.PROTOCOL_ID,
        "p3_id": base.P3_ID,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED",
        "validated": True,
        "transport_equivalence": "VALID",
        "analysis_implementation_and_mutations": "VALID",
        "identity_count": 8,
        "profiles": {
            "W4_PREDECESSOR": {
                "role": "NUMERIC_EQUIVALENCE_COMPARATOR_ONLY",
                "eligible_for_selection": False,
                "reason": "EFFECTIVE_OPENBLAS_THREAD_COUNT_NOT_OBSERVED",
            },
            "W8_SCHEDULER_SUCCESSOR": {
                "role": "USER_AUTHORIZED_ELIGIBLE_SCHEDULER",
                "eligible_for_selection": True,
                "workers": 8,
                "opencv_threads_per_worker": 1,
                "openblas_threads_per_worker": 18,
            },
        },
        "profile_numeric_equivalence": "PASS",
        "selected_profile": "W8",
        "selection_rule": (
            "USER_AUTHORIZED_W8_SUCCESSOR_AFTER_RESOURCE_NUMERIC_AND_"
            "12_HOUR_GUARDS"
        ),
        "projection": projection,
        "measured_w8_wall_seconds": successor["timing"]["wall_seconds"],
        "bindings": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": base.sha256_file(path),
            }
            for path in binding_paths
        ],
        "formal_paths_absent": [
            path.relative_to(root).as_posix() for path in formal_paths
        ],
        "formal_seed_access": False,
        "formal_480_plus_16_run": False,
        "scientific_outcome_interpreted": False,
        "strength_retuned": False,
        "r3_threshold_or_three_pair_modified": False,
        "sequence16_android_realtime": False,
        "formal_execution_authorized": False,
        "p4_activated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--identity-lock", type=Path, required=True)
    parser.add_argument("--transport-lock", type=Path, required=True)
    parser.add_argument("--analysis-lock", type=Path, required=True)
    parser.add_argument("--predecessor-w4-directory", type=Path, required=True)
    parser.add_argument("--successor-w8-directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    receipt = validate_successor(
        root,
        (root / args.identity_lock).resolve(),
        (root / args.transport_lock).resolve(),
        (root / args.analysis_lock).resolve(),
        (root / args.predecessor_w4_directory).resolve(),
        (root / args.successor_w8_directory).resolve(),
    )
    base.write_exclusive((root / args.receipt).resolve(), receipt)
    print(
        json.dumps(
            {
                "terminal": receipt["terminal"],
                "selected_profile": receipt["selected_profile"],
                "projected_hours": (
                    receipt["projection"]["total_seconds"] / 3600.0
                ),
                "validated": receipt["validated"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
