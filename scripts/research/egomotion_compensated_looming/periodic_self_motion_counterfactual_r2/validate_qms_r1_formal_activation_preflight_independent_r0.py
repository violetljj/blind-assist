"""Independent validator for the QMS-R1 successor activation preflight.

The validator does not import the preflight producer or the QMS operator.  It
recomputes identity derivation, exclusion, geometry, transport, analysis,
guarded-host, and formal-firewall evidence before issuing a one-shot decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from . import p3_analysis_r0 as analysis
from . import p3_transport_r0 as transport
from . import validate_geometry_independent_r2_keyset_repair_r0 as geometry_validator


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_FORMAL_ACTIVATION_PREFLIGHT_R0"
)
EXPECTED_OPERATOR_SOURCE_SHA256 = (
    "5e66d270c1267d36e927cf47808337e6c1c0da68566e039c9a6ad35eb7c7e8c6"
)
EXPECTED_OPERATOR_IDENTITY = {
    "operator_id": "QMS_R1_MATERIAL_RESIDUAL_CONTRACTION",
    "clean_modulation_formula": "0.65+0.35*checker",
    "material_mean_modulation": 0.825,
    "low_modulation_formula": "0.825+0.15*(clean-0.825)",
    "alpha": 0.15,
    "pairing": "ONE_RAYCAST_SHARED_GEOMETRY",
    "domain": "PREQUANTIZATION_LINEAR_RGB",
    "psf_none": True,
}
GEOMETRY_EVIDENCE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0"
)
GEOMETRY_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R2_KEYSET_REPAIR_R0_2026-07-29.json"
)
TRANSPORT_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "R3_TRANSPORT_EQUIVALENCE_LOCK_R0_2026-07-29.json"
)
ANALYSIS_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "ANALYSIS_IMPLEMENTATION_LOCK_R0_2026-07-29.json"
)
GIB = 1024**3
WALL_CEILING_SECONDS = 12 * 3600


class InvalidIndependentPreflight(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(os.fspath(path), flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _collect_values(value: Any, key: str) -> set[Any]:
    found: set[Any] = set()
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key and isinstance(child, (str, int)):
                found.add(child)
            found.update(_collect_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_values(child, key))
    return found


def _derive(domain: str, block: str, kind: str, ordinal: int) -> dict[str, Any]:
    token = f"{TASK_ID}|{domain}|{block}|{kind}|{ordinal:02d}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return {
        "token": token,
        "token_sha256": digest,
        "numeric_seed_uint64": int.from_bytes(bytes.fromhex(digest)[:8], "big"),
    }


def validate_operator(root: Path, path: Path) -> dict[str, Any]:
    value = load_json(path)
    source = root / value.get("operator_source_path", "")
    identity_sha = hashlib.sha256(canonical_bytes(EXPECTED_OPERATOR_IDENTITY)).hexdigest()
    if (
        value.get("task_id") != TASK_ID
        or value.get("operator_identity") != EXPECTED_OPERATOR_IDENTITY
        or value.get("operator_identity_sha256") != identity_sha
        or value.get("operator_source_sha256") != EXPECTED_OPERATOR_SOURCE_SHA256
        or not source.is_file()
        or sha256_file(source) != EXPECTED_OPERATOR_SOURCE_SHA256
        or value.get("formal_execution_authorized") is not False
    ):
        raise InvalidIndependentPreflight("OPERATOR_LOCK")
    return {
        "operator_lock_sha256": sha256_file(path),
        "operator_source_sha256": EXPECTED_OPERATOR_SOURCE_SHA256,
        "operator_identity_sha256": identity_sha,
    }


def _exclusion_union(root: Path, sources: list[dict[str, Any]]) -> dict[str, set[Any]]:
    fields = (
        "numeric_seed_uint64",
        "token",
        "token_sha256",
        "cluster_id",
        "sequence_id",
        "scene_geometry_sha256",
    )
    union = {field: set() for field in fields}
    for source in sources:
        path_literal = source.get("path")
        if not isinstance(path_literal, str):
            raise InvalidIndependentPreflight("EXCLUSION_PATH")
        path = root / path_literal
        if source.get("label") == "OLD_PREFLIGHT_W8_SCENE_RECEIPTS":
            if not path.is_dir():
                raise InvalidIndependentPreflight("OLD_PREFLIGHT_RECEIPTS")
            for receipt in path.glob("*/receipt.json"):
                value = load_json(receipt)
                union["scene_geometry_sha256"].update(
                    _collect_values(value, "scene_geometry_sha256")
                )
            continue
        if not path.is_file() or sha256_file(path) != source.get("sha256"):
            raise InvalidIndependentPreflight("EXCLUSION_SOURCE_DRIFT")
        value = load_json(path)
        for field in fields:
            union[field].update(_collect_values(value, field))
    return union


def _assert_zero_overlap(
    candidates: list[dict[str, Any]], excluded: dict[str, set[Any]]
) -> dict[str, int]:
    overlaps = {}
    for field, old_values in excluded.items():
        new_values = _collect_values(candidates, field)
        overlaps[field] = len(new_values & old_values)
        if overlaps[field]:
            raise InvalidIndependentPreflight(f"IDENTITY_OVERLAP:{field}")
    return overlaps


def validate_formal_lock(
    root: Path, path: Path, operator_lock_sha256: str
) -> tuple[dict[str, Any], dict[str, int]]:
    value = load_json(path)
    identities = value.get("identities")
    seeds = value.get("seeds")
    if (
        value.get("task_id") != TASK_ID
        or not isinstance(identities, list)
        or len(identities) != 496
        or not isinstance(seeds, list)
        or len(seeds) != 88
        or value.get("counts", {}).get("main_sequences") != 480
        or value.get("counts", {}).get("guardrail_sequences") != 16
        or value.get("formal_execution_authorized") is not False
        or value.get("formal_sequences_run") != 0
        or value.get("operator_lock", {}).get("sha256") != operator_lock_sha256
    ):
        raise InvalidIndependentPreflight("FORMAL_LOCK_CARDINALITY")
    if len({item.get("sequence_id") for item in identities}) != 496:
        raise InvalidIndependentPreflight("FORMAL_SEQUENCE_UNIQUENESS")
    if len({item.get("numeric_seed_uint64") for item in seeds}) != 88:
        raise InvalidIndependentPreflight("FORMAL_SEED_UNIQUENESS")
    for seed in seeds:
        expected = _derive(
            "SUCCESSOR_FORMAL",
            seed.get("block"),
            seed.get("kind"),
            seed.get("ordinal"),
        )
        if any(seed.get(key) != expected[key] for key in expected):
            raise InvalidIndependentPreflight("FORMAL_SEED_DERIVATION")
    expected_set_sha = hashlib.sha256(canonical_bytes(identities)).hexdigest()
    if value.get("identity_set_sha256") != expected_set_sha:
        raise InvalidIndependentPreflight("FORMAL_IDENTITY_SET_HASH")
    excluded = _exclusion_union(root, value.get("exclusion_sources", []))
    overlaps = _assert_zero_overlap(identities + seeds, excluded)
    arm_keys = {
        (
            item.get("scene_geometry_sha256"),
            item.get("trajectory_sha256"),
            item.get("arm"),
        )
        for item in identities
    }
    if len(arm_keys) != 496:
        raise InvalidIndependentPreflight("FORMAL_ARM_IDENTITY_UNIQUENESS")
    return value, overlaps


def validate_preflight_lock(
    root: Path,
    path: Path,
    formal: dict[str, Any],
    formal_path: Path,
    operator_lock_sha256: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    value = load_json(path)
    identities = value.get("identities")
    seeds = value.get("seeds")
    if (
        value.get("task_id") != TASK_ID
        or not isinstance(identities, list)
        or len(identities) != 8
        or not isinstance(seeds, list)
        or len(seeds) != 2
        or value.get("workers") != 8
        or value.get("native_threads_per_worker") != 18
        or value.get("operator_lock_sha256") != operator_lock_sha256
        or value.get("successor_formal_identity_lock_sha256")
        != sha256_file(formal_path)
        or value.get("formal_execution_authorized") is not False
    ):
        raise InvalidIndependentPreflight("PREFLIGHT_LOCK_CARDINALITY")
    for seed in seeds:
        expected = _derive("PREFLIGHT", "ADVIO_14", seed.get("kind"), 0)
        if any(seed.get(key) != expected[key] for key in expected):
            raise InvalidIndependentPreflight("PREFLIGHT_SEED_DERIVATION")
    if value.get("identity_set_sha256") != hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest():
        raise InvalidIndependentPreflight("PREFLIGHT_IDENTITY_SET_HASH")
    excluded = _exclusion_union(root, value.get("exclusion_sources", []))
    for field in excluded:
        excluded[field].update(_collect_values(formal, field))
    overlaps = _assert_zero_overlap(identities + seeds, excluded)
    return value, overlaps


def validate_scientific_locks(root: Path) -> dict[str, Any]:
    geometry_receipt = geometry_validator.validate(
        root / GEOMETRY_EVIDENCE, root / GEOMETRY_LOCK
    )
    if (
        geometry_receipt.get("status") != "VALID"
        or geometry_receipt.get("terminal") != "GENERATOR_GEOMETRY_PASS"
        or geometry_receipt.get("gate_pass_count") != 14
        or geometry_receipt.get("failed_gates") != []
        or geometry_receipt.get("errors") != []
        or geometry_receipt.get("formal_sequences_run") is not False
    ):
        raise InvalidIndependentPreflight("ALL_SEED_GEOMETRY")
    tracked_transport = load_json(root / TRANSPORT_LOCK)
    recomputed_transport = transport.run_equivalence()
    if canonical_bytes(tracked_transport) != canonical_bytes(recomputed_transport):
        raise InvalidIndependentPreflight("R3_TRANSPORT_DRIFT")
    tracked_analysis = load_json(root / ANALYSIS_LOCK)
    recomputed_analysis = analysis.implementation_lock()
    if canonical_bytes(tracked_analysis) != canonical_bytes(recomputed_analysis):
        raise InvalidIndependentPreflight("ANALYSIS_LOCK_DRIFT")
    return {
        "all_seed_geometry": {
            "terminal": geometry_receipt["terminal"],
            "gate_pass_count": 14,
            "manifest_sha256": sha256_file(
                root / GEOMETRY_EVIDENCE / "all_seed_geometry_manifest.jsonl"
            ),
            "recomputed_receipt_sha256": hashlib.sha256(
                canonical_bytes(geometry_receipt)
            ).hexdigest(),
        },
        "r3_transport": {
            "terminal": recomputed_transport["terminal"],
            "lock_sha256": sha256_file(root / TRANSPORT_LOCK),
            "rows_equal": recomputed_transport["fixture"]["rows_equal"],
            "state_equal": recomputed_transport["fixture"]["state_equal"],
        },
        "analysis": {
            "terminal": recomputed_analysis["terminal"],
            "lock_sha256": sha256_file(root / ANALYSIS_LOCK),
            "family_count": len(recomputed_analysis["frozen_contract"]["family"]),
        },
    }


def _projection(profile: dict[str, Any]) -> dict[str, Any]:
    receipts = profile["sequence_receipts"]
    factorial = [
        item for item in receipts if item["cluster_kind"] == "FACTORIAL"
    ]
    guardrail = [
        item for item in receipts if item["cluster_kind"] == "GUARDRAIL"
    ]
    grouped_render_work = 0.0
    for motion in ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION"):
        group = [
            item for item in factorial if item["arm"].startswith(motion)
        ]
        if len(group) != 3:
            raise InvalidIndependentPreflight("W8_FACTORIAL_MOTION_GROUP")
        # QMS-R1 returns clean and low from one shared raycast; blur is derived
        # from that same clean frame.  The maximum independently measured
        # render time is a conservative bound for the grouped render.
        grouped_render_work += 80 * max(
            item["timing"]["render_seconds"] for item in group
        )
    grouped_render_work += 8 * sum(
        item["timing"]["render_seconds"] for item in guardrail
    )
    r3_work = sum(
        item["timing"]["r3_seconds"]
        * (80 if item["cluster_kind"] == "FACTORIAL" else 8)
        for item in receipts
    )
    validation_work = sum(
        item["timing"]["validation_and_hash_seconds"]
        * (80 if item["cluster_kind"] == "FACTORIAL" else 8)
        for item in receipts
    )
    projected_render = grouped_render_work / 8.0
    projected_r3 = r3_work / 8.0
    projected_validation = validation_work / 8.0
    core = projected_render + projected_r3 + projected_validation
    return {
        "render_seconds": projected_render,
        "r3_seconds": projected_r3,
        "validation_and_hash_seconds": projected_validation,
        "projected_core_seconds": core,
        "retry_reserve_seconds": core * 0.10,
        "total_seconds": core * 1.10,
        "wall_ceiling_seconds": WALL_CEILING_SECONDS,
        "projection_method": (
            "QMS_R1_CLUSTER_GROUPED_SHARED_RENDER_MAX_BOUND_PLUS_"
            "ALL_ARM_R3_AND_VALIDATION_DIVIDED_BY_W8"
        ),
        "required_formal_scheduler": (
            "GROUP_BY_SCENE_AND_MOTION_ONE_QMS_RENDER_PAIR_"
            "DERIVE_CLEAN_LOW_AND_BLUR"
        ),
    }


def validate_w8(
    root: Path, directory: Path, identity_path: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    profile_path = directory / "success.json"
    profile = load_json(profile_path)
    receipts = profile.get("sequence_receipts")
    expected_ids = [item["sequence_id"] for item in identity["identities"]]
    if (
        profile.get("task_id") != TASK_ID
        or profile.get("profile") != "W8"
        or profile.get("workers") != 8
        or profile.get("native_threads_per_worker") != 18
        or profile.get("terminal") != "PROFILE_COMPLETE / PREFLIGHT_ONLY"
        or profile.get("identity_lock_sha256") != sha256_file(identity_path)
        or profile.get("identity_set_sha256") != identity["identity_set_sha256"]
        or profile.get("sequence_count") != 8
        or profile.get("frame_count") != 4816
        or profile.get("pair_count") != 4808
        or profile.get("residual_worker_pids") != []
        or profile.get("formal_execution_authorized") is not False
        or not isinstance(receipts, list)
        or [item.get("sequence_id") for item in receipts] != expected_ids
    ):
        raise InvalidIndependentPreflight("W8_PROFILE")
    by_id = {item["sequence_id"]: item for item in identity["identities"]}
    for receipt in receipts:
        locked = by_id[receipt["sequence_id"]]
        if (
            receipt.get("numeric_seed_uint64") != locked["numeric_seed_uint64"]
            or receipt.get("scene_geometry_sha256")
            != locked["scene_geometry_sha256"]
            or receipt.get("frame_count") != 602
            or receipt.get("pair_count") != 601
            or receipt.get("qms_r1_operator_source_sha256")
            != EXPECTED_OPERATOR_SOURCE_SHA256
            or receipt.get("outcome_firewall")
            != {
                "response_values_emitted": False,
                "trigger_values_emitted": False,
                "scientific_interpretation": False,
            }
        ):
            raise InvalidIndependentPreflight("W8_SEQUENCE_RECEIPT")
        receipt_path = directory / "sequences" / receipt["sequence_id"] / "receipt.json"
        if not receipt_path.is_file():
            raise InvalidIndependentPreflight("W8_SEQUENCE_FILE")
    resource = profile.get("resource", {})
    if (
        resource.get("available_ram_at_launch_bytes", 0) < 8 * GIB
        or resource.get("minimum_available_ram_bytes", 0) < 4 * GIB
        or resource.get("sustained_paging") is not False
        or resource.get("swap_in_delta") != 0
        or resource.get("swap_out_delta") != 0
        or resource.get("heartbeat_max_interval_seconds", 31) > 30
    ):
        raise InvalidIndependentPreflight("W8_RESOURCE")
    progress = load_json(directory / "progress.json")
    telemetry = load_json(directory / "telemetry.json")
    if (
        progress.get("terminal_state") != "SUCCESS"
        or progress.get("completed_units") != 8
        or telemetry.get("outcome_fields_present") is not False
    ):
        raise InvalidIndependentPreflight("W8_PROGRESS_TELEMETRY")
    firewall = profile.get("formal_path_firewall", {})
    successor_path = root / firewall.get("successor_formal_path", "")
    if (
        firewall.get("predecessor_formal_unchanged") is not True
        or firewall.get("predecessor_formal_tree_sha256_before")
        != firewall.get("predecessor_formal_tree_sha256_after")
        or firewall.get("successor_formal_path_absent") is not True
        or successor_path.exists()
        or firewall.get("formal_sequences_run") != 0
    ):
        raise InvalidIndependentPreflight("FORMAL_PATH_FIREWALL")
    projection = _projection(profile)
    if not math.isfinite(projection["total_seconds"]) or (
        projection["total_seconds"] > WALL_CEILING_SECONDS
    ):
        raise InvalidIndependentPreflight("W8_PROJECTION")
    return {
        "terminal": "W8_GUARDED_HOST_QUALIFIED / PREFLIGHT_ONLY",
        "profile_sha256": sha256_file(profile_path),
        "measured_wall_seconds": profile["timing"]["wall_seconds"],
        "resource": resource,
        "projection": projection,
        "formal_path_firewall": firewall,
    }


def validate_all(
    root: Path,
    operator_path: Path,
    formal_path: Path,
    preflight_path: Path,
    w8_directory: Path,
) -> dict[str, Any]:
    operator_result = validate_operator(root, operator_path)
    formal, formal_overlaps = validate_formal_lock(
        root, formal_path, operator_result["operator_lock_sha256"]
    )
    preflight, preflight_overlaps = validate_preflight_lock(
        root,
        preflight_path,
        formal,
        formal_path,
        operator_result["operator_lock_sha256"],
    )
    scientific = validate_scientific_locks(root)
    w8 = validate_w8(root, w8_directory, preflight_path, preflight)
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_activation_preflight_independent_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "validated": True,
        "errors": [],
        "operator": operator_result,
        "successor_formal_identity_lock": {
            "sha256": sha256_file(formal_path),
            "identity_set_sha256": formal["identity_set_sha256"],
            "sequence_count": 496,
            "overlap_counts": formal_overlaps,
        },
        "preflight_identity_lock": {
            "sha256": sha256_file(preflight_path),
            "identity_set_sha256": preflight["identity_set_sha256"],
            "identity_count": 8,
            "overlap_counts": preflight_overlaps,
        },
        "scientific_locks": scientific,
        "guarded_host": w8,
        "formal_execution": {
            "successor_sequences_run": 0,
            "formal_r3_pair_core_calls": 0,
            "scientific_outcome_interpreted": False,
        },
        "formal_execution_authorized": False,
        "terminal": "ACTIVATION_PREFLIGHT_PASS / VALID / FORMAL_NOT_RUN",
    }


def activation_decision(
    root: Path,
    receipt_path: Path,
    receipt: dict[str, Any],
    operator_path: Path,
    formal_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    if (
        receipt.get("validated") is not True
        or receipt.get("terminal")
        != "ACTIVATION_PREFLIGHT_PASS / VALID / FORMAL_NOT_RUN"
    ):
        raise InvalidIndependentPreflight("ACTIVATION_PRECONDITION")
    validator_path = Path(__file__).resolve()
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_successor_activation_decision.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "decision_id": f"{TASK_ID}_ONE_SHOT_DECISION",
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": [
            {
                "role": role,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for role, path in (
                ("QMS_R1_OPERATOR_LOCK", operator_path),
                ("SUCCESSOR_FORMAL_IDENTITY_LOCK", formal_path),
                ("PREFLIGHT_IDENTITY_LOCK", preflight_path),
                ("INDEPENDENT_RECEIPT", receipt_path),
                ("INDEPENDENT_VALIDATOR", validator_path),
                ("R3_TRANSPORT_LOCK", root / TRANSPORT_LOCK),
                ("ANALYSIS_LOCK", root / ANALYSIS_LOCK),
            )
        ],
        "execution": {
            "class": "QMS_R1_SUCCESSOR_FORMAL_480_PLUS_16",
            "one_shot": True,
            "workers": 8,
            "scheduler_strategy": (
                "GROUP_BY_SCENE_AND_MOTION_ONE_QMS_RENDER_PAIR_"
                "DERIVE_CLEAN_LOW_AND_BLUR"
            ),
            "formal_sequences": 496,
            "formal_frames": 298592,
            "formal_pairs": 298096,
            "identity_set_sha256": receipt["successor_formal_identity_lock"][
                "identity_set_sha256"
            ],
            "operator_source_sha256": EXPECTED_OPERATOR_SOURCE_SHA256,
        },
        "preflight_only_execution_completed": True,
        "formal_sequences_run_at_decision": 0,
        "formal_execution_authorized": True,
        "authority_ceiling": {
            "one_shot_successor_formal_only": True,
            "identity_replacement": False,
            "operator_change": False,
            "r3_or_threshold_or_analysis_change": False,
            "sequence16": False,
            "android_or_realtime": False,
            "product_or_safety_claim": False,
        },
        "terminal": "QMS_R1_SUCCESSOR_FORMAL_EXECUTION_AUTHORIZED / ONE_SHOT",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--operator-lock", type=Path, required=True)
    parser.add_argument("--formal-identity-lock", type=Path, required=True)
    parser.add_argument("--preflight-identity-lock", type=Path, required=True)
    parser.add_argument("--w8-directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--activation-decision", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    operator_path = (root / args.operator_lock).resolve()
    formal_path = (root / args.formal_identity_lock).resolve()
    preflight_path = (root / args.preflight_identity_lock).resolve()
    w8_directory = (root / args.w8_directory).resolve()
    receipt_path = (root / args.receipt).resolve()
    decision_path = (root / args.activation_decision).resolve()
    receipt = validate_all(
        root, operator_path, formal_path, preflight_path, w8_directory
    )
    write_exclusive(receipt_path, receipt)
    decision = activation_decision(
        root,
        receipt_path,
        receipt,
        operator_path,
        formal_path,
        preflight_path,
    )
    write_exclusive(decision_path, decision)
    print(
        json.dumps(
            {
                "receipt_terminal": receipt["terminal"],
                "activation_terminal": decision["terminal"],
                "formal_sequences_run": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
