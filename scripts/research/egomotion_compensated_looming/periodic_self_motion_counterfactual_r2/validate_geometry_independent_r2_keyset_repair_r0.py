"""Independent keyset-repair full G01-G14 source-known geometry validator.

The validator imports only the already-independent R0 geometry math, never a
generator and never RCLE. It revalidates every gate on byte-identical R2
geometry, fixes only the consumed R0 evidence-key literal, and preserves all
three failed predecessor receipts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import cv2
import numpy as np
import scipy

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_geometry_independent as base,
)


IMPLEMENTATION_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_R2_KEYSET_REPAIR_R0"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r2_keyset_repair_r0"
)
R2_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r2"
)
R1_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r1"
)
R0_EVIDENCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2"
    / "p1_geometry_r0"
)
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_"
    "R2_KEYSET_REPAIR_R0_2026-07-29.json"
)
AMENDMENT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_KEYSET_REPAIR_R0_2026-07-29.json"
)
R0_LOCK_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R0_2026-07-28.json"
)
EXPECTED_R0_RECEIPT_SHA256 = (
    "72e0b8e042be9eb6208389eb8d83e9e9e4ad28e54ec82f7064b5387cc1abd279"
)
EXPECTED_R1_FAILED_RECEIPT_SHA256 = (
    "af00df05c115036ea31bb3d05addbebfcebad73122d2b354f7e52170c2277e9a"
)
EXPECTED_R1_LOCK_SHA256 = (
    "b49efb5ef2d267dbcb50a3ff85f1890b4026272d77188a914dca2e9a91cc624d"
)
EXPECTED_R1_AMENDMENT_SHA256 = (
    "521fd5fe523e9970c437c82e0dd5f3091a283e57de78e953db48b5d0cb0bfe48"
)
EXPECTED_R1_CONSUMED_VALIDATOR_SHA256 = (
    "5be754efdcd04e4fcaa3fafc64a6b39ce92a5e36594e4b3fd2e141a62c5b9d8b"
)
EXPECTED_R1_RECEIPT_RECORDED_VALIDATOR_SHA256 = (
    "fd80e5b2d12f30fe7ba02c37e8311b9af37f52148362fc3b53cc8580a0166539"
)
EXPECTED_R2_FAILED_RECEIPT_SHA256 = (
    "75899978919b67be260bbba1161d69ea09b42384f1730ec866a243f6d0f41a32"
)
EXPECTED_R2_LOCK_SHA256 = (
    "2a1921201c8215efbc0f05f5007908674e247325885d7640a592e731537d719f"
)
EXPECTED_R2_VALIDATOR_SHA256 = (
    "76e63a735635f95e767d801813e90500a5c12fc2a09ec1bc91ed31e3056d3cb6"
)
EXPECTED_R2_AMENDMENT_SHA256 = (
    "a81c922bca4bfa9b6211515bf16a4d7bae46748197082ec7d2f88ca2481e7283"
)
LOG_1P20 = 0.1823215567939546
EXPECTED_TARGET_ID = 9
EXPECTED_TARGET_POINT = [0.2, 0.1, 4.0]
EXPECTED_GUARD_OBJECT_COUNT = 12
EXPECTED_REPAIR_IDENTITY = (
    "R1_FIXED_CENTRAL_TARGET_LAYOUT_ALL_BLOCKS_ALL_GUARD_SEEDS"
)
REQUIRED_SOURCE_KEYS = {
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/generator_geometry.py",
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/validate_geometry_independent.py",
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/generator_geometry_r1.py",
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/validate_geometry_independent_r1.py",
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/generator_geometry_r2.py",
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/validate_geometry_independent_r2.py",
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "generator_geometry_r2_keyset_repair_r0.py",
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "validate_geometry_independent_r2_keyset_repair_r0.py",
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_VALIDATOR_REPAIR_R2_2026-07-28.json",
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R2_2026-07-28.json",
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_KEYSET_REPAIR_R0_2026-07-29.json",
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_SPEC_REPAIR_R1_2026-07-28.json",
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R1_2026-07-28.json",
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GEOMETRY_VALIDATION_R0_2026-07-28.json",
}
REQUIRED_EVIDENCE_KEYS = {
    "all_seed_geometry_manifest.jsonl",
    "analytic_fixture_ledger.json",
    "deterministic_replay_ledger.json",
    "guard_scene_replay_ledger.json",
    "package_manifest.json",
    "projective_sample_ledger.json",
    "runtime_manifest.json",
    "trajectory_manifest.json",
    "generator_r2_keyset_repair_r0_receipt.json",
}
REQUIRED_PRODUCER_ARTIFACT_KEYS = REQUIRED_EVIDENCE_KEYS - {
    "generator_r2_keyset_repair_r0_receipt.json"
}
EXPECTED_R0_EVIDENCE_KEYS = {
    "all_seed_geometry_manifest.jsonl",
    "analytic_fixture_ledger.json",
    "deterministic_replay_ledger.json",
    "producer_receipt.json",
    "projective_sample_ledger.json",
    "runtime_manifest.json",
    "trajectory_manifest.json",
}


def _has_exact_r0_evidence_keyset(receipt: dict[str, Any]) -> bool:
    evidence_hashes = receipt.get("evidence_sha256")
    return (
        isinstance(evidence_hashes, dict)
        and set(evidence_hashes) == EXPECTED_R0_EVIDENCE_KEYS
    )


def _validate_keyset_repair_lock(
    lock: dict[str, Any], evidence: Path, errors: list[str]
) -> None:
    if lock.get("implementation_id") != IMPLEMENTATION_ID:
        errors.append("KEYSET_REPAIR_LOCK_IMPLEMENTATION_ID")
    for field in (
        "formal_execution_authorized",
        "quality_calibration_authorized",
        "automatic_p2_authority",
    ):
        if lock.get(field) is not False:
            errors.append(f"KEYSET_REPAIR_LOCK_AUTHORITY:{field}")
    source_hashes = lock.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != REQUIRED_SOURCE_KEYS:
        errors.append("KEYSET_REPAIR_LOCK_SOURCE_HASH_KEYSET")
        source_hashes = {}
    for relative, expected in source_hashes.items():
        path = REPO_ROOT / relative
        if not path.is_file() or base.sha256_file(path) != expected:
            errors.append(f"KEYSET_REPAIR_LOCK_SOURCE_HASH:{relative}")
    evidence_hashes = lock.get("evidence_sha256")
    if (
        not isinstance(evidence_hashes, dict)
        or set(evidence_hashes) != REQUIRED_EVIDENCE_KEYS
    ):
        errors.append("KEYSET_REPAIR_LOCK_EVIDENCE_HASH_KEYSET")
        evidence_hashes = {}
    for name, expected in evidence_hashes.items():
        path = evidence / name
        if not path.is_file() or base.sha256_file(path) != expected:
            errors.append(f"KEYSET_REPAIR_LOCK_EVIDENCE_HASH:{name}")
    expected_environment = lock.get("environment", {})
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
    }
    for field, value in actual.items():
        if expected_environment.get(field) != value:
            errors.append(f"KEYSET_REPAIR_LOCK_ENVIRONMENT:{field}")
    predecessor = lock.get("immutable_r0", {})
    if predecessor.get("receipt_sha256") != EXPECTED_R0_RECEIPT_SHA256:
        errors.append("KEYSET_REPAIR_LOCK_R0_RECEIPT_IDENTITY")
    if predecessor.get("terminal") != "INTERVENTION_NOT_EVALUABLE / HOLD_P1":
        errors.append("KEYSET_REPAIR_LOCK_R0_TERMINAL")
    r1_predecessor = lock.get("immutable_r1_failure", {})
    if (
        r1_predecessor.get("receipt_sha256")
        != EXPECTED_R1_FAILED_RECEIPT_SHA256
    ):
        errors.append("KEYSET_REPAIR_LOCK_R1_RECEIPT_IDENTITY")
    if r1_predecessor.get("terminal") != "INTERVENTION_NOT_EVALUABLE / HOLD_P1":
        errors.append("KEYSET_REPAIR_LOCK_R1_TERMINAL")
    r2_predecessor = lock.get("immutable_r2_failure", {})
    if (
        r2_predecessor.get("receipt_sha256")
        != EXPECTED_R2_FAILED_RECEIPT_SHA256
    ):
        errors.append("KEYSET_REPAIR_LOCK_R2_RECEIPT_IDENTITY")
    if r2_predecessor.get("terminal") != "INTERVENTION_NOT_EVALUABLE / HOLD_P1":
        errors.append("KEYSET_REPAIR_LOCK_R2_TERMINAL")


def _validate_r0_immutability(
    current_records: list[dict[str, Any]],
    current_evidence: Path,
    errors: list[str],
) -> None:
    r0_receipt_path = R0_EVIDENCE / "independent_geometry_validation_receipt.json"
    if base.sha256_file(r0_receipt_path) != EXPECTED_R0_RECEIPT_SHA256:
        errors.append("R0_RECEIPT_HASH")
    else:
        receipt = base.load_json(r0_receipt_path)
        if (
            receipt.get("terminal") != "INTERVENTION_NOT_EVALUABLE"
            or receipt.get("state") != "HOLD_P1"
            or receipt.get("gate_pass_count") != 13
        ):
            errors.append("R0_TERMINAL_CONTENT")
        evidence_hashes = receipt.get("evidence_sha256")
        if not _has_exact_r0_evidence_keyset(receipt):
            errors.append("R0_RECEIPT_EVIDENCE_HASH_KEYSET")
        else:
            for name, expected in evidence_hashes.items():
                path = R0_EVIDENCE / name
                if not path.is_file() or base.sha256_file(path) != expected:
                    errors.append(f"R0_EVIDENCE_HASH_DRIFT:{name}")
    r1_failed_receipt_path = (
        R1_EVIDENCE / "independent_geometry_validation_receipt.json"
    )
    if (
        not r1_failed_receipt_path.is_file()
        or base.sha256_file(r1_failed_receipt_path)
        != EXPECTED_R1_FAILED_RECEIPT_SHA256
    ):
        errors.append("R1_FAILED_RECEIPT_HASH")
    else:
        r1_receipt = base.load_json(r1_failed_receipt_path)
        if (
            r1_receipt.get("terminal") != "INTERVENTION_NOT_EVALUABLE"
            or r1_receipt.get("state") != "HOLD_P1"
            or r1_receipt.get("gate_pass_count") != 13
            or r1_receipt.get("failed_gates")
            != ["G13_MONOTONIC_APPROACH_TRUTH"]
            or r1_receipt.get("implementation_lock_sha256")
            != EXPECTED_R1_LOCK_SHA256
            or r1_receipt.get("amendment_sha256")
            != EXPECTED_R1_AMENDMENT_SHA256
            or r1_receipt.get("validator_source_sha256")
            != EXPECTED_R1_RECEIPT_RECORDED_VALIDATOR_SHA256
        ):
            errors.append("R1_FAILED_RECEIPT_CONTENT")
    r2_failed_receipt_path = (
        R2_EVIDENCE / "independent_geometry_validation_receipt.json"
    )
    if (
        not r2_failed_receipt_path.is_file()
        or base.sha256_file(r2_failed_receipt_path)
        != EXPECTED_R2_FAILED_RECEIPT_SHA256
    ):
        errors.append("R2_FAILED_RECEIPT_HASH")
    else:
        r2_receipt = base.load_json(r2_failed_receipt_path)
        if (
            r2_receipt.get("status") != "INVALID"
            or r2_receipt.get("terminal") != "INTERVENTION_NOT_EVALUABLE"
            or r2_receipt.get("state") != "HOLD_P1"
            or r2_receipt.get("gate_pass_count") != 14
            or r2_receipt.get("failed_gates") != []
            or r2_receipt.get("errors")
            != ["R0_RECEIPT_EVIDENCE_HASH_KEYSET"]
            or r2_receipt.get("implementation_lock_sha256")
            != EXPECTED_R2_LOCK_SHA256
            or r2_receipt.get("amendment_sha256")
            != EXPECTED_R2_AMENDMENT_SHA256
            or r2_receipt.get("validator_source_sha256")
            != EXPECTED_R2_VALIDATOR_SHA256
        ):
            errors.append("R2_FAILED_RECEIPT_CONTENT")
    r1_lock_path = (
        REPO_ROOT
        / "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R1_2026-07-28.json"
    )
    r1_amendment_path = (
        REPO_ROOT
        / "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "GEOMETRY_SPEC_REPAIR_R1_2026-07-28.json"
    )
    r1_validator_path = (
        REPO_ROOT
        / "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "validate_geometry_independent_r1.py"
    )
    for label, path, expected in (
        ("LOCK", r1_lock_path, EXPECTED_R1_LOCK_SHA256),
        ("AMENDMENT", r1_amendment_path, EXPECTED_R1_AMENDMENT_SHA256),
        (
            "CONSUMED_VALIDATOR",
            r1_validator_path,
            EXPECTED_R1_CONSUMED_VALIDATOR_SHA256,
        ),
    ):
        if not path.is_file() or base.sha256_file(path) != expected:
            errors.append(f"R1_HISTORICAL_{label}_HASH")
    r2_lock_path = (
        REPO_ROOT
        / "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R2_2026-07-28.json"
    )
    r2_amendment_path = (
        REPO_ROOT
        / "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "GEOMETRY_VALIDATOR_REPAIR_R2_2026-07-28.json"
    )
    r2_validator_path = (
        REPO_ROOT
        / "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "validate_geometry_independent_r2.py"
    )
    for label, path, expected in (
        ("LOCK", r2_lock_path, EXPECTED_R2_LOCK_SHA256),
        ("AMENDMENT", r2_amendment_path, EXPECTED_R2_AMENDMENT_SHA256),
        ("VALIDATOR", r2_validator_path, EXPECTED_R2_VALIDATOR_SHA256),
    ):
        if not path.is_file() or base.sha256_file(path) != expected:
            errors.append(f"R2_HISTORICAL_{label}_HASH")
    r0_records = base.load_jsonl(R0_EVIDENCE / "all_seed_geometry_manifest.jsonl")
    r1_records = base.load_jsonl(R1_EVIDENCE / "all_seed_geometry_manifest.jsonl")
    r0_lines = (
        R0_EVIDENCE / "all_seed_geometry_manifest.jsonl"
    ).read_bytes().splitlines(keepends=True)
    current_lines = (
        current_evidence / "all_seed_geometry_manifest.jsonl"
    ).read_bytes().splitlines(keepends=True)
    r1_lines = (
        R1_EVIDENCE / "all_seed_geometry_manifest.jsonl"
    ).read_bytes().splitlines(keepends=True)
    r2_lines = (
        R2_EVIDENCE / "all_seed_geometry_manifest.jsonl"
    ).read_bytes().splitlines(keepends=True)
    if (
        len(r0_lines) != 88
        or len(r1_lines) != 88
        or len(r2_lines) != 88
        or len(current_lines) != 88
    ):
        errors.append("KEYSET_REPAIR_MANIFEST_LINE_COUNT")
    else:
        if r0_lines[:80] != current_lines[:80]:
            errors.append("KEYSET_REPAIR_MAIN_RECORD_BYTES_DRIFT_FROM_R0")
        if r1_lines != current_lines:
            errors.append("KEYSET_REPAIR_RECORD_BYTES_DRIFT_FROM_R1")
        if r2_lines != current_lines:
            errors.append("KEYSET_REPAIR_RECORD_BYTES_DRIFT_FROM_R2")
    r0_main = {
        item["cluster_id"]: item
        for item in r0_records
        if item.get("record_type") == "main_cluster"
    }
    r2_main = {
        item["cluster_id"]: item
        for item in current_records
        if item.get("record_type") == "main_cluster"
    }
    if set(r0_main) != set(r2_main) or len(r2_main) != 80:
        errors.append("R2_MAIN_IDENTITY_SET")
    else:
        for cluster_id in sorted(r0_main):
            if base.canonical_bytes(r0_main[cluster_id]) != base.canonical_bytes(
                r2_main[cluster_id]
            ):
                errors.append(f"R2_MAIN_RECORD_DRIFT:{cluster_id}")
    r0_guard = {
        item["cluster_id"]: item
        for item in r0_records
        if item.get("record_type") == "guardrail_cluster"
    }
    r2_guard = {
        item["cluster_id"]: item
        for item in current_records
        if item.get("record_type") == "guardrail_cluster"
    }
    if set(r0_guard) != set(r2_guard) or len(r2_guard) != 8:
        errors.append("R2_GUARD_IDENTITY_SET")
        return
    for cluster_id in sorted(r0_guard):
        before = r0_guard[cluster_id]
        after = r2_guard[cluster_id]
        if int(before["numeric_seed_uint64"]) != int(after["numeric_seed_uint64"]):
            errors.append(f"R2_GUARD_SEED_DRIFT:{cluster_id}")
        before_arms = {item["arm_id"]: item for item in before["arms"]}
        after_arms = {item["arm_id"]: item for item in after["arms"]}
        if set(before_arms) != set(after_arms):
            errors.append(f"R2_GUARD_ARM_SET:{cluster_id}")
            continue
        for arm_id in sorted(before_arms):
            for field in ("trajectory_sha256", "trajectory"):
                if base.canonical_bytes(before_arms[arm_id][field]) != base.canonical_bytes(
                    after_arms[arm_id][field]
                ):
                    errors.append(
                        f"R2_GUARD_TRAJECTORY_DRIFT:{cluster_id}:{arm_id}:{field}"
                    )


def _unit_hash(*parts: object) -> float:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return integer / float(2**64 - 1)


def _expected_bounds(
    u0: float, u1: float, v0: float, v1: float, z: float
) -> list[float]:
    return [
        (u0 - base.K[0, 2]) / base.K[0, 0] * z,
        (u1 - base.K[0, 2]) / base.K[0, 0] * z,
        (v0 - base.K[1, 2]) / base.K[1, 1] * z,
        (v1 - base.K[1, 2]) / base.K[1, 1] * z,
    ]


def _validate_guard_scene_contract(
    guards: list[dict[str, Any]], errors: list[str]
) -> None:
    expected_camera = {
        "projection": "pinhole",
        "width_px": base.WIDTH,
        "height_px": base.HEIGHT,
        "intrinsic": base.K.tolist(),
        "near_clip_m": 0.5,
        "far_clip_m": 25.0,
        "distortion": "none",
        "camera_axes": "+x right, +y down, +z optical forward",
    }
    near_layout = [
        (u0, u1, v0, v1)
        for u0, u1 in ((0.0, 80.0), (280.0, 360.0))
        for v0, v1 in (
            (0.0, 213.0),
            (213.0, 426.0),
            (426.0, 640.0),
        )
    ]
    target_edges = np.linspace(220.0, 450.0, 6).tolist()
    for record in guards:
        cluster_id = str(record.get("cluster_id"))
        block = str(record.get("block"))
        ordinal = int(record.get("ordinal", -1))
        expected_seed = base.derive_seed("GUARD", block, ordinal)
        if int(record.get("numeric_seed_uint64", -1)) != expected_seed:
            errors.append(f"R1_GUARD_SEED_DERIVATION:{cluster_id}")
        scene = record.get("scene", {})
        if scene.get("schema") != (
            "rcle.periodic_self_motion_counterfactual."
            "p1_geometry_manifest.v2"
        ):
            errors.append(f"R1_GUARD_SCHEMA:{cluster_id}")
        if scene.get("namespace") != "GUARD":
            errors.append(f"R1_GUARD_NAMESPACE:{cluster_id}")
        if scene.get("block") != block or int(scene.get("ordinal", -1)) != ordinal:
            errors.append(f"R1_GUARD_RECORD_BINDING:{cluster_id}")
        if int(scene.get("numeric_seed_uint64", -1)) != expected_seed:
            errors.append(f"R1_GUARD_SCENE_SEED:{cluster_id}")
        if scene.get("camera") != expected_camera:
            errors.append(f"R1_GUARD_CAMERA:{cluster_id}")
        if scene.get("repair_identity") != EXPECTED_REPAIR_IDENTITY:
            errors.append(f"R1_GUARD_REPAIR_IDENTITY:{cluster_id}")
        if scene.get("designated_target") != {
            "object_id": EXPECTED_TARGET_ID,
            "world_point_m": EXPECTED_TARGET_POINT,
            "role": "PERSISTENT_RENDERED_MIDDLE_DEPTH_GUARDRAIL_TARGET",
        }:
            errors.append(f"R1_GUARD_TARGET_IDENTITY:{cluster_id}")
        objects = scene.get("world", {}).get("objects", [])
        if (
            len(objects) != EXPECTED_GUARD_OBJECT_COUNT
            or [item.get("object_id") for item in objects] != list(range(1, 13))
        ):
            errors.append(f"R1_GUARD_OBJECT_SET:{cluster_id}")
            continue
        for index, (u0, u1, v0, v1) in enumerate(near_layout, start=1):
            obj = objects[index - 1]
            expected_z = 1.5 + 0.25 * _unit_hash(
                expected_seed, index, "depth"
            )
            expected_bounds = _expected_bounds(u0, u1, v0, v1, expected_z)
            if abs(float(obj.get("plane_z_m", -1.0)) - expected_z) > 1e-12:
                errors.append(f"R1_GUARD_NEAR_DEPTH:{cluster_id}:{index}")
            if not np.allclose(
                np.asarray(obj.get("bounds_xy_m", []), dtype=np.float64),
                expected_bounds,
                atol=1e-12,
                rtol=0.0,
            ):
                errors.append(f"R1_GUARD_NEAR_BOUNDS:{cluster_id}:{index}")
        for offset, (v0, v1) in enumerate(
            zip(target_edges[:-1], target_edges[1:]), start=7
        ):
            obj = objects[offset - 1]
            expected_bounds = _expected_bounds(80.0, 280.0, v0, v1, 4.0)
            if float(obj.get("plane_z_m", -1.0)) != 4.0:
                errors.append(f"R1_GUARD_TARGET_DEPTH:{cluster_id}:{offset}")
            if not np.allclose(
                np.asarray(obj.get("bounds_xy_m", []), dtype=np.float64),
                expected_bounds,
                atol=1e-12,
                rtol=0.0,
            ):
                errors.append(f"R1_GUARD_TARGET_BOUNDS:{cluster_id}:{offset}")
        far = objects[11]
        if (
            float(far.get("plane_z_m", -1.0)) != 18.0
            or far.get("bounds_xy_m") != [-12.0, 12.0, -16.0, 16.0]
        ):
            errors.append(f"R1_GUARD_FAR_LAYOUT:{cluster_id}")
        for obj in objects:
            x0, x1, y0, y1 = obj.get("bounds_xy_m", [None] * 4)
            z = obj.get("plane_z_m")
            expected_vertices = [
                [x0, y0, z],
                [x1, y0, z],
                [x1, y1, z],
                [x0, y1, z],
            ]
            if (
                obj.get("primitive") != "rectangle_mesh_2tri"
                or obj.get("triangles") != [[0, 1, 2], [0, 2, 3]]
                or obj.get("vertices_world_m") != expected_vertices
            ):
                errors.append(
                    f"R1_GUARD_EXPLICIT_MESH:{cluster_id}:{obj.get('object_id')}"
                )
        scene_copy = copy.deepcopy(scene)
        reported_scene_hash = scene_copy.pop("scene_geometry_sha256", None)
        recomputed_scene_hash = base.sha256_bytes(
            base.canonical_bytes(scene_copy)
        )
        if reported_scene_hash != recomputed_scene_hash:
            errors.append(f"R1_GUARD_SCENE_HASH:{cluster_id}")
        for arm in record.get("arms", []):
            if arm.get("scene_geometry_sha256") != reported_scene_hash:
                errors.append(
                    f"R1_GUARD_ARM_SCENE_HASH:{cluster_id}:{arm.get('arm_id')}"
                )


def gate_g01_g02_r2(
    main: list[dict[str, Any]], guards: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    u, v = np.meshgrid(
        np.arange(base.WIDTH, dtype=np.float64),
        np.arange(base.HEIGHT, dtype=np.float64),
    )
    uv = np.column_stack((u.reshape(-1), v.reshape(-1)))
    g01_failures = []
    g02_failures = []
    hash_mismatches = []
    for record in [*main, *guards]:
        depth_flat, object_flat, _ = base.raycast(
            record["scene"], np.eye(3), np.zeros(3), uv
        )
        depth = depth_flat.reshape(base.HEIGHT, base.WIDTH)
        object_id = object_flat.reshape(base.HEIGHT, base.WIDTH)
        valid = np.isfinite(depth)
        band_fractions = {
            "near": float(np.mean(valid & (depth >= 0.75) & (depth < 2.0))),
            "middle": float(
                np.mean(valid & (depth >= 2.0) & (depth < 5.0))
            ),
            "far": float(np.mean(valid & (depth >= 5.0) & (depth <= 20.0))),
        }
        metrics = record["reference_metrics"]
        actual_hashes = {
            "reference_depth_sha256": base.sha256_bytes(
                depth.astype("<f8").tobytes()
            ),
            "reference_object_id_sha256": base.sha256_bytes(
                object_id.astype("<i4").tobytes()
            ),
            "reference_visibility_sha256": base.sha256_bytes(
                valid.astype(np.uint8).tobytes()
            ),
        }
        for field, actual in actual_hashes.items():
            if metrics.get(field) != actual:
                hash_mismatches.append(f"{record['cluster_id']}:{field}")
        if (
            len(record["scene"]["world"]["objects"]) < 12
            or float(np.mean(valid)) < 0.90
            or min(band_fractions.values()) < 0.10
        ):
            g01_failures.append(record["cluster_id"])
        if record["record_type"] == "main_cluster":
            diverse = 0
            for row in range(3):
                for column in range(3):
                    tile = depth[
                        row * base.HEIGHT // 3 : (row + 1) * base.HEIGHT // 3,
                        column * base.WIDTH // 3 : (column + 1) * base.WIDTH // 3,
                    ]
                    tile_valid = np.isfinite(tile)
                    denominator = int(np.count_nonzero(tile_valid))
                    fractions = (
                        float(
                            np.count_nonzero(tile_valid & (tile < 2.0))
                            / denominator
                        ),
                        float(
                            np.count_nonzero(
                                tile_valid & (tile >= 2.0) & (tile < 5.0)
                            )
                            / denominator
                        ),
                        float(
                            np.count_nonzero(tile_valid & (tile >= 5.0))
                            / denominator
                        ),
                    )
                    diverse += int(sum(item >= 0.05 for item in fractions) >= 2)
            if diverse < 7:
                g02_failures.append(record["cluster_id"])
    g01_pass = not g01_failures and not hash_mismatches
    g02_pass = not g02_failures and not hash_mismatches
    return (
        {
            "id": "G01_FINITE_MULTI_DEPTH",
            "status": "PASS" if g01_pass else "FAIL",
            "evaluated_scene_count": len(main) + len(guards),
            "failures": g01_failures,
            "reference_hash_mismatches": hash_mismatches,
        },
        {
            "id": "G02_GRID_DEPTH_DIVERSITY",
            "status": "PASS" if g02_pass else "FAIL",
            "evaluated_scene_count": len(main),
            "failures": g02_failures,
            "reference_hash_mismatches": hash_mismatches,
        },
    )


def _manual_pinhole_project(
    world: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera = (rotation.T @ (world - translation).T).T
    depth = camera[:, 2]
    uv = np.column_stack(
        (
            base.K[0, 0] * camera[:, 0] / depth + base.K[0, 2],
            base.K[1, 1] * camera[:, 1] / depth + base.K[1, 2],
        )
    )
    return uv, depth


def gate_g03_r2(
    fixtures: dict[str, Any],
    samples: dict[str, Any],
    trajectories: dict[str, Any],
    main_by_key: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    errors_px: list[float] = []
    fixture_count = 0
    fixture_failures = []
    for fixture in fixtures.get("fixtures", []):
        if "points_camera0_m" not in fixture:
            continue
        points0 = np.asarray(fixture["points_camera0_m"], dtype=np.float64)
        pose0, pose1 = fixture["camera_poses"]
        rotation0 = base.Rotation.from_rotvec(
            pose0["rotation_rotvec_rad"]
        ).as_matrix()
        rotation1 = base.Rotation.from_rotvec(
            pose1["rotation_rotvec_rad"]
        ).as_matrix()
        translation0 = np.asarray(pose0["translation_m"], dtype=np.float64)
        translation1 = np.asarray(pose1["translation_m"], dtype=np.float64)
        world = (rotation0 @ points0.T).T + translation0
        manual_uv1, manual_depth1 = _manual_pinhole_project(
            world, rotation1, translation1
        )
        base_uv1, base_depth1 = base.project(world, rotation1, translation1)
        errors_px.extend(np.linalg.norm(manual_uv1 - base_uv1, axis=1).tolist())
        if (
            not np.all(np.isfinite(manual_uv1))
            or not np.all(manual_depth1 > 0.0)
            or not np.allclose(
                manual_depth1, base_depth1, atol=1e-12, rtol=0.0
            )
        ):
            fixture_failures.append(str(fixture.get("id")))
        fixture_count += len(points0)
    block_counts: dict[str, int] = {}
    visibility_errors = 0
    sample_identity_errors = 0
    blocks = samples.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != set(base.BLOCKS):
        blocks = {}
        sample_identity_errors += 1
    for block in base.BLOCKS:
        _, translations, rotations = base.pose_arrays(trajectories[block])
        block_samples = blocks.get(block, {}).get("samples", [])
        block_counts[block] = len(block_samples)
        for sample in block_samples:
            try:
                ordinal = int(sample["scene_ordinal"])
                frame = int(sample["frame_index"])
                record = main_by_key[(block, ordinal)]
                if not (0 <= ordinal < 20 and 0 <= frame < base.PAIR_COUNT):
                    raise ValueError("sample index")
                world = np.asarray(
                    [sample["world_point_m"]], dtype=np.float64
                )
                reported = np.asarray(
                    sample["renderer_uv_next"], dtype=np.float64
                )
                object_id = np.asarray(
                    [int(sample["object_id"])], dtype=np.int32
                )
            except (KeyError, TypeError, ValueError):
                sample_identity_errors += 1
                continue
            uv, depth = _manual_pinhole_project(
                world, rotations[frame + 1], translations[frame + 1]
            )
            errors_px.append(float(np.linalg.norm(uv[0] - reported)))
            visible = _visibility_by_explicit_occlusion(
                record["scene"],
                rotations[frame + 1],
                translations[frame + 1],
                uv,
                world,
                object_id,
            )
            if (
                sample.get("visible_next") is not True
                or not bool(visible[0])
                or not np.isfinite(depth[0])
            ):
                visibility_errors += 1
    values = np.asarray(errors_px, dtype=np.float64)
    rms = float(np.sqrt(np.mean(values**2))) if len(values) else math.inf
    p99 = float(np.quantile(values, 0.99)) if len(values) else math.inf
    passed = bool(
        fixture_count == 48
        and not fixture_failures
        and all(value == 10000 for value in block_counts.values())
        and rms <= 0.05
        and p99 <= 0.25
        and visibility_errors == 0
        and sample_identity_errors == 0
    )
    return {
        "id": "G03_PROJECTIVE_PARITY",
        "status": "PASS" if passed else "FAIL",
        "fixture_sample_count": fixture_count,
        "fixture_failures": fixture_failures,
        "block_sample_counts": block_counts,
        "rms_px": rms,
        "p99_px": p99,
        "visibility_error_count": visibility_errors,
        "sample_identity_error_count": sample_identity_errors,
        "analytic_cross_check": "closed_form_pinhole",
    }


def _visibility_by_explicit_occlusion(
    scene: dict[str, Any],
    rotation: np.ndarray,
    translation: np.ndarray,
    uv: np.ndarray,
    world: np.ndarray,
    source_object: np.ndarray,
) -> np.ndarray:
    projected_uv, projected_depth = base.project(world, rotation, translation)
    if not np.allclose(projected_uv, uv, atol=1e-9, rtol=0.0, equal_nan=True):
        raise ValueError("OCCLUSION_PROJECTED_UV_MISMATCH")
    pixels = np.column_stack((uv, np.ones(len(uv), dtype=np.float64)))
    directions = (rotation @ (base.K_INV @ pixels.T)).T
    closest_depth = np.full(len(uv), np.inf, dtype=np.float64)
    closest_object = np.zeros(len(uv), dtype=np.int32)
    for obj in scene["world"]["objects"]:
        z = float(obj["plane_z_m"])
        scale = (z - translation[2]) / directions[:, 2]
        intersection = translation + scale[:, None] * directions
        camera = (rotation.T @ (intersection - translation).T).T
        depth = camera[:, 2]
        x0, x1, y0, y1 = [float(item) for item in obj["bounds_xy_m"]]
        eligible = (
            np.isfinite(scale)
            & (scale > 0.0)
            & (intersection[:, 0] >= x0)
            & (intersection[:, 0] <= x1)
            & (intersection[:, 1] >= y0)
            & (intersection[:, 1] <= y1)
            & (depth >= 0.5)
            & (depth <= 25.0)
            & (depth < closest_depth)
        )
        closest_depth[eligible] = depth[eligible]
        closest_object[eligible] = int(obj["object_id"])
    inside = (
        (uv[:, 0] >= 0.0)
        & (uv[:, 0] < base.WIDTH)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < base.HEIGHT)
        & np.isfinite(projected_depth)
    )
    return (
        inside
        & (closest_object == source_object)
        & (np.abs(closest_depth - projected_depth) <= 1e-7)
    )


def _analytic_occlusion_disocclusion_count() -> int:
    def bounds(
        u0: float, u1: float, v0: float, v1: float, z: float
    ) -> list[float]:
        return _expected_bounds(u0, u1, v0, v1, z)

    scene = {
        "world": {
            "objects": [
                {
                    "object_id": 1,
                    "plane_z_m": 1.5,
                    "bounds_xy_m": bounds(120.0, 240.0, 180.0, 460.0, 1.5),
                },
                {
                    "object_id": 2,
                    "plane_z_m": 6.0,
                    "bounds_xy_m": bounds(0.0, 359.0, 0.0, 639.0, 6.0),
                },
            ]
        }
    }
    grid_u = np.linspace(100.0, 260.0, 81)
    grid_v = np.linspace(190.0, 450.0, 81)
    uv0 = np.asarray([(u, v) for v in grid_v for u in grid_u])
    rays = np.column_stack((uv0, np.ones(len(uv0)))) @ base.K_INV.T
    far_world = rays * (6.0 / rays[:, 2:3])
    object_id = np.full(len(far_world), 2, dtype=np.int32)
    rendered0_depth, rendered0_object, _ = base.raycast(
        scene, np.eye(3), np.zeros(3), uv0
    )
    uv1, _ = base.project(
        far_world, np.eye(3), np.asarray([0.10, 0.0, 0.0])
    )
    rendered1_depth, rendered1_object, _ = base.raycast(
        scene, np.eye(3), np.asarray([0.10, 0.0, 0.0]), uv1
    )
    occluded0 = rendered0_object != object_id
    visible1 = _visibility_by_explicit_occlusion(
        scene,
        np.eye(3),
        np.asarray([0.10, 0.0, 0.0]),
        uv1,
        far_world,
        object_id,
    )
    visible1 &= (
        (rendered1_object == object_id)
        & np.isfinite(rendered1_depth)
    )
    return int(np.count_nonzero(occluded0 & visible1))


def _raycast_batched(
    scene: dict[str, Any],
    rotations: np.ndarray,
    translations: np.ndarray,
    uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.concatenate(
        (uv, np.ones((*uv.shape[:2], 1), dtype=np.float64)), axis=2
    )
    camera_rays = np.einsum("ij,nsj->nsi", base.K_INV, pixels)
    directions = np.einsum("nij,nsj->nsi", rotations, camera_rays)
    depth_out = np.full(uv.shape[:2], np.inf, dtype=np.float64)
    object_out = np.zeros(uv.shape[:2], dtype=np.int32)
    world_out = np.full((*uv.shape[:2], 3), np.nan, dtype=np.float64)
    for obj in scene["world"]["objects"]:
        z = float(obj["plane_z_m"])
        scale = (z - translations[:, None, 2]) / directions[:, :, 2]
        intersection = translations[:, None, :] + scale[:, :, None] * directions
        camera = np.einsum(
            "nji,nsj->nsi",
            rotations,
            intersection - translations[:, None, :],
        )
        depth = camera[:, :, 2]
        x0, x1, y0, y1 = [float(item) for item in obj["bounds_xy_m"]]
        eligible = (
            np.isfinite(scale)
            & (scale > 0.0)
            & (intersection[:, :, 0] >= x0)
            & (intersection[:, :, 0] <= x1)
            & (intersection[:, :, 1] >= y0)
            & (intersection[:, :, 1] <= y1)
            & (depth >= 0.5)
            & (depth <= 25.0)
            & (depth < depth_out)
        )
        depth_out[eligible] = depth[eligible]
        object_out[eligible] = int(obj["object_id"])
        world_out[eligible] = intersection[eligible]
    return depth_out, object_out, world_out


def _project_batched(
    world: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera = np.einsum(
        "nji,nsj->nsi",
        rotations,
        world - translations[:, None, :],
    )
    depth = camera[:, :, 2]
    uv = np.stack(
        (
            base.K[0, 0] * camera[:, :, 0] / depth + base.K[0, 2],
            base.K[1, 1] * camera[:, :, 1] / depth + base.K[1, 2],
        ),
        axis=2,
    )
    return uv, depth


def _explicit_visibility_batched(
    scene: dict[str, Any],
    rotations: np.ndarray,
    translations: np.ndarray,
    uv: np.ndarray,
    world: np.ndarray,
    source_object: np.ndarray,
) -> np.ndarray:
    projected_uv, projected_depth = _project_batched(
        world, rotations, translations
    )
    if not np.allclose(
        projected_uv, uv, atol=1e-9, rtol=0.0, equal_nan=True
    ):
        raise ValueError("BATCH_OCCLUSION_PROJECTED_UV_MISMATCH")
    pixels = np.concatenate(
        (uv, np.ones((*uv.shape[:2], 1), dtype=np.float64)), axis=2
    )
    camera_rays = np.einsum("ij,nsj->nsi", base.K_INV, pixels)
    directions = np.einsum("nij,nsj->nsi", rotations, camera_rays)
    closest_depth = np.full(uv.shape[:2], np.inf, dtype=np.float64)
    closest_object = np.zeros(uv.shape[:2], dtype=np.int32)
    for obj in scene["world"]["objects"]:
        z = float(obj["plane_z_m"])
        scale = (z - translations[:, None, 2]) / directions[:, :, 2]
        intersection = translations[:, None, :] + scale[:, :, None] * directions
        camera = np.einsum(
            "nji,nsj->nsi",
            rotations,
            intersection - translations[:, None, :],
        )
        depth = camera[:, :, 2]
        x0, x1, y0, y1 = [float(item) for item in obj["bounds_xy_m"]]
        eligible = (
            np.isfinite(scale)
            & (scale > 0.0)
            & (intersection[:, :, 0] >= x0)
            & (intersection[:, :, 0] <= x1)
            & (intersection[:, :, 1] >= y0)
            & (intersection[:, :, 1] <= y1)
            & (depth >= 0.5)
            & (depth <= 25.0)
            & (depth < closest_depth)
        )
        closest_depth[eligible] = depth[eligible]
        closest_object[eligible] = int(obj["object_id"])
    inside = (
        (uv[:, :, 0] >= 0.0)
        & (uv[:, :, 0] < base.WIDTH)
        & (uv[:, :, 1] >= 0.0)
        & (uv[:, :, 1] < base.HEIGHT)
        & np.isfinite(projected_depth)
    )
    return (
        inside
        & (closest_object == source_object)
        & (np.abs(closest_depth - projected_depth) <= 1e-7)
    )


def gate_g08_r2(
    main: list[dict[str, Any]], trajectories: dict[str, Any]
) -> dict[str, Any]:
    grid = np.asarray(
        [
            (
                (column + 0.5) * base.WIDTH / 3.0,
                (row + 0.5) * base.HEIGHT / 3.0,
            )
            for row in range(3)
            for column in range(3)
        ],
        dtype=np.float64,
    )
    mismatch = 0
    invalid_visible = 0
    sequence_count = 0
    sample_count = 0
    for record in main:
        _, translations, rotations = base.pose_arrays(
            trajectories[record["block"]]
        )
        scene = record["scene"]
        for motion in ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION"):
            sequence_count += 1
            if motion == "STATIC_CAMERA":
                rotation0 = np.repeat(
                    np.eye(3)[None, :, :], base.PAIR_COUNT, axis=0
                )
                translation0 = np.zeros((base.PAIR_COUNT, 3))
                rotation1 = rotation0
                translation1 = translation0
            else:
                rotation0 = rotations[:-1]
                translation0 = translations[:-1]
                rotation1 = rotations[1:]
                translation1 = translations[1:]
            uv0 = np.repeat(grid[None, :, :], base.PAIR_COUNT, axis=0)
            _, source_object, world = _raycast_batched(
                scene, rotation0, translation0, uv0
            )
            uv1, expected_depth = _project_batched(
                world, rotation1, translation1
            )
            rendered_depth, rendered_object, _ = _raycast_batched(
                scene, rotation1, translation1, uv1
            )
            renderer_visible = (
                (rendered_object == source_object)
                & np.isfinite(rendered_depth)
                & (np.abs(rendered_depth - expected_depth) <= 1e-7)
            )
            independent_visible = _explicit_visibility_batched(
                scene,
                rotation1,
                translation1,
                uv1,
                world,
                source_object,
            )
            mismatch += int(
                np.count_nonzero(renderer_visible != independent_visible)
            )
            invalid_visible += int(
                np.count_nonzero(renderer_visible & ~independent_visible)
            )
            sample_count += int(np.prod(renderer_visible.shape))
    mismatch_fraction = mismatch / sample_count if sample_count else 1.0
    disocclusion_count = _analytic_occlusion_disocclusion_count()
    passed = bool(
        sequence_count == 160
        and mismatch_fraction <= 0.001
        and invalid_visible == 0
        and disocclusion_count > 0
    )
    return {
        "id": "G08_OCCLUSION_VISIBILITY",
        "status": "PASS" if passed else "FAIL",
        "evaluated_motion_sequence_count": sequence_count,
        "sample_count": sample_count,
        "visibility_mismatch_fraction": mismatch_fraction,
        "invalid_visible_correspondence_count": invalid_visible,
        "analytic_disocclusion_point_count": disocclusion_count,
    }


ARM_KEYS = {
    "arm_id",
    "cluster_id",
    "depth_sha256",
    "geometry_identity_sha256",
    "intrinsic_sha256",
    "motion",
    "object_id_sha256",
    "pose_sha256",
    "quality",
    "quality_operator_status",
    "scene_geometry_sha256",
    "timestamp_sha256",
    "trajectory_sha256",
    "visibility_sha256",
}


def gate_g11_r2(main: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    shared_across_motion = {
        "cluster_id",
        "intrinsic_sha256",
        "quality_operator_status",
        "scene_geometry_sha256",
        "timestamp_sha256",
    }
    for record in main:
        cluster_id = str(record.get("cluster_id"))
        arms = record.get("arms")
        if not isinstance(arms, list):
            failures.append(f"{cluster_id}:ARMS_NOT_LIST")
            continue
        ids = [arm.get("arm_id") for arm in arms if isinstance(arm, dict)]
        if (
            len(arms) != 6
            or len(ids) != 6
            or tuple(ids) != base.ARMS
            or len(set(ids)) != 6
        ):
            failures.append(f"{cluster_id}:ARM_CARTESIAN_IDENTITY")
            continue
        for arm in arms:
            if set(arm) != ARM_KEYS:
                failures.append(f"{cluster_id}:{arm.get('arm_id')}:ARM_SCHEMA")
                continue
            expected_arm_id = f"{arm['motion']}__{arm['quality']}"
            if (
                arm["arm_id"] != expected_arm_id
                or arm["cluster_id"] != cluster_id
                or arm["motion"]
                not in ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
                or arm["quality"] not in ("CLEAN", "BLUR", "LOW_TEXTURE")
                or arm["quality_operator_status"]
                != "NOT_CALIBRATED_P1_IDENTITY_ONLY"
            ):
                failures.append(f"{cluster_id}:{arm['arm_id']}:ARM_CONTENT")
        for motion in ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION"):
            group = [arm for arm in arms if arm.get("motion") == motion]
            if len(group) != 3:
                failures.append(f"{cluster_id}:{motion}:QUALITY_GROUP")
                continue
            normalized = [
                {key: value for key, value in arm.items() if key not in {"arm_id", "quality"}}
                for arm in group
            ]
            if any(
                base.canonical_bytes(item)
                != base.canonical_bytes(normalized[0])
                for item in normalized[1:]
            ):
                failures.append(f"{cluster_id}:{motion}:UNDECLARED_QUALITY_DIFF")
        for field in shared_across_motion:
            if len({arm.get(field) for arm in arms}) != 1:
                failures.append(f"{cluster_id}:{field}:UNDECLARED_MOTION_DIFF")
    return {
        "id": "G11_SIX_ARM_PAIRING",
        "status": "PASS" if len(main) == 80 and not failures else "FAIL",
        "cluster_count": len(main),
        "exact_arm_schema_field_count": len(ARM_KEYS),
        "failures": failures,
    }


def _scene_hash(scene: dict[str, Any]) -> str:
    core = copy.deepcopy(scene)
    core.pop("scene_geometry_sha256", None)
    return base.sha256_bytes(base.canonical_bytes(core))


def gate_g12_r2(
    main: list[dict[str, Any]], trajectories: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    intrinsic_hash = base.sha256_bytes(base.K.astype("<f8").tobytes())
    static_trajectory_hash = base.sha256_bytes(
        base.canonical_bytes({"static": base.FRAME_COUNT})
    )
    derived_group_hashes: dict[str, str] = {}
    for record in main:
        cluster_id = str(record["cluster_id"])
        block = str(record["block"])
        scene_hash = _scene_hash(record["scene"])
        if (
            record["scene"].get("scene_geometry_sha256") != scene_hash
            or any(
                arm.get("scene_geometry_sha256") != scene_hash
                for arm in record["arms"]
            )
        ):
            failures.append(f"{cluster_id}:SCENE_HASH")
        periodic_hash = str(trajectories[block]["periodic_pose_sha256"])
        for motion in ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION"):
            trajectory_hash = (
                static_trajectory_hash
                if motion == "STATIC_CAMERA"
                else periodic_hash
            )
            identity = {
                "scene_geometry_sha256": scene_hash,
                "camera_intrinsic_sha256": intrinsic_hash,
                "timestamp_sha256": periodic_hash,
                "motion": motion,
            }
            geometry_hash = base.sha256_bytes(base.canonical_bytes(identity))
            derived_group_hashes[f"{cluster_id}:{motion}"] = geometry_hash
            expected = {
                "scene_geometry_sha256": scene_hash,
                "trajectory_sha256": trajectory_hash,
                "geometry_identity_sha256": geometry_hash,
                "depth_sha256": geometry_hash,
                "object_id_sha256": geometry_hash,
                "pose_sha256": geometry_hash,
                "intrinsic_sha256": intrinsic_hash,
                "timestamp_sha256": periodic_hash,
                "visibility_sha256": geometry_hash,
            }
            group = [
                arm for arm in record["arms"] if arm.get("motion") == motion
            ]
            if len(group) != 3:
                failures.append(f"{cluster_id}:{motion}:GROUP_COUNT")
                continue
            for arm in group:
                for field, value in expected.items():
                    if arm.get(field) != value:
                        failures.append(
                            f"{cluster_id}:{arm.get('arm_id')}:{field}"
                        )
    return {
        "id": "G12_QUALITY_GEOMETRY_IDENTITY",
        "status": "PASS" if len(main) == 80 and not failures else "FAIL",
        "motion_level_group_count": len(derived_group_hashes),
        "identity_derivation": (
            "scene+intrinsic+timestamp+motion; producer equality strings are "
            "checked against independently derived source-known identities"
        ),
        "failures": failures,
    }


def gate_g14_r2(
    replay: dict[str, Any],
    guard_replay: dict[str, Any],
    fixtures: dict[str, Any],
    guards: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    items = replay.get("items")
    if not isinstance(items, list) or len(items) != 14:
        failures.append("BASE_REPLAY_ITEM_COUNT")
        items = []
    fixture_by_id = {
        str(item["id"]): item for item in fixtures.get("fixtures", [])
    }
    expected_fixture_ids = set(fixture_by_id)
    observed_fixture_ids: set[str] = set()
    observed_calibration: set[tuple[str, int, int]] = set()
    recomputed_base_mismatch = 0
    for index, item in enumerate(items):
        kind = item.get("kind")
        actual_match = False
        if kind == "analytic_fixture_manifest":
            if set(item) != {
                "kind",
                "fixture_id",
                "first_sha256",
                "second_sha256",
                "match",
            }:
                failures.append(f"BASE_FIXTURE_SCHEMA:{index}")
            fixture_id = str(item.get("fixture_id"))
            observed_fixture_ids.add(fixture_id)
            expected_fixture_hash = (
                base.sha256_bytes(base.canonical_bytes(fixture_by_id[fixture_id]))
                if fixture_id in fixture_by_id
                else None
            )
            actual_match = bool(
                expected_fixture_hash is not None
                and item.get("first_sha256")
                == item.get("second_sha256")
                == expected_fixture_hash
            )
        elif kind == "calibration_seed":
            if set(item) != {
                "kind",
                "block",
                "ordinal",
                "frame_index",
                "first",
                "second",
                "match",
            }:
                failures.append(f"BASE_CALIBRATION_SCHEMA:{index}")
            identity = (
                str(item.get("block")),
                int(item.get("ordinal", -1)),
                int(item.get("frame_index", -1)),
            )
            observed_calibration.add(identity)
            first = item.get("first")
            second = item.get("second")
            expected_hash_keys = {
                "depth_sha256",
                "object_id_sha256",
                "rgb_sha256",
                "visibility_sha256",
            }
            actual_match = bool(
                isinstance(first, dict)
                and isinstance(second, dict)
                and set(first) == expected_hash_keys
                and set(second) == expected_hash_keys
                and first == second
            )
        else:
            failures.append(f"BASE_REPLAY_KIND:{index}")
        recomputed_base_mismatch += int(not actual_match)
        if item.get("match") is not actual_match:
            failures.append(f"BASE_REPLAY_DECLARED_MATCH:{index}")
    expected_calibration = {
        (block, 0, frame)
        for block in base.BLOCKS
        for frame in (0, base.FRAME_COUNT - 1)
    }
    if observed_fixture_ids != expected_fixture_ids or len(observed_fixture_ids) != 6:
        failures.append("BASE_FIXTURE_IDENTITY_SET")
    if observed_calibration != expected_calibration:
        failures.append("BASE_CALIBRATION_IDENTITY_SET")
    if replay.get("mismatch_count") != recomputed_base_mismatch:
        failures.append("BASE_REPLAY_MISMATCH_COUNT")

    guard_items = guard_replay.get("items")
    if not isinstance(guard_items, list) or len(guard_items) != 8:
        failures.append("GUARD_REPLAY_ITEM_COUNT")
        guard_items = []
    guard_by_identity = {
        (str(item["block"]), int(item["ordinal"])): item for item in guards
    }
    observed_guards: set[tuple[str, int]] = set()
    recomputed_guard_mismatch = 0
    for index, item in enumerate(guard_items):
        if set(item) != {
            "kind",
            "block",
            "ordinal",
            "numeric_seed_uint64",
            "first_scene_sha256",
            "second_scene_sha256",
            "manifest_scene_sha256",
            "match",
        }:
            failures.append(f"GUARD_REPLAY_SCHEMA:{index}")
        identity = (str(item.get("block")), int(item.get("ordinal", -1)))
        observed_guards.add(identity)
        record = guard_by_identity.get(identity)
        expected_seed = base.derive_seed("GUARD", *identity)
        expected_scene_hash = (
            base.sha256_bytes(base.canonical_bytes(record["scene"]))
            if record is not None
            else None
        )
        actual_match = bool(
            record is not None
            and int(item.get("numeric_seed_uint64", -1)) == expected_seed
            and item.get("first_scene_sha256")
            == item.get("second_scene_sha256")
            == item.get("manifest_scene_sha256")
            == expected_scene_hash
        )
        recomputed_guard_mismatch += int(not actual_match)
        if item.get("match") is not actual_match:
            failures.append(f"GUARD_REPLAY_DECLARED_MATCH:{index}")
    if observed_guards != set(guard_by_identity):
        failures.append("GUARD_REPLAY_IDENTITY_SET")
    if guard_replay.get("mismatch_count") != recomputed_guard_mismatch:
        failures.append("GUARD_REPLAY_MISMATCH_COUNT")
    if guard_replay.get("implementation_id") != IMPLEMENTATION_ID:
        failures.append("GUARD_REPLAY_IMPLEMENTATION_ID")
    passed = not failures and recomputed_base_mismatch == recomputed_guard_mismatch == 0
    return {
        "id": "G14_DETERMINISTIC_REPLAY",
        "status": "PASS" if passed else "FAIL",
        "base_recomputed_mismatch_count": recomputed_base_mismatch,
        "guard_recomputed_mismatch_count": recomputed_guard_mismatch,
        "scope_counts": {
            "analytic_fixture_manifest": len(observed_fixture_ids),
            "calibration_seed_frame": len(observed_calibration),
            "r1_guard_scene": len(observed_guards),
        },
        "failures": failures,
    }


def _target_visibility_and_measurements(
    scene: dict[str, Any], arm: dict[str, Any]
) -> dict[str, Any]:
    target = scene["designated_target"]
    target_id = int(target["object_id"])
    point = np.asarray([target["world_point_m"]], dtype=np.float64)
    objects = {
        int(item["object_id"]): item for item in scene["world"]["objects"]
    }
    if target_id not in objects:
        raise ValueError("TARGET_OBJECT_MISSING")
    target_object = objects[target_id]
    x0, x1, y0, y1 = target_object["bounds_xy_m"]
    on_mesh = bool(
        abs(float(point[0, 2]) - float(target_object["plane_z_m"])) <= 1e-12
        and x0 <= point[0, 0] <= x1
        and y0 <= point[0, 1] <= y1
    )
    depths = []
    radii = []
    visible = []
    translations = []
    rotations = []
    timestamps = []
    for pose in arm["trajectory"]:
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(pose["translation_m"], dtype=np.float64)
        uv, projected_depth = base.project(point, rotation, translation)
        rendered_depth, rendered_object, _ = base.raycast(
            scene, rotation, translation, uv
        )
        is_visible = bool(
            0.0 <= uv[0, 0] < base.WIDTH
            and 0.0 <= uv[0, 1] < base.HEIGHT
            and np.isfinite(rendered_depth[0])
            and int(rendered_object[0]) == target_id
            and abs(float(rendered_depth[0] - projected_depth[0])) <= 1e-7
        )
        visible.append(is_visible)
        depths.append(float(projected_depth[0]))
        radii.append(float(np.linalg.norm(uv[0] - base.K[:2, 2])))
        translations.append(translation)
        rotations.append(rotation)
        timestamps.append(float(pose["timestamp_s"]))
    depth_array = np.asarray(depths, dtype=np.float64)
    radius_array = np.asarray(radii, dtype=np.float64)
    inverse_increase = float(
        (1.0 / depth_array[-1]) / (1.0 / depth_array[0]) - 1.0
    )
    integrated_log_radial = float(
        math.log(radius_array[-1] / radius_array[0])
    )
    return {
        "on_mesh": on_mesh,
        "visible": np.asarray(visible, dtype=bool),
        "depth": depth_array,
        "radius": radius_array,
        "inverse_increase": inverse_increase,
        "integrated_log_radial": integrated_log_radial,
        "translation": np.asarray(translations, dtype=np.float64),
        "rotation": np.asarray(rotations, dtype=np.float64),
        "timestamp": np.asarray(timestamps, dtype=np.float64),
        "point": point,
    }


def gate_g13_r2(
    guards: list[dict[str, Any]], trajectories: dict[str, Any]
) -> dict[str, Any]:
    summaries = []
    failures = []
    for record in guards:
        block = str(record["block"])
        _, periodic_translation, periodic_rotation = base.pose_arrays(
            trajectories[block]
        )
        for arm in record["arms"]:
            arm_id = str(arm["arm_id"])
            try:
                measured = _target_visibility_and_measurements(
                    record["scene"], arm
                )
            except (KeyError, ValueError, ZeroDivisionError) as error:
                failures.append(
                    f"{record['cluster_id']}:{arm_id}:TARGET:{type(error).__name__}"
                )
                continue
            timestamps = measured["timestamp"]
            phase = (
                (timestamps - timestamps[0])
                / (timestamps[-1] - timestamps[0])
            )
            expected_approach = np.column_stack(
                (
                    np.zeros(base.FRAME_COUNT),
                    np.zeros(base.FRAME_COUNT),
                    0.8 * phase,
                )
            )
            if arm_id == "MONOTONIC_APPROACH":
                approach_translation = measured["translation"]
                approach_rotation = measured["rotation"]
            elif arm_id == "MONOTONIC_APPROACH_PLUS_PERIODIC":
                approach_translation = (
                    measured["translation"] - periodic_translation
                )
                rotation_component_identity = np.einsum(
                    "nij,njk->nik",
                    np.transpose(periodic_rotation, (0, 2, 1)),
                    measured["rotation"],
                )
                approach_rotation = np.repeat(
                    np.eye(3, dtype=np.float64)[None, :, :],
                    base.FRAME_COUNT,
                    axis=0,
                )
            else:
                failures.append(
                    f"{record['cluster_id']}:{arm_id}:UNEXPECTED_ARM"
                )
                continue
            translation_error = float(
                np.max(
                    np.abs(approach_translation - expected_approach)
                )
            )
            rotation_error = float(
                np.max(
                    np.abs(
                        (
                            rotation_component_identity
                            if arm_id == "MONOTONIC_APPROACH_PLUS_PERIODIC"
                            else approach_rotation
                        )
                        - np.eye(3, dtype=np.float64)[None, :, :]
                    )
                )
            )
            approach_component_depth = np.asarray(
                [
                    (
                        approach_rotation[index].T
                        @ (
                            measured["point"]
                            - approach_translation[index]
                        ).T
                    ).T[0, 2]
                    for index in range(base.FRAME_COUNT)
                ],
                dtype=np.float64,
            )
            monotonic = bool(
                np.all(np.diff(approach_component_depth) <= 1e-12)
            )
            passed = bool(
                measured["on_mesh"]
                and np.all(measured["visible"])
                and measured["inverse_increase"] >= 0.20
                and measured["integrated_log_radial"] >= LOG_1P20
                and translation_error <= 1e-12
                and rotation_error <= 1e-12
                and monotonic
            )
            if not passed:
                failures.append(f"{record['cluster_id']}:{arm_id}")
            summaries.append(
                {
                    "cluster_id": record["cluster_id"],
                    "arm_id": arm_id,
                    "target_on_rendered_mesh": measured["on_mesh"],
                    "persistent_visible_frame_count": int(
                        np.count_nonzero(measured["visible"])
                    ),
                    "inverse_depth_endpoint_increase": measured[
                        "inverse_increase"
                    ],
                    "integrated_endpoint_log_radial_expansion": measured[
                        "integrated_log_radial"
                    ],
                    "source_component_translation_max_error_m": translation_error,
                    "source_component_rotation_matrix_max_error": (
                        rotation_error
                    ),
                    "approach_component_depth_monotonic": monotonic,
                }
            )
    return {
        "id": "G13_MONOTONIC_APPROACH_TRUTH",
        "status": (
            "PASS" if len(summaries) == 16 and not failures else "FAIL"
        ),
        "sequence_count": len(summaries),
        "integrated_log_radial_gate": LOG_1P20,
        "failures": failures,
        "summaries": summaries,
    }


def validate(evidence: Path, lock_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    lock = base.load_json(lock_path)
    amendment = base.load_json(AMENDMENT_PATH)
    geometry_spec = base.load_json(base.GEOMETRY_PATH)
    runtime = base.load_json(evidence / "runtime_manifest.json")
    trajectories = base.load_json(evidence / "trajectory_manifest.json")
    fixtures = base.load_json(evidence / "analytic_fixture_ledger.json")
    replay = base.load_json(evidence / "deterministic_replay_ledger.json")
    guard_replay = base.load_json(evidence / "guard_scene_replay_ledger.json")
    package_manifest = base.load_json(evidence / "package_manifest.json")
    samples = base.load_json(evidence / "projective_sample_ledger.json")
    producer = base.load_json(
        evidence / "generator_r2_keyset_repair_r0_receipt.json"
    )
    records = base.load_jsonl(evidence / "all_seed_geometry_manifest.jsonl")
    _validate_keyset_repair_lock(lock, evidence, errors)
    _validate_r0_immutability(records, evidence, errors)
    if (
        amendment.get("status")
        != "FROZEN_BEFORE_R2_KEYSET_REPAIR_R0_MATERIALIZATION"
    ):
        errors.append("KEYSET_REPAIR_AMENDMENT_STATUS")
    for field in (
        "formal_execution_authorized",
        "quality_calibration_authorized",
        "automatic_p2_authority",
    ):
        if amendment.get(field) is not False:
            errors.append(f"KEYSET_REPAIR_AMENDMENT_AUTHORITY:{field}")
    if amendment.get("machine_gate_lock") != {
        "main_cluster_count": 80,
        "guard_cluster_count": 8,
        "guard_object_count": EXPECTED_GUARD_OBJECT_COUNT,
        "target_object_id": EXPECTED_TARGET_ID,
        "target_world_point_m": EXPECTED_TARGET_POINT,
        "required_persistent_frames_per_arm": base.FRAME_COUNT,
        "inverse_depth_endpoint_increase_gte": 0.20,
        "integrated_endpoint_log_radial_expansion_gte": LOG_1P20,
        "deperiodized_translation_max_error_lte_m": 1e-12,
        "deperiodized_rotation_matrix_component_max_error_lte": 1e-12,
        "visibility_mismatch_fraction_lte": 0.001,
        "guard_scene_replay_count": 8,
        "required_gate_count": 14,
    }:
        errors.append("KEYSET_REPAIR_AMENDMENT_MACHINE_GATE_LOCK")
    if (
        runtime.get("rcle_imported_or_executed") is not False
        or runtime.get("implementation_id")
        != (
            "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
            "GENERATOR_GEOMETRY_IMPLEMENTATION_R0"
        )
        or runtime.get("manifest_schema")
        != (
            "rcle.periodic_self_motion_counterfactual."
            "p1_geometry_manifest.v1"
        )
    ):
        errors.append("KEYSET_REPAIR_RUNTIME_RCLE_FIREWALL")
    expected_package = {
        "main_cluster": (
            "rcle.periodic_self_motion_counterfactual."
            "p1_geometry_manifest.v1"
        ),
        "guardrail_cluster": (
            "rcle.periodic_self_motion_counterfactual."
            "p1_geometry_manifest.v2"
        ),
    }
    if (
        package_manifest.get("schema")
        != (
            "rcle.periodic_self_motion_counterfactual."
            "p1_geometry_package_r2_keyset_repair_r0.v1"
        )
        or package_manifest.get("implementation_id") != IMPLEMENTATION_ID
        or package_manifest.get("scene_schema_union") != expected_package
        or package_manifest.get("record_counts")
        != {"main_cluster": 80, "guardrail_cluster": 8}
        or package_manifest.get("formal_execution_authorized") is not False
        or package_manifest.get("automatic_p2_authority") is not False
        or package_manifest.get("all_seed_manifest_sha256")
        != base.sha256_file(evidence / "all_seed_geometry_manifest.jsonl")
        or package_manifest.get("guard_scene_replay_sha256")
        != base.sha256_file(evidence / "guard_scene_replay_ledger.json")
        or package_manifest.get("runtime_manifest_role")
        != (
            "byte-identical inherited R0 renderer/environment evidence; "
            "its embedded R0 implementation_id is intentionally historical"
        )
    ):
        errors.append("KEYSET_REPAIR_PACKAGE_MANIFEST")
    for field in (
        "rcle_output_accessed_or_executed",
        "quality_strength_calibrated",
        "performance_preflight_run",
        "formal_sequences_run",
        "formal_execution_authorized",
    ):
        if producer.get(field) is not False:
            errors.append(f"KEYSET_REPAIR_PRODUCER_BOUNDARY:{field}")
    expected_producer_identity = {
        "status": "R2_KEYSET_REPAIR_R0_EVIDENCE_MATERIALIZED",
        "implementation_id": IMPLEMENTATION_ID,
        "r2_failed_receipt_sha256": EXPECTED_R2_FAILED_RECEIPT_SHA256,
        "main_cluster_count": 80,
        "guardrail_cluster_count": 8,
        "main_record_change_count_from_r2": 0,
        "guardrail_record_change_count_from_r2": 0,
        "numeric_seed_replacement_count": 0,
        "trajectory_change_count": 0,
        "guard_scene_replay_count": 8,
    }
    for field, expected in expected_producer_identity.items():
        if producer.get(field) != expected:
            errors.append(f"KEYSET_REPAIR_PRODUCER_IDENTITY:{field}")
    producer_hashes = producer.get("artifact_sha256")
    if (
        not isinstance(producer_hashes, dict)
        or set(producer_hashes) != REQUIRED_PRODUCER_ARTIFACT_KEYS
    ):
        errors.append("KEYSET_REPAIR_PRODUCER_ARTIFACT_HASH_KEYSET")
        producer_hashes = {}
    for name, expected in producer_hashes.items():
        path = evidence / name
        if not path.is_file() or base.sha256_file(path) != expected:
            errors.append(f"KEYSET_REPAIR_PRODUCER_ARTIFACT_HASH:{name}")
    main = [item for item in records if item.get("record_type") == "main_cluster"]
    guards = [
        item for item in records if item.get("record_type") == "guardrail_cluster"
    ]
    _validate_guard_scene_contract(guards, errors)
    main_by_key = {(item["block"], int(item["ordinal"])): item for item in main}
    fixture_by_id = {item["id"]: item for item in fixtures["fixtures"]}
    gates: list[dict[str, Any]] = []
    gates.extend(gate_g01_g02_r2(main, guards))
    gates.append(gate_g03_r2(fixtures, samples, trajectories, main_by_key))
    gates.append(base.gate_g04(fixture_by_id, main))
    gates.append(
        base.gate_g05(
            fixture_by_id["PURE_TRANSLATION_LATERAL_MULTI_DEPTH"]
        )
    )
    gates.append(
        base.gate_g06(
            fixture_by_id["PURE_TRANSLATION_LATERAL_MULTI_DEPTH"],
            trajectories,
            main_by_key,
        )
    )
    gates.append(
        base.gate_g07(
            fixture_by_id["PURE_ROTATION_SHARED_BEARINGS_MULTI_DEPTH"]
        )
    )
    gates.append(gate_g08_r2(main, trajectories))
    gates.extend(base.gates_g09_g10(trajectories))
    gates.append(gate_g11_r2(main))
    gates.append(gate_g12_r2(main, trajectories))
    gates.append(gate_g13_r2(guards, trajectories))
    gates.append(gate_g14_r2(replay, guard_replay, fixtures, guards))
    required = [gate["id"] for gate in geometry_spec["required_gates"]]
    if [gate["id"] for gate in gates] != required:
        errors.append("KEYSET_REPAIR_GATE_ORDER_OR_IDENTITY")
    failed = [gate["id"] for gate in gates if gate["status"] != "PASS"]
    if errors or failed:
        status = "INVALID" if errors else "VALID_FAIL_CLOSED"
        terminal = "INTERVENTION_NOT_EVALUABLE"
        state = "HOLD_P1"
    else:
        status = "VALID"
        terminal = "GENERATOR_GEOMETRY_PASS"
        state = "EXECUTION_NOT_AUTHORIZED"
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "p1_independent_geometry_r2_keyset_repair_r0_receipt.v1"
        ),
        "protocol_id": base.PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": status,
        "terminal": terminal,
        "state": state,
        "gate_pass_count": sum(
            gate["status"] == "PASS" for gate in gates
        ),
        "gate_required_count": 14,
        "failed_gates": failed,
        "errors": sorted(errors),
        "gates": gates,
        "immutable_r0": {
            "receipt_sha256": EXPECTED_R0_RECEIPT_SHA256,
            "terminal": "INTERVENTION_NOT_EVALUABLE / HOLD_P1",
        },
        "immutable_r1_failure": {
            "receipt_sha256": EXPECTED_R1_FAILED_RECEIPT_SHA256,
            "terminal": "INTERVENTION_NOT_EVALUABLE / HOLD_P1",
        },
        "immutable_r2_failure": {
            "receipt_sha256": EXPECTED_R2_FAILED_RECEIPT_SHA256,
            "terminal": "INTERVENTION_NOT_EVALUABLE / HOLD_P1",
        },
        "implementation_lock_sha256": base.sha256_file(lock_path),
        "amendment_sha256": base.sha256_file(AMENDMENT_PATH),
        "evidence_sha256": {
            path.name: base.sha256_file(path)
            for path in sorted(evidence.iterdir())
            if path.is_file()
            and path.name != "independent_geometry_validation_receipt.json"
        },
        "validator_source_sha256": base.sha256_file(Path(__file__)),
        "generator_imported": False,
        "rcle_output_accessed_or_executed": False,
        "quality_strength_calibrated": False,
        "performance_preflight_run": False,
        "formal_sequences_run": False,
        "formal_execution_authorized": False,
        "automatic_p2_authority": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--formal-receipt", type=Path)
    return parser.parse_args()


def _write_receipt_exclusive(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(base.canonical_bytes(receipt))


def main() -> int:
    args = parse_args()
    evidence = args.evidence.resolve()
    lock_path = args.lock.resolve()
    try:
        receipt = validate(evidence, lock_path)
    except Exception as error:
        receipt = {
            "status": "INVALID",
            "terminal": "INTERVENTION_NOT_EVALUABLE",
            "state": "HOLD_P1",
            "errors": [f"{type(error).__name__}:{error}"],
            "formal_execution_authorized": False,
        }
    if args.formal_receipt is not None:
        formal_receipt = args.formal_receipt.resolve()
        expected_receipt = (
            evidence / "independent_geometry_validation_receipt.json"
        )
        if formal_receipt != expected_receipt:
            receipt = {
                "status": "INVALID",
                "terminal": "INTERVENTION_NOT_EVALUABLE",
                "state": "HOLD_P1",
                "errors": ["FORMAL_RECEIPT_PATH_MUST_MATCH_EVIDENCE"],
                "formal_execution_authorized": False,
            }
        else:
            try:
                _write_receipt_exclusive(formal_receipt, receipt)
            except Exception as error:
                receipt = {
                    "status": "INVALID",
                    "terminal": "INTERVENTION_NOT_EVALUABLE",
                    "state": "HOLD_P1",
                    "errors": [
                        f"FORMAL_RECEIPT_EXCLUSIVE_CREATE:"
                        f"{type(error).__name__}:{error}"
                    ],
                    "formal_execution_authorized": False,
                }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt.get("status") == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
