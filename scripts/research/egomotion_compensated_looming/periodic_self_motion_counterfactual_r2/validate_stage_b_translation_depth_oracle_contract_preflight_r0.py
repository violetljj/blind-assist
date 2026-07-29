"""Independent contract-only preflight for RCLE Stage B.

The validator reads frozen control-plane documents and runs small analytic
geometry fixtures.  It does not import RCLE evaluation/local-fit code, create
Stage B render identities, or read any Stage B response/workload output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_"
    "TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
ARMS = (
    "STATIC_SCENE",
    "EGO_ROTATION_STATIC_SCENE",
    "EGO_TRANSLATION_STATIC_SCENE",
    "OBJECT_APPROACH_STATIC_CAMERA",
    "OBJECT_APPROACH_PLUS_EGO_6DOF",
)
POSE_HASHES = {
    "ADVIO_13": "ec6ac2dedf8a767755d0388aa2100287dbf245f1ea02016b003a7398441a60ae",
    "ADVIO_14": "2f093d63b1dfb5b51f08be3dd01ab2ed3479a5d31f89cef564f5b297ddc7dadb",
    "ADVIO_15": "4f7e0f79303b91b2aa56a915a56cd8b23b4393d68b1314908b2ade5c5e2b90f1",
    "ADVIO_17": "a3b44d5c17d0899747b9cbd3ffb35e68d458d889f148f7c469a3d0e99140b7ac",
}
DOC_STEM = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_"
    "CONTRACT_PREFLIGHT_R0"
)
CONTRACT_RELATIVE = f"{DOC_STEM}_CONTRACT_2026-07-29.json"
IDENTITY_RELATIVE = f"{DOC_STEM}_IDENTITY_LOCK_2026-07-29.json"
RECEIPT_RELATIVE = f"{DOC_STEM}_INDEPENDENT_RECEIPT_2026-07-29.json"
DECISION_RELATIVE = (
    f"{DOC_STEM}_EXECUTION_ACTIVATION_DECISION_2026-07-29.json"
)
VALIDATOR_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "validate_stage_b_translation_depth_oracle_contract_preflight_r0.py"
)
STAGE_B_RESPONSE_ROOT = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_stage_b_translation_depth_oracle_object_approach_r0"
)


class InvalidStageBContractPreflight(RuntimeError):
    """Raised when a frozen preflight gate does not pass."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InvalidStageBContractPreflight(f"NOT_OBJECT:{path}")
    return value


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise InvalidStageBContractPreflight(label)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise InvalidStageBContractPreflight(
            f"{label}:expected={expected!r}:actual={actual!r}"
        )


def _close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise InvalidStageBContractPreflight(
            f"{label}:expected={expected!r}:actual={actual!r}:tol={tolerance}"
        )


def _rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    projected = (intrinsic @ points.T).T
    return projected[:, :2] / projected[:, 2:3]


def oracle_displacement(
    pixels_previous: np.ndarray,
    depth_previous_m: np.ndarray,
    intrinsic: np.ndarray,
    rotation_world_from_previous: np.ndarray,
    translation_world_from_previous_m: np.ndarray,
    rotation_world_from_current: np.ndarray,
    translation_world_from_current_m: np.ndarray,
) -> np.ndarray:
    """Return camera-translation displacement in the R3-aligned frame."""

    homogeneous = np.column_stack(
        (pixels_previous, np.ones(len(pixels_previous), dtype=np.float64))
    )
    points_previous = (
        np.linalg.inv(intrinsic) @ homogeneous.T
    ).T * depth_previous_m[:, None]
    rotation_current_from_previous = (
        rotation_world_from_current.T @ rotation_world_from_previous
    )
    translation_current_from_previous = rotation_world_from_current.T @ (
        translation_world_from_previous_m - translation_world_from_current_m
    )
    points_current = (
        rotation_current_from_previous @ points_previous.T
    ).T + translation_current_from_previous
    pixels_current = _project(points_current, intrinsic)
    homography = (
        intrinsic
        @ rotation_current_from_previous
        @ np.linalg.inv(intrinsic)
    )
    homogeneous_current = np.column_stack(
        (pixels_current, np.ones(len(pixels_current), dtype=np.float64))
    )
    aligned = (np.linalg.inv(homography) @ homogeneous_current.T).T
    pixels_r3 = aligned[:, :2] / aligned[:, 2:3]
    return pixels_r3 - pixels_previous


def _fit_affine_expansion(
    previous: np.ndarray, current: np.ndarray, dt_s: float
) -> tuple[np.ndarray, float]:
    design = np.column_stack(
        (previous, np.ones(len(previous), dtype=np.float64))
    )
    coefficients = np.linalg.lstsq(design, current, rcond=None)[0]
    linear = coefficients[:2, :].T - np.eye(2)
    return coefficients, float(0.5 * np.trace(linear) / dt_s)


def _visibility_valid(
    *,
    source_depth_valid: bool = True,
    actual_current_z_positive: bool = True,
    actual_endpoint_in_bounds: bool = True,
    same_material_object: bool = True,
    actual_zbuffer_visible: bool = True,
    generator_masks_valid: bool = True,
    r3_warp_mask_valid: bool = True,
    sparse_track_valid: bool = True,
) -> bool:
    return all(
        (
            source_depth_valid,
            actual_current_z_positive,
            actual_endpoint_in_bounds,
            same_material_object,
            actual_zbuffer_visible,
            generator_masks_valid,
            r3_warp_mask_valid,
            sparse_track_valid,
        )
    )


def analytic_geometry_gates(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate formula direction, units and refit semantics without RCLE."""

    intrinsic = np.asarray(
        contract["coordinate_and_unit_contract"]["image"]["intrinsic"],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    pixels = np.asarray(
        (
            (110.0, 180.0),
            (182.3389, 321.654),
            (260.0, 470.0),
        ),
        dtype=np.float64,
    )
    depths = np.full(len(pixels), 4.0, dtype=np.float64)
    rotation = _rotation_y(0.13)

    zero_translation = oracle_displacement(
        pixels, depths, intrinsic, identity, zero, rotation, zero
    )
    zero_translation_error = float(np.max(np.abs(zero_translation)))
    _require(zero_translation_error <= 1e-12, "G03_ZERO_TRANSLATION")

    forward_m = 0.4
    forward = oracle_displacement(
        pixels,
        depths,
        intrinsic,
        identity,
        zero,
        identity,
        np.asarray((0.0, 0.0, forward_m)),
    )
    center = np.asarray((intrinsic[0, 2], intrinsic[1, 2]))
    analytic_forward = (pixels - center) * (
        depths[:, None] / (depths[:, None] - forward_m) - 1.0
    )
    plane_error = float(np.max(np.abs(forward - analytic_forward)))
    _require(plane_error <= 1e-10, "G04_CONSTANT_DEPTH_PLANE")

    backward = oracle_displacement(
        pixels,
        depths,
        intrinsic,
        identity,
        zero,
        identity,
        np.asarray((0.0, 0.0, -forward_m)),
    )
    radial = pixels - center
    selected = np.linalg.norm(radial, axis=1) > 1e-9
    forward_dot = float(np.median(np.sum(forward[selected] * radial[selected], axis=1)))
    backward_dot = float(
        np.median(np.sum(backward[selected] * radial[selected], axis=1))
    )
    _require(forward_dot > 0.0 and backward_dot < 0.0, "G05_DIRECTION_REVERSAL")

    scale = 0.5
    scale_matrix = np.diag((scale, scale, 1.0))
    scaled_intrinsic = scale_matrix @ intrinsic
    scaled_pixels = pixels * scale
    scaled = oracle_displacement(
        scaled_pixels,
        depths,
        scaled_intrinsic,
        identity,
        zero,
        identity,
        np.asarray((0.0, 0.0, forward_m)),
    )
    scale_error = float(np.max(np.abs(scaled / scale - forward)))
    _require(scale_error <= 1e-10, "G06_INTRINSIC_SCALE_EQUIVALENCE")

    previous_object_point = np.asarray(((0.2, 0.1, 6.0),), dtype=np.float64)
    actual_object_point = np.asarray(((0.2, 0.1, 5.0),), dtype=np.float64)
    camera_current = np.asarray((0.0, 0.0, 0.3), dtype=np.float64)
    previous_object_pixel = _project(previous_object_point, intrinsic)
    actual_current_pixel = _project(
        actual_object_point - camera_current, intrinsic
    )
    rigid_current_pixel = _project(
        previous_object_point - camera_current, intrinsic
    )
    camera_only = oracle_displacement(
        previous_object_pixel,
        np.asarray((6.0,), dtype=np.float64),
        intrinsic,
        identity,
        zero,
        identity,
        camera_current,
    )
    observed = actual_current_pixel - previous_object_pixel
    oracle_residual = observed - camera_only
    expected_object_residual = actual_current_pixel - rigid_current_pixel
    anti_swallow_error = float(
        np.max(np.abs(oracle_residual - expected_object_residual))
    )
    _require(anti_swallow_error <= 1e-10, "G08_MOVING_OBJECT_NOT_SWALLOWED")
    _require(
        float(np.linalg.norm(oracle_residual)) > 0.0,
        "G08_MOVING_OBJECT_RESIDUAL_ZERO",
    )

    _require(_visibility_valid(), "G07_VALID_VISIBILITY_REJECTED")
    visibility_fields = (
        "source_depth_valid",
        "actual_current_z_positive",
        "actual_endpoint_in_bounds",
        "same_material_object",
        "actual_zbuffer_visible",
        "generator_masks_valid",
        "r3_warp_mask_valid",
        "sparse_track_valid",
    )
    for field in visibility_fields:
        _require(
            not _visibility_valid(**{field: False}),
            f"G07_INVALID_VISIBILITY_ACCEPTED:{field}",
        )

    grid = np.asarray(
        [(x, y) for y in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)],
        dtype=np.float64,
    )
    dt_s = 0.1
    translation_linear = np.asarray(((0.03, 0.01), (-0.01, 0.02)))
    object_linear = np.asarray(((0.04, 0.00), (0.00, 0.04)))
    translation_offset = np.asarray((0.2, -0.1))
    object_offset = np.asarray((-0.05, 0.03))
    translation_term = grid @ translation_linear.T + translation_offset
    object_term = grid @ object_linear.T + object_offset
    baseline_endpoint = grid + translation_term + object_term
    oracle_endpoint = baseline_endpoint - translation_term
    baseline_coefficients, baseline_expansion = _fit_affine_expansion(
        grid, baseline_endpoint, dt_s
    )
    oracle_coefficients, oracle_expansion = _fit_affine_expansion(
        grid, oracle_endpoint, dt_s
    )
    expected_oracle_expansion = float(
        0.5 * np.trace(object_linear) / dt_s
    )
    _close(
        oracle_expansion,
        expected_oracle_expansion,
        1e-12,
        "G09_LOCAL_REFIT_ORACLE_EXPANSION",
    )
    _require(
        not np.allclose(
            baseline_coefficients,
            oracle_coefficients,
            rtol=0.0,
            atol=1e-12,
        ),
        "G09_BASELINE_FIT_REUSED",
    )
    _require(
        baseline_expansion > oracle_expansion > 0.0,
        "G09_LOCAL_REFIT_DIRECTION",
    )

    object_contract = contract["object_approach_positive_control"]
    _close(
        float(object_contract["initial_depth_m"])
        / float(object_contract["final_depth_m"]),
        float(object_contract["analytic_endpoint_radial_scale"]),
        1e-15,
        "G10_OBJECT_RADIAL_SCALE",
    )
    _close(
        math.log(float(object_contract["analytic_endpoint_radial_scale"])),
        float(object_contract["analytic_endpoint_log_radial_expansion"]),
        1e-15,
        "G10_OBJECT_LOG_EXPANSION",
    )

    return {
        "G03_ZERO_TRANSLATION": {
            "status": "PASS",
            "max_abs_px": zero_translation_error,
        },
        "G04_CONSTANT_DEPTH_PLANE": {
            "status": "PASS",
            "max_abs_px": plane_error,
        },
        "G05_DIRECTION_REVERSAL": {
            "status": "PASS",
            "forward_radial_dot": forward_dot,
            "backward_radial_dot": backward_dot,
        },
        "G06_INTRINSIC_SCALE_EQUIVALENCE": {
            "status": "PASS",
            "max_abs_px": scale_error,
        },
        "G07_VISIBILITY_AND_MASK": {
            "status": "PASS",
            "invalid_cases_rejected": len(visibility_fields),
        },
        "G08_MOVING_OBJECT_NOT_SWALLOWED": {
            "status": "PASS",
            "max_abs_px": anti_swallow_error,
            "residual_norm_px": float(np.linalg.norm(oracle_residual)),
        },
        "G09_ORACLE_LOCAL_REFIT": {
            "status": "PASS",
            "baseline_expansion_per_s": baseline_expansion,
            "oracle_expansion_per_s": oracle_expansion,
        },
        "G10_OBJECT_GEOMETRY": {
            "status": "PASS",
            "endpoint_radial_scale": float(
                object_contract["analytic_endpoint_radial_scale"]
            ),
        },
    }


def _walk_identity_values(value: Any) -> Iterable[tuple[str, Any]]:
    identity_keys = {
        "cluster_id",
        "sequence_id",
        "numeric_seed_uint64",
        "token",
        "token_sha256",
        "object_token",
        "object_token_sha256",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if key in identity_keys and isinstance(item, (str, int)):
                yield key, item
            yield from _walk_identity_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_identity_values(item)


def _exclusion_values(
    root: Path, authority: dict[str, Any]
) -> set[tuple[str, Any]]:
    values = set(_walk_identity_values(authority))
    for source in authority.get("exclusion_sources", []):
        relative = source.get("path")
        if not isinstance(relative, str):
            continue
        path = root / relative
        if path.is_file() and path.suffix.lower() == ".json":
            values.update(
                _walk_identity_values(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            )
    return values


def validate_identity_lock(
    root: Path, contract: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    _equal(identity.get("protocol_id"), PROTOCOL_ID, "IDENTITY_PROTOCOL")
    _equal(identity.get("task_id"), TASK_ID, "IDENTITY_TASK")
    _equal(identity.get("arms"), list(ARMS), "IDENTITY_ARMS")
    _equal(
        identity.get("counts"),
        {
            "arms_per_cluster": 5,
            "blocks": 4,
            "clusters": 8,
            "frames": 24080,
            "ordinals_per_block": 2,
            "pairs": 24040,
            "sequences": 40,
            "target_identities": 8,
        },
        "IDENTITY_COUNTS",
    )
    _equal(
        identity.get("terminal"),
        "STAGE_B_IDENTITIES_FROZEN / GEOMETRY_NOT_MATERIALIZED / NOT_EXECUTABLE",
        "IDENTITY_TERMINAL",
    )
    _equal(identity.get("execution_authorized"), False, "IDENTITY_EXECUTION")
    _equal(identity.get("stage_b_response_read"), False, "IDENTITY_RESPONSE")
    _equal(identity.get("stage_b_workload_run"), False, "IDENTITY_WORKLOAD")
    _equal(
        identity["contract"].get("path"), CONTRACT_RELATIVE, "IDENTITY_CONTRACT_PATH"
    )
    _equal(
        identity["contract"].get("sha256"),
        sha256_file(root / CONTRACT_RELATIVE),
        "IDENTITY_CONTRACT_HASH",
    )

    authority_path = root / identity["exclusion_authority"]["path"]
    _require(authority_path.is_file(), "EXCLUSION_AUTHORITY_MISSING")
    _equal(
        sha256_file(authority_path),
        identity["exclusion_authority"]["sha256"],
        "EXCLUSION_AUTHORITY_HASH",
    )
    exclusion = _exclusion_values(root, load_json(authority_path))
    clusters = identity.get("clusters", [])
    _equal(len(clusters), 8, "CLUSTER_COUNT")
    expected_pairs = [(block, ordinal) for block in BLOCKS for ordinal in (0, 1)]
    actual_pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, Any]] = set()
    for cluster in clusters:
        block = cluster["block"]
        ordinal = int(cluster["ordinal"])
        actual_pairs.append((block, ordinal))
        token = f"{TASK_ID}|SCENE|{block}|{ordinal:02d}"
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        object_token = (
            f"{TASK_ID}|OBJECT|{block}|{ordinal:02d}|TARGET_1001"
        )
        object_digest = hashlib.sha256(object_token.encode("utf-8")).hexdigest()
        cluster_id = f"B_TDO_OA_R0_{block}_S{ordinal + 1}"
        _equal(cluster["cluster_id"], cluster_id, "CLUSTER_ID")
        _equal(cluster["token"], token, "CLUSTER_TOKEN")
        _equal(cluster["token_sha256"], digest, "CLUSTER_TOKEN_HASH")
        _equal(
            cluster["numeric_seed_uint64"],
            int.from_bytes(bytes.fromhex(digest)[:8], "big"),
            "CLUSTER_NUMERIC_SEED",
        )
        _equal(
            cluster["source_periodic_pose_sha256"],
            POSE_HASHES[block],
            "CLUSTER_POSE_HASH",
        )
        _equal(cluster["target_object_id"], 1001, "TARGET_OBJECT_ID")
        _equal(cluster["object_token"], object_token, "OBJECT_TOKEN")
        _equal(
            cluster["object_token_sha256"],
            object_digest,
            "OBJECT_TOKEN_HASH",
        )
        _equal(cluster["state"], "FROZEN_NOT_EXECUTABLE", "CLUSTER_STATE")
        expected_sequences = [
            f"{cluster_id}__{arm}__CLEAN" for arm in ARMS
        ]
        _equal(cluster["sequence_ids"], expected_sequences, "SEQUENCE_IDS")
        for key, item in _walk_identity_values(cluster):
            pair = (key, item)
            _require(pair not in seen, f"WITHIN_STAGE_B_IDENTITY_DUPLICATE:{pair}")
            _require(pair not in exclusion, f"EXTERNAL_IDENTITY_COLLISION:{pair}")
            seen.add(pair)
    _equal(actual_pairs, expected_pairs, "BLOCK_ORDINAL_ORDER")
    identity_set_hash = hashlib.sha256(canonical_bytes(clusters)).hexdigest()
    _equal(
        identity.get("identity_set_sha256"),
        identity_set_hash,
        "IDENTITY_SET_HASH",
    )
    _equal(
        contract["design"]["cluster_count"], len(clusters), "CONTRACT_CLUSTER_COUNT"
    )
    _equal(
        contract["design"]["sequence_count"],
        len(clusters) * len(ARMS),
        "CONTRACT_SEQUENCE_COUNT",
    )
    return {
        "status": "PASS",
        "cluster_count": len(clusters),
        "sequence_count": len(clusters) * len(ARMS),
        "exclusion_identity_value_count": len(exclusion),
        "identity_set_sha256": identity_set_hash,
        "collision_count": 0,
    }


def validate_and_build(
    root: Path,
    *,
    contract_path: Path | None = None,
    identity_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = contract_path or root / CONTRACT_RELATIVE
    identity_path = identity_path or root / IDENTITY_RELATIVE
    validator_path = root / VALIDATOR_RELATIVE
    for path in (contract_path, identity_path, validator_path):
        _require(path.is_file(), f"MISSING_INPUT:{path}")
    contract = load_json(contract_path)
    identity = load_json(identity_path)

    _equal(contract.get("protocol_id"), PROTOCOL_ID, "CONTRACT_PROTOCOL")
    _equal(contract.get("task_id"), TASK_ID, "CONTRACT_TASK")
    _equal(
        contract.get("terminal"),
        "STAGE_B_CONTRACT_FROZEN / PREFLIGHT_PENDING / NOT_RUN",
        "CONTRACT_TERMINAL",
    )
    _equal(contract.get("execution_authorized"), False, "CONTRACT_EXECUTION")
    _equal(
        contract.get("response_access_authorized"),
        False,
        "CONTRACT_RESPONSE_ACCESS",
    )
    _equal(
        contract["formal_firewall"]["formal_sequences_run"],
        0,
        "FORMAL_SEQUENCES",
    )
    _equal(
        contract["formal_firewall"]["formal_r3_pair_core_calls"],
        0,
        "FORMAL_R3_CALLS",
    )
    _equal(
        contract["formal_firewall"]["formal_authority_consumed"],
        False,
        "FORMAL_AUTHORITY",
    )
    _equal(
        contract["analysis_and_reporting"]["unchanged_threshold"],
        {
            "abstention_resets_streak": True,
            "consecutive_pairs": 3,
            "operator": "strict_greater_than",
            "per_s": 0.01,
        },
        "UNCHANGED_THRESHOLD",
    )
    _equal(
        contract["analysis_and_reporting"][
            "signed_response_and_absolute_leakage_separate"
        ],
        True,
        "SIGNED_ABSOLUTE_SEPARATION",
    )
    _equal(
        contract["future_execution_gates"]["rotation_boundary"][
            "required_clusters"
        ],
        8,
        "ROTATION_REQUIRED_CLUSTERS",
    )
    _equal(
        contract["stopping_and_routing"]["automatic_entry_to_c_or_d"],
        False,
        "AUTO_C_D",
    )
    _equal(
        contract["stopping_and_routing"]["automatic_algorithm_change"],
        False,
        "AUTO_ALGORITHM_CHANGE",
    )

    binding_checks: dict[str, str] = {}
    binding_specs = (
        (
            "stage_a_closeout",
            contract["authority_source"]["stage_a_closeout_path"],
            contract["authority_source"]["stage_a_closeout_sha256"],
        ),
        (
            "stage_b_contract_preparation_decision",
            contract["authority_source"][
                "stage_b_contract_preparation_decision_path"
            ],
            contract["authority_source"][
                "stage_b_contract_preparation_decision_sha256"
            ],
        ),
        (
            "r3_implementation_amendment",
            contract["rotation_warp_contract"][
                "r3_implementation_amendment_path"
            ],
            contract["rotation_warp_contract"][
                "r3_implementation_amendment_sha256"
            ],
        ),
        (
            "r3_transport_lock",
            contract["rotation_warp_contract"]["r3_transport_lock_path"],
            contract["rotation_warp_contract"]["r3_transport_lock_sha256"],
        ),
        (
            "rotation_compensation_source",
            contract["rotation_warp_contract"][
                "rotation_compensation_source_path"
            ],
            contract["rotation_warp_contract"][
                "rotation_compensation_source_sha256"
            ],
        ),
    )
    for label, relative, expected_hash in binding_specs:
        path = root / relative
        _require(path.is_file(), f"BINDING_MISSING:{label}")
        actual_hash = sha256_file(path)
        _equal(actual_hash, expected_hash, f"BINDING_HASH:{label}")
        binding_checks[label] = actual_hash

    stage_a_closeout = load_json(
        root / contract["authority_source"]["stage_a_closeout_path"]
    )
    _equal(
        stage_a_closeout.get("terminal"),
        "VALID / STAGE_A_COMPLETE",
        "STAGE_A_AUTHORITY_TERMINAL",
    )
    _equal(
        stage_a_closeout.get("formal_sequences_run"),
        0,
        "STAGE_A_FORMAL_SEQUENCES",
    )
    _equal(
        stage_a_closeout.get("formal_r3_pair_core_calls"),
        0,
        "STAGE_A_FORMAL_CALLS",
    )
    _equal(
        stage_a_closeout.get("formal_authority"),
        "UNCHANGED_NOT_CONSUMED",
        "STAGE_A_FORMAL_AUTHORITY",
    )
    _require(
        not (root / STAGE_B_RESPONSE_ROOT).exists(),
        "STAGE_B_RESPONSE_OR_WORKLOAD_PATH_EXISTS",
    )

    identity_gate = validate_identity_lock(root, contract, identity)
    analytic_gates = analytic_geometry_gates(contract)
    gates: dict[str, Any] = {
        "G01_CONTRACT_AND_AUTHORITY_BINDINGS": {
            "status": "PASS",
            "bindings": binding_checks,
        },
        "G02_IDENTITY_ROLE_AND_DISJOINTNESS": identity_gate,
        **analytic_gates,
        "G11_ROTATION_LEAKAGE_LIMIT_FROZEN": {
            "status": "PASS",
            "required_clusters": 8,
            "absolute_p90_max_per_s": 0.01,
            "three_pair_trigger_density_fixed_required": 0.0,
            "failure_terminal": (
                "B_ORACLE_NOT_EVALUABLE / ROTATION_LEAKAGE_BOUNDARY_FAIL"
            ),
        },
        "G12_SCOPE_AND_RESPONSE_FIREWALL": {
            "status": "PASS",
            "stage_b_response_root_absent": True,
            "formal_sequences_run": 0,
            "formal_r3_pair_core_calls": 0,
            "execution_authorized": False,
        },
    }
    _equal(len(gates), 12, "GATE_COUNT")
    _require(
        all(gate.get("status") == "PASS" for gate in gates.values()),
        "GATE_NOT_PASS",
    )

    receipt = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual.qms_r1."
            "stage_b_translation_depth_oracle_contract_preflight_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "contract_path": str(contract_path.relative_to(root)).replace("\\", "/"),
        "contract_sha256": sha256_file(contract_path),
        "identity_lock_path": str(identity_path.relative_to(root)).replace(
            "\\", "/"
        ),
        "identity_lock_sha256": sha256_file(identity_path),
        "identity_set_sha256": identity["identity_set_sha256"],
        "validator_source_path": VALIDATOR_RELATIVE,
        "validator_source_sha256": sha256_file(validator_path),
        "gates": gates,
        "gate_pass_count": 12,
        "gate_fail_count": 0,
        "stage_b_response_files_read": 0,
        "stage_b_response_files_written": 0,
        "stage_b_workload_calls": 0,
        "r3_source_modified": False,
        "threshold_modified": False,
        "three_pair_modified": False,
        "abstention_modified": False,
        "formal_sequences_run": 0,
        "formal_r3_pair_core_calls": 0,
        "formal_authority_consumed": False,
        "stage_c_authorized": False,
        "stage_d_authorized": False,
        "validated": True,
        "errors": [],
        "terminal": (
            "CONTRACT_PREFLIGHT_PASS / VALID / EXECUTION_NOT_ACTIVATED"
        ),
    }
    decision = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual.qms_r1."
            "stage_b_translation_depth_oracle_execution_activation_decision.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "decision": "HOLD_STAGE_B_EXECUTION_PENDING_SEPARATE_ACTIVATION",
        "contract_preflight_valid": True,
        "geometry_contract_frozen": True,
        "identity_and_role_contract_frozen": True,
        "rotation_leakage_boundary_mandatory": True,
        "stage_b_response_access_authorized": False,
        "stage_b_execution_authorized": False,
        "stage_b_workload_run": False,
        "r3_change_authorized": False,
        "threshold_or_three_pair_change_authorized": False,
        "stage_c_authorized": False,
        "stage_d_authorized": False,
        "formal_480_plus_16_run": False,
        "formal_authority_consumed": False,
        "activation_requirements": [
            "separate explicit Stage B execution authority",
            "materialized scene and target geometry hashes for the exact frozen identities",
            "independent geometry-only receipt bound to those hashes",
            "write-once response output root and unchanged formal firewall",
        ],
        "automatic_activation": False,
        "terminal": "CONTRACT_PREFLIGHT_PASS / EXECUTION_NOT_ACTIVATED",
    }
    return receipt, decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--decision-output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    receipt, decision = validate_and_build(root)
    receipt_path = args.receipt_output or root / RECEIPT_RELATIVE
    decision_path = args.decision_output or root / DECISION_RELATIVE
    write_exclusive(receipt_path, receipt)
    decision["preflight_receipt_path"] = str(
        receipt_path.relative_to(root)
    ).replace("\\", "/")
    decision["preflight_receipt_sha256"] = sha256_file(receipt_path)
    decision["validator_source_sha256"] = receipt["validator_source_sha256"]
    decision["contract_sha256"] = receipt["contract_sha256"]
    decision["identity_lock_sha256"] = receipt["identity_lock_sha256"]
    decision["identity_set_sha256"] = receipt["identity_set_sha256"]
    write_exclusive(decision_path, decision)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "decision": str(decision_path),
                "decision_sha256": sha256_file(decision_path),
                "terminal": decision["terminal"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
