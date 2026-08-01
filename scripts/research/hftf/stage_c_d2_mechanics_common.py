#!/usr/bin/env python3
"""Shared frozen mechanics for HFTF Stage C D2."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from audit_stage_c_f0_1_teacher_opportunity import _probe_passes
from audit_swept_envelope_label_mechanics import (
    _swept_prism_probes_world,
)
from run_geometry_teacher_canary import (
    _obstacle_points_world,
    _pose,
    _theta_edges,
)
from run_stage_c_g0_signed_clearance_mechanics import (
    _signed_clearance_field,
)
from verify_sanpo_pose_geometry_authority import (
    GROUND_PROXY_CLASSES,
    _fit_local_ground_plane,
)


DESIGN_SCHEMA = (
    "blindassist_hftf_stage_c_causal_signed_clearance_transport_d2"
)
CLARIFICATION_SCHEMA = (
    "blindassist_hftf_stage_c_causal_signed_clearance_transport_"
    "clarification_d2_1"
)
CLARIFICATION_STATUS = (
    "FROZEN_AFTER_METADATA_ONLY_COHORT_LOCK_BEFORE_ANY_D2_MEDIA_"
    "POSE_CONTENT_OR_MECHANICS_OUTCOME"
)
CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_mechanics_execution_contract"
)
CONTRACT_STATUS = (
    "FROZEN_AFTER_D2_MEDIA_ACQUISITION_BEFORE_PREPROCESSOR_OR_TRUTH_OUTCOME"
)
SOURCE_INDEX_SCHEMA = (
    "blindassist_hftf_stage_c_d2_per_frame_acquisition_index"
)
SOURCE_INDEX_TERMINAL = "D2_SIX_SOURCE_PER_FRAME_MEDIA_POSE_SLICES_READY"
PREPROCESSOR_COMPLETION_SCHEMA = (
    "blindassist_hftf_stage_c_d2_future_blind_predictions_frozen"
)
PREPROCESSOR_TERMINAL = "D2_FUTURE_BLIND_PREDICTIONS_FROZEN"
RESULT_SCHEMA = "blindassist_hftf_stage_c_d2_transport_effect_result"
SUPPORTED = "CAUSAL_SIGNED_CLEARANCE_TRANSPORT_SUPPORTED_FOR_RGB_STUDENT_PROTOCOL"
STOP = "CAUSAL_SIGNED_CLEARANCE_TRANSPORT_NOT_SUPPORTED_STOP"
NOT_EVALUABLE = (
    "D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT"
)
HEIGHTS = ("body", "head")
HORIZONS = (0.4, 0.8)
ANCHORS = tuple(range(2, 9))
EXPECTED_SOURCE_COUNT = 6
EXPECTED_ANCHOR_COUNT = 42
EXPECTED_HORIZON_RECORD_COUNT = 84
DESIGN_STATUS = "FROZEN_BEFORE_D2_METADATA_SCAN_OR_SOURCE_OUTCOME"
MEDIA_RESULT_SCHEMA = (
    "blindassist_hftf_stage_c_d2_six_source_media_acquisition_result"
)
MEDIA_RESULT_TERMINAL = "D2_SIX_SOURCE_SHORT_PATH_MEDIA_COHORT_ACQUIRED"
COMMON_RELATIVE_PATH = (
    "scripts/research/hftf/stage_c_d2_mechanics_common.py"
)
PREPROCESSOR_RELATIVE_PATH = (
    "scripts/research/hftf/preprocess_stage_c_d2_future_blind.py"
)
EVALUATOR_RELATIVE_PATH = (
    "scripts/research/hftf/evaluate_stage_c_d2_transport_effect.py"
)
TEST_RELATIVE_PATH = (
    "scripts/research/hftf/test_stage_c_d2_mechanics.py"
)
SWEPT_MECHANICS_RELATIVE_PATH = (
    "scripts/research/hftf/audit_swept_envelope_label_mechanics.py"
)
VISIBILITY_MECHANICS_RELATIVE_PATH = (
    "scripts/research/hftf/audit_stage_c_f0_1_teacher_opportunity.py"
)
PREDICTION_RELATIVE_ROOT = (
    "artifacts.local/evidence/hftf/"
    "stage-c-d2-future-blind-predictions-20260802"
)
TRUTH_JOIN_RECEIPT_RELATIVE_PATH = (
    "artifacts.local/evidence/hftf/"
    "stage-c-d2-truth-join-once-20260802.json"
)
EFFECT_RESULT_RELATIVE_PATH = (
    "artifacts.local/evidence/hftf/"
    "stage-c-d2-transport-effect-result-20260802/result.json"
)
PREPROCESSOR_FAILURE_RELATIVE_PATH = (
    PREDICTION_RELATIVE_ROOT + "/failure.json"
)
EFFECT_FAILURE_RELATIVE_PATH = (
    "artifacts.local/evidence/hftf/"
    "stage-c-d2-transport-effect-result-20260802/failure.json"
)
PREPROCESSOR_FAILURE_TERMINAL = (
    "D2_NOT_EVALUABLE_PREPROCESSOR_EXECUTION_FAILED_"
    "NO_RERUN_NO_SOURCE_REPLACEMENT"
)
EVALUATOR_PRETRUTH_FAILURE_TERMINAL = (
    "D2_NOT_EVALUABLE_EVALUATOR_PRETRUTH_VALIDATION_FAILED_"
    "NO_RERUN_NO_SOURCE_REPLACEMENT"
)
TRUTH_JOIN_INTERRUPTED_TERMINAL = (
    "D2_NOT_EVALUABLE_TRUTH_JOIN_INTERRUPTED_"
    "NO_SECOND_JOIN_NO_SOURCE_REPLACEMENT"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def test_definition_count(path: Path) -> int:
    return len(
        re.findall(
            r"^\s+def test_",
            path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


def _require_tracked_clean(path: Path, label: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        relative = path.resolve().relative_to(repo_root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must stay in repository") from error

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(
                f"{label} must be tracked and clean: "
                + (result.stderr.strip() or "git verification failed")
            )
        return result.stdout.strip()

    git("ls-files", "--error-unmatch", "--", relative)
    git("diff", "--quiet", "--", relative)
    git("diff", "--cached", "--quiet", "--", relative)


def require_pushed_state(contract_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        relative = contract_path.resolve().relative_to(repo_root)
    except ValueError as error:
        raise ValueError(
            "D2 mechanics contract must stay in repository"
        ) from error
    if (
        relative.parts[:3] != ("docs", "research", "hftf")
        or contract_path.suffix.lower() != ".json"
    ):
        raise ValueError("D2 mechanics contract must be an HFTF JSON")
    paths = (
        (contract_path, "D2 mechanics execution contract"),
        (
            repo_root / COMMON_RELATIVE_PATH,
            "D2 mechanics common implementation",
        ),
        (
            repo_root / PREPROCESSOR_RELATIVE_PATH,
            "D2 future-blind preprocessor",
        ),
        (
            repo_root / EVALUATOR_RELATIVE_PATH,
            "D2 truth/effect evaluator",
        ),
        (repo_root / TEST_RELATIVE_PATH, "D2 mechanics test"),
        (
            repo_root / SWEPT_MECHANICS_RELATIVE_PATH,
            "D2 swept-probe mechanics",
        ),
        (
            repo_root / VISIBILITY_MECHANICS_RELATIVE_PATH,
            "D2 probe-visibility mechanics",
        ),
    )
    for path, label in paths:
        _require_tracked_clean(path.resolve(), label)

    def rev(name: str) -> str:
        result = subprocess.run(
            ["git", "rev-parse", name],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"cannot resolve Git revision {name}")
        return result.stdout.strip()

    if rev("HEAD") != rev("origin/master"):
        raise ValueError(
            "HEAD must equal origin/master before D2 mechanics execution"
        )


def resolve(repo_root: Path, parent: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] in {
        "artifacts.local",
        "docs",
        "scripts",
    }:
        return (repo_root / path).resolve()
    return (parent / path).resolve()


def bound_json(
    contract_path: Path,
    contract: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[3]
    receipt = contract["parents"][key]
    path = resolve(repo_root, contract_path.parent, str(receipt["path"]))
    if sha256(path) != str(receipt["sha256"]):
        raise ValueError(f"D2 mechanics parent hash mismatch: {key}")
    return path, load_json(path)


def require_implementation(
    contract: dict[str, Any],
    key: str,
    path: Path,
) -> None:
    receipt = contract["implementations"][key]
    expected = path.resolve().relative_to(path.resolve().parents[3])
    if (
        Path(str(receipt["path"])).as_posix() != expected.as_posix()
        or str(receipt["sha256"]) != sha256(path)
    ):
        raise ValueError(f"D2 implementation receipt mismatch: {key}")


def validate_acquired_source_binding(
    qualified: dict[str, Any],
    source: dict[str, Any],
) -> None:
    frames = source.get("frames", [])
    if (
        [int(item["normalized_index"]) for item in frames]
        != list(range(13))
        or [int(item["source_frame_index"]) for item in frames]
        != [
            int(value)
            for value in qualified["selected_source_frames"]
        ]
        or source.get("camera") != qualified.get("camera")
    ):
        raise ValueError(
            "D2 acquired frame or camera binding differs from qualification"
        )


def validate_cross_parent_bindings(
    clarification: dict[str, Any],
    metadata_path: Path,
    g0: dict[str, Any],
    mechanics_path: Path,
) -> None:
    if (
        clarification["parents"]["metadata_qualification_result"][
            "sha256"
        ]
        != sha256(metadata_path)
        or g0["parents"]["swept_envelope_mechanics"]["sha256"]
        != sha256(mechanics_path)
    ):
        raise ValueError("D2 cross-parent hash binding mismatch")


def load_context(
    contract_path: Path,
    implementation_key: str,
    implementation_path: Path,
    *,
    verify_git: bool = True,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
        or contract.get("workflow_profile") != "THESIS_DEVELOPMENT"
    ):
        raise ValueError("D2 mechanics contract identity mismatch")
    require_implementation(
        contract,
        "mechanics_common",
        Path(__file__),
    )
    repo_root = Path(__file__).resolve().parents[3]
    require_implementation(
        contract,
        "swept_probe_mechanics",
        repo_root / SWEPT_MECHANICS_RELATIVE_PATH,
    )
    require_implementation(
        contract,
        "probe_visibility_mechanics",
        repo_root / VISIBILITY_MECHANICS_RELATIVE_PATH,
    )
    for key, path in (
        (
            "future_blind_preprocessor",
            repo_root / PREPROCESSOR_RELATIVE_PATH,
        ),
        (
            "truth_effect_evaluator",
            repo_root / EVALUATOR_RELATIVE_PATH,
        ),
    ):
        require_implementation(contract, key, path)
    expected_implementation_paths = {
        "future_blind_preprocessor": (
            repo_root / PREPROCESSOR_RELATIVE_PATH
        ).resolve(),
        "truth_effect_evaluator": (
            repo_root / EVALUATOR_RELATIVE_PATH
        ).resolve(),
    }
    if (
        implementation_key not in expected_implementation_paths
        or implementation_path.resolve()
        != expected_implementation_paths[implementation_key]
    ):
        raise ValueError("D2 active mechanics implementation mismatch")
    test_path = (repo_root / TEST_RELATIVE_PATH).resolve()
    test_receipt = contract.get("implementation_tests", {}).get(
        "mechanics_test", {}
    )
    actual_test_count = test_definition_count(test_path)
    if (
        Path(str(test_receipt.get("path", ""))).as_posix()
        != TEST_RELATIVE_PATH
        or str(test_receipt.get("sha256", "")) != sha256(test_path)
        or contract.get("implementation_tests", {}).get("test_count")
        != actual_test_count
        or contract.get("implementation_tests", {}).get("tests_passed")
        != actual_test_count
    ):
        raise ValueError("D2 mechanics test receipt mismatch")
    authorization = contract.get("authorization", {})
    required_false = (
        "future_truth_open_authorized_before_completion",
        "second_truth_join_authorized",
        "source_replacement_authorized",
        "same_cohort_retuning_authorized",
        "rgb_student_execution_authorized",
        "reserved_official_test_open_authorized",
        "research_mainline_changed",
        "default_app_changed",
        "android_changed",
        "production_authorized",
        "safety_claim_authorized",
    )
    if (
        authorization.get(
            "future_blind_preprocessor_execution_authorized"
        )
        is not True
        or authorization.get(
            "truth_effect_execution_after_completion_authorized"
        )
        is not True
        or authorization.get(
            "geometry_teacher_is_synthetic_proxy_only"
        )
        is not True
        or any(
            authorization.get(key) is not False
            for key in required_false
        )
    ):
        raise ValueError("D2 mechanics authorization firewall mismatch")
    canonical = contract.get("canonical_artifacts", {})
    expected_canonical = {
        "future_blind_prediction_root": PREDICTION_RELATIVE_ROOT,
        "truth_join_once_receipt": TRUTH_JOIN_RECEIPT_RELATIVE_PATH,
        "effect_result": EFFECT_RESULT_RELATIVE_PATH,
        "preprocessor_failure": PREPROCESSOR_FAILURE_RELATIVE_PATH,
        "truth_effect_failure": EFFECT_FAILURE_RELATIVE_PATH,
    }
    if any(
        Path(str(canonical.get(key, ""))).as_posix() != value
        for key, value in expected_canonical.items()
    ):
        raise ValueError("D2 mechanics canonical artifact path mismatch")
    failure_policy = contract.get("failure_policy", {})
    if (
        failure_policy.get("preprocessor_failure_terminal")
        != PREPROCESSOR_FAILURE_TERMINAL
        or failure_policy.get("evaluator_pretruth_failure_terminal")
        != EVALUATOR_PRETRUTH_FAILURE_TERMINAL
        or failure_policy.get("truth_join_interruption_terminal")
        != TRUTH_JOIN_INTERRUPTED_TERMINAL
        or any(
            failure_policy.get(key) is not False
            for key in (
                "preprocessor_rerun_authorized",
                "second_truth_join_authorized",
                "source_replacement_authorized",
                "same_cohort_retuning_authorized",
            )
        )
        or failure_policy.get("preserve_partial_artifacts_on_failure")
        is not True
    ):
        raise ValueError("D2 mechanics failure policy mismatch")
    design_path, design = bound_json(contract_path, contract, "d2_design")
    clarification_path, clarification = bound_json(
        contract_path,
        contract,
        "d2_1_clarification",
    )
    metadata_path, metadata = bound_json(
        contract_path,
        contract,
        "metadata_qualification_result",
    )
    qualification_path, qualification = bound_json(
        contract_path,
        contract,
        "metadata_qualification",
    )
    g0_path, g0 = bound_json(
        contract_path,
        contract,
        "g0_signed_clearance_definition",
    )
    mechanics_path, mechanics = bound_json(
        contract_path,
        contract,
        "swept_envelope_mechanics",
    )
    index_path, source_index = bound_json(
        contract_path,
        contract,
        "per_frame_acquisition_index",
    )
    media_result_path, media_result = bound_json(
        contract_path,
        contract,
        "media_acquisition_result",
    )
    media_index_receipt = media_result.get("durable_evidence", {}).get(
        "per_frame_acquisition_index", {}
    )
    media_index_path = resolve(
        repo_root,
        media_result_path.parent,
        str(media_index_receipt.get("path", "")),
    )
    tracked_qualification = metadata.get("durable_evidence", {}).get(
        "qualification", {}
    )
    tracked_qualification_path = resolve(
        repo_root,
        metadata_path.parent,
        str(tracked_qualification.get("path", "")),
    )
    validate_cross_parent_bindings(
        clarification,
        metadata_path,
        g0,
        mechanics_path,
    )
    if (
        design.get("schema") != DESIGN_SCHEMA
        or design.get("status") != DESIGN_STATUS
        or clarification.get("schema") != CLARIFICATION_SCHEMA
        or clarification.get("status") != CLARIFICATION_STATUS
        or clarification["parents"]["d2_design"]["sha256"]
        != sha256(design_path)
        or clarification["parents"]["g0_signed_clearance_definition"][
            "sha256"
        ]
        != sha256(g0_path)
        or metadata.get("schema")
        != (
            "blindassist_hftf_stage_c_d2_official_train_"
            "metadata_qualification_result"
        )
        or metadata.get("terminal")
        != "D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED"
        or qualification.get("schema")
        != (
            "blindassist_hftf_stage_c_d2_official_train_"
            "metadata_qualification"
        )
        or qualification.get("terminal")
        != "D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED"
        or tracked_qualification_path != qualification_path
        or str(tracked_qualification.get("sha256", ""))
        != sha256(qualification_path)
        or tracked_qualification.get("required_terminal")
        != "D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED"
        or g0.get("schema")
        != "blindassist_hftf_stage_c_signed_clearance_current_bridge_g0"
        or mechanics.get("schema")
        != (
            "blindassist_hftf_stage_b_swept_envelope_"
            "label_mechanics_canary_d0"
        )
        or mechanics.get("status")
        != "FROZEN_DEVELOPMENT_CANARY_RESULT_NOT_RUN"
        or media_result.get("schema") != MEDIA_RESULT_SCHEMA
        or media_result.get("terminal") != MEDIA_RESULT_TERMINAL
        or media_result.get("authorization", {}).get(
            "freeze_d2_mechanics_execution_contract"
        )
        is not True
        or media_result.get("authorization", {}).get(
            "execute_d2_mechanics_now"
        )
        is not False
        or media_index_path != index_path
        or str(media_index_receipt.get("sha256", "")) != sha256(index_path)
        or media_index_receipt.get("required_terminal")
        != SOURCE_INDEX_TERMINAL
        or source_index.get("schema") != SOURCE_INDEX_SCHEMA
        or source_index.get("terminal") != SOURCE_INDEX_TERMINAL
        or source_index.get("candidate_or_truth_executed") is not False
        or source_index.get(
            "future_blind_preprocessor_execution_authorized"
        )
        is not False
    ):
        raise ValueError("D2 mechanics parent identity mismatch")
    qualified_ids = [
        str(item["session_id"])
        for item in qualification["qualified_parents"]
    ]
    sources = source_index.get("sources", [])
    if (
        len(qualified_ids) != EXPECTED_SOURCE_COUNT
        or [str(item["session_id"]) for item in sources] != qualified_ids
        or [str(item) for item in source_index.get("source_order", [])]
        != qualified_ids
    ):
        raise ValueError("D2 acquired source order differs from qualification")
    for qualified, source in zip(
        qualification["qualified_parents"],
        sources,
    ):
        validate_acquired_source_binding(qualified, source)
    primitive_paths = {
        "exact_g0_signed_clearance_runner": (
            "scripts/research/hftf/run_stage_c_g0_signed_clearance_mechanics.py"
        ),
        "exact_g0_geometry_primitives": (
            "scripts/research/hftf/run_geometry_teacher_canary.py"
        ),
        "frozen_sanpo_pose_and_ground_authority": (
            "scripts/research/hftf/verify_sanpo_pose_geometry_authority.py"
        ),
    }
    for key, relative in primitive_paths.items():
        receipt = clarification["implementation_receipts"][key]
        path = (repo_root / relative).resolve()
        if (
            Path(str(receipt["path"])).as_posix() != relative
            or sha256(path) != str(receipt["sha256"])
        ):
            raise ValueError(f"D2.1 primitive drift: {key}")
    if verify_git:
        require_pushed_state(contract_path)
    return {
        "contract": contract,
        "design": design,
        "clarification": clarification,
        "g0": g0,
        "mechanics": mechanics,
        "source_index": source_index,
        "source_index_path": index_path,
        "sources": sources,
    }


def field_parameters(
    g0: dict[str, Any],
    mechanics: dict[str, Any],
) -> dict[str, Any]:
    field = g0["field_contract"]
    clearance = g0["signed_clearance_contract"]
    view = field["teacher_view_for_every_role"]
    if (
        view
        != {
            "point_sample_stride_xy": 4,
            "point_sample_offset_xy": 2,
            "name": "reference",
        }
        or clearance["order_statistic"] != 2
        or clearance["raw_clearance_clip_m"] != [-0.5, 1.0]
    ):
        raise ValueError("D2 exact G0 field contract drifted")
    return {
        "theta_edges": _theta_edges(field),
        "distance_edges": np.asarray(
            field["distance_edges_m"], dtype=np.float64
        ),
        "height_bands": [
            tuple(
                float(value)
                for value in field["height_bands_m"][height]
            )
            for height in HEIGHTS
        ],
        "widths": np.asarray(
            [
                field["effective_lateral_half_width_m"][height]
                for height in HEIGHTS
            ],
            dtype=np.float64,
        ),
        "stride": 4,
        "offset": 2,
        "excluded_classes": set(
            mechanics["obstacle_support"][
                "excluded_semantic_class_ids"
            ]
        ),
        "dynamic_classes": set(
            mechanics["obstacle_support"][
                "dynamic_provenance_class_ids"
            ]
        ),
        "known_tolerance_m": float(
            mechanics["known_support"]["depth_front_tolerance_m"]
        ),
        "order_statistic": 2,
        "final_edge_atol_m": float(
            clearance["final_distance_edge_isclose"]["atol_m"]
        ),
        "final_edge_rtol": float(
            clearance["final_distance_edge_isclose"]["rtol"]
        ),
        "clip_min_m": -0.5,
        "clip_max_m": 1.0,
    }


def fit_current_plane(
    depth: np.ndarray,
    semantic: np.ndarray,
    binding: dict[str, Any],
    camera: dict[str, Any],
    source_frame_index: int,
) -> dict[str, Any]:
    height, width = depth.shape
    y_grid = np.arange(int(height * 0.55), height, 16)
    x_grid = np.arange(8, width, 16)
    u, v = np.meshgrid(x_grid, y_grid)
    z = depth[v, u]
    valid = (
        np.isfinite(z)
        & (z >= 0.5)
        & (z <= 8.0)
        & np.isin(semantic[v, u], list(GROUND_PROXY_CLASSES))
    )
    z = z[valid].astype(np.float64)
    if z.size < 20:
        raise ValueError("D2 current ground sample is inadequate")
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    points_camera = np.stack(
        (
            (u - float(camera["cx"])) * z / float(camera["fx"]),
            (v - float(camera["cy"])) * z / float(camera["fy"]),
            z,
        ),
        axis=0,
    )
    position, rotation = _pose(binding)
    world = rotation @ points_camera + position[:, None]
    plane = _fit_local_ground_plane(
        world,
        position,
        int(source_frame_index),
    )
    if plane is None:
        raise ValueError("D2 current local ground plane is null")
    return plane


def _project_forward(
    binding: dict[str, Any],
    up: np.ndarray,
) -> np.ndarray:
    _, rotation = _pose(binding)
    forward = rotation @ np.asarray([0.0, 0.0, 1.0])
    forward -= float(forward @ up) * up
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("D2 ground-aligned forward projection is degenerate")
    return forward / norm


def predicted_bases(
    history_binding: dict[str, Any],
    current_binding: dict[str, Any],
    plane: dict[str, Any],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    dict[str, float | list[float]],
]:
    current_position, _ = _pose(current_binding)
    history_position, _ = _pose(history_binding)
    origin = np.asarray(
        plane["camera_ground_projection_m"], dtype=np.float64
    )
    up = np.asarray(plane["normal_toward_camera"], dtype=np.float64)
    up /= np.linalg.norm(up)
    current_forward = _project_forward(current_binding, up)
    history_forward = _project_forward(history_binding, up)
    current_right = np.cross(current_forward, up)
    current_right /= np.linalg.norm(current_right)
    raw = math.atan2(
        float(up @ np.cross(history_forward, current_forward)),
        float(np.clip(history_forward @ current_forward, -1.0, 1.0)),
    )
    yaw_delta = ((raw + math.pi) % (2.0 * math.pi)) - math.pi
    yaw_rate = yaw_delta / 0.4
    velocity = (current_position - history_position) / 0.4
    tangent_velocity = velocity - float(velocity @ up) * up
    current_basis = (origin, current_forward, current_right, up)
    predicted: dict[
        float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for horizon in HORIZONS:
        alpha = yaw_rate * horizon
        forward = (
            current_forward * math.cos(alpha)
            + np.cross(up, current_forward) * math.sin(alpha)
            + up
            * float(up @ current_forward)
            * (1.0 - math.cos(alpha))
        )
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        predicted[horizon] = (
            origin + tangent_velocity * horizon,
            forward,
            right,
            up,
        )
    receipt: dict[str, float | list[float]] = {
        "yaw_delta_rad": yaw_delta,
        "yaw_rate_rad_s": yaw_rate,
        "tangent_translation_velocity_m_s": tangent_velocity.tolist(),
    }
    return current_basis, predicted, receipt


def compute_points(
    depth: np.ndarray,
    semantic: np.ndarray,
    row: dict[str, Any],
    binding: dict[str, Any],
    camera: dict[str, Any],
    parameters: dict[str, Any],
) -> np.ndarray:
    points, _ = _obstacle_points_world(
        Path("."),
        row,
        binding,
        camera,
        stride=parameters["stride"],
        offset=parameters["offset"],
        excluded_classes=parameters["excluded_classes"],
        dynamic_classes=parameters["dynamic_classes"],
        depth_override=depth,
        semantic_override=semantic,
    )
    if points.shape[0] != 3 or not np.isfinite(points).all():
        raise ValueError("D2 obstacle point population is invalid")
    return points


def compute_field(
    points_world: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    parameters: dict[str, Any],
) -> np.ndarray:
    _, clipped, _ = _signed_clearance_field(
        points_world,
        basis,
        parameters["theta_edges"],
        parameters["distance_edges"],
        parameters["height_bands"],
        parameters["widths"],
        order_statistic=parameters["order_statistic"],
        final_edge_atol_m=parameters["final_edge_atol_m"],
        final_edge_rtol=parameters["final_edge_rtol"],
        clip_min_m=parameters["clip_min_m"],
        clip_max_m=parameters["clip_max_m"],
    )
    return clipped.transpose(2, 0, 1)


def compute_known(
    depth: np.ndarray,
    semantic: np.ndarray,
    row: dict[str, Any],
    binding: dict[str, Any],
    camera: dict[str, Any],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    probes = _swept_prism_probes_world(
        basis,
        parameters["theta_edges"],
        parameters["distance_edges"],
        parameters["height_bands"],
        parameters["widths"],
    )
    passing = _probe_passes(
        probes,
        row,
        binding,
        camera,
        depth,
        semantic,
        parameters["known_tolerance_m"],
    )
    counts = passing.sum(axis=1).reshape((6, 6, 2)).transpose(2, 0, 1)
    return counts, counts >= 5


def nullable_field(
    known: np.ndarray,
    clearance: np.ndarray,
) -> list[list[list[float | None]]]:
    if known.shape != (2, 6, 6) or clearance.shape != (2, 6, 6):
        raise ValueError("D2 field shape must be 2x6x6")
    output: list[list[list[float | None]]] = []
    for height in range(2):
        rows: list[list[float | None]] = []
        for theta in range(6):
            rows.append(
                [
                    float(clearance[height, theta, distance])
                    if bool(known[height, theta, distance])
                    else None
                    for distance in range(6)
                ]
            )
        output.append(rows)
    return output


def basis_receipt(
    basis: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, list[float]]:
    return {
        "origin_m": basis[0].tolist(),
        "forward": basis[1].tolist(),
        "right": basis[2].tolist(),
        "up": basis[3].tolist(),
    }


def arrays_from_arm(
    arm: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    known = np.asarray(arm["known"], dtype=bool)
    counts = np.asarray(arm["probe_pass_counts"])
    nullable = np.asarray(arm["clearance_m"], dtype=object)
    if (
        known.shape != (2, 6, 6)
        or counts.shape != (2, 6, 6)
        or nullable.shape != (2, 6, 6)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or np.any(counts > 9)
        or not np.array_equal(known, counts >= 5)
    ):
        raise ValueError("D2 serialized arm shape mismatch")
    clearance = np.full((2, 6, 6), np.nan, dtype=np.float64)
    for index in np.ndindex((2, 6, 6)):
        if known[index]:
            if nullable[index] is None:
                raise ValueError("D2 known cell has null clearance")
            clearance[index] = float(nullable[index])
            if not math.isfinite(clearance[index]):
                raise ValueError("D2 known clearance is non-finite")
        elif nullable[index] is not None:
            raise ValueError("D2 UNKNOWN cell became numeric SAFE/risk")
    return known, clearance
