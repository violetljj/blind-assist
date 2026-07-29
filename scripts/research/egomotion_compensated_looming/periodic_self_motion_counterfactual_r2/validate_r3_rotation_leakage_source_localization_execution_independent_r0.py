"""Independent R3 rotation-leakage localization execution validator.

This validator deliberately does not import the localization runner, the R3
producer, tracking/evaluation code, or the local-fit implementation.  It
reconstructs the fixed-grid source-error and final-consensus reductions from
write-once numeric primitives, applies the frozen coverage and routing rules,
and is the sole writer of the execution analysis and route decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
import uuid

import cv2
import numpy as np


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
    "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0"
)
RUNNER_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
    "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_RUNNER_R0"
)
PAIR_COUNT = 601
CLUSTER_COUNT = 8
PILOT_CLUSTER_COUNT = 4
WORKERS = 4
GIB = 1024**3
LAUNCH_REFILL_BYTES = 6 * GIB
IN_FLIGHT_FLOOR_BYTES = 4 * GIB
GRID_CELLS = 9
COVERAGE_FRACTION = 0.75
COVERAGE_MINIMUM = 451
WARP_OVERLAP_FLOOR = 0.75
LEAKAGE_THRESHOLD_PER_S = 0.01
GRAY_RESIDUAL_THRESHOLD = 1.0 / 255.0
GEOMETRY_ROUNDTRIP_MAX_PX = 1e-9
WARP_ROUNDTRIP_P99_MAX_PX = 1e-6
WIDTH = 360
HEIGHT = 640
ACTIVE_VALIDATION_CLAIMS: dict[Path, str] = {}
NUMERIC_ATOL = 1e-12
CLUSTER_METRIC_CLAIM_CEILING = (
    "CONTROLLED_GENERATOR_INTERNAL_MECHANISM_LOCALIZATION_ONLY"
)
CLUSTER_METRIC_AMBIGUITY_RULES = {
    "boundary_definition": (
        "common AND erode(previous_valid,21x21) AND "
        "erode(warped_current_valid,21x21)"
    ),
    "cluster_reduction": "pair scalar then Hyndman-Fan type-7 percentile",
    "formal_minimum_evaluable_pairs": COVERAGE_MINIMUM,
    "flow_gate": "accepted AND final-managed source-error P90 <= 0.01/s",
    "coordinate_roundtrip_population": "all common-valid pixel centers",
    "local_fit_mismatch": "NOT_EVALUABLE",
}

CONTRACT_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_"
    "CONTRACT_2026-07-29.json"
)
IDENTITY_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_"
    "IDENTITY_INPUT_LOCK_2026-07-29.json"
)
PREFLIGHT_RECEIPT_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_"
    "INDEPENDENT_RECEIPT_2026-07-29.json"
)
ACTIVATION_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_EXECUTION_ACTIVATION_DECISION_"
    "R1_2026-07-29.json"
)
PROTOCOL_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/configs/"
    "phase_a_synthetic_signal_audit_r0.json"
)
STAGE_B_IDENTITY_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0_"
    "IDENTITY_LOCK_2026-07-29.json"
)
STAGE_B_GEOMETRY_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_stage_b_translation_depth_oracle_object_approach_r0/control/"
    "geometry_manifest.json"
)
TRAJECTORY_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
)
FORMAL_ROOT_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_r3_rotation_leakage_source_localization_r0"
)
PILOT_PARENT_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_r3_rotation_leakage_source_localization_pilots"
)
GENERATOR_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/generator_geometry.py"
)
RENDERER_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "material_residual_contraction_r1.py"
)
RUNNER_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "r3_rotation_leakage_source_localization_r0.py"
)
TEST_SUITE_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "tests_periodic_self_motion_counterfactual_r2/"
    "test_r3_rotation_leakage_source_localization_execution_r0.py"
)
SPEC_RELATIVE = (
    "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
    "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_EXECUTABLE_SPEC_R1_"
    "2026-07-29.json"
)
EXECUTABLE_SPEC_STATUS = "FROZEN_IMPLEMENTATION_READY"
IMPLEMENTATION_BINDINGS_STATUS = "FROZEN_IMPLEMENTATIONS_PASS"
IMPLEMENTATION_READY_SCHEMA = (
    "rcle.r3_rotation_leakage_localization."
    "implementation_ready_receipt.v1"
)
IMPLEMENTATION_READY_TERMINAL = (
    "IMPLEMENTATION_READY_PASS / "
    "EXISTING_ACTIVATION_REQUIRED_FOR_FORMAL_EXECUTION"
)
PILOT_EQUIVALENCE_SCHEMA = (
    "rcle.r3_rotation_leakage_localization."
    "pilot_w1_w4_equivalence_receipt.v1"
)
PILOT_EQUIVALENCE_TERMINAL = (
    "PILOT_W1_W4_EQUIVALENCE_PASS / FORMAL_NOT_CONSUMED"
)
PREFORMAL_TEST_SCHEMA = (
    "rcle.r3_rotation_leakage_localization."
    "preformal_test_receipt.v1"
)
PREFORMAL_TEST_TERMINAL = (
    "PREFORMAL_UNIT_MUTATION_AND_TRANSPARENCY_PASS / "
    "FORMAL_NOT_CONSUMED"
)
HOST_PREFLIGHT_SCHEMA = "blindassist.host_research_preflight.v1"
HOST_PREFLIGHT_TASK_ID = "RCLE_R3_ROTATION_LEAKAGE_LOCALIZATION_R0"
GUARDED_HOST_LAUNCHER_RELATIVE = "scripts/run_guarded_host_research.ps1"
SPEC_SCHEMA = (
    "rcle.r3_rotation_leakage_source_localization.executable_spec.v1"
)
SPEC_TERMINAL = (
    "FROZEN_IMPLEMENTATION_READY / PILOT_W1_W4_EQUIVALENCE_PASS / "
    "EXISTING_ACTIVATION_REQUIRED_FOR_FORMAL_EXECUTION"
)
SPEC_CLAIM_CEILING = (
    "CONTROLLED_GENERATOR_INTERNAL_MECHANISM_LOCALIZATION_ONLY; "
    "FIRST_VISIBLE_LAYER_IS_NOT_CAUSAL_IDENTIFICATION; NO R3 REPAIR, "
    "PERFORMANCE, REAL_SCENE, ANDROID, PRODUCT, DANGER OR SAFETY CLAIM"
)
SPEC_REQUIRED_STATUS = (
    "FROZEN_IMPLEMENTATION_READY / FROZEN_IMPLEMENTATIONS_PASS / "
    "PILOT_W1_W4_EQUIVALENCE_PASS / PREFORMAL_TEST_PASS / "
    "QUALIFIED_HOST_PREFLIGHT / BOUND_IMPLEMENTATION_READY_RECEIPT / "
    "EXISTING_R1_ACTIVATION"
)
SPEC_DIRECT_BINDING_PATHS = {
    "frozen_contract": CONTRACT_RELATIVE,
    "identity_input_lock": IDENTITY_RELATIVE,
    "independent_preflight_receipt": PREFLIGHT_RECEIPT_RELATIVE,
    "execution_activation_r1": ACTIVATION_RELATIVE,
    "stage_b_closeout": (
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_"
        "TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_INDEPENDENT_CLOSEOUT_"
        "RECEIPT_R1_2026-07-29.json"
    ),
    "r3_transport_lock": (
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_R3_TRANSPORT_"
        "EQUIVALENCE_LOCK_R0_2026-07-29.json"
    ),
    "memory_gate_amendment": (
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_MEMORY_GATE_"
        "6GIB_SUCCESSOR_AMENDMENT_R0_2026-07-29.json"
    ),
    "numeric_representation_amendment": (
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_"
        "INDEPENDENT_VALIDATOR_NUMERIC_REPRESENTATION_AMENDMENT_R0_"
        "2026-07-29.json"
    ),
}
SPEC_FROZEN_DEPENDENCY_PATHS = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "stage_b_translation_depth_oracle_object_approach_r0.py",
    GENERATOR_RELATIVE,
    RENDERER_RELATIVE,
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/p3_transport_r0.py",
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/p3_runtime_preflight_r0.py",
    "scripts/research/egomotion_compensated_looming/"
    "rgb_algorithm_development_canary_cid_sims_r0/producer.py",
    "scripts/research/egomotion_compensated_looming/"
    "rcle_minimal/rotation_compensation.py",
    "scripts/research/egomotion_compensated_looming/"
    "rcle_minimal_r1/sparse_flow.py",
    "scripts/research/egomotion_compensated_looming/"
    "rcle_minimal_r1/local_expansion.py",
    "scripts/research/egomotion_compensated_looming/"
    "rcle_observable_support_r0/__init__.py",
    "scripts/research/egomotion_compensated_looming/"
    "rcle_observable_support_r0/evaluation.py",
    "scripts/research/egomotion_compensated_looming/configs/"
    "phase_a_synthetic_signal_audit_r0.json",
)
ACTIVATION_BINDING_PATHS = {
    "FROZEN_CONTRACT": CONTRACT_RELATIVE,
    "IDENTITY_INPUT_LOCK": IDENTITY_RELATIVE,
    "INDEPENDENT_PREFLIGHT_RECEIPT": PREFLIGHT_RECEIPT_RELATIVE,
    "INDEPENDENT_PREFLIGHT_VALIDATOR": (
        "scripts/research/egomotion_compensated_looming/"
        "periodic_self_motion_counterfactual_r2/"
        "validate_r3_rotation_leakage_source_localization_contract_"
        "preflight_r0.py"
    ),
    "HISTORICAL_PREFLIGHT_HOLD_DECISION": (
        "docs/research/rcle/"
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
        "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0_"
        "EXECUTION_ACTIVATION_DECISION_2026-07-29.json"
    ),
}
CONTRACT_FROZEN_BINDING_PATHS = {
    "identity_input_lock": IDENTITY_RELATIVE,
    "stage_b_closeout": SPEC_DIRECT_BINDING_PATHS["stage_b_closeout"],
    "r3_transport_lock": SPEC_DIRECT_BINDING_PATHS["r3_transport_lock"],
    "r3_pair_core": SPEC_FROZEN_DEPENDENCY_PATHS[5],
    "rotation_warp": SPEC_FROZEN_DEPENDENCY_PATHS[6],
    "sparse_lk": SPEC_FROZEN_DEPENDENCY_PATHS[7],
    "local_affine": SPEC_FROZEN_DEPENDENCY_PATHS[8],
    "support_manager": SPEC_FROZEN_DEPENDENCY_PATHS[9],
    "support_manager_evaluation": SPEC_FROZEN_DEPENDENCY_PATHS[10],
    "r3_parameters": SPEC_FROZEN_DEPENDENCY_PATHS[11],
    "memory_gate_amendment": SPEC_DIRECT_BINDING_PATHS[
        "memory_gate_amendment"
    ],
}
EXPECTED_LK_CONTRACT = {
    "grid": "3x3",
    "max_features_per_cell": 80,
    "quality_level": 0.01,
    "min_distance_px": 5.0,
    "block_size": 7,
    "window_size_px": [21, 21],
    "max_pyramid_level": 3,
    "termination_count": 30,
    "termination_epsilon": 0.01,
    "forward_backward_max_error_px": 1.0,
}
EXPECTED_LOCAL_AFFINE_CONTRACT = {
    "grid": "3x3",
    "minimum_tracks_per_cell": 12,
    "minimum_hull_fraction": 0.1,
    "maximum_design_condition_number": 1000.0,
    "maximum_median_fit_residual_px_per_frame": 0.75,
    "minimum_common_evaluable_cells": 5,
}

RUN_RECEIPT_SCHEMAS = {
    "rcle.r3_rotation_leakage_source_localization.run_receipt.v1",
    "rcle.r3_rotation_leakage_source_localization.execution_receipt.v1",
    "rcle.r3_rotation_leakage_localization.run.v1",
}
CLUSTER_RECEIPT_SCHEMAS = {
    "rcle.r3_rotation_leakage_source_localization.cluster_receipt.v1",
    "rcle.r3_rotation_leakage_localization.cluster_receipt.v1",
}
CLUSTER_METRIC_SCHEMAS = {
    "rcle.r3_rotation_leakage_source_localization.cluster_metrics.v1",
    "rcle.r3_rotation_leakage_localization.cluster.v1",
}

NPZ_KEYS = frozenset(
    {
        "initial_offsets",
        "initial_previous",
        "initial_source_offsets",
        "initial_source_index",
        "initial_source_depth_m",
        "initial_source_object_id",
        "initial_source_world",
        "initial_source_current_pixel",
        "initial_source_aligned",
        "initial_source_valid",
        "lk_forward_offsets",
        "lk_forward_source_index",
        "lk_forward_points",
        "lk_forward_available",
        "lk_backward_offsets",
        "lk_backward_source_index",
        "lk_backward_points",
        "lk_backward_level",
        "lk_backward_error",
        "lk_backward_available",
        "lk_fb_pass",
        "lk_mask_pass",
        "lk_accepted",
        "lk_rejection_reason",
        "accepted_offsets",
        "accepted_previous",
        "accepted_current",
        "accepted_fb_error",
        "accepted_source_offsets",
        "accepted_source_index",
        "accepted_source_aligned",
        "accepted_source_valid",
        "managed_offsets",
        "managed_previous",
        "managed_current",
        "managed_fb_error",
        "managed_source_offsets",
        "managed_source_index",
        "managed_source_aligned",
        "managed_source_valid",
        "managed_path_class",
        "fit_offsets",
        "fit_previous",
        "fit_current",
        "fit_fb_error",
        "fit_group_path",
        "activated_offsets",
        "activated_cell_index",
        "merge_candidate_offsets",
        "merge_candidate_previous",
        "merge_candidate_current",
        "merge_candidate_fb_error",
        "merge_candidate_path_class",
        "merge_candidate_selected",
        "merge_group_path",
        "consensus_offsets",
        "consensus_previous",
        "consensus_current",
        "consensus_pair_cell",
        "consensus_path",
    }
)

FIT_GROUP_PATHS = (
    "RAW_INITIAL",
    "COMPENSATED_INITIAL",
    "RAW_MANAGED",
    "COMPENSATED_MANAGED",
)
MERGE_GROUP_PATHS = ("RAW_MANAGED", "COMPENSATED_MANAGED")

ROUTES = frozenset(
    {
        "LEAKAGE_ALREADY_PRESENT_IN_INPUT_GEOMETRY",
        "LEAKAGE_FIRST_VISIBLE_AT_WARP",
        "LEAKAGE_FIRST_VISIBLE_AT_MASK_BOUNDARY",
        "LEAKAGE_FIRST_VISIBLE_AT_FLOW",
        "LEAKAGE_FIRST_VISIBLE_AT_LOCAL_FIT",
        "MULTIPLE_SOURCES_NOT_SEPARABLE",
        "NOT_EVALUABLE",
    }
)


class InvalidExecution(RuntimeError):
    """The sealed execution cannot support a valid independent decision."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda item: (
                item.tolist()
                if isinstance(item, np.ndarray)
                else item.item()
            ),
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise InvalidExecution(f"NONFINITE_JSON_CONSTANT:{value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidExecution(f"DUPLICATE_JSON_KEY:{key}")
        result[key] = value
    return result


def load_json(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    raw = path.read_bytes()
    require(not raw.startswith(b"\xef\xbb\xbf"), f"JSON_BOM:{path.name}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidExecution(f"JSON_PARSE:{path.name}:{error}") from error
    require(isinstance(value, dict), f"JSON_OBJECT:{path.name}")
    require_finite_tree(value, f"JSON_FINITE:{path.name}")
    if canonical:
        require(raw == canonical_bytes(value), f"JSON_NOT_CANONICAL:{path.name}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    require(not raw.startswith(b"\xef\xbb\xbf"), f"JSONL_BOM:{path.name}")
    require(raw.endswith(b"\n"), f"JSONL_FINAL_LF:{path.name}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        require(line, f"JSONL_BLANK_LINE:{path.name}:{line_number}")
        try:
            row = json.loads(
                line.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidExecution(
                f"JSONL_PARSE:{path.name}:{line_number}:{error}"
            ) from error
        require(isinstance(row, dict), f"JSONL_OBJECT:{line_number}")
        require_finite_tree(row, f"JSONL_FINITE:{line_number}")
        require(
            line + b"\n" == canonical_bytes(row),
            f"JSONL_NOT_CANONICAL:{line_number}",
        )
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace an operational sidecar, tolerating brief readers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        os.fspath(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    last_error: OSError | None = None
    for _ in range(40):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.05)
    try:
        temporary.unlink(missing_ok=True)
    finally:
        raise InvalidExecution(
            f"VALIDATION_PROGRESS_REPLACE:{last_error}"
        ) from last_error


def write_validation_progress(
    response_root: Path,
    *,
    validation_claim_id: str,
    completed_units: int,
    total_units: int,
    started: float,
    status: str,
    initial: bool = False,
) -> None:
    progress_path = response_root / "validation_progress.json"
    require(
        isinstance(validation_claim_id, str)
        and len(validation_claim_id) == 32
        and all(character in "0123456789abcdef" for character in validation_claim_id),
        "VALIDATION_CLAIM_ID",
    )
    if initial:
        sample_index = 0
    else:
        previous = load_json(progress_path, canonical=True)
        require(
            previous.get("validation_claim_id") == validation_claim_id,
            "VALIDATION_PROGRESS_OWNERSHIP",
        )
        sample_index = int(previous.get("sample_index", -1)) + 1
    elapsed = max(time.perf_counter() - started, 1e-9)
    throughput = completed_units / elapsed
    eta = (
        (total_units - completed_units) / throughput
        if throughput > 0.0
        else None
    )
    record = {
        "schema": (
            "rcle.r3_rotation_leakage_localization."
            "validation_progress.v1"
        ),
        "phase": "independent_validation",
        "validation_claim_id": validation_claim_id,
        "sample_index": sample_index,
        "completed_units": int(completed_units),
        "total_units": int(total_units),
        "throughput": float(throughput),
        "eta_seconds": eta,
        "last_progress_at": utc_now(),
        "status": status,
    }
    if initial:
        write_exclusive(progress_path, record)
    else:
        write_atomic(progress_path, record)


def require(condition: Any, label: str) -> None:
    if not bool(condition):
        raise InvalidExecution(label)


def require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            require(isinstance(key, str), f"{label}:NONSTRING_KEY")
            require_finite_tree(item, f"{label}:{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_tree(item, f"{label}:{index}")
    elif isinstance(value, float):
        require(math.isfinite(value), label)


def require_schema(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    require(value.get("schema") in allowed, f"{label}:SCHEMA")


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        label,
    )
    return float(value)


def optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return finite_number(value, label)


def close(
    recorded: float | None,
    recomputed: float | None,
    label: str,
    *,
    atol: float = NUMERIC_ATOL,
) -> None:
    if recorded is None or recomputed is None:
        require(recorded is None and recomputed is None, label)
        return
    left = finite_number(recorded, label)
    right = finite_number(recomputed, label)
    require(abs(left - right) <= atol, label)


def hf7(values: Sequence[float], probability: float) -> float | None:
    """Hyndman-Fan type-7 quantile, implemented without runner code."""

    require(0.0 <= probability <= 1.0, "HF7_PROBABILITY")
    if not values:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    require(
        ordered.ndim == 1 and np.isfinite(ordered).all(),
        "HF7_NONFINITE",
    )
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * probability
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    weight = index - lower
    return float(
        ordered[lower] + weight * (ordered[upper] - ordered[lower])
    )


def cell_bounds(index: int) -> tuple[int, int, int, int]:
    require(0 <= index < GRID_CELLS, "CELL_INDEX")
    row, column = divmod(index, 3)
    return (
        int(round(column * WIDTH / 3)),
        int(round(row * HEIGHT / 3)),
        int(round((column + 1) * WIDTH / 3)),
        int(round((row + 1) * HEIGHT / 3)),
    )


def _cell_fit(
    previous: np.ndarray,
    current: np.ndarray,
    dt_s: float,
    cell_index: int,
) -> dict[str, Any]:
    """Recompute the frozen deterministic float64 OLS audit for one cell."""

    require(dt_s > 0.0 and math.isfinite(dt_s), "FIT_DT")
    require(
        previous.ndim == 2
        and current.ndim == 2
        and previous.shape == current.shape
        and previous.shape[1:] == (2,),
        "FIT_SHAPE",
    )
    require(
        np.isfinite(previous).all() and np.isfinite(current).all(),
        "FIT_NONFINITE",
    )
    x0, y0, x1, y1 = cell_bounds(cell_index)
    selected = (
        (previous[:, 0] >= x0)
        & (previous[:, 0] < x1)
        & (previous[:, 1] >= y0)
        & (previous[:, 1] < y1)
    )
    points = previous[selected].astype(np.float64, copy=False)
    endpoints = current[selected].astype(np.float64, copy=False)
    support = int(len(points))
    output: dict[str, Any] = {
        "cell_index": cell_index,
        "support_count": support,
        "hull_fraction": 0.0,
        "condition_number": None,
        "median_fit_residual_px_per_frame": None,
        "coefficients": None,
        "expansion_per_s": None,
        "evaluable": False,
        "abstention_reason": None,
    }
    if support < 12:
        output["abstention_reason"] = "SUPPORT_BELOW_12"
        return output
    hull = cv2.convexHull(points.astype(np.float32))
    hull_fraction = float(cv2.contourArea(hull)) / max(
        float((x1 - x0) * (y1 - y0)), 1.0
    )
    output["hull_fraction"] = hull_fraction
    if hull_fraction < 0.1:
        output["abstention_reason"] = "HULL_FRACTION_BELOW_0_1"
        return output
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    half_width = max(0.5 * (x1 - x0), 1.0)
    half_height = max(0.5 * (y1 - y0), 1.0)
    design = np.column_stack(
        (
            (points[:, 0] - center_x) / half_width,
            (points[:, 1] - center_y) / half_height,
            np.ones(support, dtype=np.float64),
        )
    )
    condition = float(np.linalg.cond(design))
    output["condition_number"] = condition
    if not math.isfinite(condition) or condition > 1000.0:
        output["abstention_reason"] = "CONDITION_ABOVE_1000"
        return output
    velocity = (endpoints - points) / dt_s
    coefficients, _, rank, _ = np.linalg.lstsq(design, velocity, rcond=None)
    require(rank == 3, "FIT_RANK")
    residual = float(
        np.median(np.linalg.norm(design @ coefficients - velocity, axis=1))
        * dt_s
    )
    output["median_fit_residual_px_per_frame"] = residual
    if residual > 0.75:
        output["abstention_reason"] = "RESIDUAL_ABOVE_0_75"
        return output
    expansion = float(
        0.5
        * (
            coefficients[0, 0] / half_width
            + coefficients[1, 1] / half_height
        )
    )
    require(math.isfinite(expansion), "FIT_EXPANSION_NONFINITE")
    output.update(
        {
            "coefficients": coefficients.tolist(),
            "expansion_per_s": expansion,
            "evaluable": True,
            "abstention_reason": None,
        }
    )
    return output


def _native_consensus_mask(
    previous: np.ndarray, current: np.ndarray
) -> np.ndarray:
    """Recompute the frozen native RANSAC mask without importing local-fit."""

    points = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
    endpoints = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    require(
        points.shape == endpoints.shape
        and np.isfinite(points).all()
        and np.isfinite(endpoints).all(),
        "NATIVE_CONSENSUS_INPUT",
    )
    if len(points) < 3:
        return np.zeros(len(points), dtype=bool)
    _, inliers = cv2.estimateAffine2D(
        points.astype(np.float32),
        endpoints.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=0.75,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if inliers is None:
        return np.zeros(len(points), dtype=bool)
    return np.ascontiguousarray(inliers.reshape(-1) > 0)


def _native_cell_fit(
    previous: np.ndarray,
    current: np.ndarray,
    dt_s: float,
    cell_index: int,
) -> dict[str, Any]:
    """Independently reproduce one native cell, including exact consensus."""

    points = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
    endpoints = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    require(
        points.shape == endpoints.shape
        and np.isfinite(points).all()
        and np.isfinite(endpoints).all(),
        "NATIVE_CELL_INPUT",
    )
    x0, y0, x1, y1 = cell_bounds(cell_index)
    in_cell = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    tracked_previous = points[in_cell]
    tracked_current = endpoints[in_cell]
    consensus = _native_consensus_mask(tracked_previous, tracked_current)
    consensus_previous = tracked_previous[consensus]
    consensus_current = tracked_current[consensus]
    audit = _cell_fit(
        consensus_previous,
        consensus_current,
        dt_s,
        cell_index,
    )
    reason_map = {
        "SUPPORT_BELOW_12": "LK_TRACK_SUPPORT_BELOW_12",
        "HULL_FRACTION_BELOW_0_1": "TRACK_HULL_COVERAGE_BELOW_0_10",
        "CONDITION_ABOVE_1000": "AFFINE_DESIGN_CONDITION_ABOVE_1000",
        "RESIDUAL_ABOVE_0_75": (
            "AFFINE_MEDIAN_RESIDUAL_ABOVE_0_75PX_PER_FRAME"
        ),
        None: None,
    }
    require(
        audit["abstention_reason"] in reason_map,
        "NATIVE_CELL_ABSTENTION_REASON",
    )
    confidence = 0.0
    if audit["evaluable"]:
        support = int(audit["support_count"])
        residual = float(audit["median_fit_residual_px_per_frame"])
        hull = float(audit["hull_fraction"])
        confidence = (
            min(1.0, support / 24.0)
            * max(0.0, 1.0 - residual / 0.75)
            * min(1.0, hull / 0.20)
        )
    audit.update(
        {
            "tracked_support_count": int(len(tracked_previous)),
            "region": [x0, y0, x1, y1],
            "confidence": float(confidence),
            "abstention_reason": reason_map[audit["abstention_reason"]],
            "consensus_previous": np.ascontiguousarray(
                consensus_previous.astype(np.float32)
            ),
            "consensus_current": np.ascontiguousarray(
                consensus_current.astype(np.float32)
            ),
        }
    )
    return audit


def _consensus_ols_audit(
    previous: np.ndarray,
    current: np.ndarray,
    dt_s: float,
    cell_index: int,
) -> dict[str, Any]:
    points = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
    endpoints = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    require(
        points.shape == endpoints.shape
        and np.isfinite(points).all()
        and np.isfinite(endpoints).all(),
        "CONSENSUS_AUDIT_INPUT",
    )
    output = {
        "coefficients": None,
        "condition_number": None,
        "median_fit_residual_px_per_frame": None,
        "expansion_per_s": None,
    }
    if len(points) < 3:
        return output
    x0, y0, x1, y1 = cell_bounds(cell_index)
    center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half_width = max(0.5 * (x1 - x0), 1.0)
    half_height = max(0.5 * (y1 - y0), 1.0)
    design = np.column_stack(
        (
            (points[:, 0] - center_x) / half_width,
            (points[:, 1] - center_y) / half_height,
            np.ones(len(points), dtype=np.float64),
        )
    )
    velocity = (endpoints - points) / dt_s
    coefficients, _, _, _ = np.linalg.lstsq(design, velocity, rcond=None)
    residual = float(
        np.median(np.linalg.norm(design @ coefficients - velocity, axis=1))
        * dt_s
    )
    output.update(
        {
            "coefficients": coefficients.tolist(),
            "condition_number": float(np.linalg.cond(design)),
            "median_fit_residual_px_per_frame": residual,
            "expansion_per_s": float(
                0.5
                * (
                    coefficients[0, 0] / half_width
                    + coefficients[1, 1] / half_height
                )
            ),
        }
    )
    require_finite_tree(output, "CONSENSUS_AUDIT_FINITE")
    return output


def fit_pair(
    previous: np.ndarray,
    current: np.ndarray,
    dt_s: float,
) -> dict[str, Any]:
    cells = [
        _cell_fit(previous, current, dt_s, cell_index)
        for cell_index in range(GRID_CELLS)
    ]
    evaluable = [
        float(cell["expansion_per_s"])
        for cell in cells
        if cell["evaluable"]
    ]
    pair_evaluable = len(evaluable) >= 5
    return {
        "evaluable": pair_evaluable,
        "evaluable_cell_count": len(evaluable),
        "common_cell_indices": [
            int(cell["cell_index"]) for cell in cells if cell["evaluable"]
        ],
        "signed_per_s": (
            float(np.median(np.asarray(evaluable, dtype=np.float64)))
            if pair_evaluable
            else None
        ),
        "absolute_per_s": (
            float(
                np.median(
                    np.abs(np.asarray(evaluable, dtype=np.float64))
                )
            )
            if pair_evaluable
            else None
        ),
        "cells": cells,
    }


def _source_error_fit(
    previous: np.ndarray,
    observed_current: np.ndarray,
    source_aligned: np.ndarray,
    source_valid: np.ndarray,
    dt_s: float,
) -> dict[str, Any]:
    require(
        len(previous)
        == len(observed_current)
        == len(source_aligned)
        == len(source_valid),
        "SOURCE_ERROR_LENGTH",
    )
    selected = source_valid.astype(bool, copy=False)
    error_endpoint = (
        previous[selected].astype(np.float64, copy=False)
        + observed_current[selected].astype(np.float64, copy=False)
        - source_aligned[selected].astype(np.float64, copy=False)
    )
    result = fit_pair(
        previous[selected].astype(np.float64, copy=False),
        error_endpoint,
        dt_s,
    )
    result["source_valid_count"] = int(np.count_nonzero(selected))
    return result


def boundary_masks(
    previous_valid: np.ndarray, warped_current_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boundary, interior, common using the frozen 21x21 footprint."""

    previous = np.asarray(previous_valid) > 0
    current = np.asarray(warped_current_valid) > 0
    require(
        previous.ndim == 2 and previous.shape == current.shape,
        "BOUNDARY_MASK_SHAPE",
    )
    common = previous & current
    kernel = np.ones((21, 21), dtype=np.uint8)
    previous_eroded = cv2.erode(
        previous.astype(np.uint8),
        kernel,
        anchor=(10, 10),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    current_eroded = cv2.erode(
        current.astype(np.uint8),
        kernel,
        anchor=(10, 10),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    interior = common & previous_eroded & current_eroded
    boundary = common & ~interior
    return (
        np.ascontiguousarray(boundary),
        np.ascontiguousarray(interior),
        np.ascontiguousarray(common),
    )


def _sample_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    if not len(points):
        return result
    require(
        points.ndim == 2
        and points.shape[1] == 2
        and np.isfinite(points).all(),
        "MASK_SAMPLE_POINTS",
    )
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    inside = (
        (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    )
    indices = np.flatnonzero(inside)
    result[indices] = np.asarray(mask)[y[indices], x[indices]] > 0
    return result


def _homography_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points64 = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack(
        (points64, np.ones(len(points64), dtype=np.float64))
    )
    mapped = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    require(
        np.isfinite(mapped).all()
        and np.all(np.abs(mapped[:, 2]) > 1e-15),
        "HOMOGRAPHY_POINTS",
    )
    return np.ascontiguousarray(mapped[:, :2] / mapped[:, 2, None])


def _rotation_homography(
    intrinsic: np.ndarray,
    previous_world_from_camera: np.ndarray,
    current_world_from_camera: np.ndarray,
) -> np.ndarray:
    intrinsic64 = np.asarray(intrinsic, dtype=np.float64)
    previous = np.asarray(previous_world_from_camera, dtype=np.float64)
    current = np.asarray(current_world_from_camera, dtype=np.float64)
    require(
        intrinsic64.shape == (3, 3)
        and previous.shape == (3, 3)
        and current.shape == (3, 3)
        and np.isfinite(intrinsic64).all()
        and np.isfinite(previous).all()
        and np.isfinite(current).all(),
        "HOMOGRAPHY_INPUT",
    )
    homography = (
        intrinsic64 @ (current.T @ previous) @ np.linalg.inv(intrinsic64)
    )
    require(np.isfinite(homography).all(), "HOMOGRAPHY_NONFINITE")
    return homography


def _independent_warp_audit(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_valid: np.ndarray,
    current_valid: np.ndarray,
    previous_rotation: np.ndarray,
    current_rotation: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-render the frozen warp with OpenCV, without importing R3 transport."""

    previous_gray = cv2.cvtColor(
        np.asarray(previous_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY
    )
    current_gray = cv2.cvtColor(
        np.asarray(current_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY
    )
    previous_mask = (
        (np.asarray(previous_valid) > 0).astype(np.uint8) * 255
    )
    current_mask = (
        (np.asarray(current_valid) > 0).astype(np.uint8) * 255
    )
    require(
        previous_gray.shape == current_gray.shape == previous_mask.shape
        and previous_mask.shape == current_mask.shape,
        "WARP_INPUT_SHAPE",
    )
    height, width = previous_gray.shape
    homography = _rotation_homography(
        intrinsic, previous_rotation, current_rotation
    )
    inverse = np.linalg.inv(homography)
    warped_gray = cv2.warpPerspective(
        current_gray,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = cv2.warpPerspective(
        current_mask,
        inverse,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    boundary, interior, common = boundary_masks(
        previous_mask, warped_mask
    )
    previous_count = int(np.count_nonzero(previous_mask))
    common_count = int(np.count_nonzero(common))
    interior_count = int(np.count_nonzero(interior))
    boundary_count = int(np.count_nonzero(boundary))
    residual = (
        warped_gray.astype(np.float64) - previous_gray.astype(np.float64)
    ) / 255.0
    interior_values = residual[interior]
    boundary_values = residual[boundary]
    common_y, common_x = np.nonzero(common)
    audit_points = np.column_stack((common_x, common_y)).astype(np.float64)
    roundtrip = _homography_points(
        _homography_points(audit_points, homography), inverse
    )
    errors = np.linalg.norm(
        roundtrip - np.asarray(audit_points, dtype=np.float64), axis=1
    )
    overlap = common_count / previous_count if previous_count else 0.0
    interior_fraction = (
        interior_count / previous_count if previous_count else 0.0
    )
    evaluable = (
        previous_count > 0
        and overlap >= 0.75
        and interior_fraction >= 0.75
        and len(interior_values) > 0
        and len(boundary_values) > 0
        and np.isfinite(errors).all()
        and np.isfinite(interior_values).all()
        and np.isfinite(boundary_values).all()
    )
    audit = {
        "evaluable": evaluable,
        "previous_valid_count": previous_count,
        "common_count": common_count,
        "interior_count": interior_count,
        "boundary_count": boundary_count,
        "overlap_fraction": overlap,
        "interior_support_fraction": interior_fraction,
        "coordinate_roundtrip_p99_px": (
            hf7(errors.tolist(), 0.99) if evaluable else None
        ),
        "interior_signed_p50": (
            hf7(interior_values.tolist(), 0.5) if evaluable else None
        ),
        "interior_signed_p90": (
            hf7(interior_values.tolist(), 0.9) if evaluable else None
        ),
        "interior_absolute_p50": (
            hf7(np.abs(interior_values).tolist(), 0.5)
            if evaluable
            else None
        ),
        "interior_absolute_p90": (
            hf7(np.abs(interior_values).tolist(), 0.9)
            if evaluable
            else None
        ),
        "interior_absolute_p99": (
            hf7(np.abs(interior_values).tolist(), 0.99)
            if evaluable
            else None
        ),
        "boundary_absolute_p90": (
            hf7(np.abs(boundary_values).tolist(), 0.9)
            if evaluable
            else None
        ),
        "previous_gray_sha256": sha256_array(previous_gray),
        "warped_gray_sha256": sha256_array(warped_gray),
        "previous_valid_sha256": sha256_array(previous_mask),
        "warped_valid_sha256": sha256_array(warped_mask),
    }
    return audit, boundary, interior, homography, warped_mask


def _dynamic_static_scene(base: Mapping[str, Any], frame_index: int) -> dict[str, Any]:
    scene = json.loads(json.dumps(base))
    target = next(
        (
            item
            for item in scene["world"]["objects"]
            if int(item["object_id"]) == 1001
        ),
        None,
    )
    require(target is not None, "TARGET_OBJECT_1001")
    target["plane_z_m"] = 6.0
    target["bounds_xy_m"] = [-0.4, 0.8, -0.7, 0.9]
    target["vertices_world_m"] = [
        [-0.4, -0.7, 6.0],
        [0.8, -0.7, 6.0],
        [0.8, 0.9, 6.0],
        [-0.4, 0.9, 6.0],
    ]
    scene.pop("scene_geometry_sha256", None)
    scene["frame_index"] = int(frame_index)
    scene["target_motion"] = "STATIC"
    scene["scene_geometry_sha256"] = sha256_value(scene)
    return scene


def _rotation_poses(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    poses = []
    for source in trajectory["poses"]:
        poses.append(
            {
                "frame_index": source["frame_index"],
                "timestamp_s": source["timestamp_s"],
                "rotation_matrix": source["rotation_matrix"],
                "translation_m": [0.0, 0.0, 0.0],
            }
        )
    return poses


def _load_generator_modules(
    root: Path, bound_hashes: Mapping[str, str]
) -> tuple[Any, Any]:
    generator_path = root / GENERATOR_RELATIVE
    renderer_path = root / RENDERER_RELATIVE
    require(
        sha256_file(generator_path) == bound_hashes.get(GENERATOR_RELATIVE),
        "GENERATOR_HASH_BINDING",
    )
    require(
        sha256_file(renderer_path) == bound_hashes.get(RENDERER_RELATIVE),
        "RENDERER_HASH_BINDING",
    )
    root_text = os.fspath(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    generator = importlib.import_module(
        "scripts.research.egomotion_compensated_looming."
        "periodic_self_motion_counterfactual_r2.generator_geometry"
    )
    renderer = importlib.import_module(
        "scripts.research.egomotion_compensated_looming."
        "periodic_self_motion_counterfactual_r2."
        "material_residual_contraction_r1"
    )
    return generator, renderer


def _project_world(
    world: np.ndarray,
    rotation_wc: np.ndarray,
    translation_wc: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera = (
        np.asarray(rotation_wc, dtype=np.float64).T
        @ (
            np.asarray(world, dtype=np.float64)
            - np.asarray(translation_wc, dtype=np.float64)
        ).T
    ).T
    depth = camera[:, 2]
    pixels = np.zeros((len(world), 2), dtype=np.float64)
    valid = np.isfinite(camera).all(axis=1) & (depth > 0.0)
    pixels[valid, 0] = (
        intrinsic[0, 0] * camera[valid, 0] / depth[valid]
        + intrinsic[0, 2]
    )
    pixels[valid, 1] = (
        intrinsic[1, 1] * camera[valid, 1] / depth[valid]
        + intrinsic[1, 2]
    )
    return pixels, depth


def _source_correspondence(
    generator: Any,
    previous_points: np.ndarray,
    previous_scene: Mapping[str, Any],
    current_scene: Mapping[str, Any],
    previous_rotation: np.ndarray,
    current_rotation: np.ndarray,
    previous_translation: np.ndarray,
    current_translation: np.ndarray,
    homography: np.ndarray,
) -> dict[str, np.ndarray]:
    points = np.asarray(previous_points, dtype=np.float64).reshape(-1, 2)
    depth, object_id, world = generator._raycast(
        previous_scene, previous_rotation, previous_translation, points
    )
    current_pixels, current_depth = _project_world(
        world,
        current_rotation,
        current_translation,
        np.asarray(generator.K, dtype=np.float64),
    )
    safe = (
        np.isfinite(depth)
        & (depth >= 0.5)
        & (depth <= 25.0)
        & np.isfinite(current_pixels).all(axis=1)
        & (current_pixels[:, 0] >= 0.0)
        & (current_pixels[:, 0] < int(generator.WIDTH) - 1.0)
        & (current_pixels[:, 1] >= 0.0)
        & (current_pixels[:, 1] < int(generator.HEIGHT) - 1.0)
    )
    visible = np.zeros(len(points), dtype=bool)
    if np.any(safe):
        current_z, current_id, current_world = generator._raycast(
            current_scene,
            current_rotation,
            current_translation,
            current_pixels[safe],
        )
        tolerance = np.maximum(1e-7, 1e-6 * current_depth[safe])
        visible[safe] = (
            np.isfinite(current_z)
            & (current_id == object_id[safe])
            & (np.abs(current_z - current_depth[safe]) <= tolerance)
            & (
                np.linalg.norm(current_world - world[safe], axis=1)
                <= tolerance
            )
        )
    aligned = np.zeros((len(points), 2), dtype=np.float64)
    aligned_all = _homography_points(current_pixels[safe], np.linalg.inv(homography))
    aligned[safe] = aligned_all
    valid = (
        safe
        & visible
        & (object_id > 0)
        & np.isfinite(world).all(axis=1)
        & np.isfinite(aligned).all(axis=1)
    )
    return {
        "valid": np.ascontiguousarray(valid),
        "object_id": np.asarray(object_id, dtype=np.int32),
        "depth_m": np.asarray(depth, dtype=np.float64),
        "world_xyz": np.asarray(world, dtype=np.float64),
        "current_xy": current_pixels,
        "aligned_xy": aligned,
    }


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    dtype: np.dtype[Any] | type,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    array = arrays[name]
    require(array.dtype == np.dtype(dtype), f"NPZ_DTYPE:{name}:{array.dtype}")
    require(array.ndim == 1 + len(trailing_shape), f"NPZ_NDIM:{name}")
    require(array.shape[1:] == trailing_shape, f"NPZ_SHAPE:{name}")
    return array


def _require_finite_array(array: np.ndarray, name: str) -> None:
    if array.dtype.kind in "fc":
        require(np.isfinite(array).all(), f"NPZ_NONFINITE:{name}")


def _validate_offsets(
    offsets: np.ndarray,
    expected_groups: int,
    terminal_length: int,
    label: str,
) -> None:
    require(offsets.dtype == np.int64, f"{label}:DTYPE")
    require(offsets.shape == (expected_groups + 1,), f"{label}:SHAPE")
    require(int(offsets[0]) == 0, f"{label}:START")
    require(int(offsets[-1]) == terminal_length, f"{label}:END")
    require(np.all(offsets[1:] >= offsets[:-1]), f"{label}:MONOTONIC")


def load_primitives(path: Path, pair_count: int) -> dict[str, np.ndarray]:
    """Load and validate the fixed, pickle-free ragged primitive package."""

    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == NPZ_KEYS, "NPZ_KEYSET")
        arrays = {name: archive[name].copy() for name in archive.files}

    initial_offsets = _require_array(arrays, "initial_offsets", np.int64, ())
    initial_previous = _require_array(
        arrays, "initial_previous", np.float32, (2,)
    )
    initial_length = len(initial_previous)
    _validate_offsets(initial_offsets, pair_count, initial_length, "INITIAL_OFFSETS")
    initial_source_offsets = _require_array(
        arrays, "initial_source_offsets", np.int64, ()
    )
    initial_source_index = _require_array(
        arrays, "initial_source_index", np.int32, ()
    )
    initial_source_aligned = _require_array(
        arrays, "initial_source_aligned", np.float64, (2,)
    )
    initial_source_depth = _require_array(
        arrays, "initial_source_depth_m", np.float64, ()
    )
    initial_source_object = _require_array(
        arrays, "initial_source_object_id", np.int32, ()
    )
    initial_source_world = _require_array(
        arrays, "initial_source_world", np.float64, (3,)
    )
    initial_source_current = _require_array(
        arrays, "initial_source_current_pixel", np.float64, (2,)
    )
    initial_source_valid = _require_array(
        arrays, "initial_source_valid", np.bool_, ()
    )
    _validate_offsets(
        initial_source_offsets,
        pair_count,
        len(initial_source_aligned),
        "INITIAL_SOURCE_OFFSETS",
    )
    require(
        len(initial_source_index)
        == len(initial_source_aligned)
        == len(initial_source_depth)
        == len(initial_source_object)
        == len(initial_source_world)
        == len(initial_source_current)
        and len(initial_source_valid) == initial_length,
        "INITIAL_SOURCE_LENGTHS",
    )

    lk_forward_offsets = _require_array(
        arrays, "lk_forward_offsets", np.int64, ()
    )
    lk_forward_source_index = _require_array(
        arrays, "lk_forward_source_index", np.int32, ()
    )
    lk_forward_points = _require_array(
        arrays, "lk_forward_points", np.float32, (2,)
    )
    _validate_offsets(
        lk_forward_offsets,
        pair_count,
        len(lk_forward_points),
        "LK_FORWARD_OFFSETS",
    )
    require(
        len(lk_forward_source_index) == len(lk_forward_points),
        "LK_FORWARD_LENGTHS",
    )
    lk_backward_offsets = _require_array(
        arrays, "lk_backward_offsets", np.int64, ()
    )
    lk_backward_source_index = _require_array(
        arrays, "lk_backward_source_index", np.int32, ()
    )
    lk_backward_points = _require_array(
        arrays, "lk_backward_points", np.float32, (2,)
    )
    lk_backward_level = _require_array(
        arrays, "lk_backward_level", np.int16, ()
    )
    lk_backward_error = _require_array(
        arrays, "lk_backward_error", np.float32, ()
    )
    _validate_offsets(
        lk_backward_offsets,
        pair_count,
        len(lk_backward_points),
        "LK_BACKWARD_OFFSETS",
    )
    require(
        len(lk_backward_source_index)
        == len(lk_backward_points)
        == len(lk_backward_level)
        == len(lk_backward_error),
        "LK_BACKWARD_LENGTHS",
    )
    for name in (
        "lk_forward_available",
        "lk_backward_available",
        "lk_fb_pass",
        "lk_mask_pass",
        "lk_accepted",
    ):
        require(
            len(_require_array(arrays, name, np.bool_, ())) == initial_length,
            f"NPZ_LENGTH:{name}",
        )
    rejection = arrays["lk_rejection_reason"]
    require(
        rejection.dtype.kind == "U"
        and rejection.dtype.itemsize == 24 * 4
        and rejection.shape == (initial_length,),
        "NPZ_DTYPE:lk_rejection_reason",
    )
    require(
        np.array_equal(
            arrays["lk_accepted"],
            arrays["lk_forward_available"]
            & arrays["lk_backward_available"]
            & arrays["lk_fb_pass"]
            & arrays["lk_mask_pass"],
        ),
        "LK_ACCEPTED_STATUS_COMPOSITION",
    )

    def track_family(prefix: str) -> tuple[np.ndarray, int]:
        offsets = _require_array(arrays, f"{prefix}_offsets", np.int64, ())
        previous = _require_array(
            arrays, f"{prefix}_previous", np.float32, (2,)
        )
        current = _require_array(
            arrays, f"{prefix}_current", np.float32, (2,)
        )
        fb_error = _require_array(
            arrays, f"{prefix}_fb_error", np.float32, ()
        )
        length = len(previous)
        _validate_offsets(offsets, pair_count, length, f"{prefix.upper()}_OFFSETS")
        require(len(current) == len(fb_error) == length, f"{prefix.upper()}_LENGTHS")
        require(
            np.all(fb_error >= 0.0) and np.all(fb_error <= 1.0 + 1e-6),
            f"{prefix.upper()}_FB_RANGE",
        )
        source_offsets = _require_array(
            arrays, f"{prefix}_source_offsets", np.int64, ()
        )
        source_index = _require_array(
            arrays, f"{prefix}_source_index", np.int32, ()
        )
        source_aligned = _require_array(
            arrays, f"{prefix}_source_aligned", np.float64, (2,)
        )
        source_valid = _require_array(
            arrays, f"{prefix}_source_valid", np.bool_, ()
        )
        _validate_offsets(
            source_offsets,
            pair_count,
            len(source_aligned),
            f"{prefix.upper()}_SOURCE_OFFSETS",
        )
        require(
            len(source_index) == len(source_aligned)
            and len(source_valid) == length,
            f"{prefix.upper()}_SOURCE_LENGTHS",
        )
        return offsets, length

    accepted_offsets, accepted_length = track_family("accepted")
    managed_offsets, managed_length = track_family("managed")
    managed_path_class = arrays["managed_path_class"]
    require(
        managed_path_class.dtype.kind == "U"
        and managed_path_class.dtype.itemsize == 16 * 4
        and managed_path_class.shape == (managed_length,),
        "NPZ_DTYPE:managed_path_class",
    )
    require(
        {str(value) for value in managed_path_class.tolist()}
        <= {"BASELINE", "CARRIED", "SUPPLEMENT"},
        "MANAGED_PATH_CLASS",
    )

    fit_offsets = _require_array(arrays, "fit_offsets", np.int64, ())
    fit_previous = _require_array(
        arrays, "fit_previous", np.float32, (2,)
    )
    fit_current = _require_array(arrays, "fit_current", np.float32, (2,))
    fit_error = _require_array(arrays, "fit_fb_error", np.float32, ())
    fit_group_path = arrays["fit_group_path"]
    fit_length = len(fit_previous)
    _validate_offsets(
        fit_offsets,
        pair_count * len(FIT_GROUP_PATHS),
        fit_length,
        "FIT_OFFSETS",
    )
    require(
        len(fit_current) == len(fit_error) == fit_length,
        "FIT_LENGTHS",
    )
    require(
        fit_group_path.dtype.kind == "U"
        and fit_group_path.shape
        == (pair_count * len(FIT_GROUP_PATHS),),
        "NPZ_DTYPE:fit_group_path",
    )
    require(
        [
            str(value)
            for value in fit_group_path.reshape(pair_count, -1)[0].tolist()
        ]
        == list(FIT_GROUP_PATHS)
        if pair_count
        else True,
        "FIT_GROUP_PATH_FIRST",
    )
    for pair_index in range(pair_count):
        group_paths = [
            str(value)
            for value in fit_group_path[
                pair_index
                * len(FIT_GROUP_PATHS) : (pair_index + 1)
                * len(FIT_GROUP_PATHS)
            ].tolist()
        ]
        require(
            group_paths == list(FIT_GROUP_PATHS),
            f"FIT_GROUP_PATH:{pair_index}",
        )
    require(
        np.all(fit_error >= 0.0) and np.all(fit_error <= 1.0 + 1e-6),
        "FIT_FB_RANGE",
    )

    activated_offsets = _require_array(
        arrays, "activated_offsets", np.int64, ()
    )
    activated_cell_index = _require_array(
        arrays, "activated_cell_index", np.int8, ()
    )
    _validate_offsets(
        activated_offsets,
        pair_count,
        len(activated_cell_index),
        "ACTIVATED_OFFSETS",
    )

    merge_offsets = _require_array(
        arrays, "merge_candidate_offsets", np.int64, ()
    )
    merge_previous = _require_array(
        arrays, "merge_candidate_previous", np.float32, (2,)
    )
    merge_current = _require_array(
        arrays, "merge_candidate_current", np.float32, (2,)
    )
    merge_error = _require_array(
        arrays, "merge_candidate_fb_error", np.float32, ()
    )
    merge_selected = _require_array(
        arrays, "merge_candidate_selected", np.bool_, ()
    )
    merge_path = arrays["merge_candidate_path_class"]
    require(
        merge_path.dtype.kind == "U"
        and merge_path.dtype.itemsize == 16 * 4
        and merge_path.ndim == 1,
        "NPZ_DTYPE:merge_candidate_path_class",
    )
    merge_group_path = arrays["merge_group_path"]
    require(
        merge_group_path.dtype.kind == "U"
        and merge_group_path.shape
        == (pair_count * len(MERGE_GROUP_PATHS),),
        "NPZ_DTYPE:merge_group_path",
    )
    merge_length = len(merge_previous)
    _validate_offsets(
        merge_offsets,
        pair_count * len(MERGE_GROUP_PATHS),
        merge_length,
        "MERGE_OFFSETS",
    )
    require(
        len(merge_current)
        == len(merge_error)
        == len(merge_selected)
        == len(merge_path)
        == merge_length,
        "MERGE_LENGTHS",
    )
    require(
        {str(value) for value in merge_path.tolist()}
        <= {"BASELINE", "CARRIED", "SUPPLEMENT"},
        "MERGE_PATH_CLASS",
    )
    for pair_index in range(pair_count):
        group_paths = [
            str(value)
            for value in merge_group_path[
                pair_index
                * len(MERGE_GROUP_PATHS) : (pair_index + 1)
                * len(MERGE_GROUP_PATHS)
            ].tolist()
        ]
        require(
            group_paths == list(MERGE_GROUP_PATHS),
            f"MERGE_GROUP_PATH:{pair_index}",
        )

    consensus_offsets = _require_array(
        arrays, "consensus_offsets", np.int64, ()
    )
    consensus_previous = _require_array(
        arrays, "consensus_previous", np.float32, (2,)
    )
    consensus_current = _require_array(
        arrays, "consensus_current", np.float32, (2,)
    )
    consensus_pair_cell = _require_array(
        arrays, "consensus_pair_cell", np.int32, (2,)
    )
    consensus_path = arrays["consensus_path"]
    require(
        consensus_path.dtype.kind == "U"
        and consensus_path.dtype.itemsize == 20 * 4
        and consensus_path.ndim == 1,
        "NPZ_DTYPE:consensus_path",
    )
    consensus_groups = pair_count * GRID_CELLS * 2
    _validate_offsets(
        consensus_offsets,
        consensus_groups,
        len(consensus_previous),
        "CONSENSUS_OFFSETS",
    )
    require(
        len(consensus_current) == len(consensus_previous)
        and len(consensus_pair_cell) == len(consensus_path) == consensus_groups,
        "CONSENSUS_LENGTHS",
    )

    for name, array in arrays.items():
        require(array.dtype.kind != "O", f"NPZ_OBJECT:{name}")
        _require_finite_array(array, name)

    for pair_index in range(pair_count):
        il, ir = map(int, initial_offsets[pair_index : pair_index + 2])
        al, ar = map(int, accepted_offsets[pair_index : pair_index + 2])
        selected = arrays["lk_accepted"][il:ir]
        require(int(np.count_nonzero(selected)) == ar - al, f"ACCEPTED_COUNT:{pair_index}")
        require(
            np.array_equal(
                arrays["accepted_previous"][al:ar],
                initial_previous[il:ir][selected],
            ),
            f"ACCEPTED_PREVIOUS_IDENTITY:{pair_index}",
        )

        fl, fr = map(int, lk_forward_offsets[pair_index : pair_index + 2])
        forward_index = lk_forward_source_index[fl:fr]
        require(
            np.array_equal(
                forward_index,
                np.flatnonzero(arrays["lk_forward_available"][il:ir]).astype(np.int32),
            ),
            f"LK_FORWARD_INDEX:{pair_index}",
        )
        forward_lookup = np.zeros((ir - il, 2), dtype=np.float32)
        forward_lookup[forward_index] = lk_forward_points[fl:fr]
        require(
            np.array_equal(
                arrays["accepted_current"][al:ar],
                forward_lookup[selected],
            ),
            f"ACCEPTED_CURRENT_IDENTITY:{pair_index}",
        )

        bl, br = map(int, lk_backward_offsets[pair_index : pair_index + 2])
        backward_index = lk_backward_source_index[bl:br]
        require(
            np.array_equal(
                backward_index,
                np.flatnonzero(arrays["lk_backward_available"][il:ir]).astype(np.int32),
            ),
            f"LK_BACKWARD_INDEX:{pair_index}",
        )
        error_lookup = np.zeros(ir - il, dtype=np.float32)
        error_lookup[backward_index] = lk_backward_error[bl:br]
        recomputed_backward_error = np.linalg.norm(
            lk_backward_points[bl:br].astype(np.float32)
            - initial_previous[il:ir][backward_index].astype(np.float32),
            axis=1,
        ).astype(np.float32)
        require(
            np.array_equal(
                lk_backward_error[bl:br], recomputed_backward_error
            ),
            f"LK_BACKWARD_ERROR_IDENTITY:{pair_index}",
        )
        require(
            np.all(lk_backward_level[bl:br] >= 0)
            and np.all(lk_backward_level[bl:br] <= 4),
            f"LK_BACKWARD_LEVEL_RANGE:{pair_index}",
        )
        require(
            np.array_equal(arrays["accepted_fb_error"][al:ar], error_lookup[selected]),
            f"ACCEPTED_FB_IDENTITY:{pair_index}",
        )
        expected_rejection = np.full(ir - il, "ACCEPTED", dtype="<U24")
        forward_available = arrays["lk_forward_available"][il:ir]
        backward_available = arrays["lk_backward_available"][il:ir]
        fb_pass = arrays["lk_fb_pass"][il:ir]
        mask_pass = arrays["lk_mask_pass"][il:ir]
        expected_rejection[~forward_available] = "FORWARD_UNAVAILABLE"
        expected_rejection[forward_available & ~backward_available] = (
            "BACKWARD_UNAVAILABLE"
        )
        expected_rejection[
            forward_available & backward_available & ~fb_pass
        ] = "FORWARD_BACKWARD_FAIL"
        expected_rejection[fb_pass & ~mask_pass] = "MASK_FAIL"
        require(
            np.array_equal(
                arrays["lk_rejection_reason"][il:ir], expected_rejection
            ),
            f"LK_REJECTION_REASON:{pair_index}",
        )

        for prefix, offsets, valid_name, source_offsets_name, source_index_name in (
            ("initial", initial_offsets, "initial_source_valid", "initial_source_offsets", "initial_source_index"),
            ("accepted", accepted_offsets, "accepted_source_valid", "accepted_source_offsets", "accepted_source_index"),
            ("managed", managed_offsets, "managed_source_valid", "managed_source_offsets", "managed_source_index"),
        ):
            left, right = map(int, offsets[pair_index : pair_index + 2])
            source_offsets = arrays[source_offsets_name]
            sl, sr = map(int, source_offsets[pair_index : pair_index + 2])
            require(
                np.array_equal(
                    arrays[source_index_name][sl:sr],
                    np.flatnonzero(arrays[valid_name][left:right]).astype(np.int32),
                ),
                f"{prefix.upper()}_SOURCE_INDEX:{pair_index}",
            )

        fit_groups = [
            _fit_group(arrays, pair_index, ordinal)
            for ordinal in range(len(FIT_GROUP_PATHS))
        ]
        require(
            np.array_equal(
                fit_groups[1]["previous"],
                arrays["accepted_previous"][al:ar],
            )
            and np.array_equal(
                fit_groups[1]["current"],
                arrays["accepted_current"][al:ar],
            )
            and np.array_equal(
                fit_groups[1]["fb_error"],
                arrays["accepted_fb_error"][al:ar],
            ),
            f"COMPENSATED_INITIAL_ACCEPTED_IDENTITY:{pair_index}",
        )

        activated_left, activated_right = map(
            int, activated_offsets[pair_index : pair_index + 2]
        )
        activated = activated_cell_index[
            activated_left:activated_right
        ].astype(np.int64)
        require(
            np.all((activated >= 0) & (activated < GRID_CELLS))
            and np.array_equal(activated, np.unique(activated)),
            f"ACTIVATED_CELL_ORDER:{pair_index}",
        )
        active = {int(value) for value in activated.tolist()}

        managed_group_labels: dict[str, np.ndarray] = {}
        for merge_ordinal, merge_name in enumerate(MERGE_GROUP_PATHS):
            merge_group = (
                pair_index * len(MERGE_GROUP_PATHS) + merge_ordinal
            )
            ml, mr = map(
                int, merge_offsets[merge_group : merge_group + 2]
            )
            candidate_previous = merge_previous[ml:mr]
            candidate_current = merge_current[ml:mr]
            candidate_error = merge_error[ml:mr]
            candidate_path = merge_path[ml:mr]
            recorded_selected = merge_selected[ml:mr]
            ranks = np.asarray(
                [
                    {"BASELINE": 0, "CARRIED": 1, "SUPPLEMENT": 2}[
                        str(value)
                    ]
                    for value in candidate_path.tolist()
                ],
                dtype=np.int8,
            )
            require(
                np.all(ranks[1:] >= ranks[:-1]),
                f"MERGE_PATH_ORDER:{pair_index}:{merge_name}",
            )
            expected_selected = np.zeros(len(candidate_previous), dtype=bool)
            occupied: list[np.ndarray] = []
            for candidate_index, point in enumerate(candidate_previous):
                accept = True
                if occupied:
                    squared = np.sum(
                        (
                            np.vstack(occupied).astype(np.float64)
                            - point.astype(np.float64)
                        )
                        ** 2,
                        axis=1,
                    )
                    accept = not bool(np.any(squared < 25.0))
                expected_selected[candidate_index] = accept
                if accept:
                    occupied.append(point)
            require(
                np.array_equal(recorded_selected, expected_selected),
                f"MERGE_SELECTION:{pair_index}:{merge_name}",
            )
            managed_fit = fit_groups[2 + merge_ordinal]
            if active:
                require(
                    len(candidate_previous) > 0
                    and np.array_equal(
                        managed_fit["previous"],
                        candidate_previous[expected_selected],
                    )
                    and np.array_equal(
                        managed_fit["current"],
                        candidate_current[expected_selected],
                    )
                    and np.array_equal(
                        managed_fit["fb_error"],
                        candidate_error[expected_selected],
                    ),
                    f"MERGE_MANAGED_IDENTITY:{pair_index}:{merge_name}",
                )
            else:
                require(
                    len(candidate_previous)
                    == len(managed_fit["previous"])
                    == 0,
                    f"MERGE_WITHOUT_ACTIVATION:{pair_index}:{merge_name}",
                )
            managed_group_labels[merge_name] = candidate_path[
                expected_selected
            ]

        if active:
            compensated_managed = fit_groups[3]
            compensated_labels = managed_group_labels[
                "COMPENSATED_MANAGED"
            ]
            final_previous_parts: list[np.ndarray] = []
            final_current_parts: list[np.ndarray] = []
            final_error_parts: list[np.ndarray] = []
            final_label_parts: list[np.ndarray] = []
            for cell_index in range(GRID_CELLS):
                if cell_index in active:
                    source = compensated_managed
                    labels = compensated_labels
                else:
                    source = fit_groups[1]
                    labels = np.full(
                        len(source["previous"]), "BASELINE", dtype="<U16"
                    )
                x0, y0, x1, y1 = cell_bounds(cell_index)
                cell_selected = (
                    (source["previous"][:, 0] >= x0)
                    & (source["previous"][:, 0] < x1)
                    & (source["previous"][:, 1] >= y0)
                    & (source["previous"][:, 1] < y1)
                )
                final_previous_parts.append(
                    source["previous"][cell_selected]
                )
                final_current_parts.append(source["current"][cell_selected])
                final_error_parts.append(source["fb_error"][cell_selected])
                final_label_parts.append(labels[cell_selected])
            expected_managed_previous = np.concatenate(
                final_previous_parts, axis=0
            )
            expected_managed_current = np.concatenate(
                final_current_parts, axis=0
            )
            expected_managed_error = np.concatenate(
                final_error_parts, axis=0
            )
            expected_managed_labels = np.concatenate(
                final_label_parts, axis=0
            )
        else:
            expected_managed_previous = fit_groups[1]["previous"]
            expected_managed_current = fit_groups[1]["current"]
            expected_managed_error = fit_groups[1]["fb_error"]
            expected_managed_labels = np.full(
                len(expected_managed_previous), "BASELINE", dtype="<U16"
            )
        managed_left, managed_right = map(
            int, managed_offsets[pair_index : pair_index + 2]
        )
        require(
            np.array_equal(
                arrays["managed_previous"][managed_left:managed_right],
                expected_managed_previous,
            )
            and np.array_equal(
                arrays["managed_current"][managed_left:managed_right],
                expected_managed_current,
            )
            and np.array_equal(
                arrays["managed_fb_error"][managed_left:managed_right],
                expected_managed_error,
            )
            and np.array_equal(
                arrays["managed_path_class"][managed_left:managed_right],
                expected_managed_labels,
            ),
            f"ACTIVATED_FINAL_SPLICE:{pair_index}",
        )

        for path_ordinal, path_name in enumerate(("RAW_FINAL", "COMPENSATED_FINAL")):
            for cell_index in range(GRID_CELLS):
                group = pair_index * GRID_CELLS * 2 + path_ordinal * GRID_CELLS + cell_index
                require(
                    np.array_equal(
                        consensus_pair_cell[group],
                        np.asarray([pair_index, cell_index], dtype=np.int32),
                    )
                    and str(consensus_path[group]) == path_name,
                    f"CONSENSUS_GROUP:{pair_index}:{path_name}:{cell_index}",
                )
                left, right = map(int, consensus_offsets[group : group + 2])
                points = consensus_previous[left:right]
                x0, y0, x1, y1 = cell_bounds(cell_index)
                require(
                    np.all(points[:, 0] >= x0)
                    and np.all(points[:, 0] < x1)
                    and np.all(points[:, 1] >= y0)
                    and np.all(points[:, 1] < y1),
                    f"CONSENSUS_CELL_MEMBERSHIP:{pair_index}:{path_name}:{cell_index}",
                )
    return arrays


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), label)
    return value


def _validate_pair_state_record(
    value: Mapping[str, Any], label: str
) -> None:
    record = dict(value)
    semantic = record.pop("semantic_sha256", None)
    require(
        isinstance(semantic, str)
        and len(semantic) == 64
        and semantic == sha256_value(record),
        f"{label}:SEMANTIC_HASH",
    )
    count = record.get("survivor_count")
    requested = record.get("survivor_requested_count")
    require(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and isinstance(requested, int)
        and not isinstance(requested, bool)
        and requested >= count,
        f"{label}:COUNTS",
    )
    hash_keys = (
        "survivor_previous_sha256",
        "survivor_current_sha256",
        "survivor_fb_error_sha256",
    )
    present = [key in record for key in hash_keys]
    require(all(present) or not any(present), f"{label}:HASH_KEYSET")
    if all(present):
        require(
            all(
                isinstance(record[key], str) and len(record[key]) == 64
                for key in hash_keys
            )
            and finite_number(
                record.get("dt_seconds"), f"{label}:DT"
            )
            > 0.0,
            f"{label}:SURVIVOR_STATE",
        )
    else:
        require(
            count == requested == 0
            and record.get("dt_seconds") is None,
            f"{label}:EMPTY_STATE",
        )


def _bool(value: Any, label: str) -> bool:
    require(type(value) is bool, label)
    return bool(value)


def _layer(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(row.get(name), f"LAYER:{name}")


def _required_number(
    value: Mapping[str, Any], name: str, label: str
) -> float:
    require(name in value, f"{label}:MISSING:{name}")
    return finite_number(value[name], f"{label}:{name}")


def _optional_layer_number(
    value: Mapping[str, Any], name: str, label: str
) -> float | None:
    require(name in value, f"{label}:MISSING:{name}")
    return optional_number(value[name], f"{label}:{name}")


def _compare_pair_reduction(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    label: str,
    *,
    cell_count_key: str = "evaluable_cell_count",
    signed_key: str = "signed_per_s",
    absolute_key: str = "absolute_per_s",
) -> None:
    require(
        _bool(recorded.get("evaluable"), f"{label}:EVALUABLE")
        is bool(recomputed["evaluable"]),
        f"{label}:EVALUABLE_MISMATCH",
    )
    require(
        int(recorded.get(cell_count_key, -1))
        == int(recomputed["evaluable_cell_count"]),
        f"{label}:CELL_COUNT",
    )
    close(
        _optional_layer_number(recorded, signed_key, label),
        recomputed["signed_per_s"],
        f"{label}:SIGNED",
    )
    close(
        _optional_layer_number(recorded, absolute_key, label),
        recomputed["absolute_per_s"],
        f"{label}:ABSOLUTE",
    )


def _boundary_evaluable_states(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    label: str,
) -> tuple[bool, bool, bool]:
    joint = _bool(recorded.get("evaluable"), f"{label}:EVALUABLE")
    boundary = _bool(
        recorded.get("boundary_evaluable"),
        f"{label}:BOUNDARY_EVALUABLE",
    )
    interior = _bool(
        recorded.get("interior_evaluable"),
        f"{label}:INTERIOR_EVALUABLE",
    )
    require(
        joint is (boundary and interior),
        f"{label}:JOINT_EVALUABLE",
    )
    require(
        joint is bool(recomputed["evaluable"])
        and boundary is bool(recomputed["boundary"]["evaluable"])
        and interior is bool(recomputed["interior"]["evaluable"]),
        f"{label}:REPLAY_EVALUABLE",
    )
    return joint, boundary, interior


def _validate_audit_coefficients(
    coefficients: Any,
    expansion_per_s: float,
    cell_index: int,
    label: str,
) -> None:
    """Apply the frozen numeric-representation amendment to audit coefficients."""

    matrix = np.asarray(coefficients, dtype=np.float64)
    require(
        matrix.shape == (3, 2) and np.isfinite(matrix).all(),
        f"{label}:SHAPE_OR_FINITE",
    )
    x0, y0, x1, y1 = cell_bounds(cell_index)
    half_width = max(0.5 * (x1 - x0), 1.0)
    half_height = max(0.5 * (y1 - y0), 1.0)
    expansion_from_coefficients = float(
        0.5
        * (
            matrix[0, 0] / half_width
            + matrix[1, 1] / half_height
        )
    )
    require(
        expansion_from_coefficients == expansion_per_s,
        f"{label}:EXPANSION_FORMULA",
    )


def _fit_group(
    arrays: Mapping[str, np.ndarray],
    pair_index: int,
    group_ordinal: int,
) -> dict[str, Any]:
    group = pair_index * len(FIT_GROUP_PATHS) + group_ordinal
    offsets = arrays["fit_offsets"]
    left, right = map(int, offsets[group : group + 2])
    return {
        "path": str(arrays["fit_group_path"][group]),
        "previous": arrays["fit_previous"][left:right],
        "current": arrays["fit_current"][left:right],
        "fb_error": arrays["fit_fb_error"][left:right],
    }


def _early_warp_abstention_local_fit(
    arrays: Mapping[str, np.ndarray], pair_index: int
) -> dict[str, Any]:
    """Validate the frozen R3 overlap short-circuit and its empty primitives."""

    for ordinal, expected_path in enumerate(FIT_GROUP_PATHS):
        group = pair_index * len(FIT_GROUP_PATHS) + ordinal
        left, right = map(int, arrays["fit_offsets"][group : group + 2])
        require(
            left == right
            and str(arrays["fit_group_path"][group]) == expected_path,
            f"EARLY_WARP_FIT_GROUP:{pair_index}:{expected_path}",
        )

    activated_left, activated_right = map(
        int, arrays["activated_offsets"][pair_index : pair_index + 2]
    )
    require(
        activated_left == activated_right,
        f"EARLY_WARP_ACTIVATED_NONEMPTY:{pair_index}",
    )

    for ordinal, expected_path in enumerate(MERGE_GROUP_PATHS):
        group = pair_index * len(MERGE_GROUP_PATHS) + ordinal
        left, right = map(
            int, arrays["merge_candidate_offsets"][group : group + 2]
        )
        require(
            left == right
            and str(arrays["merge_group_path"][group]) == expected_path,
            f"EARLY_WARP_MERGE_GROUP:{pair_index}:{expected_path}",
        )

    for path_ordinal, path_name in enumerate(
        ("RAW_FINAL", "COMPENSATED_FINAL")
    ):
        for cell_index in range(GRID_CELLS):
            group = (
                pair_index * GRID_CELLS * 2
                + path_ordinal * GRID_CELLS
                + cell_index
            )
            left, right = map(
                int, arrays["consensus_offsets"][group : group + 2]
            )
            require(
                left == right
                and np.array_equal(
                    arrays["consensus_pair_cell"][group],
                    np.asarray([pair_index, cell_index], dtype=np.int32),
                )
                and str(arrays["consensus_path"][group]) == path_name,
                (
                    f"EARLY_WARP_CONSENSUS_GROUP:{pair_index}:"
                    f"{path_name}:{cell_index}"
                ),
            )

    return {
        "evaluable": False,
        "evaluable_cell_count": 0,
        "common_cell_indices": [],
        "raw_signed_per_s": None,
        "raw_absolute_per_s": None,
        "signed_per_s": None,
        "absolute_per_s": None,
        "raw_cells": [],
        "cells": [],
        "early_warp_abstention": True,
    }


def _consensus_fit(
    arrays: Mapping[str, np.ndarray],
    pair_index: int,
    dt_s: float,
    activated: Sequence[int],
) -> dict[str, Any]:
    """Reproduce native RANSAC in call order, then apply the final splice."""

    active = {int(value) for value in activated}
    groups = [
        _fit_group(arrays, pair_index, ordinal)
        for ordinal in range(len(FIT_GROUP_PATHS))
    ]
    independently_fitted: dict[str, list[dict[str, Any]] | None] = {}
    for expected_path, group in zip(FIT_GROUP_PATHS, groups, strict=True):
        require(group["path"] == expected_path, f"FIT_GROUP_PATH:{pair_index}")
        previous = np.asarray(group["previous"])
        current = np.asarray(group["current"])
        if expected_path.endswith("_MANAGED") and not active:
            require(
                len(previous) == len(current) == len(group["fb_error"]) == 0,
                f"FIT_MANAGED_WITHOUT_ACTIVATION:{pair_index}:{expected_path}",
            )
            independently_fitted[expected_path] = None
            continue
        independently_fitted[expected_path] = [
            _native_cell_fit(previous, current, dt_s, cell_index)
            for cell_index in range(GRID_CELLS)
        ]

    if active:
        require(
            independently_fitted["RAW_MANAGED"] is not None
            and independently_fitted["COMPENSATED_MANAGED"] is not None,
            f"FIT_MANAGED_MISSING:{pair_index}",
        )
    raw_initial_cells = independently_fitted["RAW_INITIAL"]
    compensated_initial_cells = independently_fitted["COMPENSATED_INITIAL"]
    require(
        raw_initial_cells is not None and compensated_initial_cells is not None,
        f"FIT_INITIAL_MISSING:{pair_index}",
    )
    independently_activated = [
        cell_index
        for cell_index in range(GRID_CELLS)
        if (
            int(raw_initial_cells[cell_index]["support_count"]) < 12
            or float(raw_initial_cells[cell_index]["hull_fraction"]) < 0.10
            or int(compensated_initial_cells[cell_index]["support_count"]) < 12
            or float(compensated_initial_cells[cell_index]["hull_fraction"])
            < 0.10
        )
    ]
    expected_activated = (
        [] if pair_index == 0 else independently_activated
    )
    require(
        expected_activated == sorted(active),
        f"ACTIVATED_CELL_REPLAY:{pair_index}",
    )
    path_cells: dict[str, list[dict[str, Any]]] = {}
    offsets = arrays["consensus_offsets"]
    for path_ordinal, path in enumerate(("RAW_FINAL", "COMPENSATED_FINAL")):
        initial_name = "RAW_INITIAL" if path == "RAW_FINAL" else "COMPENSATED_INITIAL"
        managed_name = "RAW_MANAGED" if path == "RAW_FINAL" else "COMPENSATED_MANAGED"
        initial_cells = independently_fitted[initial_name]
        managed_cells = independently_fitted[managed_name]
        require(initial_cells is not None, f"FIT_INITIAL_MISSING:{pair_index}:{path}")
        cells: list[dict[str, Any]] = []
        for cell_index in range(GRID_CELLS):
            source_cells = managed_cells if cell_index in active else initial_cells
            require(
                source_cells is not None,
                f"FIT_ACTIVATED_MANAGED_MISSING:{pair_index}:{path}:{cell_index}",
            )
            cell = dict(source_cells[cell_index])
            group = (
                pair_index * GRID_CELLS * 2
                + path_ordinal * GRID_CELLS
                + cell_index
            )
            left, right = map(int, offsets[group : group + 2])
            recorded_previous = arrays["consensus_previous"][left:right]
            recorded_current = arrays["consensus_current"][left:right]
            require(
                np.array_equal(
                    recorded_previous, cell["consensus_previous"]
                )
                and np.array_equal(
                    recorded_current, cell["consensus_current"]
                ),
                f"NATIVE_CONSENSUS_MISMATCH:{pair_index}:{path}:{cell_index}",
            )
            cell["ols_audit"] = _consensus_ols_audit(
                cell["consensus_previous"],
                cell["consensus_current"],
                dt_s,
                cell_index,
            )
            cells.append(cell)
        path_cells[path] = cells
    common = [
        cell_index
        for cell_index in range(GRID_CELLS)
        if path_cells["RAW_FINAL"][cell_index]["evaluable"]
        and path_cells["COMPENSATED_FINAL"][cell_index]["evaluable"]
    ]
    raw_expansions = [
        float(path_cells["RAW_FINAL"][index]["expansion_per_s"])
        for index in common
    ]
    expansions = [
        float(path_cells["COMPENSATED_FINAL"][index]["expansion_per_s"])
        for index in common
    ]
    evaluable = len(common) >= 5
    return {
        "evaluable": evaluable,
        "evaluable_cell_count": len(common),
        "common_cell_indices": common,
        "raw_signed_per_s": (
            float(np.median(np.asarray(raw_expansions, dtype=np.float64)))
            if evaluable
            else None
        ),
        "raw_absolute_per_s": (
            float(
                np.median(
                    np.abs(np.asarray(raw_expansions, dtype=np.float64))
                )
            )
            if evaluable
            else None
        ),
        "signed_per_s": (
            float(np.median(np.asarray(expansions, dtype=np.float64)))
            if evaluable
            else None
        ),
        "absolute_per_s": (
            float(np.median(np.abs(np.asarray(expansions, dtype=np.float64))))
            if evaluable
            else None
        ),
        "raw_cells": path_cells["RAW_FINAL"],
        "cells": path_cells["COMPENSATED_FINAL"],
    }


def _reconstruct_local_fit(
    arrays: Mapping[str, np.ndarray],
    pair_index: int,
    dt_s: float,
    activated: Sequence[int],
    overlap_fraction: float,
) -> dict[str, Any]:
    if overlap_fraction < WARP_OVERLAP_FLOOR:
        return _early_warp_abstention_local_fit(arrays, pair_index)
    return _consensus_fit(arrays, pair_index, dt_s, activated)


def _safe_repo_file(root: Path, relative: str, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label}:PATH")
    require(not Path(relative).is_absolute(), f"{label}:ABSOLUTE_PATH")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise InvalidExecution(f"{label}:OUTSIDE_REPO") from error
    require(candidate.is_file(), f"{label}:MISSING")
    return candidate


def _binding_records(run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = run.get("bindings", [])
    if isinstance(value, Mapping):
        records: list[Mapping[str, Any]] = []
        for role, item in value.items():
            record = dict(_mapping(item, f"RUN_BINDING:{role}"))
            record.setdefault("role", role)
            records.append(record)
        return records
    require(isinstance(value, list), "RUN_BINDINGS")
    return [_mapping(item, "RUN_BINDING") for item in value]


def _verify_bound_file(
    root: Path,
    run: Mapping[str, Any],
    *,
    role: str,
    expected_relative: str | None = None,
) -> tuple[Path, str]:
    records = [
        record
        for record in _binding_records(run)
        if record.get("role") == role
    ]
    if records:
        require(len(records) == 1, f"RUN_BINDING_DUPLICATE:{role}")
        record = records[0]
        relative = record.get("path")
        digest = record.get("sha256")
    else:
        stem = role.lower()
        relative = run.get(f"{stem}_path")
        digest = run.get(f"{stem}_sha256")
    require(isinstance(relative, str), f"RUN_BINDING_PATH:{role}")
    require(
        isinstance(digest, str) and len(digest) == 64,
        f"RUN_BINDING_SHA:{role}",
    )
    if expected_relative is not None:
        require(relative == expected_relative, f"RUN_BINDING_RELATIVE:{role}")
    path = _safe_repo_file(root, relative, f"RUN_BINDING:{role}")
    require(sha256_file(path) == digest, f"RUN_BINDING_HASH:{role}")
    return path, digest


def _receipt_hash(
    receipt: Mapping[str, Any], filename: str, label: str
) -> str:
    files = receipt.get("files")
    if isinstance(files, Mapping) and filename in files:
        item = files[filename]
        if isinstance(item, Mapping):
            digest = item.get("sha256")
        else:
            digest = item
    else:
        stem = filename.replace(".", "_")
        digest = receipt.get(f"{stem}_sha256")
        if digest is None:
            digest = receipt.get(
                {
                    "pair_ledger.jsonl": "pair_ledger_sha256",
                    "primitives.npz": "primitives_sha256",
                    "cluster_metrics.json": "cluster_metrics_sha256",
                }[filename]
            )
    require(
        isinstance(digest, str) and len(digest) == 64,
        f"{label}:HASH_FIELD:{filename}",
    )
    return digest


def _cluster_paths(
    response_root: Path, cluster_id: str
) -> tuple[Path, Path, Path, Path]:
    cluster_dir = response_root / "clusters" / cluster_id
    require(cluster_dir.is_dir(), f"CLUSTER_DIRECTORY:{cluster_id}")
    resolved = cluster_dir.resolve()
    try:
        resolved.relative_to(response_root.resolve())
    except ValueError as error:
        raise InvalidExecution(f"CLUSTER_OUTSIDE_ROOT:{cluster_id}") from error
    ledger = resolved / "pair_ledger.jsonl"
    primitives = resolved / "primitives.npz"
    metrics = resolved / "cluster_metrics.json"
    receipt = resolved / "receipt.json"
    for path in (ledger, primitives, metrics, receipt):
        require(path.is_file(), f"CLUSTER_FILE:{cluster_id}:{path.name}")
        require(not path.is_symlink(), f"CLUSTER_SYMLINK:{cluster_id}:{path.name}")
    return ledger, primitives, metrics, receipt


def _compare_stored_source_family(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
    pair_index: int,
    replay: Mapping[str, np.ndarray],
) -> None:
    offsets = arrays[f"{prefix}_offsets"]
    left, right = map(int, offsets[pair_index : pair_index + 2])
    source_offsets = arrays[f"{prefix}_source_offsets"]
    source_left, source_right = map(
        int, source_offsets[pair_index : pair_index + 2]
    )
    replay_valid = np.asarray(replay["valid"], dtype=bool)
    require(
        len(replay_valid) == right - left,
        f"SOURCE_REPLAY_LENGTH:{prefix}:{pair_index}",
    )
    require(
        np.array_equal(
            arrays[f"{prefix}_source_valid"][left:right], replay_valid
        ),
        f"SOURCE_REPLAY_VALID:{prefix}:{pair_index}",
    )
    replay_index = np.flatnonzero(replay_valid).astype(np.int32)
    require(
        np.array_equal(
            arrays[f"{prefix}_source_index"][source_left:source_right],
            replay_index,
        ),
        f"SOURCE_REPLAY_INDEX:{prefix}:{pair_index}",
    )
    recorded = arrays[f"{prefix}_source_aligned"][
        source_left:source_right
    ]
    expected = np.asarray(replay["aligned_xy"], dtype=np.float64)[
        replay_valid
    ]
    require(
        recorded.shape == expected.shape
        and np.allclose(recorded, expected, rtol=0.0, atol=1e-12),
        f"SOURCE_REPLAY_ALIGNED:{prefix}:{pair_index}",
    )
    if prefix == "initial":
        comparisons = (
            ("initial_source_depth_m", "depth_m", 1e-12),
            ("initial_source_object_id", "object_id", 0.0),
            ("initial_source_world", "world_xyz", 1e-12),
            ("initial_source_current_pixel", "current_xy", 1e-12),
        )
        for array_name, replay_name, tolerance in comparisons:
            recorded_value = arrays[array_name][source_left:source_right]
            expected_value = np.asarray(replay[replay_name])[replay_valid]
            if tolerance == 0.0:
                matches = np.array_equal(recorded_value, expected_value)
            else:
                matches = np.allclose(
                    recorded_value,
                    expected_value,
                    rtol=0.0,
                    atol=tolerance,
                )
            require(
                recorded_value.shape == expected_value.shape and matches,
                f"SOURCE_REPLAY_{array_name}:{pair_index}",
            )


def _pair_reconstruction(
    arrays: Mapping[str, np.ndarray],
    row: Mapping[str, Any],
    pair_index: int,
    replay_pair: Mapping[str, Any],
) -> dict[str, Any]:
    require(row.get("pair_index") == pair_index, f"PAIR_ORDER:{pair_index}")
    dt_s = finite_number(row.get("dt_s"), f"PAIR_DT:{pair_index}")
    require(dt_s > 0.0, f"PAIR_DT_POSITIVE:{pair_index}")
    require(
        replay_pair.get("pair_index") == pair_index,
        f"REPLAY_PAIR_ORDER:{pair_index}",
    )
    previous_replay = replay_pair["previous"]
    current_replay = replay_pair["current"]
    replay_dt = finite_number(
        current_replay["pose"]["timestamp_s"], "REPLAY_CURRENT_TIMESTAMP"
    ) - finite_number(
        previous_replay["pose"]["timestamp_s"], "REPLAY_PREVIOUS_TIMESTAMP"
    )
    close(dt_s, replay_dt, f"REPLAY_DT:{pair_index}", atol=1e-12)
    close(
        finite_number(
            row.get("previous_timestamp_s"),
            f"PAIR_PREVIOUS_TIMESTAMP:{pair_index}",
        ),
        float(previous_replay["pose"]["timestamp_s"]),
        f"REPLAY_PREVIOUS_TIMESTAMP:{pair_index}",
        atol=1e-12,
    )
    close(
        finite_number(
            row.get("current_timestamp_s"),
            f"PAIR_CURRENT_TIMESTAMP:{pair_index}",
        ),
        float(current_replay["pose"]["timestamp_s"]),
        f"REPLAY_CURRENT_TIMESTAMP:{pair_index}",
        atol=1e-12,
    )

    initial_left = int(arrays["initial_offsets"][pair_index])
    initial_right = int(arrays["initial_offsets"][pair_index + 1])
    array_offsets = _mapping(
        row.get("array_offsets"), f"PAIR_ARRAY_OFFSETS:{pair_index}"
    )
    for name in ("initial", "accepted", "managed"):
        expected_offsets = [
            int(arrays[f"{name}_offsets"][pair_index]),
            int(arrays[f"{name}_offsets"][pair_index + 1]),
        ]
        require(
            array_offsets.get(name) == expected_offsets,
            f"PAIR_ARRAY_OFFSET:{pair_index}:{name}",
        )
    initial_previous = arrays["initial_previous"][
        initial_left:initial_right
    ].astype(np.float64, copy=False)
    warp_audit, boundary_mask, interior_mask, homography, warped_mask = (
        _independent_warp_audit(
            previous_replay["rgb"],
            current_replay["rgb"],
            previous_replay["valid"],
            current_replay["valid"],
            previous_replay["rotation"],
            current_replay["rotation"],
            np.asarray(replay_pair["generator"].K, dtype=np.float64),
        )
    )
    forward_left, forward_right = map(
        int,
        arrays["lk_forward_offsets"][pair_index : pair_index + 2],
    )
    forward_index = arrays["lk_forward_source_index"][
        forward_left:forward_right
    ]
    expected_mask_pass = np.zeros(initial_right - initial_left, dtype=bool)
    expected_mask_pass[forward_index] = _sample_mask(
        warped_mask,
        arrays["lk_forward_points"][forward_left:forward_right],
    )
    require(
        np.array_equal(
            arrays["lk_mask_pass"][initial_left:initial_right][
                forward_index
            ],
            expected_mask_pass[forward_index],
        ),
        f"LK_MASK_REPLAY:{pair_index}",
    )
    backward_left, backward_right = map(
        int,
        arrays["lk_backward_offsets"][pair_index : pair_index + 2],
    )
    backward_index = arrays["lk_backward_source_index"][
        backward_left:backward_right
    ]
    backward_error = np.zeros(initial_right - initial_left, dtype=np.float32)
    backward_error[backward_index] = arrays["lk_backward_error"][
        backward_left:backward_right
    ]
    expected_fb_pass = (
        arrays["lk_forward_available"][initial_left:initial_right]
        & arrays["lk_backward_available"][initial_left:initial_right]
        & (backward_error <= 1.0)
    )
    require(
        np.array_equal(
            arrays["lk_fb_pass"][initial_left:initial_right],
            expected_fb_pass,
        ),
        f"LK_FB_REPLAY:{pair_index}",
    )
    initial_source_result = _source_correspondence(
        replay_pair["generator"],
        initial_previous,
        previous_replay["scene"],
        current_replay["scene"],
        previous_replay["rotation"],
        current_replay["rotation"],
        previous_replay["translation"],
        current_replay["translation"],
        homography,
    )
    _compare_stored_source_family(
        arrays, "initial", pair_index, initial_source_result
    )
    initial_source = initial_source_result["aligned_xy"]
    initial_valid = initial_source_result["valid"]
    geometry = fit_pair(
        initial_previous[initial_valid],
        initial_source[initial_valid],
        dt_s,
    )
    geometry["source_valid_count"] = int(np.count_nonzero(initial_valid))
    geometry_audit_points = np.vstack(
        (
            initial_previous,
            np.asarray(
                [
                    [0.0, 0.0],
                    [WIDTH - 1.0, 0.0],
                    [0.0, HEIGHT - 1.0],
                    [WIDTH - 1.0, HEIGHT - 1.0],
                ],
                dtype=np.float64,
            ),
        )
    )
    roundtrip = _homography_points(
        _homography_points(geometry_audit_points, homography),
        np.linalg.inv(homography),
    )
    geometry["roundtrip_max_px"] = float(
        np.max(np.linalg.norm(roundtrip - geometry_audit_points, axis=1))
    )

    accepted_left = int(arrays["accepted_offsets"][pair_index])
    accepted_right = int(arrays["accepted_offsets"][pair_index + 1])
    accepted_previous = arrays["accepted_previous"][
        accepted_left:accepted_right
    ]
    accepted_current = arrays["accepted_current"][
        accepted_left:accepted_right
    ]
    accepted_source_result = _source_correspondence(
        replay_pair["generator"],
        accepted_previous,
        previous_replay["scene"],
        current_replay["scene"],
        previous_replay["rotation"],
        current_replay["rotation"],
        previous_replay["translation"],
        current_replay["translation"],
        homography,
    )
    _compare_stored_source_family(
        arrays, "accepted", pair_index, accepted_source_result
    )
    accepted = _source_error_fit(
        accepted_previous,
        accepted_current,
        accepted_source_result["aligned_xy"],
        accepted_source_result["valid"],
        dt_s,
    )

    managed_left = int(arrays["managed_offsets"][pair_index])
    managed_right = int(arrays["managed_offsets"][pair_index + 1])
    managed_previous = arrays["managed_previous"][managed_left:managed_right]
    managed_current = arrays["managed_current"][managed_left:managed_right]
    managed_source_result = _source_correspondence(
        replay_pair["generator"],
        managed_previous,
        previous_replay["scene"],
        current_replay["scene"],
        previous_replay["rotation"],
        current_replay["rotation"],
        previous_replay["translation"],
        current_replay["translation"],
        homography,
    )
    _compare_stored_source_family(
        arrays, "managed", pair_index, managed_source_result
    )
    managed = _source_error_fit(
        managed_previous,
        managed_current,
        managed_source_result["aligned_xy"],
        managed_source_result["valid"],
        dt_s,
    )
    managed_valid = managed_source_result["valid"]
    managed_error_endpoint = (
        managed_previous.astype(np.float64)
        + managed_current.astype(np.float64)
        - managed_source_result["aligned_xy"]
    )
    boundary_selected = managed_valid & _sample_mask(
        boundary_mask, managed_previous
    )
    interior_selected = managed_valid & _sample_mask(
        interior_mask, managed_previous
    )
    boundary_fit = fit_pair(
        managed_previous[boundary_selected].astype(np.float64, copy=False),
        managed_error_endpoint[boundary_selected],
        dt_s,
    )
    interior_fit = fit_pair(
        managed_previous[interior_selected].astype(np.float64, copy=False),
        managed_error_endpoint[interior_selected],
        dt_s,
    )
    activated_left, activated_right = map(
        int,
        arrays["activated_offsets"][pair_index : pair_index + 2],
    )
    activated = arrays["activated_cell_index"][
        activated_left:activated_right
    ]
    local_fit = _reconstruct_local_fit(
        arrays,
        pair_index,
        dt_s,
        activated.tolist(),
        float(warp_audit["overlap_fraction"]),
    )
    lk_slice = slice(initial_left, initial_right)
    consensus_left = int(
        arrays["consensus_offsets"][pair_index * GRID_CELLS * 2]
    )
    consensus_right = int(
        arrays["consensus_offsets"][(pair_index + 1) * GRID_CELLS * 2]
    )
    return {
        "pair_index": pair_index,
        "dt_s": dt_s,
        "geometry": geometry,
        "warp": warp_audit,
        "boundary": {
            "evaluable": (
                boundary_fit["evaluable"] and interior_fit["evaluable"]
            ),
            "boundary": boundary_fit,
            "interior": interior_fit,
            "initial_boundary_count": int(
                np.count_nonzero(
                    _sample_mask(boundary_mask, initial_previous)
                )
            ),
            "initial_interior_count": int(
                np.count_nonzero(
                    _sample_mask(interior_mask, initial_previous)
                )
            ),
            "accepted_boundary_count": int(
                np.count_nonzero(
                    _sample_mask(boundary_mask, accepted_previous)
                )
            ),
            "accepted_interior_count": int(
                np.count_nonzero(
                    _sample_mask(interior_mask, accepted_previous)
                )
            ),
            "managed_boundary_count": int(
                np.count_nonzero(
                    _sample_mask(boundary_mask, managed_previous)
                )
            ),
            "managed_interior_count": int(
                np.count_nonzero(
                    _sample_mask(interior_mask, managed_previous)
                )
            ),
        },
        "accepted": accepted,
        "managed": managed,
        "local_fit": local_fit,
        "coverage": {
            "initial_feature_count": initial_right - initial_left,
            "forward_valid_count": int(
                np.count_nonzero(arrays["lk_forward_available"][lk_slice])
            ),
            "backward_available_count": int(
                np.count_nonzero(arrays["lk_backward_available"][lk_slice])
            ),
            "fb_pass_count": int(
                np.count_nonzero(arrays["lk_fb_pass"][lk_slice])
            ),
            "mask_pass_count": int(
                np.count_nonzero(arrays["lk_mask_pass"][lk_slice])
            ),
            "accepted_track_count": accepted_right - accepted_left,
            "managed_track_count": managed_right - managed_left,
            "consensus_track_count": consensus_right - consensus_left,
            "evaluable_cell_count": int(
                local_fit["evaluable_cell_count"]
            ),
        },
    }


def _validate_early_warp_abstention_records(
    local_fit: Mapping[str, Any],
    r3: Mapping[str, Any],
    label: str,
) -> None:
    require(
        _bool(local_fit.get("evaluable"), f"{label}:LOCAL_FIT:EVALUABLE")
        is False
        and _bool(
            local_fit.get("numeric_reproduced"),
            f"{label}:LOCAL_NUMERIC_REPRODUCED",
        )
        is False
        and local_fit.get("reproduction_error_max_abs") is None
        and local_fit.get("common_cell_indices") == []
        and local_fit.get("signed_per_s") is None
        and local_fit.get("absolute_per_s") is None
        and local_fit.get("raw_cells") == []
        and local_fit.get("compensated_cells") == [],
        f"{label}:EARLY_WARP_LOCAL_FIT",
    )
    require(
        _bool(r3.get("evaluable"), f"{label}:R3:EVALUABLE") is False
        and r3.get("reason")
        == "ROTATION_WARP_VALID_COVERAGE_BELOW_0_75"
        and r3.get("signed_per_s") is None
        and r3.get("absolute_per_s") is None
        and r3.get("common_cell_count") is None,
        f"{label}:EARLY_WARP_R3",
    )


def _validate_and_compare_pair(
    row: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    pair_index: int,
) -> dict[str, Any]:
    label = f"PAIR:{pair_index}"
    geometry = _layer(row, "input_geometry")
    warp = _layer(row, "rotation_warp")
    boundary = _layer(row, "mask_boundary")
    flow = _layer(row, "sparse_lk_and_track_filtering")
    local_fit = _layer(row, "local_affine_and_final_aggregation")
    r3 = _layer(row, "r3_pair_row")
    coverage = _layer(row, "coverage")

    _compare_pair_reduction(
        geometry, recomputed["geometry"], f"{label}:GEOMETRY"
    )
    roundtrip = _required_number(
        geometry, "roundtrip_max_px", f"{label}:GEOMETRY"
    )
    require(roundtrip >= 0.0, f"{label}:GEOMETRY_ROUNDTRIP_RANGE")
    close(
        roundtrip,
        recomputed["geometry"]["roundtrip_max_px"],
        f"{label}:GEOMETRY_ROUNDTRIP_REPLAY",
        atol=1e-9,
    )

    warp_evaluable = _bool(
        warp.get("evaluable"), f"{label}:WARP:EVALUABLE"
    )
    warp_roundtrip = _optional_layer_number(
        warp, "coordinate_roundtrip_p99_px", f"{label}:WARP"
    )
    warp_interior = _optional_layer_number(
        warp, "interior_gray_absolute_p90_normalized", f"{label}:WARP"
    )
    warp_overlap = _optional_layer_number(
        warp, "overlap_fraction", f"{label}:WARP"
    )
    warp_interior_support = _optional_layer_number(
        warp, "interior_support_fraction", f"{label}:WARP"
    )
    if warp_evaluable:
        require(
            None
            not in (
                warp_roundtrip,
                warp_interior,
                warp_overlap,
                warp_interior_support,
            ),
            f"{label}:WARP_VALUES",
        )
        require(
            0.0 <= float(warp_overlap) <= 1.0
            and 0.0 <= float(warp_interior_support) <= 1.0
            and float(warp_interior) >= 0.0
            and float(warp_roundtrip) >= 0.0,
            f"{label}:WARP_RANGE",
        )
    else:
        require(
            all(
                value is None
                for value in (
                    warp_roundtrip,
                    warp_interior,
                )
            ),
            f"{label}:WARP_ABSTENTION_VALUES",
        )
        require(
            warp_overlap is not None
            and warp_interior_support is not None
            and 0.0 <= warp_overlap <= 1.0
            and 0.0 <= warp_interior_support <= 1.0,
            f"{label}:WARP_ABSTENTION_COVERAGE",
        )
    replay_warp = recomputed["warp"]
    require(
        warp_evaluable is bool(replay_warp["evaluable"]),
        f"{label}:WARP_REPLAY_EVALUABLE",
    )
    close(
        warp_roundtrip,
        replay_warp["coordinate_roundtrip_p99_px"],
        f"{label}:WARP_REPLAY_ROUNDTRIP",
        atol=1e-9,
    )
    close(
        warp_interior,
        replay_warp["interior_absolute_p90"],
        f"{label}:WARP_REPLAY_INTERIOR",
    )
    for recorded_key, replay_key in (
        ("interior_gray_signed_p50_normalized", "interior_signed_p50"),
        ("interior_gray_absolute_p50_normalized", "interior_absolute_p50"),
        ("interior_gray_absolute_p99_normalized", "interior_absolute_p99"),
        ("boundary_gray_absolute_p90_normalized", "boundary_absolute_p90"),
    ):
        close(
            _optional_layer_number(
                warp, recorded_key, f"{label}:WARP"
            ),
            replay_warp[replay_key],
            f"{label}:WARP_REPLAY:{recorded_key}",
        )
    close(
        warp_overlap,
        replay_warp["overlap_fraction"],
        f"{label}:WARP_REPLAY_OVERLAP",
        atol=1e-12,
    )
    close(
        warp_interior_support,
        replay_warp["interior_support_fraction"],
        f"{label}:WARP_REPLAY_INTERIOR_SUPPORT",
        atol=1e-12,
    )
    for key in (
        "previous_valid_pixel_count",
        "common_pixel_count",
        "interior_pixel_count",
        "boundary_pixel_count",
    ):
        replay_key = {
            "previous_valid_pixel_count": "previous_valid_count",
            "common_pixel_count": "common_count",
            "interior_pixel_count": "interior_count",
            "boundary_pixel_count": "boundary_count",
        }[key]
        require(
            warp.get(key) == replay_warp[replay_key],
            f"{label}:WARP_REPLAY_COUNT:{key}",
        )
    for key, replay_key in (
        ("previous_gray_sha256", "previous_gray_sha256"),
        ("warped_current_gray_sha256", "warped_gray_sha256"),
        ("previous_valid_sha256", "previous_valid_sha256"),
        ("warped_current_valid_sha256", "warped_valid_sha256"),
    ):
        require(
            warp.get(key) == replay_warp[replay_key],
            f"{label}:WARP_REPLAY_HASH:{key}",
        )

    replay_boundary = recomputed["boundary"]
    (
        joint_boundary_evaluable,
        boundary_evaluable,
        interior_evaluable,
    ) = _boundary_evaluable_states(
        boundary,
        replay_boundary,
        f"{label}:BOUNDARY",
    )
    boundary_absolute = _optional_layer_number(
        boundary,
        "boundary_absolute_per_s",
        f"{label}:BOUNDARY",
    )
    interior_absolute = _optional_layer_number(
        boundary,
        "interior_absolute_per_s",
        f"{label}:BOUNDARY",
    )
    require(
        (boundary_absolute is not None) is boundary_evaluable,
        f"{label}:BOUNDARY_VALUE",
    )
    require(
        (interior_absolute is not None) is interior_evaluable,
        f"{label}:INTERIOR_VALUE",
    )
    if boundary_absolute is not None:
        require(boundary_absolute >= 0.0, f"{label}:BOUNDARY_RANGE")
    if interior_absolute is not None:
        require(interior_absolute >= 0.0, f"{label}:INTERIOR_RANGE")
    boundary_recorded = {
        "evaluable": boundary_evaluable,
        "evaluable_cell_count": boundary.get(
            "boundary_evaluable_cell_count"
        ),
        "signed_per_s": boundary.get("boundary_signed_per_s"),
        "absolute_per_s": boundary.get("boundary_absolute_per_s"),
    }
    interior_recorded = {
        "evaluable": interior_evaluable,
        "evaluable_cell_count": boundary.get(
            "interior_evaluable_cell_count"
        ),
        "signed_per_s": boundary.get("interior_signed_per_s"),
        "absolute_per_s": boundary.get("interior_absolute_per_s"),
    }
    _compare_pair_reduction(
        boundary_recorded,
        replay_boundary["boundary"],
        f"{label}:BOUNDARY_REPLAY",
    )
    _compare_pair_reduction(
        interior_recorded,
        replay_boundary["interior"],
        f"{label}:INTERIOR_REPLAY",
    )
    for key in (
        "initial_boundary_count",
        "initial_interior_count",
        "accepted_boundary_count",
        "accepted_interior_count",
        "managed_boundary_count",
        "managed_interior_count",
    ):
        require(
            boundary.get(key) == replay_boundary[key],
            f"{label}:BOUNDARY_REPLAY_COUNT:{key}",
        )

    accepted_recorded = _mapping(
        flow.get("accepted"), f"{label}:FLOW_ACCEPTED"
    )
    managed_recorded = _mapping(
        flow.get("managed"), f"{label}:FLOW_MANAGED"
    )
    _compare_pair_reduction(
        accepted_recorded,
        recomputed["accepted"],
        f"{label}:FLOW_ACCEPTED",
    )
    _compare_pair_reduction(
        managed_recorded,
        recomputed["managed"],
        f"{label}:FLOW_MANAGED",
    )

    early_warp_abstention = bool(
        recomputed["local_fit"].get("early_warp_abstention", False)
    )
    if early_warp_abstention:
        _validate_early_warp_abstention_records(local_fit, r3, label)
    else:
        _compare_pair_reduction(
            local_fit,
            recomputed["local_fit"],
            f"{label}:LOCAL_FIT",
            cell_count_key="common_cell_count",
        )
        close(
            _optional_layer_number(
                local_fit, "raw_signed_per_s", f"{label}:LOCAL_FIT"
            ),
            recomputed["local_fit"]["raw_signed_per_s"],
            f"{label}:LOCAL_RAW_SIGNED",
        )
        close(
            _optional_layer_number(
                local_fit, "raw_absolute_per_s", f"{label}:LOCAL_FIT"
            ),
            recomputed["local_fit"]["raw_absolute_per_s"],
            f"{label}:LOCAL_RAW_ABSOLUTE",
        )
    for path_name, recorded_key, recomputed_key in (
        ("RAW_FINAL", "raw_cells", "raw_cells"),
        ("COMPENSATED_FINAL", "compensated_cells", "cells"),
    ):
        recorded_cells = local_fit.get(recorded_key)
        require(
            isinstance(recorded_cells, list)
            and len(recorded_cells)
            == (0 if early_warp_abstention else GRID_CELLS),
            f"{label}:LOCAL_CELLS:{path_name}",
        )
        for cell_index, (recorded_cell, recomputed_cell) in enumerate(
            zip(
                recorded_cells,
                recomputed["local_fit"][recomputed_key],
                strict=True,
            )
        ):
            recorded_cell = _mapping(
                recorded_cell,
                f"{label}:LOCAL_CELL:{path_name}:{cell_index}",
            )
            require(
                recorded_cell.get("path") == path_name
                and recorded_cell.get("cell_index") == cell_index
                and bool(recorded_cell.get("evaluable"))
                is bool(recomputed_cell["evaluable"])
                and recorded_cell.get("consensus_support_count")
                == recomputed_cell["support_count"],
                f"{label}:LOCAL_CELL_IDENTITY:{path_name}:{cell_index}",
            )
            require(
                recorded_cell.get("support_count")
                == recomputed_cell["support_count"]
                and recorded_cell.get("tracked_support_count")
                == recomputed_cell["tracked_support_count"]
                and recorded_cell.get("region")
                == recomputed_cell["region"]
                and recorded_cell.get("abstention_reason")
                == recomputed_cell["abstention_reason"],
                f"{label}:LOCAL_CELL_PROVENANCE:{path_name}:{cell_index}",
            )
            close(
                finite_number(
                    recorded_cell.get("confidence"),
                    f"{label}:LOCAL_CELL_CONFIDENCE",
                ),
                recomputed_cell["confidence"],
                f"{label}:LOCAL_CELL_CONFIDENCE:{path_name}:{cell_index}",
            )
            close(
                optional_number(
                    recorded_cell.get("expansion"),
                    f"{label}:LOCAL_NATIVE_EXPANSION",
                ),
                recomputed_cell["expansion_per_s"],
                f"{label}:LOCAL_NATIVE_EXPANSION:{path_name}:{cell_index}",
            )
            require(
                _bool(
                    recorded_cell.get("numeric_reproduced"),
                    f"{label}:LOCAL_CELL_REPRODUCED",
                ),
                f"{label}:LOCAL_CELL_REPRODUCED_FALSE:{path_name}:{cell_index}",
            )
            recorded_ols_expansion = optional_number(
                recorded_cell.get("ols_expansion_per_s"),
                f"{label}:LOCAL_CELL_EXPANSION",
            )
            close(
                recorded_ols_expansion,
                recomputed_cell["ols_audit"]["expansion_per_s"],
                f"{label}:LOCAL_CELL_EXPANSION:{path_name}:{cell_index}",
            )
            recorded_coefficients = recorded_cell.get("ols_coefficients")
            expected_coefficients = recomputed_cell["ols_audit"][
                "coefficients"
            ]
            if recorded_coefficients is None or expected_coefficients is None:
                require(
                    recorded_coefficients is None
                    and expected_coefficients is None,
                    f"{label}:LOCAL_CELL_COEFFICIENTS:{path_name}:{cell_index}",
                )
            else:
                require(
                    recorded_ols_expansion is not None,
                    f"{label}:LOCAL_CELL_COEFFICIENTS_EXPANSION:"
                    f"{path_name}:{cell_index}",
                )
                _validate_audit_coefficients(
                    recorded_coefficients,
                    float(recorded_ols_expansion),
                    cell_index,
                    f"{label}:LOCAL_CELL_COEFFICIENTS:"
                    f"{path_name}:{cell_index}",
                )
            close(
                optional_number(
                    recorded_cell.get("condition_number"),
                    f"{label}:LOCAL_CELL_CONDITION",
                ),
                recomputed_cell["condition_number"],
                f"{label}:LOCAL_CELL_CONDITION:{path_name}:{cell_index}",
            )
            close(
                optional_number(
                    recorded_cell.get("hull_fraction"),
                    f"{label}:LOCAL_CELL_HULL",
                ),
                recomputed_cell["hull_fraction"],
                f"{label}:LOCAL_CELL_HULL:{path_name}:{cell_index}",
            )
            close(
                optional_number(
                    recorded_cell.get(
                        "fit_residual_pixels_per_frame"
                    ),
                    f"{label}:LOCAL_CELL_RESIDUAL",
                ),
                recomputed_cell["median_fit_residual_px_per_frame"],
                f"{label}:LOCAL_CELL_RESIDUAL:{path_name}:{cell_index}",
            )
    if not early_warp_abstention:
        _compare_pair_reduction(
            r3,
            recomputed["local_fit"],
            f"{label}:R3_REPRODUCTION",
            cell_count_key="common_cell_count",
            signed_key="signed_per_s",
            absolute_key="absolute_per_s",
        )
    common = local_fit.get("common_cell_indices")
    require(
        isinstance(common, list)
        and common == recomputed["local_fit"]["common_cell_indices"],
        f"{label}:LOCAL_COMMON_CELLS",
    )
    numeric_reproduced = _bool(
        local_fit.get("numeric_reproduced"),
        f"{label}:LOCAL_NUMERIC_REPRODUCED",
    )
    require(
        numeric_reproduced is (not early_warp_abstention),
        f"{label}:LOCAL_NUMERIC_REPRODUCED_STATE",
    )
    reproduction_error = optional_number(
        local_fit.get("reproduction_error_max_abs"),
        f"{label}:LOCAL_REPRODUCTION_ERROR",
    )
    if early_warp_abstention:
        require(
            reproduction_error is None,
            f"{label}:EARLY_WARP_REPRODUCTION_ERROR",
        )
    elif local_fit.get("evaluable") is True:
        require(
            reproduction_error is not None
            and reproduction_error <= NUMERIC_ATOL,
            f"{label}:LOCAL_REPRODUCTION_ERROR_RANGE",
        )

    coverage_key_map = {
        "initial_features": "initial_feature_count",
        "accepted_tracks": "accepted_track_count",
        "managed_tracks": "managed_track_count",
        "common_cells": "evaluable_cell_count",
    }
    for key, recomputed_key in coverage_key_map.items():
        require(
            isinstance(coverage.get(key), int)
            and not isinstance(coverage.get(key), bool)
            and int(coverage[key]) >= 0,
            f"{label}:COVERAGE:{key}",
        )
        require(
            int(coverage[key])
            == int(recomputed["coverage"][recomputed_key]),
            f"{label}:COVERAGE_RECOMPUTED:{key}",
        )
    require(
        int(coverage.get("source_geometry_valid", -1))
        == int(recomputed["geometry"]["source_valid_count"]),
        f"{label}:COVERAGE_SOURCE_GEOMETRY",
    )
    flow_count_map = {
        "requested_count": "initial_feature_count",
        "forward_valid_count": "forward_valid_count",
        "backward_available_count": "backward_available_count",
        "fb_pass_count": "fb_pass_count",
        "mask_pass_count": "mask_pass_count",
        "accepted_count": "accepted_track_count",
        "managed_count": "managed_track_count",
    }
    for key, recomputed_key in flow_count_map.items():
        require(
            isinstance(flow.get(key), int)
            and not isinstance(flow.get(key), bool)
            and int(flow[key]) == int(recomputed["coverage"][recomputed_key]),
            f"{label}:FLOW_COUNT:{key}",
        )

    return {
        "pair_index": pair_index,
        "dt_s": recomputed["dt_s"],
        "geometry": {
            **recomputed["geometry"],
            "roundtrip_max_px": roundtrip,
        },
        "warp": {
            "evaluable": replay_warp["evaluable"],
            "coordinate_roundtrip_p99_px": replay_warp[
                "coordinate_roundtrip_p99_px"
            ],
            "interior_gray_absolute_p90": replay_warp[
                "interior_absolute_p90"
            ],
            "overlap_fraction": replay_warp["overlap_fraction"],
            "interior_support_fraction": replay_warp[
                "interior_support_fraction"
            ],
        },
        "boundary": {
            "boundary_evaluable": replay_boundary["boundary"][
                "evaluable"
            ],
            "interior_evaluable": replay_boundary["interior"]["evaluable"],
            "boundary_source_error_absolute_per_s": replay_boundary[
                "boundary"
            ]["absolute_per_s"],
            "interior_source_error_absolute_per_s": replay_boundary[
                "interior"
            ]["absolute_per_s"],
        },
        "flow": {
            "accepted": recomputed["accepted"],
            "managed": recomputed["managed"],
        },
        "local_fit": recomputed["local_fit"],
        "r3": recomputed["local_fit"],
        "coverage": dict(coverage),
    }


def _status(coverage_ok: bool, gate_pass: bool) -> str:
    if not coverage_ok:
        return "NOT_EVALUABLE"
    return "PASS" if gate_pass else "FAIL"


def _signed_absolute_reduction(
    pairs: Sequence[Mapping[str, Any]],
    *,
    evaluable: callable,
    signed: callable,
    absolute: callable,
    planned_pair_count: int,
) -> dict[str, Any]:
    signed_values: list[float] = []
    absolute_values: list[float] = []
    ordered_signed: list[float | None] = []
    for pair in pairs:
        if bool(evaluable(pair)):
            signed_value = finite_number(signed(pair), "REDUCE_SIGNED")
            absolute_value = finite_number(absolute(pair), "REDUCE_ABSOLUTE")
            require(absolute_value >= 0.0, "REDUCE_ABSOLUTE_RANGE")
            signed_values.append(signed_value)
            absolute_values.append(absolute_value)
            ordered_signed.append(signed_value)
        else:
            ordered_signed.append(None)
    streak = 0
    longest = 0
    trigger_count = 0
    for value in ordered_signed:
        if value is None:
            streak = 0
        elif value > LEAKAGE_THRESHOLD_PER_S:
            streak += 1
            longest = max(longest, streak)
            if streak >= 3:
                trigger_count += 1
        else:
            streak = 0
    return {
        "planned_pair_count": planned_pair_count,
        "evaluable_pair_count": len(signed_values),
        "evaluable_pair_fraction": (
            len(signed_values) / planned_pair_count
            if planned_pair_count
            else 0.0
        ),
        "signed_p50_per_s": hf7(signed_values, 0.5),
        "signed_p90_per_s": hf7(signed_values, 0.9),
        "absolute_p50_per_s": hf7(absolute_values, 0.5),
        "absolute_p90_per_s": hf7(absolute_values, 0.9),
        "three_pair_trigger_count": trigger_count,
        "three_pair_trigger_density_fixed": (
            trigger_count / planned_pair_count if planned_pair_count else 0.0
        ),
        "longest_positive_streak": longest,
    }


def reduce_cluster(
    pairs: Sequence[Mapping[str, Any]],
    planned_pair_count: int,
) -> dict[str, Any]:
    coverage_minimum = int(math.ceil(COVERAGE_FRACTION * planned_pair_count))
    require(
        coverage_minimum == (
            COVERAGE_MINIMUM if planned_pair_count == PAIR_COUNT else coverage_minimum
        ),
        "COVERAGE_MINIMUM",
    )

    geometry_reduction = _signed_absolute_reduction(
        pairs,
        evaluable=lambda pair: pair["geometry"]["evaluable"],
        signed=lambda pair: pair["geometry"]["signed_per_s"],
        absolute=lambda pair: pair["geometry"]["absolute_per_s"],
        planned_pair_count=planned_pair_count,
    )
    geometry_roundtrips = [
        float(pair["geometry"]["roundtrip_max_px"])
        for pair in pairs
        if pair["geometry"]["evaluable"]
    ]
    geometry_roundtrip_max = (
        max(geometry_roundtrips) if geometry_roundtrips else None
    )
    geometry_coverage = (
        geometry_reduction["evaluable_pair_count"] >= coverage_minimum
    )
    geometry_gate = (
        geometry_roundtrip_max is not None
        and geometry_roundtrip_max <= GEOMETRY_ROUNDTRIP_MAX_PX
        and geometry_reduction["absolute_p90_per_s"] is not None
        and geometry_reduction["absolute_p90_per_s"]
        <= LEAKAGE_THRESHOLD_PER_S
    )
    geometry_layer = {
        **geometry_reduction,
        "roundtrip_max_px": geometry_roundtrip_max,
        "coverage_minimum": coverage_minimum,
        "status": _status(geometry_coverage, geometry_gate),
    }

    warp_evaluable = [
        pair for pair in pairs if pair["warp"]["evaluable"]
    ]
    warp_count = len(warp_evaluable)
    warp_coordinate = [
        float(pair["warp"]["coordinate_roundtrip_p99_px"])
        for pair in warp_evaluable
    ]
    warp_gray = [
        float(pair["warp"]["interior_gray_absolute_p90"])
        for pair in warp_evaluable
    ]
    warp_overlap = [
        float(pair["warp"]["overlap_fraction"]) for pair in warp_evaluable
    ]
    warp_support = [
        float(pair["warp"]["interior_support_fraction"])
        for pair in warp_evaluable
    ]
    warp_layer = {
        "planned_pair_count": planned_pair_count,
        "evaluable_pair_count": warp_count,
        "evaluable_pair_fraction": warp_count / planned_pair_count,
        "coverage_minimum": coverage_minimum,
        "coordinate_roundtrip_pair_p99_cluster_p90_px": hf7(
            warp_coordinate, 0.9
        ),
        "interior_gray_absolute_pair_p90_cluster_p90": hf7(
            warp_gray, 0.9
        ),
        "overlap_fraction_cluster_p10": hf7(warp_overlap, 0.1),
        "interior_support_fraction_cluster_p10": hf7(warp_support, 0.1),
    }
    warp_gate = (
        warp_layer["coordinate_roundtrip_pair_p99_cluster_p90_px"]
        is not None
        and warp_layer["coordinate_roundtrip_pair_p99_cluster_p90_px"]
        <= WARP_ROUNDTRIP_P99_MAX_PX
        and warp_layer["interior_gray_absolute_pair_p90_cluster_p90"]
        is not None
        and warp_layer["interior_gray_absolute_pair_p90_cluster_p90"]
        <= GRAY_RESIDUAL_THRESHOLD
        and warp_layer["overlap_fraction_cluster_p10"] is not None
        and warp_layer["overlap_fraction_cluster_p10"] >= 0.75
        and warp_layer["interior_support_fraction_cluster_p10"] is not None
        and warp_layer["interior_support_fraction_cluster_p10"] >= 0.75
    )
    warp_layer["status"] = _status(
        warp_count >= coverage_minimum, warp_gate
    )

    boundary_individually_evaluable = [
        pair for pair in pairs if pair["boundary"]["boundary_evaluable"]
    ]
    interior_individually_evaluable = [
        pair for pair in pairs if pair["boundary"]["interior_evaluable"]
    ]
    boundary_matched = [
        pair
        for pair in pairs
        if (
            pair["boundary"]["boundary_evaluable"]
            and pair["boundary"]["interior_evaluable"]
        )
    ]
    boundary_values = [
        float(pair["boundary"]["boundary_source_error_absolute_per_s"])
        for pair in boundary_matched
    ]
    interior_values = [
        float(pair["boundary"]["interior_source_error_absolute_per_s"])
        for pair in boundary_matched
    ]
    matched_count = len(boundary_matched)
    boundary_p90 = hf7(boundary_values, 0.9)
    interior_p90 = hf7(interior_values, 0.9)
    boundary_coverage = matched_count >= coverage_minimum
    if not boundary_coverage:
        boundary_status = "NOT_EVALUABLE"
        boundary_failure_mode = None
    elif (
        boundary_p90 is not None
        and interior_p90 is not None
        and boundary_p90 <= LEAKAGE_THRESHOLD_PER_S
        and interior_p90 <= LEAKAGE_THRESHOLD_PER_S
    ):
        boundary_status = "PASS"
        boundary_failure_mode = None
    elif (
        boundary_p90 is not None
        and interior_p90 is not None
        and boundary_p90 > LEAKAGE_THRESHOLD_PER_S
        and interior_p90 <= LEAKAGE_THRESHOLD_PER_S
    ):
        boundary_status = "FAIL"
        boundary_failure_mode = "BOUNDARY_ONLY"
    else:
        boundary_status = "FAIL"
        boundary_failure_mode = "MASK_NONSEPARABLE"
    boundary_layer = {
        "planned_pair_count": planned_pair_count,
        "matched_evaluable_pair_count": matched_count,
        "matched_evaluable_pair_fraction": (
            matched_count / planned_pair_count
            if planned_pair_count
            else 0.0
        ),
        "boundary_individually_evaluable_pair_count": len(
            boundary_individually_evaluable
        ),
        "interior_individually_evaluable_pair_count": len(
            interior_individually_evaluable
        ),
        "comparison_pair_set": "MATCHED_BOUNDARY_AND_INTERIOR",
        "coverage_minimum": coverage_minimum,
        "boundary_source_error_absolute_p90_per_s": boundary_p90,
        "interior_source_error_absolute_p90_per_s": interior_p90,
        "status": boundary_status,
        "failure_mode": boundary_failure_mode,
    }

    accepted_reduction = _signed_absolute_reduction(
        pairs,
        evaluable=lambda pair: pair["flow"]["accepted"]["evaluable"],
        signed=lambda pair: pair["flow"]["accepted"]["signed_per_s"],
        absolute=lambda pair: pair["flow"]["accepted"]["absolute_per_s"],
        planned_pair_count=planned_pair_count,
    )
    managed_reduction = _signed_absolute_reduction(
        pairs,
        evaluable=lambda pair: pair["flow"]["managed"]["evaluable"],
        signed=lambda pair: pair["flow"]["managed"]["signed_per_s"],
        absolute=lambda pair: pair["flow"]["managed"]["absolute_per_s"],
        planned_pair_count=planned_pair_count,
    )
    flow_coverage = (
        accepted_reduction["evaluable_pair_count"] >= coverage_minimum
        and managed_reduction["evaluable_pair_count"] >= coverage_minimum
    )
    flow_gate = (
        accepted_reduction["absolute_p90_per_s"] is not None
        and managed_reduction["absolute_p90_per_s"] is not None
        and accepted_reduction["absolute_p90_per_s"]
        <= LEAKAGE_THRESHOLD_PER_S
        and managed_reduction["absolute_p90_per_s"]
        <= LEAKAGE_THRESHOLD_PER_S
    )
    flow_layer = {
        "accepted": accepted_reduction,
        "managed": managed_reduction,
        "coverage_minimum": coverage_minimum,
        "status": _status(flow_coverage, flow_gate),
    }

    local_reduction = _signed_absolute_reduction(
        pairs,
        evaluable=lambda pair: pair["local_fit"]["evaluable"],
        signed=lambda pair: pair["local_fit"]["signed_per_s"],
        absolute=lambda pair: pair["local_fit"]["absolute_per_s"],
        planned_pair_count=planned_pair_count,
    )
    local_coverage = (
        local_reduction["evaluable_pair_count"] >= coverage_minimum
    )
    local_gate = (
        local_reduction["absolute_p90_per_s"] is not None
        and local_reduction["absolute_p90_per_s"]
        <= LEAKAGE_THRESHOLD_PER_S
    )
    local_layer = {
        **local_reduction,
        "coverage_minimum": coverage_minimum,
        "independent_consensus_ols_reproduction": "PASS",
        "status": _status(local_coverage, local_gate),
    }

    final_precondition = {
        "coverage_ok": local_coverage,
        "absolute_p90_per_s": local_reduction["absolute_p90_per_s"],
        "boundary_failed": (
            local_coverage
            and local_reduction["absolute_p90_per_s"] is not None
            and local_reduction["absolute_p90_per_s"]
            > LEAKAGE_THRESHOLD_PER_S
        ),
    }
    layers = {
        "INPUT_GEOMETRY": geometry_layer,
        "ROTATION_WARP": warp_layer,
        "MASK_BOUNDARY": boundary_layer,
        "SPARSE_LK_AND_TRACK_FILTERING": flow_layer,
        "LOCAL_AFFINE_AND_FINAL_AGGREGATION": local_layer,
    }
    route = route_cluster(layers, final_precondition)
    return {
        "pair_count": planned_pair_count,
        "coverage_minimum": coverage_minimum,
        "layers": layers,
        "final_precondition": final_precondition,
        "route": route,
    }


def route_cluster(
    layers: Mapping[str, Mapping[str, Any]],
    final_precondition: Mapping[str, Any],
) -> str:
    ordered = (
        "INPUT_GEOMETRY",
        "ROTATION_WARP",
        "MASK_BOUNDARY",
        "SPARSE_LK_AND_TRACK_FILTERING",
        "LOCAL_AFFINE_AND_FINAL_AGGREGATION",
    )
    if (
        not bool(final_precondition.get("coverage_ok"))
        or not bool(final_precondition.get("boundary_failed"))
        or any(layers[name].get("status") == "NOT_EVALUABLE" for name in ordered)
    ):
        return "NOT_EVALUABLE"

    geometry = layers["INPUT_GEOMETRY"]["status"]
    warp = layers["ROTATION_WARP"]["status"]
    boundary = layers["MASK_BOUNDARY"]
    flow = layers["SPARSE_LK_AND_TRACK_FILTERING"]["status"]
    local_fit = layers["LOCAL_AFFINE_AND_FINAL_AGGREGATION"]["status"]

    if geometry == "FAIL":
        return "LEAKAGE_ALREADY_PRESENT_IN_INPUT_GEOMETRY"
    if warp == "FAIL":
        if boundary["status"] == "PASS" and flow == "PASS":
            return "LEAKAGE_FIRST_VISIBLE_AT_WARP"
        return "MULTIPLE_SOURCES_NOT_SEPARABLE"
    if boundary["status"] == "FAIL":
        if boundary.get("failure_mode") == "BOUNDARY_ONLY" and flow == "PASS":
            return "LEAKAGE_FIRST_VISIBLE_AT_MASK_BOUNDARY"
        return "MULTIPLE_SOURCES_NOT_SEPARABLE"
    if flow == "FAIL":
        return "LEAKAGE_FIRST_VISIBLE_AT_FLOW"
    if local_fit == "FAIL":
        return "LEAKAGE_FIRST_VISIBLE_AT_LOCAL_FIT"
    return "NOT_EVALUABLE"


def _resolve_response_root(input_root: Path) -> Path:
    candidate = input_root.resolve()
    if (candidate / "run_receipt.json").is_file():
        return candidate
    if (candidate / "response" / "run_receipt.json").is_file():
        return candidate / "response"
    raise InvalidExecution("RUN_RECEIPT_MISSING")


def _activation_binding_hash(
    activation: Mapping[str, Any], role: str
) -> str:
    records = [
        item
        for item in activation.get("bindings", [])
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    require(len(records) == 1, f"ACTIVATION_BINDING:{role}")
    digest = records[0].get("sha256")
    require(
        isinstance(digest, str) and len(digest) == 64,
        f"ACTIVATION_BINDING_HASH:{role}",
    )
    return digest


def _run_cluster_records(
    run: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records = run.get("clusters")
    require(isinstance(records, list), "RUN_CLUSTERS")
    result: dict[str, Mapping[str, Any]] = {}
    for value in records:
        record = _mapping(value, "RUN_CLUSTER_RECORD")
        cluster_id = record.get("cluster_id")
        require(
            isinstance(cluster_id, str) and cluster_id not in result,
            "RUN_CLUSTER_ID",
        )
        result[cluster_id] = record
    return result


def _verify_executable_spec_governance(
    spec: Mapping[str, Any],
) -> None:
    expected_authority = {
        "separate_r1_activation_exists": True,
        "this_spec_adds_execution_authority": False,
        "formal_one_shot_may_start_from_this_spec_alone": False,
        "required_status_before_formal_launch": SPEC_REQUIRED_STATUS,
        "stage_b_retry_replacement_or_reseed": False,
        "r3_modification": False,
        "single_variable_repair": False,
        "formal_480_plus_16": False,
    }
    require(
        spec.get("schema") == SPEC_SCHEMA
        and spec.get("protocol_id") == PROTOCOL_ID
        and spec.get("task_id") == TASK_ID
        and spec.get("status") == EXECUTABLE_SPEC_STATUS
        and spec.get("terminal") == SPEC_TERMINAL
        and spec.get("claim_ceiling") == SPEC_CLAIM_CEILING
        and spec.get("authority") == expected_authority,
        "SPEC_GOVERNANCE",
    )


def _verify_all_spec_bindings(
    root: Path, spec: Mapping[str, Any]
) -> dict[str, str]:
    """Independently hash-check every frozen file named by the spec."""

    bindings = _mapping(spec.get("bindings"), "SPEC_BINDINGS")
    require(bool(bindings), "SPEC_BINDINGS_EMPTY")
    require(
        set(bindings)
        == set(SPEC_DIRECT_BINDING_PATHS)
        | {"frozen_execution_dependencies"},
        "SPEC_BINDING_KEYSET",
    )
    for name, expected_path in SPEC_DIRECT_BINDING_PATHS.items():
        record = _mapping(
            bindings.get(name), f"SPEC_DIRECT_BINDING:{name}"
        )
        require(
            record.get("path") == expected_path,
            f"SPEC_DIRECT_BINDING_PATH:{name}",
        )
    dependencies = bindings.get("frozen_execution_dependencies")
    require(
        isinstance(dependencies, list)
        and len(dependencies) == len(SPEC_FROZEN_DEPENDENCY_PATHS)
        and [
            _mapping(item, "SPEC_FROZEN_DEPENDENCY_PATH").get("path")
            for item in dependencies
        ]
        == list(SPEC_FROZEN_DEPENDENCY_PATHS),
        "SPEC_FROZEN_DEPENDENCY_PATHSET",
    )
    records: list[tuple[str, Mapping[str, Any]]] = []
    for name, value in bindings.items():
        if name == "frozen_execution_dependencies":
            require(
                isinstance(value, list) and bool(value),
                "SPEC_FROZEN_DEPENDENCIES",
            )
            for index, item in enumerate(value):
                records.append(
                    (
                        f"{name}:{index}",
                        _mapping(
                            item, f"SPEC_FROZEN_DEPENDENCY:{index}"
                        ),
                    )
                )
        else:
            records.append(
                (name, _mapping(value, f"SPEC_BINDING:{name}"))
            )
    hashes: dict[str, str] = {}
    for name, record in records:
        relative = record.get("path")
        digest = record.get("sha256")
        require(
            isinstance(relative, str)
            and isinstance(digest, str)
            and len(digest) == 64
            and relative not in hashes,
            f"SPEC_BINDING_SCHEMA:{name}",
        )
        path = _safe_repo_file(root, relative, f"SPEC_BINDING:{name}")
        require(
            sha256_file(path) == digest,
            f"SPEC_BINDING_HASH:{name}",
        )
        hashes[relative] = digest
    return hashes


def _verify_formal_readiness_bindings(
    root: Path,
    claim: Mapping[str, Any],
    run: Mapping[str, Any],
    activation: Mapping[str, Any],
) -> None:
    """Recheck spec, ready receipt, host receipt and validator self-binding."""

    require(
        claim.get("executable_spec_path") == SPEC_RELATIVE,
        "CLAIM_SPEC_PATH",
    )
    spec_path = _safe_repo_file(root, SPEC_RELATIVE, "CLAIM_SPEC")
    spec_digest = sha256_file(spec_path)
    require(
        claim.get("executable_spec_sha256") == spec_digest
        and run.get("executable_spec_sha256") == spec_digest,
        "CLAIM_SPEC_HASH",
    )
    spec = load_json(spec_path)
    _verify_executable_spec_governance(spec)
    _verify_all_spec_bindings(root, spec)
    implementations = _mapping(
        spec.get("implementation_bindings"),
        "SPEC_IMPLEMENTATION_BINDINGS",
    )
    require(
        implementations.get("status") == IMPLEMENTATION_BINDINGS_STATUS,
        "SPEC_IMPLEMENTATION_STATUS",
    )
    roles = (
        "runner",
        "independent_execution_validator",
        "test_suite",
    )
    require(
        set(implementations) == {"status", "finalization_rule", *roles}
        and isinstance(implementations.get("finalization_rule"), str)
        and bool(implementations.get("finalization_rule")),
        "SPEC_IMPLEMENTATION_KEYSET",
    )
    expected_implementations: dict[str, dict[str, str]] = {}
    for role in roles:
        binding = _mapping(
            implementations.get(role),
            f"SPEC_IMPLEMENTATION:{role}",
        )
        relative = binding.get("path")
        digest = binding.get("sha256")
        require(
            isinstance(relative, str)
            and isinstance(digest, str)
            and len(digest) == 64,
            f"SPEC_IMPLEMENTATION_SCHEMA:{role}",
        )
        path = _safe_repo_file(
            root, relative, f"SPEC_IMPLEMENTATION:{role}"
        )
        require(
            sha256_file(path) == digest,
            f"SPEC_IMPLEMENTATION_HASH:{role}",
        )
        expected_implementations[role] = {
            "path": relative,
            "sha256": digest,
        }
    validator_path = Path(__file__).resolve()
    validator_relative = validator_path.relative_to(root).as_posix()
    require(
        expected_implementations["runner"]
        == {
            "path": RUNNER_RELATIVE,
            "sha256": claim.get("runner_source_sha256"),
        }
        and expected_implementations["independent_execution_validator"]
        == {
            "path": validator_relative,
            "sha256": sha256_file(validator_path),
        }
        and expected_implementations["test_suite"]["path"]
        == TEST_SUITE_RELATIVE,
        "SPEC_IMPLEMENTATION_SELF_BINDINGS",
    )

    ready_relative = claim.get("implementation_ready_receipt_path")
    ready_digest = claim.get("implementation_ready_receipt_sha256")
    require(
        isinstance(ready_relative, str)
        and isinstance(ready_digest, str)
        and len(ready_digest) == 64,
        "CLAIM_READY_BINDING",
    )
    ready_path = _safe_repo_file(root, ready_relative, "CLAIM_READY")
    require(
        sha256_file(ready_path) == ready_digest
        and run.get("implementation_ready_receipt_sha256")
        == ready_digest,
        "CLAIM_READY_HASH",
    )
    ready = load_json(ready_path)
    require(
        ready.get("schema") == IMPLEMENTATION_READY_SCHEMA
        and ready.get("protocol_id") == PROTOCOL_ID
        and ready.get("task_id") == TASK_ID
        and ready.get("status") == "PASS"
        and ready.get("terminal") == IMPLEMENTATION_READY_TERMINAL
        and ready.get("formal_authority_consumed") is False
        and ready.get("scientific_interpretation") is False
        and ready.get("spec_path") == SPEC_RELATIVE
        and ready.get("spec_sha256") == spec_digest
        and ready.get("implementation_bindings")
        == expected_implementations,
        "READY_RECEIPT",
    )

    pilot_binding = _mapping(
        ready.get("pilot_w1_w4_equivalence"),
        "READY_PILOT",
    )
    pilot_relative = pilot_binding.get("receipt_path")
    pilot_digest = pilot_binding.get("receipt_sha256")
    require(
        isinstance(pilot_relative, str)
        and isinstance(pilot_digest, str)
        and len(pilot_digest) == 64,
        "READY_PILOT_BINDING_SCHEMA",
    )
    pilot_path = _safe_repo_file(
        root,
        pilot_relative,
        "READY_PILOT_RECEIPT",
    )
    pilot_parent = (root / PILOT_PARENT_RELATIVE).resolve()
    require(
        pilot_binding.get("status") == "PASS"
        and pilot_binding.get("workers") == [1, WORKERS]
        and pilot_path.is_relative_to(pilot_parent)
        and sha256_file(pilot_path) == pilot_digest,
        "READY_PILOT_BINDING",
    )
    pilot_receipt = load_json(pilot_path)
    pilot_checks = pilot_receipt.get("checks")
    require(
        pilot_receipt.get("schema") == PILOT_EQUIVALENCE_SCHEMA
        and pilot_receipt.get("protocol_id") == PROTOCOL_ID
        and pilot_receipt.get("task_id") == TASK_ID
        and pilot_receipt.get("valid") is True
        and pilot_receipt.get("terminal") == PILOT_EQUIVALENCE_TERMINAL
        and pilot_receipt.get("cluster_count") == PILOT_CLUSTER_COUNT
        and pilot_receipt.get("formal_authority_consumed") is False
        and pilot_receipt.get("scientific_interpretation") is False
        and pilot_receipt.get("runner_source_sha256")
        == expected_implementations["runner"]["sha256"]
        and pilot_receipt.get("validator_source_sha256")
        == expected_implementations[
            "independent_execution_validator"
        ]["sha256"]
        and pilot_receipt.get("semantic_payload_sha256")
        == pilot_binding.get("semantic_payload_sha256")
        and pilot_receipt.get("pilot_manifest_sha256")
        == pilot_binding.get("pilot_manifest_sha256")
        and isinstance(pilot_checks, Mapping)
        and bool(pilot_checks)
        and all(value == "PASS" for value in pilot_checks.values()),
        "READY_PILOT_RECEIPT",
    )
    pilot_paths: dict[str, Path] = {}
    for key in ("w1_root", "w4_root", "pilot_manifest_path"):
        relative = pilot_receipt.get(key)
        require(isinstance(relative, str), f"READY_PILOT_PATH:{key}")
        require(
            not Path(relative).is_absolute(),
            f"READY_PILOT_ABSOLUTE_PATH:{key}",
        )
        candidate = (root / relative).resolve()
        require(
            candidate.is_relative_to(pilot_parent),
            f"READY_PILOT_PATH_SCOPE:{key}",
        )
        pilot_paths[key] = candidate
    require(
        pilot_receipt.get("w1_root") != pilot_receipt.get("w4_root"),
        "READY_PILOT_DISTINCT_RUNS",
    )
    require(
        pilot_paths["pilot_manifest_path"].is_file()
        and sha256_file(pilot_paths["pilot_manifest_path"])
        == pilot_receipt.get("pilot_manifest_sha256"),
        "READY_PILOT_MANIFEST_HASH",
    )
    pilot_evidence = _mapping(
        _mapping(
            spec.get("pilot_w1_w4_equivalence"),
            "SPEC_PILOT_CONTRACT",
        ).get("frozen_evidence"),
        "SPEC_PILOT_EVIDENCE",
    )
    require(
        pilot_evidence.get("equivalence_receipt_path") == pilot_relative
        and pilot_evidence.get("equivalence_receipt_sha256")
        == pilot_digest
        and pilot_evidence.get("semantic_payload_sha256")
        == pilot_receipt.get("semantic_payload_sha256")
        and pilot_evidence.get("pilot_manifest_path")
        == pilot_receipt.get("pilot_manifest_path")
        and pilot_evidence.get("pilot_manifest_sha256")
        == pilot_receipt.get("pilot_manifest_sha256")
        and pilot_evidence.get("w1_root") == pilot_receipt.get("w1_root")
        and pilot_evidence.get("w4_root") == pilot_receipt.get("w4_root")
        and pilot_evidence.get("status") == PILOT_EQUIVALENCE_TERMINAL
        and pilot_evidence.get("scientific_interpretation") is False,
        "SPEC_PILOT_EVIDENCE_BINDING",
    )
    pilot_disjoint_relative = pilot_evidence.get(
        "pilot_disjoint_receipt_path"
    )
    pilot_disjoint_digest = pilot_evidence.get(
        "pilot_disjoint_receipt_sha256"
    )
    require(
        isinstance(pilot_disjoint_relative, str)
        and isinstance(pilot_disjoint_digest, str)
        and len(pilot_disjoint_digest) == 64,
        "SPEC_PILOT_DISJOINT_SCHEMA",
    )
    pilot_disjoint_path = _safe_repo_file(
        root, pilot_disjoint_relative, "SPEC_PILOT_DISJOINT"
    )
    require(
        pilot_disjoint_path.is_relative_to(pilot_parent)
        and sha256_file(pilot_disjoint_path) == pilot_disjoint_digest,
        "SPEC_PILOT_DISJOINT_HASH",
    )
    for prefix, root_key in (("w1", "w1_root"), ("w4", "w4_root")):
        pilot_run_root = pilot_paths[root_key]
        require(
            pilot_run_root.is_dir(),
            f"SPEC_PILOT_RUN_ROOT:{prefix}",
        )
        for filename, digest_key in (
            ("run_receipt.json", f"{prefix}_run_receipt_sha256"),
            (
                "independent_validation_receipt.json",
                f"{prefix}_independent_validation_receipt_sha256",
            ),
        ):
            artifact = pilot_run_root / filename
            digest = pilot_evidence.get(digest_key)
            require(
                artifact.is_file()
                and isinstance(digest, str)
                and sha256_file(artifact) == digest,
                f"SPEC_PILOT_ARTIFACT:{prefix}:{filename}",
            )

    test_binding = _mapping(
        ready.get("preformal_test_gate"), "READY_TEST"
    )
    test_relative = test_binding.get("receipt_path")
    test_digest = test_binding.get("receipt_sha256")
    require(
        isinstance(test_relative, str)
        and isinstance(test_digest, str)
        and len(test_digest) == 64,
        "READY_TEST_BINDING_SCHEMA",
    )
    test_path = _safe_repo_file(
        root,
        test_relative,
        "READY_TEST_RECEIPT",
    )
    require(
        test_binding.get("status") == "PASS"
        and test_path.is_relative_to(pilot_parent)
        and sha256_file(test_path) == test_digest,
        "READY_TEST_BINDING",
    )
    test_receipt = load_json(test_path)
    test_count = test_receipt.get("test_count")
    failure_count = test_receipt.get("failure_count")
    test_checks = test_receipt.get("checks")
    require(
        test_receipt.get("schema") == PREFORMAL_TEST_SCHEMA
        and test_receipt.get("protocol_id") == PROTOCOL_ID
        and test_receipt.get("task_id") == TASK_ID
        and test_receipt.get("status") == "PASS"
        and test_receipt.get("terminal") == PREFORMAL_TEST_TERMINAL
        and isinstance(test_count, int)
        and not isinstance(test_count, bool)
        and test_count >= 23
        and isinstance(failure_count, int)
        and not isinstance(failure_count, bool)
        and failure_count == 0
        and test_receipt.get("formal_authority_consumed") is False
        and test_receipt.get("formal_output_root_access") is False
        and test_receipt.get("sealed_input_access") is False
        and test_receipt.get("implementation_bindings")
        == expected_implementations
        and test_count == test_binding.get("test_count")
        and expected_implementations["test_suite"]["sha256"]
        == test_binding.get("test_suite_sha256")
        and isinstance(test_checks, Mapping)
        and bool(test_checks)
        and all(value == "PASS" for value in test_checks.values())
        and {
            "unit_and_mutation_suite",
            "observation_hook_transparency",
            "postclaim_failure_closure",
            "primitive_provenance_mutations",
        }.issubset(test_checks),
        "READY_TEST_RECEIPT",
    )
    test_evidence = _mapping(
        _mapping(
            spec.get("preformal_test_gate"),
            "SPEC_PREFORMAL_CONTRACT",
        ).get("frozen_evidence"),
        "SPEC_PREFORMAL_EVIDENCE",
    )
    require(
        test_evidence.get("receipt_path") == test_relative
        and test_evidence.get("receipt_sha256") == test_digest
        and test_evidence.get("test_count") == test_count
        and test_evidence.get("failure_count") == failure_count
        and test_evidence.get("terminal") == PREFORMAL_TEST_TERMINAL,
        "SPEC_PREFORMAL_EVIDENCE_BINDING",
    )

    host_binding = _mapping(
        ready.get("host_preflight"), "READY_HOST"
    )
    host_relative = claim.get("host_preflight_receipt_path")
    host_digest = claim.get("host_preflight_receipt_sha256")
    require(
        isinstance(host_relative, str)
        and isinstance(host_digest, str)
        and host_binding
        == {"path": host_relative, "sha256": host_digest},
        "READY_HOST_BINDING",
    )
    host_path = _safe_repo_file(root, host_relative, "READY_HOST_RECEIPT")
    require(
        sha256_file(host_path) == host_digest
        and run.get("host_preflight_receipt_sha256") == host_digest,
        "READY_HOST_HASH",
    )
    host = load_json(host_path)
    guarded_launcher = _mapping(
        host.get("guarded_launcher"), "HOST_GUARDED_LAUNCHER"
    )
    guarded_relative = guarded_launcher.get("script")
    guarded_digest = guarded_launcher.get("sha256")
    require(
        guarded_relative == GUARDED_HOST_LAUNCHER_RELATIVE
        and isinstance(guarded_digest, str)
        and len(guarded_digest) == 64,
        "HOST_GUARDED_LAUNCHER_BINDING",
    )
    guarded_path = _safe_repo_file(
        root,
        guarded_relative,
        "HOST_GUARDED_LAUNCHER",
    )
    host_workload = _mapping(host.get("workload"), "HOST_WORKLOAD")
    host_pilot = _mapping(host.get("pilot"), "HOST_PILOT")
    host_scheduler = _mapping(host.get("scheduler"), "HOST_SCHEDULER")
    host_progress = _mapping(host.get("progress"), "HOST_PROGRESS")
    host_terminal = _mapping(host.get("terminal"), "HOST_TERMINAL")
    host_formal = _mapping(host.get("formal"), "HOST_FORMAL")
    required_progress_fields = {
        "phase",
        "completed_units",
        "total_units",
        "throughput",
        "eta_seconds",
        "last_progress_at",
        "status",
    }
    expected_root = FORMAL_ROOT_RELATIVE
    progress_interval = host_progress.get("update_interval_seconds")
    reserve_memory = host_scheduler.get("reserve_memory_gib")
    per_worker_memory = host_scheduler.get("estimated_gib_per_worker")
    require(
        host.get("schema_version") == HOST_PREFLIGHT_SCHEMA
        and host.get("task_id") == HOST_PREFLIGHT_TASK_ID
        and host.get("execution_class") == "formal"
        and host.get("implementation")
        == {
            "script": expected_implementations["runner"]["path"],
            "sha256": expected_implementations["runner"]["sha256"],
        }
        and sha256_file(guarded_path) == guarded_digest
        and host_workload.get("class") == "cpu_data_parallel"
        and host_workload.get("real_data_mechanics_match") is True
        and host_workload.get("input_identity")
        == f"identity_lock:{claim.get('identity_lock_sha256')}"
        and host_pilot.get("representative_units") == 32
        and host_pilot.get("projected_full_units")
        == CLUSTER_COUNT * PAIR_COUNT
        and host_pilot.get("same_access_mechanics") is True
        and host_pilot.get("output_equivalence") == "PASS"
        and isinstance(host_pilot.get("progress_samples"), int)
        and not isinstance(host_pilot.get("progress_samples"), bool)
        and host_pilot.get("progress_samples") >= 2
        and host_scheduler.get("backend") == "cpu_process_pool"
        and host_scheduler.get("workers") == WORKERS
        and host_scheduler.get("comparison_performed") is True
        and host_scheduler.get("scientific_parameters_unchanged") is True
        and host_scheduler.get("inject_workers") is True
        and isinstance(reserve_memory, (int, float))
        and not isinstance(reserve_memory, bool)
        and math.isfinite(float(reserve_memory))
        and float(reserve_memory) >= 6.0
        and isinstance(per_worker_memory, (int, float))
        and not isinstance(per_worker_memory, bool)
        and math.isfinite(float(per_worker_memory))
        and float(per_worker_memory) > 0.0
        and host_progress.get("path") == f"{expected_root}/progress.json"
        and isinstance(host_progress.get("fields"), list)
        and all(
            isinstance(field, str)
            for field in host_progress.get("fields")
        )
        and required_progress_fields.issubset(
            set(host_progress.get("fields"))
        )
        and host_progress.get("verified_in_pilot") is True
        and isinstance(progress_interval, (int, float))
        and not isinstance(progress_interval, bool)
        and math.isfinite(float(progress_interval))
        and 0.0 < float(progress_interval) <= 60.0
        and host_terminal.get("success_path")
        == f"{expected_root}/success.json"
        and host_terminal.get("failure_path")
        == f"{expected_root}/failure.json"
        and host_formal.get("one_shot") is True
        and host_formal.get("claim_created_by_runner_only") is True
        and host_formal.get("claim_path") == f"{expected_root}/claim.json"
        and host_formal.get("output_path") == f"{expected_root}/success.json"
        and host_formal.get("failure_receipt_path")
        == f"{expected_root}/failure.json"
        and host_formal.get("activation_authority")
        == (
            f"{ACTIVATION_RELATIVE}:"
            f"{sha256_file(root / ACTIVATION_RELATIVE)}"
        )
        and activation.get("execution_authorized") is True,
        "HOST_PREFLIGHT_SEMANTICS",
    )
    host_evidence = _mapping(
        _mapping(
            spec.get("host_preflight_contract"),
            "SPEC_HOST_CONTRACT",
        ).get("frozen_evidence"),
        "SPEC_HOST_EVIDENCE",
    )
    require(
        host_evidence.get("receipt_path") == host_relative
        and host_evidence.get("receipt_sha256") == host_digest
        and host_evidence.get("status") == "QUALIFIED"
        and host_evidence.get("workers") == WORKERS,
        "SPEC_HOST_EVIDENCE_BINDING",
    )


def _verify_r3_protocol_parameters(
    contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    unchanged = _mapping(
        contract.get("unchanged_r3_parameters"),
        "CONTRACT_R3_PARAMETERS",
    )
    expected_lk = _mapping(
        unchanged.get("sparse_lk"), "CONTRACT_SPARSE_LK"
    )
    expected_affine = _mapping(
        unchanged.get("local_affine"), "CONTRACT_LOCAL_AFFINE"
    )
    require(
        expected_lk == EXPECTED_LK_CONTRACT
        and expected_affine == EXPECTED_LOCAL_AFFINE_CONTRACT,
        "R3_CONTRACT_PARAMETER_DRIFT",
    )
    lk = _mapping(protocol.get("sparse_lk"), "PROTOCOL_SPARSE_LK")
    affine = _mapping(
        protocol.get("local_affine"), "PROTOCOL_LOCAL_AFFINE"
    )
    lk_values = {
        "grid": f"{lk.get('grid_rows')}x{lk.get('grid_cols')}",
        "max_features_per_cell": lk.get("max_features_per_cell"),
        "quality_level": lk.get("quality_level"),
        "min_distance_px": lk.get("min_distance_pixels"),
        "block_size": lk.get("block_size"),
        "window_size_px": lk.get("window_size"),
        "max_pyramid_level": lk.get("max_pyramid_level"),
        "termination_count": lk.get("termination_count"),
        "termination_epsilon": lk.get("termination_epsilon"),
        "forward_backward_max_error_px": lk.get(
            "forward_backward_max_error_pixels"
        ),
    }
    affine_values = {
        "grid": (
            f"{affine.get('grid_rows')}x{affine.get('grid_cols')}"
        ),
        "minimum_tracks_per_cell": affine.get(
            "minimum_tracks_per_cell"
        ),
        "minimum_hull_fraction": affine.get(
            "minimum_track_convex_hull_fraction"
        ),
        "maximum_design_condition_number": affine.get(
            "maximum_design_condition_number"
        ),
        "maximum_median_fit_residual_px_per_frame": affine.get(
            "maximum_median_fit_residual_pixels_per_frame"
        ),
        "minimum_common_evaluable_cells": affine.get(
            "minimum_common_evaluable_cells_per_pair"
        ),
    }
    metrics = _mapping(protocol.get("metrics"), "PROTOCOL_METRICS")
    require(lk_values == expected_lk, "R3_LK_PARAMETER_DRIFT")
    require(
        affine_values == expected_affine,
        "R3_AFFINE_PARAMETER_DRIFT",
    )
    require(
        metrics.get("sign_accuracy_zero_band_per_s")
        == LEAKAGE_THRESHOLD_PER_S,
        "R3_THRESHOLD_DRIFT",
    )


def _verify_authority_bundle(
    root: Path,
    activation: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    require(
        activation.get("decision")
        == "AUTHORIZE_ROTATION_LEAKAGE_LOCALIZATION_ONE_SHOT_EXECUTION"
        and activation.get("execution_authorized") is True
        and activation.get("r3_modification_authorized") is False
        and activation.get("stage_b_rerun_authorized") is False
        and activation.get("single_variable_repair_authorized") is False,
        "ACTIVATION_AUTHORITY",
    )
    execution = _mapping(
        activation.get("execution"), "ACTIVATION_EXECUTION"
    )
    require(
        execution.get("cluster_count") == CLUSTER_COUNT
        and execution.get("fixed_pairs_per_cluster") == PAIR_COUNT
        and execution.get("workers") == WORKERS
        and execution.get("one_shot") is True
        and execution.get("write_once_output") is True
        and execution.get("source_localization_workload_run") is False
        and execution.get("output_root") == FORMAL_ROOT_RELATIVE,
        "ACTIVATION_EXECUTION_LOCK",
    )
    require(
        activation.get("resource_gate")
        == {
            "in_flight_emergency_floor_bytes": IN_FLIGHT_FLOOR_BYTES,
            "launch_and_refill_minimum_available_ram_bytes": (
                LAUNCH_REFILL_BYTES
            ),
            "launch_check_required": True,
            "sustained_paging_stop": True,
        },
        "ACTIVATION_RESOURCE_LOCK",
    )
    records = activation.get("bindings")
    require(isinstance(records, list), "ACTIVATION_BINDINGS")
    by_role: dict[str, Mapping[str, Any]] = {}
    for value in records:
        record = _mapping(value, "ACTIVATION_BINDING")
        role = record.get("role")
        require(
            isinstance(role, str) and role not in by_role,
            "ACTIVATION_BINDING_ROLE",
        )
        by_role[role] = record
    require(
        set(by_role) == set(ACTIVATION_BINDING_PATHS),
        "ACTIVATION_BINDING_KEYSET",
    )
    for role, relative in ACTIVATION_BINDING_PATHS.items():
        record = by_role[role]
        digest = record.get("sha256")
        require(
            record.get("path") == relative
            and isinstance(digest, str)
            and len(digest) == 64,
            f"ACTIVATION_BINDING_SCHEMA:{role}",
        )
        path = _safe_repo_file(
            root, relative, f"ACTIVATION_BINDING:{role}"
        )
        require(
            sha256_file(path) == digest,
            f"ACTIVATION_BINDING_HASH:{role}",
        )
    preflight_path = _safe_repo_file(
        root, PREFLIGHT_RECEIPT_RELATIVE, "PREFLIGHT_RECEIPT"
    )
    preflight = load_json(preflight_path)
    checks = preflight.get("checks")
    require(
        contract.get("task_id") == TASK_ID
        and preflight.get("task_id") == TASK_ID
        and preflight.get("protocol_status") == "VALID"
        and preflight.get("check_count") == 10
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(value == "PASS" for value in checks.values()),
        "PREFLIGHT_IDENTITY",
    )
    require(
        sha256_file(root / CONTRACT_RELATIVE)
        == by_role["FROZEN_CONTRACT"].get("sha256")
        and sha256_file(root / IDENTITY_RELATIVE)
        == by_role["IDENTITY_INPUT_LOCK"].get("sha256")
        and sha256_file(preflight_path)
        == by_role["INDEPENDENT_PREFLIGHT_RECEIPT"].get("sha256"),
        "ACTIVATION_PRIMARY_BINDING",
    )
    frozen = _mapping(
        contract.get("frozen_bindings"),
        "CONTRACT_FROZEN_BINDINGS",
    )
    require(
        set(frozen) == set(CONTRACT_FROZEN_BINDING_PATHS),
        "CONTRACT_BINDING_KEYSET",
    )
    for name, relative in CONTRACT_FROZEN_BINDING_PATHS.items():
        record = _mapping(
            frozen.get(name), f"CONTRACT_BINDING:{name}"
        )
        if name == "identity_input_lock":
            require(
                set(record)
                == {"path", "sha256_filled_by_validator_receipt"}
                and record.get("path") == relative
                and record.get("sha256_filled_by_validator_receipt") is True,
                "CONTRACT_BINDING_PLACEHOLDER:identity_input_lock",
            )
            continue
        digest = record.get("sha256")
        require(
            record.get("path") == relative
            and isinstance(digest, str)
            and len(digest) == 64,
            f"CONTRACT_BINDING_SCHEMA:{name}",
        )
        path = _safe_repo_file(
            root, relative, f"CONTRACT_BINDING:{name}"
        )
        require(
            sha256_file(path) == digest,
            f"CONTRACT_BINDING_HASH:{name}",
        )
    protocol_path = _safe_repo_file(
        root, PROTOCOL_RELATIVE, "R3_PROTOCOL"
    )
    protocol = load_json(protocol_path)
    _verify_r3_protocol_parameters(contract, protocol)


def _verify_control_bindings(
    root: Path,
    input_root: Path,
    response_root: Path,
    run: Mapping[str, Any],
    *,
    fixture: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    claim_path_candidates = (
        input_root / "control" / "execution_claim.json",
        input_root / "claim.json",
        response_root.parent / "control" / "execution_claim.json",
        response_root / "claim.json",
    )
    claim_paths = [path for path in claim_path_candidates if path.is_file()]
    require(len({path.resolve() for path in claim_paths}) == 1, "CLAIM_PATH")
    claim_path = claim_paths[0]
    claim = load_json(claim_path, canonical=True)
    claim_id = claim.get("claim_id")
    require(
        claim.get("schema")
        == "rcle.r3_rotation_leakage_localization.claim.v1"
        and claim.get("protocol_id") == PROTOCOL_ID
        and claim.get("task_id") == TASK_ID
        and claim.get("runner_id") == RUNNER_ID
        and claim.get("write_once") is True,
        "CLAIM_IDENTITY",
    )
    require(
        isinstance(claim_id, str)
        and len(claim_id) == 32
        and all(character in "0123456789abcdef" for character in claim_id)
        and isinstance(claim.get("claimed_utc"), str)
        and claim.get("terminal") == "EXECUTION_CLAIMED / NO_RETRY",
        "CLAIM_LIFECYCLE",
    )
    runner_relative = claim.get("runner_source_path")
    require(runner_relative == RUNNER_RELATIVE, "CLAIM_RUNNER_PATH")
    runner_path = _safe_repo_file(root, runner_relative, "CLAIM_RUNNER")
    require(
        sha256_file(runner_path) == claim.get("runner_source_sha256"),
        "CLAIM_RUNNER_HASH",
    )
    activation_path = root / ACTIVATION_RELATIVE
    contract_path = root / CONTRACT_RELATIVE
    identity_path = root / IDENTITY_RELATIVE
    preflight_path = root / PREFLIGHT_RECEIPT_RELATIVE
    protocol_path = root / PROTOCOL_RELATIVE
    activation = load_json(activation_path)
    contract = load_json(contract_path)
    identity = load_json(identity_path)
    _verify_authority_bundle(root, activation, contract)
    require(
        activation.get("decision")
        == "AUTHORIZE_ROTATION_LEAKAGE_LOCALIZATION_ONE_SHOT_EXECUTION"
        and activation.get("execution_authorized") is True
        and activation.get("r3_modification_authorized") is False
        and activation.get("stage_b_rerun_authorized") is False,
        "ACTIVATION_AUTHORITY",
    )
    require(
        contract.get("protocol_id") == PROTOCOL_ID
        and identity.get("protocol_id") == PROTOCOL_ID,
        "FROZEN_IDENTITY",
    )
    require(
        sha256_file(contract_path)
        == _activation_binding_hash(activation, "FROZEN_CONTRACT"),
        "CONTRACT_ACTIVATION_HASH",
    )
    require(
        sha256_file(identity_path)
        == _activation_binding_hash(activation, "IDENTITY_INPUT_LOCK"),
        "IDENTITY_ACTIVATION_HASH",
    )
    for key, path in (
        ("activation_sha256", activation_path),
        ("contract_sha256", contract_path),
        ("identity_lock_sha256", identity_path),
        ("preflight_receipt_sha256", preflight_path),
        ("protocol_sha256", protocol_path),
    ):
        require(path.is_file(), f"CLAIM_BINDING_FILE:{key}")
        require(claim.get(key) == sha256_file(path), f"CLAIM_BINDING:{key}")
    require(
        run.get("claim_sha256") == sha256_file(claim_path),
        "RUN_CLAIM_HASH",
    )
    if not fixture:
        expected = (root / activation["execution"]["output_root"]).resolve()
        require(input_root.resolve() == expected, "FORMAL_OUTPUT_ROOT")
        require(
            claim.get("mode") == "FORMAL"
            and run.get("mode") == "FORMAL",
            "FORMAL_MODE",
        )
        require(
            claim.get("cluster_count") == CLUSTER_COUNT
            and claim.get("workers") == WORKERS,
            "FORMAL_CLAIM_COUNTS",
        )
        require(
            claim.get("formal_authority_consumed") is True
            and claim.get("future_repair_authorized") is False,
            "FORMAL_CLAIM_FIREWALL",
        )
        require(
            run.get("schema")
            == "rcle.r3_rotation_leakage_localization.run.v1"
            and run.get("runner_id") == RUNNER_ID
            and run.get("identity_lock_sha256")
            == claim.get("identity_lock_sha256")
            and run.get("formal_authority_consumed") is True
            and run.get("terminal")
            == (
                "LOCALIZATION_EXECUTION_COMPLETE / "
                "INDEPENDENT_VALIDATION_REQUIRED"
            ),
            "FORMAL_RUN_LIFECYCLE",
        )
        _verify_formal_readiness_bindings(
            root, claim, run, activation
        )
    return activation, contract, identity, claim


def _claim_source_hashes(
    root: Path, claim: Mapping[str, Any]
) -> dict[str, str]:
    records = claim.get("source_bindings")
    require(isinstance(records, list), "CLAIM_SOURCE_BINDINGS")
    expected = {
        "GENERATOR_GEOMETRY": GENERATOR_RELATIVE,
        "MATERIAL_RESIDUAL_RENDERER": RENDERER_RELATIVE,
    }
    hashes: dict[str, str] = {}
    for role, relative in expected.items():
        matches = [
            _mapping(item, f"CLAIM_SOURCE_BINDING:{role}")
            for item in records
            if isinstance(item, Mapping) and item.get("role") == role
        ]
        require(len(matches) == 1, f"CLAIM_SOURCE_BINDING_COUNT:{role}")
        record = matches[0]
        require(record.get("path") == relative, f"CLAIM_SOURCE_PATH:{role}")
        digest = record.get("sha256")
        require(
            isinstance(digest, str) and len(digest) == 64,
            f"CLAIM_SOURCE_SHA:{role}",
        )
        path = _safe_repo_file(root, relative, f"CLAIM_SOURCE:{role}")
        require(sha256_file(path) == digest, f"CLAIM_SOURCE_HASH:{role}")
        hashes[relative] = digest
    return hashes


def _formal_replay_descriptors(
    root: Path,
    identity: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    geometry_path = root / STAGE_B_GEOMETRY_RELATIVE
    trajectory_path = root / TRAJECTORY_RELATIVE
    require(
        sha256_file(geometry_path)
        == identity["bindings"]["stage_b_geometry_manifest"]["sha256"],
        "REPLAY_GEOMETRY_MANIFEST_HASH",
    )
    geometry_manifest = load_json(geometry_path)
    require(
        sha256_file(trajectory_path)
        == geometry_manifest.get("trajectory_manifest_sha256"),
        "REPLAY_TRAJECTORY_HASH",
    )
    trajectory_manifest = load_json(trajectory_path)
    by_cluster = {
        item["cluster_id"]: item
        for item in geometry_manifest.get("clusters", [])
    }
    output: dict[str, dict[str, Any]] = {}
    for locked in identity.get("clusters", []):
        cluster_id = locked["cluster_id"]
        require(cluster_id in by_cluster, f"REPLAY_CLUSTER:{cluster_id}")
        materialized = by_cluster[cluster_id]
        poses = _rotation_poses(trajectory_manifest[locked["block"]])
        base_scene = materialized["base_scene"]
        require(
            base_scene.get("scene_geometry_sha256")
            == locked["scene_geometry_sha256"],
            f"REPLAY_SCENE_HASH:{cluster_id}",
        )
        require(
            sha256_value(poses) == locked["pose_sha256"],
            f"REPLAY_POSE_HASH:{cluster_id}",
        )
        output[cluster_id] = {
            "cluster_id": cluster_id,
            "sequence_id": locked["sequence_id"],
            "base_scene": base_scene,
            "poses": poses,
        }
    return output


def _pilot_replay_descriptors(
    root: Path, claim: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    relative = claim.get("pilot_manifest_path")
    digest = claim.get("pilot_manifest_sha256")
    require(
        isinstance(relative, str)
        and isinstance(digest, str)
        and len(digest) == 64,
        "PILOT_MANIFEST_BINDING",
    )
    manifest_path = _safe_repo_file(root, relative, "PILOT_MANIFEST")
    require(sha256_file(manifest_path) == digest, "PILOT_MANIFEST_HASH")
    manifest = load_json(manifest_path)
    require(
        manifest.get("schema")
        == "rcle.r3_rotation_leakage_source_localization.pilot_input.v1"
        and manifest.get("mode") == "DISJOINT_PILOT"
        and manifest.get("response_blind") is True,
        "PILOT_MANIFEST_SCHEMA",
    )
    output = {}
    for cluster in manifest.get("clusters", []):
        cluster_id = cluster.get("cluster_id")
        require(
            isinstance(cluster_id, str) and cluster_id not in output,
            "PILOT_CLUSTER_ID",
        )
        output[cluster_id] = {
            "cluster_id": cluster_id,
            "sequence_id": cluster["sequence_id"],
            "block": cluster["block"],
            "ordinal": cluster["ordinal"],
            "base_scene": cluster["base_scene"],
            "poses": cluster["poses"],
        }
    return output


def _render_replay_pairs(
    descriptor: Mapping[str, Any], generator: Any, renderer: Any
) -> Iterable[dict[str, Any]]:
    base_scene = descriptor["base_scene"]
    poses = descriptor["poses"]
    previous: dict[str, Any] | None = None
    for frame_index, pose in enumerate(poses):
        scene = _dynamic_static_scene(base_scene, frame_index)
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(pose["translation_m"], dtype=np.float64)
        rendered = renderer.render_pair(scene, rotation, translation)
        current = {
            "rgb": np.asarray(rendered["rgb_pair"]["clean"], dtype=np.uint8),
            "valid": np.asarray(rendered["valid_mask"]),
            "scene": scene,
            "pose": pose,
            "rotation": rotation,
            "translation": translation,
        }
        if previous is not None:
            yield {
                "pair_index": frame_index - 1,
                "previous": previous,
                "current": current,
                "generator": generator,
            }
        previous = current


def _compare_runner_metrics(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    cluster: Mapping[str, Any],
    expected_mode: str = "PILOT",
) -> None:
    cluster_id = str(cluster["cluster_id"])
    require_schema(recorded, CLUSTER_METRIC_SCHEMAS, "CLUSTER_METRICS")
    required_top_level = {
        "schema",
        "mode",
        "block",
        "ordinal",
        "cluster_id",
        "sequence_id",
        "planned_pair_count",
        "minimum_evaluable_pair_count",
        "final",
        "geometry",
        "warp",
        "mask",
        "flow",
        "local",
        "layers",
        "route",
        "ambiguity_rules",
        "claim_ceiling",
    }
    require(
        set(recorded) == required_top_level,
        "METRIC_TOP_LEVEL_KEYSET",
    )
    require(
        recorded.get("cluster_id") == cluster_id
        and recorded.get("sequence_id") == cluster["sequence_id"]
        and recorded.get("block") == cluster["block"]
        and recorded.get("ordinal") == cluster["ordinal"],
        "METRIC_CLUSTER_IDENTITY",
    )
    require(
        recorded.get("mode") == expected_mode,
        "METRIC_MODE",
    )
    require(
        recorded.get("claim_ceiling") == CLUSTER_METRIC_CLAIM_CEILING,
        "METRIC_CLAIM_CEILING",
    )
    require(
        recorded.get("ambiguity_rules")
        == CLUSTER_METRIC_AMBIGUITY_RULES,
        "METRIC_AMBIGUITY_RULES",
    )
    require(
        recorded.get("planned_pair_count") == recomputed["pair_count"],
        "METRIC_PAIR_COUNT",
    )
    require(
        recorded.get("minimum_evaluable_pair_count")
        == recomputed["coverage_minimum"],
        "METRIC_COVERAGE_MINIMUM",
    )
    require(
        recorded.get("route") == "PENDING_INDEPENDENT_VALIDATION",
        "RUNNER_FINAL_ROUTE_FORBIDDEN",
    )
    recorded_layers = _mapping(recorded.get("layers"), "METRIC_LAYERS")
    require(
        set(recorded_layers) == set(recomputed["layers"]),
        "METRIC_LAYER_KEYSET",
    )
    for layer_name, layer in recomputed["layers"].items():
        runner_layer = _mapping(
            recorded_layers.get(layer_name), f"METRIC_LAYER:{layer_name}"
        )
        expected_status = layer["status"]
        if layer_name == "MASK_BOUNDARY" and expected_status == "FAIL":
            expected_status = (
                "BOUNDARY_ONLY_FAIL"
                if layer.get("failure_mode") == "BOUNDARY_ONLY"
                else "MASK_NONSEPARABLE"
            )
        require(
            runner_layer.get("status") == expected_status,
            f"METRIC_LAYER_STATUS:{layer_name}",
        )

    geometry_summary = _mapping(
        recorded.get("geometry"), "METRIC_GEOMETRY_SUMMARY"
    )
    warp_summary = _mapping(
        recorded.get("warp"), "METRIC_WARP_SUMMARY"
    )
    mask_summary = _mapping(
        recorded.get("mask"), "METRIC_MASK_SUMMARY"
    )
    flow_summary = _mapping(
        recorded.get("flow"), "METRIC_FLOW_SUMMARY"
    )
    local_summary = _mapping(
        recorded.get("local"), "METRIC_LOCAL_SUMMARY"
    )
    require(
        geometry_summary == recorded_layers["INPUT_GEOMETRY"]
        and warp_summary == recorded_layers["ROTATION_WARP"]
        and local_summary
        == recorded_layers["LOCAL_AFFINE_AND_FINAL_AGGREGATION"],
        "METRIC_REDUNDANT_SUMMARY_BINDING",
    )
    boundary_layer = _mapping(
        recorded_layers["MASK_BOUNDARY"],
        "METRIC_BOUNDARY_LAYER",
    )
    require(
        boundary_layer
        == {
            **mask_summary,
            "boundary_only_failure": (
                mask_summary.get("status") == "BOUNDARY_ONLY_FAIL"
            ),
            "interior_flow_status": flow_summary.get("interior_status"),
        },
        "METRIC_MASK_SUMMARY_BINDING",
    )
    accepted_status = flow_summary.get("accepted_status")
    managed_status = flow_summary.get("managed_status")
    flow_layer_status = (
        "NOT_EVALUABLE"
        if "NOT_EVALUABLE" in (accepted_status, managed_status)
        else (
            "PASS"
            if accepted_status == managed_status == "PASS"
            else "FAIL"
        )
    )
    require(
        recorded_layers["SPARSE_LK_AND_TRACK_FILTERING"]
        == {
            "status": flow_layer_status,
            "accepted_status": accepted_status,
            "managed_status": managed_status,
            "accepted": flow_summary.get("accepted"),
            "managed": flow_summary.get("managed"),
        },
        "METRIC_FLOW_SUMMARY_BINDING",
    )

    def compare_count(
        runner: Mapping[str, Any],
        runner_key: str,
        independent: Mapping[str, Any],
        independent_key: str,
        label: str,
    ) -> None:
        require(
            runner.get(runner_key) == independent.get(independent_key),
            label,
        )

    def compare_value(
        runner: Mapping[str, Any],
        runner_key: str,
        independent: Mapping[str, Any],
        independent_key: str,
        label: str,
    ) -> None:
        close(
            runner.get(runner_key),
            independent.get(independent_key),
            label,
        )

    geometry_runner = _mapping(
        recorded_layers["INPUT_GEOMETRY"], "METRIC_GEOMETRY"
    )
    geometry_independent = recomputed["layers"]["INPUT_GEOMETRY"]
    compare_count(
        geometry_runner,
        "evaluable_pair_count",
        geometry_independent,
        "evaluable_pair_count",
        "METRIC_GEOMETRY_COUNT",
    )
    compare_value(
        geometry_runner,
        "absolute_per_s_p90",
        geometry_independent,
        "absolute_p90_per_s",
        "METRIC_GEOMETRY_ABSOLUTE_P90",
    )
    compare_value(
        geometry_runner,
        "roundtrip_max_px_max",
        geometry_independent,
        "roundtrip_max_px",
        "METRIC_GEOMETRY_ROUNDTRIP_MAX",
    )

    warp_runner = _mapping(
        recorded_layers["ROTATION_WARP"], "METRIC_WARP"
    )
    warp_independent = recomputed["layers"]["ROTATION_WARP"]
    compare_count(
        warp_runner,
        "evaluable_pair_count",
        warp_independent,
        "evaluable_pair_count",
        "METRIC_WARP_COUNT",
    )
    compare_value(
        warp_runner,
        "coordinate_roundtrip_p99_px_p90",
        warp_independent,
        "coordinate_roundtrip_pair_p99_cluster_p90_px",
        "METRIC_WARP_ROUNDTRIP_P90",
    )
    compare_value(
        warp_runner,
        "interior_gray_absolute_p90_normalized_p90",
        warp_independent,
        "interior_gray_absolute_pair_p90_cluster_p90",
        "METRIC_WARP_GRAY_P90",
    )

    boundary_runner = _mapping(
        recorded_layers["MASK_BOUNDARY"], "METRIC_BOUNDARY"
    )
    boundary_independent = recomputed["layers"]["MASK_BOUNDARY"]
    compare_count(
        boundary_runner,
        "evaluable_pair_count",
        boundary_independent,
        "matched_evaluable_pair_count",
        "METRIC_BOUNDARY_MATCHED_COUNT",
    )
    compare_value(
        boundary_runner,
        "boundary_absolute_per_s_p90",
        boundary_independent,
        "boundary_source_error_absolute_p90_per_s",
        "METRIC_BOUNDARY_ABSOLUTE_P90",
    )
    compare_value(
        boundary_runner,
        "interior_absolute_per_s_p90",
        boundary_independent,
        "interior_source_error_absolute_p90_per_s",
        "METRIC_INTERIOR_ABSOLUTE_P90",
    )

    flow_runner = _mapping(
        recorded_layers["SPARSE_LK_AND_TRACK_FILTERING"],
        "METRIC_FLOW",
    )
    flow_independent = recomputed["layers"][
        "SPARSE_LK_AND_TRACK_FILTERING"
    ]
    for path in ("accepted", "managed"):
        runner_subprimitive = _mapping(
            flow_runner.get(path), f"METRIC_FLOW:{path}"
        )
        independent_subprimitive = _mapping(
            flow_independent.get(path), f"METRIC_FLOW_REPLAY:{path}"
        )
        compare_count(
            runner_subprimitive,
            "evaluable_pair_count",
            independent_subprimitive,
            "evaluable_pair_count",
            f"METRIC_FLOW_COUNT:{path}",
        )
        compare_value(
            runner_subprimitive,
            "absolute_per_s_p90",
            independent_subprimitive,
            "absolute_p90_per_s",
            f"METRIC_FLOW_ABSOLUTE_P90:{path}",
        )

    local_runner = _mapping(
        recorded_layers["LOCAL_AFFINE_AND_FINAL_AGGREGATION"],
        "METRIC_LOCAL",
    )
    local_independent = recomputed["layers"][
        "LOCAL_AFFINE_AND_FINAL_AGGREGATION"
    ]
    compare_count(
        local_runner,
        "evaluable_pair_count",
        local_independent,
        "evaluable_pair_count",
        "METRIC_LOCAL_COUNT",
    )
    compare_value(
        local_runner,
        "absolute_per_s_p90",
        local_independent,
        "absolute_p90_per_s",
        "METRIC_LOCAL_ABSOLUTE_P90",
    )
    require(
        local_runner.get("numeric_reproduced") is True,
        "METRIC_LOCAL_REPRODUCTION",
    )

    final_runner = _mapping(recorded.get("final"), "METRIC_FINAL")
    compare_count(
        final_runner,
        "evaluable_pair_count",
        local_independent,
        "evaluable_pair_count",
        "METRIC_FINAL_COUNT",
    )
    compare_value(
        final_runner,
        "absolute_per_s_p90",
        recomputed["final_precondition"],
        "absolute_p90_per_s",
        "METRIC_FINAL_ABSOLUTE_P90",
    )
    require(
        final_runner.get("reproduced") is True,
        "METRIC_FINAL_REPRODUCTION",
    )


def _validate_cluster(
    response_root: Path,
    cluster: Mapping[str, Any],
    run_record: Mapping[str, Any],
    pair_count: int,
    replay_descriptor: Mapping[str, Any],
    generator: Any,
    renderer: Any,
    expected_mode: str,
    progress_callback: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cluster_id = str(cluster["cluster_id"])
    ledger_path, primitives_path, metrics_path, receipt_path = _cluster_paths(
        response_root, cluster_id
    )
    receipt = load_json(receipt_path, canonical=True)
    require_schema(receipt, CLUSTER_RECEIPT_SCHEMAS, "CLUSTER_RECEIPT")
    require(
        receipt.get("protocol_id") == PROTOCOL_ID
        and receipt.get("task_id") == TASK_ID
        and receipt.get("cluster_id") == cluster_id
        and receipt.get("sequence_id") == cluster["sequence_id"]
        and receipt.get("pair_count") == pair_count,
        f"CLUSTER_RECEIPT_IDENTITY:{cluster_id}",
    )
    require(
        receipt.get("route_status") == "PENDING_INDEPENDENT_VALIDATION",
        f"CLUSTER_RUNNER_ROUTE:{cluster_id}",
    )
    for filename, path in (
        ("pair_ledger.jsonl", ledger_path),
        ("primitives.npz", primitives_path),
        ("cluster_metrics.json", metrics_path),
    ):
        require(
            _receipt_hash(receipt, filename, cluster_id) == sha256_file(path),
            f"CLUSTER_ARTIFACT_HASH:{cluster_id}:{filename}",
        )
    require(
        run_record.get("receipt_sha256") == sha256_file(receipt_path),
        f"RUN_CLUSTER_RECEIPT_HASH:{cluster_id}",
    )
    relative_receipt = Path(
        str(run_record.get("receipt_path", ""))
    ).as_posix()
    require(
        relative_receipt
        in {
            f"clusters/{cluster_id}/receipt.json",
            f"response/clusters/{cluster_id}/receipt.json",
        },
        f"RUN_CLUSTER_RECEIPT_PATH:{cluster_id}",
    )

    rows = load_jsonl(ledger_path)
    require(len(rows) == pair_count, f"PAIR_COUNT:{cluster_id}")
    arrays = load_primitives(primitives_path, pair_count)
    recomputed_pairs = []
    replay_pairs = iter(
        _render_replay_pairs(replay_descriptor, generator, renderer)
    )
    trigger_streak = 0
    previous_state_after: Mapping[str, Any] | None = None
    for pair_index, row in enumerate(rows):
        require(
            row.get("schema")
            == "rcle.r3_rotation_leakage_localization.pair.v1",
            f"PAIR_SCHEMA:{cluster_id}:{pair_index}",
        )
        require(
            row.get("cluster_id") == cluster_id
            and row.get("sequence_id") == cluster["sequence_id"],
            f"PAIR_IDENTITY:{cluster_id}:{pair_index}",
        )
        try:
            replay_pair = next(replay_pairs)
        except StopIteration as error:
            raise InvalidExecution(
                f"REPLAY_PAIR_SHORT:{cluster_id}:{pair_index}"
            ) from error
        primitive_reconstruction = _pair_reconstruction(
            arrays, row, pair_index, replay_pair
        )
        reconstructed = _validate_and_compare_pair(
            row, primitive_reconstruction, pair_index
        )
        recomputed_pairs.append(reconstructed)
        r3_row = _layer(row, "r3_pair_row")
        expected_strict = (
            reconstructed["r3"]["evaluable"]
            and reconstructed["r3"]["signed_per_s"]
            > LEAKAGE_THRESHOLD_PER_S
        )
        trigger_streak = trigger_streak + 1 if expected_strict else 0
        require(
            _bool(
                r3_row.get("strict_trigger"),
                f"PAIR_STRICT_TRIGGER:{pair_index}",
            )
            is expected_strict
            and _bool(
                r3_row.get("three_pair_trigger"),
                f"PAIR_THREE_TRIGGER:{pair_index}",
            )
            is (trigger_streak >= 3),
            f"PAIR_TRIGGER_STATE:{pair_index}",
        )
        state_before = _mapping(
            row.get("pair_state_before"),
            f"PAIR_STATE_BEFORE:{pair_index}",
        )
        state_after = _mapping(
            row.get("pair_state_after"),
            f"PAIR_STATE_AFTER:{pair_index}",
        )
        _validate_pair_state_record(
            state_before, f"PAIR_STATE_BEFORE:{pair_index}"
        )
        _validate_pair_state_record(
            state_after, f"PAIR_STATE_AFTER:{pair_index}"
        )
        if previous_state_after is not None:
            require(
                state_before == previous_state_after,
                f"PAIR_STATE_CONTINUITY:{pair_index}",
            )
        previous_state_after = state_after
        if progress_callback is not None:
            progress_callback()
    try:
        next(replay_pairs)
    except StopIteration:
        pass
    else:
        raise InvalidExecution(f"REPLAY_PAIR_LONG:{cluster_id}")
    reduced = reduce_cluster(recomputed_pairs, pair_count)
    reduced.update(
        {
            "cluster_id": cluster_id,
            "sequence_id": cluster["sequence_id"],
            "block": cluster.get("block"),
            "ordinal": cluster.get("ordinal"),
        }
    )
    metrics = load_json(metrics_path, canonical=True)
    _compare_runner_metrics(
        metrics,
        reduced,
        cluster,
        expected_mode=expected_mode,
    )
    artifact_hashes = {
        "pair_ledger_sha256": sha256_file(ledger_path),
        "primitives_sha256": sha256_file(primitives_path),
        "cluster_metrics_sha256": sha256_file(metrics_path),
        "cluster_receipt_sha256": sha256_file(receipt_path),
    }
    return reduced, artifact_hashes


def validate_execution(
    input_root: Path,
    *,
    validate_only: bool = False,
    fixture: bool = False,
    pilot_manifest: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = repo_root()
    input_resolved = input_root.resolve()
    response_root = _resolve_response_root(input_resolved)
    persist_validation_progress = not validate_only
    validation_claim_id = (
        uuid.uuid4().hex if persist_validation_progress else None
    )
    validation_started = time.perf_counter()
    run_path = response_root / "run_receipt.json"
    run = load_json(run_path, canonical=True)
    require_schema(run, RUN_RECEIPT_SCHEMAS, "RUN_RECEIPT")
    require(
        run.get("protocol_id") == PROTOCOL_ID
        and run.get("task_id") == TASK_ID,
        "RUN_IDENTITY",
    )
    require(
        run.get("route_status") == "PENDING_INDEPENDENT_VALIDATION",
        "RUNNER_FINAL_ROUTE_FORBIDDEN",
    )
    success_path = response_root / "success.json"
    require(
        success_path.is_file()
        and not (response_root / "failure.json").exists(),
        "RUN_TERMINAL_FILES",
    )
    success = load_json(success_path, canonical=True)
    require(
        success.get("run_receipt_sha256") == sha256_file(run_path),
        "SUCCESS_RUN_RECEIPT_HASH",
    )
    require(
        set(success) == set(run) | {"run_receipt_sha256", "completed_utc"}
        and all(success.get(key) == value for key, value in run.items())
        and isinstance(success.get("completed_utc"), str),
        "SUCCESS_RUN_RECEIPT_SEMANTICS",
    )
    if persist_validation_progress:
        for path in (
            response_root / "analysis_result.json",
            response_root / "independent_validation_receipt.json",
            response_root / "execution_decision.json",
            response_root / "validation_failure.json",
            response_root / "validation_progress.json",
        ):
            require(
                not path.exists(),
                f"VALIDATOR_OUTPUT_COLLISION:{path.name}",
            )
        run_cluster_count = int(run.get("cluster_count", 0))
        require(run_cluster_count > 0, "RUN_PROGRESS_CLUSTER_COUNT")
        if not fixture:
            require(
                validation_claim_id is not None,
                "VALIDATION_CLAIM_ID_MISSING",
            )
            ACTIVE_VALIDATION_CLAIMS[
                response_root.resolve()
            ] = validation_claim_id
        write_validation_progress(
            response_root,
            validation_claim_id=str(validation_claim_id),
            completed_units=0,
            total_units=run_cluster_count,
            started=validation_started,
            status="running",
            initial=True,
        )
    activation, _, identity, claim = _verify_control_bindings(
        root, input_resolved, response_root, run, fixture=fixture
    )
    source_hashes = _claim_source_hashes(root, claim)
    generator, renderer = _load_generator_modules(root, source_hashes)

    run_records = _run_cluster_records(run)
    if fixture:
        require(pilot_manifest is not None, "PILOT_MANIFEST_REQUIRED")
        pilot_path = pilot_manifest.resolve()
        try:
            pilot_relative = pilot_path.relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise InvalidExecution("PILOT_MANIFEST_OUTSIDE_REPO") from error
        claim_with_path = {
            **claim,
            "pilot_manifest_path": pilot_relative,
        }
        replay_descriptors = _pilot_replay_descriptors(
            root, claim_with_path
        )
        clusters = [
            {
                "cluster_id": cluster_id,
                "sequence_id": record["sequence_id"],
                "block": replay_descriptors[cluster_id]["block"],
                "ordinal": replay_descriptors[cluster_id]["ordinal"],
            }
            for cluster_id, record in sorted(run_records.items())
        ]
        pair_count = int(run.get("pairs_per_cluster", 0))
        if pair_count <= 0:
            first_cluster_id = next(iter(sorted(run_records)), "")
            first_receipt = load_json(
                response_root
                / "clusters"
                / first_cluster_id
                / "receipt.json",
                canonical=True,
            )
            pair_count = int(first_receipt.get("pair_count", 0))
        require(pair_count > 0, "FIXTURE_PAIR_COUNT")
    else:
        replay_descriptors = _formal_replay_descriptors(root, identity)
        clusters = identity.get("clusters")
        require(
            isinstance(clusters, list)
            and len(clusters) == CLUSTER_COUNT,
            "FORMAL_CLUSTERS",
        )
        pair_count = PAIR_COUNT
        require(run.get("cluster_count") == CLUSTER_COUNT, "RUN_CLUSTER_COUNT")
    cluster_ids = [str(item["cluster_id"]) for item in clusters]
    require(
        set(run_records) == set(cluster_ids)
        and len(run_records) == len(cluster_ids),
        "RUN_CLUSTER_SET",
    )
    require(
        int(run.get("cluster_count", 0)) == len(cluster_ids),
        "RUN_CLUSTER_COUNT_PROGRESS",
    )
    require(
        set(replay_descriptors) == set(cluster_ids),
        "REPLAY_CLUSTER_SET",
    )
    actual_directories = {
        path.name
        for path in (response_root / "clusters").iterdir()
        if path.is_dir()
    }
    require(actual_directories == set(cluster_ids), "CLUSTER_DIRECTORY_SET")

    resource = _mapping(run.get("resource"), "RUN_RESOURCE")
    require(
        not resource.get("residual_worker_pids"),
        "RUN_RESIDUAL_WORKERS",
    )
    require(
        run.get("r3_modified") is False
        and run.get("formal_480_plus_16_sequences_run") == 0
        and run.get("future_repair_authorized") is False
        and run.get("first_visible_layer_is_not_causal_identification") is True,
        "RUN_FIREWALL",
    )
    if not fixture:
        require(
            resource.get("workers") == 4
            and int(resource.get("available_ram_at_launch_bytes", 0))
            >= 6 * 1024**3
            and int(resource.get("minimum_available_ram_bytes", 0))
            >= 4 * 1024**3,
            "RUN_RESOURCE_GATE",
        )
        require(
            activation["execution"]["fixed_pairs_per_cluster"] == PAIR_COUNT,
            "ACTIVATION_PAIR_COUNT",
        )

    clusters_out = []
    artifact_hashes: dict[str, Any] = {}
    last_progress_write = time.perf_counter()

    def heartbeat() -> None:
        nonlocal last_progress_write
        now = time.perf_counter()
        if (
            persist_validation_progress
            and now - last_progress_write >= 5.0
        ):
            write_validation_progress(
                response_root,
                validation_claim_id=str(validation_claim_id),
                completed_units=len(clusters_out),
                total_units=len(clusters),
                started=validation_started,
                status="running",
            )
            last_progress_write = now

    for cluster in clusters:
        cluster_id = str(cluster["cluster_id"])
        reduced, hashes = _validate_cluster(
            response_root,
            cluster,
            run_records[cluster_id],
            pair_count,
            replay_descriptors[cluster_id],
            generator,
            renderer,
            str(run.get("mode")),
            progress_callback=heartbeat,
        )
        clusters_out.append(reduced)
        artifact_hashes[cluster_id] = hashes
        if persist_validation_progress:
            write_validation_progress(
                response_root,
                validation_claim_id=str(validation_claim_id),
                completed_units=len(clusters_out),
                total_units=len(clusters),
                started=validation_started,
                status="running",
            )
    route_counts = {
        route: sum(cluster["route"] == route for cluster in clusters_out)
        for route in sorted(ROUTES)
    }
    analysis = {
        "schema": "rcle.r3_rotation_leakage_localization.analysis_result.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "analysis_unit": "cluster",
        "cluster_count": len(clusters_out),
        "pairs_per_cluster": pair_count,
        "fixed_formal_denominator": PAIR_COUNT,
        "pair_frame_track_cell_are_longitudinal_repeats": True,
        "clusters": clusters_out,
        "route_counts": route_counts,
        "cross_cluster_majority_route": False,
        "first_visible_layer_is_not_causal_identification": True,
        "formal_480_plus_16_consumed": False,
        "future_repair_authorized": False,
        "terminal": (
            "ROTATION_LEAKAGE_LOCALIZATION_ANALYSIS_COMPLETE / "
            "PER_CLUSTER_ROUTES_ONLY"
        ),
    }
    analysis_digest = hashlib.sha256(canonical_bytes(analysis)).hexdigest()
    validator_path = Path(__file__).resolve()
    receipt = {
        "schema": (
            "rcle.r3_rotation_leakage_localization."
            "independent_validation_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "valid": True,
        "fixture": fixture,
        "run_receipt_sha256": sha256_file(run_path),
        "analysis_result_sha256": analysis_digest,
        "validator_source_path": validator_path.relative_to(root).as_posix(),
        "validator_source_sha256": sha256_file(validator_path),
        "cluster_artifact_hashes": artifact_hashes,
        "checks": {
            "bindings_hashes_and_write_once": "PASS",
            "canonical_schema_order_and_finite_values": "PASS",
            "pickle_free_fixed_ragged_primitives": "PASS",
            "hash_bound_generator_and_renderer_replay": "PASS",
            "independent_cv2_warp_mask_and_gray_reconstruction": "PASS",
            "independent_source_geometry_ols": "PASS",
            "finite_lk_endpoint_status_error_and_mask_accounting": "PASS",
            "independent_merge_provenance_and_activated_splice": "PASS",
            "independent_accepted_and_managed_source_error_ols": "PASS",
            "independent_native_ransac_consensus_local_fit_and_r3_reproduction": "PASS",
            "signed_absolute_separation": "PASS",
            "manual_hyndman_fan_type7": "PASS",
            "fixed_451_of_601_coverage": "PASS",
            "five_layer_gate_and_route_precedence": "PASS",
            "formal_firewall": "PASS",
        },
        "terminal": (
            "ROTATION_LEAKAGE_LOCALIZATION_INDEPENDENT_VALIDATION_PASS / "
            "VALID"
        ),
    }
    receipt_digest = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    decision = {
        "schema": (
            "rcle.r3_rotation_leakage_localization.execution_decision.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "decision": "PER_CLUSTER_ROUTES_RECORDED",
        "cluster_routes": [
            {
                "cluster_id": cluster["cluster_id"],
                "route": cluster["route"],
            }
            for cluster in clusters_out
        ],
        "route_counts": route_counts,
        "analysis_result_sha256": analysis_digest,
        "independent_validation_receipt_sha256": receipt_digest,
        "first_visible_layer_is_not_causal_identification": True,
        "r3_modification_authorized": False,
        "stage_b_retry_replacement_or_reseed_authorized": False,
        "future_single_variable_repair_authorized": False,
        "formal_480_plus_16_authority_consumed": False,
        "terminal": (
            "ROTATION_LEAKAGE_LOCALIZATION_EXECUTION_DECISION / "
            "STOP_AFTER_PER_CLUSTER_ROUTING"
        ),
    }

    if not validate_only:
        analysis_path = response_root / "analysis_result.json"
        receipt_path = response_root / "independent_validation_receipt.json"
        decision_path = response_root / "execution_decision.json"
        for path in (analysis_path, receipt_path, decision_path):
            require(not path.exists(), f"VALIDATOR_OUTPUT_COLLISION:{path.name}")
        write_exclusive(analysis_path, analysis)
        require(sha256_file(analysis_path) == analysis_digest, "ANALYSIS_HASH")
        write_exclusive(receipt_path, receipt)
        require(sha256_file(receipt_path) == receipt_digest, "RECEIPT_HASH")
        if persist_validation_progress:
            write_validation_progress(
                response_root,
                validation_claim_id=str(validation_claim_id),
                completed_units=len(clusters_out),
                total_units=len(clusters),
                started=validation_started,
                status="complete",
            )
        require(
            not (response_root / "validation_failure.json").exists(),
            "VALIDATOR_FAILURE_DECISION_COLLISION",
        )
        write_exclusive(decision_path, decision)
        if not fixture:
            ACTIVE_VALIDATION_CLAIMS.pop(response_root.resolve(), None)
    return analysis, receipt, decision


def _npz_semantic_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == NPZ_KEYS, "PILOT_NPZ_KEYSET")
        for name in sorted(archive.files):
            array = archive[name]
            require(array.dtype.kind != "O", f"PILOT_NPZ_OBJECT:{name}")
            _require_finite_array(array, f"PILOT:{name}")
            digest.update(
                canonical_bytes(
                    {
                        "name": name,
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                    }
                )
            )
            digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _stripped_cluster_receipt(path: Path) -> dict[str, Any]:
    value = load_json(path, canonical=True)
    for key in (
        "wall_seconds",
        "minimum_available_ram_bytes",
        "pair_ledger_sha256",
        "primitives_sha256",
        "cluster_metrics_sha256",
    ):
        value.pop(key, None)
    return value


def _pilot_semantic_payload(root_path: Path) -> dict[str, Any]:
    response_root = _resolve_response_root(root_path.resolve())
    run = load_json(response_root / "run_receipt.json", canonical=True)
    claim = load_json(
        (
            root_path / "claim.json"
            if (root_path / "claim.json").is_file()
            else root_path / "control" / "execution_claim.json"
        ),
        canonical=True,
    )
    records = _run_cluster_records(run)
    clusters = []
    for cluster_id in sorted(records):
        directory = response_root / "clusters" / cluster_id
        ledger = directory / "pair_ledger.jsonl"
        primitives = directory / "primitives.npz"
        metrics = directory / "cluster_metrics.json"
        receipt = directory / "receipt.json"
        clusters.append(
            {
                "cluster_id": cluster_id,
                "sequence_id": records[cluster_id]["sequence_id"],
                "pair_ledger_sha256": sha256_file(ledger),
                "primitives_semantic_sha256": _npz_semantic_digest(
                    primitives
                ),
                "cluster_metrics_sha256": sha256_file(metrics),
                "cluster_receipt_semantic_sha256": hashlib.sha256(
                    canonical_bytes(_stripped_cluster_receipt(receipt))
                ).hexdigest(),
            }
        )
    return {
        "mode": run.get("mode"),
        "cluster_count": run.get("cluster_count"),
        "route_status": run.get("route_status"),
        "pilot_manifest_sha256": claim.get("pilot_manifest_sha256"),
        "pilot_disjoint_receipt_sha256": claim.get(
            "pilot_disjoint_receipt_sha256"
        ),
        "runner_source_path": claim.get("runner_source_path"),
        "runner_source_sha256": claim.get("runner_source_sha256"),
        "source_bindings": claim.get("source_bindings"),
        "clusters": clusters,
    }


def compare_pilots(
    w1_root: Path,
    w4_root: Path,
    pilot_manifest: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    root = repo_root()
    output = output_receipt.resolve()
    formal_root = (
        root
        / "artifacts.local/evidence/"
        "rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1_r3_rotation_leakage_source_localization_r0"
    ).resolve()
    require(
        output != formal_root
        and formal_root not in output.parents
        and not output.exists(),
        "PILOT_EQUIVALENCE_OUTPUT",
    )
    manifest_path = pilot_manifest.resolve()
    manifest_hash = sha256_file(manifest_path)
    left_analysis, _, _ = validate_execution(
        w1_root,
        validate_only=True,
        fixture=True,
        pilot_manifest=manifest_path,
    )
    right_analysis, _, _ = validate_execution(
        w4_root,
        validate_only=True,
        fixture=True,
        pilot_manifest=manifest_path,
    )
    left_response = _resolve_response_root(w1_root.resolve())
    right_response = _resolve_response_root(w4_root.resolve())
    left_run = load_json(left_response / "run_receipt.json", canonical=True)
    right_run = load_json(right_response / "run_receipt.json", canonical=True)
    require(
        left_run.get("mode") == right_run.get("mode") == "PILOT",
        "PILOT_EQUIVALENCE_MODE",
    )
    require(
        _mapping(left_run.get("resource"), "PILOT_W1_RESOURCE").get(
            "workers"
        )
        == 1
        and _mapping(right_run.get("resource"), "PILOT_W4_RESOURCE").get(
            "workers"
        )
        == 4,
        "PILOT_EQUIVALENCE_WORKERS",
    )
    left_payload = _pilot_semantic_payload(w1_root.resolve())
    right_payload = _pilot_semantic_payload(w4_root.resolve())
    require(
        left_payload["pilot_manifest_sha256"]
        == right_payload["pilot_manifest_sha256"]
        == manifest_hash,
        "PILOT_EQUIVALENCE_MANIFEST",
    )
    require(
        left_payload == right_payload,
        "PILOT_EQUIVALENCE_SEMANTIC_PAYLOAD",
    )
    require(
        left_analysis["clusters"] == right_analysis["clusters"]
        and left_analysis["route_counts"] == right_analysis["route_counts"],
        "PILOT_EQUIVALENCE_ANALYSIS",
    )
    semantic_digest = hashlib.sha256(
        canonical_bytes(left_payload)
    ).hexdigest()
    validator_path = Path(__file__).resolve()
    receipt = {
        "schema": (
            "rcle.r3_rotation_leakage_localization."
            "pilot_w1_w4_equivalence_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "valid": True,
        "pilot_manifest_path": manifest_path.relative_to(root).as_posix(),
        "pilot_manifest_sha256": manifest_hash,
        "w1_root": w1_root.resolve().relative_to(root).as_posix(),
        "w4_root": w4_root.resolve().relative_to(root).as_posix(),
        "runner_source_path": left_payload["runner_source_path"],
        "runner_source_sha256": left_payload["runner_source_sha256"],
        "validator_source_path": validator_path.relative_to(root).as_posix(),
        "validator_source_sha256": sha256_file(validator_path),
        "semantic_payload_sha256": semantic_digest,
        "cluster_count": left_payload["cluster_count"],
        "checks": {
            "same_disjoint_manifest": "PASS",
            "workers_1_and_4": "PASS",
            "runner_and_validator_hashes": "PASS",
            "cluster_pair_identity_and_order": "PASS",
            "pair_ledgers": "PASS",
            "primitive_array_dtype_shape_content": "PASS",
            "cluster_metrics_and_routes": "PASS",
            "utc_pid_wall_progress_ignored_only": "PASS",
            "formal_authority_not_consumed": "PASS",
        },
        "formal_authority_consumed": False,
        "scientific_interpretation": False,
        "terminal": "PILOT_W1_W4_EQUIVALENCE_PASS / FORMAL_NOT_CONSUMED",
    }
    write_exclusive(output, receipt)
    return receipt


def write_formal_validation_failure(
    input_root: Path, error: BaseException
) -> None:
    """Write one terminal validator failure only at the exact formal root."""

    root = repo_root()
    expected = (root / FORMAL_ROOT_RELATIVE).resolve()
    actual = input_root.resolve()
    validation_claim_id = ACTIVE_VALIDATION_CLAIMS.get(actual)
    if actual != expected or not actual.is_dir() or validation_claim_id is None:
        return
    progress_path = actual / "validation_progress.json"
    try:
        progress = load_json(progress_path, canonical=True)
    except (OSError, InvalidExecution):
        ACTIVE_VALIDATION_CLAIMS.pop(actual, None)
        return
    if progress.get("validation_claim_id") != validation_claim_id:
        ACTIVE_VALIDATION_CLAIMS.pop(actual, None)
        return
    failure_path = actual / "validation_failure.json"
    if (
        failure_path.exists()
        or (actual / "execution_decision.json").exists()
    ):
        return
    validator_path = Path(__file__).resolve()
    failure = {
        "schema": (
            "rcle.r3_rotation_leakage_localization."
            "validation_failure.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "valid": False,
        "failed_utc": utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(limit=40),
        "validator_source_path": validator_path.relative_to(root).as_posix(),
        "validator_source_sha256": sha256_file(validator_path),
        "input_root": FORMAL_ROOT_RELATIVE,
        "validation_claim_id": validation_claim_id,
        "runner_retry_authorized": False,
        "validator_retry_authorized": False,
        "replacement_reseed_resume_or_output_delete_authorized": False,
        "future_repair_authorized": False,
        "terminal": (
            "INDEPENDENT_VALIDATION_INVALID / "
            "ONE_SHOT_CONSUMED / NO_RERUN"
        ),
    }
    try:
        write_exclusive(failure_path, failure)
        if progress_path.is_file():
            progress.update(
                {
                    "last_progress_at": utc_now(),
                    "status": "failed",
                    "eta_seconds": None,
                }
            )
            write_atomic(progress_path, progress)
    finally:
        ACTIVE_VALIDATION_CLAIMS.pop(actual, None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independently validate and route the write-once R3 "
            "rotation-leakage localization execution."
        )
    )
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--compare-pilots",
        type=Path,
        nargs=2,
        metavar=("W1_ROOT", "W4_ROOT"),
    )
    parser.add_argument("--output-receipt", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Validate an explicitly disjoint pilot root, never formal evidence.",
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        help="Bound disjoint pilot manifest, required with --fixture.",
    )
    args = parser.parse_args()
    try:
        if args.compare_pilots is not None:
            require(
                args.input_root is None
                and args.pilot_manifest is not None
                and args.output_receipt is not None,
                "COMPARE_PILOTS_ARGUMENTS",
            )
            receipt = compare_pilots(
                args.compare_pilots[0],
                args.compare_pilots[1],
                args.pilot_manifest,
                args.output_receipt,
            )
            print(
                json.dumps(
                    {
                        "valid": receipt["valid"],
                        "semantic_payload_sha256": receipt[
                            "semantic_payload_sha256"
                        ],
                        "terminal": receipt["terminal"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        require(args.input_root is not None, "INPUT_ROOT_REQUIRED")
        analysis, receipt, _ = validate_execution(
            args.input_root,
            validate_only=args.validate_only,
            fixture=args.fixture,
            pilot_manifest=args.pilot_manifest,
        )
    except BaseException as error:
        if (
            args.compare_pilots is None
            and args.input_root is not None
            and not args.validate_only
            and not args.fixture
        ):
            try:
                write_formal_validation_failure(args.input_root, error)
            except Exception as failure_error:
                print(
                    json.dumps(
                        {
                            "valid": False,
                            "error": str(error),
                            "failure_receipt_error": str(failure_error),
                            "terminal": "INDEPENDENT_VALIDATION_INVALID",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return 2
        print(
            json.dumps(
                {
                    "valid": False,
                    "error": str(error),
                    "terminal": "INDEPENDENT_VALIDATION_INVALID",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "valid": receipt["valid"],
                "cluster_count": analysis["cluster_count"],
                "route_counts": analysis["route_counts"],
                "terminal": receipt["terminal"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
