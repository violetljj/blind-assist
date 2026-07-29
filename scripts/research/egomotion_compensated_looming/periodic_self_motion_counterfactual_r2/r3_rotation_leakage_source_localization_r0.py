"""One-shot R3 rotation-leakage source-localization runner.

This module is an observation-only companion around the frozen R3 pipeline.  It
never replaces a return value from R3: pair-local monkeypatches call the bound
implementation once, copy native primitives, and restore the original symbols
before the next pair.  Formal execution is limited to the eight sealed Stage B
rotation-only identities.  Pilot execution requires an explicitly disjoint
manifest and writes outside the formal target.

The preflight contract intentionally leaves a few reduction details in prose.
They are frozen here, before payload access, in the most conservative form:

* a 21 x 21 LK footprint means a 10-pixel Chebyshev boundary; ``interior`` is
  the intersection of separately eroded previous and warped-current masks;
* every layer first forms a pair scalar, then a type-7 cluster percentile;
  formal cluster coverage is at least ceil(0.75 * 601) == 451 pairs;
* source-error cell fits use deterministic float64 OLS and the unchanged
  support, hull, condition, residual, and five-common-cell gates;
* accepted and final-managed compensated tracks are both required by FLOW;
  failure of either is a same-layer FLOW failure, not cross-layer ambiguity;
* an independently reproduced local-fit mismatch is NOT_EVALUABLE, not a
  LOCAL_FIT scientific failure;
* coordinate round-trip P99 and photometric summaries use all common-valid
  pixel centers in deterministic row-major order.

``first_visible_layer`` remains an ordered audit observation, never a causal
identification or authority to repair R3.  Missing, nonfinite, misbound, or
under-covered primitives fail closed.  Ragged NPZ families use explicit
offsets, indices and validity arrays; NaN, infinity and object dtypes are
forbidden.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = (
        "scripts.research.egomotion_compensated_looming."
        "periodic_self_motion_counterfactual_r2"
    )

from . import p3_runtime_preflight_r0 as guarded

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import time
import traceback
from typing import Any, Iterable, Sequence
from unittest import mock
import uuid

import cv2
import numpy as np
import psutil

from ..rcle_minimal import evaluation as r0_evaluation
from ..rcle_minimal_r1 import local_expansion as local_affine_module
from ..rcle_minimal_r1 import sparse_flow as sparse_flow_module
from ..rcle_minimal_r1.sparse_flow import SparseTrackResult
from ..rgb_algorithm_development_canary_cid_sims_r0 import producer as r3
from . import generator_geometry as geometry
from . import material_residual_contraction_r1 as qms
from . import p3_transport_r0 as transport


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
    "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_CONTRACT_PREFLIGHT_R0"
)
RUNNER_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_R3_"
    "ROTATION_LEAKAGE_SOURCE_LOCALIZATION_RUNNER_R0"
)
FORMAL_MODE = "FORMAL"
PILOT_MODE = "PILOT"
FRAME_COUNT = 602
PAIR_COUNT = 601
WORKERS = 4
GIB = 1024**3
LAUNCH_REFILL_BYTES = 6 * GIB
IN_FLIGHT_FLOOR_BYTES = 4 * GIB
MINIMUM_COVERAGE_FRACTION = 0.75
MINIMUM_FORMAL_EVALUABLE_PAIRS = math.ceil(
    MINIMUM_COVERAGE_FRACTION * PAIR_COUNT
)
THRESHOLD_PER_S = 0.01
BOUNDARY_RADIUS_PX = 10
LOCAL_FIT_NUMERIC_ATOL = 1e-12
TARGET_ID = 1001
PILOT_FIXTURE_ID = (
    "R3_ROTATION_LEAKAGE_LOCALIZATION_IMPLEMENTATION_PILOT_R0"
)
PILOT_SOURCE_ROLE = "DISJOINT_PILOT_FIXTURE"
PILOT_CLUSTER_COUNT = 4
PILOT_FRAME_COUNT = 9
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
SPEC_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "R3_ROTATION_LEAKAGE_SOURCE_LOCALIZATION_EXECUTABLE_SPEC_R1_"
    "2026-07-29.json"
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
SUCCESSOR_FORMAL_RELATIVE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_successor_formal"
)
GENERATOR_GEOMETRY_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/generator_geometry.py"
)
MATERIAL_RENDERER_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "material_residual_contraction_r1.py"
)
INDEPENDENT_VALIDATOR_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "validate_r3_rotation_leakage_source_localization_execution_"
    "independent_r0.py"
)
TEST_SUITE_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/"
    "tests_periodic_self_motion_counterfactual_r2/"
    "test_r3_rotation_leakage_source_localization_execution_r0.py"
)
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
    GENERATOR_GEOMETRY_RELATIVE,
    MATERIAL_RENDERER_RELATIVE,
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

LAYER_NAMES = (
    "INPUT_GEOMETRY",
    "ROTATION_WARP",
    "MASK_BOUNDARY",
    "SPARSE_LK_AND_TRACK_FILTERING",
    "LOCAL_AFFINE_AND_FINAL_AGGREGATION",
)
ROUTES = (
    "LEAKAGE_ALREADY_PRESENT_IN_INPUT_GEOMETRY",
    "LEAKAGE_FIRST_VISIBLE_AT_WARP",
    "LEAKAGE_FIRST_VISIBLE_AT_MASK_BOUNDARY",
    "LEAKAGE_FIRST_VISIBLE_AT_FLOW",
    "LEAKAGE_FIRST_VISIBLE_AT_LOCAL_FIT",
    "MULTIPLE_SOURCES_NOT_SEPARABLE",
    "NOT_EVALUABLE",
)
AMBIGUITY_RULES = {
    "boundary_definition": (
        "common AND erode(previous_valid,21x21) AND "
        "erode(warped_current_valid,21x21)"
    ),
    "cluster_reduction": "pair scalar then Hyndman-Fan type-7 percentile",
    "formal_minimum_evaluable_pairs": MINIMUM_FORMAL_EVALUABLE_PAIRS,
    "flow_gate": "accepted AND final-managed source-error P90 <= 0.01/s",
    "coordinate_roundtrip_population": "all common-valid pixel centers",
    "local_fit_mismatch": "NOT_EVALUABLE",
}


class InvalidLocalization(ValueError):
    """Fail-closed runner error."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise InvalidLocalization(f"NONFINITE_JSON_CONSTANT:{value}")


def _unique_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidLocalization(f"DUPLICATE_JSON_KEY:{key}")
        result[key] = value
    return result


def _require_finite_json_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidLocalization(f"{label}:NONSTRING_KEY")
            _require_finite_json_tree(item, f"{label}:{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json_tree(item, f"{label}:{index}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise InvalidLocalization(label)


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise InvalidLocalization(f"JSON_BOM:{path.name}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidLocalization(
            f"JSON_PARSE:{path.name}:{error}"
        ) from error
    if not isinstance(value, dict):
        raise InvalidLocalization(f"JSON_OBJECT_REQUIRED:{path}")
    _require_finite_json_tree(value, f"JSON_FINITE:{path.name}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def write_atomic_json(path: Path, value: Any) -> bool:
    """Best-effort replace of an operational sidecar on Windows.

    Numbered progress snapshots remain the authoritative write-once record.
    A transient reader that denies delete sharing must not abort scientific
    execution after that snapshot has already been persisted.
    """
    payload = canonical_bytes(value)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    for _ in range(20):
        try:
            os.replace(temporary, path)
            return True
        except PermissionError:
            time.sleep(0.05)
    try:
        with path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.unlink(missing_ok=True)
        return True
    except PermissionError:
        return False


def write_exclusive_jsonl(
    path: Path, rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_bytes(row))
        stream.flush()
        os.fsync(stream.fileno())


def _finite(values: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(values)).all())


def hf7(values: Sequence[float], probability: float) -> float:
    """Hyndman-Fan type-7 quantile; empty or nonfinite inputs fail closed."""
    if not 0.0 <= probability <= 1.0:
        raise InvalidLocalization("HF7_PROBABILITY")
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise InvalidLocalization("HF7_EMPTY")
    if not np.isfinite(array).all():
        raise InvalidLocalization("HF7_NONFINITE")
    ordered = np.sort(array)
    if ordered.size == 1:
        return float(ordered[0])
    h = (ordered.size - 1) * probability
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    fraction = h - lower
    return float(
        ordered[lower] + fraction * (ordered[upper] - ordered[lower])
    )


def boundary_masks(
    previous_valid: np.ndarray, warped_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common, interior, boundary masks under the frozen 21x21 rule."""
    previous = np.asarray(previous_valid)
    warped = np.asarray(warped_valid)
    if previous.shape != warped.shape or previous.ndim != 2:
        raise InvalidLocalization("BOUNDARY_MASK_SHAPE")
    if previous.dtype != np.uint8 or warped.dtype != np.uint8:
        raise InvalidLocalization("BOUNDARY_MASK_UINT8")
    if not set(np.unique(previous)).issubset({0, 255}):
        raise InvalidLocalization("PREVIOUS_MASK_NONBINARY")
    if not set(np.unique(warped)).issubset({0, 255}):
        raise InvalidLocalization("WARPED_MASK_NONBINARY")
    kernel = np.ones(
        (2 * BOUNDARY_RADIUS_PX + 1, 2 * BOUNDARY_RADIUS_PX + 1),
        dtype=np.uint8,
    )
    previous_bool = previous > 0
    warped_bool = warped > 0
    common = previous_bool & warped_bool
    previous_interior = cv2.erode(
        previous,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    warped_interior = cv2.erode(
        warped,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    interior = common & previous_interior & warped_interior
    boundary = common & ~interior
    return (
        np.ascontiguousarray(boundary),
        np.ascontiguousarray(interior),
        np.ascontiguousarray(common),
    )


def _dig(row: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    value: Any = row
    for part in prefix.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, dict) else None


def reduce_layer(
    pair_rows: Sequence[dict[str, Any]],
    prefix: str,
    planned_pairs: int = PAIR_COUNT,
) -> dict[str, Any]:
    """Reduce an evaluable pair-layer dict without treating pairs as IID."""
    if planned_pairs <= 0 or len(pair_rows) != planned_pairs:
        raise InvalidLocalization(f"PAIR_COUNT:{prefix}")
    records = []
    for row in pair_rows:
        value = _dig(row, prefix)
        if value is None:
            value = {
                key[len(prefix) + 1 :]: item
                for key, item in row.items()
                if key.startswith(f"{prefix}_")
            }
        if value is not None and value.get("evaluable") is True:
            records.append(value)
    result: dict[str, Any] = {
        "planned_pair_count": planned_pairs,
        "evaluable_pair_count": len(records),
        "evaluable_fraction": len(records) / planned_pairs,
        "evaluable_pairs": len(records),
        "coverage_fraction_fixed": len(records) / planned_pairs,
        "coverage_status": (
            "PASS"
            if len(records)
            >= math.ceil(MINIMUM_COVERAGE_FRACTION * planned_pairs)
            else "NOT_EVALUABLE"
        ),
    }
    scalar_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if (
                key.endswith("_per_s")
                or key.endswith("_px")
                or key.endswith("_normalized")
            )
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    for key in scalar_keys:
        values = [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
            and not isinstance(record.get(key), bool)
        ]
        if values:
            result[f"{key}_p50"] = hf7(values, 0.50)
            result[f"{key}_p90"] = hf7(values, 0.90)
            result[f"{key}_p99"] = hf7(values, 0.99)
            result[f"{key}_max"] = max(values)
            if key == "signed_per_s":
                result["signed_p50_per_s"] = result[f"{key}_p50"]
            if key == "absolute_per_s":
                result["absolute_p90_per_s"] = result[f"{key}_p90"]
    return result


def route_cluster(metrics: dict[str, Any]) -> str:
    """Apply the frozen ordered per-cluster routing rules."""
    if "layers" in metrics:
        layers = metrics["layers"]
        boundary_layer = layers.get("MASK_BOUNDARY", {})
        flow_layer = layers.get("SPARSE_LK_AND_TRACK_FILTERING", {})
        metrics = {
            "final": {
                "evaluable_pairs": metrics.get("final_evaluable_pairs", 0),
                "absolute_p90_per_s": metrics.get(
                    "final_absolute_p90_per_s"
                ),
                "reproduced": metrics.get("final_precondition") == "FAIL",
            },
            "geometry": layers.get("INPUT_GEOMETRY", {}),
            "warp": layers.get("ROTATION_WARP", {}),
            "mask": {
                **boundary_layer,
                "status": (
                    "BOUNDARY_ONLY_FAIL"
                    if boundary_layer.get("boundary_only_failure") is True
                    else boundary_layer.get("status")
                ),
            },
            "flow": {
                "accepted_status": flow_layer.get(
                    "accepted_status", flow_layer.get("status")
                ),
                "managed_status": flow_layer.get(
                    "managed_status", flow_layer.get("status")
                ),
                "interior_status": boundary_layer.get(
                    "interior_flow_status", flow_layer.get("status")
                ),
            },
            "local": layers.get(
                "LOCAL_AFFINE_AND_FINAL_AGGREGATION", {}
            ),
        }
    final = metrics.get("final", {})
    geometry = metrics.get("geometry", {})
    warp = metrics.get("warp", {})
    mask = metrics.get("mask", {})
    flow = metrics.get("flow", {})
    local = metrics.get("local", {})
    required_statuses = (
        geometry.get("status"),
        warp.get("status"),
        mask.get("status"),
        flow.get("accepted_status"),
        flow.get("managed_status"),
        flow.get("interior_status"),
        local.get("status"),
    )
    if (
        final.get("reproduced") is not True
        or int(final.get("evaluable_pairs", 0))
        < MINIMUM_FORMAL_EVALUABLE_PAIRS
        or not isinstance(final.get("absolute_p90_per_s"), (int, float))
        or not math.isfinite(float(final["absolute_p90_per_s"]))
        or float(final["absolute_p90_per_s"]) <= THRESHOLD_PER_S
        or any(
            status in (None, "NOT_EVALUABLE")
            for status in required_statuses
        )
    ):
        return "NOT_EVALUABLE"
    if geometry["status"] == "FAIL":
        return "LEAKAGE_ALREADY_PRESENT_IN_INPUT_GEOMETRY"
    if warp["status"] == "FAIL":
        if (
            mask["status"] == "PASS"
            and flow["accepted_status"] == "PASS"
            and flow["managed_status"] == "PASS"
            and flow["interior_status"] == "PASS"
        ):
            return "LEAKAGE_FIRST_VISIBLE_AT_WARP"
        return "MULTIPLE_SOURCES_NOT_SEPARABLE"
    if mask["status"] == "BOUNDARY_ONLY_FAIL":
        if flow["interior_status"] == "PASS":
            return "LEAKAGE_FIRST_VISIBLE_AT_MASK_BOUNDARY"
        return "MULTIPLE_SOURCES_NOT_SEPARABLE"
    if mask["status"] in ("FAIL", "MASK_NONSEPARABLE"):
        return "MULTIPLE_SOURCES_NOT_SEPARABLE"
    if (
        flow["accepted_status"] == "FAIL"
        or flow["managed_status"] == "FAIL"
    ):
        return "LEAKAGE_FIRST_VISIBLE_AT_FLOW"
    if flow["interior_status"] == "FAIL":
        return "MULTIPLE_SOURCES_NOT_SEPARABLE"
    if local["status"] == "FAIL":
        return "LEAKAGE_FIRST_VISIBLE_AT_LOCAL_FIT"
    return "NOT_EVALUABLE"


def resource_action(
    available_bytes: int,
    active_workers: int,
    refill_requested: bool,
    paging_streak: int,
) -> str:
    """Return the frozen preclaim/in-flight resource action."""
    if paging_streak >= 2:
        return "STOP_IN_FLIGHT" if active_workers else "STOP_BEFORE_CLAIM"
    if active_workers > 0 and available_bytes < IN_FLIGHT_FLOOR_BYTES:
        return "STOP_IN_FLIGHT"
    if refill_requested and available_bytes < LAUNCH_REFILL_BYTES:
        return "STOP_IN_FLIGHT" if active_workers else "STOP_BEFORE_CLAIM"
    if active_workers == 0 and available_bytes < LAUNCH_REFILL_BYTES:
        return "STOP_BEFORE_CLAIM"
    return "ALLOW"


def validate_formal_target_absent(root: Path, output_root: Path) -> None:
    """Require the exact activation-bound formal root and write-once absence."""
    expected = (root / FORMAL_ROOT_RELATIVE).resolve()
    actual = output_root.resolve()
    if actual != expected:
        raise InvalidLocalization("FORMAL_TARGET_PATH")
    if actual.exists():
        raise InvalidLocalization("FORMAL_TARGET_ALREADY_EXISTS")
    if (root / SUCCESSOR_FORMAL_RELATIVE).exists():
        raise InvalidLocalization("SUCCESSOR_FORMAL_FIREWALL")


def _binding(root: Path, path: str, expected: str, name: str) -> Path:
    if not isinstance(path, str) or Path(path).is_absolute():
        raise InvalidLocalization(f"BINDING_PATH:{name}")
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise InvalidLocalization(f"BINDING_PATH:{name}") from error
    if (
        not candidate.is_file()
        or not expected
        or sha256_file(candidate) != expected
    ):
        raise InvalidLocalization(f"BINDING:{name}")
    return candidate


def _validate_executable_spec_governance(spec: dict[str, Any]) -> None:
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
    if (
        spec.get("schema") != SPEC_SCHEMA
        or spec.get("protocol_id") != PROTOCOL_ID
        or spec.get("task_id") != TASK_ID
        or spec.get("status") != EXECUTABLE_SPEC_STATUS
        or spec.get("terminal") != SPEC_TERMINAL
        or spec.get("claim_ceiling") != SPEC_CLAIM_CEILING
        or spec.get("authority") != expected_authority
    ):
        raise InvalidLocalization("EXECUTABLE_SPEC_GOVERNANCE")


def _validate_frozen_spec_bindings(
    root: Path, spec: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Hash-check every frozen file named by the executable spec."""

    bindings = spec.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise InvalidLocalization("SPEC_BINDINGS")
    expected_keys = set(SPEC_DIRECT_BINDING_PATHS) | {
        "frozen_execution_dependencies"
    }
    if set(bindings) != expected_keys:
        raise InvalidLocalization("SPEC_BINDING_KEYSET")
    for name, expected_path in SPEC_DIRECT_BINDING_PATHS.items():
        value = bindings.get(name)
        if not isinstance(value, dict) or value.get("path") != expected_path:
            raise InvalidLocalization(f"SPEC_BINDING_PATH:{name}")
    dependencies = bindings.get("frozen_execution_dependencies")
    if (
        not isinstance(dependencies, list)
        or len(dependencies) != len(SPEC_FROZEN_DEPENDENCY_PATHS)
        or [
            item.get("path") if isinstance(item, dict) else None
            for item in dependencies
        ]
        != list(SPEC_FROZEN_DEPENDENCY_PATHS)
    ):
        raise InvalidLocalization("SPEC_FROZEN_DEPENDENCY_PATHSET")
    records: list[tuple[str, dict[str, Any]]] = []
    for name, value in bindings.items():
        if name == "frozen_execution_dependencies":
            if not isinstance(value, list) or not value:
                raise InvalidLocalization("SPEC_FROZEN_DEPENDENCIES")
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise InvalidLocalization(
                        f"SPEC_FROZEN_DEPENDENCY:{index}"
                    )
                records.append((f"{name}:{index}", item))
        else:
            if not isinstance(value, dict):
                raise InvalidLocalization(f"SPEC_BINDING_RECORD:{name}")
            records.append((name, value))
    by_path: dict[str, dict[str, Any]] = {}
    for name, record in records:
        path = record.get("path")
        expected = record.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or path in by_path
        ):
            raise InvalidLocalization(f"SPEC_BINDING_SCHEMA:{name}")
        _binding(root, path, expected, f"SPEC_BINDING:{name}")
        by_path[path] = record
    return by_path


def _protocol_parameter_guard(
    contract: dict[str, Any], protocol: dict[str, Any]
) -> None:
    lk = protocol.get("sparse_lk", {})
    affine = protocol.get("local_affine", {})
    expected_lk = contract["unchanged_r3_parameters"]["sparse_lk"]
    expected_affine = contract["unchanged_r3_parameters"]["local_affine"]
    if (
        expected_lk != EXPECTED_LK_CONTRACT
        or expected_affine != EXPECTED_LOCAL_AFFINE_CONTRACT
    ):
        raise InvalidLocalization("R3_CONTRACT_PARAMETER_DRIFT")
    comparisons = {
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
    if any(comparisons[key] != expected_lk[key] for key in comparisons):
        raise InvalidLocalization("R3_LK_PARAMETER_DRIFT")
    affine_comparisons = {
        "grid": f"{affine.get('grid_rows')}x{affine.get('grid_cols')}",
        "minimum_tracks_per_cell": affine.get("minimum_tracks_per_cell"),
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
    if any(
        affine_comparisons[key] != expected_affine[key]
        for key in affine_comparisons
    ):
        raise InvalidLocalization("R3_AFFINE_PARAMETER_DRIFT")
    threshold = protocol.get("metrics", {}).get(
        "sign_accuracy_zero_band_per_s"
    )
    if threshold != THRESHOLD_PER_S:
        raise InvalidLocalization("R3_THRESHOLD_DRIFT")


def validate_implementation_ready(
    root: Path, receipt_path: Path
) -> dict[str, Any]:
    """Validate the frozen executable spec and its separate ready receipt."""
    spec_path = (root / SPEC_RELATIVE).resolve()
    if not spec_path.is_file():
        raise InvalidLocalization("EXECUTABLE_SPEC_MISSING")
    spec = load_json(spec_path)
    _validate_executable_spec_governance(spec)
    spec_binding_by_path = _validate_frozen_spec_bindings(root, spec)
    implementations = spec.get("implementation_bindings", {})
    roles = (
        "runner",
        "independent_execution_validator",
        "test_suite",
    )
    if (
        not isinstance(implementations, dict)
        or implementations.get("status")
        != IMPLEMENTATION_BINDINGS_STATUS
        or set(implementations)
        != {"status", "finalization_rule", *roles}
        or not isinstance(implementations.get("finalization_rule"), str)
        or not implementations.get("finalization_rule")
    ):
        raise InvalidLocalization("SPEC_IMPLEMENTATION_BINDINGS_STATUS")
    implementation_hashes: dict[str, str] = {}
    expected_role_paths = {
        "runner": Path(__file__).resolve().relative_to(root).as_posix(),
        "independent_execution_validator": INDEPENDENT_VALIDATOR_RELATIVE,
        "test_suite": TEST_SUITE_RELATIVE,
    }
    for role in roles:
        binding = implementations.get(role)
        if not isinstance(binding, dict):
            raise InvalidLocalization(f"SPEC_IMPLEMENTATION_BINDING:{role}")
        path = binding.get("path")
        expected = binding.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or path != expected_role_paths[role]
        ):
            raise InvalidLocalization(f"SPEC_IMPLEMENTATION_HASH:{role}")
        _binding(root, path, expected, f"SPEC_{role}")
        implementation_hashes[role] = expected
    runner_path = Path(__file__).resolve()
    runner_binding = implementations["runner"]
    if (
        (root / runner_binding["path"]).resolve() != runner_path
        or implementation_hashes["runner"] != sha256_file(runner_path)
    ):
        raise InvalidLocalization("SPEC_RUNNER_SELF_HASH")
    ready_path = receipt_path.resolve()
    if (
        not ready_path.is_relative_to(root.resolve())
        or not ready_path.is_file()
    ):
        raise InvalidLocalization("IMPLEMENTATION_READY_RECEIPT_MISSING")
    receipt = load_json(ready_path)
    if (
        receipt.get("schema") != IMPLEMENTATION_READY_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("task_id") != TASK_ID
        or receipt.get("status") != "PASS"
        or receipt.get("terminal") != IMPLEMENTATION_READY_TERMINAL
        or receipt.get("formal_authority_consumed") is not False
        or receipt.get("scientific_interpretation") is not False
        or receipt.get("spec_path") != SPEC_RELATIVE
    ):
        raise InvalidLocalization("IMPLEMENTATION_READY_RECEIPT_STATUS")
    spec_sha256 = sha256_file(spec_path)
    if receipt.get("spec_sha256") != spec_sha256:
        raise InvalidLocalization("IMPLEMENTATION_READY_SPEC_BINDING")
    receipt_bindings = receipt.get("implementation_bindings")
    if (
        not isinstance(receipt_bindings, dict)
        or set(receipt_bindings) != set(roles)
    ):
        raise InvalidLocalization("IMPLEMENTATION_READY_BINDING_KEYSET")
    for role, expected in implementation_hashes.items():
        expected_path = implementations[role]["path"]
        binding = receipt_bindings.get(role)
        if (
            not isinstance(binding, dict)
            or binding
            != {
                "path": expected_path,
                "sha256": expected,
            }
        ):
            raise InvalidLocalization(
                f"IMPLEMENTATION_READY_HASH_BINDING:{role}"
            )
    pilot = receipt.get("pilot_w1_w4_equivalence")
    if (
        not isinstance(pilot, dict)
        or pilot.get("status") != "PASS"
        or pilot.get("workers") != [1, WORKERS]
        or not isinstance(pilot.get("receipt_path"), str)
        or not isinstance(pilot.get("receipt_sha256"), str)
    ):
        raise InvalidLocalization("PILOT_W1_W4_EQUIVALENCE_NOT_PASS")
    pilot_receipt_path = _binding(
        root,
        pilot["receipt_path"],
        pilot["receipt_sha256"],
        "PILOT_EQUIVALENCE_RECEIPT",
    )
    pilot_parent = (root / PILOT_PARENT_RELATIVE).resolve()
    if (
        pilot_receipt_path == (root / FORMAL_ROOT_RELATIVE).resolve()
        or not pilot_receipt_path.is_relative_to(pilot_parent)
    ):
        raise InvalidLocalization("PILOT_EQUIVALENCE_RECEIPT_PATH")
    pilot_receipt = load_json(pilot_receipt_path)
    if (
        pilot_receipt.get("schema") != PILOT_EQUIVALENCE_SCHEMA
        or pilot_receipt.get("protocol_id") != PROTOCOL_ID
        or pilot_receipt.get("task_id") != TASK_ID
        or pilot_receipt.get("valid") is not True
        or pilot_receipt.get("terminal") != PILOT_EQUIVALENCE_TERMINAL
        or pilot_receipt.get("cluster_count") != PILOT_CLUSTER_COUNT
        or pilot_receipt.get("formal_authority_consumed") is not False
        or pilot_receipt.get("scientific_interpretation") is not False
        or pilot_receipt.get("runner_source_sha256")
        != implementation_hashes["runner"]
        or pilot_receipt.get("validator_source_sha256")
        != implementation_hashes["independent_execution_validator"]
    ):
        raise InvalidLocalization("PILOT_EQUIVALENCE_RECEIPT_INVALID")
    pilot_checks = pilot_receipt.get("checks")
    if (
        not isinstance(pilot_checks, dict)
        or not pilot_checks
        or not all(value == "PASS" for value in pilot_checks.values())
    ):
        raise InvalidLocalization("PILOT_EQUIVALENCE_CHECKS")
    pilot_paths: dict[str, Path] = {}
    for key in ("w1_root", "w4_root", "pilot_manifest_path"):
        value = pilot_receipt.get(key)
        if not isinstance(value, str) or Path(value).is_absolute():
            raise InvalidLocalization(f"PILOT_EQUIVALENCE_PATH:{key}")
        candidate = (root / value).resolve()
        if not candidate.is_relative_to(pilot_parent):
            raise InvalidLocalization(f"PILOT_EQUIVALENCE_PATH:{key}")
        pilot_paths[key] = candidate
    if pilot_receipt["w1_root"] == pilot_receipt["w4_root"]:
        raise InvalidLocalization("PILOT_EQUIVALENCE_DISTINCT_RUNS")
    if (
        pilot.get("semantic_payload_sha256")
        != pilot_receipt.get("semantic_payload_sha256")
        or pilot.get("pilot_manifest_sha256")
        != pilot_receipt.get("pilot_manifest_sha256")
    ):
        raise InvalidLocalization("PILOT_EQUIVALENCE_SUMMARY_BINDING")
    if (
        not pilot_paths["pilot_manifest_path"].is_file()
        or sha256_file(pilot_paths["pilot_manifest_path"])
        != pilot_receipt.get("pilot_manifest_sha256")
    ):
        raise InvalidLocalization("PILOT_EQUIVALENCE_MANIFEST_HASH")
    pilot_contract = spec.get("pilot_w1_w4_equivalence")
    pilot_evidence = (
        pilot_contract.get("frozen_evidence")
        if isinstance(pilot_contract, dict)
        else None
    )
    if (
        not isinstance(pilot_evidence, dict)
        or pilot_evidence.get("equivalence_receipt_path")
        != pilot["receipt_path"]
        or pilot_evidence.get("equivalence_receipt_sha256")
        != pilot["receipt_sha256"]
        or pilot_evidence.get("semantic_payload_sha256")
        != pilot_receipt.get("semantic_payload_sha256")
        or pilot_evidence.get("pilot_manifest_path")
        != pilot_receipt.get("pilot_manifest_path")
        or pilot_evidence.get("pilot_manifest_sha256")
        != pilot_receipt.get("pilot_manifest_sha256")
        or pilot_evidence.get("w1_root") != pilot_receipt.get("w1_root")
        or pilot_evidence.get("w4_root") != pilot_receipt.get("w4_root")
        or pilot_evidence.get("status") != PILOT_EQUIVALENCE_TERMINAL
        or pilot_evidence.get("scientific_interpretation") is not False
    ):
        raise InvalidLocalization("SPEC_PILOT_EVIDENCE_BINDING")
    disjoint_relative = pilot_evidence.get("pilot_disjoint_receipt_path")
    disjoint_digest = pilot_evidence.get("pilot_disjoint_receipt_sha256")
    if (
        not isinstance(disjoint_relative, str)
        or not isinstance(disjoint_digest, str)
        or len(disjoint_digest) != 64
    ):
        raise InvalidLocalization("SPEC_PILOT_DISJOINT_BINDING")
    disjoint_path = _binding(
        root,
        disjoint_relative,
        disjoint_digest,
        "SPEC_PILOT_DISJOINT",
    )
    if not disjoint_path.is_relative_to(pilot_parent):
        raise InvalidLocalization("SPEC_PILOT_DISJOINT_PATH")
    for prefix, root_key in (("w1", "w1_root"), ("w4", "w4_root")):
        pilot_run_root = pilot_paths[root_key]
        if not pilot_run_root.is_dir():
            raise InvalidLocalization(f"SPEC_PILOT_RUN_ROOT:{prefix}")
        for filename, digest_key in (
            ("run_receipt.json", f"{prefix}_run_receipt_sha256"),
            (
                "independent_validation_receipt.json",
                f"{prefix}_independent_validation_receipt_sha256",
            ),
        ):
            artifact = pilot_run_root / filename
            digest = pilot_evidence.get(digest_key)
            if (
                not artifact.is_file()
                or not isinstance(digest, str)
                or sha256_file(artifact) != digest
            ):
                raise InvalidLocalization(
                    f"SPEC_PILOT_ARTIFACT:{prefix}:{filename}"
                )
    test_gate = receipt.get("preformal_test_gate")
    if (
        not isinstance(test_gate, dict)
        or test_gate.get("status") != "PASS"
        or not isinstance(test_gate.get("receipt_path"), str)
        or not isinstance(test_gate.get("receipt_sha256"), str)
    ):
        raise InvalidLocalization("PREFORMAL_TEST_GATE_NOT_PASS")
    test_receipt_path = _binding(
        root,
        test_gate["receipt_path"],
        test_gate["receipt_sha256"],
        "PREFORMAL_TEST_RECEIPT",
    )
    if not test_receipt_path.is_relative_to(pilot_parent):
        raise InvalidLocalization("PREFORMAL_TEST_RECEIPT_PATH")
    test_receipt = load_json(test_receipt_path)
    test_checks = test_receipt.get("checks")
    test_count = test_receipt.get("test_count")
    failure_count = test_receipt.get("failure_count")
    if (
        test_receipt.get("schema") != PREFORMAL_TEST_SCHEMA
        or test_receipt.get("protocol_id") != PROTOCOL_ID
        or test_receipt.get("task_id") != TASK_ID
        or test_receipt.get("status") != "PASS"
        or test_receipt.get("terminal") != PREFORMAL_TEST_TERMINAL
        or not isinstance(test_count, int)
        or isinstance(test_count, bool)
        or test_count < 23
        or not isinstance(failure_count, int)
        or isinstance(failure_count, bool)
        or failure_count != 0
        or test_receipt.get("formal_authority_consumed") is not False
        or test_receipt.get("formal_output_root_access") is not False
        or test_receipt.get("sealed_input_access") is not False
        or not isinstance(test_checks, dict)
        or not test_checks
        or not all(value == "PASS" for value in test_checks.values())
        or not {
            "unit_and_mutation_suite",
            "observation_hook_transparency",
            "postclaim_failure_closure",
            "primitive_provenance_mutations",
        }.issubset(test_checks)
    ):
        raise InvalidLocalization("PREFORMAL_TEST_RECEIPT_INVALID")
    test_sources = test_receipt.get("implementation_bindings")
    if (
        not isinstance(test_sources, dict)
        or set(test_sources) != set(roles)
        or any(
            test_sources.get(role)
            != {
                "path": implementations[role]["path"],
                "sha256": implementation_hashes[role],
            }
            for role in roles
        )
    ):
        raise InvalidLocalization("PREFORMAL_TEST_SOURCE_BINDING")
    if (
        test_gate.get("test_count") != test_receipt.get("test_count")
        or test_gate.get("test_suite_sha256")
        != implementation_hashes["test_suite"]
    ):
        raise InvalidLocalization("PREFORMAL_TEST_SUMMARY_BINDING")
    preformal_contract = spec.get("preformal_test_gate")
    preformal_evidence = (
        preformal_contract.get("frozen_evidence")
        if isinstance(preformal_contract, dict)
        else None
    )
    if (
        not isinstance(preformal_evidence, dict)
        or preformal_evidence.get("receipt_path")
        != test_gate["receipt_path"]
        or preformal_evidence.get("receipt_sha256")
        != test_gate["receipt_sha256"]
        or preformal_evidence.get("test_count") != test_count
        or preformal_evidence.get("failure_count") != failure_count
        or preformal_evidence.get("terminal") != PREFORMAL_TEST_TERMINAL
    ):
        raise InvalidLocalization("SPEC_PREFORMAL_EVIDENCE_BINDING")
    host = receipt.get("host_preflight")
    if (
        not isinstance(host, dict)
        or set(host) != {"path", "sha256"}
        or not isinstance(host.get("path"), str)
        or not isinstance(host.get("sha256"), str)
        or len(host["sha256"]) != 64
    ):
        raise InvalidLocalization("HOST_PREFLIGHT_READY_BINDING")
    host_receipt_path = _binding(
        root,
        host["path"],
        host["sha256"],
        "HOST_PREFLIGHT_READY",
    )
    host_contract = spec.get("host_preflight_contract")
    host_evidence = (
        host_contract.get("frozen_evidence")
        if isinstance(host_contract, dict)
        else None
    )
    if (
        not isinstance(host_evidence, dict)
        or host_evidence.get("receipt_path") != host["path"]
        or host_evidence.get("receipt_sha256") != host["sha256"]
        or host_evidence.get("status") != "QUALIFIED"
        or host_evidence.get("workers") != WORKERS
        or not host_receipt_path.is_relative_to(root.resolve())
    ):
        raise InvalidLocalization("SPEC_HOST_EVIDENCE_BINDING")
    source_bindings = []
    for role, path in (
        ("GENERATOR_GEOMETRY", GENERATOR_GEOMETRY_RELATIVE),
        ("MATERIAL_RESIDUAL_RENDERER", MATERIAL_RENDERER_RELATIVE),
    ):
        item = spec_binding_by_path.get(path)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
        ):
            raise InvalidLocalization(f"SPEC_SOURCE_BINDING:{role}")
        _binding(root, path, item["sha256"], f"SPEC_{role}")
        source_bindings.append(
            {"role": role, "path": path, "sha256": item["sha256"]}
        )
    return {
        "spec": spec,
        "spec_path": spec_path,
        "spec_sha256": spec_sha256,
        "implementation_ready_receipt": receipt,
        "implementation_ready_receipt_path": ready_path,
        "implementation_ready_receipt_sha256": sha256_file(ready_path),
        "implementation_hashes": implementation_hashes,
        "source_bindings": source_bindings,
        "pilot_equivalence_receipt": pilot_receipt,
        "preformal_test_receipt": test_receipt,
        "host_preflight_binding": host,
    }


def validate_host_preflight_receipt(
    root: Path,
    receipt_path: Path,
    implementation_ready: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Require a bound guarded-host W4 receipt before a formal claim."""
    candidate = receipt_path.resolve()
    if not candidate.is_file():
        raise InvalidLocalization("HOST_PREFLIGHT_RECEIPT_MISSING")
    ready_binding = implementation_ready["host_preflight_binding"]
    expected_path = (root / ready_binding["path"]).resolve()
    if candidate != expected_path:
        raise InvalidLocalization("HOST_PREFLIGHT_RECEIPT_PATH")
    digest = sha256_file(candidate)
    if ready_binding["sha256"] != digest:
        raise InvalidLocalization("HOST_PREFLIGHT_RECEIPT_UNBOUND")
    receipt = load_json(candidate)
    implementation = receipt.get("implementation", {})
    workload = receipt.get("workload", {})
    pilot = receipt.get("pilot", {})
    scheduler = receipt.get("scheduler", {})
    progress = receipt.get("progress", {})
    terminal = receipt.get("terminal", {})
    formal = receipt.get("formal", {})
    guarded_launcher = receipt.get("guarded_launcher", {})
    mappings = (
        implementation,
        workload,
        pilot,
        scheduler,
        progress,
        terminal,
        formal,
        guarded_launcher,
    )
    if not all(isinstance(value, dict) for value in mappings):
        raise InvalidLocalization("HOST_PREFLIGHT_RECEIPT_INVALID")
    runner_path = Path(__file__).resolve()
    runner_relative = runner_path.relative_to(root).as_posix()
    expected_root = FORMAL_ROOT_RELATIVE
    required_progress_fields = {
        "phase",
        "completed_units",
        "total_units",
        "throughput",
        "eta_seconds",
        "last_progress_at",
        "status",
    }
    progress_samples = pilot.get("progress_samples")
    reserve_memory = scheduler.get("reserve_memory_gib")
    estimated_per_worker = scheduler.get("estimated_gib_per_worker")
    progress_fields = progress.get("fields")
    update_interval = progress.get("update_interval_seconds")
    valid_progress_samples = (
        isinstance(progress_samples, int)
        and not isinstance(progress_samples, bool)
        and progress_samples >= 2
    )
    valid_reserve_memory = (
        isinstance(reserve_memory, (int, float))
        and not isinstance(reserve_memory, bool)
        and math.isfinite(float(reserve_memory))
        and float(reserve_memory) >= 6.0
    )
    valid_estimated_per_worker = (
        isinstance(estimated_per_worker, (int, float))
        and not isinstance(estimated_per_worker, bool)
        and math.isfinite(float(estimated_per_worker))
        and float(estimated_per_worker) > 0.0
    )
    valid_progress_fields = (
        isinstance(progress_fields, list)
        and all(isinstance(field, str) for field in progress_fields)
        and required_progress_fields.issubset(set(progress_fields))
    )
    valid_update_interval = (
        isinstance(update_interval, (int, float))
        and not isinstance(update_interval, bool)
        and math.isfinite(float(update_interval))
        and 0.0 < float(update_interval) <= 60.0
    )
    if (
        receipt.get("schema_version") != HOST_PREFLIGHT_SCHEMA
        or receipt.get("task_id") != HOST_PREFLIGHT_TASK_ID
        or receipt.get("execution_class") != "formal"
        or implementation
        != {
            "script": runner_relative,
            "sha256": sha256_file(runner_path),
        }
        or workload.get("class") != "cpu_data_parallel"
        or workload.get("real_data_mechanics_match") is not True
        or workload.get("input_identity")
        != f"identity_lock:{authority['identity_sha256']}"
        or pilot.get("representative_units") != 32
        or pilot.get("projected_full_units") != 8 * PAIR_COUNT
        or pilot.get("same_access_mechanics") is not True
        or pilot.get("output_equivalence") != "PASS"
        or not valid_progress_samples
        or scheduler.get("backend") != "cpu_process_pool"
        or scheduler.get("workers") != WORKERS
        or scheduler.get("comparison_performed") is not True
        or scheduler.get("scientific_parameters_unchanged") is not True
        or scheduler.get("inject_workers") is not True
        or not valid_reserve_memory
        or not valid_estimated_per_worker
        or progress.get("path") != f"{expected_root}/progress.json"
        or not valid_progress_fields
        or progress.get("verified_in_pilot") is not True
        or not valid_update_interval
        or terminal.get("success_path") != f"{expected_root}/success.json"
        or terminal.get("failure_path") != f"{expected_root}/failure.json"
        or formal.get("one_shot") is not True
        or formal.get("claim_created_by_runner_only") is not True
        or formal.get("claim_path") != f"{expected_root}/claim.json"
        or formal.get("output_path") != f"{expected_root}/success.json"
        or formal.get("failure_receipt_path")
        != f"{expected_root}/failure.json"
        or formal.get("activation_authority")
        != f"{ACTIVATION_RELATIVE}:{authority['activation_sha256']}"
        or guarded_launcher.get("script")
        != GUARDED_HOST_LAUNCHER_RELATIVE
        or not isinstance(guarded_launcher.get("sha256"), str)
    ):
        raise InvalidLocalization("HOST_PREFLIGHT_RECEIPT_INVALID")
    _binding(
        root,
        guarded_launcher["script"],
        guarded_launcher["sha256"],
        "GUARDED_HOST_LAUNCHER",
    )
    return {"path": candidate, "sha256": digest, "receipt": receipt}


def validate_authority(root: Path) -> dict[str, Any]:
    """Validate common authority without opening sealed Stage-B payloads."""
    activation_path = root / ACTIVATION_RELATIVE
    activation = load_json(activation_path)
    if (
        activation.get("decision")
        != "AUTHORIZE_ROTATION_LEAKAGE_LOCALIZATION_ONE_SHOT_EXECUTION"
        or activation.get("execution_authorized") is not True
        or activation.get("r3_modification_authorized") is not False
        or activation.get("stage_b_rerun_authorized") is not False
        or activation.get("single_variable_repair_authorized") is not False
    ):
        raise InvalidLocalization("ACTIVATION_AUTHORITY")
    execution = activation.get("execution", {})
    if (
        execution.get("cluster_count") != 8
        or execution.get("fixed_pairs_per_cluster") != PAIR_COUNT
        or execution.get("workers") != WORKERS
        or execution.get("one_shot") is not True
        or execution.get("write_once_output") is not True
        or execution.get("source_localization_workload_run") is not False
        or execution.get("output_root") != FORMAL_ROOT_RELATIVE
    ):
        raise InvalidLocalization("ACTIVATION_EXECUTION_LOCK")
    if activation.get("resource_gate") != {
        "in_flight_emergency_floor_bytes": IN_FLIGHT_FLOOR_BYTES,
        "launch_and_refill_minimum_available_ram_bytes": LAUNCH_REFILL_BYTES,
        "launch_check_required": True,
        "sustained_paging_stop": True,
    }:
        raise InvalidLocalization("ACTIVATION_RESOURCE_LOCK")
    activation_records = activation.get("bindings")
    if not isinstance(activation_records, list):
        raise InvalidLocalization("ACTIVATION_BINDING_KEYSET")
    activation_bindings: dict[str, dict[str, Any]] = {}
    for item in activation_records:
        if not isinstance(item, dict):
            raise InvalidLocalization("ACTIVATION_BINDING_SCHEMA")
        role = item.get("role")
        if not isinstance(role, str) or role in activation_bindings:
            raise InvalidLocalization("ACTIVATION_BINDING_SCHEMA")
        activation_bindings[role] = item
    if set(activation_bindings) != set(ACTIVATION_BINDING_PATHS):
        raise InvalidLocalization("ACTIVATION_BINDING_KEYSET")
    for role, expected_path in ACTIVATION_BINDING_PATHS.items():
        item = activation_bindings[role]
        digest = item.get("sha256")
        if (
            item.get("path") != expected_path
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise InvalidLocalization(f"ACTIVATION_BINDING_PATH:{role}")
        _binding(root, expected_path, digest, f"ACTIVATION_{role}")

    contract_path = root / CONTRACT_RELATIVE
    identity_path = root / IDENTITY_RELATIVE
    receipt_path = root / PREFLIGHT_RECEIPT_RELATIVE
    contract = load_json(contract_path)
    receipt = load_json(receipt_path)
    receipt_checks = receipt.get("checks")
    if (
        contract.get("task_id") != TASK_ID
        or receipt.get("task_id") != TASK_ID
        or receipt.get("protocol_status") != "VALID"
        or receipt.get("check_count") != 10
        or not isinstance(receipt_checks, dict)
        or not receipt_checks
        or not all(value == "PASS" for value in receipt_checks.values())
    ):
        raise InvalidLocalization("PREFLIGHT_IDENTITY")
    if (
        sha256_file(contract_path)
        != activation_bindings["FROZEN_CONTRACT"]["sha256"]
        or sha256_file(identity_path)
        != activation_bindings["IDENTITY_INPUT_LOCK"]["sha256"]
        or sha256_file(receipt_path)
        != activation_bindings["INDEPENDENT_PREFLIGHT_RECEIPT"]["sha256"]
    ):
        raise InvalidLocalization("ACTIVATION_PRIMARY_BINDING")
    contract_bindings = contract.get("frozen_bindings")
    if (
        not isinstance(contract_bindings, dict)
        or set(contract_bindings) != set(CONTRACT_FROZEN_BINDING_PATHS)
    ):
        raise InvalidLocalization("CONTRACT_BINDING_KEYSET")
    for name, expected_path in CONTRACT_FROZEN_BINDING_PATHS.items():
        item = contract_bindings.get(name)
        if (
            not isinstance(item, dict)
            or item.get("path") != expected_path
        ):
            raise InvalidLocalization(f"CONTRACT_BINDING_PATH:{name}")
        if name == "identity_input_lock":
            if (
                set(item) != {
                    "path",
                    "sha256_filled_by_validator_receipt",
                }
                or item.get("sha256_filled_by_validator_receipt") is not True
            ):
                raise InvalidLocalization(
                    "CONTRACT_BINDING_PLACEHOLDER:identity_input_lock"
                )
            continue
        expected = item.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise InvalidLocalization(f"CONTRACT_BINDING_HASH:{name}")
        _binding(root, item["path"], expected, f"CONTRACT_{name}")
    protocol_path = root / CONTRACT_FROZEN_BINDING_PATHS["r3_parameters"]
    protocol = load_json(protocol_path)
    _protocol_parameter_guard(contract, protocol)
    return {
        "activation": activation,
        "activation_sha256": sha256_file(activation_path),
        "contract": contract,
        "contract_sha256": sha256_file(contract_path),
        "identity_path": identity_path,
        "identity_sha256": sha256_file(identity_path),
        "preflight_receipt_sha256": sha256_file(receipt_path),
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "source_bindings": [
            {
                "role": "GENERATOR_GEOMETRY",
                "path": GENERATOR_GEOMETRY_RELATIVE,
                "sha256": sha256_file(root / GENERATOR_GEOMETRY_RELATIVE),
            },
            {
                "role": "MATERIAL_RESIDUAL_RENDERER",
                "path": MATERIAL_RENDERER_RELATIVE,
                "sha256": sha256_file(root / MATERIAL_RENDERER_RELATIVE),
            },
        ],
    }


def validate_formal_inputs(
    root: Path, common: dict[str, Any]
) -> dict[str, Any]:
    """Open and bind sealed Stage-B inputs only after the formal claim."""
    identity = load_json(common["identity_path"])
    if identity.get("counts") != {
        "clusters": 8,
        "sequences": 8,
        "frames_per_sequence": FRAME_COUNT,
        "pairs_per_sequence": PAIR_COUNT,
        "pair_records_are_longitudinal_repeats": True,
        "analysis_unit": "cluster",
    }:
        raise InvalidLocalization("LOCALIZATION_COUNTS")
    for name, item in identity.get("bindings", {}).items():
        _binding(root, item["path"], item["sha256"], f"IDENTITY_{name}")
    stage_b_identity = load_json(root / STAGE_B_IDENTITY_RELATIVE)
    geometry_manifest = load_json(root / STAGE_B_GEOMETRY_RELATIVE)
    if (
        sha256_file(root / STAGE_B_IDENTITY_RELATIVE)
        != identity["bindings"]["stage_b_identity_lock"]["sha256"]
        or sha256_file(root / STAGE_B_GEOMETRY_RELATIVE)
        != identity["bindings"]["stage_b_geometry_manifest"]["sha256"]
        or geometry_manifest.get("cluster_count") != 8
    ):
        raise InvalidLocalization("STAGE_B_INPUT_BINDING")
    trajectory_path = root / TRAJECTORY_RELATIVE
    if (
        not trajectory_path.is_file()
        or sha256_file(trajectory_path)
        != geometry_manifest.get("trajectory_manifest_sha256")
    ):
        raise InvalidLocalization("TRAJECTORY_BINDING")
    return {
        **common,
        "identity": identity,
        "stage_b_identity": stage_b_identity,
        "geometry_manifest": geometry_manifest,
        "trajectory_manifest": load_json(trajectory_path),
        "trajectory_manifest_sha256": sha256_file(trajectory_path),
    }


def _dynamic_static_scene(
    base: dict[str, Any], frame_index: int
) -> dict[str, Any]:
    scene = json.loads(json.dumps(base))
    target = next(
        item
        for item in scene["world"]["objects"]
        if int(item["object_id"]) == TARGET_ID
    )
    target["plane_z_m"] = 6.0
    target["bounds_xy_m"] = [-0.4, 0.8, -0.7, 0.9]
    target["vertices_world_m"] = [
        [-0.4, -0.7, 6.0],
        [0.8, -0.7, 6.0],
        [0.8, 0.9, 6.0],
        [-0.4, 0.9, 6.0],
    ]
    scene.pop("scene_geometry_sha256", None)
    scene["frame_index"] = frame_index
    scene["target_motion"] = "STATIC"
    scene["scene_geometry_sha256"] = sha256_value(scene)
    return scene


def _rotation_poses(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
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


def _render_input_sha(
    base_scene: dict[str, Any], poses: Sequence[dict[str, Any]]
) -> str:
    rows = []
    for frame_index, pose in enumerate(poses):
        scene = _dynamic_static_scene(base_scene, frame_index)
        rows.append(
            {
                "frame_index": frame_index,
                "scene_geometry_sha256": scene["scene_geometry_sha256"],
                "rotation_matrix": pose["rotation_matrix"],
                "translation_m": pose["translation_m"],
                "target_z_m": 6.0,
            }
        )
    return sha256_value(rows)


def formal_tasks(authority: dict[str, Any]) -> list[dict[str, Any]]:
    localization = {
        item["cluster_id"]: item for item in authority["identity"]["clusters"]
    }
    stage_b = {
        item["cluster_id"]: item
        for item in authority["stage_b_identity"]["clusters"]
    }
    manifests = {
        item["cluster_id"]: item
        for item in authority["geometry_manifest"]["clusters"]
    }
    if set(localization) != set(stage_b) or set(localization) != set(manifests):
        raise InvalidLocalization("FORMAL_CLUSTER_JOIN")
    tasks = []
    for cluster_id in sorted(
        localization,
        key=lambda value: (
            localization[value]["block"],
            localization[value]["ordinal"],
        ),
    ):
        locked = localization[cluster_id]
        source = stage_b[cluster_id]
        materialized = manifests[cluster_id]
        arm = next(
            item
            for item in materialized["arms"]
            if item["arm"] == "EGO_ROTATION_STATIC_SCENE"
        )
        trajectory = authority["trajectory_manifest"][locked["block"]]
        poses = _rotation_poses(trajectory)
        base_scene = materialized["base_scene"]
        checks = {
            "scene_geometry_sha256": base_scene["scene_geometry_sha256"],
            "pose_sha256": sha256_value(poses),
            "render_input_sha256": _render_input_sha(base_scene, poses),
        }
        if any(checks[key] != locked[key] for key in checks):
            raise InvalidLocalization(f"FORMAL_IDENTITY:{cluster_id}")
        if (
            arm["pose_sha256"] != locked["pose_sha256"]
            or arm["render_input_sha256"] != locked["render_input_sha256"]
            or len(poses) != FRAME_COUNT
            or source["numeric_seed_uint64"]
            != materialized["numeric_seed_uint64"]
        ):
            raise InvalidLocalization(f"FORMAL_MATERIALIZATION:{cluster_id}")
        tasks.append(
            {
                "mode": FORMAL_MODE,
                "cluster_id": cluster_id,
                "sequence_id": locked["sequence_id"],
                "block": locked["block"],
                "ordinal": locked["ordinal"],
                "base_scene": base_scene,
                "poses": poses,
                "pair_count": PAIR_COUNT,
                "identity_hashes": checks,
                "protocol_sha256": authority["protocol_sha256"],
            }
        )
    if len(tasks) != 8:
        raise InvalidLocalization("FORMAL_TASK_COUNT")
    return tasks


def pilot_tasks(
    authority: dict[str, Any], pilot_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate an explicit source-known pilot without reading sealed payloads."""
    fixture = pilot_manifest.get("fixture", {})
    fixture_id = pilot_manifest.get(
        "fixture_id",
        fixture.get("id") if isinstance(fixture, dict) else None,
    )
    source_role = pilot_manifest.get(
        "source_role",
        fixture.get("source_role") if isinstance(fixture, dict) else None,
    )
    if (
        pilot_manifest.get("schema")
        != "rcle.r3_rotation_leakage_source_localization.pilot_input.v1"
        or pilot_manifest.get("mode") != "DISJOINT_PILOT"
        or pilot_manifest.get("response_blind") is not True
        or fixture_id != PILOT_FIXTURE_ID
        or source_role != PILOT_SOURCE_ROLE
    ):
        raise InvalidLocalization("PILOT_MANIFEST")
    clusters = pilot_manifest.get("clusters")
    if (
        not isinstance(clusters, list)
        or len(clusters) != PILOT_CLUSTER_COUNT
    ):
        raise InvalidLocalization("PILOT_CLUSTER_COUNT")
    tasks = []
    pilot_seeds: set[int] = set()
    for item in clusters:
        poses = item.get("poses")
        base_scene = item.get("base_scene")
        seed = item.get("numeric_seed_uint64")
        if (
            not isinstance(poses, list)
            or len(poses) != PILOT_FRAME_COUNT
            or not isinstance(base_scene, dict)
            or not isinstance(seed, int)
            or seed < 0
            or seed >= 2**64
            or seed in pilot_seeds
            or not str(item.get("cluster_id", "")).startswith("PILOT_ONLY_")
            or not str(item.get("sequence_id", "")).startswith("PILOT_ONLY_")
        ):
            raise InvalidLocalization("PILOT_INPUT_SHAPE")
        pilot_seeds.add(seed)
        hashes = {
            "scene_geometry_sha256": base_scene.get(
                "scene_geometry_sha256"
            ),
            "pose_sha256": sha256_value(poses),
            "render_input_sha256": _render_input_sha(base_scene, poses),
            "numeric_seed_uint64": seed,
        }
        if any(item.get(key) != value for key, value in hashes.items()):
            raise InvalidLocalization("PILOT_HASH")
        tasks.append(
            {
                "mode": PILOT_MODE,
                "cluster_id": item["cluster_id"],
                "sequence_id": item["sequence_id"],
                "block": item.get("block", "PILOT"),
                "ordinal": int(item.get("ordinal", 0)),
                "base_scene": base_scene,
                "poses": poses,
                "pair_count": len(poses) - 1,
                "identity_hashes": hashes,
                "protocol_sha256": authority["protocol_sha256"],
            }
        )
    return tasks


def validate_pilot_disjoint_receipt(
    authority: dict[str, Any],
    pilot_manifest_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Bind a separately produced disjointness PASS without parsing identity."""
    root = repo_root()
    pilot_parent = (root / PILOT_PARENT_RELATIVE).resolve()
    candidate = receipt_path.resolve()
    manifest_path = pilot_manifest_path.resolve()
    if (
        not candidate.is_file()
        or not candidate.is_relative_to(pilot_parent)
        or not manifest_path.is_file()
        or not manifest_path.is_relative_to(pilot_parent)
    ):
        raise InvalidLocalization("PILOT_DISJOINT_RECEIPT_MISSING")
    receipt = load_json(candidate)
    checks = receipt.get("checks")
    counts = receipt.get("counts", {})
    manifest_binding = receipt.get("pilot_manifest", {})
    validator = receipt.get("validator", {})
    if (
        receipt.get("schema")
        != (
            "rcle.r3_rotation_leakage_source_localization."
            "pilot_fixture_independent_receipt.v1"
        )
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("task_id") != TASK_ID
        or receipt.get("status") != "PASS"
        or receipt.get("terminal")
        != (
            "PILOT_FIXTURE_DISJOINT_PASS / "
            "FORMAL_AUTHORITY_NOT_CONSUMED / "
            "SCIENTIFICALLY_NOT_INTERPRETABLE"
        )
        or receipt.get("fixture_id") != PILOT_FIXTURE_ID
        or receipt.get("source_role") != PILOT_SOURCE_ROLE
        or receipt.get("pilot_manifest_sha256")
        != sha256_file(manifest_path)
        or manifest_binding
        != {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
        }
        or receipt.get("formal_identity_lock_sha256")
        != authority["identity_sha256"]
        or receipt.get("sealed_cluster_access") is not False
        or receipt.get("sealed_response_payload_access") is not False
        or receipt.get("identity_lock_payload_access") is not False
        or receipt.get("formal_output_root_access") is not False
        or receipt.get("formal_authority_consumed") is not False
        or receipt.get("scientific_interpretation") is not False
        or counts.get("clusters") != PILOT_CLUSTER_COUNT
        or counts.get("frames_per_cluster") != PILOT_FRAME_COUNT
        or counts.get("pairs_per_cluster") != PILOT_FRAME_COUNT - 1
        or not isinstance(checks, list)
        or len(checks) != 11
        or any(
            not isinstance(item, dict) or item.get("status") != "PASS"
            for item in checks
        )
        or validator.get("imports_runner_or_r3") is not False
        or not isinstance(validator.get("path"), str)
        or not isinstance(validator.get("sha256"), str)
        or sha256_file(_binding(
            root,
            validator["path"],
            validator["sha256"],
            "PILOT_FIXTURE_VALIDATOR",
        ))
        != validator["sha256"]
    ):
        raise InvalidLocalization("PILOT_DISJOINT_RECEIPT_INVALID")
    return {
        "path": candidate,
        "sha256": sha256_file(candidate),
        "receipt": receipt,
    }


def _sample_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    if not len(points):
        return result
    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)
    inside = (
        (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    )
    indices = np.flatnonzero(inside)
    result[indices] = mask[y[indices], x[indices]] > 0
    return result


def _cell_mask(
    points: np.ndarray, cell_index: int, image_shape: tuple[int, int]
) -> np.ndarray:
    height, width = image_shape
    row, column = divmod(cell_index, 3)
    x0 = int(round(column * width / 3))
    x1 = int(round((column + 1) * width / 3))
    y0 = int(round(row * height / 3))
    y1 = int(round((row + 1) * height / 3))
    return (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )


def _copy_tracks(value: SparseTrackResult) -> SparseTrackResult:
    return SparseTrackResult(
        previous_points=value.previous_points.copy(),
        current_points=value.current_points.copy(),
        forward_backward_errors=value.forward_backward_errors.copy(),
        requested_count=int(value.requested_count),
    )


def _pair_state_record(state: Any) -> dict[str, Any]:
    survivors = state.survivors
    record: dict[str, Any] = {
        "dt_seconds": state.dt_seconds,
        "survivor_count": 0 if survivors is None else survivors.valid_count,
        "survivor_requested_count": (
            0 if survivors is None else int(survivors.requested_count)
        ),
    }
    if survivors is not None:
        record.update(
            {
                "survivor_previous_sha256": sha256_array(
                    survivors.previous_points
                ),
                "survivor_current_sha256": sha256_array(
                    survivors.current_points
                ),
                "survivor_fb_error_sha256": sha256_array(
                    survivors.forward_backward_errors
                ),
            }
        )
    record["semantic_sha256"] = sha256_value(record)
    return record


def _lk_audit(
    initial: np.ndarray,
    current_mask: np.ndarray,
    parameters: dict[str, Any],
    calls: list[tuple[Any, Any, Any]],
    result: SparseTrackResult,
) -> dict[str, Any]:
    count = len(initial)
    forward_points = np.full((count, 2), np.nan, dtype=np.float32)
    forward_available = np.zeros(count, dtype=bool)
    backward_available = np.zeros(count, dtype=bool)
    best_error = np.full(count, np.inf, dtype=np.float64)
    best_backward_points = np.full((count, 2), np.nan, dtype=np.float32)
    best_backward_level = np.full(count, -1, dtype=np.int16)
    if calls:
        forward, status, _ = calls[0]
        if forward is not None:
            forward_points = np.ascontiguousarray(
                forward.reshape(-1, 2).astype(np.float32)
            )
        if status is not None:
            forward_available = (
                status.reshape(-1) > 0
            ) & np.isfinite(forward_points).all(axis=1)
        maximum_level = int(parameters["max_pyramid_level"])
        for ordinal, (backward, status, _) in enumerate(calls[1:]):
            if backward is None or status is None:
                continue
            backward_flat = backward.reshape(-1, 2)
            errors = np.linalg.norm(
                backward_flat - initial.astype(np.float32), axis=1
            )
            finite = (
                (status.reshape(-1) > 0)
                & np.isfinite(backward_flat).all(axis=1)
                & np.isfinite(errors)
            )
            improve = finite & (errors < best_error)
            best_error[improve] = errors[improve]
            best_backward_points[improve] = backward_flat[improve]
            best_backward_level[improve] = maximum_level - ordinal
            backward_available |= finite
    fb_pass = (
        forward_available
        & backward_available
        & (
            best_error
            <= float(parameters["forward_backward_max_error_pixels"])
        )
    )
    mask_pass = _sample_mask(current_mask, forward_points)
    accepted = fb_pass & mask_pass
    rejection_reason = np.full(count, "ACCEPTED", dtype="<U24")
    rejection_reason[~forward_available] = "FORWARD_UNAVAILABLE"
    rejection_reason[forward_available & ~backward_available] = (
        "BACKWARD_UNAVAILABLE"
    )
    rejection_reason[
        forward_available & backward_available & ~fb_pass
    ] = "FORWARD_BACKWARD_FAIL"
    rejection_reason[fb_pass & ~mask_pass] = "MASK_FAIL"
    if not (
        np.array_equal(
            result.previous_points, initial.astype(np.float32)[accepted]
        )
        and np.array_equal(
            result.current_points, forward_points[accepted]
        )
        and np.array_equal(
            result.forward_backward_errors,
            best_error[accepted].astype(np.float32),
        )
    ):
        raise InvalidLocalization("LK_AUDIT_MISMATCH")
    return {
        "initial": np.ascontiguousarray(initial.astype(np.float32)),
        "forward_points": forward_points,
        "forward_available": forward_available,
        "best_backward_points": best_backward_points,
        "best_backward_level": best_backward_level,
        "best_forward_backward_error": best_error,
        "backward_available": backward_available,
        "fb_pass": fb_pass,
        "mask_pass": mask_pass,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "result": _copy_tracks(result),
    }


def _cell_audit_from_consensus(
    tracks: SparseTrackResult,
    dt: float,
    image_shape: tuple[int, int],
    cell_index: int,
    consensus: np.ndarray,
) -> dict[str, Any]:
    selected = _cell_mask(tracks.previous_points, cell_index, image_shape)
    previous = tracks.previous_points[selected].astype(np.float64)
    current = tracks.current_points[selected].astype(np.float64)
    mask = np.asarray(consensus, dtype=bool)
    if mask.shape != (len(previous),):
        raise InvalidLocalization("CONSENSUS_SHAPE")
    points = previous[mask]
    endpoints = current[mask]
    row, column = divmod(cell_index, 3)
    height, width = image_shape
    x0 = int(round(column * width / 3))
    x1 = int(round((column + 1) * width / 3))
    y0 = int(round(row * height / 3))
    y1 = int(round((row + 1) * height / 3))
    value: dict[str, Any] = {
        "cell_index": cell_index,
        "tracked_support_count": int(len(previous)),
        "consensus_support_count": int(len(points)),
        "coefficients": None,
        "condition_number": None,
        "median_fit_residual_px_per_frame": None,
        "expansion_per_s": None,
        "consensus_previous": points.astype(np.float32),
        "consensus_current": endpoints.astype(np.float32),
    }
    if len(points) < 3:
        return value
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    half_width = max(0.5 * (x1 - x0), 1.0)
    half_height = max(0.5 * (y1 - y0), 1.0)
    design = np.column_stack(
        (
            (points[:, 0] - center_x) / half_width,
            (points[:, 1] - center_y) / half_height,
            np.ones(len(points)),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, (endpoints - points) / dt, rcond=None
    )
    residual = np.linalg.norm(
        design @ coefficients - (endpoints - points) / dt, axis=1
    )
    value.update(
        {
            "coefficients": coefficients.tolist(),
            "condition_number": float(np.linalg.cond(design)),
            "median_fit_residual_px_per_frame": float(
                np.median(residual) * dt
            ),
            "expansion_per_s": float(
                0.5
                * (
                    coefficients[0, 0] / half_width
                    + coefficients[1, 1] / half_height
                )
            ),
        }
    )
    return value


class PairCapture:
    """Pair-local wrappers that return every frozen R3 value unchanged."""

    def __init__(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_mask: np.ndarray,
        current_mask: np.ndarray,
        dt: float,
    ) -> None:
        self.previous_gray = previous_gray
        self.current_gray = current_gray
        self.previous_mask = previous_mask
        self.current_mask = current_mask
        self.dt = dt
        self.compensation: dict[str, Any] | None = None
        self.initial_points: np.ndarray | None = None
        self.track_calls: list[dict[str, Any]] = []
        self.fit_calls: list[dict[str, Any]] = []
        self.observable_calls: list[Any] = []
        self.merge_calls: list[dict[str, Any]] = []
        self.common: dict[str, Any] | None = None
        self._original_compensate = r3.compensate_current_to_previous
        self._original_detect = r3.detect_fixed_grid_features
        self._original_track = r3.track_features
        self._original_fit = r3.fit_fixed_grid_local_affine
        self._original_observable = r3.track_observable_points
        self._original_merge = r3.merge_path_correspondences
        self._original_common = r0_evaluation._common_cell_expansions

    def compensate(self, *args: Any, **kwargs: Any) -> Any:
        result = self._original_compensate(*args, **kwargs)
        self.compensation = {
            "result": result,
            "homography": np.asarray(args[3], dtype=np.float64).copy(),
        }
        return result

    def detect(self, *args: Any, **kwargs: Any) -> np.ndarray:
        result = self._original_detect(*args, **kwargs)
        if self.initial_points is not None:
            raise InvalidLocalization("MULTIPLE_INITIAL_FEATURE_CALLS")
        self.initial_points = result.copy()
        return result

    def track(self, *args: Any, **kwargs: Any) -> SparseTrackResult:
        calls: list[tuple[Any, Any, Any]] = []
        original_lk = sparse_flow_module.cv2.calcOpticalFlowPyrLK

        def spy(*lk_args: Any, **lk_kwargs: Any) -> Any:
            result = original_lk(*lk_args, **lk_kwargs)
            calls.append(
                tuple(
                    None if item is None else np.asarray(item).copy()
                    for item in result
                )
            )
            return result

        with mock.patch.object(
            sparse_flow_module.cv2,
            "calcOpticalFlowPyrLK",
            side_effect=spy,
        ):
            result = self._original_track(*args, **kwargs)
        audit = _lk_audit(
            np.asarray(args[2]),
            np.asarray(args[3]),
            args[4],
            calls,
            result,
        )
        self.track_calls.append(audit)
        return result

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        tracks = _copy_tracks(args[0])
        consensus: list[np.ndarray] = []
        original_consensus = local_affine_module._consensus_mask

        def capture_consensus(*inner_args: Any, **inner_kwargs: Any) -> Any:
            result = original_consensus(*inner_args, **inner_kwargs)
            consensus.append(result.copy())
            return result

        with mock.patch.object(
            local_affine_module,
            "_consensus_mask",
            side_effect=capture_consensus,
        ):
            results = self._original_fit(*args, **kwargs)
        if len(consensus) != 9 or len(results) != 9:
            raise InvalidLocalization("LOCAL_FIT_CELL_COUNT")
        audits = [
            _cell_audit_from_consensus(
                tracks, float(args[1]), args[2], index, consensus[index]
            )
            for index in range(9)
        ]
        self.fit_calls.append(
            {
                "tracks": tracks,
                "results": results,
                "audits": audits,
            }
        )
        return results

    def observable(self, *args: Any, **kwargs: Any) -> Any:
        result = self._original_observable(*args, **kwargs)
        self.observable_calls.append(result)
        return result

    def merge(self, *args: Any, **kwargs: Any) -> SparseTrackResult:
        result = self._original_merge(*args, **kwargs)
        audit = _merged_path_audit(args[0], args[1], args[2], result)
        self.merge_calls.append(
            {"result": _copy_tracks(result), **audit}
        )
        return result

    def common_cells(self, *args: Any, **kwargs: Any) -> Any:
        result = self._original_common(*args, **kwargs)
        if self.common is not None:
            raise InvalidLocalization("MULTIPLE_COMMON_CELL_CALLS")
        self.common = {
            "raw_cells": args[0],
            "compensated_cells": args[1],
            "raw_values": np.asarray(result[0], dtype=np.float64),
            "compensated_values": np.asarray(result[1], dtype=np.float64),
            "indices": list(result[2]),
        }
        return result

    def patches(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                r3,
                "compensate_current_to_previous",
                side_effect=self.compensate,
            )
        )
        stack.enter_context(
            mock.patch.object(
                r3, "detect_fixed_grid_features", side_effect=self.detect
            )
        )
        stack.enter_context(
            mock.patch.object(r3, "track_features", side_effect=self.track)
        )
        stack.enter_context(
            mock.patch.object(
                r3, "fit_fixed_grid_local_affine", side_effect=self.fit
            )
        )
        stack.enter_context(
            mock.patch.object(
                r3, "track_observable_points", side_effect=self.observable
            )
        )
        stack.enter_context(
            mock.patch.object(
                r3, "merge_path_correspondences", side_effect=self.merge
            )
        )
        stack.enter_context(
            mock.patch.object(
                r0_evaluation,
                "_common_cell_expansions",
                side_effect=self.common_cells,
            )
        )
        return stack


def _merged_path_audit(
    baseline: SparseTrackResult,
    carried: Any,
    supplements: Any,
    merged: SparseTrackResult,
) -> dict[str, np.ndarray]:
    candidate_sources: list[np.ndarray] = []
    candidate_targets: list[np.ndarray] = []
    candidate_errors: list[float] = []
    candidate_labels: list[str] = []
    selected: list[bool] = []
    occupied: list[np.ndarray] = []

    def append(
        source: np.ndarray, target: np.ndarray, error: float, label: str
    ) -> None:
        candidate_sources.append(source.astype(np.float32))
        candidate_targets.append(target.astype(np.float32))
        candidate_errors.append(float(error))
        candidate_labels.append(label)
        accept = True
        if occupied:
            squared = np.sum(
                (
                    np.vstack(occupied).astype(np.float64)
                    - source.astype(np.float64)
                )
                ** 2,
                axis=1,
            )
            if bool(np.any(squared < 25.0)):
                accept = False
        selected.append(accept)
        if accept:
            occupied.append(source.astype(np.float32))

    for source, target, error in zip(
        baseline.previous_points,
        baseline.current_points,
        baseline.forward_backward_errors,
        strict=True,
    ):
        append(source, target, float(error), "BASELINE")
    for diagnostics, label in (
        (carried, "CARRIED"),
        (supplements, "SUPPLEMENT"),
    ):
        for source, target, error in zip(
            diagnostics.initial_points[diagnostics.accepted],
            diagnostics.forward_points[diagnostics.accepted],
            diagnostics.forward_backward_errors[diagnostics.accepted],
            strict=True,
        ):
            append(source, target, float(error), label)
    candidates_previous = (
        np.vstack(candidate_sources).astype(np.float32)
        if candidate_sources
        else np.empty((0, 2), dtype=np.float32)
    )
    candidates_current = (
        np.vstack(candidate_targets).astype(np.float32)
        if candidate_targets
        else np.empty((0, 2), dtype=np.float32)
    )
    candidates_fb_error = np.asarray(candidate_errors, dtype=np.float32)
    selected_mask = np.asarray(selected, dtype=bool)
    expected_previous = candidates_previous[selected_mask]
    expected_current = candidates_current[selected_mask]
    expected_error = candidates_fb_error[selected_mask]
    if not (
        np.array_equal(expected_previous, merged.previous_points)
        and np.array_equal(expected_current, merged.current_points)
        and np.array_equal(expected_error, merged.forward_backward_errors)
    ):
        raise InvalidLocalization("MANAGED_PATH_CAPTURE_MISMATCH")
    candidates_path = np.asarray(candidate_labels, dtype="<U16")
    return {
        "candidate_previous": np.ascontiguousarray(candidates_previous),
        "candidate_current": np.ascontiguousarray(candidates_current),
        "candidate_fb_error": np.ascontiguousarray(candidates_fb_error),
        "candidate_path_class": candidates_path,
        "selected_mask": selected_mask,
        "selected_index": np.flatnonzero(selected_mask).astype(np.int32),
        "path_class": candidates_path[selected_mask],
    }


def _final_tracks(
    capture: PairCapture, activated: Sequence[int]
) -> tuple[SparseTrackResult, np.ndarray]:
    if len(capture.fit_calls) not in (2, 4):
        raise InvalidLocalization("FIT_CALL_COUNT")
    initial = capture.fit_calls[1]["tracks"]
    if len(capture.fit_calls) == 2:
        return initial, np.full(initial.valid_count, "BASELINE", dtype="<U16")
    managed = capture.fit_calls[3]["tracks"]
    if len(capture.merge_calls) != 2:
        raise InvalidLocalization("MERGE_CALL_COUNT")
    managed_labels = capture.merge_calls[1]["path_class"]
    previous_parts = []
    current_parts = []
    error_parts = []
    label_parts = []
    active = set(int(value) for value in activated)
    for index in range(9):
        if index in active:
            source = managed
            labels = managed_labels
        else:
            source = initial
            labels = np.full(initial.valid_count, "BASELINE", dtype="<U16")
        selected = _cell_mask(
            source.previous_points,
            index,
            (geometry.HEIGHT, geometry.WIDTH),
        )
        previous_parts.append(source.previous_points[selected])
        current_parts.append(source.current_points[selected])
        error_parts.append(source.forward_backward_errors[selected])
        label_parts.append(labels[selected])
    return (
        SparseTrackResult(
            np.ascontiguousarray(np.vstack(previous_parts).astype(np.float32)),
            np.ascontiguousarray(np.vstack(current_parts).astype(np.float32)),
            np.ascontiguousarray(
                np.concatenate(error_parts).astype(np.float32)
            ),
            initial.requested_count,
        ),
        np.concatenate(label_parts),
    )


def _project_world(
    world: np.ndarray, rotation_wc: np.ndarray, translation_wc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera = (rotation_wc.T @ (world - translation_wc).T).T
    depth = camera[:, 2]
    pixels = np.full((len(world), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera).all(axis=1) & (depth > 0.0)
    pixels[valid, 0] = (
        geometry.K[0, 0] * camera[valid, 0] / depth[valid]
        + geometry.K[0, 2]
    )
    pixels[valid, 1] = (
        geometry.K[1, 1] * camera[valid, 1] / depth[valid]
        + geometry.K[1, 2]
    )
    return pixels, depth


def _homography_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (points.astype(np.float64), np.ones(len(points)))
    )
    mapped = (matrix @ homogeneous.T).T
    valid = np.isfinite(mapped).all(axis=1) & (np.abs(mapped[:, 2]) > 1e-15)
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[valid] = mapped[valid, :2] / mapped[valid, 2, None]
    return result


def _source_correspondence(
    previous_points: np.ndarray,
    previous_scene: dict[str, Any],
    current_scene: dict[str, Any],
    previous_rotation: np.ndarray,
    current_rotation: np.ndarray,
    previous_translation: np.ndarray,
    current_translation: np.ndarray,
    homography: np.ndarray,
) -> dict[str, Any]:
    points = np.asarray(previous_points, dtype=np.float64).reshape(-1, 2)
    depth, object_id, world = geometry._raycast(
        previous_scene, previous_rotation, previous_translation, points
    )
    current_pixels, current_depth = _project_world(
        world, current_rotation, current_translation
    )
    safe = (
        np.isfinite(depth)
        & (depth >= 0.5)
        & (depth <= 25.0)
        & np.isfinite(current_pixels).all(axis=1)
        & (current_pixels[:, 0] >= 0.0)
        & (current_pixels[:, 0] < geometry.WIDTH - 1.0)
        & (current_pixels[:, 1] >= 0.0)
        & (current_pixels[:, 1] < geometry.HEIGHT - 1.0)
    )
    visible = np.zeros(len(points), dtype=bool)
    if np.any(safe):
        current_z, current_id, current_world = geometry._raycast(
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
    aligned = _homography_points(current_pixels, np.linalg.inv(homography))
    valid = (
        safe
        & visible
        & (object_id > 0)
        & np.isfinite(world).all(axis=1)
        & np.isfinite(aligned).all(axis=1)
    )
    return {
        "valid": np.ascontiguousarray(valid),
        "object_id": object_id.astype(np.int32),
        "depth_m": depth.astype(np.float64),
        "world": world.astype(np.float64),
        "current_pixel": current_pixels,
        "aligned_current": aligned,
    }


def _ols_error_cells(
    previous: np.ndarray,
    endpoints: np.ndarray,
    valid: np.ndarray,
    dt: float,
    parameters: dict[str, Any],
    stratum: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    points = np.asarray(previous, dtype=np.float64)
    targets = np.asarray(endpoints, dtype=np.float64)
    keep = np.asarray(valid, dtype=bool).copy()
    if stratum is not None:
        keep &= _sample_mask(
            np.asarray(stratum, dtype=np.uint8) * 255, points
        )
    values = []
    for cell_index in range(9):
        chosen = keep & _cell_mask(
            points, cell_index, (geometry.HEIGHT, geometry.WIDTH)
        )
        source = points[chosen]
        target = targets[chosen]
        row, column = divmod(cell_index, 3)
        x0 = int(round(column * geometry.WIDTH / 3))
        x1 = int(round((column + 1) * geometry.WIDTH / 3))
        y0 = int(round(row * geometry.HEIGHT / 3))
        y1 = int(round((row + 1) * geometry.HEIGHT / 3))
        record: dict[str, Any] = {
            "cell_index": cell_index,
            "support_count": int(len(source)),
            "evaluable": False,
            "reason": "SOURCE_SUPPORT_BELOW_12",
            "signed_per_s": None,
        }
        if len(source) < int(parameters["minimum_tracks_per_cell"]):
            values.append(record)
            continue
        hull = cv2.convexHull(source.astype(np.float32))
        hull_fraction = float(cv2.contourArea(hull)) / max(
            float((x1 - x0) * (y1 - y0)), 1.0
        )
        if hull_fraction < float(
            parameters["minimum_track_convex_hull_fraction"]
        ):
            record["reason"] = "SOURCE_HULL_BELOW_0_10"
            values.append(record)
            continue
        center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        half_width = max(0.5 * (x1 - x0), 1.0)
        half_height = max(0.5 * (y1 - y0), 1.0)
        design = np.column_stack(
            (
                (source[:, 0] - center_x) / half_width,
                (source[:, 1] - center_y) / half_height,
                np.ones(len(source)),
            )
        )
        condition = float(np.linalg.cond(design))
        if (
            not math.isfinite(condition)
            or condition
            > float(parameters["maximum_design_condition_number"])
        ):
            record["reason"] = "SOURCE_CONDITION_ABOVE_1000"
            values.append(record)
            continue
        velocity = (target - source) / dt
        coefficients, _, _, _ = np.linalg.lstsq(
            design, velocity, rcond=None
        )
        residual = float(
            np.median(np.linalg.norm(design @ coefficients - velocity, axis=1))
            * dt
        )
        if residual > float(
            parameters["maximum_median_fit_residual_pixels_per_frame"]
        ):
            record["reason"] = "SOURCE_RESIDUAL_ABOVE_0_75"
            values.append(record)
            continue
        signed = float(
            0.5
            * (
                coefficients[0, 0] / half_width
                + coefficients[1, 1] / half_height
            )
        )
        record.update(
            {
                "evaluable": math.isfinite(signed),
                "reason": None if math.isfinite(signed) else "NONFINITE",
                "signed_per_s": signed if math.isfinite(signed) else None,
                "hull_fraction": hull_fraction,
                "condition_number": condition,
                "median_fit_residual_px_per_frame": residual,
            }
        )
        values.append(record)
    return values


def _pair_from_cells(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(item["signed_per_s"])
        for item in cells
        if item.get("evaluable") is True
        and item.get("signed_per_s") is not None
    ]
    if len(values) < 5:
        return {
            "evaluable": False,
            "evaluable_cell_count": len(values),
            "signed_per_s": None,
            "absolute_per_s": None,
        }
    return {
        "evaluable": True,
        "evaluable_cell_count": len(values),
        "signed_per_s": float(np.median(values)),
        "absolute_per_s": float(np.median(np.abs(values))),
    }


def _warp_audit(
    capture: PairCapture, initial: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if capture.compensation is None:
        raise InvalidLocalization("WARP_CAPTURE_MISSING")
    result = capture.compensation["result"]
    homography = capture.compensation["homography"]
    boundary, interior, common = boundary_masks(
        capture.previous_mask, result.valid_mask
    )
    denominator = int(np.count_nonzero(capture.previous_mask))
    interior_fraction = (
        int(np.count_nonzero(interior)) / denominator if denominator else 0.0
    )
    residual = (
        result.image.astype(np.float64)
        - capture.previous_gray.astype(np.float64)
    ) / 255.0
    interior_values = residual[interior]
    boundary_values = residual[boundary]
    common_y, common_x = np.nonzero(common)
    audit_points = np.column_stack((common_x, common_y)).astype(np.float64)
    roundtrip = _homography_points(
        _homography_points(audit_points, homography),
        np.linalg.inv(homography),
    )
    errors = np.linalg.norm(roundtrip - audit_points, axis=1)
    evaluable = (
        denominator > 0
        and interior_fraction >= MINIMUM_COVERAGE_FRACTION
        and interior_values.size > 0
        and boundary_values.size > 0
        and audit_points.size > 0
        and _finite(errors)
        and _finite(interior_values)
        and _finite(boundary_values)
    )
    return (
        {
            "evaluable": evaluable,
            "overlap_fraction": float(result.overlap_fraction),
            "interior_support_fraction": interior_fraction,
            "previous_gray_sha256": sha256_array(capture.previous_gray),
            "warped_current_gray_sha256": sha256_array(result.image),
            "previous_valid_sha256": sha256_array(capture.previous_mask),
            "warped_current_valid_sha256": sha256_array(
                result.valid_mask
            ),
            "coordinate_roundtrip_p99_px": (
                hf7(errors.tolist(), 0.99) if evaluable else None
            ),
            "interior_gray_signed_p50_normalized": (
                hf7(interior_values.tolist(), 0.5) if evaluable else None
            ),
            "interior_gray_absolute_p50_normalized": (
                hf7(np.abs(interior_values).tolist(), 0.5)
                if evaluable
                else None
            ),
            "interior_gray_absolute_p90_normalized": (
                hf7(np.abs(interior_values).tolist(), 0.9)
                if evaluable
                else None
            ),
            "interior_gray_absolute_p99_normalized": (
                hf7(np.abs(interior_values).tolist(), 0.99)
                if evaluable
                else None
            ),
            "boundary_gray_absolute_p90_normalized": (
                hf7(np.abs(boundary_values).tolist(), 0.9)
                if evaluable
                else None
            ),
            "common_pixel_count": int(np.count_nonzero(common)),
            "previous_valid_pixel_count": denominator,
            "interior_pixel_count": int(np.count_nonzero(interior)),
            "boundary_pixel_count": int(np.count_nonzero(boundary)),
        },
        interior,
        boundary,
    )


def _local_cells(
    capture: PairCapture, row: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    list[tuple[str, int, np.ndarray]],
]:
    if capture.common is None or len(capture.fit_calls) not in (2, 4):
        empty_consensus = [
            (path, cell_index, np.empty((0, 4), dtype=np.float32))
            for path in ("RAW_FINAL", "COMPENSATED_FINAL")
            for cell_index in range(9)
        ]
        return (
            {
                "evaluable": False,
                "numeric_reproduced": False,
                "reproduction_error_max_abs": None,
                "common_cell_indices": [],
                "signed_per_s": None,
                "absolute_per_s": None,
            },
            {"RAW_FINAL": [], "COMPENSATED_FINAL": []},
            empty_consensus,
        )
    activated = set(
        int(value)
        for value in row.get("support_manager", {}).get(
            "activated_cell_indices", []
        )
    )
    raw_initial = capture.fit_calls[0]
    comp_initial = capture.fit_calls[1]
    raw_managed = capture.fit_calls[2] if len(capture.fit_calls) == 4 else None
    comp_managed = capture.fit_calls[3] if len(capture.fit_calls) == 4 else None
    summaries: dict[str, list[dict[str, Any]]] = {
        "RAW_FINAL": [],
        "COMPENSATED_FINAL": [],
    }
    consensus_arrays: list[tuple[str, int, np.ndarray]] = []
    reproduced = True
    reproduction_errors: list[float] = []
    for path, native_cells, initial_source, managed_source in (
        (
            "RAW_FINAL",
            capture.common["raw_cells"],
            raw_initial,
            raw_managed,
        ),
        (
            "COMPENSATED_FINAL",
            capture.common["compensated_cells"],
            comp_initial,
            comp_managed,
        ),
    ):
        for index, result in enumerate(native_cells):
            source = (
                managed_source
                if managed_source is not None and index in activated
                else initial_source
            )
            audit = source["audits"][index]
            error: float | None = None
            if (
                result.evaluable
                and result.expansion is not None
                and audit["expansion_per_s"] is not None
            ):
                error = abs(
                    float(result.expansion)
                    - float(audit["expansion_per_s"])
                )
                match = error <= LOCAL_FIT_NUMERIC_ATOL
                reproduction_errors.append(error)
            else:
                match = (
                    not result.evaluable and result.expansion is None
                )
            reproduced &= match
            summaries[path].append(
                {
                    **asdict(result),
                    "cell_index": index,
                    "path": path,
                    "consensus_support_count": audit[
                        "consensus_support_count"
                    ],
                    "ols_coefficients": audit["coefficients"],
                    "ols_expansion_per_s": audit["expansion_per_s"],
                    "numeric_reproduced": match,
                    "reproduction_error_abs": error,
                }
            )
            consensus_arrays.append(
                (
                    path,
                    index,
                    np.column_stack(
                        (
                            audit["consensus_previous"],
                            audit["consensus_current"],
                        )
                    ),
                )
            )
    indices = capture.common["indices"]
    independently_common = [
        index
        for index in range(9)
        if capture.common["raw_cells"][index].evaluable
        and capture.common["compensated_cells"][index].evaluable
    ]
    reproduced &= independently_common == indices
    raw_values = [
        float(capture.common["raw_cells"][index].expansion)
        for index in indices
    ]
    compensated_values = [
        float(capture.common["compensated_cells"][index].expansion)
        for index in indices
    ]
    evaluable = (
        row.get("evaluable") is True
        and len(raw_values) >= 5
        and len(compensated_values) >= 5
    )
    raw_signed = float(np.median(raw_values)) if evaluable else None
    raw_absolute = (
        float(np.median(np.abs(raw_values))) if evaluable else None
    )
    signed = float(np.median(compensated_values)) if evaluable else None
    absolute = (
        float(np.median(np.abs(compensated_values))) if evaluable else None
    )
    if evaluable and (
        not math.isclose(
            raw_signed,
            float(row["raw_expansion_median_per_s"]),
            abs_tol=LOCAL_FIT_NUMERIC_ATOL,
            rel_tol=0.0,
        )
        or not math.isclose(
            raw_absolute,
            float(row["raw_abs_expansion_median_per_s"]),
            abs_tol=LOCAL_FIT_NUMERIC_ATOL,
            rel_tol=0.0,
        )
        or not math.isclose(
            signed,
            float(row["compensated_expansion_median_per_s"]),
            abs_tol=LOCAL_FIT_NUMERIC_ATOL,
            rel_tol=0.0,
        )
        or not math.isclose(
            absolute,
            float(row["compensated_abs_expansion_median_per_s"]),
            abs_tol=LOCAL_FIT_NUMERIC_ATOL,
            rel_tol=0.0,
        )
    ):
        reproduced = False
    return (
        {
            "evaluable": evaluable,
            "numeric_reproduced": reproduced,
            "reproduction_error_max_abs": (
                max(reproduction_errors) if reproduction_errors else None
            ),
            "common_cell_count": len(indices),
            "common_cell_indices": indices,
            "raw_signed_per_s": raw_signed,
            "raw_absolute_per_s": raw_absolute,
            "signed_per_s": signed,
            "absolute_per_s": absolute,
        },
        summaries,
        consensus_arrays,
    )


def _cluster_metrics(
    rows: list[dict[str, Any]], planned_pairs: int
) -> dict[str, Any]:
    minimum = math.ceil(MINIMUM_COVERAGE_FRACTION * planned_pairs)
    geometry = reduce_layer(rows, "input_geometry", planned_pairs)
    warp = reduce_layer(rows, "rotation_warp", planned_pairs)
    boundary = reduce_layer(rows, "mask_boundary", planned_pairs)
    accepted = reduce_layer(
        rows, "sparse_lk_and_track_filtering.accepted", planned_pairs
    )
    managed = reduce_layer(
        rows, "sparse_lk_and_track_filtering.managed", planned_pairs
    )
    local = reduce_layer(
        rows, "local_affine_and_final_aggregation", planned_pairs
    )
    final = reduce_layer(rows, "r3_pair_row", planned_pairs)

    def covered(value: dict[str, Any]) -> bool:
        return value["evaluable_pair_count"] >= minimum

    geometry_abs = geometry.get("absolute_per_s_p90")
    geometry_roundtrip = geometry.get("roundtrip_max_px_max")
    if not covered(geometry) or geometry_abs is None or geometry_roundtrip is None:
        geometry_status = "NOT_EVALUABLE"
    else:
        geometry_status = (
            "PASS"
            if geometry_abs <= THRESHOLD_PER_S
            and geometry_roundtrip <= 1e-9
            else "FAIL"
        )
    warp_coord = warp.get("coordinate_roundtrip_p99_px_p90")
    warp_photo = warp.get(
        "interior_gray_absolute_p90_normalized_p90"
    )
    warp_overlap = min(
        (
            float(row["rotation_warp"]["overlap_fraction"])
            for row in rows
            if row["rotation_warp"].get("evaluable") is True
        ),
        default=None,
    )
    if (
        not covered(warp)
        or warp_coord is None
        or warp_photo is None
        or warp_overlap is None
    ):
        warp_status = "NOT_EVALUABLE"
    else:
        warp_status = (
            "PASS"
            if warp_coord <= 1e-6
            and warp_photo <= 1.0 / 255.0
            and warp_overlap >= 0.75
            else "FAIL"
        )
    boundary_abs = boundary.get("boundary_absolute_per_s_p90")
    interior_abs = boundary.get("interior_absolute_per_s_p90")
    if not covered(boundary) or boundary_abs is None or interior_abs is None:
        boundary_status = "NOT_EVALUABLE"
        interior_flow_status = "NOT_EVALUABLE"
    else:
        boundary_only = (
            boundary_abs > THRESHOLD_PER_S
            and interior_abs <= THRESHOLD_PER_S
        )
        if boundary_only:
            boundary_status = "BOUNDARY_ONLY_FAIL"
        elif interior_abs > THRESHOLD_PER_S:
            boundary_status = "MASK_NONSEPARABLE"
        else:
            boundary_status = "PASS"
        interior_flow_status = (
            "PASS" if interior_abs <= THRESHOLD_PER_S else "FAIL"
        )
    accepted_abs = accepted.get("absolute_per_s_p90")
    managed_abs = managed.get("absolute_per_s_p90")
    if (
        not covered(accepted)
        or not covered(managed)
        or accepted_abs is None
        or managed_abs is None
    ):
        accepted_status = (
            "NOT_EVALUABLE"
            if not covered(accepted) or accepted_abs is None
            else ("PASS" if accepted_abs <= THRESHOLD_PER_S else "FAIL")
        )
        managed_status = (
            "NOT_EVALUABLE"
            if not covered(managed) or managed_abs is None
            else ("PASS" if managed_abs <= THRESHOLD_PER_S else "FAIL")
        )
    else:
        accepted_status = (
            "PASS" if accepted_abs <= THRESHOLD_PER_S else "FAIL"
        )
        managed_status = (
            "PASS" if managed_abs <= THRESHOLD_PER_S else "FAIL"
        )
    local_abs = local.get("absolute_per_s_p90")
    reproduction = all(
        row["local_affine_and_final_aggregation"].get(
            "numeric_reproduced"
        )
        is True
        for row in rows
        if row["local_affine_and_final_aggregation"].get("evaluable")
        is True
    )
    if not covered(local) or local_abs is None or not reproduction:
        local_status = "NOT_EVALUABLE"
    else:
        local_status = (
            "PASS" if local_abs <= THRESHOLD_PER_S else "FAIL"
        )
    final_abs = final.get("absolute_per_s_p90")
    final_reproduced = reproduction and covered(final)
    metrics = {
        "planned_pair_count": planned_pairs,
        "minimum_evaluable_pair_count": minimum,
        "final": {
            **final,
            "evaluable_pairs": final["evaluable_pair_count"],
            "absolute_p90_per_s": final_abs,
            "reproduced": final_reproduced,
        },
        "geometry": {**geometry, "status": geometry_status},
        "warp": {**warp, "status": warp_status},
        "mask": {**boundary, "status": boundary_status},
        "flow": {
            "accepted_status": accepted_status,
            "managed_status": managed_status,
            "interior_status": interior_flow_status,
            "accepted": accepted,
            "managed": managed,
        },
        "local": {
            **local,
            "status": local_status,
            "numeric_reproduced": reproduction,
        },
        "layers": {
            "INPUT_GEOMETRY": {
                **geometry,
                "status": geometry_status,
            },
            "ROTATION_WARP": {**warp, "status": warp_status},
            "MASK_BOUNDARY": {
                **boundary,
                "status": boundary_status,
                "boundary_only_failure": (
                    boundary_status == "BOUNDARY_ONLY_FAIL"
                ),
                "interior_flow_status": interior_flow_status,
            },
            "SPARSE_LK_AND_TRACK_FILTERING": {
                "status": (
                    "NOT_EVALUABLE"
                    if "NOT_EVALUABLE"
                    in (accepted_status, managed_status)
                    else (
                        "PASS"
                        if accepted_status == managed_status == "PASS"
                        else "FAIL"
                    )
                ),
                "accepted_status": accepted_status,
                "managed_status": managed_status,
                "accepted": accepted,
                "managed": managed,
            },
            "LOCAL_AFFINE_AND_FINAL_AGGREGATION": {
                **local,
                "status": local_status,
                "numeric_reproduced": reproduction,
            },
        },
        "ambiguity_rules": AMBIGUITY_RULES,
        "route": "PENDING_INDEPENDENT_VALIDATION",
    }
    return metrics


class _Ragged:
    def __init__(self) -> None:
        self.offsets = [0]
        self.parts: list[np.ndarray] = []

    def add(self, value: np.ndarray) -> None:
        array = np.asarray(value)
        self.parts.append(array)
        self.offsets.append(self.offsets[-1] + len(array))

    def array(
        self, shape_tail: tuple[int, ...], dtype: np.dtype[Any] | str
    ) -> np.ndarray:
        if self.parts and self.offsets[-1]:
            return np.ascontiguousarray(np.concatenate(self.parts).astype(dtype))
        return np.empty((0, *shape_tail), dtype=dtype)


def _cluster_worker(task: dict[str, Any]) -> dict[str, Any]:
    guarded._initialize_worker()
    root = repo_root()
    output_root = Path(task["output_root"])
    cluster_id = task["cluster_id"]
    staging = (
        output_root / "staging" / f"{cluster_id}.{os.getpid()}.tmp"
    )
    final = output_root / "clusters" / cluster_id
    staging.mkdir(parents=True, exist_ok=False)
    protocol_path = root / task["protocol_path"]
    if sha256_file(protocol_path) != task["protocol_sha256"]:
        raise InvalidLocalization("WORKER_PROTOCOL_DRIFT")
    protocol = load_json(protocol_path)
    affine_parameters = protocol["local_affine"]
    base_scene = task["base_scene"]
    poses = task["poses"]
    rows: list[dict[str, Any]] = []
    initial_ragged = _Ragged()
    initial_source_ragged = _Ragged()
    initial_source_index_ragged = _Ragged()
    initial_valid_ragged = _Ragged()
    initial_source_depth_ragged = _Ragged()
    initial_source_object_ragged = _Ragged()
    initial_source_world_ragged = _Ragged()
    initial_source_current_pixel_ragged = _Ragged()
    lk_forward_ragged = _Ragged()
    lk_forward_source_index_ragged = _Ragged()
    lk_forward_valid_ragged = _Ragged()
    lk_backward_ragged = _Ragged()
    lk_backward_source_index_ragged = _Ragged()
    lk_backward_level_ragged = _Ragged()
    lk_backward_error_ragged = _Ragged()
    lk_backward_available_ragged = _Ragged()
    lk_fb_ragged = _Ragged()
    lk_mask_ragged = _Ragged()
    lk_accepted_ragged = _Ragged()
    lk_rejection_ragged = _Ragged()
    accepted_previous_ragged = _Ragged()
    accepted_current_ragged = _Ragged()
    accepted_fb_ragged = _Ragged()
    accepted_source_ragged = _Ragged()
    accepted_source_index_ragged = _Ragged()
    accepted_valid_ragged = _Ragged()
    managed_previous_ragged = _Ragged()
    managed_current_ragged = _Ragged()
    managed_fb_ragged = _Ragged()
    managed_source_ragged = _Ragged()
    managed_source_index_ragged = _Ragged()
    managed_valid_ragged = _Ragged()
    managed_path_ragged = _Ragged()
    fit_previous_ragged = _Ragged()
    fit_current_ragged = _Ragged()
    fit_fb_ragged = _Ragged()
    fit_group_path: list[str] = []
    activated_cell_ragged = _Ragged()
    merge_candidate_previous_ragged = _Ragged()
    merge_candidate_current_ragged = _Ragged()
    merge_candidate_fb_ragged = _Ragged()
    merge_candidate_path_ragged = _Ragged()
    merge_candidate_selected_ragged = _Ragged()
    merge_group_path: list[str] = []
    consensus_ragged = _Ragged()
    consensus_pair_cell: list[np.ndarray] = []
    consensus_path: list[str] = []
    previous: tuple[
        np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]
    ] | None = None
    state = r3.PairState()
    streak = 0
    minimum_memory = psutil.virtual_memory().available
    started = time.perf_counter()
    for frame_index, pose in enumerate(poses):
        available = int(psutil.virtual_memory().available)
        minimum_memory = min(minimum_memory, available)
        if resource_action(available, 1, False, 0) != "ALLOW":
            raise InvalidLocalization("WORKER_RAM_BELOW_4_GIB")
        scene = _dynamic_static_scene(base_scene, frame_index)
        rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(pose["translation_m"], dtype=np.float64)
        rendered = qms.render_pair(scene, rotation, translation)
        rgb = rendered["rgb_pair"]["clean"]
        mask = rendered["valid_mask"]
        if previous is not None:
            previous_rgb, previous_mask, previous_pose, previous_scene = previous
            previous_gray = transport.rgb_to_gray(previous_rgb)
            current_gray = transport.rgb_to_gray(rgb)
            previous_u8 = transport.valid_mask(
                previous_mask, previous_gray.shape
            )
            current_u8 = transport.valid_mask(mask, current_gray.shape)
            dt = float(pose["timestamp_s"]) - float(
                previous_pose["timestamp_s"]
            )
            capture = PairCapture(
                previous_gray, current_gray, previous_u8, current_u8, dt
            )
            pair_state_before = _pair_state_record(state)
            with capture.patches():
                r3_row = transport.evaluate_pair(
                    pair_index=frame_index - 1,
                    previous_rgb=previous_rgb,
                    current_rgb=rgb,
                    previous_valid=previous_mask,
                    current_valid=mask,
                    previous_timestamp_s=previous_pose["timestamp_s"],
                    current_timestamp_s=pose["timestamp_s"],
                    previous_world_from_camera=np.asarray(
                        previous_pose["rotation_matrix"], dtype=np.float64
                    ),
                    current_world_from_camera=rotation,
                    intrinsic=geometry.K,
                    protocol=protocol,
                    state=state,
                )
            pair_state_after = _pair_state_record(state)
            initial = (
                capture.initial_points
                if capture.initial_points is not None
                else np.empty((0, 2), dtype=np.float32)
            )
            if capture.compensation is None:
                raise InvalidLocalization("PAIR_WARP_CAPTURE")
            homography = capture.compensation["homography"]
            previous_rotation = np.asarray(
                previous_pose["rotation_matrix"], dtype=np.float64
            )
            previous_translation = np.asarray(
                previous_pose["translation_m"], dtype=np.float64
            )
            source_initial = _source_correspondence(
                initial,
                previous_scene,
                scene,
                previous_rotation,
                rotation,
                previous_translation,
                translation,
                homography,
            )
            roundtrip_points = np.vstack(
                (
                    initial.astype(np.float64),
                    np.asarray(
                        [
                            [0.0, 0.0],
                            [geometry.WIDTH - 1.0, 0.0],
                            [0.0, geometry.HEIGHT - 1.0],
                            [
                                geometry.WIDTH - 1.0,
                                geometry.HEIGHT - 1.0,
                            ],
                        ]
                    ),
                )
            )
            roundtrip = _homography_points(
                _homography_points(roundtrip_points, homography),
                np.linalg.inv(homography),
            )
            roundtrip_error = np.linalg.norm(
                roundtrip - roundtrip_points, axis=1
            )
            geometry_cells = _ols_error_cells(
                initial,
                source_initial["aligned_current"],
                source_initial["valid"],
                dt,
                affine_parameters,
            )
            geometry_pair = _pair_from_cells(geometry_cells)
            geometry_pair.update(
                {
                    "roundtrip_max_px": (
                        float(np.max(roundtrip_error))
                        if roundtrip_error.size
                        and _finite(roundtrip_error)
                        else None
                    )
                }
            )
            warp_pair, interior, boundary = _warp_audit(capture, initial)

            if len(capture.track_calls) in (0, 2):
                accepted = (
                    capture.fit_calls[1]["tracks"]
                    if len(capture.fit_calls) in (2, 4)
                    else SparseTrackResult(
                        np.empty((0, 2), dtype=np.float32),
                        np.empty((0, 2), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                        0,
                    )
                )
                lk = (
                    capture.track_calls[1]
                    if len(capture.track_calls) == 2
                    else {
                        "forward_points": np.empty((0, 2), dtype=np.float32),
                        "forward_available": np.empty((0,), dtype=bool),
                        "best_backward_points": np.empty(
                            (0, 2), dtype=np.float32
                        ),
                        "best_backward_level": np.empty(
                            (0,), dtype=np.int16
                        ),
                        "best_forward_backward_error": np.empty(
                            (0,), dtype=np.float64
                        ),
                        "backward_available": np.empty((0,), dtype=bool),
                        "fb_pass": np.empty((0,), dtype=bool),
                        "mask_pass": np.empty((0,), dtype=bool),
                        "accepted": np.empty((0,), dtype=bool),
                        "rejection_reason": np.empty((0,), dtype="<U24"),
                    }
                )
            else:
                raise InvalidLocalization("TRACK_CALL_COUNT")
            managed, path_class = (
                _final_tracks(
                    capture,
                    r3_row.get("support_manager", {}).get(
                        "activated_cell_indices", []
                    ),
                )
                if len(capture.fit_calls) in (2, 4)
                else (
                    accepted,
                    np.full(accepted.valid_count, "BASELINE", dtype="<U16"),
                )
            )
            accepted_source = _source_correspondence(
                accepted.previous_points,
                previous_scene,
                scene,
                previous_rotation,
                rotation,
                previous_translation,
                translation,
                homography,
            )
            managed_source = _source_correspondence(
                managed.previous_points,
                previous_scene,
                scene,
                previous_rotation,
                rotation,
                previous_translation,
                translation,
                homography,
            )
            accepted_error_endpoints = (
                accepted.previous_points.astype(np.float64)
                + accepted.current_points.astype(np.float64)
                - accepted_source["aligned_current"]
            )
            managed_error_endpoints = (
                managed.previous_points.astype(np.float64)
                + managed.current_points.astype(np.float64)
                - managed_source["aligned_current"]
            )
            accepted_cells = _ols_error_cells(
                accepted.previous_points,
                accepted_error_endpoints,
                accepted_source["valid"],
                dt,
                affine_parameters,
            )
            managed_cells = _ols_error_cells(
                managed.previous_points,
                managed_error_endpoints,
                managed_source["valid"],
                dt,
                affine_parameters,
            )
            accepted_pair = _pair_from_cells(accepted_cells)
            managed_pair = _pair_from_cells(managed_cells)
            boundary_cells = _ols_error_cells(
                managed.previous_points,
                managed_error_endpoints,
                managed_source["valid"],
                dt,
                affine_parameters,
                boundary,
            )
            interior_cells = _ols_error_cells(
                managed.previous_points,
                managed_error_endpoints,
                managed_source["valid"],
                dt,
                affine_parameters,
                interior,
            )
            boundary_pair = _pair_from_cells(boundary_cells)
            interior_pair = _pair_from_cells(interior_cells)
            boundary_summary = {
                "evaluable": (
                    boundary_pair["evaluable"]
                    and interior_pair["evaluable"]
                ),
                "boundary_evaluable": boundary_pair["evaluable"],
                "interior_evaluable": interior_pair["evaluable"],
                "boundary_absolute_per_s": boundary_pair["absolute_per_s"],
                "boundary_signed_per_s": boundary_pair["signed_per_s"],
                "interior_absolute_per_s": interior_pair["absolute_per_s"],
                "interior_signed_per_s": interior_pair["signed_per_s"],
                "boundary_evaluable_cell_count": boundary_pair[
                    "evaluable_cell_count"
                ],
                "interior_evaluable_cell_count": interior_pair[
                    "evaluable_cell_count"
                ],
                "initial_boundary_count": int(
                    np.count_nonzero(_sample_mask(boundary * 255, initial))
                ),
                "initial_interior_count": int(
                    np.count_nonzero(_sample_mask(interior * 255, initial))
                ),
                "accepted_boundary_count": int(
                    np.count_nonzero(
                        _sample_mask(boundary * 255, accepted.previous_points)
                    )
                ),
                "accepted_interior_count": int(
                    np.count_nonzero(
                        _sample_mask(interior * 255, accepted.previous_points)
                    )
                ),
                "managed_boundary_count": int(
                    np.count_nonzero(
                        _sample_mask(boundary * 255, managed.previous_points)
                    )
                ),
                "managed_interior_count": int(
                    np.count_nonzero(
                        _sample_mask(interior * 255, managed.previous_points)
                    )
                ),
            }
            local_pair, cell_summaries, consensus_arrays = _local_cells(
                capture, r3_row
            )
            if r3_row.get("evaluable") is True:
                response = float(
                    r3_row["compensated_expansion_median_per_s"]
                )
                streak = streak + 1 if response > THRESHOLD_PER_S else 0
            else:
                streak = 0
            pair_index = frame_index - 1
            row = {
                "schema": "rcle.r3_rotation_leakage_localization.pair.v1",
                "cluster_id": cluster_id,
                "sequence_id": task["sequence_id"],
                "pair_index": pair_index,
                "previous_timestamp_s": float(
                    previous_pose["timestamp_s"]
                ),
                "current_timestamp_s": float(pose["timestamp_s"]),
                "dt_s": dt,
                "array_offsets": {
                    "initial": [
                        initial_ragged.offsets[-1],
                        initial_ragged.offsets[-1] + len(initial),
                    ],
                    "accepted": [
                        accepted_previous_ragged.offsets[-1],
                        accepted_previous_ragged.offsets[-1]
                        + accepted.valid_count,
                    ],
                    "managed": [
                        managed_previous_ragged.offsets[-1],
                        managed_previous_ragged.offsets[-1]
                        + managed.valid_count,
                    ],
                },
                "input_geometry": geometry_pair,
                "rotation_warp": warp_pair,
                "mask_boundary": boundary_summary,
                "sparse_lk_and_track_filtering": {
                    "accepted": accepted_pair,
                    "managed": managed_pair,
                    "requested_count": int(len(initial)),
                    "forward_valid_count": int(
                        np.count_nonzero(lk["forward_available"])
                    ),
                    "backward_available_count": int(
                        np.count_nonzero(lk["backward_available"])
                    ),
                    "fb_pass_count": int(np.count_nonzero(lk["fb_pass"])),
                    "mask_pass_count": int(
                        np.count_nonzero(lk["mask_pass"])
                    ),
                    "accepted_count": accepted.valid_count,
                    "managed_count": managed.valid_count,
                    "carried_count": int(
                        np.count_nonzero(path_class == "CARRIED")
                    ),
                    "supplement_count": int(
                        np.count_nonzero(path_class == "SUPPLEMENT")
                    ),
                },
                "local_affine_and_final_aggregation": {
                    **local_pair,
                    "raw_cells": cell_summaries["RAW_FINAL"],
                    "compensated_cells": cell_summaries[
                        "COMPENSATED_FINAL"
                    ],
                },
                "r3_pair_row": {
                    "evaluable": bool(r3_row.get("evaluable", False)),
                    "reason": r3_row.get("reason"),
                    "signed_per_s": r3_row.get(
                        "compensated_expansion_median_per_s"
                    ),
                    "absolute_per_s": r3_row.get(
                        "compensated_abs_expansion_median_per_s"
                    ),
                    "common_cell_count": r3_row.get("common_cell_count"),
                    "strict_trigger": bool(r3_row.get("trigger", False)),
                    "three_pair_trigger": streak >= 3,
                },
                "pair_state_before": pair_state_before,
                "pair_state_after": pair_state_after,
                "coverage": {
                    "source_geometry_valid": int(
                        np.count_nonzero(source_initial["valid"])
                    ),
                    "initial_features": int(len(initial)),
                    "accepted_tracks": accepted.valid_count,
                    "managed_tracks": managed.valid_count,
                    "common_cells": int(
                        local_pair.get("common_cell_count", 0)
                    ),
                },
            }
            rows.append(row)

            initial_ragged.add(initial)
            initial_source_index = np.flatnonzero(
                source_initial["valid"]
            ).astype(np.int32)
            initial_source_ragged.add(
                source_initial["aligned_current"][source_initial["valid"]]
            )
            initial_source_index_ragged.add(initial_source_index)
            initial_valid_ragged.add(source_initial["valid"])
            initial_source_depth_ragged.add(
                source_initial["depth_m"][source_initial["valid"]]
            )
            initial_source_object_ragged.add(
                source_initial["object_id"][source_initial["valid"]]
            )
            initial_source_world_ragged.add(
                source_initial["world"][source_initial["valid"]]
            )
            initial_source_current_pixel_ragged.add(
                source_initial["current_pixel"][source_initial["valid"]]
            )
            lk_forward_index = np.flatnonzero(
                lk["forward_available"]
                & np.isfinite(lk["forward_points"]).all(axis=1)
            ).astype(np.int32)
            lk_forward_ragged.add(lk["forward_points"][lk_forward_index])
            lk_forward_source_index_ragged.add(lk_forward_index)
            lk_forward_valid_ragged.add(lk["forward_available"])
            lk_backward_index = np.flatnonzero(
                lk["backward_available"]
                & np.isfinite(lk["best_backward_points"]).all(axis=1)
                & np.isfinite(lk["best_forward_backward_error"])
            ).astype(np.int32)
            lk_backward_ragged.add(
                lk["best_backward_points"][lk_backward_index]
            )
            lk_backward_source_index_ragged.add(lk_backward_index)
            lk_backward_level_ragged.add(
                lk["best_backward_level"][lk_backward_index]
            )
            lk_backward_error_ragged.add(
                lk["best_forward_backward_error"][lk_backward_index]
            )
            lk_backward_available_ragged.add(lk["backward_available"])
            lk_fb_ragged.add(lk["fb_pass"])
            lk_mask_ragged.add(lk["mask_pass"])
            lk_accepted_ragged.add(lk["accepted"])
            lk_rejection_ragged.add(lk["rejection_reason"])
            accepted_previous_ragged.add(accepted.previous_points)
            accepted_current_ragged.add(accepted.current_points)
            accepted_fb_ragged.add(accepted.forward_backward_errors)
            accepted_source_index = np.flatnonzero(
                accepted_source["valid"]
            ).astype(np.int32)
            accepted_source_ragged.add(
                accepted_source["aligned_current"][
                    accepted_source["valid"]
                ]
            )
            accepted_source_index_ragged.add(accepted_source_index)
            accepted_valid_ragged.add(accepted_source["valid"])
            managed_previous_ragged.add(managed.previous_points)
            managed_current_ragged.add(managed.current_points)
            managed_fb_ragged.add(managed.forward_backward_errors)
            managed_source_index = np.flatnonzero(
                managed_source["valid"]
            ).astype(np.int32)
            managed_source_ragged.add(
                managed_source["aligned_current"][
                    managed_source["valid"]
                ]
            )
            managed_source_index_ragged.add(managed_source_index)
            managed_valid_ragged.add(managed_source["valid"])
            managed_path_ragged.add(path_class)
            empty_tracks = SparseTrackResult(
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                0,
            )
            if len(capture.fit_calls) not in (0, 2, 4):
                raise InvalidLocalization("FIT_CALL_COUNT_FOR_PERSISTENCE")
            fit_sources = (
                capture.fit_calls[0]["tracks"]
                if len(capture.fit_calls) >= 2
                else empty_tracks,
                capture.fit_calls[1]["tracks"]
                if len(capture.fit_calls) >= 2
                else empty_tracks,
                capture.fit_calls[2]["tracks"]
                if len(capture.fit_calls) == 4
                else empty_tracks,
                capture.fit_calls[3]["tracks"]
                if len(capture.fit_calls) == 4
                else empty_tracks,
            )
            for fit_name, fit_tracks in zip(
                (
                    "RAW_INITIAL",
                    "COMPENSATED_INITIAL",
                    "RAW_MANAGED",
                    "COMPENSATED_MANAGED",
                ),
                fit_sources,
                strict=True,
            ):
                fit_previous_ragged.add(fit_tracks.previous_points)
                fit_current_ragged.add(fit_tracks.current_points)
                fit_fb_ragged.add(fit_tracks.forward_backward_errors)
                fit_group_path.append(fit_name)
            activated_cell_ragged.add(
                np.asarray(
                    sorted(
                        int(value)
                        for value in r3_row.get(
                            "support_manager", {}
                        ).get("activated_cell_indices", [])
                    ),
                    dtype=np.int8,
                )
            )
            if len(capture.merge_calls) not in (0, 2):
                raise InvalidLocalization("MERGE_CALL_COUNT_FOR_PERSISTENCE")
            for merge_index, merge_name in enumerate(
                ("RAW_MANAGED", "COMPENSATED_MANAGED")
            ):
                captured_merge = (
                    capture.merge_calls[merge_index]
                    if len(capture.merge_calls) == 2
                    else None
                )
                if captured_merge is None:
                    merge_candidate_previous_ragged.add(
                        np.empty((0, 2), dtype=np.float32)
                    )
                    merge_candidate_current_ragged.add(
                        np.empty((0, 2), dtype=np.float32)
                    )
                    merge_candidate_fb_ragged.add(
                        np.empty((0,), dtype=np.float32)
                    )
                    merge_candidate_path_ragged.add(
                        np.empty((0,), dtype="<U16")
                    )
                    merge_candidate_selected_ragged.add(
                        np.empty((0,), dtype=bool)
                    )
                else:
                    merge_candidate_previous_ragged.add(
                        captured_merge["candidate_previous"]
                    )
                    merge_candidate_current_ragged.add(
                        captured_merge["candidate_current"]
                    )
                    merge_candidate_fb_ragged.add(
                        captured_merge["candidate_fb_error"]
                    )
                    merge_candidate_path_ragged.add(
                        captured_merge["candidate_path_class"]
                    )
                    merge_candidate_selected_ragged.add(
                        captured_merge["selected_mask"]
                    )
                merge_group_path.append(merge_name)
            for path, cell_index, values in consensus_arrays:
                consensus_ragged.add(values)
                consensus_pair_cell.append(
                    np.asarray([pair_index, cell_index], dtype=np.int32)
                )
                consensus_path.append(path)
        previous = (rgb, mask, pose, scene)
    if len(rows) != task["pair_count"]:
        raise InvalidLocalization("WORKER_PAIR_COUNT")
    metrics = _cluster_metrics(rows, task["pair_count"])
    metrics.update(
        {
            "schema": "rcle.r3_rotation_leakage_localization.cluster.v1",
            "cluster_id": cluster_id,
            "sequence_id": task["sequence_id"],
            "block": task["block"],
            "ordinal": task["ordinal"],
            "mode": task["mode"],
            "claim_ceiling": (
                "CONTROLLED_GENERATOR_INTERNAL_MECHANISM_LOCALIZATION_ONLY"
            ),
        }
    )
    ledger_path = staging / "pair_ledger.jsonl"
    primitive_path = staging / "primitives.npz"
    metrics_path = staging / "cluster_metrics.json"
    write_exclusive_jsonl(ledger_path, rows)
    primitive_payload = {
        "initial_offsets": np.asarray(
            initial_ragged.offsets, dtype=np.int64
        ),
        "initial_previous": initial_ragged.array((2,), np.float32),
        "initial_source_offsets": np.asarray(
            initial_source_ragged.offsets, dtype=np.int64
        ),
        "initial_source_index": initial_source_index_ragged.array(
            (), np.int32
        ),
        "initial_source_aligned": initial_source_ragged.array(
            (2,), np.float64
        ),
        "initial_source_valid": initial_valid_ragged.array((), np.bool_),
        "initial_source_depth_m": initial_source_depth_ragged.array(
            (), np.float64
        ),
        "initial_source_object_id": initial_source_object_ragged.array(
            (), np.int32
        ),
        "initial_source_world": initial_source_world_ragged.array(
            (3,), np.float64
        ),
        "initial_source_current_pixel": (
            initial_source_current_pixel_ragged.array((2,), np.float64)
        ),
        "lk_forward_offsets": np.asarray(
            lk_forward_ragged.offsets, dtype=np.int64
        ),
        "lk_forward_source_index": (
            lk_forward_source_index_ragged.array((), np.int32)
        ),
        "lk_forward_points": lk_forward_ragged.array((2,), np.float32),
        "lk_forward_available": lk_forward_valid_ragged.array(
            (), np.bool_
        ),
        "lk_backward_offsets": np.asarray(
            lk_backward_ragged.offsets, dtype=np.int64
        ),
        "lk_backward_source_index": (
            lk_backward_source_index_ragged.array((), np.int32)
        ),
        "lk_backward_points": lk_backward_ragged.array((2,), np.float32),
        "lk_backward_level": lk_backward_level_ragged.array((), np.int16),
        "lk_backward_error": lk_backward_error_ragged.array((), np.float32),
        "lk_backward_available": lk_backward_available_ragged.array(
            (), np.bool_
        ),
        "lk_fb_pass": lk_fb_ragged.array((), np.bool_),
        "lk_mask_pass": lk_mask_ragged.array((), np.bool_),
        "lk_accepted": lk_accepted_ragged.array((), np.bool_),
        "lk_rejection_reason": lk_rejection_ragged.array((), "<U24"),
        "accepted_offsets": np.asarray(
            accepted_previous_ragged.offsets, dtype=np.int64
        ),
        "accepted_previous": accepted_previous_ragged.array(
            (2,), np.float32
        ),
        "accepted_current": accepted_current_ragged.array(
            (2,), np.float32
        ),
        "accepted_fb_error": accepted_fb_ragged.array((), np.float32),
        "accepted_source_offsets": np.asarray(
            accepted_source_ragged.offsets, dtype=np.int64
        ),
        "accepted_source_index": accepted_source_index_ragged.array(
            (), np.int32
        ),
        "accepted_source_aligned": accepted_source_ragged.array(
            (2,), np.float64
        ),
        "accepted_source_valid": accepted_valid_ragged.array((), np.bool_),
        "managed_offsets": np.asarray(
            managed_previous_ragged.offsets, dtype=np.int64
        ),
        "managed_previous": managed_previous_ragged.array(
            (2,), np.float32
        ),
        "managed_current": managed_current_ragged.array((2,), np.float32),
        "managed_fb_error": managed_fb_ragged.array((), np.float32),
        "managed_source_offsets": np.asarray(
            managed_source_ragged.offsets, dtype=np.int64
        ),
        "managed_source_index": managed_source_index_ragged.array(
            (), np.int32
        ),
        "managed_source_aligned": managed_source_ragged.array(
            (2,), np.float64
        ),
        "managed_source_valid": managed_valid_ragged.array((), np.bool_),
        "managed_path_class": managed_path_ragged.array((), "<U16"),
        "fit_offsets": np.asarray(
            fit_previous_ragged.offsets, dtype=np.int64
        ),
        "fit_previous": fit_previous_ragged.array((2,), np.float32),
        "fit_current": fit_current_ragged.array((2,), np.float32),
        "fit_fb_error": fit_fb_ragged.array((), np.float32),
        "fit_group_path": np.asarray(fit_group_path, dtype="<U24"),
        "activated_offsets": np.asarray(
            activated_cell_ragged.offsets, dtype=np.int64
        ),
        "activated_cell_index": activated_cell_ragged.array(
            (), np.int8
        ),
        "merge_candidate_offsets": np.asarray(
            merge_candidate_previous_ragged.offsets, dtype=np.int64
        ),
        "merge_candidate_previous": merge_candidate_previous_ragged.array(
            (2,), np.float32
        ),
        "merge_candidate_current": merge_candidate_current_ragged.array(
            (2,), np.float32
        ),
        "merge_candidate_fb_error": merge_candidate_fb_ragged.array(
            (), np.float32
        ),
        "merge_candidate_path_class": merge_candidate_path_ragged.array(
            (), "<U16"
        ),
        "merge_candidate_selected": (
            merge_candidate_selected_ragged.array((), np.bool_)
        ),
        "merge_group_path": np.asarray(merge_group_path, dtype="<U24"),
        "consensus_offsets": np.asarray(
            consensus_ragged.offsets, dtype=np.int64
        ),
        "consensus_previous": (
            consensus_ragged.array((4,), np.float32)[:, :2]
        ),
        "consensus_current": (
            consensus_ragged.array((4,), np.float32)[:, 2:]
        ),
        "consensus_pair_cell": np.asarray(
            consensus_pair_cell, dtype=np.int32
        ).reshape(-1, 2),
        "consensus_path": np.asarray(consensus_path, dtype="<U20"),
    }
    for name, array in primitive_payload.items():
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(
            array
        ).all():
            raise InvalidLocalization(f"NONFINITE_PRIMITIVE:{name}")
    np.savez_compressed(primitive_path, **primitive_payload)
    write_exclusive_json(metrics_path, metrics)
    receipt = {
        "schema": "rcle.r3_rotation_leakage_localization.cluster_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "runner_id": RUNNER_ID,
        "cluster_id": cluster_id,
        "sequence_id": task["sequence_id"],
        "identity_hashes": task["identity_hashes"],
        "pair_count": len(rows),
        "pair_ledger_sha256": sha256_file(ledger_path),
        "primitives_sha256": sha256_file(primitive_path),
        "cluster_metrics_sha256": sha256_file(metrics_path),
        "route_status": "PENDING_INDEPENDENT_VALIDATION",
        "wall_seconds": time.perf_counter() - started,
        "minimum_available_ram_bytes": minimum_memory,
        "r3_return_values_unchanged": True,
        "terminal": (
            "LOCALIZATION_CLUSTER_COMPLETE / "
            "INDEPENDENT_VALIDATION_REQUIRED"
        ),
    }
    receipt_path = staging / "receipt.json"
    write_exclusive_json(receipt_path, receipt)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise InvalidLocalization("FINAL_CLUSTER_EXISTS")
    os.replace(staging, final)
    return {
        "cluster_id": cluster_id,
        "sequence_id": task["sequence_id"],
        "route_status": "PENDING_INDEPENDENT_VALIDATION",
        "receipt_path": (
            Path("clusters") / cluster_id / "receipt.json"
        ).as_posix(),
        "receipt_sha256": sha256_file(final / "receipt.json"),
        "wall_seconds": receipt["wall_seconds"],
        "minimum_available_ram_bytes": minimum_memory,
    }


def _progress(
    output_root: Path, index: int, payload: dict[str, Any]
) -> None:
    required = {
        "phase",
        "completed_units",
        "total_units",
        "throughput",
        "eta_seconds",
        "status",
    }
    if not required.issubset(payload):
        raise InvalidLocalization("PROGRESS_SCHEMA")
    record = {
        "schema": "rcle.r3_rotation_leakage_localization.progress.v1",
        "task_id": TASK_ID,
        "progress_index": index,
        "last_progress_at": utc_now(),
        **payload,
    }
    write_exclusive_json(
        output_root / "progress" / f"{index:06d}.json", record
    )
    write_atomic_json(output_root / "progress.json", record)


def _terminate_pool(pool: ProcessPoolExecutor | None) -> list[int]:
    if pool is None:
        return []
    pids = sorted(int(pid) for pid in getattr(pool, "_processes", {}))
    for process in list(getattr(pool, "_processes", {}).values()):
        if process.is_alive():
            process.terminate()
    for process in list(getattr(pool, "_processes", {}).values()):
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
    pool.shutdown(wait=False, cancel_futures=True)
    return [pid for pid in pids if psutil.pid_exists(pid)]


def _write_failure_receipt(
    output_root: Path,
    *,
    mode: str,
    error: BaseException,
    results: Sequence[dict[str, Any]],
    residual_worker_pids: Sequence[int],
) -> None:
    """Close every consumed claim, including partial post-root setup."""
    claim_path = output_root / "claim.json"
    failure = {
        "schema": "rcle.r3_rotation_leakage_localization.failure.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "runner_id": RUNNER_ID,
        "mode": mode,
        "claim_sha256": (
            sha256_file(claim_path) if claim_path.is_file() else None
        ),
        "failed_utc": utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(limit=20),
        "completed_clusters": len(results),
        "completed_cluster_ids": sorted(
            item["cluster_id"] for item in results
        ),
        "residual_worker_pids": sorted(
            int(pid) for pid in residual_worker_pids
        ),
        "retry_authorized": False,
        "terminal": (
            "INVALID_INCOMPLETE_ONE_SHOT_CONSUMED / "
            "NO_RETRY_REPLACEMENT_RESEED_RESUME_OR_OUTPUT_DELETE"
        ),
    }
    write_exclusive_json(output_root / "failure.json", failure)


def _atomic_claim_root(
    output_root: Path, claim: dict[str, Any]
) -> None:
    """Publish the output root and its write-once claim in one rename."""

    staging_root = output_root.with_name(
        f".{output_root.name}.claim-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging_root.mkdir(exist_ok=False)
    published = False
    try:
        write_exclusive_json(staging_root / "claim.json", claim)
        os.rename(staging_root, output_root)
        published = True
    finally:
        if not published and staging_root.exists():
            (staging_root / "claim.json").unlink(missing_ok=True)
            staging_root.rmdir()


def _formal_claim_rehash_barrier(
    *,
    root: Path,
    output_root: Path,
    claim: Mapping[str, Any],
    implementation_ready_receipt_path: Path,
    host_preflight_receipt_path: Path,
) -> None:
    """Revalidate every mutable formal binding immediately before claiming."""

    authority = validate_authority(root)
    readiness = validate_implementation_ready(
        root, implementation_ready_receipt_path
    )
    host_preflight = validate_host_preflight_receipt(
        root,
        host_preflight_receipt_path,
        readiness,
        authority,
    )
    runner_path = Path(__file__).resolve()
    expected = {
        "runner_source_path": runner_path.relative_to(root).as_posix(),
        "runner_source_sha256": sha256_file(runner_path),
        "activation_sha256": authority["activation_sha256"],
        "contract_sha256": authority["contract_sha256"],
        "identity_lock_sha256": authority["identity_sha256"],
        "preflight_receipt_sha256": authority[
            "preflight_receipt_sha256"
        ],
        "protocol_sha256": authority["protocol_sha256"],
        "source_bindings": readiness["source_bindings"],
        "executable_spec_path": SPEC_RELATIVE,
        "executable_spec_sha256": readiness["spec_sha256"],
        "implementation_ready_receipt_path": (
            readiness["implementation_ready_receipt_path"]
            .relative_to(root)
            .as_posix()
        ),
        "implementation_ready_receipt_sha256": readiness[
            "implementation_ready_receipt_sha256"
        ],
        "host_preflight_receipt_path": (
            host_preflight["path"].relative_to(root).as_posix()
        ),
        "host_preflight_receipt_sha256": host_preflight["sha256"],
    }
    for key, expected_value in expected.items():
        if claim.get(key) != expected_value:
            raise InvalidLocalization(f"FINAL_REHASH_BARRIER:{key}")
    validate_formal_target_absent(root, output_root)
    if (
        resource_action(
            int(psutil.virtual_memory().available),
            active_workers=0,
            refill_requested=True,
            paging_streak=0,
        )
        != "ALLOW"
    ):
        raise InvalidLocalization("FINAL_REHASH_BARRIER:RESOURCE")


def _claim_owned_by(output_root: Path, claim_id: str) -> bool:
    """Return true only when the published claim belongs to this invocation."""

    claim_path = output_root / "claim.json"
    if not claim_path.is_file():
        return False
    try:
        published = load_json(claim_path)
    except (OSError, UnicodeError, json.JSONDecodeError, InvalidLocalization):
        return False
    return published.get("claim_id") == claim_id


def execute(
    *,
    mode: str,
    output_root: Path,
    pilot_manifest_path: Path | None = None,
    pilot_disjoint_receipt_path: Path | None = None,
    pilot_workers: int = 1,
    formal_workers: int | None = None,
    implementation_ready_receipt_path: Path | None = None,
    host_preflight_receipt_path: Path | None = None,
) -> dict[str, Any]:
    root = repo_root()
    authority = validate_authority(root)
    output_root = output_root.resolve()
    readiness: dict[str, Any] | None = None
    host_preflight: dict[str, Any] | None = None
    pilot_disjoint: dict[str, Any] | None = None
    pilot_manifest_sha256: str | None = None
    tasks: list[dict[str, Any]] = []
    if mode == FORMAL_MODE:
        if formal_workers != WORKERS:
            raise InvalidLocalization("FORMAL_WORKERS_MUST_EQUAL_4")
        if implementation_ready_receipt_path is None:
            raise InvalidLocalization("IMPLEMENTATION_READY_RECEIPT_REQUIRED")
        if host_preflight_receipt_path is None:
            raise InvalidLocalization("HOST_PREFLIGHT_RECEIPT_REQUIRED")
        readiness = validate_implementation_ready(
            root, implementation_ready_receipt_path
        )
        host_preflight = validate_host_preflight_receipt(
            root,
            host_preflight_receipt_path,
            readiness,
            authority,
        )
        validate_formal_target_absent(root, output_root)
        workers = WORKERS
        cluster_count = 8
    elif mode == PILOT_MODE:
        pilot_parent = (root / PILOT_PARENT_RELATIVE).resolve()
        if (
            output_root == (root / FORMAL_ROOT_RELATIVE).resolve()
            or not output_root.is_relative_to(pilot_parent)
            or output_root.exists()
        ):
            raise InvalidLocalization("PILOT_TARGET")
        if pilot_workers not in (1, WORKERS):
            raise InvalidLocalization("PILOT_WORKERS")
        if (
            pilot_manifest_path is None
            or pilot_disjoint_receipt_path is None
        ):
            raise InvalidLocalization(
                "PILOT_MANIFEST_AND_DISJOINT_RECEIPT_REQUIRED"
            )
        manifest_path = pilot_manifest_path.resolve()
        pilot_disjoint = validate_pilot_disjoint_receipt(
            authority, manifest_path, pilot_disjoint_receipt_path
        )
        manifest = load_json(manifest_path)
        tasks = pilot_tasks(authority, manifest)
        pilot_manifest_sha256 = sha256_file(manifest_path)
        workers = pilot_workers
        cluster_count = len(tasks)
    else:
        raise InvalidLocalization("MODE")
    available = int(psutil.virtual_memory().available)
    action = resource_action(available, 0, True, 0)
    if action != "ALLOW":
        raise InvalidLocalization(action)
    if output_root.parent.exists() is False:
        if mode == FORMAL_MODE:
            raise InvalidLocalization("FORMAL_PARENT_MISSING")
        output_root.parent.mkdir(parents=True, exist_ok=True)
    claim_id = uuid.uuid4().hex
    claim = {
        "schema": "rcle.r3_rotation_leakage_localization.claim.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "runner_id": RUNNER_ID,
        "claim_id": claim_id,
        "mode": mode,
        "claimed_utc": utc_now(),
        "runner_source_path": (
            Path(__file__).resolve().relative_to(root).as_posix()
        ),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "activation_sha256": authority["activation_sha256"],
        "contract_sha256": authority["contract_sha256"],
        "identity_lock_sha256": authority["identity_sha256"],
        "preflight_receipt_sha256": authority[
            "preflight_receipt_sha256"
        ],
        "protocol_sha256": authority["protocol_sha256"],
        "source_bindings": (
            readiness["source_bindings"]
            if readiness is not None
            else authority["source_bindings"]
        ),
        "executable_spec_path": (
            SPEC_RELATIVE if readiness is not None else None
        ),
        "executable_spec_sha256": (
            readiness["spec_sha256"] if readiness is not None else None
        ),
        "implementation_ready_receipt_path": (
            readiness["implementation_ready_receipt_path"]
            .relative_to(root)
            .as_posix()
            if readiness is not None
            else None
        ),
        "implementation_ready_receipt_sha256": (
            readiness["implementation_ready_receipt_sha256"]
            if readiness is not None
            else None
        ),
        "host_preflight_receipt_path": (
            host_preflight["path"].relative_to(root).as_posix()
            if host_preflight is not None
            else None
        ),
        "host_preflight_receipt_sha256": (
            host_preflight["sha256"]
            if host_preflight is not None
            else None
        ),
        "pilot_manifest_sha256": pilot_manifest_sha256,
        "pilot_disjoint_receipt_sha256": (
            pilot_disjoint["sha256"]
            if pilot_disjoint is not None
            else None
        ),
        "workers": workers,
        "cluster_count": cluster_count,
        "write_once": True,
        "formal_authority_consumed": mode == FORMAL_MODE,
        "future_repair_authorized": False,
        "terminal": "EXECUTION_CLAIMED / NO_RETRY",
    }
    if mode == FORMAL_MODE:
        assert implementation_ready_receipt_path is not None
        assert host_preflight_receipt_path is not None
        _formal_claim_rehash_barrier(
            root=root,
            output_root=output_root,
            claim=claim,
            implementation_ready_receipt_path=(
                implementation_ready_receipt_path
            ),
            host_preflight_receipt_path=host_preflight_receipt_path,
        )
    claim_consumed = False
    try:
        _atomic_claim_root(output_root, claim)
        claim_consumed = True
        (output_root / "staging").mkdir()
        (output_root / "clusters").mkdir()
        (output_root / "progress").mkdir()
        _progress(
            output_root,
            0,
            {
                "phase": "CLAIM",
                "status": "CLAIMED",
                "completed_units": 0,
                "total_units": cluster_count,
                "throughput": 0.0,
                "eta_seconds": None,
                "active_clusters": [],
                "remaining_clusters": cluster_count,
            },
        )
    except BaseException as error:
        if claim_consumed or _claim_owned_by(output_root, claim_id):
            _write_failure_receipt(
                output_root,
                mode=mode,
                error=error,
                results=[],
                residual_worker_pids=[],
            )
        raise
    started = time.perf_counter()
    started_swap = psutil.swap_memory()
    last_swap = started_swap
    minimum_available = available
    paging_streak = 0
    progress_index = 1
    results: list[dict[str, Any]] = []
    pool: ProcessPoolExecutor | None = None
    worker_pids: list[int] = []
    active: dict[Any, dict[str, Any]] = {}
    remaining: list[dict[str, Any]] = []
    try:
        if mode == FORMAL_MODE:
            authority = validate_formal_inputs(root, authority)
            tasks = formal_tasks(authority)
            if len(tasks) != cluster_count:
                raise InvalidLocalization("FORMAL_TASK_COUNT_AFTER_CLAIM")
        remaining = list(tasks)
        context = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=guarded._initialize_worker,
            mp_context=context,
        )
        while remaining or active:
            while remaining and len(active) < workers:
                current_available = int(psutil.virtual_memory().available)
                action = resource_action(
                    current_available, len(active), True, paging_streak
                )
                if action != "ALLOW":
                    raise InvalidLocalization(action)
                task = remaining.pop(0)
                task = {
                    **task,
                    "output_root": str(output_root),
                    "protocol_path": authority["contract"][
                        "frozen_bindings"
                    ]["r3_parameters"]["path"],
                }
                future = pool.submit(_cluster_worker, task)
                active[future] = task
                worker_pids = sorted(
                    int(pid) for pid in getattr(pool, "_processes", {})
                )
            done, _ = wait(
                active, timeout=5.0, return_when=FIRST_COMPLETED
            )
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            minimum_available = min(minimum_available, int(memory.available))
            paging_delta = max(
                0,
                int(swap.sin - last_swap.sin)
                + int(swap.sout - last_swap.sout),
            )
            paging_streak = paging_streak + 1 if paging_delta else 0
            last_swap = swap
            action = resource_action(
                int(memory.available), len(active), False, paging_streak
            )
            if action != "ALLOW":
                raise InvalidLocalization(action)
            for future in done:
                active.pop(future)
                results.append(future.result())
            elapsed = max(time.perf_counter() - started, 1e-9)
            completed_units = len(results)
            throughput = completed_units / elapsed
            eta = (
                (cluster_count - completed_units) / throughput
                if throughput > 0.0
                else None
            )
            _progress(
                output_root,
                progress_index,
                {
                    "phase": "EXECUTION",
                    "status": "RUNNING" if active or remaining else "COMPLETE",
                    "completed_units": completed_units,
                    "total_units": cluster_count,
                    "throughput": throughput,
                    "eta_seconds": eta,
                    "active_clusters": sorted(
                        task["cluster_id"] for task in active.values()
                    ),
                    "remaining_clusters": len(remaining),
                    "available_ram_bytes": int(memory.available),
                    "swap_in_total": int(swap.sin),
                    "swap_out_total": int(swap.sout),
                    "paging_streak": paging_streak,
                },
            )
            progress_index += 1
        pool.shutdown(wait=True)
        pool = None
        residual = [pid for pid in worker_pids if psutil.pid_exists(pid)]
        if residual:
            raise InvalidLocalization("RESIDUAL_WORKERS")
        results.sort(key=lambda item: item["cluster_id"])
        receipt = {
            "schema": "rcle.r3_rotation_leakage_localization.run.v1",
            "protocol_id": PROTOCOL_ID,
            "task_id": TASK_ID,
            "runner_id": RUNNER_ID,
            "mode": mode,
            "claim_sha256": sha256_file(output_root / "claim.json"),
            "cluster_count": len(results),
            "clusters": results,
            "route_status": "PENDING_INDEPENDENT_VALIDATION",
            "identity_lock_sha256": authority["identity_sha256"],
            "trajectory_manifest_sha256": authority.get(
                "trajectory_manifest_sha256"
            ),
            "executable_spec_sha256": (
                readiness["spec_sha256"] if readiness is not None else None
            ),
            "implementation_ready_receipt_sha256": (
                readiness["implementation_ready_receipt_sha256"]
                if readiness is not None
                else None
            ),
            "host_preflight_receipt_sha256": (
                host_preflight["sha256"]
                if host_preflight is not None
                else None
            ),
            "pilot_manifest_sha256": pilot_manifest_sha256,
            "pilot_disjoint_receipt_sha256": (
                pilot_disjoint["sha256"]
                if pilot_disjoint is not None
                else None
            ),
            "resource": {
                "workers": workers,
                "available_ram_at_launch_bytes": available,
                "minimum_available_ram_bytes": minimum_available,
                "launch_refill_gate_bytes": LAUNCH_REFILL_BYTES,
                "in_flight_floor_bytes": IN_FLIGHT_FLOOR_BYTES,
                "swap_in_delta": int(last_swap.sin - started_swap.sin),
                "swap_out_delta": int(last_swap.sout - started_swap.sout),
                "worker_pids": worker_pids,
                "residual_worker_pids": residual,
                "wall_seconds": time.perf_counter() - started,
            },
            "r3_modified": False,
            "formal_480_plus_16_sequences_run": 0,
            "formal_authority_consumed": mode == FORMAL_MODE,
            "first_visible_layer_is_not_causal_identification": True,
            "future_repair_authorized": False,
            "terminal": (
                "LOCALIZATION_EXECUTION_COMPLETE / "
                "INDEPENDENT_VALIDATION_REQUIRED"
            ),
        }
        write_exclusive_json(output_root / "run_receipt.json", receipt)
        success = {
            **receipt,
            "run_receipt_sha256": sha256_file(
                output_root / "run_receipt.json"
            ),
            "completed_utc": utc_now(),
        }
        write_exclusive_json(output_root / "success.json", success)
        return success
    except BaseException as error:
        residual = _terminate_pool(pool)
        _write_failure_receipt(
            output_root,
            mode=mode,
            error=error,
            results=results,
            residual_worker_pids=residual,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R3 rotation-leakage source-localization runner"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--pilot-manifest", type=Path, required=True)
    pilot.add_argument(
        "--pilot-disjoint-receipt", type=Path, required=True
    )
    pilot.add_argument("--output-root", type=Path, required=True)
    pilot.add_argument("--workers", type=int, choices=(1, WORKERS), default=1)
    formal = sub.add_parser("execute-formal")
    formal.add_argument("--output-root", type=Path)
    formal.add_argument("--workers", type=int, required=True)
    formal.add_argument(
        "--implementation-ready-receipt", type=Path, required=True
    )
    formal.add_argument(
        "--host-preflight-receipt", type=Path, required=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if args.command == "pilot":
        result = execute(
            mode=PILOT_MODE,
            output_root=args.output_root,
            pilot_manifest_path=args.pilot_manifest,
            pilot_disjoint_receipt_path=args.pilot_disjoint_receipt,
            pilot_workers=args.workers,
        )
    else:
        output_root = (
            args.output_root
            if args.output_root is not None
            else root / FORMAL_ROOT_RELATIVE
        )
        result = execute(
            mode=FORMAL_MODE,
            output_root=output_root,
            formal_workers=args.workers,
            implementation_ready_receipt_path=(
                args.implementation_ready_receipt
            ),
            host_preflight_receipt_path=args.host_preflight_receipt,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
