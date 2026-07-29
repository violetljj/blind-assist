"""Two-stage clean motion-component localization for RCLE residual expansion.

All 32 identities are frozen before Stage 1.  Stage 1 executes ordinal zero
only.  Ordinal one remains sealed unless an independent Stage 1 routing
decision opens Stage 2.  R3 is unchanged; a read-only hook retains the nine
local-cell records already computed inside the frozen pair core.
"""

from __future__ import annotations

import os

# Install the measured numeric-thread mode before NumPy/OpenCV imports.
from . import p3_runtime_preflight_r0 as guarded

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import time
from typing import Any
from unittest import mock

import numpy as np
import psutil

from ..rcle_minimal import evaluation as r0_evaluation
from ..rgb_algorithm_development_canary_cid_sims_r0 import producer as r3
from . import generator_geometry as geometry
from . import material_residual_contraction_r1 as qms
from . import p3_transport_r0 as transport


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = (
    "STATIC",
    "ROTATION_ONLY",
    "TRANSLATION_ONLY",
    "FULL_6DOF",
)
FRAME_COUNT = 602
PAIR_COUNT = 601
WORKERS = 4
NATIVE_THREADS = 18
GIB = 1024**3
THRESHOLD = 0.01
EXCLUSION_SOURCES = (
    (
        "OLD_FORMAL",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "P4_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
    (
        "QMS_R0_PREDECESSOR_DEV",
        "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r0/predecessor_dev/identity_manifest.json",
    ),
    (
        "QMS_R1_PREDECESSOR_DEV",
        "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1/predecessor_dev/identity_manifest.json",
    ),
    (
        "QMS_R1_CAL",
        "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1/new_cal/identity_manifest.json",
    ),
    (
        "OLD_PREFLIGHT",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "PREFLIGHT_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
    (
        "QMS_R1_SUCCESSOR_FORMAL",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "QMS_R1_SUCCESSOR_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
    (
        "QMS_R1_ACTIVATION_PREFLIGHT",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "QMS_R1_FORMAL_ACTIVATION_PREFLIGHT_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
    (
        "QMS_R1_FOUR_BLOCK_DEV",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "QMS_R1_FOUR_BLOCK_DEV_DIAGNOSTIC_R0_IDENTITY_LOCK_2026-07-29.json",
    ),
)
PREFLIGHT_RECEIPT_DIRS = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r0/w4/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r0/w8/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r1/w4/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r2/w8/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r3/w8/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_formal_activation_preflight_r0/w8/sequences",
)
TRAJECTORY_MANIFEST = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
)
GENERATOR_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/generator_geometry.py"
)
GENERATOR_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R2_KEYSET_REPAIR_R0_2026-07-29.json"
)
OPERATOR_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/material_residual_contraction_r1.py"
)
OPERATOR_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_OPERATOR_LOCK_R0_2026-07-29.json"
)
TRANSPORT_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/p3_transport_r0.py"
)
TRANSPORT_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "R3_TRANSPORT_EQUIVALENCE_LOCK_R0_2026-07-29.json"
)
ANALYSIS_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "ANALYSIS_IMPLEMENTATION_LOCK_R0_2026-07-29.json"
)
FORMAL_DECISION = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_SUCCESSOR_FORMAL_ACTIVATION_DECISION_R0_2026-07-29.json"
)
SUCCESSOR_FORMAL_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_SUCCESSOR_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json"
)
PREVIOUS_DEV_BUNDLE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_four_block_dev_diagnostic_r0"
)


class InvalidMotionComponent(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    rows = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(os.fspath(path), flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(
        b"\n".join(canonical_bytes(row) for row in rows) + b"\n"
    )


def _collect(value: Any, key: str) -> set[Any]:
    found: set[Any] = set()
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key and isinstance(child, (str, int)):
                found.add(child)
            found.update(_collect(child, key))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect(child, key))
    return found


def _seed(block: str, ordinal: int) -> dict[str, Any]:
    if block not in BLOCKS or ordinal not in (0, 1):
        raise InvalidMotionComponent("SEED_COORDINATE")
    token = f"{TASK_ID}|SCENE|{block}|{ordinal:02d}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return {
        "token": token,
        "token_sha256": digest,
        "numeric_seed_uint64": int.from_bytes(bytes.fromhex(digest)[:8], "big"),
    }


def _scene(block: str, ordinal: int) -> dict[str, Any]:
    seed = _seed(block, ordinal)
    with mock.patch.object(
        geometry, "derive_seed", return_value=seed["numeric_seed_uint64"]
    ):
        return geometry.build_scene(block, ordinal, "MOTION_COMPONENT")


def motion_poses(
    trajectory: dict[str, Any], motion: str
) -> list[dict[str, Any]]:
    if motion not in MOTIONS:
        raise InvalidMotionComponent("MOTION")
    identity = np.eye(3, dtype=np.float64).tolist()
    zero = [0.0, 0.0, 0.0]
    poses = []
    for source in trajectory["poses"]:
        poses.append(
            {
                "frame_index": source["frame_index"],
                "timestamp_s": source["timestamp_s"],
                "rotation_matrix": (
                    source["rotation_matrix"]
                    if motion in {"ROTATION_ONLY", "FULL_6DOF"}
                    else identity
                ),
                "translation_m": (
                    source["translation_m"]
                    if motion in {"TRANSLATION_ONLY", "FULL_6DOF"}
                    else zero
                ),
            }
        )
    return poses


def trajectory_sha256(trajectory: dict[str, Any], motion: str) -> str:
    return hashlib.sha256(
        geometry.canonical_bytes(motion_poses(trajectory, motion))
    ).hexdigest()


def _bindings(root: Path, contract_path: Path, identity_path: Path) -> dict[str, str]:
    return {
        "contract_sha256": sha256_file(contract_path),
        "identity_lock_sha256": sha256_file(identity_path),
        "qms_r1_operator_source_sha256": sha256_file(root / OPERATOR_SOURCE),
        "qms_r1_operator_lock_sha256": sha256_file(root / OPERATOR_LOCK),
        "r3_transport_source_sha256": sha256_file(root / TRANSPORT_SOURCE),
        "r3_transport_lock_sha256": sha256_file(root / TRANSPORT_LOCK),
        "analysis_lock_sha256": sha256_file(root / ANALYSIS_LOCK),
        "generator_source_sha256": sha256_file(root / GENERATOR_SOURCE),
        "generator_lock_sha256": sha256_file(root / GENERATOR_LOCK),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
    }


def build_contract(root: Path) -> dict[str, Any]:
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.motion_component_contract.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "stage": "DEVELOPMENT_ROUTING_AUDIT",
        "research_question": (
            "Which frozen ego-motion components stably alter unchanged R3 "
            "residual local expansion under clean QMS-R1 rendering?"
        ),
        "signal_semantics": (
            "rotation-compensated residual local expansion; not danger, "
            "object identity, or alarm truth"
        ),
        "design": {
            "blocks": list(BLOCKS),
            "scene_seeds_per_block": 2,
            "identities_frozen_before_stage_1": 32,
            "stage_1_ordinal": 0,
            "stage_1_clusters": 4,
            "stage_1_sequences": 16,
            "stage_2_ordinal": 1,
            "stage_2_clusters": 4,
            "stage_2_sequences": 16,
            "stage_2_initial_state": "SEALED_NOT_EXECUTABLE",
            "motion_arms": list(MOTIONS),
            "quality": "CLEAN_ONLY",
            "frame_count_per_sequence": FRAME_COUNT,
            "pair_count_per_sequence": PAIR_COUNT,
            "observational_unit": "block_x_scene_seed_cluster",
            "paired_repeated_measure": "four motion arms within cluster",
            "pair_rows_are_independent_samples": False,
        },
        "motion_definitions": {
            "STATIC": "R=I,T=0",
            "ROTATION_ONLY": "R=periodic,T=0",
            "TRANSLATION_ONLY": "R=I,T=periodic",
            "FULL_6DOF": "R=periodic,T=periodic",
        },
        "primary_metrics": {
            "compensated_signed_p50_p90": "type7 over evaluable pair medians",
            "compensated_absolute_p50_p90": (
                "type7 over pair-level median_g(abs(cell expansion)); "
                "not abs(median_g(cell expansion))"
            ),
            "threshold_positive_fraction_evaluable": "P(e>0.01|evaluable)",
            "threshold_positive_density_fixed": "count(e>0.01)/601",
            "trigger_density_fixed": (
                "three-consecutive strict e>0.01 with abstention reset / 601"
            ),
            "longest_positive_streak": (
                "maximum consecutive evaluable strict e>0.01 pairs"
            ),
            "evaluable_fraction": "evaluable pairs / 601",
        },
        "secondary_metrics": {
            "raw_signed_and_absolute_p50_p90": True,
            "center_and_periphery_cell_absolute_expansion": True,
            "same_sign_spatial_fraction": True,
            "cell_fit_residual_p50_p90": True,
            "track_and_common_cell_support": True,
            "response_speed_spearman": True,
            "response_angular_translation_dominant_frequency": True,
            "secondary_metrics_drive_routing": False,
        },
        "observation_only_instrumentation": {
            "purpose": (
                "retain the nine raw and compensated LocalExpansionResult "
                "records already computed by unchanged R3"
            ),
            "hooked_function": (
                "rcle_minimal.evaluation._common_cell_expansions"
            ),
            "return_value_passthrough": "exact original tuple",
            "r3_source_modified": False,
            "required_equivalence": (
                "all original pair scalar fields equal values independently "
                "recomputed from retained common cells"
            ),
        },
        "threshold": {
            "operator": "strict_greater_than",
            "per_s": THRESHOLD,
            "consecutive_pairs": 3,
            "abstention_resets_streak": True,
            "not_retuned_from_stage_outputs": True,
        },
        "routing": {
            "paired_contrasts": {
                "ROTATION_MINUS_STATIC": {
                    "metric": "compensated_absolute_p90",
                    "formula": "ROTATION_ONLY-STATIC",
                },
                "TRANSLATION_MINUS_ROTATION": (
                    {
                        "metric": "compensated_signed_p90",
                        "formula": "TRANSLATION_ONLY-ROTATION_ONLY",
                    }
                ),
                "FULL_MINUS_MAX_SINGLE": {
                    "metric": "compensated_signed_p90",
                    "formula": (
                        "FULL_6DOF-max(ROTATION_ONLY,TRANSLATION_ONLY)"
                    ),
                },
            },
            "consistent_sign_rule": (
                "strict positive delta in >=3/4 exact blocks"
            ),
            "open_stage_2": (
                "at least one frozen contrast is strictly positive in >=3/4; "
                "missing block is NOT_EVALUABLE and cannot open"
            ),
            "stage_2_decision_maker": "independent_validator_only",
            "no_p_values": True,
            "no_confirmatory_inference": True,
        },
        "reporting": {
            "cluster_level_paired_effects": True,
            "direction_counts": True,
            "heterogeneity_ranges": True,
            "bootstrap_or_confidence_interval": False,
            "pair_pooled_inference": False,
            "formal_classification": False,
        },
        "scope_exclusions": [
            "object-approach positive control",
            "translation-depth oracle subtraction",
            "feature-contract freeze",
            "fusion model",
            "danger or false-alarm label",
            "sequence16",
            "Android, product, or safety claim",
        ],
        "bindings": {
            "qms_r1_operator_source_sha256": sha256_file(
                root / OPERATOR_SOURCE
            ),
            "qms_r1_operator_lock_sha256": sha256_file(root / OPERATOR_LOCK),
            "r3_transport_source_sha256": sha256_file(root / TRANSPORT_SOURCE),
            "r3_transport_lock_sha256": sha256_file(root / TRANSPORT_LOCK),
            "analysis_lock_sha256": sha256_file(root / ANALYSIS_LOCK),
            "trajectory_manifest_sha256": sha256_file(
                root / TRAJECTORY_MANIFEST
            ),
            "generator_source_sha256": sha256_file(root / GENERATOR_SOURCE),
            "generator_lock_sha256": sha256_file(root / GENERATOR_LOCK),
            "formal_activation_decision_sha256": sha256_file(
                root / FORMAL_DECISION
            ),
            "successor_formal_identity_lock_sha256": sha256_file(
                root / SUCCESSOR_FORMAL_LOCK
            ),
        },
        "r3_algorithm_changed": False,
        "formal_identity_execution": False,
        "formal_authority_consumed": False,
        "terminal": "MOTION_COMPONENT_CONTRACT_FROZEN / NOT_RUN",
    }


def build_identity_lock(root: Path, contract_path: Path) -> dict[str, Any]:
    fields = (
        "numeric_seed_uint64",
        "token",
        "token_sha256",
        "cluster_id",
        "sequence_id",
        "scene_geometry_sha256",
    )
    excluded = {field: set() for field in fields}
    sources = []
    for label, relative in EXCLUSION_SOURCES:
        path = root / relative
        value = load_json(path)
        counts = {}
        for field in fields:
            values = _collect(value, field)
            if field == "token_sha256":
                values.update(_collect(value, "cluster_token_sha256"))
            excluded[field].update(values)
            counts[field] = len(values)
        sources.append(
            {
                "label": label,
                "path": relative,
                "sha256": sha256_file(path),
                "unique_counts": counts,
            }
        )
    runtime_scenes = set()
    for relative in PREFLIGHT_RECEIPT_DIRS:
        directory = root / relative
        receipts = list(directory.glob("*/receipt.json"))
        if not receipts:
            raise InvalidMotionComponent("PREFLIGHT_RECEIPTS_MISSING")
        directory_scenes = set()
        for receipt in receipts:
            directory_scenes.update(
                _collect(load_json(receipt), "scene_geometry_sha256")
            )
        runtime_scenes.update(directory_scenes)
        sources.append(
            {
                "label": "PREFLIGHT_RUNTIME_SCENE_RECEIPTS",
                "path": relative,
                "tree_sha256": tree_sha256(directory),
                "receipt_count": len(receipts),
                "scene_geometry_sha256": sorted(directory_scenes),
            }
        )
    if len(runtime_scenes) != 4:
        raise InvalidMotionComponent("PREFLIGHT_SCENE_SET")
    excluded["scene_geometry_sha256"].update(runtime_scenes)
    trajectories = load_json(root / TRAJECTORY_MANIFEST)
    seeds = []
    identities = []
    for block in BLOCKS:
        trajectory = trajectories[block]
        for ordinal in (0, 1):
            stage_number = ordinal + 1
            stage_state = (
                "STAGE_1_EXECUTABLE"
                if ordinal == 0
                else "STAGE_2_SEALED_NOT_EXECUTABLE"
            )
            seed = _seed(block, ordinal)
            scene = _scene(block, ordinal)
            cluster_id = f"MCL_R0_{block}_S{stage_number}"
            latent = {
                **seed,
                "cluster_id": cluster_id,
                "scene_geometry_sha256": scene["scene_geometry_sha256"],
            }
            for field, old_values in excluded.items():
                if latent.get(field) in old_values:
                    raise InvalidMotionComponent(f"IDENTITY_OVERLAP:{field}")
            seeds.append(
                {
                    "block": block,
                    "ordinal": ordinal,
                    "stage_number": stage_number,
                    "stage_state": stage_state,
                    **latent,
                }
            )
            for arm_ordinal, motion in enumerate(MOTIONS):
                identity = {
                    "sequence_id": f"{cluster_id}__{motion}__CLEAN",
                    "cluster_id": cluster_id,
                    "block": block,
                    "ordinal": ordinal,
                    "stage_number": stage_number,
                    "stage_state": stage_state,
                    "role": "MOTION_COMPONENT_ROUTING_DEV",
                    "arm": f"{motion}__CLEAN",
                    "arm_ordinal": arm_ordinal,
                    "motion": motion,
                    "quality": "CLEAN",
                    **seed,
                    "scene_geometry_sha256": scene[
                        "scene_geometry_sha256"
                    ],
                    "trajectory_sha256": trajectory_sha256(
                        trajectory, motion
                    ),
                    "source_periodic_pose_sha256": trajectory[
                        "periodic_pose_sha256"
                    ],
                    "frame_count": FRAME_COUNT,
                    "pair_count": PAIR_COUNT,
                }
                for field, old_values in excluded.items():
                    if identity.get(field) in old_values:
                        raise InvalidMotionComponent(
                            f"IDENTITY_OVERLAP:{field}"
                        )
                identities.append(identity)
    if len(_collect(identities, "sequence_id")) != 32:
        raise InvalidMotionComponent("INTERNAL_SEQUENCE_COLLISION")
    payload = {
        "schema": "rcle.periodic_self_motion_counterfactual.motion_component_identity_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "role": "DEVELOPMENT_ROUTING_ONLY",
        "counts": {
            "blocks": 4,
            "seeds_per_block": 2,
            "clusters": 8,
            "arms_per_cluster": 4,
            "sequences": 32,
            "frames": 32 * FRAME_COUNT,
            "pairs": 32 * PAIR_COUNT,
            "stage_1_sequences": 16,
            "stage_2_sequences": 16,
        },
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "sha256": sha256_file(contract_path),
        },
        "exclusion_sources": sources,
        "zero_overlap_fields": list(fields),
        "seeds": seeds,
        "identities": identities,
        "stage_1": {
            "ordinal": 0,
            "state": "EXECUTABLE",
            "sequence_count": 16,
        },
        "stage_2": {
            "ordinal": 1,
            "state": "SEALED_NOT_EXECUTABLE",
            "sequence_count": 16,
            "opening_condition": (
                "independent Stage 1 decision OPEN_STAGE_2"
            ),
        },
        "formal_execution_authorized": False,
        "formal_authority_consumed": False,
        "terminal": "MOTION_COMPONENT_IDENTITIES_FROZEN / STAGE_1_NOT_RUN",
    }
    payload["identity_set_sha256"] = hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest()
    return payload


def build_stage_1_activation(
    root: Path, contract_path: Path, identity_path: Path
) -> dict[str, Any]:
    lock = load_json(identity_path)
    identities = [
        item for item in lock["identities"] if item["stage_number"] == 1
    ]
    sealed = [
        item for item in lock["identities"] if item["stage_number"] == 2
    ]
    if len(identities) != 16 or len(sealed) != 16:
        raise InvalidMotionComponent("ACTIVATION_IDENTITY_COUNTS")
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.motion_component_stage1_activation.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "decision": "STAGE_1_EXECUTION_AUTHORIZED",
        "contract_sha256": sha256_file(contract_path),
        "identity_lock_sha256": sha256_file(identity_path),
        "identity_set_sha256": lock["identity_set_sha256"],
        "stage_1_sequence_ids": [item["sequence_id"] for item in identities],
        "stage_1_allowlist_sha256": hashlib.sha256(
            canonical_bytes(identities)
        ).hexdigest(),
        "stage_2_sealed_sequence_ids_sha256": hashlib.sha256(
            canonical_bytes([item["sequence_id"] for item in sealed])
        ).hexdigest(),
        "stage_2_state": "SEALED_NOT_EXECUTABLE",
        "bindings": build_contract(root)["bindings"],
        "formal_execution_authorized": False,
        "formal_authority_consumed": False,
        "terminal": "STAGE_1_AUTHORIZED / STAGE_2_SEALED",
    }


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + 1 + end)
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    a = _ranks(left)
    b = _ranks(right)
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    numerator = sum(
        (x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True)
    )
    denominator = math.sqrt(
        sum((x - mean_a) ** 2 for x in a)
        * sum((y - mean_b) ** 2 for y in b)
    )
    return numerator / denominator if denominator > 0.0 else None


def _dominant_frequency(
    values: list[float], timestamps: list[float]
) -> float | None:
    if len(values) < 3 or len(values) != len(timestamps):
        return None
    series = np.asarray(values, dtype=np.float64)
    series = series - float(np.mean(series))
    if not np.any(np.abs(series) > 0.0):
        return None
    dt = float(np.median(np.diff(np.asarray(timestamps, dtype=np.float64))))
    frequencies = np.fft.rfftfreq(len(series), d=dt)
    power = np.abs(np.fft.rfft(series)) ** 2
    selected = np.flatnonzero((frequencies >= 0.7) & (frequencies <= 3.0))
    if selected.size == 0 or not np.any(power[selected] > 0.0):
        return None
    winner = int(selected[int(np.argmax(power[selected]))])
    return float(frequencies[winner])


def _sinusoid_r2(
    values: list[float], timestamps: list[float], frequency: float | None
) -> float | None:
    if frequency is None or len(values) < 3 or len(values) != len(timestamps):
        return None
    response = np.asarray(values, dtype=np.float64)
    total = float(np.sum((response - np.mean(response)) ** 2))
    if total <= 0.0:
        return None
    phase = 2.0 * np.pi * frequency * np.asarray(
        timestamps, dtype=np.float64
    )
    design = np.column_stack(
        (np.ones(len(response)), np.sin(phase), np.cos(phase))
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    return max(0.0, min(1.0, 1.0 - float(np.sum(residual**2)) / total))


def _angular_speed(left: np.ndarray, right: np.ndarray, dt: float) -> float:
    relative = right.T @ left
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine) / dt


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != PAIR_COUNT:
        raise InvalidMotionComponent("PAIR_COUNT")
    signed: list[float] = []
    raw_signed: list[float] = []
    positive_count = trigger_count = evaluable_count = 0
    streak = longest = 0
    compensated_cells: list[float] = []
    center_cells: list[float] = []
    peripheral_cells: list[float] = []
    fit_residuals: list[float] = []
    common_counts: list[float] = []
    raw_tracks: list[float] = []
    compensated_tracks: list[float] = []
    same_sign_fractions: list[float] = []
    angular_values: list[float] = []
    translation_values: list[float] = []
    timestamps: list[float] = []
    evaluable_timestamps: list[float] = []
    evaluable_signed_response: list[float] = []
    correlation_response: list[float] = []
    correlation_angular: list[float] = []
    correlation_translation: list[float] = []
    for index, row in enumerate(rows):
        if row.get("pair_index") != index:
            raise InvalidMotionComponent("PAIR_ORDER")
        evaluable = row.get("evaluable")
        response = row.get("compensated_expansion_median_per_s")
        if evaluable is True:
            value = float(response)
            if not math.isfinite(value):
                raise InvalidMotionComponent("NONFINITE_RESPONSE")
            signed.append(value)
            raw_signed.append(float(row["raw_expansion_median_per_s"]))
            evaluable_count += 1
            positive = value > THRESHOLD
            positive_count += int(positive)
            streak = streak + 1 if positive else 0
            longest = max(longest, streak)
            correlation_response.append(abs(value))
            correlation_angular.append(float(row["angular_speed_rad_s"]))
            correlation_translation.append(
                float(row["translation_speed_m_s"])
            )
            evaluable_signed_response.append(value)
            evaluable_timestamps.append(
                0.5
                * (
                    float(row["previous_timestamp_s"])
                    + float(row["current_timestamp_s"])
                )
            )
        elif evaluable is False:
            if response is not None:
                raise InvalidMotionComponent("ABSTENTION_RESPONSE")
            streak = 0
        else:
            raise InvalidMotionComponent("EVALUABLE")
        expected_trigger = streak >= 3
        if row.get("compensated_three_pair_trigger") is not expected_trigger:
            raise InvalidMotionComponent("TRIGGER")
        trigger_count += int(expected_trigger)
        angular_values.append(float(row["angular_speed_rad_s"]))
        translation_values.append(float(row["translation_speed_m_s"]))
        timestamps.append(
            0.5
            * (
                float(row["previous_timestamp_s"])
                + float(row["current_timestamp_s"])
            )
        )
        diagnostics = row.get("cell_diagnostics", {})
        raw_cells = diagnostics.get("raw_cells")
        compensated_diagnostics = diagnostics.get("compensated_cells")
        if (
            not isinstance(raw_cells, list)
            or not isinstance(compensated_diagnostics, list)
            or len(raw_cells) != 9
            or len(compensated_diagnostics) != 9
        ):
            raise InvalidMotionComponent("CELL_DIAGNOSTICS")
        pair_values = []
        for cell_index, (raw_cell, compensated_cell) in enumerate(
            zip(raw_cells, compensated_diagnostics, strict=True)
        ):
            if (
                raw_cell.get("evaluable") is True
                and compensated_cell.get("evaluable") is True
            ):
                cell_value = float(compensated_cell["expansion"])
                compensated_cells.append(cell_value)
                pair_values.append(cell_value)
                if cell_index == 4:
                    center_cells.append(cell_value)
                else:
                    peripheral_cells.append(cell_value)
                residual = compensated_cell.get(
                    "fit_residual_pixels_per_frame"
                )
                if residual is not None:
                    fit_residuals.append(float(residual))
        if pair_values and evaluable is True:
            pair_sign = 1 if float(response) > 0.0 else -1 if float(response) < 0.0 else 0
            same = sum(
                (1 if value > 0.0 else -1 if value < 0.0 else 0) == pair_sign
                for value in pair_values
            )
            same_sign_fractions.append(same / len(pair_values))
        common_counts.append(float(row["common_cell_count"]))
        raw_tracks.append(float(row["raw_track_count"]))
        compensated_tracks.append(float(row["compensated_track_count"]))
    pair_absolute = [
        float(row["compensated_abs_expansion_median_per_s"])
        for row in rows
        if row.get("evaluable") is True
    ]
    raw_pair_absolute = [
        float(row["raw_abs_expansion_median_per_s"])
        for row in rows
        if row.get("evaluable") is True
    ]
    center_absolute = [abs(value) for value in center_cells]
    peripheral_absolute = [abs(value) for value in peripheral_cells]
    angular_frequency = _dominant_frequency(angular_values, timestamps)
    translation_frequency = _dominant_frequency(
        translation_values, timestamps
    )
    return {
        "scheduled_pair_count": PAIR_COUNT,
        "evaluable_pair_count": evaluable_count,
        "evaluable_fraction": evaluable_count / PAIR_COUNT,
        "compensated_signed_p50": _quantile(signed, 0.50),
        "compensated_signed_p90": _quantile(signed, 0.90),
        "compensated_absolute_p50": _quantile(pair_absolute, 0.50),
        "compensated_absolute_p90": _quantile(pair_absolute, 0.90),
        "raw_signed_p50": _quantile(raw_signed, 0.50),
        "raw_signed_p90": _quantile(raw_signed, 0.90),
        "raw_absolute_p50": _quantile(raw_pair_absolute, 0.50),
        "raw_absolute_p90": _quantile(raw_pair_absolute, 0.90),
        "sign_positive_ratio_evaluable": (
            sum(value > 0.0 for value in signed) / evaluable_count
            if evaluable_count
            else None
        ),
        "threshold_positive_fraction_evaluable": (
            positive_count / evaluable_count if evaluable_count else None
        ),
        "threshold_positive_density_fixed": positive_count / PAIR_COUNT,
        "trigger_count": trigger_count,
        "trigger_density_fixed": trigger_count / PAIR_COUNT,
        "longest_positive_streak_pairs": longest,
        "compensated_cell_absolute_p50": _quantile(
            [abs(value) for value in compensated_cells], 0.50
        ),
        "compensated_cell_absolute_p90": _quantile(
            [abs(value) for value in compensated_cells], 0.90
        ),
        "center_cell_absolute_p50": _quantile(center_absolute, 0.50),
        "periphery_cell_absolute_p50": _quantile(
            peripheral_absolute, 0.50
        ),
        "same_sign_spatial_fraction_p50": _quantile(
            same_sign_fractions, 0.50
        ),
        "cell_fit_residual_px_per_frame_p50": _quantile(
            fit_residuals, 0.50
        ),
        "cell_fit_residual_px_per_frame_p90": _quantile(
            fit_residuals, 0.90
        ),
        "common_cell_count_p50": _quantile(common_counts, 0.50),
        "raw_track_count_p50": _quantile(raw_tracks, 0.50),
        "compensated_track_count_p50": _quantile(
            compensated_tracks, 0.50
        ),
        "absolute_response_vs_angular_speed_spearman": _spearman(
            correlation_response, correlation_angular
        ),
        "absolute_response_vs_translation_speed_spearman": _spearman(
            correlation_response, correlation_translation
        ),
        "angular_speed_dominant_frequency_hz": angular_frequency,
        "translation_speed_dominant_frequency_hz": translation_frequency,
        "signed_response_at_angular_frequency_r2": _sinusoid_r2(
            evaluable_signed_response,
            evaluable_timestamps,
            angular_frequency,
        ),
        "signed_response_at_translation_frequency_r2": _sinusoid_r2(
            evaluable_signed_response,
            evaluable_timestamps,
            translation_frequency,
        ),
    }


def _cluster_worker(task: dict[str, Any]) -> dict[str, Any]:
    guarded._initialize_worker()
    root = repo_root()
    identities = task["identities"]
    block = task["block"]
    ordinal = task["ordinal"]
    output_root = Path(task["output_root"])
    cluster_id = identities[0]["cluster_id"]
    staging = output_root / "staging" / f"{cluster_id}.{os.getpid()}.tmp"
    final = output_root / "clusters" / cluster_id
    staging.mkdir(parents=True, exist_ok=False)
    scene = _scene(block, ordinal)
    if any(
        item["scene_geometry_sha256"] != scene["scene_geometry_sha256"]
        for item in identities
    ):
        raise InvalidMotionComponent("SCENE_IDENTITY")
    trajectory = load_json(root / TRAJECTORY_MANIFEST)[block]
    protocol = load_json(root / transport.PROTOCOL_RELATIVE)
    arm_outputs = []
    render_calls = 0
    started = time.perf_counter()
    for identity in identities:
        motion = identity["motion"]
        poses = motion_poses(trajectory, motion)
        if trajectory_sha256(trajectory, motion) != identity["trajectory_sha256"]:
            raise InvalidMotionComponent("TRAJECTORY_IDENTITY")
        state = r3.PairState()
        previous: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
        rows = []
        streak = 0
        static_frame: tuple[np.ndarray, np.ndarray] | None = None
        original_common = r0_evaluation._common_cell_expansions
        for frame_index, pose in enumerate(poses):
            rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
            translation = np.asarray(pose["translation_m"], dtype=np.float64)
            if motion == "STATIC" and static_frame is not None:
                rgb, mask = static_frame
            else:
                rendered = qms.render_pair(scene, rotation, translation)
                rgb = rendered["rgb_pair"]["clean"]
                mask = rendered["valid_mask"]
                render_calls += 1
                if motion == "STATIC":
                    static_frame = (rgb, mask)
            if previous is not None:
                previous_rgb, previous_mask, previous_pose = previous
                captured: dict[str, Any] = {}

                def capture_cells(
                    raw_cells: Any, compensated_cells: Any
                ) -> Any:
                    if captured:
                        raise InvalidMotionComponent("MULTIPLE_CELL_CAPTURE")
                    output = original_common(raw_cells, compensated_cells)
                    captured["raw_cells"] = [
                        cell.to_dict() for cell in raw_cells
                    ]
                    captured["compensated_cells"] = [
                        cell.to_dict() for cell in compensated_cells
                    ]
                    captured["common_cell_indices"] = list(output[2])
                    return output

                with mock.patch.object(
                    r0_evaluation,
                    "_common_cell_expansions",
                    side_effect=capture_cells,
                ):
                    row = transport.evaluate_pair(
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
                if not captured:
                    raise InvalidMotionComponent("CELL_CAPTURE_MISSING")
                if (
                    len(captured["common_cell_indices"])
                    != row["common_cell_count"]
                    or (
                        row.get("evaluable") is True
                        and captured["common_cell_indices"]
                        != row.get("common_cell_indices")
                    )
                ):
                    raise InvalidMotionComponent("CELL_CAPTURE_MISMATCH")
                response = row.get("compensated_expansion_median_per_s")
                if row.get("evaluable") is True:
                    streak = (
                        streak + 1 if float(response) > THRESHOLD else 0
                    )
                else:
                    streak = 0
                    row.setdefault(
                        "compensated_expansion_median_per_s", None
                    )
                    row.setdefault("raw_expansion_median_per_s", None)
                row["compensated_three_pair_trigger"] = streak >= 3
                dt = float(pose["timestamp_s"]) - float(
                    previous_pose["timestamp_s"]
                )
                row["angular_speed_rad_s"] = _angular_speed(
                    np.asarray(
                        previous_pose["rotation_matrix"], dtype=np.float64
                    ),
                    rotation,
                    dt,
                )
                row["translation_speed_m_s"] = float(
                    np.linalg.norm(
                        translation
                        - np.asarray(
                            previous_pose["translation_m"],
                            dtype=np.float64,
                        )
                    )
                    / dt
                )
                row["cell_diagnostics"] = captured
                envelope = {
                    key: identity[key]
                    for key in (
                        "sequence_id",
                        "cluster_id",
                        "block",
                        "ordinal",
                        "stage_number",
                        "role",
                        "arm",
                        "motion",
                    )
                }
                envelope["pair_index"] = frame_index - 1
                envelope.update(row)
                rows.append(envelope)
            previous = (rgb, mask, pose)
        summary = reduce_rows(rows)
        arm_dir = staging / identity["sequence_id"]
        arm_dir.mkdir()
        ledger_path = arm_dir / "pair_ledger.jsonl"
        summary_path = arm_dir / "reduced_metrics.json"
        write_jsonl(ledger_path, rows)
        write_json(summary_path, summary)
        receipt = {
            **identity,
            "bindings": task["bindings"],
            "pair_ledger_sha256": sha256_file(ledger_path),
            "reduced_metrics_sha256": sha256_file(summary_path),
            "r3_source_unchanged": True,
            "cell_capture_hook_return_values_unchanged": True,
            "terminal": "MOTION_COMPONENT_ARM_COMPLETE",
        }
        receipt_path = arm_dir / "receipt.json"
        write_json(receipt_path, receipt)
        arm_outputs.append(
            {
                "sequence_id": identity["sequence_id"],
                "receipt_path": (
                    Path("clusters")
                    / cluster_id
                    / identity["sequence_id"]
                    / "receipt.json"
                ).as_posix(),
                "receipt_sha256": sha256_file(receipt_path),
            }
        )
    cluster_receipt = {
        "task_id": TASK_ID,
        "cluster_id": cluster_id,
        "block": block,
        "ordinal": ordinal,
        "stage_number": ordinal + 1,
        "sequence_count": 4,
        "arm_outputs": arm_outputs,
        "qms_render_pair_calls": render_calls,
        "expected_qms_render_pair_calls": 1 + 3 * FRAME_COUNT,
        "wall_seconds": time.perf_counter() - started,
        "operator_source_sha256": sha256_file(root / OPERATOR_SOURCE),
        "terminal": "MOTION_COMPONENT_CLUSTER_COMPLETE",
    }
    write_json(staging / "cluster_receipt.json", cluster_receipt)
    if final.exists():
        raise InvalidMotionComponent("FINAL_CLUSTER_EXISTS")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    return {
        **cluster_receipt,
        "cluster_receipt_path": (
            Path("clusters") / cluster_id / "cluster_receipt.json"
        ).as_posix(),
        "cluster_receipt_sha256": sha256_file(
            final / "cluster_receipt.json"
        ),
    }


def _validate_stage_2_decision(
    root: Path, decision_path: Path, identity_path: Path
) -> dict[str, Any]:
    decision = load_json(decision_path)
    if (
        decision.get("task_id") != TASK_ID
        or decision.get("decision") != "OPEN_STAGE_2"
        or decision.get("stage_2_ordinal") != 1
        or decision.get("identity_lock_sha256") != sha256_file(identity_path)
        or decision.get("formal_authority_consumed") is not False
    ):
        raise InvalidMotionComponent("STAGE_2_DECISION")
    summary = decision.get("routing_direction_summary", {})
    opened = [
        name
        for name, value in summary.items()
        if isinstance(value, dict)
        and value.get("positive_count", 0) >= 3
        and value.get("opens_stage_2") is True
    ]
    if not opened or decision.get("opened_by_contrasts") != opened:
        raise InvalidMotionComponent("STAGE_2_ROUTING_RULE")
    source = root / decision.get("validator_source_path", "")
    if (
        not source.is_file()
        or decision.get("validator_source_sha256") != sha256_file(source)
    ):
        raise InvalidMotionComponent("STAGE_2_DECISION_VALIDATOR")
    artifact_fields = (
        ("stage_1_run_receipt_path", "stage_1_run_receipt_sha256"),
        ("stage_1_analysis_result_path", "stage_1_analysis_result_sha256"),
        (
            "stage_1_independent_receipt_path",
            "stage_1_independent_receipt_sha256",
        ),
    )
    loaded = {}
    for path_key, hash_key in artifact_fields:
        artifact = root / decision.get(path_key, "")
        if (
            not artifact.is_file()
            or decision.get(hash_key) != sha256_file(artifact)
        ):
            raise InvalidMotionComponent(f"STAGE_2_DECISION_BINDING:{path_key}")
        loaded[path_key] = load_json(artifact)
    if (
        loaded["stage_1_run_receipt_path"].get("terminal")
        != "STAGE_1_EXECUTION_COMPLETE / INDEPENDENT_VALIDATION_REQUIRED"
        or loaded["stage_1_independent_receipt_path"].get("terminal")
        != "VALID / STAGE_1_ROUTING_COMPLETE"
        or loaded["stage_1_independent_receipt_path"].get(
            "validator_source_sha256"
        )
        != sha256_file(source)
    ):
        raise InvalidMotionComponent("STAGE_2_DECISION_TERMINALS")
    return decision


def _validate_stage_1_activation(
    root: Path,
    activation_path: Path,
    contract_path: Path,
    identity_path: Path,
) -> dict[str, Any]:
    observed = load_json(activation_path)
    expected = build_stage_1_activation(
        root, contract_path, identity_path
    )
    if canonical_bytes(observed) != canonical_bytes(expected):
        raise InvalidMotionComponent("STAGE_1_ACTIVATION_DRIFT")
    return observed


def run_stage(
    identity_path: Path,
    output_root: Path,
    stage_number: int,
    stage_1_activation_path: Path | None = None,
    decision_path: Path | None = None,
) -> dict[str, Any]:
    root = repo_root()
    lock = load_json(identity_path)
    contract_path = root / lock["contract"]["path"]
    if canonical_bytes(lock) != canonical_bytes(
        build_identity_lock(root, contract_path)
    ):
        raise InvalidMotionComponent("IDENTITY_LOCK_DRIFT")
    if stage_number not in (1, 2):
        raise InvalidMotionComponent("STAGE_NUMBER")
    if stage_number == 1 and (output_root.parent / "stage2").exists():
        raise InvalidMotionComponent("STAGE_2_OUTPUT_PRESENT_BEFORE_ROUTING")
    decision = None
    activation = None
    if stage_number == 1:
        if stage_1_activation_path is None:
            raise InvalidMotionComponent("STAGE_1_ACTIVATION_REQUIRED")
        activation = _validate_stage_1_activation(
            root,
            stage_1_activation_path,
            contract_path,
            identity_path,
        )
        if decision_path is not None:
            raise InvalidMotionComponent("STAGE_1_DECISION_PROHIBITED")
    elif stage_number == 2:
        if stage_1_activation_path is not None:
            raise InvalidMotionComponent("STAGE_2_STAGE_1_ACTIVATION_PROHIBITED")
        if decision_path is None:
            raise InvalidMotionComponent("STAGE_2_DECISION_REQUIRED")
        decision = _validate_stage_2_decision(
            root, decision_path, identity_path
        )
    predecessor_formal = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "p4_formal"
    )
    successor_formal = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1_successor_formal"
    )
    previous_dev = root / PREVIOUS_DEV_BUNDLE
    if (
        not predecessor_formal.is_dir()
        or successor_formal.exists()
        or not previous_dev.is_dir()
    ):
        raise InvalidMotionComponent("FORMAL_OR_DEV_FIREWALL_PATH")
    firewall_before = {
        "predecessor_formal_tree_sha256": tree_sha256(predecessor_formal),
        "previous_dev_tree_sha256": tree_sha256(previous_dev),
        "formal_activation_decision_sha256": sha256_file(
            root / FORMAL_DECISION
        ),
        "successor_formal_identity_lock_sha256": sha256_file(
            root / SUCCESSOR_FORMAL_LOCK
        ),
    }
    memory = psutil.virtual_memory()
    if memory.available < 8 * GIB:
        raise InvalidMotionComponent("LAUNCH_AVAILABLE_RAM_BELOW_8_GIB")
    bindings = _bindings(root, contract_path, identity_path)
    identities = [
        item
        for item in lock["identities"]
        if item["stage_number"] == stage_number
    ]
    if len(identities) != 16:
        raise InvalidMotionComponent("STAGE_IDENTITY_COUNT")
    if output_root.exists():
        raise InvalidMotionComponent("OUTPUT_EXISTS")
    output_root.mkdir(parents=True, exist_ok=False)
    tasks = []
    ordinal = stage_number - 1
    for block in BLOCKS:
        cluster = [
            item
            for item in identities
            if item["block"] == block and item["ordinal"] == ordinal
        ]
        if len(cluster) != 4:
            raise InvalidMotionComponent("CLUSTER_IDENTITY_COUNT")
        tasks.append(
            {
                "block": block,
                "ordinal": ordinal,
                "identities": cluster,
                "output_root": str(output_root),
                "bindings": bindings,
            }
        )
    started = time.perf_counter()
    started_swap = psutil.swap_memory()
    last_swap = started_swap
    minimum_available = memory.available
    paging_streak = 0
    samples = []
    completed: dict[str, dict[str, Any]] = {}
    worker_pids: list[int] = []
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=guarded._initialize_worker,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(_cluster_worker, task): task for task in tasks
        }
        worker_pids = sorted(int(pid) for pid in executor._processes)
        while futures:
            done, _ = wait(
                futures, timeout=20.0, return_when=FIRST_COMPLETED
            )
            current_memory = psutil.virtual_memory()
            current_swap = psutil.swap_memory()
            minimum_available = min(
                minimum_available, current_memory.available
            )
            paging_delta = max(
                0,
                int(current_swap.sin - last_swap.sin)
                + int(current_swap.sout - last_swap.sout),
            )
            paging_streak = paging_streak + 1 if paging_delta else 0
            last_swap = current_swap
            if current_memory.available < 4 * GIB:
                raise InvalidMotionComponent("RUN_AVAILABLE_RAM_BELOW_4_GIB")
            if paging_streak >= 2:
                raise InvalidMotionComponent("SUSTAINED_PAGING")
            for future in done:
                futures.pop(future)
                result = future.result()
                completed[result["cluster_id"]] = result
            sample = {
                "sample_index": len(samples),
                "elapsed_seconds": time.perf_counter() - started,
                "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                "available_ram_bytes": current_memory.available,
                "swap_in_total": int(current_swap.sin),
                "swap_out_total": int(current_swap.sout),
                "completed_clusters": len(completed),
            }
            samples.append(sample)
            write_json(
                output_root / "progress.json",
                {
                    "task_id": TASK_ID,
                    "stage_number": stage_number,
                    "completed_clusters": len(completed),
                    "total_clusters": 4,
                    "completed_sequences": len(completed) * 4,
                    "total_sequences": 16,
                    "status": "SUCCESS" if not futures else "RUNNING",
                    "last_heartbeat_utc": sample["sampled_at_utc"],
                },
            )
            write_json(
                output_root / "telemetry.json",
                {
                    "task_id": TASK_ID,
                    "stage_number": stage_number,
                    "samples": samples,
                    "scientific_response_fields_present": False,
                },
            )
    residual = [pid for pid in worker_pids if psutil.pid_exists(pid)]
    if residual:
        raise InvalidMotionComponent("RESIDUAL_WORKERS")
    firewall_after = {
        "predecessor_formal_tree_sha256": tree_sha256(predecessor_formal),
        "previous_dev_tree_sha256": tree_sha256(previous_dev),
        "formal_activation_decision_sha256": sha256_file(
            root / FORMAL_DECISION
        ),
        "successor_formal_identity_lock_sha256": sha256_file(
            root / SUCCESSOR_FORMAL_LOCK
        ),
    }
    if firewall_after != firewall_before or successor_formal.exists():
        raise InvalidMotionComponent("FORMAL_OR_DEV_FIREWALL_DRIFT")
    ordered = [
        completed[f"MCL_R0_{block}_S{stage_number}"] for block in BLOCKS
    ]
    result = {
        "schema": "rcle.periodic_self_motion_counterfactual.motion_component_run.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "stage_number": stage_number,
        "terminal": (
            f"STAGE_{stage_number}_EXECUTION_COMPLETE / "
            "INDEPENDENT_VALIDATION_REQUIRED"
        ),
        "identity_lock_sha256": sha256_file(identity_path),
        "identity_set_sha256": lock["identity_set_sha256"],
        "bindings": bindings,
        "stage_2_decision_sha256": (
            sha256_file(decision_path) if decision is not None else None
        ),
        "stage_1_activation_sha256": (
            sha256_file(stage_1_activation_path)
            if activation is not None
            else None
        ),
        "clusters": ordered,
        "counts": {
            "clusters": 4,
            "sequences": 16,
            "frames": 16 * FRAME_COUNT,
            "pairs": 16 * PAIR_COUNT,
        },
        "resource": {
            "available_ram_at_launch_bytes": memory.available,
            "minimum_available_ram_bytes": minimum_available,
            "swap_in_delta": int(last_swap.sin - started_swap.sin),
            "swap_out_delta": int(last_swap.sout - started_swap.sout),
            "heartbeat_max_interval_seconds": max(
                (
                    right["elapsed_seconds"] - left["elapsed_seconds"]
                    for left, right in zip(samples, samples[1:])
                ),
                default=0.0,
            ),
            "worker_pids": worker_pids,
            "residual_worker_pids": residual,
            "wall_seconds": time.perf_counter() - started,
        },
        "formal_firewall": {
            "before": firewall_before,
            "after": firewall_after,
            "successor_formal_path_absent": True,
            "formal_sequences_run": 0,
            "formal_r3_pair_core_calls": 0,
            "formal_authority_consumed": False,
        },
        "formal_execution_authorized_by_this_run": False,
    }
    write_exclusive(output_root / "run_receipt.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--contract", type=Path, required=True)
    freeze.add_argument("--identity-lock", type=Path, required=True)
    freeze.add_argument("--stage-1-activation", type=Path, required=True)
    execute = sub.add_parser("run-stage")
    execute.add_argument("--identity-lock", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    execute.add_argument("--stage-number", type=int, required=True)
    execute.add_argument("--stage-1-activation", type=Path)
    execute.add_argument("--decision", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        root = args.repo_root.resolve()
        contract_path = (root / args.contract).resolve()
        identity_path = (root / args.identity_lock).resolve()
        activation_path = (root / args.stage_1_activation).resolve()
        write_exclusive(contract_path, build_contract(root))
        write_exclusive(
            identity_path, build_identity_lock(root, contract_path)
        )
        write_exclusive(
            activation_path,
            build_stage_1_activation(
                root, contract_path, identity_path
            ),
        )
        value = load_json(identity_path)
    else:
        value = run_stage(
            args.identity_lock.resolve(),
            args.output_root.resolve(),
            args.stage_number,
            (
                args.stage_1_activation.resolve()
                if args.stage_1_activation
                else None
            ),
            args.decision.resolve() if args.decision else None,
        )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "terminal": value["terminal"],
                "formal_sequences_run": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
