"""Independent validation for two-stage clean motion-component localization.

This module does not import the Stage A runner, QMS operator, R3 transport,
pair core, or formal analysis.  It independently derives identities and pose
components, validates the observation-only cell sidecar, reconstructs pair
scalars from common cells, reduces every ledger, and makes the frozen routing
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any
from unittest import mock

import numpy as np

from . import generator_geometry as geometry


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC", "ROTATION_ONLY", "TRANSLATION_ONLY", "FULL_6DOF")
FRAME_COUNT = 602
PAIR_COUNT = 601
THRESHOLD = 0.01
GIB = 1024**3
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
RUNNER_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "motion_component_localization_r0.py"
)
VALIDATOR_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "validate_motion_component_localization_independent_r0.py"
)


class InvalidIndependentMotionComponent(ValueError):
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    token = f"{TASK_ID}|SCENE|{block}|{ordinal:02d}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return {
        "token": token,
        "token_sha256": digest,
        "numeric_seed_uint64": int.from_bytes(bytes.fromhex(digest)[:8], "big"),
    }


def _scene_hash(block: str, ordinal: int) -> str:
    seed = _seed(block, ordinal)
    with mock.patch.object(
        geometry, "derive_seed", return_value=seed["numeric_seed_uint64"]
    ):
        return geometry.build_scene(
            block, ordinal, "MOTION_COMPONENT"
        )["scene_geometry_sha256"]


def _poses(
    trajectory: dict[str, Any], motion: str
) -> list[dict[str, Any]]:
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    zero = [0.0, 0.0, 0.0]
    return [
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
        for source in trajectory["poses"]
    ]


def _trajectory_hash(trajectory: dict[str, Any], motion: str) -> str:
    return hashlib.sha256(
        geometry.canonical_bytes(_poses(trajectory, motion))
    ).hexdigest()


def validate_contract(root: Path, path: Path) -> dict[str, Any]:
    value = load_json(path)
    design = value.get("design", {})
    routing = value.get("routing", {})
    contrasts = routing.get("paired_contrasts", {})
    expected_contrasts = {
        "ROTATION_MINUS_STATIC": {
            "metric": "compensated_absolute_p90",
            "formula": "ROTATION_ONLY-STATIC",
        },
        "TRANSLATION_MINUS_ROTATION": {
            "metric": "compensated_signed_p90",
            "formula": "TRANSLATION_ONLY-ROTATION_ONLY",
        },
        "FULL_MINUS_MAX_SINGLE": {
            "metric": "compensated_signed_p90",
            "formula": (
                "FULL_6DOF-max(ROTATION_ONLY,TRANSLATION_ONLY)"
            ),
        },
    }
    if (
        value.get("task_id") != TASK_ID
        or value.get("stage") != "DEVELOPMENT_ROUTING_AUDIT"
        or design.get("blocks") != list(BLOCKS)
        or design.get("scene_seeds_per_block") != 2
        or design.get("identities_frozen_before_stage_1") != 32
        or design.get("stage_1_sequences") != 16
        or design.get("stage_2_sequences") != 16
        or design.get("stage_2_initial_state") != "SEALED_NOT_EXECUTABLE"
        or design.get("motion_arms") != list(MOTIONS)
        or design.get("pair_rows_are_independent_samples") is not False
        or contrasts != expected_contrasts
        or routing.get("no_p_values") is not True
        or routing.get("no_confirmatory_inference") is not True
        or value.get("r3_algorithm_changed") is not False
        or value.get("formal_identity_execution") is not False
        or value.get("formal_authority_consumed") is not False
    ):
        raise InvalidIndependentMotionComponent("CONTRACT")
    expected_bindings = {
        "qms_r1_operator_source_sha256": sha256_file(root / OPERATOR_SOURCE),
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
    }
    if value.get("bindings") != expected_bindings:
        raise InvalidIndependentMotionComponent("CONTRACT_BINDINGS")
    return value


def _exclusion_union(
    root: Path, records: list[dict[str, Any]]
) -> dict[str, set[Any]]:
    fields = (
        "numeric_seed_uint64",
        "token",
        "token_sha256",
        "cluster_id",
        "sequence_id",
        "scene_geometry_sha256",
    )
    union = {field: set() for field in fields}
    expected = dict(EXCLUSION_SOURCES)
    observed = {
        item.get("label"): item
        for item in records
        if item.get("label") in expected
    }
    if set(observed) != set(expected):
        raise InvalidIndependentMotionComponent("EXCLUSION_FILE_SET")
    for label, relative in EXCLUSION_SOURCES:
        record = observed[label]
        path = root / relative
        if (
            record.get("path") != relative
            or record.get("sha256") != sha256_file(path)
        ):
            raise InvalidIndependentMotionComponent(
                f"EXCLUSION_FILE:{label}"
            )
        value = load_json(path)
        for field in fields:
            values = _collect(value, field)
            if field == "token_sha256":
                values.update(_collect(value, "cluster_token_sha256"))
            union[field].update(values)
    runtime_records = [
        item
        for item in records
        if item.get("label") == "PREFLIGHT_RUNTIME_SCENE_RECEIPTS"
    ]
    if len(runtime_records) != len(PREFLIGHT_RECEIPT_DIRS):
        raise InvalidIndependentMotionComponent("RUNTIME_EXCLUSION_SET")
    by_path = {item.get("path"): item for item in runtime_records}
    runtime_scenes = set()
    for relative in PREFLIGHT_RECEIPT_DIRS:
        record = by_path.get(relative)
        directory = root / relative
        if (
            record is None
            or record.get("tree_sha256") != tree_sha256(directory)
        ):
            raise InvalidIndependentMotionComponent("RUNTIME_EXCLUSION")
        scenes = set(record.get("scene_geometry_sha256", []))
        runtime_scenes.update(scenes)
    if len(runtime_scenes) != 4:
        raise InvalidIndependentMotionComponent("RUNTIME_SCENE_SET")
    union["scene_geometry_sha256"].update(runtime_scenes)
    return union


def validate_identity_lock(
    root: Path, path: Path, contract_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = load_json(path)
    identities = value.get("identities")
    seeds = value.get("seeds")
    if (
        value.get("task_id") != TASK_ID
        or value.get("counts")
        != {
            "blocks": 4,
            "seeds_per_block": 2,
            "clusters": 8,
            "arms_per_cluster": 4,
            "sequences": 32,
            "frames": 19264,
            "pairs": 19232,
            "stage_1_sequences": 16,
            "stage_2_sequences": 16,
        }
        or not isinstance(identities, list)
        or len(identities) != 32
        or not isinstance(seeds, list)
        or len(seeds) != 8
        or value.get("contract", {}).get("path")
        != contract_path.relative_to(root).as_posix()
        or value.get("contract", {}).get("sha256")
        != sha256_file(contract_path)
        or value.get("stage_1", {}).get("state") != "EXECUTABLE"
        or value.get("stage_2", {}).get("state")
        != "SEALED_NOT_EXECUTABLE"
        or value.get("formal_execution_authorized") is not False
        or value.get("formal_authority_consumed") is not False
    ):
        raise InvalidIndependentMotionComponent("IDENTITY_HEADER")
    trajectories = load_json(root / TRAJECTORY_MANIFEST)
    expected_seeds = []
    expected: dict[str, dict[str, Any]] = {}
    for block in BLOCKS:
        trajectory = trajectories[block]
        for ordinal in (0, 1):
            stage_number = ordinal + 1
            state = (
                "STAGE_1_EXECUTABLE"
                if stage_number == 1
                else "STAGE_2_SEALED_NOT_EXECUTABLE"
            )
            seed = _seed(block, ordinal)
            scene_hash = _scene_hash(block, ordinal)
            cluster_id = f"MCL_R0_{block}_S{stage_number}"
            expected_seeds.append(
                {
                    "block": block,
                    "ordinal": ordinal,
                    "stage_number": stage_number,
                    "stage_state": state,
                    **seed,
                    "cluster_id": cluster_id,
                    "scene_geometry_sha256": scene_hash,
                }
            )
            for arm_ordinal, motion in enumerate(MOTIONS):
                sequence_id = f"{cluster_id}__{motion}__CLEAN"
                expected[sequence_id] = {
                    "sequence_id": sequence_id,
                    "cluster_id": cluster_id,
                    "block": block,
                    "ordinal": ordinal,
                    "stage_number": stage_number,
                    "stage_state": state,
                    "role": "MOTION_COMPONENT_ROUTING_DEV",
                    "arm": f"{motion}__CLEAN",
                    "arm_ordinal": arm_ordinal,
                    "motion": motion,
                    "quality": "CLEAN",
                    **seed,
                    "scene_geometry_sha256": scene_hash,
                    "trajectory_sha256": _trajectory_hash(
                        trajectory, motion
                    ),
                    "source_periodic_pose_sha256": trajectory[
                        "periodic_pose_sha256"
                    ],
                    "frame_count": FRAME_COUNT,
                    "pair_count": PAIR_COUNT,
                }
    if seeds != expected_seeds or identities != list(expected.values()):
        raise InvalidIndependentMotionComponent("IDENTITY_GRID")
    if value.get("identity_set_sha256") != hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest():
        raise InvalidIndependentMotionComponent("IDENTITY_SET_HASH")
    for block in BLOCKS:
        full = next(
            item
            for item in expected.values()
            if item["block"] == block
            and item["ordinal"] == 0
            and item["motion"] == "FULL_6DOF"
        )
        if (
            full["trajectory_sha256"]
            != trajectories[block]["periodic_pose_sha256"]
        ):
            raise InvalidIndependentMotionComponent("FULL_TRAJECTORY_CHAIN")
    excluded = _exclusion_union(root, value.get("exclusion_sources", []))
    for field, old_values in excluded.items():
        new_values = _collect(identities + seeds, field)
        if new_values & old_values:
            raise InvalidIndependentMotionComponent(
                f"IDENTITY_OVERLAP:{field}"
            )
    return value, expected


def validate_stage_1_activation(
    root: Path,
    path: Path,
    contract_path: Path,
    identity_path: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    value = load_json(path)
    stage_1 = [
        item for item in lock["identities"] if item["stage_number"] == 1
    ]
    stage_2 = [
        item for item in lock["identities"] if item["stage_number"] == 2
    ]
    if (
        value.get("task_id") != TASK_ID
        or value.get("decision") != "STAGE_1_EXECUTION_AUTHORIZED"
        or value.get("contract_sha256") != sha256_file(contract_path)
        or value.get("identity_lock_sha256") != sha256_file(identity_path)
        or value.get("identity_set_sha256") != lock["identity_set_sha256"]
        or value.get("stage_1_sequence_ids")
        != [item["sequence_id"] for item in stage_1]
        or value.get("stage_1_allowlist_sha256")
        != hashlib.sha256(canonical_bytes(stage_1)).hexdigest()
        or value.get("stage_2_sealed_sequence_ids_sha256")
        != hashlib.sha256(
            canonical_bytes([item["sequence_id"] for item in stage_2])
        ).hexdigest()
        or value.get("stage_2_state") != "SEALED_NOT_EXECUTABLE"
        or value.get("formal_execution_authorized") is not False
        or value.get("formal_authority_consumed") is not False
        or value.get("bindings") != load_json(contract_path).get("bindings")
    ):
        raise InvalidIndependentMotionComponent("STAGE_1_ACTIVATION")
    return value


def validate_stage_1_decision(
    root: Path, path: Path, identity_path: Path
) -> dict[str, Any]:
    value = load_json(path)
    summary = value.get("routing_direction_summary", {})
    opened = [
        name
        for name, item in summary.items()
        if isinstance(item, dict)
        and item.get("positive_count", 0) >= 3
        and item.get("opens_stage_2") is True
    ]
    if (
        value.get("task_id") != TASK_ID
        or value.get("decision") != "OPEN_STAGE_2"
        or value.get("stage_2_ordinal") != 1
        or value.get("opened_by_contrasts") != opened
        or not opened
        or value.get("identity_lock_sha256") != sha256_file(identity_path)
        or value.get("validator_source_sha256")
        != sha256_file(root / VALIDATOR_SOURCE)
        or value.get("formal_sequences_run") != 0
        or value.get("formal_r3_pair_core_calls") != 0
        or value.get("formal_authority_consumed") is not False
    ):
        raise InvalidIndependentMotionComponent("STAGE_1_DECISION")
    loaded: dict[str, dict[str, Any]] = {}
    for path_key, hash_key in (
        ("stage_1_run_receipt_path", "stage_1_run_receipt_sha256"),
        ("stage_1_analysis_result_path", "stage_1_analysis_result_sha256"),
        (
            "stage_1_independent_receipt_path",
            "stage_1_independent_receipt_sha256",
        ),
    ):
        artifact = root / value.get(path_key, "")
        if (
            not artifact.is_file()
            or value.get(hash_key) != sha256_file(artifact)
        ):
            raise InvalidIndependentMotionComponent(
                f"STAGE_1_DECISION_BINDING:{path_key}"
            )
        loaded[path_key] = load_json(artifact)
    if (
        loaded["stage_1_run_receipt_path"].get("terminal")
        != "STAGE_1_EXECUTION_COMPLETE / INDEPENDENT_VALIDATION_REQUIRED"
        or loaded["stage_1_analysis_result_path"].get("terminal")
        != "STAGE_1_ROUTING_COMPLETE / DESCRIPTIVE_ONLY"
        or loaded["stage_1_independent_receipt_path"].get("terminal")
        != "VALID / STAGE_1_ROUTING_COMPLETE"
        or loaded["stage_1_independent_receipt_path"].get(
            "validator_source_sha256"
        )
        != sha256_file(root / VALIDATOR_SOURCE)
    ):
        raise InvalidIndependentMotionComponent("STAGE_1_DECISION_TERMINALS")
    return value


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
    return float(
        frequencies[int(selected[int(np.argmax(power[selected]))])]
    )


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


def _close(actual: Any, expected: Any, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise InvalidIndependentMotionComponent(label)
    elif not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise InvalidIndependentMotionComponent(label)


def reduce_rows(
    rows: list[dict[str, Any]],
    identity: dict[str, Any],
    poses: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(rows) != PAIR_COUNT:
        raise InvalidIndependentMotionComponent("PAIR_COUNT")
    signed: list[float] = []
    pair_absolute: list[float] = []
    raw_signed: list[float] = []
    raw_pair_absolute: list[float] = []
    positive_count = trigger_count = evaluable_count = 0
    streak = longest = 0
    cell_values: list[float] = []
    center_values: list[float] = []
    periphery_values: list[float] = []
    fit_residuals: list[float] = []
    same_sign: list[float] = []
    common_counts: list[float] = []
    raw_tracks: list[float] = []
    compensated_tracks: list[float] = []
    angular_values: list[float] = []
    translation_values: list[float] = []
    timestamps: list[float] = []
    evaluable_timestamps: list[float] = []
    evaluable_signed: list[float] = []
    correlation_response: list[float] = []
    correlation_angular: list[float] = []
    correlation_translation: list[float] = []
    envelope_fields = (
        "sequence_id",
        "cluster_id",
        "block",
        "ordinal",
        "stage_number",
        "role",
        "arm",
        "motion",
    )
    for index, row in enumerate(rows):
        if (
            row.get("pair_index") != index
            or any(row.get(key) != identity[key] for key in envelope_fields)
        ):
            raise InvalidIndependentMotionComponent("PAIR_ENVELOPE")
        previous = poses[index]
        current = poses[index + 1]
        dt = float(current["timestamp_s"]) - float(
            previous["timestamp_s"]
        )
        angular = _angular_speed(
            np.asarray(previous["rotation_matrix"], dtype=np.float64),
            np.asarray(current["rotation_matrix"], dtype=np.float64),
            dt,
        )
        translation = float(
            np.linalg.norm(
                np.asarray(current["translation_m"], dtype=np.float64)
                - np.asarray(previous["translation_m"], dtype=np.float64)
            )
            / dt
        )
        _close(row.get("angular_speed_rad_s"), angular, "ANGULAR_SPEED")
        _close(
            row.get("translation_speed_m_s"),
            translation,
            "TRANSLATION_SPEED",
        )
        midpoint = 0.5 * (
            float(previous["timestamp_s"]) + float(current["timestamp_s"])
        )
        angular_values.append(angular)
        translation_values.append(translation)
        timestamps.append(midpoint)
        diagnostics = row.get("cell_diagnostics", {})
        raw_cells = diagnostics.get("raw_cells")
        compensated_cells = diagnostics.get("compensated_cells")
        if (
            not isinstance(raw_cells, list)
            or not isinstance(compensated_cells, list)
            or len(raw_cells) != 9
            or len(compensated_cells) != 9
        ):
            raise InvalidIndependentMotionComponent("CELL_SLOT_COUNT")
        common = []
        raw_common = []
        compensated_common = []
        for cell_index, (raw_cell, comp_cell) in enumerate(
            zip(raw_cells, compensated_cells, strict=True)
        ):
            for cell in (raw_cell, comp_cell):
                if cell.get("evaluable") is True:
                    expansion = cell.get("expansion")
                    if (
                        not isinstance(expansion, (int, float))
                        or not math.isfinite(float(expansion))
                    ):
                        raise InvalidIndependentMotionComponent(
                            "CELL_EXPANSION"
                        )
                elif cell.get("evaluable") is False:
                    if cell.get("expansion") is not None:
                        raise InvalidIndependentMotionComponent(
                            "CELL_ABSTENTION_EXPANSION"
                        )
                else:
                    raise InvalidIndependentMotionComponent(
                        "CELL_EVALUABLE"
                    )
                residual = cell.get("fit_residual_pixels_per_frame")
                if residual is not None:
                    if (
                        not isinstance(residual, (int, float))
                        or not math.isfinite(float(residual))
                        or float(residual) < 0.0
                    ):
                        raise InvalidIndependentMotionComponent(
                            "CELL_RESIDUAL"
                        )
            if (
                raw_cell["evaluable"] is True
                and comp_cell["evaluable"] is True
            ):
                common.append(cell_index)
                raw_common.append(float(raw_cell["expansion"]))
                comp_value = float(comp_cell["expansion"])
                compensated_common.append(comp_value)
                cell_values.append(comp_value)
                if cell_index == 4:
                    center_values.append(comp_value)
                else:
                    periphery_values.append(comp_value)
                residual = comp_cell.get("fit_residual_pixels_per_frame")
                if residual is not None:
                    fit_residuals.append(float(residual))
        if diagnostics.get("common_cell_indices") != common:
            raise InvalidIndependentMotionComponent("COMMON_CELL_INDICES")
        if row.get("common_cell_count") != len(common):
            raise InvalidIndependentMotionComponent("COMMON_CELL_COUNT")
        evaluable = len(common) >= 5
        if row.get("evaluable") is not evaluable:
            raise InvalidIndependentMotionComponent("PAIR_EVALUABLE")
        if evaluable:
            signed_value = float(statistics.median(compensated_common))
            raw_value = float(statistics.median(raw_common))
            absolute_value = float(
                statistics.median(abs(value) for value in compensated_common)
            )
            raw_absolute_value = float(
                statistics.median(abs(value) for value in raw_common)
            )
            _close(
                row.get("compensated_expansion_median_per_s"),
                signed_value,
                "PAIR_SIGNED",
            )
            _close(
                row.get("raw_expansion_median_per_s"),
                raw_value,
                "PAIR_RAW_SIGNED",
            )
            _close(
                row.get("compensated_abs_expansion_median_per_s"),
                absolute_value,
                "PAIR_ABSOLUTE",
            )
            _close(
                row.get("raw_abs_expansion_median_per_s"),
                raw_absolute_value,
                "PAIR_RAW_ABSOLUTE",
            )
            signed.append(signed_value)
            raw_signed.append(raw_value)
            pair_absolute.append(absolute_value)
            raw_pair_absolute.append(raw_absolute_value)
            evaluable_count += 1
            positive = signed_value > THRESHOLD
            positive_count += int(positive)
            streak = streak + 1 if positive else 0
            longest = max(longest, streak)
            pair_sign = (
                1 if signed_value > 0.0 else -1 if signed_value < 0.0 else 0
            )
            same_sign.append(
                sum(
                    (
                        1
                        if value > 0.0
                        else -1
                        if value < 0.0
                        else 0
                    )
                    == pair_sign
                    for value in compensated_common
                )
                / len(compensated_common)
            )
            evaluable_timestamps.append(midpoint)
            evaluable_signed.append(signed_value)
            correlation_response.append(abs(signed_value))
            correlation_angular.append(angular)
            correlation_translation.append(translation)
        else:
            if row.get("compensated_expansion_median_per_s") is not None:
                raise InvalidIndependentMotionComponent(
                    "ABSTENTION_RESPONSE"
                )
            streak = 0
        trigger = streak >= 3
        if row.get("compensated_three_pair_trigger") is not trigger:
            raise InvalidIndependentMotionComponent("TRIGGER")
        trigger_count += int(trigger)
        common_counts.append(float(len(common)))
        raw_tracks.append(float(row["raw_track_count"]))
        compensated_tracks.append(float(row["compensated_track_count"]))
    angular_frequency = _dominant_frequency(angular_values, timestamps)
    translation_frequency = _dominant_frequency(
        translation_values, timestamps
    )
    center_absolute = [abs(value) for value in center_values]
    periphery_absolute = [abs(value) for value in periphery_values]
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
            [abs(value) for value in cell_values], 0.50
        ),
        "compensated_cell_absolute_p90": _quantile(
            [abs(value) for value in cell_values], 0.90
        ),
        "center_cell_absolute_p50": _quantile(center_absolute, 0.50),
        "periphery_cell_absolute_p50": _quantile(
            periphery_absolute, 0.50
        ),
        "same_sign_spatial_fraction_p50": _quantile(same_sign, 0.50),
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
            evaluable_signed, evaluable_timestamps, angular_frequency
        ),
        "signed_response_at_translation_frequency_r2": _sinusoid_r2(
            evaluable_signed,
            evaluable_timestamps,
            translation_frequency,
        ),
    }


def _compare_summary(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    if set(observed) != set(expected):
        raise InvalidIndependentMotionComponent("SUMMARY_KEYSET")
    for key, value in expected.items():
        if value is None:
            if observed[key] is not None:
                raise InvalidIndependentMotionComponent(f"SUMMARY:{key}")
        elif isinstance(value, float):
            _close(observed[key], value, f"SUMMARY:{key}")
        elif observed[key] != value:
            raise InvalidIndependentMotionComponent(f"SUMMARY:{key}")


def validate_bundle(
    root: Path,
    bundle: Path,
    identity_path: Path,
    activation_path: Path,
    expected: dict[str, dict[str, Any]],
    stage_number: int,
    decision_path: Path | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if stage_number == 1 and (bundle.parent / "stage2").exists():
        raise InvalidIndependentMotionComponent(
            "STAGE_2_OUTPUT_PRESENT_BEFORE_ROUTING"
        )
    run = load_json(bundle / "run_receipt.json")
    contract_path = root / load_json(identity_path)["contract"]["path"]
    expected_bindings = {
        "contract_sha256": sha256_file(contract_path),
        "identity_lock_sha256": sha256_file(identity_path),
        "qms_r1_operator_source_sha256": sha256_file(root / OPERATOR_SOURCE),
        "qms_r1_operator_lock_sha256": sha256_file(root / OPERATOR_LOCK),
        "r3_transport_source_sha256": sha256_file(root / TRANSPORT_SOURCE),
        "r3_transport_lock_sha256": sha256_file(root / TRANSPORT_LOCK),
        "analysis_lock_sha256": sha256_file(root / ANALYSIS_LOCK),
        "generator_source_sha256": sha256_file(root / GENERATOR_SOURCE),
        "generator_lock_sha256": sha256_file(root / GENERATOR_LOCK),
        "runner_source_sha256": sha256_file(root / RUNNER_SOURCE),
    }
    if (
        run.get("task_id") != TASK_ID
        or run.get("stage_number") != stage_number
        or run.get("terminal")
        != (
            f"STAGE_{stage_number}_EXECUTION_COMPLETE / "
            "INDEPENDENT_VALIDATION_REQUIRED"
        )
        or run.get("identity_lock_sha256") != sha256_file(identity_path)
        or run.get("bindings") != expected_bindings
        or run.get("counts")
        != {
            "clusters": 4,
            "sequences": 16,
            "frames": 9632,
            "pairs": 9616,
        }
        or run.get("formal_execution_authorized_by_this_run") is not False
    ):
        raise InvalidIndependentMotionComponent("RUN_RECEIPT")
    if stage_number == 1:
        if (
            run.get("stage_1_activation_sha256")
            != sha256_file(activation_path)
            or run.get("stage_2_decision_sha256") is not None
        ):
            raise InvalidIndependentMotionComponent("STAGE_1_RUN_BINDING")
    else:
        if (
            decision_path is None
            or run.get("stage_1_activation_sha256") is not None
            or run.get("stage_2_decision_sha256")
            != sha256_file(decision_path)
        ):
            raise InvalidIndependentMotionComponent("STAGE_2_RUN_BINDING")
        validate_stage_1_decision(root, decision_path, identity_path)
    resource = run.get("resource", {})
    if (
        resource.get("available_ram_at_launch_bytes", 0) < 8 * GIB
        or resource.get("minimum_available_ram_bytes", 0) < 4 * GIB
        or resource.get("swap_in_delta") != 0
        or resource.get("swap_out_delta") != 0
        or resource.get("heartbeat_max_interval_seconds", 31) > 30
        or resource.get("residual_worker_pids") != []
    ):
        raise InvalidIndependentMotionComponent("RESOURCE")
    firewall = run.get("formal_firewall", {})
    before = firewall.get("before")
    after = firewall.get("after")
    predecessor = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "p4_formal"
    )
    previous_dev = root / PREVIOUS_DEV_BUNDLE
    successor = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1_successor_formal"
    )
    expected_firewall = {
        "predecessor_formal_tree_sha256": tree_sha256(predecessor),
        "previous_dev_tree_sha256": tree_sha256(previous_dev),
        "formal_activation_decision_sha256": sha256_file(
            root / FORMAL_DECISION
        ),
        "successor_formal_identity_lock_sha256": sha256_file(
            root / SUCCESSOR_FORMAL_LOCK
        ),
    }
    if (
        before != expected_firewall
        or after != expected_firewall
        or successor.exists()
        or firewall.get("successor_formal_path_absent") is not True
        or firewall.get("formal_sequences_run") != 0
        or firewall.get("formal_r3_pair_core_calls") != 0
        or firewall.get("formal_authority_consumed") is not False
    ):
        raise InvalidIndependentMotionComponent("FORMAL_FIREWALL")
    if any((bundle / "staging").glob("*")):
        raise InvalidIndependentMotionComponent("STAGING_REMAINS")
    clusters = run.get("clusters")
    if not isinstance(clusters, list) or len(clusters) != 4:
        raise InvalidIndependentMotionComponent("CLUSTERS")
    trajectories = load_json(root / TRAJECTORY_MANIFEST)
    summaries: dict[str, dict[str, Any]] = {}
    expected_stage = {
        key: value
        for key, value in expected.items()
        if value["stage_number"] == stage_number
    }
    for cluster in clusters:
        relative = cluster.get("cluster_receipt_path", "")
        path = (bundle / relative).resolve()
        try:
            path.relative_to(bundle.resolve())
        except ValueError as error:
            raise InvalidIndependentMotionComponent(
                "CLUSTER_PATH_OUTSIDE"
            ) from error
        if (
            not path.is_file()
            or sha256_file(path) != cluster.get("cluster_receipt_sha256")
        ):
            raise InvalidIndependentMotionComponent("CLUSTER_RECEIPT")
        disk_cluster = load_json(path)
        if (
            disk_cluster.get("qms_render_pair_calls") != 1807
            or disk_cluster.get("expected_qms_render_pair_calls") != 1807
            or disk_cluster.get("sequence_count") != 4
            or disk_cluster.get("terminal")
            != "MOTION_COMPONENT_CLUSTER_COMPLETE"
        ):
            raise InvalidIndependentMotionComponent("CLUSTER_EXECUTION")
        outputs = disk_cluster.get("arm_outputs")
        if not isinstance(outputs, list) or len(outputs) != 4:
            raise InvalidIndependentMotionComponent("ARM_OUTPUTS")
        for output in outputs:
            sequence_id = output.get("sequence_id")
            identity = expected_stage.get(sequence_id)
            if identity is None:
                raise InvalidIndependentMotionComponent("ARM_IDENTITY")
            receipt_path = (bundle / output.get("receipt_path", "")).resolve()
            try:
                receipt_path.relative_to(bundle.resolve())
            except ValueError as error:
                raise InvalidIndependentMotionComponent(
                    "ARM_PATH_OUTSIDE"
                ) from error
            if (
                not receipt_path.is_file()
                or sha256_file(receipt_path)
                != output.get("receipt_sha256")
            ):
                raise InvalidIndependentMotionComponent("ARM_RECEIPT")
            receipt = load_json(receipt_path)
            for key, value in identity.items():
                if receipt.get(key) != value:
                    raise InvalidIndependentMotionComponent(
                        f"ARM_RECEIPT_IDENTITY:{key}"
                    )
            if (
                receipt.get("bindings") != expected_bindings
                or receipt.get("r3_source_unchanged") is not True
                or receipt.get("cell_capture_hook_return_values_unchanged")
                is not True
                or receipt.get("terminal") != "MOTION_COMPONENT_ARM_COMPLETE"
            ):
                raise InvalidIndependentMotionComponent("ARM_BINDINGS")
            arm_dir = receipt_path.parent
            ledger_path = arm_dir / "pair_ledger.jsonl"
            summary_path = arm_dir / "reduced_metrics.json"
            if (
                receipt.get("pair_ledger_sha256")
                != sha256_file(ledger_path)
                or receipt.get("reduced_metrics_sha256")
                != sha256_file(summary_path)
            ):
                raise InvalidIndependentMotionComponent("ARM_HASHES")
            poses = _poses(
                trajectories[identity["block"]], identity["motion"]
            )
            summary = reduce_rows(
                load_jsonl(ledger_path), identity, poses
            )
            _compare_summary(load_json(summary_path), summary)
            summaries[sequence_id] = summary
    if set(summaries) != set(expected_stage):
        raise InvalidIndependentMotionComponent("STAGE_KEYSET")
    return run, summaries


def analyze_stage(
    expected: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    stage_number: int,
) -> dict[str, Any]:
    units = []
    for block in BLOCKS:
        identities = [
            item
            for item in expected.values()
            if item["block"] == block
            and item["stage_number"] == stage_number
        ]
        by_motion = {
            item["motion"]: summaries[item["sequence_id"]]
            for item in identities
        }
        if set(by_motion) != set(MOTIONS):
            raise InvalidIndependentMotionComponent("ANALYSIS_ARM_SET")
        rotation = (
            by_motion["ROTATION_ONLY"]["compensated_absolute_p90"]
            - by_motion["STATIC"]["compensated_absolute_p90"]
        )
        translation = (
            by_motion["TRANSLATION_ONLY"]["compensated_signed_p90"]
            - by_motion["ROTATION_ONLY"]["compensated_signed_p90"]
        )
        full = (
            by_motion["FULL_6DOF"]["compensated_signed_p90"]
            - max(
                by_motion["ROTATION_ONLY"]["compensated_signed_p90"],
                by_motion["TRANSLATION_ONLY"]["compensated_signed_p90"],
            )
        )
        units.append(
            {
                "block": block,
                "stage_number": stage_number,
                "cluster_id": identities[0]["cluster_id"],
                "arm_metrics": by_motion,
                "routing_contrasts": {
                    "ROTATION_MINUS_STATIC": rotation,
                    "TRANSLATION_MINUS_ROTATION": translation,
                    "FULL_MINUS_MAX_SINGLE": full,
                },
            }
        )
    directions = {}
    for name in (
        "ROTATION_MINUS_STATIC",
        "TRANSLATION_MINUS_ROTATION",
        "FULL_MINUS_MAX_SINGLE",
    ):
        values = [unit["routing_contrasts"][name] for unit in units]
        directions[name] = {
            "values_by_block": {
                unit["block"]: unit["routing_contrasts"][name]
                for unit in units
            },
            "positive_count": sum(value > 0.0 for value in values),
            "negative_count": sum(value < 0.0 for value in values),
            "zero_count": sum(value == 0.0 for value in values),
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
            "opens_stage_2": (
                stage_number == 1
                and sum(value > 0.0 for value in values) >= 3
            ),
        }
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.motion_component_analysis.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "stage_number": stage_number,
        "terminal": (
            f"STAGE_{stage_number}_ROUTING_COMPLETE / DESCRIPTIVE_ONLY"
        ),
        "units": units,
        "routing_direction_summary": directions,
        "analysis_controls": {
            "pair_pooled_inference": False,
            "p_values": False,
            "confidence_intervals": False,
            "bootstrap": False,
            "max_t": False,
            "formal_classification": False,
        },
        "claim_ceiling": "CONTROLLED_GENERATOR_INTERNAL_ROUTING_AUDIT_ONLY",
        "formal_authority": "UNCHANGED_NOT_CONSUMED",
    }


def validate_all(
    root: Path,
    contract_path: Path,
    identity_path: Path,
    activation_path: Path,
    bundle: Path,
    stage_number: int,
    decision_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_contract(root, contract_path)
    lock, expected = validate_identity_lock(
        root, identity_path, contract_path
    )
    validate_stage_1_activation(
        root, activation_path, contract_path, identity_path, lock
    )
    run, summaries = validate_bundle(
        root,
        bundle,
        identity_path,
        activation_path,
        expected,
        stage_number,
        decision_path,
    )
    result = analyze_stage(expected, summaries, stage_number)
    receipt = {
        "schema": "rcle.periodic_self_motion_counterfactual.motion_component_independent_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "stage_number": stage_number,
        "validated": True,
        "errors": [],
        "terminal": f"VALID / STAGE_{stage_number}_ROUTING_COMPLETE",
        "identity_set_sha256": lock["identity_set_sha256"],
        "run_receipt_sha256": sha256_file(bundle / "run_receipt.json"),
        "analysis_result_sha256": hashlib.sha256(
            canonical_bytes(result)
        ).hexdigest(),
        "resource": run["resource"],
        "formal_firewall": run["formal_firewall"],
        "analysis_controls": result["analysis_controls"],
        "formal_authority": "UNCHANGED_NOT_CONSUMED",
        "formal_sequences_run": 0,
        "formal_r3_pair_core_calls": 0,
        "validator_source_sha256": sha256_file(root / VALIDATOR_SOURCE),
    }
    return result, receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--identity-lock", type=Path, required=True)
    parser.add_argument("--stage-1-activation", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage-number", type=int, choices=(1, 2), required=True)
    parser.add_argument("--stage-1-decision", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--routing-decision", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    contract_path = (root / args.contract).resolve()
    identity_path = (root / args.identity_lock).resolve()
    activation_path = (root / args.stage_1_activation).resolve()
    bundle = (root / args.bundle).resolve()
    decision_path = (
        (root / args.stage_1_decision).resolve()
        if args.stage_1_decision
        else None
    )
    if args.stage_number == 1 and decision_path is not None:
        raise InvalidIndependentMotionComponent("STAGE_1_DECISION_INPUT")
    if args.stage_number == 2 and decision_path is None:
        raise InvalidIndependentMotionComponent("STAGE_2_DECISION_INPUT")
    result, receipt = validate_all(
        root,
        contract_path,
        identity_path,
        activation_path,
        bundle,
        args.stage_number,
        decision_path,
    )
    result_path = (root / args.result).resolve()
    receipt_path = (root / args.receipt).resolve()
    write_exclusive(result_path, result)
    receipt["analysis_result_path"] = result_path.relative_to(root).as_posix()
    receipt["analysis_result_file_sha256"] = sha256_file(result_path)
    write_exclusive(receipt_path, receipt)
    routing = None
    if args.stage_number == 1:
        if args.routing_decision is None:
            raise InvalidIndependentMotionComponent(
                "ROUTING_DECISION_OUTPUT_REQUIRED"
            )
        summaries = result["routing_direction_summary"]
        opened_by = [
            name
            for name, value in summaries.items()
            if value["opens_stage_2"] is True
        ]
        decision = "OPEN_STAGE_2" if opened_by else "KEEP_STAGE_2_SEALED"
        routing = {
            "schema": "rcle.periodic_self_motion_counterfactual.motion_component_routing_decision.v1",
            "protocol_id": PROTOCOL_ID,
            "task_id": TASK_ID,
            "decision": decision,
            "stage_2_ordinal": 1,
            "opened_by_contrasts": opened_by,
            "routing_direction_summary": summaries,
            "contract_sha256": sha256_file(contract_path),
            "identity_lock_sha256": sha256_file(identity_path),
            "stage_1_activation_sha256": sha256_file(activation_path),
            "stage_1_run_receipt_sha256": sha256_file(
                bundle / "run_receipt.json"
            ),
            "stage_1_run_receipt_path": (
                bundle / "run_receipt.json"
            ).relative_to(root).as_posix(),
            "stage_1_analysis_result_sha256": sha256_file(result_path),
            "stage_1_analysis_result_path": result_path.relative_to(
                root
            ).as_posix(),
            "stage_1_independent_receipt_sha256": sha256_file(receipt_path),
            "stage_1_independent_receipt_path": receipt_path.relative_to(
                root
            ).as_posix(),
            "validator_source_path": VALIDATOR_SOURCE,
            "validator_source_sha256": sha256_file(root / VALIDATOR_SOURCE),
            "formal_sequences_run": 0,
            "formal_r3_pair_core_calls": 0,
            "formal_authority_consumed": False,
            "terminal": (
                "STAGE_2_OPENED_BY_FROZEN_ROUTING"
                if decision == "OPEN_STAGE_2"
                else "STAGE_2_REMAINS_SEALED"
            ),
        }
        routing_path = (root / args.routing_decision).resolve()
        write_exclusive(routing_path, routing)
    print(
        json.dumps(
            {
                "terminal": receipt["terminal"],
                "stage_number": args.stage_number,
                "routing_decision": (
                    routing["decision"] if routing is not None else None
                ),
                "formal_sequences_run": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
