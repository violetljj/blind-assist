"""Four-block QMS-R1 RCLE development diagnostic over 48 fresh sequences.

The observational unit is one of eight scene×motion-block clusters.  Each
cluster contains the frozen six paired arms and retains all 601 ordered pairs.
This module is development-only and cannot consume successor formal identities.
"""

from __future__ import annotations

import os

# Install the measured W8 numeric-thread mode before importing NumPy/OpenCV.
from . import p3_runtime_preflight_r0 as guarded

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any
from unittest import mock

import numpy as np
import psutil

from ..rgb_algorithm_development_canary_cid_sims_r0 import producer as r3
from ..temporal_structure_diagnostic_r1 import extract as diagnostic
from . import generator_geometry as geometry
from . import material_residual_contraction_r1 as qms
from . import p3_transport_r0 as transport
from . import quality_interventions_r0 as quality


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_FOUR_BLOCK_DEV_DIAGNOSTIC_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
QUALITIES = ("CLEAN", "BLUR", "LOW_TEXTURE")
ARMS = tuple(f"{motion}__{quality}" for motion in MOTIONS for quality in QUALITIES)
FRAME_COUNT = 602
PAIR_COUNT = 601
WORKERS = 8
NATIVE_THREADS = 18
GIB = 1024**3
EXCLUSION_SOURCES = (
    (
        "OLD_FORMAL",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "P4_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json",
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
)
TRAJECTORY_MANIFEST = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
)
OPERATOR_PATH = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/material_residual_contraction_r1.py"
)
OPERATOR_LOCK_PATH = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_OPERATOR_LOCK_R0_2026-07-29.json"
)
FORMAL_DECISION_PATH = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_SUCCESSOR_FORMAL_ACTIVATION_DECISION_R0_2026-07-29.json"
)
SUCCESSOR_FORMAL_LOCK_PATH = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_SUCCESSOR_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json"
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


class InvalidDevDiagnostic(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


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


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


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


def _seed(block: str, ordinal: int) -> dict[str, Any]:
    if block not in BLOCKS or ordinal not in (0, 1):
        raise InvalidDevDiagnostic("SEED_COORDINATE")
    token = f"{TASK_ID}|DEV_DIAGNOSTIC|{block}|MAIN|{ordinal:02d}"
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
        return geometry.build_scene(block, ordinal, "MAIN")


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


def build_contract(root: Path) -> dict[str, Any]:
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_four_block_dev_contract.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "stage": "DEVELOPMENT_DIAGNOSTIC",
        "research_question": (
            "Do unchanged R3 responses show stable descriptive motion, quality, "
            "or interaction direction across four motion blocks under QMS-R1?"
        ),
        "design": {
            "blocks": list(BLOCKS),
            "new_scene_seeds_per_block": 2,
            "latent_cluster_count": 8,
            "arms": list(ARMS),
            "sequence_count": 48,
            "frame_count_per_sequence": FRAME_COUNT,
            "pair_count_per_sequence": PAIR_COUNT,
            "observational_unit": "scene_seed_x_motion_block_cluster",
            "repeated_measure": "six paired arms within cluster",
        },
        "outcomes": {
            "primary_descriptive": [
                "trigger_density_fixed_601_denominator",
                "evaluable_fraction",
                "response_median_among_evaluable",
                "quality_failure_union_density",
            ],
            "threshold_operator": "strict_greater_than",
            "threshold_per_s": 0.01,
            "required_consecutive_pairs": 3,
            "abstention_resets_streak": True,
        },
        "contrasts": [
            {
                "name": "MOTION_CLEAN",
                "formula": "PERIODIC_CLEAN-STATIC_CLEAN",
            },
            {
                "name": "BLUR_STATIC",
                "formula": "STATIC_BLUR-STATIC_CLEAN",
            },
            {
                "name": "LOW_TEXTURE_STATIC",
                "formula": "STATIC_LOW_TEXTURE-STATIC_CLEAN",
            },
            {
                "name": "MOTION_X_BLUR",
                "formula": (
                    "(PERIODIC_BLUR-PERIODIC_CLEAN)-"
                    "(STATIC_BLUR-STATIC_CLEAN)"
                ),
            },
            {
                "name": "MOTION_X_LOW_TEXTURE",
                "formula": (
                    "(PERIODIC_LOW_TEXTURE-PERIODIC_CLEAN)-"
                    "(STATIC_LOW_TEXTURE-STATIC_CLEAN)"
                ),
            },
        ],
        "reporting": {
            "cluster_contrasts": True,
            "overall_mean_median_range_and_sign_count": True,
            "block_two_cluster_descriptives": True,
            "bootstrap_or_confidence_interval": False,
            "p_values": False,
            "multiplicity_claim": False,
            "formal_max_t": False,
        },
        "stop_rule": (
            "RUN_EXACT_48_ONCE; NO_IDENTITY_REPLACEMENT, RETUNING, "
            "THRESHOLD_CHANGE, OR POST_HOC_EXTRA_SEEDS"
        ),
        "claim_ceiling": (
            "CONTROLLED_GENERATOR_INTERNAL_DEVELOPMENT_DIAGNOSTIC_ONLY"
        ),
        "bindings": {
            "qms_r1_operator_source_sha256": sha256_file(
                root / OPERATOR_PATH
            ),
            "qms_r1_operator_lock_sha256": sha256_file(
                root / OPERATOR_LOCK_PATH
            ),
            "r3_transport_lock_sha256": sha256_file(
                root
                / "docs/research/rcle/"
                "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
                "R3_TRANSPORT_EQUIVALENCE_LOCK_R0_2026-07-29.json"
            ),
            "r3_transport_source_sha256": sha256_file(
                root
                / "scripts/research/egomotion_compensated_looming/"
                "periodic_self_motion_counterfactual_r2/p3_transport_r0.py"
            ),
            "formal_analysis_lock_sha256": sha256_file(
                root
                / "docs/research/rcle/"
                "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
                "ANALYSIS_IMPLEMENTATION_LOCK_R0_2026-07-29.json"
            ),
            "formal_activation_decision_sha256": sha256_file(
                root / FORMAL_DECISION_PATH
            ),
        },
        "formal_identity_execution": False,
        "formal_identity_lock_read_for_exclusion_only": True,
        "formal_execution_authorized_by_this_contract": False,
        "formal_sequences_run": 0,
        "terminal": "DEV_DIAGNOSTIC_CONTRACT_FROZEN / NOT_RUN",
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
    old_preflight_scene_hashes = set()
    for relative in PREFLIGHT_RECEIPT_DIRS:
        directory = root / relative
        receipts = list(directory.glob("*/receipt.json"))
        if not receipts:
            raise InvalidDevDiagnostic("PREFLIGHT_RECEIPTS_MISSING")
        directory_scenes = set()
        for receipt in receipts:
            directory_scenes.update(
                _collect(load_json(receipt), "scene_geometry_sha256")
            )
        old_preflight_scene_hashes.update(directory_scenes)
        sources.append(
            {
                "label": "PREFLIGHT_RUNTIME_SCENE_RECEIPTS",
                "path": relative,
                "tree_sha256": tree_sha256(directory),
                "receipt_count": len(receipts),
                "scene_geometry_sha256": sorted(directory_scenes),
            }
        )
    if len(old_preflight_scene_hashes) != 4:
        raise InvalidDevDiagnostic("OLD_PREFLIGHT_SCENE_IDENTITY")
    excluded["scene_geometry_sha256"].update(old_preflight_scene_hashes)
    trajectories = load_json(root / TRAJECTORY_MANIFEST)
    seeds = []
    identities = []
    for block in BLOCKS:
        trajectory = trajectories[block]
        for ordinal in (0, 1):
            seed = _seed(block, ordinal)
            scene = _scene(block, ordinal)
            cluster_id = f"QMSR1_DEV_{block}_MAIN_{ordinal:02d}"
            latent = {
                **seed,
                "cluster_id": cluster_id,
                "scene_geometry_sha256": scene["scene_geometry_sha256"],
            }
            for field, old_values in excluded.items():
                if latent.get(field) in old_values:
                    raise InvalidDevDiagnostic(f"IDENTITY_OVERLAP:{field}")
            seeds.append({"block": block, "ordinal": ordinal, **latent})
            for arm_ordinal, arm in enumerate(ARMS):
                motion = arm.split("__", 1)[0]
                identity = {
                    "sequence_id": (
                        f"QMSR1_DEV_{block}_MAIN_{ordinal:02d}__{arm}"
                    ),
                    "cluster_id": cluster_id,
                    "block": block,
                    "ordinal": ordinal,
                    "role": "DEV_DIAGNOSTIC",
                    "arm": arm,
                    "arm_ordinal": arm_ordinal,
                    "motion": motion,
                    "quality": arm.split("__", 1)[1],
                    **seed,
                    "scene_geometry_sha256": scene["scene_geometry_sha256"],
                    "trajectory_sha256": (
                        trajectory["periodic_pose_sha256"]
                        if motion == "PERIODIC_6DOF_SELF_MOTION"
                        else hashlib.sha256(
                            geometry.canonical_bytes({"static": FRAME_COUNT})
                        ).hexdigest()
                    ),
                    "frame_count": FRAME_COUNT,
                    "pair_count": PAIR_COUNT,
                }
                for field, old_values in excluded.items():
                    if identity.get(field) in old_values:
                        raise InvalidDevDiagnostic(f"IDENTITY_OVERLAP:{field}")
                identities.append(identity)
    payload = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_four_block_dev_identity_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "role": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "counts": {
            "blocks": 4,
            "seeds_per_block": 2,
            "clusters": 8,
            "arms_per_cluster": 6,
            "sequences": 48,
            "frames": 48 * FRAME_COUNT,
            "pairs": 48 * PAIR_COUNT,
        },
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "sha256": sha256_file(contract_path),
        },
        "exclusion_sources": sources,
        "zero_overlap_fields": list(fields),
        "seeds": seeds,
        "identities": identities,
        "formal_execution_authorized": False,
        "formal_sequences_run": 0,
        "terminal": "DEV_DIAGNOSTIC_IDENTITY_LOCK_VALID / NOT_RUN",
    }
    payload["identity_set_sha256"] = hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest()
    return payload


def _poses(trajectory: dict[str, Any], motion: str) -> list[dict[str, Any]]:
    if motion == "PERIODIC_6DOF_SELF_MOTION":
        return trajectory["poses"]
    timestamps = [item["timestamp_s"] for item in trajectory["poses"]]
    return [
        {
            "frame_index": index,
            "timestamp_s": timestamp,
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_matrix": np.eye(3).tolist(),
        }
        for index, timestamp in enumerate(timestamps)
    ]


def _response_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["compensated_expansion_median_per_s"])
        for row in rows
        if row["evaluable"] is True
    ]
    return {
        "evaluable_fraction": len(values) / PAIR_COUNT,
        "response_median_per_s": float(np.median(values)) if values else None,
        "response_mean_per_s": float(np.mean(values)) if values else None,
        "response_p90_per_s": (
            float(np.quantile(values, 0.90, method="linear"))
            if values
            else None
        ),
    }


def _reduce_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != PAIR_COUNT:
        raise InvalidDevDiagnostic("PAIR_COUNT")
    streak = trigger_count = failure_count = clean_count = evaluable_count = 0
    for index, row in enumerate(rows):
        if row.get("pair_index") != index:
            raise InvalidDevDiagnostic("PAIR_ORDER")
        evaluable = row.get("evaluable")
        response = row.get("compensated_expansion_median_per_s")
        if evaluable is True:
            response_value = float(response)
            if not np.isfinite(response_value):
                raise InvalidDevDiagnostic("RESPONSE_NONFINITE")
            evaluable_count += 1
            streak = streak + 1 if response_value > 0.01 else 0
        elif evaluable is False:
            if response is not None:
                raise InvalidDevDiagnostic("ABSTENTION_RESPONSE")
            streak = 0
        else:
            raise InvalidDevDiagnostic("EVALUABLE_FLAG")
        trigger = streak >= 3
        if row.get("compensated_three_pair_trigger") is not trigger:
            raise InvalidDevDiagnostic("TRIGGER_DRIFT")
        trigger_count += int(trigger)
        detected = row.get("detected_feature_count")
        consistent = row.get("forward_backward_consistent_count")
        fraction = row.get("forward_backward_consistent_fraction")
        occupied = row.get("occupied_3x3_cells")
        fb_error = row.get("median_forward_backward_error_px")
        if (
            not isinstance(detected, int)
            or not isinstance(consistent, int)
            or consistent > detected
            or not isinstance(occupied, int)
            or occupied not in range(10)
        ):
            raise InvalidDevDiagnostic("TRACKING_COUNTS")
        fraction_value = float(fraction)
        if not np.isfinite(fraction_value) or not 0.0 <= fraction_value <= 1.0:
            raise InvalidDevDiagnostic("TRACKING_FRACTION")
        if fb_error is not None:
            fb_error = float(fb_error)
            if not np.isfinite(fb_error) or fb_error < 0.0:
                raise InvalidDevDiagnostic("FB_ERROR")
        collapse = detected < 60 or consistent < 60 or fraction_value < 0.50
        fb_failure = fb_error is None or fb_error > 0.75
        failure_count += int(collapse or fb_failure)
        clean_count += int(
            not collapse and not fb_failure and occupied >= 5
        )
    return {
        "scheduled_pair_count": PAIR_COUNT,
        "evaluable_pair_count": evaluable_count,
        "trigger_count": trigger_count,
        "trigger_density": trigger_count / PAIR_COUNT,
        "quality_failure_union_count": failure_count,
        "quality_failure_union_density": failure_count / PAIR_COUNT,
        "clean_trackable_pair_count": clean_count,
        "clean_sequence_trackable": clean_count >= 421,
    }


def _cluster_worker(task: dict[str, Any]) -> dict[str, Any]:
    guarded._initialize_worker()
    thread_guard = guarded._native_thread_guard()
    if not guarded._native_thread_guard_valid(thread_guard):
        raise InvalidDevDiagnostic("NATIVE_THREAD_GUARD")
    block = task["block"]
    ordinal = task["ordinal"]
    identities = task["identities"]
    output_root = Path(task["output_root"])
    cluster_id = identities[0]["cluster_id"]
    staging = (
        output_root
        / "staging"
        / f"{cluster_id}.{os.getpid()}.tmp"
    )
    final = output_root / "clusters" / cluster_id
    if staging.exists() or final.exists():
        raise InvalidDevDiagnostic("CLUSTER_OUTPUT_EXISTS")
    staging.mkdir(parents=True, exist_ok=False)
    scene = _scene(block, ordinal)
    if any(
        item["numeric_seed_uint64"] != scene["numeric_seed_uint64"]
        or item["scene_geometry_sha256"] != scene["scene_geometry_sha256"]
        for item in identities
    ):
        raise InvalidDevDiagnostic("CLUSTER_SCENE_IDENTITY")
    trajectory = load_json(repo_root() / TRAJECTORY_MANIFEST)[block]
    protocol = load_json(repo_root() / transport.PROTOCOL_RELATIVE)
    started = time.perf_counter()
    arm_outputs = []
    for motion in MOTIONS:
        motion_identities = [
            item for item in identities if item["motion"] == motion
        ]
        poses = _poses(trajectory, motion)
        states = {item["arm"]: r3.PairState() for item in motion_identities}
        previous: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
        rows_by_arm = {item["arm"]: [] for item in motion_identities}
        streaks = {item["arm"]: 0 for item in motion_identities}
        static_triplet: dict[str, np.ndarray] | None = None
        static_mask: np.ndarray | None = None
        for frame_index, pose in enumerate(poses):
            rotation = np.asarray(pose["rotation_matrix"], dtype=np.float64)
            translation = np.asarray(pose["translation_m"], dtype=np.float64)
            if motion == "STATIC_CAMERA" and static_triplet is not None:
                triplet = static_triplet
                mask = static_mask
            else:
                rendered = qms.render_pair(scene, rotation, translation)
                clean = rendered["rgb_pair"]["clean"]
                triplet = {
                    "CLEAN": clean,
                    "LOW_TEXTURE": rendered["rgb_pair"]["low"],
                    "BLUR": quality.apply_blur(clean, 0.475),
                }
                mask = rendered["valid_mask"]
                if motion == "STATIC_CAMERA":
                    static_triplet = triplet
                    static_mask = mask
            for identity in motion_identities:
                arm = identity["arm"]
                rgb = triplet[identity["quality"]]
                if arm in previous:
                    previous_rgb, previous_mask, previous_pose = previous[arm]
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
                        state=states[arm],
                    )
                    metrics = diagnostic.flow_direction_metrics(
                        transport.rgb_to_gray(previous_rgb),
                        transport.rgb_to_gray(rgb),
                        transport.valid_mask(
                            previous_mask, previous_rgb.shape[:2]
                        ),
                    )
                    row.update(metrics)
                    row["occupied_3x3_cells"] = row["occupied_grid_cells"]
                    response = row.get("compensated_expansion_median_per_s")
                    if row.get("evaluable") is True:
                        streaks[arm] = (
                            streaks[arm] + 1
                            if float(response) > 0.01
                            else 0
                        )
                    else:
                        streaks[arm] = 0
                        row.setdefault(
                            "compensated_expansion_median_per_s", None
                        )
                    row["compensated_three_pair_trigger"] = (
                        streaks[arm] >= 3
                    )
                    envelope = {
                        key: identity[key]
                        for key in (
                            "sequence_id",
                            "cluster_id",
                            "block",
                            "ordinal",
                            "role",
                            "arm",
                        )
                    }
                    envelope["pair_index"] = frame_index - 1
                    envelope.update(row)
                    rows_by_arm[arm].append(envelope)
                previous[arm] = (rgb, mask, pose)
        for identity in motion_identities:
            rows = rows_by_arm[identity["arm"]]
            if len(rows) != PAIR_COUNT:
                raise InvalidDevDiagnostic("PAIR_COUNT")
            arm_dir = staging / "arms" / identity["sequence_id"]
            arm_dir.mkdir(parents=True, exist_ok=False)
            ledger_path = arm_dir / "pair_ledger.jsonl"
            ledger_path.write_bytes(
                b"".join(canonical_bytes(row) + b"\n" for row in rows)
            )
            reduced = {
                **_reduce_pair_rows(rows),
                **_response_summary(rows),
            }
            reduced_path = arm_dir / "reduced_metrics.json"
            write_exclusive(reduced_path, reduced)
            receipt = {
                "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_dev_arm_receipt.v1",
                "task_id": TASK_ID,
                **{
                    key: identity[key]
                    for key in (
                        "sequence_id",
                        "cluster_id",
                        "block",
                        "ordinal",
                        "role",
                        "arm",
                        "numeric_seed_uint64",
                        "scene_geometry_sha256",
                        "trajectory_sha256",
                        "token",
                        "token_sha256",
                        "motion",
                        "quality",
                    )
                },
                "frame_count": FRAME_COUNT,
                "pair_count": PAIR_COUNT,
                "pair_ledger_sha256": sha256_file(ledger_path),
                "reduced_metrics_sha256": sha256_file(reduced_path),
                "bindings": task["bindings"],
                "thread_guard": thread_guard,
                "terminal": "DEV_ARM_COMPLETE",
            }
            receipt_path = arm_dir / "receipt.json"
            write_exclusive(receipt_path, receipt)
            arm_outputs.append(
                {
                    "sequence_id": identity["sequence_id"],
                    "receipt_path": (
                        Path("clusters")
                        / cluster_id
                        / "arms"
                        / identity["sequence_id"]
                        / "receipt.json"
                    ).as_posix(),
                    "receipt_sha256": sha256_file(receipt_path),
                }
            )
    cluster_receipt = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_dev_cluster_receipt.v1",
        "task_id": TASK_ID,
        "cluster_id": cluster_id,
        "block": block,
        "ordinal": ordinal,
        "sequence_count": 6,
        "arm_outputs": arm_outputs,
        "qms_render_pair_calls": 603,
        "static_render_pair_calls": 1,
        "static_render_reuse_hits": 601,
        "periodic_render_pair_calls": 602,
        "operator_source_sha256": sha256_file(repo_root() / OPERATOR_PATH),
        "wall_seconds": time.perf_counter() - started,
        "thread_guard": thread_guard,
        "terminal": "DEV_CLUSTER_COMPLETE",
    }
    write_exclusive(staging / "cluster_receipt.json", cluster_receipt)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    return {
        **cluster_receipt,
        "cluster_receipt_path": (
            Path("clusters") / cluster_id / "cluster_receipt.json"
        ).as_posix(),
        "cluster_receipt_sha256": sha256_file(
            final / "cluster_receipt.json"
        ),
    }


def run(identity_lock_path: Path, output_root: Path) -> dict[str, Any]:
    root = repo_root()
    lock = load_json(identity_lock_path)
    expected = build_identity_lock(
        root, root / lock["contract"]["path"]
    )
    if canonical_bytes(lock) != canonical_bytes(expected):
        raise InvalidDevDiagnostic("IDENTITY_LOCK_DRIFT")
    if output_root.exists():
        raise FileExistsError("OUTPUT_ROOT_EXISTS")
    memory = psutil.virtual_memory()
    if memory.available < 8 * GIB:
        raise InvalidDevDiagnostic("LAUNCH_AVAILABLE_RAM_BELOW_8_GIB")
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
    if not predecessor_formal.is_dir() or successor_formal.exists():
        raise InvalidDevDiagnostic("FORMAL_PATH_FIREWALL")
    predecessor_hash = tree_sha256(predecessor_formal)
    decision_hash = sha256_file(root / FORMAL_DECISION_PATH)
    successor_lock_hash = sha256_file(root / SUCCESSOR_FORMAL_LOCK_PATH)
    bindings = {
        "contract_sha256": sha256_file(root / lock["contract"]["path"]),
        "identity_lock_sha256": sha256_file(identity_lock_path),
        "operator_source_sha256": sha256_file(root / OPERATOR_PATH),
        "operator_lock_sha256": sha256_file(root / OPERATOR_LOCK_PATH),
        "r3_transport_lock_sha256": build_contract(root)["bindings"][
            "r3_transport_lock_sha256"
        ],
        "r3_transport_source_sha256": build_contract(root)["bindings"][
            "r3_transport_source_sha256"
        ],
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_swap = psutil.swap_memory()
    last_swap = started_swap
    paging_streak = 0
    minimum_available = memory.available
    completed: dict[str, dict[str, Any]] = {}
    samples = []
    identities = lock["identities"]
    tasks = []
    for block in BLOCKS:
        for ordinal in (0, 1):
            cluster = [
                item
                for item in identities
                if item["block"] == block and item["ordinal"] == ordinal
            ]
            tasks.append(
                {
                    "block": block,
                    "ordinal": ordinal,
                    "identities": cluster,
                    "output_root": str(output_root),
                    "bindings": bindings,
                }
            )
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
                raise InvalidDevDiagnostic("RUN_AVAILABLE_RAM_BELOW_4_GIB")
            if paging_streak >= 2:
                raise InvalidDevDiagnostic("SUSTAINED_PAGING")
            for future in done:
                task = futures.pop(future)
                result = future.result()
                completed[result["cluster_id"]] = result
            elapsed = time.perf_counter() - started
            sample = {
                "sample_index": len(samples),
                "elapsed_seconds": elapsed,
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
                    "completed_clusters": len(completed),
                    "total_clusters": 8,
                    "completed_sequences": len(completed) * 6,
                    "total_sequences": 48,
                    "status": "SUCCESS" if not futures else "RUNNING",
                    "last_heartbeat_utc": sample["sampled_at_utc"],
                },
            )
            write_json(
                output_root / "telemetry.json",
                {
                    "task_id": TASK_ID,
                    "samples": samples,
                    "scientific_response_fields_present": False,
                },
            )
    residual = [pid for pid in worker_pids if psutil.pid_exists(pid)]
    if residual:
        raise InvalidDevDiagnostic("RESIDUAL_WORKERS")
    if (
        tree_sha256(predecessor_formal) != predecessor_hash
        or successor_formal.exists()
        or sha256_file(root / FORMAL_DECISION_PATH) != decision_hash
        or sha256_file(root / SUCCESSOR_FORMAL_LOCK_PATH)
        != successor_lock_hash
    ):
        raise InvalidDevDiagnostic("FORMAL_FIREWALL_DRIFT")
    ordered = [
        completed[f"QMSR1_DEV_{block}_MAIN_{ordinal:02d}"]
        for block in BLOCKS
        for ordinal in (0, 1)
    ]
    result = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_four_block_dev_run.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "terminal": "DEV_DIAGNOSTIC_EXECUTION_COMPLETE / VALIDATION_REQUIRED",
        "identity_lock_sha256": sha256_file(identity_lock_path),
        "identity_set_sha256": lock["identity_set_sha256"],
        "bindings": bindings,
        "clusters": ordered,
        "counts": {
            "clusters": 8,
            "sequences": 48,
            "frames": 48 * FRAME_COUNT,
            "pairs": 48 * PAIR_COUNT,
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
        },
        "timing": {"wall_seconds": time.perf_counter() - started},
        "formal_firewall": {
            "predecessor_formal_tree_sha256_before": predecessor_hash,
            "predecessor_formal_tree_sha256_after": tree_sha256(
                predecessor_formal
            ),
            "successor_formal_path_absent": not successor_formal.exists(),
            "formal_activation_decision_sha256_before": decision_hash,
            "formal_activation_decision_sha256_after": sha256_file(
                root / FORMAL_DECISION_PATH
            ),
            "successor_formal_identity_lock_sha256_before": (
                successor_lock_hash
            ),
            "successor_formal_identity_lock_sha256_after": sha256_file(
                root / SUCCESSOR_FORMAL_LOCK_PATH
            ),
            "formal_sequences_run": 0,
            "successor_formal_authority_consumed": False,
        },
        "claim_ceiling": (
            "CONTROLLED_GENERATOR_INTERNAL_DEVELOPMENT_DIAGNOSTIC_ONLY"
        ),
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
    execute = sub.add_parser("run")
    execute.add_argument("--identity-lock", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        root = args.repo_root.resolve()
        contract_path = (root / args.contract).resolve()
        lock_path = (root / args.identity_lock).resolve()
        write_exclusive(contract_path, build_contract(root))
        write_exclusive(
            lock_path, build_identity_lock(root, contract_path)
        )
        result = load_json(lock_path)
    else:
        result = run(args.identity_lock.resolve(), args.output_root.resolve())
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "terminal": result["terminal"],
                "formal_sequences_run": result.get(
                    "formal_sequences_run",
                    result.get("formal_firewall", {}).get(
                        "formal_sequences_run", 0
                    ),
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
