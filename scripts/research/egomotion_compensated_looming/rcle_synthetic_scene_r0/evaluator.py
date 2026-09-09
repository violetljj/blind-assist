from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_low_reference_false_trigger_r1.temporal_confirmation import (
    THRESHOLD,
    apply_confirmation,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    LOCK_PATH as PHASE_A_LOCK_PATH,
    PROTOCOL_PATH as PHASE_A_PROTOCOL_PATH,
    PROTOCOL_SHA256 as PHASE_A_PROTOCOL_SHA256,
    validate_protocol as validate_phase_a_protocol,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.rotation_compensation import (
    compensate_current_to_previous,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.local_expansion import (
    fit_fixed_grid_local_affine,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.sparse_flow import (
    detect_fixed_grid_features,
    track_features,
)

from .io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .truth import evaluate_truth_pair


def _load_frozen_phase_a_protocol() -> dict[str, Any]:
    """Load the frozen protocol while neutralizing checkout-only CRLF changes."""
    raw = PHASE_A_PROTOCOL_PATH.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    if hashlib.sha256(normalized).hexdigest() != PHASE_A_PROTOCOL_SHA256:
        raise ValueError("PHASE_A_PROTOCOL_HASH_MISMATCH")
    lock = json.loads(PHASE_A_LOCK_PATH.read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != PHASE_A_PROTOCOL_SHA256:
        raise ValueError("PHASE_A_PROTOCOL_LOCK_HASH_MISMATCH")
    protocol = json.loads(normalized.decode("utf-8"))
    validate_phase_a_protocol(protocol)
    return protocol


def _load_gray(root: Path, row: dict[str, Any]) -> np.ndarray:
    image = cv2.imread(str(root / row["rgb_path"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"RGB_READ:{row['rgb_path']}")
    return np.ascontiguousarray(image)


def _load_valid_mask(root: Path, row: dict[str, Any]) -> np.ndarray:
    mask = cv2.imread(
        str(root / row["valid_mask_path"]), cv2.IMREAD_GRAYSCALE
    )
    if mask is None:
        raise ValueError(f"VALID_MASK_READ:{row['valid_mask_path']}")
    return np.ascontiguousarray(np.where(mask > 0, 255, 0).astype(np.uint8))


def _common_expansions(
    raw_cells: list[Any], compensated_cells: list[Any]
) -> tuple[list[float], list[float], list[int]]:
    raw_by_index = {
        index: cell
        for index, cell in enumerate(raw_cells)
        if cell.evaluable and cell.expansion is not None
    }
    compensated_by_index = {
        index: cell
        for index, cell in enumerate(compensated_cells)
        if cell.evaluable and cell.expansion is not None
    }
    common = sorted(raw_by_index.keys() & compensated_by_index.keys())
    return (
        [float(raw_by_index[index].expansion) for index in common],
        [
            float(compensated_by_index[index].expansion)
            for index in common
        ],
        common,
    )


def _evaluate_algorithm_pair(
    root: Path,
    previous: dict[str, Any],
    current: dict[str, Any],
    phase_a_protocol: dict[str, Any],
) -> dict[str, Any]:
    previous_image = _load_gray(root, previous)
    current_image = _load_gray(root, current)
    previous_valid = _load_valid_mask(root, previous)
    current_valid = _load_valid_mask(root, current)
    dt = (
        float(current["timestamp"]["seconds"])
        - float(previous["timestamp"]["seconds"])
    )
    if not np.isfinite(dt) or dt <= 0:
        return {"evaluable": False, "reason": "NON_POSITIVE_DT"}
    intrinsics = np.asarray(previous["intrinsics"], dtype=np.float64)
    rotation_current_from_previous = np.asarray(
        current["rotation_current_from_previous"], dtype=np.float64
    )
    rotation_h = (
        intrinsics
        @ rotation_current_from_previous
        @ np.linalg.inv(intrinsics)
    )
    compensation = compensate_current_to_previous(
        current_image, current_valid, previous_valid, rotation_h
    )
    if compensation.overlap_fraction < 0.75:
        return {
            "evaluable": False,
            "reason": "ROTATION_WARP_VALID_COVERAGE_BELOW_0_75",
            "warp_overlap_fraction": compensation.overlap_fraction,
        }
    lk = phase_a_protocol["sparse_lk"]
    affine = phase_a_protocol["local_affine"]
    initial = detect_fixed_grid_features(previous_image, previous_valid, lk)
    raw_tracks = track_features(
        previous_image, current_image, initial, current_valid, lk
    )
    compensated_tracks = track_features(
        previous_image,
        compensation.image,
        initial,
        compensation.valid_mask,
        lk,
    )
    raw_cells = fit_fixed_grid_local_affine(
        raw_tracks, dt, previous_image.shape, affine
    )
    compensated_cells = fit_fixed_grid_local_affine(
        compensated_tracks, dt, previous_image.shape, affine
    )
    raw_values, compensated_values, common = _common_expansions(
        raw_cells, compensated_cells
    )
    if len(common) < int(
        affine["minimum_common_evaluable_cells_per_pair"]
    ):
        return {
            "evaluable": False,
            "reason": "COMMON_GRID_SUPPORT_BELOW_5_OF_9",
            "common_cell_count": len(common),
            "raw_track_count": raw_tracks.valid_count,
            "compensated_track_count": compensated_tracks.valid_count,
            "warp_overlap_fraction": compensation.overlap_fraction,
        }
    compensated_median = float(np.median(compensated_values))
    return {
        "evaluable": True,
        "reason": None,
        "common_cell_count": len(common),
        "common_cell_indices": common,
        "raw_track_count": raw_tracks.valid_count,
        "compensated_track_count": compensated_tracks.valid_count,
        "warp_overlap_fraction": compensation.overlap_fraction,
        "raw_expansion_median_per_s": float(np.median(raw_values)),
        "raw_abs_expansion_median_per_s": float(
            np.median(np.abs(raw_values))
        ),
        "compensated_expansion_median_per_s": compensated_median,
        "compensated_abs_expansion_median_per_s": float(
            np.median(np.abs(compensated_values))
        ),
        "trigger": compensated_median > THRESHOLD,
    }


def _summarize(
    spec: dict[str, Any],
    split: str,
    truth_rows: list[dict[str, Any]],
    algorithm_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    truth_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comparison_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    algorithm_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth_rows:
        truth_by_sequence[row["sequence_id"]].append(row)
    for row in comparison_rows:
        comparison_by_sequence[row["window_id"]].append(row)
    for row in algorithm_rows:
        algorithm_by_sequence[row["sequence_id"]].append(row)
    gates = spec["development_gates"]
    sequences: list[dict[str, Any]] = []
    all_pass = True
    for sequence_id in sorted(truth_by_sequence):
        truth_selected = truth_by_sequence[sequence_id]
        algorithm_selected = comparison_by_sequence[sequence_id]
        raw_algorithm_selected = algorithm_by_sequence[sequence_id]
        if len(truth_selected) != 100 or len(algorithm_selected) != 100:
            raise ValueError("FIXED_PAIR_DENOMINATOR_DRIFT")
        truth_evaluable = [
            row for row in truth_selected if row["evaluable"] is True
        ]
        algorithm_evaluable = [
            row for row in algorithm_selected if row["evaluable"] is True
        ]
        raw_algorithm_evaluable = [
            row
            for row in raw_algorithm_selected
            if row["evaluable"] is True
        ]
        role = truth_selected[0]["role"]
        scene_id = truth_selected[0]["scene_id"]
        motion_id = truth_selected[0]["motion_id"]
        truth_median = (
            float(
                np.median(
                    [
                        row["compensated_expansion_median_per_s"]
                        for row in truth_evaluable
                    ]
                )
            )
            if truth_evaluable
            else None
        )
        evaluable_fraction = len(algorithm_evaluable) / len(
            algorithm_selected
        )
        algorithm_median = (
            float(
                np.median(
                    [
                        row["compensated_expansion_median_per_s"]
                        for row in raw_algorithm_evaluable
                    ]
                )
            )
            if raw_algorithm_evaluable
            else None
        )
        algorithm_truth_error = (
            abs(algorithm_median - truth_median)
            if algorithm_median is not None and truth_median is not None
            else None
        )
        old_count = sum(row["old_trigger"] for row in algorithm_selected)
        r1_count = sum(row["revised_trigger"] for row in algorithm_selected)
        retention = r1_count / old_count if old_count else None
        coverage = r1_count / len(algorithm_selected)
        old_first = next(
            (
                row["current_timestamp_s"]
                for row in algorithm_selected
                if row["old_trigger"]
            ),
            None,
        )
        r1_first = next(
            (
                row["current_timestamp_s"]
                for row in algorithm_selected
                if row["revised_trigger"]
            ),
            None,
        )
        delay = (
            float(r1_first) - float(old_first)
            if old_first is not None and r1_first is not None
            else None
        )
        checks = {
            "algorithm_evaluable_pair_fraction": (
                evaluable_fraction
                >= gates["minimum_sequence_evaluable_pair_fraction"]
            )
        }
        if role == "POSITIVE_APPROACH_WINDOW":
            checks.update(
                {
                    "positive_truth": (
                        truth_median is not None
                        and truth_median
                        >= gates[
                            "positive_truth_median_expansion_per_s_min"
                        ]
                    ),
                    "positive_algorithm_expansion": (
                        algorithm_median is not None
                        and algorithm_median
                        >= gates[
                            "positive_algorithm_compensated_expansion_median_per_s_min"
                        ]
                    ),
                    "positive_r1_coverage": (
                        coverage
                        >= gates["positive_r1_trigger_coverage_min"]
                    ),
                    "positive_r1_retention": (
                        retention is not None
                        and retention
                        >= gates["positive_r1_trigger_retention_min"]
                    ),
                    "algorithm_truth_error": (
                        algorithm_truth_error is not None
                        and algorithm_truth_error
                        <= gates[
                            "algorithm_truth_sequence_median_absolute_error_per_s_max"
                        ]
                    ),
                    "positive_delay": (
                        delay is not None
                        and delay
                        <= gates[
                            "maximum_positive_first_trigger_delay_s"
                        ]
                    ),
                }
            )
        elif role == "BELOW_TRIGGER_REFERENCE_WINDOW":
            checks.update(
                {
                    "below_truth": (
                        truth_median is not None
                        and abs(truth_median)
                        <= gates[
                            "below_truth_abs_median_expansion_per_s_max"
                        ]
                    ),
                    "below_r1_coverage": (
                        coverage
                        <= gates["below_r1_trigger_coverage_max"]
                    ),
                    "algorithm_truth_error": (
                        algorithm_truth_error is not None
                        and algorithm_truth_error
                        <= gates[
                            "algorithm_truth_sequence_median_absolute_error_per_s_max"
                        ]
                    ),
                }
            )
            if motion_id == "below_pure_yaw":
                raw_abs = float(
                    np.median(
                        [
                            row["raw_abs_expansion_median_per_s"]
                            for row in raw_algorithm_evaluable
                        ]
                    )
                )
                compensated_abs = float(
                    np.median(
                        [
                            row[
                                "compensated_abs_expansion_median_per_s"
                            ]
                            for row in raw_algorithm_evaluable
                        ]
                    )
                )
                checks.update(
                    {
                        "pure_yaw_compensated_abs": (
                            compensated_abs
                            <= gates[
                                "below_pure_yaw_compensated_abs_expansion_median_per_s_max"
                            ]
                        ),
                        "pure_yaw_suppression": (
                            raw_abs - compensated_abs
                            >= gates[
                                "below_pure_yaw_raw_minus_compensated_abs_expansion_median_per_s_min"
                            ]
                        ),
                    }
                )
        sequence_pass = all(checks.values())
        all_pass &= sequence_pass
        sequences.append(
            {
                "sequence_id": sequence_id,
                "scene_id": scene_id,
                "motion_id": motion_id,
                "role": role,
                "truth_evaluable_pair_count": len(truth_evaluable),
                "algorithm_evaluable_pair_count": len(algorithm_evaluable),
                "algorithm_evaluable_pair_fraction": evaluable_fraction,
                "truth_compensated_expansion_median_per_s": truth_median,
                "algorithm_compensated_expansion_median_per_s": algorithm_median,
                "algorithm_truth_sequence_median_absolute_error_per_s": (
                    algorithm_truth_error
                ),
                "old_trigger_count": old_count,
                "r1_trigger_count": r1_count,
                "r1_trigger_coverage": coverage,
                "r1_trigger_retention": retention,
                "first_trigger_delay_s": delay,
                "checks": checks,
                "pass": sequence_pass,
            }
        )
    by_scene_motion = {
        (row["scene_id"], row["motion_id"]): row for row in sequences
    }
    rotation_approach_checks: list[dict[str, Any]] = []
    for scene in spec["splits"][split]["scene_families"]:
        scene_id = scene["scene_id"]
        plain = by_scene_motion[(scene_id, "positive_approach")]
        rotated = by_scene_motion[
            (scene_id, "positive_rotation_approach")
        ]
        plain_value = plain[
            "algorithm_compensated_expansion_median_per_s"
        ]
        rotated_value = rotated[
            "algorithm_compensated_expansion_median_per_s"
        ]
        difference = (
            abs(float(rotated_value) - float(plain_value))
            if rotated_value is not None and plain_value is not None
            else None
        )
        passed = (
            difference is not None
            and difference
            <= gates[
                "positive_rotation_approach_vs_positive_approach_algorithm_median_absolute_difference_per_s_max"
            ]
        )
        all_pass &= passed
        rotation_approach_checks.append(
            {
                "scene_id": scene_id,
                "absolute_difference_per_s": difference,
                "pass": passed,
            }
        )
    return {
        "schema": "rcle.synthetic_scene.scientific_summary.v1",
        "protocol_id": spec["protocol_id"],
        "split": split,
        "sequence_count": len(sequences),
        "sequences": sequences,
        "positive_rotation_approach_comparisons": rotation_approach_checks,
        "all_local_gates_pass": all_pass,
        "scientific_outcome": (
            f"SYNTHETIC_MECHANISM_{split.upper()}_PASS"
            if all_pass
            else f"SYNTHETIC_MECHANISM_{split.upper()}_REVISE"
        ),
        "claim_ceiling": spec["authority"]["maximum_claim"],
        "pooled_rescue_used": False,
    }


def evaluate_algorithm_ledger(dataset_root: Path) -> list[dict[str, Any]]:
    """Build the algorithm ledger from the strict staging contract only."""
    cv2.setNumThreads(1)
    dataset_root = dataset_root.resolve()
    algorithm_inputs = read_jsonl(
        dataset_root / "algorithm_input_manifest.jsonl"
    )
    algorithm_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in algorithm_inputs:
        forbidden = {
            "depth_npy_path",
            "depth_png_path",
            "surface_id_path",
            "t_world_camera",
            "t_camera_world",
            "role",
            "motion_id",
            "scene_id",
            "translation_world_m_per_s",
        }
        if forbidden & row.keys():
            raise ValueError("ALGORITHM_INPUT_FIREWALL")
        allowed = {
            "schema",
            "protocol_id",
            "split",
            "sequence_id",
            "frame_index",
            "timestamp",
            "rgb_path",
            "rgb_sha256",
            "valid_mask_path",
            "valid_mask_sha256",
            "intrinsics",
            "rotation_current_from_previous",
            "algorithm_seed",
        }
        if set(row) != allowed:
            raise ValueError("ALGORITHM_INPUT_ALLOWLIST_DRIFT")
        for field in ("rgb_path", "valid_mask_path"):
            path = (dataset_root / row[field]).resolve()
            staging = (dataset_root / "algorithm_staging").resolve()
            if not path.is_relative_to(staging):
                raise ValueError("ALGORITHM_STAGING_PATH_ESCAPE")
            if not path.is_file():
                raise ValueError(f"ALGORITHM_STAGING_FILE_MISSING:{field}")
        if sha256_file(dataset_root / row["rgb_path"]) != row["rgb_sha256"]:
            raise ValueError("ALGORITHM_STAGING_RGB_HASH_MISMATCH")
        if (
            sha256_file(dataset_root / row["valid_mask_path"])
            != row["valid_mask_sha256"]
        ):
            raise ValueError("ALGORITHM_STAGING_VALID_MASK_HASH_MISMATCH")
        algorithm_by_sequence[row["sequence_id"]].append(row)
    phase_a_protocol = _load_frozen_phase_a_protocol()
    algorithm_rows: list[dict[str, Any]] = []
    for sequence_id in sorted(algorithm_by_sequence):
        selected = sorted(
            algorithm_by_sequence[sequence_id],
            key=lambda row: row["frame_index"],
        )
        for pair_index, (previous, current) in enumerate(
            zip(selected, selected[1:])
        ):
            cv2.setRNGSeed(int(previous["algorithm_seed"]))
            base = {
                "window_id": sequence_id,
                "sequence_id": sequence_id,
                "pair_index": pair_index,
                "previous_timestamp_s": previous["timestamp"]["seconds"],
                "current_timestamp_s": current["timestamp"]["seconds"],
                "dt_s": (
                    current["timestamp"]["seconds"]
                    - previous["timestamp"]["seconds"]
                ),
                "trigger": False,
            }
            algorithm_rows.append(
                {
                    **base,
                    **_evaluate_algorithm_pair(
                        dataset_root,
                        previous,
                        current,
                        phase_a_protocol,
                    ),
                }
            )
    return algorithm_rows


def evaluate_dataset(
    dataset_root: Path, *, sealed_execution_authorized: bool = False
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    dataset_root = dataset_root.resolve()
    spec = read_json(dataset_root / "dataset_spec.json")
    receipt = read_json(dataset_root / "receipt.json")
    split = receipt.get("split")
    if split not in spec["splits"]:
        raise ValueError("DATASET_SPLIT_INVALID")
    if split == "development" and spec["splits"]["development"]["execution"] != "OPEN":
        raise ValueError("DEVELOPMENT_NOT_OPEN")
    if split == "sealed" and receipt.get("status") != (
        "CANDIDATE_DATASET_GENERATED_VALIDATION_REQUIRED"
    ):
        raise ValueError("SEALED_DATASET_RECEIPT_INVALID")
    if split == "sealed" and not sealed_execution_authorized:
        raise PermissionError(
            "SEALED_EVALUATION_REQUIRES_GOVERNED_EXECUTOR"
        )
    algorithm_rows = evaluate_algorithm_ledger(dataset_root)
    algorithm_sha = write_jsonl(
        dataset_root / "algorithm_pair_ledger.jsonl", algorithm_rows
    )

    frames = read_jsonl(dataset_root / "frame_manifest.jsonl")
    truth_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        truth_by_sequence[row["sequence_id"]].append(row)
    truth_rows: list[dict[str, Any]] = []
    role_by_sequence: dict[str, str] = {}
    metadata_by_sequence: dict[str, tuple[str, str]] = {}
    for sequence_id in sorted(truth_by_sequence):
        selected = sorted(
            truth_by_sequence[sequence_id],
            key=lambda row: row["frame_index"],
        )
        role_by_sequence[sequence_id] = selected[0]["role"]
        metadata_by_sequence[sequence_id] = (
            selected[0]["scene_id"],
            selected[0]["motion_id"],
        )
        for pair_index, (previous, current) in enumerate(
            zip(selected, selected[1:])
        ):
            truth_rows.append(
                {
                    "window_id": sequence_id,
                    "sequence_id": sequence_id,
                    "scene_id": previous["scene_id"],
                    "motion_id": previous["motion_id"],
                    "role": previous["role"],
                    "pair_index": pair_index,
                    "previous_timestamp_s": previous["timestamp"]["seconds"],
                    "current_timestamp_s": current["timestamp"]["seconds"],
                    "dt_s": (
                        current["timestamp"]["seconds"]
                        - previous["timestamp"]["seconds"]
                    ),
                    **evaluate_truth_pair(
                        dataset_root, previous, current
                    ),
                }
            )
    truth_sha = write_jsonl(
        dataset_root / "truth_pair_ledger.jsonl", truth_rows
    )
    joined_for_comparison = [
        {
            **row,
            "role": role_by_sequence[row["sequence_id"]],
            "scene_id": metadata_by_sequence[row["sequence_id"]][0],
            "motion_id": metadata_by_sequence[row["sequence_id"]][1],
        }
        for row in algorithm_rows
    ]
    comparison_rows = apply_confirmation(joined_for_comparison)
    comparison_sha = write_jsonl(
        dataset_root / "old_r1_comparison.jsonl", comparison_rows
    )
    summary = _summarize(
        spec, split, truth_rows, algorithm_rows, comparison_rows
    )
    summary["ledger_sha256"] = {
        "truth": truth_sha,
        "algorithm": algorithm_sha,
        "old_r1_comparison": comparison_sha,
    }
    write_json(dataset_root / "scientific_summary.json", summary)
    return summary
