"""Produce the response-blind R2 P2 quality calibration ledger and lock."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np

from . import generator_geometry as p1
from . import quality_interventions_r0 as quality


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
CALIBRATION_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "RESPONSE_BLIND_QUALITY_CALIBRATION_R0"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_CONTRACT_2026-07-28.json"
)
BUDGET_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_RUN_BUDGET_R0_2026-07-28.json"
)
P1_LOCK_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R2_KEYSET_REPAIR_R0_2026-07-29.json"
)
P1_EVIDENCE_ROOT = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0"
)
P1_RECEIPT_PATH = (
    P1_EVIDENCE_ROOT / "independent_geometry_validation_receipt.json"
)
TRAJECTORY_PATH = P1_EVIDENCE_ROOT / "trajectory_manifest.json"
IMPLEMENTATION_PATH = Path(__file__).resolve()
INTERVENTION_PATH = Path(quality.__file__).resolve()
P1_SOURCE_PATH = Path(p1.__file__).resolve()
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p2_quality_calibration_r0"
)
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_R0_STRENGTH_LOCK_2026-07-29.json"
)

EXPECTED_INPUT_SHA256 = {
    CONTRACT_PATH: "73705144d0d7a8162c0a47694676364e00d3b27917904f78a39642900aeac0c5",
    BUDGET_PATH: "4f0c3204e6ea5bcb7daf22b017b1819325bae47c20e7f170677d085a41708798",
    P1_LOCK_PATH: "a7fa41c0406908baf05805904111ba43fdbd8dd93b8c4e496706f1990438adc9",
    P1_RECEIPT_PATH: "95646437fbe0ef0cf03844f94467303f5d90ca15c3e22fc1785157b037a8c079",
    TRAJECTORY_PATH: "c394641b6419c7a58a58c1bc485e2783cd710e8ba2893415dd6687c20b7d2652",
    P1_SOURCE_PATH: "e6153f52f89674947f4960faf05ae8d5b90d1e9e69fef627a74c719127d6ca43",
}
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
CAL_ORDINALS = (0, 1, 2, 3)
FRAME_POSITIONS = (
    0,
    40,
    80,
    120,
    160,
    200,
    240,
    280,
    320,
    360,
    400,
    440,
    480,
    520,
    560,
    601,
)
BLUR_CANDIDATES = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
LOW_TEXTURE_CANDIDATES = (0.75, 0.60, 0.45, 0.30, 0.15)
BLUR_TARGET = (0.35, 0.55)
BLUR_RMS_MINIMUM = 0.70
LOW_TEXTURE_TARGET = (0.35, 0.55)
EDGE_SPREAD_TARGET = (0.90, 1.10)
EXPECTED_LEDGER_ROWS = 6144


class CalibrationIdentityError(ValueError):
    """Raised when the P2 claim cannot be signed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _read_json(
    path: Path,
    read_ledger: list[dict[str, str]],
) -> dict[str, Any]:
    expected = EXPECTED_INPUT_SHA256.get(path)
    if expected is None:
        raise CalibrationIdentityError(f"UNDECLARED_READ_PATH:{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise CalibrationIdentityError(f"INPUT_SHA256_MISMATCH:{path.name}")
    read_ledger.append(
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": actual,
        }
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_import_firewall(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden_fragments = (
        "ecological_response_discovery_r0",
        "rgb_algorithm_development_canary",
        "temporal_confirmation",
        "degradation_flow_quality_diagnostic",
        "cotracker",
        "android",
    )
    for module in imported:
        lower = module.lower()
        if any(fragment in lower for fragment in forbidden_fragments):
            forbidden.append(module)
    if forbidden:
        raise CalibrationIdentityError(
            "RCLE_FIREWALL_IMPORT:" + ",".join(sorted(forbidden))
        )
    try:
        display_path = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "imports": sorted(set(imported)),
        "forbidden_imports": [],
    }


def _verify_freeze(
    contract: dict[str, Any],
    budget: dict[str, Any],
    p1_lock: dict[str, Any],
    p1_receipt: dict[str, Any],
) -> None:
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise CalibrationIdentityError("CONTRACT_PROTOCOL_ID")
    if contract.get("formal_execution_authorized") is not False:
        raise CalibrationIdentityError("FORMAL_EXECUTION_AUTHORITY")
    quality_contract = contract["quality_interventions"]
    if tuple(quality_contract["calibration_seed_ordinals_per_block"]) != CAL_ORDINALS:
        raise CalibrationIdentityError("CAL_ORDINALS")
    if tuple(quality_contract["calibration_frame_positions"]) != FRAME_POSITIONS:
        raise CalibrationIdentityError("CAL_FRAME_POSITIONS")
    if tuple(quality_contract["blur"]["candidate_sigma_px"]) != BLUR_CANDIDATES:
        raise CalibrationIdentityError("BLUR_CANDIDATES")
    if tuple(quality_contract["low_texture"]["candidate_alpha"]) != LOW_TEXTURE_CANDIDATES:
        raise CalibrationIdentityError("LOW_TEXTURE_CANDIDATES")
    if tuple(contract["factorial_design"]["motion_blocks"]) != BLOCKS:
        raise CalibrationIdentityError("MOTION_BLOCKS")
    counts = budget["count_budget"]
    expected_counts = {
        "quality_calibration_blocks": 4,
        "quality_calibration_seeds_per_block": 4,
        "quality_calibration_motion_levels": 2,
        "quality_calibration_candidate_states_per_motion": 12,
        "quality_calibration_frames_per_arm": 16,
        "quality_calibration_candidate_image_evaluations": 6144,
        "quality_calibration_final_six_arm_panel_images": 1536,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise CalibrationIdentityError("RUN_BUDGET_CAL_COUNTS")
    if budget.get("formal_execution_authorized") is not False:
        raise CalibrationIdentityError("BUDGET_FORMAL_AUTHORITY")
    if p1_lock.get("quality_calibration_authorized") is not False:
        raise CalibrationIdentityError("P1_MUST_NOT_SELF_AUTHORIZE_P2")
    if p1_lock.get("formal_execution_authorized") is not False:
        raise CalibrationIdentityError("P1_FORMAL_AUTHORITY")
    if (
        p1_receipt.get("terminal") != "GENERATOR_GEOMETRY_PASS"
        or p1_receipt.get("status") != "VALID"
        or p1_receipt.get("automatic_p2_authority") is not False
        or p1_receipt.get("formal_execution_authorized") is not False
        or p1_receipt.get("rcle_output_accessed_or_executed") is not False
    ):
        raise CalibrationIdentityError("P1_RECEIPT_TERMINAL")


def _pose(
    trajectories: dict[str, Any],
    block: str,
    motion: str,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if motion == "STATIC_CAMERA":
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    item = trajectories[block]["poses"][frame_index]
    if item["frame_index"] != frame_index:
        raise CalibrationIdentityError("TRAJECTORY_FRAME_INDEX")
    return (
        np.asarray(item["rotation_matrix"], dtype=np.float64),
        np.asarray(item["translation_m"], dtype=np.float64),
    )


def _candidate_id(kind: str, strength: float) -> str:
    if kind == "BLUR":
        return f"BLUR_SIGMA_{strength:.2f}"
    return f"LOW_TEXTURE_ALPHA_{strength:.2f}"


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def _summarize_candidate(
    sequence_values: dict[tuple[str, int, str], dict[str, list[float]]],
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    sequence_medians: list[dict[str, Any]] = []
    for (block, ordinal, motion), metrics in sorted(sequence_values.items()):
        if any(len(metrics[name]) != len(FRAME_POSITIONS) for name in metric_names):
            raise CalibrationIdentityError("SEQUENCE_METRIC_FRAME_COUNT")
        medians = {
            name: quality.average_rank_median(metrics[name])
            for name in metric_names
        }
        sequence_medians.append(
            {
                "block": block,
                "cal_ordinal": ordinal,
                "motion": motion,
                "frame_count": len(FRAME_POSITIONS),
                "metrics": medians,
            }
        )
    if len(sequence_medians) != 32:
        raise CalibrationIdentityError("SEQUENCE_MEDIAN_COUNT")
    subgroup_medians: list[dict[str, Any]] = []
    for block in BLOCKS:
        for motion in MOTIONS:
            selected = [
                row["metrics"]
                for row in sequence_medians
                if row["block"] == block and row["motion"] == motion
            ]
            if len(selected) != 4:
                raise CalibrationIdentityError("SUBGROUP_SEED_COUNT")
            subgroup_medians.append(
                {
                    "block": block,
                    "motion": motion,
                    "sequence_count": len(selected),
                    "metrics": {
                        name: quality.average_rank_median(
                            [row[name] for row in selected]
                        )
                        for name in metric_names
                    },
                }
            )
    overall = {
        name: quality.average_rank_median(
            [row["metrics"][name] for row in sequence_medians]
        )
        for name in metric_names
    }
    return {
        "sequence_medians": sequence_medians,
        "subgroup_medians": subgroup_medians,
        "overall_median": overall,
    }


def _blur_pass(summary: dict[str, Any]) -> bool:
    values = [
        summary["overall_median"],
        *(row["metrics"] for row in summary["subgroup_medians"]),
    ]
    return all(
        BLUR_TARGET[0] <= row["laplacian_variance_ratio"] <= BLUR_TARGET[1]
        and row["local_rms_contrast_ratio"] >= BLUR_RMS_MINIMUM
        for row in values
    )


def _low_texture_pass(
    summary: dict[str, Any],
    fixture_ratio: float,
) -> bool:
    values = [
        summary["overall_median"],
        *(row["metrics"] for row in summary["subgroup_medians"]),
    ]
    return (
        all(
            LOW_TEXTURE_TARGET[0]
            <= row["multiscale_gradient_density_ratio"]
            <= LOW_TEXTURE_TARGET[1]
            and EDGE_SPREAD_TARGET[0]
            <= row["edge_spread_ratio"]
            <= EDGE_SPREAD_TARGET[1]
            for row in values
        )
        and EDGE_SPREAD_TARGET[0]
        <= fixture_ratio
        <= EDGE_SPREAD_TARGET[1]
    )


def _fixture_summaries() -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for alpha in LOW_TEXTURE_CANDIDATES:
        clean, degraded = quality.analytic_edge_fixture(alpha)
        clean_y = quality.linear_luminance(clean["rgb"])
        degraded_y = quality.linear_luminance(degraded["rgb"])
        clean_width, clean_count = quality.source_known_edge_spread(
            clean_y,
            clean["object_id"],
            clean["edges"],
        )
        degraded_width, degraded_count = quality.source_known_edge_spread(
            degraded_y,
            degraded["object_id"],
            degraded["edges"],
        )
        summaries[_candidate_id("LOW_TEXTURE", alpha)] = {
            "alpha": alpha,
            "clean_edge_spread_px": clean_width,
            "degraded_edge_spread_px": degraded_width,
            "edge_spread_ratio": quality.safe_ratio(
                degraded_width,
                clean_width,
                "ANALYTIC_FIXTURE_EDGE_SPREAD",
            ),
            "clean_valid_edge_count": clean_count,
            "degraded_valid_edge_count": degraded_count,
            "fixture_clean_rgb_sha256": quality.sha256_bytes(
                clean["rgb"].tobytes()
            ),
            "fixture_degraded_rgb_sha256": quality.sha256_bytes(
                degraded["rgb"].tobytes()
            ),
        }
    return summaries


def produce(output: Path, lock_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{output}")
    if lock_path.exists():
        raise FileExistsError(f"LOCK_ALREADY_EXISTS:{lock_path}")
    output.mkdir(parents=True, exist_ok=False)
    read_ledger: list[dict[str, str]] = []
    contract = _read_json(CONTRACT_PATH, read_ledger)
    budget = _read_json(BUDGET_PATH, read_ledger)
    p1_lock = _read_json(P1_LOCK_PATH, read_ledger)
    p1_receipt = _read_json(P1_RECEIPT_PATH, read_ledger)
    trajectories = _read_json(TRAJECTORY_PATH, read_ledger)
    if sha256_file(P1_SOURCE_PATH) != EXPECTED_INPUT_SHA256[P1_SOURCE_PATH]:
        raise CalibrationIdentityError("P1_SOURCE_SHA256")
    read_ledger.append(
        {
            "path": P1_SOURCE_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": EXPECTED_INPUT_SHA256[P1_SOURCE_PATH],
        }
    )
    _verify_freeze(contract, budget, p1_lock, p1_receipt)
    firewall_audit = [
        _validate_import_firewall(INTERVENTION_PATH),
        _validate_import_firewall(IMPLEMENTATION_PATH),
    ]
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260729)

    ledger_path = output / "response_blind_metric_ledger.jsonl"
    ledger_digest = hashlib.sha256()
    row_count = 0
    completed_frozen_frames = 0
    total_frozen_frames = 512
    manifest_identities: list[dict[str, Any]] = []
    blur_values: dict[
        str,
        dict[tuple[str, int, str], dict[str, list[float]]],
    ] = {
        _candidate_id("BLUR", strength): {}
        for strength in BLUR_CANDIDATES
    }
    low_values: dict[
        str,
        dict[tuple[str, int, str], dict[str, list[float]]],
    ] = {
        _candidate_id("LOW_TEXTURE", strength): {}
        for strength in LOW_TEXTURE_CANDIDATES
    }

    with ledger_path.open("xb") as ledger:
        for block in BLOCKS:
            if len(trajectories[block]["poses"]) != 602:
                raise CalibrationIdentityError("TRAJECTORY_POSE_COUNT")
            for ordinal in CAL_ORDINALS:
                base_scene = p1.build_scene(block, ordinal, "CAL")
                scene = quality.add_calibration_plate(base_scene)
                manifest_identities.append(
                    {
                        "block": block,
                        "cal_ordinal": ordinal,
                        "numeric_seed_uint64": scene["numeric_seed_uint64"],
                        "base_scene_geometry_sha256": scene[
                            "base_scene_geometry_sha256"
                        ],
                        "calibration_scene_sha256": scene[
                            "calibration_scene_sha256"
                        ],
                    }
                )
                for motion in MOTIONS:
                    sequence_key = (block, ordinal, motion)
                    for candidate in blur_values.values():
                        candidate[sequence_key] = {
                            "laplacian_variance_ratio": [],
                            "local_rms_contrast_ratio": [],
                        }
                    for candidate in low_values.values():
                        candidate[sequence_key] = {
                            "multiscale_gradient_density_ratio": [],
                            "edge_spread_ratio": [],
                        }
                    for frame_index in FRAME_POSITIONS:
                        rotation, translation = _pose(
                            trajectories,
                            block,
                            motion,
                            frame_index,
                        )
                        clean = quality.render_calibration_frame(
                            scene,
                            rotation,
                            translation,
                        )
                        prepared = quality.prepare_clean_frame_metrics(clean)
                        clean_hash = quality.sha256_bytes(
                            clean["rgb"].tobytes()
                        )
                        edge_manifest_hash = quality.sha256_bytes(
                            quality.canonical_bytes(clean["edges"])
                        )
                        common = {
                            "schema": (
                                "rcle.periodic_self_motion_counterfactual."
                                "response_blind_quality_metric_row.v1"
                            ),
                            "protocol_id": PROTOCOL_ID,
                            "calibration_id": CALIBRATION_ID,
                            "block": block,
                            "cal_ordinal": ordinal,
                            "numeric_seed_uint64": scene[
                                "numeric_seed_uint64"
                            ],
                            "motion": motion,
                            "frame_index": frame_index,
                            "base_scene_geometry_sha256": scene[
                                "base_scene_geometry_sha256"
                            ],
                            "calibration_scene_sha256": scene[
                                "calibration_scene_sha256"
                            ],
                            "valid_mask_sha256": clean["geometry_identity"][
                                "valid_mask_sha256"
                            ],
                            "object_id_sha256": clean["geometry_identity"][
                                "object_id_sha256"
                            ],
                            "source_known_edge_manifest_sha256": edge_manifest_hash,
                            "clean_rgb_sha256": clean_hash,
                        }
                        clean_row = {
                            **common,
                            "candidate_kind": "CLEAN",
                            "candidate_strength": None,
                            "degraded_rgb_sha256": clean_hash,
                            "metrics": {
                                "laplacian_variance": prepared[
                                    "laplacian_variance"
                                ],
                                "local_rms_contrast": prepared[
                                    "local_rms_contrast"
                                ],
                                "multiscale_gradient_density": prepared[
                                    "multiscale_gradient_density"
                                ],
                                "edge_spread_px": prepared["edge_spread_px"],
                                "valid_edge_count": prepared[
                                    "valid_edge_count"
                                ],
                            },
                        }
                        payload = quality.canonical_bytes(clean_row)
                        ledger.write(payload)
                        ledger_digest.update(payload)
                        row_count += 1

                        for sigma in BLUR_CANDIDATES:
                            candidate_id = _candidate_id("BLUR", sigma)
                            degraded = quality.apply_blur(clean["rgb"], sigma)
                            metrics = quality.blur_frame_metrics(
                                prepared,
                                degraded,
                            )
                            for name in (
                                "laplacian_variance_ratio",
                                "local_rms_contrast_ratio",
                            ):
                                blur_values[candidate_id][sequence_key][
                                    name
                                ].append(float(metrics[name]))
                            row = {
                                **common,
                                "candidate_kind": "BLUR",
                                "candidate_strength": sigma,
                                "degraded_rgb_sha256": quality.sha256_bytes(
                                    degraded.tobytes()
                                ),
                                "metrics": metrics,
                            }
                            if not _finite_tree(row):
                                raise quality.InvalidQualityMetric(
                                    "NONFINITE_BLUR_ROW"
                                )
                            payload = quality.canonical_bytes(row)
                            ledger.write(payload)
                            ledger_digest.update(payload)
                            row_count += 1

                        for alpha in LOW_TEXTURE_CANDIDATES:
                            candidate_id = _candidate_id(
                                "LOW_TEXTURE",
                                alpha,
                            )
                            degraded = quality.apply_low_texture(clean, alpha)
                            metrics = quality.low_texture_frame_metrics(
                                clean,
                                prepared,
                                degraded,
                            )
                            for name in (
                                "multiscale_gradient_density_ratio",
                                "edge_spread_ratio",
                            ):
                                low_values[candidate_id][sequence_key][
                                    name
                                ].append(float(metrics[name]))
                            row = {
                                **common,
                                "candidate_kind": "LOW_TEXTURE",
                                "candidate_strength": alpha,
                                "degraded_rgb_sha256": quality.sha256_bytes(
                                    degraded.tobytes()
                                ),
                                "metrics": metrics,
                                "psf_operator_applied": False,
                            }
                            if not _finite_tree(row):
                                raise quality.InvalidQualityMetric(
                                    "NONFINITE_LOW_TEXTURE_ROW"
                                )
                            payload = quality.canonical_bytes(row)
                            ledger.write(payload)
                            ledger_digest.update(payload)
                            row_count += 1
                        completed_frozen_frames += 1
                        if (
                            completed_frozen_frames % 16 == 0
                            or completed_frozen_frames == total_frozen_frames
                        ):
                            elapsed = time.perf_counter() - start
                            throughput = completed_frozen_frames / elapsed
                            eta = (
                                total_frozen_frames - completed_frozen_frames
                            ) / throughput
                            print(
                                json.dumps(
                                    {
                                        "protocol_id": PROTOCOL_ID,
                                        "phase": (
                                            "P2_RESPONSE_BLIND_QUALITY_"
                                            "CALIBRATION"
                                        ),
                                        "completed_units": (
                                            completed_frozen_frames
                                        ),
                                        "total_units": total_frozen_frames,
                                        "throughput": throughput,
                                        "eta_seconds": eta,
                                        "status": "RUNNING",
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
        ledger.flush()
        os.fsync(ledger.fileno())

    if row_count != EXPECTED_LEDGER_ROWS:
        raise CalibrationIdentityError("LEDGER_ROW_COUNT")
    fixture_summaries = _fixture_summaries()
    blur_summaries = {
        candidate_id: {
            "sigma_px": strength,
            **_summarize_candidate(
                blur_values[candidate_id],
                (
                    "laplacian_variance_ratio",
                    "local_rms_contrast_ratio",
                ),
            ),
        }
        for strength in BLUR_CANDIDATES
        for candidate_id in (_candidate_id("BLUR", strength),)
    }
    low_summaries = {
        candidate_id: {
            "alpha": strength,
            **_summarize_candidate(
                low_values[candidate_id],
                (
                    "multiscale_gradient_density_ratio",
                    "edge_spread_ratio",
                ),
            ),
            "analytic_fixture": fixture_summaries[candidate_id],
        }
        for strength in LOW_TEXTURE_CANDIDATES
        for candidate_id in (_candidate_id("LOW_TEXTURE", strength),)
    }
    for summary in blur_summaries.values():
        summary["passes_all_gates"] = _blur_pass(summary)
    for summary in low_summaries.values():
        summary["passes_all_gates"] = _low_texture_pass(
            summary,
            summary["analytic_fixture"]["edge_spread_ratio"],
        )
    selected_blur = next(
        (
            strength
            for strength in BLUR_CANDIDATES
            if blur_summaries[_candidate_id("BLUR", strength)][
                "passes_all_gates"
            ]
        ),
        None,
    )
    selected_low = next(
        (
            strength
            for strength in LOW_TEXTURE_CANDIDATES
            if low_summaries[_candidate_id("LOW_TEXTURE", strength)][
                "passes_all_gates"
            ]
        ),
        None,
    )
    if selected_blur is not None and selected_low is not None:
        scientific_status = "QUALITY_CALIBRATION_PASS"
        execution_authority = "P3_NOT_AUTHORIZED"
    else:
        scientific_status = "NO_GLOBAL_QUALITY_STRENGTH"
        execution_authority = "HOLD_P2"
    manifest_sha256 = quality.sha256_bytes(
        quality.canonical_bytes(manifest_identities)
    )
    lock = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "response_blind_quality_strength_lock.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "calibration_id": CALIBRATION_ID,
        "date": "2026-07-29",
        "phase": "P2_RESPONSE_BLIND_QUALITY_CALIBRATION",
        "scientific_status": scientific_status,
        "protocol_status": "VALID",
        "execution_authority": execution_authority,
        "selected_global_strengths": {
            "blur_sigma_px": selected_blur,
            "low_texture_alpha": selected_low,
        },
        "selection_policy": {
            "blur": (
                "smallest frozen sigma passing overall and all eight "
                "block_x_motion subgroups"
            ),
            "low_texture": (
                "largest frozen alpha passing overall, all eight "
                "block_x_motion subgroups, and analytic fixture"
            ),
            "no_global_value": (
                "do not expand grid, replace seed, or fit block-specific "
                "strengths"
            ),
        },
        "frozen_grid_and_gates": {
            "blur_sigma_px": list(BLUR_CANDIDATES),
            "laplacian_variance_ratio_inclusive": list(BLUR_TARGET),
            "local_rms_contrast_ratio_minimum": BLUR_RMS_MINIMUM,
            "low_texture_alpha": list(LOW_TEXTURE_CANDIDATES),
            "multiscale_gradient_density_ratio_inclusive": list(
                LOW_TEXTURE_TARGET
            ),
            "cal_fixture_edge_spread_ratio_inclusive": list(
                EDGE_SPREAD_TARGET
            ),
        },
        "calibration_panel": {
            "blocks": list(BLOCKS),
            "cal_ordinals_per_block": list(CAL_ORDINALS),
            "motions": list(MOTIONS),
            "frame_positions": list(FRAME_POSITIONS),
            "sequence_count": 32,
            "candidate_state_count_per_sequence": 12,
            "candidate_image_evaluations": row_count,
            "calibration_scene_manifest_sha256": manifest_sha256,
            "manifest": manifest_identities,
        },
        "candidate_summaries": {
            "blur": blur_summaries,
            "low_texture": low_summaries,
        },
        "evidence": {
            "response_blind_metric_ledger": ledger_path.relative_to(
                REPO_ROOT
            ).as_posix(),
            "response_blind_metric_ledger_sha256": ledger_digest.hexdigest(),
            "response_blind_metric_ledger_row_count": row_count,
        },
        "implementation_identity": {
            "quality_interventions_source": INTERVENTION_PATH.relative_to(
                REPO_ROOT
            ).as_posix(),
            "quality_interventions_sha256": sha256_file(INTERVENTION_PATH),
            "producer_source": IMPLEMENTATION_PATH.relative_to(
                REPO_ROOT
            ).as_posix(),
            "producer_sha256": sha256_file(IMPLEMENTATION_PATH),
            "p1_generator_source": P1_SOURCE_PATH.relative_to(
                REPO_ROOT
            ).as_posix(),
            "p1_generator_sha256": sha256_file(P1_SOURCE_PATH),
            "low_texture_psf_operator": "NONE",
        },
        "input_read_ledger": read_ledger,
        "firewall": {
            "source_import_audit": firewall_audit,
            "algorithm_output_read_or_run": False,
            "strength_selected_from_algorithm_output": False,
            "p3_preflight_run": False,
            "formal_480_plus_16_sequences_run": False,
            "r3_threshold_or_three_pair_modified": False,
            "sequence16_cotracker_android_realtime": False,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_threads": int(cv2.getNumThreads()),
            "wall_seconds": time.perf_counter() - start,
        },
        "independent_validation": "REQUIRED_NOT_YET_CREATED",
        "formal_execution_authorized": False,
        "p3_authorized": False,
    }
    if not _finite_tree(lock):
        raise quality.InvalidQualityMetric("NONFINITE_LOCK")
    lock_payload = quality.canonical_bytes(lock)
    _exclusive_write(lock_path, lock_payload)
    producer_receipt = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "response_blind_quality_producer_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "calibration_id": CALIBRATION_ID,
        "scientific_status": scientific_status,
        "protocol_status": "VALID",
        "execution_authority": execution_authority,
        "ledger_sha256": ledger_digest.hexdigest(),
        "lock_path": lock_path.relative_to(REPO_ROOT).as_posix(),
        "lock_sha256": hashlib.sha256(lock_payload).hexdigest(),
        "row_count": row_count,
        "algorithm_output_read_or_run": False,
        "p3_or_formal_execution_run": False,
    }
    return producer_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = produce(args.output.resolve(), args.lock.resolve())
    except (
        CalibrationIdentityError,
        quality.InvalidQualityMetric,
        FileExistsError,
        KeyError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "scientific_status": "CLAIM_NOT_SIGNABLE",
                    "protocol_status": "INVALID",
                    "execution_authority": "HOLD_P2",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
