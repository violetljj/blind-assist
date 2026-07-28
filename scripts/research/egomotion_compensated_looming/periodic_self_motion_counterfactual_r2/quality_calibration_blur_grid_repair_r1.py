"""One-shot response-blind small-sigma repair for R2 P2."""

from __future__ import annotations

import argparse
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
from . import quality_calibration_r0 as r0
from . import quality_interventions_r0 as quality


PROTOCOL_ID = r0.PROTOCOL_ID
REPAIR_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1"
)
REPO_ROOT = r0.REPO_ROOT
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_CONTRACT_2026-07-29.json"
)
R0_LOCK_PATH = r0.DEFAULT_LOCK
R0_RECEIPT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_R0_INDEPENDENT_VALIDATION_RECEIPT_2026-07-29.json"
)
TRAJECTORY_PATH = r0.TRAJECTORY_PATH
P1_SOURCE_PATH = Path(p1.__file__).resolve()
QUALITY_SOURCE_PATH = Path(quality.__file__).resolve()
R0_PRODUCER_PATH = Path(r0.__file__).resolve()
IMPLEMENTATION_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p2_quality_calibration_blur_grid_repair_r1"
)
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_GLOBAL_STRENGTH_LOCK_2026-07-29.json"
)
EXPECTED_SHA256 = {
    CONTRACT_PATH: "ac2c9fa9b499d60492d01d542e7401bca4058cd5c786870804a1ebfff3a845ca",
    R0_LOCK_PATH: "d04c06d544c6780e8b86e4eb32b3c181ffe3940626e1f4127fa9cbf497dd41ea",
    R0_RECEIPT_PATH: "37d2f09d0e8764aee904ebc7998d2b629e1cba9dde024e739fbe4d1ea667fd06",
    TRAJECTORY_PATH: "c394641b6419c7a58a58c1bc485e2783cd710e8ba2893415dd6687c20b7d2652",
    P1_SOURCE_PATH: "e6153f52f89674947f4960faf05ae8d5b90d1e9e69fef627a74c719127d6ca43",
    QUALITY_SOURCE_PATH: "750bcdced7c92f96946f9c3d28ed11042edc8b43b8911f817e80772a66c01a5b",
    R0_PRODUCER_PATH: "4c6469131242756dbbf654cd017779e37f89c5677bf0e14290d252d6e8dbfb64",
}
SIGMA_CANDIDATES = (
    0.35,
    0.40,
    0.425,
    0.45,
    0.475,
    0.50,
    0.55,
    0.60,
    0.65,
)
EXPECTED_ROWS = 5120
INHERITED_ALPHA = 0.15
R0_LEDGER_SHA256 = (
    "0356a85d0901426a5ed7142d348863501f6d1d9f14fd1c76520c15ee2f9d0c3a"
)


class RepairIdentityError(ValueError):
    pass


def _read_json(path: Path, reads: list[dict[str, str]]) -> dict[str, Any]:
    expected = EXPECTED_SHA256.get(path)
    if expected is None:
        raise RepairIdentityError(f"UNDECLARED_READ:{path}")
    actual = r0.sha256_file(path)
    if actual != expected:
        raise RepairIdentityError(f"INPUT_SHA256:{path.name}")
    reads.append(
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": actual,
        }
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _record_source(path: Path, reads: list[dict[str, str]]) -> None:
    expected = EXPECTED_SHA256[path]
    actual = r0.sha256_file(path)
    if actual != expected:
        raise RepairIdentityError(f"SOURCE_SHA256:{path.name}")
    reads.append(
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": actual,
        }
    )


def _verify_contract(
    contract: dict[str, Any],
    r0_lock: dict[str, Any],
    r0_receipt: dict[str, Any],
) -> None:
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise RepairIdentityError("CONTRACT_PROTOCOL")
    if contract.get("repair_id") != REPAIR_ID:
        raise RepairIdentityError("CONTRACT_REPAIR_ID")
    authorization = contract["authorization"]
    if (
        authorization.get("user_authorized_once") is not True
        or authorization.get("allowed_now") is not True
        or authorization.get("automatic_second_repair") is not False
        or authorization.get("p3_authorized") is not False
        or authorization.get("formal_execution_authorized") is not False
    ):
        raise RepairIdentityError("CONTRACT_AUTHORITY")
    frozen = contract["frozen_repair"]
    if tuple(frozen["candidate_sigma_px"]) != SIGMA_CANDIDATES:
        raise RepairIdentityError("SIGMA_GRID")
    panel = frozen["calibration_panel"]
    if (
        tuple(panel["blocks"]) != r0.BLOCKS
        or tuple(panel["cal_ordinals_per_block"]) != r0.CAL_ORDINALS
        or tuple(panel["motions"]) != r0.MOTIONS
        or tuple(panel["frame_positions"]) != r0.FRAME_POSITIONS
        or panel["candidate_image_evaluations"] != EXPECTED_ROWS
    ):
        raise RepairIdentityError("CAL_PANEL")
    if (
        r0_lock.get("scientific_status") != "NO_GLOBAL_QUALITY_STRENGTH"
        or r0_lock.get("protocol_status") != "VALID"
        or r0_lock.get("execution_authority") != "HOLD_P2"
        or r0_lock.get("selected_global_strengths", {}).get(
            "low_texture_alpha"
        )
        != INHERITED_ALPHA
        or r0_lock.get("selected_global_strengths", {}).get(
            "blur_sigma_px"
        )
        is not None
    ):
        raise RepairIdentityError("R0_LOCK_TERMINAL")
    if (
        r0_receipt.get("validated") is not True
        or r0_receipt.get("errors") != []
        or r0_receipt.get("scientific_status")
        != "NO_GLOBAL_QUALITY_STRENGTH"
        or r0_receipt.get("protocol_status") != "VALID"
        or r0_receipt.get("execution_authority") != "HOLD_P2"
        or r0_receipt.get("selected_global_strengths", {}).get(
            "low_texture_alpha"
        )
        != INHERITED_ALPHA
    ):
        raise RepairIdentityError("R0_RECEIPT_TERMINAL")
    if (
        contract["predecessor"].get("r0_metric_ledger_sha256")
        != R0_LEDGER_SHA256
        or r0_lock.get("evidence", {}).get(
            "response_blind_metric_ledger_sha256"
        )
        != R0_LEDGER_SHA256
        or r0_receipt.get("evidence_sha256", {}).get(
            "response_blind_metric_ledger"
        )
        != R0_LEDGER_SHA256
    ):
        raise RepairIdentityError("R0_LEDGER_INHERITANCE")


def _manifest_map(r0_lock: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    items = r0_lock["calibration_panel"]["manifest"]
    result = {
        (item["block"], item["cal_ordinal"]): item
        for item in items
    }
    if len(result) != 16:
        raise RepairIdentityError("R0_MANIFEST_COUNT")
    return result


def produce(output: Path, lock_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"OUTPUT_ALREADY_EXISTS:{output}")
    if lock_path.exists():
        raise FileExistsError(f"LOCK_ALREADY_EXISTS:{lock_path}")
    output.mkdir(parents=True, exist_ok=False)
    reads: list[dict[str, str]] = []
    contract = _read_json(CONTRACT_PATH, reads)
    r0_lock = _read_json(R0_LOCK_PATH, reads)
    r0_receipt = _read_json(R0_RECEIPT_PATH, reads)
    trajectories = _read_json(TRAJECTORY_PATH, reads)
    for source in (P1_SOURCE_PATH, QUALITY_SOURCE_PATH, R0_PRODUCER_PATH):
        _record_source(source, reads)
    _verify_contract(contract, r0_lock, r0_receipt)
    predecessor_manifest = _manifest_map(r0_lock)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260729)

    ledger_path = output / "response_blind_blur_metric_ledger.jsonl"
    digest = hashlib.sha256()
    row_count = 0
    completed = 0
    manifest: list[dict[str, Any]] = []
    values: dict[
        str,
        dict[tuple[str, int, str], dict[str, list[float]]],
    ] = {
        r0._candidate_id("BLUR", sigma): {}
        for sigma in SIGMA_CANDIDATES
    }
    with ledger_path.open("xb") as ledger:
        for block in r0.BLOCKS:
            if len(trajectories[block]["poses"]) != 602:
                raise RepairIdentityError("TRAJECTORY_COUNT")
            for ordinal in r0.CAL_ORDINALS:
                scene = quality.add_calibration_plate(
                    p1.build_scene(block, ordinal, "CAL")
                )
                identity = {
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
                if predecessor_manifest[(block, ordinal)] != identity:
                    raise RepairIdentityError("R0_CAL_MANIFEST_DRIFT")
                manifest.append(identity)
                for motion in r0.MOTIONS:
                    sequence_key = (block, ordinal, motion)
                    for candidate in values.values():
                        candidate[sequence_key] = {
                            "laplacian_variance_ratio": [],
                            "local_rms_contrast_ratio": [],
                        }
                    for frame_index in r0.FRAME_POSITIONS:
                        rotation, translation = r0._pose(
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
                        common = {
                            "schema": (
                                "rcle.periodic_self_motion_counterfactual."
                                "response_blind_blur_grid_repair_row.v1"
                            ),
                            "protocol_id": PROTOCOL_ID,
                            "repair_id": REPAIR_ID,
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
                            },
                        }
                        payload = quality.canonical_bytes(clean_row)
                        ledger.write(payload)
                        digest.update(payload)
                        row_count += 1
                        for sigma in SIGMA_CANDIDATES:
                            candidate_id = r0._candidate_id("BLUR", sigma)
                            degraded = quality.apply_blur(
                                clean["rgb"],
                                sigma,
                            )
                            metrics = quality.blur_frame_metrics(
                                prepared,
                                degraded,
                            )
                            for name in values[candidate_id][sequence_key]:
                                values[candidate_id][sequence_key][name].append(
                                    float(metrics[name])
                                )
                            row = {
                                **common,
                                "candidate_kind": "BLUR",
                                "candidate_strength": sigma,
                                "degraded_rgb_sha256": quality.sha256_bytes(
                                    degraded.tobytes()
                                ),
                                "metrics": metrics,
                            }
                            if not r0._finite_tree(row):
                                raise quality.InvalidQualityMetric(
                                    "NONFINITE_R1_ROW"
                                )
                            payload = quality.canonical_bytes(row)
                            ledger.write(payload)
                            digest.update(payload)
                            row_count += 1
                        completed += 1
                        if completed % 16 == 0 or completed == 512:
                            elapsed = time.perf_counter() - start
                            throughput = completed / elapsed
                            print(
                                json.dumps(
                                    {
                                        "protocol_id": PROTOCOL_ID,
                                        "phase": "P2_BLUR_GRID_REPAIR_R1",
                                        "completed_units": completed,
                                        "total_units": 512,
                                        "throughput": throughput,
                                        "eta_seconds": (
                                            512 - completed
                                        )
                                        / throughput,
                                        "status": "RUNNING",
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
        ledger.flush()
        os.fsync(ledger.fileno())
    if row_count != EXPECTED_ROWS:
        raise RepairIdentityError("LEDGER_ROW_COUNT")

    summaries = {}
    for sigma in SIGMA_CANDIDATES:
        candidate_id = r0._candidate_id("BLUR", sigma)
        summary = r0._summarize_candidate(
            values[candidate_id],
            (
                "laplacian_variance_ratio",
                "local_rms_contrast_ratio",
            ),
        )
        summaries[candidate_id] = {
            "sigma_px": sigma,
            **summary,
            "passes_all_gates": r0._blur_pass(summary),
        }
    selected_blur = next(
        (
            sigma
            for sigma in SIGMA_CANDIDATES
            if summaries[r0._candidate_id("BLUR", sigma)][
                "passes_all_gates"
            ]
        ),
        None,
    )
    if selected_blur is None:
        axes = (
            "NO_GLOBAL_QUALITY_STRENGTH",
            "VALID",
            "HOLD_P2",
        )
    else:
        axes = (
            "QUALITY_CALIBRATION_PASS",
            "VALID",
            "P3_NOT_AUTHORIZED",
        )
    manifest_hash = quality.sha256_bytes(
        quality.canonical_bytes(manifest)
    )
    lock = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "response_blind_quality_global_strength_lock.v2"
        ),
        "protocol_id": PROTOCOL_ID,
        "repair_id": REPAIR_ID,
        "date": "2026-07-29",
        "phase": "P2_RESPONSE_BLIND_QUALITY_CALIBRATION",
        "scientific_status": axes[0],
        "protocol_status": axes[1],
        "execution_authority": axes[2],
        "selected_global_strengths": {
            "blur_sigma_px": selected_blur,
            "low_texture_alpha": INHERITED_ALPHA,
        },
        "low_texture_inheritance": {
            "source": R0_LOCK_PATH.relative_to(REPO_ROOT).as_posix(),
            "source_sha256": EXPECTED_SHA256[R0_LOCK_PATH],
            "independent_receipt_sha256": EXPECTED_SHA256[
                R0_RECEIPT_PATH
            ],
            "metric_ledger_sha256": R0_LEDGER_SHA256,
            "alpha": INHERITED_ALPHA,
            "rerendered_or_retuned": False,
        },
        "frozen_repair": {
            "candidate_sigma_px": list(SIGMA_CANDIDATES),
            "laplacian_variance_ratio_inclusive": list(r0.BLUR_TARGET),
            "local_rms_contrast_ratio_minimum": r0.BLUR_RMS_MINIMUM,
            "selection": "smallest passing sigma",
            "automatic_second_repair": False,
        },
        "calibration_panel": {
            "blocks": list(r0.BLOCKS),
            "cal_ordinals_per_block": list(r0.CAL_ORDINALS),
            "motions": list(r0.MOTIONS),
            "frame_positions": list(r0.FRAME_POSITIONS),
            "frozen_frame_identities": 512,
            "states_per_frame": 10,
            "candidate_image_evaluations": row_count,
            "manifest": manifest,
            "manifest_sha256": manifest_hash,
            "r0_manifest_sha256": r0_lock["calibration_panel"][
                "calibration_scene_manifest_sha256"
            ],
        },
        "blur_candidate_summaries": summaries,
        "evidence": {
            "ledger": ledger_path.relative_to(REPO_ROOT).as_posix(),
            "ledger_sha256": digest.hexdigest(),
            "ledger_row_count": row_count,
        },
        "implementation_identity": {
            "quality_interventions_sha256": r0.sha256_file(
                QUALITY_SOURCE_PATH
            ),
            "r0_producer_dependency_sha256": r0.sha256_file(
                R0_PRODUCER_PATH
            ),
            "r1_producer_sha256": r0.sha256_file(IMPLEMENTATION_PATH),
            "p1_generator_sha256": r0.sha256_file(P1_SOURCE_PATH),
        },
        "input_read_ledger": reads,
        "firewall": {
            "algorithm_output_read_or_run": False,
            "strength_selected_from_trigger_or_response": False,
            "new_or_replacement_cal_seed": False,
            "low_texture_rerun_or_retune": False,
            "grid_extended_after_access": False,
            "per_block_strength": False,
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
    if not r0._finite_tree(lock):
        raise quality.InvalidQualityMetric("NONFINITE_R1_LOCK")
    payload = quality.canonical_bytes(lock)
    r0._exclusive_write(lock_path, payload)
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "response_blind_quality_blur_grid_repair_producer_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "repair_id": REPAIR_ID,
        "scientific_status": axes[0],
        "protocol_status": axes[1],
        "execution_authority": axes[2],
        "selected_global_strengths": lock[
            "selected_global_strengths"
        ],
        "ledger_sha256": digest.hexdigest(),
        "lock_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": row_count,
        "algorithm_output_read_or_run": False,
        "p3_or_formal_execution_run": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = produce(args.output.resolve(), args.lock.resolve())
    except (
        RepairIdentityError,
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
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
