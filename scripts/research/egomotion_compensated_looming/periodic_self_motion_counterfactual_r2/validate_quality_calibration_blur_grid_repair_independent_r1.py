"""Independent validator for the one-shot R1 small-sigma repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from . import validate_quality_calibration_independent_r0 as base


PROTOCOL_ID = base.PROTOCOL_ID
REPAIR_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1"
)
REPO_ROOT = base.REPO_ROOT
CONTRACT_PATH = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_CONTRACT_2026-07-29.json"
)
R0_LOCK_PATH = base.DEFAULT_LOCK
R0_RECEIPT_PATH = base.DEFAULT_RECEIPT
TRAJECTORY_PATH = base.TRAJECTORY_PATH
P1_SOURCE_PATH = base.P1_SOURCE_PATH
QUALITY_SOURCE_PATH = base.INTERVENTION_PATH
R0_PRODUCER_PATH = base.PRODUCER_PATH
R1_PRODUCER_PATH = (
    REPO_ROOT
    / "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "quality_calibration_blur_grid_repair_r1.py"
)
R0_VALIDATOR_PATH = Path(base.__file__).resolve()
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p2_quality_calibration_blur_grid_repair_r1"
)
DEFAULT_LEDGER = DEFAULT_EVIDENCE_ROOT / "response_blind_blur_metric_ledger.jsonl"
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_GLOBAL_STRENGTH_LOCK_2026-07-29.json"
)
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_INDEPENDENT_VALIDATION_RECEIPT_2026-07-29.json"
)
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
R0_LEDGER_SHA256 = (
    "0356a85d0901426a5ed7142d348863501f6d1d9f14fd1c76520c15ee2f9d0c3a"
)
EXPECTED_FIXED_SHA256 = {
    CONTRACT_PATH: "ac2c9fa9b499d60492d01d542e7401bca4058cd5c786870804a1ebfff3a845ca",
    R0_LOCK_PATH: "d04c06d544c6780e8b86e4eb32b3c181ffe3940626e1f4127fa9cbf497dd41ea",
    R0_RECEIPT_PATH: "37d2f09d0e8764aee904ebc7998d2b629e1cba9dde024e739fbe4d1ea667fd06",
    TRAJECTORY_PATH: "c394641b6419c7a58a58c1bc485e2783cd710e8ba2893415dd6687c20b7d2652",
    P1_SOURCE_PATH: "e6153f52f89674947f4960faf05ae8d5b90d1e9e69fef627a74c719127d6ca43",
    QUALITY_SOURCE_PATH: "750bcdced7c92f96946f9c3d28ed11042edc8b43b8911f817e80772a66c01a5b",
    R0_PRODUCER_PATH: "4c6469131242756dbbf654cd017779e37f89c5677bf0e14290d252d6e8dbfb64",
}


def _validate_fixed_hashes(errors: list[str]) -> None:
    for path, expected in EXPECTED_FIXED_SHA256.items():
        if not path.is_file():
            errors.append(f"MISSING_FIXED_INPUT:{path.name}")
        elif base.sha256_file(path) != expected:
            errors.append(f"FIXED_INPUT_SHA256:{path.name}")


def _validate_lock(lock: dict[str, Any], errors: list[str]) -> None:
    if lock.get("protocol_id") != PROTOCOL_ID:
        errors.append("LOCK_PROTOCOL")
    if lock.get("repair_id") != REPAIR_ID:
        errors.append("LOCK_REPAIR_ID")
    if lock.get("phase") != "P2_RESPONSE_BLIND_QUALITY_CALIBRATION":
        errors.append("LOCK_PHASE")
    if lock.get("protocol_status") != "VALID":
        errors.append("LOCK_PROTOCOL_STATUS")
    if (
        lock.get("formal_execution_authorized") is not False
        or lock.get("p3_authorized") is not False
    ):
        errors.append("LOCK_AUTHORITY")
    frozen = lock.get("frozen_repair", {})
    if tuple(frozen.get("candidate_sigma_px", ())) != SIGMA_CANDIDATES:
        errors.append("LOCK_SIGMA_GRID")
    if tuple(
        frozen.get("laplacian_variance_ratio_inclusive", ())
    ) != base.BLUR_RANGE:
        errors.append("LOCK_LAPLACIAN_GATE")
    if (
        frozen.get("local_rms_contrast_ratio_minimum")
        != base.RMS_MINIMUM
    ):
        errors.append("LOCK_RMS_GATE")
    if frozen.get("automatic_second_repair") is not False:
        errors.append("LOCK_SECOND_REPAIR")
    panel = lock.get("calibration_panel", {})
    if (
        tuple(panel.get("blocks", ())) != base.BLOCKS
        or tuple(panel.get("cal_ordinals_per_block", ())) != base.ORDINALS
        or tuple(panel.get("motions", ())) != base.MOTIONS
        or tuple(panel.get("frame_positions", ())) != base.FRAMES
        or panel.get("frozen_frame_identities") != 512
        or panel.get("states_per_frame") != 10
        or panel.get("candidate_image_evaluations") != EXPECTED_ROWS
    ):
        errors.append("LOCK_PANEL")
    inheritance = lock.get("low_texture_inheritance", {})
    if (
        inheritance.get("source_sha256")
        != EXPECTED_FIXED_SHA256[R0_LOCK_PATH]
        or inheritance.get("independent_receipt_sha256")
        != EXPECTED_FIXED_SHA256[R0_RECEIPT_PATH]
        or inheritance.get("metric_ledger_sha256")
        != R0_LEDGER_SHA256
        or inheritance.get("alpha") != 0.15
        or inheritance.get("rerendered_or_retuned") is not False
    ):
        errors.append("LOW_TEXTURE_INHERITANCE")
    required_false = (
        "algorithm_output_read_or_run",
        "strength_selected_from_trigger_or_response",
        "new_or_replacement_cal_seed",
        "low_texture_rerun_or_retune",
        "grid_extended_after_access",
        "per_block_strength",
        "p3_preflight_run",
        "formal_480_plus_16_sequences_run",
        "r3_threshold_or_three_pair_modified",
        "sequence16_cotracker_android_realtime",
    )
    firewall = lock.get("firewall", {})
    for key in required_false:
        if firewall.get(key) is not False:
            errors.append(f"FIREWALL:{key}")
    if lock.get("independent_validation") != "REQUIRED_NOT_YET_CREATED":
        errors.append("LOCK_PREVALIDATION_STATE")


def _validate_hash_chain(lock: dict[str, Any], errors: list[str]) -> None:
    identity = lock.get("implementation_identity", {})
    expected = {
        "quality_interventions_sha256": QUALITY_SOURCE_PATH,
        "r0_producer_dependency_sha256": R0_PRODUCER_PATH,
        "r1_producer_sha256": R1_PRODUCER_PATH,
        "p1_generator_sha256": P1_SOURCE_PATH,
    }
    for key, path in expected.items():
        if not path.is_file() or identity.get(key) != base.sha256_file(path):
            errors.append(f"IMPLEMENTATION_SHA256:{key}")
    expected_reads = {
        path.relative_to(REPO_ROOT).as_posix(): digest
        for path, digest in EXPECTED_FIXED_SHA256.items()
    }
    actual_reads = {
        item.get("path"): item.get("sha256")
        for item in lock.get("input_read_ledger", [])
        if isinstance(item, dict)
    }
    if actual_reads != expected_reads:
        errors.append("INPUT_READ_LEDGER")


def _read_ledger(
    path: Path,
    errors: list[str],
) -> tuple[
    dict[str, dict[tuple[str, int, str], dict[str, list[float]]]],
    list[dict[str, Any]],
    int,
]:
    values = {
        base._candidate_id("BLUR", sigma): {}
        for sigma in SIGMA_CANDIDATES
    }
    groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    manifest: dict[tuple[str, int], dict[str, Any]] = {}
    row_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"LEDGER_JSON:{line_number}")
                continue
            row_count += 1
            if not base._finite_tree(row):
                errors.append(f"NONFINITE_ROW:{line_number}")
                continue
            if len(errors) < 100:
                base._validate_row_keys(row, errors)
            if (
                row.get("protocol_id") != PROTOCOL_ID
                or row.get("repair_id") != REPAIR_ID
            ):
                errors.append(f"ROW_IDENTITY:{line_number}")
                continue
            block = row.get("block")
            ordinal = row.get("cal_ordinal")
            motion = row.get("motion")
            frame = row.get("frame_index")
            if (
                block not in base.BLOCKS
                or ordinal not in base.ORDINALS
                or motion not in base.MOTIONS
                or frame not in base.FRAMES
            ):
                errors.append(f"ROW_PANEL:{line_number}")
                continue
            frame_key = (block, ordinal, motion, frame)
            groups.setdefault(frame_key, []).append(row)
            item = {
                "block": block,
                "cal_ordinal": ordinal,
                "numeric_seed_uint64": row.get("numeric_seed_uint64"),
                "base_scene_geometry_sha256": row.get(
                    "base_scene_geometry_sha256"
                ),
                "calibration_scene_sha256": row.get(
                    "calibration_scene_sha256"
                ),
            }
            existing = manifest.setdefault((block, ordinal), item)
            if existing != item:
                errors.append(f"MANIFEST_DRIFT:{line_number}")
            kind = row.get("candidate_kind")
            strength = row.get("candidate_strength")
            metrics = row.get("metrics", {})
            if kind == "CLEAN":
                if strength is not None:
                    errors.append(f"CLEAN_STRENGTH:{line_number}")
                if row.get("clean_rgb_sha256") != row.get(
                    "degraded_rgb_sha256"
                ):
                    errors.append(f"CLEAN_RGB:{line_number}")
            elif kind == "BLUR" and strength in SIGMA_CANDIDATES:
                candidate = base._candidate_id("BLUR", float(strength))
                sequence_key = (block, ordinal, motion)
                target = values[candidate].setdefault(
                    sequence_key,
                    {
                        "laplacian_variance_ratio": [],
                        "local_rms_contrast_ratio": [],
                    },
                )
                base._ratio(
                    metrics,
                    "degraded_laplacian_variance",
                    "clean_laplacian_variance",
                    "laplacian_variance_ratio",
                    errors,
                    str(line_number),
                )
                base._ratio(
                    metrics,
                    "degraded_local_rms_contrast",
                    "clean_local_rms_contrast",
                    "local_rms_contrast_ratio",
                    errors,
                    str(line_number),
                )
                for name in target:
                    target[name].append(float(metrics[name]))
            else:
                errors.append(f"ROW_CANDIDATE:{line_number}")
    if row_count != EXPECTED_ROWS:
        errors.append("LEDGER_ROW_COUNT")
    if len(groups) != 512:
        errors.append("FRAME_GROUP_COUNT")
    expected_states = {
        ("CLEAN", None),
        *(("BLUR", sigma) for sigma in SIGMA_CANDIDATES),
    }
    for key, rows in groups.items():
        states = {
            (row.get("candidate_kind"), row.get("candidate_strength"))
            for row in rows
        }
        if len(rows) != 10 or states != expected_states:
            errors.append(f"FRAME_STATES:{key}")
            continue
        for common in (
            "numeric_seed_uint64",
            "base_scene_geometry_sha256",
            "calibration_scene_sha256",
            "valid_mask_sha256",
            "object_id_sha256",
            "clean_rgb_sha256",
        ):
            if len({row.get(common) for row in rows}) != 1:
                errors.append(f"FRAME_IDENTITY:{key}:{common}")
    manifest_list = [manifest[key] for key in sorted(manifest)]
    return values, manifest_list, row_count


def validate(lock_path: Path, ledger_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    _validate_fixed_hashes(errors)
    try:
        r0_lock = json.loads(R0_LOCK_PATH.read_text(encoding="utf-8"))
        r0_receipt = json.loads(R0_RECEIPT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        r0_lock, r0_receipt = {}, {}
        errors.append("R0_JSON")
    if (
        r0_lock.get("evidence", {}).get(
            "response_blind_metric_ledger_sha256"
        )
        != R0_LEDGER_SHA256
        or r0_receipt.get("evidence_sha256", {}).get(
            "response_blind_metric_ledger"
        )
        != R0_LEDGER_SHA256
        or r0_receipt.get("validated") is not True
        or r0_receipt.get("errors") != []
    ):
        errors.append("R0_INHERITANCE_CHAIN")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lock = {}
        errors.append("LOCK_JSON")
    if not base._finite_tree(lock):
        errors.append("LOCK_NONFINITE")
    _validate_lock(lock, errors)
    _validate_hash_chain(lock, errors)
    if not ledger_path.is_file():
        errors.append("LEDGER_MISSING")
        values, manifest, row_count = {}, [], 0
    else:
        if base.sha256_file(ledger_path) != lock.get("evidence", {}).get(
            "ledger_sha256"
        ):
            errors.append("LEDGER_SHA256")
        values, manifest, row_count = _read_ledger(ledger_path, errors)
    manifest_hash = hashlib.sha256(base.canonical_bytes(manifest)).hexdigest()
    panel = lock.get("calibration_panel", {})
    if panel.get("manifest") != manifest:
        errors.append("MANIFEST")
    if panel.get("manifest_sha256") != manifest_hash:
        errors.append("MANIFEST_SHA256")
    if manifest != r0_lock.get("calibration_panel", {}).get("manifest"):
        errors.append("R0_MANIFEST_IDENTITY")
    if panel.get("r0_manifest_sha256") != r0_lock.get(
        "calibration_panel",
        {},
    ).get("calibration_scene_manifest_sha256"):
        errors.append("R0_MANIFEST_SHA256")

    independent: dict[str, Any] = {}
    observed = lock.get("blur_candidate_summaries", {})
    if values:
        for sigma in SIGMA_CANDIDATES:
            candidate = base._candidate_id("BLUR", sigma)
            summary = base._summary(
                values[candidate],
                (
                    "laplacian_variance_ratio",
                    "local_rms_contrast_ratio",
                ),
                errors,
            )
            expected = {
                "sigma_px": sigma,
                **summary,
                "passes_all_gates": base._passes_blur(summary),
            }
            independent[candidate] = expected
            base._compare_tree(
                observed.get(candidate),
                expected,
                candidate,
                errors,
            )
    selected_blur = next(
        (
            sigma
            for sigma in SIGMA_CANDIDATES
            if independent.get(
                base._candidate_id("BLUR", sigma),
                {},
            ).get("passes_all_gates")
        ),
        None,
    )
    selected = lock.get("selected_global_strengths", {})
    if selected.get("blur_sigma_px") != selected_blur:
        errors.append("SELECTED_BLUR")
    if selected.get("low_texture_alpha") != 0.15:
        errors.append("SELECTED_LOW_TEXTURE")
    if selected_blur is None:
        expected_axes = (
            "NO_GLOBAL_QUALITY_STRENGTH",
            "VALID",
            "HOLD_P2",
        )
    else:
        expected_axes = (
            "QUALITY_CALIBRATION_PASS",
            "VALID",
            "P3_NOT_AUTHORIZED",
        )
    observed_axes = (
        lock.get("scientific_status"),
        lock.get("protocol_status"),
        lock.get("execution_authority"),
    )
    if observed_axes != expected_axes:
        errors.append("TERMINAL_AXES")
    axes = (
        ("CLAIM_NOT_SIGNABLE", "INVALID", "HOLD_P2")
        if errors
        else expected_axes
    )
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "response_blind_quality_blur_grid_repair_validation_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "repair_id": REPAIR_ID,
        "date": "2026-07-29",
        "scientific_status": axes[0],
        "protocol_status": axes[1],
        "execution_authority": axes[2],
        "errors": errors,
        "validated": not errors,
        "selected_global_strengths": {
            "blur_sigma_px": selected_blur,
            "low_texture_alpha": 0.15,
        },
        "counts": {
            "ledger_rows": row_count,
            "frozen_frame_identities": 512,
            "states_per_frame": 10,
            "block_x_motion_subgroups": 8,
        },
        "independence": {
            "r1_producer_imported": False,
            "quality_intervention_module_imported": False,
            "algorithm_module_imported_or_run": False,
            "predecessor_independent_validator_reused": True,
            "validation_scope": (
                "all R1 rows, raw ratios, sequence/subgroup/overall "
                "hierarchy, gates, smallest-sigma selection, R0 alpha "
                "inheritance, manifests, hashes, read allowlist and firewall"
            ),
        },
        "evidence_sha256": {
            "contract": EXPECTED_FIXED_SHA256[CONTRACT_PATH],
            "r0_strength_lock": EXPECTED_FIXED_SHA256[R0_LOCK_PATH],
            "r0_validation_receipt": EXPECTED_FIXED_SHA256[R0_RECEIPT_PATH],
            "r0_metric_ledger": R0_LEDGER_SHA256,
            "r1_strength_lock": base.sha256_file(lock_path)
            if lock_path.is_file()
            else None,
            "r1_metric_ledger": base.sha256_file(ledger_path)
            if ledger_path.is_file()
            else None,
            "r1_producer_source": base.sha256_file(R1_PRODUCER_PATH),
            "validator_source": base.sha256_file(Path(__file__).resolve()),
            "predecessor_validator_source": base.sha256_file(
                R0_VALIDATOR_PATH
            ),
        },
        "firewall": {
            "algorithm_output_read_or_run": False,
            "p3_preflight_run": False,
            "formal_480_plus_16_sequences_run": False,
            "r3_threshold_or_three_pair_modified": False,
            "sequence16_cotracker_android_realtime": False,
        },
        "formal_execution_authorized": False,
        "p3_authorized": False,
        "automatic_second_repair": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = validate(args.lock.resolve(), args.ledger.resolve())
    payload = base.canonical_bytes(receipt)
    if not args.preflight:
        try:
            base._exclusive_write(args.receipt.resolve(), payload)
        except FileExistsError:
            print(
                json.dumps(
                    {
                        "scientific_status": "CLAIM_NOT_SIGNABLE",
                        "protocol_status": "INVALID",
                        "execution_authority": "HOLD_P2",
                        "errors": ["RECEIPT_ALREADY_EXISTS"],
                    },
                    sort_keys=True,
                )
            )
            return 2
    print(payload.decode("utf-8").strip())
    return 0 if receipt["validated"] else 2


if __name__ == "__main__":
    sys.exit(main())
