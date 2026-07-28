"""Independent ledger/selection validator for R2 P2 quality calibration.

The validator intentionally does not import the P2 producer, intervention
implementation, or any RCLE algorithm module. It independently checks every
ledger row, ratio, hierarchy, gate, selection direction, identity hash, and
firewall declaration before writing one exclusive validation receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
CALIBRATION_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "RESPONSE_BLIND_QUALITY_CALIBRATION_R0"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p2_quality_calibration_r0"
)
DEFAULT_LEDGER = DEFAULT_EVIDENCE_ROOT / "response_blind_metric_ledger.jsonl"
DEFAULT_LOCK = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_R0_STRENGTH_LOCK_2026-07-29.json"
)
DEFAULT_RECEIPT = (
    REPO_ROOT
    / "docs/research/rcle/"
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QUALITY_CALIBRATION_R0_INDEPENDENT_VALIDATION_RECEIPT_2026-07-29.json"
)
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
P1_RECEIPT_PATH = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/"
    "independent_geometry_validation_receipt.json"
)
TRAJECTORY_PATH = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
)
P1_SOURCE_PATH = (
    REPO_ROOT
    / "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/generator_geometry.py"
)
INTERVENTION_PATH = (
    REPO_ROOT
    / "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/quality_interventions_r0.py"
)
PRODUCER_PATH = (
    REPO_ROOT
    / "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/quality_calibration_r0.py"
)
EXPECTED_FIXED_SHA256 = {
    CONTRACT_PATH: "73705144d0d7a8162c0a47694676364e00d3b27917904f78a39642900aeac0c5",
    BUDGET_PATH: "4f0c3204e6ea5bcb7daf22b017b1819325bae47c20e7f170677d085a41708798",
    P1_LOCK_PATH: "a7fa41c0406908baf05805904111ba43fdbd8dd93b8c4e496706f1990438adc9",
    P1_RECEIPT_PATH: "95646437fbe0ef0cf03844f94467303f5d90ca15c3e22fc1785157b037a8c079",
    TRAJECTORY_PATH: "c394641b6419c7a58a58c1bc485e2783cd710e8ba2893415dd6687c20b7d2652",
    P1_SOURCE_PATH: "e6153f52f89674947f4960faf05ae8d5b90d1e9e69fef627a74c719127d6ca43",
}
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
ORDINALS = (0, 1, 2, 3)
FRAMES = (
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
BLUR = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
LOW_TEXTURE = (0.75, 0.60, 0.45, 0.30, 0.15)
BLUR_RANGE = (0.35, 0.55)
RMS_MINIMUM = 0.70
GRADIENT_RANGE = (0.35, 0.55)
EDGE_RANGE = (0.90, 1.10)
EXPECTED_ROW_COUNT = 6144
FORBIDDEN_KEY_FRAGMENTS = (
    "trigger",
    "response_value",
    "response_metric",
    "support",
    "feature_collapse",
    "fb_output",
    "forward_backward",
    "pair_state",
    "three_pair",
    "cotracker",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
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


def _median(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("INVALID_MEDIAN_INPUT")
    result = float(np.median(array))
    if not math.isfinite(result):
        raise ValueError("NONFINITE_MEDIAN")
    return result


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def _candidate_id(kind: str, strength: float) -> str:
    if kind == "BLUR":
        return f"BLUR_SIGMA_{strength:.2f}"
    return f"LOW_TEXTURE_ALPHA_{strength:.2f}"


def _validate_fixed_hashes(errors: list[str]) -> None:
    for path, expected in EXPECTED_FIXED_SHA256.items():
        if not path.is_file():
            errors.append(f"MISSING_FIXED_INPUT:{path.name}")
        elif sha256_file(path) != expected:
            errors.append(f"FIXED_INPUT_SHA256:{path.name}")


def _validate_lock_identity(lock: dict[str, Any], errors: list[str]) -> None:
    if lock.get("protocol_id") != PROTOCOL_ID:
        errors.append("LOCK_PROTOCOL_ID")
    if lock.get("calibration_id") != CALIBRATION_ID:
        errors.append("LOCK_CALIBRATION_ID")
    if lock.get("phase") != "P2_RESPONSE_BLIND_QUALITY_CALIBRATION":
        errors.append("LOCK_PHASE")
    if lock.get("protocol_status") != "VALID":
        errors.append("LOCK_PROTOCOL_STATUS")
    if lock.get("formal_execution_authorized") is not False:
        errors.append("FORMAL_EXECUTION_AUTHORITY")
    if lock.get("p3_authorized") is not False:
        errors.append("P3_AUTHORITY")
    panel = lock.get("calibration_panel", {})
    if tuple(panel.get("blocks", ())) != BLOCKS:
        errors.append("LOCK_BLOCKS")
    if tuple(panel.get("cal_ordinals_per_block", ())) != ORDINALS:
        errors.append("LOCK_ORDINALS")
    if tuple(panel.get("motions", ())) != MOTIONS:
        errors.append("LOCK_MOTIONS")
    if tuple(panel.get("frame_positions", ())) != FRAMES:
        errors.append("LOCK_FRAMES")
    if panel.get("candidate_image_evaluations") != EXPECTED_ROW_COUNT:
        errors.append("LOCK_IMAGE_EVALUATIONS")
    gates = lock.get("frozen_grid_and_gates", {})
    if tuple(gates.get("blur_sigma_px", ())) != BLUR:
        errors.append("LOCK_BLUR_GRID")
    if tuple(gates.get("low_texture_alpha", ())) != LOW_TEXTURE:
        errors.append("LOCK_LOW_TEXTURE_GRID")
    if tuple(gates.get("laplacian_variance_ratio_inclusive", ())) != BLUR_RANGE:
        errors.append("LOCK_BLUR_RANGE")
    if gates.get("local_rms_contrast_ratio_minimum") != RMS_MINIMUM:
        errors.append("LOCK_RMS_MINIMUM")
    if (
        tuple(
            gates.get(
                "multiscale_gradient_density_ratio_inclusive",
                (),
            )
        )
        != GRADIENT_RANGE
    ):
        errors.append("LOCK_GRADIENT_RANGE")
    if (
        tuple(gates.get("cal_fixture_edge_spread_ratio_inclusive", ()))
        != EDGE_RANGE
    ):
        errors.append("LOCK_EDGE_RANGE")
    firewall = lock.get("firewall", {})
    required_false = (
        "algorithm_output_read_or_run",
        "strength_selected_from_algorithm_output",
        "p3_preflight_run",
        "formal_480_plus_16_sequences_run",
        "r3_threshold_or_three_pair_modified",
        "sequence16_cotracker_android_realtime",
    )
    for key in required_false:
        if firewall.get(key) is not False:
            errors.append(f"FIREWALL:{key}")
    if lock.get("independent_validation") != "REQUIRED_NOT_YET_CREATED":
        errors.append("LOCK_PREVALIDATION_STATE")


def _validate_implementation_hashes(
    lock: dict[str, Any],
    errors: list[str],
) -> None:
    identity = lock.get("implementation_identity", {})
    expected = {
        "quality_interventions_sha256": INTERVENTION_PATH,
        "producer_sha256": PRODUCER_PATH,
        "p1_generator_sha256": P1_SOURCE_PATH,
    }
    for key, path in expected.items():
        if not path.is_file() or identity.get(key) != sha256_file(path):
            errors.append(f"IMPLEMENTATION_SHA256:{key}")
    if identity.get("low_texture_psf_operator") != "NONE":
        errors.append("LOW_TEXTURE_PSF_IDENTITY")
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
    for audit in lock.get("firewall", {}).get("source_import_audit", []):
        if audit.get("forbidden_imports") != []:
            errors.append("SOURCE_IMPORT_FIREWALL")


def _validate_row_keys(row: dict[str, Any], errors: list[str]) -> None:
    stack = [row]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                lower = key.lower()
                if any(fragment in lower for fragment in FORBIDDEN_KEY_FRAGMENTS):
                    errors.append(f"FORBIDDEN_LEDGER_KEY:{key}")
                stack.append(item)
        elif isinstance(value, list):
            stack.extend(value)


def _ratio(
    metrics: dict[str, Any],
    numerator: str,
    denominator: str,
    ratio: str,
    errors: list[str],
    row_label: str,
) -> None:
    try:
        expected = float(metrics[numerator]) / float(metrics[denominator])
        observed = float(metrics[ratio])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append(f"ROW_RATIO_INPUT:{row_label}:{ratio}")
        return
    if not math.isfinite(expected) or not _close(expected, observed):
        errors.append(f"ROW_RATIO_MISMATCH:{row_label}:{ratio}")


def _read_and_validate_ledger(
    ledger_path: Path,
    errors: list[str],
) -> tuple[
    dict[str, dict[tuple[str, int, str], dict[str, list[float]]]],
    dict[str, dict[tuple[str, int, str], dict[str, list[float]]]],
    list[dict[str, Any]],
    int,
]:
    blur_values = {
        _candidate_id("BLUR", strength): {}
        for strength in BLUR
    }
    low_values = {
        _candidate_id("LOW_TEXTURE", strength): {}
        for strength in LOW_TEXTURE
    }
    frame_groups: dict[
        tuple[str, int, str, int],
        list[dict[str, Any]],
    ] = {}
    manifest: dict[tuple[str, int], dict[str, Any]] = {}
    row_count = 0
    with ledger_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"LEDGER_JSON:{line_number}")
                continue
            row_count += 1
            if not _finite_tree(row):
                errors.append(f"NONFINITE_ROW:{line_number}")
                continue
            if len(errors) < 100:
                _validate_row_keys(row, errors)
            label = f"{line_number}"
            if (
                row.get("protocol_id") != PROTOCOL_ID
                or row.get("calibration_id") != CALIBRATION_ID
            ):
                errors.append(f"ROW_IDENTITY:{line_number}")
                continue
            block = row.get("block")
            ordinal = row.get("cal_ordinal")
            motion = row.get("motion")
            frame = row.get("frame_index")
            if (
                block not in BLOCKS
                or ordinal not in ORDINALS
                or motion not in MOTIONS
                or frame not in FRAMES
            ):
                errors.append(f"ROW_PANEL_IDENTITY:{line_number}")
                continue
            frame_key = (block, ordinal, motion, frame)
            frame_groups.setdefault(frame_key, []).append(row)
            manifest_key = (block, ordinal)
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
            existing = manifest.setdefault(manifest_key, item)
            if existing != item:
                errors.append(f"ROW_SCENE_IDENTITY_DRIFT:{line_number}")
            kind = row.get("candidate_kind")
            strength = row.get("candidate_strength")
            metrics = row.get("metrics", {})
            sequence_key = (block, ordinal, motion)
            if kind == "CLEAN":
                if strength is not None:
                    errors.append(f"CLEAN_STRENGTH:{line_number}")
                if row.get("degraded_rgb_sha256") != row.get(
                    "clean_rgb_sha256"
                ):
                    errors.append(f"CLEAN_RGB_IDENTITY:{line_number}")
            elif kind == "BLUR" and strength in BLUR:
                candidate = _candidate_id("BLUR", float(strength))
                target = blur_values[candidate].setdefault(
                    sequence_key,
                    {
                        "laplacian_variance_ratio": [],
                        "local_rms_contrast_ratio": [],
                    },
                )
                _ratio(
                    metrics,
                    "degraded_laplacian_variance",
                    "clean_laplacian_variance",
                    "laplacian_variance_ratio",
                    errors,
                    label,
                )
                _ratio(
                    metrics,
                    "degraded_local_rms_contrast",
                    "clean_local_rms_contrast",
                    "local_rms_contrast_ratio",
                    errors,
                    label,
                )
                for name in target:
                    target[name].append(float(metrics[name]))
            elif kind == "LOW_TEXTURE" and strength in LOW_TEXTURE:
                if row.get("psf_operator_applied") is not False:
                    errors.append(f"LOW_TEXTURE_PSF:{line_number}")
                candidate = _candidate_id("LOW_TEXTURE", float(strength))
                target = low_values[candidate].setdefault(
                    sequence_key,
                    {
                        "multiscale_gradient_density_ratio": [],
                        "edge_spread_ratio": [],
                    },
                )
                _ratio(
                    metrics,
                    "degraded_multiscale_gradient_density",
                    "clean_multiscale_gradient_density",
                    "multiscale_gradient_density_ratio",
                    errors,
                    label,
                )
                _ratio(
                    metrics,
                    "degraded_edge_spread_px",
                    "clean_edge_spread_px",
                    "edge_spread_ratio",
                    errors,
                    label,
                )
                if int(metrics.get("clean_valid_edge_count", 0)) <= 0:
                    errors.append(f"CLEAN_EDGE_COUNT:{line_number}")
                if int(metrics.get("degraded_valid_edge_count", 0)) <= 0:
                    errors.append(f"DEGRADED_EDGE_COUNT:{line_number}")
                for name in target:
                    target[name].append(float(metrics[name]))
            else:
                errors.append(f"ROW_CANDIDATE:{line_number}")
    if row_count != EXPECTED_ROW_COUNT:
        errors.append("LEDGER_ROW_COUNT")
    if len(frame_groups) != 512:
        errors.append("LEDGER_FRAME_IDENTITY_COUNT")
    expected_states = {
        ("CLEAN", None),
        *(("BLUR", strength) for strength in BLUR),
        *(("LOW_TEXTURE", strength) for strength in LOW_TEXTURE),
    }
    for frame_key, rows in frame_groups.items():
        states = {
            (row.get("candidate_kind"), row.get("candidate_strength"))
            for row in rows
        }
        if len(rows) != 12 or states != expected_states:
            errors.append(f"FRAME_CANDIDATE_STATES:{frame_key}")
            continue
        common_keys = (
            "numeric_seed_uint64",
            "base_scene_geometry_sha256",
            "calibration_scene_sha256",
            "valid_mask_sha256",
            "object_id_sha256",
            "source_known_edge_manifest_sha256",
            "clean_rgb_sha256",
        )
        for key in common_keys:
            if len({row.get(key) for row in rows}) != 1:
                errors.append(f"FRAME_COMMON_IDENTITY:{frame_key}:{key}")
    manifest_list = [
        manifest[key]
        for key in sorted(manifest)
    ]
    return blur_values, low_values, manifest_list, row_count


def _summary(
    values: dict[tuple[str, int, str], dict[str, list[float]]],
    names: tuple[str, ...],
    errors: list[str],
) -> dict[str, Any]:
    sequences: list[dict[str, Any]] = []
    expected_keys = {
        (block, ordinal, motion)
        for block in BLOCKS
        for ordinal in ORDINALS
        for motion in MOTIONS
    }
    if set(values) != expected_keys:
        errors.append("SEQUENCE_KEYSET")
    for block, ordinal, motion in sorted(expected_keys):
        key = (block, ordinal, motion)
        metrics = values.get(key)
        if metrics is None:
            errors.append(f"MISSING_SEQUENCE:{key}")
            continue
        if any(len(metrics[name]) != 16 for name in names):
            errors.append(f"SEQUENCE_FRAME_COUNT:{key}")
            continue
        sequences.append(
            {
                "block": block,
                "cal_ordinal": ordinal,
                "motion": motion,
                "frame_count": 16,
                "metrics": {
                    name: _median(metrics[name]) for name in names
                },
            }
        )
    subgroups: list[dict[str, Any]] = []
    for block in BLOCKS:
        for motion in MOTIONS:
            chosen = [
                row["metrics"]
                for row in sequences
                if row["block"] == block and row["motion"] == motion
            ]
            if len(chosen) != 4:
                errors.append(f"SUBGROUP_SEQUENCE_COUNT:{block}:{motion}")
                continue
            subgroups.append(
                {
                    "block": block,
                    "motion": motion,
                    "sequence_count": 4,
                    "metrics": {
                        name: _median([row[name] for row in chosen])
                        for name in names
                    },
                }
            )
    overall = {
        name: _median([row["metrics"][name] for row in sequences])
        for name in names
    }
    return {
        "sequence_medians": sequences,
        "subgroup_medians": subgroups,
        "overall_median": overall,
    }


def _compare_tree(
    observed: Any,
    expected: Any,
    label: str,
    errors: list[str],
) -> None:
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not _close(
            observed,
            expected,
        ):
            errors.append(f"SUMMARY_MISMATCH:{label}")
        return
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            errors.append(f"SUMMARY_KEYSET:{label}")
            return
        for key in expected:
            _compare_tree(
                observed[key],
                expected[key],
                f"{label}.{key}",
                errors,
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            errors.append(f"SUMMARY_LIST:{label}")
            return
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _compare_tree(left, right, f"{label}[{index}]", errors)
        return
    if observed != expected:
        errors.append(f"SUMMARY_VALUE:{label}")


def _passes_blur(summary: dict[str, Any]) -> bool:
    rows = [
        summary["overall_median"],
        *(item["metrics"] for item in summary["subgroup_medians"]),
    ]
    return all(
        BLUR_RANGE[0] <= row["laplacian_variance_ratio"] <= BLUR_RANGE[1]
        and row["local_rms_contrast_ratio"] >= RMS_MINIMUM
        for row in rows
    )


def _passes_low(summary: dict[str, Any], fixture_ratio: float) -> bool:
    rows = [
        summary["overall_median"],
        *(item["metrics"] for item in summary["subgroup_medians"]),
    ]
    return (
        all(
            GRADIENT_RANGE[0]
            <= row["multiscale_gradient_density_ratio"]
            <= GRADIENT_RANGE[1]
            and EDGE_RANGE[0] <= row["edge_spread_ratio"] <= EDGE_RANGE[1]
            for row in rows
        )
        and EDGE_RANGE[0] <= fixture_ratio <= EDGE_RANGE[1]
    )


def validate(lock_path: Path, ledger_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    _validate_fixed_hashes(errors)
    if not lock_path.is_file():
        errors.append("LOCK_MISSING")
        lock: dict[str, Any] = {}
    else:
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            lock = {}
            errors.append("LOCK_JSON")
    if not ledger_path.is_file():
        errors.append("LEDGER_MISSING")
    if not _finite_tree(lock):
        errors.append("LOCK_NONFINITE")
    _validate_lock_identity(lock, errors)
    _validate_implementation_hashes(lock, errors)
    if ledger_path.is_file():
        expected_hash = lock.get("evidence", {}).get(
            "response_blind_metric_ledger_sha256"
        )
        if sha256_file(ledger_path) != expected_hash:
            errors.append("LEDGER_SHA256")
        blur_values, low_values, manifest, row_count = (
            _read_and_validate_ledger(ledger_path, errors)
        )
    else:
        blur_values, low_values, manifest, row_count = {}, {}, [], 0
    manifest_hash = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    if (
        lock.get("calibration_panel", {}).get(
            "calibration_scene_manifest_sha256"
        )
        != manifest_hash
    ):
        errors.append("CALIBRATION_MANIFEST_SHA256")
    if lock.get("calibration_panel", {}).get("manifest") != manifest:
        errors.append("CALIBRATION_MANIFEST")

    independent_blur: dict[str, Any] = {}
    independent_low: dict[str, Any] = {}
    lock_blur = lock.get("candidate_summaries", {}).get("blur", {})
    lock_low = lock.get("candidate_summaries", {}).get("low_texture", {})
    if blur_values:
        for strength in BLUR:
            candidate = _candidate_id("BLUR", strength)
            summary = _summary(
                blur_values[candidate],
                (
                    "laplacian_variance_ratio",
                    "local_rms_contrast_ratio",
                ),
                errors,
            )
            passed = _passes_blur(summary)
            expected = {
                "sigma_px": strength,
                **summary,
                "passes_all_gates": passed,
            }
            independent_blur[candidate] = expected
            _compare_tree(lock_blur.get(candidate), expected, candidate, errors)
        for strength in LOW_TEXTURE:
            candidate = _candidate_id("LOW_TEXTURE", strength)
            summary = _summary(
                low_values[candidate],
                (
                    "multiscale_gradient_density_ratio",
                    "edge_spread_ratio",
                ),
                errors,
            )
            fixture = lock_low.get(candidate, {}).get("analytic_fixture", {})
            try:
                fixture_ratio = float(fixture["degraded_edge_spread_px"]) / float(
                    fixture["clean_edge_spread_px"]
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                fixture_ratio = float("nan")
                errors.append(f"FIXTURE_RATIO:{candidate}")
            if (
                not math.isfinite(fixture_ratio)
                or not _close(fixture_ratio, fixture.get("edge_spread_ratio", float("nan")))
                or fixture.get("clean_valid_edge_count") != 32
                or fixture.get("degraded_valid_edge_count") != 32
            ):
                errors.append(f"FIXTURE_IDENTITY:{candidate}")
            passed = _passes_low(summary, fixture_ratio)
            expected = {
                "alpha": strength,
                **summary,
                "analytic_fixture": fixture,
                "passes_all_gates": passed,
            }
            independent_low[candidate] = expected
            _compare_tree(lock_low.get(candidate), expected, candidate, errors)

    selected_blur = next(
        (
            strength
            for strength in BLUR
            if independent_blur.get(
                _candidate_id("BLUR", strength),
                {},
            ).get("passes_all_gates")
        ),
        None,
    )
    selected_low = next(
        (
            strength
            for strength in LOW_TEXTURE
            if independent_low.get(
                _candidate_id("LOW_TEXTURE", strength),
                {},
            ).get("passes_all_gates")
        ),
        None,
    )
    selected = lock.get("selected_global_strengths", {})
    if selected.get("blur_sigma_px") != selected_blur:
        errors.append("SELECTED_BLUR")
    if selected.get("low_texture_alpha") != selected_low:
        errors.append("SELECTED_LOW_TEXTURE")
    if selected_blur is not None and selected_low is not None:
        expected_axes = (
            "QUALITY_CALIBRATION_PASS",
            "VALID",
            "P3_NOT_AUTHORIZED",
        )
    else:
        expected_axes = (
            "NO_GLOBAL_QUALITY_STRENGTH",
            "VALID",
            "HOLD_P2",
        )
    observed_axes = (
        lock.get("scientific_status"),
        lock.get("protocol_status"),
        lock.get("execution_authority"),
    )
    if observed_axes != expected_axes:
        errors.append("TERMINAL_AXES")
    if errors:
        axes = ("CLAIM_NOT_SIGNABLE", "INVALID", "HOLD_P2")
    else:
        axes = expected_axes
    return {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "response_blind_quality_independent_validation_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "calibration_id": CALIBRATION_ID,
        "date": "2026-07-29",
        "scientific_status": axes[0],
        "protocol_status": axes[1],
        "execution_authority": axes[2],
        "errors": errors,
        "validated": not errors,
        "independence": {
            "producer_imported": False,
            "quality_intervention_module_imported": False,
            "algorithm_module_imported_or_run": False,
            "validation_scope": (
                "all ledger rows, raw ratios, 16-frame sequence medians, "
                "four-seed subgroup medians, overall medians, analytic "
                "fixture identities, gates, selection direction, hashes, "
                "read allowlist, and firewall"
            ),
        },
        "counts": {
            "ledger_rows": row_count,
            "frozen_frame_identities": 512,
            "candidate_states_per_frame": 12,
            "block_x_motion_subgroups": 8,
        },
        "selected_global_strengths": {
            "blur_sigma_px": selected_blur,
            "low_texture_alpha": selected_low,
        },
        "evidence_sha256": {
            "strength_lock": sha256_file(lock_path)
            if lock_path.is_file()
            else None,
            "response_blind_metric_ledger": sha256_file(ledger_path)
            if ledger_path.is_file()
            else None,
            "quality_interventions_source": sha256_file(INTERVENTION_PATH),
            "producer_source": sha256_file(PRODUCER_PATH),
            "validator_source": sha256_file(Path(__file__).resolve()),
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
    payload = canonical_bytes(receipt)
    if not args.preflight:
        try:
            _exclusive_write(args.receipt.resolve(), payload)
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
