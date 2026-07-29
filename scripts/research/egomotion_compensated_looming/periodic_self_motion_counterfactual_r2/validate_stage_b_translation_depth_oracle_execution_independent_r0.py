"""Independent Stage B scientific reconstruction and routing.

This module intentionally does not import the Stage B runner, R3, RCLE
evaluation, tracking, or local-fit implementation.  It reconstructs local
affine fits from the sealed paired-track arrays and is the only component that
writes the Stage B routing decision.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_"
    "TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0"
)
ARMS = (
    "STATIC_SCENE",
    "EGO_ROTATION_STATIC_SCENE",
    "EGO_TRANSLATION_STATIC_SCENE",
    "OBJECT_APPROACH_STATIC_CAMERA",
    "OBJECT_APPROACH_PLUS_EGO_6DOF",
)
POSITIVE_ARMS = (
    "OBJECT_APPROACH_STATIC_CAMERA",
    "OBJECT_APPROACH_PLUS_EGO_6DOF",
)
WIDTH = 360
HEIGHT = 640
PAIR_COUNT = 601
THRESHOLD = 0.01
SPEC_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_EXECUTABLE_SPEC_R1_"
    "2026-07-29.json"
)
IDENTITY_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0_"
    "IDENTITY_LOCK_2026-07-29.json"
)
ACTIVATION_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_EXECUTION_ACTIVATION_R1_"
    "2026-07-29.json"
)
PROTOCOL_RELATIVE = (
    "scripts/research/egomotion_compensated_looming/configs/"
    "phase_a_synthetic_signal_audit_r0.json"
)
DEFAULT_ROOT = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_stage_b_translation_depth_oracle_object_approach_r0"
)
NUMERIC_REPRESENTATION_AMENDMENT_RELATIVE = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_"
    "STAGE_B_INDEPENDENT_VALIDATOR_NUMERIC_REPRESENTATION_AMENDMENT_R0_"
    "2026-07-29.json"
)


class InvalidExecution(RuntimeError):
    pass


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
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else item.item(),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def require(condition: bool, label: str) -> None:
    if not condition:
        raise InvalidExecution(label)


def close(left: float | None, right: float | None, label: str) -> None:
    if left is None or right is None:
        require(left is None and right is None, label)
        return
    require(
        math.isfinite(float(left))
        and math.isfinite(float(right))
        and abs(float(left) - float(right)) <= 2e-7,
        label,
    )


def cell_bounds(index: int) -> tuple[int, int, int, int]:
    row, column = divmod(index, 3)
    return (
        int(round(column * WIDTH / 3)),
        int(round(row * HEIGHT / 3)),
        int(round((column + 1) * WIDTH / 3)),
        int(round((row + 1) * HEIGHT / 3)),
    )


def fit_cells(
    previous: np.ndarray,
    current: np.ndarray,
    dt: float,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    velocity = (current.astype(np.float64) - previous.astype(np.float64)) / dt
    for index in range(9):
        x0, y0, x1, y1 = cell_bounds(index)
        selected = (
            (previous[:, 0] >= x0)
            & (previous[:, 0] < x1)
            & (previous[:, 1] >= y0)
            & (previous[:, 1] < y1)
        )
        points = previous[selected].astype(np.float64)
        cell_velocity = velocity[selected]
        support = len(points)
        result: dict[str, Any] = {
            "cell_index": index,
            "support_count": support,
            "evaluable": False,
            "expansion": None,
            "residual": None,
            "coefficients": None,
        }
        if support < int(parameters["minimum_tracks_per_cell"]):
            results.append(result)
            continue
        hull = cv2.convexHull(points.astype(np.float32))
        hull_fraction = float(cv2.contourArea(hull)) / max(
            float((x1 - x0) * (y1 - y0)), 1.0
        )
        if hull_fraction < float(
            parameters["minimum_track_convex_hull_fraction"]
        ):
            results.append(result)
            continue
        center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        half_width, half_height = max(0.5 * (x1 - x0), 1.0), max(
            0.5 * (y1 - y0), 1.0
        )
        design = np.column_stack(
            (
                (points[:, 0] - center_x) / half_width,
                (points[:, 1] - center_y) / half_height,
                np.ones(support),
            )
        )
        condition = float(np.linalg.cond(design))
        if (
            not math.isfinite(condition)
            or condition
            > float(parameters["maximum_design_condition_number"])
        ):
            results.append(result)
            continue
        coefficients, _, _, _ = np.linalg.lstsq(
            design, cell_velocity, rcond=None
        )
        residual = float(
            np.median(np.linalg.norm(design @ coefficients - cell_velocity, axis=1))
            * dt
        )
        if residual > float(
            parameters["maximum_median_fit_residual_pixels_per_frame"]
        ):
            results.append(result)
            continue
        expansion = float(
            0.5
            * (
                coefficients[0, 0] / half_width
                + coefficients[1, 1] / half_height
            )
        )
        require(math.isfinite(expansion), "NONFINITE_RECOMPUTED_EXPANSION")
        result.update(
            {
                "evaluable": True,
                "expansion": expansion,
                "residual": residual,
                "coefficients": coefficients.tolist(),
            }
        )
        results.append(result)
    return results


def recompute_pair(
    previous: np.ndarray,
    baseline: np.ndarray,
    oracle: np.ndarray,
    target: np.ndarray,
    dt: float,
    r3_evaluable: bool,
    parameters: dict[str, Any],
    target_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = target.astype(bool) if target_only else np.ones(len(target), dtype=bool)
    left_cells = fit_cells(previous[selected], baseline[selected], dt, parameters)
    right_cells = fit_cells(previous[selected], oracle[selected], dt, parameters)
    common = [
        index
        for index in range(9)
        if left_cells[index]["evaluable"] and right_cells[index]["evaluable"]
    ]
    minimum = 1 if target_only else 5
    evaluable = r3_evaluable and len(common) >= minimum
    result: dict[str, Any] = {
        "evaluable": evaluable,
        "common_cell_indices": common,
        "baseline_signed_per_s": None,
        "oracle_signed_per_s": None,
        "baseline_absolute_per_s": None,
        "oracle_absolute_per_s": None,
    }
    if evaluable:
        left = np.asarray(
            [left_cells[index]["expansion"] for index in common],
            dtype=np.float64,
        )
        right = np.asarray(
            [right_cells[index]["expansion"] for index in common],
            dtype=np.float64,
        )
        result.update(
            {
                "baseline_signed_per_s": float(np.median(left)),
                "oracle_signed_per_s": float(np.median(right)),
                "baseline_absolute_per_s": float(np.median(np.abs(left))),
                "oracle_absolute_per_s": float(np.median(np.abs(right))),
            }
        )
    return result, left_cells, right_cells


def compare_pair(
    recorded: dict[str, Any],
    recomputed: dict[str, Any],
    left_cells: list[dict[str, Any]],
    right_cells: list[dict[str, Any]],
    channel: str,
) -> None:
    source = recorded[channel]
    require(source["evaluable"] is recomputed["evaluable"], "PAIR_EVALUABLE")
    require(
        source["common_cell_indices"] == recomputed["common_cell_indices"],
        "PAIR_COMMON_CELLS",
    )
    for key in (
        "baseline_signed_per_s",
        "oracle_signed_per_s",
        "baseline_absolute_per_s",
        "oracle_absolute_per_s",
    ):
        close(source.get(key), recomputed.get(key), f"PAIR_SCALAR:{key}")
    for index in range(9):
        for name, cells in (
            ("baseline", left_cells),
            ("oracle", right_cells),
        ):
            recorded_cell = source[f"{name}_cells"][index]
            require(
                recorded_cell["evaluable"] is cells[index]["evaluable"],
                "CELL_EVALUABLE",
            )
            close(
                recorded_cell.get("expansion"),
                cells[index].get("expansion"),
                "CELL_EXPANSION",
            )
            if cells[index]["evaluable"]:
                close(
                    recorded_cell.get(
                        "fit_residual_pixels_per_frame"
                    ),
                    cells[index].get("residual"),
                    "CELL_FIT_RESIDUAL",
                )
            audit = source[f"{name}_cell_audit"][index]
            require(audit["support_count"] == cells[index]["support_count"], "CELL_SUPPORT")
            coefficients = audit.get("coefficients")
            if coefficients is not None:
                coefficient_array = np.asarray(coefficients, dtype=np.float64)
                require(
                    coefficient_array.shape == (3, 2)
                    and np.isfinite(coefficient_array).all(),
                    "CELL_AUDIT_COEFFICIENTS_FINITE",
                )
                x0, y0, x1, y1 = cell_bounds(index)
                half_width = max(0.5 * (x1 - x0), 1.0)
                half_height = max(0.5 * (y1 - y0), 1.0)
                audit_expansion = float(
                    0.5
                    * (
                        coefficient_array[0, 0] / half_width
                        + coefficient_array[1, 1] / half_height
                    )
                )
                close(
                    audit.get("expansion_from_coefficients_per_s"),
                    audit_expansion,
                    "CELL_AUDIT_EXPANSION_SELF_CONSISTENCY",
                )
                residual = audit.get(
                    "median_fit_residual_pixels_per_frame"
                )
                require(
                    residual is not None
                    and math.isfinite(float(residual))
                    and float(residual) >= 0.0,
                    "CELL_AUDIT_RESIDUAL_FINITE",
                )


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values), probability, method="linear"))


def triggers(
    pairs: list[dict[str, Any]], channel: str, metric: str
) -> tuple[int, int]:
    streak = 0
    count = 0
    longest = 0
    for pair in pairs:
        record = pair[channel]
        value = record.get(metric) if record["evaluable"] else None
        if value is None:
            streak = 0
        elif float(value) > THRESHOLD:
            streak += 1
            longest = max(longest, streak)
            if streak >= 3:
                count += 1
        else:
            streak = 0
    return count, longest


def reduce_pairs(
    pairs: list[dict[str, Any]], channel: str
) -> dict[str, Any]:
    evaluable = [pair[channel] for pair in pairs if pair[channel]["evaluable"]]
    fields = {}
    for side in ("baseline", "oracle"):
        for metric in ("signed", "absolute"):
            values = [
                float(item[f"{side}_{metric}_per_s"]) for item in evaluable
            ]
            fields[f"{side}_{metric}_p50_per_s"] = quantile(values, 0.5)
            fields[f"{side}_{metric}_p90_per_s"] = quantile(values, 0.9)
        count, longest = triggers(
            pairs, channel, f"{side}_signed_per_s"
        )
        fields[f"{side}_three_pair_trigger_count"] = count
        fields[f"{side}_three_pair_trigger_density_fixed"] = count / PAIR_COUNT
        fields[f"{side}_longest_positive_streak"] = longest
    return {
        "channel": channel,
        "planned_pair_count": PAIR_COUNT,
        "paired_evaluable_pair_count": len(evaluable),
        "paired_evaluable_fraction": len(evaluable) / PAIR_COUNT,
        **fields,
    }


def compare_reduction(recorded: dict[str, Any], actual: dict[str, Any]) -> None:
    require(recorded.keys() == actual.keys(), "REDUCTION_KEYS")
    for key, value in actual.items():
        if isinstance(value, float) or value is None:
            close(recorded[key], value, f"REDUCTION:{key}")
        else:
            require(recorded[key] == value, f"REDUCTION:{key}")


def validate(output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = repo_root()
    response = output_root / "response"
    run_path = response / "run_receipt.json"
    analysis_path = response / "analysis_result.json"
    receipt_path = response / "independent_validation_receipt.json"
    decision_path = response / "execution_decision.json"
    for path in (analysis_path, receipt_path, decision_path):
        require(not path.exists(), f"OUTPUT_PREEXISTS:{path.name}")
    activation_path = root / ACTIVATION_RELATIVE
    require(activation_path.is_file(), "ACTIVATION_MISSING")
    activation = load_json(activation_path)
    spec = load_json(root / SPEC_RELATIVE)
    identity = load_json(root / IDENTITY_RELATIVE)
    run = load_json(run_path)
    protocol = load_json(root / PROTOCOL_RELATIVE)
    numeric_amendment_path = root / NUMERIC_REPRESENTATION_AMENDMENT_RELATIVE
    numeric_amendment = load_json(numeric_amendment_path)
    require(
        numeric_amendment.get("classification")
        == "THIN_PROTOCOL_ONLY_VALIDATOR_REPAIR / SCIENTIFIC_LEDGER_UNCHANGED",
        "NUMERIC_REPRESENTATION_AMENDMENT",
    )
    require(
        activation["stage_b_execution_authorized"] is True
        and activation["stage_b_response_access_authorized"] is True,
        "ACTIVATION_FALSE",
    )
    runner_source = root / activation["runner_source_path"]
    require(
        runner_source.is_file()
        and sha256_file(runner_source) == activation["runner_source_sha256"],
        "RUNNER_BINDING",
    )
    require(run["activation_sha256"] == sha256_file(activation_path), "RUN_ACTIVATION")
    require(run["cluster_count"] == 8 and run["sequence_count"] == 40, "RUN_COUNTS")
    require(run["planned_pair_count"] == 40 * PAIR_COUNT, "RUN_PAIR_COUNT")
    require(run["launch_and_refill_gate_bytes"] == 6 * 1024**3, "RUN_RAM6")
    require(run["in_flight_emergency_floor_bytes"] == 4 * 1024**3, "RUN_RAM4")
    require(run["available_ram_at_launch_bytes"] >= 6 * 1024**3, "RUN_LAUNCH_RAM")
    require(run["minimum_available_ram_bytes"] >= 4 * 1024**3, "RUN_MIN_RAM")
    require(
        run["formal_480_plus_16_sequences_run"] == 0
        and run["formal_r3_pair_core_calls"] == 0
        and run["formal_firewall_before"] == run["formal_firewall_after"]
        and run["successor_formal_path_absent"] is True,
        "FORMAL_FIREWALL",
    )
    require(not run["residual_worker_pids"], "RESIDUAL_WORKERS")
    parameters = protocol["local_affine"]
    recomputed_arms: dict[str, dict[str, Any]] = {}
    pair_count = 0
    cell_fit_count = 0
    for cluster in identity["clusters"]:
        cluster_dir = response / "clusters" / cluster["cluster_id"]
        require(cluster_dir.is_dir(), "CLUSTER_MISSING")
        for arm in ARMS:
            arm_dir = cluster_dir / arm
            ledger_path = arm_dir / "pair_ledger.jsonl"
            tracks_path = arm_dir / "paired_tracks.npz"
            metrics_path = arm_dir / "reduced_metrics.json"
            receipt = load_json(arm_dir / "receipt.json")
            require(receipt["pair_ledger_sha256"] == sha256_file(ledger_path), "LEDGER_HASH")
            require(receipt["paired_tracks_sha256"] == sha256_file(tracks_path), "TRACK_HASH")
            require(receipt["reduced_metrics_sha256"] == sha256_file(metrics_path), "METRIC_HASH")
            rows = load_jsonl(ledger_path)
            require(len(rows) == PAIR_COUNT, "LEDGER_PAIR_COUNT")
            with np.load(tracks_path) as arrays:
                offsets = arrays["offsets"]
                previous = arrays["previous"]
                baseline = arrays["baseline"]
                oracle = arrays["oracle"]
                target = arrays["target"]
            require(len(offsets) == PAIR_COUNT + 1, "TRACK_OFFSETS")
            require(offsets[0] == 0 and offsets[-1] == len(previous), "TRACK_OFFSET_RANGE")
            require(
                len(previous) == len(baseline) == len(oracle) == len(target),
                "TRACK_LENGTHS",
            )
            pairs = []
            max_displacement = 0.0
            for index, row in enumerate(rows):
                require(row["pair_index"] == index, "PAIR_ORDER")
                left, right = int(offsets[index]), int(offsets[index + 1])
                full, full_left, full_right = recompute_pair(
                    previous[left:right],
                    baseline[left:right],
                    oracle[left:right],
                    target[left:right],
                    float(row["dt_s"]),
                    bool(row["r3_pair_evaluable"]),
                    parameters,
                    False,
                )
                target_result, target_left, target_right = recompute_pair(
                    previous[left:right],
                    baseline[left:right],
                    oracle[left:right],
                    target[left:right],
                    float(row["dt_s"]),
                    bool(row["r3_pair_evaluable"]) and full["evaluable"],
                    parameters,
                    True,
                )
                compare_pair(row, full, full_left, full_right, "full_scene")
                compare_pair(
                    row,
                    target_result,
                    target_left,
                    target_right,
                    "target_mask",
                )
                pairs.append(
                    {"full_scene": full, "target_mask": target_result}
                )
                max_displacement = max(
                    max_displacement,
                    float(row["oracle_displacement_max_abs_px"]),
                )
                pair_count += 1
                cell_fit_count += 36
            reduced = {
                "cluster_id": cluster["cluster_id"],
                "block": cluster["block"],
                "ordinal": cluster["ordinal"],
                "arm": arm,
                "full_scene": reduce_pairs(pairs, "full_scene"),
                "target_mask": reduce_pairs(pairs, "target_mask"),
                "oracle_displacement_max_abs_px": max_displacement,
            }
            recorded_metrics = load_json(metrics_path)
            require(
                {
                    key: recorded_metrics[key]
                    for key in ("cluster_id", "block", "ordinal", "arm")
                }
                == {
                    key: reduced[key]
                    for key in ("cluster_id", "block", "ordinal", "arm")
                },
                "METRIC_IDENTITY",
            )
            compare_reduction(
                recorded_metrics["full_scene"], reduced["full_scene"]
            )
            compare_reduction(
                recorded_metrics["target_mask"], reduced["target_mask"]
            )
            close(
                recorded_metrics["oracle_displacement_max_abs_px"],
                max_displacement,
                "ORACLE_DISPLACEMENT_MAX",
            )
            recomputed_arms[f"{cluster['cluster_id']}|{arm}"] = reduced

    coverage_failures = []
    rotation_failures = []
    clusters_out = []
    signed_positive = 0
    signed_suppressions = []
    absolute_go = 0
    positive_pass = Counter()
    for cluster in identity["clusters"]:
        cluster_result: dict[str, Any] = {
            "cluster_id": cluster["cluster_id"],
            "block": cluster["block"],
            "ordinal": cluster["ordinal"],
            "arms": {},
        }
        for arm in ARMS:
            item = recomputed_arms[f"{cluster['cluster_id']}|{arm}"]
            coverage_channel = (
                "target_mask" if arm in POSITIVE_ARMS else "full_scene"
            )
            coverage = item[coverage_channel]["paired_evaluable_fraction"]
            if coverage < 0.75:
                coverage_failures.append(
                    f"{cluster['cluster_id']}:{arm}:{coverage_channel}"
                )
            cluster_result["arms"][arm] = item
        rotation = cluster_result["arms"]["EGO_ROTATION_STATIC_SCENE"]
        rotation_full = rotation["full_scene"]
        rotation_values_present = all(
            rotation_full[key] is not None
            for key in (
                "baseline_absolute_p90_per_s",
                "oracle_absolute_p90_per_s",
            )
        )
        rotation_pass = (
            rotation_values_present
            and
            rotation["oracle_displacement_max_abs_px"] <= 1e-12
            and abs(
                rotation_full["baseline_absolute_p90_per_s"]
                - rotation_full["oracle_absolute_p90_per_s"]
            )
            <= 1e-12
            and rotation_full["oracle_absolute_p90_per_s"] <= 0.01
            and rotation_full["oracle_three_pair_trigger_density_fixed"] == 0.0
        )
        if not rotation_pass:
            rotation_failures.append(cluster["cluster_id"])
        translation = cluster_result["arms"]["EGO_TRANSLATION_STATIC_SCENE"][
            "full_scene"
        ]
        translation_values_present = all(
            translation[key] is not None
            for key in (
                "baseline_signed_p90_per_s",
                "oracle_signed_p90_per_s",
                "baseline_absolute_p90_per_s",
                "oracle_absolute_p90_per_s",
            )
        )
        if translation_values_present:
            signed = (
                translation["baseline_signed_p90_per_s"]
                - translation["oracle_signed_p90_per_s"]
            )
            absolute = (
                translation["baseline_absolute_p90_per_s"]
                - translation["oracle_absolute_p90_per_s"]
            ) / max(translation["baseline_absolute_p90_per_s"], 1e-12)
            signed_suppressions.append(signed)
            signed_positive += int(signed > 0.0)
            absolute_go += int(absolute >= 0.5)
        else:
            signed = None
            absolute = None
        cluster_result["translation_estimands"] = {
            "signed_suppression_per_s": signed,
            "absolute_leakage_suppression_fraction": absolute,
        }
        cluster_result["positive_controls"] = {}
        for arm in POSITIVE_ARMS:
            positive = cluster_result["arms"][arm]["target_mask"]
            baseline_p90 = positive["baseline_signed_p90_per_s"]
            oracle_p90 = positive["oracle_signed_p90_per_s"]
            established = (
                baseline_p90 is not None and baseline_p90 > 0.01
            )
            ratio = oracle_p90 / baseline_p90 if established else None
            passed = (
                established
                and oracle_p90 is not None
                and oracle_p90 > 0.01
                and ratio is not None
                and ratio >= 0.8
            )
            positive_pass[arm] += int(passed)
            cluster_result["positive_controls"][arm] = {
                "baseline_signed_p90_per_s": baseline_p90,
                "oracle_signed_p90_per_s": oracle_p90,
                "retention_ratio": ratio,
                "pass": passed,
            }
        cluster_result["rotation_boundary_pass"] = rotation_pass
        clusters_out.append(cluster_result)
    median_signed = (
        float(np.median(np.asarray(signed_suppressions)))
        if signed_suppressions
        else None
    )
    if coverage_failures or rotation_failures:
        route = "B_ORACLE_NOT_EVALUABLE"
    elif (
        any(positive_pass[arm] < 6 for arm in POSITIVE_ARMS)
        or signed_positive < 4
    ):
        route = "STOP_OR_DOWNGRADE_RCLE"
    elif not (
        signed_positive >= 6
        and median_signed is not None
        and median_signed > 0.0
        and absolute_go >= 6
        and all(positive_pass[arm] >= 6 for arm in POSITIVE_ARMS)
    ):
        route = "FREEZE_AS_CONDITIONAL_RESIDUAL_FEATURE"
    else:
        route = "GO_SINGLE_TARGETED_UPGRADE"
    analysis = {
        "schema": "rcle.stage_b.analysis_result.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "analysis_unit": "cluster",
        "cluster_count": 8,
        "sequence_count": 40,
        "pair_count_recomputed": pair_count,
        "cell_fits_recomputed": cell_fit_count,
        "coverage_failures": coverage_failures,
        "rotation_failures": rotation_failures,
        "translation": {
            "signed_positive_clusters": signed_positive,
            "median_signed_suppression_per_s": median_signed,
            "absolute_suppression_at_least_0_5_clusters": absolute_go,
        },
        "positive_control_pass_counts": dict(positive_pass),
        "clusters": clusters_out,
        "route": route,
        "automatic_entry_to_c_or_d": False,
        "formal_480_plus_16_consumed": False,
        "terminal": f"STAGE_B_ANALYSIS_COMPLETE / {route}",
    }
    write_exclusive(analysis_path, analysis)
    receipt = {
        "schema": "rcle.stage_b.independent_validation_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "valid": True,
        "run_receipt_sha256": sha256_file(run_path),
        "analysis_result_sha256": sha256_file(analysis_path),
        "validator_source_path": Path(__file__).resolve().relative_to(root).as_posix(),
        "validator_source_sha256": sha256_file(Path(__file__).resolve()),
        "numeric_representation_amendment_path": (
            NUMERIC_REPRESENTATION_AMENDMENT_RELATIVE
        ),
        "numeric_representation_amendment_sha256": sha256_file(
            numeric_amendment_path
        ),
        "checks": {
            "bindings_and_activation": "PASS",
            "6gib_launch_refill_and_4gib_floor": "PASS",
            "40_sequence_completeness": "PASS",
            "sealed_track_hashes": "PASS",
            "independent_local_affine_reconstruction": "PASS",
            "float64_audit_coefficient_self_consistency": "PASS",
            "signed_vs_absolute_reduction": "PASS",
            "type7_quantiles": "PASS",
            "fixed_601_trigger_denominator": "PASS",
            "rotation_precedence": "PASS",
            "formal_firewall": "PASS",
        },
        "pair_count_recomputed": pair_count,
        "cell_fits_recomputed": cell_fit_count,
        "route": route,
        "terminal": "STAGE_B_INDEPENDENT_VALIDATION_PASS / VALID",
    }
    write_exclusive(receipt_path, receipt)
    decision = {
        "schema": "rcle.stage_b.execution_decision.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "decision": route,
        "analysis_result_sha256": sha256_file(analysis_path),
        "independent_receipt_sha256": sha256_file(receipt_path),
        "rotation_boundary_required_and_precedence_applied": True,
        "automatic_entry_to_feature_contract_c": False,
        "automatic_entry_to_fusion_experiment_d": False,
        "automatic_algorithm_change": False,
        "formal_480_plus_16_authority_consumed": False,
        "retry_replacement_or_reseed_authorized": False,
        "terminal": f"STAGE_B_EXECUTION_DECISION / {route}",
    }
    write_exclusive(decision_path, decision)
    return analysis, receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = repo_root()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (root / DEFAULT_ROOT).resolve()
    )
    analysis, receipt = validate(output_root)
    print(
        json.dumps(
            {
                "route": analysis["route"],
                "terminal": receipt["terminal"],
                "pair_count_recomputed": receipt["pair_count_recomputed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
