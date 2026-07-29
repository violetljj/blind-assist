"""Frozen P3 analysis implementation for RCLE periodic counterfactual R2.

This module is deliberately outcome-blind in P3: it exposes pure validation and
analysis functions plus synthetic fixtures.  It has no CLI that can discover or
read formal R2 outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any, Iterable

import numpy as np


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
ANALYSIS_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
QUALITIES = ("CLEAN", "BLUR", "LOW_TEXTURE")
ARMS = tuple(f"{motion}__{quality}" for motion in MOTIONS for quality in QUALITIES)
FAMILY = (
    "MOTION_CLEAN",
    "BLUR_STATIC",
    "LOW_TEXTURE_STATIC",
    "MOTION_X_BLUR",
    "MOTION_X_LOW_TEXTURE",
    "MOTION_BLUR_VS_STATIC_CLEAN",
    "MOTION_LOW_TEXTURE_VS_STATIC_CLEAN",
    "BLUR_FAILURE_UNION_STATIC",
    "LOW_TEXTURE_FAILURE_UNION_STATIC",
)
PAIR_COUNT = 601
THRESHOLD = 0.01
REQUIRED_CONSECUTIVE = 3
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_REPLICATES = 20_000


class InvalidAnalysis(ValueError):
    """The frozen analysis contract was violated."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
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


def reduce_pair_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recompute fixed-denominator trigger and failure-union densities."""

    materialized = list(rows)
    if len(materialized) != PAIR_COUNT:
        raise InvalidAnalysis(f"PAIR_COUNT:{len(materialized)}")
    expected = list(range(PAIR_COUNT))
    actual = [row.get("pair_index") for row in materialized]
    if actual != expected:
        raise InvalidAnalysis("PAIR_ORDER_OR_IDENTITY")
    streak = 0
    trigger_count = 0
    failure_count = 0
    evaluable_count = 0
    for row in materialized:
        evaluable = row.get("evaluable")
        response = row.get("compensated_expansion_median_per_s")
        if evaluable is True:
            if not isinstance(response, (int, float)) or not math.isfinite(response):
                raise InvalidAnalysis("EVALUABLE_RESPONSE_INVALID")
            evaluable_count += 1
            streak = streak + 1 if float(response) > THRESHOLD else 0
        elif evaluable is False:
            if response is not None:
                raise InvalidAnalysis("ABSTENTION_RESPONSE_PRESENT")
            streak = 0
        else:
            raise InvalidAnalysis("EVALUABLE_FLAG_INVALID")
        recomputed_trigger = streak >= REQUIRED_CONSECUTIVE
        if "compensated_three_pair_trigger" in row and (
            row["compensated_three_pair_trigger"] is not recomputed_trigger
        ):
            raise InvalidAnalysis("FORGED_THREE_PAIR_TRIGGER")
        trigger_count += int(recomputed_trigger)

        detected = row.get("detected_feature_count")
        consistent = row.get("forward_backward_consistent_count")
        fraction = row.get("forward_backward_consistent_fraction")
        fb_error = row.get("median_forward_backward_error_px")
        numeric = (detected, consistent, fraction)
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in numeric
        ):
            raise InvalidAnalysis("TRACKING_DIAGNOSTIC_INVALID")
        collapse = detected < 60 or consistent < 60 or fraction < 0.50
        fb_failure = (
            fb_error is None
            or not isinstance(fb_error, (int, float))
            or not math.isfinite(fb_error)
            or fb_error > 0.75
        )
        failure_count += int(collapse or fb_failure)
    return {
        "scheduled_pair_count": PAIR_COUNT,
        "evaluable_pair_count": evaluable_count,
        "trigger_count": trigger_count,
        "trigger_density": trigger_count / PAIR_COUNT,
        "quality_failure_union_count": failure_count,
        "quality_failure_union_density": failure_count / PAIR_COUNT,
    }


def _finite_unit_interval(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidAnalysis(f"{label}_NONFINITE")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise InvalidAnalysis(f"{label}_OUTSIDE_UNIT_INTERVAL")
    return result


def validate_clusters(clusters: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized = list(clusters)
    if len(materialized) != 80:
        raise InvalidAnalysis(f"CLUSTER_COUNT:{len(materialized)}")
    seen: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for cluster in materialized:
        block = cluster.get("block")
        ordinal = cluster.get("ordinal")
        if block not in BLOCKS or not isinstance(ordinal, int) or ordinal not in range(20):
            raise InvalidAnalysis("CLUSTER_IDENTITY")
        identity = (block, ordinal)
        if identity in seen:
            raise InvalidAnalysis("DUPLICATE_CLUSTER")
        seen.add(identity)
        arms = cluster.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise InvalidAnalysis("SIX_ARM_KEYSET")
        normalized_arms: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            payload = arms[arm]
            if not isinstance(payload, dict):
                raise InvalidAnalysis("ARM_PAYLOAD")
            normalized_arms[arm] = {
                "trigger_density": _finite_unit_interval(
                    payload.get("trigger_density"), "TRIGGER_DENSITY"
                ),
                "quality_failure_union_density": _finite_unit_interval(
                    payload.get("quality_failure_union_density"),
                    "FAILURE_UNION_DENSITY",
                ),
            }
        normalized.append(
            {"block": block, "ordinal": ordinal, "arms": normalized_arms}
        )
    if seen != {(block, ordinal) for block in BLOCKS for ordinal in range(20)}:
        raise InvalidAnalysis("FROZEN_CLUSTER_GRID")
    return sorted(normalized, key=lambda item: (BLOCKS.index(item["block"]), item["ordinal"]))


def unit_contrasts(cluster: dict[str, Any]) -> dict[str, float]:
    arms = cluster["arms"]

    def y(motion: str, quality: str) -> float:
        return arms[f"{motion}__{quality}"]["trigger_density"]

    def failure(motion: str, quality: str) -> float:
        return arms[f"{motion}__{quality}"]["quality_failure_union_density"]

    motion_clean = y("PERIODIC_6DOF_SELF_MOTION", "CLEAN") - y(
        "STATIC_CAMERA", "CLEAN"
    )
    return {
        "MOTION_CLEAN": motion_clean,
        "BLUR_STATIC": y("STATIC_CAMERA", "BLUR") - y("STATIC_CAMERA", "CLEAN"),
        "LOW_TEXTURE_STATIC": y("STATIC_CAMERA", "LOW_TEXTURE")
        - y("STATIC_CAMERA", "CLEAN"),
        "MOTION_X_BLUR": (
            y("PERIODIC_6DOF_SELF_MOTION", "BLUR")
            - y("STATIC_CAMERA", "BLUR")
            - motion_clean
        ),
        "MOTION_X_LOW_TEXTURE": (
            y("PERIODIC_6DOF_SELF_MOTION", "LOW_TEXTURE")
            - y("STATIC_CAMERA", "LOW_TEXTURE")
            - motion_clean
        ),
        "MOTION_BLUR_VS_STATIC_CLEAN": y(
            "PERIODIC_6DOF_SELF_MOTION", "BLUR"
        )
        - y("STATIC_CAMERA", "CLEAN"),
        "MOTION_LOW_TEXTURE_VS_STATIC_CLEAN": y(
            "PERIODIC_6DOF_SELF_MOTION", "LOW_TEXTURE"
        )
        - y("STATIC_CAMERA", "CLEAN"),
        "BLUR_FAILURE_UNION_STATIC": failure("STATIC_CAMERA", "BLUR")
        - failure("STATIC_CAMERA", "CLEAN"),
        "LOW_TEXTURE_FAILURE_UNION_STATIC": failure(
            "STATIC_CAMERA", "LOW_TEXTURE"
        )
        - failure("STATIC_CAMERA", "CLEAN"),
    }


def analyze(
    clusters: Iterable[dict[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the frozen shared-resample, equal-block, nine-member max-t analysis."""

    if replicates != BOOTSTRAP_REPLICATES or seed != BOOTSTRAP_SEED:
        raise InvalidAnalysis("BOOTSTRAP_LOCK")
    normalized = validate_clusters(clusters)
    values = np.empty((4, 20, len(FAMILY)), dtype=np.float64)
    for block_index, block in enumerate(BLOCKS):
        by_ordinal = {
            item["ordinal"]: unit_contrasts(item)
            for item in normalized
            if item["block"] == block
        }
        for ordinal in range(20):
            values[block_index, ordinal] = [
                by_ordinal[ordinal][name] for name in FAMILY
            ]
    if not np.isfinite(values).all():
        raise InvalidAnalysis("UNIT_CONTRAST_NONFINITE")
    block_points = values.mean(axis=1)
    theta = block_points.mean(axis=0)

    rng = np.random.default_rng(seed)
    # One (replicate, block, ordinal-slot) draw matrix is shared by all arms and
    # all nine contrasts.
    draws = rng.integers(0, 20, size=(replicates, 4, 20), endpoint=False)
    estimates = np.empty((replicates, len(FAMILY)), dtype=np.float64)
    for block_index in range(4):
        sampled = values[block_index][draws[:, block_index]]
        block_estimate = sampled.mean(axis=1)
        estimates[:] = block_estimate / 4.0 if block_index == 0 else (
            estimates + block_estimate / 4.0
        )
    if not np.isfinite(estimates).all():
        raise InvalidAnalysis("BOOTSTRAP_NONFINITE")
    sd = estimates.std(axis=0, ddof=1)
    zero = sd == 0.0
    for index in np.flatnonzero(zero):
        if not np.all(values[:, :, index] == values[0, 0, index]):
            raise InvalidAnalysis(f"ZERO_SD_INCONSISTENT:{FAMILY[index]}")
    nonzero = ~zero
    if nonzero.any():
        z = (estimates[:, nonzero] - theta[nonzero]) / sd[nonzero]
        critical_samples = np.max(np.abs(z), axis=1)
        critical = float(np.quantile(critical_samples, 0.95, method="linear"))
    else:
        critical = 0.0
    if not math.isfinite(critical):
        raise InvalidAnalysis("CRITICAL_VALUE_NONFINITE")

    estimands: dict[str, Any] = {}
    for index, name in enumerate(FAMILY):
        lower = float(theta[index] - critical * sd[index])
        upper = float(theta[index] + critical * sd[index])
        blocks_at_material = int(np.sum((block_points[:, index] >= 0.10) & (block_points[:, index] > 0)))
        classification = (
            "SUPPORTED"
            if theta[index] >= 0.10 and lower > 0.0 and blocks_at_material >= 3
            else (
                "RULED_OUT_AS_MATERIAL"
                if upper < 0.10
                else "INCONCLUSIVE"
            )
        )
        estimands[name] = {
            "theta": float(theta[index]),
            "bootstrap_sd_ddof1": float(sd[index]),
            "simultaneous_interval": [lower, upper],
            "block_point_estimates": block_points[:, index].tolist(),
            "blocks_positive_at_least_0_10": blocks_at_material,
            "classification": classification,
        }
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_analysis.v1",
        "protocol_id": PROTOCOL_ID,
        "analysis_id": ANALYSIS_ID,
        "input_role": "SYNTHETIC_FIXTURE_OR_FUTURE_FORMAL_ONLY",
        "cluster_count": 80,
        "family": list(FAMILY),
        "bootstrap": {
            "seed": seed,
            "replicates": replicates,
            "bit_generator": type(rng.bit_generator).__name__,
            "shared_draw_matrix_sha256": hashlib.sha256(
                draws.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "sd_ddof": 1,
            "critical_quantile": 0.95,
            "quantile_method": "linear_type_7",
            "critical_value": critical,
        },
        "estimands": estimands,
    }


def fixture_clusters() -> list[dict[str, Any]]:
    """Deterministic non-scientific fixture used only by P3 mutation tests."""

    result: list[dict[str, Any]] = []
    for block_index, block in enumerate(BLOCKS):
        for ordinal in range(20):
            base = 0.15 + block_index * 0.01 + ordinal * 0.0001
            arms: dict[str, dict[str, float]] = {}
            for motion in MOTIONS:
                for quality in QUALITIES:
                    trigger = base
                    if motion == "PERIODIC_6DOF_SELF_MOTION":
                        trigger += 0.03
                    if quality == "BLUR":
                        trigger += 0.02
                    if quality == "LOW_TEXTURE":
                        trigger += 0.01
                    failure_union = 0.05 + (0.12 if quality == "BLUR" else 0.0)
                    failure_union += 0.11 if quality == "LOW_TEXTURE" else 0.0
                    arms[f"{motion}__{quality}"] = {
                        "trigger_density": trigger,
                        "quality_failure_union_density": failure_union,
                    }
            result.append({"block": block, "ordinal": ordinal, "arms": arms})
    return result


def implementation_lock() -> dict[str, Any]:
    result = analyze(fixture_clusters())
    root = Path(__file__).resolve().parents[4]
    test_path = (
        root
        / "scripts/research/egomotion_compensated_looming/"
        "tests_periodic_self_motion_counterfactual_r2/"
        "test_p3_analysis_mutations.py"
    )
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_analysis_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "analysis_id": ANALYSIS_ID,
        "terminal": "ANALYSIS_IMPLEMENTATION_AND_FIXTURE_VALID / PREFLIGHT_ONLY",
        "formal_input_read": False,
        "formal_execution_authorized": False,
        "scientific_outcome_interpreted": False,
        "frozen_contract": {
            "blocks": list(BLOCKS),
            "clusters_per_block": 20,
            "arms": list(ARMS),
            "pair_count": PAIR_COUNT,
            "threshold_operator": "strict_greater_than",
            "threshold_per_s": THRESHOLD,
            "required_consecutive_pairs": REQUIRED_CONSECUTIVE,
            "family": list(FAMILY),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_sd_ddof": 1,
            "quantile_method": "linear_type_7",
            "overall_weighting": "equal_four_block_mean",
            "resampling": "one_shared_cluster_draw_matrix_for_all_nine",
        },
        "fixture": {
            "role": "NON_SCIENTIFIC_MUTATION_TEST_FIXTURE",
            "result_sha256": hashlib.sha256(canonical_bytes(result)).hexdigest(),
            "critical_value": result["bootstrap"]["critical_value"],
            "shared_draw_matrix_sha256": result["bootstrap"][
                "shared_draw_matrix_sha256"
            ],
            "estimand_theta": {
                name: result["estimands"][name]["theta"] for name in FAMILY
            },
        },
        "implementation": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
            "mutation_test_path": test_path.relative_to(root).as_posix(),
            "mutation_test_sha256": sha256_file(test_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-lock", type=Path, required=True)
    args = parser.parse_args()
    lock = implementation_lock()
    _write_exclusive(args.write_lock.resolve(), lock)
    print(
        json.dumps(
            {
                "terminal": lock["terminal"],
                "family_count": len(lock["frozen_contract"]["family"]),
                "formal_execution_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
