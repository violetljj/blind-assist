"""Independent validator for the RCLE R2 P4 formal result.

This module intentionally does not import the formal assembler, P3 analysis,
formal runner, transport adapter, or R3 pair core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
RESULT_SCHEMA = "rcle.periodic_self_motion_counterfactual.p4_formal_result.v1"
RECEIPT_SCHEMA = (
    "rcle.periodic_self_motion_counterfactual.p4_formal_validation_receipt.v1"
)
IDENTITY_SCHEMA = (
    "rcle.periodic_self_motion_counterfactual.p4_formal_identity_lock.v1"
)
BUNDLE_SCHEMA = "rcle.periodic_self_motion_counterfactual.p4_bundle_success.v1"
ARM_SCHEMA = "rcle.periodic_self_motion_counterfactual.p4_arm_receipt.v1"
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MAIN_ARMS = (
    "STATIC_CAMERA__CLEAN",
    "STATIC_CAMERA__BLUR",
    "STATIC_CAMERA__LOW_TEXTURE",
    "PERIODIC_6DOF_SELF_MOTION__CLEAN",
    "PERIODIC_6DOF_SELF_MOTION__BLUR",
    "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE",
)
GUARD_ARMS = (
    "MONOTONIC_APPROACH_ONLY__CLEAN",
    "MONOTONIC_APPROACH_PLUS_PERIODIC_6DOF__CLEAN",
)
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
FRAME_COUNT = 602
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_REPLICATES = 20_000
CLEAN_PAIR_MINIMUM = math.ceil(0.70 * PAIR_COUNT)


class ValidationError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(code: str) -> None:
    raise ValidationError(code)


def _finite(value: Any, code: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        _fail(code)
    return float(value)


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _seed(role: str, block: str, ordinal: int) -> int:
    namespace = "MAIN" if role == "MAIN_FACTORIAL" else "GUARD"
    token = f"{PROTOCOL_ID}|{namespace}|{block}|{ordinal:02d}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _expected_keys() -> set[tuple[str, str, int, str]]:
    result = set()
    for block in BLOCKS:
        for ordinal in range(20):
            for arm in MAIN_ARMS:
                result.add(("MAIN_FACTORIAL", block, ordinal, arm))
        for ordinal in range(2):
            for arm in GUARD_ARMS:
                result.add(("POSITIVE_GUARDRAIL", block, ordinal, arm))
    return result


def _identities(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        lock.get("schema") != IDENTITY_SCHEMA
        or lock.get("protocol_id") != PROTOCOL_ID
        or lock.get("formal_execution_authorized") is not False
    ):
        _fail("IDENTITY_HEADER")
    values = lock.get("identities")
    if not isinstance(values, list) or len(values) != 496:
        _fail("IDENTITY_COUNT")
    if hashlib.sha256(canonical_bytes(values)).hexdigest() != lock.get(
        "identity_set_sha256"
    ):
        _fail("IDENTITY_SET_HASH")
    expected = _expected_keys()
    seen: set[tuple[str, str, int, str]] = set()
    by_sequence: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            _fail("IDENTITY_OBJECT")
        key = (
            value.get("role"),
            value.get("block"),
            value.get("ordinal"),
            value.get("arm"),
        )
        if key not in expected or key in seen:
            _fail("IDENTITY_KEYSET")
        seen.add(key)
        sequence_id = value.get("sequence_id")
        if (
            not isinstance(sequence_id, str)
            or not sequence_id
            or sequence_id in by_sequence
            or not isinstance(value.get("cluster_id"), str)
            or not value["cluster_id"]
        ):
            _fail("SEQUENCE_ID")
        if (
            value.get("numeric_seed_uint64")
            != _seed(value["role"], value["block"], value["ordinal"])
            or value.get("frame_count") != FRAME_COUNT
            or value.get("pair_count") != PAIR_COUNT
            or not _is_sha(value.get("scene_geometry_sha256"))
            or not _is_sha(value.get("trajectory_sha256"))
        ):
            _fail("IDENTITY_BINDING")
        by_sequence[sequence_id] = value
    if seen != expected:
        _fail("IDENTITY_GRID")
    expected_counts = {
        "main_clusters": 80,
        "guardrail_clusters": 8,
        "main_sequences": 480,
        "guardrail_sequences": 16,
        "total_sequences": 496,
        "frames": 298592,
        "pairs": 298096,
    }
    if lock.get("counts") != expected_counts:
        _fail("IDENTITY_COUNTS")
    return by_sequence


def _reduce(
    rows: Iterable[dict[str, Any]], identity: dict[str, Any]
) -> dict[str, Any]:
    values = list(rows)
    if len(values) != PAIR_COUNT:
        _fail("PAIR_COUNT")
    streak = triggers = failures = clean = evaluable_count = 0
    fields = ("sequence_id", "cluster_id", "block", "ordinal", "role", "arm")
    for pair_index, row in enumerate(values):
        if not isinstance(row, dict) or row.get("pair_index") != pair_index:
            _fail("PAIR_ORDER")
        if any(row.get(field) != identity.get(field) for field in fields):
            _fail("PAIR_ENVELOPE")
        evaluable = row.get("evaluable")
        response = row.get("compensated_expansion_median_per_s")
        if evaluable is True:
            response_value = _finite(response, "RESPONSE")
            evaluable_count += 1
            streak = streak + 1 if response_value > 0.01 else 0
        elif evaluable is False and response is None:
            streak = 0
        else:
            _fail("ABSTENTION")
        recomputed = streak >= 3
        if (
            "compensated_three_pair_trigger" in row
            and row["compensated_three_pair_trigger"] is not recomputed
        ):
            _fail("FORGED_TRIGGER")
        triggers += int(recomputed)
        detected = row.get("detected_feature_count")
        consistent = row.get("forward_backward_consistent_count")
        occupied = row.get("occupied_3x3_cells")
        if (
            not isinstance(detected, int)
            or isinstance(detected, bool)
            or detected < 0
            or not isinstance(consistent, int)
            or isinstance(consistent, bool)
            or consistent < 0
            or consistent > detected
            or not isinstance(occupied, int)
            or isinstance(occupied, bool)
            or occupied not in range(10)
        ):
            _fail("TRACK_COUNTS")
        fraction = _finite(
            row.get("forward_backward_consistent_fraction"), "TRACK_FRACTION"
        )
        if not 0.0 <= fraction <= 1.0:
            _fail("TRACK_FRACTION_RANGE")
        error_raw = row.get("median_forward_backward_error_px")
        if error_raw is None:
            error = None
        else:
            error = _finite(error_raw, "TRACK_ERROR")
            if error < 0:
                _fail("TRACK_ERROR_RANGE")
        collapse = detected < 60 or consistent < 60 or fraction < 0.50
        fb_failure = error is None or error > 0.75
        failures += int(collapse or fb_failure)
        clean += int(not collapse and not fb_failure and occupied >= 5)
    return {
        "trigger_density": triggers / PAIR_COUNT,
        "failure_density": failures / PAIR_COUNT,
        "clean_sequence": clean >= CLEAN_PAIR_MINIMUM,
        "trigger_count": triggers,
        "failure_count": failures,
        "clean_pair_count": clean,
        "evaluable_count": evaluable_count,
    }


def _unit(arms: dict[str, dict[str, float]]) -> dict[str, float]:
    def y(arm: str) -> float:
        return arms[arm]["trigger_density"]

    def q(arm: str) -> float:
        return arms[arm]["failure_density"]

    motion = y("PERIODIC_6DOF_SELF_MOTION__CLEAN") - y(
        "STATIC_CAMERA__CLEAN"
    )
    return {
        "MOTION_CLEAN": motion,
        "BLUR_STATIC": y("STATIC_CAMERA__BLUR") - y("STATIC_CAMERA__CLEAN"),
        "LOW_TEXTURE_STATIC": y("STATIC_CAMERA__LOW_TEXTURE")
        - y("STATIC_CAMERA__CLEAN"),
        "MOTION_X_BLUR": y("PERIODIC_6DOF_SELF_MOTION__BLUR")
        - y("STATIC_CAMERA__BLUR")
        - motion,
        "MOTION_X_LOW_TEXTURE": y(
            "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE"
        )
        - y("STATIC_CAMERA__LOW_TEXTURE")
        - motion,
        "MOTION_BLUR_VS_STATIC_CLEAN": y(
            "PERIODIC_6DOF_SELF_MOTION__BLUR"
        )
        - y("STATIC_CAMERA__CLEAN"),
        "MOTION_LOW_TEXTURE_VS_STATIC_CLEAN": y(
            "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE"
        )
        - y("STATIC_CAMERA__CLEAN"),
        "BLUR_FAILURE_UNION_STATIC": q("STATIC_CAMERA__BLUR")
        - q("STATIC_CAMERA__CLEAN"),
        "LOW_TEXTURE_FAILURE_UNION_STATIC": q("STATIC_CAMERA__LOW_TEXTURE")
        - q("STATIC_CAMERA__CLEAN"),
    }


def _analysis(
    summaries: dict[str, dict[str, Any]],
    identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    array = np.empty((4, 20, 9), dtype=np.float64)
    for block_index, block in enumerate(BLOCKS):
        for ordinal in range(20):
            arms: dict[str, dict[str, float]] = {}
            for sequence_id, identity in identities.items():
                if (
                    identity["role"] == "MAIN_FACTORIAL"
                    and identity["block"] == block
                    and identity["ordinal"] == ordinal
                ):
                    arms[identity["arm"]] = summaries[sequence_id]
            if set(arms) != set(MAIN_ARMS):
                _fail("MAIN_CLUSTER")
            contrasts = _unit(arms)
            array[block_index, ordinal] = [
                contrasts[name] for name in FAMILY
            ]
    points = array.mean(axis=1)
    theta = points.mean(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(
        0, 20, size=(BOOTSTRAP_REPLICATES, 4, 20), endpoint=False
    )
    estimates = np.zeros((BOOTSTRAP_REPLICATES, 9), dtype=np.float64)
    for block_index in range(4):
        estimates += (
            array[block_index][draws[:, block_index]].mean(axis=1) / 4.0
        )
    sd = estimates.std(axis=0, ddof=1)
    zero = sd == 0.0
    for index in np.flatnonzero(zero):
        if not np.all(array[:, :, index] == array[0, 0, index]):
            _fail("ZERO_SD_INCONSISTENT")
    if (~zero).any():
        z = (estimates[:, ~zero] - theta[~zero]) / sd[~zero]
        critical = float(
            np.quantile(np.max(np.abs(z), axis=1), 0.95, method="linear")
        )
    else:
        critical = 0.0
    estimands: dict[str, Any] = {}
    for index, name in enumerate(FAMILY):
        lower = float(theta[index] - critical * sd[index])
        upper = float(theta[index] + critical * sd[index])
        block_count = int(
            np.sum((points[:, index] >= 0.10) & (points[:, index] > 0))
        )
        classification = (
            "SUPPORTED"
            if theta[index] >= 0.10 and lower > 0 and block_count >= 3
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
            "block_point_estimates": points[:, index].tolist(),
            "blocks_positive_at_least_0_10": block_count,
            "classification": classification,
        }
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.p3_analysis.v1",
        "protocol_id": PROTOCOL_ID,
        "analysis_id": (
            "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
            "TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0"
        ),
        "input_role": "SYNTHETIC_FIXTURE_OR_FUTURE_FORMAL_ONLY",
        "cluster_count": 80,
        "family": list(FAMILY),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
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


def _gates(
    summaries: dict[str, dict[str, Any]],
    identities: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_blocks: dict[str, Any] = {}
    clean_all = True
    for block in BLOCKS:
        clean_blocks[block] = {}
        for arm in (
            "STATIC_CAMERA__CLEAN",
            "PERIODIC_6DOF_SELF_MOTION__CLEAN",
        ):
            values = [
                summaries[sequence]["clean_sequence"]
                for sequence, identity in identities.items()
                if identity["role"] == "MAIN_FACTORIAL"
                and identity["block"] == block
                and identity["arm"] == arm
            ]
            count = sum(values)
            passed = len(values) == 20 and count >= 18
            clean_all &= passed
            clean_blocks[block][arm] = {
                "sequence_count": len(values),
                "trackable_sequence_count": count,
                "passed": passed,
            }
    guard_result: dict[str, Any] = {}
    guard_all = True
    for arm in GUARD_ARMS:
        block_means: dict[str, float] = {}
        for block in BLOCKS:
            values = [
                summaries[sequence]["trigger_density"]
                for sequence, identity in identities.items()
                if identity["role"] == "POSITIVE_GUARDRAIL"
                and identity["block"] == block
                and identity["arm"] == arm
            ]
            if len(values) != 2:
                _fail("GUARD_COUNT")
            block_means[block] = sum(values) / 2
        overall = sum(block_means.values()) / 4
        blocks_passing = sum(value >= 0.50 for value in block_means.values())
        passed = overall >= 0.50 and blocks_passing >= 3
        guard_all &= passed
        guard_result[arm] = {
            "block_means": block_means,
            "overall_mean": overall,
            "blocks_at_least_0_50": blocks_passing,
            "passed": passed,
        }
    return (
        {"passed": clean_all, "blocks": clean_blocks},
        {"passed": guard_all, "arms": guard_result},
    )


def _terminal(
    analysis: dict[str, Any], clean: bool, guard: bool
) -> tuple[str, str | None]:
    result = analysis["estimands"]

    def s(name: str) -> str:
        return result[name]["classification"]

    if not clean or not guard:
        return "INTERVENTION_NOT_EVALUABLE", None
    motion, blur, low = s("MOTION_CLEAN"), s("BLUR_STATIC"), s(
        "LOW_TEXTURE_STATIC"
    )
    xb, xl = s("MOTION_X_BLUR"), s("MOTION_X_LOW_TEXTURE")
    if (
        motion == "SUPPORTED"
        and (blur == "SUPPORTED" or low == "SUPPORTED")
    ) or (
        xb == "SUPPORTED"
        and s("MOTION_BLUR_VS_STATIC_CLEAN") == "SUPPORTED"
    ) or (
        xl == "SUPPORTED"
        and s("MOTION_LOW_TEXTURE_VS_STATIC_CLEAN") == "SUPPORTED"
    ):
        return "MIXED", None
    ruled = "RULED_OUT_AS_MATERIAL"
    if (
        motion == "SUPPORTED"
        and all(value == ruled for value in (blur, low, xb, xl))
    ):
        return "MOTION_SUPPORTED", None
    kinds = []
    if blur == "SUPPORTED" and s("BLUR_FAILURE_UNION_STATIC") == "SUPPORTED":
        kinds.append("BLUR")
    if (
        low == "SUPPORTED"
        and s("LOW_TEXTURE_FAILURE_UNION_STATIC") == "SUPPORTED"
    ):
        kinds.append("LOW_TEXTURE")
    if kinds and motion == ruled and xb == ruled and xl == ruled:
        return (
            "QUALITY_SUPPORTED",
            "BOTH" if len(kinds) == 2 else kinds[0],
        )
    return "NO_SEPARATION_HOLD", None


def independently_recompute(
    bundle: dict[str, Any],
    identity_lock: dict[str, Any],
    receipt_loader: Callable[[str], dict[str, Any]],
    receipt_hash_loader: Callable[[str], str],
    ledger_loader: Callable[[str], list[dict[str, Any]]],
    ledger_hash_loader: Callable[[str], str],
    *,
    identity_lock_sha256: str,
) -> dict[str, Any]:
    identities = _identities(identity_lock)
    if (
        bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("protocol_id") != PROTOCOL_ID
        or bundle.get("terminal")
        not in {"BUNDLE_COMPLETE", "FORMAL_BUNDLE_COMPLETE"}
        or bundle.get("identity_manifest_sha256") != identity_lock_sha256
        or not _is_sha(bundle.get("activation_sha256"))
    ):
        _fail("BUNDLE_HEADER")
    authority_fields = (
        "manipulation_receipt_sha256",
        "manipulation_producer_receipt_sha256",
        "trajectory_manifest_sha256",
        "runner_sha256",
    )
    if any(not _is_sha(bundle.get(field)) for field in authority_fields):
        _fail("BUNDLE_AUTHORITY")
    if (
        bundle.get("arm_count") != 496
        or bundle.get("frame_count") != 298592
        or bundle.get("pair_count") != 298096
        or bundle.get("residual_worker_pids") != []
        or bundle.get("rgb_frames_retained") is not False
        or bundle.get("sequence16_android_realtime") is not False
        or bundle.get("scientific_outcome_interpreted") is not False
    ):
        _fail("BUNDLE_CLOSURE")
    gates = bundle.get("prerequisite_gates")
    if not isinstance(gates, dict) or any(
        gates.get(name) != "PASS"
        for name in (
            "geometry_validation",
            "quality_strength_lock",
            "formal_main_manipulation",
        )
    ):
        _fail("PREREQUISITE_GATES")
    entries = bundle.get("arms")
    if not isinstance(entries, list) or len(entries) != 496:
        _fail("BUNDLE_COUNT")
    summaries: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            _fail("BUNDLE_ENTRY")
        sequence = entry.get("sequence_id")
        if sequence not in identities or sequence in seen:
            _fail("BUNDLE_KEYSET")
        seen.add(sequence)
        receipt_path = entry.get("receipt_path")
        if (
            not isinstance(receipt_path, str)
            or receipt_hash_loader(receipt_path)
            != entry.get("receipt_sha256")
        ):
            _fail("RECEIPT_HASH")
        receipt = receipt_loader(receipt_path)
        identity = identities[sequence]
        if (
            receipt.get("schema") != ARM_SCHEMA
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("terminal") != "ARM_COMPLETE"
            or receipt.get("activation_sha256")
            != bundle.get("activation_sha256")
            or receipt.get("identity_manifest_sha256")
            != identity_lock_sha256
        ):
            _fail("RECEIPT_HEADER")
        if any(
            receipt.get(field) != bundle.get(field)
            for field in authority_fields
        ):
            _fail("RECEIPT_AUTHORITY")
        for field in (
            "sequence_id",
            "cluster_id",
            "block",
            "ordinal",
            "role",
            "arm",
            "numeric_seed_uint64",
            "frame_count",
            "pair_count",
        ):
            if receipt.get(field) != identity.get(field):
                _fail("RECEIPT_IDENTITY")
        ledger_path = receipt.get("pair_ledger_path")
        if (
            not isinstance(ledger_path, str)
            or ledger_hash_loader(ledger_path)
            != receipt.get("pair_ledger_sha256")
        ):
            _fail("LEDGER_HASH")
        frame_path = receipt.get("frame_manifest_path")
        reduced_path = receipt.get("reduced_metrics_path")
        if (
            not isinstance(frame_path, str)
            or ledger_hash_loader(frame_path)
            != receipt.get("frame_manifest_sha256")
            or not isinstance(reduced_path, str)
            or receipt_hash_loader(reduced_path)
            != receipt.get("reduced_metrics_sha256")
        ):
            _fail("ATOMIC_ARTIFACT_HASH")
        frame_rows = ledger_loader(frame_path)
        if len(frame_rows) != FRAME_COUNT:
            _fail("FRAME_COUNT")
        for frame_index, frame in enumerate(frame_rows):
            if (
                not isinstance(frame, dict)
                or frame.get("frame_index") != frame_index
                or not _is_sha(frame.get("rgb_sha256"))
                or not _is_sha(frame.get("valid_mask_sha256"))
            ):
                _fail("FRAME_ROW")
        summary = _reduce(ledger_loader(ledger_path), identity)
        if receipt_loader(reduced_path) != {
            "scheduled_pair_count": PAIR_COUNT,
            "evaluable_pair_count": summary["evaluable_count"],
            "trigger_count": summary["trigger_count"],
            "trigger_density": summary["trigger_density"],
            "quality_failure_union_count": summary["failure_count"],
            "quality_failure_union_density": summary["failure_density"],
            "clean_trackable_pair_count": summary["clean_pair_count"],
            "clean_sequence_trackable": summary["clean_sequence"],
        }:
            _fail("REDUCED_METRICS")
        summaries[sequence] = summary
    if seen != set(identities):
        _fail("BUNDLE_GRID")
    analysis = _analysis(summaries, identities)
    clean, guard = _gates(summaries, identities)
    terminal, subtype = _terminal(
        analysis, clean["passed"], guard["passed"]
    )
    return {
        "analysis": analysis,
        "clean_tracking_gate": clean,
        "positive_guardrail_gate": guard,
        "scientific_terminal": terminal,
        "quality_subtype": subtype,
    }


def validate_loaded_result(
    result: dict[str, Any],
    result_sha256: str,
    bundle: dict[str, Any],
    identity_lock: dict[str, Any],
    receipt_loader: Callable[[str], dict[str, Any]],
    receipt_hash_loader: Callable[[str], str],
    ledger_loader: Callable[[str], list[dict[str, Any]]],
    ledger_hash_loader: Callable[[str], str],
    *,
    bundle_sha256: str,
    identity_lock_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        recomputed = independently_recompute(
            bundle,
            identity_lock,
            receipt_loader,
            receipt_hash_loader,
            ledger_loader,
            ledger_hash_loader,
            identity_lock_sha256=identity_lock_sha256,
        )
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("protocol_id") != PROTOCOL_ID
            or result.get("execution_state") != "VALID_COMPLETE"
        ):
            errors.append("RESULT_HEADER")
        if result.get("inputs") != {
            "bundle_success_sha256": bundle_sha256,
            "identity_lock_sha256": identity_lock_sha256,
            "activation_sha256": bundle.get("activation_sha256"),
        }:
            errors.append("RESULT_INPUT_BINDINGS")
        for field in (
            "analysis",
            "clean_tracking_gate",
            "positive_guardrail_gate",
            "scientific_terminal",
            "quality_subtype",
        ):
            if result.get(field) != recomputed[field]:
                errors.append(f"RESULT_MISMATCH:{field}")
        expected_stage = {
            "MOTION_SUPPORTED": "IMPLEMENTATION_READY_FOR_CONFIRMATION",
            "QUALITY_SUPPORTED": "IMPLEMENTATION_READY_FOR_CONFIRMATION",
            "MIXED": "IMPLEMENTATION_READY_FOR_CONFIRMATION",
            "NO_SEPARATION_HOLD": "IMPLEMENTATION_NOT_READY",
            "INTERVENTION_NOT_EVALUABLE": "NOT_EVALUABLE",
        }[recomputed["scientific_terminal"]]
        if result.get("stage_terminal") != expected_stage:
            errors.append("RESULT_STAGE_TERMINAL")
        if result.get("run_id") != bundle.get("run_id"):
            errors.append("RESULT_RUN_ID")
        if result.get("authority") != (
            "SYNTHETIC_DEVELOPMENT_MECHANISM_ONLY_NO_AUTOMATIC_"
            "ALGORITHM_ANDROID_PRODUCT_OR_SAFETY_AUTHORITY"
        ):
            errors.append("RESULT_AUTHORITY")
        closure = result.get("closure")
        if not isinstance(closure, dict) or closure != {
            "atomic_receipt_count": 496,
            "main_sequence_count": 480,
            "guardrail_sequence_count": 16,
            "frame_count": 298592,
            "pair_count": 298096,
            "missing_sequence_count": 0,
            "duplicate_sequence_count": 0,
        }:
            errors.append("RESULT_CLOSURE")
        if result.get("prerequisite_gates") != bundle.get(
            "prerequisite_gates"
        ):
            errors.append("RESULT_PREREQUISITE_GATES")
    except (
        ValidationError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:
        errors.append(f"INDEPENDENT_RECOMPUTE:{error}")
    validated = not errors
    return {
        "schema": RECEIPT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "validated": validated,
        "terminal": (
            "FORMAL_RESULT_VALID / SCIENTIFIC_TERMINAL_SIGNED"
            if validated
            else "FORMAL_RESULT_INVALID / NO_SCIENTIFIC_TERMINAL"
        ),
        "result_sha256": result_sha256,
        "bundle_success_sha256": bundle_sha256,
        "identity_lock_sha256": identity_lock_sha256,
        "independence": {
            "formal_assembler_imported": False,
            "p3_analysis_imported": False,
            "formal_runner_imported": False,
            "r3_pair_core_imported": False,
            "scope": (
                "496 identities, receipts, ledgers, pair reductions, gates, "
                "shared max-t analysis, terminal precedence and result"
            ),
        },
        "errors": errors,
    }


def _resolve(repo: Path, bundle_root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        _fail("PATH_SCOPE")
    resolved = (bundle_root / value).resolve()
    try:
        resolved.relative_to((repo / "artifacts.local").resolve())
    except ValueError:
        _fail("PATH_SCOPE")
    return resolved


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        _fail("OBJECT")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    _fail("LEDGER_OBJECT")
                rows.append(value)
    return rows


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bundle-success", type=Path, required=True)
    parser.add_argument("--identity-lock", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    bundle = _object(args.bundle_success.resolve())
    identity = _object(args.identity_lock.resolve())
    result = _object(args.result.resolve())
    bundle_root = args.bundle_success.resolve().parent

    def path(relative: str) -> Path:
        return _resolve(repo, bundle_root, relative)

    receipt = validate_loaded_result(
        result,
        sha256_file(args.result.resolve()),
        bundle,
        identity,
        lambda relative: _object(path(relative)),
        lambda relative: sha256_file(path(relative)),
        lambda relative: _jsonl(path(relative)),
        lambda relative: sha256_file(path(relative)),
        bundle_sha256=sha256_file(args.bundle_success.resolve()),
        identity_lock_sha256=sha256_file(args.identity_lock.resolve()),
    )
    _write_exclusive(args.receipt.resolve(), receipt)
    print(
        json.dumps(
            {
                "validated": receipt["validated"],
                "terminal": receipt["terminal"],
                "errors": receipt["errors"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
