"""Independent Stage A closeout for clean motion-component localization.

This validator is deliberately separate from both the producer and the
per-stage validator.  It checks the two immutable stage receipts, independently
recomputes the three block-level routing contrasts from analysis primitives,
and emits a descriptive Stage A closeout plus a contract-preparation-only
Stage B decision.
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


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
CONTRASTS = (
    "ROTATION_MINUS_STATIC",
    "TRANSLATION_MINUS_ROTATION",
    "FULL_MINUS_MAX_SINGLE",
)
EXPECTED_COUNTS = {
    "clusters": 4,
    "frames": 9632,
    "pairs": 9616,
    "sequences": 16,
}
EXPECTED_QMS_RENDER_PAIR_CALLS = 7228
ANALYSIS_CONTROLS = {
    "bootstrap": False,
    "confidence_intervals": False,
    "formal_classification": False,
    "max_t": False,
    "p_values": False,
    "pair_pooled_inference": False,
}
DEFAULT_BASE = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_clean_motion_component_localization_r0"
)
DEFAULT_DOC_STEM = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0"
)
VALIDATOR_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "validate_motion_component_localization_independent_r0.py"
)
CLOSEOUT_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "validate_motion_component_stage_a_closeout_independent_r0.py"
)
FORMAL_DECISION = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_SUCCESSOR_FORMAL_ACTIVATION_DECISION_R0_2026-07-29.json"
)


class InvalidStageACloseout(ValueError):
    """Raised when a Stage A evidence binding or invariant fails."""


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
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidStageACloseout(f"JSON_READ_FAILED:{path}:{error}") from error


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


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise InvalidStageACloseout(code)


def _equal(actual: Any, expected: Any, code: str) -> None:
    _require(actual == expected, f"{code}:actual={actual!r}:expected={expected!r}")


def _float_equal(actual: Any, expected: Any, code: str) -> None:
    _require(
        isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
        and math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ),
        f"{code}:actual={actual!r}:expected={expected!r}",
    )


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise InvalidStageACloseout(f"PATH_OUTSIDE_REPO:{path}") from error


def _validate_common_document(value: dict[str, Any], label: str) -> None:
    _equal(value.get("protocol_id"), PROTOCOL_ID, f"{label}_PROTOCOL")
    _equal(value.get("task_id"), TASK_ID, f"{label}_TASK")


def _validate_formal_firewall(
    root: Path, firewall: dict[str, Any], label: str
) -> None:
    _equal(firewall.get("before"), firewall.get("after"), f"{label}_DRIFT")
    _equal(
        firewall.get("formal_authority_consumed"),
        False,
        f"{label}_AUTHORITY",
    )
    _equal(firewall.get("formal_sequences_run"), 0, f"{label}_SEQUENCES")
    _equal(
        firewall.get("formal_r3_pair_core_calls"), 0, f"{label}_PAIR_CALLS"
    )
    _equal(
        firewall.get("successor_formal_path_absent"),
        True,
        f"{label}_SUCCESSOR_PATH",
    )
    current_formal_hash = sha256_file(root / FORMAL_DECISION)
    _equal(
        firewall["before"].get("formal_activation_decision_sha256"),
        current_formal_hash,
        f"{label}_FORMAL_DECISION_HASH",
    )


def _validate_analysis_controls(
    controls: dict[str, Any], label: str
) -> None:
    _equal(controls, ANALYSIS_CONTROLS, f"{label}_ANALYSIS_CONTROLS")


def recompute_contrasts(
    analysis: dict[str, Any], stage_number: int
) -> dict[str, dict[str, Any]]:
    """Recompute routing contrasts from unit arm metrics, never pair rows."""

    _validate_common_document(analysis, f"STAGE_{stage_number}_ANALYSIS")
    _equal(
        analysis.get("stage_number"),
        stage_number,
        f"STAGE_{stage_number}_ANALYSIS_STAGE",
    )
    _equal(
        analysis.get("terminal"),
        f"STAGE_{stage_number}_ROUTING_COMPLETE / DESCRIPTIVE_ONLY",
        f"STAGE_{stage_number}_ANALYSIS_TERMINAL",
    )
    _validate_analysis_controls(
        analysis.get("analysis_controls", {}),
        f"STAGE_{stage_number}_ANALYSIS",
    )
    _equal(
        analysis.get("claim_ceiling"),
        "CONTROLLED_GENERATOR_INTERNAL_ROUTING_AUDIT_ONLY",
        f"STAGE_{stage_number}_CLAIM_CEILING",
    )
    _equal(
        analysis.get("formal_authority"),
        "UNCHANGED_NOT_CONSUMED",
        f"STAGE_{stage_number}_ANALYSIS_FORMAL_AUTHORITY",
    )

    units = analysis.get("units")
    _require(isinstance(units, list), f"STAGE_{stage_number}_UNITS")
    _equal(len(units), len(BLOCKS), f"STAGE_{stage_number}_UNIT_COUNT")
    values: dict[str, dict[str, float]] = {
        contrast: {} for contrast in CONTRASTS
    }
    for unit in units:
        block = unit.get("block")
        _require(block in BLOCKS, f"STAGE_{stage_number}_BLOCK:{block}")
        _require(
            block not in values[CONTRASTS[0]],
            f"STAGE_{stage_number}_DUPLICATE_BLOCK:{block}",
        )
        _equal(
            unit.get("stage_number"),
            stage_number,
            f"STAGE_{stage_number}_{block}_STAGE",
        )
        arms = unit.get("arm_metrics", {})
        _equal(
            set(arms),
            {"STATIC", "ROTATION_ONLY", "TRANSLATION_ONLY", "FULL_6DOF"},
            f"STAGE_{stage_number}_{block}_ARMS",
        )
        rotation = float(
            arms["ROTATION_ONLY"]["compensated_absolute_p90"]
        ) - float(arms["STATIC"]["compensated_absolute_p90"])
        translation = float(
            arms["TRANSLATION_ONLY"]["compensated_signed_p90"]
        ) - float(arms["ROTATION_ONLY"]["compensated_signed_p90"])
        full = float(arms["FULL_6DOF"]["compensated_signed_p90"]) - max(
            float(arms["ROTATION_ONLY"]["compensated_signed_p90"]),
            float(arms["TRANSLATION_ONLY"]["compensated_signed_p90"]),
        )
        computed = {
            "ROTATION_MINUS_STATIC": rotation,
            "TRANSLATION_MINUS_ROTATION": translation,
            "FULL_MINUS_MAX_SINGLE": full,
        }
        for contrast, value in computed.items():
            values[contrast][block] = value
            _float_equal(
                unit.get("routing_contrasts", {}).get(contrast),
                value,
                f"STAGE_{stage_number}_{block}_{contrast}_UNIT",
            )

    _equal(
        set(values[CONTRASTS[0]]), set(BLOCKS), f"STAGE_{stage_number}_BLOCKS"
    )
    result: dict[str, dict[str, Any]] = {}
    frozen_summary = analysis.get("routing_direction_summary", {})
    for contrast in CONTRASTS:
        block_values = values[contrast]
        ordered = [block_values[block] for block in BLOCKS]
        summary = {
            "values_by_block": block_values,
            "positive_count": sum(value > 0.0 for value in ordered),
            "negative_count": sum(value < 0.0 for value in ordered),
            "zero_count": sum(value == 0.0 for value in ordered),
            "minimum": min(ordered),
            "median": float(statistics.median(ordered)),
            "maximum": max(ordered),
        }
        source_summary = frozen_summary.get(contrast, {})
        for key, expected in summary.items():
            actual = source_summary.get(key)
            if key == "values_by_block":
                _equal(set(actual or {}), set(expected), f"{contrast}_BLOCK_KEYS")
                for block, value in expected.items():
                    _float_equal(
                        actual[block],
                        value,
                        f"STAGE_{stage_number}_{contrast}_{block}_SUMMARY",
                    )
            elif isinstance(expected, float):
                _float_equal(
                    actual,
                    expected,
                    f"STAGE_{stage_number}_{contrast}_{key}",
                )
            else:
                _equal(
                    actual,
                    expected,
                    f"STAGE_{stage_number}_{contrast}_{key}",
                )
        result[contrast] = summary
    return result


def _validate_stage_bundle(
    root: Path,
    *,
    stage_number: int,
    run_path: Path,
    analysis_path: Path,
    receipt_path: Path,
    expected_validator_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run = load_json(run_path)
    analysis = load_json(analysis_path)
    receipt = load_json(receipt_path)
    for label, value in (
        ("RUN", run),
        ("ANALYSIS", analysis),
        ("RECEIPT", receipt),
    ):
        _validate_common_document(value, f"STAGE_{stage_number}_{label}")
        _equal(
            value.get("stage_number"),
            stage_number,
            f"STAGE_{stage_number}_{label}_STAGE",
        )

    _equal(
        run.get("terminal"),
        f"STAGE_{stage_number}_EXECUTION_COMPLETE / "
        "INDEPENDENT_VALIDATION_REQUIRED",
        f"STAGE_{stage_number}_RUN_TERMINAL",
    )
    _equal(run.get("counts"), EXPECTED_COUNTS, f"STAGE_{stage_number}_COUNTS")
    clusters = run.get("clusters")
    _require(
        isinstance(clusters, list),
        f"STAGE_{stage_number}_RUN_CLUSTERS",
    )
    _equal(
        len(clusters),
        EXPECTED_COUNTS["clusters"],
        f"STAGE_{stage_number}_RUN_CLUSTER_COUNT",
    )
    total_qms_calls = 0
    arm_receipt_count = 0
    for cluster in clusters:
        block = cluster.get("block")
        _require(block in BLOCKS, f"STAGE_{stage_number}_RUN_BLOCK:{block}")
        _equal(
            cluster.get("terminal"),
            "MOTION_COMPONENT_CLUSTER_COMPLETE",
            f"STAGE_{stage_number}_{block}_CLUSTER_TERMINAL",
        )
        _equal(
            cluster.get("sequence_count"),
            4,
            f"STAGE_{stage_number}_{block}_SEQUENCE_COUNT",
        )
        _equal(
            cluster.get("qms_render_pair_calls"),
            cluster.get("expected_qms_render_pair_calls"),
            f"STAGE_{stage_number}_{block}_QMS_CALLS",
        )
        total_qms_calls += int(cluster["qms_render_pair_calls"])
        cluster_receipt_path = (
            run_path.parent / cluster["cluster_receipt_path"]
        )
        _require(
            cluster_receipt_path.is_file(),
            f"STAGE_{stage_number}_{block}_CLUSTER_RECEIPT_MISSING",
        )
        _equal(
            sha256_file(cluster_receipt_path),
            cluster.get("cluster_receipt_sha256"),
            f"STAGE_{stage_number}_{block}_CLUSTER_RECEIPT_HASH",
        )
        arms = cluster.get("arm_outputs")
        _require(
            isinstance(arms, list),
            f"STAGE_{stage_number}_{block}_ARM_OUTPUTS",
        )
        _equal(
            len(arms), 4, f"STAGE_{stage_number}_{block}_ARM_OUTPUT_COUNT"
        )
        for arm in arms:
            arm_path = run_path.parent / arm["receipt_path"]
            _require(
                arm_path.is_file(),
                f"STAGE_{stage_number}_{block}_ARM_RECEIPT_MISSING",
            )
            _equal(
                sha256_file(arm_path),
                arm.get("receipt_sha256"),
                f"STAGE_{stage_number}_{block}_ARM_RECEIPT_HASH",
            )
            arm_receipt_count += 1
    _equal(
        total_qms_calls,
        EXPECTED_QMS_RENDER_PAIR_CALLS,
        f"STAGE_{stage_number}_TOTAL_QMS_CALLS",
    )
    _equal(
        arm_receipt_count,
        EXPECTED_COUNTS["sequences"],
        f"STAGE_{stage_number}_ARM_RECEIPT_COUNT",
    )
    _equal(
        run.get("formal_execution_authorized_by_this_run"),
        False,
        f"STAGE_{stage_number}_RUN_FORMAL_AUTH",
    )
    _validate_formal_firewall(
        root, run.get("formal_firewall", {}), f"STAGE_{stage_number}_RUN"
    )

    _equal(receipt.get("validated"), True, f"STAGE_{stage_number}_VALIDATED")
    _equal(receipt.get("errors"), [], f"STAGE_{stage_number}_ERRORS")
    _equal(
        receipt.get("terminal"),
        f"VALID / STAGE_{stage_number}_ROUTING_COMPLETE",
        f"STAGE_{stage_number}_RECEIPT_TERMINAL",
    )
    _equal(
        receipt.get("formal_authority"),
        "UNCHANGED_NOT_CONSUMED",
        f"STAGE_{stage_number}_RECEIPT_FORMAL_AUTHORITY",
    )
    _validate_analysis_controls(
        receipt.get("analysis_controls", {}),
        f"STAGE_{stage_number}_RECEIPT",
    )
    _validate_formal_firewall(
        root,
        receipt.get("formal_firewall", {}),
        f"STAGE_{stage_number}_RECEIPT",
    )
    _equal(
        receipt.get("formal_sequences_run"),
        0,
        f"STAGE_{stage_number}_RECEIPT_FORMAL_SEQUENCES",
    )
    _equal(
        receipt.get("formal_r3_pair_core_calls"),
        0,
        f"STAGE_{stage_number}_RECEIPT_FORMAL_CALLS",
    )
    run_hash = sha256_file(run_path)
    analysis_hash = sha256_file(analysis_path)
    _equal(
        receipt.get("run_receipt_sha256"),
        run_hash,
        f"STAGE_{stage_number}_RUN_HASH",
    )
    _equal(
        receipt.get("analysis_result_file_sha256"),
        analysis_hash,
        f"STAGE_{stage_number}_ANALYSIS_FILE_HASH",
    )
    _equal(
        receipt.get("analysis_result_sha256"),
        analysis_hash,
        f"STAGE_{stage_number}_ANALYSIS_HASH",
    )
    _equal(
        receipt.get("analysis_result_path"),
        _relative(root, analysis_path),
        f"STAGE_{stage_number}_ANALYSIS_PATH",
    )
    _equal(
        receipt.get("validator_source_sha256"),
        expected_validator_hash,
        f"STAGE_{stage_number}_VALIDATOR_HASH",
    )
    recompute_contrasts(analysis, stage_number)
    return run, analysis, receipt


def validate_and_build(
    root: Path,
    *,
    contract_path: Path,
    identity_lock_path: Path,
    activation_path: Path,
    stage_1_run_path: Path,
    stage_1_analysis_path: Path,
    stage_1_receipt_path: Path,
    stage_1_decision_path: Path,
    stage_2_run_path: Path,
    stage_2_analysis_path: Path,
    stage_2_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate all immutable inputs and build closeout/activation documents."""

    paths = (
        contract_path,
        identity_lock_path,
        activation_path,
        stage_1_run_path,
        stage_1_analysis_path,
        stage_1_receipt_path,
        stage_1_decision_path,
        stage_2_run_path,
        stage_2_analysis_path,
        stage_2_receipt_path,
        root / VALIDATOR_SOURCE,
        root / CLOSEOUT_SOURCE,
        root / FORMAL_DECISION,
    )
    for path in paths:
        _require(path.is_file(), f"MISSING_INPUT:{path}")

    contract = load_json(contract_path)
    identity_lock = load_json(identity_lock_path)
    activation = load_json(activation_path)
    stage_1_decision = load_json(stage_1_decision_path)
    for label, value in (
        ("CONTRACT", contract),
        ("IDENTITY_LOCK", identity_lock),
        ("ACTIVATION", activation),
        ("STAGE_1_DECISION", stage_1_decision),
    ):
        _validate_common_document(value, label)

    contract_hash = sha256_file(contract_path)
    identity_lock_hash = sha256_file(identity_lock_path)
    activation_hash = sha256_file(activation_path)
    stage_1_decision_hash = sha256_file(stage_1_decision_path)
    per_stage_validator_hash = sha256_file(root / VALIDATOR_SOURCE)
    closeout_validator_hash = sha256_file(root / CLOSEOUT_SOURCE)
    _equal(
        contract.get("terminal"),
        "MOTION_COMPONENT_CONTRACT_FROZEN / NOT_RUN",
        "CONTRACT_TERMINAL",
    )
    _equal(
        contract.get("formal_authority_consumed"),
        False,
        "CONTRACT_FORMAL_CONSUMED",
    )
    _equal(
        contract.get("formal_identity_execution"),
        False,
        "CONTRACT_FORMAL_EXECUTION",
    )
    _equal(
        identity_lock.get("terminal"),
        "MOTION_COMPONENT_IDENTITIES_FROZEN / STAGE_1_NOT_RUN",
        "IDENTITY_LOCK_TERMINAL",
    )
    _equal(
        identity_lock.get("counts"),
        {
            "arms_per_cluster": 4,
            "blocks": 4,
            "clusters": 8,
            "frames": 19264,
            "pairs": 19232,
            "seeds_per_block": 2,
            "sequences": 32,
            "stage_1_sequences": 16,
            "stage_2_sequences": 16,
        },
        "IDENTITY_LOCK_COUNTS",
    )
    _equal(
        len(identity_lock.get("identities", [])),
        32,
        "IDENTITY_LOCK_IDENTITY_COUNT",
    )
    _equal(
        activation.get("terminal"),
        "STAGE_1_AUTHORIZED / STAGE_2_SEALED",
        "ACTIVATION_TERMINAL",
    )
    _equal(
        activation.get("contract_sha256"), contract_hash, "ACTIVATION_CONTRACT_HASH"
    )
    _equal(
        activation.get("identity_lock_sha256"),
        identity_lock_hash,
        "ACTIVATION_LOCK_HASH",
    )
    _equal(
        activation.get("identity_set_sha256"),
        identity_lock.get("identity_set_sha256"),
        "ACTIVATION_IDENTITY_SET",
    )
    _equal(
        activation.get("formal_execution_authorized"),
        False,
        "ACTIVATION_FORMAL_AUTH",
    )
    _equal(
        activation.get("formal_authority_consumed"),
        False,
        "ACTIVATION_FORMAL_CONSUMED",
    )
    _equal(
        stage_1_decision.get("decision"), "OPEN_STAGE_2", "STAGE_1_DECISION"
    )
    _equal(
        stage_1_decision.get("terminal"),
        "STAGE_2_OPENED_BY_FROZEN_ROUTING",
        "STAGE_1_DECISION_TERMINAL",
    )
    _equal(
        stage_1_decision.get("validator_source_sha256"),
        per_stage_validator_hash,
        "STAGE_1_DECISION_VALIDATOR_HASH",
    )

    run_1, analysis_1, receipt_1 = _validate_stage_bundle(
        root,
        stage_number=1,
        run_path=stage_1_run_path,
        analysis_path=stage_1_analysis_path,
        receipt_path=stage_1_receipt_path,
        expected_validator_hash=per_stage_validator_hash,
    )
    run_2, analysis_2, receipt_2 = _validate_stage_bundle(
        root,
        stage_number=2,
        run_path=stage_2_run_path,
        analysis_path=stage_2_analysis_path,
        receipt_path=stage_2_receipt_path,
        expected_validator_hash=per_stage_validator_hash,
    )

    _equal(
        run_1.get("stage_1_activation_sha256"),
        activation_hash,
        "STAGE_1_RUN_ACTIVATION_HASH",
    )
    _equal(
        run_2.get("stage_2_decision_sha256"),
        stage_1_decision_hash,
        "STAGE_2_RUN_DECISION_HASH",
    )
    identity_set_hash = identity_lock.get("identity_set_sha256")
    for label, value in (
        ("STAGE_1_RUN", run_1),
        ("STAGE_2_RUN", run_2),
        ("STAGE_1_RECEIPT", receipt_1),
        ("STAGE_2_RECEIPT", receipt_2),
    ):
        _equal(
            value.get("identity_set_sha256"),
            identity_set_hash,
            f"{label}_IDENTITY_SET",
        )
    _equal(run_1.get("bindings"), run_2.get("bindings"), "RUN_BINDING_DRIFT")
    _equal(
        run_1["bindings"].get("contract_sha256"),
        contract_hash,
        "RUN_CONTRACT_HASH",
    )
    _equal(
        run_1["bindings"].get("identity_lock_sha256"),
        identity_lock_hash,
        "RUN_IDENTITY_LOCK_HASH",
    )
    _equal(
        run_1["bindings"].get("runner_source_sha256"),
        sha256_file(
            root
            / "scripts/research/egomotion_compensated_looming/"
            "periodic_self_motion_counterfactual_r2/"
            "motion_component_localization_r0.py"
        ),
        "RUNNER_SOURCE_DRIFT",
    )
    _equal(
        run_1.get("formal_firewall"),
        run_2.get("formal_firewall"),
        "CROSS_STAGE_FORMAL_FIREWALL_DRIFT",
    )

    contrasts_1 = recompute_contrasts(analysis_1, 1)
    contrasts_2 = recompute_contrasts(analysis_2, 2)
    replicated = [
        contrast
        for contrast in CONTRASTS
        if contrasts_1[contrast]["positive_count"] >= 3
        and contrasts_2[contrast]["positive_count"] >= 3
    ]
    _equal(
        replicated,
        ["ROTATION_MINUS_STATIC", "TRANSLATION_MINUS_ROTATION"],
        "REPLICATED_COMPONENT_CLASSIFICATION",
    )
    unstable = [
        contrast for contrast in CONTRASTS if contrast not in replicated
    ]
    _equal(
        unstable,
        ["FULL_MINUS_MAX_SINGLE"],
        "UNSTABLE_COMPONENT_CLASSIFICATION",
    )

    evidence_paths = {
        "contract": _relative(root, contract_path),
        "identity_lock": _relative(root, identity_lock_path),
        "stage_1_activation": _relative(root, activation_path),
        "stage_1_run_receipt": _relative(root, stage_1_run_path),
        "stage_1_analysis": _relative(root, stage_1_analysis_path),
        "stage_1_independent_receipt": _relative(root, stage_1_receipt_path),
        "stage_1_routing_decision": _relative(root, stage_1_decision_path),
        "stage_2_run_receipt": _relative(root, stage_2_run_path),
        "stage_2_analysis": _relative(root, stage_2_analysis_path),
        "stage_2_independent_receipt": _relative(root, stage_2_receipt_path),
    }
    evidence_hashes = {
        name: sha256_file(root / path)
        for name, path in evidence_paths.items()
    }
    closeout = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "motion_component_stage_a_closeout.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "closeout_convention": {
            "classification": "DESCRIPTIVE_ROUTING_ONLY",
            "robust_replicated_component_rule": (
                "positive block direction in at least 3 of 4 blocks in both "
                "Stage 1 and Stage 2"
            ),
            "preregistered_confirmatory_inference": False,
            "cluster_unit": "block_seed_cluster",
            "pair_is_independent_unit": False,
        },
        "analysis_controls": ANALYSIS_CONTROLS,
        "counts_per_stage": EXPECTED_COUNTS,
        "qms_render_pair_calls_per_stage": EXPECTED_QMS_RENDER_PAIR_CALLS,
        "stage_1_contrasts": contrasts_1,
        "stage_2_contrasts": contrasts_2,
        "replicated_components": replicated,
        "unstable_components": unstable,
        "scientific_routing": {
            "rotation_residual_leakage": (
                "REPLICATED_BOUNDARY_CONDITION / NO_ALGORITHM_CHANGE"
            ),
            "translation_signed_response": (
                "REPLICATED / ROUTE_TO_TRANSLATION_DEPTH_ORACLE"
            ),
            "full_6dof_interaction": (
                "UNSTABLE / DO_NOT_ACTIVATE_INTERACTION_BRANCH"
            ),
            "object_approach_evidence": "NOT_TESTED_IN_STAGE_A",
        },
        "evidence_paths": evidence_paths,
        "evidence_sha256": evidence_hashes,
        "per_stage_validator_source_sha256": per_stage_validator_hash,
        "closeout_validator_source_sha256": closeout_validator_hash,
        "formal_firewall": run_2["formal_firewall"],
        "formal_external_authority": (
            "AUTHORIZED_ONE_SHOT / NOT_CONSUMED / NOT_RUN"
        ),
        "formal_execution_authorized_by_this_closeout": False,
        "formal_authority": "UNCHANGED_NOT_CONSUMED",
        "formal_sequences_run": 0,
        "formal_r3_pair_core_calls": 0,
        "scientific_terminal": "A_COMPONENT_DIRECTION_REPLICATED",
        "errors": [],
        "validated": True,
        "terminal": "VALID / STAGE_A_COMPLETE",
    }
    activation_decision = {
        "schema": (
            "rcle.periodic_self_motion_counterfactual."
            "stage_b_activation_decision.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "decision": "ACTIVATE_STAGE_B_CONTRACT_PREPARATION_ONLY",
        "next_stage": (
            "TRANSLATION_DEPTH_ORACLE_WITH_OBJECT_APPROACH_CONTROL"
        ),
        "authorized_actions": [
            "draft_and_freeze_stage_b_contract",
            "draft_stage_b_identity_and_role_design",
            "define_oracle_and_object_approach_estimands",
            "define_independent_validation_and_stop_rules",
        ],
        "stage_b_execution_authorized": False,
        "stage_c_feature_contract_authorized": False,
        "stage_d_incremental_value_authorized": False,
        "rotation_algorithm_change_authorized": False,
        "full_interaction_branch_authorized": False,
        "formal_execution_authorized_by_this_decision": False,
        "formal_authority_consumed": False,
        "carried_boundaries": {
            "rotation_residual_leakage": (
                "audit boundary condition, not an algorithm-change trigger"
            ),
            "signed_vs_absolute": (
                "signed expansion and absolute leakage remain separate"
            ),
            "independence": (
                "cluster is the analysis unit; pair is a longitudinal repeat"
            ),
            "claim_ceiling": (
                "Stage A is controlled-generator internal routing evidence"
            ),
        },
        "terminal": "STAGE_A_CLOSED / STAGE_B_CONTRACT_ONLY",
    }
    return closeout, activation_decision


def _default_paths(root: Path) -> dict[str, Path]:
    stem = root / DEFAULT_DOC_STEM
    base = root / DEFAULT_BASE
    return {
        "contract_path": Path(f"{stem}_CONTRACT_2026-07-29.json"),
        "identity_lock_path": Path(
            f"{stem}_IDENTITY_LOCK_2026-07-29.json"
        ),
        "activation_path": Path(
            f"{stem}_STAGE_1_ACTIVATION_2026-07-29.json"
        ),
        "stage_1_run_path": base / "stage1/run_receipt.json",
        "stage_1_analysis_path": (
            base / "stage1/independent_analysis_result.json"
        ),
        "stage_1_receipt_path": Path(
            f"{stem}_STAGE_1_INDEPENDENT_RECEIPT_2026-07-29.json"
        ),
        "stage_1_decision_path": Path(
            f"{stem}_STAGE_1_ROUTING_DECISION_2026-07-29.json"
        ),
        "stage_2_run_path": base / "stage2/run_receipt.json",
        "stage_2_analysis_path": (
            base / "stage2/independent_analysis_result.json"
        ),
        "stage_2_receipt_path": Path(
            f"{stem}_STAGE_2_INDEPENDENT_RECEIPT_2026-07-29.json"
        ),
        "closeout_path": Path(
            f"{stem}_STAGE_A_INDEPENDENT_CLOSEOUT_RECEIPT_2026-07-29.json"
        ),
        "decision_path": Path(
            f"{stem}_STAGE_B_ACTIVATION_DECISION_2026-07-29.json"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--closeout-output", type=Path)
    parser.add_argument("--decision-output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    paths = _default_paths(root)
    closeout, decision = validate_and_build(
        root,
        **{
            key: value
            for key, value in paths.items()
            if key not in {"closeout_path", "decision_path"}
        },
    )
    closeout_path = args.closeout_output or paths["closeout_path"]
    decision_path = args.decision_output or paths["decision_path"]
    write_exclusive(closeout_path, closeout)
    decision["stage_a_closeout_receipt_path"] = _relative(
        root, closeout_path
    )
    decision["stage_a_closeout_receipt_sha256"] = sha256_file(closeout_path)
    decision["closeout_validator_source_sha256"] = sha256_file(
        root / CLOSEOUT_SOURCE
    )
    write_exclusive(decision_path, decision)
    print(
        json.dumps(
            {
                "closeout": str(closeout_path),
                "closeout_sha256": sha256_file(closeout_path),
                "decision": str(decision_path),
                "decision_sha256": sha256_file(decision_path),
                "terminal": decision["terminal"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
