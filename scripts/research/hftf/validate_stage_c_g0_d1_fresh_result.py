#!/usr/bin/env python3
"""Independently validate the frozen HFTF G0-D1 fresh effect terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_current_clearance_"
    "fresh_execution_contract_d1"
)
CONTRACT_STATUS = "FROZEN_BEFORE_D1_FRESH_SOURCE_OPENING_OR_PREDICTION"
PACKAGE_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_package_validation"
PACKAGE_READY = "G0_D1_FRESH_PACKAGE_VALIDATED_AND_OPPORTUNITY_ADEQUATE"
PREDICTION_AUTHORIZATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction_authorization"
)
PREDICTION_AUTHORIZED = (
    "G0_D1_FRESH_PREDICTION_AUTHORIZATION_READY"
)
PREDICTION_AUTHORIZATION_KEYS = {
    "schema",
    "terminal",
    "contract_sha256",
    "package_validator_sha256",
    "prediction_inputs_path",
    "prediction_inputs_sha256",
    "prediction_input_count",
    "source_order",
    "source_frame_indices",
    "authorization",
}
PREDICTION_AUTHORIZATION_DECISION_KEYS = {
    "fresh_prediction_authorized",
    "truth_join_authorized_before_predictions_frozen",
    "source_replacement_or_package_rematerialization",
}
TRUTH_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_truth"
PREDICTION_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_prediction"
COMPLETION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_predictions_frozen"
)
PREDICTIONS_READY = "G0_D1_FRESH_PREDICTIONS_FROZEN"
JOIN_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_truth_join_receipt"
RESULT_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_effect_result"
VALIDATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_effect_validation"
)
VALIDATED = "G0_D1_FRESH_EFFECT_TERMINAL_VALIDATED"
SUPPORTED = (
    "SIGNED_CLEARANCE_CURRENT_BRIDGE_SUPPORTED_"
    "FOR_CAUSAL_TRANSPORT_CONTRACT_ONLY"
)
NOT_SUPPORTED = (
    "SIGNED_CLEARANCE_CURRENT_CROSS_SOURCE_"
    "LEARNABILITY_NOT_SUPPORTED_STOP"
)
NOT_EVALUABLE = "G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT"
IMPLEMENTATION_KEY = "fresh_result_validator"
IMPLEMENTATION_PATH = (
    "scripts/research/hftf/validate_stage_c_g0_d1_fresh_result.py"
)
EVALUATOR_PATH = "scripts/research/hftf/evaluate_stage_c_g0_d1_fresh.py"
PREDICTOR_PATH = "scripts/research/hftf/predict_stage_c_g0_d1_fresh.py"
PACKAGE_VALIDATOR_PATH = (
    "scripts/research/hftf/validate_stage_c_g0_d1_fresh_package.py"
)
PREDICTION_RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction_execution_receipt"
)
ARMS = ("DIRECT_RISK_CURRENT", "SIGNED_CLEARANCE_CURRENT")
SEEDS = (17, 29, 43)
HEIGHTS = ("body", "head")
GROUPS = ("overall", "risk", "safe", "near")
TRUTH_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "source_frame_index",
    "manifest_id",
    "labels",
}
PREDICTION_KEYS = {
    "schema",
    "prediction_index",
    "seed",
    "arm",
    "checkpoint_sha256",
    "sample_id",
    "session_id",
    "source_frame_index",
    "manifest_id",
    "raw_task_output",
    "risk_probability",
    "known_probability",
}
JOIN_KEYS = {
    "schema",
    "status",
    "execution_contract_sha256",
    "package_validation_sha256",
    "completion_sha256",
    "predictions_sha256",
    "expected_truth_sha256",
    "truth_join_exactly_once",
    "second_model_forward_authorized",
    "source_replacement_authorized",
}
RESULT_KEYS = {
    "schema",
    "terminal",
    "workflow_profile",
    "claim_ceiling",
    "parents",
    "run_metrics",
    "aggregates",
    "gates",
    "all_gates_pass",
    "fresh_firewall",
    "authorization",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected JSONL object at {path}:{line_number}"
            )
        records.append(value)
    return records


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_paths(
    contract_path: Path,
    package_validation_path: Path,
    truth_path: Path,
    completion_path: Path,
    predictions_path: Path,
    join_path: Path,
    result_path: Path,
    output_root: Path,
) -> None:
    repository = _repository_root()
    expected = {
        "contract": repository
        / "docs/research/hftf/"
        "HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_"
        "EXECUTION_CONTRACT_D1_2026-08-01.json",
        "package_validation": repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-validation-20260801/validation.json",
        "truth": repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-20260801/truth_labels.jsonl",
        "completion": repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-predictions-20260801/completion.json",
        "predictions": repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-predictions-20260801/predictions.jsonl",
        "join": repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-effect-20260801/truth_join_receipt.json",
        "result": repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-effect-20260801/effect_result.json",
        "output": repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-effect-validation-20260801",
    }
    actual = {
        "contract": contract_path,
        "package_validation": package_validation_path,
        "truth": truth_path,
        "completion": completion_path,
        "predictions": predictions_path,
        "join": join_path,
        "result": result_path,
        "output": output_root,
    }
    for key, expected_path in expected.items():
        if actual[key].resolve() != expected_path.resolve():
            raise ValueError(f"Fresh result validator noncanonical {key} path")


def _implementation_receipts(contract: dict[str, Any]) -> None:
    receipt = contract.get("implementations", {}).get(IMPLEMENTATION_KEY)
    if (
        not isinstance(receipt, dict)
        or receipt.get("path") != IMPLEMENTATION_PATH
        or receipt.get("sha256") != _sha256(Path(__file__).resolve())
        or receipt.get("execution_authorized") is not True
    ):
        raise ValueError("Fresh result validator implementation mismatch")
    evaluator = contract.get("implementations", {}).get("fresh_evaluator")
    evaluator_path = (_repository_root() / EVALUATOR_PATH).resolve()
    if (
        not isinstance(evaluator, dict)
        or evaluator.get("path") != EVALUATOR_PATH
        or not evaluator_path.is_file()
        or evaluator.get("sha256") != _sha256(evaluator_path)
        or evaluator.get("execution_authorized") is not True
    ):
        raise ValueError("Fresh evaluator implementation receipt mismatch")
    predictor = contract.get("implementations", {}).get("fresh_predictor")
    predictor_path = (_repository_root() / PREDICTOR_PATH).resolve()
    if (
        not isinstance(predictor, dict)
        or predictor.get("path") != PREDICTOR_PATH
        or not predictor_path.is_file()
        or predictor.get("sha256") != _sha256(predictor_path)
        or predictor.get("execution_authorized") is not True
    ):
        raise ValueError("Fresh predictor implementation receipt mismatch")
    package_validator = contract.get("implementations", {}).get(
        "fresh_package_validator"
    )
    package_validator_path = (
        _repository_root() / PACKAGE_VALIDATOR_PATH
    ).resolve()
    if (
        not isinstance(package_validator, dict)
        or package_validator.get("path") != PACKAGE_VALIDATOR_PATH
        or not package_validator_path.is_file()
        or package_validator.get("sha256")
        != _sha256(package_validator_path)
        or package_validator.get("execution_authorized") is not True
    ):
        raise ValueError(
            "Fresh package validator implementation receipt mismatch"
        )


def _exact_matrix(
    value: Any, *, probability: bool = False
) -> list[list[list[float]]]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Independent matrix height shape mismatch")
    result: list[list[list[float]]] = []
    for height in value:
        if not isinstance(height, list) or len(height) != 6:
            raise ValueError("Independent matrix row shape mismatch")
        rows: list[list[float]] = []
        for row in height:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError("Independent matrix column shape mismatch")
            numbers: list[float] = []
            for item in row:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or (
                        probability
                        and not 0.0 <= float(item) <= 1.0
                    )
                ):
                    raise ValueError("Independent prediction value invalid")
                numbers.append(float(item))
            rows.append(numbers)
        result.append(rows)
    return result


def _truth_labels(
    labels: Any,
) -> tuple[
    list[list[list[bool]]],
    list[list[list[bool]]],
    list[list[list[float]]],
]:
    if not isinstance(labels, dict) or set(labels) != {
        "known_target",
        "risk_target_nullable",
        "clearance_target_m_nullable",
    }:
        raise ValueError("Independent truth label keys mismatch")
    known_raw = labels["known_target"]
    risk_raw = labels["risk_target_nullable"]
    clearance_raw = labels["clearance_target_m_nullable"]
    for matrix in (known_raw, risk_raw, clearance_raw):
        if (
            not isinstance(matrix, list)
            or len(matrix) != 2
            or any(
                not isinstance(height, list)
                or len(height) != 6
                or any(
                    not isinstance(row, list) or len(row) != 6
                    for row in height
                )
                for height in matrix
            )
        ):
            raise ValueError("Independent truth matrix shape mismatch")
    known = [[[False] * 6 for _ in range(6)] for _ in range(2)]
    risk = [[[False] * 6 for _ in range(6)] for _ in range(2)]
    clearance = [[[0.0] * 6 for _ in range(6)] for _ in range(2)]
    for height in range(2):
        for row in range(6):
            for column in range(6):
                known_value = known_raw[height][row][column]
                risk_value = risk_raw[height][row][column]
                clearance_value = clearance_raw[height][row][column]
                if type(known_value) is not int or known_value not in (0, 1):
                    raise ValueError("Independent known truth is not binary")
                known[height][row][column] = bool(known_value)
                if not known_value:
                    if risk_value is not None or clearance_value is not None:
                        raise ValueError(
                            "Independent UNKNOWN truth is not null"
                        )
                    continue
                if (
                    type(risk_value) is not int
                    or risk_value not in (0, 1)
                    or isinstance(clearance_value, bool)
                    or not isinstance(clearance_value, (int, float))
                    or not math.isfinite(float(clearance_value))
                    or not -0.5 <= float(clearance_value) <= 1.0
                    or bool(risk_value)
                    != (float(clearance_value) < 0.0)
                ):
                    raise ValueError(
                        "Independent known risk/clearance truth mismatch"
                    )
                risk[height][row][column] = bool(risk_value)
                clearance[height][row][column] = float(clearance_value)
    return known, risk, clearance


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _add_count(
    counts: dict[str, int], prediction: bool, truth: bool
) -> None:
    if prediction and truth:
        counts["tp"] += 1
    elif prediction:
        counts["fp"] += 1
    elif truth:
        counts["fn"] += 1
    else:
        counts["tn"] += 1


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"],
    )
    denominator = 2 * tp + fp + fn
    return {
        **counts,
        "f1": 2 * tp / denominator if denominator else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def _checkpoint_contract(contract: dict[str, Any]) -> list[dict[str, Any]]:
    checkpoints = contract.get("checkpoint_contract", {}).get("checkpoints")
    expected_order = [
        (seed, arm) for seed in SEEDS for arm in ARMS
    ]
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != 6
        or [
            (item.get("seed"), item.get("arm"))
            for item in checkpoints
            if isinstance(item, dict)
        ]
        != expected_order
    ):
        raise ValueError("Independent checkpoint order mismatch")
    for item in checkpoints:
        digest = item.get("sha256")
        if (
            isinstance(item.get("seed"), bool)
            or not isinstance(item.get("seed"), int)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("Independent checkpoint receipt invalid")
    return checkpoints


def _exact_records(
    contract: dict[str, Any],
    package_validation: dict[str, Any],
    truth_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, str, str], dict[str, Any]],
]:
    source_order = package_validation.get("source_order")
    source_frames = package_validation.get("source_frame_indices")
    if (
        not isinstance(source_order, list)
        or len(source_order) != 3
        or len(set(source_order)) != 3
        or not all(
            isinstance(source, str) and source for source in source_order
        )
        or not isinstance(source_frames, dict)
        or set(source_frames) != set(source_order)
    ):
        raise ValueError("Independent frozen source contract mismatch")
    expected_pairs: list[tuple[str, int]] = []
    for source in source_order:
        frames = source_frames[source]
        if (
            not isinstance(frames, list)
            or len(frames) != 25
            or len(set(frames)) != 25
            or any(
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame < 0
                for frame in frames
            )
        ):
            raise ValueError("Independent fresh source frames mismatch")
        expected_pairs.extend((source, frame) for frame in frames)
    if len(truth_records) != 75:
        raise ValueError("Independent truth record count mismatch")
    samples: set[str] = set()
    for record, expected_pair in zip(
        truth_records, expected_pairs, strict=True
    ):
        sample_id = record.get("sample_id")
        if (
            set(record) != TRUTH_KEYS
            or record.get("schema") != TRUTH_SCHEMA
            or not isinstance(sample_id, str)
            or not sample_id
            or sample_id in samples
            or (
                record.get("session_id"),
                record.get("source_frame_index"),
            )
            != expected_pair
            or not isinstance(record.get("manifest_id"), str)
            or not record["manifest_id"]
        ):
            raise ValueError("Independent truth schema/order mismatch")
        _truth_labels(record["labels"])
        samples.add(sample_id)
    checkpoints = _checkpoint_contract(contract)
    expected_keys = [
        (int(checkpoint["seed"]), str(checkpoint["arm"]), truth["sample_id"])
        for checkpoint in checkpoints
        for truth in truth_records
    ]
    if len(predictions) != 450:
        raise ValueError("Independent prediction record count mismatch")
    prediction_by_key: dict[
        tuple[int, str, str], dict[str, Any]
    ] = {}
    truth_by_sample = {
        record["sample_id"]: record for record in truth_records
    }
    checkpoint_hashes = {
        (int(item["seed"]), str(item["arm"])): item["sha256"]
        for item in checkpoints
    }
    for index, (record, expected_key) in enumerate(
        zip(predictions, expected_keys, strict=True)
    ):
        key = (
            record.get("seed"),
            record.get("arm"),
            record.get("sample_id"),
        )
        truth = truth_by_sample.get(record.get("sample_id"))
        if (
            set(record) != PREDICTION_KEYS
            or record.get("schema") != PREDICTION_SCHEMA
            or record.get("prediction_index") != index
            or key != expected_key
            or key in prediction_by_key
            or truth is None
            or record.get("checkpoint_sha256")
            != checkpoint_hashes[(key[0], key[1])]
            or record.get("session_id") != truth["session_id"]
            or record.get("source_frame_index")
            != truth["source_frame_index"]
            or record.get("manifest_id") != truth["manifest_id"]
        ):
            raise ValueError("Independent prediction Cartesian key mismatch")
        _exact_matrix(record["raw_task_output"])
        raw = np.asarray(
            _exact_matrix(record["raw_task_output"]), dtype=np.float64
        )
        probability = np.asarray(
            _exact_matrix(
                record["risk_probability"], probability=True
            ),
            dtype=np.float64,
        )
        _exact_matrix(record["known_probability"], probability=True)
        expected_probability = (
            1.0 / (1.0 + np.exp(-raw))
            if key[1] == ARMS[0]
            else (raw < 0.0).astype(np.float64)
        )
        if not np.allclose(
            probability, expected_probability, rtol=0.0, atol=1e-7
        ):
            raise ValueError(
                "Independent frozen risk derivation mismatch"
            )
        prediction_by_key[key] = record
    if set(prediction_by_key) != set(expected_keys):
        raise ValueError("Independent prediction Cartesian set mismatch")
    return truth_records, prediction_by_key


def _recompute_run(
    checkpoint: dict[str, Any],
    source_order: list[str],
    truth_records: list[dict[str, Any]],
    prediction_by_key: dict[
        tuple[int, str, str], dict[str, Any]
    ],
) -> dict[str, Any]:
    seed = int(checkpoint["seed"])
    arm = str(checkpoint["arm"])
    micro = _empty_counts()
    by_height = {height: _empty_counts() for height in HEIGHTS}
    by_source = {source: _empty_counts() for source in source_order}
    known_correct = 0
    known_total = 0
    clearance_groups = {
        (source, height): {
            group: {"sum": 0.0, "count": 0} for group in GROUPS
        }
        for source in source_order
        for height in HEIGHTS
    }
    out_of_range = 0
    clearance_known = 0
    for truth_record in truth_records:
        prediction = prediction_by_key[
            (seed, arm, truth_record["sample_id"])
        ]
        risk_probability = _exact_matrix(
            prediction["risk_probability"], probability=True
        )
        known_probability = _exact_matrix(
            prediction["known_probability"], probability=True
        )
        raw = _exact_matrix(prediction["raw_task_output"])
        known, risk, clearance = _truth_labels(truth_record["labels"])
        source = truth_record["session_id"]
        for height_index, height in enumerate(HEIGHTS):
            for row in range(6):
                for column in range(6):
                    is_known = known[height_index][row][column]
                    known_correct += int(
                        (known_probability[height_index][row][column] >= 0.5)
                        == is_known
                    )
                    known_total += 1
                    if not is_known:
                        continue
                    truth_risk = risk[height_index][row][column]
                    predicted_risk = (
                        risk_probability[height_index][row][column] >= 0.5
                    )
                    _add_count(micro, predicted_risk, truth_risk)
                    _add_count(
                        by_height[height], predicted_risk, truth_risk
                    )
                    _add_count(
                        by_source[source], predicted_risk, truth_risk
                    )
        if arm == "SIGNED_CLEARANCE_CURRENT":
            raw_array = np.asarray(raw, dtype=np.float64)
            known_array = np.asarray(known, dtype=np.bool_)
            risk_array = np.asarray(risk, dtype=np.bool_)
            clearance_array = np.asarray(clearance, dtype=np.float64)
            absolute_error = np.abs(raw_array - clearance_array)
            out_of_range += int(
                (
                    (
                        (raw_array < -0.5)
                        | (raw_array > 1.0)
                    )
                    & known_array
                ).sum()
            )
            clearance_known += int(known_array.sum())
            for height_index, height in enumerate(HEIGHTS):
                masks = {
                    "overall": known_array[height_index],
                    "risk": (
                        known_array[height_index]
                        & risk_array[height_index]
                    ),
                    "safe": (
                        known_array[height_index]
                        & ~risk_array[height_index]
                    ),
                    "near": (
                        known_array[height_index]
                        & (
                            np.abs(clearance_array[height_index])
                            <= 0.2
                        )
                    ),
                }
                for group, mask in masks.items():
                    values = clearance_groups[(source, height)][group]
                    values["sum"] = float(values["sum"]) + float(
                        absolute_error[height_index][mask].sum()
                    )
                    values["count"] = int(values["count"]) + int(
                        mask.sum()
                    )
    run: dict[str, Any] = {
        "seed": seed,
        "arm": arm,
        "checkpoint_sha256": checkpoint["sha256"],
        "risk_micro": _metrics(micro),
        "risk_by_height": {
            height: _metrics(counts)
            for height, counts in by_height.items()
        },
        "risk_by_source": {
            source: _metrics(counts)
            for source, counts in by_source.items()
        },
        "known_accuracy_diagnostic": known_correct / known_total,
    }
    if arm == "SIGNED_CLEARANCE_CURRENT":
        source_height_mae: dict[str, dict[str, float]] = {}
        for (source, height), groups in clearance_groups.items():
            source_height_mae[f"{source}:{height}"] = {}
            for group, values in groups.items():
                if values["count"] <= 0:
                    raise ValueError(
                        "Independent clearance denominator is empty"
                    )
                source_height_mae[f"{source}:{height}"][group] = (
                    values["sum"] / values["count"]
                )
        run["clearance_mae_by_source_height_m"] = source_height_mae
        run["clearance_source_height_macro_mae_m"] = {
            group: float(
                np.mean(
                    [
                        values[group]
                        for values in source_height_mae.values()
                    ]
                )
            )
            for group in GROUPS
        }
        run["raw_prediction_out_of_target_range_fraction"] = (
            out_of_range / clearance_known
        )
    return run


def _recompute_decision(
    runs: list[dict[str, Any]],
    source_order: list[str],
    unknown_to_safe_violation_count: int,
) -> tuple[dict[str, Any], dict[str, bool], str]:
    by = {(run["seed"], run["arm"]): run for run in runs}
    if set(by) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("Independent run key set mismatch")

    def paired(getter: Any) -> list[float]:
        return [
            float(getter(by[(seed, ARMS[1])]))
            - float(getter(by[(seed, ARMS[0])]))
            for seed in SEEDS
        ]

    f1_deltas = paired(lambda run: run["risk_micro"]["f1"])
    recall_deltas = paired(lambda run: run["risk_micro"]["recall"])
    fpr_deltas = paired(
        lambda run: run["risk_micro"]["false_positive_rate"]
    )
    height_deltas = {
        height: paired(
            lambda run, chosen=height: run["risk_by_height"][chosen]["f1"]
        )
        for height in HEIGHTS
    }
    source_deltas = {
        source: paired(
            lambda run, chosen=source: run["risk_by_source"][chosen]["f1"]
        )
        for source in source_order
    }
    clearance_f1 = [
        by[(seed, ARMS[1])]["risk_micro"]["f1"] for seed in SEEDS
    ]
    source_clearance_f1 = {
        source: [
            by[(seed, ARMS[1])]["risk_by_source"][source]["f1"]
            for seed in SEEDS
        ]
        for source in source_order
    }
    clearance_mae = {
        group: [
            by[(seed, ARMS[1])][
                "clearance_source_height_macro_mae_m"
            ][group]
            for seed in SEEDS
        ]
        for group in GROUPS
    }
    aggregates = {
        "clearance_micro_f1_by_seed": dict(
            zip(map(str, SEEDS), clearance_f1)
        ),
        "clearance_median_seed_micro_f1": float(
            statistics.median(clearance_f1)
        ),
        "micro_f1_delta_by_seed": dict(
            zip(map(str, SEEDS), f1_deltas)
        ),
        "median_micro_f1_delta": float(statistics.median(f1_deltas)),
        "median_recall_delta": float(statistics.median(recall_deltas)),
        "median_false_positive_rate_delta": float(
            statistics.median(fpr_deltas)
        ),
        "median_f1_delta_by_height": {
            height: float(statistics.median(values))
            for height, values in height_deltas.items()
        },
        "median_f1_delta_by_source": {
            source: float(statistics.median(values))
            for source, values in source_deltas.items()
        },
        "median_clearance_f1_by_source": {
            source: float(statistics.median(values))
            for source, values in source_clearance_f1.items()
        },
        "clearance_source_height_macro_mae_by_seed_m": {
            group: dict(zip(map(str, SEEDS), values))
            for group, values in clearance_mae.items()
        },
        "max_seed_clearance_source_height_macro_mae_m": {
            group: max(values) for group, values in clearance_mae.items()
        },
        "raw_prediction_out_of_target_range_fraction_by_seed": {
            str(seed): by[(seed, ARMS[1])][
                "raw_prediction_out_of_target_range_fraction"
            ]
            for seed in SEEDS
        },
    }
    gates = {
        "clearance_median_seed_micro_f1": (
            aggregates["clearance_median_seed_micro_f1"] >= 0.6
        ),
        "median_micro_f1_delta": (
            aggregates["median_micro_f1_delta"] >= 0.05
        ),
        "each_seed_micro_f1_delta_positive": all(
            value > 0.0 for value in f1_deltas
        ),
        "body_median_seed_f1_delta": (
            aggregates["median_f1_delta_by_height"]["body"] >= 0.0
        ),
        "head_median_seed_f1_delta": (
            aggregates["median_f1_delta_by_height"]["head"] >= 0.0
        ),
        "worst_fresh_parent_median_seed_f1_delta": (
            min(aggregates["median_f1_delta_by_source"].values()) >= 0.0
        ),
        "worst_fresh_parent_absolute_median_seed_f1": (
            min(aggregates["median_clearance_f1_by_source"].values())
            >= 0.4
        ),
        "median_seed_recall_delta": (
            aggregates["median_recall_delta"] >= -0.02
        ),
        "median_seed_false_positive_rate_delta": (
            aggregates["median_false_positive_rate_delta"] <= 0.02
        ),
        "overall_clearance_mae": (
            aggregates["max_seed_clearance_source_height_macro_mae_m"][
                "overall"
            ]
            <= 0.1
        ),
        "risk_clearance_mae": (
            aggregates["max_seed_clearance_source_height_macro_mae_m"][
                "risk"
            ]
            <= 0.15
        ),
        "safe_clearance_mae": (
            aggregates["max_seed_clearance_source_height_macro_mae_m"][
                "safe"
            ]
            <= 0.15
        ),
        "near_boundary_clearance_mae": (
            aggregates["max_seed_clearance_source_height_macro_mae_m"][
                "near"
            ]
            <= 0.1
        ),
        "target_unknown_to_safe_violations": (
            unknown_to_safe_violation_count == 0
        ),
    }
    terminal = SUPPORTED if all(gates.values()) else NOT_SUPPORTED
    return aggregates, gates, terminal


def _ordered_prediction_key_sha256(
    predictions: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for record in predictions:
        digest.update(
            _canonical_bytes(
                {
                    "seed": record["seed"],
                    "arm": record["arm"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "sample_id": record["sample_id"],
                }
            )
            + b"\n"
        )
    return digest.hexdigest()


def validate(
    contract_path: Path,
    package_validation_path: Path,
    truth_path: Path,
    completion_path: Path,
    predictions_path: Path,
    join_path: Path,
    result_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    _canonical_paths(
        contract_path,
        package_validation_path,
        truth_path,
        completion_path,
        predictions_path,
        join_path,
        result_path,
        output_root,
    )
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
    ):
        raise ValueError("Frozen fresh contract identity mismatch")
    _implementation_receipts(contract)
    package_validation = _load_json(package_validation_path)
    prediction_authorization_path = (
        package_validation_path.parent / "prediction_authorization.json"
    )
    prediction_authorization = _load_json(
        prediction_authorization_path
    )
    prediction_authorization_decision = (
        prediction_authorization.get("authorization", {})
    )
    completion = _load_json(completion_path)
    prediction_receipt_path = (
        completion_path.parent / "execution_receipt.json"
    )
    prediction_receipt = _load_json(prediction_receipt_path)
    join = _load_json(join_path)
    result = _load_json(result_path)
    contract_sha = _sha256(contract_path)
    package_sha = _sha256(package_validation_path)
    prediction_authorization_sha = _sha256(
        prediction_authorization_path
    )
    completion_sha = _sha256(completion_path)
    predictions_sha = _sha256(predictions_path)
    truth_sha = _sha256(truth_path)
    join_sha = _sha256(join_path)
    checkpoints = _checkpoint_contract(contract)
    expected_checkpoint_receipts = [
        {
            "seed": item["seed"],
            "arm": item["arm"],
            "checkpoint_sha256": item["sha256"],
        }
        for item in checkpoints
    ]
    if (
        {path.name for path in completion_path.parent.iterdir()}
        != {
            "execution_receipt.json",
            "predictions.jsonl",
            "completion.json",
        }
        or {path.name for path in result_path.parent.iterdir()}
        != {"truth_join_receipt.json", "effect_result.json"}
        or package_validation.get("schema") != PACKAGE_SCHEMA
        or package_validation.get("terminal") != PACKAGE_READY
        or package_validation.get("contract_sha256") != contract_sha
        or package_validation.get("package_validator_sha256")
        != contract["implementations"]["fresh_package_validator"]["sha256"]
        or package_validation.get("truth_labels_path")
        != str(truth_path.resolve())
        or package_validation.get("truth_labels_sha256") != truth_sha
        or package_validation.get("truth_label_count") != 75
        or package_validation.get("unknown_to_safe_violation_count") != 0
        or set(prediction_authorization)
        != PREDICTION_AUTHORIZATION_KEYS
        or set(prediction_authorization_decision)
        != PREDICTION_AUTHORIZATION_DECISION_KEYS
        or prediction_authorization.get("schema")
        != PREDICTION_AUTHORIZATION_SCHEMA
        or prediction_authorization.get("terminal")
        != PREDICTION_AUTHORIZED
        or prediction_authorization.get("contract_sha256") != contract_sha
        or prediction_authorization.get("prediction_inputs_path")
        != package_validation.get("prediction_inputs_path")
        or prediction_authorization.get("prediction_inputs_sha256")
        != package_validation.get("prediction_inputs_sha256")
        or prediction_authorization.get("source_order")
        != package_validation.get("source_order")
        or prediction_authorization.get("source_frame_indices")
        != package_validation.get("source_frame_indices")
        or prediction_authorization_decision.get(
            "fresh_prediction_authorized"
        )
        is not True
        or prediction_authorization_decision.get(
            "truth_join_authorized_before_predictions_frozen"
        )
        is not False
        or prediction_authorization_decision.get(
            "source_replacement_or_package_rematerialization"
        )
        is not False
        or prediction_receipt.get("schema")
        != PREDICTION_RECEIPT_SCHEMA
        or prediction_receipt.get("status")
        != "STARTED_BEFORE_FIRST_FRESH_FORWARD"
        or prediction_receipt.get("contract_sha256") != contract_sha
        or prediction_receipt.get("prediction_authorization_sha256")
        != prediction_authorization_sha
        or prediction_receipt.get("predictor_sha256")
        != contract["implementations"]["fresh_predictor"]["sha256"]
        or prediction_receipt.get("checkpoint_receipts")
        != expected_checkpoint_receipts
        or prediction_receipt.get("truth_files_opened") is not False
        or prediction_receipt.get("teacher_files_opened") is not False
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("terminal") != PREDICTIONS_READY
        or completion.get("contract_sha256") != contract_sha
        or completion.get("prediction_authorization_sha256")
        != prediction_authorization_sha
        or completion.get("predictor_sha256")
        != contract["implementations"]["fresh_predictor"]["sha256"]
        or completion.get("execution_receipt_sha256")
        != _sha256(prediction_receipt_path)
        or completion.get("prediction_inputs_sha256")
        != prediction_receipt.get("prediction_inputs_sha256")
        or completion.get("training_validation_sha256")
        != prediction_receipt.get("training_validation_sha256")
        or completion.get("predictions_path")
        != str(predictions_path.resolve())
        or completion.get("predictions_sha256") != predictions_sha
        or completion.get("prediction_count") != 450
        or completion.get("ordered_prediction_key_sha256")
        != _ordered_prediction_key_sha256(_load_jsonl(predictions_path))
        or completion.get("checkpoint_receipts")
        != expected_checkpoint_receipts
        or completion.get("raw_task_output_shape") != [2, 6, 6]
        or completion.get("risk_probability_shape") != [2, 6, 6]
        or completion.get("known_probability_shape") != [2, 6, 6]
        or completion.get("all_outputs_finite") is not True
        or completion.get("truth_files_opened") is not False
        or completion.get("teacher_files_opened") is not False
        or completion.get("truth_join_authorized") is not True
        or completion.get("second_prediction_run_authorized") is not False
        or set(join) != JOIN_KEYS
        or join.get("schema") != JOIN_SCHEMA
        or join.get("status")
        != "FROZEN_PREDICTIONS_GLOBALLY_CONSUMED_BEFORE_TRUTH_OPEN"
        or join.get("execution_contract_sha256") != contract_sha
        or join.get("package_validation_sha256") != package_sha
        or join.get("completion_sha256") != completion_sha
        or join.get("predictions_sha256") != predictions_sha
        or join.get("expected_truth_sha256") != truth_sha
        or join.get("truth_join_exactly_once") is not True
        or join.get("second_model_forward_authorized") is not False
        or join.get("source_replacement_authorized") is not False
        or set(result) != RESULT_KEYS
        or result.get("schema") != RESULT_SCHEMA
    ):
        raise ValueError("Independent fresh parent/hash/firewall mismatch")
    predictions = _load_jsonl(predictions_path)
    truth_records = _load_jsonl(truth_path)
    truth_records, prediction_by_key = _exact_records(
        contract,
        package_validation,
        truth_records,
        predictions,
    )
    source_order = package_validation["source_order"]
    runs = [
        _recompute_run(
            checkpoint,
            source_order,
            truth_records,
            prediction_by_key,
        )
        for checkpoint in checkpoints
    ]
    aggregates, gates, terminal = _recompute_decision(
        runs,
        source_order,
        int(package_validation["unknown_to_safe_violation_count"]),
    )
    expected_result = {
        "schema": RESULT_SCHEMA,
        "terminal": terminal,
        "workflow_profile": "FORMAL_ONE_SHOT_FRESH_EVALUATION",
        "claim_ceiling": (
            "FRESH_CURRENT_SYNTHETIC_PROXY_LEARNABILITY_ONLY"
        ),
        "parents": {
            "execution_contract_sha256": contract_sha,
            "package_validation_sha256": package_sha,
            "completion_sha256": completion_sha,
            "predictions_sha256": predictions_sha,
            "truth_sha256": truth_sha,
            "truth_join_receipt_sha256": join_sha,
        },
        "run_metrics": runs,
        "aggregates": aggregates,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "fresh_firewall": {
            "prediction_forward_count": 450,
            "truth_join_count": 1,
            "second_model_forward_executed": False,
            "checkpoint_substitution_executed": False,
            "threshold_change_executed": False,
            "source_replacement_executed": False,
            "reserved_heldout_opened": False,
        },
        "authorization": {
            "causal_transport_contract_may_be_frozen": (
                terminal == SUPPORTED
            ),
            "same_cohort_rescue_authorized": False,
            "reserved_official_test_authorized": False,
            "future_or_temporal_experiment_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }
    if result != expected_result:
        raise ValueError("Independent fresh effect recomputation mismatch")
    return {
        "schema": VALIDATION_SCHEMA,
        "terminal": VALIDATED,
        "effect_terminal": terminal,
        "contract_sha256": contract_sha,
        "package_validation_sha256": package_sha,
        "prediction_completion_sha256": completion_sha,
        "prediction_execution_receipt_sha256": _sha256(
            prediction_receipt_path
        ),
        "predictions_sha256": predictions_sha,
        "truth_sha256": truth_sha,
        "truth_join_receipt_sha256": join_sha,
        "effect_result_sha256": _sha256(result_path),
        "result_validator_sha256": _sha256(Path(__file__).resolve()),
        "checks": {
            "exact_75_truth_records": True,
            "exact_450_prediction_cartesian_records": True,
            "prediction_completion_hashes_exact": True,
            "truth_join_receipt_hashes_exact": True,
            "risk_metrics_independently_recomputed": True,
            "source_height_clearance_mae_independently_recomputed": True,
            "all_frozen_gates_independently_recomputed": True,
            "every_seed_clearance_mae_gate_enforced": True,
            "effect_result_and_terminal_exact": True,
            "fresh_firewalls_preserved": True,
            "second_model_forward_performed": False,
        },
        "authorization": {
            "fresh_effect_terminal_final": True,
            "same_cohort_rescue_authorized": False,
            "reserved_official_test_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }


def main() -> int:
    repository = _repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=repository
        / "docs/research/hftf/"
        "HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_"
        "EXECUTION_CONTRACT_D1_2026-08-01.json",
    )
    parser.add_argument(
        "--package-validation",
        type=Path,
        default=repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-validation-20260801/validation.json",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-package-20260801/truth_labels.jsonl",
    )
    parser.add_argument(
        "--completion",
        type=Path,
        default=repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-predictions-20260801/completion.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-predictions-20260801/predictions.jsonl",
    )
    parser.add_argument(
        "--truth-join-receipt",
        type=Path,
        default=repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-effect-20260801/truth_join_receipt.json",
    )
    parser.add_argument(
        "--effect-result",
        type=Path,
        default=repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-effect-20260801/effect_result.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-d1-fresh-effect-validation-20260801",
    )
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    try:
        if output_root.exists():
            raise FileExistsError(
                "Refusing to overwrite fresh effect validation"
            )
        report = validate(
            arguments.contract.resolve(),
            arguments.package_validation.resolve(),
            arguments.truth.resolve(),
            arguments.completion.resolve(),
            arguments.predictions.resolve(),
            arguments.truth_join_receipt.resolve(),
            arguments.effect_result.resolve(),
            output_root,
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        output_root.mkdir()
        _atomic_json(output_root / "validation.json", report)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "effect_terminal": report["effect_terminal"],
                    "validation_sha256": _sha256(
                        output_root / "validation.json"
                    ),
                }
            )
        )
        return 0
    except (
        FileExistsError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
