#!/usr/bin/env python3
"""Join frozen truth-blind predictions to heldout truth and apply F0.1 gates."""

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

from materialize_stage_c_f0_1_heldout_package import (
    CONTRACT_SCHEMA,
    EXPECTED_CONTRACT_STATUS,
    _canonical_artifact_path,
    _implementation_receipt,
)
from run_geometry_teacher_canary import _sha256
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = "blindassist_hftf_stage_c_f0_1_heldout_effect_result"
PREDICTION_SCHEMA = "blindassist_hftf_stage_c_f0_1_heldout_prediction"
PREDICTION_RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_prediction_execution_receipt"
)
COMPLETION_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_prediction_completion"
)
PREDICTIONS_READY = "F0_1_SANPO_HELDOUT_PREDICTIONS_FROZEN"
JOIN_RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_truth_join_execution_receipt"
)
SUPPORTED = (
    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_SIGNAL_SUPPORTED"
)
NOT_SUPPORTED = (
    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_SIGNAL_"
    "NOT_SUPPORTED_STOP"
)
NOT_EVALUABLE = (
    "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_STUDENT_CANARY_NOT_EVALUABLE"
)
PACKAGE_VALIDATED = "F0_1_SANPO_HELDOUT_PACKAGE_VALIDATED"
PREDICTION_KEYS = {
    "schema",
    "prediction_index",
    "seed",
    "arm",
    "checkpoint_sha256",
    "sample_id",
    "session_id",
    "anchor_timeline_index",
    "risk_probability",
    "known_probability",
}
TRUTH_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "anchor_timeline_index",
    "anchor_source_frame_index",
    "labels",
}
EXECUTION_RECEIPT_KEYS = {
    "schema",
    "status",
    "contract_sha256",
    "package_validation_sha256",
    "inference_inputs_sha256",
    "predictor_sha256",
    "checkpoint_sha256_by_seed_arm",
    "first_forward_consumes_one_shot",
    "truth_or_teacher_receipt_opened",
}
COMPLETION_KEYS = {
    "schema",
    "terminal",
    "contract_sha256",
    "package_validation_sha256",
    "inference_inputs_sha256",
    "predictor_sha256",
    "execution_receipt_sha256",
    "predictions_file",
    "predictions_sha256",
    "prediction_record_count",
    "ordered_join_key_sha256",
    "all_probabilities_shape_2x6x6_finite_and_in_unit_interval",
    "truth_or_teacher_receipt_opened",
    "one_shot_consumed",
    "consumption_ledger_sha256",
    "second_prediction_run_authorized",
    "truth_join_authorized",
}
JOIN_RECEIPT_KEYS = {
    "schema",
    "status",
    "contract_sha256",
    "package_validation_sha256",
    "prediction_completion_sha256",
    "truth_opened_before_receipt",
    "rerun_authorized",
}
PACKAGE_VALIDATION_KEYS = {
    "schema",
    "terminal",
    "contract_sha256",
    "package_validator_sha256",
    "source_lock_sha256",
    "teacher_opportunity_sha256",
    "package_manifest_sha256",
    "package_root",
    "files",
    "checks",
    "authorization",
}
LEDGER_KEYS = {
    "schema",
    "status",
    "contract_sha256",
    "package_validation_sha256",
    "inference_inputs_sha256",
    "predictor_sha256",
    "prediction_output_root",
    "first_seed",
    "first_arm",
    "rerun_authorized",
}


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _strict_binary_matrix(value: Any, description: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (2, 6, 6):
        raise ValueError(f"{description} shape mismatch")
    for item in raw.flat:
        if type(item) is not int or item not in (0, 1):
            raise ValueError(f"{description} must contain exact JSON integers 0/1")
    return np.asarray(raw, dtype=np.uint8)


def _probability_matrix(value: Any) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (2, 6, 6):
        raise ValueError("Prediction probability matrix is invalid")
    for item in raw.flat:
        if type(item) not in (int, float):
            raise ValueError("Prediction probability values must be JSON numbers")
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != (2, 6, 6)
        or not np.isfinite(array).all()
        or (array < 0.0).any()
        or (array > 1.0).any()
    ):
        raise ValueError("Prediction probability matrix is invalid")
    return array


def _truth_matrix(label: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if set(label) != {"known_target", "risk_target_nullable"}:
        raise ValueError("Heldout truth label key set mismatch")
    known = _strict_binary_matrix(
        label["known_target"], "Heldout truth known target"
    )
    risk_object = np.asarray(label["risk_target_nullable"], dtype=object)
    if (
        known.shape != (2, 6, 6)
        or risk_object.shape != (2, 6, 6)
    ):
        raise ValueError("Heldout truth label shape/value mismatch")
    numeric = np.vectorize(lambda value: value is not None)(risk_object)
    if not np.array_equal(numeric, known.astype(bool)):
        raise ValueError("Heldout truth UNKNOWN/null mask mismatch")
    risk = np.zeros((2, 6, 6), dtype=np.uint8)
    for index in np.argwhere(numeric):
        position = tuple(int(value) for value in index)
        value = risk_object[position]
        if type(value) is not int or value not in (0, 1):
            raise ValueError("Heldout known risk target must be binary")
        risk[position] = int(value)
    return known.astype(bool), risk.astype(bool)


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key in target:
        target[key] += source[key]


def _counts(
    probability: np.ndarray,
    truth: np.ndarray,
    known: np.ndarray,
) -> dict[str, int]:
    prediction = probability >= 0.5
    return {
        "tp": int((prediction & truth & known).sum()),
        "fp": int((prediction & ~truth & known).sum()),
        "fn": int((~prediction & truth & known).sum()),
        "tn": int((~prediction & ~truth & known).sum()),
    }


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"],
    )
    f1_denominator = 2 * tp + fp + fn
    recall_denominator = tp + fn
    fpr_denominator = fp + tn
    return {
        **counts,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "recall": tp / recall_denominator if recall_denominator else 0.0,
        "false_positive_rate": (
            fp / fpr_denominator if fpr_denominator else 0.0
        ),
        "positive_truth_count": recall_denominator,
        "negative_truth_count": fpr_denominator,
    }


def _validate_exact_sets(
    contract: dict[str, Any],
    predictions: list[dict[str, Any]],
    truth: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[int, str, str, str], dict[str, Any]],
]:
    sample_ids = [
        f"hftf_f0_1_heldout_{session_id}_{anchor:02d}"
        for session_id in contract["heldout_source_contract"]["source_order"]
        for anchor in range(8, 21)
    ]
    truth_by_sample: dict[str, dict[str, Any]] = {}
    if len(truth) != 39:
        raise ValueError("Heldout truth record count mismatch")
    for expected_sample_id, record in zip(sample_ids, truth, strict=True):
        if (
            set(record) != TRUTH_KEYS
            or record.get("schema")
            != "blindassist_hftf_f0_1_heldout_truth"
            or record.get("sample_id") != expected_sample_id
            or record["sample_id"] in truth_by_sample
            or set(record.get("labels", {})) != {"current", "future"}
        ):
            raise ValueError("Heldout truth exact schema/order mismatch")
        _truth_matrix(record["labels"]["current"])
        _truth_matrix(record["labels"]["future"])
        truth_by_sample[record["sample_id"]] = record
    expected_keys = [
        (
            int(checkpoint["seed"]),
            str(checkpoint["arm"]),
            str(checkpoint["sha256"]),
            sample_id,
        )
        for checkpoint in contract["checkpoint_contract"]["checkpoints"]
        for sample_id in sample_ids
    ]
    prediction_by_key: dict[
        tuple[int, str, str, str], dict[str, Any]
    ] = {}
    if len(predictions) != 351:
        raise ValueError("Heldout prediction record count mismatch")
    for expected_index, (expected_key, record) in enumerate(
        zip(expected_keys, predictions, strict=True)
    ):
        key = (
            int(record.get("seed", -1)),
            str(record.get("arm", "")),
            str(record.get("checkpoint_sha256", "")),
            str(record.get("sample_id", "")),
        )
        truth_record = truth_by_sample.get(key[3])
        if (
            set(record) != PREDICTION_KEYS
            or record.get("schema") != PREDICTION_SCHEMA
            or record.get("prediction_index") != expected_index
            or key != expected_key
            or key in prediction_by_key
            or truth_record is None
            or record.get("session_id") != truth_record["session_id"]
            or record.get("anchor_timeline_index")
            != truth_record["anchor_timeline_index"]
        ):
            raise ValueError("Heldout prediction exact Cartesian key mismatch")
        _probability_matrix(record["risk_probability"])
        _probability_matrix(record["known_probability"])
        prediction_by_key[key] = record
    if set(prediction_by_key) != set(expected_keys):
        raise ValueError("Heldout prediction missing/extra key mismatch")
    return truth_by_sample, prediction_by_key


def _ordered_join_key_sha256(
    predictions: list[dict[str, Any]],
) -> str:
    hasher = hashlib.sha256()
    for record in predictions:
        key = {
            "seed": record["seed"],
            "arm": record["arm"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "sample_id": record["sample_id"],
        }
        hasher.update(_canonical_bytes(key) + b"\n")
    return hasher.hexdigest()


def _run_metrics(
    contract: dict[str, Any],
    checkpoint: dict[str, Any],
    truth_by_sample: dict[str, dict[str, Any]],
    prediction_by_key: dict[
        tuple[int, str, str, str], dict[str, Any]
    ],
) -> dict[str, Any]:
    seed = int(checkpoint["seed"])
    arm = str(checkpoint["arm"])
    checkpoint_sha256 = str(checkpoint["sha256"])
    horizon = contract["metric_contract"]["arm_target_mapping"][arm]
    micro = _empty_counts()
    by_height = {
        "body": _empty_counts(),
        "head": _empty_counts(),
    }
    by_source = {
        session_id: _empty_counts()
        for session_id in contract["heldout_source_contract"]["source_order"]
    }
    known_correct = 0
    known_total = 0
    for sample_id, truth_record in truth_by_sample.items():
        prediction = prediction_by_key[
            (seed, arm, checkpoint_sha256, sample_id)
        ]
        risk_probability = _probability_matrix(
            prediction["risk_probability"]
        )
        known_probability = _probability_matrix(
            prediction["known_probability"]
        )
        known, risk = _truth_matrix(truth_record["labels"][horizon])
        sample_counts = _counts(risk_probability, risk, known)
        _add_counts(micro, sample_counts)
        for height_index, height in enumerate(("body", "head")):
            _add_counts(
                by_height[height],
                _counts(
                    risk_probability[height_index],
                    risk[height_index],
                    known[height_index],
                ),
            )
        _add_counts(by_source[truth_record["session_id"]], sample_counts)
        known_correct += int(
            ((known_probability >= 0.5) == known).sum()
        )
        known_total += known.size
    return {
        "seed": seed,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha256,
        "truth_horizon": horizon,
        "risk_micro": _metrics(micro),
        "risk_by_height": {
            height: _metrics(counts)
            for height, counts in by_height.items()
        },
        "risk_by_source": {
            session_id: _metrics(counts)
            for session_id, counts in by_source.items()
        },
        "known_accuracy_diagnostic": known_correct / known_total,
    }


def _median(values: list[float]) -> float:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Frozen median requires three finite seed values")
    return float(statistics.median(values))


def _effect_decision(
    contract: dict[str, Any], runs: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, bool]]:
    by_seed_arm = {
        (run["seed"], run["arm"]): run for run in runs
    }
    seeds = (17, 29, 43)
    f1_delta = [
        by_seed_arm[(seed, "HIST_FUTURE")]["risk_micro"]["f1"]
        - by_seed_arm[(seed, "SF_FUTURE")]["risk_micro"]["f1"]
        for seed in seeds
    ]
    recall_delta = [
        by_seed_arm[(seed, "HIST_FUTURE")]["risk_micro"]["recall"]
        - by_seed_arm[(seed, "SF_FUTURE")]["risk_micro"]["recall"]
        for seed in seeds
    ]
    fpr_delta = [
        by_seed_arm[(seed, "HIST_FUTURE")]["risk_micro"][
            "false_positive_rate"
        ]
        - by_seed_arm[(seed, "SF_FUTURE")]["risk_micro"][
            "false_positive_rate"
        ]
        for seed in seeds
    ]
    height_delta = {
        height: [
            by_seed_arm[(seed, "HIST_FUTURE")]["risk_by_height"][height][
                "f1"
            ]
            - by_seed_arm[(seed, "SF_FUTURE")]["risk_by_height"][height][
                "f1"
            ]
            for seed in seeds
        ]
        for height in ("body", "head")
    }
    source_delta = {
        session_id: [
            by_seed_arm[(seed, "HIST_FUTURE")]["risk_by_source"][
                session_id
            ]["f1"]
            - by_seed_arm[(seed, "SF_FUTURE")]["risk_by_source"][
                session_id
            ]["f1"]
            for seed in seeds
        ]
        for session_id in contract["heldout_source_contract"]["source_order"]
    }
    source_medians = {
        session_id: _median(values)
        for session_id, values in source_delta.items()
    }
    sf_current_f1 = [
        by_seed_arm[(seed, "SF_CURRENT")]["risk_micro"]["f1"]
        for seed in seeds
    ]
    aggregates = {
        "micro_f1_delta_by_seed": dict(zip(map(str, seeds), f1_delta)),
        "median_micro_f1_delta": _median(f1_delta),
        "recall_delta_by_seed": dict(zip(map(str, seeds), recall_delta)),
        "median_recall_delta": _median(recall_delta),
        "false_positive_rate_delta_by_seed": dict(
            zip(map(str, seeds), fpr_delta)
        ),
        "median_false_positive_rate_delta": _median(fpr_delta),
        "f1_delta_by_height_and_seed": {
            height: dict(zip(map(str, seeds), values))
            for height, values in height_delta.items()
        },
        "median_f1_delta_by_height": {
            height: _median(values)
            for height, values in height_delta.items()
        },
        "f1_delta_by_source_and_seed": {
            session_id: dict(zip(map(str, seeds), values))
            for session_id, values in source_delta.items()
        },
        "median_f1_delta_by_source": source_medians,
        "worst_source_median_seed_f1_delta": min(
            source_medians.values()
        ),
        "sf_current_micro_f1_by_seed": dict(
            zip(map(str, seeds), sf_current_f1)
        ),
        "sf_current_median_seed_micro_f1": _median(sf_current_f1),
    }
    gates_contract = contract["student_effect_gates"]
    gates = {
        "median_micro_f1_delta": (
            aggregates["median_micro_f1_delta"]
            >= gates_contract[
                "primary_median_seed_micro_f1_delta_minimum"
            ]
        ),
        "each_seed_micro_f1_delta_positive": all(
            value > 0.0 for value in f1_delta
        ),
        "median_recall_delta": (
            aggregates["median_recall_delta"]
            >= gates_contract["median_seed_recall_delta_minimum"]
        ),
        "median_false_positive_rate_delta": (
            aggregates["median_false_positive_rate_delta"]
            <= gates_contract[
                "median_seed_false_positive_rate_delta_maximum"
            ]
        ),
        "median_body_f1_delta": (
            aggregates["median_f1_delta_by_height"]["body"]
            >= gates_contract["median_seed_f1_delta_body_minimum"]
        ),
        "median_head_f1_delta": (
            aggregates["median_f1_delta_by_height"]["head"]
            >= gates_contract["median_seed_f1_delta_head_minimum"]
        ),
        "worst_source_median_seed_f1_delta": (
            aggregates["worst_source_median_seed_f1_delta"]
            >= gates_contract[
                "worst_source_median_seed_f1_delta_minimum"
            ]
        ),
        "sf_current_learnability": (
            aggregates["sf_current_median_seed_micro_f1"]
            >= gates_contract[
                "sf_current_median_seed_micro_f1_minimum"
            ]
        ),
    }
    return aggregates, gates


def evaluate(
    contract_path: Path,
    package_validation_path: Path,
    truth_path: Path,
    prediction_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != EXPECTED_CONTRACT_STATUS
    ):
        raise ValueError("Frozen heldout contract identity mismatch")
    _implementation_receipt(
        contract, "heldout_truth_join", Path(__file__).resolve()
    )
    _canonical_artifact_path(
        contract,
        "heldout_package_validation_root",
        package_validation_path.parent,
    )
    _canonical_artifact_path(
        contract, "heldout_package_root", truth_path.parent
    )
    _canonical_artifact_path(
        contract, "heldout_predictions_root", prediction_root
    )
    _canonical_artifact_path(
        contract, "heldout_effect_result_root", output_root
    )
    if package_validation_path.name != "validation.json":
        raise ValueError("Heldout package validation filename mismatch")
    if truth_path.name != "heldout_truth.jsonl":
        raise ValueError("Heldout truth filename mismatch")
    if {path.name for path in output_root.iterdir()} != {
        "execution_receipt.json"
    }:
        raise ValueError("Truth join output root did not start from receipt only")
    package_validation = _load_json(package_validation_path)
    join_receipt_path = output_root / "execution_receipt.json"
    join_receipt = _load_json(join_receipt_path)
    if (
        set(join_receipt) != JOIN_RECEIPT_KEYS
        or set(package_validation) != PACKAGE_VALIDATION_KEYS
        or join_receipt.get("schema") != JOIN_RECEIPT_SCHEMA
        or join_receipt.get("status") != "STARTED_BEFORE_TRUTH_OPEN"
        or join_receipt.get("contract_sha256") != _sha256(contract_path)
        or join_receipt.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or join_receipt.get("prediction_completion_sha256")
        != _sha256(prediction_root / "completion.json")
        or join_receipt.get("truth_opened_before_receipt") is not False
        or join_receipt.get("rerun_authorized") is not False
        or package_validation.get("terminal") != PACKAGE_VALIDATED
        or package_validation.get("schema")
        != "blindassist_hftf_stage_c_f0_1_heldout_package_validation"
        or package_validation.get("contract_sha256") != _sha256(contract_path)
        or package_validation.get("package_validator_sha256")
        != contract["implementations"]["heldout_package_validator"][
            "sha256"
        ]
        or package_validation.get("source_lock_sha256")
        != contract["parents"]["source_lock"]["sha256"]
        or package_validation.get("teacher_opportunity_sha256")
        != contract["parents"]["teacher_opportunity_report"]["sha256"]
        or package_validation.get("files", {})
        .get("heldout_truth.jsonl", {})
        .get("sha256")
        != _sha256(truth_path)
        or set(package_validation.get("files", {}))
        != {
            "inference_inputs.jsonl",
            "heldout_truth.jsonl",
            "teacher_receipts.jsonl",
        }
        or any(
            set(metadata) != {"sha256", "record_count"}
            or metadata.get("record_count") != 39
            for metadata in package_validation.get("files", {}).values()
        )
        or package_validation.get("authorization")
        != {
            "one_shot_prediction_authorized": True,
            "truth_join_authorized_before_predictions_frozen": False,
            "package_rematerialization_authorized": False,
        }
        or package_validation.get("checks")
        != {
            "exact_source_anchor_and_sample_order": True,
            "inference_schema_and_history_hashes_exact": True,
            "truth_schema_shape_null_mask_exact": True,
            "receipt_schema_causal_binding_exact": True,
            "truth_reaggregates_to_frozen_reference_opportunity": True,
            "student_output_computed": False,
        }
    ):
        raise ValueError("Heldout truth validation authority mismatch")
    expected_truth_path = (
        Path(str(package_validation["package_root"]))
        / "heldout_truth.jsonl"
    ).resolve()
    if truth_path.resolve() != expected_truth_path:
        raise ValueError("Truth join path is not validated heldout truth")
    if {path.name for path in prediction_root.iterdir()} != {
        "execution_receipt.json",
        "predictions.jsonl",
        "completion.json",
    }:
        raise ValueError("Frozen prediction directory file set mismatch")
    completion = _load_json(prediction_root / "completion.json")
    execution_receipt = _load_json(
        prediction_root / "execution_receipt.json"
    )
    predictions_path = prediction_root / "predictions.jsonl"
    expected_checkpoint_receipts = {
        f"{checkpoint['seed']}:{checkpoint['arm']}": checkpoint["sha256"]
        for checkpoint in contract["checkpoint_contract"]["checkpoints"]
    }
    inference_inputs_sha256 = package_validation["files"][
        "inference_inputs.jsonl"
    ]["sha256"]
    predictor_sha256 = contract["implementations"]["heldout_predictor"][
        "sha256"
    ]
    repository_root = Path(__file__).resolve().parents[3]
    ledger_path = (
        repository_root
        / contract["canonical_artifact_paths"][
            "one_shot_consumption_ledger"
        ]
    ).resolve()
    ledger = _load_json(ledger_path)
    if (
        set(execution_receipt) != EXECUTION_RECEIPT_KEYS
        or set(completion) != COMPLETION_KEYS
        or execution_receipt.get("schema")
        != "blindassist_hftf_stage_c_f0_1_heldout_prediction_execution_receipt"
        or execution_receipt.get("status")
        != "STARTED_BEFORE_FIRST_HELDOUT_FORWARD"
        or execution_receipt.get("contract_sha256")
        != _sha256(contract_path)
        or execution_receipt.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or execution_receipt.get("inference_inputs_sha256")
        != inference_inputs_sha256
        or execution_receipt.get("predictor_sha256")
        != predictor_sha256
        or execution_receipt.get("checkpoint_sha256_by_seed_arm")
        != expected_checkpoint_receipts
        or execution_receipt.get("truth_or_teacher_receipt_opened")
        is not False
        or execution_receipt.get("first_forward_consumes_one_shot")
        is not True
        or completion.get("execution_receipt_sha256")
        != _sha256(prediction_root / "execution_receipt.json")
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("terminal") != PREDICTIONS_READY
        or completion.get("contract_sha256") != _sha256(contract_path)
        or completion.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or completion.get("inference_inputs_sha256")
        != inference_inputs_sha256
        or completion.get("predictor_sha256") != predictor_sha256
        or completion.get("consumption_ledger_sha256")
        != _sha256(ledger_path)
        or completion.get("predictions_sha256") != _sha256(predictions_path)
        or completion.get("prediction_record_count") != 351
        or completion.get("truth_or_teacher_receipt_opened") is not False
        or completion.get("truth_join_authorized") is not True
        or set(ledger) != LEDGER_KEYS
        or ledger.get("schema")
        != "blindassist_hftf_stage_c_f0_1_heldout_one_shot_consumption"
        or ledger.get("status")
        != "CONSUMED_CONSERVATIVELY_IMMEDIATELY_BEFORE_FIRST_FORWARD"
        or ledger.get("contract_sha256") != _sha256(contract_path)
        or ledger.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or ledger.get("inference_inputs_sha256")
        != inference_inputs_sha256
        or ledger.get("predictor_sha256") != predictor_sha256
        or Path(str(ledger.get("prediction_output_root", ""))).resolve()
        != prediction_root.resolve()
        or ledger.get("first_seed")
        != contract["checkpoint_contract"]["checkpoints"][0]["seed"]
        or ledger.get("first_arm")
        != contract["checkpoint_contract"]["checkpoints"][0]["arm"]
        or ledger.get("rerun_authorized") is not False
    ):
        raise ValueError("Frozen prediction completion mismatch")
    predictions = _load_jsonl(predictions_path)
    truth = _load_jsonl(truth_path)
    truth_by_sample, prediction_by_key = _validate_exact_sets(
        contract, predictions, truth
    )
    if completion.get("ordered_join_key_sha256") != _ordered_join_key_sha256(
        predictions
    ):
        raise ValueError("Frozen prediction ordered join-key digest mismatch")
    runs = [
        _run_metrics(
            contract,
            checkpoint,
            truth_by_sample,
            prediction_by_key,
        )
        for checkpoint in contract["checkpoint_contract"]["checkpoints"]
    ]
    for run in runs:
        if run["arm"] in {"SF_FUTURE", "HIST_FUTURE"}:
            if (
                run["risk_micro"]["positive_truth_count"] <= 0
                or run["risk_micro"]["negative_truth_count"] <= 0
                or any(
                    metrics["positive_truth_count"] <= 0
                    or metrics["negative_truth_count"] <= 0
                    for metrics in run["risk_by_height"].values()
                )
                or any(
                    metrics["positive_truth_count"] <= 0
                    or metrics["negative_truth_count"] <= 0
                    for metrics in run["risk_by_source"].values()
                )
            ):
                raise ValueError("Frozen future metric denominator is zero")
    aggregates, gates = _effect_decision(contract, runs)
    terminal = SUPPORTED if all(gates.values()) else NOT_SUPPORTED
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "SYNTHETIC_BODY_HEAD_GEOMETRY_PROXY_SIGNAL_ONLY",
        "contract_sha256": _sha256(contract_path),
        "package_validation_sha256": _sha256(package_validation_path),
        "heldout_truth_sha256": _sha256(truth_path),
        "prediction_completion_sha256": _sha256(
            prediction_root / "completion.json"
        ),
        "predictions_sha256": _sha256(predictions_path),
        "truth_join_implementation_sha256": _sha256(
            Path(__file__).resolve()
        ),
        "truth_join_execution_receipt_sha256": _sha256(join_receipt_path),
        "run_count": len(runs),
        "runs": runs,
        "primary_comparison": aggregates,
        "effect_gates": gates,
        "all_effect_gates_pass": all(gates.values()),
        "one_shot": {
            "prediction_record_count": len(predictions),
            "truth_record_count": len(truth),
            "prediction_cartesian_key_set_exact": True,
            "truth_join_by_exact_keys_not_position": True,
            "checkpoint_model_or_inference_input_opened_by_join": False,
        },
        "authorization": {
            "independent_terminal_validation_authorized": True,
            "second_prediction_run_authorized": False,
            "after_outcome_model_checkpoint_threshold_source_metric_or_gate_"
            "change_authorized": False,
            "mainline_promotion_authorized": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--package-validation", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    receipt_started = False
    try:
        if output_root.exists():
            raise FileExistsError("Refusing to overwrite heldout effect result")
        contract_path = args.contract.resolve()
        package_validation_path = args.package_validation.resolve()
        prediction_root = args.prediction_root.resolve()
        contract = _load_json(contract_path)
        if (
            contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("status") != EXPECTED_CONTRACT_STATUS
        ):
            raise ValueError("Frozen heldout contract identity mismatch")
        _canonical_artifact_path(
            contract, "heldout_effect_result_root", output_root
        )
        _canonical_artifact_path(
            contract,
            "heldout_package_validation_root",
            package_validation_path.parent,
        )
        _canonical_artifact_path(
            contract, "heldout_package_root", args.truth.resolve().parent
        )
        _canonical_artifact_path(
            contract, "heldout_predictions_root", prediction_root
        )
        if package_validation_path.name != "validation.json":
            raise ValueError("Heldout package validation filename mismatch")
        if args.truth.resolve().name != "heldout_truth.jsonl":
            raise ValueError("Heldout truth filename mismatch")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        output_root.mkdir()
        receipt = {
            "schema": JOIN_RECEIPT_SCHEMA,
            "status": "STARTED_BEFORE_TRUTH_OPEN",
            "contract_sha256": _sha256(contract_path),
            "package_validation_sha256": _sha256(
                package_validation_path
            ),
            "prediction_completion_sha256": _sha256(
                prediction_root / "completion.json"
            ),
            "truth_opened_before_receipt": False,
            "rerun_authorized": False,
        }
        _atomic_json(output_root / "execution_receipt.json", receipt)
        receipt_started = True
        report = evaluate(
            contract_path,
            package_validation_path,
            args.truth.resolve(),
            prediction_root,
            output_root,
        )
        _atomic_json(output_root / "result.json", report)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "all_effect_gates_pass": report[
                        "all_effect_gates_pass"
                    ],
                    "result_sha256": _sha256(
                        output_root / "result.json"
                    ),
                }
            )
        )
        return 0
    except Exception as error:
        if (
            receipt_started
            and not (output_root / "result.json").exists()
            and not (output_root / "failure.json").exists()
        ):
            try:
                _atomic_json(
                    output_root / "failure.json",
                    {
                        "terminal": NOT_EVALUABLE,
                        "stage": "truth_join",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "truth_join_rerun_authorized": False,
                        "prediction_rerun_authorized": False,
                    },
                )
            except OSError:
                pass
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
