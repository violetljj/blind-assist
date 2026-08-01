#!/usr/bin/env python3
"""Independently recompute the frozen F0.1 heldout terminal."""

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


SCHEMA = "blindassist_hftf_stage_c_f0_1_heldout_result_validation"
READY = "F0_1_SANPO_HELDOUT_EFFECT_TERMINAL_VALIDATED"
PREDICTION_SCHEMA = "blindassist_hftf_stage_c_f0_1_heldout_prediction"
RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_prediction_execution_receipt"
)
COMPLETION_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_prediction_completion"
)
PREDICTIONS_READY = "F0_1_SANPO_HELDOUT_PREDICTIONS_FROZEN"
JOIN_RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_truth_join_execution_receipt"
)
VALIDATION_RECEIPT_SCHEMA = (
    "blindassist_hftf_stage_c_f0_1_heldout_terminal_validation_execution_receipt"
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
RESULT_SCHEMA = "blindassist_hftf_stage_c_f0_1_heldout_effect_result"
TRUTH_KEYS = {
    "schema",
    "sample_id",
    "session_id",
    "anchor_timeline_index",
    "anchor_source_frame_index",
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
    "anchor_timeline_index",
    "risk_probability",
    "known_probability",
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
VALIDATION_RECEIPT_KEYS = {
    "schema",
    "status",
    "contract_sha256",
    "package_validation_sha256",
    "prediction_completion_sha256",
    "effect_result_sha256",
    "truth_opened_before_receipt",
    "rerun_authorized",
}
RESULT_KEYS = {
    "schema",
    "terminal",
    "workflow_profile",
    "claim_ceiling",
    "contract_sha256",
    "package_validation_sha256",
    "heldout_truth_sha256",
    "prediction_completion_sha256",
    "predictions_sha256",
    "truth_join_implementation_sha256",
    "truth_join_execution_receipt_sha256",
    "run_count",
    "runs",
    "primary_comparison",
    "effect_gates",
    "all_effect_gates_pass",
    "one_shot",
    "authorization",
    "claim_boundary",
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


def _decode_truth(label: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if set(label) != {"known_target", "risk_target_nullable"}:
        raise ValueError("Independent truth label key set mismatch")
    known = _strict_binary_matrix(
        label["known_target"], "Independent known target"
    )
    raw = np.asarray(label["risk_target_nullable"], dtype=object)
    if (
        known.shape != (2, 6, 6)
        or raw.shape != (2, 6, 6)
    ):
        raise ValueError("Independent truth shape mismatch")
    risk = np.zeros((2, 6, 6), dtype=bool)
    for index in np.ndindex(raw.shape):
        value = raw[index]
        if bool(known[index]):
            if type(value) is not int or value not in (0, 1):
                raise ValueError("Independent known truth mismatch")
            risk[index] = bool(value)
        elif value is not None:
            raise ValueError("Independent unknown truth must be null")
    return known.astype(bool), risk


def _decode_probability(value: Any) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (2, 6, 6):
        raise ValueError("Independent prediction matrix mismatch")
    for item in raw.flat:
        if type(item) not in (int, float):
            raise ValueError("Independent prediction values must be JSON numbers")
    probability = np.asarray(value, dtype=np.float64)
    if (
        probability.shape != (2, 6, 6)
        or not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise ValueError("Independent prediction matrix mismatch")
    return probability


def _score(
    probability: np.ndarray, truth: np.ndarray, known: np.ndarray
) -> dict[str, Any]:
    prediction = probability >= 0.5
    tp = int((prediction & truth & known).sum())
    fp = int((prediction & ~truth & known).sum())
    fn = int((~prediction & truth & known).sum())
    tn = int((~prediction & ~truth & known).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _sum(scores: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(score[key]) for score in scores)
        for key in ("tp", "fp", "fn", "tn")
    }


def _metric(score: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (
        score["tp"],
        score["fp"],
        score["fn"],
        score["tn"],
    )
    return {
        **score,
        "f1": (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn
            else 0.0
        ),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "positive_truth_count": tp + fn,
        "negative_truth_count": fp + tn,
    }


def _recompute_runs(
    contract: dict[str, Any],
    truth_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_samples = [
        (
            f"hftf_f0_1_heldout_{session_id}_{anchor:02d}",
            session_id,
            anchor,
        )
        for session_id in contract["heldout_source_contract"]["source_order"]
        for anchor in range(8, 21)
    ]
    truth: dict[str, dict[str, Any]] = {}
    if len(truth_records) != 39:
        raise ValueError("Independent truth key set mismatch")
    for record, (sample_id, session_id, anchor) in zip(
        truth_records, expected_samples, strict=True
    ):
        if (
            set(record) != TRUTH_KEYS
            or record.get("schema")
            != "blindassist_hftf_f0_1_heldout_truth"
            or record.get("sample_id") != sample_id
            or record.get("session_id") != session_id
            or record.get("anchor_timeline_index") != anchor
            or sample_id in truth
            or set(record.get("labels", {})) != {"current", "future"}
        ):
            raise ValueError("Independent truth schema/order mismatch")
        _decode_truth(record["labels"]["current"])
        _decode_truth(record["labels"]["future"])
        truth[sample_id] = record
    prediction: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    expected_prediction_keys = [
        (
            int(checkpoint["seed"]),
            str(checkpoint["arm"]),
            str(checkpoint["sha256"]),
            sample_id,
        )
        for checkpoint in contract["checkpoint_contract"]["checkpoints"]
        for sample_id, _, _ in expected_samples
    ]
    if len(prediction_records) != 351:
        raise ValueError("Independent prediction count mismatch")
    for expected_index, (record, expected_key) in enumerate(
        zip(prediction_records, expected_prediction_keys, strict=True)
    ):
        key = (
            int(record["seed"]),
            str(record["arm"]),
            str(record["checkpoint_sha256"]),
            str(record["sample_id"]),
        )
        truth_record = truth.get(key[3])
        if (
            set(record) != PREDICTION_KEYS
            or record.get("schema") != PREDICTION_SCHEMA
            or record.get("prediction_index") != expected_index
            or key != expected_key
            or key in prediction
            or truth_record is None
            or record.get("session_id") != truth_record["session_id"]
            or record.get("anchor_timeline_index")
            != truth_record["anchor_timeline_index"]
        ):
            raise ValueError("Independent duplicate prediction key")
        _decode_probability(record["risk_probability"])
        _decode_probability(record["known_probability"])
        prediction[key] = record
    if set(prediction) != set(expected_prediction_keys):
        raise ValueError("Independent prediction Cartesian set mismatch")
    runs: list[dict[str, Any]] = []
    for checkpoint in contract["checkpoint_contract"]["checkpoints"]:
        seed = int(checkpoint["seed"])
        arm = str(checkpoint["arm"])
        digest = str(checkpoint["sha256"])
        horizon = contract["metric_contract"]["arm_target_mapping"][arm]
        sample_scores: list[dict[str, Any]] = []
        height_scores = {"body": [], "head": []}
        source_scores = {
            session_id: []
            for session_id in contract["heldout_source_contract"][
                "source_order"
            ]
        }
        known_correct = 0
        known_total = 0
        for sample_id, truth_record in truth.items():
            record = prediction[(seed, arm, digest, sample_id)]
            risk_probability = _decode_probability(
                record["risk_probability"]
            )
            known_probability = _decode_probability(
                record["known_probability"]
            )
            known, risk = _decode_truth(truth_record["labels"][horizon])
            sample_score = _score(risk_probability, risk, known)
            sample_scores.append(sample_score)
            source_scores[truth_record["session_id"]].append(sample_score)
            for index, height in enumerate(("body", "head")):
                height_scores[height].append(
                    _score(
                        risk_probability[index],
                        risk[index],
                        known[index],
                    )
                )
            known_correct += int(
                ((known_probability >= 0.5) == known).sum()
            )
            known_total += int(known.size)
        runs.append(
            {
                "seed": seed,
                "arm": arm,
                "checkpoint_sha256": digest,
                "truth_horizon": horizon,
                "risk_micro": _metric(_sum(sample_scores)),
                "risk_by_height": {
                    height: _metric(_sum(scores))
                    for height, scores in height_scores.items()
                },
                "risk_by_source": {
                    session_id: _metric(_sum(scores))
                    for session_id, scores in source_scores.items()
                },
                "known_accuracy_diagnostic": known_correct / known_total,
            }
        )
    return runs


def _recompute_decision(
    contract: dict[str, Any], runs: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, bool]]:
    lookup = {(run["seed"], run["arm"]): run for run in runs}
    seeds = (17, 29, 43)

    def paired(
        getter: Any,
    ) -> list[float]:
        return [
            float(getter(lookup[(seed, "HIST_FUTURE")]))
            - float(getter(lookup[(seed, "SF_FUTURE")]))
            for seed in seeds
        ]

    f1 = paired(lambda run: run["risk_micro"]["f1"])
    recall = paired(lambda run: run["risk_micro"]["recall"])
    fpr = paired(lambda run: run["risk_micro"]["false_positive_rate"])
    by_height = {
        height: paired(
            lambda run, selected=height: run["risk_by_height"][selected][
                "f1"
            ]
        )
        for height in ("body", "head")
    }
    by_source = {
        session_id: paired(
            lambda run, selected=session_id: run["risk_by_source"][selected][
                "f1"
            ]
        )
        for session_id in contract["heldout_source_contract"]["source_order"]
    }
    source_median = {
        source: float(statistics.median(values))
        for source, values in by_source.items()
    }
    sf_current = [
        lookup[(seed, "SF_CURRENT")]["risk_micro"]["f1"]
        for seed in seeds
    ]
    aggregate = {
        "micro_f1_delta_by_seed": dict(zip(map(str, seeds), f1)),
        "median_micro_f1_delta": float(statistics.median(f1)),
        "recall_delta_by_seed": dict(zip(map(str, seeds), recall)),
        "median_recall_delta": float(statistics.median(recall)),
        "false_positive_rate_delta_by_seed": dict(
            zip(map(str, seeds), fpr)
        ),
        "median_false_positive_rate_delta": float(
            statistics.median(fpr)
        ),
        "f1_delta_by_height_and_seed": {
            height: dict(zip(map(str, seeds), values))
            for height, values in by_height.items()
        },
        "median_f1_delta_by_height": {
            height: float(statistics.median(values))
            for height, values in by_height.items()
        },
        "f1_delta_by_source_and_seed": {
            source: dict(zip(map(str, seeds), values))
            for source, values in by_source.items()
        },
        "median_f1_delta_by_source": source_median,
        "worst_source_median_seed_f1_delta": min(
            source_median.values()
        ),
        "sf_current_micro_f1_by_seed": dict(
            zip(map(str, seeds), sf_current)
        ),
        "sf_current_median_seed_micro_f1": float(
            statistics.median(sf_current)
        ),
    }
    frozen = contract["student_effect_gates"]
    gates = {
        "median_micro_f1_delta": aggregate["median_micro_f1_delta"]
        >= frozen["primary_median_seed_micro_f1_delta_minimum"],
        "each_seed_micro_f1_delta_positive": all(value > 0 for value in f1),
        "median_recall_delta": aggregate["median_recall_delta"]
        >= frozen["median_seed_recall_delta_minimum"],
        "median_false_positive_rate_delta": aggregate[
            "median_false_positive_rate_delta"
        ]
        <= frozen["median_seed_false_positive_rate_delta_maximum"],
        "median_body_f1_delta": aggregate["median_f1_delta_by_height"][
            "body"
        ]
        >= frozen["median_seed_f1_delta_body_minimum"],
        "median_head_f1_delta": aggregate["median_f1_delta_by_height"][
            "head"
        ]
        >= frozen["median_seed_f1_delta_head_minimum"],
        "worst_source_median_seed_f1_delta": aggregate[
            "worst_source_median_seed_f1_delta"
        ]
        >= frozen["worst_source_median_seed_f1_delta_minimum"],
        "sf_current_learnability": aggregate[
            "sf_current_median_seed_micro_f1"
        ]
        >= frozen["sf_current_median_seed_micro_f1_minimum"],
    }
    return aggregate, gates


def _validate_future_denominators(runs: list[dict[str, Any]]) -> None:
    for run in runs:
        if run["arm"] not in {"SF_FUTURE", "HIST_FUTURE"}:
            continue
        scopes = [
            run["risk_micro"],
            *run["risk_by_height"].values(),
            *run["risk_by_source"].values(),
        ]
        if any(
            metrics["positive_truth_count"] <= 0
            or metrics["negative_truth_count"] <= 0
            for metrics in scopes
        ):
            raise ValueError("Independent future metric denominator is zero")


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


def validate(
    contract_path: Path,
    package_validation_path: Path,
    truth_path: Path,
    prediction_root: Path,
    result_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != EXPECTED_CONTRACT_STATUS
    ):
        raise ValueError("Frozen heldout contract identity mismatch")
    _implementation_receipt(
        contract, "heldout_terminal_validator", Path(__file__).resolve()
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
        contract, "heldout_effect_result_root", result_path.parent
    )
    _canonical_artifact_path(
        contract, "heldout_terminal_validation_root", output_root
    )
    if package_validation_path.name != "validation.json":
        raise ValueError("Independent package validation filename mismatch")
    if truth_path.name != "heldout_truth.jsonl":
        raise ValueError("Independent truth filename mismatch")
    if result_path.name != "result.json":
        raise ValueError("Independent effect result filename mismatch")
    if {path.name for path in result_path.parent.iterdir()} != {
        "execution_receipt.json",
        "result.json",
    }:
        raise ValueError("Independent effect result directory mismatch")
    if {path.name for path in output_root.iterdir()} != {
        "execution_receipt.json"
    }:
        raise ValueError(
            "Independent validation output did not start from receipt only"
        )
    package_validation = _load_json(package_validation_path)
    completion = _load_json(prediction_root / "completion.json")
    execution_receipt = _load_json(
        prediction_root / "execution_receipt.json"
    )
    join_receipt_path = result_path.parent / "execution_receipt.json"
    join_receipt = _load_json(join_receipt_path)
    validation_receipt_path = output_root / "execution_receipt.json"
    validation_receipt = _load_json(validation_receipt_path)
    result = _load_json(result_path)
    predictions_path = prediction_root / "predictions.jsonl"
    if {path.name for path in prediction_root.iterdir()} != {
        "execution_receipt.json",
        "predictions.jsonl",
        "completion.json",
    }:
        raise ValueError("Independent prediction directory file set mismatch")
    repository_root = Path(__file__).resolve().parents[3]
    ledger_path = (
        repository_root
        / contract["canonical_artifact_paths"][
            "one_shot_consumption_ledger"
        ]
    ).resolve()
    ledger = _load_json(ledger_path)
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
    if (
        set(execution_receipt) != EXECUTION_RECEIPT_KEYS
        or set(completion) != COMPLETION_KEYS
        or set(join_receipt) != JOIN_RECEIPT_KEYS
        or set(validation_receipt) != VALIDATION_RECEIPT_KEYS
        or set(result) != RESULT_KEYS
        or set(package_validation) != PACKAGE_VALIDATION_KEYS
        or set(ledger) != LEDGER_KEYS
        or validation_receipt.get("schema") != VALIDATION_RECEIPT_SCHEMA
        or validation_receipt.get("status")
        != "STARTED_BEFORE_INDEPENDENT_TRUTH_OPEN"
        or validation_receipt.get("contract_sha256")
        != _sha256(contract_path)
        or validation_receipt.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or validation_receipt.get("prediction_completion_sha256")
        != _sha256(prediction_root / "completion.json")
        or validation_receipt.get("effect_result_sha256")
        != _sha256(result_path)
        or validation_receipt.get("truth_opened_before_receipt") is not False
        or validation_receipt.get("rerun_authorized") is not False
        or package_validation.get("schema")
        != "blindassist_hftf_stage_c_f0_1_heldout_package_validation"
        or package_validation.get("terminal") != PACKAGE_VALIDATED
        or package_validation.get("contract_sha256")
        != _sha256(contract_path)
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
        or execution_receipt.get("schema") != RECEIPT_SCHEMA
        or execution_receipt.get("status")
        != "STARTED_BEFORE_FIRST_HELDOUT_FORWARD"
        or execution_receipt.get("contract_sha256")
        != _sha256(contract_path)
        or execution_receipt.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or execution_receipt.get("inference_inputs_sha256")
        != inference_inputs_sha256
        or execution_receipt.get("predictor_sha256") != predictor_sha256
        or execution_receipt.get("checkpoint_sha256_by_seed_arm")
        != expected_checkpoint_receipts
        or execution_receipt.get("first_forward_consumes_one_shot")
        is not True
        or execution_receipt.get("truth_or_teacher_receipt_opened")
        is not False
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("terminal") != PREDICTIONS_READY
        or completion.get("contract_sha256") != _sha256(contract_path)
        or completion.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or completion.get("inference_inputs_sha256")
        != inference_inputs_sha256
        or completion.get("predictor_sha256") != predictor_sha256
        or completion.get("execution_receipt_sha256")
        != _sha256(prediction_root / "execution_receipt.json")
        or completion.get("predictions_sha256") != _sha256(predictions_path)
        or completion.get("prediction_record_count") != 351
        or completion.get("predictions_file") != "predictions.jsonl"
        or completion.get(
            "all_probabilities_shape_2x6x6_finite_and_in_unit_interval"
        )
        is not True
        or completion.get("truth_or_teacher_receipt_opened") is not False
        or completion.get("one_shot_consumed") is not True
        or completion.get("consumption_ledger_sha256")
        != _sha256(ledger_path)
        or completion.get("second_prediction_run_authorized") is not False
        or completion.get("truth_join_authorized") is not True
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
        or join_receipt.get("schema") != JOIN_RECEIPT_SCHEMA
        or join_receipt.get("status") != "STARTED_BEFORE_TRUTH_OPEN"
        or join_receipt.get("contract_sha256") != _sha256(contract_path)
        or join_receipt.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or join_receipt.get("prediction_completion_sha256")
        != _sha256(prediction_root / "completion.json")
        or join_receipt.get("truth_opened_before_receipt") is not False
        or join_receipt.get("rerun_authorized") is not False
        or result.get("schema") != RESULT_SCHEMA
        or result.get("terminal") not in {SUPPORTED, NOT_SUPPORTED}
        or result.get("contract_sha256") != _sha256(contract_path)
        or result.get("package_validation_sha256")
        != _sha256(package_validation_path)
        or result.get("heldout_truth_sha256") != _sha256(truth_path)
        or result.get("prediction_completion_sha256")
        != _sha256(prediction_root / "completion.json")
        or result.get("predictions_sha256") != _sha256(predictions_path)
        or result.get("truth_join_implementation_sha256")
        != contract["implementations"]["heldout_truth_join"]["sha256"]
        or result.get("truth_join_execution_receipt_sha256")
        != _sha256(join_receipt_path)
        or result.get("workflow_profile") != "DEVELOPMENT_STANDARD"
        or result.get("claim_ceiling")
        != "SYNTHETIC_BODY_HEAD_GEOMETRY_PROXY_SIGNAL_ONLY"
        or result.get("claim_boundary") != contract["claim_boundary"]
        or result.get("run_count") != 9
        or result.get("one_shot")
        != {
            "prediction_record_count": 351,
            "truth_record_count": 39,
            "prediction_cartesian_key_set_exact": True,
            "truth_join_by_exact_keys_not_position": True,
            "checkpoint_model_or_inference_input_opened_by_join": False,
        }
        or result.get("authorization")
        != {
            "independent_terminal_validation_authorized": True,
            "second_prediction_run_authorized": False,
            "after_outcome_model_checkpoint_threshold_source_metric_or_gate_"
            "change_authorized": False,
            "mainline_promotion_authorized": False,
        }
    ):
        raise ValueError("Independent heldout parent hash mismatch")
    truth = _load_jsonl(truth_path)
    predictions = _load_jsonl(predictions_path)
    if completion.get("ordered_join_key_sha256") != _ordered_join_key_sha256(
        predictions
    ):
        raise ValueError("Independent ordered join-key digest mismatch")
    runs = _recompute_runs(contract, truth, predictions)
    _validate_future_denominators(runs)
    aggregate, gates = _recompute_decision(contract, runs)
    terminal = SUPPORTED if all(gates.values()) else NOT_SUPPORTED
    if (
        result.get("runs") != runs
        or result.get("primary_comparison") != aggregate
        or result.get("effect_gates") != gates
        or result.get("all_effect_gates_pass") != all(gates.values())
        or result.get("terminal") != terminal
    ):
        raise ValueError("Independent heldout terminal recomputation mismatch")
    return {
        "schema": SCHEMA,
        "terminal": READY,
        "effect_terminal": terminal,
        "contract_sha256": _sha256(contract_path),
        "package_validation_sha256": _sha256(package_validation_path),
        "heldout_truth_sha256": _sha256(truth_path),
        "predictions_sha256": _sha256(predictions_path),
        "result_sha256": _sha256(result_path),
        "terminal_validator_sha256": _sha256(Path(__file__).resolve()),
        "terminal_validation_execution_receipt_sha256": _sha256(
            validation_receipt_path
        ),
        "checks": {
            "truth_and_prediction_hashes_match": True,
            "exact_prediction_cartesian_key_set": True,
            "all_metrics_independently_recomputed": True,
            "all_gate_aggregations_independently_recomputed": True,
            "effect_terminal_exact": True,
            "second_model_prediction_run_performed": False,
        },
        "authorization": {
            "heldout_effect_terminal_final": True,
            "after_outcome_rescue_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--package-validation", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    receipt_started = False
    try:
        if output_root.exists():
            raise FileExistsError("Refusing to overwrite terminal validation")
        contract_path = args.contract.resolve()
        package_validation_path = args.package_validation.resolve()
        prediction_root = args.prediction_root.resolve()
        result_path = args.result.resolve()
        contract = _load_json(contract_path)
        if (
            contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("status") != EXPECTED_CONTRACT_STATUS
        ):
            raise ValueError("Frozen heldout contract identity mismatch")
        _canonical_artifact_path(
            contract, "heldout_terminal_validation_root", output_root
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
        _canonical_artifact_path(
            contract, "heldout_effect_result_root", result_path.parent
        )
        if package_validation_path.name != "validation.json":
            raise ValueError("Independent package validation filename mismatch")
        if args.truth.resolve().name != "heldout_truth.jsonl":
            raise ValueError("Independent truth filename mismatch")
        if result_path.name != "result.json":
            raise ValueError("Independent effect result filename mismatch")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        output_root.mkdir()
        receipt = {
            "schema": VALIDATION_RECEIPT_SCHEMA,
            "status": "STARTED_BEFORE_INDEPENDENT_TRUTH_OPEN",
            "contract_sha256": _sha256(contract_path),
            "package_validation_sha256": _sha256(
                package_validation_path
            ),
            "prediction_completion_sha256": _sha256(
                prediction_root / "completion.json"
            ),
            "effect_result_sha256": _sha256(result_path),
            "truth_opened_before_receipt": False,
            "rerun_authorized": False,
        }
        _atomic_json(output_root / "execution_receipt.json", receipt)
        receipt_started = True
        report = validate(
            contract_path,
            package_validation_path,
            args.truth.resolve(),
            prediction_root,
            result_path,
            output_root,
        )
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
    except Exception as error:
        if (
            receipt_started
            and not (output_root / "validation.json").exists()
            and not (output_root / "failure.json").exists()
        ):
            try:
                _atomic_json(
                    output_root / "failure.json",
                    {
                        "terminal": NOT_EVALUABLE,
                        "stage": "independent_terminal_validation",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "terminal_validation_rerun_authorized": False,
                        "prediction_rerun_authorized": False,
                    },
                )
            except OSError:
                pass
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
