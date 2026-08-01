#!/usr/bin/env python3
"""Join frozen G0-D1 fresh predictions to truth exactly once."""

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
PACKAGE_VALIDATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_package_validation"
)
PACKAGE_READY = "G0_D1_FRESH_PACKAGE_VALIDATED_AND_OPPORTUNITY_ADEQUATE"
PREDICTION_AUTHORIZATION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_prediction_authorization"
)
PREDICTION_AUTHORIZED = (
    "G0_D1_FRESH_PREDICTION_AUTHORIZATION_READY"
)
TRUTH_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_truth"
PREDICTION_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_prediction"
COMPLETION_SCHEMA = (
    "blindassist_hftf_stage_c_g0_d1_fresh_predictions_frozen"
)
PREDICTIONS_READY = "G0_D1_FRESH_PREDICTIONS_FROZEN"
JOIN_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_truth_join_receipt"
RESULT_SCHEMA = "blindassist_hftf_stage_c_g0_d1_fresh_effect_result"
SUPPORTED = (
    "SIGNED_CLEARANCE_CURRENT_BRIDGE_SUPPORTED_"
    "FOR_CAUSAL_TRANSPORT_CONTRACT_ONLY"
)
NOT_SUPPORTED = (
    "SIGNED_CLEARANCE_CURRENT_CROSS_SOURCE_"
    "LEARNABILITY_NOT_SUPPORTED_STOP"
)
NOT_EVALUABLE = "G0_D1_FRESH_EVALUATION_NOT_EVALUABLE_NO_SOURCE_REPLACEMENT"
IMPLEMENTATION_PATH = (
    "scripts/research/hftf/evaluate_stage_c_g0_d1_fresh.py"
)
ARMS = ("DIRECT_RISK_CURRENT", "SIGNED_CLEARANCE_CURRENT")
SEEDS = (17, 29, 43)
HEIGHTS = ("body", "head")
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
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSONL objects: {path}")
    return records


def _load_jsonl_once_with_sha256(
    path: Path,
) -> tuple[list[dict[str, Any]], str]:
    with path.open("rb") as stream:
        payload = stream.read()
    digest = hashlib.sha256(payload).hexdigest()
    records = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
    ]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSONL objects: {path}")
    return records, digest


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _ordered_prediction_key_sha256(
    predictions: list[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        key = {
            "seed": prediction.get("seed"),
            "arm": prediction.get("arm"),
            "checkpoint_sha256": prediction.get("checkpoint_sha256"),
            "sample_id": prediction.get("sample_id"),
        }
        digest.update(_canonical_bytes(key) + b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f"{path.name}.partial-", dir=path.parent
    )
    partial = Path(partial_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    value,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _matrix(value: Any, *, probability: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (2, 6, 6)
        or not np.isfinite(matrix).all()
        or (probability and ((matrix < 0.0) | (matrix > 1.0)).any())
    ):
        raise ValueError("D1 fresh prediction matrix is invalid")
    return matrix


def _truth(
    labels: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known_raw = np.asarray(labels["known_target"], dtype=object)
    risk_raw = np.asarray(labels["risk_target_nullable"], dtype=object)
    clearance_raw = np.asarray(
        labels["clearance_target_m_nullable"], dtype=object
    )
    if any(
        matrix.shape != (2, 6, 6)
        for matrix in (known_raw, risk_raw, clearance_raw)
    ):
        raise ValueError("D1 fresh truth shape mismatch")
    known = np.zeros((2, 6, 6), dtype=np.bool_)
    risk = np.zeros_like(known)
    clearance = np.zeros((2, 6, 6), dtype=np.float64)
    for index in np.ndindex(known.shape):
        known_value = known_raw[index]
        if type(known_value) is not int or known_value not in (0, 1):
            raise ValueError("D1 fresh known truth must be exact binary")
        known[index] = bool(known_value)
        if not known[index]:
            if risk_raw[index] is not None or clearance_raw[index] is not None:
                raise ValueError("D1 fresh UNKNOWN truth must remain null")
            continue
        risk_value = risk_raw[index]
        clearance_value = clearance_raw[index]
        if (
            type(risk_value) is not int
            or risk_value not in (0, 1)
            or isinstance(clearance_value, bool)
            or not isinstance(clearance_value, (int, float))
            or not math.isfinite(float(clearance_value))
            or not -0.5 <= float(clearance_value) <= 1.0
            or bool(risk_value) != bool(float(clearance_value) < 0.0)
        ):
            raise ValueError("D1 fresh risk/clearance truth is invalid")
        risk[index] = bool(risk_value)
        clearance[index] = float(clearance_value)
    return known, risk, clearance


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
    return {
        **counts,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def _median(values: list[float]) -> float:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("D1 frozen median requires three finite values")
    return float(statistics.median(values))


def _validate_exact_sets(
    contract: dict[str, Any],
    package_validation: dict[str, Any],
    predictions: list[dict[str, Any]],
    truth_records: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[int, str, str], dict[str, Any]],
]:
    source_order = package_validation["source_order"]
    source_frame_indices = package_validation["source_frame_indices"]
    truth_by_sample: dict[str, dict[str, Any]] = {}
    expected_truth_order = [
        (source, int(frame_index))
        for source in source_order
        for frame_index in source_frame_indices[source]
    ]
    actual_truth_order = [
        (
            str(record.get("session_id", "")),
            int(record.get("source_frame_index", -1)),
        )
        for record in truth_records
    ]
    if (
        len(source_order) != 3
        or set(source_frame_indices) != set(source_order)
        or any(
            len(source_frame_indices[source]) != 25
            for source in source_order
        )
        or len(truth_records) != 75
        or actual_truth_order != expected_truth_order
    ):
        raise ValueError("D1 fresh source or truth cardinality mismatch")
    for record in truth_records:
        if (
            set(record) != TRUTH_KEYS
            or record.get("schema") != TRUTH_SCHEMA
            or record.get("session_id") not in source_order
            or record.get("sample_id") in truth_by_sample
        ):
            raise ValueError("D1 fresh truth exact schema mismatch")
        _truth(record["labels"])
        truth_by_sample[str(record["sample_id"])] = record
    sample_order = list(truth_by_sample)
    checkpoints = contract["checkpoint_contract"]["checkpoints"]
    expected_keys = [
        (int(item["seed"]), str(item["arm"]), sample_id)
        for item in checkpoints
        for sample_id in sample_order
    ]
    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    if len(checkpoints) != 6 or len(predictions) != 450:
        raise ValueError("D1 fresh prediction Cartesian count mismatch")
    checkpoint_sha = {
        (int(item["seed"]), str(item["arm"])): str(item["sha256"])
        for item in checkpoints
    }
    for index, (expected_key, record) in enumerate(
        zip(expected_keys, predictions, strict=True)
    ):
        key = (
            int(record.get("seed", -1)),
            str(record.get("arm", "")),
            str(record.get("sample_id", "")),
        )
        truth_record = truth_by_sample.get(key[2])
        if (
            set(record) != PREDICTION_KEYS
            or record.get("schema") != PREDICTION_SCHEMA
            or record.get("prediction_index") != index
            or key != expected_key
            or key in by_key
            or truth_record is None
            or record.get("checkpoint_sha256")
            != checkpoint_sha[(key[0], key[1])]
            or record.get("session_id") != truth_record["session_id"]
            or record.get("source_frame_index")
            != truth_record["source_frame_index"]
            or record.get("manifest_id") != truth_record["manifest_id"]
        ):
            raise ValueError("D1 fresh prediction exact key mismatch")
        raw = _matrix(record["raw_task_output"])
        probability = _matrix(
            record["risk_probability"], probability=True
        )
        _matrix(record["known_probability"], probability=True)
        expected_probability = (
            1.0 / (1.0 + np.exp(-raw))
            if key[1] == "DIRECT_RISK_CURRENT"
            else (raw < 0.0).astype(np.float64)
        )
        if not np.allclose(
            probability,
            expected_probability,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("D1 fresh frozen risk derivation mismatch")
        by_key[key] = record
    return truth_by_sample, by_key


def _run_metrics(
    checkpoint: dict[str, Any],
    source_order: list[str],
    truth_by_sample: dict[str, dict[str, Any]],
    prediction_by_key: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    seed, arm = int(checkpoint["seed"]), str(checkpoint["arm"])
    micro = _empty_counts()
    by_height = {height: _empty_counts() for height in HEIGHTS}
    by_source = {source: _empty_counts() for source in source_order}
    known_correct = 0
    known_total = 0
    clearance_groups: dict[
        tuple[str, str], dict[str, dict[str, float | int]]
    ] = {
        (source, height): {
            group: {"sum": 0.0, "count": 0}
            for group in ("overall", "risk", "safe", "near")
        }
        for source in source_order
        for height in HEIGHTS
    }
    out_of_range = 0
    clearance_known = 0
    for sample_id, truth_record in truth_by_sample.items():
        prediction = prediction_by_key[(seed, arm, sample_id)]
        probability = _matrix(
            prediction["risk_probability"], probability=True
        )
        known_probability = _matrix(
            prediction["known_probability"], probability=True
        )
        raw = _matrix(prediction["raw_task_output"])
        known, risk, clearance = _truth(truth_record["labels"])
        _add_counts(micro, _counts(probability, risk, known))
        for height_index, height in enumerate(HEIGHTS):
            _add_counts(
                by_height[height],
                _counts(
                    probability[height_index],
                    risk[height_index],
                    known[height_index],
                ),
            )
        _add_counts(
            by_source[str(truth_record["session_id"])],
            _counts(probability, risk, known),
        )
        known_correct += int(((known_probability >= 0.5) == known).sum())
        known_total += known.size
        if arm == "SIGNED_CLEARANCE_CURRENT":
            absolute_error = np.abs(raw - clearance)
            out_of_range += int(
                (((raw < -0.5) | (raw > 1.0)) & known).sum()
            )
            clearance_known += int(known.sum())
            for height_index, height in enumerate(HEIGHTS):
                scope = clearance_groups[
                    (str(truth_record["session_id"]), height)
                ]
                masks = {
                    "overall": known[height_index],
                    "risk": known[height_index] & risk[height_index],
                    "safe": known[height_index] & ~risk[height_index],
                    "near": known[height_index]
                    & (np.abs(clearance[height_index]) <= 0.2),
                }
                for group, mask in masks.items():
                    scope[group]["sum"] = float(scope[group]["sum"]) + float(
                        absolute_error[height_index][mask].sum()
                    )
                    scope[group]["count"] = int(
                        scope[group]["count"]
                    ) + int(mask.sum())
    result: dict[str, Any] = {
        "seed": seed,
        "arm": arm,
        "checkpoint_sha256": checkpoint["sha256"],
        "risk_micro": _metrics(micro),
        "risk_by_height": {
            height: _metrics(counts)
            for height, counts in by_height.items()
        },
        "risk_by_source": {
            source: _metrics(counts) for source, counts in by_source.items()
        },
        "known_accuracy_diagnostic": known_correct / known_total,
    }
    if arm == "SIGNED_CLEARANCE_CURRENT":
        source_height_mae: dict[str, dict[str, float]] = {}
        for (source, height), groups in clearance_groups.items():
            key = f"{source}:{height}"
            source_height_mae[key] = {}
            for group, values in groups.items():
                if int(values["count"]) <= 0:
                    raise ValueError("D1 fresh clearance group is empty")
                source_height_mae[key][group] = float(values["sum"]) / int(
                    values["count"]
                )
        result["clearance_mae_by_source_height_m"] = source_height_mae
        result["clearance_source_height_macro_mae_m"] = {
            group: float(
                np.mean(
                    [
                        values[group]
                        for values in source_height_mae.values()
                    ]
                )
            )
            for group in ("overall", "risk", "safe", "near")
        }
        result["raw_prediction_out_of_target_range_fraction"] = (
            out_of_range / clearance_known
        )
    return result


def _decision(
    runs: list[dict[str, Any]],
    source_order: list[str],
    unknown_to_safe_violation_count: int,
) -> tuple[dict[str, Any], dict[str, bool], str]:
    by = {(run["seed"], run["arm"]): run for run in runs}
    deltas = [
        by[(seed, ARMS[1])]["risk_micro"]["f1"]
        - by[(seed, ARMS[0])]["risk_micro"]["f1"]
        for seed in SEEDS
    ]
    recall_deltas = [
        by[(seed, ARMS[1])]["risk_micro"]["recall"]
        - by[(seed, ARMS[0])]["risk_micro"]["recall"]
        for seed in SEEDS
    ]
    fpr_deltas = [
        by[(seed, ARMS[1])]["risk_micro"]["false_positive_rate"]
        - by[(seed, ARMS[0])]["risk_micro"]["false_positive_rate"]
        for seed in SEEDS
    ]
    height_deltas = {
        height: [
            by[(seed, ARMS[1])]["risk_by_height"][height]["f1"]
            - by[(seed, ARMS[0])]["risk_by_height"][height]["f1"]
            for seed in SEEDS
        ]
        for height in HEIGHTS
    }
    source_deltas = {
        source: [
            by[(seed, ARMS[1])]["risk_by_source"][source]["f1"]
            - by[(seed, ARMS[0])]["risk_by_source"][source]["f1"]
            for seed in SEEDS
        ]
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
            by[(seed, ARMS[1])]["clearance_source_height_macro_mae_m"][
                group
            ]
            for seed in SEEDS
        ]
        for group in ("overall", "risk", "safe", "near")
    }
    aggregates = {
        "clearance_micro_f1_by_seed": dict(
            zip(map(str, SEEDS), clearance_f1)
        ),
        "clearance_median_seed_micro_f1": _median(clearance_f1),
        "micro_f1_delta_by_seed": dict(zip(map(str, SEEDS), deltas)),
        "median_micro_f1_delta": _median(deltas),
        "median_recall_delta": _median(recall_deltas),
        "median_false_positive_rate_delta": _median(fpr_deltas),
        "median_f1_delta_by_height": {
            height: _median(values)
            for height, values in height_deltas.items()
        },
        "median_f1_delta_by_source": {
            source: _median(values)
            for source, values in source_deltas.items()
        },
        "median_clearance_f1_by_source": {
            source: _median(values)
            for source, values in source_clearance_f1.items()
        },
        "clearance_source_height_macro_mae_by_seed_m": {
            group: dict(zip(map(str, SEEDS), values))
            for group, values in clearance_mae.items()
        },
        "max_seed_clearance_source_height_macro_mae_m": {
            group: max(values)
            for group, values in clearance_mae.items()
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
            value > 0.0 for value in deltas
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
            min(
                aggregates["median_clearance_f1_by_source"].values()
            )
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


def _canonical_paths(
    contract_path: Path,
    package_validation_path: Path,
    truth_path: Path,
    completion_path: Path,
    predictions_path: Path,
    output_root: Path,
) -> None:
    repository = _repository_root()
    expected = {
        "contract": (
            repository
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_"
            "EXECUTION_CONTRACT_D1_2026-08-01.json"
        ).resolve(),
        "package_validation": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-validation-20260801/"
            "validation.json"
        ).resolve(),
        "truth": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801/truth_labels.jsonl"
        ).resolve(),
        "completion": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-predictions-20260801/"
            "completion.json"
        ).resolve(),
        "predictions": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-predictions-20260801/"
            "predictions.jsonl"
        ).resolve(),
        "output": (
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-effect-20260801"
        ).resolve(),
    }
    actual = {
        "contract": contract_path.resolve(),
        "package_validation": package_validation_path.resolve(),
        "truth": truth_path.resolve(),
        "completion": completion_path.resolve(),
        "predictions": predictions_path.resolve(),
        "output": output_root.resolve(),
    }
    for key in expected:
        if actual[key] != expected[key]:
            raise ValueError(f"D1 fresh evaluator noncanonical {key} path")


def evaluate(
    contract_path: Path,
    package_validation_path: Path,
    truth_path: Path,
    completion_path: Path,
    predictions_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    _canonical_paths(
        contract_path,
        package_validation_path,
        truth_path,
        completion_path,
        predictions_path,
        output_root,
    )
    if output_root.exists():
        raise FileExistsError("D1 fresh truth join is already consumed")
    contract = _load_json(contract_path)
    package_validation = _load_json(package_validation_path)
    completion = _load_json(completion_path)
    implementation = contract.get("implementations", {}).get(
        "fresh_evaluator", {}
    )
    package_validator = contract.get("implementations", {}).get(
        "fresh_package_validator", {}
    )
    predictor = contract.get("implementations", {}).get(
        "fresh_predictor", {}
    )
    package_authorization = package_validation.get("authorization", {})
    checkpoints = contract.get("checkpoint_contract", {}).get(
        "checkpoints", []
    )
    expected_checkpoint_receipts = [
        {
            "seed": item.get("seed"),
            "arm": item.get("arm"),
            "checkpoint_sha256": item.get("sha256"),
        }
        for item in checkpoints
    ]
    expected_truth_sha256 = package_validation.get(
        "truth_labels_sha256"
    )
    prediction_authorization_path = (
        package_validation_path.parent / "prediction_authorization.json"
    )
    prediction_authorization = _load_json(
        prediction_authorization_path
    )
    authorization_decision = prediction_authorization.get(
        "authorization", {}
    )
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != CONTRACT_STATUS
        or implementation.get("path") != IMPLEMENTATION_PATH
        or implementation.get("sha256")
        != _sha256(Path(__file__).resolve())
        or implementation.get("execution_authorized") is not True
        or package_validation.get("schema")
        != PACKAGE_VALIDATION_SCHEMA
        or package_validation.get("terminal") != PACKAGE_READY
        or package_validation.get("contract_sha256")
        != _sha256(contract_path)
        or package_validation.get("package_validator_sha256")
        != package_validator.get("sha256")
        or package_validation.get("source_order")
        != contract.get("fresh_source_contract", {}).get("source_order")
        or package_validation.get("truth_labels_path")
        != str(truth_path.resolve())
        or not isinstance(expected_truth_sha256, str)
        or len(expected_truth_sha256) != 64
        or package_validation.get("truth_label_count") != 75
        or package_validation.get("unknown_to_safe_violation_count") != 0
        or package_authorization.get("fresh_prediction_authorized")
        is not True
        or package_authorization.get(
            "truth_join_authorized_before_predictions_frozen"
        )
        is not False
        or prediction_authorization.get("schema")
        != PREDICTION_AUTHORIZATION_SCHEMA
        or prediction_authorization.get("terminal")
        != PREDICTION_AUTHORIZED
        or prediction_authorization.get("contract_sha256")
        != _sha256(contract_path)
        or prediction_authorization.get("package_validator_sha256")
        != package_validator.get("sha256")
        or prediction_authorization.get("prediction_inputs_path")
        != package_validation.get("prediction_inputs_path")
        or prediction_authorization.get("prediction_inputs_sha256")
        != package_validation.get("prediction_inputs_sha256")
        or prediction_authorization.get("prediction_input_count") != 75
        or prediction_authorization.get("source_order")
        != package_validation.get("source_order")
        or prediction_authorization.get("source_frame_indices")
        != package_validation.get("source_frame_indices")
        or authorization_decision.get("fresh_prediction_authorized")
        is not True
        or authorization_decision.get(
            "truth_join_authorized_before_predictions_frozen"
        )
        is not False
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("terminal") != PREDICTIONS_READY
        or completion.get("contract_sha256") != _sha256(contract_path)
        or completion.get("prediction_authorization_sha256")
        != _sha256(prediction_authorization_path)
        or completion.get("prediction_inputs_path")
        != package_validation.get("prediction_inputs_path")
        or completion.get("prediction_inputs_sha256")
        != package_validation.get("prediction_inputs_sha256")
        or completion.get("predictor_sha256") != predictor.get("sha256")
        or completion.get("predictions_sha256") != _sha256(predictions_path)
        or completion.get("checkpoint_receipts")
        != expected_checkpoint_receipts
        or completion.get("all_outputs_finite") is not True
        or completion.get("truth_files_opened") is not False
        or completion.get("teacher_files_opened") is not False
        or completion.get("prediction_count") != 450
        or completion.get("truth_join_authorized") is not True
        or completion.get("second_prediction_run_authorized") is not False
    ):
        raise ValueError("D1 fresh evaluation parent receipt mismatch")
    output_root.mkdir(parents=True)
    join = {
        "schema": JOIN_SCHEMA,
        "status": "FROZEN_PREDICTIONS_GLOBALLY_CONSUMED_BEFORE_TRUTH_OPEN",
        "execution_contract_sha256": _sha256(contract_path),
        "package_validation_sha256": _sha256(package_validation_path),
        "completion_sha256": _sha256(completion_path),
        "predictions_sha256": _sha256(predictions_path),
        "expected_truth_sha256": expected_truth_sha256,
        "truth_join_exactly_once": True,
        "second_model_forward_authorized": False,
        "source_replacement_authorized": False,
    }
    try:
        _atomic_json(output_root / "truth_join_receipt.json", join)
    except BaseException:
        if output_root.exists() and not any(output_root.iterdir()):
            output_root.rmdir()
        raise
    try:
        predictions = _load_jsonl(predictions_path)
        if completion.get("ordered_prediction_key_sha256") != (
            _ordered_prediction_key_sha256(predictions)
        ):
            raise ValueError("D1 fresh ordered prediction key hash drifted")
        truth_records, actual_truth_sha256 = (
            _load_jsonl_once_with_sha256(truth_path)
        )
        if actual_truth_sha256 != expected_truth_sha256:
            raise ValueError("D1 fresh truth hash mismatch after join receipt")
        truth_by_sample, prediction_by_key = _validate_exact_sets(
            contract,
            package_validation,
            predictions,
            truth_records,
        )
        runs = [
            _run_metrics(
                checkpoint,
                package_validation["source_order"],
                truth_by_sample,
                prediction_by_key,
            )
            for checkpoint in contract["checkpoint_contract"][
                "checkpoints"
            ]
        ]
        aggregates, gates, terminal = _decision(
            runs,
            package_validation["source_order"],
            int(package_validation["unknown_to_safe_violation_count"]),
        )
        result = {
            "schema": RESULT_SCHEMA,
            "terminal": terminal,
            "workflow_profile": "FORMAL_ONE_SHOT_FRESH_EVALUATION",
            "claim_ceiling": (
                "FRESH_CURRENT_SYNTHETIC_PROXY_LEARNABILITY_ONLY"
            ),
            "parents": {
                "execution_contract_sha256": _sha256(contract_path),
                "package_validation_sha256": _sha256(
                    package_validation_path
                ),
                "completion_sha256": _sha256(completion_path),
                "predictions_sha256": _sha256(predictions_path),
                "truth_sha256": actual_truth_sha256,
                "truth_join_receipt_sha256": _sha256(
                    output_root / "truth_join_receipt.json"
                ),
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
        _atomic_json(output_root / "effect_result.json", result)
        return result
    except BaseException as error:
        failure = {
            "schema": RESULT_SCHEMA,
            "terminal": NOT_EVALUABLE,
            "error": str(error),
            "truth_join_consumed": True,
            "rerun_authorized": False,
            "source_replacement_authorized": False,
        }
        _atomic_json(output_root / "failure.json", failure)
        raise


def main() -> int:
    repository = _repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            repository
            / "docs/research/hftf/"
            "HFTF_STAGE_C_CURRENT_CLEARANCE_FRESH_"
            "EXECUTION_CONTRACT_D1_2026-08-01.json"
        ),
    )
    parser.add_argument(
        "--package-validation",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-validation-20260801/"
            "validation.json"
        ),
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-package-20260801/truth_labels.jsonl"
        ),
    )
    parser.add_argument(
        "--completion",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-predictions-20260801/completion.json"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-predictions-20260801/"
            "predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repository
            / "artifacts.local/evidence/hftf/"
            "stage-c-g0-d1-fresh-effect-20260801"
        ),
    )
    arguments = parser.parse_args()
    try:
        result = evaluate(
            arguments.contract.resolve(),
            arguments.package_validation.resolve(),
            arguments.truth.resolve(),
            arguments.completion.resolve(),
            arguments.predictions.resolve(),
            arguments.output_root.resolve(),
        )
        print(
            json.dumps(
                {
                    "terminal": result["terminal"],
                    "all_gates_pass": result["all_gates_pass"],
                    "result_sha256": _sha256(
                        arguments.output_root / "effect_result.json"
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
    ) as error:
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
