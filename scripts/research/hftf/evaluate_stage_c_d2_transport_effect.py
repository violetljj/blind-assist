#!/usr/bin/env python3
"""Open D2 future truth once and evaluate the frozen transport effect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from preprocess_stage_c_d2_future_blind import (
    ATTEMPT_SCHEMA,
    ATTEMPT_STATUS,
    SCHEMA as PREDICTION_SCHEMA,
    _frame_map,
    _load_current,
    _load_pose,
    _write_json_durable,
)
from stage_c_d2_mechanics_common import (
    ANCHORS,
    EFFECT_FAILURE_RELATIVE_PATH,
    EFFECT_RESULT_RELATIVE_PATH,
    EVALUATOR_PRETRUTH_FAILURE_TERMINAL,
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_HORIZON_RECORD_COUNT,
    HEIGHTS,
    HORIZONS,
    NOT_EVALUABLE,
    PREPROCESSOR_COMPLETION_SCHEMA,
    PREPROCESSOR_TERMINAL,
    PREDICTION_RELATIVE_ROOT,
    RESULT_SCHEMA,
    STOP,
    SUPPORTED,
    TRUTH_JOIN_RECEIPT_RELATIVE_PATH,
    TRUTH_JOIN_INTERRUPTED_TERMINAL,
    arrays_from_arm,
    compute_field,
    compute_known,
    compute_points,
    field_parameters,
    load_context,
    load_json,
    nullable_field,
    resolve,
    sha256,
)


IMPLEMENTATION_KEY = "truth_effect_evaluator"
PERSISTENCE = "CURRENT_FIELD_PERSISTENCE"
ADVECTED = "HISTORY_CAUSAL_ADVECTED_CURRENT_FIELD"
ARM_NAMES = (PERSISTENCE, ADVECTED)
CELL_DENOMINATOR = len(ANCHORS) * 6 * 6
TOLERANCE = 1e-12
FAILURE_SCHEMA = (
    "blindassist_hftf_stage_c_d2_truth_effect_evaluation_failure"
)
PRETRUTH_FAILURE_TERMINAL = (
    EVALUATOR_PRETRUTH_FAILURE_TERMINAL
)
TRUTH_INTERRUPTED_TERMINAL = (
    TRUTH_JOIN_INTERRUPTED_TERMINAL
)


def _basis(value: dict[str, Any]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    basis = tuple(
        np.asarray(value[key], dtype=np.float64)
        for key in ("origin_m", "forward", "right", "up")
    )
    if any(item.shape != (3,) for item in basis):
        raise ValueError("D2 predicted truth-frame basis shape mismatch")
    origin, forward, right, up = basis
    if (
        not np.isfinite(np.concatenate(basis)).all()
        or abs(float(forward @ up)) > 1e-8
        or abs(float(right @ up)) > 1e-8
        or abs(float(forward @ right)) > 1e-8
        or abs(float(np.linalg.norm(forward)) - 1.0) > 1e-8
        or abs(float(np.linalg.norm(right)) - 1.0) > 1e-8
        or abs(float(np.linalg.norm(up)) - 1.0) > 1e-8
        or float(np.cross(forward, up) @ right) < 1.0 - 1e-8
    ):
        raise ValueError("D2 predicted truth-frame basis is not orthonormal")
    return origin, forward, right, up


def _f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    true_positive = int(np.count_nonzero(truth & prediction))
    false_positive = int(np.count_nonzero(~truth & prediction))
    false_negative = int(np.count_nonzero(truth & ~prediction))
    denominator = 2 * true_positive + false_positive + false_negative
    return (
        0.0
        if denominator == 0
        else float(2 * true_positive / denominator)
    )


def _empty_stratum() -> dict[str, Any]:
    return {
        "common_count": 0,
        "risk_count": 0,
        "safe_count": 0,
        "absolute_error": {arm: [] for arm in ARM_NAMES},
    }


def summarize_observations(
    source_ids: list[str],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    strata: dict[tuple[str, str, float], dict[str, Any]] = {
        (source_id, height, horizon): _empty_stratum()
        for source_id in source_ids
        for height in HEIGHTS
        for horizon in HORIZONS
    }
    parent_truth: dict[str, list[np.ndarray]] = {
        source_id: [] for source_id in source_ids
    }
    parent_prediction: dict[str, dict[str, list[np.ndarray]]] = {
        source_id: {arm: [] for arm in ARM_NAMES}
        for source_id in source_ids
    }
    unknown_violations = 0
    for observation in observations:
        source_id = str(observation["session_id"])
        horizon = float(observation["horizon_s"])
        truth_known = observation["truth_known"]
        truth_clearance = observation["truth_clearance"]
        arm_arrays = observation["arms"]
        unknown_violations += int(
            observation.get("unknown_to_safe_violations", 0)
        )
        for height_index, height in enumerate(HEIGHTS):
            common = truth_known[height_index].copy()
            for arm in ARM_NAMES:
                common &= arm_arrays[arm][0][height_index]
            target = truth_clearance[height_index][common]
            stratum = strata[(source_id, height, horizon)]
            stratum["common_count"] += int(common.sum())
            stratum["risk_count"] += int(np.count_nonzero(target < 0.0))
            stratum["safe_count"] += int(np.count_nonzero(target >= 0.0))
            parent_truth[source_id].append(target < 0.0)
            for arm in ARM_NAMES:
                predicted = arm_arrays[arm][1][height_index][common]
                stratum["absolute_error"][arm].extend(
                    np.abs(predicted - target).tolist()
                )
                parent_prediction[source_id][arm].append(predicted < 0.0)

    opportunity_rows: list[dict[str, Any]] = []
    opportunity_adequate = unknown_violations == 0
    stratum_mae: dict[tuple[str, str, float], dict[str, float]] = {}
    for key, stratum in strata.items():
        source_id, height, horizon = key
        coverage = stratum["common_count"] / CELL_DENOMINATOR
        passed = (
            coverage + TOLERANCE >= 0.1
            and stratum["risk_count"] >= 5
            and stratum["safe_count"] >= 20
        )
        opportunity_adequate &= passed
        opportunity_rows.append(
            {
                "session_id": source_id,
                "height": height,
                "horizon_s": horizon,
                "common_known_count": stratum["common_count"],
                "denominator": CELL_DENOMINATOR,
                "common_known_coverage": coverage,
                "known_risk_count": stratum["risk_count"],
                "known_safe_count": stratum["safe_count"],
                "passed": passed,
            }
        )
        if stratum["common_count"]:
            stratum_mae[key] = {
                arm: float(np.mean(stratum["absolute_error"][arm]))
                for arm in ARM_NAMES
            }

    base: dict[str, Any] = {
        "opportunity_adequate": bool(opportunity_adequate),
        "opportunity_strata": opportunity_rows,
        "unknown_to_safe_violations": unknown_violations,
    }
    if not opportunity_adequate:
        return base

    source_mae = {
        source_id: {
            arm: float(
                np.mean(
                    [
                        stratum_mae[(source_id, height, horizon)][arm]
                        for height in HEIGHTS
                        for horizon in HORIZONS
                    ]
                )
            )
            for arm in ARM_NAMES
        }
        for source_id in source_ids
    }
    macro = {
        arm: float(
            np.mean([source_mae[source_id][arm] for source_id in source_ids])
        )
        for arm in ARM_NAMES
    }
    height_macro = {
        height: {
            arm: float(
                np.mean(
                    [
                        stratum_mae[(source_id, height, horizon)][arm]
                        for source_id in source_ids
                        for horizon in HORIZONS
                    ]
                )
            )
            for arm in ARM_NAMES
        }
        for height in HEIGHTS
    }
    horizon_macro = {
        str(horizon): {
            arm: float(
                np.mean(
                    [
                        stratum_mae[(source_id, height, horizon)][arm]
                        for source_id in source_ids
                        for height in HEIGHTS
                    ]
                )
            )
            for arm in ARM_NAMES
        }
        for horizon in HORIZONS
    }
    parent_f1: dict[str, dict[str, float]] = {}
    for source_id in source_ids:
        truth = np.concatenate(parent_truth[source_id])
        parent_f1[source_id] = {
            arm: _f1(
                truth,
                np.concatenate(parent_prediction[source_id][arm]),
            )
            for arm in ARM_NAMES
        }
    macro_f1 = {
        arm: float(
            np.mean([parent_f1[source_id][arm] for source_id in source_ids])
        )
        for arm in ARM_NAMES
    }
    absolute_reduction = macro[PERSISTENCE] - macro[ADVECTED]
    relative_reduction = (
        0.0
        if macro[PERSISTENCE] <= TOLERANCE
        else absolute_reduction / macro[PERSISTENCE]
    )
    gates = {
        "relative_mae_reduction_at_least_0_10": (
            relative_reduction + TOLERANCE >= 0.1
        ),
        "absolute_mae_reduction_m_at_least_0_03": (
            absolute_reduction + TOLERANCE >= 0.03
        ),
        "each_height_noninferior": all(
            height_macro[height][PERSISTENCE]
            - height_macro[height][ADVECTED]
            >= -TOLERANCE
            for height in HEIGHTS
        ),
        "each_horizon_noninferior": all(
            horizon_macro[str(horizon)][PERSISTENCE]
            - horizon_macro[str(horizon)][ADVECTED]
            >= -TOLERANCE
            for horizon in HORIZONS
        ),
        "at_least_5_of_6_parents_improve": sum(
            source_mae[source_id][PERSISTENCE]
            - source_mae[source_id][ADVECTED]
            > TOLERANCE
            for source_id in source_ids
        )
        >= 5,
        "parent_macro_risk_f1_delta_at_least_0_03": (
            macro_f1[ADVECTED] - macro_f1[PERSISTENCE] + TOLERANCE
            >= 0.03
        ),
        "unknown_to_safe_violations_zero": unknown_violations == 0,
    }
    base.update(
        {
            "stratum_mae_m": [
                {
                    "session_id": source_id,
                    "height": height,
                    "horizon_s": horizon,
                    **stratum_mae[(source_id, height, horizon)],
                }
                for source_id in source_ids
                for height in HEIGHTS
                for horizon in HORIZONS
            ],
            "source_mae_m": source_mae,
            "six_source_macro_mae_m": macro,
            "height_macro_mae_m": height_macro,
            "horizon_macro_mae_m": horizon_macro,
            "parent_risk_f1": parent_f1,
            "parent_macro_risk_f1": macro_f1,
            "absolute_mae_reduction_m": absolute_reduction,
            "relative_mae_reduction": relative_reduction,
            "effect_gates": gates,
            "all_effect_gates_passed": all(gates.values()),
        }
    )
    return base


def result_authorization(terminal: str) -> dict[str, bool]:
    return {
        "freeze_rgb_student_contract_authorized": terminal == SUPPORTED,
        "rgb_student_training_authorized": False,
        "rgb_student_execution_authorized": False,
        "reserved_official_test_open_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
        "android_changed": False,
        "production_authorized": False,
        "safety_claim_authorized": False,
    }


def _load_completion(
    contract_path: Path,
    prediction_root: Path,
    source_ids: list[str],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    attempt_path = prediction_root / "attempt.json"
    attempt = load_json(attempt_path)
    completion_path = prediction_root / "completion.json"
    completion = load_json(completion_path)
    if (
        attempt.get("schema") != ATTEMPT_SCHEMA
        or attempt.get("status") != ATTEMPT_STATUS
        or attempt.get("contract_sha256") != sha256(contract_path)
        or attempt.get("pose_or_media_opened_before_attempt") is not False
        or attempt.get("future_depth_mask_or_pose_opened") is not False
        or attempt.get("second_preprocessor_run_authorized") is not False
        or attempt.get("truth_join_authorized_before_completion")
        is not False
        or completion.get("schema") != PREPROCESSOR_COMPLETION_SCHEMA
        or completion.get("terminal") != PREPROCESSOR_TERMINAL
        or completion.get("contract_sha256") != sha256(contract_path)
        or completion.get("preprocessor_attempt_sha256")
        != sha256(attempt_path)
        or completion.get("prediction_record_count")
        != EXPECTED_ANCHOR_COUNT
        or completion.get("anchor_horizon_record_count")
        != EXPECTED_HORIZON_RECORD_COUNT
        or completion.get("all_records_durable_before_truth_join") is not True
        or completion.get("future_depth_mask_or_pose_opened") is not False
        or completion.get("truth_join_authorized") is not True
    ):
        raise ValueError("D2 preprocessor completion identity mismatch")
    expected_keys = [
        (source_id, anchor)
        for source_id in source_ids
        for anchor in ANCHORS
    ]
    receipts = completion["records"]
    if [
        (
            str(receipt["session_id"]),
            int(receipt["anchor_normalized_index"]),
        )
        for receipt in receipts
    ] != expected_keys:
        raise ValueError("D2 frozen prediction order differs from contract")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for receipt, key in zip(receipts, expected_keys):
        path = Path(str(receipt["path"])).resolve()
        expected_path = (
            prediction_root / key[0] / f"anchor-{key[1]}.json"
        ).resolve()
        expected_points_path = (
            prediction_root / key[0] / f"anchor-{key[1]}.points.npy"
        ).resolve()
        if path != expected_path:
            raise ValueError("D2 prediction record escaped canonical root")
        if sha256(path) != str(receipt["sha256"]):
            raise ValueError("D2 frozen prediction record hash mismatch")
        record = load_json(path)
        points_path = Path(str(record["points"]["path"])).resolve()
        if (
            record.get("schema") != PREDICTION_SCHEMA
            or str(record.get("session_id")) != key[0]
            or int(record.get("anchor_normalized_index", -1)) != key[1]
            or record.get("future_depth_mask_or_pose_read") is not False
            or int(record.get("unknown_to_safe_violations", -1)) != 0
            or points_path != expected_points_path
            or sha256(points_path) != str(receipt["points_sha256"])
        ):
            raise ValueError("D2 frozen prediction record identity mismatch")
        records[key] = record
    if len(records) != EXPECTED_ANCHOR_COUNT:
        raise ValueError("D2 frozen predictions are not unique 42 anchors")
    return completion, records


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    context = load_context(contract_path, IMPLEMENTATION_KEY, Path(__file__))
    authorization = context["contract"]["authorization"]
    if (
        authorization.get("truth_effect_execution_after_completion_authorized")
        is not True
        or authorization.get("second_truth_join_authorized") is not False
    ):
        raise ValueError("D2 truth/effect authorization mismatch")
    prior_failure_path = resolve(
        Path(__file__).resolve().parents[3],
        contract_path.parent,
        context["contract"]["canonical_artifacts"][
            "truth_effect_failure"
        ],
    )
    if prior_failure_path.exists():
        prior_failure = load_json(prior_failure_path)
        terminal = prior_failure.get("terminal")
        if (
            prior_failure.get("schema") != FAILURE_SCHEMA
            or terminal
            not in {
                PRETRUTH_FAILURE_TERMINAL,
                TRUTH_INTERRUPTED_TERMINAL,
            }
        ):
            raise FileExistsError(
                "D2 truth/effect failure artifact exists and is invalid"
            )
        raise FileExistsError(
            f"D2 truth/effect failure already sealed: {terminal}"
        )
    expected_output = resolve(
        Path(__file__).resolve().parents[3],
        contract_path.parent,
        context["contract"]["canonical_artifacts"]["effect_result"],
    )
    if output_path.resolve() != expected_output or output_path.exists():
        raise FileExistsError("D2 effect result is noncanonical or exists")
    truth_join_receipt = resolve(
        Path(__file__).resolve().parents[3],
        contract_path.parent,
        context["contract"]["canonical_artifacts"][
            "truth_join_once_receipt"
        ],
    )
    if truth_join_receipt.exists():
        raise FileExistsError("D2 future truth was already opened once")
    if truth_join_receipt == output_path.resolve():
        raise ValueError("D2 truth receipt and result paths must differ")
    prediction_root = resolve(
        Path(__file__).resolve().parents[3],
        contract_path.parent,
        context["contract"]["canonical_artifacts"][
            "future_blind_prediction_root"
        ],
    )
    source_ids = [str(source["session_id"]) for source in context["sources"]]
    completion, predictions = _load_completion(
        contract_path,
        prediction_root,
        source_ids,
    )
    parameters = field_parameters(context["g0"], context["mechanics"])
    truth_join_receipt.parent.mkdir(parents=True, exist_ok=True)
    _write_json_durable(
        truth_join_receipt,
        {
            "schema": "blindassist_hftf_stage_c_d2_truth_join_once_receipt",
            "status": "D2_TRUTH_JOIN_STARTED_NO_SECOND_JOIN_AUTHORIZED",
            "contract_sha256": sha256(contract_path),
            "preprocessor_completion_sha256": sha256(
                prediction_root / "completion.json"
            ),
            "all_predictions_durable_before_receipt": True,
            "future_truth_opened_before_receipt": False,
            "second_truth_join_authorized": False,
        },
    )
    repo_root = Path(__file__).resolve().parents[3]
    index_root = context["source_index_path"].parent
    observations: list[dict[str, Any]] = []
    truth_receipts: list[dict[str, Any]] = []
    for source in context["sources"]:
        source_id = str(source["session_id"])
        frames = _frame_map(source)
        camera = source["camera"]
        for anchor in ANCHORS:
            prediction = predictions[(source_id, anchor)]
            horizon_records = {
                float(item["horizon_s"]): item
                for item in prediction["horizons"]
            }
            if set(horizon_records) != set(HORIZONS):
                raise ValueError("D2 anchor horizon records differ from freeze")
            for horizon in HORIZONS:
                future_index = anchor + (2 if horizon == 0.4 else 4)
                future_frame = frames[future_index]
                future_binding = _load_pose(
                    repo_root,
                    index_root,
                    future_frame,
                )
                depth, semantic = _load_current(
                    repo_root,
                    index_root,
                    future_frame,
                    camera,
                )
                row = {
                    "id": future_frame["manifest_id"],
                    "width": int(camera["image_width"]),
                    "height": int(camera["image_height"]),
                }
                basis = _basis(horizon_records[horizon]["predicted_basis"])
                truth_points = compute_points(
                    depth,
                    semantic,
                    row,
                    future_binding,
                    camera,
                    parameters,
                )
                truth_clearance = compute_field(
                    truth_points,
                    basis,
                    parameters,
                )
                truth_counts, truth_known = compute_known(
                    depth,
                    semantic,
                    row,
                    future_binding,
                    camera,
                    basis,
                    parameters,
                )
                arms = {
                    arm: arrays_from_arm(
                        horizon_records[horizon]["arms"][arm]
                    )
                    for arm in ARM_NAMES
                }
                observations.append(
                    {
                        "session_id": source_id,
                        "anchor_normalized_index": anchor,
                        "horizon_s": horizon,
                        "truth_known": truth_known,
                        "truth_clearance": truth_clearance,
                        "arms": arms,
                        "unknown_to_safe_violations": 0,
                    }
                )
                truth_receipts.append(
                    {
                        "session_id": source_id,
                        "anchor_normalized_index": anchor,
                        "horizon_s": horizon,
                        "future_normalized_index": future_index,
                        "truth_probe_pass_counts": truth_counts.tolist(),
                        "truth_known": truth_known.tolist(),
                        "truth_clearance_m": nullable_field(
                            truth_known,
                            truth_clearance,
                        ),
                    }
                )
                del depth, semantic, truth_points
    if len(observations) != EXPECTED_HORIZON_RECORD_COUNT:
        raise ValueError("D2 truth join did not produce exactly 84 records")
    metrics = summarize_observations(source_ids, observations)
    terminal = (
        NOT_EVALUABLE
        if not metrics["opportunity_adequate"]
        else (
            SUPPORTED
            if metrics["all_effect_gates_passed"]
            else STOP
        )
    )
    result = {
        "schema": RESULT_SCHEMA,
        "terminal": terminal,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "contract_sha256": sha256(contract_path),
        "preprocessor_completion_sha256": sha256(
            prediction_root / "completion.json"
        ),
        "truth_join_once_receipt_sha256": sha256(truth_join_receipt),
        "truth_join_count": len(truth_receipts),
        "truth_join_performed_once_after_all_predictions_durable": True,
        "source_replacement_authorized": False,
        "same_cohort_retuning_authorized": False,
        "authorization": result_authorization(terminal),
        "metrics": metrics,
        "truth_records": truth_receipts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_durable(output_path, result)
    return result


def seal_failure_if_attempted(
    output_path: Path,
    error: BaseException,
) -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    expected_output = (repo_root / EFFECT_RESULT_RELATIVE_PATH).resolve()
    if output_path.resolve() != expected_output or expected_output.exists():
        return None
    failure_path = (repo_root / EFFECT_FAILURE_RELATIVE_PATH).resolve()
    if failure_path.exists():
        failure = load_json(failure_path)
        terminal = failure.get("terminal")
        return (
            str(terminal)
            if terminal
            in {PRETRUTH_FAILURE_TERMINAL, TRUTH_INTERRUPTED_TERMINAL}
            else None
        )
    prediction_root = (repo_root / PREDICTION_RELATIVE_ROOT).resolve()
    completion_path = prediction_root / "completion.json"
    if not completion_path.is_file():
        return None
    truth_receipt = (
        repo_root / TRUTH_JOIN_RECEIPT_RELATIVE_PATH
    ).resolve()
    truth_opened = truth_receipt.is_file()
    terminal = (
        TRUTH_INTERRUPTED_TERMINAL
        if truth_opened
        else PRETRUTH_FAILURE_TERMINAL
    )
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_durable(
        failure_path,
        {
            "schema": FAILURE_SCHEMA,
            "terminal": terminal,
            "preprocessor_completion_sha256": sha256(completion_path),
            "truth_join_once_receipt_sha256": (
                sha256(truth_receipt) if truth_opened else None
            ),
            "future_truth_opened": truth_opened,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "partial_truth_or_evaluation_artifacts_preserved": True,
            "second_truth_join_authorized": False,
            "source_replacement_authorized": False,
            "same_cohort_retuning_authorized": False,
            "freeze_rgb_student_contract_authorized": False,
            "rgb_student_execution_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
            "android_changed": False,
            "production_authorized": False,
            "safety_claim_authorized": False,
        },
    )
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.contract.resolve(), args.output.resolve())
        print(json.dumps({"terminal": result["terminal"]}))
        return 0
    except (
        OSError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        terminal = seal_failure_if_attempted(
            args.output.resolve(),
            error,
        )
        print(
            json.dumps(
                {
                    "ok": False,
                    "terminal": terminal,
                    "error": str(error),
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
