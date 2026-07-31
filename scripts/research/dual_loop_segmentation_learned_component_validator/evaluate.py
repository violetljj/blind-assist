"""Run the single frozen grouped Logistic Regression component validator."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .core import (
    ARM_IDS,
    CAUSAL_ARM_ID,
    CANDIDATE_ID,
    CONFIDENCE_ARM_ID,
    EVALUATION_SCHEMA_VERSION,
    HISTORICAL_ARM_ID,
    LEARNED_ARM_ID,
    MODEL_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    PROTOCOL_ID,
    RAW_ARM_ID,
    aggregate_arm_report,
    atomic_output_directory,
    build_frame_contexts,
    build_frame_metric_rows,
    build_reference_masks,
    feature_matrix,
    gate_checks,
    load_bound_inputs,
    masks_from_component_decisions,
    near_miss_eligible,
    normalized_gate_margins,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    sigmoid_scores,
    stable_high_confidence_diagnostic,
    utility_values,
    validate_static_config,
    validate_table_contract,
    verify_output_scope,
    write_json,
    write_jsonl,
)


HISTORICAL_ARM_MAPPING = {
    RAW_ARM_ID: "BASELINE_UNFILTERED",
    CAUSAL_ARM_ID: "REFERENCE_CAUSAL_2_OF_3_UNION",
    CONFIDENCE_ARM_ID: "REFERENCE_CONFIDENCE_GE_0_65",
    HISTORICAL_ARM_ID: "CLASS_CONDITIONED_MULTI_NEGATIVE",
}


def current_git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def session_class_balanced_weights(
    labels: np.ndarray,
    sessions: Sequence[str],
) -> np.ndarray:
    if labels.ndim != 1 or labels.shape[0] != len(sessions):
        raise ValueError("labels/session shape mismatch")
    session_counts = Counter(str(value) for value in sessions)
    class_counts = Counter(int(value) for value in labels)
    if set(class_counts) != {0, 1}:
        raise ValueError("training labels must contain both classes")
    total = len(labels)
    session_count = len(session_counts)
    result = np.asarray(
        [
            (total / (session_count * session_counts[str(session)]))
            * (total / (2 * class_counts[int(label)]))
            for label, session in zip(labels, sessions)
        ],
        dtype=np.float64,
    )
    result /= float(np.mean(result))
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("sample weights are invalid")
    return result


def fit_frozen_logistic(
    *,
    matrix: np.ndarray,
    labels: np.ndarray,
    sessions: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    weights = session_class_balanced_weights(labels, sessions)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(matrix, sample_weight=weights)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("invalid fitted scaler")
    transformed = scaler.transform(matrix)
    model = LogisticRegression(
        penalty=str(contract["penalty"]),
        C=float(contract["C"]),
        solver=str(contract["solver"]),
        fit_intercept=bool(contract["fit_intercept"]),
        max_iter=int(contract["max_iter"]),
        tol=float(contract["tol"]),
        random_state=int(contract["random_state"]),
    )
    model.fit(transformed, labels, sample_weight=weights)
    coefficients = np.asarray(model.coef_[0], dtype=np.float64)
    intercept = float(model.intercept_[0])
    pure_scores = sigmoid_scores(
        matrix,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=scale,
        coefficients=coefficients,
        intercept=intercept,
    )
    sklearn_scores = model.predict_proba(transformed)[:, 1]
    if not np.allclose(pure_scores, sklearn_scores, atol=1e-12, rtol=1e-12):
        raise ValueError("pure NumPy sigmoid does not reproduce sklearn")
    return {
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": scale,
        "coefficients": coefficients,
        "intercept": intercept,
        "n_iter": int(model.n_iter_[0]),
        "training_row_count": int(matrix.shape[0]),
        "training_session_ids": sorted(set(str(value) for value in sessions)),
        "training_label_counts": {
            "reject": int(np.count_nonzero(labels == 0)),
            "keep": int(np.count_nonzero(labels == 1)),
        },
        "sample_weight": {
            "minimum": float(np.min(weights)),
            "maximum": float(np.max(weights)),
            "mean": float(np.mean(weights)),
            "sum": float(np.sum(weights)),
        },
    }


def model_scores(model: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
    return sigmoid_scores(
        matrix,
        mean=np.asarray(model["mean"], dtype=np.float64),
        scale=np.asarray(model["scale"], dtype=np.float64),
        coefficients=np.asarray(model["coefficients"], dtype=np.float64),
        intercept=float(model["intercept"]),
    )


def serializable_model(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean": np.asarray(model["mean"], dtype=np.float64).tolist(),
        "scale": np.asarray(model["scale"], dtype=np.float64).tolist(),
        "coefficients": np.asarray(
            model["coefficients"], dtype=np.float64
        ).tolist(),
        "intercept": float(model["intercept"]),
        "n_iter": int(model["n_iter"]),
        "training_row_count": int(model["training_row_count"]),
        "training_session_ids": list(model["training_session_ids"]),
        "training_label_counts": dict(model["training_label_counts"]),
        "sample_weight": dict(model["sample_weight"]),
    }


def _rows_for_sessions(
    rows: Sequence[dict[str, Any]],
    sessions: set[str],
) -> list[dict[str, Any]]:
    return [
        row for row in rows if str(row["identity"]["session_id"]) in sessions
    ]


def _contexts_for_sessions(contexts: Sequence[Any], sessions: set[str]) -> list[Any]:
    return [context for context in contexts if context.session_id in sessions]


def score_probability_threshold(
    *,
    config: dict[str, Any],
    contexts: Sequence[Any],
    probabilities: Mapping[str, float],
    threshold: float,
    raw_masks: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    keep = {
        component_id: float(probability) >= float(threshold)
        for component_id, probability in probabilities.items()
    }
    learned_masks = masks_from_component_decisions(
        config=config,
        contexts=contexts,
        keep_by_component_id=keep,
    )
    frame_rows = build_frame_metric_rows(
        contexts=contexts,
        arm_masks={RAW_ARM_ID: raw_masks, LEARNED_ARM_ID: learned_masks},
    )
    classes = [str(value) for value in config["candidate_classes"]]
    baseline = aggregate_arm_report(frame_rows, RAW_ARM_ID, classes)
    candidate = aggregate_arm_report(frame_rows, LEARNED_ARM_ID, classes)
    values = utility_values(candidate=candidate, baseline=baseline)
    gate_result = gate_checks(values, config["utility_gates"])
    margins = normalized_gate_margins(values, config["utility_gates"])
    return {
        "threshold": float(threshold),
        "values": values,
        "gates": gate_result,
        "normalized_gate_margins": margins,
        "minimum_normalized_gate_margin": min(margins.values()),
    }


def select_threshold(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("threshold candidate list is empty")
    return max(
        candidates,
        key=lambda value: (
            float(value["minimum_normalized_gate_margin"]),
            float(value["values"]["minimum_session_recall_retention"]),
            float(value["values"]["fp_pixel_reduction"]),
            -float(value["threshold"]),
        ),
    )


def verify_historical_reproduction(
    *,
    new_frame_rows: Sequence[dict[str, Any]],
    historical_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    historical_by_id = {
        str(row["view_row_id"]): row for row in historical_rows
    }
    if len(historical_by_id) != len(historical_rows):
        raise ValueError("duplicate historical frame identity")
    checks = 0
    for new in new_frame_rows:
        old = historical_by_id.get(str(new["view_row_id"]))
        if old is None:
            raise ValueError("historical frame is missing")
        for new_arm, old_arm in HISTORICAL_ARM_MAPPING.items():
            new_metrics = new["arms"][new_arm]
            old_metrics = old["arms"][old_arm]
            for field in ("tp", "fp", "fn", "tn"):
                if int(new_metrics["candidate"][field]) != int(
                    old_metrics["pixel"][field]
                ):
                    raise ValueError(
                        f"historical reproduction drift: "
                        f"{new['view_row_id']}/{new_arm}/{field}"
                    )
                checks += 1
            if int(
                new_metrics["components"]["false_activation_component_count"]
            ) != int(old_metrics["any_hazard_false_component_count"]):
                raise ValueError(
                    f"historical component reproduction drift: "
                    f"{new['view_row_id']}/{new_arm}"
                )
            checks += 1
            for class_name in ("boundary_step_curb", "obstacle"):
                for field in ("tp", "fp", "fn", "tn"):
                    if int(
                        new_metrics["classes"][class_name]["pixel"][field]
                    ) != int(
                        old_metrics["classes"][class_name]["pixel"][field]
                    ):
                        raise ValueError(
                            f"historical class reproduction drift: "
                            f"{new['view_row_id']}/{new_arm}/{class_name}/{field}"
                        )
                    checks += 1
    return {
        "status": "EXACT_INTEGER_REPRODUCED",
        "check_count": checks,
        "frame_count": len(new_frame_rows),
        "arm_mapping": HISTORICAL_ARM_MAPPING,
    }


def coefficient_summary(
    *,
    feature_names: Sequence[str],
    models: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    matrix = np.asarray(
        [model["model"]["coefficients"] for model in models],
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        values = matrix[:, index]
        rows.append(
            {
                "feature": str(name),
                "mean_standardized_coefficient": float(np.mean(values)),
                "mean_absolute_standardized_coefficient": float(
                    np.mean(np.abs(values))
                ),
                "minimum_standardized_coefficient": float(np.min(values)),
                "maximum_standardized_coefficient": float(np.max(values)),
                "positive_fold_count": int(np.count_nonzero(values > 0)),
                "negative_fold_count": int(np.count_nonzero(values < 0)),
                "zero_fold_count": int(np.count_nonzero(values == 0)),
            }
        )
    rows.sort(
        key=lambda value: (
            -float(value["mean_absolute_standardized_coefficient"]),
            str(value["feature"]),
        )
    )
    return rows


def crossfit_ablation_predictions(
    *,
    config: dict[str, Any],
    table_rows: Sequence[dict[str, Any]],
    fold_models: Sequence[dict[str, Any]],
) -> dict[str, dict[str, bool]]:
    feature_names = [str(value) for value in config["feature_contract"]["feature_names"]]
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    table_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        table_by_session[str(row["identity"]["session_id"])].append(row)
    outputs: dict[str, dict[str, bool]] = {
        block: {} for block in config["feature_contract"]["diagnostic_ablation_blocks"]
    }
    for fold in fold_models:
        held_out = str(fold["held_out_session_id"])
        rows = table_by_session[held_out]
        matrix = feature_matrix(config, rows)
        model = fold["model"]
        mean = np.asarray(model["mean"], dtype=np.float64)
        for block, names in config["feature_contract"][
            "diagnostic_ablation_blocks"
        ].items():
            ablated = matrix.copy()
            for name in names:
                ablated[:, name_to_index[str(name)]] = mean[name_to_index[str(name)]]
            scores = model_scores(model, ablated)
            threshold = float(fold["selected_threshold"])
            for row, score in zip(rows, scores):
                outputs[str(block)][str(row["identity"]["component_id"])] = bool(
                    score >= threshold
                )
    expected_ids = {
        str(row["identity"]["component_id"]) for row in table_rows
    }
    for block, decisions in outputs.items():
        if set(decisions) != expected_ids:
            raise ValueError(f"ablation decision membership drifted: {block}")
    return outputs


def run_evaluation(
    *,
    repo_root: Path,
    config_path: Path,
    prepared_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    prepared_root = prepared_root.resolve()
    output_root = output_root.resolve()
    verify_output_scope(repo_root, output_root)
    config = read_json(config_path)
    validate_static_config(config)
    prepare_receipt_path = prepared_root / "prepare_receipt.json"
    component_table_path = prepared_root / "component_table.jsonl"
    prepare_receipt = read_json(prepare_receipt_path)
    if (
        prepare_receipt.get("status") != "COMPLETE"
        or prepare_receipt.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("prepared evidence is not COMPLETE")
    expected_table = prepare_receipt["output_files"]["component_table.jsonl"]
    if (
        sha256_file(component_table_path) != expected_table["sha256"]
        or len(read_jsonl(component_table_path)) != int(expected_table["row_count"])
    ):
        raise ValueError("prepared component table binding drifted")
    table_rows = read_jsonl(component_table_path)
    validate_table_contract(config, table_rows)
    inputs = load_bound_inputs(repo_root, config)
    contexts = build_frame_contexts(
        repo_root=repo_root,
        config=config,
        inputs=inputs,
        table_rows=table_rows,
    )
    reference_masks = build_reference_masks(config=config, contexts=contexts)
    sessions = sorted(
        {str(row["identity"]["session_id"]) for row in table_rows}
    )
    if len(sessions) != int(config["grouped_evaluation"]["outer_fold_count"]):
        raise ValueError("outer source-session count drifted")
    rows_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        rows_by_session[str(row["identity"]["session_id"])].append(row)

    outer_predictions: dict[str, dict[str, Any]] = {}
    fold_models: list[dict[str, Any]] = []
    threshold_grid = [
        float(value) for value in config["grouped_evaluation"]["threshold_grid"]
    ]
    model_contract = config["model_contract"]
    all_feature_names = [
        str(value) for value in config["feature_contract"]["feature_names"]
    ]

    for outer_index, held_out_session in enumerate(sessions):
        outer_training_sessions = [
            value for value in sessions if value != held_out_session
        ]
        inner_probabilities: dict[str, float] = {}
        inner_receipts: list[dict[str, Any]] = []
        for inner_session in outer_training_sessions:
            inner_fit_sessions = {
                value
                for value in outer_training_sessions
                if value != inner_session
            }
            fit_rows = _rows_for_sessions(table_rows, inner_fit_sessions)
            validation_rows = rows_by_session[inner_session]
            fit_matrix = feature_matrix(config, fit_rows)
            fit_labels = np.asarray(
                [int(row["target"]["keep_label"]) for row in fit_rows],
                dtype=np.int64,
            )
            fit_sessions = [
                str(row["identity"]["session_id"]) for row in fit_rows
            ]
            model = fit_frozen_logistic(
                matrix=fit_matrix,
                labels=fit_labels,
                sessions=fit_sessions,
                contract=model_contract,
            )
            validation_scores = model_scores(
                model, feature_matrix(config, validation_rows)
            )
            for row, score in zip(validation_rows, validation_scores):
                component_id = str(row["identity"]["component_id"])
                if component_id in inner_probabilities:
                    raise ValueError("duplicate inner OOF prediction")
                inner_probabilities[component_id] = float(score)
            inner_receipts.append(
                {
                    "inner_held_out_session_id": inner_session,
                    "training_session_ids": sorted(inner_fit_sessions),
                    "training_row_count": len(fit_rows),
                    "held_out_row_count": len(validation_rows),
                    "model_definition_sha256": _model_definition_sha(config),
                }
            )
        inner_sessions = set(outer_training_sessions)
        expected_inner_ids = {
            str(row["identity"]["component_id"])
            for row in _rows_for_sessions(table_rows, inner_sessions)
        }
        if set(inner_probabilities) != expected_inner_ids:
            raise ValueError("inner OOF membership is incomplete")
        inner_contexts = _contexts_for_sessions(contexts, inner_sessions)
        inner_raw_masks = {
            context.view_row_id: reference_masks[RAW_ARM_ID][context.view_row_id]
            for context in inner_contexts
        }
        threshold_candidates = [
            score_probability_threshold(
                config=config,
                contexts=inner_contexts,
                probabilities=inner_probabilities,
                threshold=threshold,
                raw_masks=inner_raw_masks,
            )
            for threshold in threshold_grid
        ]
        selected = select_threshold(threshold_candidates)

        outer_fit_rows = _rows_for_sessions(table_rows, set(outer_training_sessions))
        outer_model = fit_frozen_logistic(
            matrix=feature_matrix(config, outer_fit_rows),
            labels=np.asarray(
                [int(row["target"]["keep_label"]) for row in outer_fit_rows],
                dtype=np.int64,
            ),
            sessions=[
                str(row["identity"]["session_id"]) for row in outer_fit_rows
            ],
            contract=model_contract,
        )
        held_out_rows = rows_by_session[held_out_session]
        held_out_scores = model_scores(
            outer_model, feature_matrix(config, held_out_rows)
        )
        threshold = float(selected["threshold"])
        for row, score in zip(held_out_rows, held_out_scores):
            component_id = str(row["identity"]["component_id"])
            if component_id in outer_predictions:
                raise ValueError("duplicate outer held-out prediction")
            outer_predictions[component_id] = {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "candidate_id": CANDIDATE_ID,
                "fold_id": f"OUTER_{outer_index:02d}",
                "held_out_session_id": held_out_session,
                "component_id": component_id,
                "view_row_id": str(row["identity"]["view_row_id"]),
                "predicted_class": str(row["identity"]["predicted_class"]),
                "probability_keep": float(score),
                "selected_threshold": threshold,
                "keep": bool(float(score) >= threshold),
                "target_keep_label": int(row["target"]["keep_label"]),
            }
        fold_models.append(
            {
                "schema_version": MODEL_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "candidate_id": CANDIDATE_ID,
                "fold_id": f"OUTER_{outer_index:02d}",
                "held_out_session_id": held_out_session,
                "outer_training_session_ids": outer_training_sessions,
                "outer_held_out_row_count": len(held_out_rows),
                "model_definition_sha256": _model_definition_sha(config),
                "model": serializable_model(outer_model),
                "inner_oof": {
                    "method": config["grouped_evaluation"]["inner_method"],
                    "fold_count": len(inner_receipts),
                    "folds": inner_receipts,
                    "prediction_row_count": len(inner_probabilities),
                    "outer_held_out_session_present": False,
                },
                "threshold_selection": {
                    "grid": threshold_grid,
                    "candidate_count": len(threshold_candidates),
                    "rule": config["grouped_evaluation"]["threshold_selection"],
                    "candidates": threshold_candidates,
                    "selected": selected,
                },
                "selected_threshold": threshold,
            }
        )
    expected_ids = {
        str(row["identity"]["component_id"]) for row in table_rows
    }
    if set(outer_predictions) != expected_ids:
        raise ValueError("outer held-out prediction membership is incomplete")
    learned_keep = {
        component_id: bool(value["keep"])
        for component_id, value in outer_predictions.items()
    }
    learned_masks = masks_from_component_decisions(
        config=config,
        contexts=contexts,
        keep_by_component_id=learned_keep,
    )
    all_arm_masks = {
        **reference_masks,
        LEARNED_ARM_ID: learned_masks,
    }
    frame_rows = build_frame_metric_rows(
        contexts=contexts, arm_masks=all_arm_masks
    )
    reproduction = verify_historical_reproduction(
        new_frame_rows=frame_rows,
        historical_rows=inputs.historical_frame_rows,
    )
    classes = [str(value) for value in config["candidate_classes"]]
    reports = {
        arm_id: aggregate_arm_report(frame_rows, arm_id, classes)
        for arm_id in ARM_IDS
    }
    utilities: dict[str, Any] = {}
    for arm_id in ARM_IDS[1:]:
        values = utility_values(
            candidate=reports[arm_id],
            baseline=reports[RAW_ARM_ID],
        )
        utilities[arm_id] = {
            "values": values,
            "gates": gate_checks(values, config["utility_gates"]),
            "normalized_gate_margins": normalized_gate_margins(
                values, config["utility_gates"]
            ),
        }
    stable_diagnostic = stable_high_confidence_diagnostic(
        table_rows=table_rows,
        keep_by_component_id=learned_keep,
    )

    ablation_decisions = crossfit_ablation_predictions(
        config=config,
        table_rows=table_rows,
        fold_models=fold_models,
    )
    ablations: dict[str, Any] = {}
    for block, decisions in ablation_decisions.items():
        masks = masks_from_component_decisions(
            config=config,
            contexts=contexts,
            keep_by_component_id=decisions,
        )
        rows = build_frame_metric_rows(
            contexts=contexts,
            arm_masks={RAW_ARM_ID: reference_masks[RAW_ARM_ID], block: masks},
        )
        baseline = aggregate_arm_report(rows, RAW_ARM_ID, classes)
        candidate = aggregate_arm_report(rows, block, classes)
        values = utility_values(candidate=candidate, baseline=baseline)
        ablations[block] = {
            "method": "SET_BLOCK_TO_OUTER_TRAINING_SCALER_MEAN_NO_REFIT_NO_RETHRESHOLD",
            "diagnostic_only": True,
            "values": values,
            "gates": gate_checks(values, config["utility_gates"]),
        }

    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "stage": config["stage"],
        "evidence_instance": config["evidence_instance"],
        "claim_ceiling": config["claim_ceiling"],
        "execution_status": "EVALUATION_COMPLETE_ENGINEERING_PENDING",
        "scientific_terminal": "PENDING_ENGINEERING_GATES",
        "git_head": current_git_head(repo_root),
        "config": {
            "path": str(config_path.relative_to(repo_root)).replace("\\", "/"),
            "sha256": sha256_file(config_path),
        },
        "prepared_evidence": {
            "path": str(prepared_root.relative_to(repo_root)).replace("\\", "/"),
            "prepare_receipt_sha256": sha256_file(prepare_receipt_path),
            "component_table_sha256": sha256_file(component_table_path),
        },
        "runtime_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "grouped_evaluation": {
            "outer_method": config["grouped_evaluation"]["outer_method"],
            "outer_fold_count": len(fold_models),
            "inner_method": config["grouped_evaluation"]["inner_method"],
            "outer_held_out_prediction_count": len(outer_predictions),
            "claim": config["grouped_evaluation"]["claim"],
            "participant_route_parent_capture_independence": (
                "NOT_EVALUABLE_MISSING_IDENTIFIERS"
            ),
            "fresh_holdout_accessed": False,
        },
        "historical_reference_reproduction": reproduction,
        "arms": reports,
        "utility": utilities,
        "learned_utility_all_nine_passed": bool(
            utilities[LEARNED_ARM_ID]["gates"]["all_passed"]
        ),
        "stable_high_confidence_residual_diagnostic": stable_diagnostic,
        "feature_block_zero_ablation": ablations,
        "coefficient_summary": coefficient_summary(
            feature_names=all_feature_names,
            models=fold_models,
        ),
        "model_family_count": 1,
        "candidate_selection_or_hyperparameter_search": False,
        "android_or_alert_authority": "NONE",
        "confirmation": "NOT_ACTIVATED",
        "drives_alerts": False,
    }

    temporary, finalize = atomic_output_directory(output_root)
    try:
        models_path = temporary / "fold_models.jsonl"
        predictions_path = temporary / "held_out_predictions.jsonl"
        frames_path = temporary / "frame_metrics.jsonl"
        write_jsonl(models_path, fold_models)
        write_jsonl(
            predictions_path,
            [outer_predictions[key] for key in sorted(outer_predictions)],
        )
        write_jsonl(frames_path, frame_rows)
        result["output_files"] = {
            "fold_models.jsonl": {
                "sha256": sha256_file(models_path),
                "row_count": len(fold_models),
            },
            "held_out_predictions.jsonl": {
                "sha256": sha256_file(predictions_path),
                "row_count": len(outer_predictions),
            },
            "frame_metrics.jsonl": {
                "sha256": sha256_file(frames_path),
                "row_count": len(frame_rows),
            },
        }
        write_json(temporary / "result.json", result)
        finalize(True)
    except BaseException:
        finalize(False)
        raise
    return result


def _model_definition_sha(config: Mapping[str, Any]) -> str:
    frozen = {
        "feature_names": config["feature_contract"]["feature_names"],
        "model_contract": config["model_contract"],
        "grouped_evaluation": config["grouped_evaluation"],
        "utility_gates": config["utility_gates"],
    }
    return sha256_json(frozen)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_evaluation(
        repo_root=args.repo_root,
        config_path=args.config,
        prepared_root=args.prepared_root,
        output_root=args.output_root,
    )
    learned = result["utility"][LEARNED_ARM_ID]
    print(
        json.dumps(
            {
                "status": result["execution_status"],
                "fold_count": result["grouped_evaluation"]["outer_fold_count"],
                "prediction_count": result["grouped_evaluation"][
                    "outer_held_out_prediction_count"
                ],
                "utility_gates_passed": learned["gates"]["passed_count"],
                "utility_all_passed": learned["gates"]["all_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
