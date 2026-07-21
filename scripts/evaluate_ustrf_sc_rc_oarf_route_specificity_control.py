#!/usr/bin/env python3
"""Evaluate a receipt-bound within-image wrong-route control over frozen r816 predictions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


CONTRACT_SCHEMA = "blindassist_ustrf_sc_rc_oarf_route_specificity_control_contract_v1"
REPORT_SCHEMA = "blindassist_ustrf_sc_rc_oarf_route_specificity_control_report_v1"


class ContractError(ValueError):
    """The frozen control cannot be evaluated from the supplied receipts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ContractError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read JSONL {path}: {error}") from error
    return rows


def binary_metrics(truth: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if len(truth) != len(predictions) or not truth:
        raise ContractError("metrics require aligned non-empty truth and predictions")
    if any(value not in (0, 1) for value in (*truth, *predictions)):
        raise ContractError("truth and predictions must be binary")
    tn = sum(expected == 0 and actual == 0 for expected, actual in zip(truth, predictions))
    fp = sum(expected == 0 and actual == 1 for expected, actual in zip(truth, predictions))
    fn = sum(expected == 1 and actual == 0 for expected, actual in zip(truth, predictions))
    tp = sum(expected == 1 and actual == 1 for expected, actual in zip(truth, predictions))
    no_alert_recall = tn / (tn + fp) if tn + fp else 0.0
    alert_recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "confusion_matrix_rows_truth_columns_prediction": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / len(truth),
        "candidate_no_alert_recall": no_alert_recall,
        "candidate_alert_recall": alert_recall,
        "balanced_accuracy": (no_alert_recall + alert_recall) / 2.0,
    }


def _bound_file(repo_root: Path, binding: dict[str, Any], *, where: str) -> Path:
    relative = binding.get("path")
    expected_hash = binding.get("sha256")
    if not isinstance(relative, str) or not relative or not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ContractError(f"{where} path/hash binding is invalid")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ContractError(f"{where} escapes repository root") from error
    if not path.is_file() or sha256_file(path).lower() != expected_hash.lower():
        raise ContractError(f"{where} is missing or its SHA256 changed")
    return path


def _nested(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ContractError(f"missing bound report field: {dotted_path}")
        current = current[key]
    return current


def _nested_optional(value: dict[str, Any], dotted_path: str) -> Any | None:
    current: Any = value
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def evaluate(contract: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ContractError("unexpected route-specificity contract schema")
    report_binding = contract.get("bound_r816_report")
    legacy_binding = contract.get("legacy_prediction_replay")
    frozen_execution = contract.get("frozen_execution")
    post_run_disclosure = contract.get("post_run_binding_disclosure")
    stability_binding = contract.get("bound_stability_gate")
    rows_binding = contract.get("bound_route_examples")
    policy = contract.get("permutation_policy")
    gate = contract.get("gate")
    authority = contract.get("authority")
    if not all(isinstance(value, dict) for value in (report_binding, rows_binding, policy, gate, authority)):
        raise ContractError("contract bindings, policy, gate, and authority must be objects")
    if authority.get("train_only_synthetic_mechanism_evidence") is not True or any(
        authority.get(key) is not False
        for key in (
            "real_event_truth", "route_provider_evaluation", "student_training", "calibration",
            "blind", "android_runtime_change", "production_model_replacement",
        )
    ):
        raise ContractError("contract authority must remain train-only synthetic and fail closed")

    report_path = _bound_file(repo_root, report_binding, where="bound_r816_report")
    rows_path = _bound_file(repo_root, rows_binding, where="bound_route_examples")
    report = load_json(report_path)
    rows = load_jsonl(rows_path)
    if report.get("schema") != report_binding.get("schema"):
        raise ContractError("bound r816 report schema changed")
    if report.get("dataset_build_receipt_sha256") != rows_binding.get("dataset_build_receipt_sha256"):
        raise ContractError("r816 dataset build receipt binding changed")
    if report.get("dataset_manual_review_sha256") != rows_binding.get("manual_review_sha256"):
        raise ContractError("r816 manual-review binding changed")

    predictions = _nested(report, str(report_binding.get("prediction_path")))
    expected_count = report_binding.get("example_count")
    if not isinstance(predictions, list) or len(predictions) != len(rows) or len(rows) != expected_count:
        raise ContractError("prediction, route-row, or frozen example count differs")
    if any(not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1) for value in predictions):
        raise ContractError("r816 predictions must be binary integers")

    legacy_predictions_exact: bool | None = None
    legacy_evaluation_exact: bool | None = None
    legacy_report_path: Path | None = None
    if legacy_binding is not None:
        if (
            not isinstance(legacy_binding, dict)
            or legacy_binding.get("require_exact_predictions") is not True
            or legacy_binding.get("require_exact_evaluation_except_example_ids") is not True
        ):
            raise ContractError("legacy replay must require exact predictions and evaluation parity")
        legacy_report_path = _bound_file(repo_root, legacy_binding, where="legacy_prediction_replay")
        legacy_report = load_json(legacy_report_path)
        if legacy_report.get("schema") != legacy_binding.get("schema"):
            raise ContractError("legacy r816 report schema changed")
        legacy_predictions = _nested(legacy_report, str(legacy_binding.get("prediction_path")))
        legacy_predictions_exact = legacy_predictions == predictions
        if not legacy_predictions_exact:
            raise ContractError("identity-bound r816 predictions differ from the frozen legacy receipt")
        current_evaluation = copy.deepcopy(_nested(report, "evaluation"))
        legacy_evaluation = copy.deepcopy(_nested(legacy_report, "evaluation"))
        for readout in ("global_readout", "route_conditioned_readout", "exact_field_linear_head"):
            current_branch = _nested(current_evaluation, readout)
            legacy_branch = _nested(legacy_evaluation, readout)
            if not isinstance(current_branch, dict) or not isinstance(legacy_branch, dict):
                raise ContractError("r816 evaluation readout is not an object")
            current_branch.pop("example_ids", None)
            legacy_branch.pop("example_ids", None)
        legacy_evaluation_exact = current_evaluation == legacy_evaluation
        if not legacy_evaluation_exact:
            raise ContractError("identity-bound r816 evaluation differs from the frozen legacy receipt")
        if not isinstance(frozen_execution, dict):
            raise ContractError("identity-bound replay requires a frozen execution contract")
        if (
            not isinstance(post_run_disclosure, dict)
            or post_run_disclosure.get("identity_report_was_created_before_this_hash_binding") is not True
        ):
            raise ContractError("identity-bound replay must disclose its post-run hash binding")
        legacy_contract_binding = post_run_disclosure.get("legacy_preregistered_contract")
        if not isinstance(legacy_contract_binding, dict):
            raise ContractError("identity-bound replay lacks the preregistered legacy contract binding")
        legacy_contract_path = _bound_file(repo_root, legacy_contract_binding, where="legacy_preregistered_contract")
        legacy_contract = load_json(legacy_contract_path)
        unchanged = post_run_disclosure.get("unchanged_from_legacy_contract")
        if unchanged != ["route rows", "wrong-route permutations", "gate thresholds", "authority"]:
            raise ContractError("identity-bound replay disclosure changed the inherited frozen scope")
        for key in ("bound_route_examples", "permutation_policy", "gate", "authority"):
            if contract.get(key) != legacy_contract.get(key):
                raise ContractError(f"identity-bound replay changed inherited legacy scope: {key}")
        try:
            report_created_at = datetime.fromisoformat(str(report["created_at_utc"]).replace("Z", "+00:00"))
            contract_bound_at = datetime.fromisoformat(str(contract["frozen_at_utc"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("identity-bound replay timestamps are invalid") from error
        if report_created_at.tzinfo is None or contract_bound_at.tzinfo is None or contract_bound_at < report_created_at:
            raise ContractError("identity-bound contract must postdate the report it hash-binds")
        representation = report.get("frozen_risk_representation")
        thresholds = _nested(report, "route_interaction_gate.thresholds")
        if not isinstance(representation, dict) or not isinstance(thresholds, dict):
            raise ContractError("identity-bound r816 report lacks frozen execution metadata")
        for key in (
            "checkpoint_sha256", "input_size", "layer_index", "teacher_target",
            "distance_sigma_patches", "seed", "teacher_ridge", "head_ridge",
        ):
            if representation.get(key) != frozen_execution.get(key):
                raise ContractError(f"identity-bound r816 execution changed: {key}")
        for key in (
            "route_balanced_accuracy_gte", "each_class_recall_gte",
            "balanced_accuracy_gain_over_global_gte",
        ):
            if thresholds.get(key) != frozen_execution.get(key):
                raise ContractError(f"identity-bound r816 gate changed: {key}")

    required_choices = rows_binding.get("required_route_choices")
    controls = policy.get("controls")
    if policy.get("kind") != "within_image_cyclic_derangements_only" or policy.get("seed") is not None or policy.get("threshold_or_model_refit") is not False:
        raise ContractError("permutation policy is not the frozen no-refit cyclic control")
    if not isinstance(required_choices, list) or len(required_choices) != 3 or len(set(required_choices)) != 3:
        raise ContractError("exactly three distinct route choices are required")
    if not isinstance(controls, dict) or len(controls) != 2:
        raise ContractError("exactly two frozen wrong-route controls are required")
    choice_set = set(required_choices)
    for name, mapping in controls.items():
        if not isinstance(mapping, dict) or set(mapping) != choice_set or set(mapping.values()) != choice_set:
            raise ContractError(f"control {name} is not a route permutation")
        if any(mapping[choice] == choice for choice in required_choices):
            raise ContractError(f"control {name} is not a derangement")
    mapping_signatures = {
        tuple(mapping[choice] for choice in required_choices)
        for mapping in controls.values()
    }
    if len(mapping_signatures) != len(controls):
        raise ContractError("wrong-route controls must be distinct derangements")

    by_image: dict[str, dict[str, int]] = defaultdict(dict)
    image_source: dict[str, str] = {}
    truth: list[int] = []
    sources: list[str] = []
    seen_examples: set[str] = set()
    for index, row in enumerate(rows):
        try:
            example_id = str(row["example_id"])
            image_id = str(row["image_id"])
            source_id = str(row["parent_source_id"])
            route_choice = str(row["route_choice"])
            raw_label = row["route_blocked"]
            if not isinstance(raw_label, (bool, int)) or isinstance(raw_label, int) and not isinstance(raw_label, bool) and raw_label not in (0, 1):
                raise ValueError("route_blocked is not binary")
            label = int(raw_label)
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"route row {index} lacks a valid identity, source, route, or label") from error
        if not example_id or example_id in seen_examples or not image_id or not source_id or route_choice not in choice_set or label not in (0, 1) or row.get("train_only") is not True:
            raise ContractError(f"route row {index} violates frozen identity/route/label constraints")
        seen_examples.add(example_id)
        if image_id in image_source and image_source[image_id] != source_id:
            raise ContractError(f"image {image_id} crosses parent sources")
        image_source[image_id] = source_id
        if route_choice in by_image[image_id]:
            raise ContractError(f"image {image_id} repeats route choice {route_choice}")
        by_image[image_id][route_choice] = index
        truth.append(label)
        sources.append(source_id)
    if any(set(group) != choice_set for group in by_image.values()):
        raise ContractError("each image must contain exactly LEFT/STRAIGHT/RIGHT route rows")

    row_example_ids = [str(row["example_id"]) for row in rows]
    prediction_example_ids = _nested_optional(
        report, str(report_binding.get("prediction_example_id_path"))
    )
    ordering_identity_bound = (
        isinstance(prediction_example_ids, list)
        and prediction_example_ids == row_example_ids
    )

    true_metrics = binary_metrics(truth, predictions)
    bound_metrics = _nested(report, "evaluation.route_conditioned_readout.metrics")
    for key in ("balanced_accuracy", "candidate_no_alert_recall", "candidate_alert_recall"):
        if not isinstance(bound_metrics, dict) or abs(float(bound_metrics[key]) - float(true_metrics[key])) > 1e-12:
            raise ContractError(f"recomputed true-route metric differs from r816: {key}")

    wrong_predictions: dict[str, list[int]] = {}
    wrong_metrics: dict[str, dict[str, Any]] = {}
    for name, mapping in controls.items():
        permuted = [
            predictions[by_image[str(row["image_id"])][mapping[str(row["route_choice"])]]]
            for row in rows
        ]
        wrong_predictions[name] = permuted
        wrong_metrics[name] = binary_metrics(truth, permuted)

    per_source: dict[str, Any] = {}
    for source_id in dict.fromkeys(sources):
        indices = [index for index, value in enumerate(sources) if value == source_id]
        source_truth = [truth[index] for index in indices]
        source_true = binary_metrics(source_truth, [predictions[index] for index in indices])
        source_wrong = {
            name: binary_metrics(source_truth, [values[index] for index in indices])
            for name, values in wrong_predictions.items()
        }
        per_source[source_id] = {"example_count": len(indices), "true_route": source_true, "wrong_route_controls": source_wrong}

    minimum_ba = float(gate.get("minimum_true_route_balanced_accuracy", -1))
    minimum_recall = float(gate.get("minimum_true_route_each_class_recall", -1))
    minimum_gain = float(gate.get("minimum_true_minus_each_wrong_route_balanced_accuracy", -1))
    if not 0 <= minimum_ba <= 1 or not 0 <= minimum_recall <= 1 or not 0 <= minimum_gain <= 1:
        raise ContractError("gate thresholds must be within 0..1")
    if gate.get("every_parent_source_true_route_must_exceed_every_wrong_route") is not True:
        raise ContractError("per-source same-direction gate must remain enabled")
    if gate.get("formal_gate_requires_identity_bound_predictions") is not True:
        raise ContractError("formal gate must require identity-bound predictions")
    checks = {
        "true_route_balanced_accuracy": true_metrics["balanced_accuracy"] >= minimum_ba,
        "true_route_each_class_recall": min(
            true_metrics["candidate_no_alert_recall"], true_metrics["candidate_alert_recall"]
        ) >= minimum_recall,
        "gain_over_each_wrong_route": all(
            true_metrics["balanced_accuracy"] - metrics["balanced_accuracy"] >= minimum_gain
            for metrics in wrong_metrics.values()
        ),
        "every_parent_source_same_direction": all(
            values["true_route"]["balanced_accuracy"] > max(
                metrics["balanced_accuracy"] for metrics in values["wrong_route_controls"].values()
            )
            for values in per_source.values()
        ),
    }
    mechanism_checks_passed = all(checks.values())
    stability_summary: dict[str, Any] = {
        "bound": False,
        "passed": None,
        "stability": None,
    }
    if stability_binding is not None:
        if not isinstance(stability_binding, dict) or stability_binding.get("required_for_student_training") is not True:
            raise ContractError("bound stability receipt must remain required for student training")
        stability_contract_binding = stability_binding.get("contract")
        stability_report_binding = stability_binding.get("report")
        if not isinstance(stability_contract_binding, dict) or not isinstance(stability_report_binding, dict):
            raise ContractError("stability contract/report bindings must be objects")
        stability_contract_path = _bound_file(repo_root, stability_contract_binding, where="bound_stability_contract")
        stability_report_path = _bound_file(repo_root, stability_report_binding, where="bound_stability_report")
        stability_contract = load_json(stability_contract_path)
        stability_report = load_json(stability_report_path)
        if stability_report.get("schema") != stability_report_binding.get("schema"):
            raise ContractError("bound stability report schema changed")
        if stability_report.get("contract_sha256") != sha256_file(stability_contract_path):
            raise ContractError("stability report no longer binds its frozen contract")
        frame_binding = stability_contract.get("bound_frame_probe")
        if not isinstance(frame_binding, dict) or not isinstance(legacy_binding, dict):
            raise ContractError("stability receipt lacks the frozen legacy r816 binding")
        if frame_binding.get("sha256") != legacy_binding.get("sha256"):
            raise ContractError("stability receipt targets a different r816 prediction set")
        stability_metrics = stability_report.get("stability")
        stability_gate = stability_report.get("prototype_bootstrap_gate")
        stability_thresholds = stability_contract.get("gate")
        if (
            not isinstance(stability_metrics, dict)
            or not isinstance(stability_gate, dict)
            or not isinstance(stability_gate.get("passed"), bool)
            or not isinstance(stability_thresholds, dict)
        ):
            raise ContractError("stability report lacks a valid gate and metrics")
        try:
            stability_checks = {
                "worst_seed_balanced_accuracy": float(stability_metrics["worst_seed_balanced_accuracy"])
                >= float(stability_thresholds["minimum_worst_seed_balanced_accuracy"]),
                "worst_seed_each_class_recall": min(
                    float(stability_metrics["worst_seed_candidate_no_alert_recall"]),
                    float(stability_metrics["worst_seed_candidate_alert_recall"]),
                ) >= float(stability_thresholds["minimum_worst_seed_each_class_recall"]),
                "mean_balanced_accuracy": float(stability_metrics["mean_balanced_accuracy"])
                >= float(stability_thresholds["minimum_mean_balanced_accuracy"]),
                "balanced_accuracy_stddev": float(stability_metrics["stddev_balanced_accuracy"])
                <= float(stability_thresholds["maximum_seed_balanced_accuracy_stddev"]),
                "worst_seed_parent_source_balanced_accuracy": float(
                    stability_metrics["worst_seed_parent_source_balanced_accuracy"]
                ) >= float(stability_thresholds["minimum_worst_seed_parent_source_balanced_accuracy"]),
                "repeat_exact": stability_report.get("repeat_exact") is stability_thresholds.get("repeat_exact") is True,
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("stability report cannot be recomputed from its frozen thresholds") from error
        recomputed_stability_passed = all(stability_checks.values())
        if stability_gate["passed"] is not recomputed_stability_passed:
            raise ContractError("stability gate boolean differs from recomputed frozen checks")
        stability_summary = {
            "bound": True,
            "passed": recomputed_stability_passed,
            "stability": stability_metrics,
            "recomputed_checks": stability_checks,
            "contract_path": str(stability_contract_path),
            "contract_sha256": sha256_file(stability_contract_path),
            "report_path": str(stability_report_path),
            "report_sha256": sha256_file(stability_report_path),
        }
    route_gate_passed = mechanism_checks_passed and ordering_identity_bound
    combined_readiness = (
        "BLOCKED_ON_R818_STABILITY"
        if route_gate_passed and stability_summary["bound"] and stability_summary["passed"] is False
        else "SYNTHETIC_GATES_READY_FOR_HUMAN_U0"
        if route_gate_passed and stability_summary["bound"] and stability_summary["passed"] is True
        else "STABILITY_GATE_NOT_BOUND"
        if route_gate_passed
        else "BLOCKED_ON_ROUTE_SPECIFICITY"
    )
    return {
        "schema": REPORT_SCHEMA,
        "contract_id": contract.get("contract_id"),
        "bound_inputs": {
            "r816_report_path": str(report_path),
            "r816_report_sha256": sha256_file(report_path),
            "route_examples_path": str(rows_path),
            "route_examples_sha256": sha256_file(rows_path),
        },
        "ordering_check": {
            "prediction_count_equals_row_count": len(predictions) == len(rows),
            "each_image_has_exact_route_triplet": True,
            "binary_prediction_and_label": True,
            "prediction_example_ids_present_and_exact": ordering_identity_bound,
        },
        "legacy_prediction_replay": {
            "required": legacy_binding is not None,
            "exact": legacy_predictions_exact,
            "evaluation_exact_except_example_ids": legacy_evaluation_exact,
            "legacy_report_path": str(legacy_report_path) if legacy_report_path is not None else None,
            "legacy_report_sha256": sha256_file(legacy_report_path) if legacy_report_path is not None else None,
        },
        "example_count": len(rows),
        "image_count": len(by_image),
        "parent_source_count": len(per_source),
        "permutation_policy": policy,
        "true_route": true_metrics,
        "wrong_route_controls": wrong_metrics,
        "per_parent_source": per_source,
        "checks": checks,
        "mechanism_signal_observed": mechanism_checks_passed,
        "gate_passed": route_gate_passed,
        "decision": (
            "PASS_IDENTITY_BOUND_SYNTHETIC_ROUTE_SPECIFICITY"
            if mechanism_checks_passed and ordering_identity_bound
            else "BLOCKED_ON_PREDICTION_IDENTITY_BINDING"
            if mechanism_checks_passed
            else "FAIL_ROUTE_SPECIFICITY"
        ),
        "blocking_reason": (
            None if route_gate_passed
            else "r816 report lacks an example_id sequence binding each prediction to route_examples order"
            if not ordering_identity_bound
            else "route-specificity mechanism checks failed"
        ),
        "stability_gate": stability_summary,
        "combined_readiness_decision": combined_readiness,
        "repeat_exact": True,
        "interpretation": "Passing establishes synthetic within-image route specificity only; it does not establish route-provider, human-event, device, or production validity.",
        "authority": authority,
    }


def run(contract_path: Path, output: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    repo_root = contract_path.parent.parent
    contract = load_json(contract_path)
    first = evaluate(contract, repo_root=repo_root)
    second = evaluate(contract, repo_root=repo_root)
    if first != second:
        raise ContractError("route-specificity evaluation is not exactly repeatable")
    result = dict(first)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["contract_path"] = str(contract_path)
    result["contract_sha256"] = sha256_file(contract_path)
    result["evaluator_path"] = str(Path(__file__).resolve())
    result["evaluator_sha256"] = sha256_file(Path(__file__).resolve())
    if output.exists():
        raise ContractError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        report = run(args.contract, args.output)
    except (ContractError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "gate_passed": report["gate_passed"],
        "decision": report["decision"],
        "mechanism_signal_observed": report["mechanism_signal_observed"],
        "combined_readiness_decision": report["combined_readiness_decision"],
        "true_route_balanced_accuracy": report["true_route"]["balanced_accuracy"],
        "wrong_route_balanced_accuracy": {
            name: metrics["balanced_accuracy"] for name, metrics in report["wrong_route_controls"].items()
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
