#!/usr/bin/env python3
"""Evaluate preregistered USTRF U0 arms against eligible GPT/Codex consensus event truth.

The evaluator is deliberately fail-closed. It recomputes the route-conditioned
truth gate, binds every arm to the exact config/manifest hashes, rejects future
or blind inputs, and never authorizes student training or production changes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "blindassist_ustrf_sc_u0_teacher_upper_bound_report_v1"
PREDICTION_SCHEMA = "blindassist_ustrf_sc_u0_six_arm_predictions_v2"


class ContractError(ValueError):
    """The supplied U0 evidence cannot be evaluated under the frozen contract."""


_VALIDATOR_PATH = Path(__file__).with_name("validate_sanpo_counterfactual_episodes.py")
_SPEC = importlib.util.spec_from_file_location("ustrf_u0_truth_validator", _VALIDATOR_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load truth validator: {_VALIDATOR_PATH}")
_TRUTH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRUTH)

_PREDICTION_EVIDENCE_PATH = Path(__file__).with_name("validate_ustrf_sc_u0_prediction_bundle.py")
_PREDICTION_EVIDENCE_SPEC = importlib.util.spec_from_file_location(
    "ustrf_u0_prediction_evidence_validator", _PREDICTION_EVIDENCE_PATH,
)
if _PREDICTION_EVIDENCE_SPEC is None or _PREDICTION_EVIDENCE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load U0 prediction evidence validator: {_PREDICTION_EVIDENCE_PATH}")
_PREDICTION_EVIDENCE = importlib.util.module_from_spec(_PREDICTION_EVIDENCE_SPEC)
_PREDICTION_EVIDENCE_SPEC.loader.exec_module(_PREDICTION_EVIDENCE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _metrics(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, float | int]:
    positives = 0
    critical = 0
    recalled = 0
    critical_misses = 0
    false_alerts = 0
    delivered = 0
    repeated = 0
    cleared = 0
    regenerations = 0
    non_abstained = 0
    duration_ms = 0
    negative_total = 0
    negative_clear = 0
    clearance_latencies: list[int] = []
    for truth, prediction in rows:
        duration = truth["duration_ms"]
        duration_ms += duration
        abstained = prediction["abstained"]
        non_abstained += int(not abstained)
        alerts = prediction["alert_timestamps_ms"]
        if truth["pair_role"] == "matched_negative":
            negative_total += 1
            negative_clear += int(not alerts)
            false_alerts += len(alerts)
            continue
        positives += 1
        critical += int(truth["expected_critical"])
        start = truth["alertable_start_ms"]
        end = truth["passed_or_cleared_ms"]
        in_event = [value for value in alerts if start <= value < end]
        out_of_event = [value for value in alerts if value < start or value >= end]
        hit = bool(in_event)
        recalled += int(hit)
        critical_misses += int(truth["expected_critical"] and not hit)
        delivered += len(in_event)
        repeated += max(0, len(in_event) - 1)
        false_alerts += len(out_of_event)
        post = [value for value in alerts if value >= end]
        cleared += int(not post)
        regenerations += int(bool(post))
        clearance_latencies.append(0 if not post else max(post) - end)
    minutes = duration_ms / 60000.0
    sorted_latencies = sorted(clearance_latencies)
    p95_latency = 0 if not sorted_latencies else sorted_latencies[math.ceil(0.95 * len(sorted_latencies)) - 1]
    return {
        "episode_count": len(rows),
        "positive_event_count": positives,
        "critical_event_count": critical,
        "event_recall": _ratio(recalled, positives),
        "critical_miss_rate": _ratio(critical_misses, critical),
        "false_alerts_per_minute": 0.0 if minutes == 0 else false_alerts / minutes,
        "delivered_alerts_per_event": _ratio(delivered, positives),
        "delivered_repeated_alert_rate": _ratio(repeated, delivered),
        "post_event_clearance_rate": _ratio(cleared, positives),
        "p95_clearance_latency_ms": p95_latency,
        "event_regeneration_rate": _ratio(regenerations, positives),
        "matched_negative_specificity": _ratio(negative_clear, negative_total),
        "matched_pair_balanced_accuracy": (
            _ratio(recalled, positives) + _ratio(negative_clear, negative_total)
        ) / 2.0,
        "non_abstain_coverage": _ratio(non_abstained, len(rows)),
    }


def _validate_predictions(
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    predictions: Mapping[str, Any],
    *,
    truth_root: Path,
    prediction_root: Path,
    truth_config_sha256: str,
    truth_manifest_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if predictions.get("schema") != contract.get("prediction_schema", PREDICTION_SCHEMA):
        raise ContractError("unexpected U0 prediction schema")
    if predictions.get("contract_id") != contract.get("contract_id"):
        raise ContractError("prediction contract_id does not match U0 contract")
    if predictions.get("truth_config_sha256") != truth_config_sha256:
        raise ContractError("prediction truth_config_sha256 mismatch")
    if predictions.get("truth_manifest_sha256") != truth_manifest_sha256:
        raise ContractError("prediction truth_manifest_sha256 mismatch")
    if predictions.get("blind_accessed") is not False:
        raise ContractError("U0 predictions must declare blind_accessed=false")
    if predictions.get("future_inputs_used") is not False:
        raise ContractError("U0 predictions must be causal and future-input-free")
    if predictions.get("production_model_replacement_authorized") is not False:
        raise ContractError("U0 predictions cannot authorize production replacement")
    if not isinstance(predictions.get("synthetic_fixture"), bool):
        raise ContractError("U0 predictions must declare synthetic_fixture explicitly")

    expected_episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    required = {row["arm_id"]: row for row in contract["required_arms"]}
    arms = predictions.get("arms")
    if (
        not isinstance(arms, list)
        or len(arms) != len(required)
        or any(not isinstance(row, dict) for row in arms)
        or len({row.get("arm_id") for row in arms}) != len(arms)
        or {row.get("arm_id") for row in arms} != set(required)
    ):
        raise ContractError("prediction arms must exactly match the preregistered arms")
    result: dict[str, dict[str, Any]] = {}
    frame_ledgers: set[str] = set()
    for arm in arms:
        arm_id = arm["arm_id"]
        expected_arm = required[arm_id]
        if arm.get("candidate_adapter_id") != expected_arm["candidate_adapter_id"]:
            raise ContractError(f"{arm_id}.candidate_adapter_id differs from the preregistered arm")
        for key in ("uses_explicit_route", "uses_dense_teacher", "uses_causal_lifecycle", "control"):
            if not isinstance(arm.get(key), bool) or arm[key] != expected_arm[key]:
                raise ContractError(f"{arm_id}.{key} differs from the preregistered arm")
        for key in ("fit_policy", "event_identity_policy", "route_input_policy"):
            if arm.get(key) != expected_arm.get(key):
                raise ContractError(f"{arm_id}.{key} differs from the preregistered arm")
        if arm.get("shared_decision_kernel_contract_id") != "blindassist_shared_decision_kernel_v1":
            raise ContractError(f"{arm_id} must use the shared production decision kernel")
        for key in ("implementation_sha256", "artifact_sha256", "threshold_config_sha256", "ordered_frame_ledger_sha256"):
            value = arm.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ContractError(f"{arm_id}.{key} must be a lowercase SHA-256")
        frame_ledgers.add(arm["ordered_frame_ledger_sha256"])
        episodes = arm.get("episodes")
        if not isinstance(episodes, list):
            raise ContractError(f"{arm_id}.episodes must be a list")
        by_id = {row.get("episode_id"): row for row in episodes if isinstance(row, dict)}
        if len(by_id) != len(episodes) or set(by_id) != set(expected_episodes):
            raise ContractError(f"{arm_id} must contain each truth episode exactly once")
        for episode_id, row in by_id.items():
            truth = expected_episodes[episode_id]
            if row.get("fold_held_out_session_id") != truth["session_id"]:
                raise ContractError(f"{arm_id}/{episode_id} is not bound to its LOSO holdout session")
            if row.get("input_video_sha256") != truth["video_sha256"]:
                raise ContractError(f"{arm_id}/{episode_id} input video hash mismatch")
            if row.get("source_route_intent_sha256") != truth["route_intent_sha256"]:
                raise ContractError(f"{arm_id}/{episode_id} source route hash mismatch")
            if row.get("source_capture_frame_ledger_sha256") != truth.get("capture_frame_ledger_sha256"):
                raise ContractError(f"{arm_id}/{episode_id} source capture frame ledger hash mismatch")
            for key in ("frame_ids_sha256", "prediction_trace_sha256"):
                value = row.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                    raise ContractError(f"{arm_id}/{episode_id}.{key} must be a lowercase SHA-256")
            if not isinstance(row.get("abstained"), bool):
                raise ContractError(f"{arm_id}/{episode_id}.abstained must be boolean")
            alerts = row.get("alert_timestamps_ms")
            if (
                not isinstance(alerts, list)
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > truth["duration_ms"] for value in alerts)
                or alerts != sorted(set(alerts))
            ):
                raise ContractError(f"{arm_id}/{episode_id}.alert_timestamps_ms must be sorted unique in-range integers")
            if row["abstained"] and alerts:
                raise ContractError(f"{arm_id}/{episode_id} cannot alert while abstained")
        result[arm_id] = by_id
    if len(frame_ledgers) != 1:
        raise ContractError("all U0 arms must use the same ordered frame ledger")
    try:
        evidence_report = _PREDICTION_EVIDENCE.validate_bundle(
            contract,
            manifest,
            predictions,
            truth_root=truth_root,
            prediction_root=prediction_root,
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ContractError(f"U0 prediction evidence gate failed: {error}") from error
    return result, evidence_report


def evaluate(
    contract: dict[str, Any],
    truth_config: dict[str, Any],
    truth_manifest: dict[str, Any],
    predictions: dict[str, Any],
    *,
    truth_root: Path,
    prediction_root: Path,
    truth_config_sha256: str,
    truth_manifest_sha256: str,
) -> dict[str, Any]:
    if contract.get("schema") != "blindassist_ustrf_sc_u0_teacher_upper_bound_contract_v1":
        raise ContractError("unexpected U0 contract schema")
    prediction_evidence_contract = contract.get("prediction_evidence_contract")
    if not isinstance(prediction_evidence_contract, dict):
        raise ContractError("U0 contract lacks prediction_evidence_contract")
    if normalized_text_sha256(_PREDICTION_EVIDENCE_PATH) != prediction_evidence_contract.get("validator_implementation_sha256"):
        raise ContractError("U0 prediction evidence validator implementation SHA mismatch")
    if truth_config.get("contract_id") != contract.get("truth_contract_id"):
        raise ContractError("U0 truth contract_id mismatch")
    requirements = contract.get("truth_requirements")
    if not isinstance(requirements, dict):
        raise ContractError("U0 contract lacks truth_requirements")
    if truth_config.get("collection_scope") != requirements.get("collection_scope", "official_full_matrix"):
        raise ContractError("U0 truth config must be the official_full_matrix scope")
    if requirements.get("hash_contract") != {
        "json": "utf8_canonical_sorted_keys_compact",
        "validator_text": "utf8_lf_normalized",
    }:
        raise ContractError("U0 truth hash contract mismatch")
    if truth_config_sha256 != requirements.get("official_truth_config_sha256"):
        raise ContractError("U0 truth config SHA does not match the preregistered official config")
    if normalized_text_sha256(_VALIDATOR_PATH) != requirements.get("truth_validator_implementation_sha256"):
        raise ContractError("U0 truth validator implementation SHA mismatch")
    dependency_hashes = requirements.get("truth_validator_dependency_sha256")
    expected_dependencies = {
        "validate_explicit_route_intent_episode.py",
        "validate_ustrf_sc_capture_frame_ledger.py",
        "validate_ai_review_receipt.py",
    }
    if not isinstance(dependency_hashes, dict) or set(dependency_hashes) != expected_dependencies:
        raise ContractError("U0 truth validator dependency hash inventory mismatch")
    for filename, expected_hash in dependency_hashes.items():
        if normalized_text_sha256(Path(__file__).with_name(filename)) != expected_hash:
            raise ContractError(f"U0 truth validator dependency SHA mismatch: {filename}")
    if truth_manifest.get("collection_scope") != "official_full_matrix" or truth_manifest.get("pilot") is not False:
        raise ContractError("U0 truth manifest must be official_full_matrix with pilot=false")
    for key in ("training_eligible", "android_runtime_change_authorized", "production_model_replacement_authorized"):
        if truth_manifest.get(key) is not False:
            raise ContractError(f"U0 truth manifest must declare {key}=false")
    try:
        truth_gate = _TRUTH.validate(truth_config, truth_manifest, root=truth_root, require_complete=True)
    except (ValueError, KeyError, TypeError) as error:
        raise ContractError(f"route-conditioned GPT/Codex truth gate failed: {error}") from error
    if truth_gate.get("route_conditioned_truth_eligible") is not True:
        raise ContractError("route-conditioned GPT/Codex consensus truth is not eligible for U0")
    for key in ("episode_count", "matched_pair_count", "route_bound_episode_count"):
        if truth_gate.get(key) != requirements.get(key):
            raise ContractError(f"truth gate {key} does not meet the preregistered U0 denominator")
    if truth_gate.get("training_eligible") is not False or truth_gate.get("production_model_replacement_authorized") is not False:
        raise ContractError("U0 truth authority must remain evaluation-only")
    episode_ids = [row.get("episode_id") for row in truth_manifest["episodes"]]
    risk_event_ids = [row.get("risk_event_id") for row in truth_manifest["episodes"]]
    if len(set(episode_ids)) != len(episode_ids) or len(set(risk_event_ids)) != len(risk_event_ids):
        raise ContractError("truth episode_id and risk_event_id values must be unique")
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth_manifest["episodes"]:
        pairs[row["matched_pair_id"]].append(row)
    for pair_id, members in pairs.items():
        route_semantics = []
        for member in members:
            path = (truth_root / member["route_intent_path"]).resolve()
            route = json.loads(path.read_text(encoding="utf-8"))
            provider = route.get("provider")
            if not isinstance(provider, dict):
                raise ContractError(f"matched_pair {pair_id} route provider is invalid")
            route_semantics.append((
                route.get("route_plan_id"),
                {key: provider.get(key) for key in ("type", "provider_id", "implementation_sha256", "config_sha256", "input_space")},
                member.get("capture_context", {}).get("route_choice"),
            ))
        if len(route_semantics) != 2 or route_semantics[0] != route_semantics[1]:
            raise ContractError(f"matched_pair {pair_id} does not share route plan, provider policy, and route choice")

    arms, prediction_evidence_gate = _validate_predictions(
        contract,
        truth_manifest,
        predictions,
        truth_root=truth_root,
        prediction_root=prediction_root,
        truth_config_sha256=truth_config_sha256,
        truth_manifest_sha256=truth_manifest_sha256,
    )
    truth_by_id = {row["episode_id"]: row for row in truth_manifest["episodes"]}
    arm_reports: dict[str, Any] = {}
    for arm_id, by_id in arms.items():
        joined = [(truth_by_id[episode_id], by_id[episode_id]) for episode_id in sorted(truth_by_id)]
        sessions: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        scenes: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        session_scenes: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for item in joined:
            sessions[item[0]["session_id"]].append(item)
            scenes[item[0]["scene_id"]].append(item)
            session_scenes[f"{item[0]['session_id']}|{item[0]['scene_id']}"].append(item)
        arm_reports[arm_id] = {
            "aggregate": _metrics(joined),
            "by_session": {key: _metrics(value) for key, value in sorted(sessions.items())},
            "by_scene": {key: _metrics(value) for key, value in sorted(scenes.items())},
            "by_session_scene": {key: _metrics(value) for key, value in sorted(session_scenes.items())},
        }

    causal = arm_reports["teacher_dense_explicit_route_causal"]
    thresholds = contract["evaluation"]["absolute_thresholds_each_fold_for_causal_arm"]
    minimum_coverage = contract["evaluation"]["minimum_non_abstain_coverage_each_fold"]
    fold_checks: dict[str, dict[str, bool]] = {}
    for session_id, metrics in causal["by_session"].items():
        fold_checks[session_id] = {
            "coverage": metrics["non_abstain_coverage"] >= minimum_coverage,
            "event_recall": metrics["event_recall"] >= thresholds["minimum_event_recall"],
            "critical_miss_rate": metrics["critical_miss_rate"] <= thresholds["maximum_critical_miss_rate"],
            "false_alerts_per_minute": metrics["false_alerts_per_minute"] <= thresholds["maximum_false_alerts_per_minute"],
            "delivered_alerts_per_event": metrics["delivered_alerts_per_event"] <= thresholds["maximum_delivered_alerts_per_event"],
            "repeated_alert_rate": metrics["delivered_repeated_alert_rate"] <= thresholds["maximum_delivered_repeated_alert_rate"],
            "post_event_clearance_rate": metrics["post_event_clearance_rate"] >= thresholds["minimum_post_event_clearance_rate"],
            "clearance_latency": metrics["p95_clearance_latency_ms"] <= thresholds["maximum_p95_clearance_latency_ms"],
            "event_regeneration_rate": metrics["event_regeneration_rate"] <= thresholds["maximum_event_regeneration_rate"],
            "critical_denominator": metrics["critical_event_count"] >= requirements["minimum_critical_events_each_fold"],
        }

    mechanism = contract["evaluation"]["mechanism_thresholds_aggregate"]
    dense = arm_reports["teacher_dense_explicit_route"]["aggregate"]
    causal_aggregate = causal["aggregate"]
    detector_route = arm_reports["detector_bbox_explicit_route"]["aggregate"]
    uniform = arm_reports["teacher_dense_uniform_route_control"]["aggregate"]
    shuffled = arm_reports["teacher_dense_shuffled_route_control"]["aggregate"]
    bbox_unknown = max(
        arm_reports["baseline_yolo_geometry"]["by_scene"]["unknown_low_obstacle"]["event_recall"],
        arm_reports["detector_bbox_explicit_route"]["by_scene"]["unknown_low_obstacle"]["event_recall"],
    )
    dense_unknown = arm_reports["teacher_dense_explicit_route"]["by_scene"]["unknown_low_obstacle"]["event_recall"]
    unknown_session_gains = 0
    for session_id in causal["by_session"]:
        key = f"{session_id}|unknown_low_obstacle"
        best_bbox = max(
            arm_reports["baseline_yolo_geometry"]["by_session_scene"][key]["event_recall"],
            arm_reports["detector_bbox_explicit_route"]["by_session_scene"][key]["event_recall"],
        )
        if arm_reports["teacher_dense_explicit_route"]["by_session_scene"][key]["event_recall"] - best_bbox >= mechanism["minimum_unknown_low_obstacle_recall_gain_vs_best_bbox_arm"]:
            unknown_session_gains += 1
    mechanism_checks = {
        "dense_gain_vs_detector_route": dense["matched_pair_balanced_accuracy"] - detector_route["matched_pair_balanced_accuracy"] >= mechanism["minimum_dense_route_balanced_accuracy_gain_vs_detector_route"],
        "route_gain_vs_uniform": dense["matched_pair_balanced_accuracy"] - uniform["matched_pair_balanced_accuracy"] >= mechanism["minimum_explicit_route_balanced_accuracy_gain_vs_uniform_control"],
        "route_gain_vs_shuffled": dense["matched_pair_balanced_accuracy"] - shuffled["matched_pair_balanced_accuracy"] >= mechanism["minimum_explicit_route_balanced_accuracy_gain_vs_shuffled_control"],
        "causal_recall_preserved": dense["event_recall"] - causal_aggregate["event_recall"] <= mechanism["maximum_causal_event_recall_drop_vs_dense_route"],
        "causal_critical_miss_not_increased": causal_aggregate["critical_miss_rate"] <= dense["critical_miss_rate"],
        "causal_false_alerts_not_increased": causal_aggregate["false_alerts_per_minute"] <= dense["false_alerts_per_minute"],
        "causal_repeats_not_increased": causal_aggregate["delivered_repeated_alert_rate"] <= dense["delivered_repeated_alert_rate"],
        "causal_clearance_not_decreased": causal_aggregate["post_event_clearance_rate"] >= dense["post_event_clearance_rate"],
        "unknown_low_obstacle_gain": dense_unknown - bbox_unknown >= mechanism["minimum_unknown_low_obstacle_recall_gain_vs_best_bbox_arm"],
        "unknown_low_obstacle_gain_across_sessions": unknown_session_gains >= mechanism["minimum_sessions_with_unknown_low_obstacle_gain"],
    }
    route_gain_by_session = all(
        arm_reports["teacher_dense_explicit_route"]["by_session"][session_id]["matched_pair_balanced_accuracy"]
        - arm_reports[control]["by_session"][session_id]["matched_pair_balanced_accuracy"]
        >= mechanism[threshold]
        for session_id in causal["by_session"]
        for control, threshold in (
            ("teacher_dense_uniform_route_control", "minimum_explicit_route_balanced_accuracy_gain_vs_uniform_control"),
            ("teacher_dense_shuffled_route_control", "minimum_explicit_route_balanced_accuracy_gain_vs_shuffled_control"),
        )
    )
    bbox_aggregates = [
        arm_reports["baseline_yolo_geometry"]["aggregate"],
        arm_reports["detector_bbox_explicit_route"]["aggregate"],
    ]
    mechanism_checks.update({
        "route_control_gain_each_session": route_gain_by_session,
        "causal_critical_miss_not_worse_than_bbox": all(causal_aggregate["critical_miss_rate"] <= row["critical_miss_rate"] for row in bbox_aggregates),
        "causal_false_alerts_not_worse_than_bbox": all(causal_aggregate["false_alerts_per_minute"] <= row["false_alerts_per_minute"] for row in bbox_aggregates),
        "causal_repeats_not_worse_than_bbox": all(causal_aggregate["delivered_repeated_alert_rate"] <= row["delivered_repeated_alert_rate"] for row in bbox_aggregates),
        "causal_clearance_not_worse_than_bbox": all(causal_aggregate["post_event_clearance_rate"] >= row["post_event_clearance_rate"] for row in bbox_aggregates),
        "causal_unknown_low_obstacle_gain": causal["by_scene"]["unknown_low_obstacle"]["event_recall"] - bbox_unknown >= mechanism["minimum_unknown_low_obstacle_recall_gain_vs_best_bbox_arm"],
    })
    checks_passed = all(all(checks.values()) for checks in fold_checks.values()) and all(mechanism_checks.values())
    synthetic_allowed = contract["authority"].get("synthetic_fixture_can_authorize_u0") is True
    u0_passed = bool(checks_passed and (not predictions["synthetic_fixture"] or synthetic_allowed))
    failed_checks = [
        f"fold:{session_id}:{name}"
        for session_id, checks in fold_checks.items()
        for name, passed in checks.items()
        if not passed
    ] + [f"mechanism:{name}" for name, passed in mechanism_checks.items() if not passed]
    if predictions["synthetic_fixture"] and not synthetic_allowed:
        failed_checks.append("authority:synthetic_fixture_cannot_authorize_u0")
    return {
        "schema": SCHEMA,
        "contract_id": contract["contract_id"],
        "truth_config_sha256": truth_config_sha256,
        "truth_manifest_sha256": truth_manifest_sha256,
        "truth_gate": truth_gate,
        "prediction_evidence_gate": prediction_evidence_gate,
        "arms": arm_reports,
        "fold_checks": fold_checks,
        "mechanism_checks": mechanism_checks,
        "u0_passed": u0_passed,
        "failed_checks": failed_checks,
        "decision": "GO_TO_S0_RECOMMENDED" if u0_passed else "NO_GO_UPPER_BOUND_INSUFFICIENT",
        "s0_probe_eligible": bool(u0_passed and contract["authority"]["u0_pass_can_authorize_s0_probe"]),
        "student_training_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--truth-config", required=True, type=Path)
    parser.add_argument("--truth-manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(
            _load(args.contract),
            _load(args.truth_config),
            _load(args.truth_manifest),
            _load(args.predictions),
            truth_root=args.truth_manifest.resolve().parent,
            prediction_root=args.predictions.resolve().parent,
            truth_config_sha256=canonical_json_sha256(args.truth_config),
            truth_manifest_sha256=canonical_json_sha256(args.truth_manifest),
        )
    except (ContractError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "decision": report["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
