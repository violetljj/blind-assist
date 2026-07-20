"""Probe pair-relative lifecycle transitions without an absolute risk threshold.

The mechanism-specific episode scores are frozen inputs from the preceding
temporal-range probe.  Within each qualified same-source counterfactual pair,
the earlier and later episodes are ordered by their SHA-bound frame indices.
An increasing score opens an event and a decreasing score closes it.

This is a provisional lifecycle-change diagnostic.  It requires a trusted
recent reference state and cannot authorize calibration, blind evaluation,
Android integration, or production promotion.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_pair_relative_lifecycle_probe_v1"
MECHANISM_SCHEMA = "blindassist_public_silver_mechanism_coverage_v1"
TEMPORAL_SCHEMA = "blindassist_public_silver_mechanism_temporal_range_probe_v1"
OPEN_EVENT = "open_event"
CLOSE_EVENT = "close_event"
ABSTAIN = "abstain"


def _qualified_pair_contract(mechanism_report: dict[str, Any]) -> dict[str, list[str]]:
    if mechanism_report.get("schema") != MECHANISM_SCHEMA:
        raise ValueError("unexpected mechanism coverage schema")
    gate = mechanism_report.get("mechanism_coverage_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("mechanism coverage gate is not passed")
    isolation = mechanism_report.get("isolation_contract")
    if not isinstance(isolation, dict) or isolation.get("independent_model_direction_data_used") is not False:
        raise ValueError("mechanism report does not preserve independent-direction isolation")

    contract: dict[str, list[str]] = {}
    coverage = mechanism_report.get("coverage")
    if not isinstance(coverage, dict) or not coverage:
        raise ValueError("mechanism coverage is missing")
    for mechanism in mechanism_report.get("required_mechanisms", []):
        row = coverage.get(mechanism)
        pair_ids = row.get("counterfactual_pair_ids") if isinstance(row, dict) else None
        if not isinstance(pair_ids, list) or not pair_ids or not all(isinstance(value, str) for value in pair_ids):
            raise ValueError(f"qualified pair IDs are missing for mechanism: {mechanism}")
        contract[str(mechanism)] = list(pair_ids)
    if not contract:
        raise ValueError("no required mechanism was declared")
    return contract


def _score_rows(temporal_report: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    if temporal_report.get("schema") != TEMPORAL_SCHEMA:
        raise ValueError("unexpected temporal range schema")
    isolation = temporal_report.get("isolation_contract")
    if not isinstance(isolation, dict) or any(
        isolation.get(key) is not False
        for key in (
            "independent_model_direction_data_used",
            "independent_model_direction_code_used",
            "independent_model_direction_metrics_used_as_gate",
        )
    ):
        raise ValueError("temporal report does not preserve independent-direction isolation")
    pair_scores = temporal_report.get("pair_scores")
    if not isinstance(pair_scores, dict):
        raise ValueError("temporal pair scores are missing")
    indexed: dict[str, tuple[str, dict[str, Any]]] = {}
    for mechanism, rows in pair_scores.items():
        if not isinstance(rows, list):
            raise ValueError(f"temporal pair scores must be a list: {mechanism}")
        for row in rows:
            pair_id = row.get("counterfactual_pair_id") if isinstance(row, dict) else None
            if not isinstance(pair_id, str) or pair_id in indexed:
                raise ValueError("temporal pair IDs must be unique non-empty strings")
            for key in ("no_alert_score", "alert_score"):
                value = row.get(key)
                if not isinstance(value, (int, float)):
                    raise ValueError(f"temporal score is missing: {pair_id}: {key}")
            indexed[pair_id] = (str(mechanism), row)
    return indexed


def _frame_bounds(episode: dict[str, Any]) -> tuple[int, int]:
    frames = episode.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"episode has no bound frames: {episode.get('episode_id')}")
    indices = [frame.get("frame_index") for frame in frames if isinstance(frame, dict)]
    if len(indices) != len(frames) or not all(isinstance(value, int) for value in indices):
        raise ValueError(f"episode has no authoritative integer frame indices: {episode.get('episode_id')}")
    return min(indices), max(indices)


def _transition_from_delta(delta: float) -> str:
    if delta > 0.0:
        return OPEN_EVENT
    if delta < 0.0:
        return CLOSE_EVENT
    return ABSTAIN


def evaluate(
    *,
    episodes: Sequence[dict[str, Any]],
    mechanism_report: dict[str, Any],
    temporal_report: dict[str, Any],
) -> dict[str, Any]:
    contract = _qualified_pair_contract(mechanism_report)
    indexed_scores = _score_rows(temporal_report)
    temporal_contract = temporal_report.get("qualified_pair_contract")
    if temporal_contract != contract:
        raise ValueError("qualified pair contract differs between mechanism and temporal reports")

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        pair_id = episode.get("counterfactual_pair_id")
        if isinstance(pair_id, str):
            by_pair.setdefault(pair_id, []).append(episode)

    transitions: list[dict[str, Any]] = []
    required_pair_ids = {pair_id for pair_ids in contract.values() for pair_id in pair_ids}
    if set(indexed_scores) != required_pair_ids:
        raise ValueError("temporal score rows do not exactly match the qualified pair contract")

    for mechanism, pair_ids in contract.items():
        for pair_id in pair_ids:
            members = by_pair.get(pair_id, [])
            if len(members) != 2:
                raise ValueError(f"qualified pair must bind exactly two episodes: {pair_id}")
            if {int(row.get("label", -1)) for row in members} != {0, 1}:
                raise ValueError(f"qualified pair must contain one alert and one no-alert episode: {pair_id}")
            source_ids = {row.get("source_id") for row in members}
            if len(source_ids) != 1:
                raise ValueError(f"qualified pair must remain within one source: {pair_id}")

            score_mechanism, score_row = indexed_scores[pair_id]
            if score_mechanism != mechanism:
                raise ValueError(f"pair mechanism differs from the qualified contract: {pair_id}")
            if score_row.get("source_id") not in source_ids:
                raise ValueError(f"pair source differs from the temporal report: {pair_id}")
            score_by_label = {
                0: float(score_row["no_alert_score"]),
                1: float(score_row["alert_score"]),
            }

            ordered = sorted(members, key=lambda row: (_frame_bounds(row)[0], _frame_bounds(row)[1]))
            first_bounds = _frame_bounds(ordered[0])
            second_bounds = _frame_bounds(ordered[1])
            if first_bounds[1] >= second_bounds[0]:
                raise ValueError(f"qualified pair episodes overlap or have ambiguous chronology: {pair_id}")
            first_label = int(ordered[0]["label"])
            second_label = int(ordered[1]["label"])
            expected = OPEN_EVENT if (first_label, second_label) == (0, 1) else CLOSE_EVENT
            delta = score_by_label[second_label] - score_by_label[first_label]
            normalized_margin = abs(delta) / max(
                abs(score_by_label[first_label]), abs(score_by_label[second_label]), 1e-12
            )
            predicted = _transition_from_delta(delta)
            transitions.append({
                "counterfactual_pair_id": pair_id,
                "mechanism": mechanism,
                "source_id": next(iter(source_ids)),
                "earlier_episode_id": ordered[0]["episode_id"],
                "later_episode_id": ordered[1]["episode_id"],
                "earlier_frame_bounds": list(first_bounds),
                "later_frame_bounds": list(second_bounds),
                "earlier_state": "risk" if first_label else "clear",
                "later_state": "risk" if second_label else "clear",
                "earlier_score": score_by_label[first_label],
                "later_score": score_by_label[second_label],
                "signed_score_delta": delta,
                "normalized_absolute_margin": normalized_margin,
                "expected_transition": expected,
                "predicted_transition": predicted,
                "correct": predicted == expected,
            })

    transition_types = {row["expected_transition"] for row in transitions}
    represented_mechanisms = {row["mechanism"] for row in transitions}
    checks = {
        "all_qualified_pairs_evaluated": len(transitions) == len(required_pair_ids),
        "all_pair_relative_transitions_correct": all(row["correct"] for row in transitions),
        "open_event_coverage_present": OPEN_EVENT in transition_types,
        "close_event_coverage_present": CLOSE_EVENT in transition_types,
        "all_required_mechanisms_represented": represented_mechanisms == set(contract),
    }
    stress_thresholds = (0.0, 0.01, 0.02, 0.05, 0.10)
    sensitivity = []
    for threshold in stress_thresholds:
        retained = [row for row in transitions if row["normalized_absolute_margin"] > threshold]
        sensitivity.append({
            "minimum_normalized_margin_exclusive": threshold,
            "retained_transition_count": len(retained),
            "abstain_count": len(transitions) - len(retained),
            "retained_transition_accuracy": (
                sum(bool(row["correct"]) for row in retained) / len(retained)
                if retained else None
            ),
            "all_qualified_pairs_retained": len(retained) == len(transitions),
        })
    return {
        "qualified_pair_contract": contract,
        "score_contract": {
            "prediction": "sign(later_episode_score - earlier_episode_score)",
            "positive": OPEN_EVENT,
            "negative": CLOSE_EVENT,
            "zero": ABSTAIN,
            "absolute_threshold_used": False,
            "trainable_parameters": 0,
            "chronology_source": "SHA-bound source frame_index only",
        },
        "transitions": transitions,
        "metrics": {
            "qualified_pair_count": len(transitions),
            "correct_transition_count": sum(bool(row["correct"]) for row in transitions),
            "transition_accuracy": sum(bool(row["correct"]) for row in transitions) / len(transitions),
            "open_event_count": sum(row["expected_transition"] == OPEN_EVENT for row in transitions),
            "close_event_count": sum(row["expected_transition"] == CLOSE_EVENT for row in transitions),
            "minimum_normalized_absolute_margin": min(
                row["normalized_absolute_margin"] for row in transitions
            ),
        },
        "margin_sensitivity": sensitivity,
        "robustness_interpretation": "The raw sign test is threshold-free. Margin rows are post-result stress diagnostics only; they are not calibrated acceptance thresholds.",
        "acceptance": {**checks, "passed": all(checks.values())},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    mil.reject_independent_direction(package_root)
    mechanism_path = args.mechanism_report.resolve()
    temporal_path = args.temporal_report.resolve()
    mechanism_report = lifecycle.verify_json_sidecar(mechanism_path)
    temporal_report = lifecycle.verify_json_sidecar(temporal_path)
    for label, report in (("mechanism", mechanism_report), ("temporal", temporal_report)):
        bound_root = report.get("package_root")
        if not isinstance(bound_root, str) or Path(bound_root).resolve() != package_root:
            raise ValueError(f"{label} report package root differs from the supplied package root")
    if temporal_report.get("mechanism_report_sha256") != common.sha256_file(mechanism_path):
        raise ValueError("temporal report is not bound to the supplied mechanism report")
    episodes, excluded = common.load_episode_specs(package_root)
    result = evaluate(
        episodes=episodes,
        mechanism_report=mechanism_report,
        temporal_report=temporal_report,
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "inputs": {
            "mechanism_report": {"path": str(mechanism_path), "sha256": common.sha256_file(mechanism_path)},
            "temporal_report": {"path": str(temporal_path), "sha256": common.sha256_file(temporal_path)},
        },
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        **result,
        "runtime_precondition": "A recent trusted reference state is required; arbitrary-frame cold start is not solved.",
        "isolation_contract": {
            "public_video_mainline_only": True,
            "independent_model_direction_data_used": False,
            "independent_model_direction_code_used": False,
            "independent_model_direction_metrics_used_as_gate": False,
        },
        "evidence_limit": "Six qualified GPT/VLM provisional same-source pairs only; lifecycle-change feasibility, not human truth, calibration, blind evaluation, Android integration, or production promotion.",
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_integration_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output or sidecar: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--temporal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
