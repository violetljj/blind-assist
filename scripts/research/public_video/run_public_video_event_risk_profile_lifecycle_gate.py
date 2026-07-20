#!/usr/bin/env python3
"""Audit a zero-parameter event risk-profile and lifecycle composition.

Entry requires both a frozen r7.25 radial approach and a positive explicit
obstacle-to-route change.  After entry, the frozen r7.30 chromatic lifecycle
owns persistence and clear.  Segmentation remains an auxiliary relation cue;
it cannot open an event without radial evidence.
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


SCHEMA = "blindassist_public_video_event_risk_profile_lifecycle_gate_v1"
JAPAN_SOURCE = "wikimedia_commons_japan_rural_riverside_walk_2025"
EDMONTON_SOURCE = "youtube_cc_edmonton_city_construction_chaos_pov_2025"


def passed_event_count(source: dict[str, Any]) -> int:
    return sum(event.get("radial_approach_passed") is True for event in source.get("events", []))


def candidate_count(report: dict[str, Any], source_id: str) -> int:
    rows = [row for row in report.get("sources", []) if row.get("source_id") == source_id]
    if len(rows) != 1:
        raise ValueError(f"candidate report must contain one source: {source_id}")
    return passed_event_count(rows[0])


def route_sample(report: dict[str, Any], sample_id: str) -> dict[str, Any]:
    rows = [row for row in report.get("real_video_pressure", []) if row.get("sample_id") == sample_id]
    if len(rows) != 1:
        raise ValueError(f"route report must contain one sample: {sample_id}")
    return rows[0]


def event_row(*, sample_id: str, label: int, radial_count: int, route_delta: float) -> dict[str, Any]:
    radial_entry = radial_count > 0
    route_support = route_delta > 0.0
    prediction = int(radial_entry and route_support)
    return {
        "sample_id": sample_id,
        "label": label,
        "frozen_radial_entry_count": radial_count,
        "explicit_route_relation_delta": route_delta,
        "radial_entry_present": radial_entry,
        "route_relation_supports_entry": route_support,
        "predicted_event_alert": prediction,
        "correct": prediction == label,
    }


def metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["predicted_event_alert"] for row in rows], dtype=np.int64)
    return common.binary_metrics(labels, predictions)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [args.r725_probe, args.edmonton_candidates, args.r730_gap_bridge, args.route_relation, *args.negative_candidates, args.output]
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    r725 = lifecycle.verify_json_sidecar(args.r725_probe)
    edmonton = lifecycle.verify_json_sidecar(args.edmonton_candidates)
    gap_bridge = lifecycle.verify_json_sidecar(args.r730_gap_bridge)
    route = lifecycle.verify_json_sidecar(args.route_relation)
    if route.get("schema") != "blindassist_public_video_explicit_ego_route_relation_probe_v1":
        raise ValueError("unexpected explicit route relation report")

    japan_groups = r725.get("groups", {}).get("prospective_positive_japan", [])
    if len(japan_groups) != 1 or japan_groups[0].get("source_id") != JAPAN_SOURCE:
        raise ValueError("r7.25 Japan positive group is missing")
    positives = [
        event_row(
            sample_id="japan_path_intrusion",
            label=1,
            radial_count=passed_event_count(japan_groups[0]),
            route_delta=float(route_sample(route, "japan_path_intrusion")["marker_minus_clear_intrusion"]),
        ),
        event_row(
            sample_id="edmonton_left_corridor_intrusion",
            label=1,
            radial_count=candidate_count(edmonton, EDMONTON_SOURCE),
            route_delta=float(route_sample(route, "edmonton_left_corridor_intrusion")["marker_minus_clear_intrusion"]),
        ),
    ]

    negative_reports = [lifecycle.verify_json_sidecar(path) for path in args.negative_candidates]
    report_by_source = {}
    negative_inputs = []
    for path, report in zip(args.negative_candidates, negative_reports):
        for source in report.get("sources", []):
            source_id = source.get("source_id")
            if source_id in report_by_source:
                raise ValueError(f"duplicate negative candidate source: {source_id}")
            report_by_source[source_id] = report
        negative_inputs.append({"path": str(path.resolve()), "sha256": common.sha256_file(path)})
    negative_specs = [
        ("jakarta_dense_boundary", "youtube_cc_jakarta_car_free_reopening_2026"),
        ("cape_town_wide_forecourt", "youtube_cc_cape_town_waterfront_construction_walk_2026"),
        ("bramwell_grassy_shoulder_cone", "wikimedia_commons_bramwell_west_virginia_walk_2019"),
        ("dallas_grass_detour_panel", "youtube_cc_boring_dallas_cigarroa_sidewalk_cones_2025"),
        ("dallas_road_edge_cone", "youtube_cc_boring_dallas_cigarroa_sidewalk_cones_2025"),
    ]
    negatives = []
    for sample_id, source_id in negative_specs:
        if source_id not in report_by_source:
            raise ValueError(f"negative candidate report missing source: {source_id}")
        negatives.append(event_row(
            sample_id=sample_id,
            label=0,
            radial_count=candidate_count(report_by_source[source_id], source_id),
            route_delta=float(route_sample(route, sample_id)["marker_minus_clear_intrusion"]),
        ))
    rows = positives + negatives
    metrics = metric_rows(rows)
    selected = gap_bridge.get("selection", {})
    edmonton_lifecycle_passed = bool(
        selected.get("minimum_passing_clear_absent_samples") == 9
        and selected.get("selected_for_future_freeze") is True
    )
    true_radial_negative_count = sum(row["frozen_radial_entry_count"] > 0 for row in negatives)
    diagnostic_pass = bool(metrics["balanced_accuracy"] == 1.0 and edmonton_lifecycle_passed)
    closure_pass = bool(diagnostic_pass and true_radial_negative_count >= 1)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_zero_parameter_event_risk_profile_lifecycle_composition",
        "inputs": {
            "r725_probe_sha256": common.sha256_file(args.r725_probe),
            "edmonton_candidates_sha256": common.sha256_file(args.edmonton_candidates),
            "r730_gap_bridge_sha256": common.sha256_file(args.r730_gap_bridge),
            "explicit_route_relation_sha256": common.sha256_file(args.route_relation),
            "negative_candidate_reports": negative_inputs,
        },
        "architecture": {
            "entry": "frozen r7.25 radial approach AND positive explicit route-relation change from trusted clear reference",
            "persistence": "after entry, frozen chromatic marker evidence resets the absence run without another reminder",
            "uncertain": "marker absence shorter than nine one-second samples",
            "clear": "nine consecutive absent one-second samples",
            "reopen": "new frozen radial approach and route-relation support required",
            "segmentation_role": "auxiliary route relation only; cannot open without radial evidence",
            "pixel_supervision_role": "auxiliary only",
            "learned_parameters": 0,
        },
        "event_rows": rows,
        "event_metrics": metrics,
        "lifecycle_pressure": {
            "edmonton_gap_bridge_passed": edmonton_lifecycle_passed,
            "same_episode_reminder_once": edmonton_lifecycle_passed,
            "japan_full_lifecycle_under_r730_tested": False,
        },
        "hard_missing_evidence": {
            "independent_real_negative_with_true_frozen_radial_entry_required": True,
            "observed_count": true_radial_negative_count,
            "satisfied": true_radial_negative_count >= 1,
            "reason": "Existing reviewed negatives have zero frozen r7.25 entries, so they test entry isolation but cannot test whether route relation vetoes a real radial false entry.",
        },
        "diagnostic_gate": {
            "event_metrics_and_edmonton_lifecycle_passed": diagnostic_pass,
            "full_closure_passed": closure_pass,
        },
        "authorizations": {
            "five_prototype_bootstrap_short_runs": False,
            "future_prospective_contract_freeze": closure_pass,
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
        "evidence_limit": "Perfect current event metrics remain retrospective GPT/VLM silver and do not close the missing true-radial negative stress. No runtime or training authorization follows.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r725-probe", type=Path, required=True)
    parser.add_argument("--edmonton-candidates", type=Path, required=True)
    parser.add_argument("--r730-gap-bridge", type=Path, required=True)
    parser.add_argument("--route-relation", type=Path, required=True)
    parser.add_argument("--negative-candidates", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "event_metrics": value["event_metrics"],
        "diagnostic_passed": value["diagnostic_gate"]["event_metrics_and_edmonton_lifecycle_passed"],
        "full_closure_passed": value["diagnostic_gate"]["full_closure_passed"],
        "output_sha256": common.sha256_file(parsed.output),
    }, ensure_ascii=False))
