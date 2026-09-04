"""Normalize existing DTR evidence before the full Baseline Reckoning run.

This is a read-only audit. It deliberately keeps CARLA detector evidence,
JRDB native-track ceilings, and JRDB detector-derived X21 in separate panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "blindassist-dtr-baseline-reckoning-readiness-v1"
REPO = Path(__file__).resolve().parents[3]
DEFAULT_CARLA = (
    REPO
    / "artifacts.local"
    / "evidence"
    / "dtr-carla-x95-consumed-cross-validation"
    / "x95-v2-cv-20260901-211602"
    / "summary.json"
)
DEFAULT_JRDB_NATIVE = (
    REPO / "artifacts.local" / "evidence" / "dtr-r3" / "jrdb-test" / "result.json"
)
DEFAULT_JRDB_X21 = (
    REPO
    / "artifacts.local"
    / "evidence"
    / "dtr-x21"
    / "track-carried-component-ancestry-replay-20260829"
    / "result.json"
)


def _read(path: Path) -> dict[str, Any]:
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _row(panel: str, arm: str, input_contract: str, evidence_role: str) -> dict[str, Any]:
    return {
        "panel": panel,
        "arm": arm,
        "input_contract": input_contract,
        "evidence_role": evidence_role,
        "event_precision": None,
        "event_recall": None,
        "event_f1": None,
        "false_segments": None,
        "false_segments_per_minute": None,
        "median_first_alert_lead_s": None,
        "p10_first_alert_lead_s": None,
        "fragmentation_count": None,
        "fragmented_event_rate": None,
        "median_clear_delay_s": None,
        "clear_right_censored_events": None,
        "frame_precision": None,
        "frame_recall": None,
        "frame_f1": None,
    }


def normalize_carla(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema") != "blindassist-dtr-carla-x95-consumed-cross-validation-v1":
        raise ValueError("carla_summary_schema")
    names = {
        "BASELINE_X94_ZERO_POINT_SIX_SECOND_HYSTERESIS": "0.60 s hysteresis over X94 emissions",
        "BASELINE_X94_PLAIN_FULL_DROPOUT_FORWARD_FILL": "plain full-dropout forward fill",
        "X95_LOGISTIC_EMISSION_ONLY": "legacy tiny logistic emission",
        "X94_ISSUED_PLAN_ONE_FRAME_FULL_DROPOUT_CONTINUITY": "X94 complete mechanism",
        "X95_CREDENTIALED_ROUTE_CONDITIONED_HAZARD_STATE": "X95 event challenger",
    }
    rows: list[dict[str, Any]] = []
    for key, metrics in value["aggregate"].items():
        row = _row(
            "CARLA_11_CONSUMED",
            names.get(key, key),
            "X94-derived sealed frame ledger",
            "CONSUMED_POSTHOC_DEVELOPMENT_DIAGNOSTIC",
        )
        row.update(
            event_precision=float(metrics["event_precision"]),
            event_recall=float(metrics["event_recall"]),
            event_f1=float(metrics["event_f1"]),
            false_segments=int(metrics["false_alert_segments"]),
            false_segments_per_minute=float(metrics["false_alert_segments_per_minute"]),
            median_first_alert_lead_s=metrics["median_lead_s"],
            p10_first_alert_lead_s=metrics["p10_lead_s"],
            fragmentation_count=int(metrics["fragment_false_runs"]),
            median_clear_delay_s=metrics["median_clear_latency_s"],
            clear_right_censored_events=int(metrics["clear_latency_censored_events"]),
            frame_precision=float(metrics["precision"]),
            frame_recall=float(metrics["recall"]),
            frame_f1=float(metrics["f1"]),
        )
        rows.append(row)
    return rows


def normalize_jrdb_native(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema_version") != "dtr-r3-jrdb-native-ceiling-v1":
        raise ValueError("jrdb_native_schema")
    names = {
        "B2_radial_ttc": "distance / radial TTC",
        "C_route_intersection": "straight CV route intersection",
        "A_curved_ctrv_robust_target_cv": "CTRV route + robust target CV",
        "E_r2_guarded_occupancy_consensus": "R2 guarded occupancy consensus",
    }
    rows: list[dict[str, Any]] = []
    for key, name in names.items():
        metrics = value["pooled"][key]
        row = _row(
            "JRDB_27_NATIVE_TRACK_CEILING",
            name,
            "privileged native 3-D identity trajectories",
            "PRIVILEGED_INPUT_DIAGNOSTIC_NOT_X21_HEAD_TO_HEAD",
        )
        row.update(
            event_precision=float(metrics["event_detection_precision"]),
            event_recall=float(metrics["event_detection_recall"]),
            event_f1=float(metrics["event_detection_f1"]),
            false_segments=int(metrics["false_alert_segments"]),
            false_segments_per_minute=float(metrics["false_alert_segments_per_target_track_minute"]),
            median_first_alert_lead_s=metrics["median_first_alert_lead_s"],
            fragmentation_count=int(metrics["extra_alert_fragments"]),
            fragmented_event_rate=float(metrics["fragmented_event_rate"]),
            median_clear_delay_s=metrics["median_clear_delay_s"],
            clear_right_censored_events=int(metrics["clear_followup_right_censored_events"]),
        )
        rows.append(row)
    return rows


def normalize_jrdb_x21(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema") != "blindassist-dtr-x21-track-carried-component-ancestry-replay-v1":
        raise ValueError("jrdb_x21_schema")
    rows: list[dict[str, Any]] = []
    for key, metrics in value["metrics"].items():
        hits = int(metrics["contact_recall"])
        events = int(metrics["contact_events"])
        false_segments = int(metrics["false_alert_segments"])
        precision = hits / (hits + false_segments) if hits + false_segments else 0.0
        recall = hits / events if events else 0.0
        computed_f1 = _f1(precision, recall)
        stored_f1 = float(metrics["event_f1"])
        if not math.isclose(computed_f1, stored_f1, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"jrdb_x21_event_f1_contract:{key}")
        row = _row(
            "JRDB_6_DETECTOR_DERIVED",
            key,
            "frozen detector-derived causal occupancy ledger",
            "CONSUMED_DEVELOPMENT",
        )
        row.update(
            event_precision=precision,
            event_recall=recall,
            event_f1=stored_f1,
            false_segments=false_segments,
            median_first_alert_lead_s=metrics["median_first_alert_lead_s"],
        )
        rows.append(row)
    return rows


def build_audit(carla_path: Path, jrdb_native_path: Path, jrdb_x21_path: Path) -> dict[str, Any]:
    carla = _read(carla_path)
    jrdb_native = _read(jrdb_native_path)
    jrdb_x21 = _read(jrdb_x21_path)
    return {
        "schema": SCHEMA,
        "status": "PRELIMINARY_EXISTING_EVIDENCE_ONLY_NOT_BASELINE_RECKONING",
        "rows": [
            *normalize_carla(carla),
            *normalize_jrdb_native(jrdb_native),
            *normalize_jrdb_x21(jrdb_x21),
        ],
        "missing_before_full_reckoning": {
            "carla_same_raw_input_arms": [
                "distance_or_ttc",
                "cv_plus_route_tube",
                "kalman_sort_cv_plus_route_tube_plus_0.60_s_horizon",
                "ctrv_or_constant_turn",
                "independent_tiny_learned_predictor",
                "x24",
                "x73",
            ],
            "jrdb_same_detector_input_arms": [
                "distance_or_ttc",
                "cv_plus_route_tube",
                "kalman_sort_cv_plus_route_tube_plus_0.60_s_horizon",
                "ctrv_if_causal_yaw_rate_is_evaluable",
                "independent_tiny_learned_predictor",
                "x24_x73_x94_not_portable_without_an_explicit_jrdb_adapter",
                "x95_fresh_held_out_confirmation",
            ],
            "legacy_metric_gaps": [
                "JRDB native result does not record frame confusion or frame F1",
                "JRDB X21 result does not record event fragmentation, CLEAR, or frame F1",
            ],
        },
        "comparability": {
            "rank_only_within_panel_and_identical_input_contract": True,
            "do_not_rank_jrdb_native_ceiling_against_detector_x21": True,
            "do_not_promote_from_consumed_carla_or_jrdb": True,
        },
        "sources": {
            "carla_summary": {"path": str(carla_path.resolve()), "sha256": _sha256(carla_path)},
            "jrdb_native": {"path": str(jrdb_native_path.resolve()), "sha256": _sha256(jrdb_native_path)},
            "jrdb_x21": {"path": str(jrdb_x21_path.resolve()), "sha256": _sha256(jrdb_x21_path)},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-summary", type=Path, default=DEFAULT_CARLA)
    parser.add_argument("--jrdb-native-result", type=Path, default=DEFAULT_JRDB_NATIVE)
    parser.add_argument("--jrdb-x21-result", type=Path, default=DEFAULT_JRDB_X21)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = build_audit(args.carla_summary, args.jrdb_native_result, args.jrdb_x21_result)
    encoded = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(encoded, encoding="utf-8")
        print(json.dumps({"status": audit["status"], "output": str(args.output.resolve()), "rows": len(audit["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
