"""Pool the frozen DTR-R3 retrospective ceilings and apply the R3 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from dtr_r0 import Arm
from dtr_r3 import R3Arm


SCHEMA = "dtr-r3-cross-source-summary-v1"
R2 = Arm.E_R2_GUARDED_CONSENSUS.value
R3_C = R3Arm.C_CURVED_DISTRIBUTIONAL_GUARDED.value
EXPECTED_R2 = {
    "THOR": {"recalled": 10, "events": 10, "false": 42},
    "JRDB": {"recalled": 164, "events": 175, "false": 256},
    "CODA_CORE": {"recalled": 119, "events": 122, "false": 285},
}
GATE_RECALL = 0.95
GATE_FALSE_REDUCTION = 0.30
HEADLINE_RECALL = 0.97
HEADLINE_FALSE_REDUCTION = 0.40


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def mean_or_none(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_arms(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    arms = result.get("pooled") or result.get("pooled_arms")
    if not isinstance(arms, dict):
        raise ValueError("result has no pooled arm metrics")
    return arms


def validate_arms(source: str, arms: dict[str, dict[str, Any]]) -> None:
    required = {R2, *(arm.value for arm in R3Arm)}
    missing = required.difference(arms)
    if missing:
        raise ValueError(f"{source} is missing arms: {sorted(missing)}")


def baseline_check(
    source_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    observed: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []
    for source, expected in EXPECTED_R2.items():
        metrics = source_arms(source_results[source])[R2]
        actual = {
            "recalled": int(metrics["critical_events_recalled"]),
            "events": int(metrics["critical_events"]),
            "false": int(metrics["false_alert_segments"]),
        }
        observed[source] = actual
        if actual != expected:
            mismatches.append(
                {"source": source, "expected": expected, "observed": actual}
            )
    totals = {
        key: sum(item[key] for item in observed.values())
        for key in ("recalled", "events", "false")
    }
    return {
        "status": "EXACT_REPRODUCTION" if not mismatches else "MISMATCH",
        "expected_by_source": EXPECTED_R2,
        "observed_by_source": observed,
        "mismatches": mismatches,
        "pooled": {
            **totals,
            "recall": ratio(totals["recalled"], totals["events"]),
        },
    }


def pool_arm(
    source_results: dict[str, dict[str, Any]],
    arm: str,
    sources: Sequence[str],
) -> dict[str, Any]:
    metrics = {
        source: source_arms(source_results[source])[arm] for source in sources
    }

    def total(field: str) -> float:
        return sum(float(item[field]) for item in metrics.values())

    events = total("critical_events")
    recalled = total("critical_events_recalled")
    alerts = total("alert_segments")
    false_alerts = total("false_alert_segments")
    matched = total("event_detection_true_positives")
    eligible_alerts = total("event_detection_evaluable_alert_segments")
    target_minutes = total("evaluated_target_track_minutes")
    negative_minutes = total("known_negative_target_track_minutes")
    rankable = total("frame_auprc_rankable_frames")
    scored = total("frame_auprc_evaluable_frames")
    evaluated_frames = total("evaluated_prediction_frames")
    unknown_frames = total("unknown_prediction_frames")
    fragmented = total("fragmented_events")
    extra_onsets = total("extra_alert_fragments")
    clear_eligible = total("clear_eligible_events")
    cleared = total("cleared_events")
    precision = ratio(matched, eligible_alerts)
    detection_recall = ratio(matched, events)
    f1 = (
        2.0 * precision * detection_recall / (precision + detection_recall)
        if precision is not None
        and detection_recall is not None
        and precision + detection_recall > 0.0
        else 0.0
        if precision == 0.0 and detection_recall == 0.0
        else None
    )
    return {
        "sources": list(sources),
        "critical_events": int(events),
        "critical_events_recalled_legacy_overlap": int(recalled),
        "critical_event_recall_legacy_overlap": ratio(recalled, events),
        "alert_segments": int(alerts),
        "false_alert_segments_legacy_overlap": int(false_alerts),
        "alert_segment_overlap_precision": ratio(alerts - false_alerts, alerts),
        "event_detection_true_positives": int(matched),
        "event_detection_evaluable_alert_segments": int(eligible_alerts),
        "event_detection_precision_onset_matched": precision,
        "event_detection_recall_onset_matched": detection_recall,
        "event_detection_f1_onset_matched": f1,
        "evaluated_target_track_minutes": target_minutes,
        "known_negative_target_track_minutes": negative_minutes,
        "false_alert_segments_per_known_negative_target_track_minute": ratio(
            false_alerts, negative_minutes
        ),
        "fragmented_matched_events": int(fragmented),
        "fragmented_matched_event_rate": ratio(fragmented, matched),
        "extra_onsets": int(extra_onsets),
        "extra_onsets_per_matched_event": ratio(extra_onsets, matched),
        "clear_eligible_matched_events": int(clear_eligible),
        "cleared_matched_events": int(cleared),
        "clear_rate": ratio(cleared, clear_eligible),
        "known_prediction_coverage": ratio(
            evaluated_frames - unknown_frames, evaluated_frames
        ),
        "frame_auprc_score_coverage": ratio(scored, rankable),
        "frame_auprc_by_source": {
            source: item["frame_auprc"] for source, item in metrics.items()
        },
        "frame_auprc_source_macro": mean_or_none(
            [item["frame_auprc"] for item in metrics.values()]
        ),
        "median_first_alert_lead_s_by_source": {
            source: item["median_first_alert_lead_s"]
            for source, item in metrics.items()
        },
        "median_clear_delay_s_by_source": {
            source: item["median_clear_delay_s"]
            for source, item in metrics.items()
        },
    }


def source_gain(
    source_results: dict[str, dict[str, Any]],
    source: str,
    arm: str,
) -> dict[str, Any]:
    arms = source_arms(source_results[source])
    baseline = arms[R2]
    challenger = arms[arm]
    baseline_recall = float(baseline["critical_event_recall"])
    challenger_recall = float(challenger["critical_event_recall"])
    baseline_false = int(baseline["false_alert_segments"])
    challenger_false = int(challenger["false_alert_segments"])
    units = (
        source_results[source].get("by_session")
        or source_results[source].get("by_sequence")
        or source_results[source].get("sequences")
        or []
    )
    unit_rows = []
    for unit in units:
        unit_arms = unit.get("arms", {})
        if arm not in unit_arms or R2 not in unit_arms:
            continue
        r2_unit = unit_arms[R2]
        r3_unit = unit_arms[arm]
        if not r2_unit["critical_events"]:
            continue
        r2_unit_recall = float(r2_unit["critical_event_recall"])
        r3_unit_recall = float(r3_unit["critical_event_recall"])
        unit_rows.append(
            {
                "unit": str(
                    unit.get("source_session_id")
                    or unit.get("file_id")
                    or unit.get("session")
                    or unit.get("sequence")
                    or "unknown"
                ),
                "events": int(r2_unit["critical_events"]),
                "r2_recall": r2_unit_recall,
                "r3_recall": r3_unit_recall,
                "recall_delta": r3_unit_recall - r2_unit_recall,
            }
        )
    return {
        "r2_recall": baseline_recall,
        "r3_recall": challenger_recall,
        "recall_delta": challenger_recall - baseline_recall,
        "r2_false_alert_segments": baseline_false,
        "r3_false_alert_segments": challenger_false,
        "false_alert_reduction_fraction": ratio(
            baseline_false - challenger_false, baseline_false
        ),
        "event_bearing_unit_count": len(unit_rows),
        "event_bearing_unit_macro_r2_recall": mean_or_none(
            [item["r2_recall"] for item in unit_rows]
        ),
        "event_bearing_unit_macro_r3_recall": mean_or_none(
            [item["r3_recall"] for item in unit_rows]
        ),
        "worst_event_bearing_unit_r3_recall": (
            min(item["r3_recall"] for item in unit_rows) if unit_rows else None
        ),
        "worst_event_bearing_unit_recall_delta": (
            min(item["recall_delta"] for item in unit_rows) if unit_rows else None
        ),
    }


def admission_coverage(
    source_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source, result in source_results.items():
        coverage = result.get("coverage") or result.get("totals") or {}
        denominator_field = next(
            (
                field
                for field in ("target_samples", "objects_used", "dynamic_objects_used")
                if field in coverage
            ),
            None,
        )
        denominator = float(coverage[denominator_field]) if denominator_field else 0.0
        admitted = float(
            source_arms(result)[R2]["evaluated_prediction_frames"]
        )
        output[source] = {
            "admitted_evaluator_known_target_frames": int(admitted),
            "source_target_observation_denominator": int(denominator),
            "denominator_field": denominator_field,
            "admission_coverage": ratio(admitted, denominator),
        }
    return output


def arm_summary(
    source_results: dict[str, dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    sources = tuple(source_results)
    pooled = pool_arm(source_results, arm, sources)
    baseline = pool_arm(source_results, R2, sources)
    reductions = {
        source: source_gain(source_results, source, arm) for source in sources
    }
    pooled["r2_comparison"] = {
        "recall_delta": (
            pooled["critical_event_recall_legacy_overlap"]
            - baseline["critical_event_recall_legacy_overlap"]
        ),
        "false_alert_segment_reduction_fraction": ratio(
            baseline["false_alert_segments_legacy_overlap"]
            - pooled["false_alert_segments_legacy_overlap"],
            baseline["false_alert_segments_legacy_overlap"],
        ),
        "by_source": reductions,
        "source_macro_false_alert_reduction_fraction": mean_or_none(
            [item["false_alert_reduction_fraction"] for item in reductions.values()]
        ),
        "worst_source_recall_delta": min(
            item["recall_delta"] for item in reductions.values()
        ),
    }
    return pooled


def gate(
    baseline: dict[str, Any],
    champion: dict[str, Any],
) -> dict[str, Any]:
    if baseline["status"] != "EXACT_REPRODUCTION":
        return {
            "status": "INVALID_BASELINE_MISMATCH",
            "r3_passed": False,
            "headline_passed": False,
            "authorize_r4": False,
        }
    recall = champion["critical_event_recall_legacy_overlap"]
    reduction = champion["r2_comparison"][
        "false_alert_segment_reduction_fraction"
    ]
    r3_passed = bool(
        recall is not None
        and reduction is not None
        and recall >= GATE_RECALL
        and reduction >= GATE_FALSE_REDUCTION
    )
    headline_passed = bool(
        recall is not None
        and reduction is not None
        and recall >= HEADLINE_RECALL
        and reduction >= HEADLINE_FALSE_REDUCTION
    )
    return {
        "status": "R3_GATE_PASSED" if r3_passed else "R3_GATE_NOT_MET",
        "champion_arm": R3_C,
        "observed_recall": recall,
        "observed_false_alert_reduction_fraction": reduction,
        "required_recall": GATE_RECALL,
        "required_false_alert_reduction_fraction": GATE_FALSE_REDUCTION,
        "r3_passed": r3_passed,
        "headline_required_recall": HEADLINE_RECALL,
        "headline_required_false_alert_reduction_fraction": (
            HEADLINE_FALSE_REDUCTION
        ),
        "headline_passed": headline_passed,
        "authorize_r4": r3_passed,
    }


def extension_summary(result: dict[str, Any]) -> dict[str, Any]:
    arms = source_arms(result)
    return {
        "outside_frozen_r3_gate": True,
        "critical_events": int(result["totals"]["critical_events"]),
        "sequences": [item["sequence"] for item in result["sequences"]],
        "by_arm_and_object_group": {
            arm: metrics.get("by_object_group", {}) for arm, metrics in arms.items()
        },
        "limitations": (
            "Natural source-native positive availability extension; two events do "
            "not establish class-level generalization."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thor", type=Path, required=True)
    parser.add_argument("--jrdb", type=Path, required=True)
    parser.add_argument("--coda-core", type=Path, required=True)
    parser.add_argument("--coda-extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "THOR": args.thor,
        "JRDB": args.jrdb,
        "CODA_CORE": args.coda_core,
    }
    source_results = {source: load(path) for source, path in paths.items()}
    for source, result in source_results.items():
        validate_arms(source, source_arms(result))

    baseline = baseline_check(source_results)
    arms = {
        arm.value: arm_summary(source_results, arm.value) for arm in R3Arm
    }
    curved_sources = ("THOR", "CODA_CORE")
    curve_authority = {
        arm.value: {
            "r2": pool_arm(source_results, R2, curved_sources),
            "r3": pool_arm(source_results, arm.value, curved_sources),
        }
        for arm in (
            R3Arm.A_CURVED_ROBUST_CV,
            R3Arm.C_CURVED_DISTRIBUTIONAL_GUARDED,
        )
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "input_results": {
            source: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for source, path in paths.items()
        },
        "baseline_reproduction": baseline,
        "r2_pooled_metrics": pool_arm(
            source_results,
            R2,
            tuple(source_results),
        ),
        "r3_arms": arms,
        "r3_c_gate": gate(baseline, arms[R3_C]),
        "curved_route_authority_subset": {
            "sources": list(curved_sources),
            "excluded": {
                "JRDB": "NOT_EVALUABLE_CURVED_ROUTE_AUTHORITY_ABSENT"
            },
            "arms": curve_authority,
        },
        "evaluator_admission_coverage": admission_coverage(source_results),
        "metric_semantics": {
            "frozen_gate": (
                "legacy source-pooled target-track event overlap recall and false "
                "segment count, retained only to compare exactly with R2"
            ),
            "event_precision_f1": "one-to-one ONSET in event-start-to-contact window",
            "false_rate": "per known-negative target-track minute",
            "user_wall_clock_false_alerts_per_minute": "NOT_EVALUABLE_TARGET_STREAMS_NOT_MERGED",
            "auprc": (
                "per-source descriptive frame ranking on score-known frames; source "
                "macro is not a pooled PR curve"
            ),
            "ablation": (
                "coupled-arm performance comparison; no single-component causal attribution"
            ),
        },
        "claim_ceiling": (
            "RETROSPECTIVE_PUBLIC_REAL_PRIVILEGED_TRACK_POSE_CEILING_ONLY"
        ),
    }
    if args.coda_extension is not None:
        extension = load(args.coda_extension)
        result["input_results"]["CODA_EXTENSION"] = {
            "path": str(args.coda_extension.resolve()),
            "sha256": sha256_file(args.coda_extension),
        }
        result["coda_multiclass_extension"] = extension_summary(extension)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["r3_c_gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
