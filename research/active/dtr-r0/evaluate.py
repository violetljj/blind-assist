"""Evaluate all four DTR-R0 arms on a JSONL episode file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dtr_r0 import Arm, DTRConfig, compute_event_metrics, frame_from_dict, run_arm


CLAIM_CEILING = "CONTROLLED_SYNTHETIC_MECHANICS_ONLY"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("schema_version") != "dtr-r0-episode-v1":
                raise ValueError(f"unsupported schema on line {line_number}")
            if value.get("mechanism_smoke_only") is not True:
                raise ValueError(
                    "this evaluator is sealed to mechanism-smoke inputs; "
                    "a controlled cohort needs a new frozen evaluator version"
                )
            episodes.append(value)
    if not episodes:
        raise ValueError("input contains no episodes")
    ids = [str(episode["episode_id"]) for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episode ids must be unique")
    return episodes


def _difference(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return None
    return left - right


def evaluate_episodes(
    episodes: list[dict[str, Any]], config: DTRConfig | None = None
) -> dict[str, Any]:
    config = config or DTRConfig()
    arm_metrics: dict[str, dict[str, Any]] = {}
    for arm in Arm:
        episode_predictions = {
            str(episode["episode_id"]): run_arm(
                (frame_from_dict(frame) for frame in episode["frames"]),
                arm,
                config,
            )
            for episode in episodes
        }
        arm_metrics[arm.value] = compute_event_metrics(
            episodes,
            episode_predictions,
            clear_grace_s=config.clear_grace_s,
        )

    baseline = arm_metrics[Arm.B0_DETECTION.value]
    challenger = arm_metrics[Arm.C_ROUTE_INTERSECTION.value]
    baseline_irrelevant = baseline["irrelevant_alert_segments"]
    irrelevant_reduction = None
    if baseline_irrelevant:
        irrelevant_reduction = (
            baseline_irrelevant - challenger["irrelevant_alert_segments"]
        ) / baseline_irrelevant

    return {
        "schema_version": "dtr-r0-evaluation-v1",
        "claim_ceiling": CLAIM_CEILING,
        "result_status": "NO_SCIENTIFIC_RESULT",
        "gate_evaluation": {
            "status": "EXCLUDED_SYNTHETIC_MECHANICS_SMOKE",
            "controlled_event_cohort_status": "CONTROLLED_EVENT_COHORT_PENDING",
            "reason": (
                "Synthetic episodes validate mechanics only and are excluded from "
                "the DTR-R0 scientific advancement gate."
            ),
            "future_controlled_gate": {
                "comparison_rule_status": "MUST_FREEZE_BEFORE_CONTROLLED_OUTCOME",
                "comparator_pool": [arm.value for arm in Arm if arm is not Arm.C_ROUTE_INTERSECTION],
                "require_critical_event_recall_non_decrease": True,
                "minimum_irrelevant_alert_reduction_fraction": 0.40,
                "minimum_median_first_alert_lead_s": 1.0,
                "maximum_mean_alert_segments_per_event": 1.5,
                "require_stable_clear_after_route_exit": True,
                "target_cohort_size_events": "100-160",
                "require_six_scene_classes_balanced": True,
            },
        },
        "arms": arm_metrics,
        "synthetic_diagnostics_not_a_gate_verdict": {
            "challenger_critical_event_recall_delta_vs_b0": _difference(
                challenger["critical_event_recall"], baseline["critical_event_recall"]
            ),
            "challenger_irrelevant_alert_reduction_fraction_vs_b0": irrelevant_reduction,
            "challenger_median_first_alert_lead_s": challenger[
                "median_first_alert_lead_s"
            ],
            "challenger_mean_alert_segments_per_critical_event": challenger[
                "mean_alert_segments_per_critical_event"
            ],
            "challenger_stable_clear_rate": challenger["stable_clear_rate"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_episodes(read_jsonl(args.input))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "claim_ceiling": CLAIM_CEILING,
                    "result_status": "NO_SCIENTIFIC_RESULT",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
