#!/usr/bin/env python3
"""Evaluate a synthetic teacher-derived continuous-event HFTF proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_stage_c_d5_tartanground_development_student import (
    HftfDataset,
    TemporalStudent,
    decode_labels,
    load_jsonl,
    sha256,
)


HORIZONS = ("near", "far")
HEIGHTS = ("body", "head")
DECISION_POLICIES = (
    "hard_known_and_risk",
    "height_temporal_selective_v0",
    "height_temporal_selective_v1",
)


def decision_policy_spec(policy: str) -> dict[str, dict[str, Any]]:
    if policy == "hard_known_and_risk":
        return {
            height: {
                "risk_threshold": 0.5,
                "base_mode": "known_and_risk",
                "known_threshold": 0.5,
                "risk_override_threshold": None,
                "causal_confirmation_steps": 1,
                "confirm_override": True,
            }
            for height in HEIGHTS
        }
    if policy == "height_temporal_selective_v0":
        return {
            "body": {
                "risk_threshold": 0.5,
                "base_mode": "risk_only",
                "known_threshold": None,
                "risk_override_threshold": None,
                "causal_confirmation_steps": 3,
                "confirm_override": True,
            },
            "head": {
                "risk_threshold": 0.5,
                "base_mode": "known_and_risk",
                "known_threshold": 0.5,
                "risk_override_threshold": 0.9,
                "causal_confirmation_steps": 2,
                "confirm_override": True,
            },
        }
    if policy == "height_temporal_selective_v1":
        return {
            "body": {
                "risk_threshold": 0.5,
                "base_mode": "risk_only",
                "known_threshold": None,
                "risk_override_threshold": None,
                "causal_confirmation_steps": 3,
                "confirm_override": True,
            },
            "head": {
                "risk_threshold": 0.5,
                "base_mode": "known_and_risk",
                "known_threshold": 0.5,
                "risk_override_threshold": 0.8,
                "causal_confirmation_steps": 2,
                "confirm_override": False,
            },
        }
    raise ValueError(f"Unknown decision policy: {policy}")


def raw_lane_signals(
    risk_probability: np.ndarray,
    known_probability: np.ndarray,
    height: str,
    decision_policy: str,
) -> tuple[bool, bool]:
    spec = decision_policy_spec(decision_policy)[height]
    base = risk_probability >= spec["risk_threshold"]
    if spec["base_mode"] == "known_and_risk":
        base &= known_probability >= spec["known_threshold"]
    elif spec["base_mode"] != "risk_only":
        raise ValueError(f"Unknown base mode: {spec['base_mode']}")
    override = spec["risk_override_threshold"]
    return (
        bool(np.any(base)),
        bool(np.any(risk_probability >= override))
        if override is not None
        else False,
    )


def raw_lane_active(
    risk_probability: np.ndarray,
    known_probability: np.ndarray,
    height: str,
    decision_policy: str,
) -> bool:
    base, override = raw_lane_signals(
        risk_probability,
        known_probability,
        height,
        decision_policy,
    )
    return base or override


def causal_confirmation(
    values: list[bool],
    steps: int,
) -> list[bool]:
    if steps < 1:
        raise ValueError("Confirmation steps must be positive")
    return [
        index >= steps - 1
        and all(values[index - steps + 1 : index + 1])
        for index in range(len(values))
    ]


def apply_decision_confirmation(
    base_values: list[bool],
    override_values: list[bool],
    spec: dict[str, Any],
) -> list[bool]:
    steps = spec["causal_confirmation_steps"]
    if spec["confirm_override"]:
        return causal_confirmation(
            [
                base or override
                for base, override in zip(
                    base_values,
                    override_values,
                    strict=True,
                )
            ],
            steps,
        )
    confirmed_base = causal_confirmation(base_values, steps)
    return [
        base or override
        for base, override in zip(
            confirmed_base,
            override_values,
            strict=True,
        )
    ]


def lane_truth_state(
    risk: np.ndarray,
    known: np.ndarray,
) -> bool | None:
    mask = known.astype(bool)
    if np.any(mask & (risk >= 0.5)):
        return True
    if np.all(mask):
        return False
    return None


def contiguous_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def trace_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [row["truth"] is not None for row in rows]
    positives = [row["truth"] is True for row in rows]
    negatives = [row["truth"] is False for row in rows]
    active = [bool(row["active"]) for row in rows]
    positive_runs = contiguous_runs(positives)
    false_runs = contiguous_runs(
        [negative and prediction for negative, prediction in zip(
            negatives,
            active,
            strict=True,
        )]
    )
    hits = 0
    response_delays = []
    clearance_eligible = 0
    cleared = 0
    clearance_delays = []
    for start, end in positive_runs:
        active_indices = [
            index for index in range(start, end + 1) if active[index]
        ]
        if active_indices:
            hits += 1
            response_delays.append(active_indices[0] - start)
        next_positive = next(
            (
                index
                for index in range(end + 1, len(rows))
                if positives[index]
            ),
            len(rows),
        )
        negative_indices = [
            index
            for index in range(end + 1, next_positive)
            if negatives[index]
        ]
        if negative_indices:
            clearance_eligible += 1
            inactive = [
                index for index in negative_indices if not active[index]
            ]
            if inactive:
                cleared += 1
                clearance_delays.append(inactive[0] - end)
    positive_frames = sum(positives)
    negative_frames = sum(negatives)
    return {
        "observed_lane_frames": sum(observed),
        "positive_lane_frames": positive_frames,
        "negative_lane_frames": negative_frames,
        "positive_event_count": len(positive_runs),
        "hit_event_count": hits,
        "missed_event_count": len(positive_runs) - hits,
        "event_recall": (
            hits / len(positive_runs) if positive_runs else None
        ),
        "false_active_lane_frames": sum(
            negative and prediction
            for negative, prediction in zip(
                negatives,
                active,
                strict=True,
            )
        ),
        "false_active_lane_frame_rate": (
            sum(
                negative and prediction
                for negative, prediction in zip(
                    negatives,
                    active,
                    strict=True,
                )
            )
            / negative_frames
            if negative_frames
            else None
        ),
        "false_alert_event_count": len(false_runs),
        "clearance_eligible_event_count": clearance_eligible,
        "cleared_event_count": cleared,
        "clearance_rate": (
            cleared / clearance_eligible if clearance_eligible else None
        ),
        "response_delay_anchor_steps": response_delays,
        "clearance_delay_anchor_steps": clearance_delays,
    }


def aggregate_trace_metrics(
    traces: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = [trace_metrics(trace) for trace in traces]
    summed = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "observed_lane_frames",
            "positive_lane_frames",
            "negative_lane_frames",
            "positive_event_count",
            "hit_event_count",
            "missed_event_count",
            "false_active_lane_frames",
            "false_alert_event_count",
            "clearance_eligible_event_count",
            "cleared_event_count",
        )
    }
    response_delays = [
        value
        for row in rows
        for value in row["response_delay_anchor_steps"]
    ]
    clearance_delays = [
        value
        for row in rows
        for value in row["clearance_delay_anchor_steps"]
    ]
    summed.update(
        {
            "event_recall": (
                summed["hit_event_count"]
                / summed["positive_event_count"]
                if summed["positive_event_count"]
                else None
            ),
            "false_active_lane_frame_rate": (
                summed["false_active_lane_frames"]
                / summed["negative_lane_frames"]
                if summed["negative_lane_frames"]
                else None
            ),
            "clearance_rate": (
                summed["cleared_event_count"]
                / summed["clearance_eligible_event_count"]
                if summed["clearance_eligible_event_count"]
                else None
            ),
            "response_delay_anchor_steps_median": (
                float(np.median(response_delays))
                if response_delays
                else None
            ),
            "clearance_delay_anchor_steps_median": (
                float(np.median(clearance_delays))
                if clearance_delays
                else None
            ),
        }
    )
    return summed


def predict(
    records: list[dict[str, Any]],
    checkpoint_path: Path,
    pretrained_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = TemporalStudent(
        pretrained_path,
        architecture=checkpoint.get("architecture", "pooled"),
        temporal_mode=checkpoint.get("temporal_mode", "joint"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    risk_rows = []
    known_rows = []
    loader = DataLoader(
        HftfDataset(records, "single", train=False, seed=0),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    for frames, _, _ in loader:
        with torch.no_grad():
            risk_logits, known_logits = model(frames.to(device))
        risk_rows.append(torch.sigmoid(risk_logits).cpu().numpy())
        known_rows.append(torch.sigmoid(known_logits).cpu().numpy())
    return np.concatenate(risk_rows), np.concatenate(known_rows)


def build_traces(
    records: list[dict[str, Any]],
    risk_probability: np.ndarray,
    known_probability: np.ndarray,
    decision_policy: str = "hard_known_and_risk",
) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    policy_spec = decision_policy_spec(decision_policy)
    traces: dict[
        tuple[str, str, str, int],
        list[dict[str, Any]],
    ] = {}
    for index, record in enumerate(records):
        truth_risk, truth_known = decode_labels(record)
        truth_risk_array = truth_risk.numpy()
        truth_known_array = truth_known.numpy()
        for horizon_index, horizon in enumerate(HORIZONS, start=1):
            for height_index, height in enumerate(HEIGHTS, start=1):
                for direction in range(6):
                    key = (
                        record["environment"],
                        horizon,
                        height,
                        direction,
                    )
                    base_active, override_active = raw_lane_signals(
                        risk_probability[
                            index,
                            horizon_index,
                            height_index,
                            direction,
                        ],
                        known_probability[
                            index,
                            horizon_index,
                            height_index,
                            direction,
                        ],
                        height,
                        decision_policy,
                    )
                    traces.setdefault(key, []).append(
                        {
                            "anchor_frame_id": record["anchor_frame_id"],
                            "truth": lane_truth_state(
                                truth_risk_array[
                                    horizon_index,
                                    height_index,
                                    direction,
                                ],
                                truth_known_array[
                                    horizon_index,
                                    height_index,
                                    direction,
                                ],
                            ),
                            "active": base_active or override_active,
                            "_base_active": base_active,
                            "_override_active": override_active,
                        }
                    )
    for key, rows in traces.items():
        rows.sort(key=lambda row: row["anchor_frame_id"])
        confirmed = apply_decision_confirmation(
            [bool(row["_base_active"]) for row in rows],
            [bool(row["_override_active"]) for row in rows],
            policy_spec[key[2]],
        )
        for row, active in zip(rows, confirmed, strict=True):
            row["active"] = active
            del row["_base_active"]
            del row["_override_active"]
    return traces


def model_metrics(
    records: list[dict[str, Any]],
    risk_probability: np.ndarray,
    known_probability: np.ndarray,
    decision_policy: str = "hard_known_and_risk",
) -> dict[str, Any]:
    traces = build_traces(
        records,
        risk_probability,
        known_probability,
        decision_policy=decision_policy,
    )
    overall = aggregate_trace_metrics(list(traces.values()))
    by_environment = {
        environment: aggregate_trace_metrics(
            [
                rows
                for key, rows in traces.items()
                if key[0] == environment
            ]
        )
        for environment in sorted({key[0] for key in traces})
    }
    by_height = {
        height: aggregate_trace_metrics(
            [rows for key, rows in traces.items() if key[2] == height]
        )
        for height in HEIGHTS
    }
    return {
        "overall": overall,
        "by_environment": by_environment,
        "by_height": by_height,
    }


def comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "event_recall",
        "false_active_lane_frame_rate",
        "clearance_rate",
    )
    output = {
        key: candidate["overall"][key] - reference["overall"][key]
        for key in keys
        if candidate["overall"][key] is not None
        and reference["overall"][key] is not None
    }
    environment_deltas = {
        environment: {
            key: metrics[key]
            - reference["by_environment"][environment][key]
            for key in keys
            if metrics[key] is not None
            and reference["by_environment"][environment][key] is not None
        }
        for environment, metrics in candidate["by_environment"].items()
    }
    output["by_environment"] = environment_deltas
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("NAME", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--role", default="dev")
    parser.add_argument(
        "--decision-policy",
        choices=DECISION_POLICIES,
        default="hard_known_and_risk",
    )
    args = parser.parse_args()

    names = [name for name, _ in args.model]
    if len(names) != len(set(names)):
        parser.error("Model names must be unique")
    if args.reference not in names:
        parser.error("--reference must name one of --model")
    records = [
        record
        for record in load_jsonl(args.samples)
        if record["role"] == args.role
    ]
    records.sort(
        key=lambda record: (
            record["parent_id"],
            record["anchor_frame_id"],
        )
    )
    if not records:
        raise ValueError("No records for requested role")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, checkpoint_text in args.model:
        checkpoint_path = Path(checkpoint_text)
        risk, known = predict(
            records,
            checkpoint_path,
            args.pretrained,
            device,
        )
        models[name] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
            **model_metrics(
                records,
                risk,
                known,
                decision_policy=args.decision_policy,
            ),
        }
        print(
            json.dumps(
                {
                    "model": name,
                    "event_recall": models[name]["overall"][
                        "event_recall"
                    ],
                    "false_active_lane_frame_rate": models[name][
                        "overall"
                    ]["false_active_lane_frame_rate"],
                }
            ),
            flush=True,
        )
    reference = models[args.reference]
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d5_tartanground_"
            "synthetic_event_proxy_v1"
        ),
        "status": "DEVELOPMENT_SYNTHETIC_EVENT_PROXY_COMPLETE",
        "policy": {
            "outcome_open": True,
            "repairable": True,
            "human_event_truth": False,
            "system_claim": False,
            "promotion_evidence": False,
        },
        "definition": {
            "unit": "environment_x_horizon_x_height_x_direction_lane",
            "positive": (
                "any teacher-known distance cell with risk_score >= 0.5"
            ),
            "negative": (
                "all six distance cells teacher-known and none positive"
            ),
            "unknown": "otherwise",
            "candidate_active": (
                decision_policy_spec(args.decision_policy)
            ),
            "decision_policy": args.decision_policy,
            "anchor_period_s": 0.2,
        },
        "samples_path": str(args.samples.resolve()),
        "samples_sha256": sha256(args.samples),
        "sample_count": len(records),
        "role": args.role,
        "reference": args.reference,
        "models": models,
        "comparisons": {
            name: comparison(metrics, reference)
            for name, metrics in models.items()
            if name != args.reference
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
