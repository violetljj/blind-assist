#!/usr/bin/env python3
"""Evaluate D23 history checkpoints with a zero-dynamics input ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    load_jsonl,
    sha256,
)
from materialize_stage_c_d8_thor_magni_local_route_supervision import (
    FUTURE_HORIZON_SECONDS,
    FUTURE_SAMPLE_SECONDS,
    PROXIMITY_THRESHOLD_M,
    read_scenario,
)
from run_stage_c_d17_tartanground_early_temporal_onset_canary import (
    DEFAULT_PRETRAINED,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_FLOW,
    DEFAULT_RGB_CACHE,
    DEFAULT_SAMPLES,
    ThorDenseFlowDynamicsEncoder,
    predict,
)


SCHEMA = "blindassist_hftf_stage_c_d24_thor_magni_event_ablation_v0"
SEEDS = (17, 23, 41)
FOLDS = tuple(range(5))
FALSE_ACTIVE_CAP = 0.10
EVENT_MAX_GAP_FRAMES = 45
DEFAULT_D8_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d8-thor-magni-local-route-supervision-v0/samples.jsonl"
)
DEFAULT_D22_REPORT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d22-thor-magni-dense-flow-transfer-v0/report.json"
)
DEFAULT_D23_ADDITIONAL = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d23-thor-magni-proximity-multiseed-v0/"
    "additional_seeds_report.json"
)
DEFAULT_D23_REPORT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d23-thor-magni-proximity-multiseed-v0/report.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d24-thor-magni-proximity-event-ablation-v0/report.json"
)
METRICS = (
    "event_auroc",
    "event_average_precision",
    "event_recall_at_false_active_cap",
    "anchor_recall_at_false_active_cap",
    "lead_time_credit_seconds",
    "clearance_rate",
)


def operating_threshold(
    negative_scores: np.ndarray,
    false_active_cap: float = FALSE_ACTIVE_CAP,
) -> float:
    """Return the most permissive strict-greater threshold under the cap."""
    scores = np.asarray(negative_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("D24 operating threshold needs negative scores")
    if not np.isfinite(scores).all():
        raise ValueError("D24 operating threshold received non-finite scores")
    if not 0.0 <= false_active_cap < 1.0:
        raise ValueError("D24 false-active cap must be in [0,1)")
    allowed = int(math.floor(false_active_cap * len(scores) + 1e-12))
    descending = np.sort(scores)[::-1]
    if allowed == 0:
        return float(descending[0])
    return float(descending[allowed])


def positive_event_groups(
    records: list[dict[str, Any]],
) -> list[list[int]]:
    """Group consecutive positive anchors without crossing a missing stride."""
    positives = sorted(
        (
            index
            for index, record in enumerate(records)
            if bool(record["future_onset_target"]["proximity_onset"])
        ),
        key=lambda index: int(records[index]["anchor_scene_frame"]),
    )
    groups: list[list[int]] = []
    for index in positives:
        frame = int(records[index]["anchor_scene_frame"])
        if (
            not groups
            or frame
            - int(records[groups[-1][-1]]["anchor_scene_frame"])
            > EVENT_MAX_GAP_FRAMES
        ):
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def infer_scene_column(path: Path, camera_body: str) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        for _ in range(16):
            next(reader)
        header = next(reader)
    candidates = [
        name
        for name in header
        if name.startswith(f"{camera_body} ")
        and name.endswith("_SceneFNr")
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"D24 expected one scene-frame column for {camera_body}: "
            f"{candidates}"
        )
    return candidates[0]


def first_crossing_offset(
    record: dict[str, Any],
    trajectory_cache: dict[tuple[str, str], dict[str, Any]],
) -> float:
    path = Path(str(record["scenario_csv_path"]))
    camera_body = str(record["camera_body"])
    key = (str(path.resolve()), camera_body)
    data = trajectory_cache.get(key)
    if data is None:
        scene_column = infer_scene_column(path, camera_body)
        data = read_scenario(path, camera_body, scene_column)
        trajectory_cache[key] = data
    matches = np.flatnonzero(
        data["frames"] == int(record["qtm_frame"])
    )
    if len(matches) != 1:
        raise ValueError(
            f"D24 QTM anchor is not unique: {record['sample_id']}"
        )
    index = int(matches[0])
    times = data["times"]
    end_time = times[index] + FUTURE_HORIZON_SECONDS
    future_end = int(np.searchsorted(times, end_time, side="right"))
    local = np.diff(times[max(0, index - 20): index + 21])
    step = max(1, int(round(FUTURE_SAMPLE_SECONDS / np.median(local))))
    for future_index in range(index, future_end, step):
        if not np.isfinite(data["camera"][future_index, :2]).all():
            continue
        minimum = math.inf
        for positions in data["others"].values():
            if not np.isfinite(positions[future_index, :2]).all():
                continue
            relative = (
                positions[future_index, :2]
                - data["camera"][future_index, :2]
            )
            minimum = min(minimum, float(np.linalg.norm(relative)))
        if minimum <= PROXIMITY_THRESHOLD_M:
            return float(times[future_index] - times[index])
    raise ValueError(
        f"D24 positive onset has no reconstructed crossing: "
        f"{record['sample_id']}"
    )


def build_crossing_offsets(
    d12_records: list[dict[str, Any]],
    d8_records: list[dict[str, Any]],
) -> dict[str, float]:
    d8_by_id = {str(record["sample_id"]): record for record in d8_records}
    if len(d8_by_id) != len(d8_records):
        raise ValueError("D24 D8 sample IDs are not unique")
    missing = [
        str(record["sample_id"])
        for record in d12_records
        if str(record["sample_id"]) not in d8_by_id
    ]
    if missing:
        raise ValueError(f"D24 D12-to-D8 join failed: {missing[:3]}")
    trajectory_cache: dict[tuple[str, str], dict[str, Any]] = {}
    offsets = {}
    for record in d12_records:
        target = record["future_onset_target"]
        if not bool(target["proximity_onset"]):
            continue
        sample_id = str(record["sample_id"])
        offsets[sample_id] = first_crossing_offset(
            d8_by_id[sample_id],
            trajectory_cache,
        )
    if any(
        not 0.0 <= value <= FUTURE_HORIZON_SECONDS + 1e-6
        for value in offsets.values()
    ):
        raise ValueError("D24 reconstructed crossing offset is out of range")
    return offsets


def evaluate_arm(
    records: list[dict[str, Any]],
    scores: np.ndarray,
    crossing_offsets: dict[str, float],
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(records),):
        raise ValueError("D24 arm score shape mismatch")
    if not np.isfinite(scores).all():
        raise ValueError("D24 arm scores contain non-finite values")
    sources = sorted(
        {str(record["source_session_id"]) for record in records}
    )
    by_source = []
    for source in sources:
        source_indices = [
            index
            for index, record in enumerate(records)
            if str(record["source_session_id"]) == source
            and bool(
                record["future_onset_target"]["proximity_eligible"]
            )
        ]
        source_records = [records[index] for index in source_indices]
        source_scores = scores[source_indices]
        groups = positive_event_groups(source_records)
        negative_indices = [
            index
            for index, record in enumerate(source_records)
            if not bool(
                record["future_onset_target"]["proximity_onset"]
            )
        ]
        if not groups or not negative_indices:
            continue
        negative_scores = source_scores[negative_indices]
        event_scores = np.asarray(
            [
                max(float(source_scores[index]) for index in group)
                for group in groups
            ],
            dtype=np.float64,
        )
        event_metric = binary_metrics(
            np.concatenate(
                (
                    np.ones(len(event_scores), dtype=np.int64),
                    np.zeros(len(negative_scores), dtype=np.int64),
                )
            ),
            np.concatenate((event_scores, negative_scores)),
        )
        threshold = operating_threshold(negative_scores)
        alerts = source_scores > threshold
        false_active_rate = float(
            np.mean(source_scores[negative_indices] > threshold)
        )
        if false_active_rate > FALSE_ACTIVE_CAP + 1e-12:
            raise ValueError("D24 false-active cap was violated")
        positive_indices = [
            index for group in groups for index in group
        ]
        event_recall = float(
            np.mean(
                [
                    any(bool(alerts[index]) for index in group)
                    for group in groups
                ]
            )
        )
        anchor_recall = float(np.mean(alerts[positive_indices]))
        lead_time_credit = float(
            np.mean(
                [
                    (
                        crossing_offsets[
                            str(source_records[index]["sample_id"])
                        ]
                        if bool(alerts[index])
                        else 0.0
                    )
                    for index in positive_indices
                ]
            )
        )
        clearance = []
        for group in groups:
            end_frame = int(
                source_records[group[-1]]["anchor_scene_frame"]
            )
            subsequent = [
                index
                for index in negative_indices
                if int(source_records[index]["anchor_scene_frame"])
                > end_frame
            ]
            if subsequent:
                first = min(
                    subsequent,
                    key=lambda index: int(
                        source_records[index]["anchor_scene_frame"]
                    ),
                )
                clearance.append(not bool(alerts[first]))
        by_source.append(
            {
                "source_session_id": source,
                "positive_events": len(groups),
                "positive_anchors": len(positive_indices),
                "negative_anchors": len(negative_indices),
                "threshold": threshold,
                "event_auroc": float(event_metric["auroc"]),
                "event_average_precision": float(
                    event_metric["average_precision"]
                ),
                "event_recall_at_false_active_cap": event_recall,
                "anchor_recall_at_false_active_cap": anchor_recall,
                "false_active_rate": false_active_rate,
                "lead_time_credit_seconds": lead_time_credit,
                "clearance_rate": (
                    float(np.mean(clearance)) if clearance else None
                ),
                "clearance_events": len(clearance),
            }
        )
    if not by_source:
        raise ValueError("D24 arm has no evaluable sources")
    source_macro = {}
    for metric in METRICS:
        values = [
            float(row[metric])
            for row in by_source
            if row[metric] is not None
        ]
        source_macro[metric] = float(np.mean(values))
        source_macro[f"{metric}_evaluable_sources"] = len(values)
    source_macro["false_active_rate"] = float(
        np.mean([row["false_active_rate"] for row in by_source])
    )
    source_macro["evaluable_sources"] = len(by_source)
    return {"source_macro": source_macro, "by_source": by_source}


def summarize_delta(
    units: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    values = [
        float(unit["history_minus_zero_dynamics"][metric])
        for unit in units
    ]
    by_seed = [
        float(
            np.mean(
                [
                    unit["history_minus_zero_dynamics"][metric]
                    for unit in units
                    if int(unit["seed"]) == seed
                ]
            )
        )
        for seed in SEEDS
    ]
    by_fold = [
        float(
            np.mean(
                [
                    unit["history_minus_zero_dynamics"][metric]
                    for unit in units
                    if int(unit["fold"]) == fold
                ]
            )
        )
        for fold in FOLDS
    ]
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive_units": int(sum(value > 0 for value in values)),
        "seeds": list(SEEDS),
        "by_seed_mean": by_seed,
        "positive_seeds": int(sum(value > 0 for value in by_seed)),
        "folds": list(FOLDS),
        "by_fold_seed_mean": by_fold,
        "positive_folds": int(sum(value > 0 for value in by_fold)),
    }


def build_gate(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    event_auroc = aggregate["event_auroc"]
    event_recall = aggregate["event_recall_at_false_active_cap"]
    anchor_recall = aggregate["anchor_recall_at_false_active_cap"]
    lead_time = aggregate["lead_time_credit_seconds"]
    checks = {
        "event_auroc_effect": event_auroc["mean"] >= 0.010,
        "event_recall_effect": event_recall["mean"] >= 0.020,
        "event_recall_all_seeds_positive": (
            event_recall["positive_seeds"] == 3
        ),
        "event_recall_positive_folds": (
            event_recall["positive_folds"] >= 3
        ),
        "event_recall_positive_units": (
            event_recall["positive_units"] >= 10
        ),
        "anchor_recall_noninferiority": anchor_recall["mean"] >= -0.010,
        "lead_time_credit_positive": lead_time["mean"] > 0.0,
    }
    return {
        "frozen_thresholds": {
            "false_active_cap": FALSE_ACTIVE_CAP,
            "event_auroc_mean_delta_floor": 0.010,
            "event_recall_mean_delta_floor": 0.020,
            "event_recall_positive_seeds": 3,
            "event_recall_positive_folds": 3,
            "event_recall_positive_units": 10,
            "anchor_recall_mean_delta_floor": -0.010,
            "lead_time_credit_mean_delta_strictly_positive": True,
        },
        "checks": checks,
        "supported": all(checks.values()),
    }


def load_and_validate_checkpoints(
    d22_report_path: Path,
    additional_report_path: Path,
    d23_report_path: Path,
    samples_path: Path,
    rgb_cache_path: Path,
    flow_path: Path,
    pretrained_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d22 = json.loads(d22_report_path.read_text(encoding="utf-8"))
    additional = json.loads(
        additional_report_path.read_text(encoding="utf-8")
    )
    d23 = json.loads(d23_report_path.read_text(encoding="utf-8"))
    if d23["status"] != (
        "D23_THOR_MAGNI_PROXIMITY_MULTI_SEED_ROBUSTNESS_SUPPORTED"
    ):
        raise ValueError("D24 requires the supported D23 terminal")
    bindings = {
        "seed17_report_sha256": sha256(d22_report_path),
        "additional_seeds_report_sha256": sha256(
            additional_report_path
        ),
        "samples_sha256": sha256(samples_path),
        "rgb_cache_sha256": sha256(rgb_cache_path),
        "flow_sha256": sha256(flow_path),
        "pretrained_sha256": sha256(pretrained_path),
    }
    for key, actual in bindings.items():
        if str(d23["inputs"][key]) != actual:
            raise ValueError(f"D24 D23 binding mismatch: {key}")
    checkpoints = [
        *d22["history_checkpoints"],
        *additional["history_checkpoints"],
    ]
    identities = {
        (int(row["fold"]), int(row["seed"])) for row in checkpoints
    }
    if identities != {
        (fold, seed) for fold in FOLDS for seed in SEEDS
    }:
        raise ValueError("D24 requires exact 5 folds x 3 seeds checkpoints")
    for row in checkpoints:
        path = Path(str(row["path"]))
        if not path.is_file() or sha256(path) != str(row["sha256"]):
            raise ValueError(f"D24 checkpoint hash mismatch: {path}")
    return sorted(
        checkpoints,
        key=lambda row: (int(row["fold"]), int(row["seed"])),
    ), d23


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--d8-samples",
        type=Path,
        default=DEFAULT_D8_SAMPLES,
    )
    parser.add_argument(
        "--rgb-cache",
        type=Path,
        default=DEFAULT_RGB_CACHE,
    )
    parser.add_argument("--flow", type=Path, default=DEFAULT_FLOW)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument(
        "--d22-report",
        type=Path,
        default=DEFAULT_D22_REPORT,
    )
    parser.add_argument(
        "--d23-additional",
        type=Path,
        default=DEFAULT_D23_ADDITIONAL,
    )
    parser.add_argument(
        "--d23-report",
        type=Path,
        default=DEFAULT_D23_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    score_path = args.output.with_name("scores.npz")
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    score_sidecar = score_path.with_suffix(score_path.suffix + ".sha256")
    outputs = (args.output, sidecar, score_path, score_sidecar)
    if any(path.exists() for path in outputs):
        raise FileExistsError("D24 outputs are non-overwriting")

    records = load_jsonl(args.samples)
    if len(records) != 1078:
        raise ValueError("D24 requires the exact 1,078 D12 samples")
    for cache_index, record in enumerate(records):
        record["_d22_cache_index"] = cache_index
    d8_records = load_jsonl(args.d8_samples)
    crossing_offsets = build_crossing_offsets(records, d8_records)
    checkpoints, d23 = load_and_validate_checkpoints(
        args.d22_report,
        args.d23_additional,
        args.d23_report,
        args.samples,
        args.rgb_cache,
        args.flow,
        args.pretrained,
    )
    eligible = [
        record
        for record in records
        if bool(record["future_onset_target"]["proximity_eligible"])
    ]
    positive_count = sum(
        bool(record["future_onset_target"]["proximity_onset"])
        for record in eligible
    )
    event_count = sum(
        len(
            positive_event_groups(
                [
                    record
                    for record in eligible
                    if str(record["source_session_id"]) == source
                ]
            )
        )
        for source in sorted(
            {str(record["source_session_id"]) for record in eligible}
        )
    )

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("D24 requires CUDA for frozen checkpoint inference")
    history_scores = np.full(
        (len(SEEDS), len(records)),
        np.nan,
        dtype=np.float32,
    )
    zero_scores = np.full_like(history_scores, np.nan)
    seed_index = {seed: index for index, seed in enumerate(SEEDS)}
    units = []
    for checkpoint_row in checkpoints:
        fold = int(checkpoint_row["fold"])
        seed = int(checkpoint_row["seed"])
        test_indices = [
            index
            for index, record in enumerate(records)
            if int(record["fold"]) == fold
        ]
        test_records = [records[index] for index in test_indices]
        checkpoint = torch.load(
            Path(str(checkpoint_row["path"])),
            map_location="cpu",
            weights_only=True,
        )
        if (
            int(checkpoint["fold"]) != fold
            or int(checkpoint["seed"]) != seed
            or str(checkpoint["arm"]) != "history"
            or str(checkpoint["flow_sha256"])
            != str(d23["inputs"]["flow_sha256"])
        ):
            raise ValueError("D24 checkpoint metadata mismatch")
        model = ThorDenseFlowDynamicsEncoder(args.pretrained).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        history = predict(
            model,
            test_records,
            "history",
            args.rgb_cache,
            args.flow,
            seed,
            device,
        )[:, 0]
        zero = predict(
            model,
            test_records,
            "current",
            args.rgb_cache,
            args.flow,
            seed,
            device,
        )[:, 0]
        history_scores[seed_index[seed], test_indices] = history
        zero_scores[seed_index[seed], test_indices] = zero
        history_metric = evaluate_arm(
            test_records,
            history,
            crossing_offsets,
        )
        zero_metric = evaluate_arm(
            test_records,
            zero,
            crossing_offsets,
        )
        delta = {
            metric: (
                history_metric["source_macro"][metric]
                - zero_metric["source_macro"][metric]
            )
            for metric in METRICS
        }
        units.append(
            {
                "fold": fold,
                "seed": seed,
                "heldout_source_sessions": sorted(
                    {
                        str(record["source_session_id"])
                        for record in test_records
                    }
                ),
                "history": history_metric,
                "zero_dynamics": zero_metric,
                "history_minus_zero_dynamics": delta,
            }
        )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "seed": seed,
                    "event_auroc_delta": delta["event_auroc"],
                    "event_recall_delta": delta[
                        "event_recall_at_false_active_cap"
                    ],
                    "lead_time_credit_delta": delta[
                        "lead_time_credit_seconds"
                    ],
                }
            ),
            flush=True,
        )
        del model, checkpoint
        torch.cuda.empty_cache()
    if not np.isfinite(history_scores).all() or not np.isfinite(
        zero_scores
    ).all():
        raise ValueError("D24 score matrix is incomplete")
    units.sort(key=lambda row: (int(row["fold"]), int(row["seed"])))
    aggregate = {
        metric: summarize_delta(units, metric) for metric in METRICS
    }
    gate = build_gate(aggregate)
    status = (
        "D24_THOR_MAGNI_PROXIMITY_EVENT_DYNAMICS_SUPPORTED"
        if gate["supported"]
        else "D24_THOR_MAGNI_PROXIMITY_EVENT_DYNAMICS_NOT_SUPPORTED"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        score_path,
        sample_ids=np.asarray(
            [str(record["sample_id"]) for record in records],
            dtype="U31",
        ),
        seeds=np.asarray(SEEDS, dtype=np.int64),
        history=history_scores,
        zero_dynamics=zero_scores,
    )
    score_digest = sha256(score_path)
    score_sidecar.write_text(
        f"{score_digest}  {score_path.name}\n",
        encoding="utf-8",
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development real-recorded-trajectory geometric event proxy"
            ),
            "source_native_geometric_proxy": True,
            "human_event_truth": False,
            "deployable_threshold": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "d8_samples_path": str(args.d8_samples.resolve()),
            "d8_samples_sha256": sha256(args.d8_samples),
            "rgb_cache_path": str(args.rgb_cache.resolve()),
            "rgb_cache_sha256": sha256(args.rgb_cache),
            "flow_path": str(args.flow.resolve()),
            "flow_sha256": sha256(args.flow),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
            "d22_report_path": str(args.d22_report.resolve()),
            "d22_report_sha256": sha256(args.d22_report),
            "d23_additional_path": str(args.d23_additional.resolve()),
            "d23_additional_sha256": sha256(args.d23_additional),
            "d23_report_path": str(args.d23_report.resolve()),
            "d23_report_sha256": sha256(args.d23_report),
            "scores_path": str(score_path.resolve()),
            "scores_sha256": score_digest,
        },
        "design": {
            "comparison": (
                "same D23 history-trained checkpoint with actual history "
                "and flow versus repeated current RGB and zero flow"
            ),
            "event_grouping": (
                "within-source consecutive positive onset anchors with "
                "frame gap <= 45; event score is maximum anchor score"
            ),
            "negative_observations": "eligible negative anchors",
            "operating_point": (
                "per-source, per-arm most permissive threshold based only "
                "on heldout negative scores with observed false-active "
                "rate <= 0.10; diagnostic envelope, not deployable"
            ),
            "lead_time": (
                "source-native first 1.25m crossing on the original 0.10s "
                "future scan; missed positive anchors receive zero credit"
            ),
            "folds": 5,
            "seeds": list(SEEDS),
            "training_runs": 0,
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(
                {str(record["source_session_id"]) for record in records}
            ),
            "proximity_eligible_anchors": len(eligible),
            "positive_onset_anchors": positive_count,
            "negative_anchors": len(eligible) - positive_count,
            "positive_events": event_count,
            "crossing_offsets": len(crossing_offsets),
            "paired_units": len(units),
            "checkpoint_inference_passes": len(units) * 2,
        },
        "gate": gate,
        "aggregate_history_minus_zero_dynamics": aggregate,
        "units": units,
        "next_action": (
            "freeze an independent real-sequence decision replication"
            if gate["supported"]
            else (
                "retain D23 representation robustness and localize the "
                "event-layer conversion failure before any new model"
            )
        ),
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    sidecar.write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "gate": gate,
                "aggregate": aggregate,
                "report_sha256": digest,
                "scores_sha256": score_digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
