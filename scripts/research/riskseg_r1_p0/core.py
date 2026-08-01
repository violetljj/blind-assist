"""Pure scoring and nested-development helpers for RISKSEG-R1 P0."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLASS_ORDER = (
    "walkable",
    "blocking_obstacle",
    "boundary_level_change",
    "unknown_nonwalkable",
)
POSITIVE_BUCKETS = (
    "blocking_obstacle_positive",
    "boundary_level_change_positive",
)
MODEL_ARMS = ("seed-20260801", "seed-20260802", "seed-20260803")
ORACLE_ARM = "truth-mask"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def adapter_configs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    grid = contract["adapter_grid"]
    lateral_profiles = grid["lateral_profiles"]
    configs: list[dict[str, Any]] = []
    for boundary_weight in grid["boundary_weights"]:
        for top_fraction in grid["top_fractions"]:
            for profile_name in sorted(lateral_profiles):
                config = {
                    "boundary_weight": float(boundary_weight),
                    "top_fraction": float(top_fraction),
                    "lateral_profile": profile_name,
                    "lateral_weights": [
                        float(value) for value in lateral_profiles[profile_name]
                    ],
                }
                config["config_id"] = (
                    f"bw{boundary_weight:g}_top{top_fraction:g}_{profile_name}"
                )
                configs.append(config)
    return configs


def thresholds(contract: dict[str, Any]) -> list[float]:
    spec = contract["threshold_grid"]
    start = int(spec["start_milli"])
    stop = int(spec["stop_milli"])
    step = int(spec["step_milli"])
    if start <= 0 or stop > 1000 or step <= 0 or start > stop:
        raise ValueError("invalid threshold grid")
    return [value / 1000.0 for value in range(start, stop + 1, step)]


def fold_assignments(
    manifest: dict[str, Any], contract: dict[str, Any]
) -> dict[str, int]:
    fold_count = int(contract["nested_development"]["outer_fold_count"])
    offsets = {
        str(bucket): int(offset)
        for bucket, offset in contract["nested_development"][
            "bucket_fold_offsets"
        ].items()
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in manifest["events"]:
        grouped[str(event["bucket"])].append(event)
    assignments: dict[str, int] = {}
    for bucket in sorted(grouped):
        if bucket not in offsets:
            raise ValueError(f"missing fold offset for bucket {bucket}")
        events = sorted(grouped[bucket], key=lambda item: item["parent_event_id"])
        for position, event in enumerate(events):
            assignments[str(event["parent_event_id"])] = (
                position + offsets[bucket]
            ) % fold_count
    fold_sizes = [
        sum(fold == value for fold in assignments.values())
        for value in range(fold_count)
    ]
    if fold_sizes != [6] * fold_count:
        raise ValueError(f"outer fold sizes {fold_sizes} != {[6] * fold_count}")
    return assignments


def stable_softmax_from_int8(
    quantized: np.ndarray, scale: float
) -> np.ndarray:
    if quantized.dtype != np.int8 or quantized.ndim != 4 or quantized.shape[-1] != 4:
        raise ValueError(f"expected NHWC INT8 four-channel output, got {quantized}")
    # A shared zero point cancels in softmax. Subtracting the per-pixel maximum
    # before scaling avoids unnecessary float range.
    centered = quantized.astype(np.float32)
    centered -= centered.max(axis=-1, keepdims=True)
    logits = centered * np.float32(scale)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=-1, keepdims=True)
    if not np.isfinite(logits).all():
        raise ValueError("non-finite softmax output")
    return logits


def corridor_zone_masks(
    height: int, width: int, geometry: dict[str, Any]
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    top = max(0, min(height - 1, int(height * float(geometry["top_ratio"]))))
    corridor = np.zeros((height, width), dtype=bool)
    zones = [np.zeros((height, width), dtype=bool) for _ in range(3)]
    denominator = max(1, height - 1 - top)
    for y in range(top, height):
        progress = (y - top) / denominator
        half = float(geometry["top_half_width_ratio"]) + (
            float(geometry["bottom_half_width_ratio"])
            - float(geometry["top_half_width_ratio"])
        ) * progress
        left = max(0, min(width - 1, int(width * (0.5 - half))))
        right = max(left + 1, min(width, int(width * (0.5 + half))))
        corridor[y, left:right] = True
        span = right - left
        for x in range(left, right):
            zone = min(2, (3 * (x - left)) // span)
            zones[zone][y, x] = True
    if not corridor.any() or any(not zone.any() for zone in zones):
        raise ValueError("empty corridor or lateral zone")
    return corridor, (zones[0], zones[1], zones[2])


def _top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("top-fraction input must be a non-empty vector")
    count = max(1, math.ceil(values.size * fraction))
    if count == values.size:
        return float(values.mean())
    start = values.size - count
    return float(np.partition(values, start)[start:].mean())


def pool_probabilities(
    probabilities: np.ndarray,
    contract: dict[str, Any],
    configs: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    if probabilities.ndim == 4:
        if probabilities.shape[0] != 1:
            raise ValueError("batch size must be one")
        probabilities = probabilities[0]
    if probabilities.ndim != 3 or probabilities.shape[-1] != 4:
        raise ValueError(f"expected HWC probabilities, got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    height, width, _ = probabilities.shape
    corridor, zones = corridor_zone_masks(
        height, width, contract["corridor_geometry"]
    )
    walkable = probabilities[..., 0]
    obstacle = probabilities[..., 1]
    boundary = probabilities[..., 2]
    unknown = probabilities[..., 3]
    known = 1.0 - unknown
    sorted_probabilities = np.sort(probabilities, axis=-1)
    margin = sorted_probabilities[..., -1] - sorted_probabilities[..., -2]
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=-1
    ) / math.log(4.0)

    scores: dict[str, float] = {}
    for config in configs:
        evidence = known * (
            obstacle + float(config["boundary_weight"]) * boundary
        )
        zone_values = [
            _top_fraction_mean(evidence[zone], float(config["top_fraction"]))
            for zone in zones
        ]
        scores[str(config["config_id"])] = max(
            weight * value
            for weight, value in zip(config["lateral_weights"], zone_values)
        )

    def percentile(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q))

    diagnostic = {
        "corridor_pixel_count": int(corridor.sum()),
        "class_mean": {
            CLASS_ORDER[index]: float(probabilities[..., index][corridor].mean())
            for index in range(4)
        },
        "derived_known_mean": float(known[corridor].mean()),
        "unknown_p90": percentile(unknown[corridor], 90),
        "margin_mean": float(margin[corridor].mean()),
        "margin_p10": percentile(margin[corridor], 10),
        "normalized_entropy_mean": float(entropy[corridor].mean()),
        "normalized_entropy_p90": percentile(entropy[corridor], 90),
        "walkable_mean": float(walkable[corridor].mean()),
    }
    return scores, diagnostic


def score_event(
    event: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    config_id: str,
    threshold: float,
) -> dict[str, Any]:
    rows = sorted(frame_rows, key=lambda item: int(item["frame_index"]))
    if [int(row["frame_index"]) for row in rows] != list(range(len(event["frames"]))):
        raise ValueError(f"{event['parent_event_id']}: frame membership drift")
    active = [
        float(row["adapter_scores"][config_id]) >= threshold for row in rows
    ]
    positive = bool(event["positive"])
    hit_frames: list[int] = []
    passed_frames: list[int] = []
    if positive:
        alert_start, alert_end = map(int, event["alertable_interval_frames"])
        passed_start, passed_end = map(int, event["passed_interval_frames"])
        hit_frames = [
            index for index in range(alert_start, alert_end + 1) if active[index]
        ]
        passed_frames = [
            index for index in range(passed_start, passed_end + 1) if active[index]
        ]
    return {
        "parent_event_id": event["parent_event_id"],
        "source_session_id": event["source_session_id"],
        "bucket": event["bucket"],
        "positive": positive,
        "event_hit": positive and bool(hit_frames),
        "critical_miss": positive and not hit_frames,
        "false_alert_event": (not positive) and any(active),
        "passed_cleared": positive and not passed_frames,
        "first_alertable_alert_frame": hit_frames[0] if hit_frames else None,
    }


def aggregate(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scores = list(scores)
    positives = [item for item in scores if item["positive"]]
    negatives = [item for item in scores if not item["positive"]]
    hits = sum(bool(item["event_hit"]) for item in positives)
    cleared = sum(bool(item["passed_cleared"]) for item in positives)
    return {
        "positive_event_count": len(positives),
        "hit_event_count": hits,
        "event_recall": hits / len(positives) if positives else None,
        "critical_miss_count": len(positives) - hits,
        "negative_event_count": len(negatives),
        "false_alert_event_count": sum(
            bool(item["false_alert_event"]) for item in negatives
        ),
        "cleared_event_count": cleared,
        "clearance_rate": cleared / len(positives) if positives else None,
        "bucket_hit_counts": {
            bucket: sum(
                bool(item["event_hit"])
                for item in positives
                if item["bucket"] == bucket
            )
            for bucket in POSITIVE_BUCKETS
        },
    }


def candidate_rank(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    config_id: str,
    threshold: float,
) -> tuple[Any, ...]:
    hit_deficit = max(
        0, int(baseline["hit_event_count"]) - int(metrics["hit_event_count"])
    )
    false_excess = max(
        0,
        int(metrics["false_alert_event_count"])
        - int(baseline["false_alert_event_count"]),
    )
    clearance_deficit = max(
        0,
        int(baseline["cleared_event_count"])
        - int(metrics["cleared_event_count"]),
    )
    violation_sum = hit_deficit + false_excess + clearance_deficit
    return (
        violation_sum,
        hit_deficit,
        false_excess,
        clearance_deficit,
        -int(metrics["hit_event_count"]),
        int(metrics["false_alert_event_count"]),
        -int(metrics["cleared_event_count"]),
        config_id,
        threshold,
    )


def nested_oof_score(
    *,
    arm: str,
    manifest: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    yolo_events: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    events = {str(event["parent_event_id"]): event for event in manifest["events"]}
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        if row["arm"] == arm:
            by_event[str(row["parent_event_id"])].append(row)
    if set(by_event) != set(events):
        raise ValueError(f"{arm}: event membership mismatch")
    assignments = fold_assignments(manifest, contract)
    configs = adapter_configs(contract)
    threshold_values = thresholds(contract)
    fold_count = int(contract["nested_development"]["outer_fold_count"])
    outer_scores: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []

    for outer_fold in range(fold_count):
        inner_ids = sorted(
            event_id
            for event_id, fold in assignments.items()
            if fold != outer_fold
        )
        outer_ids = sorted(
            event_id
            for event_id, fold in assignments.items()
            if fold == outer_fold
        )
        baseline_inner = aggregate(yolo_events[event_id] for event_id in inner_ids)
        ranked: list[tuple[tuple[Any, ...], dict[str, Any], float, dict[str, Any]]] = []
        for config in configs:
            config_id = str(config["config_id"])
            for threshold in threshold_values:
                inner_scores = [
                    score_event(events[event_id], by_event[event_id], config_id, threshold)
                    for event_id in inner_ids
                ]
                metrics = aggregate(inner_scores)
                ranked.append(
                    (
                        candidate_rank(
                            metrics, baseline_inner, config_id, threshold
                        ),
                        config,
                        threshold,
                        metrics,
                    )
                )
        rank, selected_config, selected_threshold, inner_metrics = min(
            ranked, key=lambda item: item[0]
        )
        selections.append(
            {
                "outer_fold": outer_fold,
                "inner_event_count": len(inner_ids),
                "outer_event_count": len(outer_ids),
                "selected_config": selected_config,
                "selected_threshold": selected_threshold,
                "selection_rank": list(rank),
                "inner_adapter_metrics": inner_metrics,
                "inner_yolo_metrics": baseline_inner,
            }
        )
        for event_id in outer_ids:
            score = score_event(
                events[event_id],
                by_event[event_id],
                str(selected_config["config_id"]),
                selected_threshold,
            )
            score["outer_fold"] = outer_fold
            score["selected_config_id"] = selected_config["config_id"]
            score["selected_threshold"] = selected_threshold
            outer_scores.append(score)

    if len(outer_scores) != len(events) or len(
        {item["parent_event_id"] for item in outer_scores}
    ) != len(events):
        raise ValueError(f"{arm}: each event must be scored exactly once out of fold")
    return {
        "arm": arm,
        "fold_selections": selections,
        "oof_event_scores": sorted(
            outer_scores, key=lambda item: item["parent_event_id"]
        ),
        "oof_aggregate": aggregate(outer_scores),
    }


def timing_against_yolo(
    scored_events: list[dict[str, Any]],
    yolo_events: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    delays: list[int] = []
    for score in scored_events:
        yolo = yolo_events[score["parent_event_id"]]
        if score["event_hit"] and yolo["event_hit"]:
            delays.append(
                int(score["first_alertable_alert_frame"])
                - int(yolo["first_alertable_alert_frame"])
            )
    return {
        "common_hit_count": len(delays),
        "median_delay_frames": (
            float(statistics.median(delays)) if delays else None
        ),
        "late_over_two_frames_rate": (
            sum(value > 2 for value in delays) / len(delays) if delays else None
        ),
        "delays": delays,
    }
