#!/usr/bin/env python3
"""Evaluate a fixed HFTF checkpoint on the consumed SANPO event cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from evaluate_stage_c_d5_tartanground_event_proxy import (
    apply_decision_confirmation,
    decision_policy_spec,
    raw_lane_signals,
)
from train_stage_c_d5_tartanground_development_student import (
    MEAN,
    STD,
    TemporalStudent,
    sha256,
)


DECISION_POLICY = "height_spatiotemporal_selective_v2"
POSITIVE_BUCKETS = (
    "blocking_obstacle_positive",
    "boundary_level_change_positive",
)
HEIGHTS = {"body": 1, "head": 2}
HORIZONS = {"near": 1, "far": 2}
CENTRAL_DIRECTIONS = (2, 3)
TARGET_STEP_MS = 200

DEFAULT_MANIFEST = Path(
    "artifacts.local/evidence/riskseg-r0/event-eval/"
    "device-view-v2/manifest.json"
)
DEFAULT_BASELINE = Path(
    "docs/research/dual-loop/"
    "RISKSEG_R0_FINAL_RESULT_2026-08-01.json"
)
DEFAULT_PRETRAINED = Path(
    "artifacts.local/models/hftf/torch/hub/checkpoints/"
    "mobilenet_v3_small-047dcff4.pth"
)


class ManifestFrames(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(
        self,
        manifest_path: Path,
        manifest: dict[str, Any],
    ) -> None:
        self.root = manifest_path.parent
        self.rows = [
            (event_index, frame_index, frame)
            for event_index, event in enumerate(manifest["events"])
            for frame_index, frame in enumerate(event["frames"])
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int, int]:
        event_index, frame_index, frame = self.rows[index]
        path = self.root / frame["image_path"]
        with Image.open(path) as source:
            value = tvf.resize(
                source.convert("RGB"),
                [128, 224],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        tensor = tvf.pil_to_tensor(value).float().div_(255.0)
        tensor = tvf.normalize(tensor, MEAN, STD)
        return tensor, event_index, frame_index


def single_frame_logits(
    model: TemporalStudent,
    frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact repeated-current single arm without 5x encoding."""
    encoded = model.encoder(frames)
    current_kernel = model.temporal_depthwise.weight.sum(dim=2)
    fused = nnf.conv2d(
        encoded,
        current_kernel,
        groups=encoded.shape[1],
    )
    fused = model.pointwise(fused)
    if model.architecture == "pooled":
        output = model.head(model.dropout(model.pool(fused).flatten(1)))
        output = output.reshape(frames.shape[0], 2, 3, 3, 6, 6)
    elif model.architecture == "directional":
        directional = model.pool(fused).squeeze(2)
        output = model.head(model.dropout(directional))
        output = output.reshape(frames.shape[0], 2, 3, 3, 6, 6)
        output = output.permute(0, 1, 2, 3, 5, 4)
    else:
        grid = model.pool(fused)
        output = model.head(model.dropout(grid))
        output = output.reshape(frames.shape[0], 2, 3, 6, 3, 6)
        output = output.permute(0, 1, 2, 4, 5, 3)
    return output[:, 0], output[:, 1]


def sampled_indices(event: dict[str, Any]) -> list[int]:
    indices = [
        index
        for index, frame in enumerate(event["frames"])
        if int(frame["timestamp_ms"]) % TARGET_STEP_MS == 0
    ]
    if not indices or indices[0] != 0:
        raise ValueError(
            f"{event['parent_event_id']}: 5 Hz sampling must start at zero"
        )
    return indices


def hold_sampled_values(
    values: list[bool],
    sample_indices: list[int],
    frame_count: int,
) -> list[bool]:
    if len(values) != len(sample_indices):
        raise ValueError("Sample value/index count mismatch")
    output = [False] * frame_count
    for sample_number, start in enumerate(sample_indices):
        end = (
            sample_indices[sample_number + 1]
            if sample_number + 1 < len(sample_indices)
            else frame_count
        )
        output[start:end] = [values[sample_number]] * (end - start)
    return output


def contiguous_runs(values: list[bool]) -> list[list[int]]:
    runs: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(values + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append([start, index - 1])
            start = None
    return runs


def event_alerts(
    event: dict[str, Any],
    risk: np.ndarray,
    known: np.ndarray,
) -> tuple[list[bool], dict[str, Any]]:
    indices = sampled_indices(event)
    route_lanes: list[list[bool]] = []
    lane_diagnostics = {}
    for horizon, horizon_index in HORIZONS.items():
        for height, height_index in HEIGHTS.items():
            spec = decision_policy_spec(DECISION_POLICY)[height]
            for direction in CENTRAL_DIRECTIONS:
                bases = []
                overrides = []
                for frame_index in indices:
                    base, override = raw_lane_signals(
                        risk[
                            frame_index,
                            horizon_index,
                            height_index,
                            direction,
                        ],
                        known[
                            frame_index,
                            horizon_index,
                            height_index,
                            direction,
                        ],
                        height,
                        DECISION_POLICY,
                    )
                    bases.append(base)
                    overrides.append(override)
                confirmed = apply_decision_confirmation(
                    bases,
                    overrides,
                    spec,
                )
                route_lanes.append(
                    hold_sampled_values(
                        confirmed,
                        indices,
                        len(event["frames"]),
                    )
                )
                lane_diagnostics[
                    f"{horizon}/{height}/direction-{direction}"
                ] = {
                    "sampled_active_count": sum(confirmed),
                    "sampled_count": len(confirmed),
                }
    active = [
        any(lane[index] for lane in route_lanes)
        for index in range(len(event["frames"]))
    ]
    return active, {
        "sampled_frame_count": len(indices),
        "route_lane_count": len(route_lanes),
        "lane_activity": lane_diagnostics,
    }


def score_event(
    event: dict[str, Any],
    active: list[bool],
) -> dict[str, Any]:
    positive = event["bucket"] in POSITIVE_BUCKETS
    hit_frames: list[int] = []
    passed_active: list[int] = []
    if positive:
        alert_start, alert_end = map(
            int,
            event["alertable_interval_frames"],
        )
        passed_start, passed_end = map(
            int,
            event["passed_interval_frames"],
        )
        hit_frames = [
            index
            for index in range(alert_start, alert_end + 1)
            if active[index]
        ]
        passed_active = [
            index
            for index in range(passed_start, passed_end + 1)
            if active[index]
        ]
    return {
        "parent_event_id": event["parent_event_id"],
        "source_session_id": event["source_session_id"],
        "bucket": event["bucket"],
        "positive": positive,
        "event_hit": positive and bool(hit_frames),
        "critical_miss": positive and not hit_frames,
        "false_alert_event": (not positive) and any(active),
        "passed_cleared": positive and not passed_active,
        "first_alertable_alert_frame": (
            hit_frames[0] if hit_frames else None
        ),
        "response_delay_frames": (
            hit_frames[0] - int(event["alertable_interval_frames"][0])
            if hit_frames
            else None
        ),
        "active_frame_count": sum(active),
        "active_runs": contiguous_runs(active),
        "passed_active_frame_count": len(passed_active),
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in scores if row["positive"]]
    negatives = [row for row in scores if not row["positive"]]
    hits = sum(bool(row["event_hit"]) for row in positives)
    cleared = sum(bool(row["passed_cleared"]) for row in positives)
    response_delays = [
        int(row["response_delay_frames"])
        for row in positives
        if row["response_delay_frames"] is not None
    ]
    return {
        "positive_event_count": len(positives),
        "hit_event_count": hits,
        "event_recall": hits / len(positives),
        "critical_miss_count": len(positives) - hits,
        "negative_event_count": len(negatives),
        "false_alert_event_count": sum(
            bool(row["false_alert_event"]) for row in negatives
        ),
        "cleared_event_count": cleared,
        "clearance_rate": cleared / len(positives),
        "response_delay_frames_median": (
            float(np.median(response_delays))
            if response_delays
            else None
        ),
        "bucket_hit_counts": {
            bucket: sum(
                bool(row["event_hit"])
                for row in positives
                if row["bucket"] == bucket
            )
            for bucket in POSITIVE_BUCKETS
        },
        "bucket_miss_counts": {
            bucket: sum(
                bool(row["critical_miss"])
                for row in positives
                if row["bucket"] == bucket
            )
            for bucket in POSITIVE_BUCKETS
        },
        "negative_bucket_false_alert_counts": {
            bucket: sum(
                bool(row["false_alert_event"])
                for row in negatives
                if row["bucket"] == bucket
            )
            for bucket in sorted(
                {row["bucket"] for row in negatives}
            )
        },
    }


def reference_comparison(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        "hit_event_count": (
            int(metrics["hit_event_count"])
            - int(baseline["hit_event_count"])
        ),
        "critical_miss_count": (
            int(metrics["critical_miss_count"])
            - int(baseline["critical_miss_count"])
        ),
        "false_alert_event_count": (
            int(metrics["false_alert_event_count"])
            - int(baseline["false_alert_event_count"])
        ),
        "cleared_event_count": (
            int(metrics["cleared_event_count"])
            - int(baseline["cleared_event_count"])
        ),
    }
    nonworse = {
        "hit_event_count": deltas["hit_event_count"] >= 0,
        "false_alert_event_count": deltas["false_alert_event_count"] <= 0,
        "cleared_event_count": deltas["cleared_event_count"] >= 0,
    }
    strict = {
        "hit_event_count": deltas["hit_event_count"] > 0,
        "false_alert_event_count": deltas["false_alert_event_count"] < 0,
        "cleared_event_count": deltas["cleared_event_count"] > 0,
    }
    return {
        "deltas": deltas,
        "pareto_nonworse_all_three": all(nonworse.values()),
        "pareto_strict_on_at_least_one": any(strict.values()),
        "development_pareto_dominates_current_yolo": (
            all(nonworse.values()) and any(strict.values())
        ),
    }


def load_model(
    pretrained: Path,
    checkpoint_path: Path,
) -> tuple[TemporalStudent, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = TemporalStudent(
        pretrained,
        checkpoint.get("architecture", "pooled"),
        checkpoint.get("temporal_mode", "joint"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def infer_manifest_probabilities(
    model: TemporalStudent,
    dataset: ManifestFrames,
    manifest: dict[str, Any],
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    risk_by_event = [
        np.zeros((len(event["frames"]), 3, 3, 6, 6), np.float32)
        for event in manifest["events"]
    ]
    known_by_event = [
        np.zeros((len(event["frames"]), 3, 3, 6, 6), np.float32)
        for event in manifest["events"]
    ]
    with torch.inference_mode():
        for frames, event_indices, frame_indices in loader:
            risk_logits, known_logits = single_frame_logits(model, frames)
            risks = torch.sigmoid(risk_logits).cpu().numpy()
            knowns = torch.sigmoid(known_logits).cpu().numpy()
            for batch_index in range(len(frames)):
                event_index = int(event_indices[batch_index])
                frame_index = int(frame_indices[batch_index])
                risk_by_event[event_index][frame_index] = risks[batch_index]
                known_by_event[event_index][frame_index] = knowns[batch_index]
    return risk_by_event, known_by_event


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        int(manifest["event_count"]) != 30
        or sum(len(event["frames"]) for event in manifest["events"])
        != 1920
    ):
        raise ValueError("Expected the 30-event / 1,920-frame SANPO view")
    baseline_result = json.loads(
        args.baseline.read_text(encoding="utf-8")
    )
    baseline = baseline_result["event_evaluation"][
        "current_yolo_reference"
    ]
    model, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    dataset = ManifestFrames(args.manifest, manifest)
    risk_by_event, known_by_event = infer_manifest_probabilities(
        model,
        dataset,
        manifest,
        args.batch_size,
    )

    event_rows = []
    for event_index, event in enumerate(manifest["events"]):
        active, diagnostics = event_alerts(
            event,
            risk_by_event[event_index],
            known_by_event[event_index],
        )
        score = score_event(event, active)
        score["diagnostics"] = diagnostics
        event_rows.append(score)
    metrics = aggregate(event_rows)
    result = {
        "schema": "blindassist_hftf_stage_c_d6_sanpo_real_event_transfer_v0",
        "status": "SANPO_REAL_EVENT_TRANSFER_EVALUATION_COMPLETE",
        "policy": {
            "data_role": "consumed_development",
            "real_rgb": True,
            "human_reviewed_parent_event_truth": True,
            "checkpoint_trained_without_sanpo": True,
            "adapter_selected_before_hftf_outcomes": True,
            "decision_policy": DECISION_POLICY,
            "target_frequency_hz": 5,
            "route_direction_degrees": [-15, 15],
            "central_direction_indices": list(CENTRAL_DIRECTIONS),
            "human_safety_or_app_claim": False,
        },
        "model": {
            "name": args.name,
            "architecture": checkpoint.get("architecture", "pooled"),
            "temporal_mode": checkpoint.get("temporal_mode", "joint"),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "inputs": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "event_count": manifest["event_count"],
            "frame_count": len(dataset),
            "bucket_counts": manifest["bucket_counts"],
        },
        "current_yolo_reference": baseline,
        "metrics": metrics,
        "comparison_to_current_yolo": reference_comparison(
            metrics,
            baseline,
        ),
        "events": event_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
