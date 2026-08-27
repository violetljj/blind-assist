"""Run the R5 detector-dropout residual-occupancy Development canary.

The intervention removes the evaluator-associated RGB track during the final
0.2/0.4/0.8 seconds before a real JRDB contact.  The frozen R2 route matcher and
ONSET/HOLD/ESCALATE/CLEAR lifecycle are not changed.  Three arms are compared:

* R2 with the induced track dropout;
* R2 with fixed, at-most-0.4-second constant-velocity imputation;
* R2 plus a detector-independent dense RGB semantic residual.

The dense source is sealed before JRDB labels are opened.  It is the native
argmax ``person`` mask from an ADE20K semantic head, with no confidence or NMS
sweep.  At evaluation time, current detector boxes are removed from that mask;
connected residual components use the already documented fixed 1.70 m height
geometry and evaluator-only 2-D identity binding.  The residual contributes
positive evidence only.  Its absence cannot manufacture CLEAR.

This is a curated Development stress canary, not a natural-distribution,
calibrated-occupancy, product, user-benefit, or safety result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dtr_r0 import (
    CausalFrame,
    DTRConfig,
    EgoPose,
    Observation,
    Prediction,
    Signal,
    Vec2,
)
from dtr_r1 import RiskEventLifecycle, _first_tube_entry_s
from dtr_r2 import FROZEN_R2_CONFIG, DTRR2Arm
from jrdb_known_height_bridge import estimate_known_height_geometry
from jrdb_native_ceiling import (
    ArmAccumulator,
    TruthEvent,
    future_hits,
    score_arm,
    truth_events,
)
from jrdb_range_acquire import sha256_file
from jrdb_rgb_bridge import (
    FIRST_FRAME,
    HORIZON_S,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LAST_FRAME,
    ROUTE_HALF_WIDTH_M,
    SEQUENCE,
    associate_frame,
    interpolate_pose,
    load_image_timestamps,
    read_bag_pose_and_rgb,
    require,
)
from jrdb_sensor_geometry_bridge import (
    PERSON_RADIUS_M,
    SensorSample,
    contiguous_segments,
    load_truth_and_associate,
    read_jsonl,
    write_json,
)

SCHEMA = "blindassist-dtr-r5-track-dropout-canary-v1"
MASK_SCHEMA = "blindassist-dtr-r5-dense-semantic-mask-ledger-v1"
CLAIM_CEILING = "CURATED_PUBLIC_REAL_INDUCED_DROPOUT_DEVELOPMENT_CANARY_ONLY"
PERSON_CLASS_ID = 12
SEMANTIC_IMAGE_SIZE = 640
DROPOUT_DURATIONS_S = (0.2, 0.4, 0.8)
IMPUTATION_MAX_GAP_S = 0.4
EGO_HISTORY_S = 0.5
MINIMUM_CLOSING_SPEED_MPS = 0.05
ACTIVE_SIGNALS = {Signal.ONSET, Signal.HOLD, Signal.ESCALATE}


@dataclass(frozen=True)
class SegmentCase:
    label_id: str
    segment_index: int
    samples: tuple[SensorSample, ...]
    truth: tuple[bool | None, ...]
    known: tuple[bool, ...]
    events: tuple[TruthEvent, ...]


@dataclass(frozen=True)
class ArmRun:
    predictions: tuple[Prediction, ...]
    imputed_frames: int = 0
    residual_available_frames: int = 0
    residual_risk_frames: int = 0
    residual_distances_m: tuple[float, ...] = ()
    residual_evaluator_ious: tuple[float, ...] = ()


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)


def dense_ledger_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_name(output.stem + ".dense-masks.npz"),
        output.with_name(output.stem + ".dense-masks.json"),
    )


def materialize_dense_ledger(
    images_dir: Path,
    model_path: Path,
    ledger_path: Path,
    manifest_path: Path,
    batch_frames: int,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    require(torch.cuda.is_available(), "cuda_required_for_dense_semantic_source")
    image_paths = [images_dir / f"{frame:06d}.jpg" for frame in range(FIRST_FRAME, LAST_FRAME + 1)]
    for image_path in image_paths:
        image_path.resolve(strict=True)
    model_path = model_path.resolve(strict=True)
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(model_path))
    require(model.task == "semantic", f"semantic_model_required:{model.task}")
    require(model.names.get(PERSON_CLASS_ID) == "person", "semantic_person_class_drift")
    results = model.predict(
        [str(path) for path in image_paths],
        device=0,
        imgsz=SEMANTIC_IMAGE_SIZE,
        batch=batch_frames,
        verbose=False,
    )
    require(len(results) == len(image_paths), "semantic_result_count_drift")
    packed = []
    person_pixels = []
    for result in results:
        require(result.semantic_mask is not None, "semantic_mask_missing")
        values = result.semantic_mask.data
        require(tuple(values.shape) == (IMAGE_HEIGHT, IMAGE_WIDTH), "semantic_mask_shape_drift")
        mask = values.eq(PERSON_CLASS_ID).to(torch.uint8).cpu().numpy()
        person_pixels.append(int(mask.sum()))
        packed.append(np.packbits(mask, axis=1))
    frames = np.arange(FIRST_FRAME, LAST_FRAME + 1, dtype=np.int32)
    packed_array = np.stack(packed, axis=0)
    atomic_npz(ledger_path, frames=frames, packed_person_masks=packed_array)
    manifest = {
        "schema_version": MASK_SCHEMA,
        "truth_blind": True,
        "sequence": SEQUENCE,
        "frames": {"first": FIRST_FRAME, "last": LAST_FRAME, "count": len(image_paths)},
        "source": {
            "images_dir": str(images_dir.resolve()),
            "image_sha256": {path.name: sha256_file(path) for path in image_paths},
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
        },
        "inference": {
            "task": model.task,
            "person_class_id": PERSON_CLASS_ID,
            "person_class_name": model.names[PERSON_CLASS_ID],
            "decision": "native_argmax_semantic_class_no_confidence_or_nms_sweep",
            "image_size": SEMANTIC_IMAGE_SIZE,
            "batch_frames": batch_frames,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics_version,
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "observed_nonzero_person_frames": sum(value > 0 for value in person_pixels),
            "person_pixels_by_frame": dict(zip((path.stem for path in image_paths), person_pixels)),
        },
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": sha256_file(ledger_path),
    }
    write_json(manifest_path.resolve(), manifest)
    return manifest


def load_dense_ledger(ledger_path: Path, manifest_path: Path) -> tuple[dict[int, Any], dict[str, Any]]:
    import numpy as np

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == MASK_SCHEMA, "dense_manifest_schema")
    require(sha256_file(ledger_path) == manifest["ledger_sha256"], "dense_ledger_hash_drift")
    with np.load(ledger_path) as payload:
        frames = payload["frames"]
        packed = payload["packed_person_masks"]
    require(len(frames) == LAST_FRAME - FIRST_FRAME + 1, "dense_frame_count_drift")
    masks = {
        int(frame): np.unpackbits(row, axis=1, count=IMAGE_WIDTH).astype(bool)
        for frame, row in zip(frames, packed)
    }
    return masks, manifest


def load_truth_boxes(labels_path: Path) -> dict[int, list[dict[str, Any]]]:
    with zipfile.ZipFile(labels_path) as bundle:
        labels = json.loads(bundle.read(f"labels/labels_2d_stitched/{SEQUENCE}.json"))["labels"]
    output: dict[int, list[dict[str, Any]]] = {}
    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        rows = []
        for item in labels[f"{frame:06d}.jpg"]:
            if bool(item.get("attributes", {}).get("no_eval", False)):
                continue
            x, y, width, height = (float(value) for value in item["box"])
            if width > 0.0 and height > 0.0:
                rows.append(
                    {
                        "label_id": str(item["label_id"]),
                        "bbox_xyxy": [x, y, x + width, y + height],
                    }
                )
        output[frame] = rows
    return output


class ResidualLookup:
    def __init__(
        self,
        masks: dict[int, Any],
        detector_rows: Sequence[dict[str, Any]],
        truth_boxes: dict[int, list[dict[str, Any]]],
        focal_y_px: float,
    ) -> None:
        self.masks = masks
        self.truth_boxes = truth_boxes
        self.focal_y_px = focal_y_px
        self.detector_by_frame: dict[int, list[dict[str, Any]]] = {}
        for row in detector_rows:
            self.detector_by_frame.setdefault(int(row["frame_index"]), []).append(row)
        self.cache: dict[tuple[int, str | None], dict[str, dict[str, Any]]] = {}

    def by_target(
        self, frame: int, excluded_track_id: str | None
    ) -> dict[str, dict[str, Any]]:
        key = (frame, excluded_track_id)
        if key in self.cache:
            return self.cache[key]
        import cv2
        import numpy as np

        residual = self.masks[frame].copy()
        for row in self.detector_by_frame.get(frame, []):
            if excluded_track_id is not None and str(row["track_id"]) == excluded_track_id:
                continue
            x1, y1, x2, y2 = (float(value) for value in row["bbox_xyxy"])
            left = max(0, min(IMAGE_WIDTH, math.floor(x1)))
            top = max(0, min(IMAGE_HEIGHT, math.floor(y1)))
            right = max(0, min(IMAGE_WIDTH, math.ceil(x2)))
            bottom = max(0, min(IMAGE_HEIGHT, math.ceil(y2)))
            residual[top:bottom, left:right] = False
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            residual.astype(np.uint8), connectivity=8
        )
        candidates = []
        for component in range(1, count):
            left, top, width, height, pixels = (int(value) for value in stats[component])
            bbox = [float(left), float(top), float(left + width), float(top + height)]
            geometry = estimate_known_height_geometry(bbox, self.focal_y_px)
            if geometry is not None:
                candidates.append(
                    {
                        "bbox_xyxy": bbox,
                        "geometry": geometry,
                        "person_pixels": pixels,
                    }
                )
        matched = {}
        truth = self.truth_boxes[frame]
        for source_index, truth_index, overlap in associate_frame(candidates, truth):
            target = str(truth[truth_index]["label_id"])
            matched[target] = {**candidates[source_index], "evaluator_iou": overlap}
        self.cache[key] = matched
        return matched


def frame_context(timestamps_path: Path, bag_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    timestamps = load_image_timestamps(timestamps_path)
    poses, _rgb_times, bag_authority = read_bag_pose_and_rgb(bag_path)
    output = {}
    for frame in range(FIRST_FRAME, LAST_FRAME + 1):
        time_s = timestamps[frame]
        output[frame] = {
            "image_time_s": time_s,
            "pose": interpolate_pose(poses, round(time_s * 1e9)),
        }
    return output, bag_authority


def cases_from_tracks(tracks: dict[str, list[SensorSample]]) -> list[SegmentCase]:
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    output = []
    for label_id, values in sorted(tracks.items()):
        for segment_index, samples_value in enumerate(contiguous_segments(values)):
            samples = tuple(samples_value)
            if len(samples) < 2 or samples[-1].time_s - samples[0].time_s < config.minimum_track_span_s + HORIZON_S:
                continue
            truth, contacts = future_hits(samples)
            events, known = truth_events(samples, truth, contacts, config.minimum_track_span_s)
            if any(known):
                output.append(
                    SegmentCase(
                        label_id=label_id,
                        segment_index=segment_index,
                        samples=samples,
                        truth=tuple(truth),
                        known=tuple(known),
                        events=tuple(events),
                    )
                )
    return output


def sensor_observation(sample: SensorSample) -> Observation | None:
    if (
        sample.detector_track_id is None
        or sample.observed_forward_m is None
        or sample.observed_left_m is None
    ):
        return None
    return Observation(
        track_id=sample.detector_track_id,
        forward_m=sample.observed_forward_m,
        left_m=sample.observed_left_m,
        radius_m=sample.observed_radius_m,
    )


def sample_pose(sample: SensorSample) -> EgoPose:
    return EgoPose(
        sample.ego_x_m,
        sample.ego_y_m,
        sample.ego_yaw_rad,
        sample.ego_yaw_rad,
    )


def world_position(sample: SensorSample, observation: Observation) -> Vec2:
    return sample_pose(sample).local_to_world(observation.forward_m, observation.left_m)


def imputed_observation(
    sample: SensorSample,
    history: Sequence[tuple[SensorSample, Observation]],
) -> Observation | None:
    if len(history) < 2:
        return None
    previous_sample, previous = history[-2]
    last_sample, last = history[-1]
    gap_s = sample.time_s - last_sample.time_s
    span_s = last_sample.time_s - previous_sample.time_s
    if (
        gap_s <= 0.0
        or gap_s > IMPUTATION_MAX_GAP_S + 1e-9
        or span_s <= 0.0
        or previous.track_id != last.track_id
    ):
        return None
    previous_world = world_position(previous_sample, previous)
    last_world = world_position(last_sample, last)
    velocity = (last_world - previous_world) * (1.0 / span_s)
    predicted_world = last_world + velocity * gap_s
    local = predicted_world - sample_pose(sample).position
    cosine = math.cos(sample.ego_yaw_rad)
    sine = math.sin(sample.ego_yaw_rad)
    forward = local.x * cosine + local.y * sine
    left = -local.x * sine + local.y * cosine
    return Observation(last.track_id, forward, left, last.radius_m)


def base_urgent(prediction: Prediction, guard_boundary_s: float) -> bool:
    if prediction.raw_alert is not True:
        return False
    diagnostic = prediction.diagnostic
    if diagnostic.get("imminent_guard_active") == "true":
        return True
    value = diagnostic.get("r1_median_entry_s")
    return isinstance(value, (int, float)) and float(value) <= guard_boundary_s + 1e-9


def residual_entry_s(samples: Sequence[SensorSample], index: int, geometry: dict[str, Any]) -> float | None:
    current = samples[index]
    position = Vec2(float(geometry["forward_m"]), float(geometry["left_m"]))
    radius_m = ROUTE_HALF_WIDTH_M + PERSON_RADIUS_M
    if position.norm() <= radius_m + 1e-9:
        return 0.0
    earliest = index
    while earliest > 0 and current.time_s - samples[earliest - 1].time_s <= EGO_HISTORY_S + 1e-9:
        earliest -= 1
    if earliest == index:
        return None
    previous = samples[earliest]
    span_s = current.time_s - previous.time_s
    if span_s <= 0.0:
        return None
    dx = current.ego_x_m - previous.ego_x_m
    dy = current.ego_y_m - previous.ego_y_m
    cosine = math.cos(current.ego_yaw_rad)
    sine = math.sin(current.ego_yaw_rad)
    ego_forward_mps = (dx * cosine + dy * sine) / span_s
    ego_left_mps = (-dx * sine + dy * cosine) / span_s
    return _first_tube_entry_s(
        position,
        Vec2(-ego_forward_mps, -ego_left_mps),
        radius_m,
        HORIZON_S,
        MINIMUM_CLOSING_SPEED_MPS,
    )


def run_arm(
    case: SegmentCase,
    arm: str,
    dropout_frames: set[int],
    residual_lookup: ResidualLookup,
) -> ArmRun:
    require(arm in {"track_only", "bounded_imputation", "rgb_residual"}, f"unknown_arm:{arm}")
    config = DTRConfig(route_horizon_s=HORIZON_S, route_half_width_m=ROUTE_HALF_WIDTH_M)
    runner = DTRR2Arm(config)
    origin = case.samples[0].time_s
    real_history: list[tuple[SensorSample, Observation]] = []
    base_predictions = []
    imputed_frames = 0
    actual_observations: list[Observation | None] = []
    for sample in case.samples:
        real = sensor_observation(sample)
        dropped = sample.frame_index in dropout_frames and real is not None
        observation = None if dropped else real
        if arm == "bounded_imputation" and observation is None:
            observation = imputed_observation(sample, real_history)
            imputed_frames += int(observation is not None)
        if real is not None and not dropped:
            real_history.append((sample, real))
            real_history[:] = [
                item for item in real_history if sample.time_s - item[0].time_s <= 1.5 + 1e-9
            ]
        frame = CausalFrame(
            time_s=sample.time_s - origin,
            ego_pose=sample_pose(sample),
            observations=() if observation is None else (observation,),
            person_detection_count=int(observation is not None),
        )
        actual_observations.append(observation)
        base_predictions.append(runner.step(frame))
    if arm != "rgb_residual":
        return ArmRun(tuple(base_predictions), imputed_frames=imputed_frames)

    lifecycle = RiskEventLifecycle(config.clear_grace_s)
    fused = []
    residual_available_frames = 0
    residual_risk_frames = 0
    residual_distances_m = []
    residual_evaluator_ious = []
    guard_boundary_s = config.route_horizon_s * FROZEN_R2_CONFIG.imminent_horizon_fraction
    for index, (sample, base, observation) in enumerate(
        zip(case.samples, base_predictions, actual_observations)
    ):
        residual = None
        entry_s = None
        if observation is None:
            excluded = (
                sample.detector_track_id
                if sample.frame_index in dropout_frames
                else None
            )
            residual = residual_lookup.by_target(sample.frame_index, excluded).get(case.label_id)
            residual_available_frames += int(residual is not None)
            if residual is not None:
                residual_distances_m.append(
                    math.hypot(
                        float(residual["geometry"]["forward_m"]),
                        float(residual["geometry"]["left_m"]),
                    )
                )
                residual_evaluator_ious.append(float(residual["evaluator_iou"]))
                entry_s = residual_entry_s(case.samples, index, residual["geometry"])
        residual_risk = entry_s is not None
        residual_risk_frames += int(residual_risk)
        raw_alert = True if residual_risk else base.raw_alert
        urgent = residual_risk and entry_s <= guard_boundary_s + 1e-9
        urgent = urgent or base_urgent(base, guard_boundary_s)
        fused.append(
            Prediction(
                time_s=base.time_s,
                signal=lifecycle.update(base.time_s, raw_alert, urgent=urgent),
                raw_alert=raw_alert,
                reason=("rgb_residual_occupancy" if residual_risk else base.reason),
                track_id=(f"residual:{case.label_id}" if residual_risk else base.track_id),
                diagnostic={
                    **base.diagnostic,
                    "residual_available": str(residual is not None).lower(),
                    "residual_entry_s": entry_s if entry_s is not None else "none",
                    "residual_person_pixels": (
                        int(residual["person_pixels"]) if residual is not None else 0
                    ),
                    "residual_evaluator_iou": (
                        float(residual["evaluator_iou"]) if residual is not None else "none"
                    ),
                },
            )
        )
    return ArmRun(
        tuple(fused),
        residual_available_frames=residual_available_frames,
        residual_risk_frames=residual_risk_frames,
        residual_distances_m=tuple(residual_distances_m),
        residual_evaluator_ious=tuple(residual_evaluator_ious),
    )


def metrics_for_run(case: SegmentCase, run: ArmRun) -> ArmAccumulator:
    return score_arm(
        case.samples,
        run.predictions,
        case.events,
        case.known,
        case.truth,
    )


def original_cohort(cases: Sequence[SegmentCase], residual_lookup: ResidualLookup) -> dict[str, Any]:
    arms = {name: ArmAccumulator() for name in ("track_only", "bounded_imputation", "rgb_residual")}
    diagnostics = {
        name: {"imputed_frames": 0, "residual_available_frames": 0, "residual_risk_frames": 0}
        for name in arms
    }
    for case in cases:
        for name, accumulator in arms.items():
            current = run_arm(case, name, set(), residual_lookup)
            accumulator.merge(metrics_for_run(case, current))
            diagnostics[name]["imputed_frames"] += current.imputed_frames
            diagnostics[name]["residual_available_frames"] += current.residual_available_frames
            diagnostics[name]["residual_risk_frames"] += current.residual_risk_frames
    return {
        name: {**accumulator.to_dict(include_escalation=True), **diagnostics[name]}
        for name, accumulator in arms.items()
    }


def dropout_frames(samples: Sequence[SensorSample], event: TruthEvent, duration_s: float) -> set[int]:
    contact_time = samples[event.contact_index].time_s
    return {
        sample.frame_index
        for sample in samples
        if contact_time - duration_s - 1e-9 <= sample.time_s <= contact_time + 1e-9
    }


def stress_trials(cases: Sequence[SegmentCase], residual_lookup: ResidualLookup) -> dict[str, Any]:
    output = {}
    for duration_s in DROPOUT_DURATIONS_S:
        accumulators = {
            name: ArmAccumulator()
            for name in ("track_only", "bounded_imputation", "rgb_residual")
        }
        counts = {
            name: {
                "dropout_window_alerted_trials": 0,
                "dropout_window_known_evidence_frames": 0,
                "dropout_window_frames": 0,
                "imputed_frames": 0,
                "residual_available_frames": 0,
                "residual_risk_frames": 0,
            }
            for name in accumulators
        }
        trial_rows = []
        for case in cases:
            for event_index, event in enumerate(case.events):
                dropped = dropout_frames(case.samples, event, duration_s)
                arm_rows = {}
                for name, accumulator in accumulators.items():
                    current = run_arm(case, name, dropped, residual_lookup)
                    event_only_case = SegmentCase(
                        case.label_id,
                        case.segment_index,
                        case.samples,
                        case.truth,
                        case.known,
                        (event,),
                    )
                    metrics = metrics_for_run(event_only_case, current)
                    accumulator.merge(metrics)
                    window_indices = [
                        index
                        for index, sample in enumerate(case.samples)
                        if sample.frame_index in dropped
                    ]
                    alerted = any(
                        current.predictions[index].signal in ACTIVE_SIGNALS
                        and current.predictions[index].raw_alert is True
                        for index in window_indices
                    )
                    known_evidence = sum(
                        current.predictions[index].raw_alert is not None
                        for index in window_indices
                    )
                    counts[name]["dropout_window_alerted_trials"] += int(alerted)
                    counts[name]["dropout_window_known_evidence_frames"] += known_evidence
                    counts[name]["dropout_window_frames"] += len(window_indices)
                    counts[name]["imputed_frames"] += current.imputed_frames
                    counts[name]["residual_available_frames"] += current.residual_available_frames
                    counts[name]["residual_risk_frames"] += current.residual_risk_frames
                    arm_rows[name] = {
                        "dropout_window_alerted": alerted,
                        "dropout_window_known_evidence_frames": known_evidence,
                        "imputed_frames": current.imputed_frames,
                        "residual_available_frames": current.residual_available_frames,
                        "residual_risk_frames": current.residual_risk_frames,
                        "residual_min_estimated_distance_m": (
                            min(current.residual_distances_m)
                            if current.residual_distances_m
                            else None
                        ),
                        "residual_max_evaluator_iou": (
                            max(current.residual_evaluator_ious)
                            if current.residual_evaluator_ious
                            else None
                        ),
                        "metrics": metrics.to_dict(include_escalation=True),
                    }
                trial_rows.append(
                    {
                        "label_id": case.label_id,
                        "segment_index": case.segment_index,
                        "event_index": event_index,
                        "category": event.category,
                        "event_start_frame": case.samples[event.start_index].frame_index,
                        "contact_frame": case.samples[event.contact_index].frame_index,
                        "contact_truth_distance_m": case.samples[event.contact_index].distance_m,
                        "contact_truth_tube_threshold_m": case.samples[event.contact_index].tube_threshold_m,
                        "dropout_frames": sorted(dropped),
                        "arms": arm_rows,
                    }
                )
        trials = len(trial_rows)
        track_misses = sum(
            not row["arms"]["track_only"]["dropout_window_alerted"]
            for row in trial_rows
        )
        for name, values in counts.items():
            values["dropout_window_alert_recall"] = ratio(
                values["dropout_window_alerted_trials"], trials
            )
            values["dropout_window_known_evidence_rate"] = ratio(
                values["dropout_window_known_evidence_frames"],
                values["dropout_window_frames"],
            )
            values["recovered_track_only_window_misses"] = sum(
                not row["arms"]["track_only"]["dropout_window_alerted"]
                and row["arms"][name]["dropout_window_alerted"]
                for row in trial_rows
            )
            values["recovery_of_track_only_window_misses"] = ratio(
                values["recovered_track_only_window_misses"], track_misses
            )
        output[f"{duration_s:.1f}"] = {
            "duration_s": duration_s,
            "trials": trials,
            "track_only_dropout_window_misses": track_misses,
            "arms": {
                name: {**accumulators[name].to_dict(include_escalation=True), **counts[name]}
                for name in accumulators
            },
            "by_trial": trial_rows,
        }
    return output


def gate(original: dict[str, Any], stress: dict[str, Any]) -> dict[str, Any]:
    baseline = original["track_only"]
    residual = original["rgb_residual"]
    total_misses = sum(row["track_only_dropout_window_misses"] for row in stress.values())
    total_recovered = sum(
        row["arms"]["rgb_residual"]["recovered_track_only_window_misses"]
        for row in stress.values()
    )
    stress_track_fragmented = sum(
        row["arms"]["track_only"]["fragmented_events"] for row in stress.values()
    )
    stress_residual_fragmented = sum(
        row["arms"]["rgb_residual"]["fragmented_events"] for row in stress.values()
    )
    false_limit = baseline["false_alert_segments"] * 1.10
    checks = {
        "track_only_induced_window_misses_exist": total_misses > 0,
        "rgb_residual_recovers_at_least_half": (
            total_misses > 0 and total_recovered / total_misses >= 0.50
        ),
        "original_one_to_one_event_recall_not_lower": (
            residual["event_detection_recall"] is not None
            and baseline["event_detection_recall"] is not None
            and residual["event_detection_recall"] >= baseline["event_detection_recall"]
        ),
        "original_false_segments_within_ten_percent": (
            residual["false_alert_segments"] <= false_limit + 1e-9
        ),
        "original_clear_evaluable_and_not_lower": (
            baseline["clear_rate"] is not None
            and residual["clear_rate"] is not None
            and residual["clear_rate"] >= baseline["clear_rate"]
        ),
        "stress_fragmentation_evaluable_and_reduced": (
            stress_track_fragmented > 0
            and stress_residual_fragmented < stress_track_fragmented
        ),
    }
    passed = all(checks.values())
    if total_misses == 0:
        verdict = "R5_DROPOUT_EVENT_GATE_NOT_EVALUABLE_NO_INDUCED_WINDOW_MISSES"
    elif passed:
        verdict = "R5_DROPOUT_DEVELOPMENT_GATE_MET"
    else:
        verdict = "R5_DROPOUT_DEVELOPMENT_GATE_NOT_MET"
    return {
        "verdict": verdict,
        "passed": passed,
        "checks": checks,
        "track_only_induced_window_misses": total_misses,
        "rgb_residual_recovered_window_misses": total_recovered,
        "rgb_residual_recovery_rate": ratio(total_recovered, total_misses),
        "false_segment_limit": false_limit,
        "stress_fragmented_events": {
            "track_only": stress_track_fragmented,
            "rgb_residual": stress_residual_fragmented,
        },
        "clear_evidence": (
            "NOT_EVALUABLE_NO_CLEAR_ELIGIBLE_EVENTS"
            if baseline["clear_rate"] is None
            else "EVALUABLE"
        ),
        "fragmentation_evidence": (
            "NOT_EVALUABLE_TRACK_ONLY_FRAGMENTATION_IS_ZERO"
            if stress_track_fragmented == 0
            else "EVALUABLE"
        ),
        "note": (
            "Dropout-window alert recall is a controlled evidence-availability metric. "
            "Standard event recall/F1/CLEAR remain the functional metrics."
        ),
    }


def plot_result(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    durations = [float(value) for value in result["stress_by_duration_s"]]
    arms = ("track_only", "bounded_imputation", "rgb_residual")
    labels = {
        "track_only": "R2 track-only",
        "bounded_imputation": "R2 + bounded imputation",
        "rgb_residual": "R2 + RGB residual occupancy",
    }
    colors = {"track_only": "#C23B22", "bounded_imputation": "#6B7280", "rgb_residual": "#0F766E"}
    markers = {"track_only": "x", "bounded_imputation": "s", "rgb_residual": "o"}
    styles = {"track_only": "--", "bounded_imputation": "-.", "rgb_residual": "-"}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for arm in arms:
        window_recall = [
            result["stress_by_duration_s"][f"{duration:.1f}"]["arms"][arm][
                "dropout_window_alert_recall"
            ]
            for duration in durations
        ]
        event_f1 = [
            result["stress_by_duration_s"][f"{duration:.1f}"]["arms"][arm][
                "event_detection_f1"
            ]
            for duration in durations
        ]
        plot_options = {
            "marker": markers[arm],
            "linestyle": styles[arm],
            "linewidth": 2.2,
            "markersize": 7,
            "label": labels[arm],
            "color": colors[arm],
        }
        axes[0].plot(durations, window_recall, **plot_options)
        axes[1].plot(durations, event_f1, **plot_options)
    axes[0].set_title("Evidence during induced dropout")
    axes[0].set_ylabel("Dropout-window alert recall")
    axes[1].set_title("Frozen standard event evaluator")
    axes[1].set_ylabel("One-to-one event F1")
    for axis in axes:
        axis.set_xlabel("Track dropout duration (s)")
        axis.set_xticks(durations)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("DTR R5 detector-independent residual-occupancy canary", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    known_result_path = args.known_height_result.resolve(strict=True)
    known_tracks_path = args.known_height_tracks.resolve(strict=True)
    labels_path = args.labels_zip.resolve(strict=True)
    timestamps_path = args.timestamps_zip.resolve(strict=True)
    bag_path = args.bag.resolve(strict=True)
    images_dir = args.images_dir.resolve(strict=True)
    model_path = args.semantic_model.resolve(strict=True)
    known_result = json.loads(known_result_path.read_text(encoding="utf-8"))
    require(
        sha256_file(known_tracks_path)
        == known_result["truth_blind_sensor_geometry"]["ledger_sha256"],
        "known_height_ledger_hash_drift",
    )
    ledger_path, manifest_path = dense_ledger_paths(args.output.resolve())
    if args.reuse_dense_ledger and ledger_path.exists() and manifest_path.exists():
        masks, dense_manifest = load_dense_ledger(ledger_path, manifest_path)
    else:
        dense_manifest = materialize_dense_ledger(
            images_dir,
            model_path,
            ledger_path,
            manifest_path,
            args.batch_frames,
        )
        masks, dense_manifest = load_dense_ledger(ledger_path, manifest_path)

    # Privileged identity and future truth are opened only after the dense RGB
    # ledger above has been written and hash-sealed.
    sensor_rows = read_jsonl(known_tracks_path)
    context, bag_authority = frame_context(timestamps_path, bag_path)
    tracks, geometry_quality = load_truth_and_associate(labels_path, sensor_rows, context)
    cases = cases_from_tracks(tracks)
    truth_boxes = load_truth_boxes(labels_path)
    residual_lookup = ResidualLookup(
        masks,
        sensor_rows,
        truth_boxes,
        float(known_result["source"]["calibration"]["median_focal_y_px"]),
    )
    original = original_cohort(cases, residual_lookup)
    stress = stress_trials(cases, residual_lookup)
    result = {
        "schema_version": SCHEMA,
        "status": "DTR_R5_TRACK_DROPOUT_CANARY_COMPLETE",
        "claim_ceiling": CLAIM_CEILING,
        "question": (
            "Does a detector-independent dense RGB residual create usable route-risk "
            "evidence when a true detector track is actively removed?"
        ),
        "frozen": {
            "r2": FROZEN_R2_CONFIG.to_dict(),
            "route_horizon_s": HORIZON_S,
            "route_half_width_m": ROUTE_HALF_WIDTH_M,
            "lifecycle": "unchanged ONSET/HOLD/ESCALATE/CLEAR; missing remains UNKNOWN",
        },
        "intervention": {
            "durations_s": list(DROPOUT_DURATIONS_S),
            "placement": "final contiguous duration ending at evaluator contact",
            "target": "evaluator-associated current RGB track only",
            "bounded_imputation_max_gap_s": IMPUTATION_MAX_GAP_S,
        },
        "dense_rgb_source": dense_manifest,
        "source": {
            "dataset": "JRDB public train split",
            "sequence": SEQUENCE,
            "window": {"first_frame": FIRST_FRAME, "last_frame": LAST_FRAME},
            "known_height_result": str(known_result_path),
            "known_height_result_sha256": sha256_file(known_result_path),
            "known_height_tracks": str(known_tracks_path),
            "known_height_tracks_sha256": sha256_file(known_tracks_path),
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "timestamps": str(timestamps_path),
            "timestamps_sha256": sha256_file(timestamps_path),
            "bag": str(bag_path),
            "bag_sha256": sha256_file(bag_path),
            "bag_authority": bag_authority,
            "evaluable_target_segments": len(cases),
            "critical_events": sum(len(case.events) for case in cases),
        },
        "privileged_evaluator": {
            "identity_binding": "residual component to native current 2-D box at frozen IoU >= 0.30",
            "future_truth": "native future 3-D contact only after dense ledger seal",
            "geometry_quality": geometry_quality,
            "occupancy_calibration": "NOT_EVALUABLE_NO_PIXEL_OCCUPANCY_TRUTH_OR_PROBABILITY_OUTPUT",
        },
        "original_cohort": original,
        "stress_by_duration_s": stress,
        "gate": gate(original, stress),
        "limitations": [
            "The 143-frame window is transparently curated Development evidence with three events.",
            "Repeated duration trials are stress cases over the same events, not independent natural events.",
            "The ADE20K head provides a hard semantic argmax rather than calibrated occupancy probability.",
            "Fixed known-height geometry is wrong for truncation, children, seated/crouched people, and mask fragmentation.",
            "Residual short-horizon occupancy assumes a currently segmented target is static in the world and uses causal ego motion only.",
            "Evaluator-only current 2-D identity binding and future 3-D contact do not enter RGB inference.",
            "No source-disjoint generalization, Android runtime, user benefit, product reliability, or safety performance is established.",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-height-result", type=Path, required=True)
    parser.add_argument("--known-height-tracks", type=Path, required=True)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--timestamps-zip", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--batch-frames", type=int, default=4)
    parser.add_argument("--reuse-dense-ledger", action="store_true")
    args = parser.parse_args()
    require(args.output.suffix.lower() == ".json", "output_must_be_json")
    require(args.plot.suffix.lower() == ".png", "plot_must_be_png")
    require(args.batch_frames > 0, "batch_frames_must_be_positive")
    result = run(args)
    write_json(args.output.resolve(), result)
    plot_result(result, args.plot.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["gate"]["verdict"],
                "output": str(args.output.resolve()),
                "plot": str(args.plot.resolve()),
                "gate": result["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
