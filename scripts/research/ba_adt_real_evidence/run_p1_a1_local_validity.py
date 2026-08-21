#!/usr/bin/env python3
"""Run the one-shot P1-A1 conservative local-flow validity discovery."""

from __future__ import annotations

import argparse
from collections import deque
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
from typing import Any

import run_p1_consumed_adt_baseline as r0
from materialize_p1_temporal_cohort import SourceSpec, load_source, sha256
from run_rgb_observer import TargetMemory, seed_flow_points
from scripts.research.goal_copilot_bridge.p1_persistence import baseline


TRACE_SCHEMA = "blindassist_p1_a1_flow_health_trace_v1"
SWEEP_SCHEMA = "blindassist_p1_a1_local_validity_sweep_v1"
FEATURE_DIRECTIONS = {
    "point_survival_ratio": "ge",
    "fb_error_median_px": "le",
    "affine_ransac_inlier_ratio": "ge",
    "tracked_point_spatial_coverage": "ge",
    "flow_residual_dispersion": "le",
    "bbox_center_jump": "le",
    "affine_scale_jump": "le",
    "initial_anchor_appearance": "ge",
}
QUANTILES = tuple(index / 10.0 for index in range(1, 10))
TRIPLE_FEATURES = (
    "fb_error_median_px",
    "affine_ransac_inlier_ratio",
    "tracked_point_spatial_coverage",
)
RETENTION_MIN = 0.90
MEANINGFUL_REDUCTION_MIN = 0.50


def _finite(value: float, fallback: float = 1_000_000.0) -> float:
    return float(value) if math.isfinite(float(value)) else fallback


def flow_bbox_with_health(previous_gray, current_gray, points, bbox, width: int, height: int):
    """Replay the frozen forward LK propagation and add non-intervening health evidence."""
    import cv2
    import numpy as np

    if previous_gray is None or points is None or len(points) < 3 or bbox is None:
        return None, None, None
    moved, status, errors = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
    if moved is None or status is None:
        return None, None, None
    valid = status.reshape(-1).astype(bool)
    if errors is not None:
        valid &= errors.reshape(-1) < 30.0
    old_points = points.reshape(-1, 2)[valid]
    new_points = moved.reshape(-1, 2)[valid]
    if len(new_points) < 3:
        return None, None, None

    displacement_vectors = new_points - old_points
    displacement = np.median(displacement_vectors, axis=0)
    if abs(float(displacement[0])) > width * 0.08 or abs(float(displacement[1])) > height * 0.08:
        return None, None, None
    propagated = [
        max(0.0, min(width - 1.0, bbox[0] + float(displacement[0]))),
        max(0.0, min(height - 1.0, bbox[1] + float(displacement[1]))),
        max(0.0, min(width - 1.0, bbox[2] + float(displacement[0]))),
        max(0.0, min(height - 1.0, bbox[3] + float(displacement[1]))),
    ]
    if propagated[2] - propagated[0] < 3 or propagated[3] - propagated[1] < 3:
        return None, None, None

    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, new_points.reshape(-1, 1, 2).astype("float32"), None
    )
    if backward is None or backward_status is None:
        fb_error = 1_000_000.0
    else:
        backward_valid = backward_status.reshape(-1).astype(bool)
        fb_distances = np.linalg.norm(backward.reshape(-1, 2)[backward_valid] - old_points[backward_valid], axis=1)
        fb_error = float(np.median(fb_distances)) if len(fb_distances) else 1_000_000.0

    cv2.setRNGSeed(0)
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    inlier_ratio = float(inlier_mask.mean()) if inlier_mask is not None else 0.0
    if affine is None:
        scale_jump = 1_000_000.0
    else:
        affine_scale = math.sqrt(max(1e-12, float(affine[0, 0] ** 2 + affine[0, 1] ** 2)))
        scale_jump = abs(math.log(affine_scale))

    bbox_width = max(1.0, float(bbox[2] - bbox[0]))
    bbox_height = max(1.0, float(bbox[3] - bbox[1]))
    bbox_area = bbox_width * bbox_height
    hull_area = float(cv2.contourArea(cv2.convexHull(old_points.astype("float32")))) if len(old_points) >= 3 else 0.0
    spatial_coverage = max(0.0, min(1.0, hull_area / bbox_area))
    bbox_diagonal = max(1.0, math.hypot(bbox_width, bbox_height))
    residuals = np.linalg.norm(displacement_vectors - displacement, axis=1)
    dispersion = float(np.median(residuals)) / bbox_diagonal
    center_jump = math.hypot(float(displacement[0]), float(displacement[1])) / bbox_diagonal
    health = {
        "point_survival_ratio": float(len(new_points) / len(points)),
        "fb_error_median_px": _finite(fb_error),
        "affine_ransac_inlier_ratio": inlier_ratio,
        "tracked_point_spatial_coverage": spatial_coverage,
        "flow_residual_dispersion": _finite(dispersion),
        "bbox_center_jump": _finite(center_jump),
        "affine_scale_jump": _finite(scale_jump),
    }
    return propagated, new_points.reshape(-1, 1, 2).astype("float32"), health


def _decode_episode_frames(video_path: Path, episode: dict[str, Any]):
    import cv2

    wanted = [int(frame["video_frame_index"]) for frame in episode["frames"]]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open RGB video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    decoded_position = wanted[0]
    images = []
    for wanted_position in wanted:
        while decoded_position <= wanted_position:
            ok, image = capture.read()
            if not ok:
                capture.release()
                raise ValueError(f"{episode['episode_id']}: RGB decode failed at {decoded_position}")
            if decoded_position == wanted_position:
                images.append(image)
            decoded_position += 1
    capture.release()
    return images


def replay_episode(public_episode: dict[str, Any], video_path: Path) -> dict[str, Any]:
    import cv2

    images = _decode_episode_frames(video_path, public_episode)
    initial_box = [float(item) for item in public_episode["initial_target_bbox_xyxy"]]
    first_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    template = r0._crop_gray(first_gray, initial_box)
    if template is None:
        raise ValueError(f"{public_episode['episode_id']}: invalid initial crop")
    memory = TargetMemory(max_templates=1)
    if not memory.remember(images[0], initial_box, 0, force=True):
        raise ValueError(f"{public_episode['episode_id']}: initial appearance anchor failed")

    current_box: list[float] | None = initial_box
    previous_gray = first_gray
    points = seed_flow_points(first_gray, initial_box)
    missing_frames = 0
    reacquisition_history: deque[dict[str, Any] | None] = deque(maxlen=r0.REACQUIRE_HISTORY - 1)
    public_frames = []
    bbox_records = []
    health_by_candidate = {}

    for frame_index, (frame_spec, image) in enumerate(zip(public_episode["frames"], images)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidate = None
        bbox = None
        source = None
        health = None
        if frame_index == 0:
            bbox = initial_box
            candidate = {
                "candidate_id": "frame-0-rgb-candidate",
                "identity_support": 1.0,
                "identity_contradiction": 0.0,
                "stability": 1.0,
                "oscillation": 0.0,
            }
            source = "oracle_initialization"
        elif current_box is not None:
            propagated, moved_points, raw_health = flow_bbox_with_health(
                previous_gray, gray, points, current_box, image.shape[1], image.shape[0]
            )
            if propagated is not None:
                appearance = memory.score(image, propagated, frame_index)
                if appearance["appearance"] >= r0.TRACK_APPEARANCE_MIN:
                    bbox = propagated
                    current_box = propagated
                    points = seed_flow_points(gray, propagated)
                    stability = min(1.0, (0 if moved_points is None else len(moved_points)) / 12.0)
                    candidate = {
                        "candidate_id": f"frame-{frame_index}-rgb-candidate",
                        "identity_support": max(0.0, min(1.0, float(appearance["score"]))),
                        "identity_contradiction": 1.0 - max(0.0, min(1.0, float(appearance["appearance"]))),
                        "stability": stability,
                        "oscillation": 0.0,
                    }
                    health = {**raw_health, "initial_anchor_appearance": float(appearance["appearance"])}
                    source = "sparse_lk_flow"
                    missing_frames = 0
            if bbox is None:
                current_box = None
                points = None
                missing_frames = 1
        else:
            missing_frames += 1

        if frame_index > 0 and current_box is None and missing_frames % r0.REACQUIRE_SEARCH_INTERVAL == 0:
            proposal = r0.template_search(gray, template)
            eligible = (
                proposal is not None
                and proposal["score"] >= r0.REACQUIRE_SCORE_MIN
                and proposal["margin"] >= r0.REACQUIRE_MARGIN_MIN
            )
            confirmed = eligible and r0._compatible_reacquisition_hits(reacquisition_history, proposal) >= r0.REACQUIRE_CONFIRM_HITS
            reacquisition_history.append(proposal if eligible else None)
            if confirmed and proposal is not None:
                bbox = [float(item) for item in proposal["bbox_xyxy"]]
                current_box = bbox
                points = seed_flow_points(gray, bbox)
                candidate = {
                    "candidate_id": f"frame-{frame_index}-rgb-candidate",
                    "identity_support": float(proposal["score"]),
                    "identity_contradiction": 1.0 - float(proposal["score"]),
                    "stability": 0.70,
                    "oscillation": max(0.0, min(0.59, 0.60 - float(proposal["margin"]))),
                }
                source = "fixed_template_reacquisition"
                missing_frames = 0
                reacquisition_history.clear()

        public_frames.append({
            "frame_index": frame_index,
            "timestamp_ms": int(frame_spec["timestamp_ms"]),
            "candidates": [] if candidate is None else [candidate],
        })
        bbox_records.append({
            "frame_index": frame_index,
            "candidate_id": None if candidate is None else candidate["candidate_id"],
            "bbox_xyxy": bbox,
            "source": source,
        })
        if candidate is not None and health is not None:
            health_by_candidate[candidate["candidate_id"]] = health
        previous_gray = gray

    p1_input = {
        "schema_version": 1,
        "protocol_id": baseline.PROTOCOL_ID,
        "episode_id": public_episode["episode_id"],
        "handoff": public_episode["handoff"],
        "frames": public_frames,
    }
    return {
        "episode_id": public_episode["episode_id"],
        "p1_input": p1_input,
        "p1_output": baseline.run_baseline(p1_input),
        "candidate_bboxes": bbox_records,
        "health_by_candidate": health_by_candidate,
    }


def _assert_parity(replay: dict[str, Any], sealed: dict[str, Any]) -> None:
    if replay["p1_output"] != sealed["p1_output"]:
        raise ValueError(f"{replay['episode_id']}: frozen P1 output parity failed")
    if replay["candidate_bboxes"] != sealed["candidate_bboxes"]:
        raise ValueError(f"{replay['episode_id']}: frozen bbox/source parity failed")


def instrument_replay(public_path: Path, sealed_prediction_path: Path, trace_path: Path) -> dict[str, Any]:
    public = r0.validate_public_input(r0.read_json(public_path))
    sealed = r0.read_json(sealed_prediction_path)
    if sealed["public_input_sha256"] != r0.object_sha256(public):
        raise ValueError("sealed prediction/public input binding drift")
    sealed_by_id = {item["episode_id"]: item for item in sealed["episodes"]}
    episodes = []
    for source in public["sources"]:
        video_path = Path(source["rgb_video_path"])
        if sha256(video_path) != source["rgb_video_sha256"]:
            raise ValueError(f"RGB hash drift: {source['source_sequence_id']}")
        for episode in source["episodes"]:
            replay = replay_episode(episode, video_path)
            _assert_parity(replay, sealed_by_id[episode["episode_id"]])
            episodes.append(replay)
    trace = {
        "schema_version": TRACE_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "role": "RGB_ONLY_INSTRUMENTATION_REPLAY",
        "public_input_sha256": sealed["public_input_sha256"],
        "sealed_prediction_sha256": sha256(sealed_prediction_path),
        "instrumentation_parity": "PASS",
        "post_initialization_gt_reads": 0,
        "episodes": episodes,
    }
    r0.write_json(trace_path, trace)
    return trace


def _labels(private: dict[str, Any], trace_by_id: dict[str, Any]):
    labels = {}
    tags = {}
    for source_row in private["sources"]:
        gt_path = Path(source_row["groundtruth_path"])
        if sha256(gt_path) != source_row["groundtruth_sha256"]:
            raise ValueError(f"private GT hash drift: {source_row['source_sequence_id']}")
        source = load_source(
            SourceSpec(source_row["source_sequence_id"], gt_path, Path(source_row["rgb_video_path"])),
            probe_video=False,
        )
        for episode_row in source_row["episodes"]:
            source_episode = r0.read_json(Path(episode_row["episode_path"]))
            episode_id = episode_row["public_episode_id"]
            trace = trace_by_id[episode_id]
            target_id = source_episode["physical_target_id"]
            frame_labels = []
            for frame, bbox_record in zip(source_episode["frames"], trace["candidate_bboxes"]):
                if bbox_record["candidate_id"] is None:
                    identity_class = "NONE"
                else:
                    identity, _ = r0._match_bbox(
                        source, int(frame["timestamp_ns"]), bbox_record["bbox_xyxy"], episode_id
                    )
                    if identity == target_id:
                        identity_class = "CORRECT"
                    elif identity.startswith("background:"):
                        identity_class = "BACKGROUND_DRIFT"
                    else:
                        identity_class = "OTHER_INSTANCE"
                frame_labels.append({
                    "identity_class": identity_class,
                    "target_observable": bool(frame["target_visible"]),
                })
            labels[episode_id] = frame_labels
            tags[episode_id] = source_episode["temporal_mode_tags"]
    return labels, tags


def _quantile_grid(trace: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    import numpy as np

    values = {feature: [] for feature in FEATURE_DIRECTIONS}
    for episode in trace["episodes"]:
        for health in episode["health_by_candidate"].values():
            for feature in values:
                values[feature].append(float(health[feature]))
    return {
        feature: [
            {"quantile": quantile, "threshold": float(np.quantile(feature_values, quantile))}
            for quantile in QUANTILES
        ]
        for feature, feature_values in values.items()
    }


def _predicate(feature: str, row: dict[str, float]) -> dict[str, Any]:
    return {
        "feature": feature,
        "op": FEATURE_DIRECTIONS[feature],
        "quantile": row["quantile"],
        "threshold": row["threshold"],
    }


def _gate_family(grid: dict[str, list[dict[str, float]]]):
    features = list(FEATURE_DIRECTIONS)
    for feature in features:
        for row in grid[feature]:
            yield [_predicate(feature, row)]
    for left, right in combinations(features, 2):
        for left_row in grid[left]:
            for right_row in grid[right]:
                yield [_predicate(left, left_row), _predicate(right, right_row)]
    first, second, third = TRIPLE_FEATURES
    for first_row in grid[first]:
        for second_row in grid[second]:
            for third_row in grid[third]:
                yield [
                    _predicate(first, first_row),
                    _predicate(second, second_row),
                    _predicate(third, third_row),
                ]


def _passes(health: dict[str, float], predicates: list[dict[str, Any]]) -> bool:
    for predicate in predicates:
        value = float(health[predicate["feature"]])
        threshold = float(predicate["threshold"])
        if predicate["op"] == "ge" and value < threshold:
            return False
        if predicate["op"] == "le" and value > threshold:
            return False
    return True


def _gated_output(episode: dict[str, Any], predicates: list[dict[str, Any]]):
    gated_input = json.loads(json.dumps(episode["p1_input"]))
    health_by_candidate = episode["health_by_candidate"]
    for frame in gated_input["frames"]:
        kept = []
        for candidate in frame["candidates"]:
            health = health_by_candidate.get(candidate["candidate_id"])
            if health is None or _passes(health, predicates):
                kept.append(candidate)
        frame["candidates"] = kept
    return baseline.run_baseline(gated_input)


def _wrong_run(outputs: list[dict[str, Any]], frame_labels: list[dict[str, Any]], timestamps: list[int]):
    wrong = []
    for output, label in zip(outputs, frame_labels):
        wrong.append(output["current_candidate_id"] is not None and label["identity_class"] not in {"CORRECT", "NONE"})
    cadence = int(statistics.median([right - left for left, right in zip(timestamps, timestamps[1:])])) if len(timestamps) > 1 else 0
    maximum_frames = 0
    maximum_ms = 0
    start = None
    for index, flag in enumerate([*wrong, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            end = index - 1
            maximum_frames = max(maximum_frames, end - start + 1)
            maximum_ms = max(maximum_ms, timestamps[end] - timestamps[start] + cadence)
            start = None
    return maximum_frames, maximum_ms


def _score_gate(trace: dict[str, Any], labels: dict[str, Any], predicates: list[dict[str, Any]], baseline_counts: dict[str, Any]):
    correct = 0
    wrong = 0
    background_wrong = 0
    other_wrong = 0
    per_episode_wrong = {}
    max_wrong_frames = 0
    max_wrong_ms = 0
    outputs_by_episode = {}
    for episode in trace["episodes"]:
        episode_id = episode["episode_id"]
        output = _gated_output(episode, predicates)
        outputs_by_episode[episode_id] = output
        episode_wrong = 0
        for output_frame, label in zip(output["frames"], labels[episode_id]):
            if output_frame["current_candidate_id"] is None:
                continue
            if label["identity_class"] == "CORRECT":
                correct += 1
            elif label["identity_class"] == "BACKGROUND_DRIFT":
                wrong += 1
                background_wrong += 1
                episode_wrong += 1
            elif label["identity_class"] == "OTHER_INSTANCE":
                wrong += 1
                other_wrong += 1
                episode_wrong += 1
        per_episode_wrong[episode_id] = episode_wrong
        frames, duration = _wrong_run(
            output["frames"], labels[episode_id], [int(frame["timestamp_ms"]) for frame in episode["p1_input"]["frames"]]
        )
        max_wrong_frames = max(max_wrong_frames, frames)
        max_wrong_ms = max(max_wrong_ms, duration)

    episode_reductions = [
        (baseline_wrong - per_episode_wrong[episode_id]) / baseline_wrong
        for episode_id, baseline_wrong in baseline_counts["per_episode_wrong"].items()
        if baseline_wrong > 0
    ]
    retention = correct / baseline_counts["correct"]
    aggregate_reduction = (baseline_counts["wrong"] - wrong) / baseline_counts["wrong"]
    macro_reduction = statistics.mean(episode_reductions)
    lock_reduction = (baseline_counts["max_wrong_ms"] - max_wrong_ms) / baseline_counts["max_wrong_ms"]
    canonical = " AND ".join(
        f"{item['feature']} {item['op']} q{int(item['quantile'] * 100):02d}={item['threshold']:.9g}"
        for item in predicates
    )
    return {
        "predicates": predicates,
        "canonical": canonical,
        "predicate_count": len(predicates),
        "correct_assertions": correct,
        "correct_assertion_retention": retention,
        "retention_hard_pass": retention >= RETENTION_MIN,
        "wrong_instance_assertions": wrong,
        "background_wrong_assertions": background_wrong,
        "other_instance_wrong_assertions": other_wrong,
        "episode_macro_wrong_reduction": macro_reduction,
        "frame_aggregate_wrong_reduction": aggregate_reduction,
        "max_wrong_lock_frames": max_wrong_frames,
        "max_wrong_lock_duration_ms": max_wrong_ms,
        "max_wrong_lock_duration_reduction": lock_reduction,
        "meaningful_mechanism_pass": (
            macro_reduction >= MEANINGFUL_REDUCTION_MIN
            and aggregate_reduction >= MEANINGFUL_REDUCTION_MIN
            and lock_reduction >= MEANINGFUL_REDUCTION_MIN
        ),
        "per_episode_wrong_assertions": per_episode_wrong,
        "outputs_by_episode": outputs_by_episode,
    }


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (
        -row["episode_macro_wrong_reduction"],
        -row["frame_aggregate_wrong_reduction"],
        -row["max_wrong_lock_duration_reduction"],
        -row["correct_assertion_retention"],
        row["predicate_count"],
        row["canonical"],
    ))


def _choose_terminal(scored: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    admissible_signal = [row for row in scored if row["retention_hard_pass"] and row["meaningful_mechanism_pass"]]
    any_signal = [row for row in scored if row["meaningful_mechanism_pass"]]
    admissible = [row for row in scored if row["retention_hard_pass"]]
    if admissible_signal:
        return "CONSERVATIVE_LOCAL_VALIDITY_SIGNAL_ESTABLISHED", _rank(admissible_signal)[0]
    if any_signal:
        return "VALIDITY_GAIN_ONLY_BY_ABSTENTION", _rank(any_signal)[0]
    return "LOCAL_FLOW_VALIDITY_NOT_IDENTIFIABLE_FROM_CURRENT_RGB_HEALTH_FEATURES", _rank(admissible if admissible else scored)[0]


def _baseline_counts(trace: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    identity_gate = []
    counts = _score_gate_raw(trace, labels, identity_gate)
    if counts["correct"] != 87 or counts["wrong"] != 1221 or counts["max_wrong_ms"] != 8498:
        raise ValueError(f"P1-R0 baseline identity drift: {counts}")
    return counts


def _score_gate_raw(trace: dict[str, Any], labels: dict[str, Any], predicates: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    wrong = 0
    per_episode_wrong = {}
    max_wrong_frames = 0
    max_wrong_ms = 0
    for episode in trace["episodes"]:
        output = episode["p1_output"] if not predicates else _gated_output(episode, predicates)
        episode_wrong = 0
        for output_frame, label in zip(output["frames"], labels[episode["episode_id"]]):
            if output_frame["current_candidate_id"] is None:
                continue
            correct += int(label["identity_class"] == "CORRECT")
            is_wrong = label["identity_class"] in {"BACKGROUND_DRIFT", "OTHER_INSTANCE"}
            wrong += int(is_wrong)
            episode_wrong += int(is_wrong)
        per_episode_wrong[episode["episode_id"]] = episode_wrong
        frames, duration = _wrong_run(
            output["frames"], labels[episode["episode_id"]], [int(frame["timestamp_ms"]) for frame in episode["p1_input"]["frames"]]
        )
        max_wrong_frames = max(max_wrong_frames, frames)
        max_wrong_ms = max(max_wrong_ms, duration)
    return {"correct": correct, "wrong": wrong, "per_episode_wrong": per_episode_wrong, "max_wrong_frames": max_wrong_frames, "max_wrong_ms": max_wrong_ms}


def run_sweep(trace_path: Path, private_path: Path, sealed_prediction_path: Path, output_dir: Path) -> dict[str, Any]:
    trace = r0.read_json(trace_path)
    private = r0.read_json(private_path)
    sealed = r0.read_json(sealed_prediction_path)
    if trace.get("instrumentation_parity") != "PASS" or trace.get("post_initialization_gt_reads") != 0:
        raise ValueError("instrumentation parity/truth firewall did not pass")
    if trace["sealed_prediction_sha256"] != sha256(sealed_prediction_path):
        raise ValueError("sealed prediction identity drift")
    trace_by_id = {item["episode_id"]: item for item in trace["episodes"]}
    labels, tags = _labels(private, trace_by_id)
    baseline_counts = _baseline_counts(trace, labels)
    grid = _quantile_grid(trace)

    scored = []
    for predicates in _gate_family(grid):
        scored.append(_score_gate(trace, labels, predicates, baseline_counts))
    expected_count = len(FEATURE_DIRECTIONS) * 9 + math.comb(len(FEATURE_DIRECTIONS), 2) * 81 + 729
    if len(scored) != expected_count:
        raise ValueError(f"compact gate family drift: {len(scored)} != {expected_count}")

    admissible = [row for row in scored if row["retention_hard_pass"]]
    terminal, winner = _choose_terminal(scored)

    winner_episodes = []
    sealed_by_id = {item["episode_id"]: item for item in sealed["episodes"]}
    for episode in trace["episodes"]:
        sealed_episode = sealed_by_id[episode["episode_id"]]
        winner_episodes.append({
            **sealed_episode,
            "p1_output": winner["outputs_by_episode"][episode["episode_id"]],
        })
    winner_prediction = {
        **sealed,
        "episodes": winner_episodes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    winner_prediction_path = output_dir / "winner_prediction.json"
    winner_evaluation_path = output_dir / "winner_evaluation.json"
    r0.write_json(winner_prediction_path, winner_prediction)
    winner_evaluation = r0.evaluate_predictions(private_path, winner_prediction_path, winner_evaluation_path)

    for row in scored:
        row.pop("outputs_by_episode")
    winner.pop("outputs_by_episode", None)
    result = {
        "schema_version": SWEEP_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "claim_role": "CONSUMED_DEVELOPMENT_ONLY_NO_POLICY_ADMISSION_NO_SCIENTIFIC_VERDICT",
        "terminal": terminal,
        "policy_admission": "NO_POLICY_ADMISSION",
        "instrumentation": {
            "parity": trace["instrumentation_parity"],
            "post_initialization_gt_reads": trace["post_initialization_gt_reads"],
            "candidate_generator": r0.TRACKER_NAME,
            "candidate_generator_changed": False,
            "model_calls": 0,
        },
        "search": {
            "quantiles": list(QUANTILES),
            "feature_directions": FEATURE_DIRECTIONS,
            "triple_features": list(TRIPLE_FEATURES),
            "candidate_count": len(scored),
            "second_round_search": False,
        },
        "baseline": baseline_counts,
        "winner": winner,
        "winner_frozen_evaluator": winner_evaluation,
        "top_admissible": [{key: value for key, value in row.items() if key != "per_episode_wrong_assertions"} for row in _rank(admissible)[:20]],
        "all_candidates": scored,
        "episode_temporal_tags": tags,
    }
    r0.write_json(output_dir / "sweep_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("instrument-replay")
    replay.add_argument("--public-input", type=Path, required=True)
    replay.add_argument("--sealed-prediction", type=Path, required=True)
    replay.add_argument("--trace", type=Path, required=True)
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--trace", type=Path, required=True)
    sweep.add_argument("--private-input", type=Path, required=True)
    sweep.add_argument("--sealed-prediction", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "instrument-replay":
        trace = instrument_replay(args.public_input, args.sealed_prediction, args.trace)
        print(json.dumps({
            "instrumentation_parity": trace["instrumentation_parity"],
            "episodes": len(trace["episodes"]),
            "flow_candidates": sum(len(item["health_by_candidate"]) for item in trace["episodes"]),
            "post_initialization_gt_reads": trace["post_initialization_gt_reads"],
        }, sort_keys=True))
    else:
        result = run_sweep(args.trace, args.private_input, args.sealed_prediction, args.output_dir)
        print(json.dumps({"terminal": result["terminal"], "winner": result["winner"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
