#!/usr/bin/env python3
"""Prepare, run, and privately evaluate the P1-R0 consumed ADT RGB baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from materialize_p1_temporal_cohort import SourceSpec, load_source, nearest_index, probe_video_timestamps, sha256
from run_rgb_observer import TargetMemory, flow_bbox, iou, seed_flow_points
from scripts.research.goal_copilot_bridge.p1_persistence import baseline, evaluator


PUBLIC_SCHEMA = "blindassist_p1_r0_consumed_adt_public_input_v1"
PRIVATE_SCHEMA = "blindassist_p1_r0_consumed_adt_private_eval_v1"
PREDICTION_SCHEMA = "blindassist_p1_r0_consumed_adt_prediction_v1"
RESULT_SCHEMA = "blindassist_p1_r0_consumed_adt_result_v1"
CLAIM_CEILING = "CONSUMED_ADT_INDOOR_OBJECT_DEVELOPMENT_BASELINE_ONLY"
TRACKER_NAME = "SPARSE_LK_PLUS_FIXED_TEMPLATE_R0"
TRACK_APPEARANCE_MIN = 0.55
REACQUIRE_SCORE_MIN = 0.72
REACQUIRE_MARGIN_MIN = 0.05
REACQUIRE_SEARCH_INTERVAL = 3
REACQUIRE_CONFIRM_HITS = 2
REACQUIRE_HISTORY = 3
TRUTH_MATCH_IOU = 0.30

PUBLIC_ROOT_KEYS = {"schema_version", "protocol_id", "claim_role", "tracker", "sources"}
PUBLIC_SOURCE_KEYS = {"source_sequence_id", "rgb_video_path", "rgb_video_sha256", "episodes"}
PUBLIC_EPISODE_KEYS = {"episode_id", "handoff", "frames", "initial_target_bbox_xyxy"}
PUBLIC_FRAME_KEYS = {"frame_index", "timestamp_ms", "video_frame_index"}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys drift: expected {sorted(expected)}, got {sorted(value)}")


def validate_public_input(value: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(value, PUBLIC_ROOT_KEYS, "public input")
    if value["schema_version"] != PUBLIC_SCHEMA or value["protocol_id"] != baseline.PROTOCOL_ID:
        raise ValueError("public input protocol drift")
    if value["claim_role"] != "CONSUMED_DEVELOPMENT_ONLY":
        raise ValueError("public input claim role drift")
    if value["tracker"] != TRACKER_NAME:
        raise ValueError("public input tracker drift")
    for source in value["sources"]:
        _exact_keys(source, PUBLIC_SOURCE_KEYS, "public source")
        for episode in source["episodes"]:
            _exact_keys(episode, PUBLIC_EPISODE_KEYS, "public episode")
            if re.fullmatch(r"p1-r0-consumed-[0-9]{3}", episode["episode_id"]) is None:
                raise ValueError("public episode id must be an opaque ordinal alias")
            if episode["handoff"]["status"] != "REFERENT_ESTABLISHED":
                raise ValueError("real baseline only accepts an established P0 referent")
            suffix = episode["episode_id"].rsplit("-", 1)[1]
            expected_ids = {
                "goal_id": f"consumed-goal-{suffix}",
                "referent_id": f"consumed-referent-{suffix}",
            }
            if any(episode["handoff"].get(key) != expected for key, expected in expected_ids.items()):
                raise ValueError("public handoff ids must use opaque ordinal aliases")
            if episode["handoff"]["grounding_provenance"].get("p0_decision_id") != f"oracle-init-{suffix}":
                raise ValueError("public P0 decision id must use an opaque ordinal alias")
            if not episode["frames"]:
                raise ValueError("public episode has no frames")
            for index, frame in enumerate(episode["frames"]):
                _exact_keys(frame, PUBLIC_FRAME_KEYS, "public frame")
                if frame["frame_index"] != index:
                    raise ValueError("public frame index drift")
            box = episode["initial_target_bbox_xyxy"]
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(item, (int, float)) for item in box):
                raise ValueError("initial oracle bbox is invalid")
    return value


def prepare_public_inputs(cohort_dir: Path, public_path: Path, private_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(cohort_dir / "p1_d0_manifest.json")
    if manifest.get("terminal") != "P1_TEMPORAL_DEVELOPMENT_COHORT_READY":
        raise ValueError("P1-D0 cohort is not in its frozen ready terminal")
    if manifest.get("role") != "CONSUMED_DEVELOPMENT_ONLY":
        raise ValueError("P1-D0 claim role drift")

    episodes = {path.stem: (path, read_json(path)) for path in sorted((cohort_dir / "episodes").glob("*.json"))}
    if sorted(episodes) != sorted(manifest["episode_ids"]):
        raise ValueError("P1-D0 manifest/episode identity drift")

    public_sources = []
    private_sources = []
    for source_row in manifest["sources"]:
        video_path = Path(source_row["rgb_video_path"])
        gt_path = Path(source_row["groundtruth_path"])
        if sha256(video_path) != source_row["rgb_video_sha256"] or sha256(gt_path) != source_row["groundtruth_sha256"]:
            raise ValueError(f"source hash drift: {source_row['source_sequence_id']}")
        video_times = probe_video_timestamps(video_path)
        public_episodes = []
        private_episodes = []
        for manifest_position, episode_id in enumerate(manifest["episode_ids"], start=1):
            episode_path, episode = episodes[episode_id]
            if episode["source_sequence_id"] != source_row["source_sequence_id"]:
                continue
            public_episode_id = f"p1-r0-consumed-{manifest_position:03d}"
            first = episode["frames"][0]
            initial_box = first["target_bbox_xyxy"]
            if not first["target_visible"] or initial_box is None:
                raise ValueError(f"{episode_id}: frame-0 oracle initialization is unavailable")
            public_frames = []
            for frame_index, frame in enumerate(episode["frames"]):
                video_index = nearest_index(video_times, int(frame["timestamp_ns"]))
                if video_index is None:
                    raise ValueError(f"{episode_id}: RGB timestamp alignment failed at frame {frame_index}")
                public_frames.append({
                    "frame_index": frame_index,
                    "timestamp_ms": int(frame["timestamp_ns"]) // 1_000_000,
                    "video_frame_index": video_index,
                })
            public_episodes.append({
                "episode_id": public_episode_id,
                "handoff": {
                    "status": "REFERENT_ESTABLISHED",
                    "goal_id": f"consumed-goal-{manifest_position:03d}",
                    "referent_id": f"consumed-referent-{manifest_position:03d}",
                    "grounding_provenance": {
                        "p0_decision_id": f"oracle-init-{manifest_position:03d}",
                        "source_frame_index": 0,
                        "authority": "P0_ESTABLISHED_REFERENT",
                    },
                },
                "frames": public_frames,
                "initial_target_bbox_xyxy": [float(item) for item in initial_box],
            })
            private_episodes.append({
                "public_episode_id": public_episode_id,
                "source_episode_id": episode_id,
                "episode_path": str(episode_path.resolve()),
            })
        public_sources.append({
            "source_sequence_id": source_row["source_sequence_id"],
            "rgb_video_path": str(video_path.resolve()),
            "rgb_video_sha256": source_row["rgb_video_sha256"],
            "episodes": public_episodes,
        })
        private_sources.append({
            "source_sequence_id": source_row["source_sequence_id"],
            "groundtruth_path": str(gt_path.resolve()),
            "groundtruth_sha256": source_row["groundtruth_sha256"],
            "rgb_video_path": str(video_path.resolve()),
            "episodes": private_episodes,
        })

    public = {
        "schema_version": PUBLIC_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "claim_role": "CONSUMED_DEVELOPMENT_ONLY",
        "tracker": TRACKER_NAME,
        "sources": public_sources,
    }
    validate_public_input(public)
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "claim_role": "CONSUMED_DEVELOPMENT_ONLY",
        "cohort_manifest_path": str((cohort_dir / "p1_d0_manifest.json").resolve()),
        "public_input_sha256": object_sha256(public),
        "sources": private_sources,
    }
    write_json(public_path, public)
    write_json(private_path, private)
    return public, private


def _crop_gray(gray, bbox: list[float]):
    x1, y1, x2, y2 = [int(round(item)) for item in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return gray[y1:y2, x1:x2].copy()


def _second_peak(response, location: tuple[int, int], template_shape: tuple[int, int]) -> float:
    import numpy as np

    suppressed = response.copy()
    x, y = location
    height, width = template_shape
    radius_x, radius_y = max(2, width // 2), max(2, height // 2)
    suppressed[max(0, y - radius_y):y + radius_y + 1, max(0, x - radius_x):x + radius_x + 1] = -1.0
    return float(np.max(suppressed)) if suppressed.size else -1.0


def template_search(gray, template) -> dict[str, Any] | None:
    import cv2

    scale = min(1.0, 96.0 / max(template.shape))
    search = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    anchor = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else template
    candidates = []
    for template_scale in (0.8, 1.0, 1.2):
        width = max(4, int(round(anchor.shape[1] * template_scale)))
        height = max(4, int(round(anchor.shape[0] * template_scale)))
        if width >= search.shape[1] or height >= search.shape[0]:
            continue
        resized = cv2.resize(anchor, (width, height), interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
        _, maximum, _, location = cv2.minMaxLoc(response)
        candidates.append((float(maximum), _second_peak(response, location, resized.shape), location, width, height))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, within_scale_second, (x, y), width, height = candidates[0]
    cross_scale_second = candidates[1][0] if len(candidates) > 1 else -1.0
    second = max(within_scale_second, cross_scale_second)
    return {
        "bbox_xyxy": [x / scale, y / scale, (x + width) / scale, (y + height) / scale],
        "score": max(0.0, min(1.0, score)),
        "margin": score - second,
    }


def _compatible_reacquisition_hits(history: Iterable[dict[str, Any] | None], current: dict[str, Any]) -> int:
    return 1 + sum(
        earlier is not None and iou(earlier["bbox_xyxy"], current["bbox_xyxy"]) >= 0.05
        for earlier in history
    )


def track_episode(public_episode: dict[str, Any], video_path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open RGB video: {video_path}")
    frames = public_episode["frames"]
    wanted = [int(frame["video_frame_index"]) for frame in frames]
    if any(right <= left for left, right in zip(wanted, wanted[1:])):
        raise ValueError(f"{public_episode['episode_id']}: non-monotonic RGB frame mapping")
    capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    decoded_position = wanted[0]
    images = []
    for wanted_position in wanted:
        while decoded_position <= wanted_position:
            ok, image = capture.read()
            if not ok:
                capture.release()
                raise ValueError(f"{public_episode['episode_id']}: RGB decode failed at {decoded_position}")
            if decoded_position == wanted_position:
                images.append(image)
            decoded_position += 1
    capture.release()

    initial_box = [float(item) for item in public_episode["initial_target_bbox_xyxy"]]
    first_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    template = _crop_gray(first_gray, initial_box)
    if template is None:
        raise ValueError(f"{public_episode['episode_id']}: invalid initial crop")
    memory = TargetMemory(max_templates=1)
    if not memory.remember(images[0], initial_box, 0, force=True):
        raise ValueError(f"{public_episode['episode_id']}: initial appearance anchor failed")

    current_box: list[float] | None = initial_box
    previous_gray = first_gray
    points = seed_flow_points(first_gray, initial_box)
    missing_frames = 0
    reacquisition_history: deque[dict[str, Any] | None] = deque(maxlen=REACQUIRE_HISTORY - 1)
    public_frames = []
    bbox_records = []

    for frame_index, (frame_spec, image) in enumerate(zip(frames, images)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidate = None
        bbox = None
        source = None
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
            propagated, moved_points = flow_bbox(
                previous_gray, gray, points, current_box, image.shape[1], image.shape[0]
            )
            if propagated is not None:
                appearance = memory.score(image, propagated, frame_index)
                if appearance["appearance"] >= TRACK_APPEARANCE_MIN:
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
                    source = "sparse_lk_flow"
                    missing_frames = 0
            if bbox is None:
                current_box = None
                points = None
                missing_frames = 1
        else:
            missing_frames += 1

        if frame_index > 0 and current_box is None and missing_frames % REACQUIRE_SEARCH_INTERVAL == 0:
            proposal = template_search(gray, template)
            eligible = (
                proposal is not None
                and proposal["score"] >= REACQUIRE_SCORE_MIN
                and proposal["margin"] >= REACQUIRE_MARGIN_MIN
            )
            confirmed = eligible and _compatible_reacquisition_hits(reacquisition_history, proposal) >= REACQUIRE_CONFIRM_HITS
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
                    "oscillation": max(0.0, min(1.0, 1.0 - float(proposal["margin"]))),
                }
                # The frozen baseline regards high oscillation as ambiguous. A confirmed
                # top-1 margin is represented directly as low oscillation evidence.
                candidate["oscillation"] = max(0.0, min(0.59, 0.60 - float(proposal["margin"])))
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
        previous_gray = gray

    p1_input = {
        "schema_version": 1,
        "protocol_id": baseline.PROTOCOL_ID,
        "episode_id": public_episode["episode_id"],
        "handoff": public_episode["handoff"],
        "frames": public_frames,
    }
    p1_output = baseline.run_baseline(p1_input)
    return {
        "episode_id": public_episode["episode_id"],
        "p1_output": p1_output,
        "candidate_bboxes": bbox_records,
        "oracle_initializations": 1,
        "post_initialization_gt_reads": 0,
    }


def run_tracker(public_path: Path, prediction_path: Path) -> dict[str, Any]:
    public = validate_public_input(read_json(public_path))
    episodes = []
    for source in public["sources"]:
        video_path = Path(source["rgb_video_path"])
        if sha256(video_path) != source["rgb_video_sha256"]:
            raise ValueError(f"RGB hash drift: {source['source_sequence_id']}")
        for episode in source["episodes"]:
            episodes.append(track_episode(episode, video_path))
    prediction = {
        "schema_version": PREDICTION_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "tracker": TRACKER_NAME,
        "public_input_sha256": object_sha256(public),
        "truth_access": {
            "oracle_initializations": sum(item["oracle_initializations"] for item in episodes),
            "post_initialization_gt_reads": sum(item["post_initialization_gt_reads"] for item in episodes),
        },
        "episodes": episodes,
    }
    write_json(prediction_path, prediction)
    return prediction


def _visible_segments(mask: list[bool]) -> list[tuple[int, int]]:
    segments = []
    start = None
    for index, visible in enumerate([*mask, False]):
        if visible and start is None:
            start = index
        elif not visible and start is not None:
            segments.append((start, index))
            start = None
    return segments


def _phases(episode: dict[str, Any]) -> list[str]:
    visible = [bool(frame["target_visible"]) for frame in episode["frames"]]
    phases = ["VISIBLE" if item else "SHORT_UNOBSERVABLE" for item in visible]
    segments = _visible_segments(visible)
    for (_, left_end), (right_start, _) in zip(segments, segments[1:]):
        gap = right_start - left_end
        for index in range(left_end, right_start):
            phases[index] = "LOSS_ELIGIBLE" if gap >= 45 else "SHORT_UNOBSERVABLE"
        if gap >= 45 or "OUT_OF_VIEW_RETURN" in episode["temporal_mode_tags"]:
            for index in range(right_start, min(len(phases), right_start + 12)):
                phases[index] = "REACQUISITION_WINDOW"
    return phases


def _scenario_class(tags: list[str]) -> str:
    if "LONG_LOSS" in tags:
        return "LONG_TARGET_ABSENCE"
    if "TEMP_OCCLUSION" in tags:
        return "SHORT_OCCLUSION"
    if "OUT_OF_VIEW_RETURN" in tags:
        return "TURN_AWAY_AND_RETURN"
    if "DISTRACTOR_PRESENT" in tags:
        return "SAME_CLASS_DISTRACTOR_CROSSING"
    return "CONTINUOUS_VISIBLE_CAMERA_MOTION"


def _match_bbox(source, timestamp: int, bbox: list[float], episode_id: str) -> tuple[str, float]:
    best_uid = None
    best_iou = 0.0
    for uid, rows in source.boxes.items():
        row = rows.get(timestamp)
        if row is None or row.get("visibility_ratio", 0.0) < 0.10:
            continue
        truth_box = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
        overlap = iou(bbox, truth_box)
        if overlap > best_iou:
            best_iou, best_uid = overlap, uid
    if best_uid is not None and best_iou >= TRUTH_MATCH_IOU:
        return f"adt:{best_uid}", best_iou
    return f"background:{episode_id}", best_iou


def _instance_for_bbox(source, timestamp: int, bbox: list[float], episode_id: str) -> str:
    return _match_bbox(source, timestamp, bbox, episode_id)[0]


def _build_evaluator_episode(source, episode: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
    phases = _phases(episode)
    p1_output = predicted["p1_output"]
    alias_suffix = episode["episode_id"].rsplit("-", 1)[1]
    public_frames = []
    truth_frames = []
    for frame_index, (frame, bbox_record, output_frame) in enumerate(
        zip(episode["frames"], predicted["candidate_bboxes"], p1_output["frames"])
    ):
        candidate_id = bbox_record["candidate_id"]
        candidate = None
        if candidate_id is not None:
            # Recover exactly the evidence seen by the frozen state machine from its output.
            # Non-asserted uncertain candidates still remain evaluator-visible.
            identity = output_frame["identity_score"]
            stability = output_frame["stability_score"]
            oscillation = output_frame["oscillation_score"]
            candidate = {
                "candidate_id": candidate_id,
                "identity_support": 0.0 if identity is None else identity,
                "identity_contradiction": 0.0,
                "stability": 0.0 if stability is None else stability,
                "oscillation": 0.0 if oscillation is None else oscillation,
            }
        public_frames.append({
            "frame_index": frame_index,
            "timestamp_ms": int(frame["timestamp_ns"]) // 1_000_000,
            "candidates": [] if candidate is None else [candidate],
        })
        candidate_map = {}
        if candidate_id is not None:
            candidate_map[candidate_id] = _instance_for_bbox(
                source, int(frame["timestamp_ns"]), bbox_record["bbox_xyxy"], episode["episode_id"]
            )
        truth_frames.append({
            "frame_index": frame_index,
            "referent_observable": bool(frame["target_visible"]),
            "candidate_instance_map": candidate_map,
            "allowed_states": ["TRACKING", "UNCERTAIN", "TEMP_UNOBSERVABLE", "LOST"],
            "allowed_events": ["NONE", "TARGET_TEMP_UNOBSERVABLE", "LOSS_DETECTED", "REACQUIRED", "REGROUND_REQUIRED"],
            "phase": phases[frame_index],
        })
    return {
        "schema_version": 1,
        "protocol_id": baseline.PROTOCOL_ID,
        "episode_id": episode["episode_id"],
        "scenario_class": _scenario_class(episode["temporal_mode_tags"]),
        "handoff": {
            "status": "REFERENT_ESTABLISHED",
            "goal_id": f"consumed-goal-{alias_suffix}",
            "referent_id": f"consumed-referent-{alias_suffix}",
            "grounding_provenance": {
                "p0_decision_id": f"oracle-init-{alias_suffix}",
                "source_frame_index": 0,
                "authority": "P0_ESTABLISHED_REFERENT",
            },
        },
        "frames": public_frames,
        "truth": {"referent_instance_id": episode["physical_target_id"], "frames": truth_frames},
    }


def _terminal(aggregate: dict[str, Any]) -> str:
    coverage = aggregate["correct_identity_coverage"]["value"] or 0.0
    safety_failures = (
        aggregate["wrong_instance_asserted_frames"]
        + aggregate["identity_switches"]
        + aggregate["false_reacquisitions"]
    )
    if coverage < 0.10:
        return "P1_R0_BASELINE_BELOW_PERSISTENCE_RESEARCH_FLOOR_OBSERVATION_FIRST"
    if safety_failures > 0:
        return "REAL_RGB_PERSISTENCE_HEADROOM_ESTABLISHED_ON_CONSUMED_ADT"
    return "NO_MATERIAL_PERSISTENCE_HEADROOM_ON_CURRENT_DEVELOPMENT_COHORT"


def evaluate_predictions(private_path: Path, prediction_path: Path, result_path: Path) -> dict[str, Any]:
    private = read_json(private_path)
    prediction = read_json(prediction_path)
    if private.get("schema_version") != PRIVATE_SCHEMA or prediction.get("schema_version") != PREDICTION_SCHEMA:
        raise ValueError("private/prediction schema drift")
    if prediction["public_input_sha256"] != private["public_input_sha256"]:
        raise ValueError("prediction is not bound to the prepared public input")
    if prediction["truth_access"] != {"oracle_initializations": 15, "post_initialization_gt_reads": 0}:
        raise ValueError("tracker truth-access receipt drift")
    predictions = {item["episode_id"]: item for item in prediction["episodes"]}
    pairs = []
    episode_tags = {}
    observable_frames = 0
    attribution = {
        "correct_target_asserted_frames": 0,
        "wrong_background_asserted_frames": 0,
        "wrong_other_adt_instance_asserted_frames": 0,
        "visible_frames_without_assertion": 0,
        "wrong_asserted_while_target_observable": 0,
    }
    for source_row in private["sources"]:
        gt_path = Path(source_row["groundtruth_path"])
        if sha256(gt_path) != source_row["groundtruth_sha256"]:
            raise ValueError(f"private GT hash drift: {source_row['source_sequence_id']}")
        source = load_source(SourceSpec(source_row["source_sequence_id"], gt_path, Path(source_row["rgb_video_path"])), probe_video=False)
        for episode_row in source_row["episodes"]:
            source_episode = read_json(Path(episode_row["episode_path"]))
            if source_episode["episode_id"] != episode_row["source_episode_id"]:
                raise ValueError("private source episode identity drift")
            episode = {**source_episode, "episode_id": episode_row["public_episode_id"]}
            predicted = predictions[episode["episode_id"]]
            evaluator_episode = _build_evaluator_episode(source, episode, predicted)
            evaluator.validate_episode(evaluator_episode)
            pairs.append((evaluator_episode, predicted["p1_output"]))
            episode_tags[episode["episode_id"]] = episode["temporal_mode_tags"]
            observable_frames += sum(bool(frame["target_visible"]) for frame in episode["frames"])
            for frame, bbox_record, output_frame in zip(
                episode["frames"], predicted["candidate_bboxes"], predicted["p1_output"]["frames"]
            ):
                if output_frame["current_candidate_id"] is None:
                    attribution["visible_frames_without_assertion"] += int(bool(frame["target_visible"]))
                    continue
                identity, _ = _match_bbox(
                    source, int(frame["timestamp_ns"]), bbox_record["bbox_xyxy"], episode["episode_id"]
                )
                if identity == episode["physical_target_id"]:
                    attribution["correct_target_asserted_frames"] += 1
                elif identity.startswith("background:"):
                    attribution["wrong_background_asserted_frames"] += 1
                    attribution["wrong_asserted_while_target_observable"] += int(bool(frame["target_visible"]))
                else:
                    attribution["wrong_other_adt_instance_asserted_frames"] += 1
                    attribution["wrong_asserted_while_target_observable"] += int(bool(frame["target_visible"]))
    evaluation = evaluator.evaluate_batch(pairs)
    if not evaluation["valid"]:
        errors = [
            f"{item['episode_id']}: {item['contract_error']}"
            for item in evaluation["episodes"]
            if not item["valid_system_output"]
        ]
        raise ValueError(f"frozen evaluator rejected real-RGB adapter output: {errors}")
    aggregate = evaluation["aggregate"]
    aggregate["false_loss_rate"] = evaluator._rate(aggregate["false_loss_frames"], observable_frames)

    by_mode: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "correct_coverage_numerator": 0, "correct_coverage_denominator": 0, "wrong_instance_asserted_frames": 0, "identity_switches": 0, "false_reacquisitions": 0, "recovery_opportunities": 0, "recovery_successes": 0})
    for item in evaluation["episodes"]:
        metrics = item["metrics"]
        for mode in episode_tags[item["episode_id"]]:
            row = by_mode[mode]
            row["episodes"] += 1
            row["correct_coverage_numerator"] += metrics["correct_identity_coverage"]["numerator"]
            row["correct_coverage_denominator"] += metrics["correct_identity_coverage"]["denominator"]
            for key in ("wrong_instance_asserted_frames", "identity_switches", "false_reacquisitions"):
                row[key] += metrics[key]
            opportunity = metrics["temporary_occlusion_opportunity"] if mode == "TEMP_OCCLUSION" else metrics["reacquisition_opportunity"]
            success = metrics["temporary_occlusion_recovered"] if mode == "TEMP_OCCLUSION" else metrics["reacquisition_success"]
            row["recovery_opportunities"] += int(bool(opportunity))
            row["recovery_successes"] += int(bool(success))
    mode_metrics = {}
    for mode, row in sorted(by_mode.items()):
        mode_metrics[mode] = {
            **row,
            "correct_identity_coverage": evaluator._rate(row["correct_coverage_numerator"], row["correct_coverage_denominator"]),
            "recovery_rate": evaluator._rate(row["recovery_successes"], row["recovery_opportunities"]),
        }

    result = {
        "schema_version": RESULT_SCHEMA,
        "protocol_id": baseline.PROTOCOL_ID,
        "tracker": TRACKER_NAME,
        "claim_ceiling": CLAIM_CEILING,
        "terminal": _terminal(aggregate),
        "truth_firewall": {
            "public_input_sha256": private["public_input_sha256"],
            "oracle_initializations": prediction["truth_access"]["oracle_initializations"],
            "post_initialization_gt_reads": prediction["truth_access"]["post_initialization_gt_reads"],
            "evaluator_only_gt_access": True,
        },
        "post_outcome_descriptive_failure_attribution": {
            **attribution,
            "role": "DESCRIPTIVE_ONLY_DOES_NOT_CHANGE_FROZEN_TERMINAL",
            "wrong_background_fraction": evaluator._rate(
                attribution["wrong_background_asserted_frames"],
                attribution["wrong_background_asserted_frames"] + attribution["wrong_other_adt_instance_asserted_frames"],
            ),
        },
        "evaluation": evaluation,
        "by_temporal_mode": mode_metrics,
    }
    write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-public")
    prepare.add_argument("--cohort-dir", type=Path, required=True)
    prepare.add_argument("--public-input", type=Path, required=True)
    prepare.add_argument("--private-input", type=Path, required=True)
    track = subparsers.add_parser("track")
    track.add_argument("--public-input", type=Path, required=True)
    track.add_argument("--prediction", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--private-input", type=Path, required=True)
    evaluate.add_argument("--prediction", type=Path, required=True)
    evaluate.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-public":
        public, _ = prepare_public_inputs(args.cohort_dir, args.public_input, args.private_input)
        print(json.dumps({"public_input_sha256": object_sha256(public), "episodes": sum(len(source["episodes"]) for source in public["sources"])}, sort_keys=True))
    elif args.command == "track":
        prediction = run_tracker(args.public_input, args.prediction)
        print(json.dumps({"episodes": len(prediction["episodes"]), "truth_access": prediction["truth_access"]}, sort_keys=True))
    else:
        result = evaluate_predictions(args.private_input, args.prediction, args.result)
        print(json.dumps({"terminal": result["terminal"], "aggregate": result["evaluation"]["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
