#!/usr/bin/env python3
"""Fuse two visual person passes with projected CrowdBot track-role proposals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from contract import load_json, sha256_file, validate_prereg
from fuse_seen_person_proposals import run_bytetrack


def iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_matches(left: list[dict[str, Any]], right: list[dict[str, Any]], minimum_iou: float) -> list[tuple[int, int, float]]:
    candidates = [
        (iou(left_box["bbox_xyxy"], right_box["bbox_xyxy"]), left_index, right_index)
        for left_index, left_box in enumerate(left)
        for right_index, right_box in enumerate(right)
    ]
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = []
    for overlap, left_index, right_index in sorted(candidates, reverse=True):
        if overlap < minimum_iou:
            break
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_index, right_index, overlap))
    return matches


def average_box(left: list[float], right: list[float]) -> list[float]:
    return [round((float(a) + float(b)) / 2.0, 3) for a, b in zip(left, right, strict=True)]


def contains(box: list[float], point: list[float]) -> bool:
    return float(box[0]) <= float(point[0]) <= float(box[2]) and float(box[1]) <= float(point[1]) <= float(box[3])


def segment_intersects_box(start: list[float], end: list[float], box: list[float]) -> bool:
    """Liang-Barsky intersection, including points and segments on the boundary."""
    x1, y1, x2, y2 = [float(value) for value in box]
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    lower, upper = 0.0, 1.0
    for direction, offset in (
        (-dx, float(start[0]) - x1),
        (dx, x2 - float(start[0])),
        (-dy, float(start[1]) - y1),
        (dy, y2 - float(start[1])),
    ):
        if abs(direction) <= 1e-12:
            if offset < 0.0:
                return False
            continue
        ratio = offset / direction
        if direction < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True


def frame_candidates(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    minimum_iou: float,
    acceptance_confidence: float,
) -> list[dict[str, Any]]:
    matches = greedy_matches(left, right, minimum_iou)
    used_left = {row[0] for row in matches}
    used_right = {row[1] for row in matches}
    candidates = []
    for left_index, right_index, overlap in matches:
        confidences = [
            float(left[left_index].get("proposal_confidence", 1.0)),
            float(right[right_index].get("proposal_confidence", 1.0)),
        ]
        if max(confidences) < acceptance_confidence:
            continue
        candidates.append({
            "bbox_xyxy": average_box(left[left_index]["bbox_xyxy"], right[right_index]["bbox_xyxy"]),
            "visual_signals": ["pass_b", "pass_c"],
            "visual_confidences": confidences,
            "proposal_confidence": max(confidences),
            "visual_iou": overlap,
        })
    candidates.extend({
        "bbox_xyxy": row["bbox_xyxy"],
        "visual_signals": ["pass_b"],
        "visual_confidences": [float(row.get("proposal_confidence", 1.0))],
        "proposal_confidence": float(row.get("proposal_confidence", 1.0)),
        "visual_iou": None,
    } for index, row in enumerate(left) if index not in used_left and float(row.get("proposal_confidence", 1.0)) >= acceptance_confidence)
    candidates.extend({
        "bbox_xyxy": row["bbox_xyxy"],
        "visual_signals": ["pass_c"],
        "visual_confidences": [float(row.get("proposal_confidence", 1.0))],
        "proposal_confidence": float(row.get("proposal_confidence", 1.0)),
        "visual_iou": None,
    } for index, row in enumerate(right) if index not in used_right and float(row.get("proposal_confidence", 1.0)) >= acceptance_confidence)
    return candidates


def truth_route_hit(box: list[float], route: dict[str, Any] | None, margin_fraction: float) -> bool | None:
    if route is None or route.get("status") != "known":
        return None
    margin = 480.0 * margin_fraction
    expanded = [
        float(box[0]) - margin,
        float(box[1]) - margin,
        float(box[2]) + margin,
        float(box[3]) + margin,
    ]
    polyline = route.get("uv_polyline")
    if polyline is not None:
        if len(polyline) < 2:
            return None
        return any(
            segment_intersects_box(start, end, expanded)
            for start, end in zip(polyline, polyline[1:])
        )
    if route.get("uv") is None:
        return None
    return contains(expanded, route["uv"])


def derive_visual_event_proposals(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for frame in frames:
        frame_number = int(frame["frame_id"])
        for person in frame["persons"]:
            if person.get("published_track_id") is None and person.get("visual_proposal_track_id") is not None:
                observations[(frame["source_id"], frame["sequence_id"], person["person_id"])].append((frame_number, person))
    events = []
    for (source_id, sequence_id, person_id), rows in sorted(observations.items()):
        previous_frame = None
        active_run: list[tuple[int, dict[str, Any]]] = []
        release_run = 0
        intersecting_run = 0
        current_event = None
        event_index = 0
        for frame_number, person in rows:
            if previous_frame is None or frame_number != previous_frame + 1:
                active_run = []
                release_run = 0
                intersecting_run = 0
            if person["role"] == "route_intersecting":
                active_run.append((frame_number, person))
                active_run = active_run[-2:]
                release_run = 0
                intersecting_run += 1
                if current_event is None and len(active_run) == 2:
                    event_id = f"{sequence_id}:visual:{person_id}:{event_index}"
                    event_index += 1
                    current_event = {
                        "source_id": source_id,
                        "sequence_id": sequence_id,
                        "event_id": event_id,
                        "person_id": person_id,
                        "published_track_id": None,
                        "onset_frame": active_run[0][0],
                        "alertable_start_frame": frame_number,
                        "clear_frame": None,
                        "critical": False,
                        "censored_without_clear": True,
                    }
                    for _, active_person in active_run:
                        active_person["event_id"] = event_id
                    events.append(current_event)
                if current_event is not None:
                    person["event_id"] = current_event["event_id"]
                    if intersecting_run >= 3:
                        current_event["critical"] = True
            else:
                active_run = []
                intersecting_run = 0
                if current_event is not None and person["role"] == "adjacent_safe":
                    release_run += 1
                    person["event_id"] = current_event["event_id"]
                    if release_run >= 3:
                        person["role"] = "cleared"
                        current_event["clear_frame"] = frame_number
                        current_event["censored_without_clear"] = False
                        current_event = None
                        release_run = 0
                else:
                    release_run = 0
            previous_frame = frame_number
    return events


def derive_visible_metric_event_proposals(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive camera-confirmed events instead of inheriting non-visible LiDAR onset."""
    frames_by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        frames_by_sequence[(frame["source_id"], frame["sequence_id"])].append(frame)
        for person in frame["persons"]:
            person["event_id"] = None
    events = []
    for (source_id, sequence_id), sequence_frames in sorted(frames_by_sequence.items()):
        sequence_frames.sort(key=lambda row: int(row["frame_id"]))
        states: dict[str, dict[str, Any]] = {}
        for frame in sequence_frames:
            frame_number = int(frame["frame_id"])
            visible_ids = set()
            for person in frame["persons"]:
                if person.get("published_track_id") is None:
                    continue
                person_id = person["person_id"]
                if person_id in visible_ids:
                    raise RuntimeError(f"duplicate visible metric person in frame: {sequence_id} {frame_number} {person_id}")
                visible_ids.add(person_id)
                state = states.setdefault(
                    person_id,
                    {
                        "active_run": [],
                        "intersecting_run": 0,
                        "current_event": None,
                        "event_index": 0,
                        "last_seen_frame": None,
                    },
                )
                if state["last_seen_frame"] is not None and frame_number != state["last_seen_frame"] + 1:
                    state["active_run"] = []
                    state["intersecting_run"] = 0
                    if state["current_event"] is not None:
                        state["current_event"]["identity_continuous"] = False
                        state["current_event"]["visible_gap_frame_count"] += (
                            frame_number - state["last_seen_frame"] - 1
                        )
                role = person["role"]
                active = role in {"route_intersecting", "approaching_route"}
                if active:
                    state["active_run"].append((frame_number, person))
                    state["active_run"] = state["active_run"][-2:]
                    state["intersecting_run"] = (
                        state["intersecting_run"] + 1 if role == "route_intersecting" else 0
                    )
                    if state["current_event"] is None and len(state["active_run"]) == 2:
                        event_id = f"{sequence_id}:visible-metric:{person_id}:{state['event_index']}"
                        state["event_index"] += 1
                        event = {
                            "source_id": source_id,
                            "sequence_id": sequence_id,
                            "event_id": event_id,
                            "person_id": person_id,
                            "published_track_id": person["published_track_id"],
                            "onset_frame": state["active_run"][0][0],
                            "alertable_start_frame": frame_number,
                            "clear_frame": None,
                            "critical": False,
                            "censored_without_clear": True,
                            "identity_continuous": True,
                            "visible_gap_frame_count": 0,
                            "truth_origin": "visible_metric_person_role_episode",
                        }
                        state["current_event"] = event
                        for _, active_person in state["active_run"]:
                            active_person["event_id"] = event_id
                        events.append(event)
                    if state["current_event"] is not None:
                        person["event_id"] = state["current_event"]["event_id"]
                        if state["intersecting_run"] >= 3:
                            state["current_event"]["critical"] = True
                else:
                    state["active_run"] = []
                    state["intersecting_run"] = 0
                    if state["current_event"] is not None:
                        person["event_id"] = state["current_event"]["event_id"]
                        if role == "cleared":
                            state["current_event"]["clear_frame"] = frame_number
                            state["current_event"]["censored_without_clear"] = False
                            state["current_event"] = None
                state["last_seen_frame"] = frame_number
    return events


def fuse_frame(
    left: dict[str, Any],
    right: dict[str, Any],
    projected: dict[str, Any],
    minimum_iou: float,
    acceptance_confidence: float = 0.35,
    annotation_route_truth: dict[str, Any] | None = None,
    route_margin_fraction: float = 0.08,
    visual_candidates: list[dict[str, Any]] | None = None,
    stable_visual_track_ids: set[str] | None = None,
) -> dict[str, Any]:
    identity = (left["source_id"], left["sequence_id"], left["frame_id"], left["image_sha256"])
    if identity != (right["source_id"], right["sequence_id"], right["frame_id"], right["image_sha256"]):
        raise RuntimeError("visual proposal frame identity mismatch")
    if identity[:3] != (projected["source_id"], projected["sequence_id"], projected["frame_id"]):
        raise RuntimeError("projected role frame identity mismatch")
    candidates = visual_candidates if visual_candidates is not None else frame_candidates(
        left["person_proposals"], right["person_proposals"], minimum_iou, acceptance_confidence
    )
    stable_visual_track_ids = stable_visual_track_ids or set()
    projected_tracks = projected.get("projected_tracks", []) if projected.get("status") == "known" else []
    candidate_tracks: dict[int, list[int]] = defaultdict(list)
    track_candidates: dict[int, list[int]] = defaultdict(list)
    for candidate_index, candidate in enumerate(candidates):
        for track_index, track in enumerate(projected_tracks):
            if contains(candidate["bbox_xyxy"], track["projected_track_center_uv"]):
                candidate_tracks[candidate_index].append(track_index)
                track_candidates[track_index].append(candidate_index)
    persons = []
    presence_only = []
    quarantined = []
    for candidate_index, candidate in enumerate(candidates):
        track_indices = [
            index
            for index in candidate_tracks.get(candidate_index, [])
            if len(track_candidates[index]) == 1
        ]
        track_group = [projected_tracks[index] for index in track_indices]
        roles = {track["role_proposal"] for track in track_group}
        event_ids = {track["event_id"] for track in track_group if track.get("event_id") is not None}
        role_consistent_group = track_group if len(roles) == 1 and len(event_ids) <= 1 else []
        canonical_track = None
        if role_consistent_group:
            center_x = (float(candidate["bbox_xyxy"][0]) + float(candidate["bbox_xyxy"][2])) / 2.0
            center_y = (float(candidate["bbox_xyxy"][1]) + float(candidate["bbox_xyxy"][3])) / 2.0
            canonical_track = min(
                role_consistent_group,
                key=lambda track: (
                    (float(track["projected_track_center_uv"][0]) - center_x) ** 2
                    + (float(track["projected_track_center_uv"][1]) - center_y) ** 2,
                    int(track["published_track_id"]),
                ),
            )
        two_visual_signals = len(candidate["visual_signals"]) == 2
        stable_visual_track = candidate.get("proposal_track_id") in stable_visual_track_ids
        presence_accepted = two_visual_signals or canonical_track is not None or stable_visual_track
        if not presence_accepted:
            quarantined.append({
                **candidate,
                "truth_route_hit": truth_route_hit(
                    candidate["bbox_xyxy"], annotation_route_truth, route_margin_fraction
                ),
                "reason": "single_visual_signal_without_metric_track_or_stable_annotation_track",
            })
            continue
        if canonical_track is None:
            route_hit = truth_route_hit(candidate["bbox_xyxy"], annotation_route_truth, route_margin_fraction)
            image_role_evaluable = (
                route_hit is False and (two_visual_signals or stable_visual_track)
            ) or (route_hit is True and stable_visual_track)
            if image_role_evaluable:
                visual_person_id = candidate.get("proposal_track_id") or (
                    f"{left['sequence_id']}:visual-frame-{left['frame_id']}-{candidate_index}"
                )
                persons.append({
                    "person_id": visual_person_id,
                    "published_track_id": None,
                    "published_track_alias_ids": [],
                    "bbox_xyxy": candidate["bbox_xyxy"],
                    "visual_signals": candidate["visual_signals"],
                    "visual_confidences": candidate["visual_confidences"],
                    "visual_iou": candidate["visual_iou"],
                    "visual_proposal_track_id": candidate.get("proposal_track_id"),
                    "projected_track_center_uv": None,
                    "projected_ground_uv": None,
                    "route_distance_m": None,
                    "future_track_route_distance_m": None,
                    "role": "route_intersecting" if route_hit else "adjacent_safe",
                    "event_id": None,
                    "role_authority": "visual_person_against_actual_future_route_hit_kernel",
                })
                continue
            presence_only.append({
                **candidate,
                "truth_route_hit": route_hit,
                "reason_role_unknown": "visual_consensus_without_metric_track_on_or_unknown_actual_future_route",
            })
            continue
        persons.append({
            "person_id": f"{left['sequence_id']}:published-track-{canonical_track['published_track_id']}",
            "published_track_id": canonical_track["published_track_id"],
            "published_track_alias_ids": sorted(track["published_track_id"] for track in role_consistent_group),
            "bbox_xyxy": candidate["bbox_xyxy"],
            "visual_signals": candidate["visual_signals"],
            "visual_confidences": candidate["visual_confidences"],
            "visual_iou": candidate["visual_iou"],
            "visual_proposal_track_id": candidate.get("proposal_track_id"),
            "projected_track_center_uv": canonical_track["projected_track_center_uv"],
            "projected_ground_uv": canonical_track["projected_ground_uv"],
            "route_distance_m": canonical_track["route_distance_m"],
            "future_track_route_distance_m": canonical_track["future_track_route_distance_m"],
            "role": canonical_track["role_proposal"],
            "event_id": next(iter(event_ids)) if event_ids else None,
        })
    ambiguous_track_ids = sorted(
        projected_tracks[index]["published_track_id"]
        for index, candidate_indices in track_candidates.items()
        if len(candidate_indices) != 1
    )
    unmatched_track_ids = sorted(
        projected_tracks[index]["published_track_id"]
        for index in range(len(projected_tracks))
        if not track_candidates.get(index)
    )
    unresolved_projected_route_relevant_track_ids = sorted(
        projected_tracks[index]["published_track_id"]
        for index in range(len(projected_tracks))
        if (
            len(track_candidates.get(index, [])) != 1
            and projected_tracks[index].get("role_proposal")
            in {"route_intersecting", "approaching_route"}
        )
    )
    route_relevant_person_truth_complete = (
        projected.get("status") == "known"
        and annotation_route_truth is not None
        and annotation_route_truth.get("status") == "known"
        and not any(row.get("truth_route_hit") is not False for row in presence_only)
        and not any(row.get("truth_route_hit") is not False for row in quarantined)
        and not unresolved_projected_route_relevant_track_ids
    )
    return {
        "source_id": left["source_id"],
        "sequence_id": left["sequence_id"],
        "frame_id": left["frame_id"],
        "source_capture_timestamp_ns": left["source_capture_timestamp_ns"],
        "image_sha256": left["image_sha256"],
        "projected_role_status": projected.get("status"),
        "persons": persons,
        "presence_only_role_unknown": presence_only,
        "quarantined_visual_candidates": quarantined,
        "ambiguous_projected_track_ids": ambiguous_track_ids,
        "unmatched_projected_track_ids": unmatched_track_ids,
        "unresolved_projected_route_relevant_track_ids": unresolved_projected_route_relevant_track_ids,
        "route_relevant_person_truth_complete": route_relevant_person_truth_complete,
        "full_person_role_complete": (
            projected.get("status") == "known"
            and not presence_only
            and not quarantined
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--pass-b", required=True, type=Path)
    parser.add_argument("--pass-c", required=True, type=Path)
    parser.add_argument("--projected-track-role", required=True, type=Path)
    parser.add_argument("--route-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replacement", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite fused holdout truth proposal")
    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    presence = config["sealed_holdout"]["holdout_truth_freeze_contract"]["all_person_presence"]
    replacement = load_json(args.replacement) if args.replacement else None
    planned_output = (
        replacement["planned_outputs"]["fusion"]
        if replacement
        else presence["planned_fusion_output_path"]
    )
    if args.output.resolve() != (repo / planned_output).resolve():
        raise RuntimeError("fusion output differs from preregistration")
    pass_b = load_json(args.pass_b)
    pass_c = load_json(args.pass_c)
    projected = load_json(args.projected_track_role)
    route_ledger = load_json(args.route_ledger)
    for payload in (pass_b, pass_c, projected, route_ledger):
        if payload.get("candidate_outputs_executed") is not False:
            raise RuntimeError("truth fusion input exposed candidate outputs")
        if replacement and payload.get("replacement_preregistration_sha256") != sha256_file(args.replacement):
            raise RuntimeError("truth fusion replacement input binding mismatch")
    if pass_b.get("pass_id") != "HOLDOUT_PERSON_PASS_B_YOLOV8N" or pass_c.get("pass_id") != "HOLDOUT_PERSON_PASS_C_YOLO11X":
        raise RuntimeError("visual proposal pass identities drifted")
    projected_by_frame = {}
    truth_route_by_frame = {}
    raw_published_event_proposal_count = 0
    for source in projected["sources"]:
        for sequence in source["sequences"]:
            raw_published_event_proposal_count += len(sequence["event_proposals"])
            for frame in sequence["frames"]:
                key = (source["source_id"], sequence["sequence_id"], frame["frame_id"])
                projected_by_frame[key] = {"source_id": key[0], "sequence_id": key[1], **frame}
    for source in route_ledger["sources"]:
        for sequence in source["sequences"]:
            for route in sequence["route_truth_annotation_only"]:
                truth_route_by_frame[(source["source_id"], sequence["sequence_id"], route["frame_id"])] = route
    if len(pass_b["frames"]) != len(pass_c["frames"]) or len(pass_b["frames"]) != len(projected_by_frame):
        raise RuntimeError("visual and projected proposal frame coverage mismatch")
    acceptance_confidence = float(presence["proposal_acceptance_confidence"])
    minimum_iou = float(presence["visual_consensus_iou_min"])
    tracker_frames = []
    tracker_proposals: dict[str, list[dict[str, Any]]] = {}
    candidates_by_frame: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for left, right in zip(pass_b["frames"], pass_c["frames"], strict=True):
        key = (left["source_id"], left["sequence_id"], left["frame_id"])
        if key != (right["source_id"], right["sequence_id"], right["frame_id"]):
            raise RuntimeError("visual proposal frame order mismatch")
        candidates = frame_candidates(
            left["person_proposals"], right["person_proposals"], minimum_iou, acceptance_confidence
        )
        tracker_key = "|".join(key)
        for index, candidate in enumerate(candidates):
            candidate["candidate_token"] = f"{tracker_key}|{index:04d}"
        candidates_by_frame[key] = candidates
        tracker_proposals[tracker_key] = candidates
        tracker_frames.append({
            "source_id": key[0],
            "blind_window_id": key[1],
            "frame_id": key[2],
            "image_sha256": tracker_key,
        })
    tracker_config = config["seen_truth_proposal_protocol"]["fusion"]["proposal_identity_tracker"]
    tracked, track_summaries = run_bytetrack(
        tracker_frames,
        tracker_proposals,
        model_name="holdout_visual_annotation",
        tracker_config=tracker_config,
    )
    token_to_track_id = {
        candidate["candidate_token"]: candidate["proposal_track_id"]
        for candidates in tracked.values()
        for candidate in candidates
    }
    for candidates in candidates_by_frame.values():
        for candidate in candidates:
            candidate["proposal_track_id"] = token_to_track_id.get(candidate["candidate_token"])
    minimum_temporal_support = int(presence["minimum_temporal_presence_support_frames"])
    stable_visual_track_ids = {
        row["proposal_track_id"]
        for row in track_summaries
        if row["frame_count"] >= minimum_temporal_support
    }
    frames = []
    for left, right in zip(pass_b["frames"], pass_c["frames"], strict=True):
        key = (left["source_id"], left["sequence_id"], left["frame_id"])
        if key not in projected_by_frame:
            raise RuntimeError(f"projected proposal frame missing: {key}")
        if key not in truth_route_by_frame:
            raise RuntimeError(f"annotation route truth frame missing: {key}")
        frames.append(fuse_frame(
            left,
            right,
            projected_by_frame[key],
            minimum_iou,
            acceptance_confidence,
            truth_route_by_frame[key],
            float(config["frozen_axes"]["route_point_margin_fraction"]),
            candidates_by_frame[key],
            stable_visual_track_ids,
        ))
    event_proposals = derive_visible_metric_event_proposals(frames)
    quarantined_event_ids = {
        person["event_id"]
        for frame in frames
        if not frame["full_person_role_complete"]
        for person in frame["persons"]
        if person.get("event_id") is not None
    }
    payload = {
        "schema": "blindassist_crowdbot_holdout_person_role_fusion_r1",
        "authority": "candidate_blind_model_proxy_truth_proposal_not_human_or_production_truth",
        "candidate_outputs_executed": False,
        "app_detector_or_event_outputs_exposed": False,
        "config_sha256": sha256_file(args.config),
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "input_bindings": {
            "pass_b_sha256": sha256_file(args.pass_b),
            "pass_c_sha256": sha256_file(args.pass_c),
            "projected_track_role_sha256": sha256_file(args.projected_track_role),
            "route_ledger_sha256": sha256_file(args.route_ledger),
        },
        "frame_count": len(frames),
        "full_person_role_complete_frame_count": sum(row["full_person_role_complete"] for row in frames),
        "route_relevant_person_truth_complete_frame_count": sum(
            row["route_relevant_person_truth_complete"] for row in frames
        ),
        "accepted_metric_person_frame_count": sum(len(row["persons"]) for row in frames),
        "annotation_visual_track_count": len(track_summaries),
        "stable_annotation_visual_track_count": len(stable_visual_track_ids),
        "presence_only_role_unknown_count": sum(len(row["presence_only_role_unknown"]) for row in frames),
        "quarantined_visual_candidate_count": sum(len(row["quarantined_visual_candidates"]) for row in frames),
        "raw_published_event_proposal_count_ignored_for_truth": raw_published_event_proposal_count,
        "event_proposals": event_proposals,
        "quarantined_event_ids_due_incomplete_person_role_frames": sorted(quarantined_event_ids),
        "frames": frames,
        "production_authority": False,
        "candidate_h2_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "HOLDOUT_PERSON_ROLE_FUSION_MATERIALIZED",
        "frame_count": payload["frame_count"],
        "full_person_role_complete_frame_count": payload["full_person_role_complete_frame_count"],
        "accepted_metric_person_frame_count": payload["accepted_metric_person_frame_count"],
        "presence_only_role_unknown_count": payload["presence_only_role_unknown_count"],
        "quarantined_visual_candidate_count": payload["quarantined_visual_candidate_count"],
        "output_sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
