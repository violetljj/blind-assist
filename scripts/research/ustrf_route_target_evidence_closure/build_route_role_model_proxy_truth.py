from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from contract import load_json, sha256_file, validate_prereg, validate_role_truth


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_route_predictions(payload: dict[str, Any], source_id: str) -> dict[str, dict[str, Any]]:
    for source in payload["sources"]:
        if source["source_id"] == source_id:
            return {row["frame_id"]: row for row in source["route_predictions"]}
    raise ValueError(f"route source missing: {source_id}")


def median_valid_depth(depth: np.ndarray, scale: float, x1: int, y1: int, x2: int, y2: int, minimum: int) -> float | None:
    patch = depth[max(0, y1):min(depth.shape[0], y2), max(0, x1):min(depth.shape[1], x2)]
    if patch.size == 0:
        return None
    values = patch[np.isfinite(patch) & (patch > 0)]
    if values.size < minimum:
        return None
    return float(np.median(values)) / scale


def backproject(u: float, v: float, z: float, intrinsics: list[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(value) for value in intrinsics]
    return np.asarray([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def receipt_id(source_id: str, route: dict[str, Any]) -> str:
    raw = f"{source_id}|{route['frame_id']}|{route.get('predicted_at_s')}|{route.get('uv')}"
    return "route_receipt_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite route-role truth: {args.output}")

    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    truth_config = config["route_role_truth"]
    geometry = truth_config["model_proxy_geometry_contract"]
    expected_output = repo / geometry["planned_output_path"]
    if args.output.resolve() != expected_output.resolve():
        raise ValueError("route-role output path differs from preregistration")
    identity_binding = config["seen_truth_proposal_protocol"]["fusion"]["identity_adjudication_output"]
    identity_path = repo / identity_binding["path"]
    identity = load_json(identity_path)
    scorer_binding = load_json(repo / config["seen_truth_proposal_protocol"]["isolated_scorer_binding"]["path"])
    scorer_windows = {
        window["blind_window_id"]: window
        for source in scorer_binding["sources"]
        for window in source["windows"]
    }

    source_context: dict[str, dict[str, Any]] = {}
    for source_id, bindings in config["seen_inputs"]["sources"].items():
        frames_path = repo / bindings["frames"]["path"]
        frames = {row["frame_id"]: row for row in load_jsonl(frames_path)}
        route_payload = load_json(repo / bindings["route"]["path"])
        bundle_payload = load_json(repo / bindings["bundle"]["path"])
        source_context[source_id] = {
            "frames": frames,
            "frames_root": Path(bundle_payload["source_root"]),
            "routes": source_route_predictions(route_payload, source_id),
        }

    @lru_cache(maxsize=64)
    def read_depth(path_text: str) -> np.ndarray:
        depth = cv2.imread(path_text, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"cannot read depth image: {path_text}")
        return depth

    half_width = float(geometry["corridor_width_source"]["half_width_m"])
    person_support = geometry["person_ground_support"]
    route_support = geometry["route_ground_support"]
    trend = geometry["trend_contract"]
    required_observations = int(trend["causal_observation_count"])
    approach_delta = float(trend["approaching_delta_m_min"])
    recede_delta = float(trend["receding_delta_m_min"])
    min_clear = int(config["frozen_axes"]["min_clear_frames"])

    source_episodes: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in source_context}
    quarantine_rows = []
    geometry_counts: dict[str, int] = {}
    for tracklet in identity["tracklets"]:
        if tracklet["adjudication_decision"] == "quarantined_unresolved":
            scorer_window = scorer_windows[tracklet.get("blind_window_id")]
            quarantine_rows.append({
                "proposal_track_id": tracklet["proposal_track_id"],
                "source_id": tracklet["source_id"],
                "blind_window_id": tracklet.get("blind_window_id"),
                "reasons": tracklet["quarantine_reasons"],
                "window_type": scorer_window["window_type"],
                "legacy_event_id": scorer_window["event_id"],
                "critical": scorer_window["critical"],
                "contains_frozen_seed_identity": bool(tracklet["person_identity_hints"]),
            })
            continue
        source_id = tracklet["source_id"]
        scorer_window = scorer_windows[tracklet.get("blind_window_id")]
        context = source_context[source_id]
        by_frame: dict[str, list[dict[str, Any]]] = {}
        for member in tracklet["members"]:
            by_frame.setdefault(member["frame_id"], []).append(member)
        if any(len(rows) != 1 for rows in by_frame.values()):
            quarantine_rows.append({
                "proposal_track_id": tracklet["proposal_track_id"],
                "source_id": source_id,
                "blind_window_id": tracklet.get("blind_window_id"),
                "reasons": ["multiple_person_nodes_same_tracklet_frame"],
                "window_type": scorer_window["window_type"],
                "legacy_event_id": scorer_window["event_id"],
                "critical": scorer_window["critical"],
                "contains_frozen_seed_identity": bool(tracklet["person_identity_hints"]),
            })
            continue

        observations = []
        recent: list[tuple[int, float]] = []
        previous_number: int | None = None
        for frame_id in sorted(by_frame, key=int):
            member = by_frame[frame_id][0]
            frame = context["frames"][frame_id]
            route = context["routes"].get(frame_id, {"frame_id": frame_id, "status": "unknown"})
            frame_number = int(frame_id)
            if previous_number is None or frame_number != previous_number + 1:
                recent = []
            previous_number = frame_number
            base = {
                "frame_id": frame_id,
                "visibility": "visible",
                "bbox_xyxy": member["bbox_xyxy"],
                "source_capture_timestamp_ns": int(round(float(frame["rgb_timestamp_s"]) * 1_000_000_000)),
                "route_status": route.get("status", "unknown"),
                "route_receipt_id": receipt_id(source_id, route) if route.get("status") == "known" else None,
                "route_evidence_age_ms": round(max(0.0, (float(route.get("timestamp_s", frame["rgb_timestamp_s"])) - float(route.get("predicted_at_s", frame["rgb_timestamp_s"]))) * 1000.0), 6) if route.get("status") == "known" else None,
                "geometry_status": "unknown",
                "geometry_unknown_reason": None,
                "route_distance_m": None,
                "raw_relation": "unknown",
                "role": None,
            }
            if route.get("status") != "known" or not isinstance(route.get("uv"), list):
                base["geometry_unknown_reason"] = "route_unknown"
                recent = []
                observations.append(base)
                continue
            depth_path = context["frames_root"] / frame["depth_path"]
            depth = read_depth(str(depth_path))
            height, width = depth.shape[:2]
            x1, y1, x2, y2 = [float(value) for value in member["bbox_xyxy"]]
            if y2 >= height - 1:
                base["geometry_unknown_reason"] = "person_ground_support_truncated"
                recent = []
                observations.append(base)
                continue
            box_width, box_height = x2 - x1, y2 - y1
            px1 = int(np.floor(x1 + box_width * (1.0 - float(person_support["bbox_center_width_fraction"])) / 2.0))
            px2 = int(np.ceil(x2 - box_width * (1.0 - float(person_support["bbox_center_width_fraction"])) / 2.0))
            py1 = int(np.floor(y2 - box_height * float(person_support["bbox_bottom_fraction"])))
            py2 = int(np.ceil(y2))
            scale = float(frame["depth_scale"])
            person_depth = median_valid_depth(depth, scale, px1, py1, px2, py2, int(person_support["minimum_valid_depth_pixels"]))
            route_u, route_v = [float(value) for value in route["uv"]]
            radius = int(route_support["route_uv_radius_px"])
            route_depth = median_valid_depth(depth, scale, int(round(route_u)) - radius, int(round(route_v)) - radius, int(round(route_u)) + radius + 1, int(round(route_v)) + radius + 1, int(route_support["minimum_valid_depth_pixels"]))
            if person_depth is None or route_depth is None:
                base["geometry_unknown_reason"] = "person_or_route_depth_invalid"
                recent = []
                observations.append(base)
                continue
            intrinsics = frame["intrinsics_fx_fy_cx_cy"]
            person_xyz = backproject((x1 + x2) / 2.0, y2, person_depth, intrinsics)
            route_xyz = backproject(route_u, route_v, route_depth, intrinsics)
            distance = float(np.linalg.norm(person_xyz - route_xyz))
            recent.append((frame_number, distance))
            recent = recent[-required_observations:]
            delta = None
            if len(recent) == required_observations and all(recent[index][0] == recent[index - 1][0] + 1 for index in range(1, len(recent))):
                delta = distance - recent[0][1]
            if distance <= half_width:
                raw_relation = "route_intersecting"
            elif delta is not None and delta <= -approach_delta:
                raw_relation = "approaching_route"
            elif delta is not None and delta >= recede_delta:
                raw_relation = "receding"
            else:
                raw_relation = "adjacent_safe"
            base.update({
                "geometry_status": "known",
                "route_distance_m": round(distance, 6),
                "distance_delta_over_three_observations_m": round(delta, 6) if delta is not None else None,
                "raw_relation": raw_relation,
            })
            observations.append(base)

        episode_index = 1
        current_frames: list[dict[str, Any]] = []
        is_frozen_target_tracklet = scorer_window["window_type"] == "positive" and bool(tracklet["person_identity_hints"])
        target_anchors = scorer_window.get("truth_anchors") if is_frozen_target_tracklet else None
        target_clear_number = int(target_anchors["passed_or_cleared_frame"]) if target_anchors else None
        was_active = False
        outside_receding_count = 0
        cleared_latched = False
        last_role: str | None = None

        def finish_episode() -> None:
            nonlocal current_frames, episode_index
            if not current_frames:
                return
            active_frames = [row["frame_id"] for row in current_frames if row.get("role") in ("route_intersecting", "approaching_route")]
            clear_frames = [row["frame_id"] for row in current_frames if row.get("role") == "cleared"]
            person_id = f"{tracklet['proposal_track_id']}_route_episode_{episode_index:03d}"
            is_frozen_target = is_frozen_target_tracklet
            source_episodes[source_id].append({
                "person_id": person_id,
                "proposal_track_id": tracklet["proposal_track_id"],
                "blind_window_id": tracklet.get("blind_window_id"),
                "person_identity_hints": tracklet["person_identity_hints"],
                "is_frozen_target": is_frozen_target,
                "legacy_event_id": scorer_window["event_id"] if is_frozen_target else None,
                "risk_event_id": f"risk_{source_id}_{scorer_window['event_id']}_{episode_index:03d}" if is_frozen_target else f"risk_{source_id}_{person_id}",
                "event_truth": {
                    "first_visible_frame": current_frames[0]["frame_id"],
                    "alertable_start_frame": f"{int(target_anchors['alertable_frame']):06d}" if is_frozen_target and target_anchors else (active_frames[0] if active_frames else None),
                    "clear_frame": f"{target_clear_number:06d}" if is_frozen_target and target_clear_number is not None else (clear_frames[0] if clear_frames else None),
                    "should_alert": True if is_frozen_target else bool(active_frames),
                    "critical": bool(scorer_window["critical"]) if is_frozen_target else False,
                },
                "frames": current_frames,
            })
            episode_index += 1
            current_frames = []

        for observation in observations:
            raw = observation["raw_relation"]
            target_truth_cleared = target_clear_number is not None and int(observation["frame_id"]) >= target_clear_number
            if cleared_latched and not target_truth_cleared and raw in ("route_intersecting", "approaching_route"):
                finish_episode()
                was_active = False
                outside_receding_count = 0
                cleared_latched = False
                last_role = None
            if target_truth_cleared and observation["route_status"] == "known":
                observation["geometry_status"] = "known"
                observation["geometry_unknown_reason"] = None
                observation["role"] = "cleared"
                cleared_latched = True
            elif observation["geometry_status"] != "known":
                observation["role"] = None
                outside_receding_count = 0
            elif cleared_latched:
                observation["role"] = "cleared"
            elif raw == "route_intersecting":
                observation["role"] = "route_intersecting"
                was_active = True
                outside_receding_count = 0
            elif raw == "approaching_route":
                observation["role"] = "approaching_route"
                was_active = True
                outside_receding_count = 0
            elif raw == "receding":
                observation["role"] = "receding"
                if was_active:
                    outside_receding_count += 1
                    if outside_receding_count >= min_clear and not is_frozen_target_tracklet:
                        observation["role"] = "cleared"
                        cleared_latched = True
                else:
                    outside_receding_count = 0
            elif was_active:
                observation["geometry_status"] = "unknown"
                observation["geometry_unknown_reason"] = "outside_after_active_without_receding_evidence"
                observation["role"] = None
                outside_receding_count = 0
            else:
                observation["role"] = "adjacent_safe"
                outside_receding_count = 0
            current_frames.append(observation)
            if observation["role"] is not None:
                last_role = observation["role"]
            geometry_counts[observation["geometry_status"]] = geometry_counts.get(observation["geometry_status"], 0) + 1
        finish_episode()

    payload = {
        "schema": "blindassist_ustrf_route_role_truth_r1",
        "split": "seen_diagnostic_only",
        "authority": "model_proxy_route_role_evidence_not_human_truth_not_candidate_H2_or_production_authority",
        "config_sha256": sha256_file(args.config),
        "identity_adjudication_sha256": sha256_file(identity_path),
        "roles": truth_config["roles"],
        "geometry_status_counts": geometry_counts,
        "quarantined_identity_episode_count": len(quarantine_rows),
        "quarantined_identity_episodes": quarantine_rows,
        "positive_target_coverage": {
            "window_count": sum(window["window_type"] == "positive" for window in scorer_windows.values()),
            "accepted_target_tracklet_count": sum(
                episode["is_frozen_target"]
                for episodes in source_episodes.values()
                for episode in episodes
                if episode["person_id"].endswith("_route_episode_001")
            ),
            "quarantined_target_tracklet_count": sum(
                row["window_type"] == "positive" and row["contains_frozen_seed_identity"]
                for row in quarantine_rows
            ),
        },
        "sources": [
            {"source_id": source_id, "person_episodes": source_episodes[source_id]}
            for source_id in config["seen_diagnostic"]["source_ids"]
        ],
    }
    validate_role_truth(payload, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "person_episode_count": sum(len(rows) for rows in source_episodes.values()),
        "quarantined_identity_episode_count": len(quarantine_rows),
        "geometry_status_counts": geometry_counts,
        "sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
