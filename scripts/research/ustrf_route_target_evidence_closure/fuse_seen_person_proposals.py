from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from contract import load_json, sha256_file, validate_prereg


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = area(first) + area(second) - overlap
    return overlap / union if union else 0.0


def center_distance(first: list[float], second: list[float]) -> float:
    ax, ay = (first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0
    bx, by = (second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0
    scale = max(
        first[2] - first[0], first[3] - first[1],
        second[2] - second[0], second[3] - second[1], 1.0,
    )
    return math.hypot(ax - bx, ay - by) / scale


def greedy_matches(first: list[dict], second: list[dict], minimum_iou: float) -> list[tuple[int, int, float]]:
    candidates = []
    for first_index, left in enumerate(first):
        for second_index, right in enumerate(second):
            overlap = iou(left["bbox_xyxy"], right["bbox_xyxy"])
            if overlap >= minimum_iou:
                candidates.append((-overlap, first_index, second_index))
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches = []
    for negative_overlap, first_index, second_index in sorted(candidates):
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append((first_index, second_index, -negative_overlap))
    return matches


def flatten_review_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in bundle["sources"]:
        for window in source["windows"]:
            for frame in window["frames"]:
                rows.append({
                    "source_id": source["source_id"],
                    "blind_window_id": window["blind_window_id"],
                    **frame,
                })
    return rows


def flatten_pass_a(payload: dict[str, Any]) -> dict[str, list[dict]]:
    result = {}
    for source in payload["sources"]:
        for window in source["windows"]:
            for frame in window["frames"]:
                result[frame["image_sha256"]] = frame["person_proposals"]
    return result


def filter_model_proposals(
    frames: list[dict[str, Any]],
    proposals: dict[str, list[dict]],
    *,
    model_name: str,
    seed_confidence: float,
    max_gap_frames: int,
    association_iou_min: float,
    center_distance_max: float,
) -> tuple[dict[str, list[dict]], list[dict]]:
    tracks: list[dict] = []
    by_window: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for frame in frames:
        by_window[(frame["source_id"], frame["blind_window_id"])].append(frame)
    for (source_id, blind_window_id), source_frames in by_window.items():
        active: list[int] = []
        previous_number = None
        for frame in sorted(source_frames, key=lambda row: int(row["frame_id"])):
            frame_number = int(frame["frame_id"])
            if previous_number is None or frame_number - previous_number > max_gap_frames:
                active = []
            previous_number = frame_number
            candidates = proposals[frame["image_sha256"]]
            possible = []
            for track_index in active:
                track = tracks[track_index]
                gap = frame_number - track["last_frame_number"]
                if gap > max_gap_frames:
                    continue
                for candidate_index, candidate in enumerate(candidates):
                    overlap = iou(track["last_box"], candidate["bbox_xyxy"])
                    distance = center_distance(track["last_box"], candidate["bbox_xyxy"])
                    if overlap >= association_iou_min or distance <= center_distance_max:
                        possible.append((-overlap, distance + 0.15 * (gap - 1), track_index, candidate_index))
            used_tracks: set[int] = set()
            used_candidates: set[int] = set()
            for _, _, track_index, candidate_index in sorted(possible):
                if track_index in used_tracks or candidate_index in used_candidates:
                    continue
                candidate = candidates[candidate_index]
                track = tracks[track_index]
                track["members"].append((frame["image_sha256"], candidate_index))
                track["last_frame_number"] = frame_number
                track["last_box"] = candidate["bbox_xyxy"]
                track["max_confidence"] = max(track["max_confidence"], float(candidate.get("proposal_confidence", 0.0)))
                used_tracks.add(track_index)
                used_candidates.add(candidate_index)
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in used_candidates:
                    continue
                tracks.append({
                    "source_id": source_id,
                    "members": [(frame["image_sha256"], candidate_index)],
                    "last_frame_number": frame_number,
                    "last_box": candidate["bbox_xyxy"],
                    "max_confidence": float(candidate.get("proposal_confidence", 0.0)),
                })
                used_tracks.add(len(tracks) - 1)
            active = [index for index in used_tracks | set(active) if frame_number - tracks[index]["last_frame_number"] <= max_gap_frames]
    kept: dict[str, list[dict]] = {frame["image_sha256"]: [] for frame in frames}
    summaries = []
    accepted_number = 0
    for track in tracks:
        if track["max_confidence"] < seed_confidence:
            continue
        track_id = f"{model_name}_proposal_track_{accepted_number:05d}"
        accepted_number += 1
        for image_sha256, candidate_index in track["members"]:
            kept[image_sha256].append({**proposals[image_sha256][candidate_index], "proposal_track_id": track_id})
        summaries.append({
            "proposal_track_id": track_id,
            "source_id": track["source_id"],
            "frame_count": len(track["members"]),
            "max_confidence": round(track["max_confidence"], 6),
        })
    return kept, summaries


def run_bytetrack(
    frames: list[dict[str, Any]],
    proposals: dict[str, list[dict]],
    *,
    model_name: str,
    tracker_config: dict[str, Any],
) -> tuple[dict[str, list[dict]], list[dict]]:
    import numpy as np
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    class ResultsLike:
        def __init__(self, xywh: Any, conf: Any, cls: Any) -> None:
            self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
            self.conf = np.asarray(conf, dtype=np.float32)
            self.cls = np.asarray(cls, dtype=np.float32)

        def __len__(self) -> int:
            return len(self.conf)

        def __getitem__(self, item: Any) -> "ResultsLike":
            return ResultsLike(self.xywh[item], self.conf[item], self.cls[item])

    args = IterableSimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=float(tracker_config["track_high_thresh"]),
        track_low_thresh=float(tracker_config["track_low_thresh"]),
        new_track_thresh=float(tracker_config["new_track_thresh"]),
        track_buffer=int(tracker_config["track_buffer"]),
        match_thresh=float(tracker_config["match_thresh"]),
        fuse_score=bool(tracker_config["fuse_score"]),
    )
    kept: dict[str, list[dict]] = {frame["image_sha256"]: [] for frame in frames}
    track_members: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_window: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for frame in frames:
        by_window[(frame["source_id"], frame["blind_window_id"])].append(frame)
    for (source_id, blind_window_id), source_frames in by_window.items():
        tracker = BYTETracker(args)
        previous_number = None
        segment_index = 0
        for frame in sorted(source_frames, key=lambda row: int(row["frame_id"])):
            frame_number = int(frame["frame_id"])
            if previous_number is None or frame_number != previous_number + 1:
                tracker = BYTETracker(args)
                segment_index += 1
            previous_number = frame_number
            candidates = proposals[frame["image_sha256"]]
            xywh = []
            confidence = []
            for candidate in candidates:
                x1, y1, x2, y2 = [float(value) for value in candidate["bbox_xyxy"]]
                xywh.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])
                confidence.append(float(candidate.get("proposal_confidence", 0.0)))
            tracked = tracker.update(ResultsLike(xywh, confidence, [0.0] * len(candidates)))
            for row in tracked:
                candidate_index = int(round(float(row[7])))
                raw_track_id = int(round(float(row[4])))
                if not 0 <= candidate_index < len(candidates):
                    raise ValueError("ByteTrack returned an invalid proposal index")
                track_id = (
                    f"{model_name}_{source_id}_{blind_window_id}_segment_{segment_index:03d}"
                    f"_bytetrack_{raw_track_id:05d}"
                )
                kept[frame["image_sha256"]].append({
                    **candidates[candidate_index],
                    "proposal_track_id": track_id,
                })
                track_members[track_id].append((frame["frame_id"], frame["image_sha256"]))
    summaries = [
        {
            "proposal_track_id": track_id,
            "frame_count": len(members),
            "first_frame": members[0][0],
            "last_frame": members[-1][0],
        }
        for track_id, members in sorted(track_members.items())
    ]
    return kept, summaries


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def fuse_frame(frame: dict[str, Any], pass_a: list[dict], pass_b: list[dict], minimum_iou: float) -> list[dict]:
    nodes = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for seed_index, seed in enumerate(frame["person_seed_boxes"]):
        seed_box = seed["bbox_xyxy"]
        match_a = max(
            ((iou(seed_box, proposal["bbox_xyxy"]), index) for index, proposal in enumerate(pass_a) if index not in used_a),
            default=(0.0, -1),
        )
        match_b = max(
            ((iou(seed_box, proposal["bbox_xyxy"]), index) for index, proposal in enumerate(pass_b) if index not in used_b),
            default=(0.0, -1),
        )
        if match_a[0] >= minimum_iou:
            used_a.add(match_a[1])
        if match_b[0] >= minimum_iou:
            used_b.add(match_b[1])
        nodes.append({
            "frame_node_id": f"seed_{seed_index:03d}",
            "bbox_xyxy": seed_box,
            "evidence": "frozen_seed_truth",
            "person_identity_hint": seed.get("person_identity_hint"),
            "pass_a_iou": round(match_a[0], 6),
            "pass_b_iou": round(match_b[0], 6),
            "pass_a_track_id": pass_a[match_a[1]].get("proposal_track_id") if match_a[0] >= minimum_iou else None,
            "pass_b_track_id": pass_b[match_b[1]].get("proposal_track_id") if match_b[0] >= minimum_iou else None,
        })
    remaining_a = [proposal for index, proposal in enumerate(pass_a) if index not in used_a]
    remaining_b = [proposal for index, proposal in enumerate(pass_b) if index not in used_b]
    matches = greedy_matches(remaining_a, remaining_b, minimum_iou)
    matched_a = {first for first, _, _ in matches}
    matched_b = {second for _, second, _ in matches}
    for index, (first_index, second_index, overlap) in enumerate(matches):
        left = remaining_a[first_index]["bbox_xyxy"]
        right = remaining_b[second_index]["bbox_xyxy"]
        nodes.append({
            "frame_node_id": f"dual_{index:03d}",
            "bbox_xyxy": [round((float(a) + float(b)) / 2.0, 3) for a, b in zip(left, right, strict=True)],
            "evidence": "dual_model_consensus",
            "person_identity_hint": None,
            "pass_a_confidence": remaining_a[first_index].get("proposal_confidence"),
            "pass_b_confidence": remaining_b[second_index].get("proposal_confidence"),
            "pass_a_track_id": remaining_a[first_index].get("proposal_track_id"),
            "pass_b_track_id": remaining_b[second_index].get("proposal_track_id"),
            "cross_model_iou": round(overlap, 6),
        })
    for index, proposal in enumerate(remaining_a):
        if index not in matched_a:
            nodes.append({
                "frame_node_id": f"pass_a_only_{index:03d}",
                "bbox_xyxy": proposal["bbox_xyxy"],
                "evidence": "pass_a_only",
                "person_identity_hint": None,
                "pass_a_confidence": proposal.get("proposal_confidence"),
                "pass_a_track_id": proposal.get("proposal_track_id"),
                "pass_b_track_id": None,
            })
    for index, proposal in enumerate(remaining_b):
        if index not in matched_b:
            nodes.append({
                "frame_node_id": f"pass_b_only_{index:03d}",
                "bbox_xyxy": proposal["bbox_xyxy"],
                "evidence": "pass_b_only",
                "person_identity_hint": None,
                "pass_b_confidence": proposal.get("proposal_confidence"),
                "pass_a_track_id": None,
                "pass_b_track_id": proposal.get("proposal_track_id"),
            })
    return nodes


def build_tracklets(frames: list[dict[str, Any]]) -> tuple[list[dict], int]:
    all_nodes: list[dict] = []
    for frame in frames:
        for node in frame["person_nodes"]:
            all_nodes.append({
                "source_id": frame["source_id"],
                "blind_window_id": frame["blind_window_id"],
                "frame_id": frame["frame_id"],
                "image_sha256": frame["image_sha256"],
                **node,
                "association_ambiguous": False,
            })
    union = UnionFind(len(all_nodes))
    lineage_to_nodes: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    seed_hint_to_nodes: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, node in enumerate(all_nodes):
        for key in ("pass_a_track_id", "pass_b_track_id"):
            if node.get(key) is not None:
                lineage_to_nodes[(node["source_id"], node["blind_window_id"], node[key])].append(index)
        if node.get("person_identity_hint") is not None:
            seed_hint_to_nodes[(node["source_id"], node["blind_window_id"], node["person_identity_hint"])].append(index)
    for indices in list(lineage_to_nodes.values()) + list(seed_hint_to_nodes.values()):
        for index in indices[1:]:
            union.union(indices[0], index)
    cross_pairs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for node in all_nodes:
        left, right = node.get("pass_a_track_id"), node.get("pass_b_track_id")
        if left is not None and right is not None:
            cross_pairs[(node["source_id"], node["blind_window_id"], left)].add(right)
            cross_pairs[(node["source_id"], node["blind_window_id"], right)].add(left)
    ambiguous_lineages = {key for key, values in cross_pairs.items() if len(values) > 1}
    for node in all_nodes:
        if any(
            (node["source_id"], node["blind_window_id"], node.get(key)) in ambiguous_lineages
            for key in ("pass_a_track_id", "pass_b_track_id") if node.get(key) is not None
        ):
            node["association_ambiguous"] = True
    components: dict[int, list[dict]] = defaultdict(list)
    for index, node in enumerate(all_nodes):
        components[union.find(index)].append(node)
    tracklets = []
    for track_number, members in enumerate(sorted(components.values(), key=lambda rows: (rows[0]["source_id"], int(rows[0]["frame_id"]), rows[0]["frame_node_id"]))):
        hints = sorted({member["person_identity_hint"] for member in members if member["person_identity_hint"] is not None})
        single_model_count = sum(member["evidence"] in ("pass_a_only", "pass_b_only") for member in members)
        association_ambiguous = any(member["association_ambiguous"] for member in members)
        identity_conflict = len(hints) > 1
        status = "consensus_tracklet" if single_model_count == 0 and not association_ambiguous and not identity_conflict else "third_model_adjudication_required"
        tracklets.append({
            "proposal_track_id": f"proposal_track_{track_number:05d}",
            "source_id": members[0]["source_id"],
            "blind_window_id": members[0]["blind_window_id"],
            "status": status,
            "person_identity_hints": hints,
            "frame_count": len(members),
            "first_frame": min(member["frame_id"] for member in members),
            "last_frame": max(member["frame_id"] for member in members),
            "single_model_node_count": single_model_count,
            "association_ambiguous": association_ambiguous,
            "identity_conflict": identity_conflict,
            "members": sorted(members, key=lambda member: (int(member["frame_id"]), member["frame_node_id"])),
        })
    return tracklets, len(ambiguous_lineages)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite fusion output: {args.output}")
    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    protocol = config["seen_truth_proposal_protocol"]
    review_path = repo / protocol["review_bundle"]["path"]
    pass_a_path = repo / protocol["pass_a"]["path"]
    pass_b_path = repo / protocol["pass_b"]["output_path"]
    review = load_json(review_path)
    pass_a = flatten_pass_a(load_json(pass_a_path))
    pass_b_payload = load_json(pass_b_path)
    pass_b = {frame["image_sha256"]: frame["person_proposals"] for frame in pass_b_payload["frames"]}
    frames = flatten_review_bundle(review)
    image_hashes = {frame["image_sha256"] for frame in frames}
    if set(pass_a) != image_hashes or set(pass_b) != image_hashes:
        raise ValueError("proposal pass frame identities do not match the blind review bundle")
    fusion_config = protocol["fusion"]
    tracker_config = fusion_config["proposal_identity_tracker"]
    pass_a, pass_a_tracks = run_bytetrack(frames, pass_a, model_name="pass_a", tracker_config=tracker_config)
    pass_b, pass_b_tracks = run_bytetrack(frames, pass_b, model_name="pass_b", tracker_config=tracker_config)
    minimum_iou = float(protocol["fusion"]["dual_proposal_match_iou_min"])
    fused_frames = []
    evidence_counts: dict[str, int] = defaultdict(int)
    for frame in frames:
        nodes = fuse_frame(frame, pass_a[frame["image_sha256"]], pass_b[frame["image_sha256"]], minimum_iou)
        for node in nodes:
            evidence_counts[node["evidence"]] += 1
        fused_frames.append({
            "source_id": frame["source_id"],
            "blind_window_id": frame["blind_window_id"],
            "frame_id": frame["frame_id"],
            "image_path": frame["image_path"],
            "image_sha256": frame["image_sha256"],
            "route_status": frame["route_status"],
            "route_uv": frame["route_uv"],
            "person_nodes": nodes,
        })
    tracklets, ambiguous_edges = build_tracklets(fused_frames)
    payload = {
        "schema": "blindassist_ustrf_seen_person_proposal_fusion_r1",
        "authority": "proposal_tracklets_only_not_person_truth_until_third_model_adjudication",
        "config_sha256": sha256_file(args.config),
        "blind_review_bundle_sha256": sha256_file(review_path),
        "pass_a_sha256": sha256_file(pass_a_path),
        "pass_b_sha256": sha256_file(pass_b_path),
        "candidate_alerts_exposed": False,
        "baseline_app_detector_outputs_exposed": False,
        "frame_count": len(fused_frames),
        "frame_node_count": sum(evidence_counts.values()),
        "pass_a_accepted_proposal_track_count": len(pass_a_tracks),
        "pass_b_accepted_proposal_track_count": len(pass_b_tracks),
        "evidence_counts": dict(sorted(evidence_counts.items())),
        "tracklet_count": len(tracklets),
        "consensus_tracklet_count": sum(tracklet["status"] == "consensus_tracklet" for tracklet in tracklets),
        "adjudication_tracklet_count": sum(tracklet["status"] == "third_model_adjudication_required" for tracklet in tracklets),
        "ambiguous_edge_count": ambiguous_edges,
        "pass_a_proposal_tracks": pass_a_tracks,
        "pass_b_proposal_tracks": pass_b_tracks,
        "frames": fused_frames,
        "tracklets": tracklets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "frame_count", "frame_node_count", "evidence_counts", "tracklet_count",
        "pass_a_accepted_proposal_track_count", "pass_b_accepted_proposal_track_count",
        "consensus_tracklet_count", "adjudication_tracklet_count", "ambiguous_edge_count",
    )} | {"sha256": sha256_file(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
