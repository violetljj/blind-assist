from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = area(first) + area(second) - overlap
    return overlap / union if union else 0.0


def center_distance(first: list[float], second: list[float]) -> float:
    ax, ay = (first[0] + first[2]) / 2, (first[1] + first[3]) / 2
    bx, by = (second[0] + second[2]) / 2, (second[1] + second[3]) / 2
    scale = max(
        first[2] - first[0], first[3] - first[1],
        second[2] - second[0], second[3] - second[1], 1.0,
    )
    return math.hypot(ax - bx, ay - by) / scale


def filter_window(rows: list[dict], seed_confidence: float) -> tuple[list[list[dict]], list[dict]]:
    tracks: list[dict] = []
    active: list[int] = []
    for frame_index, row in enumerate(rows):
        candidates = row["all_person_candidates"]
        possible: list[tuple[float, float, int, int]] = []
        for track_index in active:
            track = tracks[track_index]
            gap = frame_index - track["last_frame"]
            if gap > 3:
                continue
            for candidate_index, candidate in enumerate(candidates):
                overlap = iou(track["last_box"], candidate["bbox_xyxy"])
                distance = center_distance(track["last_box"], candidate["bbox_xyxy"])
                if overlap >= 0.02 or distance <= 1.15:
                    possible.append((-overlap, distance + 0.15 * (gap - 1), track_index, candidate_index))
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        for _, _, track_index, candidate_index in sorted(possible):
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            candidate = candidates[candidate_index]
            track = tracks[track_index]
            track["members"].append((frame_index, candidate_index))
            track["last_frame"] = frame_index
            track["last_box"] = candidate["bbox_xyxy"]
            track["max_confidence"] = max(track["max_confidence"], float(candidate["proposal_confidence"]))
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidates:
                continue
            tracks.append({
                "members": [(frame_index, candidate_index)],
                "last_frame": frame_index,
                "last_box": candidate["bbox_xyxy"],
                "max_confidence": float(candidate["proposal_confidence"]),
            })
            used_tracks.add(len(tracks) - 1)
        active = [
            index for index, track in enumerate(tracks)
            if frame_index - track["last_frame"] <= 3
        ]
    accepted = {
        index for index, track in enumerate(tracks)
        if track["max_confidence"] >= seed_confidence
    }
    kept: list[list[dict]] = [[] for _ in rows]
    summaries: list[dict] = []
    for track_index, track in enumerate(tracks):
        if track_index not in accepted:
            continue
        for frame_index, candidate_index in track["members"]:
            kept[frame_index].append(rows[frame_index]["all_person_candidates"][candidate_index])
        summaries.append({
            "track_id": f"annotation_person_track_{track_index:03d}",
            "frame_count": len(track["members"]),
            "first_frame": rows[track["members"][0][0]]["frame_id"],
            "last_frame": rows[track["members"][-1][0]]["frame_id"],
            "max_proposal_confidence": round(track["max_confidence"], 6),
        })
    return kept, summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-confidence", type=float, default=0.10)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    payload["schema"] = "blindassist_ustrf_detector_target_truth_filtered_candidate_v1"
    payload["authority"] = "candidate_only_requires_second_visual_review_before_truth_freeze"
    payload["negative_track_filter"] = {
        "proposal_role": "annotation_only_never_detector_selection_credit",
        "seed_confidence": args.seed_confidence,
        "max_frame_gap": 3,
        "association": "iou_ge_0.02_or_normalized_center_distance_le_1.15",
    }
    for source in payload["sources"]:
        for window in source["windows"]:
            if window["window_type"] != "negative":
                continue
            kept, tracks = filter_window(window["frames"], args.seed_confidence)
            for row, accepted in zip(window["frames"], kept, strict=True):
                row["unfiltered_all_person_candidates"] = row.pop("all_person_candidates")
                row["all_person_candidates"] = accepted
                row["confirmed_absent_candidate"] = not accepted
            window["accepted_annotation_tracks"] = tracks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seed_confidence": args.seed_confidence}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
