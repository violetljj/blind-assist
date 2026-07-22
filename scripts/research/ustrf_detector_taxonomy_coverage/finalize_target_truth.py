from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def in_intervals(frame: int, intervals: list[list[int]]) -> bool:
    return any(int(start) <= frame <= int(end) for start, end in intervals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--review-pages-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen truth: {args.output}")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    review = json.loads(args.review_decisions.read_text(encoding="utf-8"))
    absent = {
        key: {int(value) for value in values}
        for key, values in review["positive_review"]["not_visible_cleared_frames"].items()
    }
    approved_interpolation = {
        key: {int(value) for value in values}
        for key, values in review["positive_review"]["approved_interpolated_target_frames"].items()
    }
    intervals = review["negative_review"]["person_visible_intervals_inclusive"]
    sources: list[dict] = []
    event_count = negative_window_count = negative_frame_count = confirmed_absent_count = 0
    for source in candidate["sources"]:
        events: list[dict] = []
        negative_windows: list[dict] = []
        for window in source["windows"]:
            if window["window_type"] == "positive":
                event = window["event_lifecycle"]
                key = f"{source['source_id']}/{window['event_id']}"
                alertable = int(event["alertable_frame"])
                cleared = int(event["passed_or_cleared_frame"])
                frames: list[dict] = []
                for row in window["frames"]:
                    frame = int(row["frame_id"])
                    if frame < alertable or frame > cleared:
                        continue
                    if frame in absent.get(key, set()):
                        state, box, origin = "not_visible_cleared", None, "visual_review_override"
                    else:
                        box = row.get("target_person_bbox_xyxy")
                        if box is None:
                            raise ValueError(f"unresolved visible target frame: {key}/{frame}")
                        if "interpolated" in row["target_visible_state"]:
                            if frame not in approved_interpolation.get(key, set()):
                                raise ValueError(f"unapproved target interpolation: {key}/{frame}")
                            state, origin = "visible_reviewed_interpolation", "visual_reviewed_short_gap_interpolation"
                        else:
                            state, origin = "visible_reviewed_bbox", "dual_annotation_visual_review"
                    frames.append({
                        "frame_id": row["frame_id"],
                        "image_sha256": row["image_sha256"],
                        "visible_state": state,
                        "target_bbox_xyxy": box,
                        "bbox_origin": origin,
                    })
                if len(frames) != cleared - alertable + 1:
                    raise ValueError(f"target lifecycle coverage mismatch: {key}")
                events.append({
                    "event_id": window["event_id"],
                    "target_person_identity": f"{key}/target_person",
                    "identity_scope": "event_scoped_unique_no_cross_event_reuse",
                    "critical": bool(window["critical"]),
                    "onset_frame": int(event["onset_frame"]),
                    "alertable_frame": alertable,
                    "passed_or_cleared_frame": cleared,
                    "end_frame": int(event["end_frame"]),
                    "frames": frames,
                })
                event_count += 1
            else:
                window_intervals = intervals.get(window["window_id"])
                if window_intervals is None:
                    raise ValueError(f"negative review interval missing: {window['window_id']}")
                frames: list[dict] = []
                for row in window["frames"]:
                    frame = int(row["frame_id"])
                    visible = in_intervals(frame, window_intervals)
                    candidates = row["all_person_candidates"] if visible else []
                    if visible and not candidates:
                        raise ValueError(f"reviewed person-visible frame lacks bbox: {window['window_id']}/{frame}")
                    boxes = [
                        {
                            "bbox_xyxy": item["bbox_xyxy"],
                            "origin": item.get("annotation_evidence", "dual_annotation_temporal_track"),
                        }
                        for item in candidates
                    ]
                    frames.append({
                        "frame_id": row["frame_id"],
                        "image_sha256": row["image_sha256"],
                        "all_person_boxes": boxes,
                        "confirmed_absent": not visible,
                        "truth_state": "all_person_visible_reviewed" if visible else "confirmed_absent_full_frame_review",
                    })
                    negative_frame_count += 1
                    confirmed_absent_count += int(not visible)
                negative_windows.append({
                    "window_id": window["window_id"],
                    "start_frame": int(window["start_frame"]),
                    "end_frame": int(window["end_frame"]),
                    "frames": frames,
                })
                negative_window_count += 1
        sources.append({
            "source_id": source["source_id"],
            "source_name": source["source_name"],
            "route_input_sha256": source["route_input_sha256"],
            "event_consensus_sha256": source["event_consensus_sha256"],
            "target_events": events,
            "negative_windows": negative_windows,
        })
    pages = sorted(args.review_pages_dir.glob("*-all-frames.jpg"))
    if len(pages) != negative_window_count:
        raise ValueError("negative review page coverage mismatch")
    payload = {
        "schema": "blindassist_ustrf_detector_target_truth_r1",
        "authority": "frozen_benchmark_truth_no_training_or_production_authority",
        "baseline_detector_outputs_used_for_truth": False,
        "baseline_detector_output_identity_or_target_association_exposed_to_review": False,
        "candidate_sha256": sha256(args.candidate),
        "review_decisions_sha256": sha256(args.review_decisions),
        "negative_review_pages": [
            {"path": str(path), "sha256": sha256(path)} for path in pages
        ],
        "event_count": event_count,
        "negative_window_count": negative_window_count,
        "negative_frame_count": negative_frame_count,
        "confirmed_absent_frame_count": confirmed_absent_count,
        "sources": sources,
    }
    if event_count != 15 or negative_window_count != 15:
        raise ValueError("frozen truth must contain 15 positive and 15 negative windows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "event_count": event_count,
        "negative_window_count": negative_window_count,
        "negative_frame_count": negative_frame_count,
        "confirmed_absent_frame_count": confirmed_absent_count,
        "sha256": sha256(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
