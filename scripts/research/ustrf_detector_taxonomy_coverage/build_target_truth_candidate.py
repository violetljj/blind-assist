from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def iou(first: list[float], second: list[float]) -> float:
    overlap = intersection(first, second)
    union = area(first) + area(second) - overlap
    return overlap / union if union > 0 else 0.0


def deduplicate(proposals: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for proposal in sorted(proposals, key=lambda row: -float(row["proposal_confidence"])):
        box = proposal["bbox_xyxy"]
        duplicate = False
        for other in kept:
            overlap = intersection(box, other["bbox_xyxy"])
            smaller = min(area(box), area(other["bbox_xyxy"]))
            if iou(box, other["bbox_xyxy"]) >= 0.45 or (smaller > 0 and overlap / smaller >= 0.80):
                duplicate = True
                break
        if not duplicate:
            kept.append(proposal)
    return kept


def point_box_distance(point: list[float], box: list[float]) -> float:
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def select_anchor(rows: list[dict], alertable: int, cleared: int) -> tuple[int, int]:
    choices: list[tuple[float, float, int, int]] = []
    for row_index, row in enumerate(rows):
        frame = int(row["frame_id"])
        if frame < alertable or frame > cleared or row.get("route_status") != "known" or not row.get("route_uv"):
            continue
        for box_index, proposal in enumerate(row["all_person_candidates"]):
            distance = point_box_distance(row["route_uv"], proposal["bbox_xyxy"])
            choices.append((distance, -float(proposal["proposal_confidence"]), row_index, box_index))
    if not choices:
        raise ValueError("target event has no route-bound person anchor candidate")
    _, _, row_index, box_index = min(choices)
    return row_index, box_index


def association_cost(previous: list[float], candidate: list[float]) -> tuple[float, float]:
    overlap = iou(previous, candidate)
    px, py = center(previous)
    cx, cy = center(candidate)
    scale = max(previous[2] - previous[0], previous[3] - previous[1], 1.0)
    return overlap, math.hypot(px - cx, py - cy) / scale


def extend_track(rows: list[dict], start: int, step: int, initial_box: list[float], assignments: dict[int, int]) -> None:
    previous = initial_box
    misses = 0
    index = start + step
    while 0 <= index < len(rows):
        candidates = rows[index]["all_person_candidates"]
        ranked: list[tuple[float, float, int]] = []
        for box_index, candidate in enumerate(candidates):
            overlap, distance = association_cost(previous, candidate["bbox_xyxy"])
            ranked.append((-overlap, distance, box_index))
        if ranked:
            neg_overlap, distance, box_index = min(ranked)
            overlap = -neg_overlap
            if overlap >= 0.02 or distance <= 1.25 + 0.35 * misses:
                assignments[index] = box_index
                previous = candidates[box_index]["bbox_xyxy"]
                misses = 0
            else:
                misses += 1
        else:
            misses += 1
        if misses > 8:
            break
        index += step


def interpolate_short_gaps(rows: list[dict], assignments: dict[int, int], max_gap: int = 5) -> dict[int, list[float]]:
    interpolated: dict[int, list[float]] = {}
    known = sorted(assignments)
    for left, right in zip(known, known[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap:
            continue
        first = rows[left]["all_person_candidates"][assignments[left]]["bbox_xyxy"]
        second = rows[right]["all_person_candidates"][assignments[right]]["bbox_xyxy"]
        for offset in range(1, gap + 1):
            ratio = offset / (gap + 1)
            interpolated[left + offset] = [round(a + (b - a) * ratio, 3) for a, b in zip(first, second)]
    return interpolated


def render_page(rows: list[dict], indices: list[int], output: Path, target: bool) -> None:
    cell_w, cell_h = 320, 200
    cols = 5
    canvas = 255 * __import__("numpy").ones((math.ceil(len(indices) / cols) * cell_h, cols * cell_w, 3), dtype="uint8")
    for position, row_index in enumerate(indices):
        row = rows[row_index]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise ValueError(f"cannot read review image: {row['image_path']}")
        scale = min(cell_w / image.shape[1], (cell_h - 20) / image.shape[0])
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        ox = (position % cols) * cell_w
        oy = (position // cols) * cell_h + 20
        canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        for candidate in row["all_person_candidates"]:
            box = [int(round(value * scale)) for value in candidate["bbox_xyxy"]]
            cv2.rectangle(canvas, (ox + box[0], oy + box[1]), (ox + box[2], oy + box[3]), (0, 180, 0), 1)
        if row.get("route_uv"):
            x, y = [int(round(value * scale)) for value in row["route_uv"]]
            cv2.circle(canvas, (ox + x, oy + y), 4, (255, 0, 255), -1)
        if target and row.get("target_person_bbox_xyxy"):
            box = [int(round(value * scale)) for value in row["target_person_bbox_xyxy"]]
            cv2.rectangle(canvas, (ox + box[0], oy + box[1]), (ox + box[2], oy + box[3]), (0, 0, 255), 2)
        label = f"{row['frame_id']} {row.get('target_visible_state', '')}"
        cv2.putText(canvas, label[:42], (ox + 3, oy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise ValueError(f"failed to write review page: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite candidate truth output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    review = read_json(args.review_bundle)
    output_sources: list[dict] = []
    review_pages: list[dict] = []
    for source in review["sources"]:
        output_windows: list[dict] = []
        for window in source["windows"]:
            rows = []
            for original in window["frames"]:
                row = {key: value for key, value in original.items() if key != "review"}
                row["all_person_candidates"] = deduplicate(original["person_proposals"])
                rows.append(row)
            if window["truth_role"] == "target_event":
                event = window["event_lifecycle"]
                anchor_row, anchor_box = select_anchor(rows, int(event["alertable_frame"]), int(event["passed_or_cleared_frame"]))
                assignments = {anchor_row: anchor_box}
                initial = rows[anchor_row]["all_person_candidates"][anchor_box]["bbox_xyxy"]
                extend_track(rows, anchor_row, -1, initial, assignments)
                extend_track(rows, anchor_row, 1, initial, assignments)
                interpolated = interpolate_short_gaps(rows, assignments)
                for index, row in enumerate(rows):
                    frame = int(row["frame_id"])
                    in_lifecycle = int(event["alertable_frame"]) <= frame <= int(event["passed_or_cleared_frame"])
                    if index in assignments:
                        row["target_person_bbox_xyxy"] = row["all_person_candidates"][assignments[index]]["bbox_xyxy"]
                        row["target_visible_state"] = "visible_annotation_proposal"
                        row["target_box_origin"] = "annotation_model_track"
                    elif index in interpolated:
                        row["target_person_bbox_xyxy"] = interpolated[index]
                        row["target_visible_state"] = "visible_interpolated_review_required"
                        row["target_box_origin"] = "short_gap_interpolation"
                    else:
                        row["target_person_bbox_xyxy"] = None
                        row["target_visible_state"] = "unresolved_review_required" if in_lifecycle else "outside_scored_lifecycle"
                        row["target_box_origin"] = None
                review_indices = sorted(set(
                    list(range(0, len(rows), 5))
                    + [anchor_row]
                    + [index for index, row in enumerate(rows) if row["target_visible_state"] in {"visible_interpolated_review_required", "unresolved_review_required"}]
                ))
                target = True
            else:
                for row in rows:
                    row["confirmed_absent_candidate"] = len(row["all_person_candidates"]) == 0
                review_indices = sorted(set(
                    [round(value) for value in __import__("numpy").linspace(0, len(rows) - 1, min(25, len(rows)))]
                    + [index for index, row in enumerate(rows) if row["all_person_candidates"]]
                ))
                target = False
            pages = []
            for page_number, offset in enumerate(range(0, len(review_indices), 25)):
                relative = Path("review-pages") / source["source_name"] / f"{window['window_id']}-{page_number:03d}.jpg"
                render_page(rows, review_indices[offset:offset + 25], args.output_dir / relative, target)
                pages.append({"path": relative.as_posix(), "sha256": sha256(args.output_dir / relative)})
            review_pages.append({"source_id": source["source_id"], "window_id": window["window_id"], "pages": pages})
            output_windows.append({key: value for key, value in window.items() if key != "frames"} | {"frames": rows})
        output_sources.append({key: value for key, value in source.items() if key != "windows"} | {"windows": output_windows})
    payload = {
        "schema": "blindassist_ustrf_detector_target_truth_candidate_v1",
        "authority": "candidate_only_requires_visual_review_before_truth_freeze",
        "baseline_detector_outputs_accessed": False,
        "review_bundle_sha256": sha256(args.review_bundle),
        "sources": output_sources,
        "review_pages": review_pages,
    }
    output = args.output_dir / "target_truth_candidate.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(output_sources), "windows": len(review_pages), "pages": sum(len(row["pages"]) for row in review_pages), "sha256": sha256(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
