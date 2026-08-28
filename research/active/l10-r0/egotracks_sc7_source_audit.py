from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


SCHEMA = "blindassist_l10_sc7_egotracks_source_audit_v1"
MIN_TRACK_BOXES = 30
MIN_GAP_FRAMES = 5
MIN_POST_GAP_BOXES = 3
BUILDING_DOOR_TERMS = re.compile(r"\b(door|doorway|entrance|entry\s*door|gate)\b", re.I)
NON_BUILDING_DOOR_TERMS = re.compile(
    r"\b(cabinet|cupboard|fridge|refrigerator|oven|microwave|dishwasher|washer|drawer)\b",
    re.I,
)


@dataclass(frozen=True)
class EligibleTrack:
    video_uid: str
    clip_uid: str
    annotation_index: int
    query_set_id: str
    object_title: str
    track_box_count: int
    first_frame: int
    last_frame: int
    gap_count: int
    max_gap_frames: int
    first_reentry_frame: int
    visual_crop_frame: int

    @property
    def cohort_id(self) -> str:
        raw = (
            f"{self.video_uid}|{self.clip_uid}|{self.annotation_index}|"
            f"{self.query_set_id}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_building_door(title: str) -> bool:
    return bool(BUILDING_DOOR_TERMS.search(title)) and not bool(
        NON_BUILDING_DOOR_TERMS.search(title)
    )


def _visible_runs(frames: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    gaps: list[int] = []
    reentries: list[tuple[int, int]] = []
    for index in range(1, len(frames)):
        missing = frames[index] - frames[index - 1] - 1
        if missing < MIN_GAP_FRAMES:
            continue
        post = 1
        for follow in range(index + 1, len(frames)):
            if frames[follow] != frames[follow - 1] + 1:
                break
            post += 1
        gaps.append(missing)
        reentries.append((frames[index], post))
    return gaps, reentries


def iter_query_sets(payload: dict[str, Any]) -> Iterator[tuple[str, str, int, str, dict[str, Any]]]:
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("EgoTracks annotation must contain a videos array")
    for video in videos:
        video_uid = str(video["video_uid"])
        for clip in video.get("clips", []):
            clip_uid = str(clip["clip_uid"])
            for annotation_index, annotation in enumerate(clip.get("annotations", [])):
                query_sets = annotation.get("query_sets", {})
                if not isinstance(query_sets, dict):
                    continue
                for query_set_id, query_set in query_sets.items():
                    if isinstance(query_set, dict):
                        yield video_uid, clip_uid, annotation_index, str(query_set_id), query_set


def eligible_track(
    video_uid: str,
    clip_uid: str,
    annotation_index: int,
    query_set_id: str,
    query_set: dict[str, Any],
) -> EligibleTrack | None:
    if not query_set.get("is_valid", False):
        return None
    title = str(query_set.get("object_title", "")).strip()
    if not _is_building_door(title):
        return None
    visual_crop = query_set.get("visual_crop")
    track = query_set.get("lt_track")
    if not isinstance(visual_crop, dict) or not isinstance(track, list):
        return None
    frames = sorted(
        {
            int(box["frame_number"])
            for box in track
            if isinstance(box, dict) and "frame_number" in box
        }
    )
    if len(frames) < MIN_TRACK_BOXES:
        return None
    gaps, reentries = _visible_runs(frames)
    valid_reentries = [row for row in reentries if row[1] >= MIN_POST_GAP_BOXES]
    if not gaps or not valid_reentries:
        return None
    return EligibleTrack(
        video_uid=video_uid,
        clip_uid=clip_uid,
        annotation_index=annotation_index,
        query_set_id=query_set_id,
        object_title=title,
        track_box_count=len(frames),
        first_frame=frames[0],
        last_frame=frames[-1],
        gap_count=len(gaps),
        max_gap_frames=max(gaps),
        first_reentry_frame=valid_reentries[0][0],
        visual_crop_frame=int(visual_crop["frame_number"]),
    )


def audit(annotation_path: Path, split: str, limit: int) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    eligible: list[EligibleTrack] = []
    seen_titles: Counter[str] = Counter()
    query_set_count = 0
    valid_query_set_count = 0
    building_door_query_set_count = 0
    for row in iter_query_sets(payload):
        query_set_count += 1
        query_set = row[-1]
        if query_set.get("is_valid", False):
            valid_query_set_count += 1
        title = str(query_set.get("object_title", "")).strip()
        if _is_building_door(title):
            building_door_query_set_count += 1
            seen_titles[title.casefold()] += 1
        track = eligible_track(*row)
        if track is not None:
            eligible.append(track)
    cohort = eligible[:limit]
    return {
        "schema": SCHEMA,
        "stage": "L10-SC7-G0-REAL-EGOTRACKS-DOOR",
        "split": split,
        "annotation_path": str(annotation_path.resolve()),
        "annotation_sha256": _sha256(annotation_path),
        "selection_policy": (
            f"first_{limit}_source_order_valid_building_door_tracks_with_"
            f">={MIN_TRACK_BOXES}_boxes_gap>={MIN_GAP_FRAMES}_frames_"
            f"post_gap>={MIN_POST_GAP_BOXES}_boxes"
        ),
        "query_set_count": query_set_count,
        "valid_query_set_count": valid_query_set_count,
        "building_door_query_set_count": building_door_query_set_count,
        "eligible_track_count": len(eligible),
        "building_door_title_counts": dict(sorted(seen_titles.items())),
        "cohort_count": len(cohort),
        "cohort": [{"cohort_id": item.cohort_id, **asdict(item)} for item in cohort],
        "claim_ceiling": (
            "Source admission only. No appearance model, belief, active control, "
            "arrival, product, user-benefit, or safety claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.annotations, args.split, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
