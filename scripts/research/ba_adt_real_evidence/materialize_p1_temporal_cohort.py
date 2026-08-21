#!/usr/bin/env python3
"""Materialize a bounded P1 Development cohort from ADT ground truth only."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RGB_STREAM = "214-1"
VISIBLE_RATIO = 0.10
MIN_VISIBLE_RUN = 12
EPISODE_BUDGET_MIN = 12
EPISODE_BUDGET_MAX = 18
SOURCE_BUDGET_MAX = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(zf: zipfile.ZipFile, member: str) -> Iterable[dict[str, str]]:
    with zf.open(member) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))


def contiguous_segments(mask: list[bool]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


@dataclass(frozen=True)
class SourceSpec:
    sequence_id: str
    groundtruth: Path
    rgb_video: Path


@dataclass
class SourceData:
    spec: SourceSpec
    instances: dict[str, dict[str, Any]]
    frame_times: list[int]
    boxes: dict[str, dict[int, dict[str, float]]]
    groundtruth_sha256: str
    rgb_video_sha256: str
    rgb_video_frame_count: int | None
    video_frame_indices: list[int]


@dataclass(frozen=True)
class Proposal:
    source_id: str
    target_uid: str
    start: int
    end: int
    primary_mode: str
    tags: tuple[str, ...]
    quality: tuple[int, int, str]


def parse_source(fields: list[str]) -> SourceSpec:
    if len(fields) != 3 or not all(fields):
        raise argparse.ArgumentTypeError("--source requires SEQUENCE_ID GROUNDTRUTH_ZIP RGB_VIDEO")
    return SourceSpec(fields[0], Path(fields[1]), Path(fields[2]))


def probe_video_timestamps(path: Path) -> list[int]:
    import av

    with av.open(str(path)) as container:
        description = container.metadata.get("description")
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        if not description or len(video_streams) != 1:
            raise ValueError(f"RGB video lacks one timestamped video stream: {path}")
        try:
            timestamps = [int(value) for value in json.loads(description)]
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"RGB video timestamp metadata is invalid: {path}") from error
        if not timestamps or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError(f"RGB video timestamps are empty or non-monotonic: {path}")
        stream_frames = int(video_streams[0].frames)
    if stream_frames and stream_frames != len(timestamps):
        raise ValueError(f"RGB video frame/timestamp count mismatch: {path}")
    return timestamps


def nearest_index(times: list[int], timestamp: int) -> int | None:
    position = bisect.bisect_left(times, timestamp)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
    if not candidates:
        return None
    index = min(candidates, key=lambda item: abs(times[item] - timestamp))
    return index if abs(times[index] - timestamp) <= 20_000_000 else None


def load_source(spec: SourceSpec, *, probe_video: bool = True) -> SourceData:
    if not spec.groundtruth.is_file() or not spec.rgb_video.is_file():
        raise ValueError(f"source files missing for {spec.sequence_id}")
    with zipfile.ZipFile(spec.groundtruth) as zf:
        names = set(zf.namelist())
        required = {"instances.json", "aria_trajectory.csv"}
        if missing := sorted(required - names):
            raise ValueError(f"{spec.sequence_id}: required truth missing: {missing}")
        bbox_member = (
            "2d_bounding_box_with_skeleton.csv"
            if "2d_bounding_box_with_skeleton.csv" in names
            else "2d_bounding_box.csv"
        )
        if bbox_member not in names:
            raise ValueError(f"{spec.sequence_id}: 2D bbox truth missing")
        raw_instances = json.load(zf.open("instances.json"))
        instances = {
            str(uid): value
            for uid, value in raw_instances.items()
            if isinstance(value, dict) and value.get("instance_type", "object") == "object"
        }
        times = [int(row["tracking_timestamp_us"]) * 1000 for row in csv_rows(zf, "aria_trajectory.csv")]
        boxes: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
        for row in csv_rows(zf, bbox_member):
            if row["stream_id"] != RGB_STREAM:
                continue
            timestamp = int(row["timestamp[ns]"])
            index = nearest_index(times, timestamp)
            if index is None:
                continue
            uid = str(row["object_uid"])
            boxes[uid][times[index]] = {
                "visibility_ratio": float(row["visibility_ratio[%]"]),
                "x_min": float(row["x_min[pixel]"]),
                "x_max": float(row["x_max[pixel]"]),
                "y_min": float(row["y_min[pixel]"]),
                "y_max": float(row["y_max[pixel]"]),
            }
    if not times:
        raise ValueError(f"{spec.sequence_id}: no Aria trajectory timestamps")
    video_timestamps = probe_video_timestamps(spec.rgb_video) if probe_video else times
    aligned_times = []
    video_frame_indices = []
    for timestamp in times:
        index = nearest_index(video_timestamps, timestamp)
        if index is not None:
            aligned_times.append(timestamp)
            video_frame_indices.append(index)
    if len(aligned_times) < len(times) - 1:
        raise ValueError(
            f"{spec.sequence_id}: only {len(aligned_times)}/{len(times)} GT timestamps "
            "align to RGB metadata within 20 ms"
        )
    times = aligned_times
    return SourceData(
        spec=spec,
        instances=instances,
        frame_times=times,
        boxes=dict(boxes),
        groundtruth_sha256=sha256(spec.groundtruth),
        rgb_video_sha256=sha256(spec.rgb_video),
        rgb_video_frame_count=len(video_timestamps) if probe_video else None,
        video_frame_indices=video_frame_indices,
    )


def visible_mask(source: SourceData, uid: str) -> list[bool]:
    rows = source.boxes.get(uid, {})
    return [rows.get(timestamp, {}).get("visibility_ratio", 0.0) >= VISIBLE_RATIO for timestamp in source.frame_times]


def matching_distractors(source: SourceData, target_uid: str) -> set[str]:
    target = source.instances.get(target_uid, {})
    prototype = target.get("prototype_name")
    category_uid = target.get("category_uid")
    result = set()
    for uid, item in source.instances.items():
        if uid == target_uid or uid not in source.boxes:
            continue
        if (prototype and item.get("prototype_name") == prototype) or (
            category_uid is not None and item.get("category_uid") == category_uid
        ):
            result.add(uid)
    return result


def proposals_for_source(source: SourceData) -> list[Proposal]:
    proposals: list[Proposal] = []
    frame_count = len(source.frame_times)
    masks = {uid: visible_mask(source, uid) for uid in source.boxes if uid in source.instances}
    for uid, mask in masks.items():
        segments = [(start, end) for start, end in contiguous_segments(mask) if end - start >= MIN_VISIBLE_RUN]
        if not segments:
            continue

        longest = max(segments, key=lambda pair: pair[1] - pair[0])
        if longest[1] - longest[0] >= 90:
            start = longest[0] + (longest[1] - longest[0] - 90) // 2
            proposals.append(Proposal(source.spec.sequence_id, uid, start, start + 90, "CONTINUOUS_VISIBLE", ("CONTINUOUS_VISIBLE",), (-(longest[1] - longest[0]), start, uid)))

        for (left_start, left_end), (right_start, right_end) in zip(segments, segments[1:]):
            gap = right_start - left_end
            if gap < 2 or gap > 240:
                continue
            gap_times = source.frame_times[left_end:right_start]
            row_fraction = (
                sum(timestamp in source.boxes[uid] for timestamp in gap_times) / gap
                if gap_times
                else 0.0
            )
            if gap >= 45:
                mode = "LONG_LOSS"
            elif row_fraction <= 0.20:
                mode = "OUT_OF_VIEW_RETURN"
            else:
                mode = "TEMP_OCCLUSION"
            start = max(0, left_end - MIN_VISIBLE_RUN)
            end = min(frame_count, right_start + MIN_VISIBLE_RUN)
            tags = (mode, "REACQUISITION")
            proposals.append(Proposal(source.spec.sequence_id, uid, start, end, mode, tags, (-gap, start, uid)))

        distractors = matching_distractors(source, uid)
        if distractors:
            simultaneous = [
                mask[index] and any(masks.get(other, [False] * frame_count)[index] for other in distractors)
                for index in range(frame_count)
            ]
            best = None
            prefix = [0]
            for value in simultaneous:
                prefix.append(prefix[-1] + int(value))
            for start in range(0, max(1, frame_count - 89)):
                end = min(frame_count, start + 90)
                count = prefix[end] - prefix[start]
                candidate = (count, -start)
                if best is None or candidate > best[0]:
                    best = (candidate, start, end)
            if best and best[0][0] >= 10:
                tags = ["DISTRACTOR_PRESENT"]
                if all(mask[best[1]:best[2]]):
                    tags.append("CONTINUOUS_VISIBLE")
                proposals.append(Proposal(source.spec.sequence_id, uid, best[1], best[2], "DISTRACTOR_PRESENT", tuple(tags), (-best[0][0], best[1], uid)))
    return proposals


def select_proposals(sources: list[SourceData], episode_budget: int) -> list[Proposal]:
    all_proposals = [proposal for source in sources for proposal in proposals_for_source(source)]
    by_mode: dict[str, list[Proposal]] = defaultdict(list)
    for proposal in all_proposals:
        by_mode[proposal.primary_mode].append(proposal)
    for rows in by_mode.values():
        rows.sort(key=lambda row: row.quality)

    selected: list[Proposal] = []
    used_targets: set[str] = set()
    source_counts: Counter[str] = Counter()
    per_source_cap = max(1, (episode_budget + len(sources) - 1) // len(sources) + 1)

    def admit(proposal: Proposal) -> bool:
        physical_id = f"adt:{proposal.target_uid}"
        if physical_id in used_targets or source_counts[proposal.source_id] >= per_source_cap:
            return False
        selected.append(proposal)
        used_targets.add(physical_id)
        source_counts[proposal.source_id] += 1
        return True

    quotas = {
        "LONG_LOSS": 3,
        "TEMP_OCCLUSION": 3,
        "OUT_OF_VIEW_RETURN": 3,
        "DISTRACTOR_PRESENT": 3,
        "CONTINUOUS_VISIBLE": 3,
    }
    for mode, quota in quotas.items():
        admitted = 0
        for proposal in by_mode.get(mode, []):
            if len(selected) >= episode_budget:
                break
            if admit(proposal):
                admitted += 1
                if admitted >= quota:
                    break

    for proposal in sorted(all_proposals, key=lambda row: (row.primary_mode, row.quality)):
        if len(selected) >= episode_budget:
            break
        admit(proposal)
    return selected


def episode_payload(source: SourceData, proposal: Proposal, ordinal: int) -> dict[str, Any]:
    times = source.frame_times[proposal.start:proposal.end]
    target_rows = source.boxes[proposal.target_uid]
    distractor_pool = matching_distractors(source, proposal.target_uid)
    frames = []
    episode_distractors: set[str] = set()
    for source_index, timestamp in enumerate(times, start=proposal.start):
        target = target_rows.get(timestamp)
        target_visible = target is not None and target["visibility_ratio"] >= VISIBLE_RATIO
        visible_distractors = sorted(
            uid
            for uid in distractor_pool
            if source.boxes.get(uid, {}).get(timestamp, {}).get("visibility_ratio", 0.0) >= VISIBLE_RATIO
        )
        episode_distractors.update(visible_distractors)
        frames.append(
            {
                "source_frame_index": source.video_frame_indices[source_index],
                "timestamp_ns": timestamp,
                "target_visibility_ratio": 0.0 if target is None else target["visibility_ratio"],
                "target_visible": target_visible,
                "target_bbox_xyxy": None if not target_visible else [target["x_min"], target["y_min"], target["x_max"], target["y_max"]],
                "candidate_distractor_instance_ids": [f"adt:{uid}" for uid in visible_distractors],
            }
        )
    instance = source.instances[proposal.target_uid]
    return {
        "schema_version": "blindassist_p1_d0_episode_v1",
        "episode_id": f"p1-d0-{ordinal:03d}-{source.spec.sequence_id}-{proposal.target_uid}",
        "source_sequence_id": source.spec.sequence_id,
        "physical_target_id": f"adt:{proposal.target_uid}",
        "source_object_uid": proposal.target_uid,
        "target_metadata": {
            "instance_name": instance.get("instance_name"),
            "prototype_name": instance.get("prototype_name"),
            "category": instance.get("category"),
            "category_uid": instance.get("category_uid"),
        },
        "temporal_mode_tags": list(proposal.tags),
        "candidate_distractor_instance_ids": [f"adt:{uid}" for uid in sorted(episode_distractors)],
        "frame_count": len(frames),
        "frames": frames,
        "truth_authority": "ADT_SOURCE_OBJECT_UID_AND_RGB_STREAM_2D_BBOX",
        "selection_authority": "GT_TEMPORAL_PROPERTIES_ONLY_NO_TRACKER_OR_MODEL_OUTPUT",
    }


def materialize(
    specs: list[SourceSpec],
    output_dir: Path,
    *,
    episode_budget: int = 15,
    probe_video: bool = True,
) -> dict[str, Any]:
    if not EPISODE_BUDGET_MIN <= episode_budget <= EPISODE_BUDGET_MAX:
        raise ValueError(f"episode budget must be {EPISODE_BUDGET_MIN}-{EPISODE_BUDGET_MAX}")
    if not 1 <= len(specs) <= SOURCE_BUDGET_MAX:
        raise ValueError(f"source count must be 1-{SOURCE_BUDGET_MAX}")
    if len({spec.sequence_id for spec in specs}) != len(specs):
        raise ValueError("source sequence ids must be unique")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output_dir}")

    sources = [load_source(spec, probe_video=probe_video) for spec in specs]
    selected = select_proposals(sources, episode_budget)
    source_by_id = {source.spec.sequence_id: source for source in sources}
    episodes = [episode_payload(source_by_id[row.source_id], row, index + 1) for index, row in enumerate(selected)]
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir()
    for episode in episodes:
        (episode_dir / f"{episode['episode_id']}.json").write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")

    mode_counts = Counter(tag for episode in episodes for tag in episode["temporal_mode_tags"])
    missing_modes = [
        mode
        for mode in ("CONTINUOUS_VISIBLE", "TEMP_OCCLUSION", "OUT_OF_VIEW_RETURN", "DISTRACTOR_PRESENT", "LONG_LOSS", "REACQUISITION")
        if mode_counts[mode] == 0
    ]
    terminal = "P1_TEMPORAL_DEVELOPMENT_COHORT_READY" if len(episodes) >= EPISODE_BUDGET_MIN else "AVAILABLE_PUBLIC_TEMPORAL_DATA_INSUFFICIENT"
    manifest = {
        "schema_version": "blindassist_p1_d0_temporal_cohort_manifest_v1",
        "protocol_id": "BLINDASSIST-P1-D0-TEMPORAL-COHORT-V1",
        "role": "CONSUMED_DEVELOPMENT_ONLY",
        "sources": [
            {
                "source_sequence_id": source.spec.sequence_id,
                "groundtruth_path": str(source.spec.groundtruth.resolve()),
                "groundtruth_sha256": source.groundtruth_sha256,
                "rgb_video_path": str(source.spec.rgb_video.resolve()),
                "rgb_video_sha256": source.rgb_video_sha256,
                "frame_count": len(source.frame_times),
                "rgb_video_frame_count": source.rgb_video_frame_count,
                "rgb_gt_alignment": "MP4_DESCRIPTION_TIMESTAMPS_TO_ARIA_TRAJECTORY_WITHIN_20MS",
            }
            for source in sources
        ],
        "episode_budget": episode_budget,
        "episode_ids": [episode["episode_id"] for episode in episodes],
        "safety_cases": [
            {
                "case_id": f"no-referent-{source.spec.sequence_id}",
                "source_episode_id": next((episode["episode_id"] for episode in episodes if episode["source_sequence_id"] == source.spec.sequence_id), None),
                "handoff": "NO_REFERENT",
                "required_persistent_referent_id": None,
                "required_illegal_bind_rate": 0,
            }
            for source in sources
            if any(episode["source_sequence_id"] == source.spec.sequence_id for episode in episodes)
        ],
        "selection": "GROUND_TRUTH_TEMPORAL_PROPERTIES_ONLY",
        "mode_tag_semantics": {
            "CONTINUOUS_VISIBLE": "one source-identity visible run of at least 90 aligned frames",
            "TEMP_OCCLUSION": "two visible runs separated by 2-44 frames where more than 20 percent of gap frames retain a below-threshold target bbox row; mechanical visibility mode, not causal occluder truth",
            "OUT_OF_VIEW_RETURN": "two visible runs separated by 2-44 frames where at most 20 percent of gap frames retain a target bbox row; mechanical absence mode, not causal field-of-view truth",
            "LONG_LOSS": "two visible runs separated by 45-240 aligned frames",
            "REACQUISITION": "the same ADT object_uid becomes visible after a selected gap",
            "DISTRACTOR_PRESENT": "another ADT instance with the same prototype or category is simultaneously visible for at least 10 frames",
        },
        "model_detector_tracker_call_counts": {"model": 0, "detector": 0, "tracker": 0},
        "claim_ceiling": "P1_DEVELOPMENT_TEMPORAL_TRUTH_ONLY_NO_ENTRANCE_OR_ALGORITHM_CLAIM",
        "terminal": terminal,
    }
    summary = {
        "terminal": terminal,
        "source_sequences": len(sources),
        "episodes": len(episodes),
        "physical_targets": len({episode["physical_target_id"] for episode in episodes}),
        "frames": sum(episode["frame_count"] for episode in episodes),
        "temporal_mode_counts": dict(sorted(mode_counts.items())),
        "missing_modes": missing_modes,
        "ground_truth_coverage": {
            "episodes_with_source_uid": sum(bool(episode["source_object_uid"]) for episode in episodes),
            "episodes_with_per_frame_visibility": len(episodes),
            "episodes_with_per_frame_bbox_or_null": len(episodes),
        },
    }
    (output_dir / "p1_d0_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("SEQUENCE_ID", "GROUNDTRUTH_ZIP", "RGB_VIDEO"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-budget", type=int, default=15)
    args = parser.parse_args()
    summary = materialize([parse_source(fields) for fields in args.source], args.output_dir, episode_budget=args.episode_budget)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["terminal"] == "P1_TEMPORAL_DEVELOPMENT_COHORT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
