"""Raw and motion-warped temporal component metrics for candidate utility R0."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .component_metrics import Component, connected_components, mask_iou


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def warp_mask(mask: np.ndarray, matrix_previous_to_current: Sequence[Sequence[float]]) -> np.ndarray:
    """Warp a previous analysis-grid mask into the current grid using a 2x3 affine."""

    source = np.asarray(mask, dtype=bool)
    matrix = np.asarray(matrix_previous_to_current, dtype=np.float64)
    if source.ndim != 2 or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("mask must be 2D and affine must be finite 2x3")
    ys, xs = np.nonzero(source)
    if not len(xs):
        return np.zeros_like(source, dtype=bool)
    mapped_x = np.rint(matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]).astype(np.int64)
    mapped_y = np.rint(matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]).astype(np.int64)
    valid = (mapped_x >= 0) & (mapped_x < source.shape[1]) & (mapped_y >= 0) & (mapped_y < source.shape[0])
    warped = np.zeros_like(source, dtype=bool)
    warped[mapped_y[valid], mapped_x[valid]] = True
    return warped


def _overlap_pairs(previous: list[Component], current: list[Component], *, threshold: float) -> list[tuple[float, int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for previous_index, previous_component in enumerate(previous):
        for current_index, current_component in enumerate(current):
            value = mask_iou(previous_component.mask, current_component.mask)
            if value >= threshold:
                pairs.append((value, previous_index, current_index))
    return sorted(pairs, reverse=True)


def _greedy_match(previous: list[Component], current: list[Component], *, threshold: float) -> list[tuple[int, int, float]]:
    matched_previous: set[int] = set()
    matched_current: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for value, previous_index, current_index in _overlap_pairs(previous, current, threshold=threshold):
        if previous_index in matched_previous or current_index in matched_current:
            continue
        matched_previous.add(previous_index)
        matched_current.add(current_index)
        matches.append((previous_index, current_index, value))
    return matches


def summarize_temporal(
    masks: Sequence[np.ndarray],
    *,
    frame_ids: Sequence[int],
    timestamps_ns: Sequence[int] | None = None,
    motion_warps: Sequence[Sequence[Sequence[float]] | None] | None = None,
    match_iou: float = 0.10,
    timestamps_are_source_native: bool = False,
) -> dict[str, Any]:
    """Summarize adjacent stability and component lifecycle for one source/class."""

    if len(masks) != len(frame_ids):
        raise ValueError("masks and frame_ids must have equal length")
    if timestamps_ns is not None and len(timestamps_ns) != len(masks):
        raise ValueError("timestamps_ns must match masks")
    if motion_warps is not None and len(motion_warps) != max(0, len(masks) - 1):
        raise ValueError("motion_warps must contain one entry per adjacent pair")
    if not masks:
        return {"frame_count": 0, "adjacent_pair_count": 0, "motion_warped_pair_count": 0}
    components_by_frame = [connected_components(mask) for mask in masks]
    raw_ious: list[float] = []
    warped_ious: list[float] = []
    births = deaths = splits = merges = 0
    assignment_rows: list[list[int]] = [[-1] * len(components_by_frame[0])]
    active: dict[int, tuple[int, Component, int]] = {}
    next_track_id = 0
    for component_index, component in enumerate(components_by_frame[0]):
        active[next_track_id] = (0, component, component_index)
        assignment_rows[0][component_index] = next_track_id
        next_track_id += 1
    tracks: dict[int, dict[str, Any]] = {
        track_id: {"track_id": track_id, "start_index": 0, "end_index": 0, "frame_count": 1}
        for track_id in active
    }
    for index in range(1, len(masks)):
        previous_mask = masks[index - 1]
        current_mask = masks[index]
        raw_ious.append(mask_iou(previous_mask, current_mask))
        warp = motion_warps[index - 1] if motion_warps is not None else None
        warped_previous = warp_mask(previous_mask, warp) if warp is not None else None
        if warped_previous is not None:
            warped_ious.append(mask_iou(warped_previous, current_mask))
        previous_components = components_by_frame[index - 1]
        current_components = components_by_frame[index]
        overlap_pairs = _overlap_pairs(previous_components, current_components, threshold=match_iou)
        previous_degree = {i: 0 for i in range(len(previous_components))}
        current_degree = {i: 0 for i in range(len(current_components))}
        for _, previous_component_index, current_component_index in overlap_pairs:
            previous_degree[previous_component_index] += 1
            current_degree[current_component_index] += 1
        splits += sum(value > 1 for value in previous_degree.values())
        merges += sum(value > 1 for value in current_degree.values())
        matched = _greedy_match(previous_components, current_components, threshold=match_iou)
        previous_track_by_component = {
            component_index: track_id
            for track_id, (_, _, component_index) in active.items()
        }
        next_active: dict[int, tuple[int, Component, int]] = {}
        assignment = [-1] * len(current_components)
        matched_current: set[int] = set()
        matched_previous: set[int] = set()
        for previous_component_index, current_component_index, _ in matched:
            track_id = previous_track_by_component.get(previous_component_index)
            if track_id is None:
                continue
            matched_previous.add(previous_component_index)
            matched_current.add(current_component_index)
            next_active[track_id] = (index, current_components[current_component_index], current_component_index)
            assignment[current_component_index] = track_id
            tracks[track_id]["end_index"] = index
            tracks[track_id]["frame_count"] += 1
        births += len(current_components) - len(matched_current)
        deaths += len(previous_components) - len(matched_previous)
        for current_component_index, component in enumerate(current_components):
            if current_component_index in matched_current:
                continue
            track_id = next_track_id
            next_track_id += 1
            next_active[track_id] = (index, component, current_component_index)
            assignment[current_component_index] = track_id
            tracks[track_id] = {"track_id": track_id, "start_index": index, "end_index": index, "frame_count": 1}
        active = next_active
        assignment_rows.append(assignment)
    durations_frames = [int(row["frame_count"]) for row in tracks.values()]
    duration_seconds: list[float] = []
    if timestamps_are_source_native and timestamps_ns is not None:
        for row in tracks.values():
            start = int(row["start_index"])
            end = int(row["end_index"])
            duration_seconds.append(float((timestamps_ns[end] - timestamps_ns[start]) / 1e9))
    elapsed_seconds = None
    if timestamps_are_source_native and timestamps_ns is not None and len(timestamps_ns) >= 2:
        elapsed_seconds = float((timestamps_ns[-1] - timestamps_ns[0]) / 1e9)
    return {
        "frame_count": len(masks),
        "adjacent_pair_count": max(0, len(masks) - 1),
        "raw_adjacent_iou": _percentiles(raw_ious),
        "motion_warped_adjacent_iou": _percentiles(warped_ious),
        "motion_warped_pair_count": len(warped_ious),
        "motion_warp_available": bool(warped_ious),
        "candidate_birth_count": int(births),
        "candidate_death_count": int(deaths),
        "candidate_births_per_minute": float(births / (elapsed_seconds / 60.0)) if elapsed_seconds and elapsed_seconds > 0 else None,
        "candidate_deaths_per_minute": float(deaths / (elapsed_seconds / 60.0)) if elapsed_seconds and elapsed_seconds > 0 else None,
        "split_count": int(splits),
        "merge_count": int(merges),
        "flicker_track_count": int(sum(duration <= 2 for duration in durations_frames)),
        "persistence_frames": _percentiles([float(value) for value in durations_frames]),
        "persistence_seconds": _percentiles(duration_seconds) if duration_seconds else None,
        "component_track_assignments": assignment_rows,
        "component_tracks": [
            {
                "track_id": int(row["track_id"]),
                "start_frame_id": int(frame_ids[int(row["start_index"])]),
                "end_frame_id": int(frame_ids[int(row["end_index"])]),
                "duration_frames": int(row["frame_count"]),
                "duration_seconds": (
                    float((timestamps_ns[int(row["end_index"])] - timestamps_ns[int(row["start_index"])]) / 1e9)
                    if timestamps_are_source_native and timestamps_ns is not None
                    else None
                ),
            }
            for row in sorted(tracks.values(), key=lambda item: int(item["track_id"]))
        ],
    }
