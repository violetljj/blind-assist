"""Temporal active-observation belief for target-bound entrance candidates.

Single-frame SET_VALUED is treated as a useful search state, never as failure
or arrival. The controller asks for a closer centered observation, commits only
after one candidate survives across consecutive views while competitors clear,
and explicitly reacquires a lost committed entrance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Sequence


class ActiveEntranceState(str, Enum):
    SEARCH = "SEARCH"
    SET_VALUED = "SET_VALUED"
    COMMIT = "COMMIT"
    REACQUIRE = "REACQUIRE"


class ObservationAction(str, Enum):
    SWEEP = "SWEEP"
    CENTER_AND_APPROACH = "CENTER_AND_APPROACH"
    TRACK = "TRACK"
    SCAN_LAST_BEARING = "SCAN_LAST_BEARING"


@dataclass(frozen=True)
class FrameCandidate:
    candidate_id: str
    box_xyxy: tuple[float, float, float, float]
    edge_score: float


@dataclass(frozen=True)
class ActiveEntranceDecision:
    state: ActiveEntranceState
    action: ObservationAction
    selected_track_id: str | None
    candidate_track_ids: tuple[str, ...]
    guidance_target_box_xyxy: tuple[float, float, float, float] | None
    reason: str


@dataclass
class _Track:
    track_id: str
    box_xyxy: tuple[float, float, float, float]
    edge_score: float
    consecutive_hits: int
    approach_consistent_hits: int
    normalized_area: float
    last_frame_index: int


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(
    left: Sequence[float], right: Sequence[float], image_width: int, image_height: int
) -> float:
    left_x, left_y = 0.5 * (left[0] + left[2]), 0.5 * (left[1] + left[3])
    right_x, right_y = 0.5 * (right[0] + right[2]), 0.5 * (right[1] + right[3])
    dx = (left_x - right_x) / max(float(image_width), 1.0)
    dy = (left_y - right_y) / max(float(image_height), 1.0)
    return (dx * dx + dy * dy) ** 0.5


def _normalized_area(box: Sequence[float], image_width: int, image_height: int) -> float:
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return (width * height) / max(float(image_width * image_height), 1.0)


class ActiveEntranceBelief:
    """Maintain candidate identity across active closer views.

    The controller deliberately has no score-tuned commit threshold. A commit
    requires a candidate to be target-bound in two consecutive frames and to
    be the sole surviving candidate in the current frame. This turns a static
    candidate set into an observation request instead of a forced decision.
    """

    def __init__(self) -> None:
        self._frame_index = -1
        self._next_track_number = 1
        self._tracks: dict[str, _Track] = {}
        self._committed_track_id: str | None = None
        self._last_committed_box: tuple[float, float, float, float] | None = None

    def update(
        self,
        candidates: Sequence[FrameCandidate],
        image_width: int,
        image_height: int,
    ) -> ActiveEntranceDecision:
        if image_width < 1 or image_height < 1:
            raise ValueError("INVALID_IMAGE_SIZE")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("DUPLICATE_FRAME_CANDIDATE_ID")
        self._frame_index += 1
        previous = list(self._tracks.values())
        if (
            self._committed_track_id is not None
            and self._committed_track_id not in self._tracks
            and self._last_committed_box is not None
        ):
            previous.append(
                _Track(
                    self._committed_track_id,
                    self._last_committed_box,
                    0.0,
                    2,
                    2,
                    _normalized_area(self._last_committed_box, image_width, image_height),
                    self._frame_index - 1,
                )
            )
        previous_by_id = {track.track_id: track for track in previous}
        assignments: dict[int, str] = {}
        claimed_tracks: set[str] = set()
        pairs = []
        for candidate_index, candidate in enumerate(candidates):
            for track in previous:
                overlap = _iou(candidate.box_xyxy, track.box_xyxy)
                distance = _center_distance(
                    candidate.box_xyxy, track.box_xyxy, image_width, image_height
                )
                if overlap > 0.0 or distance <= 0.15:
                    pairs.append((-overlap, distance, candidate_index, track.track_id))
        for _, _, candidate_index, track_id in sorted(pairs):
            if candidate_index in assignments or track_id in claimed_tracks:
                continue
            assignments[candidate_index] = track_id
            claimed_tracks.add(track_id)
        current_tracks: dict[str, _Track] = {}
        for candidate_index, candidate in enumerate(candidates):
            track_id = assignments.get(candidate_index)
            if track_id is None:
                track_id = f"entrance-track-{self._next_track_number:03d}"
                self._next_track_number += 1
                consecutive_hits = 1
                approach_consistent_hits = 1
            else:
                prior = previous_by_id[track_id]
                consecutive_hits = prior.consecutive_hits + 1
                current_area = _normalized_area(candidate.box_xyxy, image_width, image_height)
                approach_consistent_hits = (
                    prior.approach_consistent_hits + 1
                    if current_area >= prior.normalized_area
                    else 1
                )
            current_tracks[track_id] = _Track(
                track_id,
                candidate.box_xyxy,
                float(candidate.edge_score),
                consecutive_hits,
                approach_consistent_hits,
                _normalized_area(candidate.box_xyxy, image_width, image_height),
                self._frame_index,
            )
        self._tracks = current_tracks

        if self._committed_track_id is not None:
            committed = current_tracks.get(self._committed_track_id)
            if committed is None:
                return ActiveEntranceDecision(
                    ActiveEntranceState.REACQUIRE,
                    ObservationAction.SCAN_LAST_BEARING,
                    self._committed_track_id,
                    (),
                    self._last_committed_box,
                    "COMMITTED_ENTRANCE_TEMPORARILY_LOST",
                )
            self._last_committed_box = committed.box_xyxy
            return ActiveEntranceDecision(
                ActiveEntranceState.COMMIT,
                ObservationAction.TRACK,
                committed.track_id,
                (committed.track_id,),
                committed.box_xyxy,
                "COMMITTED_ENTRANCE_REOBSERVED",
            )

        ordered = sorted(current_tracks.values(), key=lambda track: (-track.edge_score, track.track_id))
        if not ordered:
            return ActiveEntranceDecision(
                ActiveEntranceState.SEARCH,
                ObservationAction.SWEEP,
                None,
                (),
                None,
                "NO_TARGET_BOUND_ENTRANCE_CANDIDATE",
            )
        if (
            len(ordered) == 1
            and ordered[0].consecutive_hits >= 2
            and ordered[0].approach_consistent_hits >= 2
        ):
            winner = ordered[0]
            self._committed_track_id = winner.track_id
            self._last_committed_box = winner.box_xyxy
            return ActiveEntranceDecision(
                ActiveEntranceState.COMMIT,
                ObservationAction.TRACK,
                winner.track_id,
                (winner.track_id,),
                winner.box_xyxy,
                "SOLE_TARGET_BOUND_CANDIDATE_SURVIVED_CONSECUTIVE_ACTIVE_VIEWS",
            )
        probe = ordered[0]
        reason = (
            "ACTIVE_CLOSER_VIEW_DID_NOT_INCREASE_TARGET_SCALE"
            if len(ordered) == 1 and probe.consecutive_hits >= 2
            else "ACTIVE_CLOSER_VIEW_REQUIRED_TO_CLEAR_COMPETITORS"
        )
        return ActiveEntranceDecision(
            ActiveEntranceState.SET_VALUED,
            ObservationAction.CENTER_AND_APPROACH,
            None,
            tuple(track.track_id for track in ordered),
            probe.box_xyxy,
            reason,
        )


def _self_test() -> dict[str, object]:
    belief = ActiveEntranceBelief()
    first = belief.update(
        [
            FrameCandidate("left", (10, 20, 35, 90), 0.58),
            FrameCandidate("right", (60, 20, 90, 90), 0.55),
        ],
        100,
        100,
    )
    if first.state != ActiveEntranceState.SET_VALUED or first.action != ObservationAction.CENTER_AND_APPROACH:
        raise AssertionError("AMBIGUITY_DID_NOT_REQUEST_ACTIVE_VIEW")
    second = belief.update([FrameCandidate("closer-left", (8, 12, 42, 96), 0.70)], 100, 100)
    if second.state != ActiveEntranceState.COMMIT:
        raise AssertionError("SURVIVING_CANDIDATE_DID_NOT_COMMIT")
    lost = belief.update([], 100, 100)
    if lost.state != ActiveEntranceState.REACQUIRE:
        raise AssertionError("LOST_COMMIT_DID_NOT_ENTER_REACQUIRE")
    found = belief.update([FrameCandidate("return", (9, 10, 44, 97), 0.72)], 100, 100)
    if found.state != ActiveEntranceState.COMMIT or found.action != ObservationAction.TRACK:
        raise AssertionError("REACQUIRED_COMMIT_DID_NOT_RESUME_TRACKING")
    return {
        "schema": "l10-active-entrance-belief-self-test-v1",
        "status": "PASS",
        "sequence": [asdict(first), asdict(second), asdict(lost), asdict(found)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test; runtime adapters call ActiveEntranceBelief directly")
    print(json.dumps(_self_test(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
