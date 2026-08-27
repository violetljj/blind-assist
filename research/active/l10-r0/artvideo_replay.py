from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Descriptor:
    gray: np.ndarray
    edge: np.ndarray
    hist: np.ndarray


@dataclass(frozen=True)
class Proposal:
    truth_id: int
    center: tuple[float, float]
    size: tuple[float, float]
    descriptor: Descriptor


def _normalize_plane(array: np.ndarray) -> np.ndarray:
    flat = array.astype(np.float32).reshape(-1)
    flat -= float(flat.mean())
    norm = float(np.linalg.norm(flat))
    return flat / max(norm, 1e-6)


def describe(crop: np.ndarray) -> Descriptor:
    resized = cv2.resize(crop, (160, 48), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 55, 150)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return Descriptor(
        gray=_normalize_plane(gray),
        edge=_normalize_plane(edge),
        hist=hist.reshape(-1).astype(np.float32),
    )


def similarity(left: Descriptor, right: Descriptor) -> float:
    gray = (float(np.dot(left.gray, right.gray)) + 1.0) * 0.5
    edge = (float(np.dot(left.edge, right.edge)) + 1.0) * 0.5
    hist = float(cv2.compareHist(left.hist, right.hist, cv2.HISTCMP_CORREL))
    hist = (hist + 1.0) * 0.5
    return max(0.0, min(1.0, 0.48 * gray + 0.34 * edge + 0.18 * hist))


def annotation_box(row: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    points = np.asarray(row["point"], dtype=np.float32)
    x0 = max(0, int(np.floor(points[:, 0].min())))
    y0 = max(0, int(np.floor(points[:, 1].min())))
    x1 = min(width, int(np.ceil(points[:, 0].max())) + 1)
    y1 = min(height, int(np.ceil(points[:, 1].max())) + 1)
    pad_x = max(2, int((x1 - x0) * 0.04))
    pad_y = max(2, int((y1 - y0) * 0.08))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)


class ReplayVideo:
    def __init__(self, frame_root: Path, annotation_path: Path):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.name = annotation_path.stem
        self.frame_root = frame_root / self.name
        self.frames = {int(row["frame_id"]): row["frame_jpg"] for row in payload["frame"]}
        self.by_frame: dict[int, list[dict[str, Any]]] = {}
        self.by_track: dict[int, list[dict[str, Any]]] = {}
        for row in payload["annotations"]:
            frame_id = int(row["frame_id"])
            track_id = int(row["obj_id"])
            self.by_frame.setdefault(frame_id, []).append(row)
            self.by_track.setdefault(track_id, []).append(row)
        self._cache: dict[int, tuple[np.ndarray, list[Proposal]]] = {}

    def frame(self, frame_id: int) -> tuple[np.ndarray, list[Proposal]]:
        if frame_id in self._cache:
            return self._cache[frame_id]
        path = self.frame_root / self.frames[frame_id]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        height, width = image.shape[:2]
        proposals = []
        for row in self.by_frame.get(frame_id, []):
            x0, y0, x1, y1 = annotation_box(row, width, height)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            proposals.append(
                Proposal(
                    truth_id=int(row["obj_id"]),
                    center=((x0 + x1) / (2.0 * width), (y0 + y1) / (2.0 * height)),
                    size=((x1 - x0) / width, (y1 - y0) / height),
                    descriptor=describe(image[y0:y1, x0:x1]),
                )
            )
        self._cache[frame_id] = image, proposals
        return image, proposals


def spatial_similarity(center: tuple[float, float], predicted: tuple[float, float]) -> float:
    distance = ((center[0] - predicted[0]) ** 2 + (center[1] - predicted[1]) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / 0.42)


class StaticTemplate:
    name = "B0_static_template"

    def reset(self, reference: Proposal) -> None:
        self.reference = reference.descriptor

    def step(self, proposals: list[Proposal]) -> int | None:
        if not proposals:
            return None
        scores = [similarity(self.reference, proposal.descriptor) for proposal in proposals]
        best = int(np.argmax(scores))
        return best if scores[best] >= 0.54 else None


class StickyLocal:
    name = "B1_sticky_local"

    def reset(self, reference: Proposal) -> None:
        self.reference = reference.descriptor
        self.center = reference.center
        self.velocity = (0.0, 0.0)
        self.misses = 0

    def step(self, proposals: list[Proposal]) -> int | None:
        if not proposals:
            self.misses += 1
            return None
        predicted = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
        scores = []
        for proposal in proposals:
            appearance = similarity(self.reference, proposal.descriptor)
            local = spatial_similarity(proposal.center, predicted)
            scores.append(0.38 * appearance + 0.62 * local)
        best = int(np.argmax(scores))
        if scores[best] < (0.50 if self.misses <= 2 else 0.58):
            self.misses += 1
            return None
        previous = self.center
        self.center = proposals[best].center
        self.velocity = (
            0.65 * self.velocity[0] + 0.35 * (self.center[0] - previous[0]),
            0.65 * self.velocity[1] + 0.35 * (self.center[1] - previous[1]),
        )
        self.misses = 0
        return best


class DualMemoryGoalLock:
    name = "L10_R0_dual_memory"

    def reset(self, reference: Proposal) -> None:
        self.reference = reference.descriptor
        self.recent = [reference.descriptor]
        self.center = reference.center
        self.velocity = (0.0, 0.0)
        self.lost = False
        self.pending_center: tuple[float, float] | None = None
        self.pending_hits = 0

    def step(self, proposals: list[Proposal]) -> int | None:
        if not proposals:
            self.lost = True
            self.pending_hits = 0
            return None
        predicted = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
        scored = []
        for index, proposal in enumerate(proposals):
            long_term = similarity(self.reference, proposal.descriptor)
            short_term = max(similarity(item, proposal.descriptor) for item in self.recent[-3:])
            spatial = spatial_similarity(proposal.center, predicted)
            if self.lost:
                score = 0.65 * long_term + 0.25 * short_term + 0.10 * spatial
            else:
                score = 0.55 * long_term + 0.25 * short_term + 0.20 * spatial
            scored.append((score, long_term, short_term, spatial, index))
        score, long_term, short_term, spatial, best = max(scored)
        candidate = proposals[best]
        if self.lost:
            if score < 0.64 or not (long_term >= 0.67 or short_term >= 0.74):
                self.pending_hits = 0
                return None
            if self.pending_center is not None and spatial_similarity(candidate.center, self.pending_center) >= 0.55:
                self.pending_hits += 1
            else:
                self.pending_center = candidate.center
                self.pending_hits = 1
            if self.pending_hits < 2:
                return None
        elif (
            score < 0.63
            or max(long_term, short_term) < 0.70
            or (spatial < 0.30 and long_term < 0.76)
        ):
            self.lost = True
            self.pending_hits = 0
            return None

        previous = self.center
        self.center = candidate.center
        self.velocity = (
            0.60 * self.velocity[0] + 0.40 * (self.center[0] - previous[0]),
            0.60 * self.velocity[1] + 0.40 * (self.center[1] - previous[1]),
        )
        self.lost = False
        self.pending_hits = 0
        self.pending_center = None
        second_score = sorted((item[0] for item in scored), reverse=True)[1] if len(scored) > 1 else 0.0
        if (
            long_term >= 0.72
            and short_term >= 0.70
            and spatial >= 0.34
            and score - second_score >= 0.035
        ):
            self.recent.append(candidate.descriptor)
            self.recent = self.recent[-5:]
        return best


TRACKERS = (StaticTemplate, StickyLocal, DualMemoryGoalLock)


def eligible_tracks(video: ReplayVideo, minimum_frames: int) -> list[tuple[int, str, list[int]]]:
    rows = []
    for track_id, annotations in video.by_track.items():
        ordered = sorted(annotations, key=lambda item: int(item["frame_id"]))
        text = str(ordered[0].get("Transcription", ""))
        frame_ids = [int(item["frame_id"]) for item in ordered]
        if text != "###" and len(text) >= 3 and len(frame_ids) >= minimum_frames:
            rows.append((track_id, text, frame_ids))
    return rows


def run_episode(
    tracker: Any,
    video: ReplayVideo,
    target_id: int,
    frame_ids: list[int],
    gap_start: int,
    gap_length: int,
) -> dict[str, Any]:
    first_frame = frame_ids[0]
    _, initial = video.frame(first_frame)
    reference = next(item for item in initial if item.truth_id == target_id)
    tracker.reset(reference)
    gap = set(frame_ids[gap_start : gap_start + gap_length])
    gap_end_frame = frame_ids[min(len(frame_ids) - 1, gap_start + gap_length - 1)]
    correct = 0
    wrong = 0
    missed = 0
    evaluable = 0
    reacquire_latency: int | None = None
    after_gap_index = 0
    for frame_id in frame_ids[1:]:
        _, proposals = video.frame(frame_id)
        if frame_id in gap:
            proposals = [item for item in proposals if item.truth_id != target_id]
        target_available = any(item.truth_id == target_id for item in proposals)
        selected_index = tracker.step(proposals)
        if target_available:
            evaluable += 1
            if selected_index is None:
                missed += 1
            elif proposals[selected_index].truth_id == target_id:
                correct += 1
                if frame_id > gap_end_frame and reacquire_latency is None:
                    reacquire_latency = after_gap_index
            else:
                wrong += 1
        elif selected_index is not None:
            wrong += 1
        if frame_id > gap_end_frame:
            after_gap_index += 1
    return {
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "evaluable": evaluable,
        "reacquired": reacquire_latency is not None and reacquire_latency <= 5,
        "reacquire_latency": reacquire_latency,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row["correct"] for row in rows)
    wrong = sum(row["wrong"] for row in rows)
    missed = sum(row["missed"] for row in rows)
    evaluable = sum(row["evaluable"] for row in rows)
    reacquired = sum(row["reacquired"] for row in rows)
    latencies = [row["reacquire_latency"] for row in rows if row["reacquired"]]
    return {
        "episodes": len(rows),
        "evaluable_target_frames": evaluable,
        "correct_target_frames": correct,
        "wrong_selection_frames": wrong,
        "missed_target_frames": missed,
        "target_frame_accuracy": round(correct / evaluable, 4) if evaluable else 0.0,
        "wrong_selection_rate": round(wrong / max(1, correct + wrong), 4),
        "reacquired_episodes": reacquired,
        "reacquire_success_rate": round(reacquired / len(rows), 4) if rows else 0.0,
        "median_reacquire_frames": statistics.median(latencies) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Proposal-conditional real-RGB L10-R0 replay.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=["video1", "video10"])
    parser.add_argument("--minimum-frames", type=int, default=15)
    parser.add_argument("--gap-length", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    videos = [
        ReplayVideo(
            args.dataset / "Test" / "frame",
            args.dataset / "Test" / "json" / f"{name}.json",
        )
        for name in args.videos
    ]
    episode_specs = []
    for video in videos:
        for target_id, text, frame_ids in eligible_tracks(video, args.minimum_frames):
            for fraction in (0.35, 0.50, 0.65):
                start = max(2, min(len(frame_ids) - args.gap_length - 3, int(len(frame_ids) * fraction)))
                episode_specs.append((video, target_id, text, frame_ids, start))

    arms = {}
    for tracker_type in TRACKERS:
        rows = [
            run_episode(tracker_type(), video, target_id, frame_ids, start, args.gap_length)
            for video, target_id, _text, frame_ids, start in episode_specs
        ]
        arms[tracker_type.name] = summarize(rows)

    result = {
        "experiment": "L10-R0-ARTVIDEO-PROPOSAL-CONDITIONAL-REPLAY-V1",
        "claim_ceiling": "REAL_RGB_GT_PROPOSALS_ARTIFICIAL_OCCLUSION_DEVELOPMENT_ONLY",
        "result_status": "REAL_RGB_CANARY_RESULT",
        "source": "ArTVideo public test frames and annotations",
        "videos": args.videos,
        "eligible_tracks": len(episode_specs) // 3,
        "episode_count": len(episode_specs),
        "artificial_gap_frames": args.gap_length,
        "goal_texts": sorted({text for _, _, text, _, _ in episode_specs}),
        "arms": arms,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
