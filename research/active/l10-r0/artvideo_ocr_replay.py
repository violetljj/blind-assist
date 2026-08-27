from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidocr import RapidOCR


@dataclass
class Candidate:
    text: str
    score: float
    box: np.ndarray
    center: tuple[float, float]
    truth_ids: frozenset[int]


def natural_key(path: Path) -> tuple[str, int]:
    return path.parent.name, int(path.stem)


def normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def tokens(text: str) -> list[str]:
    return [normalize(item) for item in re.findall(r"[\w]+", text, flags=re.UNICODE) if normalize(item)]


def lexical(goal: str, observed: str) -> float:
    g = normalize(goal)
    o = normalize(observed)
    if not g or not o:
        return 0.0
    observed_tokens = tokens(observed)
    if g in observed_tokens:
        return 1.0
    scores = [difflib.SequenceMatcher(None, g, o).ratio()]
    scores.extend(difflib.SequenceMatcher(None, g, token).ratio() for token in observed_tokens)
    if g in o:
        scores.append(min(0.95, 0.72 + 0.23 * len(g) / len(o)))
    return max(scores)


def spatial(center: tuple[float, float], predicted: tuple[float, float]) -> float:
    distance = ((center[0] - predicted[0]) ** 2 + (center[1] - predicted[1]) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / 0.42)


def gt_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    points = np.asarray(row["point"], dtype=np.float32)
    return float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())


def overlap_over_gt(candidate_box: np.ndarray, row: dict[str, Any]) -> float:
    cx0, cy0 = candidate_box[:, 0].min(), candidate_box[:, 1].min()
    cx1, cy1 = candidate_box[:, 0].max(), candidate_box[:, 1].max()
    gx0, gy0, gx1, gy1 = gt_box(row)
    intersection = max(0.0, min(cx1, gx1) - max(cx0, gx0)) * max(0.0, min(cy1, gy1) - max(cy0, gy0))
    return intersection / max(1.0, (gx1 - gx0) * (gy1 - gy0))


class PerFrameText:
    name = "B0_per_frame_text"

    def __init__(self, goal: str):
        self.goal = goal

    def step(self, candidates: list[Candidate]) -> int | None:
        if not candidates:
            return None
        ranked = [(lexical(self.goal, item.text), item.score, index) for index, item in enumerate(candidates)]
        lex, _confidence, index = max(ranked)
        return index if lex >= 0.58 else None


class StickyText:
    name = "B1_sticky_text"

    def __init__(self, goal: str):
        self.goal = goal
        self.center: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)
        self.misses = 0

    def step(self, candidates: list[Candidate]) -> int | None:
        if not candidates:
            self.misses += 1
            return None
        if self.center is None:
            ranked = [(lexical(self.goal, item.text), item.score, index) for index, item in enumerate(candidates)]
            lex, _confidence, index = max(ranked)
            if lex < 0.58:
                self.misses += 1
                return None
        else:
            predicted = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
            ranked = []
            for index, item in enumerate(candidates):
                lex = lexical(self.goal, item.text)
                local = spatial(item.center, predicted)
                ranked.append((0.62 * lex + 0.38 * local, lex, index))
            score, lex, index = max(ranked)
            threshold = 0.52 if self.misses <= 2 else 0.60
            if score < threshold or lex < 0.38:
                self.misses += 1
                return None
        previous = self.center
        self.center = candidates[index].center
        if previous is not None:
            self.velocity = (
                0.65 * self.velocity[0] + 0.35 * (self.center[0] - previous[0]),
                0.65 * self.velocity[1] + 0.35 * (self.center[1] - previous[1]),
            )
        self.misses = 0
        return index


class CandidateBoundGoalLock:
    name = "L10_candidate_bound"

    def __init__(self, goal: str):
        self.goal = goal
        self.center: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)
        self.recent_texts: list[str] = []
        self.lost = True
        self.pending_center: tuple[float, float] | None = None
        self.pending_hits = 0

    def step(self, candidates: list[Candidate]) -> int | None:
        if not candidates:
            self.lost = True
            self.pending_center = None
            self.pending_hits = 0
            return None
        predicted = self.center if self.center is not None else (0.5, 0.5)
        if self.center is not None:
            predicted = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
        ranked = []
        for index, item in enumerate(candidates):
            goal_score = lexical(self.goal, item.text)
            recent = max((lexical(text, item.text) for text in self.recent_texts[-3:]), default=goal_score)
            local = spatial(item.center, predicted) if self.center is not None else 0.5
            score = (0.72 * goal_score + 0.18 * recent + 0.10 * local) if self.lost else (0.62 * goal_score + 0.18 * recent + 0.20 * local)
            ranked.append((score, goal_score, recent, local, item.score, index))
        score, goal_score, recent, local, _confidence, index = max(ranked)
        second = sorted((item[0] for item in ranked), reverse=True)[1] if len(ranked) > 1 else 0.0
        candidate = candidates[index]
        if self.lost:
            if score < 0.62 or goal_score < 0.58 or score - second < 0.025:
                self.pending_center = None
                self.pending_hits = 0
                return None
            if self.pending_center is not None and spatial(candidate.center, self.pending_center) >= 0.55:
                self.pending_hits += 1
            else:
                self.pending_center = candidate.center
                self.pending_hits = 1
            if self.pending_hits < 2:
                return None
        elif score < 0.56 or goal_score < 0.44 or (local < 0.25 and goal_score < 0.72):
            self.lost = True
            self.pending_center = None
            self.pending_hits = 0
            return None
        previous = self.center
        self.center = candidate.center
        if previous is not None:
            self.velocity = (
                0.60 * self.velocity[0] + 0.40 * (self.center[0] - previous[0]),
                0.60 * self.velocity[1] + 0.40 * (self.center[1] - previous[1]),
            )
        self.recent_texts.append(candidate.text)
        self.recent_texts = self.recent_texts[-5:]
        self.lost = False
        self.pending_center = None
        self.pending_hits = 0
        return index


TRACKERS = (PerFrameText, StickyText, CandidateBoundGoalLock)


def build_cache(dataset: Path, cache_path: Path, models: Path) -> dict[str, Any]:
    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(models),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    rows: dict[str, Any] = {}
    started = time.perf_counter()
    for path in sorted((dataset / "Test" / "frame").rglob("*.jpg"), key=natural_key):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        before = time.perf_counter()
        output = engine(image)
        rows[path.relative_to(dataset).as_posix()] = {
            "shape": list(image.shape),
            "wall_s": round(time.perf_counter() - before, 6),
            "detections": [
                {"box": np.asarray(box, dtype=float).tolist(), "text": text, "score": float(score)}
                for box, text, score in zip(
                    output.boxes if output.boxes is not None else (),
                    output.txts if output.txts is not None else (),
                    output.scores if output.scores is not None else (),
                )
            ],
        }
    payload = {
        "schema": "artvideo_rapidocr_cache_v1",
        "backend": "RapidOCR 3.9.2 / ONNX Runtime 1.26.0 / CPUExecutionProvider",
        "models": str(models),
        "frames": rows,
        "ocr_wall_s": round(time.perf_counter() - started, 4),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_video(dataset: Path, annotation_path: Path, cache: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_frame: dict[int, list[dict[str, Any]]] = {}
    by_track: dict[int, list[dict[str, Any]]] = {}
    for row in payload["annotations"]:
        frame_id = int(row["frame_id"])
        track_id = int(row["obj_id"])
        by_frame.setdefault(frame_id, []).append(row)
        by_track.setdefault(track_id, []).append(row)
    frames = {int(row["frame_id"]): row["frame_jpg"] for row in payload["frame"]}
    candidates: dict[int, list[Candidate]] = {}
    for frame_id, frame_name in frames.items():
        key = (Path("Test") / "frame" / annotation_path.stem / frame_name).as_posix()
        cached = cache["frames"].get(key)
        if cached is None:
            continue
        height, width = cached["shape"][:2]
        frame_candidates = []
        for detection in cached["detections"]:
            box = np.asarray(detection["box"], dtype=np.float32)
            truth_ids = frozenset(
                int(row["obj_id"])
                for row in by_frame.get(frame_id, [])
                if overlap_over_gt(box, row) >= 0.25
            )
            center = (float(box[:, 0].mean()) / width, float(box[:, 1].mean()) / height)
            frame_candidates.append(Candidate(detection["text"], float(detection["score"]), box, center, truth_ids))
        candidates[frame_id] = frame_candidates
    return {"name": annotation_path.stem, "by_frame": by_frame, "by_track": by_track, "candidates": candidates}


def eligible_tracks(video: dict[str, Any], minimum_frames: int) -> list[tuple[int, str, list[int]]]:
    result = []
    for track_id, annotations in video["by_track"].items():
        ordered = sorted(annotations, key=lambda row: int(row["frame_id"]))
        text = str(ordered[0].get("Transcription", ""))
        frame_ids = [int(row["frame_id"]) for row in ordered]
        if text != "###" and len(text) >= 3 and len(frame_ids) >= minimum_frames:
            result.append((track_id, text, frame_ids))
    return result


def run_episode(tracker_type: type, video: dict[str, Any], target_id: int, goal: str, frame_ids: list[int], gap_start: int, gap_length: int) -> dict[str, Any]:
    tracker = tracker_type(goal)
    first = video["candidates"].get(frame_ids[0], [])
    tracker.step(first)
    gap = set(frame_ids[gap_start : gap_start + gap_length])
    gap_end_frame = frame_ids[min(len(frame_ids) - 1, gap_start + gap_length - 1)]
    correct = wrong_present = false_select_gap = missed = evaluable = 0
    reacquire_latency: int | None = None
    after_gap_index = 0
    for frame_id in frame_ids[1:]:
        candidates = video["candidates"].get(frame_id, [])
        if frame_id in gap:
            candidates = [item for item in candidates if target_id not in item.truth_ids]
        selected = tracker.step(candidates)
        if frame_id not in gap:
            evaluable += 1
            if selected is None:
                missed += 1
            elif target_id in candidates[selected].truth_ids:
                correct += 1
                if frame_id > gap_end_frame and reacquire_latency is None:
                    reacquire_latency = after_gap_index
            else:
                wrong_present += 1
        elif selected is not None:
            false_select_gap += 1
        if frame_id > gap_end_frame:
            after_gap_index += 1
    return {
        "correct": correct,
        "wrong": wrong_present + false_select_gap,
        "wrong_target_present": wrong_present,
        "false_select_gap": false_select_gap,
        "missed": missed,
        "evaluable": evaluable,
        "reacquired": reacquire_latency is not None and reacquire_latency <= 5,
        "reacquire_latency": reacquire_latency,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row["correct"] for row in rows)
    wrong = sum(row["wrong"] for row in rows)
    wrong_present = sum(row["wrong_target_present"] for row in rows)
    false_select_gap = sum(row["false_select_gap"] for row in rows)
    missed = sum(row["missed"] for row in rows)
    evaluable = sum(row["evaluable"] for row in rows)
    reacquired = sum(bool(row["reacquired"]) for row in rows)
    latencies = [row["reacquire_latency"] for row in rows if row["reacquired"]]
    return {
        "episodes": len(rows),
        "evaluable_target_frames": evaluable,
        "correct_target_frames": correct,
        "wrong_selection_frames": wrong,
        "wrong_target_present_frames": wrong_present,
        "false_selection_gap_frames": false_select_gap,
        "missed_target_frames": missed,
        "target_frame_accuracy": round(correct / evaluable, 4) if evaluable else 0.0,
        "wrong_selection_rate": round(wrong / max(1, correct + wrong), 4),
        "gap_reacquired_episodes": reacquired,
        "gap_reacquire_rate": round(reacquired / len(rows), 4) if rows else 0.0,
        "median_reacquire_frames": statistics.median(latencies) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--gap-length", type=int, default=4)
    args = parser.parse_args()
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else build_cache(args.dataset, args.cache, args.models)
    videos = [load_video(args.dataset, path, cache) for path in sorted((args.dataset / "Test" / "json").glob("*.json"))]
    specs = []
    coverage_rows = []
    for video in videos:
        for target_id, goal, frame_ids in eligible_tracks(video, args.minimum_frames):
            covered = sum(any(target_id in item.truth_ids for item in video["candidates"].get(frame_id, [])) for frame_id in frame_ids)
            lexical_covered = sum(any(target_id in item.truth_ids and lexical(goal, item.text) >= 0.58 for item in video["candidates"].get(frame_id, [])) for frame_id in frame_ids)
            coverage_rows.append({"video": video["name"], "target_id": target_id, "goal": goal, "frames": len(frame_ids), "candidate_covered": covered, "lexical_covered": lexical_covered})
            for fraction in (0.30, 0.50, 0.70):
                start = max(2, min(len(frame_ids) - args.gap_length - 3, int(len(frame_ids) * fraction)))
                specs.append((video, target_id, goal, frame_ids, start))
    metrics = {}
    for tracker_type in TRACKERS:
        rows = [run_episode(tracker_type, video, target_id, goal, frame_ids, start, args.gap_length) for video, target_id, goal, frame_ids, start in specs]
        metrics[tracker_type.name] = summarize(rows)
    total_frames = sum(row["frames"] for row in coverage_rows)
    payload = {
        "schema": "l10_artvideo_proposal_free_text_goal_replay_v0",
        "status": "CONTROLLED_DEVELOPMENT_DIAGNOSTIC",
        "source": "ArTVideo RGB; GT transcription supplies goal identity; GT track boxes are evaluator-only",
        "proposal_source": "RapidOCR PP-OCRv6 full-frame detection and recognition; no GT proposals enter trackers",
        "candidate_truth_rule": "OCR box is evaluator-associated with every GT track box for which intersection / GT bbox area >= 0.25",
        "episode_protocol": {"eligible_tracks": len(coverage_rows), "episodes": len(specs), "minimum_track_frames": args.minimum_frames, "artificial_gap_frames": args.gap_length, "gap_positions": [0.30, 0.50, 0.70], "reacquire_window_frames": 5},
        "ocr_cache": {"path": str(args.cache), "frames": len(cache["frames"]), "ocr_wall_s": cache.get("ocr_wall_s")},
        "candidate_ceiling": {
            "annotated_track_frames": total_frames,
            "candidate_covered_frames": sum(row["candidate_covered"] for row in coverage_rows),
            "candidate_coverage": round(sum(row["candidate_covered"] for row in coverage_rows) / max(1, total_frames), 4),
            "lexically_covered_frames": sum(row["lexical_covered"] for row in coverage_rows),
            "lexical_coverage": round(sum(row["lexical_covered"] for row in coverage_rows) / max(1, total_frames), 4),
            "per_track": coverage_rows,
        },
        "metrics": metrics,
        "claim_ceiling": "Two-video, 10-track, detector-plus-recognizer Development replay only; merged OCR boxes make correctness candidate-bound; not natural distribution, distance, guidance, Android, product, or safety evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
