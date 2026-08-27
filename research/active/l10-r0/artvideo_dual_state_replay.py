from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import artvideo_ocr_replay as replay
import artvideo_semantic_visual_replay as sc0


COAST_LIMIT_FRAMES = 2


class SemanticState(str, Enum):
    TARGET = "TARGET"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"
    UNCERTAIN = "UNCERTAIN"


class ContinuityState(str, Enum):
    UNBOUND = "UNBOUND"
    BOUND = "BOUND"
    COASTING = "COASTING"
    PROVISIONAL = "PROVISIONAL"
    LOST = "LOST"


@dataclass(frozen=True)
class DualStateDecision:
    semantic_state: str
    continuity_state: str
    identity_index: int | None
    continuity_index: int | None
    observation_index: int | None
    action_mode: str
    evidence_deficit: str | None
    fresh_semantic: bool
    semantic_age_frames: int
    carrier_center: tuple[float, float] | None = None
    carrier_scope: str | None = None
    carrier_updated: bool = False
    lexical_score: float | None = None
    visual_score: float | None = None
    spatial_score: float | None = None
    score_margin: float | None = None


class L10SC1DualState:
    """Fresh semantics owns identity; appearance may only request a better view."""

    name = "L10_SC1W_semantic_carrier"

    def __init__(self, goal: str):
        self.goal = goal
        self.center: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)
        self.long: np.ndarray | None = None
        self.carrier: np.ndarray | None = None
        self.lost = False
        self.coast_age = 0
        self.semantic_age = 0
        self.pending_center: tuple[float, float] | None = None
        self.pending_hits = 0

    @staticmethod
    def _embedding(candidate: Any) -> np.ndarray:
        value = getattr(candidate, "pooled_patch_embedding", None)
        if value is None:
            raise ValueError("candidate is missing pooled_patch_embedding")
        return value

    def _spatial_score(self, candidate: Any) -> float:
        if self.center is None:
            return 0.5
        predicted = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
        return replay.spatial(candidate.center, predicted)

    def _carrier_geometry(self, candidate: Any) -> tuple[tuple[float, float], str]:
        """Expose the geometry scope supplied by OCR without inventing a sub-box."""
        word_results = list(getattr(candidate, "word_results", ()))
        if word_results:
            ranked = [
                (replay.lexical(self.goal, str(word["text"])), float(word["score"]), index)
                for index, word in enumerate(word_results)
            ]
            lexical_score, _word_score, index = max(ranked)
            tied = sum(abs(row[0] - lexical_score) <= 1e-9 for row in ranked)
            width = float(getattr(candidate, "frame_width", 0.0))
            height = float(getattr(candidate, "frame_height", 0.0))
            if lexical_score >= sc0.LOCKED_GATES[0] and tied == 1 and width > 0.0 and height > 0.0:
                box = np.asarray(word_results[index]["box"], dtype=np.float32)
                return (float(box[:, 0].mean()) / width, float(box[:, 1].mean()) / height), "RAPIDOCR_CTC_WORD"
        words = [word for word in str(candidate.text).strip().split() if word]
        return candidate.center, "OCR_TOKEN" if len(words) <= 1 else "OCR_MERGED_LINE"

    @staticmethod
    def _runner_up(rows: list[tuple[float, ...]]) -> float:
        scores = sorted((row[0] for row in rows), reverse=True)
        return scores[1] if len(scores) > 1 else 0.0

    def _legacy_rows(self, candidates: list[Any], *, long_only: bool = False) -> list[tuple[float, float, float, float, int]]:
        if self.long is None:
            raise RuntimeError("semantic visual memory is not initialized")
        rows: list[tuple[float, float, float, float, int]] = []
        for index, candidate in enumerate(candidates):
            embedding = self._embedding(candidate)
            lexical_score = replay.lexical(self.goal, candidate.text)
            visual_score = float(embedding @ self.long)
            if not long_only and self.carrier is not None:
                visual_score = max(visual_score, float(embedding @ self.carrier))
            spatial_score = self._spatial_score(candidate)
            combined = 0.45 * lexical_score + 0.40 * visual_score + 0.15 * spatial_score
            rows.append((combined, lexical_score, visual_score, spatial_score, index))
        return rows

    def _update_motion(self, center: tuple[float, float]) -> None:
        previous = self.center
        self.center = center
        if previous is not None:
            self.velocity = (
                0.65 * self.velocity[0] + 0.35 * (center[0] - previous[0]),
                0.65 * self.velocity[1] + 0.35 * (center[1] - previous[1]),
            )

    def _decision(
        self,
        semantic_state: SemanticState,
        continuity_state: ContinuityState,
        *,
        identity_index: int | None = None,
        continuity_index: int | None = None,
        observation_index: int | None = None,
        action_mode: str,
        deficit: str | None,
        fresh_semantic: bool = False,
        lexical_score: float | None = None,
        visual_score: float | None = None,
        spatial_score: float | None = None,
        margin: float | None = None,
        carrier_candidate: Any | None = None,
        carrier_updated: bool = False,
    ) -> DualStateDecision:
        carrier_center, carrier_scope = (None, None)
        if carrier_candidate is not None:
            carrier_center, carrier_scope = self._carrier_geometry(carrier_candidate)
        return DualStateDecision(
            semantic_state=semantic_state.value,
            continuity_state=continuity_state.value,
            identity_index=identity_index,
            continuity_index=continuity_index,
            observation_index=observation_index,
            action_mode=action_mode,
            evidence_deficit=deficit,
            fresh_semantic=fresh_semantic,
            semantic_age_frames=self.semantic_age,
            carrier_center=carrier_center,
            carrier_scope=carrier_scope,
            carrier_updated=carrier_updated,
            lexical_score=lexical_score,
            visual_score=visual_score,
            spatial_score=spatial_score,
            score_margin=margin,
        )

    def _accept(
        self,
        candidates: list[Any],
        index: int,
        *,
        lexical_score: float,
        visual_score: float | None = None,
        spatial_score: float | None = None,
        margin: float | None = None,
    ) -> DualStateDecision:
        candidate = candidates[index]
        embedding = self._embedding(candidate)
        self.long = embedding.copy() if self.long is None else sc0._normalize(0.85 * self.long + 0.15 * embedding)
        self.carrier = embedding.copy()
        self._update_motion(candidate.center)
        self.lost = False
        self.coast_age = 0
        self.semantic_age = 0
        self.pending_center = None
        self.pending_hits = 0
        return self._decision(
            SemanticState.TARGET,
            ContinuityState.BOUND,
            identity_index=index,
            continuity_index=index,
            action_mode="NAVIGATE",
            deficit=None,
            fresh_semantic=True,
            lexical_score=lexical_score,
            visual_score=visual_score,
            spatial_score=spatial_score,
            margin=margin,
            carrier_candidate=candidate,
            carrier_updated=True,
        )

    def _coast(
        self,
        candidates: list[Any],
        match: tuple[int, float, float, float],
        *,
        semantic_state: SemanticState,
        deficit: str,
    ) -> DualStateDecision:
        index, visual_score, spatial_score, margin = match
        if self.coast_age >= COAST_LIMIT_FRAMES:
            self.lost = True
            self.pending_center = None
            self.pending_hits = 0
            return self._decision(
                semantic_state,
                ContinuityState.LOST,
                action_mode="SEARCH",
                deficit="SEMANTIC_REFRESH_EXPIRED",
                visual_score=visual_score,
                spatial_score=spatial_score,
                margin=margin,
            )
        self.coast_age += 1
        candidate = candidates[index]
        self.carrier = self._embedding(candidate).copy()
        self._update_motion(candidate.center)
        return self._decision(
            semantic_state,
            ContinuityState.COASTING,
            continuity_index=index,
            observation_index=index,
            action_mode="OBSERVE",
            deficit=deficit,
            visual_score=visual_score,
            spatial_score=spatial_score,
            margin=margin,
            carrier_candidate=candidate,
            carrier_updated=True,
        )

    def _acquire(self, candidates: list[Any]) -> DualStateDecision:
        if not candidates:
            return self._decision(
                SemanticState.UNKNOWN,
                ContinuityState.UNBOUND,
                action_mode="SEARCH",
                deficit="NO_LOCALIZABLE_EVIDENCE",
            )
        lexical_score, _ocr_score, index = max(
            (replay.lexical(self.goal, candidate.text), candidate.score, index)
            for index, candidate in enumerate(candidates)
        )
        if lexical_score >= sc0.REACQUIRE_GATES[0]:
            return self._accept(
                candidates,
                index,
                lexical_score=lexical_score,
            )
        if lexical_score >= sc0.LOCKED_GATES[0]:
            return self._decision(
                SemanticState.UNCERTAIN,
                ContinuityState.UNBOUND,
                observation_index=index,
                action_mode="OBSERVE",
                deficit="DECISIVE_TOKEN_UNREADABLE",
                lexical_score=lexical_score,
                carrier_candidate=candidates[index],
            )
        return self._decision(
            SemanticState.NONE,
            ContinuityState.UNBOUND,
            action_mode="STOP",
            deficit="CLEAR_CANDIDATE_SET_NONE",
            lexical_score=lexical_score,
        )

    def _track(self, candidates: list[Any]) -> DualStateDecision:
        if not candidates:
            self.lost = True
            self.pending_center = None
            self.pending_hits = 0
            return self._decision(
                SemanticState.UNKNOWN,
                ContinuityState.LOST,
                action_mode="SEARCH",
                deficit="NO_LOCALIZABLE_EVIDENCE",
            )
        rows = self._legacy_rows(candidates)
        combined, lexical_score, visual_score, spatial_score, index = max(rows)
        margin = combined - self._runner_up(rows)
        if (
            lexical_score >= sc0.LOCKED_GATES[0]
            and visual_score >= sc0.LOCKED_GATES[1]
            and combined >= sc0.LOCKED_GATES[2]
            and margin >= sc0.LOCKED_GATES[3]
        ):
            if lexical_score >= sc0.REACQUIRE_GATES[0]:
                return self._accept(
                    candidates,
                    index,
                    lexical_score=lexical_score,
                    visual_score=visual_score,
                    spatial_score=spatial_score,
                    margin=margin,
                )
            return self._coast(
                candidates,
                (index, visual_score, spatial_score, margin),
                semantic_state=SemanticState.UNCERTAIN,
                deficit="SEMANTIC_REFRESH_DUE",
            )
        self.lost = True
        self.pending_center = None
        self.pending_hits = 0
        if lexical_score >= sc0.LOCKED_GATES[0]:
            return self._decision(
                SemanticState.UNCERTAIN,
                ContinuityState.LOST,
                observation_index=index,
                action_mode="OBSERVE",
                deficit="ASSOCIATION_GATE_FAILED",
                lexical_score=lexical_score,
                visual_score=visual_score,
                spatial_score=spatial_score,
                margin=margin,
                carrier_candidate=candidates[index],
            )
        return self._decision(
            SemanticState.NONE,
            ContinuityState.LOST,
            action_mode="STOP",
            deficit="CLEAR_CANDIDATE_SET_NONE",
            lexical_score=lexical_score,
        )

    def _reacquire(self, candidates: list[Any]) -> DualStateDecision:
        if not candidates:
            self.pending_center = None
            self.pending_hits = 0
            return self._decision(
                SemanticState.UNKNOWN,
                ContinuityState.LOST,
                action_mode="SEARCH",
                deficit="NO_LOCALIZABLE_EVIDENCE",
            )
        rows = self._legacy_rows(candidates, long_only=True)
        combined, lexical_score, visual_score, spatial_score, index = max(rows)
        margin = combined - self._runner_up(rows)
        if (
            lexical_score < sc0.REACQUIRE_GATES[0]
            or visual_score < sc0.REACQUIRE_GATES[1]
            or combined < sc0.REACQUIRE_GATES[2]
            or margin < sc0.REACQUIRE_GATES[3]
        ):
            self.pending_center = None
            self.pending_hits = 0
            if lexical_score >= sc0.LOCKED_GATES[0]:
                return self._decision(
                    SemanticState.UNCERTAIN,
                    ContinuityState.LOST,
                    observation_index=index,
                    action_mode="OBSERVE",
                    deficit="REACQUIRE_EVIDENCE_INSUFFICIENT",
                    lexical_score=lexical_score,
                    visual_score=visual_score,
                    spatial_score=spatial_score,
                    margin=margin,
                    carrier_candidate=candidates[index],
                )
            return self._decision(
                SemanticState.NONE,
                ContinuityState.LOST,
                action_mode="STOP",
                deficit="CLEAR_CANDIDATE_SET_NONE",
                lexical_score=lexical_score,
            )
        candidate = candidates[index]
        if self.pending_center is not None and replay.spatial(candidate.center, self.pending_center) >= 0.55:
            self.pending_hits += 1
        else:
            self.pending_center = candidate.center
            self.pending_hits = 1
        if self.pending_hits < 2:
            return self._decision(
                SemanticState.UNCERTAIN,
                ContinuityState.PROVISIONAL,
                continuity_index=index,
                observation_index=index,
                action_mode="OBSERVE",
                deficit="REACQUIRE_CONFIRMATION_PENDING",
                lexical_score=lexical_score,
                visual_score=visual_score,
                spatial_score=spatial_score,
                margin=margin,
                carrier_candidate=candidate,
            )
        return self._accept(
            candidates,
            index,
            lexical_score=lexical_score,
            visual_score=visual_score,
            spatial_score=spatial_score,
            margin=margin,
        )

    def step(self, candidates: list[Any]) -> DualStateDecision:
        self.semantic_age += 1
        if self.long is None:
            return self._acquire(candidates)
        return self._reacquire(candidates) if self.lost else self._track(candidates)


def build_scoped_ocr_cache(dataset: Path, videos: list[str], cache_path: Path, models: Path) -> dict[str, Any]:
    from rapidocr import RapidOCR

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
    for video in videos:
        frame_dir = dataset / "Test" / "frame" / video
        for path in sorted(frame_dir.glob("*.jpg"), key=replay.natural_key):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            before = time.perf_counter()
            output = engine(image, return_word_box=True)
            detections = []
            boxes = output.boxes if output.boxes is not None else ()
            texts = output.txts if output.txts is not None else ()
            scores = output.scores if output.scores is not None else ()
            word_lines = output.word_results if output.word_results is not None else ()
            for detection_index, (box, text, score) in enumerate(zip(boxes, texts, scores)):
                words = word_lines[detection_index] if detection_index < len(word_lines) else ()
                detections.append(
                    {
                        "box": np.asarray(box, dtype=float).tolist(),
                        "text": text,
                        "score": float(score),
                        "words": [
                            {
                                "text": word_text,
                                "score": float(word_score),
                                "box": np.asarray(word_box, dtype=float).tolist(),
                            }
                            for word_text, word_score, word_box in words
                        ],
                    }
                )
            rows[path.relative_to(dataset).as_posix()] = {
                "shape": list(image.shape),
                "wall_s": round(time.perf_counter() - before, 6),
                "detections": detections,
            }
    payload = {
        "schema": "artvideo_rapidocr_scoped_word_box_cache_v1",
        "videos": videos,
        "backend": "RapidOCR 3.9.2 / ONNX Runtime 1.26.0 / CPUExecutionProvider",
        "models": str(models),
        "frames": rows,
        "ocr_wall_s": round(time.perf_counter() - started, 4),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def attach_frame_geometry(
    videos: list[dict[str, Any]],
    annotation_paths: list[Path],
    ocr_cache: dict[str, Any],
) -> None:
    for video, annotation_path in zip(videos, annotation_paths, strict=True):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        names = {int(row["frame_id"]): str(row["frame_jpg"]) for row in annotation["frame"]}
        frame_sizes: dict[int, tuple[int, int]] = {}
        for frame_id, frame_name in names.items():
            key = (Path("Test") / "frame" / annotation_path.stem / frame_name).as_posix()
            cached = ocr_cache["frames"].get(key)
            if cached is None:
                continue
            height, width = (int(cached["shape"][0]), int(cached["shape"][1]))
            frame_sizes[frame_id] = (width, height)
            detections = cached["detections"]
            candidates = video["candidates"].get(frame_id, [])
            if len(detections) != len(candidates):
                raise ValueError(f"OCR candidate count mismatch for {key}")
            for detection_index, candidate in enumerate(candidates):
                candidate.frame_width = width
                candidate.frame_height = height
                candidate.word_results = detections[detection_index].get("words", [])
        video["frame_sizes"] = frame_sizes


def _direction(center: tuple[float, float]) -> str:
    centered_x = 2.0 * center[0] - 1.0
    if centered_x < -0.14:
        return "LEFT"
    if centered_x > 0.14:
        return "RIGHT"
    return "FORWARD"


def _target_center(video: dict[str, Any], frame_id: int, target_id: int) -> tuple[float, float]:
    row = next(item for item in video["by_frame"][frame_id] if int(item["obj_id"]) == target_id)
    width, height = video["frame_sizes"][frame_id]
    points = np.asarray(row["point"], dtype=np.float32)
    return float(points[:, 0].mean()) / width, float(points[:, 1].mean()) / height


def run_dual_episode(
    video: dict[str, Any],
    target_id: int,
    goal: str,
    frame_ids: list[int],
    gap_start: int,
    gap_length: int,
) -> dict[str, Any]:
    tracker = L10SC1DualState(goal)
    tracker.step(video["candidates"].get(frame_ids[0], []))
    gap = set(frame_ids[gap_start : gap_start + gap_length])
    gap_end_frame = frame_ids[min(len(frame_ids) - 1, gap_start + gap_length - 1)]
    counts: Counter[str] = Counter()
    states: Counter[str] = Counter()
    reacquire_latency: int | None = None
    after_gap_index = 0
    for frame_id in frame_ids[1:]:
        candidates = video["candidates"].get(frame_id, [])
        if frame_id in gap:
            candidates = [candidate for candidate in candidates if target_id not in candidate.truth_ids]
        decision = tracker.step(candidates)
        states[f"semantic:{decision.semantic_state}"] += 1
        states[f"continuity:{decision.continuity_state}"] += 1
        states[f"action:{decision.action_mode}"] += 1
        if decision.carrier_scope is not None:
            states[f"scope:{decision.carrier_scope}"] += 1
        if decision.identity_index is not None and not decision.fresh_semantic:
            counts["appearance_only_identity_violations"] += 1
        if decision.action_mode == "NAVIGATE" and decision.identity_index is None:
            counts["continuity_only_navigation_violations"] += 1
        if frame_id in gap:
            counts["gap_frames"] += 1
            if decision.identity_index is not None:
                counts["false_identity_gap_frames"] += 1
            if decision.action_mode == "OBSERVE":
                counts["gap_observation_requests"] += 1
                if decision.observation_index is not None:
                    counts["false_observation_gap_frames"] += 1
                    if decision.carrier_center is not None:
                        if _direction(decision.carrier_center) == _direction(_target_center(video, frame_id, target_id)):
                            counts["correct_gap_observation_bearing_frames"] += 1
                        else:
                            counts["wrong_gap_observation_bearing_frames"] += 1
            continue

        counts["evaluable_target_frames"] += 1
        identity_correct = False
        if decision.identity_index is None:
            counts["identity_missed_target_frames"] += 1
        elif target_id in candidates[decision.identity_index].truth_ids:
            identity_correct = True
            counts["correct_identity_target_frames"] += 1
            if decision.carrier_center is None:
                counts["identity_frames_without_carrier"] += 1
            elif _direction(decision.carrier_center) == _direction(_target_center(video, frame_id, target_id)):
                counts["correct_identity_bearing_frames"] += 1
            else:
                counts["wrong_identity_bearing_frames"] += 1
            if frame_id > gap_end_frame and reacquire_latency is None:
                reacquire_latency = after_gap_index
        else:
            counts["wrong_identity_target_present_frames"] += 1

        observation_correct = False
        if decision.action_mode == "OBSERVE":
            counts["target_observation_requests"] += 1
            if decision.observation_index is None:
                counts["unlocalized_observation_requests"] += 1
            elif target_id in candidates[decision.observation_index].truth_ids:
                observation_correct = True
                counts["correct_observation_target_frames"] += 1
                if decision.carrier_center is None:
                    counts["observation_frames_without_carrier"] += 1
                elif _direction(decision.carrier_center) == _direction(_target_center(video, frame_id, target_id)):
                    counts["correct_observation_bearing_frames"] += 1
                else:
                    counts["wrong_observation_bearing_frames"] += 1
            else:
                counts["wrong_observation_target_present_frames"] += 1
        if identity_correct or observation_correct:
            counts["correct_target_support_frames"] += 1
        if (
            (identity_correct and decision.carrier_center is not None and _direction(decision.carrier_center) == _direction(_target_center(video, frame_id, target_id)))
            or (
                observation_correct
                and decision.carrier_center is not None
                and _direction(decision.carrier_center) == _direction(_target_center(video, frame_id, target_id))
            )
        ):
            counts["correct_direction_support_frames"] += 1
        if decision.semantic_state == SemanticState.NONE.value:
            counts["false_none_target_present_frames"] += 1
        if frame_id > gap_end_frame:
            after_gap_index += 1
    counts["reacquired"] = int(reacquire_latency is not None and reacquire_latency <= 5)
    return {"counts": dict(counts), "states": dict(states), "reacquire_latency": reacquire_latency}


def run_legacy_direction_episode(
    tracker_type: type,
    video: dict[str, Any],
    target_id: int,
    goal: str,
    frame_ids: list[int],
    gap_start: int,
    gap_length: int,
) -> dict[str, int]:
    tracker = tracker_type(goal)
    tracker.step(video["candidates"].get(frame_ids[0], []))
    gap = set(frame_ids[gap_start : gap_start + gap_length])
    counts: Counter[str] = Counter()
    for frame_id in frame_ids[1:]:
        candidates = video["candidates"].get(frame_id, [])
        if frame_id in gap:
            candidates = [candidate for candidate in candidates if target_id not in candidate.truth_ids]
        selected = tracker.step(candidates)
        if selected is None:
            continue
        bearing_matches = _direction(candidates[selected].center) == _direction(_target_center(video, frame_id, target_id))
        if frame_id in gap:
            counts["gap_selection_frames"] += 1
            counts["correct_gap_bearing_frames" if bearing_matches else "wrong_gap_bearing_frames"] += 1
        elif target_id in candidates[selected].truth_ids:
            counts["correct_identity_selection_frames"] += 1
            counts["correct_identity_bearing_frames" if bearing_matches else "wrong_identity_bearing_frames"] += 1
    return dict(counts)


def summarize_legacy_direction(rows: list[dict[str, int]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row)
    identity_bearing = counts["correct_identity_bearing_frames"] + counts["wrong_identity_bearing_frames"]
    gap_bearing = counts["correct_gap_bearing_frames"] + counts["wrong_gap_bearing_frames"]
    return {
        **dict(sorted(counts.items())),
        "identity_bearing_accuracy": round(counts["correct_identity_bearing_frames"] / max(1, identity_bearing), 4),
        "gap_bearing_accuracy": round(counts["correct_gap_bearing_frames"] / max(1, gap_bearing), 4),
    }


def summarize_dual(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    states: Counter[str] = Counter()
    for row in rows:
        counts.update(row["counts"])
        states.update(row["states"])
    evaluable = counts["evaluable_target_frames"]
    identity_correct = counts["correct_identity_target_frames"]
    identity_wrong = counts["wrong_identity_target_present_frames"] + counts["false_identity_gap_frames"]
    localized_observations = (
        counts["correct_observation_target_frames"]
        + counts["wrong_observation_target_present_frames"]
        + counts["false_observation_gap_frames"]
    )
    identity_bearing = counts["correct_identity_bearing_frames"] + counts["wrong_identity_bearing_frames"]
    observation_bearing = counts["correct_observation_bearing_frames"] + counts["wrong_observation_bearing_frames"]
    gap_observation_bearing = counts["correct_gap_observation_bearing_frames"] + counts["wrong_gap_observation_bearing_frames"]
    reacquired = sum(int(row["counts"].get("reacquired", 0)) for row in rows)
    latencies = [row["reacquire_latency"] for row in rows if row["counts"].get("reacquired", 0)]
    return {
        "episodes": len(rows),
        **dict(sorted(counts.items())),
        "identity_wrong_frames": identity_wrong,
        "identity_target_recall": round(identity_correct / evaluable, 4) if evaluable else 0.0,
        "navigation_precision": round(identity_correct / max(1, identity_correct + identity_wrong), 4),
        "target_support_coverage": round(counts["correct_target_support_frames"] / evaluable, 4) if evaluable else 0.0,
        "direction_ready_coverage": round(counts["correct_direction_support_frames"] / evaluable, 4) if evaluable else 0.0,
        "identity_bearing_accuracy": round(counts["correct_identity_bearing_frames"] / max(1, identity_bearing), 4),
        "observation_bearing_accuracy": round(counts["correct_observation_bearing_frames"] / max(1, observation_bearing), 4),
        "gap_observation_bearing_accuracy": round(counts["correct_gap_observation_bearing_frames"] / max(1, gap_observation_bearing), 4),
        "localized_observation_precision": round(counts["correct_observation_target_frames"] / max(1, localized_observations), 4),
        "identity_reacquired_episodes": reacquired,
        "identity_reacquire_rate": round(reacquired / len(rows), 4) if rows else 0.0,
        "median_identity_reacquire_frames": statistics.median(latencies) if latencies else None,
        "semantic_state_frames": {key.removeprefix("semantic:"): value for key, value in sorted(states.items()) if key.startswith("semantic:")},
        "continuity_state_frames": {key.removeprefix("continuity:"): value for key, value in sorted(states.items()) if key.startswith("continuity:")},
        "action_frames": {key.removeprefix("action:"): value for key, value in sorted(states.items()) if key.startswith("action:")},
        "carrier_scope_frames": {key.removeprefix("scope:"): value for key, value in sorted(states.items()) if key.startswith("scope:")},
    }


def _dual_by_video(specs: list[tuple[Any, ...]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(spec[0]["name"], []).append(run_dual_episode(*spec))
    return {name: summarize_dual(rows) for name, rows in sorted(grouped.items())}


def _legacy_by_video(specs: list[tuple[Any, ...]], tracker_type: type) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(spec[0]["name"], []).append(replay.run_episode(tracker_type, *spec))
    return {name: replay.summarize(rows) for name, rows in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L10-SC1W semantic identity and word-carrier replay.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--gap-length", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.ocr_cache.exists():
        ocr_cache = json.loads(args.ocr_cache.read_text(encoding="utf-8"))
        ocr_build = None
    else:
        if args.models is None:
            parser.error("--models is required when --ocr-cache does not exist")
        ocr_cache = build_scoped_ocr_cache(args.dataset, args.videos, args.ocr_cache, args.models)
        ocr_build = {"frames": len(ocr_cache["frames"]), "ocr_wall_s": ocr_cache["ocr_wall_s"]}

    embedding_build = None
    if not args.embedding_cache.exists() or not args.embedding_index.exists():
        if args.model is None:
            parser.error("--model is required when embedding cache/index do not exist")
        embedding_build = sc0.build_embedding_cache(
            args.dataset,
            ocr_cache,
            args.model,
            args.embedding_cache,
            args.embedding_index,
        )
    index_payload = json.loads(args.embedding_index.read_text(encoding="utf-8"))
    with np.load(args.embedding_cache, allow_pickle=False) as payload:
        pooled_patch = payload["pooled_patch"].astype(np.float32, copy=True)

    annotation_paths = [args.dataset / "Test" / "json" / f"{video}.json" for video in args.videos]
    videos = [replay.load_video(args.dataset, path, ocr_cache) for path in annotation_paths]
    attach_frame_geometry(videos, annotation_paths, ocr_cache)
    attached = sc0.attach_embeddings(videos, annotation_paths, index_payload, pooled_patch)
    specs: list[tuple[Any, ...]] = []
    for video in videos:
        for target_id, goal, frame_ids in replay.eligible_tracks(video, args.minimum_frames):
            for fraction in (0.30, 0.50, 0.70):
                start = max(2, min(len(frame_ids) - args.gap_length - 3, int(len(frame_ids) * fraction)))
                specs.append((video, target_id, goal, frame_ids, start, args.gap_length))

    legacy_metrics: dict[str, Any] = {}
    legacy_per_video: dict[str, Any] = {}
    legacy_direction_metrics: dict[str, Any] = {}
    for tracker_type in (replay.PerFrameText, replay.StickyText, sc0.SemanticVisualMemory):
        rows = [replay.run_episode(tracker_type, *spec) for spec in specs]
        legacy_metrics[tracker_type.name] = replay.summarize(rows)
        legacy_per_video[tracker_type.name] = _legacy_by_video(specs, tracker_type)
        legacy_direction_metrics[tracker_type.name] = summarize_legacy_direction(
            [run_legacy_direction_episode(tracker_type, *spec) for spec in specs]
        )
    dual_rows = [run_dual_episode(*spec) for spec in specs]
    dual_metrics = summarize_dual(dual_rows)
    sc0_metrics = legacy_metrics[sc0.SemanticVisualMemory.name]
    sc0_precision = sc0_metrics["correct_target_frames"] / max(
        1, sc0_metrics["correct_target_frames"] + sc0_metrics["wrong_selection_frames"]
    )
    gate_checks = {
        "navigation_precision_not_below_sc0": dual_metrics["navigation_precision"] >= sc0_precision,
        "identity_wrong_not_above_frozen_sc0": dual_metrics["identity_wrong_frames"] <= sc0_metrics["wrong_selection_frames"],
        "target_support_coverage_above_sc0_selection_accuracy": dual_metrics["target_support_coverage"] > sc0_metrics["target_frame_accuracy"],
        "identity_reacquire_not_below_sc0": dual_metrics["identity_reacquire_rate"] >= sc0_metrics["gap_reacquire_rate"],
        "word_carrier_bearing_above_sc0_line_center": dual_metrics["identity_bearing_accuracy"] > legacy_direction_metrics[sc0.SemanticVisualMemory.name]["identity_bearing_accuracy"],
        "appearance_only_identity_violations_zero": dual_metrics.get("appearance_only_identity_violations", 0) == 0,
        "continuity_only_navigation_violations_zero": dual_metrics.get("continuity_only_navigation_violations", 0) == 0,
    }
    result = {
        "schema": "l10_sc1w_artvideo_semantic_carrier_replay_v0",
        "status": "SC1W_DEVELOPMENT_EFFECT" if all(gate_checks.values()) else "SC1W_GATE_NOT_MET",
        "videos": args.videos,
        "episode_protocol": {
            "eligible_tracks": len(specs) // 3,
            "episodes": len(specs),
            "minimum_track_frames": args.minimum_frames,
            "artificial_gap_frames": args.gap_length,
            "gap_positions": [0.30, 0.50, 0.70],
            "identity_reacquire_window_frames": 5,
        },
        "algorithm": {
            "identity_authority": "lexical admission at the frozen 0.58 gate, corroborated by the frozen SC0 association; two fresh semantic hits after LOST",
            "continuity_authority": "target-related lexical evidence in [0.38, 0.58) plus frozen DINOv2 and camera-relative motion, for at most two OBSERVE frames",
            "navigate_requires_fresh_semantic_target": True,
            "appearance_only_identity": False,
            "appearance_only_reacquire": False,
            "appearance_only_completion": False,
            "coast_limit_frames": COAST_LIMIT_FRAMES,
            "numeric_gates": {
                "locked": sc0.LOCKED_GATES,
                "reacquire": sc0.REACQUIRE_GATES,
                "continuity_score": "reuses SC0 locked score and gates but cannot emit identity or NAVIGATE below lexical admission",
            },
            "geometry_split": "RapidOCR CTC-aligned word box supplies steering when uniquely goal-related; otherwise the OCR token or merged-line center remains explicit fallback",
        },
        "cache": {
            "ocr": {"path": str(args.ocr_cache), "build": ocr_build},
            "embeddings": {
                "path": str(args.embedding_cache),
                "rows": int(pooled_patch.shape[0]),
                "dimensions": int(pooled_patch.shape[1]),
                "attached_candidates": attached,
                "build": embedding_build,
            },
        },
        "legacy_metrics": legacy_metrics,
        "legacy_direction_metrics": legacy_direction_metrics,
        "sc1_metrics": dual_metrics,
        "per_video": {"legacy": legacy_per_video, "sc1": _dual_by_video(specs)},
        "development_gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "runtime_s": round(time.perf_counter() - started, 4),
        "claim_ceiling": (
            "ArTVideo OCR-proposal Development replay with evaluator-injected proposal gaps. "
            "OBSERVE is an evidence-acquisition request, not navigation or identity. Not live-camera, "
            "open-world identity, metric arrival, product, user-benefit, or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "metrics": dual_metrics, "gate": result["development_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
