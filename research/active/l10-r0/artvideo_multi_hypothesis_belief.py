from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import artvideo_craft_cold_start_router as sc4
import artvideo_dual_state_replay as sc1w
import artvideo_ocr_replay as replay
import artvideo_semantic_visual_replay as sc0


MAX_HYPOTHESES = 4
HYPOTHESIS_TTL_FRAMES = 5
CONTEXT_FRACTION_PER_AXIS = 0.50
FRESH_ASSOCIATION_GATE = 0.52
CONTINUITY_ASSOCIATION_GATE = 0.62


@dataclass(frozen=True)
class Observation:
    source: str
    index: int
    candidate: Any
    fresh_semantic: bool
    lexical_score: float
    context_words: frozenset[str]


@dataclass
class Hypothesis:
    hypothesis_id: int
    birth_source: str
    local_embedding: np.ndarray
    context_embedding: np.ndarray
    center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    context_words: frozenset[str] = frozenset()
    functional_role: str = "UNKNOWN"
    misses: int = 0
    age: int = 0
    fresh_semantic_age: int = 0
    current: Observation | None = None
    last_truth_ids: frozenset[int] = frozenset()
    evaluator_switches: int = 0

    def update(self, observation: Observation) -> None:
        candidate = observation.candidate
        previous = self.center
        self.center = candidate.center
        self.velocity = (
            0.65 * self.velocity[0] + 0.35 * (self.center[0] - previous[0]),
            0.65 * self.velocity[1] + 0.35 * (self.center[1] - previous[1]),
        )
        self.local_embedding = sc0._normalize(
            0.85 * self.local_embedding + 0.15 * candidate.pooled_patch_embedding
        )
        self.context_embedding = sc0._normalize(
            0.85 * self.context_embedding + 0.15 * candidate.context_embedding
        )
        self.context_words = observation.context_words
        self.misses = 0
        self.age += 1
        self.fresh_semantic_age = 0 if observation.fresh_semantic else self.fresh_semantic_age + 1
        self.current = observation
        truth_ids = frozenset(candidate.truth_ids)
        if self.last_truth_ids and truth_ids and not (self.last_truth_ids & truth_ids):
            self.evaluator_switches += 1
        if truth_ids:
            self.last_truth_ids = truth_ids


@dataclass(frozen=True)
class BeliefDecision:
    task_state: str
    hypotheses: tuple[Hypothesis, ...]
    selected: Observation | None
    observation_action: str
    appearance_only_identity_violations: int = 0


def _normalized_embedding(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def _context_bounds(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0 = box[:, 0].min(), box[:, 1].min()
    x1, y1 = box[:, 0].max(), box[:, 1].max()
    dx = max(1.0, x1 - x0) * CONTEXT_FRACTION_PER_AXIS
    dy = max(1.0, y1 - y0) * CONTEXT_FRACTION_PER_AXIS
    return (
        max(0, int(math.floor(x0 - dx))),
        max(0, int(math.floor(y0 - dy))),
        min(width, int(math.ceil(x1 + dx))),
        min(height, int(math.ceil(y1 + dy))),
    )


def build_context_embedding_cache(
    dataset: Path,
    ocr_cache: dict[str, Any],
    model_dir: Path,
    cache_path: Path,
    index_path: Path,
    batch_size: int = 32,
) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    crops: list[Image.Image] = []
    items: list[dict[str, Any]] = []
    for frame_key in sorted(ocr_cache["frames"]):
        cached = ocr_cache["frames"][frame_key]
        image = cv2.imread(str(dataset / Path(frame_key)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(dataset / Path(frame_key))
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for detection_index, detection in enumerate(cached["detections"]):
            box = np.asarray(detection["box"], dtype=np.float32)
            x0, y0, x1, y1 = _context_bounds(box, width, height)
            crops.append(Image.fromarray(rgb[y0:y1, x0:x1]))
            items.append(
                {
                    "embedding_index": len(items),
                    "frame_key": frame_key,
                    "detection_index": detection_index,
                    "ocr_text": detection["text"],
                    "crop_xyxy": [x0, y0, x1, y1],
                }
            )
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = processor(images=crops[start : start + batch_size], return_tensors="pt")
            hidden = model(pixel_values=batch["pixel_values"].to(device)).last_hidden_state
            pooled = torch.nn.functional.normalize(hidden[:, 1:].mean(dim=1), dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
    pooled_patch = np.concatenate(vectors, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, pooled_patch=pooled_patch)
    index_path.write_text(
        json.dumps(
            {
                "schema": "l10_sc5_dinov2_context_embedding_index_v0",
                "context_fraction_per_axis": CONTEXT_FRACTION_PER_AXIS,
                "model": str(model_dir),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "crops": len(items),
        "dimensions": int(pooled_patch.shape[1]),
        "device": device,
        "embedding_wall_s": round(time.perf_counter() - started, 4),
    }


def attach_context_embeddings(
    videos: list[dict[str, Any]],
    annotation_paths: list[Path],
    index_payload: dict[str, Any],
    embeddings: np.ndarray,
) -> int:
    by_key: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in index_payload["items"]:
        by_key[item["frame_key"]][int(item["detection_index"])] = item
    attached = 0
    for video, annotation_path in zip(videos, annotation_paths, strict=True):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        names = {int(row["frame_id"]): str(row["frame_jpg"]) for row in annotation["frame"]}
        for frame_id, candidates in video["candidates"].items():
            frame_key = (Path("Test/frame") / annotation_path.stem / names[frame_id]).as_posix()
            indexed = by_key.get(frame_key, {})
            if len(indexed) != len(candidates):
                raise ValueError(f"context embedding/candidate mismatch for {frame_key}")
            for detection_index, candidate in enumerate(candidates):
                item = indexed[detection_index]
                if item["ocr_text"] != candidate.text:
                    raise ValueError(f"context OCR/index mismatch for {frame_key} #{detection_index}")
                candidate.context_embedding = _normalized_embedding(
                    embeddings[int(item["embedding_index"])]
                )
                attached += 1
    return attached


def _context_words(candidate: Any, candidates: list[Any]) -> frozenset[str]:
    width = float(candidate.frame_width)
    height = float(candidate.frame_height)
    x0, y0, x1, y1 = _context_bounds(candidate.box, int(width), int(height))
    cx, cy = candidate.center
    words: set[str] = set()
    for other in candidates:
        if other is candidate:
            continue
        ox, oy = other.center
        if not (x0 / width <= ox <= x1 / width and y0 / height <= oy <= y1 / height):
            continue
        horizontal = "L" if ox < cx - 0.02 else "R" if ox > cx + 0.02 else "C"
        vertical = "U" if oy < cy - 0.02 else "D" if oy > cy + 0.02 else "C"
        for token in replay.tokens(other.text):
            words.add(f"{token}@{horizontal}{vertical}")
    return frozenset(words)


class MultiHypothesisBelief:
    """Keep source-latched physical hypotheses without granting DINO identity."""

    def __init__(self, goal: str):
        self.goal = goal
        self.hypotheses: list[Hypothesis] = []
        self.next_id = 1
        self.total_evictions = 0

    @staticmethod
    def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
        if not left and not right:
            return 1.0
        return len(left & right) / max(1, len(left | right))

    def _association_score(self, hypothesis: Hypothesis, observation: Observation) -> float:
        candidate = observation.candidate
        predicted = (
            hypothesis.center[0] + hypothesis.velocity[0],
            hypothesis.center[1] + hypothesis.velocity[1],
        )
        local = max(0.0, float(hypothesis.local_embedding @ candidate.pooled_patch_embedding))
        context = max(0.0, float(hypothesis.context_embedding @ candidate.context_embedding))
        spatial = replay.spatial(candidate.center, predicted)
        words = self._jaccard(hypothesis.context_words, observation.context_words)
        return 0.25 * local + 0.40 * context + 0.25 * spatial + 0.10 * words

    def _spawn(self, observation: Observation) -> None:
        if len(self.hypotheses) >= MAX_HYPOTHESES:
            ranked = sorted(
                self.hypotheses,
                key=lambda item: (item.misses, item.fresh_semantic_age, -item.age),
                reverse=True,
            )
            victim = ranked[0]
            self.hypotheses.remove(victim)
            self.total_evictions += 1
        candidate = observation.candidate
        self.hypotheses.append(
            Hypothesis(
                hypothesis_id=self.next_id,
                birth_source=observation.source,
                local_embedding=_normalized_embedding(candidate.pooled_patch_embedding.copy()),
                context_embedding=_normalized_embedding(candidate.context_embedding.copy()),
                center=candidate.center,
                context_words=observation.context_words,
                current=observation,
                last_truth_ids=frozenset(candidate.truth_ids),
            )
        )
        self.next_id += 1

    def update(self, fresh: list[Observation], continuity: list[Observation]) -> None:
        for hypothesis in self.hypotheses:
            hypothesis.current = None
            hypothesis.misses += 1
            hypothesis.age += 1
            hypothesis.fresh_semantic_age += 1

        assigned_hypotheses: set[int] = set()
        assigned_observations: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for hypothesis_index, hypothesis in enumerate(self.hypotheses):
            for observation_index, observation in enumerate(fresh):
                if observation.source != hypothesis.birth_source:
                    continue
                pairs.append(
                    (self._association_score(hypothesis, observation), hypothesis_index, observation_index)
                )
        for score, hypothesis_index, observation_index in sorted(pairs, reverse=True):
            if score < FRESH_ASSOCIATION_GATE:
                break
            if hypothesis_index in assigned_hypotheses or observation_index in assigned_observations:
                continue
            self.hypotheses[hypothesis_index].update(fresh[observation_index])
            assigned_hypotheses.add(hypothesis_index)
            assigned_observations.add(observation_index)

        for observation_index, observation in enumerate(fresh):
            if observation_index not in assigned_observations:
                self._spawn(observation)

        used_continuity: set[int] = set()
        for hypothesis_index, hypothesis in enumerate(self.hypotheses):
            if hypothesis.current is not None:
                continue
            ranked = [
                (self._association_score(hypothesis, observation), observation_index)
                for observation_index, observation in enumerate(continuity)
                if observation.source == hypothesis.birth_source
                and observation_index not in used_continuity
            ]
            if not ranked:
                continue
            score, observation_index = max(ranked)
            if score >= CONTINUITY_ASSOCIATION_GATE:
                hypothesis.update(continuity[observation_index])
                used_continuity.add(observation_index)

        self.hypotheses = [
            hypothesis
            for hypothesis in self.hypotheses
            if hypothesis.misses <= HYPOTHESIS_TTL_FRAMES
        ]

    def fresh_hypotheses(self) -> list[Hypothesis]:
        return [
            hypothesis
            for hypothesis in self.hypotheses
            if hypothesis.current is not None and hypothesis.current.fresh_semantic
        ]


class L10SC5MultiHypothesisController:
    name = "L10_SC5_source_latched_multi_hypothesis_belief"

    def __init__(self, goal: str, task_state: str):
        self.goal = goal
        self.task_state = task_state
        self.router = sc4.L10SC4CraftColdStartRouter(goal)
        self.belief = MultiHypothesisBelief(goal)

    def _observations(self, source: str, candidates: list[Any]) -> tuple[list[Observation], list[Observation]]:
        fresh: list[Observation] = []
        continuity: list[Observation] = []
        for index, candidate in enumerate(candidates):
            lexical_score = replay.lexical(self.goal, candidate.text)
            observation = Observation(
                source=source,
                index=index,
                candidate=candidate,
                fresh_semantic=lexical_score >= sc0.REACQUIRE_GATES[0],
                lexical_score=lexical_score,
                context_words=_context_words(candidate, candidates),
            )
            (fresh if observation.fresh_semantic else continuity).append(observation)
        return fresh, continuity

    @staticmethod
    def _approach_utility(observation: Observation) -> float:
        candidate = observation.candidate
        width = float(candidate.frame_width)
        height = float(candidate.frame_height)
        area = float(np.ptp(candidate.box[:, 0]) * np.ptp(candidate.box[:, 1])) / max(1.0, width * height)
        centered = max(0.0, 1.0 - 2.0 * abs(candidate.center[0] - 0.5))
        return 0.50 * math.sqrt(max(0.0, area)) + 0.30 * centered + 0.20 * candidate.score

    def step(self, primary: list[Any], secondary: list[Any]) -> BeliefDecision:
        routed = self.router.step(primary, secondary)
        primary_fresh, primary_continuity = self._observations("RAPIDOCR", primary)
        secondary_fresh, secondary_continuity = self._observations("CRAFT", secondary)
        if primary_fresh:
            fresh = primary_fresh
            continuity = primary_continuity
        elif self.router.primary_controller.tracker.long is None:
            admitted_secondary = [
                observation
                for observation in secondary_fresh
                if self.router._secondary_scope_match(observation.candidate)
            ]
            fresh = admitted_secondary
            continuity = secondary_continuity
        else:
            fresh = []
            continuity = primary_continuity
        self.belief.update(fresh, continuity)

        if self.task_state == "UNIQUE":
            selected = None
            if routed.active.decision.identity_index is not None:
                source = "RAPIDOCR" if routed.source_mode == "PRIMARY_RAPIDOCR" else "CRAFT"
                selected = Observation(
                    source=source,
                    index=routed.active.decision.identity_index,
                    candidate=routed.candidates[routed.active.decision.identity_index],
                    fresh_semantic=routed.active.decision.fresh_semantic,
                    lexical_score=float(routed.active.decision.lexical_score or 0.0),
                    context_words=frozenset(),
                )
            return BeliefDecision(
                task_state=self.task_state,
                hypotheses=tuple(self.belief.hypotheses),
                selected=selected,
                observation_action=routed.active.observation_action,
            )

        fresh_hypotheses = self.belief.fresh_hypotheses()
        if self.task_state == "SET_VALUED" and fresh_hypotheses:
            selected_hypothesis = max(
                fresh_hypotheses,
                key=lambda hypothesis: self._approach_utility(hypothesis.current),
            )
            return BeliefDecision(
                task_state=self.task_state,
                hypotheses=tuple(self.belief.hypotheses),
                selected=selected_hypothesis.current,
                observation_action=f"NAVIGATE_{sc1w._direction(selected_hypothesis.current.candidate.center)}",
            )

        action = "SIDESTEP_FOR_DISAMBIGUATION" if len(fresh_hypotheses) > 1 else "SCAN_FOR_FUNCTION"
        return BeliefDecision(
            task_state=self.task_state,
            hypotheses=tuple(self.belief.hypotheses),
            selected=None,
            observation_action=action,
        )


def _legal_goal_ids(annotation: dict[str, Any]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for row in annotation["annotations"]:
        normalized = replay.normalize(str(row["Transcription"]))
        if normalized and normalized != replay.normalize("###"):
            result[normalized].add(int(row["obj_id"]))
    return result


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for left in positive:
        for right in negative:
            wins += 1.0 if left > right else 0.5 if left == right else 0.0
    return round(wins / (len(positive) * len(negative)), 4)


def context_discrimination(video: dict[str, Any], tracks: list[tuple[int, str, list[int]]]) -> dict[str, Any]:
    positive_local: list[float] = []
    positive_context: list[float] = []
    negative_local: list[float] = []
    negative_context: list[float] = []
    goals = sorted({replay.normalize(goal): goal for _target, goal, _frames in tracks}.values())
    for goal in goals:
        prior: list[Any] = []
        for frame_id in sorted(video["candidates"]):
            current = [
                candidate
                for candidate in video["candidates"].get(frame_id, [])
                if replay.lexical(goal, candidate.text) >= sc0.REACQUIRE_GATES[0]
            ]
            for left in current:
                for right in prior:
                    if not left.truth_ids or not right.truth_ids:
                        continue
                    target = positive_local if left.truth_ids & right.truth_ids else negative_local
                    target.append(float(left.pooled_patch_embedding @ right.pooled_patch_embedding))
                    target = positive_context if left.truth_ids & right.truth_ids else negative_context
                    target.append(float(left.context_embedding @ right.context_embedding))
            for index, left in enumerate(current):
                for right in current[index + 1 :]:
                    if not left.truth_ids or not right.truth_ids or left.truth_ids & right.truth_ids:
                        continue
                    negative_local.append(float(left.pooled_patch_embedding @ right.pooled_patch_embedding))
                    negative_context.append(float(left.context_embedding @ right.context_embedding))
            prior = current
    return {
        "positive_pairs": len(positive_local),
        "negative_pairs": len(negative_local),
        "local_embedding_auc": _auc(positive_local, negative_local),
        "context_embedding_auc": _auc(positive_context, negative_context),
        "local_positive_median": round(statistics.median(positive_local), 4) if positive_local else None,
        "local_negative_median": round(statistics.median(negative_local), 4) if negative_local else None,
        "context_positive_median": round(statistics.median(positive_context), 4) if positive_context else None,
        "context_negative_median": round(statistics.median(negative_context), 4) if negative_context else None,
    }


def natural_task_audit(
    primary_video: dict[str, Any],
    secondary_video: dict[str, Any],
    annotation: dict[str, Any],
    tracks: list[tuple[int, str, list[int]]],
) -> dict[str, Any]:
    legal_by_goal = _legal_goal_ids(annotation)
    goals = sorted({replay.normalize(goal): goal for _target, goal, _frames in tracks}.values())
    baseline_counts: Counter[str] = Counter()
    set_counts: Counter[str] = Counter()
    exact_counts: Counter[str] = Counter()
    cardinalities: list[int] = []
    per_goal: dict[str, Any] = {}
    for goal in goals:
        legal_ids = legal_by_goal[replay.normalize(goal)]
        visible_frames = sorted(
            frame_id
            for frame_id, rows in primary_video["by_frame"].items()
            if any(int(row["obj_id"]) in legal_ids for row in rows)
        )
        baseline = sc4.L10SC4CraftColdStartRouter(goal)
        set_controller = L10SC5MultiHypothesisController(goal, "SET_VALUED")
        exact_task_state = "UNIQUE" if len(legal_ids) == 1 else "AMBIGUOUS"
        exact_controller = L10SC5MultiHypothesisController(goal, exact_task_state)
        local_baseline: Counter[str] = Counter()
        local_set: Counter[str] = Counter()
        for frame_id in visible_frames:
            primary = primary_video["candidates"].get(frame_id, [])
            secondary = secondary_video["candidates"].get(frame_id, [])
            routed = baseline.step(primary, secondary)
            if routed.active.decision.identity_index is None:
                local_baseline["missed_frames"] += 1
            else:
                candidate = routed.candidates[routed.active.decision.identity_index]
                key = "correct_frames" if candidate.truth_ids & legal_ids else "wrong_frames"
                local_baseline[key] += 1
            set_decision = set_controller.step(primary, secondary)
            exact_decision = exact_controller.step(primary, secondary)
            cardinalities.append(len(exact_decision.hypotheses))
            if len(exact_decision.hypotheses) > 1:
                exact_counts["ambiguous_belief_frames"] += 1
            if any(
                hypothesis.current is not None
                and hypothesis.current.candidate.truth_ids & legal_ids
                for hypothesis in exact_decision.hypotheses
            ):
                exact_counts["legal_instance_covered_frames"] += 1
            exact_counts["evaluable_frames"] += 1
            if exact_decision.selected is not None:
                exact_counts["navigation_outputs"] += 1
                if exact_decision.selected.candidate.truth_ids & legal_ids:
                    exact_counts["correct_navigation_outputs"] += 1
                else:
                    exact_counts["wrong_navigation_outputs"] += 1
                if exact_task_state == "AMBIGUOUS":
                    exact_counts["identity_authority_violations"] += 1
            if set_decision.selected is None:
                local_set["missed_frames"] += 1
            else:
                key = (
                    "correct_frames"
                    if set_decision.selected.candidate.truth_ids & legal_ids
                    else "wrong_frames"
                )
                local_set[key] += 1
        baseline_counts.update(local_baseline)
        set_counts.update(local_set)
        per_goal[goal] = {
            "legal_physical_ids": sorted(legal_ids),
            "task_contract": (
                "SET_VALUED for any-instance; UNIQUE for exact-instance"
                if exact_task_state == "UNIQUE"
                else "SET_VALUED for any-instance; AMBIGUOUS for exact-instance without a public binding"
            ),
            "visible_frames": len(visible_frames),
            "sc4_set_valued_reinterpretation": dict(sorted(local_baseline.items())),
            "sc5_set_valued": dict(sorted(local_set.items())),
        }

    def rates(counts: Counter[str]) -> dict[str, Any]:
        selected = counts["correct_frames"] + counts["wrong_frames"]
        total = selected + counts["missed_frames"]
        return {
            **dict(sorted(counts.items())),
            "navigation_precision": round(counts["correct_frames"] / max(1, selected), 4),
            "navigation_coverage": round(selected / max(1, total), 4),
        }

    return {
        "sc4_set_valued_natural": rates(baseline_counts),
        "sc5_set_valued_natural": rates(set_counts),
        "sc5_task_conditioned_exact_instance": {
            **dict(sorted(exact_counts.items())),
            "navigation_outputs": exact_counts["navigation_outputs"],
            "identity_authority_violations": exact_counts["identity_authority_violations"],
            "mean_belief_cardinality": round(statistics.mean(cardinalities), 4) if cardinalities else 0.0,
            "max_belief_cardinality": max(cardinalities, default=0),
            "belief_coverage": round(
                exact_counts["legal_instance_covered_frames"]
                / max(1, exact_counts["evaluable_frames"]),
                4,
            ),
            "status": (
                "AMBIGUOUS_GOALS_ABSTAIN_WITHOUT_PUBLIC_INSTANCE_BINDING"
                if any(len(legal_by_goal[replay.normalize(goal)]) > 1 for goal in goals)
                else "UNIQUE_GOALS_PRESERVE_SC4"
            ),
        },
        "per_goal": per_goal,
    }


def unique_router_parity(
    primary_video: dict[str, Any],
    secondary_video: dict[str, Any],
    tracks: list[tuple[int, str, list[int]]],
    gap_length: int,
) -> dict[str, Any]:
    mismatches: Counter[str] = Counter()
    comparisons = 0
    for target_id, goal, frame_ids in tracks:
        for fraction in (0.30, 0.50, 0.70):
            start = max(2, min(len(frame_ids) - gap_length - 3, int(len(frame_ids) * fraction)))
            gap = set(frame_ids[start : start + gap_length])
            baseline = sc4.L10SC4CraftColdStartRouter(goal)
            successor = L10SC5MultiHypothesisController(goal, "UNIQUE")
            for frame_id in frame_ids:
                primary = primary_video["candidates"].get(frame_id, [])
                secondary = secondary_video["candidates"].get(frame_id, [])
                if frame_id in gap:
                    primary = [candidate for candidate in primary if target_id not in candidate.truth_ids]
                    secondary = [candidate for candidate in secondary if target_id not in candidate.truth_ids]
                left = baseline.step(primary, secondary)
                right = successor.step(primary, secondary)
                comparisons += 1
                if (left.active.decision.identity_index is None) != (right.selected is None):
                    mismatches["identity_presence"] += 1
                elif left.active.decision.identity_index is not None and right.selected is not None:
                    left_candidate = left.candidates[left.active.decision.identity_index]
                    if left_candidate is not right.selected.candidate:
                        mismatches["candidate_identity"] += 1
                if left.active.observation_action != right.observation_action:
                    mismatches["observation_action"] += 1
    return {
        "comparisons": comparisons,
        "mismatches": dict(sorted(mismatches.items())),
        "passed": not mismatches,
    }


def _load_source(
    dataset: Path,
    videos: list[str],
    ocr_cache_path: Path,
    embedding_cache_path: Path,
    embedding_index_path: Path,
    context_cache_path: Path,
    context_index_path: Path,
    model: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    ocr_cache = json.loads(ocr_cache_path.read_text(encoding="utf-8"))
    with np.load(embedding_cache_path, allow_pickle=False) as payload:
        embeddings = payload["pooled_patch"].astype(np.float32, copy=True)
    embedding_index = json.loads(embedding_index_path.read_text(encoding="utf-8"))
    loaded, annotation_paths = sc4.load_videos(
        dataset, videos, ocr_cache, embedding_index, embeddings
    )
    context_build = None
    if not context_cache_path.exists() or not context_index_path.exists():
        context_build = build_context_embedding_cache(
            dataset, ocr_cache, model, context_cache_path, context_index_path
        )
    with np.load(context_cache_path, allow_pickle=False) as payload:
        context_embeddings = payload["pooled_patch"].astype(np.float32, copy=True)
    context_index = json.loads(context_index_path.read_text(encoding="utf-8"))
    attached = attach_context_embeddings(
        loaded, annotation_paths, context_index, context_embeddings
    )
    return loaded, ocr_cache, {
        "path": str(context_cache_path),
        "index": str(context_index_path),
        "attached": attached,
        "build": context_build,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L10-SC5 multi-hypothesis belief diagnostic.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--primary-ocr-cache", type=Path, required=True)
    parser.add_argument("--primary-embedding-cache", type=Path, required=True)
    parser.add_argument("--primary-embedding-index", type=Path, required=True)
    parser.add_argument("--primary-context-cache", type=Path, required=True)
    parser.add_argument("--primary-context-index", type=Path, required=True)
    parser.add_argument("--secondary-ocr-cache", type=Path, required=True)
    parser.add_argument("--secondary-embedding-cache", type=Path, required=True)
    parser.add_argument("--secondary-embedding-index", type=Path, required=True)
    parser.add_argument("--secondary-context-cache", type=Path, required=True)
    parser.add_argument("--secondary-context-index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--gap-length", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()

    primary_videos, _primary_cache, primary_context = _load_source(
        args.dataset,
        args.videos,
        args.primary_ocr_cache,
        args.primary_embedding_cache,
        args.primary_embedding_index,
        args.primary_context_cache,
        args.primary_context_index,
        args.model,
    )
    secondary_videos, _secondary_cache, secondary_context = _load_source(
        args.dataset,
        args.videos,
        args.secondary_ocr_cache,
        args.secondary_embedding_cache,
        args.secondary_embedding_index,
        args.secondary_context_cache,
        args.secondary_context_index,
        args.model,
    )
    secondary_by_name = {video["name"]: video for video in secondary_videos}
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    per_video: dict[str, Any] = {}
    for primary_video in primary_videos:
        secondary_video = secondary_by_name[primary_video["name"]]
        annotation_path = args.dataset / "Test/json" / f"{primary_video['name']}.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        tracks = replay.eligible_tracks(primary_video, args.minimum_frames)
        per_video[primary_video["name"]] = {
            "eligible_tracks": len(tracks),
            "natural_task_audit": natural_task_audit(
                primary_video, secondary_video, annotation, tracks
            ),
            "context_discrimination": context_discrimination(primary_video, tracks),
            "unique_router_parity": unique_router_parity(
                primary_video, secondary_video, tracks, args.gap_length
            ),
        }

    parity_passed = all(row["unique_router_parity"]["passed"] for row in per_video.values())
    video17_rows = [row for name, row in per_video.items() if name == "video17"]
    set_baseline_saturated = all(
        row["natural_task_audit"]["sc4_set_valued_natural"]["navigation_precision"] == 1.0
        for row in video17_rows
    ) if video17_rows else None
    set_coverage_effect = all(
        row["natural_task_audit"]["sc5_set_valued_natural"]["navigation_coverage"]
        > row["natural_task_audit"]["sc4_set_valued_natural"]["navigation_coverage"]
        for row in video17_rows
    ) if video17_rows else None
    context_outperforms_local = all(
        row["context_discrimination"]["context_embedding_auc"] is not None
        and row["context_discrimination"]["local_embedding_auc"] is not None
        and row["context_discrimination"]["context_embedding_auc"]
        > row["context_discrimination"]["local_embedding_auc"]
        for row in video17_rows
    ) if video17_rows else None
    status = (
        "SC5_SET_VALUED_COVERAGE_EFFECT_EXACT_INSTANCE_BLOCKED"
        if video17_rows and set_baseline_saturated and set_coverage_effect
        else "SC5_VIDEO17_TASK_AUTHORITY_BLOCKED"
        if video17_rows
        else "SC5_MULTI_HYPOTHESIS_DIAGNOSTIC_COMPLETE"
        if parity_passed
        else "SC5_MULTI_HYPOTHESIS_NON_REGRESSION_FAILED"
    )
    result = {
        "schema": "l10_sc5_source_latched_multi_hypothesis_belief_v0",
        "status": status,
        "videos": args.videos,
        "algorithm": {
            "belief": f"At most K={MAX_HYPOTHESES} source-latched physical hypotheses.",
            "birth_authority": "Fresh RapidOCR or word-scope-legal cold-start CRAFT semantic evidence only.",
            "continuity": "Each hypothesis owns independent local DINO, 2x-context DINO, OCR-context fingerprint, motion, age, and observation quality state.",
            "appearance_authority": "DINO/motion may propagate a hypothesis but never creates identity, selects a target, navigates, arrives, or completes.",
            "task_states": {
                "UNIQUE": "Preserve the frozen SC4 path; requires an external contract that exactly one physical target is legal.",
                "SET_VALUED": "Any legal same-goal physical instance may be selected; choose the lowest-cost current observation.",
                "AMBIGUOUS": "No public instance binding exists; retain the belief set and request disambiguating observation without navigation.",
            },
            "functional_role": "UNKNOWN unless an admitted source supplies evaluator-authoritative package/shelf/aisle/doorway/entrance role truth.",
            "numeric_configuration": {
                "max_hypotheses": MAX_HYPOTHESES,
                "hypothesis_ttl_frames": HYPOTHESIS_TTL_FRAMES,
                "context_fraction_per_axis": CONTEXT_FRACTION_PER_AXIS,
                "fresh_association_gate": FRESH_ASSOCIATION_GATE,
                "continuity_association_gate": CONTINUITY_ASSOCIATION_GATE,
                "association_weights": {"local_dino": 0.25, "context_dino": 0.40, "motion": 0.25, "ocr_context": 0.10},
            },
        },
        "baseline": {
            "path": str(args.baseline_result),
            "status": baseline.get("status"),
            "cascade_metrics": baseline.get("cascade_metrics"),
        },
        "context_cache": {"primary": primary_context, "secondary": secondary_context},
        "per_video": per_video,
        "adjudication": {
            "unique_router_non_regression": parity_passed,
            "video17_sc4_set_valued_precision_already_saturated": set_baseline_saturated,
            "video17_sc5_set_valued_coverage_effect": set_coverage_effect,
            "video17_context_fingerprint_outperforms_local_dino": context_outperforms_local,
            "exact_instance_from_text_only": "NOT_EVALUABLE when the same normalized public goal maps to multiple physical IDs and supplies no reference/context/function binding.",
            "functional_disambiguation": "NOT_EVALUABLE on ArTVideo video17 because package/shelf/aisle/doorway/entrance role truth is absent.",
            "active_motion_effect": "NOT_EVALUATED; actions are emitted but the replay contains no executed counterfactual view change.",
            "next_information_source": "Admit either a public exact-instance binding (reference crop/initial box/context anchor) or evaluator-authoritative functional role and aperture/entrance association before an exact-instance or functional SC5 gate.",
        },
        "runtime_s": round(time.perf_counter() - started, 4),
        "claim_ceiling": (
            "Consumed ArTVideo replay diagnostic of belief representation, task authority, context separability, "
            "and frozen UNIQUE-path parity. It is not exact-instance selection evidence without a public binding, "
            "not functional grounding without role truth, and not executed active-view, arrival, completion, product, "
            "user-benefit, or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "adjudication": result["adjudication"],
                "per_video": per_video,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
