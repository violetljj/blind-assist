from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import artvideo_craft_cold_start_router as sc4
import artvideo_dual_state_replay as sc1w
import artvideo_multi_hypothesis_belief as sc5
import artvideo_ocr_replay as replay
import artvideo_semantic_visual_replay as sc0


@dataclass(frozen=True)
class PublicInstanceBinding:
    binding_id: str
    goal_text: str
    video: str
    anchor_frame_id: int
    anchor_frame_key: str
    anchor_box: np.ndarray
    anchor_crop_path: Path
    anchor_crop_sha256: str
    opaque_anchor: str


@dataclass(frozen=True)
class ArmDecision:
    selected: sc5.Observation | None
    action: str
    authority: str | None
    bound_hypothesis_id: int | None = None
    reference_score: float | None = None
    reference_margin: float | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _runner_up(scores: list[float]) -> float:
    ranked = sorted(scores, reverse=True)
    return ranked[1] if len(ranked) > 1 else 0.0


def _box_overlap(left: np.ndarray, right: np.ndarray) -> float:
    lx0, ly0 = left[:, 0].min(), left[:, 1].min()
    lx1, ly1 = left[:, 0].max(), left[:, 1].max()
    rx0, ry0 = right[:, 0].min(), right[:, 1].min()
    rx1, ry1 = right[:, 0].max(), right[:, 1].max()
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0, min(ly1, ry1) - max(ly0, ry0)
    )
    return intersection / max(1.0, (rx1 - rx0) * (ry1 - ry0))


class FrozenSourceEligibility:
    """Expose the exact SC5 fresh-source admission without temporal identity."""

    def __init__(self, goal: str):
        self.goal = goal
        self.router = sc4.L10SC4CraftColdStartRouter(goal)
        self.observer = sc5.L10SC5MultiHypothesisController(goal, "AMBIGUOUS")

    def step(self, primary: list[Any], secondary: list[Any]) -> list[sc5.Observation]:
        self.router.step(primary, secondary)
        primary_fresh, _primary_continuity = self.observer._observations("RAPIDOCR", primary)
        secondary_fresh, _secondary_continuity = self.observer._observations("CRAFT", secondary)
        if primary_fresh:
            return primary_fresh
        if self.router.primary_controller.tracker.long is None:
            return [
                observation
                for observation in secondary_fresh
                if self.router._secondary_scope_match(observation.candidate)
            ]
        return []


class SC5TextArm:
    def __init__(self, goal: str):
        self.controller = sc5.L10SC5MultiHypothesisController(goal, "AMBIGUOUS")

    def step(self, frame_id: int, primary: list[Any], secondary: list[Any]) -> ArmDecision:
        del frame_id
        decision = self.controller.step(primary, secondary)
        return ArmDecision(selected=None, action=decision.observation_action, authority=None)


class RB0StatelessArm:
    def __init__(self, goal: str, reference_embedding: np.ndarray):
        self.eligibility = FrozenSourceEligibility(goal)
        self.reference_embedding = reference_embedding

    def step(self, frame_id: int, primary: list[Any], secondary: list[Any]) -> ArmDecision:
        del frame_id
        observations = self.eligibility.step(primary, secondary)
        if not observations:
            return ArmDecision(selected=None, action="SEARCH", authority=None)
        scores = [
            float(self.reference_embedding @ observation.candidate.pooled_patch_embedding)
            for observation in observations
        ]
        index = int(np.argmax(scores))
        margin = scores[index] - _runner_up(scores)
        if scores[index] < sc0.REACQUIRE_GATES[1] or margin < sc0.REACQUIRE_GATES[3]:
            return ArmDecision(
                selected=None,
                action="SEARCH_REFERENCE_INSUFFICIENT",
                authority=None,
                reference_score=scores[index],
                reference_margin=margin,
            )
        return ArmDecision(
            selected=observations[index],
            action=f"NAVIGATE_{sc1w._direction(observations[index].candidate.center)}",
            authority="PUBLIC_REFERENCE_STATELESS",
            reference_score=scores[index],
            reference_margin=margin,
        )


class SC6ReferenceBoundBeliefArm:
    def __init__(self, binding: PublicInstanceBinding, reference_embedding: np.ndarray):
        self.binding = binding
        self.reference_embedding = reference_embedding
        self.controller = sc5.L10SC5MultiHypothesisController(binding.goal_text, "AMBIGUOUS")
        self.bound_hypothesis_id: int | None = None
        self.bound_once = False

    def _anchor_bind(self, hypotheses: list[sc5.Hypothesis]) -> int | None:
        ranked = [
            (
                _box_overlap(hypothesis.current.candidate.box, self.binding.anchor_box),
                hypothesis.hypothesis_id,
            )
            for hypothesis in hypotheses
            if hypothesis.current is not None and hypothesis.current.fresh_semantic
        ]
        if not ranked:
            return None
        overlap, hypothesis_id = max(ranked)
        return hypothesis_id if overlap > 0.0 else None

    def _reference_rebind(self, hypotheses: list[sc5.Hypothesis]) -> tuple[int | None, float | None, float | None]:
        eligible = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.current is not None and hypothesis.current.fresh_semantic
        ]
        if not eligible:
            return None, None, None
        scores = [
            float(self.reference_embedding @ hypothesis.current.candidate.pooled_patch_embedding)
            for hypothesis in eligible
        ]
        index = int(np.argmax(scores))
        margin = scores[index] - _runner_up(scores)
        if scores[index] < sc0.REACQUIRE_GATES[1] or margin < sc0.REACQUIRE_GATES[3]:
            return None, scores[index], margin
        return eligible[index].hypothesis_id, scores[index], margin

    def step(self, frame_id: int, primary: list[Any], secondary: list[Any]) -> ArmDecision:
        self.controller.step(primary, secondary)
        hypotheses = self.controller.belief.hypotheses
        authority: str | None = None
        reference_score: float | None = None
        reference_margin: float | None = None
        if frame_id == self.binding.anchor_frame_id and self.bound_hypothesis_id is None:
            self.bound_hypothesis_id = self._anchor_bind(hypotheses)
            if self.bound_hypothesis_id is not None:
                self.bound_once = True
                authority = "PUBLIC_ANCHOR_BOX"
        bound = next(
            (
                hypothesis
                for hypothesis in hypotheses
                if hypothesis.hypothesis_id == self.bound_hypothesis_id
            ),
            None,
        )
        if bound is None and self.bound_once:
            hypothesis_id, reference_score, reference_margin = self._reference_rebind(hypotheses)
            if hypothesis_id is not None:
                self.bound_hypothesis_id = hypothesis_id
                authority = "PUBLIC_REFERENCE_REACQUIRE"
                bound = next(
                    hypothesis
                    for hypothesis in hypotheses
                    if hypothesis.hypothesis_id == hypothesis_id
                )
        if bound is None or bound.current is None or not bound.current.fresh_semantic:
            return ArmDecision(
                selected=None,
                action="SEARCH_BOUND_INSTANCE",
                authority=authority,
                bound_hypothesis_id=self.bound_hypothesis_id,
                reference_score=reference_score,
                reference_margin=reference_margin,
            )
        if authority is None:
            authority = "BOUND_TEMPORAL_FRESH_OCR"
        return ArmDecision(
            selected=bound.current,
            action=f"NAVIGATE_{sc1w._direction(bound.current.candidate.center)}",
            authority=authority,
            bound_hypothesis_id=self.bound_hypothesis_id,
            reference_score=reference_score,
            reference_margin=reference_margin,
        )


def _load_reference_embeddings(
    bindings: list[PublicInstanceBinding], model_dir: Path
) -> dict[str, np.ndarray]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    images = []
    for binding in bindings:
        if _sha256(binding.anchor_crop_path) != binding.anchor_crop_sha256:
            raise ValueError(f"anchor crop hash mismatch: {binding.binding_id}")
        bgr = cv2.imread(str(binding.anchor_crop_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(binding.anchor_crop_path)
        images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    with torch.inference_mode():
        batch = processor(images=images, return_tensors="pt")
        hidden = model(pixel_values=batch["pixel_values"].to(device)).last_hidden_state
        pooled = torch.nn.functional.normalize(hidden[:, 1:].mean(dim=1), dim=1)
    return {
        binding.binding_id: _normalize(vector)
        for binding, vector in zip(bindings, pooled.cpu().numpy().astype(np.float32), strict=True)
    }


def _load_public_bindings(path: Path) -> list[PublicInstanceBinding]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    forbidden = ("native_physical_id", "distractor_ids", "target_frame_ids", "private_truth")
    serialized = json.dumps(payload, sort_keys=True)
    if any(value in serialized for value in forbidden):
        raise ValueError("public binding manifest contains evaluator-private fields")
    root = path.parent
    return [
        PublicInstanceBinding(
            binding_id=str(row["binding_id"]),
            goal_text=str(row["goal_text"]),
            video=str(row["video"]),
            anchor_frame_id=int(row["anchor_frame_id"]),
            anchor_frame_key=str(row["anchor_frame_key"]),
            anchor_box=np.asarray(row["anchor_box_quadrilateral"], dtype=np.float32),
            anchor_crop_path=root / row["anchor_crop_path"],
            anchor_crop_sha256=str(row["anchor_crop_sha256"]),
            opaque_anchor=str(row["opaque_anchor"]),
        )
        for row in payload["bindings"]
    ]


def _load_private_truth(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["binding_id"]): row for row in payload["bindings"]}


def _selected_physical_id(
    candidate: Any, target_id: int, distractor_ids: set[int]
) -> int | None:
    truth_ids = set(candidate.truth_ids)
    if target_id in truth_ids:
        return target_id
    distractors = sorted(truth_ids & distractor_ids)
    if distractors:
        return distractors[0]
    return min(truth_ids) if truth_ids else None


def run_episode(
    arm_type: type,
    binding: PublicInstanceBinding,
    reference_embedding: np.ndarray,
    private_truth: dict[str, Any],
    primary_video: dict[str, Any],
    secondary_video: dict[str, Any],
    frame_ids: list[int],
    gap_start: int,
    gap_length: int,
) -> dict[str, Any]:
    arm = (
        arm_type(binding.goal_text)
        if arm_type is SC5TextArm
        else arm_type(binding.goal_text, reference_embedding)
        if arm_type is RB0StatelessArm
        else arm_type(binding, reference_embedding)
    )
    target_id = int(private_truth["native_physical_id"])
    distractor_ids = {int(value) for value in private_truth["same_goal_distractor_ids"]}
    target_frames = {int(value) for value in private_truth["target_frame_ids"]}
    gap = set(frame_ids[gap_start : gap_start + gap_length])
    gap_end = frame_ids[min(len(frame_ids) - 1, gap_start + gap_length - 1)]
    counts: Counter[str] = Counter()
    last_physical_id: int | None = None
    reacquire_latency: int | None = None
    after_gap_index = 0
    pre_gap_correct = False
    anchor_bound = False
    valid_authorities = {
        "PUBLIC_REFERENCE_STATELESS",
        "PUBLIC_ANCHOR_BOX",
        "PUBLIC_REFERENCE_REACQUIRE",
        "BOUND_TEMPORAL_FRESH_OCR",
    }
    for position, frame_id in enumerate(frame_ids):
        primary = primary_video["candidates"].get(frame_id, [])
        secondary = secondary_video["candidates"].get(frame_id, [])
        if frame_id in gap:
            primary = [candidate for candidate in primary if target_id not in candidate.truth_ids]
            secondary = [candidate for candidate in secondary if target_id not in candidate.truth_ids]
        decision = arm.step(frame_id, primary, secondary)
        counts["frames"] += 1
        if decision.selected is None:
            counts["search_frames"] += 1
        else:
            counts["navigation_frames"] += 1
            if decision.authority not in valid_authorities:
                counts["identity_authority_violations"] += 1
        if decision.authority == "PUBLIC_ANCHOR_BOX":
            anchor_bound = True
        if frame_id in gap:
            counts["gap_frames"] += 1
            if decision.selected is not None:
                counts["wrong_instance_gap_frames"] += 1
            continue
        target_present = frame_id in target_frames
        if target_present:
            counts["evaluable_target_frames"] += 1
        if decision.selected is None:
            if target_present:
                counts["missed_target_frames"] += 1
        else:
            selected_id = _selected_physical_id(
                decision.selected.candidate, target_id, distractor_ids
            )
            if last_physical_id is not None and selected_id is not None and selected_id != last_physical_id:
                counts["physical_id_switches"] += 1
                if selected_id != target_id:
                    counts["wrong_physical_id_switches"] += 1
            if selected_id is not None:
                last_physical_id = selected_id
            if target_present and target_id in decision.selected.candidate.truth_ids:
                counts["correct_identity_target_frames"] += 1
                if position < gap_start:
                    pre_gap_correct = True
                if frame_id > gap_end and reacquire_latency is None:
                    reacquire_latency = after_gap_index
            else:
                counts["wrong_instance_target_frames"] += 1
        if frame_id > gap_end:
            after_gap_index += 1
    reacquired = reacquire_latency is not None and reacquire_latency <= 5
    counts["reacquired"] = int(reacquired)
    counts["pre_gap_correct"] = int(pre_gap_correct)
    counts["anchor_bound"] = int(anchor_bound)
    counts["end_to_end_success"] = int(pre_gap_correct and reacquired)
    return {
        "binding_id": binding.binding_id,
        "counts": dict(counts),
        "reacquire_latency": reacquire_latency,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    latencies = []
    for row in rows:
        counts.update(row["counts"])
        if row["counts"].get("reacquired", 0):
            latencies.append(row["reacquire_latency"])
    correct = counts["correct_identity_target_frames"]
    wrong = counts["wrong_instance_target_frames"] + counts["wrong_instance_gap_frames"]
    evaluable = counts["evaluable_target_frames"]
    return {
        "episodes": len(rows),
        **dict(sorted(counts.items())),
        "exact_instance_navigation_precision": round(correct / max(1, correct + wrong), 4),
        "exact_instance_navigation_coverage": round(correct / max(1, evaluable), 4),
        "wrong_instance_frames": wrong,
        "ambiguous_search_coverage": round(counts["search_frames"] / max(1, counts["frames"]), 4),
        "gap_reacquire_rate": round(counts["reacquired"] / max(1, len(rows)), 4),
        "median_gap_reacquire_frames": statistics.median(latencies) if latencies else None,
        "end_to_end_success_rate": round(counts["end_to_end_success"] / max(1, len(rows)), 4),
    }


def _load_or_build_source(
    dataset: Path,
    videos: list[str],
    ocr_cache_path: Path,
    embedding_cache_path: Path,
    embedding_index_path: Path,
    context_cache_path: Path,
    context_index_path: Path,
    model: Path,
    primary_models: Path | None,
    secondary_runtime: Path | None,
    secondary: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if ocr_cache_path.exists():
        ocr_cache = json.loads(ocr_cache_path.read_text(encoding="utf-8"))
        ocr_build = None
    elif secondary:
        if secondary_runtime is None:
            raise ValueError("secondary runtime is required to build a missing CRAFT cache")
        ocr_cache = sc4.build_craft_cache(dataset, videos, ocr_cache_path, secondary_runtime)
        ocr_build = ocr_cache["build"]
    else:
        if primary_models is None:
            raise ValueError("primary models are required to build a missing RapidOCR cache")
        ocr_cache = sc1w.build_scoped_ocr_cache(dataset, videos, ocr_cache_path, primary_models)
        ocr_build = {"frames": len(ocr_cache["frames"]), "ocr_wall_s": ocr_cache["ocr_wall_s"]}
    embedding_build = None
    if not embedding_cache_path.exists() or not embedding_index_path.exists():
        embedding_build = sc0.build_embedding_cache(
            dataset, ocr_cache, model, embedding_cache_path, embedding_index_path
        )
    context_build = None
    if not context_cache_path.exists() or not context_index_path.exists():
        context_build = sc5.build_context_embedding_cache(
            dataset, ocr_cache, model, context_cache_path, context_index_path
        )
    with np.load(embedding_cache_path, allow_pickle=False) as payload:
        embeddings = payload["pooled_patch"].astype(np.float32, copy=True)
    embedding_index = json.loads(embedding_index_path.read_text(encoding="utf-8"))
    loaded, annotation_paths = sc4.load_videos(
        dataset, videos, ocr_cache, embedding_index, embeddings
    )
    with np.load(context_cache_path, allow_pickle=False) as payload:
        context_embeddings = payload["pooled_patch"].astype(np.float32, copy=True)
    context_index = json.loads(context_index_path.read_text(encoding="utf-8"))
    attached = sc5.attach_context_embeddings(
        loaded, annotation_paths, context_index, context_embeddings
    )
    return loaded, {
        "ocr": {"path": str(ocr_cache_path), "build": ocr_build},
        "local_embeddings": {
            "path": str(embedding_cache_path),
            "index": str(embedding_index_path),
            "build": embedding_build,
        },
        "context_embeddings": {
            "path": str(context_cache_path),
            "index": str(context_index_path),
            "attached": attached,
            "build": context_build,
        },
    }


def _parity_evidence(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for video, row in payload["per_video"].items():
            rows.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "video": video,
                    **row["unique_router_parity"],
                }
            )
    return {"rows": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L10-SC6 public reference-bound exact-instance belief.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--public-bindings", type=Path, required=True)
    parser.add_argument("--private-truth", type=Path, required=True)
    parser.add_argument("--primary-ocr-cache", type=Path, required=True)
    parser.add_argument("--primary-embedding-cache", type=Path, required=True)
    parser.add_argument("--primary-embedding-index", type=Path, required=True)
    parser.add_argument("--primary-context-cache", type=Path, required=True)
    parser.add_argument("--primary-context-index", type=Path, required=True)
    parser.add_argument("--primary-models", type=Path)
    parser.add_argument("--secondary-ocr-cache", type=Path, required=True)
    parser.add_argument("--secondary-embedding-cache", type=Path, required=True)
    parser.add_argument("--secondary-embedding-index", type=Path, required=True)
    parser.add_argument("--secondary-context-cache", type=Path, required=True)
    parser.add_argument("--secondary-context-index", type=Path, required=True)
    parser.add_argument("--secondary-runtime", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--unique-parity-result", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-length", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()

    bindings = _load_public_bindings(args.public_bindings)
    private_truth = _load_private_truth(args.private_truth)
    if {binding.binding_id for binding in bindings} != set(private_truth):
        parser.error("public/private binding IDs differ")
    videos = sorted({binding.video for binding in bindings})
    primary_videos, primary_cache = _load_or_build_source(
        args.dataset,
        videos,
        args.primary_ocr_cache,
        args.primary_embedding_cache,
        args.primary_embedding_index,
        args.primary_context_cache,
        args.primary_context_index,
        args.model,
        args.primary_models,
        None,
        False,
    )
    secondary_videos, secondary_cache = _load_or_build_source(
        args.dataset,
        videos,
        args.secondary_ocr_cache,
        args.secondary_embedding_cache,
        args.secondary_embedding_index,
        args.secondary_context_cache,
        args.secondary_context_index,
        args.model,
        None,
        args.secondary_runtime,
        True,
    )
    primary_by_name = {video["name"]: video for video in primary_videos}
    secondary_by_name = {video["name"]: video for video in secondary_videos}
    reference_embeddings = _load_reference_embeddings(bindings, args.model)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {
        "SC5_TEXT": [],
        "RB0_STATELESS": [],
        "SC6_RB_BELIEF": [],
    }
    for binding in bindings:
        truth = private_truth[binding.binding_id]
        target_frames = sorted(int(value) for value in truth["target_frame_ids"])
        last_frame = max(target_frames)
        frame_ids = list(range(binding.anchor_frame_id, last_frame + 1))
        if len(frame_ids) < args.gap_length + 6:
            continue
        for fraction in (0.30, 0.50, 0.70):
            start = max(2, min(len(frame_ids) - args.gap_length - 3, int(len(frame_ids) * fraction)))
            for name, arm_type in (
                ("SC5_TEXT", SC5TextArm),
                ("RB0_STATELESS", RB0StatelessArm),
                ("SC6_RB_BELIEF", SC6ReferenceBoundBeliefArm),
            ):
                rows_by_arm[name].append(
                    run_episode(
                        arm_type,
                        binding,
                        reference_embeddings[binding.binding_id],
                        truth,
                        primary_by_name[binding.video],
                        secondary_by_name[binding.video],
                        frame_ids,
                        start,
                        args.gap_length,
                    )
                )
    metrics = {name: summarize(rows) for name, rows in rows_by_arm.items()}
    rb0 = metrics["RB0_STATELESS"]
    successor = metrics["SC6_RB_BELIEF"]
    parity = _parity_evidence(args.unique_parity_result)
    checks = {
        "precision_at_least_95_percent": successor["exact_instance_navigation_precision"] >= 0.95,
        "coverage_at_least_70_percent": successor["exact_instance_navigation_coverage"] >= 0.70,
        "coverage_gain_over_rb0_at_least_5pp": (
            successor["exact_instance_navigation_coverage"]
            - rb0["exact_instance_navigation_coverage"]
            >= 0.05
        ),
        "wrong_commit_not_above_rb0": successor["wrong_instance_frames"] <= rb0["wrong_instance_frames"],
        "wrong_physical_id_switches_at_most_one": successor.get("wrong_physical_id_switches", 0) <= 1,
        "identity_authority_violations_zero": successor.get("identity_authority_violations", 0) == 0,
        "video14_video16_unique_parity": parity["passed"],
    }
    passed = all(checks.values())
    status = "SC6_REFERENCE_BOUND_BELIEF_EFFECT" if passed else "STOP_SC6_FROZEN_DINO_NOT_SUFFICIENT"
    result = {
        "schema": "l10_sc6_public_reference_bound_exact_instance_belief_v0",
        "status": status,
        "videos": videos,
        "bindings": len(bindings),
        "episode_protocol": {
            "episodes_per_binding": 3,
            "artificial_target_proposal_gap_frames": args.gap_length,
            "gap_positions": [0.30, 0.50, 0.70],
            "reacquire_window_frames": 5,
        },
        "frozen_algorithm": {
            "single_change": "PublicInstanceBinding(goal text + public anchor frame + public anchor box/crop).",
            "sc4_routing": "unchanged",
            "sc5": {
                "max_hypotheses": sc5.MAX_HYPOTHESES,
                "ttl_frames": sc5.HYPOTHESIS_TTL_FRAMES,
                "fresh_association_gate": sc5.FRESH_ASSOCIATION_GATE,
                "continuity_association_gate": sc5.CONTINUITY_ASSOCIATION_GATE,
                "context_fraction_per_axis": sc5.CONTEXT_FRACTION_PER_AXIS,
            },
            "reference_reacquire_gate": {
                "local_dino_cosine": sc0.REACQUIRE_GATES[1],
                "top1_margin": sc0.REACQUIRE_GATES[3],
            },
            "authority": (
                "The public anchor may create exact-instance authority. OCR admits candidates. "
                "DINO/motion may associate, propagate, or reference-reacquire but cannot create identity without the binding."
            ),
        },
        "input_firewall": {
            "public_bindings_path": str(args.public_bindings),
            "public_bindings_sha256": _sha256(args.public_bindings),
            "private_truth_path": str(args.private_truth),
            "private_truth_sha256": _sha256(args.private_truth),
            "private_truth_used_by_controllers": False,
        },
        "cache": {"primary": primary_cache, "secondary": secondary_cache},
        "metrics": metrics,
        "effect_gate": {"checks": checks, "passed": passed},
        "unique_parity_evidence": parity,
        "runtime_s": round(time.perf_counter() - started, 4),
        "claim_ceiling": (
            "Source-disjoint public-reference-bound ArTVideo replay with evaluator-private native physical IDs "
            "and injected proposal gaps. It may establish exact-instance binding/continuity/reacquisition for this "
            "cohort only. It is not executed active motion, metric arrival, completion, product, user-benefit, or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "metrics": metrics, "gate": result["effect_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
