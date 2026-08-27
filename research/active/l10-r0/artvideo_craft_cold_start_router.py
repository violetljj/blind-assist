from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import artvideo_dual_state_replay as sc1w
import artvideo_ocr_replay as replay
import artvideo_opportunity_active_search as active
import artvideo_semantic_visual_replay as sc0


@dataclass(frozen=True)
class CascadeDecision:
    active: active.ActiveDecision
    candidates: list[Any]
    source_mode: str
    primary_semantic_matches: int
    secondary_semantic_matches: int


class L10SC4CraftColdStartRouter:
    """Use CRAFT only while the primary source has never established belief.

    The RapidOCR controller advances on every frame and remains the only owner
    of its DINO/motion continuity state. CRAFT may supply a fresh, localized
    identity while that primary belief is still cold, but is permanently
    blocked after the primary has acquired once. This makes the second source
    an acquisition specialist rather than a LOST-time identity override.
    """

    name = "L10_SC4_craft_cold_start_router"

    def __init__(self, goal: str):
        self.goal = goal
        self.primary_controller = active.L10SC2OpportunityActiveSearch(goal)

    def _fresh_secondary(self, candidate: Any, lexical_score: float) -> active.ActiveDecision:
        carrier_center, carrier_scope = self.primary_controller.tracker._carrier_geometry(candidate)
        decision = sc1w.DualStateDecision(
            semantic_state=sc1w.SemanticState.TARGET.value,
            continuity_state=sc1w.ContinuityState.UNBOUND.value,
            identity_index=0,
            continuity_index=None,
            observation_index=None,
            action_mode="NAVIGATE",
            evidence_deficit=None,
            fresh_semantic=True,
            semantic_age_frames=0,
            carrier_center=carrier_center,
            carrier_scope=carrier_scope,
            carrier_updated=False,
            lexical_score=lexical_score,
        )
        return active.ActiveDecision(
            decision=decision,
            source_action_mode="NAVIGATE",
            source_semantic_state=sc1w.SemanticState.TARGET.value,
            observation_action=f"NAVIGATE_{sc1w._direction(carrier_center)}",
        )

    def _secondary_scope_match(self, candidate: Any) -> bool:
        """A one-token CRAFT box owns a word; a multi-token box owns only its full span."""
        goal_tokens = replay.tokens(self.goal)
        observed_tokens = replay.tokens(candidate.text)
        if len(goal_tokens) == 1 and len(observed_tokens) == 1:
            return replay.lexical(self.goal, candidate.text) >= sc0.REACQUIRE_GATES[0]
        return replay.normalize(candidate.text) == replay.normalize(self.goal)

    def step(self, primary: list[Any], secondary: list[Any]) -> CascadeDecision:
        primary_active = self.primary_controller.step(primary)
        primary_matches = [
            candidate for candidate in primary if replay.lexical(self.goal, candidate.text) >= sc0.REACQUIRE_GATES[0]
        ]
        secondary_lexical_matches = [
            candidate
            for candidate in secondary
            if replay.lexical(self.goal, candidate.text) >= sc0.REACQUIRE_GATES[0]
        ]
        secondary_matches = [
            candidate for candidate in secondary_lexical_matches if self._secondary_scope_match(candidate)
        ]
        primary_ever_acquired = self.primary_controller.tracker.long is not None
        if primary_matches:
            candidates = primary
            source_mode = "PRIMARY_RAPIDOCR"
            selected = primary_active
        elif primary_ever_acquired:
            candidates = primary
            source_mode = (
                "SECONDARY_CRAFT_BLOCKED_AFTER_PRIMARY_LOCK"
                if secondary_lexical_matches
                else "PRIMARY_NO_SEMANTIC_MATCH"
            )
            selected = primary_active
        elif len(secondary_matches) == 1:
            candidates = secondary_matches
            source_mode = "SECONDARY_CRAFT_WORD_SCOPE_UNIQUE"
            selected = self._fresh_secondary(
                secondary_matches[0], replay.lexical(self.goal, secondary_matches[0].text)
            )
        elif len(secondary_matches) > 1:
            candidates = primary
            source_mode = "SECONDARY_CRAFT_AMBIGUOUS_ABSTAIN"
            selected = primary_active
        elif secondary_lexical_matches:
            candidates = primary
            source_mode = "SECONDARY_CRAFT_SCOPE_ABSTAIN"
            selected = primary_active
        else:
            candidates = primary
            source_mode = "PRIMARY_NO_SEMANTIC_MATCH"
            selected = primary_active
        return CascadeDecision(
            active=selected,
            candidates=candidates,
            source_mode=source_mode,
            primary_semantic_matches=len(primary_matches),
            secondary_semantic_matches=len(secondary_matches),
        )


def normalize_probe_cache(
    dataset: Path,
    videos: list[str],
    probe_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    for video_name in videos:
        annotation_path = dataset / "Test/json" / f"{video_name}.json"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        names = {int(row["frame_id"]): str(row["frame_jpg"]) for row in annotation["frame"]}
        for frame_id, frame_name in names.items():
            probe_key = f"{video_name}/{frame_id}" if f"{video_name}/{frame_id}" in probe["frames"] else str(frame_id)
            source = probe["frames"][probe_key]
            image_path = dataset / "Test/frame" / video_name / frame_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            frame_key = (Path("Test/frame") / video_name / frame_name).as_posix()
            rows[frame_key] = {
                "shape": list(image.shape),
                "wall_s": source.get("wall_s"),
                "detections": [
                    {
                        "box": detection["box"],
                        "text": detection["text"],
                        "score": float(detection["score"]),
                        "words": [],
                    }
                    for detection in source["detections"]
                ],
            }
    payload = {
        "schema": "artvideo_easyocr_craft_cache_v0",
        "videos": videos,
        "backend": "EasyOCR 1.7.2 / CRAFT craft_mlt_25k / english_g2 / CUDA",
        "source_probe": str(probe_path),
        "frames": rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_craft_cache(
    dataset: Path,
    videos: list[str],
    cache_path: Path,
    runtime: Path,
) -> dict[str, Any]:
    """Build the secondary source cache with frozen EasyOCR/CRAFT defaults."""
    site_packages = runtime / "site-packages"
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
    import easyocr
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen CRAFT source build")
    reader = easyocr.Reader(
        ["en"],
        gpu="cuda",
        model_storage_directory=str(runtime / "models"),
        user_network_directory=str(runtime / "user-network"),
        download_enabled=False,
        verbose=False,
    )
    rows: dict[str, Any] = {}
    frame_times: list[float] = []
    started = time.perf_counter()
    for video in videos:
        frame_dir = dataset / "Test" / "frame" / video
        for path in sorted(frame_dir.glob("*.jpg"), key=replay.natural_key):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            before = time.perf_counter()
            output = reader.readtext(image, detail=1, paragraph=False)
            elapsed = time.perf_counter() - before
            frame_times.append(elapsed)
            rows[path.relative_to(dataset).as_posix()] = {
                "shape": list(image.shape),
                "wall_s": round(elapsed, 6),
                "detections": [
                    {
                        "box": np.asarray(box, dtype=float).tolist(),
                        "text": str(text),
                        "score": float(score),
                        "words": [],
                    }
                    for box, text, score in output
                ],
            }
    payload = {
        "schema": "artvideo_easyocr_craft_cache_v1",
        "videos": videos,
        "backend": "EasyOCR 1.7.2 / CRAFT craft_mlt_25k / english_g2 / CUDA",
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "readtext": {"detail": 1, "paragraph": False, "all_other_parameters": "EasyOCR 1.7.2 defaults"},
        "frames": rows,
        "build": {
            "frames": len(rows),
            "detections": sum(len(row["detections"]) for row in rows.values()),
            "readtext_wall_s": round(sum(frame_times), 4),
            "wall_s": round(time.perf_counter() - started, 4),
            "median_frame_s": round(statistics.median(frame_times), 4) if frame_times else None,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_videos(
    dataset: Path,
    names: list[str],
    cache: dict[str, Any],
    embedding_index: dict[str, Any],
    embeddings: np.ndarray,
) -> tuple[list[dict[str, Any]], list[Path]]:
    annotation_paths = [dataset / "Test/json" / f"{name}.json" for name in names]
    videos = [replay.load_video(dataset, path, cache) for path in annotation_paths]
    sc1w.attach_frame_geometry(videos, annotation_paths, cache)
    sc0.attach_embeddings(videos, annotation_paths, embedding_index, embeddings)
    return videos, annotation_paths


def _target_center(video: dict[str, Any], frame_id: int, target_id: int) -> tuple[float, float]:
    return sc1w._target_center(video, frame_id, target_id)


def run_episode(
    primary_video: dict[str, Any],
    secondary_video: dict[str, Any],
    target_id: int,
    goal: str,
    frame_ids: list[int],
    gap_start: int,
    gap_length: int,
) -> dict[str, Any]:
    controller = L10SC4CraftColdStartRouter(goal)
    gap = set(frame_ids[gap_start : gap_start + gap_length])
    gap_end_frame = frame_ids[min(len(frame_ids) - 1, gap_start + gap_length - 1)]
    counts: Counter[str] = Counter()
    states: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    reacquire_latency: int | None = None
    after_gap_index = 0
    pre_gap_identity_established = False

    first_primary = primary_video["candidates"].get(frame_ids[0], [])
    first_secondary = secondary_video["candidates"].get(frame_ids[0], [])
    first = controller.step(first_primary, first_secondary)
    if first.active.decision.identity_index is not None:
        pre_gap_identity_established = target_id in first.candidates[first.active.decision.identity_index].truth_ids

    for position, frame_id in enumerate(frame_ids[1:], start=1):
        primary_candidates = primary_video["candidates"].get(frame_id, [])
        secondary_candidates = secondary_video["candidates"].get(frame_id, [])
        if frame_id in gap:
            primary_candidates = [candidate for candidate in primary_candidates if target_id not in candidate.truth_ids]
            secondary_candidates = [candidate for candidate in secondary_candidates if target_id not in candidate.truth_ids]
        cascade = controller.step(primary_candidates, secondary_candidates)
        decision = cascade.active.decision
        candidates = cascade.candidates
        sources[cascade.source_mode] += 1
        actions[cascade.active.observation_action] += 1
        states[f"semantic:{decision.semantic_state}"] += 1
        states[f"continuity:{decision.continuity_state}"] += 1
        states[f"action:{decision.action_mode}"] += 1
        if decision.carrier_scope is not None:
            states[f"scope:{decision.carrier_scope}"] += 1
        if cascade.active.source_action_mode == "STOP":
            counts["source_stop_frames"] += 1
        if decision.action_mode == "STOP":
            counts["normalized_stop_frames"] += 1
        if decision.identity_index is not None and not decision.fresh_semantic:
            counts["appearance_only_identity_violations"] += 1
        if decision.action_mode == "NAVIGATE" and decision.identity_index is None:
            counts["continuity_only_navigation_violations"] += 1
        if cascade.source_mode == "SECONDARY_CRAFT_WORD_SCOPE_UNIQUE":
            counts["secondary_unique_source_frames"] += 1
            if decision.identity_index is not None:
                counts["secondary_identity_frames"] += 1

        if frame_id in gap:
            counts["gap_frames"] += 1
            if decision.identity_index is not None:
                counts["false_identity_gap_frames"] += 1
            if decision.action_mode == "OBSERVE":
                counts["gap_observation_requests"] += 1
                if decision.observation_index is not None:
                    counts["false_observation_gap_frames"] += 1
                    if decision.carrier_center is not None:
                        if sc1w._direction(decision.carrier_center) == sc1w._direction(
                            _target_center(primary_video, frame_id, target_id)
                        ):
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
            if cascade.source_mode == "SECONDARY_CRAFT_WORD_SCOPE_UNIQUE":
                counts["correct_secondary_identity_target_frames"] += 1
            if position < gap_start:
                pre_gap_identity_established = True
            if decision.carrier_center is None:
                counts["identity_frames_without_carrier"] += 1
            elif sc1w._direction(decision.carrier_center) == sc1w._direction(
                _target_center(primary_video, frame_id, target_id)
            ):
                counts["correct_identity_bearing_frames"] += 1
            else:
                counts["wrong_identity_bearing_frames"] += 1
            if frame_id > gap_end_frame and reacquire_latency is None:
                reacquire_latency = after_gap_index
        else:
            counts["wrong_identity_target_present_frames"] += 1
            if cascade.source_mode == "SECONDARY_CRAFT_WORD_SCOPE_UNIQUE":
                counts["wrong_secondary_identity_target_frames"] += 1

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
                elif sc1w._direction(decision.carrier_center) == sc1w._direction(
                    _target_center(primary_video, frame_id, target_id)
                ):
                    counts["correct_observation_bearing_frames"] += 1
                else:
                    counts["wrong_observation_bearing_frames"] += 1
            else:
                counts["wrong_observation_target_present_frames"] += 1
        if identity_correct or observation_correct:
            counts["correct_target_support_frames"] += 1
        if (
            identity_correct
            and decision.carrier_center is not None
            and sc1w._direction(decision.carrier_center)
            == sc1w._direction(_target_center(primary_video, frame_id, target_id))
        ) or (
            observation_correct
            and decision.carrier_center is not None
            and sc1w._direction(decision.carrier_center)
            == sc1w._direction(_target_center(primary_video, frame_id, target_id))
        ):
            counts["correct_direction_support_frames"] += 1
        if decision.semantic_state == sc1w.SemanticState.NONE.value:
            counts["false_none_target_present_frames"] += 1
        if frame_id > gap_end_frame:
            after_gap_index += 1

    reacquired = reacquire_latency is not None and reacquire_latency <= 5
    counts["reacquired"] = int(reacquired)
    counts["pre_gap_identity_established"] = int(pre_gap_identity_established)
    counts["end_to_end_success"] = int(pre_gap_identity_established and reacquired)
    return {
        "counts": dict(counts),
        "states": dict(states),
        "sources": dict(sources),
        "actions": dict(actions),
        "reacquire_latency": reacquire_latency,
        "pre_gap_identity_established": pre_gap_identity_established,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = sc1w.summarize_dual(rows)
    sources: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        sources.update(row["sources"])
        actions.update(row["actions"])
        counts.update(row["counts"])
    opportunities = counts["pre_gap_identity_established"]
    conditional_reacquired = sum(
        int(row["pre_gap_identity_established"] and row["counts"].get("reacquired", 0)) for row in rows
    )
    conditional_latencies = [
        row["reacquire_latency"]
        for row in rows
        if row["pre_gap_identity_established"] and row["counts"].get("reacquired", 0)
    ]
    metrics.update(
        {
            "pre_gap_identity_established_episodes": opportunities,
            "pre_gap_identity_not_established_episodes": len(rows) - opportunities,
            "acquisition_conditioned_reacquired_episodes": conditional_reacquired,
            "acquisition_conditioned_reacquire_rate": round(
                conditional_reacquired / max(1, opportunities), 4
            ),
            "end_to_end_successful_episodes": counts["end_to_end_success"],
            "end_to_end_success_rate": round(counts["end_to_end_success"] / max(1, len(rows)), 4),
            "median_acquisition_conditioned_reacquire_frames": (
                statistics.median(conditional_latencies) if conditional_latencies else None
            ),
            "source_frames": dict(sorted(sources.items())),
            "active_action_frames": dict(sorted(actions.items())),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L10-SC4 belief-latched CRAFT cold-start routing.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--primary-ocr-cache", type=Path, required=True)
    parser.add_argument("--primary-embedding-cache", type=Path, required=True)
    parser.add_argument("--primary-embedding-index", type=Path, required=True)
    parser.add_argument("--primary-models", type=Path)
    parser.add_argument("--secondary-ocr-cache", type=Path, required=True)
    parser.add_argument("--secondary-probe-result", type=Path)
    parser.add_argument("--secondary-runtime", type=Path)
    parser.add_argument("--secondary-embedding-cache", type=Path, required=True)
    parser.add_argument("--secondary-embedding-index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=12)
    parser.add_argument("--gap-length", type=int, default=4)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    if args.require_gpu:
        import torch

        if not torch.cuda.is_available():
            parser.error("--require-gpu was set but CUDA is unavailable")

    if args.primary_ocr_cache.exists():
        primary_cache = json.loads(args.primary_ocr_cache.read_text(encoding="utf-8"))
        primary_cache_build = None
    else:
        if args.primary_models is None:
            parser.error("--primary-models is required when --primary-ocr-cache is absent")
        primary_cache = sc1w.build_scoped_ocr_cache(
            args.dataset, args.videos, args.primary_ocr_cache, args.primary_models
        )
        primary_cache_build = {
            "frames": len(primary_cache["frames"]),
            "ocr_wall_s": primary_cache.get("ocr_wall_s"),
        }
    if args.secondary_ocr_cache.exists():
        secondary_cache = json.loads(args.secondary_ocr_cache.read_text(encoding="utf-8"))
        secondary_cache_build = None
    else:
        if args.secondary_probe_result is not None:
            secondary_cache = normalize_probe_cache(
                args.dataset, args.videos, args.secondary_probe_result, args.secondary_ocr_cache
            )
            secondary_cache_build = {"frames": len(secondary_cache["frames"]), "source": "normalized_probe"}
        elif args.secondary_runtime is not None:
            secondary_cache = build_craft_cache(
                args.dataset, args.videos, args.secondary_ocr_cache, args.secondary_runtime
            )
            secondary_cache_build = secondary_cache["build"]
        else:
            parser.error(
                "--secondary-probe-result or --secondary-runtime is required when --secondary-ocr-cache is absent"
            )

    if not args.primary_embedding_cache.exists() or not args.primary_embedding_index.exists():
        primary_embedding_build = sc0.build_embedding_cache(
            args.dataset,
            primary_cache,
            args.model,
            args.primary_embedding_cache,
            args.primary_embedding_index,
        )
    else:
        primary_embedding_build = None

    if not args.secondary_embedding_cache.exists() or not args.secondary_embedding_index.exists():
        secondary_embedding_build = sc0.build_embedding_cache(
            args.dataset,
            secondary_cache,
            args.model,
            args.secondary_embedding_cache,
            args.secondary_embedding_index,
        )
    else:
        secondary_embedding_build = None

    with np.load(args.primary_embedding_cache) as payload:
        primary_embeddings = payload["pooled_patch"].astype(np.float32, copy=True)
    primary_index = json.loads(args.primary_embedding_index.read_text(encoding="utf-8"))
    with np.load(args.secondary_embedding_cache) as payload:
        secondary_embeddings = payload["pooled_patch"].astype(np.float32, copy=True)
    secondary_index = json.loads(args.secondary_embedding_index.read_text(encoding="utf-8"))
    primary_videos, _ = load_videos(
        args.dataset, args.videos, primary_cache, primary_index, primary_embeddings
    )
    secondary_videos, _ = load_videos(
        args.dataset, args.videos, secondary_cache, secondary_index, secondary_embeddings
    )
    secondary_by_name = {video["name"]: video for video in secondary_videos}
    specs: list[tuple[Any, ...]] = []
    primary_specs: list[tuple[Any, ...]] = []
    for primary_video in primary_videos:
        secondary_video = secondary_by_name[primary_video["name"]]
        for target_id, goal, frame_ids in replay.eligible_tracks(primary_video, args.minimum_frames):
            for fraction in (0.30, 0.50, 0.70):
                start = max(2, min(len(frame_ids) - args.gap_length - 3, int(len(frame_ids) * fraction)))
                specs.append(
                    (primary_video, secondary_video, target_id, goal, frame_ids, start, args.gap_length)
                )
                primary_specs.append(
                    (primary_video, target_id, goal, frame_ids, start, args.gap_length)
                )

    primary_rows = [sc1w.run_dual_episode(*spec) for spec in primary_specs]
    primary_metrics = sc1w.summarize_dual(primary_rows)
    primary_active_rows = [active.run_episode(*spec) for spec in primary_specs]
    primary_opportunity_metrics = active.summarize(primary_active_rows)
    cascade_rows = [run_episode(*spec) for spec in specs]
    cascade_metrics = summarize(cascade_rows)
    safety_checks = {
        "navigation_precision_not_below_primary": cascade_metrics["navigation_precision"] >= primary_metrics["navigation_precision"],
        "identity_wrong_not_above_primary": cascade_metrics["identity_wrong_frames"] <= primary_metrics["identity_wrong_frames"],
        "identity_recall_not_below_primary": cascade_metrics["identity_target_recall"] >= primary_metrics["identity_target_recall"],
        "target_support_not_below_primary": cascade_metrics["target_support_coverage"] >= primary_metrics["target_support_coverage"],
        "direction_ready_not_below_primary": cascade_metrics["direction_ready_coverage"] >= primary_metrics["direction_ready_coverage"],
        "end_to_end_not_below_primary": cascade_metrics["end_to_end_success_rate"] >= primary_opportunity_metrics["end_to_end_success_rate"],
        "conditioned_reacquire_not_below_primary": cascade_metrics["acquisition_conditioned_reacquire_rate"] >= primary_opportunity_metrics["acquisition_conditioned_reacquire_rate"],
        "secondary_identity_wrong_zero": cascade_metrics.get("wrong_secondary_identity_target_frames", 0) == 0,
        "appearance_only_identity_violations_zero": cascade_metrics.get("appearance_only_identity_violations", 0) == 0,
        "continuity_only_navigation_violations_zero": cascade_metrics.get("continuity_only_navigation_violations", 0) == 0,
        "incomplete_ocr_set_never_emits_none": cascade_metrics.get("false_none_target_present_frames", 0) == 0,
    }
    effect_checks = {
        "secondary_correct_identity_present": cascade_metrics.get("correct_secondary_identity_target_frames", 0) > 0,
        "secondary_unique_source_used": cascade_metrics.get("secondary_unique_source_frames", 0) > 0,
        "strict_target_metric_gain": any(
            (
                cascade_metrics["identity_target_recall"] > primary_metrics["identity_target_recall"],
                cascade_metrics["target_support_coverage"] > primary_metrics["target_support_coverage"],
                cascade_metrics["direction_ready_coverage"] > primary_metrics["direction_ready_coverage"],
                cascade_metrics["end_to_end_success_rate"]
                > primary_opportunity_metrics["end_to_end_success_rate"],
            )
        ),
    }
    safety_passed = all(safety_checks.values())
    effect_passed = safety_passed and all(effect_checks.values())
    status = (
        "SC4_CRAFT_COLD_START_EFFECT"
        if effect_passed
        else "SC4_CRAFT_COLD_START_SAFE_NEUTRAL"
        if safety_passed
        else "SC4_CRAFT_COLD_START_GATE_NOT_MET"
    )
    result = {
        "schema": "l10_sc4_rapidocr_craft_cold_start_router_v0",
        "status": status,
        "videos": args.videos,
        "algorithm": {
            "primary": "Frozen RapidOCR SC1W candidate set and controller.",
            "fallback_trigger": "No primary candidate crosses the frozen lexical 0.58 semantic gate.",
            "fallback_admission": "Exactly one EasyOCR/CRAFT candidate owns the requested semantic scope: a one-token CRAFT word box may use the frozen lexical 0.58 gate, while a multi-token box must match the complete normalized goal. Multiple matches and sub-token claims inside a larger box abstain.",
            "belief_latch": "CRAFT is eligible only while the RapidOCR tracker has never established a long-term target anchor. After the first primary acquisition, CRAFT cannot acquire, navigate, or override a LOST/gap frame.",
            "source_isolation": "The RapidOCR tracker advances independently on every frame. A unique CRAFT fallback supplies current-frame semantic identity and bearing but never updates RapidOCR DINO/motion memory.",
            "identity_authority": "Fresh source text only. DINOv2 remains continuity/proposal evidence and cannot acquire, reacquire, navigate, or complete alone.",
            "active_search": "Non-exhaustive OCR non-match is UNKNOWN/TARGET_NOT_PROPOSED with a deficit-specific search action, never semantic NONE/terminal STOP.",
        },
        "episode_protocol": {
            "eligible_tracks": len(specs) // 3,
            "episodes": len(specs),
            "minimum_track_frames": args.minimum_frames,
            "artificial_gap_frames": args.gap_length,
            "gap_positions": [0.30, 0.50, 0.70],
            "reacquire_window_frames": 5,
        },
        "cache": {
            "primary_ocr": {"path": str(args.primary_ocr_cache), "build": primary_cache_build},
            "primary_embeddings": {
                "path": str(args.primary_embedding_cache),
                "index": str(args.primary_embedding_index),
                "build": primary_embedding_build,
            },
            "secondary_ocr": {"path": str(args.secondary_ocr_cache), "build": secondary_cache_build},
            "secondary_embeddings": {
                "path": str(args.secondary_embedding_cache),
                "index": str(args.secondary_embedding_index),
                "build": secondary_embedding_build,
            },
        },
        "primary_sc1w_metrics": primary_metrics,
        "primary_opportunity_metrics": primary_opportunity_metrics,
        "cascade_metrics": cascade_metrics,
        "gate": {
            "safety_checks": safety_checks,
            "effect_checks": effect_checks,
            "safety_passed": safety_passed,
            "effect_passed": effect_passed,
        },
        "runtime_s": round(time.perf_counter() - started, 4),
        "claim_ceiling": (
            "ArTVideo Development/consumed replay of a two-source OCR cascade with evaluator-injected gaps. "
            "It is not fresh transfer until a source-disjoint video is opened once, and not live active-view causality, "
            "open-world absence, metric arrival, product, user-benefit, or safety evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "metrics": cascade_metrics, "gate": result["gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
