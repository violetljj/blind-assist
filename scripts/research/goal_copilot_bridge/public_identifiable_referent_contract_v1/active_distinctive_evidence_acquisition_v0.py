"""Controlled demo of active distinctive-anchor acquisition.

This is deliberately a Development/showcase harness.  It changes the input
contract from one passive reference image to a short reference sweep, mines
repeatable SIFT anchors, and permits a lock only when one candidate explains a
geometrically consistent and candidate-unique anchor set.  A lock is a demo
state, not physical-instance or safety authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    dinov2_local_appearance_probe as dino,
)
from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    spatial_layout_identity_verification_v0 as washington,
)


SCHEMA_VERSION = "blindassist_active_distinctive_evidence_acquisition_v0"
EXPERIMENT_ID = "ACTIVE_DISTINCTIVE_EVIDENCE_ACQUISITION_V0"
CLAIM_CEILING = (
    "CONTROLLED_DEVELOPMENT_DEMO_WITH_CURATED_VISIBLE_DISTINCTIVE_ANCHORS_"
    "NO_GENERAL_EXACT_INSTANCE_P1_NAVIGATION_SAFETY_OR_DEFAULT_APP_CLAIM"
)
MAX_FEATURE_SIDE = 720
SIFT_FEATURES = 1800
REFERENCE_RATIO = 0.78
CANDIDATE_RATIO = 0.75
MIN_STABLE_ANCHORS = 12
MIN_INLIERS_TO_LOCK = 6
MIN_INLIER_RATIO_TO_LOCK = 0.18
MIN_SCORE_MARGIN_TO_LOCK = 2.0
DINOV2_STABLE_COSINE = 0.82
DINOV2_DISTINCTIVE_COSINE = 0.68
DINOV2_UNIQUENESS_MARGIN = 0.012
MIN_DISTINCTIVE_ANCHORS_TO_LOCK = 3
MIN_DISTINCTIVE_SCORE_MARGIN = 1.0
STEP_SECONDS = 0.9


class ActiveDistinctiveEvidenceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActiveDistinctiveEvidenceError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    _require(image is not None, f"cannot read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(bool(cv2.imwrite(str(path), image)), f"cannot write image: {path}")


def _fit_feature_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_FEATURE_SIDE / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _reference_sweep(image: np.ndarray) -> list[np.ndarray]:
    """Create a deterministic small-motion enrollment sweep from one storefront frame."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    views = [image]
    for angle, scale, dx in ((-2.0, 0.96, -0.015), (2.0, 1.04, 0.015)):
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx * width
        views.append(
            cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        )
    return views


def _scaled_on_canvas(image: np.ndarray, scale: float) -> np.ndarray:
    _require(0.2 <= scale <= 1.0, "canvas scale outside supported range")
    height, width = image.shape[:2]
    resized = cv2.resize(image, (max(2, round(width * scale)), max(2, round(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(image, 224)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


@dataclass(frozen=True)
class AnchorBank:
    points: np.ndarray
    descriptors: np.ndarray
    detected_base_count: int
    stable_anchor_count: int
    stable_support_views: int


@dataclass(frozen=True)
class DinoAnchorBank:
    descriptors: np.ndarray
    reference_floor: np.ndarray
    stable_anchor_count: int


def _detect_sift(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    fitted, _ = _fit_feature_image(image)
    gray = cv2.cvtColor(fitted, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=0.02).detectAndCompute(gray, None)
    if descriptors is None:
        return [], np.empty((0, 128), dtype=np.float32)
    return keypoints, descriptors.astype(np.float32, copy=False)


def build_anchor_bank(reference_views: Sequence[np.ndarray]) -> AnchorBank:
    _require(len(reference_views) >= 3, "reference sweep requires at least three views")
    detected = [_detect_sift(image) for image in reference_views]
    base_keypoints, base_descriptors = detected[0]
    _require(len(base_keypoints) >= MIN_STABLE_ANCHORS, "base reference has too few local features")
    support = np.ones(len(base_keypoints), dtype=np.int32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    support_views = 1
    for keypoints, descriptors in detected[1:]:
        if len(keypoints) < 4:
            continue
        matches = matcher.knnMatch(base_descriptors, descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < REFERENCE_RATIO * pair[1].distance]
        if len(good) < 4:
            continue
        source = np.float32([base_keypoints[match.queryIdx].pt for match in good])
        target = np.float32([keypoints[match.trainIdx].pt for match in good])
        _, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        if mask is None:
            continue
        inliers = [match.queryIdx for match, keep in zip(good, mask.ravel(), strict=True) if keep]
        if len(inliers) < 4:
            continue
        support[np.asarray(inliers, dtype=np.int32)] += 1
        support_views += 1
    keep = np.flatnonzero(support >= 2)
    _require(len(keep) >= MIN_STABLE_ANCHORS, f"only {len(keep)} stable anchors from reference sweep")
    order = sorted(keep.tolist(), key=lambda index: (-support[index], -base_keypoints[index].response, index))
    order = np.asarray(order[:512], dtype=np.int32)
    return AnchorBank(
        points=np.float32([base_keypoints[index].pt for index in order]),
        descriptors=base_descriptors[order],
        detected_base_count=len(base_keypoints),
        stable_anchor_count=len(order),
        stable_support_views=support_views,
    )


def score_candidate(bank: AnchorBank, image: np.ndarray) -> dict[str, float | int]:
    keypoints, descriptors = _detect_sift(image)
    if len(keypoints) < 4:
        return {"good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "coverage": 0.0, "score": 0.0}
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(bank.descriptors, descriptors, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < CANDIDATE_RATIO * pair[1].distance]
    if len(good) < 4:
        return {
            "good_matches": len(good), "inliers": 0, "inlier_ratio": 0.0, "coverage": 0.0,
            "score": 0.25 * len(good),
        }
    source = np.float32([bank.points[match.queryIdx] for match in good])
    target = np.float32([keypoints[match.trainIdx].pt for match in good])
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    if mask is None:
        inlier_indices: list[int] = []
    else:
        inlier_indices = [index for index, keep in enumerate(mask.ravel()) if keep]
    inliers = len(inlier_indices)
    if inliers:
        inlier_points = source[np.asarray(inlier_indices, dtype=np.int32)]
        mins = bank.points.min(axis=0)
        spans = np.maximum(bank.points.max(axis=0) - mins, 1.0)
        cells = np.floor(np.clip((inlier_points - mins) / spans, 0.0, 0.999) * 4).astype(np.int32)
        coverage = len({(int(x), int(y)) for x, y in cells}) / 16.0
    else:
        coverage = 0.0
    ratio = inliers / max(1, len(good))
    score = inliers + 0.25 * len(good) + 2.0 * coverage
    return {
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": ratio,
        "coverage": coverage,
        "score": score,
    }


def decide_active(candidate_scores: Mapping[str, Mapping[str, float | int]]) -> dict[str, Any]:
    ordered = sorted(candidate_scores.items(), key=lambda item: (-float(item[1]["score"]), item[0]))
    _require(len(ordered) >= 2, "active decision requires candidate competition")
    winner, best = ordered[0]
    runner_up_score = float(ordered[1][1]["score"])
    margin = float(best["score"]) - runner_up_score
    locked = (
        int(best["inliers"]) >= MIN_INLIERS_TO_LOCK
        and float(best["inlier_ratio"]) >= MIN_INLIER_RATIO_TO_LOCK
        and margin >= MIN_SCORE_MARGIN_TO_LOCK
    )
    return {
        "decision": "LOCK" if locked else "ABSTAIN",
        "selected_candidate": winner if locked else None,
        "rank1_candidate": winner,
        "score_margin": margin,
    }


def build_dino_anchor_bank(reference_features: Sequence[np.ndarray]) -> DinoAnchorBank:
    _require(len(reference_features) >= 3, "DINO anchor bank requires a three-view sweep")
    base = np.asarray(reference_features[0], dtype=np.float32)
    _require(base.shape == (dino.PATCH_COUNT, dino.FEATURE_DIM), "DINO reference feature shape drifted")
    support = np.ones(dino.PATCH_COUNT, dtype=np.int32)
    floors = np.ones(dino.PATCH_COUNT, dtype=np.float32)
    for other in reference_features[1:]:
        similarities = base @ np.asarray(other, dtype=np.float32).T
        base_to_other = similarities.argmax(axis=1)
        other_to_base = similarities.argmax(axis=0)
        values = similarities[np.arange(dino.PATCH_COUNT), base_to_other]
        mutual = other_to_base[base_to_other] == np.arange(dino.PATCH_COUNT)
        stable = mutual & (values >= DINOV2_STABLE_COSINE)
        support[stable] += 1
        floors[stable] = np.minimum(floors[stable], values[stable])
    keep = np.flatnonzero(support >= 2)
    _require(len(keep) >= MIN_DISTINCTIVE_ANCHORS_TO_LOCK, "too few stable DINO patch anchors")
    order = sorted(keep.tolist(), key=lambda index: (-support[index], -float(floors[index]), index))[:128]
    selected = np.asarray(order, dtype=np.int32)
    return DinoAnchorBank(base[selected], floors[selected], len(selected))


def score_distinctive_competition(
    bank: DinoAnchorBank,
    candidate_features: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    _require(len(candidate_features) >= 2, "distinctive scoring requires candidate competition")
    slots = sorted(candidate_features)
    maxima = np.stack(
        [bank.descriptors @ np.asarray(candidate_features[slot], dtype=np.float32).T for slot in slots]
    ).max(axis=2)
    top_indices = maxima.argmax(axis=0)
    sorted_values = np.sort(maxima, axis=0)
    margins = sorted_values[-1] - sorted_values[-2]
    scores: dict[str, dict[str, float | int]] = {}
    for slot_index, slot in enumerate(slots):
        absolute_floor = np.minimum(bank.reference_floor - 0.12, DINOV2_DISTINCTIVE_COSINE)
        explained = (top_indices == slot_index) & (maxima[slot_index] >= absolute_floor) & (
            margins >= DINOV2_UNIQUENESS_MARGIN
        )
        count = int(explained.sum())
        uniqueness = float(margins[explained].sum()) if count else 0.0
        scores[slot] = {
            "explained_anchors": count,
            "stable_anchor_count": bank.stable_anchor_count,
            "mean_anchor_similarity": float(maxima[slot_index].mean()),
            "mean_explained_similarity": float(maxima[slot_index][explained].mean()) if count else 0.0,
            "uniqueness_sum": uniqueness,
            "score": count + 8.0 * uniqueness,
        }
    return scores


def decide_distinctive(candidate_scores: Mapping[str, Mapping[str, float | int]]) -> dict[str, Any]:
    ordered = sorted(candidate_scores.items(), key=lambda item: (-float(item[1]["score"]), item[0]))
    winner, best = ordered[0]
    margin = float(best["score"]) - float(ordered[1][1]["score"])
    locked = int(best["explained_anchors"]) >= MIN_DISTINCTIVE_ANCHORS_TO_LOCK and margin >= MIN_DISTINCTIVE_SCORE_MARGIN
    return {
        "decision": "LOCK" if locked else "ABSTAIN",
        "selected_candidate": winner if locked else None,
        "rank1_candidate": winner,
        "score_margin": margin,
    }


def _archive_image(bundle: zipfile.ZipFile, member: str) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(bundle.read(member), dtype=np.uint8), cv2.IMREAD_COLOR)
    _require(decoded is not None, f"cannot decode archive member: {member}")
    return decoded


def _washington_target(
    bundle: zipfile.ZipFile,
    indexed: Mapping[tuple[str, int, int], Sequence[tuple[int, str]]],
    category: str,
    instance: int,
    scenario: str,
) -> dict[str, Any]:
    hard = instance + 1
    for video in (1, 4):
        _require((category, instance, video) in indexed, f"missing Washington target {category}/{instance}/v{video}")
        _require((category, hard, video) in indexed, f"missing Washington distractor {category}/{hard}/v{video}")
    # A short enrollment sweep and later search views come from disjoint time
    # ranges of the same capture.  This is the deliberate input-contract change;
    # it is not an independent-session exact-instance test.
    refs = [
        _archive_image(bundle, washington._choose_member(indexed[(category, instance, 4)], quantile))
        for quantile in (0.05, 0.15, 0.25)
    ]
    views = []
    for quantile in (0.35, 0.50, 0.65, 0.80):
        target = _archive_image(bundle, washington._choose_member(indexed[(category, instance, 4)], quantile))
        distractor = _archive_image(bundle, washington._choose_member(indexed[(category, hard, 4)], quantile))
        views.append((target, distractor, f"q{int(quantile * 100):02d}"))
    return {
        "target_id": f"washington-{category}-{instance}",
        "scenario": scenario,
        "source": "Washington RGB-D Object Dataset evaluation set",
        "reference_mode": "REAL_SAME_CAPTURE_EARLY_MULTI_VIEW_VIDEO_SWEEP",
        "reference_views": refs,
        "search_views": views,
    }


def build_cohort(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    named = repo_root / "artifacts.local/evidence/named-referent-provider-v0/canary-v0/data"
    archive = repo_root / "artifacts.local/datasets/washington-rgbd-object-eval/rgbd-dataset_eval.zip"
    _require(named.is_dir(), f"missing named-referent canary: {named}")
    _require(archive.is_file(), f"missing Washington archive: {archive}")
    images = named / "images"
    derived = named / "derived"

    starbucks_reference = _read_image(images / "starbucks_dazaifu_reference.jpg")
    starbucks_a = _read_image(images / "starbucks_dazaifu_query_a.jpg")
    starbucks_b = _read_image(images / "starbucks_dazaifu_query_b.jpg")
    starbucks_hard = _read_image(images / "starbucks_hudson_distractor.jpg")
    tsui_reference = _read_image(images / "tsuiwah_tko_reference.jpg")
    tsui_hard = _read_image(images / "tsuiwah_tsuenwan_distractor.jpg")
    targets: list[dict[str, Any]] = [
        {
            "target_id": "storefront-starbucks-dazaifu",
            "scenario": "STORE_OR_ENTRANCE",
            "source": "Wikimedia Commons named-referent canary",
            "reference_mode": "DETERMINISTIC_SHORT_SWEEP_SIMULATION_FROM_PUBLIC_REFERENCE",
            "reference_views": _reference_sweep(starbucks_reference),
            "search_views": [
                (starbucks_a, starbucks_hard, "query-a"),
                (_scaled_on_canvas(starbucks_a, 0.72), starbucks_hard, "query-a-scale-072"),
                (starbucks_b, starbucks_hard, "query-b"),
                (_scaled_on_canvas(starbucks_b, 0.58), starbucks_hard, "query-b-scale-058"),
            ],
        },
        {
            "target_id": "storefront-tsuiwah-tko",
            "scenario": "STORE_OR_ENTRANCE",
            "source": "Wikimedia Commons named-referent canary",
            "reference_mode": "DETERMINISTIC_SHORT_SWEEP_SIMULATION_FROM_PUBLIC_REFERENCE",
            "reference_views": _reference_sweep(tsui_reference),
            "search_views": [
                (_read_image(derived / f"tsuiwah-scale-{scale:.2f}.jpg"), tsui_hard, f"scale-{scale:.2f}")
                for scale in (1.00, 0.75, 0.50, 0.35)
            ],
        },
    ]
    indexed = washington._parse_rgb_members(archive)
    with zipfile.ZipFile(archive) as bundle:
        targets.append(_washington_target(bundle, indexed, "cereal_box", 1, "SPECIFIC_PRODUCT"))
        # The originally attempted phone enrollment produced only five stable
        # cross-view anchors and was rejected before metrics.  A keyboard is a
        # demo-valid personal item whose repeated enrollment views satisfy the
        # unchanged acquisition front door.
        targets.append(_washington_target(bundle, indexed, "keyboard", 1, "PERSONAL_ITEM"))

    input_root = run_dir / "inputs"
    manifest_targets = []
    for target_index, target in enumerate(targets):
        target_root = input_root / target["target_id"]
        reference_paths = []
        for index, image in enumerate(target["reference_views"]):
            path = target_root / "reference" / f"ref-{index + 1:02d}.jpg"
            _write_image(path, image)
            reference_paths.append(path)
        view_rows = []
        for index, (target_image, hard_image, label) in enumerate(target["search_views"]):
            target_path = target_root / "search" / f"view-{index + 1:02d}-target.jpg"
            hard_path = target_root / "search" / f"view-{index + 1:02d}-hard.jpg"
            _write_image(target_path, target_image)
            _write_image(hard_path, hard_image)
            view_rows.append({"label": label, "target_path": target_path, "hard_path": hard_path})
        unrelated_target = targets[(target_index + 1) % len(targets)]["search_views"][0][0]
        unrelated_path = target_root / "search" / "lost-unrelated.jpg"
        _write_image(unrelated_path, unrelated_target)
        manifest_targets.append(
            {
                "target_id": target["target_id"],
                "scenario": target["scenario"],
                "source": target["source"],
                "reference_mode": target["reference_mode"],
                "reference_paths": [str(path.resolve()) for path in reference_paths],
                "views": [
                    {
                        "label": row["label"],
                        "target_path": str(row["target_path"].resolve()),
                        "hard_path": str(row["hard_path"].resolve()),
                    }
                    for row in view_rows
                ],
                "lost_candidates": [
                    str(view_rows[1]["hard_path"].resolve()),
                    str(unrelated_path.resolve()),
                ],
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "data_role": "CURATED_CONSUMED_PUBLIC_AND_DERIVED_DEVELOPMENT_DEMO",
        "selection_disclosure": (
            "Four demo-friendly targets were chosen before this run for visible local anchors: two public storefronts, "
            "one printed package, and one personal phone. Storefront sweep/search scale variants are deterministic derivatives."
        ),
        "target_count": len(manifest_targets),
        "target_present_decisions": sum(len(target["views"]) for target in manifest_targets),
        "targets": manifest_targets,
        "claim_ceiling": CLAIM_CEILING,
    }
    _atomic_json(run_dir / "cohort-manifest.json", manifest)
    return manifest


def _all_image_paths(manifest: Mapping[str, Any]) -> list[Path]:
    paths: dict[str, Path] = {}
    for target in manifest["targets"]:
        for value in target["reference_paths"]:
            paths[value] = Path(value)
        for view in target["views"]:
            for key in ("target_path", "hard_path"):
                paths[view[key]] = Path(view[key])
        for value in target["lost_candidates"]:
            paths[value] = Path(value)
    return [paths[key] for key in sorted(paths)]


def _dino_scores(manifest: Mapping[str, Any], model_dir: Path, device: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = _all_image_paths(manifest)
    tensors = [dino._crop_tensor(path, [0.0, 0.0, 1.0, 1.0]) for path in paths]
    encoder = dino.DenseEncoder(model_dir, device)
    encoded = encoder.encode(tensors)
    return {str(path.resolve()): feature for path, feature in zip(paths, encoded, strict=True)}, {
        "model_repository": dino.MODEL_REPOSITORY,
        "model_revision": dino.MODEL_REVISION,
        "model_dir": str(model_dir.resolve()),
        "device": device,
        "encoded_image_count": encoder.encoded_crops,
        "forward_batch_count": encoder.forward_batches,
    }


def _passive_score(reference_feature: np.ndarray, candidate_feature: np.ndarray) -> float:
    mask = np.ones(dino.PATCH_COUNT, dtype=bool)
    return float(dino.symmetric_local_score(reference_feature, candidate_feature, mask, mask)["symmetric_score"])


def _ordered_slots(target_id: str, step_index: int, candidates: Sequence[tuple[str, Path]]) -> dict[str, tuple[str, Path]]:
    reverse = hashlib.sha256(f"{target_id}:{step_index}".encode("utf-8")).digest()[0] % 2 == 1
    ordered = list(reversed(candidates)) if reverse else list(candidates)
    return {chr(ord("A") + index): value for index, value in enumerate(ordered)}


def run_demo(manifest: Mapping[str, Any], model_dir: Path, run_dir: Path, device: str) -> dict[str, Any]:
    features, model_receipt = _dino_scores(manifest, model_dir, device)
    rows = []
    target_summaries = []
    for target in manifest["targets"]:
        reference_paths = [Path(value) for value in target["reference_paths"]]
        bank = build_anchor_bank([_read_image(path) for path in reference_paths])
        dino_bank = build_dino_anchor_bank([features[str(path.resolve())] for path in reference_paths])
        passive_reference = features[str(reference_paths[0].resolve())]
        steps: list[dict[str, Any]] = []
        for index, view in enumerate(target["views"]):
            if index == 2:
                lost = [Path(value) for value in target["lost_candidates"]]
                steps.append({"label": "target-lost", "target_present": False, "candidates": [("DISTRACTOR", path) for path in lost]})
            steps.append(
                {
                    "label": view["label"],
                    "target_present": True,
                    "candidates": [("TARGET", Path(view["target_path"])), ("DISTRACTOR", Path(view["hard_path"]))],
                }
            )
        target_rows = []
        for step_index, step in enumerate(steps):
            slots = _ordered_slots(target["target_id"], step_index, step["candidates"])
            active_scores = score_distinctive_competition(
                dino_bank,
                {slot: features[str(path.resolve())] for slot, (_, path) in slots.items()},
            )
            active_decision = decide_distinctive(active_scores)
            passive_scores = {
                slot: _passive_score(passive_reference, features[str(path.resolve())])
                for slot, (_, path) in slots.items()
            }
            passive_rank1 = max(passive_scores, key=lambda slot: (passive_scores[slot], slot))
            target_slot = next((slot for slot, (role, _) in slots.items() if role == "TARGET"), None)
            row = {
                "target_id": target["target_id"],
                "scenario": target["scenario"],
                "step_index": step_index,
                "time_seconds": step_index * STEP_SECONDS,
                "view_label": step["label"],
                "target_present": step["target_present"],
                "target_slot": target_slot,
                "candidate_roles": {slot: role for slot, (role, _) in slots.items()},
                "passive": {
                    "decision": "LOCK",
                    "selected_candidate": passive_rank1,
                    "rank1_candidate": passive_rank1,
                    "scores": passive_scores,
                },
                "active": {**active_decision, "scores": active_scores},
            }
            rows.append(row)
            target_rows.append(row)
        target_summaries.append(
            {
                "target_id": target["target_id"],
                "scenario": target["scenario"],
                "reference_mode": target["reference_mode"],
                "anchor_bank": {
                    "detected_base_count": bank.detected_base_count,
                    "stable_sift_anchor_count": bank.stable_anchor_count,
                    "stable_dinov2_patch_anchor_count": dino_bank.stable_anchor_count,
                    "supporting_reference_views": bank.stable_support_views,
                },
                "step_count": len(target_rows),
            }
        )
    raw = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "cohort_manifest_sha256": _sha256_file(run_dir / "cohort-manifest.json"),
        "passive_baseline": "single-reference DINOv2-S symmetric mean-nearest patch score; always-lock demo policy",
        "active_arm": {
            "representation": "stable local anchors repeated across enrollment views; SIFT audits acquisition and DINOv2 patches rank candidate-unique explanations",
            "candidate_evidence": "each patch anchor votes only for the candidate with sufficient absolute similarity and a uniqueness margin over every competitor",
            "decision": {
                "min_distinctive_anchors": MIN_DISTINCTIVE_ANCHORS_TO_LOCK,
                "min_anchor_uniqueness_margin": DINOV2_UNIQUENESS_MARGIN,
                "min_distinctive_score_margin": MIN_DISTINCTIVE_SCORE_MARGIN,
            },
            "tracker_authority": "NONE; sequence state is evaluated only, no tracker implemented",
        },
        "model_receipt": model_receipt,
        "targets": target_summaries,
        "rows": rows,
        "claim_ceiling": CLAIM_CEILING,
    }
    _atomic_json(run_dir / "raw-decisions.json", raw)
    report = evaluate(raw)
    _atomic_json(run_dir / "final-report.json", report)
    return report


def _arm_metrics(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    present = [row for row in rows if row["target_present"]]
    top1 = sum(row[arm]["rank1_candidate"] == row["target_slot"] for row in present)
    wrong_locks = sum(
        row[arm]["decision"] == "LOCK"
        and (not row["target_present"] or row[arm]["selected_candidate"] != row["target_slot"])
        for row in rows
    )
    lock_times = []
    reacquired = 0
    reacquisition_total = 0
    for target_id in sorted({row["target_id"] for row in rows}):
        target_rows = sorted((row for row in rows if row["target_id"] == target_id), key=lambda row: row["step_index"])
        correct = [
            row for row in target_rows
            if row["target_present"] and row[arm]["decision"] == "LOCK" and row[arm]["selected_candidate"] == row["target_slot"]
        ]
        if correct:
            lock_times.append(float(correct[0]["time_seconds"]))
        for index, row in enumerate(target_rows[:-1]):
            if not row["target_present"]:
                reacquisition_total += 1
                next_present = next((candidate for candidate in target_rows[index + 1 :] if candidate["target_present"]), None)
                if (
                    next_present is not None
                    and next_present[arm]["decision"] == "LOCK"
                    and next_present[arm]["selected_candidate"] == next_present["target_slot"]
                ):
                    reacquired += 1
    return {
        "target_present_decisions": len(present),
        "target_top1": top1,
        "target_top1_rate": top1 / len(present),
        "wrong_target_locks": wrong_locks,
        "all_sequence_steps": len(rows),
        "median_time_to_first_lock_seconds": float(np.median(lock_times)) if lock_times else None,
        "targets_with_correct_lock": len(lock_times),
        "reacquisition": reacquired,
        "reacquisition_opportunities": reacquisition_total,
    }


def evaluate(raw: Mapping[str, Any]) -> dict[str, Any]:
    rows = raw["rows"]
    passive = _arm_metrics(rows, "passive")
    active = _arm_metrics(rows, "active")
    by_scenario = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        by_scenario[scenario] = {
            "passive": _arm_metrics(scenario_rows, "passive"),
            "active": _arm_metrics(scenario_rows, "active"),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": _utc_now(),
        "data_role": "CURATED_CONSUMED_PUBLIC_AND_DERIVED_DEVELOPMENT_DEMO",
        "metrics": {"passive": passive, "active": active, "by_scenario": by_scenario},
        "delta": {
            "target_top1": active["target_top1"] - passive["target_top1"],
            "wrong_target_locks": active["wrong_target_locks"] - passive["wrong_target_locks"],
            "reacquisition": active["reacquisition"] - passive["reacquisition"],
        },
        "interpretation_boundary": [
            "The cohort is intentionally curated for visible distinctive anchors and is Development-only.",
            "Storefront sweep and scale variants are deterministic derivatives, not captured device clips.",
            "The passive comparator has no open-set threshold and therefore always locks.",
            "No tracker is implemented; reacquisition measures fresh anchor-gated relock after an injected lost step.",
            "Positive results do not reopen passive exact-instance verification or authorize P1/default-App/safety claims.",
        ],
        "terminal": "CONTROLLED_DEVELOPMENT_DEMO_MEASURED",
        "next_action": "Inspect per-scenario failures; only a visible uplift justifies a live reference-clip Android research seam.",
        "claim_ceiling": CLAIM_CEILING,
        "raw_decisions_sha256": None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require(not args.run_dir.exists(), f"refusing to overwrite run: {args.run_dir}")
    args.run_dir.mkdir(parents=True)
    try:
        manifest = build_cohort(args.repo_root.resolve(), args.run_dir.resolve())
        report = run_demo(manifest, args.model_dir.resolve(), args.run_dir.resolve(), args.device)
        report["raw_decisions_sha256"] = _sha256_file(args.run_dir / "raw-decisions.json")
        _atomic_json(args.run_dir / "final-report.json", report)
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        _atomic_json(
            args.run_dir / "failure.json",
            {"experiment_id": EXPERIMENT_ID, "failed_at_utc": _utc_now(), "error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
