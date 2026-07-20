#!/usr/bin/env python3
"""Diagnose a camera-motion-invariant static corridor feature.

The frozen r7.17 result remains untouched. This retrospective probe estimates
global motion with the same ORB/RANSAC family, then subtracts residual change
observed in same-depth peripheral background strips from the central near-field
corridor. Window scores use preregistered robust level aggregations rather than
peak-to-peak residual range. No trainable parameters or labels are fitted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_background_normalized_static_probe_v1"
FEATURE_KEYS = (
    "mean_excess",
    "q75_excess",
    "q90_excess",
    "fraction_ge_018_excess",
    "quality_weighted_q90_excess",
)
AGGREGATIONS = ("median", "q75")


def comparison_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return central near-field and same-depth peripheral control masks."""
    if size < 32:
        raise ValueError("motion size is too small")
    yy, xx = np.mgrid[:size, :size]
    y = yy / max(size - 1, 1)
    x = xx / max(size - 1, 1)
    depth = np.clip((y - 0.58) / 0.42, 0.0, 1.0)
    center_half_width = 0.14 + 0.18 * depth
    center = (y >= 0.58) & (np.abs(x - 0.5) <= center_half_width)
    background = (
        (y >= 0.58)
        & (y <= 0.94)
        & (np.abs(x - 0.5) >= 0.36)
        & (np.abs(x - 0.5) <= 0.49)
    )
    if not center.any() or not background.any() or np.any(center & background):
        raise ValueError("static comparison masks are invalid")
    return center, background


def _region(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    if not len(selected):
        raise ValueError("static comparison region is empty")
    return {
        "mean": float(selected.mean()),
        "q75": float(np.quantile(selected, 0.75)),
        "q90": float(np.quantile(selected, 0.90)),
        "fraction_ge_018": float(np.mean(selected >= 0.18)),
    }


def frame_pair_descriptor(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    size: int = 320,
) -> dict[str, Any]:
    if previous is None or current is None:
        raise ValueError("background-normalized descriptor needs two images")
    previous = cv2.resize(previous, (size, size), interpolation=cv2.INTER_AREA)
    current = cv2.resize(current, (size, size), interpolation=cv2.INTER_AREA)
    gray_previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=8)
    key_previous, desc_previous = orb.detectAndCompute(gray_previous, None)
    key_current, desc_current = orb.detectAndCompute(gray_current, None)

    good: list[Any] = []
    if desc_previous is not None and desc_current is not None:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            desc_previous, desc_current, k=2
        )
        good = [first for first, second in matches if first.distance < 0.72 * second.distance]

    homography = None
    inlier_ratio = 0.0
    if len(good) >= 8:
        source = np.float32([key_previous[row.queryIdx].pt for row in good])
        target = np.float32([key_current[row.trainIdx].pt for row in good])
        homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if homography is not None and inliers is not None:
            inlier_ratio = float(np.mean(inliers))

    success = homography is not None
    transform = homography if success else np.eye(3, dtype=np.float64)
    warped = cv2.warpPerspective(gray_previous, transform, (size, size))
    valid = cv2.warpPerspective(
        np.ones_like(gray_previous), transform, (size, size)
    ) > 0
    residual = cv2.absdiff(warped, gray_current).astype(np.float64) / 255.0
    center_mask, background_mask = comparison_masks(size)
    valid_center = center_mask & valid
    valid_background = background_mask & valid
    center_valid_fraction = float(valid[center_mask].mean())
    background_valid_fraction = float(valid[background_mask].mean())
    reliable = bool(
        success
        and center_valid_fraction >= 0.80
        and background_valid_fraction >= 0.80
    )
    if not valid_center.any() or not valid_background.any():
        reliable = False
        valid_center = center_mask
        valid_background = background_mask

    center = _region(residual, valid_center)
    background = _region(residual, valid_background)
    excess = {
        "mean_excess": max(0.0, center["mean"] - background["mean"]),
        "q75_excess": max(0.0, center["q75"] - background["q75"]),
        "q90_excess": max(0.0, center["q90"] - background["q90"]),
        "fraction_ge_018_excess": max(
            0.0,
            center["fraction_ge_018"] - background["fraction_ge_018"],
        ),
    }
    excess["quality_weighted_q90_excess"] = excess["q90_excess"] * inlier_ratio
    return {
        "homography_success": success,
        "reliable": reliable,
        "good_matches": len(good),
        "inlier_ratio": inlier_ratio,
        "center_valid_fraction": center_valid_fraction,
        "background_valid_fraction": background_valid_fraction,
        "center": center,
        "background": background,
        **excess,
    }


def score_descriptors(
    descriptors: Sequence[dict[str, Any]],
) -> dict[str, float | int | None]:
    reliable = [row for row in descriptors if row["reliable"]]
    scores: dict[str, float | int | None] = {
        "transition_count": len(descriptors),
        "reliable_transition_count": len(reliable),
    }
    for key in FEATURE_KEYS:
        values = np.asarray([float(row[key]) for row in reliable], dtype=np.float64)
        for aggregation in AGGREGATIONS:
            name = f"{aggregation}_{key}"
            if not len(values):
                scores[name] = None
            elif aggregation == "median":
                scores[name] = float(np.median(values))
            else:
                scores[name] = float(np.quantile(values, 0.75))
    return scores


def image_descriptors(images: Sequence[np.ndarray], *, size: int) -> list[dict[str, Any]]:
    if len(images) < 2 or any(image is None for image in images):
        raise ValueError("static window needs at least two decoded images")
    return [
        frame_pair_descriptor(previous, current, size=size)
        for previous, current in zip(images, images[1:])
    ]


def decode_video_window(
    video: Path,
    start_ms: int,
    end_ms: int,
    *,
    interval_ms: int,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    frames: list[np.ndarray] = []
    try:
        for timestamp_ms in range(start_ms, end_ms + 1, interval_ms):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode video at {timestamp_ms} ms")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _episode_images(episode: dict[str, Any]) -> list[np.ndarray]:
    images = [cv2.imread(row["path"], cv2.IMREAD_COLOR) for row in episode["frames"]]
    if len(images) < 2 or any(image is None for image in images):
        raise ValueError(f"episode images are missing: {episode['episode_id']}")
    return images


def _comparisons(
    rows: Sequence[dict[str, Any]],
    rice: dict[str, dict[str, float | int | None]],
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for feature in FEATURE_KEYS:
        for aggregation in AGGREGATIONS:
            key = f"{aggregation}_{feature}"
            pair_checks = [
                row["no_alert_scores"][key] is not None
                and row["alert_scores"][key] is not None
                and float(row["alert_scores"][key]) > float(row["no_alert_scores"][key])
                for row in rows
            ]
            rice_values = [rice[name][key] for name in ("pre_clear", "risk", "post_clear")]
            rice_available = all(value is not None for value in rice_values)
            rice_open = bool(
                rice_available and float(rice_values[1]) > float(rice_values[0])
            )
            rice_close = bool(
                rice_available and float(rice_values[1]) > float(rice_values[2])
            )
            candidates[key] = {
                "legacy_static_pair_ordering": pair_checks,
                "legacy_static_pair_ordering_passed": all(pair_checks),
                "rice_open_ordering_passed": rice_open,
                "rice_close_ordering_passed": rice_close,
                "all_required_orderings_passed": all(pair_checks) and rice_open and rice_close,
            }
    return candidates


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.package_root,
        args.mechanism_report,
        args.rice_video,
        args.rice_review,
        args.output,
    ):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    episodes, _excluded = common.load_episode_specs(args.package_root.resolve())
    qualified = temporal.load_qualified_pair_contract(args.mechanism_report.resolve())
    static_pair_ids = set(qualified[temporal.STATIC])
    pair_members: dict[str, list[dict[str, Any]]] = {pair_id: [] for pair_id in static_pair_ids}
    for episode in episodes:
        pair_id = episode.get("counterfactual_pair_id")
        if pair_id in pair_members:
            pair_members[pair_id].append(episode)

    static_rows: list[dict[str, Any]] = []
    for pair_id in sorted(static_pair_ids):
        members = pair_members[pair_id]
        no_alert = [row for row in members if int(row["label"]) == 0]
        alert = [row for row in members if int(row["label"]) == 1]
        if len(no_alert) != 1 or len(alert) != 1:
            raise ValueError(f"static pair is incomplete: {pair_id}")
        no_scores = score_descriptors(
            image_descriptors(_episode_images(no_alert[0]), size=args.motion_size)
        )
        alert_scores = score_descriptors(
            image_descriptors(_episode_images(alert[0]), size=args.motion_size)
        )
        static_rows.append({
            "counterfactual_pair_id": pair_id,
            "source_id": no_alert[0]["source_id"],
            "no_alert_episode_id": no_alert[0]["episode_id"],
            "alert_episode_id": alert[0]["episode_id"],
            "no_alert_scores": no_scores,
            "alert_scores": alert_scores,
        })

    review = common.load_json(args.rice_review)
    body = review.get("review") or {}
    windows = {
        "pre_clear": body.get("pre_risk_clear_window_ms"),
        "risk": body.get("risk_present_window_ms"),
        "post_clear": body.get("stable_post_clear_window_ms"),
    }
    rice_scores: dict[str, dict[str, float | int | None]] = {}
    for name, window in windows.items():
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"Rice review window is invalid: {name}")
        images = decode_video_window(
            args.rice_video.resolve(), int(window[0]), int(window[1]), interval_ms=1000
        )
        rice_scores[name] = score_descriptors(
            image_descriptors(images, size=args.motion_size)
        )

    candidates = _comparisons(static_rows, rice_scores)
    passing = sorted(
        key for key, value in candidates.items()
        if value["all_required_orderings_passed"]
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "retrospective_feature_diagnosis_after_r717_failure",
        "inputs": {
            "package_root": str(args.package_root.resolve()),
            "mechanism_report": {
                "path": str(args.mechanism_report.resolve()),
                "sha256": common.sha256_file(args.mechanism_report),
            },
            "rice_video": {
                "path": str(args.rice_video.resolve()),
                "sha256": common.sha256_file(args.rice_video),
            },
            "rice_review": {
                "path": str(args.rice_review.resolve()),
                "sha256": common.sha256_file(args.rice_review),
            },
        },
        "feature_contract": {
            "trainable_parameters": 0,
            "motion_size": args.motion_size,
            "pair_feature": "positive central near-field residual excess over same-depth peripheral background after ORB/RANSAC homography",
            "reliability": "homography success and >=.80 valid pixels in both comparison regions",
            "window_aggregations": list(AGGREGATIONS),
            "feature_keys": list(FEATURE_KEYS),
            "threshold_fitted": False,
        },
        "legacy_static_pairs": static_rows,
        "rice_street_windows": rice_scores,
        "candidate_comparisons": candidates,
        "passing_candidate_keys": passing,
        "diagnostic_gate": {
            "minimum_required_orderings": 5,
            "passed": bool(passing),
        },
        "evidence_limit": "Retrospective feature diagnosis after seeing the r7.17 failure. Passing may define a future frozen contract but cannot rescue r7.17 or authorize training, calibration, blind evaluation, Android changes, or production.",
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(str(args.output) + ".sha256").write_text(
        common.sha256_file(args.output) + "\n", encoding="ascii"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--rice-video", type=Path, required=True)
    parser.add_argument("--rice-review", type=Path, required=True)
    parser.add_argument("--motion-size", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "passing_candidate_keys": report["passing_candidate_keys"],
        "diagnostic_gate_passed": report["diagnostic_gate"]["passed"],
        "output_sha256": common.sha256_file(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
