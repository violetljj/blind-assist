#!/usr/bin/env python3
"""Add an independent single-view Teacher to the AG-ST factor-label factory.

The source sensor remains authoritative wherever it is valid.  MapAnything
remains the primary multi-view completion Teacher.  Depth Anything V2 is used
only as independent geometric evidence: its source-anchored disagreement with
MapAnything lowers pseudo-label quality and increases uncertainty.  Pixels
without sufficient agreement remain UNKNOWN.

This is a reversible WILD_LAB label materialization.  No task outcome or
deterministic reducer field is read or written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
HFTF_DIR = MODULE_DIR.parent / "hftf"
sys.path[:0] = [str(MODULE_DIR), str(HFTF_DIR)]

from arkitscenes_truth_reader import parse_trajectory  # noqa: E402
from build_ag_st_factor_labels import (  # noqa: E402
    FORBIDDEN_TASK_TOKENS,
    PROVENANCE_TEACHER,
    TEACHER_B_QUALITY,
    TEACHER_C_QUALITY,
    TIER_A_SOURCE,
    TIER_B_ANCHORED,
    TIER_C_TEACHER,
    TIER_UNKNOWN,
    _preprocess_source_for_output,
    compute_geometric_factors,
    depth_uncertainty_proxy,
)
from produce_external_rgb_metric_depth_observations import (  # noqa: E402
    DepthAnythingV2MetricSource,
)
from run_ag_st_stage0a import (  # noqa: E402
    _error_metrics,
    compute_selective_metrics,
    load_factor_source_frame,
    resolve_trajectory_path,
    select_source_videos,
    sha256_file,
)


DEFAULT_STAGE0A_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-stage0a-mapanything-apache-train16-block64-r1/result.json"
)
DEFAULT_BASE_LABEL_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-superteacher-factor-labels-train16-r5"
)
DEFAULT_DAV2_REPO = (
    REPO_ROOT / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main"
)
DEFAULT_DAV2_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-superteacher-factor-labels-multiteacher-train16-r2"
)

SCHEMA = "blindassist_ag_st_multiteacher_factor_label_factory_wild_lab_result_v1"
PAIR_DISAGREEMENT_SCALE = 0.20
MINIMUM_ANCHOR_SUPPORT = 512


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def robust_observed_scale(
    observed_depth_m: np.ndarray,
    prediction_depth_m: np.ndarray,
    *,
    minimum_support: int = MINIMUM_ANCHOR_SUPPORT,
) -> tuple[float, int]:
    """Recover one metric scale from observed source pixels only."""
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    prediction = np.asarray(prediction_depth_m, dtype=np.float32)
    require(observed.shape == prediction.shape, "anchor/prediction shape mismatch")
    valid = (
        np.isfinite(observed)
        & (observed > 0)
        & np.isfinite(prediction)
        & (prediction > 0)
    )
    support = int(valid.sum())
    require(support >= minimum_support, "second-Teacher anchor support too small")
    scale = float(np.median(observed[valid] / prediction[valid]))
    require(math.isfinite(scale) and 0.02 <= scale <= 50.0, "second-Teacher scale invalid")
    return scale, support


def teacher_pair_quality(
    primary_depth_m: np.ndarray,
    primary_valid: np.ndarray,
    secondary_depth_m: np.ndarray,
    secondary_valid: np.ndarray,
    *,
    disagreement_scale: float = PAIR_DISAGREEMENT_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return symmetric relative disagreement and its continuous quality."""
    primary = np.asarray(primary_depth_m, dtype=np.float32)
    secondary = np.asarray(secondary_depth_m, dtype=np.float32)
    first_valid = np.asarray(primary_valid, dtype=np.bool_)
    second_valid = np.asarray(secondary_valid, dtype=np.bool_)
    require(
        primary.shape == secondary.shape == first_valid.shape == second_valid.shape,
        "Teacher-pair shape mismatch",
    )
    require(disagreement_scale > 0, "disagreement scale must be positive")
    pair_valid = (
        first_valid
        & second_valid
        & np.isfinite(primary)
        & (primary > 0)
        & np.isfinite(secondary)
        & (secondary > 0)
    )
    disagreement = np.full(primary.shape, np.nan, dtype=np.float32)
    denominator = np.maximum(primary + secondary, 1e-6)
    disagreement[pair_valid] = (
        2.0 * np.abs(primary[pair_valid] - secondary[pair_valid])
        / denominator[pair_valid]
    )
    quality = np.zeros(primary.shape, dtype=np.float32)
    quality[pair_valid] = np.exp(
        -disagreement[pair_valid] / float(disagreement_scale)
    )
    return disagreement, quality, pair_valid


def regrade_teacher_labels(
    source_valid: np.ndarray,
    source_tiers: np.ndarray,
    source_provenance: np.ndarray,
    primary_valid: np.ndarray,
    base_quality: np.ndarray,
    anchor_quality: np.ndarray,
    multiview_valid: np.ndarray,
    pair_quality: np.ndarray,
    pair_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve source labels and conservatively regrade Teacher-only pixels."""
    source = np.asarray(source_valid, dtype=np.bool_)
    tiers = np.asarray(source_tiers, dtype=np.uint8).copy()
    provenance = np.asarray(source_provenance, dtype=np.uint8).copy()
    primary = np.asarray(primary_valid, dtype=np.bool_)
    base = np.asarray(base_quality, dtype=np.float32)
    anchor = np.asarray(anchor_quality, dtype=np.float32)
    multiview = np.asarray(multiview_valid, dtype=np.bool_)
    pair = np.asarray(pair_quality, dtype=np.float32)
    pair_mask = np.asarray(pair_valid, dtype=np.bool_)
    require(
        source.shape
        == tiers.shape
        == provenance.shape
        == primary.shape
        == base.shape
        == anchor.shape
        == multiview.shape
        == pair.shape
        == pair_mask.shape,
        "regrade shape mismatch",
    )

    teacher_region = ~source
    tiers[teacher_region] = TIER_UNKNOWN
    provenance[teacher_region] = 0
    scores = np.zeros(source.shape, dtype=np.float32)
    scores[source] = np.where(tiers[source] == TIER_A_SOURCE, 0.98, 0.90)

    candidate = teacher_region & primary & pair_mask
    combined = np.zeros(source.shape, dtype=np.float32)
    combined[candidate] = np.sqrt(
        np.clip(base[candidate], 0.0, 1.0)
        * np.clip(pair[candidate], 0.0, 1.0)
    )
    teacher_b = (
        candidate
        & (combined >= TEACHER_B_QUALITY)
        & (multiview | (anchor >= 0.75))
    )
    teacher_c = candidate & ~teacher_b & (combined >= TEACHER_C_QUALITY)
    tiers[teacher_b] = TIER_B_ANCHORED
    tiers[teacher_c] = TIER_C_TEACHER
    provenance[teacher_b | teacher_c] = PROVENANCE_TEACHER
    scores[teacher_b | teacher_c] = combined[teacher_b | teacher_c]
    return tiers, provenance, scores


def _base_combined_quality(payload: dict[str, np.ndarray]) -> np.ndarray:
    confidence = np.asarray(payload["teacher_confidence_quality_hw"], dtype=np.float32)
    anchor = np.asarray(payload["anchor_quality_hw"], dtype=np.float32)
    multiview = np.asarray(payload["multiview_quality_hw"], dtype=np.float32)
    valid = np.asarray(payload["teacher_candidate_valid_hw"], dtype=np.bool_)
    quality = np.cbrt(
        np.clip(confidence, 0.0, 1.0)
        * np.clip(anchor, 0.0, 1.0)
        * np.clip(multiview, 0.0, 1.0)
    )
    quality[~valid] = 0.0
    return quality.astype(np.float32)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def _split_error(
    records: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    accepted_truth: list[np.ndarray] = []
    accepted_prediction: list[np.ndarray] = []
    rejected_truth: list[np.ndarray] = []
    rejected_prediction: list[np.ndarray] = []
    total = 0
    accepted = 0
    for record in records:
        truth = np.asarray(record["truth_depth_m"], dtype=np.float32)
        prediction = np.asarray(record["prediction_depth_m"], dtype=np.float32)
        score = np.asarray(record["confidence"], dtype=np.float32)
        valid = (
            np.asarray(record["hidden_mask"], dtype=np.bool_)
            & np.asarray(record["model_mask"], dtype=np.bool_)
            & np.isfinite(truth)
            & (truth > 0)
            & np.isfinite(prediction)
            & (prediction > 0)
            & np.isfinite(score)
        )
        take = valid & (score >= threshold)
        reject = valid & ~take
        total += int(valid.sum())
        accepted += int(take.sum())
        accepted_truth.append(truth[take])
        accepted_prediction.append(prediction[take])
        rejected_truth.append(truth[reject])
        rejected_prediction.append(prediction[reject])
    require(total > 0, "split-risk denominator is zero")
    return {
        "threshold": threshold,
        "coverage": accepted / total,
        "accepted": _error_metrics(
            np.concatenate(accepted_truth), np.concatenate(accepted_prediction)
        ),
        "rejected": _error_metrics(
            np.concatenate(rejected_truth), np.concatenate(rejected_prediction)
        ),
    }


def _compact_curve(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "coverage": row["coverage_of_hidden"],
            "mae_m": row["overall"]["mae_m"],
            "bad_0_10m_rate": row["overall"]["bad_0_10m_rate"],
            "parent_macro_mae_m": row["parent_macro_mae_m"],
            "all_parents_evaluable": row["parent_macro_evaluable"],
        }
        for row in metrics["teacher_confidence_risk_coverage"]
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require(args.stage0a_result.is_file(), "Stage 0A result missing")
    require(args.base_label_dir.is_dir(), "base factor-label directory missing")
    require((args.base_label_dir / "result.json").is_file(), "base label result missing")
    require(args.dav2_repo.is_dir(), "Depth Anything V2 repository missing")
    require(args.dav2_checkpoint.is_file(), "Depth Anything V2 checkpoint missing")
    require(not args.output_dir.exists(), f"output directory already exists: {args.output_dir}")

    stage0a = json.loads(args.stage0a_result.read_text(encoding="utf-8"))
    require(stage0a.get("status") == "COMPLETED", "Stage 0A result incomplete")
    source_manifest_path = Path(stage0a["source"]["manifest_path"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    parent_ids = [str(value) for value in stage0a["source"]["parents"]]
    videos = {
        str(row["video_id"]): row
        for row in select_source_videos(
            source_manifest,
            parent_ids,
            role_token=str(stage0a["source"].get("manifest_role_token", "TRAIN")),
        )
    }
    parent_runs = {str(row["parent_id"]): row for row in stage0a["parent_runs"]}
    require(set(parent_ids) == set(parent_runs) == set(videos), "parent roster drift")

    teacher = DepthAnythingV2MetricSource(
        args.dav2_repo,
        args.dav2_checkpoint,
        args.device,
        input_size=args.dav2_input_size,
        precision=args.precision,
    )
    args.output_dir.mkdir(parents=True)

    records: dict[str, list[dict[str, Any]]] = {
        "primary_existing_signal": [],
        "primary_pair_signal": [],
        "secondary_pair_signal": [],
        "geometric_mean_pair_signal": [],
    }
    frame_receipts: list[dict[str, Any]] = []
    total_pixels = 0
    source_valid_pixels = 0
    old_teacher_added = 0
    new_teacher_added = 0
    metric_valid_pixels = 0
    support_valid_pixels = 0
    evidence_valid_pixels = 0
    pair_valid_pixels = 0
    pair_scales: list[float] = []

    for parent_id in parent_ids:
        video = videos[parent_id]
        trajectory = parse_trajectory(resolve_trajectory_path(video))
        for summary in parent_runs[parent_id]["frame_summaries"]:
            frame_index = int(summary["frame_index"])
            frame_stem = str(summary["frame_stem"])
            stage_path = args.stage0a_result.parent / f"{parent_id}_{frame_stem}.npz"
            stage = _load_npz(stage_path)
            base_path = args.base_label_dir / f"{frame_stem}.npz"
            base = _load_npz(base_path)
            primary_depth = np.asarray(stage["prediction_depth_m"], dtype=np.float32)
            primary_valid = np.asarray(
                base["teacher_candidate_valid_hw"], dtype=np.bool_
            )
            output_wh = (primary_depth.shape[1], primary_depth.shape[0])

            source_frame = load_factor_source_frame(video, frame_index, trajectory)
            raw_secondary, _ = teacher.infer(source_frame["rgb_upright"], {})
            secondary_frame = dict(source_frame)
            secondary_frame["depth_m_upright"] = np.asarray(
                raw_secondary, dtype=np.float32
            )
            secondary_frame["depth_valid_upright"] = (
                np.isfinite(raw_secondary) & (raw_secondary > 0)
            )
            secondary_frame["confidence_upright"] = secondary_frame[
                "depth_valid_upright"
            ].astype(np.uint8)
            secondary_depth, secondary_valid, _, replay_intrinsics = (
                _preprocess_source_for_output(secondary_frame, output_wh)
            )
            require(
                np.allclose(replay_intrinsics, base["intrinsics_output"], atol=1e-6),
                "second-Teacher intrinsics replay drift",
            )
            scale, scale_support = robust_observed_scale(
                stage["observed_depth_m"], secondary_depth
            )
            pair_scales.append(scale)
            secondary_depth = secondary_depth * scale
            secondary_valid &= np.isfinite(secondary_depth) & (secondary_depth > 0)
            disagreement, pair_quality, pair_valid = teacher_pair_quality(
                primary_depth,
                primary_valid,
                secondary_depth,
                secondary_valid,
            )
            base_quality = _base_combined_quality(base)
            tiers, provenance, quality_score = regrade_teacher_labels(
                base["source_native_valid_hw"],
                base["quality_tier_hw"],
                base["provenance_code_hw"],
                primary_valid,
                base_quality,
                base["anchor_quality_hw"],
                base["multiview_valid_hw"],
                pair_quality,
                pair_valid,
            )

            source_valid = np.asarray(base["source_native_valid_hw"], dtype=np.bool_)
            metric_depth = np.where(
                source_valid,
                base["metric_depth_m_hw"],
                primary_depth,
            ).astype(np.float32)
            metric_valid = tiers > TIER_UNKNOWN
            metric_depth[~metric_valid] = np.nan
            uncertainty = depth_uncertainty_proxy(
                metric_depth,
                tiers,
                provenance,
                quality_score,
                base["anchor_residual_m_hw"],
                base["multiview_residual_m_hw"],
                base["multiview_valid_hw"],
            )
            teacher_pixels = provenance == PROVENANCE_TEACHER
            pair_absolute_difference = np.abs(primary_depth - secondary_depth)
            uncertainty[teacher_pixels] = np.sqrt(
                np.square(uncertainty[teacher_pixels])
                + np.square(0.5 * pair_absolute_difference[teacher_pixels])
            )
            factors = compute_geometric_factors(
                metric_depth,
                metric_valid,
                base["intrinsics_output"],
                base["camera_to_world_output"],
                quality_score,
                tiers,
                provenance,
                uncertainty,
            )

            payload = {
                "metric_depth_m_hw": metric_depth,
                "metric_depth_valid_hw": metric_valid,
                "depth_uncertainty_proxy_m_hw": uncertainty,
                "quality_score_hw": quality_score,
                "quality_tier_hw": tiers,
                "provenance_code_hw": provenance,
                "source_native_valid_hw": source_valid,
                "teacher_candidate_valid_hw": primary_valid,
                "primary_teacher_depth_m_hw": primary_depth,
                "secondary_teacher_depth_m_hw": secondary_depth.astype(np.float32),
                "secondary_teacher_valid_hw": secondary_valid,
                "secondary_teacher_anchor_scale": np.asarray(scale, dtype=np.float32),
                "secondary_teacher_anchor_support": np.asarray(
                    scale_support, dtype=np.int64
                ),
                "teacher_pair_valid_hw": pair_valid,
                "teacher_pair_relative_disagreement_hw": disagreement,
                "teacher_pair_quality_hw": pair_quality,
                "pre_pair_combined_quality_hw": base_quality,
                "teacher_confidence_quality_hw": base[
                    "teacher_confidence_quality_hw"
                ],
                "anchor_quality_hw": base["anchor_quality_hw"],
                "anchor_residual_m_hw": base["anchor_residual_m_hw"],
                "anchor_distance_px_hw": base["anchor_distance_px_hw"],
                "multiview_quality_hw": base["multiview_quality_hw"],
                "multiview_residual_m_hw": base["multiview_residual_m_hw"],
                "multiview_valid_hw": base["multiview_valid_hw"],
                "intrinsics_output": base["intrinsics_output"],
                "camera_to_world_output": base["camera_to_world_output"],
                **factors,
            }
            for key in payload:
                lower = key.lower()
                require(
                    not any(token in lower for token in FORBIDDEN_TASK_TOKENS),
                    f"forbidden task field in output: {key}",
                )
            require(
                np.all(np.isfinite(metric_depth[metric_valid])),
                "valid metric depth contains non-finite values",
            )
            output_path = args.output_dir / f"{frame_stem}.npz"
            np.savez_compressed(output_path, **payload)

            old_added = (~source_valid) & np.asarray(
                base["metric_depth_valid_hw"], dtype=np.bool_
            )
            new_added = (~source_valid) & metric_valid
            total_pixels += metric_valid.size
            source_valid_pixels += int(source_valid.sum())
            old_teacher_added += int(old_added.sum())
            new_teacher_added += int(new_added.sum())
            metric_valid_pixels += int(metric_valid.sum())
            support_valid_pixels += int(factors["support_truth_valid_hw"].sum())
            evidence_valid_pixels += int(factors["evidence_truth_valid_hw"].sum())
            pair_valid_pixels += int(pair_valid.sum())

            hidden = np.asarray(stage["hidden_mask"], dtype=np.bool_)
            truth = np.asarray(stage["truth_depth_m"], dtype=np.float32)
            geometric_mean = np.sqrt(
                np.maximum(primary_depth, 1e-6)
                * np.maximum(secondary_depth, 1e-6)
            ).astype(np.float32)
            pair_combined = np.sqrt(
                np.clip(base_quality, 0.0, 1.0)
                * np.clip(pair_quality, 0.0, 1.0)
            ).astype(np.float32)
            common = {
                "parent_id": parent_id,
                "truth_depth_m": truth,
                "hidden_mask": hidden,
                "model_mask": pair_valid,
                "baseline_depth_m": stage["source_only_nearest_depth_m"],
            }
            for name, prediction, confidence in (
                ("primary_existing_signal", primary_depth, base_quality),
                ("primary_pair_signal", primary_depth, pair_combined),
                ("secondary_pair_signal", secondary_depth, pair_combined),
                ("geometric_mean_pair_signal", geometric_mean, pair_combined),
            ):
                records[name].append(
                    {
                        **common,
                        "prediction_depth_m": prediction,
                        "confidence": confidence,
                    }
                )
            frame_receipts.append(
                {
                    "parent_id": parent_id,
                    "frame_index": frame_index,
                    "frame_stem": frame_stem,
                    "output_path": str(output_path.resolve()),
                    "output_bytes": output_path.stat().st_size,
                    "secondary_anchor_scale": scale,
                    "secondary_anchor_support": scale_support,
                    "pair_valid_rate": float(np.mean(pair_valid)),
                    "old_teacher_added_rate": float(np.mean(old_added)),
                    "new_teacher_added_rate": float(np.mean(new_added)),
                    "support_valid_rate": float(
                        np.mean(factors["support_truth_valid_hw"])
                    ),
                    "evidence_valid_rate": float(
                        np.mean(factors["evidence_truth_valid_hw"])
                    ),
                }
            )

    selective = {
        name: compute_selective_metrics(values) for name, values in records.items()
    }
    primary_split = _split_error(
        records["primary_pair_signal"], threshold=TEACHER_C_QUALITY
    )
    accepted_mae = primary_split["accepted"]["mae_m"]
    rejected_mae = primary_split["rejected"]["mae_m"]
    uncertainty_separates_error = bool(
        accepted_mae is not None
        and rejected_mae is not None
        and accepted_mae < rejected_mae
    )
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "mode": "WILD_LAB_REVERSIBLE_EXPLORATION",
        "question": "Does an independent source-anchored single-view Teacher add useful disagreement evidence for continuous factor pseudo-labels and UNKNOWN?",
        "input": {
            "stage0a_result_path": str(args.stage0a_result.resolve()),
            "stage0a_result_sha256": sha256_file(args.stage0a_result),
            "base_label_result_path": str(
                (args.base_label_dir / "result.json").resolve()
            ),
            "base_label_result_sha256": sha256_file(
                args.base_label_dir / "result.json"
            ),
            "source_manifest_path": str(source_manifest_path.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "parent_count": len(parent_ids),
            "frame_count": len(frame_receipts),
        },
        "teachers": {
            "primary": {
                "model_id": stage0a["teacher"]["model_id"],
                "checkpoint_sha256": stage0a["teacher"]["checkpoint_sha256"],
                "role": "PRIMARY_MULTIVIEW_COMPLETION",
            },
            "secondary": {
                "model_id": "depth-anything-v2-metric-hypersim-vits",
                "checkpoint_path": str(args.dav2_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(args.dav2_checkpoint),
                "role": "INDEPENDENT_DISAGREEMENT_EVIDENCE_NOT_TRUTH",
                "precision": args.precision,
                "input_size": args.dav2_input_size,
            },
        },
        "factory": {
            "source_priority": "SOURCE_NATIVE_VALID_DEPTH_OVERRIDES_ALL_TEACHERS",
            "primary_depth_policy": "KEEP_MAPANYTHING_DEPTH_SECOND_TEACHER_ONLY_REGRADES_QUALITY",
            "pair_disagreement": "SYMMETRIC_RELATIVE_DEPTH_ERROR",
            "pair_disagreement_scale": PAIR_DISAGREEMENT_SCALE,
            "pair_quality": "EXP_NEGATIVE_DISAGREEMENT_OVER_SCALE",
            "teacher_quality_combination": "GEOMETRIC_MEAN_OF_EXISTING_AND_PAIR_QUALITY",
            "unknown_policy": "TEACHER_ONLY_PIXEL_WITHOUT_PAIR_VALIDITY_OR_C_THRESHOLD_IS_UNKNOWN",
            "secondary_scale_policy": "PER_FRAME_MEDIAN_RATIO_ON_OBSERVED_SOURCE_ANCHOR_ONLY",
            "forbidden_task_fields_written": False,
        },
        "coverage": {
            "total_pixels": total_pixels,
            "source_native": source_valid_pixels / total_pixels,
            "pair_valid": pair_valid_pixels / total_pixels,
            "old_teacher_added": old_teacher_added / total_pixels,
            "new_teacher_added": new_teacher_added / total_pixels,
            "metric_depth": metric_valid_pixels / total_pixels,
            "support": support_valid_pixels / total_pixels,
            "boundary_evidence": evidence_valid_pixels / total_pixels,
        },
        "secondary_anchor_scale": {
            "minimum": float(np.min(pair_scales)),
            "median": float(np.median(pair_scales)),
            "maximum": float(np.max(pair_scales)),
        },
        "risk_coverage": {
            name: {
                "compact_curve": _compact_curve(metrics),
                "full": metrics,
            }
            for name, metrics in selective.items()
        },
        "frozen_c_threshold_split": primary_split,
        "decision_signal": {
            "uncertainty_separates_accepted_from_rejected_error": uncertainty_separates_error,
            "accepted_mae_m": accepted_mae,
            "rejected_mae_m": rejected_mae,
            "no_requirement_to_beat_either_teacher": True,
            "interpretation": "The second Teacher is useful if disagreement identifies less reliable primary labels at non-zero coverage; it is not required to have lower global error than the primary Teacher.",
        },
        "frame_receipts": frame_receipts,
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": "Consumed source-role multi-Teacher pseudo-label and selective-risk diagnostic. Outputs are graded supervision, not complete truth, calibrated uncertainty truth, cross-source generalization, deployment, product, or safety evidence.",
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0a-result", type=Path, default=DEFAULT_STAGE0A_RESULT)
    parser.add_argument("--base-label-dir", type=Path, default=DEFAULT_BASE_LABEL_DIR)
    parser.add_argument("--dav2-repo", type=Path, default=DEFAULT_DAV2_REPO)
    parser.add_argument("--dav2-checkpoint", type=Path, default=DEFAULT_DAV2_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--dav2-input-size", type=int, default=518)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        result_path = args.output_dir / "result.json"
        _write_json_exclusive(result_path, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result": str(result_path),
                    "coverage": result["coverage"],
                    "decision_signal": result["decision_signal"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
