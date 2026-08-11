#!/usr/bin/env python3
"""Evaluate the frozen R16 angular boundary specialist on Bonn RGB-D geometry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_student_bonn_depth import (
    DEFAULT_BONN_ARCHIVE,
    DEFAULT_BONN_CATALOG,
    DEFAULT_BONN_RECEIPT,
    DEFAULT_BONN_ROOT,
    fixed_frame_pairs,
    load_cohort_indices,
    load_depth_native,
    load_rgb_native,
    validate_source_receipts,
)
from evaluate_ag_st_unified_student_bonn_boundary import BONN_INTRINSICS, parent_macro
from materialize_ag_st_source_native_boundary_corpus import conservative_source_boundary
from train_ag_st_soft_boundary_bonn_canary import boundary_metrics
from train_ag_st_source_boundary_student import DEFAULT_MOBILENET, build_decoder


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COHORT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_BONN_MIXED_DOMAIN_COHORT_R0_2026-08-10.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-angular-boundary-student-r1/boundary-decoder.pt"
)
DEFAULT_R14_BASELINE = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-unified-factor-student-depthart-multisource-r0/bonn-evaluation-boundary.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-angular-boundary-student-r1/bonn-boundary.json"
)
INFERENCE_HW = (240, 320)


def checkpoint_parent_sources(split: dict[str, list[tuple[str, str]]]) -> set[str]:
    return {str(source) for values in split.values() for source, _ in values}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torchvision.models import mobilenet_v3_small

    started = time.monotonic()
    output = args.output.resolve()
    for path in (
        args.checkpoint,
        args.mobilenet_checkpoint,
        args.cohort_manifest,
        args.r14_baseline,
    ):
        require(path.is_file(), f"input missing: {path}")
    require(not output.exists(), f"output exists: {output}")
    frame_indices = load_cohort_indices(args.cohort_manifest.resolve(), "evaluation")
    _, _, source_provenance = validate_source_receipts(
        args.dataset_root.resolve(),
        args.archive.resolve(),
        args.catalog.resolve(),
        args.receipt.resolve(),
        set(frame_indices),
    )
    pairs_by_parent = fixed_frame_pairs(args.dataset_root.resolve(), frame_indices)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(checkpoint.get("schema") == "blindassist_ag_st_source_boundary_decoder_v1", "checkpoint schema drift")
    require(checkpoint.get("target_mode") == "angular", "checkpoint is not angular-supervised")
    require(
        checkpoint["mobilenet_checkpoint_sha256"] == sha256_file(args.mobilenet_checkpoint),
        "MobileNet checkpoint digest drift",
    )
    training_sources = checkpoint_parent_sources(checkpoint["split"])
    require(training_sources == {"arkitscenes", "tum_rgbd"}, "checkpoint source contract drift")
    threshold = float(checkpoint["selected_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_model = mobilenet_v3_small(weights=None)
    backbone_model.load_state_dict(
        torch.load(args.mobilenet_checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    backbone = backbone_model.features.eval().to(device)
    decoder = build_decoder().to(device).eval()
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    # Phase 1 is RGB-only. Source depth is not opened until every prediction is frozen.
    predictions: list[tuple[str, Any, np.ndarray]] = []
    with torch.no_grad():
        for parent_id, pairs in pairs_by_parent.items():
            for pair in pairs:
                rgb_native = load_rgb_native(pair.rgb.absolute_path)
                rgb = np.asarray(
                    Image.fromarray(rgb_native).resize(
                        (INFERENCE_HW[1], INFERENCE_HW[0]), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                ).copy()
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(
                    device=device, dtype=torch.float32
                ) / 255.0
                value = ((tensor - mean) / std)[None]
                captured = []
                for layer_index, layer in enumerate(backbone):
                    value = layer(value)
                    if layer_index in (1, 3, 8, 12):
                        captured.append(value)
                require(len(captured) == 4, "MobileNet feature capture drift")
                logits = decoder(tuple(captured), INFERENCE_HW)
                probability = torch.sigmoid(logits)[:, None]
                probability = functional.interpolate(
                    probability, size=(480, 640), mode="bilinear", align_corners=False
                )[0, 0].cpu().numpy().astype(np.float32)
                predictions.append((parent_id, pair, probability))

    values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    by_parent_values: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        parent_id: [] for parent_id in pairs_by_parent
    }
    frames: list[dict[str, Any]] = []
    for parent_id, pair, probability in predictions:
        depth, source_valid = load_depth_native(pair.depth.absolute_path)
        truth_probability, boundary_valid = conservative_source_boundary(
            depth, source_valid, BONN_INTRINSICS
        )
        truth = boundary_valid & (truth_probability >= 0.5)
        item = (probability, truth, boundary_valid)
        values.append(item)
        by_parent_values[parent_id].append(item)
        frames.append(
            {
                "parent_id": parent_id,
                "rgb_row_index_zero_based": pair.rgb.row_index,
                "rgb_sha256": sha256_file(pair.rgb.absolute_path),
                "depth_sha256": sha256_file(pair.depth.absolute_path),
                "boundary_valid_pixels": int(np.sum(boundary_valid)),
                "boundary_positive_pixels": int(np.sum(truth)),
            }
        )

    tolerance = 4
    metrics = boundary_metrics(values, threshold, tolerance)
    by_parent = {
        parent_id: boundary_metrics(parent_values, threshold, tolerance)
        for parent_id, parent_values in sorted(by_parent_values.items())
    }
    macro = parent_macro(by_parent)
    r14_document = json.loads(args.r14_baseline.read_text(encoding="utf-8"))
    r14 = r14_document["overall"]["student_probability"]
    gates = {
        "checkpoint_sources_exactly_arkit_and_tum": training_sources
        == {"arkitscenes", "tum_rgbd"},
        "parent_count_eq_8_and_frame_count_eq_24": len(pairs_by_parent) == 8
        and len(frames) == 24,
        "threshold_frozen_from_internal_selection": threshold
        == float(checkpoint["selected_threshold"]),
        "average_precision_not_below_r14": float(metrics["student_average_precision"])
        >= float(r14["student_average_precision"]),
        "f1_within_4px_ge_0p25": float(metrics["f1_within_tolerance"]) >= 0.25,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_angular_boundary_student_bonn_result_v1",
        "status": "ANGULAR_BOUNDARY_BONN_EXTERNAL_PASS" if passed else "ANGULAR_BOUNDARY_BONN_EXTERNAL_FAIL",
        "question": "Does the frozen ARKit/TUM angular-boundary specialist retain the external Bonn geometry signal without fitting or threshold selection?",
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "mobilenet_checkpoint_sha256": sha256_file(args.mobilenet_checkpoint),
            "cohort": str(args.cohort_manifest.resolve()),
            "cohort_sha256": sha256_file(args.cohort_manifest),
            "source_provenance": source_provenance,
            "r14_baseline": str(args.r14_baseline.resolve()),
            "r14_baseline_sha256": sha256_file(args.r14_baseline),
        },
        "protocol": {
            "training_sources": sorted(training_sources),
            "evaluation_source": "bonn_rgbd_dynamic",
            "parent_count": len(pairs_by_parent),
            "frame_count": len(frames),
            "inference_shape_hw": list(INFERENCE_HW),
            "metric_shape_hw": [480, 640],
            "selected_threshold": threshold,
            "tolerance_px_at_metric_shape": tolerance,
            "fitting_or_threshold_selection_on_bonn": False,
            "all_rgb_predictions_frozen_before_source_depth_open": True,
        },
        "metrics": metrics,
        "parent_macro": macro,
        "by_parent": by_parent,
        "frames": frames,
        "r14_reference": {
            "average_precision": float(r14["student_average_precision"]),
            "f1_within_4px": float(r14["f1_within_tolerance"]),
            "threshold": float(r14["threshold"]),
        },
        "gates": gates,
        "execution": {"device": str(device), "elapsed_seconds": time.monotonic() - started},
        "decision": {
            "bonn_boundary_signal_retained": passed,
            "teacher_filled_boundary_authorized": False,
            "formal_f1_authority_changed": False,
        },
        "claim_boundary": "External Bonn source-depth boundary diagnostic of one frozen checkpoint. No complete truth, task utility, formal F1, safety, deployment, product, or license claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mobilenet-checkpoint", type=Path, default=DEFAULT_MOBILENET)
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--r14-baseline", type=Path, default=DEFAULT_R14_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "metrics": result["metrics"], "gates": result["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
