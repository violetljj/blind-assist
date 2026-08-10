#!/usr/bin/env python3
"""Evaluate a frozen unified-factor checkpoint on ICL exact-depth boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_student_bonn_depth import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    build_students,
    checkpoint_architecture,
    checkpoint_parent_ids,
)
from train_ag_st_masked_student import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DepthArtDenseFeatureExtractor,
    load_depthart_backbone,
)
from train_ag_st_soft_boundary_bonn_canary import boundary_metrics


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOUNDARY_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-continuous-boundary-factors-r0/result.json"
)
DEFAULT_RGB_BINDING = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-source-native-boundary-corpus-r0/rgb_binding.json"
)
ICL_OUTPUT_HW = (120, 160)
ICL_INTRINSICS_OUTPUT = np.asarray(
    [[120.3, 0.0, 79.375], [0.0, 120.0, 59.375], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def load_icl_bound_rgb(row: dict[str, Any]) -> np.ndarray:
    path = Path(row["rgb_path"])
    require(path.is_file(), f"ICL RGB missing: {path}")
    payload = path.read_bytes()
    require(
        hashlib.sha256(payload).hexdigest().upper() == str(row["rgb_sha256"]),
        "ICL RGB SHA drift",
    )
    with Image.open(path) as image:
        raw = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    require(raw.shape == (480, 640, 3), "ICL raw RGB shape drift")
    output = np.ascontiguousarray(np.flipud(raw)[2::4, 2::4])
    require(output.shape == (120, 160, 3), "ICL bound RGB transform drift")
    return output


def extract_icl_feature(
    extractor: DepthArtDenseFeatureExtractor,
    rgb: np.ndarray,
    feature_profile: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = ICL_OUTPUT_HW
    padded_height = int(math.ceil(height / 32.0) * 32)
    padded_width = int(math.ceil(width / 32.0) * 32)
    value = rgb.astype(np.float32) / 255.0
    normalized = ((value - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
    image = torch.from_numpy(normalized)[None].to(device)
    image = F.pad(
        image,
        (0, padded_width - width, 0, padded_height - height),
        mode="constant",
        value=0.0,
    )
    intrinsics = torch.from_numpy(ICL_INTRINSICS_OUTPUT.copy())[None].to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        cameras = extractor.metric_depthart.cam_embedder(
            intrinsics,
            padded_height,
            padded_width,
            device,
        )
        features = extractor.metric_depthart.pretrained.forward_with_adapters(
            image,
            adapters=[
                extractor.metric_depthart.daa1,
                extractor.metric_depthart.daa2,
                extractor.metric_depthart.daa3,
                extractor.metric_depthart.daa4,
            ],
            cams=list(cameras),
        )
        relative_depth, shared, pyramid = extractor.decode(
            list(features),
            (padded_height, padded_width),
        )
        scale = extractor.metric_depthart.sfh(features[3], cameras[3])
        base_depth = (
            relative_depth
            * scale.view(-1, 1, 1, 1)
            * extractor.metric_depthart.max_depth
        ).float().clamp(0.05, 20.0)
        selected = shared if feature_profile == "shared" else pyramid
    content_h = int(round(selected.shape[-2] * height / padded_height))
    content_w = int(round(selected.shape[-1] * width / padded_width))
    require(
        content_h * padded_height == selected.shape[-2] * height
        and content_w * padded_width == selected.shape[-1] * width,
        "ICL padded feature ratio is not integral",
    )
    selected = selected[..., :content_h, :content_w]
    base_depth = base_depth[..., :height, :width]
    require(bool(torch.isfinite(selected).all()), "non-finite ICL feature")
    require(bool(torch.isfinite(base_depth).all()), "non-finite ICL base depth")
    return selected, base_depth


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output = args.output.resolve()
    checkpoint_path = args.student_checkpoint.resolve()
    boundary_result_path = args.boundary_result.resolve()
    binding_path = args.rgb_binding.resolve()
    require(not output.exists(), f"ICL boundary output collision: {output}")
    require(
        checkpoint_path.is_file() and boundary_result_path.is_file() and binding_path.is_file(),
        "ICL boundary evaluator input missing",
    )
    require(torch.cuda.is_available(), "ICL boundary evaluation requires CUDA")
    boundary_result = json.loads(boundary_result_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    require(
        boundary_result.get("status") == "CONTINUOUS_BOUNDARY_FACTORS_PASS",
        "ICL continuous boundary input incomplete",
    )
    require(
        binding.get("status") == "SOURCE_NATIVE_BOUNDARY_RGB_BINDING_PASS",
        "ICL RGB binding incomplete",
    )
    boundary_rows = {
        str(row["frame_id"]): row
        for row in boundary_result["frames"]
        if row["source"] == "icl_exact"
    }
    rgb_rows = {
        str(row["frame_id"]): row
        for row in binding["frames"]
        if row["source"] == "icl_exact"
    }
    require(
        len(boundary_rows) == len(rgb_rows) == 12 and set(boundary_rows) == set(rgb_rows),
        "ICL boundary/RGB frame identity drift",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "student checkpoint invalid")
    architecture = checkpoint_architecture(checkpoint)
    require(
        "icl_living_room_kt1" not in checkpoint_parent_ids(checkpoint),
        "checkpoint/ICL parent overlap",
    )
    device = torch.device("cuda")
    extractor, scan = load_depthart_backbone(
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(checkpoint["seed"]),
    )
    baseline, student = build_students(checkpoint, architecture, device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    values: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "baseline_probability": [],
        "student_probability": [],
        "student_distance": [],
    }
    frames: list[dict[str, Any]] = []
    for frame_id in sorted(boundary_rows):
        boundary_row = boundary_rows[frame_id]
        rgb_row = rgb_rows[frame_id]
        rgb = load_icl_bound_rgb(rgb_row)
        feature, base_depth = extract_icl_feature(
            extractor,
            rgb,
            architecture["feature_profile"],
            device,
            amp_dtype,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            baseline_output = baseline(feature, base_depth, ICL_OUTPUT_HW)
            student_output = student(feature, base_depth, ICL_OUTPUT_HW)
        predictions = {
            "baseline_probability": torch.sigmoid(baseline_output["boundary_logits"])[
                0, 0
            ].float().cpu().numpy(),
            "student_probability": torch.sigmoid(student_output["boundary_logits"])[
                0, 0
            ].float().cpu().numpy(),
            "student_distance": np.exp(
                -0.5
                * np.square(
                    student_output["boundary_distance_px"][0, 0]
                    .float()
                    .cpu()
                    .numpy()
                    / 3.0
                )
            ).astype(np.float32),
        }
        # Exact source boundary is opened only after all RGB/K predictions exist.
        target_path = Path(boundary_row["output"])
        require(target_path.is_file(), "ICL continuous boundary payload missing")
        require(
            sha256_file(target_path) == boundary_row["output_sha256"],
            "ICL continuous boundary payload SHA drift",
        )
        with np.load(target_path) as target_payload:
            valid = np.asarray(target_payload["boundary_truth_valid_hw"], dtype=np.bool_)
            truth = valid & (
                np.asarray(
                    target_payload["boundary_core_probability_hw"], dtype=np.float32
                )
                >= 0.5
            )
        require(valid.shape == truth.shape == ICL_OUTPUT_HW, "ICL boundary target shape drift")
        for name, prediction in predictions.items():
            values[name].append((prediction, truth, valid))
        frames.append(
            {
                "frame_id": frame_id,
                "rgb_sha256": str(rgb_row["rgb_sha256"]),
                "boundary_sha256": str(boundary_row["output_sha256"]),
                "valid_pixels": int(np.sum(valid)),
                "positive_pixels": int(np.sum(truth)),
            }
        )
    threshold = 0.5
    tolerance = 2
    overall = {
        name: boundary_metrics(rows, threshold, tolerance)
        for name, rows in values.items()
    }
    result = {
        "schema": "blindassist_ag_st_unified_student_icl_boundary_evaluation_v1",
        "status": "EXTERNAL_ICL_EXACT_BOUNDARY_DIAGNOSTIC_COMPLETE",
        "question": "Does the frozen multisource unified-factor checkpoint transfer boundary localization to checkpoint-unseen ICL exact geometry without fitting or threshold selection?",
        "inputs": {
            "student_checkpoint": str(checkpoint_path),
            "student_checkpoint_sha256": sha256_file(checkpoint_path),
            "continuous_boundary_result": str(boundary_result_path),
            "continuous_boundary_result_sha256": sha256_file(boundary_result_path),
            "rgb_binding": str(binding_path),
            "rgb_binding_sha256": sha256_file(binding_path),
            "depthart_checkpoint_sha256": sha256_file(args.depthart_checkpoint.resolve()),
        },
        "protocol": {
            "source": "ICL-NUIM living room trajectory 1 exact depth",
            "parent_count": 1,
            "frame_count": 12,
            "threshold": threshold,
            "tolerance_px": tolerance,
            "exact_target_opened_after_rgb_k_predictions": True,
            "fitting_or_threshold_selection": False,
        },
        "intrinsics_output": ICL_INTRINSICS_OUTPUT.tolist(),
        "overall": overall,
        "frames": frames,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "scan_backend": scan,
        },
        "decision": {
            "formal_f1_authority_changed": False,
            "task_utility_evaluated": False,
            "support_or_obstacle_evaluated": False,
        },
        "claim_boundary": "Checkpoint-unseen external synthetic-exact boundary diagnostic over one ICL parent; no calibrated probability, real-world task utility, support, obstacle, formal F1, safety, deployment, or product claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "overall": overall,
                "execution": result["execution"],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-result", type=Path, default=DEFAULT_BOUNDARY_RESULT)
    parser.add_argument("--rgb-binding", type=Path, default=DEFAULT_RGB_BINDING)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
