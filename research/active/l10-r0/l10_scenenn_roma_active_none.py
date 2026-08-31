#!/usr/bin/env python3
"""Fresh SceneNN geometry-controlled RoMa cycle-support/NONE challenge."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_NVIDIA_ROOT = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
_DLL_DIRECTORY_HANDLES = []
if os.name == "nt" and _NVIDIA_ROOT.is_dir():
    for _dll_dir in sorted(_NVIDIA_ROOT.glob("*/bin")):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(_dll_dir)))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_efficientloftr_fresh_none as predecessor  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-roma-active-none-protocol-v1"
ADMISSION_SCHEMA = "blindassist-l10-scenenn-roma-active-none-source-admission-v1"
COHORT_SCHEMA = "blindassist-l10-scenenn-roma-active-none-cohort-v1"
RECEIPT_SCHEMA = "blindassist-l10-scenenn-roma-active-none-rgb-receipt-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-roma-active-none-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return predecessor.sha256(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    prior = protocol["predecessor"]
    prior_path = HERE / prior["result_path"]
    require(prior_path.is_file(), "PREDECESSOR_MISSING")
    require(sha256(prior_path) == prior["result_sha256"], "PREDECESSOR_HASH")
    require(load_json(prior_path)["conclusion"] == prior["required_conclusion"], "PREDECESSOR_CONCLUSION")
    model_root = ROOT / protocol["matcher"]["path"]
    require(sha256(model_root / "roma_indoor.pth") == protocol["matcher"]["roma_weights_sha256"], "ROMA_WEIGHTS")
    require(sha256(model_root / "dinov2_vitl14_pretrain.pth") == protocol["matcher"]["dinov2_weights_sha256"], "ROMA_DINOV2_WEIGHTS")
    wheel_path = ROOT / protocol["matcher"]["wheel_path"]
    require(sha256(wheel_path) == protocol["matcher"]["wheel_sha256"], "ROMA_WHEEL")
    return protocol


def minimum_eligible_baseline_pair(rows: list[dict[str, Any]], minimum_baseline: float, minimum_gap: int):
    eligible = [row for row in rows if row["eligible"]]
    candidates = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            if abs(int(left["frame"]) - int(right["frame"])) < minimum_gap:
                continue
            left_center = np.asarray(left["camera_center_world"], dtype=np.float64)
            right_center = np.asarray(right["camera_center_world"], dtype=np.float64)
            baseline = float(np.linalg.norm(left_center - right_center))
            if baseline < minimum_baseline:
                continue
            first, second = (left, right) if int(left["frame"]) < int(right["frame"]) else (right, left)
            rank = (
                baseline,
                -min(int(first["visible_pixels"]), int(second["visible_pixels"])),
                -min(float(first["visible_to_target_raster_ratio"]), float(second["visible_to_target_raster_ratio"])),
                int(first["frame"]),
                int(second["frame"]),
            )
            candidates.append((rank, first, second, baseline))
    if not candidates:
        return None
    _, reference, query, baseline = min(candidates, key=lambda row: row[0])
    return reference, query, baseline


@contextmanager
def predecessor_surface():
    saved = {
        "PROTOCOL_SCHEMA": predecessor.PROTOCOL_SCHEMA,
        "ADMISSION_SCHEMA": predecessor.ADMISSION_SCHEMA,
        "COHORT_SCHEMA": predecessor.COHORT_SCHEMA,
        "RECEIPT_SCHEMA": predecessor.RECEIPT_SCHEMA,
        "load_protocol": predecessor.load_protocol,
        "select_pair": predecessor.parent.visible.select_pair,
        "__file__": predecessor.__file__,
    }
    predecessor.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    predecessor.ADMISSION_SCHEMA = ADMISSION_SCHEMA
    predecessor.COHORT_SCHEMA = COHORT_SCHEMA
    predecessor.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    predecessor.load_protocol = load_protocol
    predecessor.parent.visible.select_pair = minimum_eligible_baseline_pair
    predecessor.__file__ = __file__
    try:
        yield
    finally:
        predecessor.PROTOCOL_SCHEMA = saved["PROTOCOL_SCHEMA"]
        predecessor.ADMISSION_SCHEMA = saved["ADMISSION_SCHEMA"]
        predecessor.COHORT_SCHEMA = saved["COHORT_SCHEMA"]
        predecessor.RECEIPT_SCHEMA = saved["RECEIPT_SCHEMA"]
        predecessor.load_protocol = saved["load_protocol"]
        predecessor.parent.visible.select_pair = saved["select_pair"]
        predecessor.__file__ = saved["__file__"]


def freeze(protocol_path: Path, admission_path: Path, source_root: Path, output_path: Path) -> None:
    with predecessor_surface():
        predecessor.freeze(protocol_path, admission_path, source_root, output_path)
    cohort = load_json(output_path)
    cohort["authority"] = "FROZEN_PRE_RGBD_GEOMETRY_CONTROLLED_ROMA_NONE_COHORT"
    cohort["selection_policy"] = "minimum eligible camera baseline, then maximum minimum visibility, then earlier frames"
    predecessor.parent.write_json(output_path, cohort)


def seal(cohort_path: Path, source_root: Path, extraction_root: Path, extractor_exe: Path, output_path: Path) -> None:
    with predecessor_surface():
        predecessor.seal(cohort_path, source_root, extraction_root, extractor_exe, output_path)
    receipt = load_json(output_path)
    receipt["authority"] = "POST_COHORT_FREEZE_GEOMETRY_CONTROLLED_SPARSE_RGB_ONLY_RECEIPT"
    predecessor.parent.write_json(output_path, receipt)


def cycle_direction(
    source_coords: torch.Tensor,
    forward_coords: torch.Tensor,
    forward_certainty: torch.Tensor,
    backward_coords: torch.Tensor,
    backward_certainty: torch.Tensor,
    matcher: dict[str, Any],
) -> dict[str, float | int]:
    sampled_backward_coords = F.grid_sample(
        backward_coords.permute(2, 0, 1)[None],
        forward_coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0)
    sampled_backward_certainty = F.grid_sample(
        backward_certainty[None, None],
        forward_coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    cycle_error = torch.linalg.vector_norm(sampled_backward_coords - source_coords, dim=-1)
    high = forward_certainty >= float(matcher["official_certainty_threshold"])
    cycle = high & (sampled_backward_certainty >= float(matcher["official_certainty_threshold"])) & (
        cycle_error <= float(matcher["maximum_cycle_error_normalized"])
    )
    total = int(high.numel())
    high_count = int(high.sum().item())
    cycle_count = int(cycle.sum().item())
    return {
        "mean_certainty": float(forward_certainty.mean().item()),
        "high_count": high_count,
        "high_fraction": high_count / total,
        "cycle_count": cycle_count,
        "cycle_fraction": cycle_count / total,
        "cycle_purity": cycle_count / high_count if high_count else 0.0,
    }


@torch.inference_mode()
def roma_support(model: Any, reference: Image.Image, query: Image.Image, matcher: dict[str, Any]) -> dict[str, Any]:
    warp, certainty = model.match(reference, query)
    warp = warp[0]
    certainty = certainty[0]
    height, double_width = certainty.shape
    width = double_width // 2
    require(height == width == int(matcher["upsample_resolution"]), "ROMA_OUTPUT_RESOLUTION")
    forward = warp[:, :width]
    backward = warp[:, width:]
    a_to_b = cycle_direction(forward[..., :2], forward[..., 2:], certainty[:, :width], backward[..., :2], certainty[:, width:], matcher)
    b_to_a = cycle_direction(backward[..., 2:], backward[..., :2], certainty[:, width:], forward[..., 2:], certainty[:, :width], matcher)
    score = math.sqrt(float(a_to_b["cycle_fraction"]) * float(b_to_a["cycle_fraction"]))
    supported = all(
        float(direction["cycle_fraction"]) >= float(matcher["minimum_directional_cycle_fraction"])
        and float(direction["cycle_purity"]) >= float(matcher["minimum_directional_cycle_purity"])
        for direction in (a_to_b, b_to_a)
    )
    return {"a_to_b": a_to_b, "b_to_a": b_to_a, "symmetric_cycle_score": score, "absolute_support": supported}


def replay(protocol_path: Path, cohort_path: Path, receipt_path: Path, source_root: Path, output_path: Path) -> None:
    from transformers import AutoImageProcessor, AutoModel
    import romatch

    protocol = load_protocol(protocol_path)
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "RECEIPT_SCHEMA")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "COHORT_PROTOCOL_HASH")
    require(cohort["entrypoint_sha256"] == sha256(Path(__file__)), "COHORT_ENTRYPOINT_HASH")
    require(receipt["cohort_sha256"] == sha256(cohort_path), "RECEIPT_COHORT_HASH")
    intrinsic = predecessor.parent.base.parse_intrinsic(predecessor.source_paths(source_root, cohort["episodes"][0]["scene_id"])["intrinsic"])
    crops: dict[str, Image.Image] = {}
    crop_receipts: dict[str, Any] = {}
    expansion = float(protocol["matcher"]["crop_expansion"])
    for episode in cohort["episodes"]:
        scene_id = episode["scene_id"]
        paths = predecessor.source_paths(source_root, scene_id)
        xyz, labels, faces = predecessor.parent.visible.read_mesh(paths["ply"])
        poses = {int(row["frame"]): row["camera_to_world"] for row in predecessor.parent.base.parse_poses(paths["trajectory"])}
        renderer = predecessor.parent.visible.VisibilityRenderer(xyz, labels, faces, int(episode["target_instance_id"]), intrinsic, protocol["pre_rgb_selector"]["renderer"])
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            sealed = receipt["sealed_frames"][f"{scene_id}:{frame}"]
            image_path = predecessor.selected_image(source_root, scene_id, frame)
            require(sha256(image_path) == sealed["image_sha256"], f"IMAGE_HASH:{scene_id}:{frame}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            visible_device, _ = renderer.visible_mask(poses[frame])
            visible_mask = renderer.cp.asnumpy(visible_device)
            require(predecessor.parent.visible.mask_sha256(visible_mask) == episode[role]["visible_mask_sha256"], f"MASK_HASH:{scene_id}:{role}")
            crop, box = predecessor.expanded_crop(image, episode[role]["visible_mask_tight_bbox_xyxy"], expansion)
            key = f"{episode['episode_id']}:{role}"
            crops[key] = crop
            crop_receipts[key] = {
                "scene_id": scene_id,
                "frame": frame,
                "source_image_sha256": sealed["image_sha256"],
                "visible_mask_sha256": episode[role]["visible_mask_sha256"],
                "tight_bbox_xyxy": episode[role]["visible_mask_tight_bbox_xyxy"],
                "expanded_bbox_xyxy": box,
                "crop_size": [crop.width, crop.height],
            }
    dino_root = ROOT / "artifacts.local/models/p1_a2_dinov2_small_ed25f3a"
    require(sha256(dino_root / "model.safetensors") == protocol["baseline"]["model_safetensors_sha256"], "DINO_WEIGHTS")
    dino_processor = AutoImageProcessor.from_pretrained(dino_root, local_files_only=True)
    dino_model = AutoModel.from_pretrained(dino_root, local_files_only=True).eval().to("cpu")
    ordered_keys = sorted(crops)
    dino_inputs = dino_processor(images=[crops[key] for key in ordered_keys], return_tensors="pt")
    with torch.inference_mode():
        dino_hidden = dino_model(**dino_inputs).last_hidden_state[:, 1:].mean(dim=1)
        dino_hidden = torch.nn.functional.normalize(dino_hidden, dim=1)
    dino = {key: dino_hidden[index].cpu().numpy().astype(np.float32) for index, key in enumerate(ordered_keys)}
    del dino_model
    model_root = ROOT / protocol["matcher"]["path"]
    weights = torch.load(model_root / "roma_indoor.pth", map_location="cpu", weights_only=True)
    dinov2_weights = torch.load(model_root / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True)
    matcher = romatch.roma_indoor(
        device="cuda",
        weights=weights,
        dinov2_weights=dinov2_weights,
        coarse_res=int(protocol["matcher"]["coarse_resolution"]),
        upsample_res=int(protocol["matcher"]["upsample_resolution"]),
        symmetric=True,
        use_custom_corr=False,
        upsample_preds=True,
    )
    episode_ids = [row["episode_id"] for row in cohort["episodes"]]
    dino_scores = np.zeros((4, 4), dtype=np.float64)
    roma_scores = np.zeros((4, 4), dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    for row, reference_id in enumerate(episode_ids):
        for column, query_id in enumerate(episode_ids):
            dino_scores[row, column] = float(dino[f"{reference_id}:reference"] @ dino[f"{query_id}:query"])
            diagnostic = roma_support(matcher, crops[f"{reference_id}:reference"], crops[f"{query_id}:query"], protocol["matcher"])
            diagnostics[f"{reference_id}->{query_id}"] = diagnostic
            roma_scores[row, column] = float(diagnostic["symmetric_cycle_score"]) if diagnostic["absolute_support"] else 0.0
    target_index = {value: index for index, value in enumerate(episode_ids)}
    scenarios: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[value] for value in references]
        columns = [target_index[value] for value in queries]
        dino_matrix = dino_scores[np.ix_(rows, columns)]
        roma_matrix = roma_scores[np.ix_(rows, columns)]
        dino_matches = predecessor.parent.open_zero.reciprocal_zero_assignment(dino_matrix)
        roma_matches = predecessor.parent.open_zero.reciprocal_zero_assignment(roma_matrix)
        scenarios.append({
            **scenario,
            "dinov2_patch_mean_cosine_matrix": dino_matrix.round(6).tolist(),
            "roma_supported_symmetric_cycle_matrix": roma_matrix.round(6).tolist(),
            "methods": {
                "dinov2_reciprocal_no_none_support": predecessor.parent.open_zero.evaluate_matches(references, queries, dino_matches),
                "roma_active_cycle_absolute_support_reciprocal": predecessor.parent.open_zero.evaluate_matches(references, queries, roma_matches),
            },
        })
    aggregates = {name: predecessor.parent.open_zero.aggregate(scenarios, name) for name in scenarios[0]["methods"]}
    baseline = aggregates["dinov2_reciprocal_no_none_support"]
    upgraded = aggregates["roma_active_cycle_absolute_support_reciprocal"]
    expected_true = int(cohort["counts"]["truth_matches_across_scenarios"])
    diagonal_supported = [bool(diagnostics[f"{episode_id}->{episode_id}"]["absolute_support"]) for episode_id in episode_ids]
    gate_met = all(diagonal_supported) and upgraded["true_positive"] == expected_true and upgraded["false_positive"] == 0 and upgraded["false_negative"] == 0 and upgraded["zero_assignment_exact_scenarios"] == len(scenarios) and upgraded["f1"] > baseline["f1"]
    predecessor.parent.write_json(output_path, {
        "schema": RESULT_SCHEMA,
        "authority": "FRESH_MULTISCENE_GEOMETRY_CONTROLLED_ROMA_CYCLE_NONE_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "receipt_path": receipt_path.name,
        "receipt_sha256": sha256(receipt_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "matcher": {"path": str(model_root.resolve()), "device": "CUDA", "model_calls": 16, "roma_weights_sha256": protocol["matcher"]["roma_weights_sha256"], "dinov2_weights_sha256": protocol["matcher"]["dinov2_weights_sha256"]},
        "conclusion": "L10_SCENENN_ROMA_ACTIVE_NONE_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_ROMA_ACTIVE_NONE_DEVELOPMENT_GATE_NOT_MET",
        "gate_met": gate_met,
        "metrics": {"aggregate": aggregates, "scenarios": scenarios, "full_dinov2_patch_mean_cosine_matrix": dino_scores.round(6).tolist(), "full_roma_supported_symmetric_cycle_matrix": roma_scores.round(6).tolist(), "pair_diagnostics": diagnostics, "true_pair_absolute_support": diagonal_supported},
        "incremental_gain_over_dinov2": {"true_positive_delta": upgraded["true_positive"] - baseline["true_positive"], "false_positive_reduction": baseline["false_positive"] - upgraded["false_positive"], "false_negative_reduction": baseline["false_negative"] - upgraded["false_negative"], "f1_delta": round(upgraded["f1"] - baseline["f1"], 6), "exact_scenario_gain": upgraded["zero_assignment_exact_scenarios"] - baseline["zero_assignment_exact_scenarios"]},
        "crop_receipts": crop_receipts,
        "claim_boundary": protocol["claim_boundary"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--admission", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--cohort", type=Path, required=True)
    seal_parser.add_argument("--source-root", type=Path, required=True)
    seal_parser.add_argument("--extraction-root", type=Path, required=True)
    seal_parser.add_argument("--extractor-exe", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--receipt", type=Path, required=True)
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.admission, args.source_root, args.output)
    elif args.action == "seal":
        seal(args.cohort, args.source_root, args.extraction_root, args.extractor_exe, args.output)
    else:
        replay(args.protocol, args.cohort, args.receipt, args.source_root, args.output)


if __name__ == "__main__":
    main()
