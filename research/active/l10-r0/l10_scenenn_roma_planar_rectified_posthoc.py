#!/usr/bin/env python3
"""Consumed SceneNN test of plane-rectified, visibility-aware RoMa support."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
import l10_scenenn_roma_active_none as roma_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-roma-planar-rectified-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-roma-planar-rectified-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return roma_base.sha256(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(protocol["implementation"]["sha256"] == sha256(Path(__file__)), "IMPLEMENTATION_HASH")
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    require(sha256(predecessor_path) == protocol["predecessor"]["result_sha256"], "PREDECESSOR_HASH")
    predecessor = load_json(predecessor_path)
    require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    receipt_path = HERE / protocol["source"]["rgb_receipt_path"]
    require(sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    require(sha256(receipt_path) == protocol["source"]["rgb_receipt_sha256"], "RECEIPT_HASH")
    model_root = ROOT / protocol["matcher"]["path"]
    require(sha256(model_root / "roma_indoor.pth") == protocol["matcher"]["roma_weights_sha256"], "ROMA_WEIGHTS")
    require(sha256(model_root / "dinov2_vitl14_pretrain.pth") == protocol["matcher"]["dinov2_weights_sha256"], "ROMA_DINOV2_WEIGHTS")
    return protocol


def intrinsic_matrix(values: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [float(values["fx"]), 0.0, float(values["cx"])],
            [0.0, float(values["fy"]), float(values["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_world(points: np.ndarray, camera_to_world: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    uvw = camera @ intrinsic.T
    pixels = uvw[:, :2] / uvw[:, 2:3]
    return pixels, depth


def stable_sign(vector: np.ndarray) -> np.ndarray:
    index = int(np.argmax(np.abs(vector)))
    return vector if vector[index] >= 0.0 else -vector


def planar_frame(points: np.ndarray) -> dict[str, Any]:
    center = np.mean(points.astype(np.float64), axis=0)
    centered = points.astype(np.float64) - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    axes = [stable_sign(vh[0]), stable_sign(vh[1])]
    coordinates = np.column_stack((centered @ axes[0], centered @ axes[1]))
    extents = np.ptp(coordinates, axis=0)
    vertical_index = int(np.argmax(extents))
    horizontal_index = 1 - vertical_index
    horizontal = axes[horizontal_index]
    vertical = axes[vertical_index]
    horizontal = stable_sign(horizontal)
    vertical = stable_sign(vertical)
    horizontal_coordinates = centered @ horizontal
    vertical_coordinates = centered @ vertical
    bounds = [
        float(np.min(horizontal_coordinates)),
        float(np.max(horizontal_coordinates)),
        float(np.min(vertical_coordinates)),
        float(np.max(vertical_coordinates)),
    ]
    require(bounds[1] > bounds[0] and bounds[3] > bounds[2], "PLANE_EXTENT")
    corners = np.asarray(
        [
            center + bounds[0] * horizontal + bounds[2] * vertical,
            center + bounds[1] * horizontal + bounds[2] * vertical,
            center + bounds[1] * horizontal + bounds[3] * vertical,
            center + bounds[0] * horizontal + bounds[3] * vertical,
        ],
        dtype=np.float64,
    )
    energy = np.square(singular)
    return {
        "center": center,
        "horizontal_axis": horizontal,
        "vertical_axis": vertical,
        "bounds_metres": bounds,
        "corners_world": corners,
        "singular_values": singular.tolist(),
        "planarity_residual_fraction": float(energy[-1] / max(float(np.sum(energy)), 1e-12)),
    }


def rectified_view(
    image: np.ndarray,
    visible_mask: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    plane: dict[str, Any],
    width: int,
    height: int,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    source_corners, depth = project_world(plane["corners_world"], camera_to_world, intrinsic)
    require(np.isfinite(source_corners).all() and bool(np.all(depth > 1e-6)), "RECTIFICATION_PROJECTION")
    destination_corners = np.asarray(
        [[0.0, float(height - 1)], [float(width - 1), float(height - 1)], [float(width - 1), 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_corners.astype(np.float32), destination_corners)
    rectified = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid = cv2.warpPerspective(
        visible_mask.astype(np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    valid_count = int(valid.sum())
    require(valid_count > 0, "RECTIFICATION_EMPTY")
    fill = np.rint(np.mean(rectified[valid], axis=0)).astype(np.uint8)
    rectified[~valid] = fill
    receipt = {
        "source_plane_corners_xy": source_corners.round(6).tolist(),
        "source_plane_corner_depths_m": depth.round(6).tolist(),
        "homography_source_to_rectified": homography.round(10).tolist(),
        "rectified_size": [width, height],
        "valid_target_pixels": valid_count,
        "valid_target_fraction": valid_count / float(width * height),
        "invalid_fill_bgr": fill.tolist(),
    }
    return Image.fromarray(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)), valid, receipt


def cycle_direction_masked(
    source_coords: torch.Tensor,
    forward_coords: torch.Tensor,
    forward_certainty: torch.Tensor,
    backward_coords: torch.Tensor,
    backward_certainty: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
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
    sampled_target_valid = F.grid_sample(
        target_valid.float()[None, None],
        forward_coords[None],
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0] >= 0.5
    cycle_error = torch.linalg.vector_norm(sampled_backward_coords - source_coords, dim=-1)
    eligible = source_valid & sampled_target_valid
    high = eligible & (forward_certainty >= float(matcher["official_certainty_threshold"]))
    cycle = high & (sampled_backward_certainty >= float(matcher["official_certainty_threshold"])) & (
        cycle_error <= float(matcher["maximum_cycle_error_normalized"])
    )
    eligible_count = int(eligible.sum().item())
    source_valid_count = int(source_valid.sum().item())
    high_count = int(high.sum().item())
    cycle_count = int(cycle.sum().item())
    return {
        "source_valid_count": source_valid_count,
        "eligible_count": eligible_count,
        "mean_certainty_on_eligible": float(forward_certainty[eligible].mean().item()) if eligible_count else 0.0,
        "high_count": high_count,
        "high_fraction_of_source_valid": high_count / source_valid_count if source_valid_count else 0.0,
        "cycle_count": cycle_count,
        "cycle_fraction": cycle_count / source_valid_count if source_valid_count else 0.0,
        "cycle_purity": cycle_count / high_count if high_count else 0.0,
    }


@torch.inference_mode()
def roma_support_masked(
    model: Any,
    reference: Image.Image,
    query: Image.Image,
    reference_valid_np: np.ndarray,
    query_valid_np: np.ndarray,
    matcher: dict[str, Any],
) -> dict[str, Any]:
    warp, certainty = model.match(reference, query)
    warp = warp[0]
    certainty = certainty[0]
    height, double_width = certainty.shape
    width = double_width // 2
    require(height == width == int(matcher["upsample_resolution"]), "ROMA_OUTPUT_RESOLUTION")
    reference_valid = F.interpolate(
        torch.from_numpy(reference_valid_np.astype(np.float32))[None, None].to(certainty.device),
        size=(height, width),
        mode="nearest",
    )[0, 0] >= 0.5
    query_valid = F.interpolate(
        torch.from_numpy(query_valid_np.astype(np.float32))[None, None].to(certainty.device),
        size=(height, width),
        mode="nearest",
    )[0, 0] >= 0.5
    forward = warp[:, :width]
    backward = warp[:, width:]
    a_to_b = cycle_direction_masked(
        forward[..., :2], forward[..., 2:], certainty[:, :width], backward[..., :2], certainty[:, width:],
        reference_valid, query_valid, matcher,
    )
    b_to_a = cycle_direction_masked(
        backward[..., 2:], backward[..., :2], certainty[:, width:], forward[..., 2:], certainty[:, :width],
        query_valid, reference_valid, matcher,
    )
    score = math.sqrt(float(a_to_b["cycle_fraction"]) * float(b_to_a["cycle_fraction"]))
    supported = all(
        float(direction["cycle_fraction"]) >= float(matcher["minimum_directional_cycle_fraction"])
        and float(direction["cycle_purity"]) >= float(matcher["minimum_directional_cycle_purity"])
        for direction in (a_to_b, b_to_a)
    )
    return {"a_to_b": a_to_b, "b_to_a": b_to_a, "symmetric_cycle_score": score, "absolute_support": supported}


def replay(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    from transformers import AutoImageProcessor, AutoModel
    import romatch

    protocol = load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    receipt_path = HERE / protocol["source"]["rgb_receipt_path"]
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    intrinsic_values = roma_base.predecessor.parent.base.parse_intrinsic(
        roma_base.predecessor.source_paths(source_root, cohort["episodes"][0]["scene_id"])["intrinsic"]
    )
    intrinsic = intrinsic_matrix(intrinsic_values)
    width = int(protocol["rectification"]["output_width"])
    height = int(protocol["rectification"]["output_height"])
    views: dict[str, Image.Image] = {}
    valid_masks: dict[str, np.ndarray] = {}
    rectification_receipts: dict[str, Any] = {}
    plane_receipts: dict[str, Any] = {}
    for episode in cohort["episodes"]:
        scene_id = str(episode["scene_id"])
        target_id = int(episode["target_instance_id"])
        paths = roma_base.predecessor.source_paths(source_root, scene_id)
        xyz, labels, faces = roma_base.predecessor.parent.visible.read_mesh(paths["ply"])
        target_points = xyz[labels == target_id]
        require(len(target_points) >= 4, f"TARGET_POINTS:{scene_id}")
        plane = planar_frame(target_points)
        plane_receipts[episode["episode_id"]] = {
            "scene_id": scene_id,
            "target_instance_id": target_id,
            "target_vertices": int(len(target_points)),
            "horizontal_axis_world": plane["horizontal_axis"].round(10).tolist(),
            "vertical_axis_world": plane["vertical_axis"].round(10).tolist(),
            "bounds_metres": plane["bounds_metres"],
            "singular_values": plane["singular_values"],
            "planarity_residual_fraction": plane["planarity_residual_fraction"],
        }
        poses = {int(row["frame"]): row["camera_to_world"] for row in roma_base.predecessor.parent.base.parse_poses(paths["trajectory"])}
        renderer = roma_base.predecessor.parent.visible.VisibilityRenderer(
            xyz, labels, faces, target_id, intrinsic_values, protocol["renderer"]
        )
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            sealed = receipt["sealed_frames"][f"{scene_id}:{frame}"]
            image_path = roma_base.predecessor.selected_image(source_root, scene_id, frame)
            require(sha256(image_path) == sealed["image_sha256"], f"IMAGE_HASH:{scene_id}:{frame}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            require(image is not None, f"IMAGE_READ:{scene_id}:{frame}")
            visible_device, _ = renderer.visible_mask(poses[frame])
            visible_mask = renderer.cp.asnumpy(visible_device)
            require(
                roma_base.predecessor.parent.visible.mask_sha256(visible_mask) == episode[role]["visible_mask_sha256"],
                f"MASK_HASH:{scene_id}:{role}",
            )
            view, valid, row_receipt = rectified_view(
                image, visible_mask, poses[frame], intrinsic, plane, width, height
            )
            key = f"{episode['episode_id']}:{role}"
            views[key] = view
            valid_masks[key] = valid
            rectification_receipts[key] = {"scene_id": scene_id, "frame": frame, **row_receipt}

    dino_root = ROOT / "artifacts.local/models/p1_a2_dinov2_small_ed25f3a"
    require(sha256(dino_root / "model.safetensors") == protocol["baseline"]["model_safetensors_sha256"], "DINO_WEIGHTS")
    dino_processor = AutoImageProcessor.from_pretrained(dino_root, local_files_only=True)
    dino_model = AutoModel.from_pretrained(dino_root, local_files_only=True).eval().to("cpu")
    ordered_keys = sorted(views)
    dino_inputs = dino_processor(images=[views[key] for key in ordered_keys], return_tensors="pt")
    with torch.inference_mode():
        dino_hidden = dino_model(**dino_inputs).last_hidden_state[:, 1:].mean(dim=1)
        dino_hidden = F.normalize(dino_hidden, dim=1)
    dino = {key: dino_hidden[index].cpu().numpy().astype(np.float32) for index, key in enumerate(ordered_keys)}
    del dino_model

    model_root = ROOT / protocol["matcher"]["path"]
    weights = torch.load(model_root / "roma_indoor.pth", map_location="cpu", weights_only=True)
    dinov2_weights = torch.load(model_root / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True)
    matcher_model = romatch.roma_indoor(
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
    count = len(episode_ids)
    dino_scores = np.zeros((count, count), dtype=np.float64)
    roma_scores = np.zeros((count, count), dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    for row, reference_id in enumerate(episode_ids):
        for column, query_id in enumerate(episode_ids):
            dino_scores[row, column] = float(dino[f"{reference_id}:reference"] @ dino[f"{query_id}:query"])
            diagnostic = roma_support_masked(
                matcher_model,
                views[f"{reference_id}:reference"],
                views[f"{query_id}:query"],
                valid_masks[f"{reference_id}:reference"],
                valid_masks[f"{query_id}:query"],
                protocol["matcher"],
            )
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
        dino_matches = roma_base.predecessor.parent.open_zero.reciprocal_zero_assignment(dino_matrix)
        roma_matches = roma_base.predecessor.parent.open_zero.reciprocal_zero_assignment(roma_matrix)
        scenarios.append(
            {
                **scenario,
                "rectified_dinov2_patch_mean_cosine_matrix": dino_matrix.round(6).tolist(),
                "rectified_roma_supported_symmetric_cycle_matrix": roma_matrix.round(6).tolist(),
                "methods": {
                    "rectified_dinov2_reciprocal_no_none_support": roma_base.predecessor.parent.open_zero.evaluate_matches(references, queries, dino_matches),
                    "rectified_roma_visibility_masked_cycle_reciprocal": roma_base.predecessor.parent.open_zero.evaluate_matches(references, queries, roma_matches),
                },
            }
        )
    aggregates = {name: roma_base.predecessor.parent.open_zero.aggregate(scenarios, name) for name in scenarios[0]["methods"]}
    baseline = aggregates["rectified_dinov2_reciprocal_no_none_support"]
    upgraded = aggregates["rectified_roma_visibility_masked_cycle_reciprocal"]
    diagonal_supported = [bool(diagnostics[f"{episode_id}->{episode_id}"]["absolute_support"]) for episode_id in episode_ids]
    gate = protocol["decision_gate"]
    gate_met = (
        sum(diagonal_supported) >= int(gate["minimum_true_pairs_with_absolute_support"])
        and upgraded["true_positive"] > int(gate["predecessor_true_positive"])
        and upgraded["false_positive"] == int(gate["maximum_false_positive"])
        and upgraded["f1"] > float(gate["predecessor_f1"])
        and upgraded["zero_assignment_exact_scenarios"] >= int(gate["minimum_exact_scenarios"])
    )
    predecessor_metrics = protocol["predecessor"]["observed_metrics"]
    roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_PLANE_RECTIFIED_VISIBILITY_MASKED_ROMA_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "source": {
                "cohort_path": cohort_path.name,
                "cohort_sha256": sha256(cohort_path),
                "rgb_receipt_path": receipt_path.name,
                "rgb_receipt_sha256": sha256(receipt_path),
                "source_root": str(source_root.resolve()),
            },
            "conclusion": "L10_SCENENN_ROMA_PLANAR_RECTIFIED_POSTHOC_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_ROMA_PLANAR_RECTIFIED_POSTHOC_DEVELOPMENT_GATE_NOT_MET",
            "gate_met": gate_met,
            "metrics": {
                "aggregate": aggregates,
                "scenarios": scenarios,
                "full_rectified_dinov2_matrix": dino_scores.round(6).tolist(),
                "full_rectified_roma_matrix": roma_scores.round(6).tolist(),
                "pair_diagnostics": diagnostics,
                "true_pair_absolute_support": diagonal_supported,
            },
            "gain_over_predecessor_roma": {
                "true_positive_delta": upgraded["true_positive"] - int(predecessor_metrics["true_positive"]),
                "false_positive_delta": upgraded["false_positive"] - int(predecessor_metrics["false_positive"]),
                "false_negative_reduction": int(predecessor_metrics["false_negative"]) - upgraded["false_negative"],
                "f1_delta": round(upgraded["f1"] - float(predecessor_metrics["f1"]), 6),
                "exact_scenario_gain": upgraded["zero_assignment_exact_scenarios"] - int(predecessor_metrics["exact_scenarios"]),
                "true_support_gain": sum(diagonal_supported) - int(predecessor_metrics["true_pairs_supported"]),
            },
            "rectification": {"planes": plane_receipts, "views": rectification_receipts},
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.source_root, args.output)


if __name__ == "__main__":
    main()
