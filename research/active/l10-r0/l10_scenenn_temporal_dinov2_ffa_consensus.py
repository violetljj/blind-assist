#!/usr/bin/env python3
"""Frozen DINOv2 foreground-feature consensus on consumed SceneNN temporal masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_observed_extent_support as parent  # noqa: E402
import l10_scenenn_temporal_fused_extent_support as temporal  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-temporal-dinov2-ffa-consensus-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-temporal-dinov2-ffa-consensus-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return parent.sha256(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    for member in ("cohort", "receipt", "result"):
        dependency = HERE / protocol["predecessor"][f"{member}_path"]
        require(sha256(dependency) == protocol["predecessor"][f"{member}_sha256"], f"PREDECESSOR_HASH:{member}")
    predecessor = load_json(HERE / protocol["predecessor"]["result_path"])
    require(predecessor.get("conclusion") == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    model_root = ROOT / protocol["model"]["path"]
    require(sha256(model_root / "model.safetensors") == protocol["model"]["model_safetensors_sha256"], "MODEL_HASH")
    return protocol


def square_target_crop(image: np.ndarray, mask: np.ndarray) -> tuple[Image.Image, np.ndarray, list[int]]:
    ys, xs = np.where(mask)
    require(len(xs) > 0, "EMPTY_VISIBLE_MASK")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb[~mask] = 0
    crop = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1].astype(np.float32)
    side = max(crop.shape[0], crop.shape[1])
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    mask_canvas = np.zeros((side, side), dtype=np.float32)
    top = (side - crop.shape[0]) // 2
    left = (side - crop.shape[1]) // 2
    canvas[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
    mask_canvas[top : top + crop.shape[0], left : left + crop.shape[1]] = crop_mask
    return Image.fromarray(canvas), mask_canvas, [x0, y0, x1, y1]


def processed_patch_weights(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    centered = resized[16:240, 16:240]
    require(centered.shape == (224, 224), "MASK_CENTER_CROP")
    weights = centered.reshape(16, 14, 16, 14).mean(axis=(1, 3)).reshape(-1).astype(np.float32)
    require(float(weights.sum()) > 0.0, "EMPTY_PATCH_WEIGHTS")
    return weights


def encode_ffa(images: list[Image.Image], weights: list[np.ndarray], model_root: Path) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(model_root, local_files_only=True).eval().to("cpu")
    batch = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state[:, 1:]
    require(hidden.shape[1] == 256, f"DINO_PATCH_COUNT:{hidden.shape[1]}")
    weight_tensor = torch.from_numpy(np.stack(weights)).to(dtype=hidden.dtype)
    pooled = (hidden * weight_tensor[:, :, None]).sum(dim=1) / weight_tensor.sum(dim=1, keepdim=True)
    pooled = torch.nn.functional.normalize(pooled, dim=1)
    return pooled.cpu().numpy().astype(np.float32)


def replay(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    predecessor = protocol["predecessor"]
    cohort_path = HERE / predecessor["cohort_path"]
    receipt_path = HERE / predecessor["receipt_path"]
    predecessor_result_path = HERE / predecessor["result_path"]
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    predecessor_result = load_json(predecessor_result_path)
    scene_id = str(cohort["scene_id"])
    paths = parent.source_paths(source_root, scene_id)
    intrinsic = parent.base.parse_intrinsic(paths["intrinsic"])
    xyz, labels, faces = parent.visible.read_mesh(paths["ply"])
    poses = {int(row["frame"]): row["camera_to_world"] for row in parent.base.parse_poses(paths["trajectory"])}
    renderer_protocol = load_json(HERE / "l10_scenenn_temporal_fused_extent_support_protocol_v1.json")["renderer"]
    images: list[Image.Image] = []
    weights: list[np.ndarray] = []
    keys: list[str] = []
    crop_receipts: dict[str, Any] = {}
    for episode in cohort["episodes"]:
        renderer = parent.visible.VisibilityRenderer(xyz, labels, faces, int(episode["target_instance_id"]), intrinsic, renderer_protocol)
        for role in ("reference", "query"):
            for frozen_frame in episode[role]["window"]:
                frame = int(frozen_frame["trajectory_frame"])
                image_path = temporal.temporal_path(source_root, scene_id, "image", frame)
                sealed = receipt["sealed_frames"][str(frame)]
                require(sha256(image_path) == sealed["image_sha256"], f"IMAGE_HASH:{frame}")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                require(image is not None and image.shape == (480, 640, 3), f"IMAGE_FORMAT:{frame}")
                visible_device, _ = renderer.visible_mask(poses[frame])
                visible_mask = renderer.cp.asnumpy(visible_device)
                require(parent.visible.mask_sha256(visible_mask) == frozen_frame["visible_mask_sha256"], f"VISIBLE_MASK_HASH:{episode['episode_id']}:{role}:{frame}")
                crop, mask, bbox = square_target_crop(image, visible_mask)
                key = f"{episode['episode_id']}:{role}:{frame}"
                images.append(crop)
                weights.append(processed_patch_weights(mask))
                keys.append(key)
                crop_receipts[key] = {"source_image_sha256": sealed["image_sha256"], "visible_mask_sha256": frozen_frame["visible_mask_sha256"], "bbox_xyxy": bbox, "square_side": crop.width, "foreground_patch_mass": round(float(weights[-1].sum()), 6)}
    encoded_array = encode_ffa(images, weights, ROOT / protocol["model"]["path"])
    encoded = {key: encoded_array[index] for index, key in enumerate(keys)}
    episode_ids = [row["episode_id"] for row in cohort["episodes"]]
    appearance = np.zeros((len(episode_ids), len(episode_ids)), dtype=np.float64)
    for row, reference in enumerate(cohort["episodes"]):
        reference_keys = [f"{reference['episode_id']}:reference:{int(value['trajectory_frame'])}" for value in reference["reference"]["window"]]
        for column, query in enumerate(cohort["episodes"]):
            query_keys = [f"{query['episode_id']}:query:{int(value['trajectory_frame'])}" for value in query["query"]["window"]]
            appearance[row, column] = max(float(encoded[left] @ encoded[right]) for left in reference_keys for right in query_keys)
    geometry = np.asarray(predecessor_result["metrics"]["full_temporal_surface_score_matrix"], dtype=np.float64)
    target_index = {value: index for index, value in enumerate(episode_ids)}
    scenarios: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[value] for value in references]
        columns = [target_index[value] for value in queries]
        geometry_scores = geometry[np.ix_(rows, columns)]
        appearance_scores = appearance[np.ix_(rows, columns)]
        geometry_matches = parent.open_zero.reciprocal_zero_assignment(geometry_scores)
        appearance_matches = parent.open_zero.reciprocal_zero_assignment(appearance_scores)
        appearance_set = set(appearance_matches)
        consensus_matches = [pair for pair in geometry_matches if pair in appearance_set]
        scenarios.append({
            **scenario,
            "geometry_score_matrix": geometry_scores.round(6).tolist(),
            "dinov2_ffa_max_view_cosine_matrix": appearance_scores.round(6).tolist(),
            "methods": {
                "geometry_reciprocal": parent.open_zero.evaluate_matches(references, queries, geometry_matches),
                "dinov2_ffa_reciprocal": parent.open_zero.evaluate_matches(references, queries, appearance_matches),
                "dinov2_ffa_geometry_consensus": parent.open_zero.evaluate_matches(references, queries, consensus_matches),
            },
        })
    aggregates = {name: parent.open_zero.aggregate(scenarios, name) for name in scenarios[0]["methods"]}
    baseline = aggregates["geometry_reciprocal"]
    consensus = aggregates["dinov2_ffa_geometry_consensus"]
    expected_true = int(cohort["counts"]["truth_matches_across_scenarios"])
    gate_met = consensus["true_positive"] == expected_true and consensus["false_positive"] == 0 and consensus["false_negative"] == 0 and consensus["zero_assignment_exact_scenarios"] == len(scenarios) and consensus["true_positive"] >= baseline["true_positive"] and consensus["f1"] >= baseline["f1"]
    parent.write_json(output_path, {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SCENE_POSTHOC_DINOV2_FFA_GEOMETRY_CONSENSUS_MECHANISM_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "receipt_path": receipt_path.name,
        "receipt_sha256": sha256(receipt_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "model": {"path": str((ROOT / protocol["model"]["path"]).resolve()), "model_safetensors_sha256": protocol["model"]["model_safetensors_sha256"], "device": "CPU"},
        "conclusion": "L10_SCENENN_TEMPORAL_DINOV2_FFA_GEOMETRY_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_TEMPORAL_DINOV2_FFA_GEOMETRY_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_NOT_MET",
        "gate_met": gate_met,
        "metrics": {"aggregate": aggregates, "scenarios": scenarios, "full_geometry_score_matrix": geometry.round(6).tolist(), "full_dinov2_ffa_max_view_cosine_matrix": appearance.round(6).tolist()},
        "crop_receipts": crop_receipts,
        "incremental_gain_over_geometry": {"true_positive_delta": consensus["true_positive"] - baseline["true_positive"], "false_positive_reduction": baseline["false_positive"] - consensus["false_positive"], "false_negative_reduction": baseline["false_negative"] - consensus["false_negative"], "f1_delta": round(consensus["f1"] - baseline["f1"], 6), "exact_scenario_gain": consensus["zero_assignment_exact_scenarios"] - baseline["zero_assignment_exact_scenarios"]},
        "claim_boundary": protocol["claim_boundary"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", choices=["replay"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.source_root, args.output)


if __name__ == "__main__":
    main()
