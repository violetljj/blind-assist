#!/usr/bin/env python3
"""Consumed SceneNN full-context, target-mask-gated RoMa challenger."""

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


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-roma-full-context-mask-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-roma-full-context-mask-posthoc-result-v1"


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
    require(
        load_json(predecessor_path)["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    cohort_path = HERE / protocol["source"]["cohort_path"]
    receipt_path = HERE / protocol["source"]["rgb_receipt_path"]
    require(sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    require(sha256(receipt_path) == protocol["source"]["rgb_receipt_sha256"], "RGB_RECEIPT_HASH")
    model_root = ROOT / protocol["matcher"]["path"]
    require(sha256(model_root / "roma_indoor.pth") == protocol["matcher"]["roma_weights_sha256"], "ROMA_WEIGHTS")
    require(
        sha256(model_root / "dinov2_vitl14_pretrain.pth") == protocol["matcher"]["dinov2_weights_sha256"],
        "ROMA_DINOV2_WEIGHTS",
    )
    return protocol


def resize_mask(mask: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
    return F.interpolate(tensor, size=(size, size), mode="nearest")[0, 0] >= 0.5


def masked_cycle_direction(
    source_coords: torch.Tensor,
    forward_coords: torch.Tensor,
    forward_certainty: torch.Tensor,
    backward_coords: torch.Tensor,
    backward_certainty: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
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
    target_at_forward = F.grid_sample(
        target_mask.float()[None, None],
        forward_coords[None],
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0] >= 0.5
    cycle_error = torch.linalg.vector_norm(sampled_backward_coords - source_coords, dim=-1)
    high = source_mask & (forward_certainty >= float(matcher["official_certainty_threshold"]))
    cycle = (
        high
        & target_at_forward
        & (sampled_backward_certainty >= float(matcher["official_certainty_threshold"]))
        & (cycle_error <= float(matcher["maximum_cycle_error_normalized"]))
    )
    source_count = int(source_mask.sum().item())
    high_count = int(high.sum().item())
    target_hit_count = int((high & target_at_forward).sum().item())
    cycle_count = int(cycle.sum().item())
    require(source_count > 0, "EMPTY_SOURCE_MASK")
    return {
        "mean_certainty_on_source_mask": float(forward_certainty[source_mask].mean().item()),
        "source_mask_count": source_count,
        "high_count": high_count,
        "high_fraction": high_count / source_count,
        "target_mask_hit_count": target_hit_count,
        "target_mask_hit_fraction": target_hit_count / source_count,
        "cycle_count": cycle_count,
        "cycle_fraction": cycle_count / source_count,
        "cycle_purity": cycle_count / high_count if high_count else 0.0,
    }


@torch.inference_mode()
def masked_roma_support(
    model: Any,
    reference: Image.Image,
    query: Image.Image,
    reference_mask: np.ndarray,
    query_mask: np.ndarray,
    matcher: dict[str, Any],
) -> dict[str, Any]:
    warp, certainty = model.match(reference, query)
    warp = warp[0]
    certainty = certainty[0]
    height, double_width = certainty.shape
    width = double_width // 2
    size = int(matcher["upsample_resolution"])
    require(height == width == size, "ROMA_OUTPUT_RESOLUTION")
    device = certainty.device
    reference_domain = resize_mask(reference_mask, size, device)
    query_domain = resize_mask(query_mask, size, device)
    forward = warp[:, :width]
    backward = warp[:, width:]
    a_to_b = masked_cycle_direction(
        forward[..., :2],
        forward[..., 2:],
        certainty[:, :width],
        backward[..., :2],
        certainty[:, width:],
        reference_domain,
        query_domain,
        matcher,
    )
    b_to_a = masked_cycle_direction(
        backward[..., 2:],
        backward[..., :2],
        certainty[:, width:],
        forward[..., 2:],
        certainty[:, :width],
        query_domain,
        reference_domain,
        matcher,
    )
    score = math.sqrt(float(a_to_b["cycle_fraction"]) * float(b_to_a["cycle_fraction"]))
    supported = all(
        float(direction["cycle_fraction"]) >= float(matcher["minimum_directional_cycle_fraction"])
        and float(direction["cycle_purity"]) >= float(matcher["minimum_directional_cycle_purity"])
        for direction in (a_to_b, b_to_a)
    )
    return {
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "symmetric_cycle_score": score,
        "absolute_support": supported,
    }


def replay(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    import romatch

    protocol = load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    receipt_path = HERE / protocol["source"]["rgb_receipt_path"]
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    predecessor_result = load_json(predecessor_path)
    intrinsic = roma_base.predecessor.parent.base.parse_intrinsic(
        roma_base.predecessor.source_paths(source_root, cohort["episodes"][0]["scene_id"])["intrinsic"]
    )
    images: dict[str, Image.Image] = {}
    masks: dict[str, np.ndarray] = {}
    input_receipts: dict[str, Any] = {}
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        scene_id = str(episode["scene_id"])
        paths = roma_base.predecessor.source_paths(source_root, scene_id)
        xyz, labels, faces = roma_base.predecessor.parent.visible.read_mesh(paths["ply"])
        poses = {
            int(row["frame"]): row["camera_to_world"]
            for row in roma_base.predecessor.parent.base.parse_poses(paths["trajectory"])
        }
        renderer = roma_base.predecessor.parent.visible.VisibilityRenderer(
            xyz,
            labels,
            faces,
            int(episode["target_instance_id"]),
            intrinsic,
            protocol["renderer"],
        )
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            sealed = receipt["sealed_frames"][f"{scene_id}:{frame}"]
            image_path = roma_base.predecessor.selected_image(source_root, scene_id, frame)
            require(sha256(image_path) == sealed["image_sha256"], f"IMAGE_HASH:{scene_id}:{frame}")
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            require(bgr is not None and bgr.shape == (480, 640, 3), f"IMAGE_FORMAT:{scene_id}:{frame}")
            visible_device, _ = renderer.visible_mask(poses[frame])
            visible_mask = renderer.cp.asnumpy(visible_device)
            mask_hash = roma_base.predecessor.parent.visible.mask_sha256(visible_mask)
            require(mask_hash == episode[role]["visible_mask_sha256"], f"MASK_HASH:{scene_id}:{role}")
            key = f"{episode_id}:{role}"
            images[key] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            masks[key] = visible_mask
            input_receipts[key] = {
                "scene_id": scene_id,
                "frame": frame,
                "image_sha256": sealed["image_sha256"],
                "visible_mask_sha256": mask_hash,
                "image_size": [640, 480],
                "visible_target_pixels": int(visible_mask.sum()),
                "tight_bbox_xyxy": episode[role]["visible_mask_tight_bbox_xyxy"],
            }

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
    scores = np.zeros((count, count), dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    for row, reference_id in enumerate(episode_ids):
        for column, query_id in enumerate(episode_ids):
            support = masked_roma_support(
                matcher_model,
                images[f"{reference_id}:reference"],
                images[f"{query_id}:query"],
                masks[f"{reference_id}:reference"],
                masks[f"{query_id}:query"],
                protocol["matcher"],
            )
            diagnostics[f"{reference_id}->{query_id}"] = support
            if support["absolute_support"]:
                scores[row, column] = float(support["symmetric_cycle_score"])

    target_index = {value: index for index, value in enumerate(episode_ids)}
    scenarios: list[dict[str, Any]] = []
    for scenario in cohort["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        rows = [target_index[value] for value in references]
        columns = [target_index[value] for value in queries]
        matrix = scores[np.ix_(rows, columns)]
        matches = roma_base.predecessor.parent.open_zero.reciprocal_zero_assignment(matrix)
        scenarios.append(
            {
                **scenario,
                "full_context_mask_gated_roma_matrix": matrix.round(6).tolist(),
                "methods": {
                    "full_context_mask_gated_roma_reciprocal": roma_base.predecessor.parent.open_zero.evaluate_matches(
                        references, queries, matches
                    )
                },
            }
        )
    upgraded = roma_base.predecessor.parent.open_zero.aggregate(
        scenarios, "full_context_mask_gated_roma_reciprocal"
    )
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
            "authority": "CONSUMED_POSTHOC_FULL_CONTEXT_MASK_GATED_ROMA_DEVELOPMENT_RESULT",
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
            "conclusion": (
                "L10_SCENENN_ROMA_FULL_CONTEXT_MASK_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met
                else "L10_SCENENN_ROMA_FULL_CONTEXT_MASK_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "aggregate": upgraded,
                "scenarios": scenarios,
                "full_matrix": scores.round(6).tolist(),
                "pair_diagnostics": diagnostics,
                "true_pair_absolute_support": diagonal_supported,
            },
            "gain_over_predecessor_roma": {
                "true_positive_delta": upgraded["true_positive"] - int(predecessor_metrics["true_positive"]),
                "false_positive_delta": upgraded["false_positive"] - int(predecessor_metrics["false_positive"]),
                "false_negative_reduction": int(predecessor_metrics["false_negative"]) - upgraded["false_negative"],
                "f1_delta": round(upgraded["f1"] - float(predecessor_metrics["f1"]), 6),
                "exact_scenario_gain": upgraded["zero_assignment_exact_scenarios"]
                - int(predecessor_metrics["exact_scenarios"]),
                "true_support_gain": sum(diagonal_supported) - int(predecessor_metrics["true_pairs_supported"]),
            },
            "predecessor_dinov2_baseline": predecessor_result["metrics"]["aggregate"][
                "dinov2_reciprocal_no_none_support"
            ],
            "input_receipts": input_receipts,
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
