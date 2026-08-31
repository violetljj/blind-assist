#!/usr/bin/env python3
"""Reference-conditioned RoMa-cycle prompts for SAM2 on 3RScan."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import zipfile
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
import l10_scenenn_roma_full_context_mask_posthoc as context_base  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as sam_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-sam-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-roma-cycle-prompt-sam-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return roma_base.sha256(path)


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes(order="C")).hexdigest()


def bbox_iou(a: list[float], b: list[float]) -> tuple[float, float, float]:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    require(area_a > 0.0 and area_b > 0.0 and union > 0.0, "INVALID_BBOX")
    return intersection / union, intersection / area_b, intersection / area_a


def rectangle_mask(size: tuple[int, int], bbox: list[float]) -> np.ndarray:
    width, height = size
    x0 = max(0, min(width, int(np.floor(float(bbox[0])))))
    y0 = max(0, min(height, int(np.floor(float(bbox[1])))))
    x1 = max(0, min(width, int(np.ceil(float(bbox[2])))))
    y1 = max(0, min(height, int(np.ceil(float(bbox[3])))))
    require(x1 > x0 and y1 > y0, "EMPTY_REFERENCE_BINDING")
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[y0:y1, x0:x1] = True
    return mask


def mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    require(len(xs) > 0, "EMPTY_MASK")
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(protocol["implementation"]["sha256"] == sha256(Path(__file__)), "IMPLEMENTATION_HASH")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    require(sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    require(sha256(predecessor_path) == protocol["predecessor"]["result_sha256"], "PREDECESSOR_HASH")
    require(
        load_json(predecessor_path)["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    for dependency in protocol["dependencies"]:
        require(sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    for model in ("masker", "matcher", "matcher_backbone"):
        row = protocol["models"][model]
        require(sha256(ROOT / row["path"]) == row["sha256"], f"MODEL_HASH:{model}")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["sequence_zips"]:
        source = artifact_root / row["path"]
        require(source.stat().st_size == int(row["bytes"]), f"ZIP_BYTES:{row['path']}")
        require(sha256(source) == row["sha256"], f"ZIP_HASH:{row['path']}")
    return protocol


def load_images(protocol: dict[str, Any], cohort: dict[str, Any]) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    images: dict[str, Image.Image] = {}
    receipts: dict[str, Any] = {}
    for row in cohort["images"].values():
        episode_id = str(row["episode_id"])
        role = str(row["role"])
        key = f"{episode_id}:{role}"
        manifest_key = f"{row['scan_id']}/sequence.zip"
        archive_path = artifact_root / cohort["source_manifest"][manifest_key]["path"]
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        require(list(image.size) == row["color_size"], f"IMAGE_SIZE:{key}")
        images[key] = image
        receipts[key] = {
            "scan_id": row["scan_id"],
            "frame": int(row["frame"]),
            "zip_member": row["zip_member"],
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "image_bytes": len(payload),
            "target_bbox_xyxy_evaluation_only": row["bbox_xyxy"],
        }
    require(len(images) == 6, "IMAGE_COUNT")
    return images, receipts


def largest_cycle_component(cycle: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    cycle_cpu = cycle.detach().cpu().numpy().astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cycle_cpu, connectivity=8)
    require(count > 1, "NO_CYCLE_COMPONENT")
    label = max(range(1, count), key=lambda value: (int(stats[value, cv2.CC_STAT_AREA]), -value))
    component = torch.from_numpy(labels == label)
    return component, {
        "component_count": count - 1,
        "selected_label": label,
        "selected_pixels": int(stats[label, cv2.CC_STAT_AREA]),
    }


def cycle_affine_prompt(
    warp: torch.Tensor,
    certainty: torch.Tensor,
    reference_mask: np.ndarray,
    query_size: tuple[int, int],
    matcher: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    height, double_width = certainty.shape
    width = double_width // 2
    require(height == width == int(matcher["upsample_resolution"]), "ROMA_OUTPUT_RESOLUTION")
    forward = warp[:, :width]
    backward = warp[:, width:]
    source_mask = context_base.resize_mask(reference_mask, width, certainty.device)
    source_coords = forward[..., :2]
    target_coords = forward[..., 2:]
    sampled_backward_coords = F.grid_sample(
        backward[..., :2].permute(2, 0, 1)[None],
        target_coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0)
    sampled_backward_certainty = F.grid_sample(
        certainty[:, width:][None, None],
        target_coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    cycle_error = torch.linalg.vector_norm(sampled_backward_coords - source_coords, dim=-1)
    threshold = float(matcher["official_certainty_threshold"])
    in_bounds = torch.all(torch.abs(target_coords) <= 1.0, dim=-1)
    high = source_mask & (certainty[:, :width] >= threshold)
    cycle = (
        high
        & in_bounds
        & (sampled_backward_certainty >= threshold)
        & (cycle_error <= float(matcher["maximum_cycle_error_normalized"]))
    )
    source_count = int(source_mask.sum().item())
    high_count = int(high.sum().item())
    cycle_count = int(cycle.sum().item())
    require(source_count > 0 and cycle_count > 0, "NO_REFERENCE_CYCLES")
    component, component_receipt = largest_cycle_component(cycle)
    component = component.to(device=warp.device)
    selected_source = source_coords[component].detach().cpu().numpy().astype(np.float64)
    selected_target = target_coords[component].detach().cpu().numpy().astype(np.float64)
    design = np.concatenate([selected_source, np.ones((len(selected_source), 1), dtype=np.float64)], axis=1)
    require(len(selected_source) >= 3 and np.linalg.matrix_rank(design) == 3, "AFFINE_SUPPORT_RANK")
    coefficients, _, _, _ = np.linalg.lstsq(design, selected_target, rcond=None)
    source_domain = source_coords[source_mask].detach().cpu().numpy().astype(np.float64)
    sx0, sy0 = np.min(source_domain, axis=0)
    sx1, sy1 = np.max(source_domain, axis=0)
    corners = np.asarray([[sx0, sy0, 1.0], [sx1, sy0, 1.0], [sx1, sy1, 1.0], [sx0, sy1, 1.0]])
    projected = corners @ coefficients
    query_width, query_height = query_size
    xs = np.clip((projected[:, 0] + 1.0) * query_width / 2.0, 0.0, float(query_width))
    ys = np.clip((projected[:, 1] + 1.0) * query_height / 2.0, 0.0, float(query_height))
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    require(box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0, "DEGENERATE_PROMPT_BOX")
    predicted = design @ coefficients
    residual = np.linalg.norm(predicted - selected_target, axis=1)
    return box, {
        "selection_authority": "REFERENCE_BBOX_ONLY_TO_LARGEST_8_CONNECTED_BIDIRECTIONAL_CYCLE_COMPONENT",
        "source_mask_pixels_at_match_resolution": source_count,
        "high_certainty_pixels": high_count,
        "all_cycle_pixels": cycle_count,
        "all_cycle_fraction": cycle_count / source_count,
        "all_cycle_purity": cycle_count / high_count if high_count else 0.0,
        "component": component_receipt,
        "selected_component_fraction_of_source": len(selected_source) / source_count,
        "selected_component_fraction_of_cycles": len(selected_source) / cycle_count,
        "affine_rank": int(np.linalg.matrix_rank(design)),
        "affine_mean_residual_normalized": float(np.mean(residual)),
        "affine_max_residual_normalized": float(np.max(residual)),
        "projected_reference_corners_normalized": projected.tolist(),
        "prompt_box_xyxy": box,
    }


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch
    from transformers import Sam2Model, Sam2Processor

    protocol = load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = load_json(cohort_path)
    images, input_receipts = load_images(protocol, cohort)
    reference_masks: dict[str, np.ndarray] = {}
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        reference_masks[episode_id] = rectangle_mask(
            images[f"{episode_id}:reference"].size,
            input_receipts[f"{episode_id}:reference"]["target_bbox_xyxy_evaluation_only"],
        )

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
    cached_matches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    prompt_receipts: dict[str, Any] = {}
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        reference = images[f"{episode_id}:reference"]
        query = images[f"{episode_id}:query"]
        with torch.inference_mode():
            warp_batch, certainty_batch = matcher_model.match(reference, query)
        warp = warp_batch[0].detach().cpu()
        certainty = certainty_batch[0].detach().cpu()
        prompt_box, receipt = cycle_affine_prompt(
            warp, certainty, reference_masks[episode_id], query.size, protocol["matcher"]
        )
        target_box = input_receipts[f"{episode_id}:query"]["target_bbox_xyxy_evaluation_only"]
        iou, recall, precision = bbox_iou(prompt_box, target_box)
        receipt.update(
            {
                "target_bbox_iou_evaluation_only": iou,
                "target_bbox_recall_evaluation_only": recall,
                "prompt_bbox_precision_evaluation_only": precision,
            }
        )
        cached_matches[episode_id] = (warp, certainty)
        prompt_receipts[episode_id] = receipt
    matcher_device = torch.cuda.get_device_name(0)
    del matcher_model, weights, dinov2_weights
    gc.collect()
    torch.cuda.empty_cache()

    sam_root = (ROOT / protocol["proposal"]["masker_root"]).resolve()
    sam_processor = Sam2Processor.from_pretrained(sam_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        sam_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    proposal_masks: dict[str, np.ndarray] = {}
    proposal_receipts: dict[str, Any] = {}
    pair_support: dict[str, Any] = {}
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        query = images[f"{episode_id}:query"]
        prompt_box = prompt_receipts[episode_id]["prompt_box_xyxy"]
        masks, sam_receipt = sam_base._sam_masks(
            sam_processor, sam_model, query, [prompt_box], query.size, torch, np
        )
        require(len(masks) == 1, f"SAM_MASK_COUNT:{episode_id}")
        mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
        require(int(mask.sum()) > 0, f"EMPTY_SAM_MASK:{episode_id}")
        proposal_box = mask_bbox(mask)
        target_box = input_receipts[f"{episode_id}:query"]["target_bbox_xyxy_evaluation_only"]
        iou, recall, precision = bbox_iou(proposal_box, target_box)
        proposal_masks[episode_id] = mask
        proposal_receipts[episode_id] = {
            "selection_authority": "ONE_NATIVE_SAM2_MASK_FROM_REFERENCE_CONDITIONED_CYCLE_AFFINE_BOX",
            "mask_sha256": mask_sha256(mask),
            "mask_pixels": int(mask.sum()),
            "mask_bbox_xyxy": proposal_box,
            "target_bbox_iou_evaluation_only": iou,
            "target_bbox_recall_evaluation_only": recall,
            "proposal_bbox_precision_evaluation_only": precision,
            "masker": sam_receipt,
        }
        warp, certainty = cached_matches[episode_id]
        width = certainty.shape[1] // 2
        reference_domain = context_base.resize_mask(reference_masks[episode_id], width, certainty.device)
        query_domain = context_base.resize_mask(mask, width, certainty.device)
        forward = warp[:, :width]
        backward = warp[:, width:]
        a_to_b = context_base.masked_cycle_direction(
            forward[..., :2], forward[..., 2:], certainty[:, :width],
            backward[..., :2], certainty[:, width:], reference_domain, query_domain,
            protocol["matcher"],
        )
        b_to_a = context_base.masked_cycle_direction(
            backward[..., 2:], backward[..., :2], certainty[:, width:],
            forward[..., 2:], certainty[:, :width], query_domain, reference_domain,
            protocol["matcher"],
        )
        supported = all(
            float(row["cycle_fraction"]) >= float(protocol["matcher"]["minimum_directional_cycle_fraction"])
            and float(row["cycle_purity"]) >= float(protocol["matcher"]["minimum_directional_cycle_purity"])
            for row in (a_to_b, b_to_a)
        )
        pair_support[episode_id] = {"a_to_b": a_to_b, "b_to_a": b_to_a, "absolute_support": supported}

    gate = protocol["decision_gate"]
    minimum_prompt_iou = min(float(row["target_bbox_iou_evaluation_only"]) for row in prompt_receipts.values())
    minimum_mask_iou = min(float(row["target_bbox_iou_evaluation_only"]) for row in proposal_receipts.values())
    supported_pairs = sum(bool(row["absolute_support"]) for row in pair_support.values())
    gate_met = (
        len(prompt_receipts) == int(gate["required_prompt_boxes"])
        and len(proposal_masks) == int(gate["required_sam_masks"])
        and minimum_prompt_iou >= float(gate["minimum_prompt_target_bbox_iou"])
        and minimum_mask_iou >= float(gate["minimum_sam_target_bbox_iou"])
        and supported_pairs >= int(gate["minimum_true_pairs_with_absolute_support"])
    )
    roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_REFERENCE_CONDITIONED_MULTI_DOOR_PROPOSAL_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "source": {"cohort_path": cohort_path.name, "cohort_sha256": sha256(cohort_path)},
            "conclusion": (
                "L10_3RSCAN_ROMA_CYCLE_PROMPT_SAM_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met else "L10_3RSCAN_ROMA_CYCLE_PROMPT_SAM_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "prompt_boxes": len(prompt_receipts),
                "sam_masks": len(proposal_masks),
                "minimum_prompt_target_bbox_iou": minimum_prompt_iou,
                "minimum_sam_target_bbox_iou": minimum_mask_iou,
                "true_pairs_with_absolute_support": supported_pairs,
                "required_true_pairs": len(cohort["episodes"]),
            },
            "prompt_receipts": prompt_receipts,
            "proposal_receipts": proposal_receipts,
            "pair_support": pair_support,
            "input_receipts": input_receipts,
            "runtime": {
                "device": matcher_device,
                "roma_calls": len(cohort["episodes"]),
                "sam2_calls": len(cohort["episodes"]),
                "grounding_dino_calls": 0,
            },
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
