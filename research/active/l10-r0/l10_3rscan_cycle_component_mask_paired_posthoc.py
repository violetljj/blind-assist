#!/usr/bin/env python3
"""Require bilateral coherent cycles between both identity-support masks."""

from __future__ import annotations

import argparse
import gc
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cycle_component_open_set_posthoc as open_set  # noqa: E402
import l10_3rscan_cycle_component_sibling_door_posthoc as sibling  # noqa: E402
import l10_3rscan_roma_cycle_prompt_dual_surface_posthoc as dual  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc as reference_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cycle-component-mask-paired-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-cycle-component-mask-paired-posthoc-result-v1"


@contextmanager
def protocol_surface():
    base = open_set.base
    saved_schema = base.PROTOCOL_SCHEMA
    saved_file = base.__file__
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.__file__ = str(Path(__file__).resolve())
    try:
        yield
    finally:
        base.PROTOCOL_SCHEMA = saved_schema
        base.__file__ = saved_file


def verify_sibling_absence(protocol_path: Path) -> dict[str, Any]:
    saved_schema = sibling.PROTOCOL_SCHEMA
    sibling.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    try:
        return sibling.verify_sibling_absence(protocol_path)
    finally:
        sibling.PROTOCOL_SCHEMA = saved_schema


def paired_cycle_component(
    source_coords: torch.Tensor,
    forward_coords: torch.Tensor,
    forward_certainty: torch.Tensor,
    backward_coords: torch.Tensor,
    backward_certainty: torch.Tensor,
    source_mask_native: np.ndarray,
    target_mask_native: np.ndarray,
    matcher: dict[str, Any],
) -> dict[str, Any]:
    base = open_set.base
    width = int(forward_certainty.shape[1])
    source_mask = base.context_base.resize_mask(source_mask_native, width, forward_certainty.device)
    target_mask = base.context_base.resize_mask(target_mask_native, width, forward_certainty.device)
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
    threshold = float(matcher["official_certainty_threshold"])
    high = source_mask & (forward_certainty >= threshold)
    cycle = (
        high
        & target_at_forward
        & (sampled_backward_certainty >= threshold)
        & (cycle_error <= float(matcher["maximum_cycle_error_normalized"]))
    )
    source_count = int(source_mask.sum().item())
    high_count = int(high.sum().item())
    target_hit_count = int((high & target_at_forward).sum().item())
    cycle_count = int(cycle.sum().item())
    base.require(source_count > 0, "EMPTY_SOURCE_MASK")
    if cycle_count == 0:
        component_receipt = {"component_count": 0, "selected_label": None, "selected_pixels": 0}
        dominance = 0.0
    else:
        _, component_receipt = base.largest_cycle_component(cycle)
        dominance = int(component_receipt["selected_pixels"]) / cycle_count
    return {
        "source_mask_pixels_at_match_resolution": source_count,
        "high_certainty_pixels": high_count,
        "target_mask_hit_pixels": target_hit_count,
        "target_mask_hit_fraction": target_hit_count / source_count,
        "paired_cycle_pixels": cycle_count,
        "paired_cycle_fraction": cycle_count / source_count,
        "component": component_receipt,
        "dominant_component_fraction_of_paired_cycles": dominance,
    }


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch
    from transformers import Sam2Model, Sam2Processor

    base = open_set.base
    absence_receipts = verify_sibling_absence(protocol_path)
    with protocol_surface():
        protocol = base.load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = base.load_json(cohort_path)
    images, inputs = base.load_images(protocol, cohort)
    episodes = {str(row["episode_id"]): row for row in cohort["episodes"]}
    reference_masks, reference_receipts, device_name = reference_base.make_reference_masks(
        protocol, cohort, images, inputs
    )
    reference_mask_by_episode: dict[str, np.ndarray] = {}
    extent_masks: dict[str, np.ndarray] = {}
    for episode_id in episodes:
        image = images[f"{episode_id}:reference"]
        bbox = inputs[f"{episode_id}:reference"]["target_bbox_xyxy_evaluation_only"]
        lookup = (int(image.size[0]), int(image.size[1]), *(float(value) for value in bbox))
        mask = reference_masks[lookup]
        reference_mask_by_episode[episode_id] = mask
        extent_masks[base.mask_sha256(mask)] = base.rectangle_mask(image.size, bbox)

    model_root = ROOT / protocol["matcher"]["path"]
    weights = torch.load(model_root / "roma_indoor.pth", map_location="cpu", weights_only=True)
    dinov2_weights = torch.load(
        model_root / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True
    )
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
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        reference_id = str(pair["reference_episode"])
        query_id = str(pair["query_episode"])
        with torch.inference_mode():
            warp_batch, certainty_batch = matcher_model.match(
                images[f"{reference_id}:reference"], images[f"{query_id}:query"]
            )
        warp = warp_batch[0].detach().cpu()
        certainty = certainty_batch[0].detach().cpu()
        cached_matches[pair_id] = (warp, certainty)
        try:
            prompt_box, receipt = dual.dual_surface_cycle_affine_prompt(
                extent_masks,
                warp,
                certainty,
                reference_mask_by_episode[reference_id],
                images[f"{query_id}:query"].size,
                protocol["matcher"],
            )
        except ValueError as exc:
            if str(exc) != "NO_REFERENCE_CYCLES" or pair["label"] != "target_absent":
                raise
            prompt_box = None
            receipt = {
                "selection_authority": "ZERO_REFERENCE_CYCLES_DETERMINISTIC_NON_COMMIT",
                "all_cycle_pixels": 0,
                "all_cycle_fraction": 0.0,
                "selected_component_fraction_of_cycles": 0.0,
                "prompt_box_xyxy": None,
            }
        if pair["label"] == "target_present":
            target_box = inputs[f"{query_id}:query"]["target_bbox_xyxy_evaluation_only"]
            receipt["target_bbox_iou_evaluation_only"] = base.bbox_iou(prompt_box, target_box)[0]
        else:
            receipt["target_bbox_iou_evaluation_only"] = None
            receipt["target_absence_authority"] = "SAME_SCENE_EXACT_TARGET_ZERO_PROJECTED_VERTICES"
        prompt_receipts[pair_id] = receipt
    del matcher_model, weights, dinov2_weights
    gc.collect()
    torch.cuda.empty_cache()

    sam_root = (ROOT / protocol["proposal"]["masker_root"]).resolve()
    processor = Sam2Processor.from_pretrained(sam_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        sam_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    query_support_receipts: dict[str, Any] = {}
    bilateral_receipts: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    gate = protocol["decision_gate"]
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        reference_id = str(pair["reference_episode"])
        query_id = str(pair["query_episode"])
        prompt = prompt_receipts[pair_id]
        if prompt["prompt_box_xyxy"] is None:
            query_support_receipts[pair_id] = {"status": "NOT_RUN_ZERO_REFERENCE_CYCLES"}
            bilateral_receipts[pair_id] = {
                "reference_to_query": {"paired_cycle_fraction": 0.0, "dominant_component_fraction_of_paired_cycles": 0.0},
                "query_to_reference": {"paired_cycle_fraction": 0.0, "dominant_component_fraction_of_paired_cycles": 0.0},
                "bilateral_mask_paired_support": False,
            }
        else:
            query = images[f"{query_id}:query"]
            masks, masker = base.sam_base._sam_masks(
                processor, model, query, [prompt["prompt_box_xyxy"]], query.size, torch, np
            )
            base.require(len(masks) == 1, f"QUERY_SAM_MASK_COUNT:{pair_id}")
            query_mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
            base.require(int(query_mask.sum()) > 0, f"EMPTY_QUERY_SAM_MASK:{pair_id}")
            query_support_receipts[pair_id] = {
                "mask_sha256": base.mask_sha256(query_mask),
                "mask_pixels": int(query_mask.sum()),
                "mask_bbox_xyxy": base.mask_bbox(query_mask),
                "masker": masker,
            }
            warp, certainty = cached_matches[pair_id]
            width = certainty.shape[1] // 2
            forward = warp[:, :width]
            backward = warp[:, width:]
            a_to_b = paired_cycle_component(
                forward[..., :2], forward[..., 2:], certainty[:, :width],
                backward[..., :2], certainty[:, width:],
                reference_mask_by_episode[reference_id], query_mask, protocol["matcher"],
            )
            b_to_a = paired_cycle_component(
                backward[..., 2:], backward[..., :2], certainty[:, width:],
                forward[..., 2:], certainty[:, :width],
                query_mask, reference_mask_by_episode[reference_id], protocol["matcher"],
            )
            supported = all(
                float(row["paired_cycle_fraction"]) >= float(gate["minimum_reference_cycle_fraction"])
                and float(row["dominant_component_fraction_of_paired_cycles"])
                >= float(gate["minimum_dominant_component_cycle_fraction"])
                for row in (a_to_b, b_to_a)
            )
            bilateral_receipts[pair_id] = {
                "reference_to_query": a_to_b,
                "query_to_reference": b_to_a,
                "bilateral_mask_paired_support": supported,
            }
        supported = bool(bilateral_receipts[pair_id]["bilateral_mask_paired_support"])
        extent_ok = (
            pair["label"] == "target_absent"
            or float(prompt["target_bbox_iou_evaluation_only"])
            >= float(gate["minimum_positive_extent_target_bbox_iou"])
        )
        commit = supported and extent_ok
        decisions[pair_id] = {
            **pair,
            "bilateral_mask_paired_support": supported,
            "positive_extent_gate": extent_ok if pair["label"] == "target_present" else None,
            "commit": commit,
            "correct": commit if pair["label"] == "target_present" else not commit,
        }

    positive_rows = [row for row in decisions.values() if row["label"] == "target_present"]
    negative_rows = [row for row in decisions.values() if row["label"] == "target_absent"]
    positive_commits = sum(bool(row["commit"]) for row in positive_rows)
    negative_false_commits = sum(bool(row["commit"]) for row in negative_rows)
    gate_met = (
        len(positive_rows) == int(gate["required_positive_pairs"])
        and len(negative_rows) == int(gate["required_target_absent_pairs"])
        and positive_commits >= int(gate["minimum_positive_commits"])
        and negative_false_commits <= int(gate["maximum_target_absent_false_commits"])
    )
    base.roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_BILATERAL_MASK_PAIRED_COHERENT_SUPPORT_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": base.sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
            "source": {"cohort_path": cohort_path.name, "cohort_sha256": base.sha256(cohort_path)},
            "conclusion": (
                "L10_3RSCAN_BILATERAL_MASK_PAIRED_CYCLE_COMPONENT_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met
                else "L10_3RSCAN_BILATERAL_MASK_PAIRED_CYCLE_COMPONENT_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "positive_pairs": len(positive_rows),
                "positive_commits": positive_commits,
                "target_absent_pairs": len(negative_rows),
                "target_absent_false_commits": negative_false_commits,
            },
            "decisions": decisions,
            "bilateral_mask_paired_receipts": bilateral_receipts,
            "prompt_receipts": prompt_receipts,
            "reference_support_receipts": reference_receipts,
            "query_support_receipts": query_support_receipts,
            "sibling_absence_receipts": absence_receipts,
            "runtime": {
                "device": device_name,
                "roma_calls": len(decisions),
                "sam2_calls": len(reference_receipts) + sum(
                    row["prompt_box_xyxy"] is not None for row in prompt_receipts.values()
                ),
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
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
