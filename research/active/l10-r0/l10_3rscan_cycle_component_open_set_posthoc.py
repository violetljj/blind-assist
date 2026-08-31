#!/usr/bin/env python3
"""Frozen cross-scene target-absent test for coherent 3RScan transport."""

from __future__ import annotations

import argparse
import gc
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_dual_surface_posthoc as dual  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc as reference_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cycle-component-open-set-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-cycle-component-open-set-posthoc-result-v1"


@contextmanager
def protocol_surface():
    saved_schema = base.PROTOCOL_SCHEMA
    saved_file = base.__file__
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.__file__ = str(Path(__file__).resolve())
    try:
        yield
    finally:
        base.PROTOCOL_SCHEMA = saved_schema
        base.__file__ = saved_file


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch
    from transformers import Sam2Model, Sam2Processor

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
    for episode_id, episode in episodes.items():
        key = f"{episode_id}:reference"
        image = images[key]
        bbox = inputs[key]["target_bbox_xyxy_evaluation_only"]
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
    prompt_receipts: dict[str, Any] = {}
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        reference_id = str(pair["reference_episode"])
        query_id = str(pair["query_episode"])
        if pair["label"] == "target_absent":
            base.require(
                episodes[reference_id]["reference_scan_id"]
                != episodes[query_id]["reference_scan_id"],
                f"NEGATIVE_NOT_CROSS_SCENE:{pair_id}",
            )
        with torch.inference_mode():
            warp_batch, certainty_batch = matcher_model.match(
                images[f"{reference_id}:reference"], images[f"{query_id}:query"]
            )
        try:
            prompt_box, receipt = dual.dual_surface_cycle_affine_prompt(
                extent_masks,
                warp_batch[0].detach().cpu(),
                certainty_batch[0].detach().cpu(),
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
            iou, recall, precision = base.bbox_iou(prompt_box, target_box)
            receipt["target_bbox_iou_evaluation_only"] = iou
            receipt["target_bbox_recall_evaluation_only"] = recall
            receipt["prompt_bbox_precision_evaluation_only"] = precision
        else:
            receipt["target_bbox_iou_evaluation_only"] = None
            receipt["target_absence_authority"] = "DIFFERENT_3RSCAN_REFERENCE_SCAN_FAMILY"
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
    decisions: dict[str, Any] = {}
    gate = protocol["decision_gate"]
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        query_id = str(pair["query_episode"])
        query = images[f"{query_id}:query"]
        prompt = prompt_receipts[pair_id]
        if prompt["prompt_box_xyxy"] is None:
            query_support_receipts[pair_id] = {
                "status": "NOT_RUN_ZERO_REFERENCE_CYCLES",
                "masker_calls": 0,
            }
        else:
            masks, masker = base.sam_base._sam_masks(
                processor, model, query, [prompt["prompt_box_xyxy"]], query.size, torch, np
            )
            base.require(len(masks) == 1, f"QUERY_SAM_MASK_COUNT:{pair_id}")
            mask = np.ascontiguousarray(masks[0], dtype=np.bool_)
            base.require(int(mask.sum()) > 0, f"EMPTY_QUERY_SAM_MASK:{pair_id}")
            mask_box = base.mask_bbox(mask)
            query_support_receipts[pair_id] = {
                "mask_sha256": base.mask_sha256(mask),
                "mask_pixels": int(mask.sum()),
                "mask_bbox_xyxy": mask_box,
                "masker": masker,
            }
            if pair["label"] == "target_present":
                target_box = inputs[f"{query_id}:query"]["target_bbox_xyxy_evaluation_only"]
                query_support_receipts[pair_id]["target_bbox_iou_diagnostic_only"] = base.bbox_iou(
                    mask_box, target_box
                )[0]
        cycle_fraction = float(prompt["all_cycle_fraction"])
        dominance = float(prompt["selected_component_fraction_of_cycles"])
        coherent = (
            cycle_fraction >= float(gate["minimum_reference_cycle_fraction"])
            and dominance >= float(gate["minimum_dominant_component_cycle_fraction"])
        )
        extent_ok = (
            pair["label"] == "target_absent"
            or float(prompt["target_bbox_iou_evaluation_only"])
            >= float(gate["minimum_positive_extent_target_bbox_iou"])
        )
        commit = coherent and extent_ok
        decisions[pair_id] = {
            **pair,
            "reference_cycle_fraction": cycle_fraction,
            "dominant_component_cycle_fraction": dominance,
            "coherent_component_support": coherent,
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
            "authority": "CONSUMED_POSTHOC_CROSS_SCENE_TARGET_ABSENT_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": base.sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
            "source": {"cohort_path": cohort_path.name, "cohort_sha256": base.sha256(cohort_path)},
            "conclusion": (
                "L10_3RSCAN_CYCLE_COMPONENT_OPEN_SET_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met else "L10_3RSCAN_CYCLE_COMPONENT_OPEN_SET_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "positive_pairs": len(positive_rows),
                "positive_commits": positive_commits,
                "target_absent_pairs": len(negative_rows),
                "target_absent_false_commits": negative_false_commits,
            },
            "decisions": decisions,
            "prompt_receipts": prompt_receipts,
            "reference_support_receipts": reference_receipts,
            "query_support_receipts": query_support_receipts,
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
    replay(args.protocol, args.output)


if __name__ == "__main__":
    main()
