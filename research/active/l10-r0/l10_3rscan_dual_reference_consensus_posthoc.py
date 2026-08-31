#!/usr/bin/env python3
"""Require primary and active reference views to converge on one query target."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cycle_component_mask_paired_posthoc as mask_paired  # noqa: E402
import l10_3rscan_cycle_component_open_set_posthoc as open_set  # noqa: E402
import l10_3rscan_cycle_component_sibling_door_posthoc as sibling  # noqa: E402
import l10_3rscan_roma_cycle_prompt_dual_surface_posthoc as dual  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc as reference_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-dual-reference-consensus-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-dual-reference-consensus-posthoc-result-v1"


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


def load_active_references(
    protocol: dict[str, Any], active_cohort: dict[str, Any]
) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    images: dict[str, Image.Image] = {}
    inputs: dict[str, Any] = {}
    for row in active_cohort["episodes"]:
        episode_id = str(row["episode_id"])
        active = row["active_reference"]
        archive_path = artifact_root / f"datasets/3rscan/{row['reference_scan_id']}/sequence.zip"
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(active["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        open_set.base.require(list(image.size) == active["color_size"], f"ACTIVE_IMAGE_SIZE:{episode_id}")
        key = f"{episode_id}:reference"
        images[key] = image
        inputs[key] = {
            "scan_id": row["reference_scan_id"],
            "frame": int(active["frame"]),
            "zip_member": active["zip_member"],
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "image_bytes": len(payload),
            "target_bbox_xyxy_evaluation_only": active["bbox_xyxy"],
            "baseline_metres": float(active["baseline_metres"]),
        }
    return images, inputs


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    open_set.base.require(union > 0, "EMPTY_MASK_UNION")
    return intersection / union


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch
    from transformers import Sam2Model, Sam2Processor

    base = open_set.base
    absence_receipts = verify_sibling_absence(protocol_path)
    with protocol_surface():
        protocol = base.load_protocol(protocol_path)
    active_protocol_path = HERE / protocol["active_reference_source"]["protocol_path"]
    base.require(
        base.sha256(active_protocol_path) == protocol["active_reference_source"]["protocol_sha256"],
        "ACTIVE_PROTOCOL_HASH",
    )
    active_path = HERE / protocol["active_reference_source"]["cohort_path"]
    base.require(base.sha256(active_path) == protocol["active_reference_source"]["cohort_sha256"], "ACTIVE_COHORT_HASH")
    active_cohort = base.load_json(active_path)
    base.require(
        active_cohort["protocol_sha256"] == protocol["active_reference_source"]["protocol_sha256"],
        "ACTIVE_COHORT_PROTOCOL_BINDING",
    )
    base.require(active_cohort["opened_members"]["rgb_members"] == 0, "ACTIVE_FREEZE_RGB")
    base.require(active_cohort["opened_members"]["model_calls"] == 0, "ACTIVE_FREEZE_MODEL")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = base.load_json(cohort_path)
    primary_images, inputs = base.load_images(protocol, cohort)
    active_images, active_inputs = load_active_references(protocol, active_cohort)
    episodes = {str(row["episode_id"]): row for row in cohort["episodes"]}

    primary_masks, primary_reference_receipts, device_name = reference_base.make_reference_masks(
        protocol, cohort, primary_images, inputs
    )
    active_masks, active_reference_receipts, _ = reference_base.make_reference_masks(
        protocol, cohort, active_images, active_inputs
    )
    reference_masks: dict[tuple[str, str], np.ndarray] = {}
    extent_masks: dict[str, np.ndarray] = {}
    for episode_id in episodes:
        for role, images, role_inputs, frozen in (
            ("primary", primary_images, inputs, primary_masks),
            ("active", active_images, active_inputs, active_masks),
        ):
            image = images[f"{episode_id}:reference"]
            bbox = role_inputs[f"{episode_id}:reference"]["target_bbox_xyxy_evaluation_only"]
            lookup = (int(image.size[0]), int(image.size[1]), *(float(value) for value in bbox))
            mask = frozen[lookup]
            reference_masks[(episode_id, role)] = mask
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
    matches: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    prompt_receipts: dict[str, Any] = {}
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        reference_id = str(pair["reference_episode"])
        query_id = str(pair["query_episode"])
        query_image = primary_images[f"{query_id}:query"]
        prompt_receipts[pair_id] = {}
        for role, role_images in (("primary", primary_images), ("active", active_images)):
            with torch.inference_mode():
                warp_batch, certainty_batch = matcher_model.match(
                    role_images[f"{reference_id}:reference"], query_image
                )
            warp = warp_batch[0].detach().cpu()
            certainty = certainty_batch[0].detach().cpu()
            matches[(pair_id, role)] = (warp, certainty)
            try:
                prompt_box, receipt = dual.dual_surface_cycle_affine_prompt(
                    extent_masks,
                    warp,
                    certainty,
                    reference_masks[(reference_id, role)],
                    query_image.size,
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
            prompt_receipts[pair_id][role] = receipt
    del matcher_model, weights, dinov2_weights
    gc.collect()
    torch.cuda.empty_cache()

    sam_root = (ROOT / protocol["proposal"]["masker_root"]).resolve()
    processor = Sam2Processor.from_pretrained(sam_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        sam_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    branch_receipts: dict[str, Any] = {}
    consensus_receipts: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    gate = protocol["decision_gate"]
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        reference_id = str(pair["reference_episode"])
        query_id = str(pair["query_episode"])
        query = primary_images[f"{query_id}:query"]
        branch_receipts[pair_id] = {}
        query_masks: dict[str, np.ndarray] = {}
        for role in ("primary", "active"):
            prompt = prompt_receipts[pair_id][role]
            if prompt["prompt_box_xyxy"] is None:
                branch_receipts[pair_id][role] = {"bilateral_mask_paired_support": False, "query_mask": None}
                continue
            masks_out, masker = base.sam_base._sam_masks(
                processor, model, query, [prompt["prompt_box_xyxy"]], query.size, torch, np
            )
            base.require(len(masks_out) == 1, f"QUERY_SAM_MASK_COUNT:{pair_id}:{role}")
            query_mask = np.ascontiguousarray(masks_out[0], dtype=np.bool_)
            base.require(int(query_mask.sum()) > 0, f"EMPTY_QUERY_SAM_MASK:{pair_id}:{role}")
            query_masks[role] = query_mask
            warp, certainty = matches[(pair_id, role)]
            width = certainty.shape[1] // 2
            forward = warp[:, :width]
            backward = warp[:, width:]
            a_to_b = mask_paired.paired_cycle_component(
                forward[..., :2], forward[..., 2:], certainty[:, :width],
                backward[..., :2], certainty[:, width:],
                reference_masks[(reference_id, role)], query_mask, protocol["matcher"],
            )
            b_to_a = mask_paired.paired_cycle_component(
                backward[..., 2:], backward[..., :2], certainty[:, width:],
                forward[..., 2:], certainty[:, :width],
                query_mask, reference_masks[(reference_id, role)], protocol["matcher"],
            )
            supported = all(
                float(row["paired_cycle_fraction"]) >= float(gate["minimum_cycle_fraction"])
                and float(row["dominant_component_fraction_of_paired_cycles"])
                >= float(gate["minimum_component_dominance"])
                for row in (a_to_b, b_to_a)
            )
            branch_receipts[pair_id][role] = {
                "query_mask_sha256": base.mask_sha256(query_mask),
                "query_mask_pixels": int(query_mask.sum()),
                "query_mask_bbox_xyxy": base.mask_bbox(query_mask),
                "reference_to_query": a_to_b,
                "query_to_reference": b_to_a,
                "bilateral_mask_paired_support": supported,
                "masker": masker,
            }
        if len(query_masks) == 2:
            prompt_iou = base.bbox_iou(
                prompt_receipts[pair_id]["primary"]["prompt_box_xyxy"],
                prompt_receipts[pair_id]["active"]["prompt_box_xyxy"],
            )[0]
            support_mask_iou = mask_iou(query_masks["primary"], query_masks["active"])
        else:
            prompt_iou = 0.0
            support_mask_iou = 0.0
        branches_supported = all(
            bool(branch_receipts[pair_id][role]["bilateral_mask_paired_support"])
            for role in ("primary", "active")
        )
        consensus = (
            branches_supported
            and prompt_iou >= float(gate["minimum_prompt_consensus_iou"])
            and support_mask_iou >= float(gate["minimum_support_mask_consensus_iou"])
        )
        if pair["label"] == "target_present":
            extent_ok = all(
                float(prompt_receipts[pair_id][role]["target_bbox_iou_evaluation_only"])
                >= float(gate["minimum_positive_extent_iou"])
                for role in ("primary", "active")
            )
        else:
            extent_ok = True
        commit = consensus and extent_ok
        consensus_receipts[pair_id] = {
            "both_branches_supported": branches_supported,
            "prompt_consensus_iou": prompt_iou,
            "support_mask_consensus_iou": support_mask_iou,
            "dual_reference_consensus": consensus,
        }
        decisions[pair_id] = {
            **pair,
            "dual_reference_consensus": consensus,
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
            "authority": "CONSUMED_POSTHOC_ACTIVE_DUAL_REFERENCE_CONSENSUS_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": base.sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
            "source": {"cohort_path": cohort_path.name, "cohort_sha256": base.sha256(cohort_path)},
            "active_reference_source": {"cohort_path": active_path.name, "cohort_sha256": base.sha256(active_path)},
            "conclusion": (
                "L10_3RSCAN_DUAL_REFERENCE_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met
                else "L10_3RSCAN_DUAL_REFERENCE_CONSENSUS_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "positive_pairs": len(positive_rows),
                "positive_commits": positive_commits,
                "target_absent_pairs": len(negative_rows),
                "target_absent_false_commits": negative_false_commits,
            },
            "decisions": decisions,
            "consensus_receipts": consensus_receipts,
            "branch_receipts": branch_receipts,
            "prompt_receipts": prompt_receipts,
            "primary_reference_receipts": primary_reference_receipts,
            "active_reference_receipts": active_reference_receipts,
            "active_input_receipts": active_inputs,
            "sibling_absence_receipts": absence_receipts,
            "runtime": {
                "device": device_name,
                "roma_calls": len(decisions) * 2,
                "sam2_calls": len(primary_reference_receipts) + len(active_reference_receipts)
                + sum(
                    prompt_receipts[pair_id][role]["prompt_box_xyxy"] is not None
                    for pair_id in prompt_receipts for role in ("primary", "active")
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
