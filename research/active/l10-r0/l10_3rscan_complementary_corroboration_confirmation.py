#!/usr/bin/env python3
"""Run frozen complementary corroboration on a new physical-target panel."""

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
import l10_3rscan_active_query_consensus_posthoc as active_base  # noqa: E402
import l10_3rscan_cycle_component_global_epipolar_posthoc as global_base  # noqa: E402
import l10_3rscan_cycle_component_open_set_posthoc as open_set  # noqa: E402
import l10_3rscan_roma_cycle_prompt_dual_surface_posthoc as dual  # noqa: E402
import l10_3rscan_roma_cycle_prompt_sam_reference_mask_posthoc as reference_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-complementary-corroboration-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-complementary-corroboration-confirmation-result-v1"


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


def load_active_queries(
    protocol: dict[str, Any], cohort: dict[str, Any]
) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    images: dict[str, Image.Image] = {}
    receipts: dict[str, Any] = {}
    required_query_ids = {str(row["query_episode"]) for row in protocol["evaluation"]["pairs"]}
    for row in cohort["episodes"]:
        episode_id = str(row["episode_id"])
        if episode_id not in required_query_ids:
            continue
        active = row["active_query"]
        open_set.base.require(active is not None, f"ACTIVE_QUERY_MISSING:{episode_id}")
        archive_path = artifact_root / f"datasets/3rscan/{row['rescan_id']}/sequence.zip"
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(active["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        open_set.base.require(list(image.size) == active["color_size"], f"ACTIVE_IMAGE_SIZE:{episode_id}")
        key = f"{episode_id}:query"
        images[key] = image
        receipts[key] = {
            "scan_id": row["rescan_id"],
            "frame": int(active["frame"]),
            "zip_member": active["zip_member"],
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "image_bytes": len(payload),
            "target_bbox_xyxy_evaluation_only": active["bbox_xyxy"],
            "baseline_metres": float(active["baseline_metres"]),
            "inside_vertex_fraction": float(active["inside_vertex_fraction"]),
            "foreign_target_instance_id": active["foreign_target_instance_id"],
            "foreign_target_inside_vertices": active["foreign_target_inside_vertices"],
        }
    open_set.base.require(set(images) == {f"{value}:query" for value in required_query_ids}, "ACTIVE_QUERY_SET")
    return images, receipts


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch
    from transformers import Sam2Model, Sam2Processor

    base = open_set.base
    with protocol_surface():
        protocol = base.load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = base.load_json(cohort_path)
    base.require(cohort["protocol_sha256"] == protocol["source"]["freeze_protocol_sha256"], "COHORT_FREEZE_BINDING")
    base.require(cohort["selection"]["opened_members"]["rgb_members"] == 0, "FREEZE_RGB")
    base.require(cohort["selection"]["opened_members"]["model_calls"] == 0, "FREEZE_MODEL")
    base.require(cohort["evaluation"]["pairs"] == protocol["evaluation"]["pairs"], "EVALUATION_BINDING")
    for receipt in cohort["sibling_absence_receipts"].values():
        base.require(
            int(receipt["primary_projected_inside_vertices"]) == 0
            and int(receipt["active_projected_inside_vertices"]) == 0,
            "NEGATIVE_ABSENCE",
        )

    primary_images, inputs = base.load_images(protocol, cohort)
    active_images, active_inputs = load_active_queries(protocol, cohort)
    episodes = {str(row["episode_id"]): row for row in cohort["episodes"]}
    reference_masks_raw, reference_receipts, device_name = reference_base.make_reference_masks(
        protocol, cohort, primary_images, inputs
    )
    reference_masks: dict[str, np.ndarray] = {}
    extent_masks: dict[str, np.ndarray] = {}
    for episode_id in episodes:
        image = primary_images[f"{episode_id}:reference"]
        bbox = inputs[f"{episode_id}:reference"]["target_bbox_xyxy_evaluation_only"]
        lookup = (int(image.size[0]), int(image.size[1]), *(float(value) for value in bbox))
        mask = reference_masks_raw[lookup]
        reference_masks[episode_id] = mask
        extent_masks[base.mask_sha256(mask)] = base.rectangle_mask(image.size, bbox)

    model_root = ROOT / protocol["matcher"]["path"]
    weights = torch.load(model_root / "roma_indoor.pth", map_location="cpu", weights_only=True)
    dinov2_weights = torch.load(model_root / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True)
    matcher_model = romatch.roma_indoor(
        device="cuda", weights=weights, dinov2_weights=dinov2_weights,
        coarse_res=int(protocol["matcher"]["coarse_resolution"]),
        upsample_res=int(protocol["matcher"]["upsample_resolution"]),
        symmetric=True, use_custom_corr=False, upsample_preds=True,
    )
    pair_matches: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    prompt_receipts: dict[str, Any] = {}
    global_receipts: dict[str, Any] = {}
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        reference_id = str(pair["reference_episode"])
        query_id = str(pair["query_episode"])
        prompt_receipts[pair_id] = {}
        for role, role_images in (("primary", primary_images), ("active", active_images)):
            query_image = role_images[f"{query_id}:query"]
            with torch.inference_mode():
                warp_batch, certainty_batch = matcher_model.match(
                    primary_images[f"{reference_id}:reference"], query_image
                )
            warp = warp_batch[0].detach().cpu()
            certainty = certainty_batch[0].detach().cpu()
            pair_matches[(pair_id, role)] = (warp, certainty)
            try:
                prompt_box, receipt = dual.dual_surface_cycle_affine_prompt(
                    extent_masks, warp, certainty, reference_masks[reference_id],
                    query_image.size, protocol["matcher"],
                )
            except ValueError as exc:
                if str(exc) != "NO_REFERENCE_CYCLES" or pair["label"] != "target_absent":
                    raise
                prompt_box = None
                receipt = {
                    "selection_authority": "ZERO_REFERENCE_CYCLES_DETERMINISTIC_NON_COMMIT",
                    "all_cycle_pixels": 0, "all_cycle_fraction": 0.0,
                    "selected_component_fraction_of_cycles": 0.0,
                    "prompt_box_xyxy": None,
                }
            if pair["label"] == "target_present":
                target_box = (
                    inputs[f"{query_id}:query"]["target_bbox_xyxy_evaluation_only"]
                    if role == "primary"
                    else active_inputs[f"{query_id}:query"]["target_bbox_xyxy_evaluation_only"]
                )
                receipt["target_bbox_iou_evaluation_only"] = base.bbox_iou(prompt_box, target_box)[0]
            else:
                receipt["target_bbox_iou_evaluation_only"] = None
            prompt_receipts[pair_id][role] = receipt

        primary_warp, primary_certainty = pair_matches[(pair_id, "primary")]
        target_source, target_query, background_source, background_query, domain_receipt = global_base.cycle_domains(
            primary_warp, primary_certainty, reference_masks[reference_id], protocol["matcher"]
        )
        global_receipts[pair_id] = {
            **domain_receipt,
            **global_base.epipolar_receipt(
                target_source, target_query, background_source, background_query, protocol
            ),
        }

    query_ids = sorted({str(row["query_episode"]) for row in protocol["evaluation"]["pairs"]})
    query_matches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for query_id in query_ids:
        with torch.inference_mode():
            warp_batch, certainty_batch = matcher_model.match(
                primary_images[f"{query_id}:query"], active_images[f"{query_id}:query"]
            )
        query_matches[query_id] = (warp_batch[0].detach().cpu(), certainty_batch[0].detach().cpu())
    del matcher_model, weights, dinov2_weights
    gc.collect()
    torch.cuda.empty_cache()

    sam_root = (ROOT / protocol["proposal"]["masker_root"]).resolve()
    processor = Sam2Processor.from_pretrained(sam_root, local_files_only=True)
    model = Sam2Model.from_pretrained(
        sam_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    branch_receipts: dict[str, Any] = {}
    corroboration_receipts: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    gate = protocol["decision_gate"]
    active_majority_floor = float(protocol["corroboration_rule"]["minimum_directional_active_query_fraction"])
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        reference_id = str(pair["reference_episode"])
        query_id = str(pair["query_episode"])
        branch_receipts[pair_id] = {}
        query_masks: dict[str, np.ndarray] = {}
        for role, role_images in (("primary", primary_images), ("active", active_images)):
            prompt = prompt_receipts[pair_id][role]
            if prompt["prompt_box_xyxy"] is None:
                branch_receipts[pair_id][role] = {"bilateral_mask_paired_support": False, "query_mask": None}
                continue
            query = role_images[f"{query_id}:query"]
            masks_out, masker = base.sam_base._sam_masks(
                processor, model, query, [prompt["prompt_box_xyxy"]], query.size, torch, np
            )
            base.require(len(masks_out) == 1, f"QUERY_SAM_MASK_COUNT:{pair_id}:{role}")
            query_mask = np.ascontiguousarray(masks_out[0], dtype=np.bool_)
            base.require(int(query_mask.sum()) > 0, f"EMPTY_QUERY_SAM_MASK:{pair_id}:{role}")
            query_masks[role] = query_mask
            warp, certainty = pair_matches[(pair_id, role)]
            supported, a_to_b, b_to_a = active_base.paired_support(
                warp, certainty, reference_masks[reference_id], query_mask,
                protocol["matcher"], gate,
            )
            branch_receipts[pair_id][role] = {
                "query_mask_sha256": base.mask_sha256(query_mask),
                "query_mask_pixels": int(query_mask.sum()),
                "query_mask_bbox_xyxy": base.mask_bbox(query_mask),
                "reference_to_query": a_to_b, "query_to_reference": b_to_a,
                "bilateral_mask_paired_support": supported, "masker": masker,
            }

        primary_supported = bool(branch_receipts[pair_id]["primary"]["bilateral_mask_paired_support"])
        both_branches_supported = len(query_masks) == 2 and primary_supported and bool(
            branch_receipts[pair_id]["active"]["bilateral_mask_paired_support"]
        )
        if len(query_masks) == 2:
            query_warp, query_certainty = query_matches[query_id]
            query_supported, primary_to_active, active_to_primary = active_base.paired_support(
                query_warp, query_certainty, query_masks["primary"], query_masks["active"],
                protocol["matcher"], gate,
            )
            forward_fraction = float(primary_to_active["paired_cycle_fraction"])
            reverse_fraction = float(active_to_primary["paired_cycle_fraction"])
        else:
            query_supported = False
            primary_to_active = active_to_primary = None
            forward_fraction = reverse_fraction = 0.0
        active_majority = bool(
            both_branches_supported and query_supported
            and forward_fraction >= active_majority_floor
            and reverse_fraction >= active_majority_floor
        )
        global_support = bool(global_receipts[pair_id]["global_epipolar_support"])
        if pair["label"] == "target_present":
            primary_extent_ok = float(prompt_receipts[pair_id]["primary"]["target_bbox_iou_evaluation_only"]) >= float(gate["minimum_positive_extent_iou"])
            active_extent_ok = float(prompt_receipts[pair_id]["active"]["target_bbox_iou_evaluation_only"]) >= float(gate["minimum_positive_extent_iou"])
        else:
            primary_extent_ok = active_extent_ok = True
        local_commit = primary_supported and primary_extent_ok
        corroborated = global_support or active_majority
        commit = local_commit and corroborated
        corroboration_receipts[pair_id] = {
            "local_bilateral_commit": local_commit,
            "global_epipolar_support": global_support,
            "both_reference_query_branches_supported": both_branches_supported,
            "primary_to_active_query": primary_to_active,
            "active_to_primary_query": active_to_primary,
            "active_query_bidirectional_majority_support": active_majority,
            "complementary_corroboration": corroborated,
        }
        decisions[pair_id] = {
            **pair,
            "positive_primary_extent_gate": primary_extent_ok if pair["label"] == "target_present" else None,
            "positive_active_extent_gate": active_extent_ok if pair["label"] == "target_present" else None,
            "local_bilateral_commit": local_commit,
            "global_epipolar_support": global_support,
            "active_query_bidirectional_majority_support": active_majority,
            "commit": commit,
            "correct": commit if pair["label"] == "target_present" else not commit,
        }

    positives = [row for row in decisions.values() if row["label"] == "target_present"]
    negatives = [row for row in decisions.values() if row["label"] == "target_absent"]
    positive_commits = sum(bool(row["commit"]) for row in positives)
    false_commits = sum(bool(row["commit"]) for row in negatives)
    gate_met = bool(
        len(positives) == int(gate["required_positive_pairs"])
        and len(negatives) == int(gate["required_target_absent_pairs"])
        and positive_commits >= int(gate["minimum_positive_commits"])
        and false_commits <= int(gate["maximum_target_absent_false_commits"])
    )
    base.roma_base.predecessor.parent.write_json(output_path, {
        "schema": RESULT_SCHEMA,
        "authority": "PHYSICAL_TARGET_DISJOINT_PARTIAL_COMPLEMENTARY_CORROBORATION_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name, "protocol_sha256": base.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
        "source": {"cohort_path": cohort_path.name, "cohort_sha256": base.sha256(cohort_path)},
        "conclusion": (
            "L10_3RSCAN_COMPLEMENTARY_CORROBORATION_PARTIAL_PHYSICAL_TARGET_CONFIRMATION_GATE_MET"
            if gate_met else "L10_3RSCAN_COMPLEMENTARY_CORROBORATION_PARTIAL_PHYSICAL_TARGET_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "positive_pairs": len(positives), "positive_commits": positive_commits,
            "target_absent_pairs": len(negatives), "target_absent_false_commits": false_commits,
            "committed_precision": positive_commits / (positive_commits + false_commits) if positive_commits + false_commits else 0.0,
        },
        "decisions": decisions,
        "corroboration_receipts": corroboration_receipts,
        "global_epipolar_receipts": global_receipts,
        "branch_receipts": branch_receipts,
        "prompt_receipts": prompt_receipts,
        "reference_receipts": reference_receipts,
        "active_input_receipts": active_inputs,
        "sibling_absence_receipts": cohort["sibling_absence_receipts"],
        "runtime": {
            "device": device_name,
            "roma_calls": len(protocol["evaluation"]["pairs"]) * 2 + len(query_ids),
            "sam2_calls": len(reference_receipts) + sum(
                prompt_receipts[pair_id][role]["prompt_box_xyxy"] is not None
                for pair_id in prompt_receipts for role in ("primary", "active")
            ),
            "grounding_dino_calls": 0,
        },
        "claim_boundary": protocol["claim_boundary"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
