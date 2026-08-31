#!/usr/bin/env python3
"""Use one target-surround context ring to transport the bound extent."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cycle_component_open_set_posthoc as open_set  # noqa: E402
import l10_3rscan_projective_extent_posthoc as projective  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-context-ring-projective-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-context-ring-projective-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def context_ring(size: tuple[int, int], bbox: list[float], scale: float) -> tuple[np.ndarray, list[float]]:
    width, height = size
    x0, y0, x1, y1 = (float(value) for value in bbox)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_width = (x1 - x0) * scale / 2.0
    half_height = (y1 - y0) * scale / 2.0
    outer_box = [
        max(0.0, cx - half_width), max(0.0, cy - half_height),
        min(float(width), cx + half_width), min(float(height), cy + half_height),
    ]
    outer = open_set.base.rectangle_mask(size, outer_box)
    inner = open_set.base.rectangle_mask(size, bbox)
    ring = np.ascontiguousarray(outer & ~inner, dtype=np.bool_)
    require(int(ring.sum()) > 0, "EMPTY_CONTEXT_RING")
    return ring, outer_box


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch

    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    require(sha256(predecessor_path) == protocol["predecessor"]["result_sha256"], "PREDECESSOR_HASH")
    predecessor = load_json(predecessor_path)
    require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    inner_path = HERE / protocol["local_carrier_protocol_path"]
    require(sha256(inner_path) == protocol["local_carrier_protocol_sha256"], "LOCAL_PROTOCOL_HASH")
    with open_set.protocol_surface():
        inner = open_set.base.load_protocol(inner_path)
    cohort = open_set.base.load_json(HERE / inner["source"]["cohort_path"])
    images, inputs = open_set.base.load_images(inner, cohort)
    pair_id = str(protocol["evaluation"]["positive_pair_id"])
    positive = next(row for row in inner["evaluation"]["pairs"] if str(row["id"]) == pair_id)
    episode_id = str(positive["reference_episode"])
    reference_image = images[f"{episode_id}:reference"]
    query_image = images[f"{episode_id}:query"]
    reference_box = inputs[f"{episode_id}:reference"]["target_bbox_xyxy_evaluation_only"]
    ring, outer_box = context_ring(
        reference_image.size, reference_box, float(protocol["context_carrier"]["outer_scale"])
    )
    extent = open_set.base.rectangle_mask(reference_image.size, reference_box)
    extent_masks = {open_set.base.mask_sha256(ring): extent}

    model_root = ROOT / inner["matcher"]["path"]
    weights = torch.load(model_root / "roma_indoor.pth", map_location="cpu", weights_only=True)
    backbone = torch.load(model_root / "dinov2_vitl14_pretrain.pth", map_location="cpu", weights_only=True)
    matcher_model = romatch.roma_indoor(
        device="cuda", weights=weights, dinov2_weights=backbone,
        coarse_res=int(inner["matcher"]["coarse_resolution"]),
        upsample_res=int(inner["matcher"]["upsample_resolution"]),
        symmetric=True, use_custom_corr=False, upsample_preds=True,
    )
    with torch.inference_mode():
        warp_batch, certainty_batch = matcher_model.match(reference_image, query_image)
    box, receipt = projective.projective_prompt(
        extent_masks,
        warp_batch[0].detach().cpu(),
        certainty_batch[0].detach().cpu(),
        ring,
        query_image.size,
        inner["matcher"],
        float(protocol["projective_model"]["reprojection_threshold_pixels"]),
    )
    del matcher_model, weights, backbone
    gc.collect()
    torch.cuda.empty_cache()
    target_box = inputs[f"{episode_id}:query"]["target_bbox_xyxy_evaluation_only"]
    iou, recall, precision = open_set.base.bbox_iou(box, target_box)
    receipt.update({
        "selection_authority": "DETERMINISTIC_TARGET_EXCLUDED_TWO_X_CONTEXT_RING_LARGEST_CYCLE_COMPONENT_USAC_MAGSAC",
        "context_outer_box_xyxy": outer_box,
        "context_ring_pixels": int(ring.sum()),
        "target_bbox_iou_evaluation_only": iou,
        "target_bbox_recall_evaluation_only": recall,
        "prompt_bbox_precision_evaluation_only": precision,
    })
    gate = protocol["decision_gate"]
    context_cycle = float(receipt["all_cycle_fraction"]) >= float(gate["minimum_context_cycle_fraction"])
    context_component = (
        float(receipt["selected_component_fraction_of_cycles"])
        >= float(gate["minimum_context_component_cycle_fraction"])
    )
    context_projective = (
        float(receipt["homography_inlier_fraction"])
        >= float(gate["minimum_projective_inlier_fraction"])
    )
    extent_ok = iou >= float(gate["minimum_positive_extent_target_bbox_iou"])
    target_identity_opportunity = bool(predecessor["decision"]["direct_exit_support"])
    commit = bool(target_identity_opportunity and context_cycle and context_component and context_projective and extent_ok)
    negatives = predecessor["inherited_negative_decisions"]
    false_commits = sum(bool(row["commit"]) for row in negatives.values())
    gate_met = bool(commit and false_commits <= int(gate["maximum_target_absent_false_commits"]))
    write_json(output_path, {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SC34_CONTEXT_RING_PROJECTIVE_POSTHOC_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "predecessor": {"path": predecessor_path.name, "sha256": sha256(predecessor_path)},
        "conclusion": (
            "L10_3RSCAN_CONTEXT_RING_PROJECTIVE_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_CONTEXT_RING_PROJECTIVE_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "positive_pairs": 1,
            "positive_commits": int(commit),
            "target_absent_pairs": len(negatives),
            "target_absent_false_commits": false_commits,
            "target_internal_affine_extent_iou": float(predecessor["metrics"]["affine_positive_extent_iou"]),
            "target_internal_projective_extent_iou": float(predecessor["metrics"]["projective_positive_extent_iou"]),
            "context_ring_projective_extent_iou": iou,
        },
        "decision": {
            "id": pair_id,
            "label": "target_present",
            "target_identity_cycle_opportunity": target_identity_opportunity,
            "context_cycle_support": context_cycle,
            "context_component_support": context_component,
            "context_projective_inlier_support": context_projective,
            "context_extent_gate": extent_ok,
            "commit": commit,
        },
        "context_receipt": receipt,
        "inherited_negative_decisions": negatives,
        "runtime": {"device": torch.cuda.get_device_name(0), "roma_calls": 1, "sam2_calls": 0, "grounding_dino_calls": 0},
        "literature_motivation": protocol["literature_motivation"],
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
