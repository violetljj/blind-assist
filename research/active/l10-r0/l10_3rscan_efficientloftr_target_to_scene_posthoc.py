#!/usr/bin/env python3
"""Test frozen EfficientLoFTR as a target-crop to full-scene PV28 carrier."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_roma_cycle_prompt_sam_posthoc as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_scenenn_efficientloftr_fresh_none as loftr  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-efficientloftr-target-to-scene-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-efficientloftr-target-to-scene-posthoc-result-v1"


def _crop(image: Image.Image, bbox: list[float], expansion: float) -> tuple[Image.Image, list[int]]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half_w = 0.5 * (x1 - x0) * expansion
    half_h = 0.5 * (y1 - y0) * expansion
    box = [
        max(0, int(math.floor(cx - half_w))),
        max(0, int(math.floor(cy - half_h))),
        min(image.width, int(math.ceil(cx + half_w))),
        min(image.height, int(math.ceil(cy + half_h))),
    ]
    base.require(box[0] < box[2] and box[1] < box[3], "REFERENCE_CROP_EMPTY")
    return image.crop(tuple(box)), box


def _support(
    processor: Any,
    model: Any,
    reference: Image.Image,
    query: Image.Image,
    matcher: dict[str, Any],
) -> dict[str, Any]:
    import torch

    inputs = processor([reference, query], return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    row = processor.post_process_keypoint_matching(
        outputs,
        [[(reference.height, reference.width), (query.height, query.width)]],
        threshold=float(matcher["postprocess_score_threshold"]),
    )[0]
    keypoints0 = row["keypoints0"].detach().cpu().numpy().astype(np.float32)
    keypoints1 = row["keypoints1"].detach().cpu().numpy().astype(np.float32)
    scores = row["matching_scores"].detach().cpu().numpy().astype(np.float32)
    receipt: dict[str, Any] = {
        "match_count": int(len(scores)),
        "mean_match_score": float(scores.mean()) if len(scores) else 0.0,
        "homography_inliers": 0,
        "homography_valid": False,
        "absolute_support": False,
        "prompt_box_xyxy": None,
    }
    if len(scores) < int(matcher["minimum_matches"]):
        return receipt
    homography, mask = cv2.findHomography(
        keypoints0,
        keypoints1,
        cv2.RANSAC,
        float(matcher["homography_ransac_reprojection_pixels"]),
    )
    if homography is None or mask is None or not np.isfinite(homography).all():
        return receipt
    inliers = int(mask.reshape(-1).sum())
    receipt["homography_inliers"] = inliers
    receipt["homography_valid"] = True
    receipt["absolute_support"] = inliers >= int(matcher["minimum_homography_inliers"])
    if not receipt["absolute_support"]:
        return receipt
    corners = np.asarray(
        [[[0.0, 0.0]], [[float(reference.width), 0.0]], [[float(reference.width), float(reference.height)]], [[0.0, float(reference.height)]]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if not np.isfinite(projected).all():
        receipt["absolute_support"] = False
        return receipt
    x0 = float(np.clip(projected[:, 0].min(), 0.0, float(query.width - 1)))
    y0 = float(np.clip(projected[:, 1].min(), 0.0, float(query.height - 1)))
    x1 = float(np.clip(projected[:, 0].max(), 0.0, float(query.width - 1)))
    y1 = float(np.clip(projected[:, 1].max(), 0.0, float(query.height - 1)))
    if not (x1 > x0 and y1 > y0):
        receipt["absolute_support"] = False
        return receipt
    receipt["prompt_box_xyxy"] = [x0, y0, x1, y1]
    return receipt


def replay(protocol_path: Path, output_path: Path) -> None:
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching

    protocol = base.load_json(protocol_path)
    base.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    base.require(base.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        base.require(base.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    predecessor = HERE / protocol["predecessor"]["path"]
    base.require(base.sha256(predecessor) == protocol["predecessor"]["sha256"], "PREDECESSOR_HASH")
    base.require(base.load_json(predecessor)["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    base.require(base.sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    cohort = base.load_json(cohort_path)
    images, inputs = base.load_images(protocol, cohort)
    matcher_root = ROOT / protocol["matcher"]["path"]
    for row in protocol["models"].values():
        base.require(base.sha256(ROOT / row["path"]) == row["sha256"], f"MODEL_HASH:{row['path']}")
    processor = AutoImageProcessor.from_pretrained(matcher_root, local_files_only=True)
    model = AutoModelForKeypointMatching.from_pretrained(matcher_root, local_files_only=True).eval().to("cpu")

    decisions = {}
    receipts = {}
    positive_id = str(protocol["evaluation"]["pairs"][0]["reference_episode"])
    reference_full = images[f"{positive_id}:reference"]
    reference_bbox = inputs[f"{positive_id}:reference"]["target_bbox_xyxy_evaluation_only"]
    reference_crop, crop_box = _crop(reference_full, reference_bbox, float(protocol["matcher"]["reference_crop_expansion"]))
    for pair in protocol["evaluation"]["pairs"]:
        pair_id = str(pair["id"])
        query_id = str(pair["query_episode"])
        query = images[f"{query_id}:query"]
        receipt = _support(processor, model, reference_crop, query, protocol["matcher"])
        if pair["label"] == "target_present" and receipt["prompt_box_xyxy"] is not None:
            target = inputs[f"{query_id}:query"]["target_bbox_xyxy_evaluation_only"]
            iou, recall, precision = base.bbox_iou(receipt["prompt_box_xyxy"], target)
            receipt["target_bbox_iou_evaluation_only"] = iou
            receipt["target_bbox_recall_evaluation_only"] = recall
            receipt["prompt_bbox_precision_evaluation_only"] = precision
            extent_ok = iou >= float(protocol["decision_gate"]["minimum_positive_extent_target_bbox_iou"])
        elif pair["label"] == "target_present":
            receipt["target_bbox_iou_evaluation_only"] = 0.0
            extent_ok = False
        else:
            receipt["target_bbox_iou_evaluation_only"] = None
            receipt["target_absence_authority"] = "DIFFERENT_3RSCAN_REFERENCE_SCAN_FAMILY"
            extent_ok = True
        commit = bool(receipt["absolute_support"] and extent_ok)
        decisions[pair_id] = {
            **pair,
            "absolute_support": bool(receipt["absolute_support"]),
            "positive_extent_gate": bool(extent_ok) if pair["label"] == "target_present" else None,
            "commit": commit,
            "correct": commit if pair["label"] == "target_present" else not commit,
        }
        receipts[pair_id] = receipt
    positives = [row for row in decisions.values() if row["label"] == "target_present"]
    negatives = [row for row in decisions.values() if row["label"] == "target_absent"]
    positive_commits = sum(bool(row["commit"]) for row in positives)
    negative_false_commits = sum(bool(row["commit"]) for row in negatives)
    gate = protocol["decision_gate"]
    gate_met = (
        positive_commits >= int(gate["minimum_positive_commits"])
        and negative_false_commits <= int(gate["maximum_target_absent_false_commits"])
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_SOURCE_STRUCTURALLY_DIFFERENT_MATCHER_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
        "source": {"cohort_path": cohort_path.name, "cohort_sha256": base.sha256(cohort_path)},
        "conclusion": (
            "L10_3RSCAN_EFFICIENTLOFTR_TARGET_TO_SCENE_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_EFFICIENTLOFTR_TARGET_TO_SCENE_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "positive_pairs": len(positives),
            "positive_commits": positive_commits,
            "target_absent_pairs": len(negatives),
            "target_absent_false_commits": negative_false_commits,
        },
        "reference_crop": {"source_bbox_xyxy": reference_bbox, "expanded_bbox_xyxy": crop_box, "size": list(reference_crop.size)},
        "decisions": decisions,
        "receipts": receipts,
        "runtime": {"device": "cpu", "efficientloftr_calls": len(decisions), "roma_calls": 0, "sam2_calls": 0},
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
