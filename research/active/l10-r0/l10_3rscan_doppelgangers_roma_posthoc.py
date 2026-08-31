#!/usr/bin/env python3
"""Apply the official Doppelgangers classifier to deterministic RoMa match masks."""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOPPEL_ROOT = ROOT / "artifacts.local/runtimes/doppelgangers-source-v1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DOPPEL_ROOT))
import l10_3rscan_cycle_component_open_set_posthoc as open_set  # noqa: E402
import l10_3rscan_cycle_component_sibling_door_posthoc as sibling  # noqa: E402
from doppelgangers.models.cnn_classifier import decoder  # noqa: E402
from doppelgangers.utils.dataset import read_loftr_matches  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-doppelgangers-roma-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-doppelgangers-roma-posthoc-result-v1"


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


def roma_keypoints(
    warp: torch.Tensor,
    certainty: torch.Tensor,
    matcher: dict[str, Any],
    adapter: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    height, double_width = certainty.shape
    width = double_width // 2
    open_set.base.require(height == width == int(matcher["upsample_resolution"]), "ROMA_OUTPUT_RESOLUTION")
    forward = warp[:, :width]
    backward = warp[:, width:]
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
    high = (certainty[:, :width] >= threshold) & torch.all(torch.abs(target_coords) <= 1.0, dim=-1)
    cycles = (
        high
        & (sampled_backward_certainty >= threshold)
        & (cycle_error <= float(matcher["maximum_cycle_error_normalized"]))
    )
    flat_indices = torch.nonzero(high.reshape(-1), as_tuple=False).flatten().detach().cpu().numpy()
    maximum = int(adapter["maximum_keypoints"])
    if len(flat_indices) > maximum:
        keep = np.linspace(0, len(flat_indices) - 1, maximum, dtype=np.int64)
        flat_indices = flat_indices[keep]
    source_flat = source_coords.reshape(-1, 2)[flat_indices].detach().cpu().numpy().astype(np.float64)
    target_flat = target_coords.reshape(-1, 2)[flat_indices].detach().cpu().numpy().astype(np.float64)
    certainty_flat = certainty[:, :width].reshape(-1)[flat_indices].detach().cpu().numpy().astype(np.float64)
    cycle_flat = cycles.reshape(-1)[flat_indices].detach().cpu().numpy().astype(bool)
    resize = int(adapter["input_long_edge"])
    resized_height = int(adapter["resized_short_edge"])
    keypoints0 = np.column_stack(
        ((source_flat[:, 0] + 1.0) * resize / 2.0, (source_flat[:, 1] + 1.0) * resized_height / 2.0)
    ).astype(np.float32)
    keypoints1 = np.column_stack(
        ((target_flat[:, 0] + 1.0) * resize / 2.0, (target_flat[:, 1] + 1.0) * resized_height / 2.0)
    ).astype(np.float32)
    confidence = np.clip(certainty_flat, 0.0, 1.0).astype(np.float32)
    cycle_indices = np.flatnonzero(cycle_flat)
    open_set.base.require(len(cycle_indices) >= 8, "INSUFFICIENT_CYCLE_MATCHES")
    cv2.setRNGSeed(int(adapter["opencv_rng_seed"]))
    fundamental, inliers = cv2.findFundamentalMat(
        keypoints0[cycle_indices],
        keypoints1[cycle_indices],
        cv2.FM_RANSAC,
        float(adapter["fundamental_ransac_threshold_pixels"]),
        float(adapter["fundamental_ransac_confidence"]),
        int(adapter["fundamental_ransac_max_iterations"]),
    )
    open_set.base.require(fundamental is not None and inliers is not None, "FUNDAMENTAL_MATRIX")
    matched = cycle_indices[np.asarray(inliers).reshape(-1).astype(bool)]
    open_set.base.require(len(matched) >= 3, "INSUFFICIENT_GEOMETRIC_MATCHES")
    matches = np.column_stack((matched, matched)).astype(np.int64)
    return keypoints0, keypoints1, confidence, matches, {
        "high_keypoints_before_cap": int(high.sum().item()),
        "adapter_keypoints": int(len(keypoints0)),
        "cycle_keypoints": int(len(cycle_indices)),
        "fundamental_inlier_matches": int(len(matches)),
    }


class ClassifierConfig:
    input_dim = 10
    initial_dim = 128


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch

    base = open_set.base
    absence_receipts = verify_sibling_absence(protocol_path)
    with protocol_surface():
        protocol = base.load_protocol(protocol_path)
    decision_source_path = HERE / protocol["decision_source"]["result_path"]
    base.require(
        base.sha256(decision_source_path) == protocol["decision_source"]["result_sha256"],
        "DECISION_SOURCE_HASH",
    )
    decision_source = base.load_json(decision_source_path)
    base.require(
        decision_source["conclusion"] == protocol["decision_source"]["required_conclusion"],
        "DECISION_SOURCE_CONCLUSION",
    )
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = base.load_json(cohort_path)
    images, _ = base.load_images(protocol, cohort)

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
    classifier_path = ROOT / protocol["doppelgangers"]["weights_path"]
    base.require(
        base.sha256(classifier_path) == protocol["doppelgangers"]["weights_sha256"],
        "DOPPELGANGERS_WEIGHTS_HASH",
    )
    checkpoint = torch.load(classifier_path, map_location="cpu", weights_only=True)
    classifier = decoder(ClassifierConfig()).eval().to("cuda:0")
    classifier.load_state_dict(checkpoint["dec"], strict=True)
    adapter_receipts: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    temp_parent = ROOT / "artifacts.local/tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="l10-doppelgangers-", dir=temp_parent) as temp_dir:
        temporary = Path(temp_dir)
        for pair in protocol["evaluation"]["pairs"]:
            pair_id = str(pair["id"])
            reference_id = str(pair["reference_episode"])
            query_id = str(pair["query_episode"])
            reference_image = images[f"{reference_id}:reference"]
            query_image = images[f"{query_id}:query"]
            with torch.inference_mode():
                warp_batch, certainty_batch = matcher_model.match(reference_image, query_image)
            keypoints0, keypoints1, confidence, matches, receipt = roma_keypoints(
                warp_batch[0].detach().cpu(),
                certainty_batch[0].detach().cpu(),
                protocol["matcher"],
                protocol["roma_adapter"],
            )
            path0 = temporary / f"{pair_id}-reference.jpg"
            path1 = temporary / f"{pair_id}-query.jpg"
            reference_image.save(path0, format="JPEG", quality=95, subsampling=0)
            query_image.save(path1, format="JPEG", quality=95, subsampling=0)
            model_input = read_loftr_matches(
                str(path0),
                str(path1),
                resize=int(protocol["roma_adapter"]["input_long_edge"]),
                df=int(protocol["roma_adapter"]["divisible_by"]),
                padding=True,
                keypoints0=keypoints0,
                keypoints1=keypoints1,
                matches=matches,
                warp=True,
                conf=confidence,
            )[None].to("cuda:0")
            with torch.inference_mode():
                logits = classifier(model_input)
                probability = torch.softmax(logits, dim=1)[0, 1].item()
            doppel_support = probability >= float(protocol["decision_gate"]["minimum_positive_probability"])
            predecessor_commit = bool(decision_source["decisions"][pair_id]["commit"])
            commit = predecessor_commit and doppel_support
            decisions[pair_id] = {
                **pair,
                "mask_paired_predecessor_commit": predecessor_commit,
                "doppelgangers_positive_probability": probability,
                "doppelgangers_support": doppel_support,
                "commit": commit,
                "correct": commit if pair["label"] == "target_present" else not commit,
            }
            adapter_receipts[pair_id] = {
                **receipt,
                "input_tensor_shape": list(model_input.shape),
                "temporary_inputs_retained": False,
            }
    del matcher_model, classifier, weights, dinov2_weights, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    positive_rows = [row for row in decisions.values() if row["label"] == "target_present"]
    negative_rows = [row for row in decisions.values() if row["label"] == "target_absent"]
    positive_commits = sum(bool(row["commit"]) for row in positive_rows)
    negative_false_commits = sum(bool(row["commit"]) for row in negative_rows)
    gate = protocol["decision_gate"]
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
            "authority": "CONSUMED_POSTHOC_OFFICIAL_DOPPELGANGERS_WITH_ROMA_ADAPTER_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": base.sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": base.sha256(Path(__file__))},
            "source": {"cohort_path": cohort_path.name, "cohort_sha256": base.sha256(cohort_path)},
            "conclusion": (
                "L10_3RSCAN_DOPPELGANGERS_ROMA_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met
                else "L10_3RSCAN_DOPPELGANGERS_ROMA_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "positive_pairs": len(positive_rows),
                "positive_commits": positive_commits,
                "target_absent_pairs": len(negative_rows),
                "target_absent_false_commits": negative_false_commits,
            },
            "decisions": decisions,
            "roma_adapter_receipts": adapter_receipts,
            "sibling_absence_receipts": absence_receipts,
            "runtime": {
                "device": torch.cuda.get_device_name(0),
                "roma_calls": len(decisions),
                "doppelgangers_calls": len(decisions),
                "sam2_calls": 0,
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
