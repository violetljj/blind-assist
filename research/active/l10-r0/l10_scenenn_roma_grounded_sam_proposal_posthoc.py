#!/usr/bin/env python3
"""Consumed SceneNN test of GroundingDINO+SAM2 proposal masks for RoMa."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_roma_active_none as roma_base  # noqa: E402
import l10_scenenn_roma_full_context_mask_posthoc as context_base  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as proposal_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-roma-grounded-sam-proposal-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-roma-grounded-sam-proposal-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return roma_base.sha256(path)


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes(order="C")).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(protocol["implementation"]["sha256"] == sha256(Path(__file__)), "IMPLEMENTATION_HASH")
    predecessor = HERE / protocol["predecessor"]["result_path"]
    require(sha256(predecessor) == protocol["predecessor"]["result_sha256"], "PREDECESSOR_HASH")
    require(load_json(predecessor)["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    cohort = HERE / protocol["source"]["cohort_path"]
    receipt = HERE / protocol["source"]["rgb_receipt_path"]
    require(sha256(cohort) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    require(sha256(receipt) == protocol["source"]["rgb_receipt_sha256"], "RGB_RECEIPT_HASH")
    for dependency in protocol["dependencies"]:
        require(sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    for model in ("grounder", "masker", "matcher", "matcher_backbone"):
        row = protocol["models"][model]
        require(sha256(ROOT / row["path"]) == row["sha256"], f"MODEL_HASH:{model}")
    require(float(protocol["proposal"]["box_threshold"]) == proposal_base.BOX_THRESHOLD, "BOX_THRESHOLD")
    require(float(protocol["proposal"]["text_threshold"]) == proposal_base.TEXT_THRESHOLD, "TEXT_THRESHOLD")
    return protocol


def truth_mask(
    episode: dict[str, Any],
    role: str,
    source_root: Path,
    intrinsic: Any,
    renderer_config: dict[str, Any],
) -> np.ndarray:
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
        renderer_config,
    )
    visible_device, _ = renderer.visible_mask(poses[int(episode[role]["frame"])])
    visible = renderer.cp.asnumpy(visible_device)
    require(
        roma_base.predecessor.parent.visible.mask_sha256(visible) == episode[role]["visible_mask_sha256"],
        f"TRUTH_MASK_HASH:{scene_id}:{role}",
    )
    return visible


def proposal_receipt(proposal: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    intersection = int(np.logical_and(proposal, truth).sum())
    union = int(np.logical_or(proposal, truth).sum())
    truth_pixels = int(truth.sum())
    proposal_pixels = int(proposal.sum())
    require(proposal_pixels > 0 and truth_pixels > 0 and union > 0, "EMPTY_MASK")
    return {
        "proposal_mask_sha256": mask_sha256(proposal),
        "proposal_pixels": proposal_pixels,
        "truth_pixels": truth_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "truth_iou_evaluation_only": intersection / union,
        "truth_recall_evaluation_only": intersection / truth_pixels,
        "proposal_precision_evaluation_only": intersection / proposal_pixels,
    }


def make_proposals(
    protocol: dict[str, Any],
    images: dict[str, Image.Image],
    truths: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

    ground_root = (ROOT / protocol["proposal"]["grounder_root"]).resolve()
    sam_root = (ROOT / protocol["proposal"]["masker_root"]).resolve()
    ground_processor = AutoProcessor.from_pretrained(ground_root, local_files_only=True)
    ground_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        ground_root,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval().to("cuda:0")
    sam_processor = Sam2Processor.from_pretrained(sam_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        sam_root,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval().to("cuda:0")
    prompt = protocol["proposal"]["prompt"]
    proposals: dict[str, np.ndarray] = {}
    receipts: dict[str, Any] = {}
    for key, image in images.items():
        detections, ground_receipt = proposal_base._ground_single(
            ground_processor,
            ground_model,
            image,
            prompt,
            image.size,
            torch,
        )
        detections.sort(key=lambda row: (-row["score"], *row["box_xyxy"], row["postprocess_index"]))
        require(bool(detections), f"NO_DOOR_PROPOSAL:{key}")
        selected = detections[0]
        masks, sam_receipt = proposal_base._sam_masks(
            sam_processor,
            sam_model,
            image,
            [selected["box_xyxy"]],
            image.size,
            torch,
            np,
        )
        require(len(masks) == 1, f"SAM_MASK_COUNT:{key}")
        proposal = np.ascontiguousarray(masks[0], dtype=np.bool_)
        proposals[key] = proposal
        receipts[key] = {
            "selection_authority": "HIGHEST_SCORE_SINGLE_DOOR_PROMPT_WITHOUT_TRUTH_ACCESS",
            "detected_box_count": len(detections),
            "selected_score": selected["score"],
            "selected_box_xyxy": selected["box_xyxy"],
            "grounder": ground_receipt,
            "masker": sam_receipt,
            **proposal_receipt(proposal, truths[key]),
        }
    runtime = {
        "grounder_model_type": type(ground_model).__name__,
        "masker_model_type": type(sam_model).__name__,
        "device": torch.cuda.get_device_name(0),
    }
    del ground_model, sam_model, ground_processor, sam_processor
    gc.collect()
    torch.cuda.empty_cache()
    return proposals, receipts, runtime


def replay(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    import romatch

    protocol = load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    receipt_path = HERE / protocol["source"]["rgb_receipt_path"]
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    cohort = load_json(cohort_path)
    receipt = load_json(receipt_path)
    predecessor = load_json(predecessor_path)
    intrinsic = roma_base.predecessor.parent.base.parse_intrinsic(
        roma_base.predecessor.source_paths(source_root, cohort["episodes"][0]["scene_id"])["intrinsic"]
    )
    images: dict[str, Image.Image] = {}
    truths: dict[str, np.ndarray] = {}
    inputs: dict[str, Any] = {}
    for episode in cohort["episodes"]:
        episode_id = str(episode["episode_id"])
        scene_id = str(episode["scene_id"])
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            sealed = receipt["sealed_frames"][f"{scene_id}:{frame}"]
            image_path = roma_base.predecessor.selected_image(source_root, scene_id, frame)
            require(sha256(image_path) == sealed["image_sha256"], f"IMAGE_HASH:{scene_id}:{frame}")
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            require(bgr is not None and bgr.shape == (480, 640, 3), f"IMAGE_FORMAT:{scene_id}:{frame}")
            key = f"{episode_id}:{role}"
            images[key] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            truths[key] = truth_mask(episode, role, source_root, intrinsic, protocol["renderer"])
            inputs[key] = {"scene_id": scene_id, "frame": frame, "image_sha256": sealed["image_sha256"]}

    proposals, proposal_receipts, proposal_runtime = make_proposals(protocol, images, truths)

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
    scores = np.zeros((len(episode_ids), len(episode_ids)), dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    for row, reference_id in enumerate(episode_ids):
        for column, query_id in enumerate(episode_ids):
            support = context_base.masked_roma_support(
                matcher_model,
                images[f"{reference_id}:reference"],
                images[f"{query_id}:query"],
                proposals[f"{reference_id}:reference"],
                proposals[f"{query_id}:query"],
                protocol["matcher"],
            )
            diagnostics[f"{reference_id}->{query_id}"] = support
            if support["absolute_support"]:
                scores[row, column] = float(support["symmetric_cycle_score"])

    target_index = {value: index for index, value in enumerate(episode_ids)}
    scenarios: list[dict[str, Any]] = []
    method = "grounded_sam_proposal_mask_gated_roma_reciprocal"
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
                "grounded_sam_proposal_mask_gated_roma_matrix": matrix.round(6).tolist(),
                "methods": {method: roma_base.predecessor.parent.open_zero.evaluate_matches(references, queries, matches)},
            }
        )
    aggregate = roma_base.predecessor.parent.open_zero.aggregate(scenarios, method)
    diagonal_supported = [bool(diagnostics[f"{episode_id}->{episode_id}"]["absolute_support"]) for episode_id in episode_ids]
    gate = protocol["decision_gate"]
    gate_met = (
        len(proposals) == int(gate["required_proposal_masks"])
        and sum(diagonal_supported) >= int(gate["minimum_true_pairs_with_absolute_support"])
        and aggregate["true_positive"] >= int(gate["minimum_true_positive"])
        and aggregate["false_positive"] == int(gate["maximum_false_positive"])
        and aggregate["false_negative"] == int(gate["maximum_false_negative"])
        and aggregate["zero_assignment_exact_scenarios"] >= int(gate["minimum_exact_scenarios"])
    )
    roma_base.predecessor.parent.write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_POSTHOC_TRUTH_BLIND_PROPOSAL_MASK_ROMA_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "source": {"cohort_path": cohort_path.name, "cohort_sha256": sha256(cohort_path), "rgb_receipt_path": receipt_path.name, "rgb_receipt_sha256": sha256(receipt_path), "source_root": str(source_root.resolve())},
            "conclusion": "L10_SCENENN_ROMA_GROUNDED_SAM_PROPOSAL_POSTHOC_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_ROMA_GROUNDED_SAM_PROPOSAL_POSTHOC_DEVELOPMENT_GATE_NOT_MET",
            "gate_met": gate_met,
            "metrics": {"aggregate": aggregate, "scenarios": scenarios, "full_matrix": scores.round(6).tolist(), "pair_diagnostics": diagnostics, "true_pair_absolute_support": diagonal_supported},
            "proposal_runtime": proposal_runtime,
            "proposal_receipts": proposal_receipts,
            "input_receipts": inputs,
            "comparison_to_exact_provider_mask": {"aggregate": predecessor["metrics"]["aggregate"], "true_pair_absolute_support": predecessor["metrics"]["true_pair_absolute_support"]},
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
