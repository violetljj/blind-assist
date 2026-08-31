#!/usr/bin/env python3
"""Provider-distinct 3RScan replay of frozen proposal-mask RoMa."""

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

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_roma_active_none as roma_base  # noqa: E402
import l10_scenenn_roma_full_context_mask_posthoc as context_base  # noqa: E402
import named_poi_grounded_sam_multiscale_part_topology as proposal_base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-roma-grounded-sam-proposal-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-roma-grounded-sam-proposal-posthoc-result-v1"


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


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(protocol["implementation"]["sha256"] == sha256(Path(__file__)), "IMPLEMENTATION_HASH")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    require(sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")
    predecessor_path = HERE / protocol["predecessor"]["result_path"]
    require(sha256(predecessor_path) == protocol["predecessor"]["result_sha256"], "PREDECESSOR_HASH")
    require(load_json(predecessor_path)["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    for dependency in protocol["dependencies"]:
        require(sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    for model in ("grounder", "masker", "matcher", "matcher_backbone"):
        row = protocol["models"][model]
        require(sha256(ROOT / row["path"]) == row["sha256"], f"MODEL_HASH:{model}")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["sequence_zips"]:
        path = artifact_root / row["path"]
        require(path.stat().st_size == int(row["bytes"]), f"ZIP_BYTES:{row['path']}")
        require(sha256(path) == row["sha256"], f"ZIP_HASH:{row['path']}")
    require(float(protocol["proposal"]["box_threshold"]) == proposal_base.BOX_THRESHOLD, "BOX_THRESHOLD")
    require(float(protocol["proposal"]["text_threshold"]) == proposal_base.TEXT_THRESHOLD, "TEXT_THRESHOLD")
    return protocol


def load_images(protocol: dict[str, Any], cohort: dict[str, Any]) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    images: dict[str, Image.Image] = {}
    receipts: dict[str, Any] = {}
    for row in cohort["images"].values():
        episode_id = str(row["episode_id"])
        role = str(row["role"])
        key = f"{episode_id}:{role}"
        scan_id = str(row["scan_id"])
        manifest_key = f"{scan_id}/sequence.zip"
        zip_row = cohort["source_manifest"][manifest_key]
        archive_path = artifact_root / zip_row["path"]
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        require(list(image.size) == row["color_size"], f"IMAGE_SIZE:{key}")
        images[key] = image
        receipts[key] = {
            "scan_id": scan_id,
            "frame": int(row["frame"]),
            "zip_member": row["zip_member"],
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "image_bytes": len(payload),
            "target_bbox_xyxy": row["bbox_xyxy"],
        }
    require(len(images) == 6, "IMAGE_COUNT")
    return images, receipts


def make_proposals(
    protocol: dict[str, Any],
    images: dict[str, Image.Image],
    inputs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

    ground_root = (ROOT / protocol["proposal"]["grounder_root"]).resolve()
    sam_root = (ROOT / protocol["proposal"]["masker_root"]).resolve()
    ground_processor = AutoProcessor.from_pretrained(ground_root, local_files_only=True)
    ground_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        ground_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    sam_processor = Sam2Processor.from_pretrained(sam_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        sam_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    prompt = protocol["proposal"]["prompt"]
    proposals: dict[str, np.ndarray] = {}
    receipts: dict[str, Any] = {}
    for key, image in images.items():
        detections, ground_receipt = proposal_base._ground_single(
            ground_processor, ground_model, image, prompt, image.size, torch
        )
        detections.sort(key=lambda row: (-row["score"], *row["box_xyxy"], row["postprocess_index"]))
        require(bool(detections), f"NO_DOOR_PROPOSAL:{key}")
        selected = detections[0]
        masks, sam_receipt = proposal_base._sam_masks(
            sam_processor, sam_model, image, [selected["box_xyxy"]], image.size, torch, np
        )
        require(len(masks) == 1, f"SAM_MASK_COUNT:{key}")
        proposal = np.ascontiguousarray(masks[0], dtype=np.bool_)
        require(int(proposal.sum()) > 0, f"EMPTY_PROPOSAL_MASK:{key}")
        iou, target_recall, proposal_precision = bbox_iou(selected["box_xyxy"], inputs[key]["target_bbox_xyxy"])
        proposals[key] = proposal
        receipts[key] = {
            "selection_authority": "HIGHEST_SCORE_SINGLE_DOOR_PROMPT_WITHOUT_PROVIDER_TARGET_ACCESS",
            "detected_box_count": len(detections),
            "selected_score": selected["score"],
            "selected_box_xyxy": selected["box_xyxy"],
            "target_bbox_iou_evaluation_only": iou,
            "target_bbox_recall_evaluation_only": target_recall,
            "proposal_bbox_precision_evaluation_only": proposal_precision,
            "proposal_mask_sha256": mask_sha256(proposal),
            "proposal_pixels": int(proposal.sum()),
            "grounder": ground_receipt,
            "masker": sam_receipt,
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


def replay(protocol_path: Path, output_path: Path) -> None:
    import romatch

    protocol = load_protocol(protocol_path)
    cohort_path = HERE / protocol["source"]["cohort_path"]
    cohort = load_json(cohort_path)
    images, inputs = load_images(protocol, cohort)
    proposals, proposal_receipts, proposal_runtime = make_proposals(protocol, images, inputs)

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
    episode_ids = [str(row["episode_id"]) for row in cohort["episodes"]]
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
    method = "grounded_sam_proposal_mask_gated_roma_reciprocal"
    scenarios: list[dict[str, Any]] = []
    for scenario in protocol["scenarios"]:
        references = scenario["reference_targets"]
        queries = scenario["query_targets"]
        matrix = scores[np.ix_([target_index[x] for x in references], [target_index[x] for x in queries])]
        matches = roma_base.predecessor.parent.open_zero.reciprocal_zero_assignment(matrix)
        scenarios.append(
            {
                **scenario,
                "proposal_mask_gated_roma_matrix": matrix.round(6).tolist(),
                "methods": {method: roma_base.predecessor.parent.open_zero.evaluate_matches(references, queries, matches)},
            }
        )
    aggregate = roma_base.predecessor.parent.open_zero.aggregate(scenarios, method)
    diagonal_supported = [bool(diagnostics[f"{episode_id}->{episode_id}"]["absolute_support"]) for episode_id in episode_ids]
    minimum_iou = min(float(row["target_bbox_iou_evaluation_only"]) for row in proposal_receipts.values())
    gate = protocol["decision_gate"]
    gate_met = (
        len(proposals) == int(gate["required_proposal_masks"])
        and minimum_iou >= float(gate["minimum_target_bbox_iou"])
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
            "authority": "CONSUMED_POSTHOC_PROVIDER_DISTINCT_MULTI_DOOR_PROPOSAL_ROMA_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "source": {"cohort_path": cohort_path.name, "cohort_sha256": sha256(cohort_path), "artifact_root": str((ROOT / protocol["source"]["artifact_root"]).resolve())},
            "conclusion": "L10_3RSCAN_ROMA_GROUNDED_SAM_PROPOSAL_POSTHOC_DEVELOPMENT_GATE_MET" if gate_met else "L10_3RSCAN_ROMA_GROUNDED_SAM_PROPOSAL_POSTHOC_DEVELOPMENT_GATE_NOT_MET",
            "gate_met": gate_met,
            "metrics": {"aggregate": aggregate, "scenarios": scenarios, "full_matrix": scores.round(6).tolist(), "pair_diagnostics": diagnostics, "true_pair_absolute_support": diagonal_supported, "minimum_target_bbox_iou": minimum_iou},
            "proposal_runtime": proposal_runtime,
            "proposal_receipts": proposal_receipts,
            "input_receipts": inputs,
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
