#!/usr/bin/env python3
"""Train the official NIDS weight adapter off-family and evaluate held-out D15."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_nids_local_appearance_small_tile_posthoc as nids  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-heldout-weight-adapter-nids-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-heldout-weight-adapter-nids-posthoc-result-v1"


class WeightAdapter(nn.Module):
    """Exact two-layer multiplicative adapter architecture published by NIDS-Net."""

    def __init__(self, channels: int, reduction: int, scalar: float) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.ReLU(inplace=True),
        )
        self.scalar = scalar

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scaled = self.scalar * inputs
        return self.fc(scaled).sigmoid() * scaled


def _info_nce(
    features: torch.Tensor, labels: torch.Tensor, temperature: float, epsilon: float
) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    similarities = features @ features.transpose(0, 1)
    labels = labels.contiguous().view(-1, 1)
    positives = torch.eq(labels, labels.transpose(0, 1)).float()
    negatives = 1.0 - positives
    positives.fill_diagonal_(0.0)
    logits = torch.exp(similarities / temperature)
    positive_sum = torch.sum(logits * positives, dim=1, keepdim=True)
    negative_sum = torch.sum(logits * negatives, dim=1, keepdim=True) + epsilon
    return (-torch.log(positive_sum / (positive_sum + negative_sum) + epsilon)).mean()


def _load_bank_images(
    protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, int]]:
    images: dict[str, Any] = {}
    image_rows: dict[str, dict[str, Any]] = {}
    labels: dict[str, int] = {}
    for label, family in enumerate(protocol["training"]["families"]):
        cohort_path = HERE / family["cohort_path"]
        nids.pixel.require(
            nids.pixel.sha256(cohort_path) == family["cohort_sha256"],
            f"TRAIN_COHORT_HASH:{family['family_id']}",
        )
        cohort = nids.pixel.load_json(cohort_path)
        pseudo_protocol = {
            "source": {"artifact_root": protocol["source"]["artifact_root"]},
            "memory": {"target_images": family["images"], "sibling_images": []},
            "evaluation": {"query_images": []},
        }
        family_images, family_rows = nids.ffa._load_images(pseudo_protocol, cohort)
        for key in family["images"]:
            unique_key = f"{family['family_id']}::{key}"
            images[unique_key] = family_images[key]
            image_rows[unique_key] = family_rows[key]
            labels[unique_key] = label
    return images, image_rows, labels


def _adapter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = nids.pixel.load_json(protocol_path)
    nids.pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    nids.pixel.require(
        nids.pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for dependency in protocol["dependencies"]:
        nids.pixel.require(
            nids.pixel.sha256(HERE / dependency["path"]) == dependency["sha256"],
            f"DEPENDENCY_HASH:{dependency['path']}",
        )
    predecessor_path = HERE / protocol["predecessor"]["path"]
    nids.pixel.require(
        nids.pixel.sha256(predecessor_path) == protocol["predecessor"]["sha256"],
        "PREDECESSOR_HASH",
    )
    predecessor = nids.pixel.load_json(predecessor_path)
    nids.pixel.require(
        predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    holdout_path = HERE / protocol["source"]["holdout_cohort_path"]
    nids.pixel.require(
        nids.pixel.sha256(holdout_path) == protocol["source"]["holdout_cohort_sha256"],
        "HOLDOUT_COHORT_HASH",
    )
    for section in ("proposal", "descriptor", "masker"):
        row = protocol[section]
        nids.pixel.require(
            nids.pixel.sha256(ROOT / row["model_path"]) == row["model_sha256"],
            f"MODEL_HASH:{section}",
        )

    training_images, training_rows, training_labels = _load_bank_images(protocol)
    holdout_cohort = nids.pixel.load_json(holdout_path)
    holdout_images, holdout_rows = nids.ffa._load_images(protocol, holdout_cohort)
    proposals, proposal_runtime = nids.tiled._tiled_proposals(protocol, holdout_images)

    from transformers import Sam2Model, Sam2Processor

    masker_root = (ROOT / protocol["masker"]["model_root"]).resolve()
    sam_processor = Sam2Processor.from_pretrained(masker_root, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(
        masker_root, local_files_only=True, use_safetensors=True, dtype=torch.float32
    ).eval().to("cuda:0")
    training_masks: dict[str, np.ndarray] = {}
    training_mask_receipts: dict[str, Any] = {}
    for key in sorted(training_images):
        masks, receipt = nids.sam_base._sam_masks(
            sam_processor,
            sam_model,
            training_images[key],
            [training_rows[key]["bbox_xyxy"]],
            training_images[key].size,
            torch,
            np,
        )
        training_masks[key] = np.ascontiguousarray(masks[0], dtype=np.bool_)
        training_mask_receipts[key] = receipt
    memory_masks: dict[str, np.ndarray] = {}
    memory_mask_receipts: dict[str, Any] = {}
    for key in protocol["memory"]["target_images"]:
        masks, receipt = nids.sam_base._sam_masks(
            sam_processor,
            sam_model,
            holdout_images[key],
            [holdout_rows[key]["bbox_xyxy"]],
            holdout_images[key].size,
            torch,
            np,
        )
        memory_masks[key] = np.ascontiguousarray(masks[0], dtype=np.bool_)
        memory_mask_receipts[key] = receipt
    query_masks: dict[str, list[np.ndarray]] = {}
    query_mask_receipts: dict[str, Any] = {}
    for key in protocol["evaluation"]["query_images"]:
        masks, receipt = nids.sam_base._sam_masks(
            sam_processor,
            sam_model,
            holdout_images[key],
            [row["box_xyxy"] for row in proposals[key]],
            holdout_images[key].size,
            torch,
            np,
        )
        query_masks[key] = [np.ascontiguousarray(mask, dtype=np.bool_) for mask in masks]
        query_mask_receipts[key] = receipt
    sam_model_type = type(sam_model).__name__
    del sam_model, sam_processor
    gc.collect()
    torch.cuda.empty_cache()

    from romatch.models.transformer import vit_large

    weights = torch.load(
        ROOT / protocol["descriptor"]["model_path"], map_location="cpu", weights_only=True
    )
    descriptor_model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]),
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
    ).eval()
    descriptor_model.load_state_dict(weights)
    descriptor_model = descriptor_model.to("cuda:0")
    training_descriptors: dict[str, np.ndarray] = {}
    for key in sorted(training_images):
        descriptor, _ = nids._foreground_representation(
            descriptor_model,
            training_images[key],
            training_masks[key],
            training_rows[key]["bbox_xyxy"],
            protocol,
        )
        training_descriptors[key] = descriptor

    training_keys = sorted(training_descriptors)
    training_tensor = torch.from_numpy(
        np.stack([training_descriptors[key] for key in training_keys])
    ).float()
    label_tensor = torch.tensor([training_labels[key] for key in training_keys], dtype=torch.long)
    settings = protocol["training"]["optimizer"]
    torch.manual_seed(int(settings["seed"]))
    adapter = WeightAdapter(
        channels=int(training_tensor.shape[1]),
        reduction=int(settings["reduction"]),
        scalar=float(settings["scalar"]),
    )
    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    losses: list[float] = []
    for _ in range(int(settings["epochs"])):
        optimizer.zero_grad()
        adapted = adapter(training_tensor)
        loss = _info_nce(
            adapted,
            label_tensor,
            float(settings["temperature"]),
            float(settings["epsilon"]),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    adapter.eval()

    target_memory: dict[str, tuple[np.ndarray, torch.Tensor]] = {}
    for key in protocol["memory"]["target_images"]:
        target_memory[key] = nids._foreground_representation(
            descriptor_model,
            holdout_images[key],
            memory_masks[key],
            holdout_rows[key]["bbox_xyxy"],
            protocol,
        )
    with torch.inference_mode():
        adapted_target_memory = {
            key: F.normalize(adapter(torch.from_numpy(value[0]).float().unsqueeze(0))[0], dim=0)
            for key, value in target_memory.items()
        }

    minimum = float(protocol["decision_gate"]["minimum_iou"])
    opportunities = 0
    successes = 0
    refined_ious: list[float] = []
    receipts: dict[str, Any] = {}
    representation_calls = len(training_descriptors) + len(target_memory)
    empty_mask_candidates = 0
    for key in protocol["evaluation"]["query_images"]:
        truth = holdout_rows[key]["bbox_xyxy"]
        candidates: list[dict[str, Any]] = []
        for source, mask in zip(proposals[key], query_masks[key]):
            try:
                descriptor, query_patches = nids._foreground_representation(
                    descriptor_model, holdout_images[key], mask, source["box_xyxy"], protocol
                )
            except ValueError as error:
                if str(error) not in {"EMPTY_PATCH_MASK", "EMPTY_BINARY_PATCH_MASK"}:
                    raise
                empty_mask_candidates += 1
                continue
            representation_calls += 1
            with torch.inference_mode():
                adapted_query = F.normalize(
                    adapter(torch.from_numpy(descriptor).float().unsqueeze(0))[0], dim=0
                )
            per_reference = {
                name: float(torch.dot(reference, adapted_query).item())
                for name, reference in adapted_target_memory.items()
            }
            winning_reference = sorted(
                per_reference, key=lambda name: (-per_reference[name], name)
            )[0]
            instance_score = per_reference[winning_reference]
            local_score = nids._appearance_score(
                query_patches, target_memory[winning_reference][1]
            )
            final_score = float((instance_score + local_score) / 2.0)
            evaluation = nids.base._bbox_metrics(source["box_xyxy"], truth)
            candidates.append(
                {
                    **source,
                    "adapted_target_memory_score": instance_score,
                    "per_reference_adapted_scores": per_reference,
                    "winning_target_reference": winning_reference,
                    "local_appearance_score": local_score,
                    "adapted_nids_fused_score": final_score,
                    "target_metrics_evaluation_only": evaluation,
                    "descriptor_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest(),
                    "mask_sha256": hashlib.sha256(
                        mask.astype(np.uint8).tobytes(order="C")
                    ).hexdigest(),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["adapted_nids_fused_score"],
                -row["adapted_target_memory_score"],
                -row["objectness_score"],
                *row["box_xyxy"],
                row["postprocess_index"],
            )
        )
        nids.pixel.require(bool(candidates), f"NO_NONEMPTY_MASK_CANDIDATE:{key}")
        best_reachable = max(
            float(row["target_metrics_evaluation_only"]["iou"]) for row in candidates
        )
        opportunity = best_reachable >= minimum
        opportunities += int(opportunity)
        selected = candidates[0]
        selected_index = next(
            index
            for index, source in enumerate(proposals[key])
            if int(source["postprocess_index"]) == int(selected["postprocess_index"])
            and source["box_xyxy"] == selected["box_xyxy"]
        )
        refined_box = nids.refine._tight_bbox(query_masks[key][selected_index])
        refined = nids.base._bbox_metrics(refined_box, truth)
        refined_iou = float(refined["iou"])
        refined_ious.append(refined_iou)
        successes += int(refined_iou >= minimum)
        receipts[key] = {
            "query_truth_used_for_selection": False,
            "target_opportunity_at_iou_gate": opportunity,
            "best_reachable_target_iou_evaluation_only": best_reachable,
            "selected": selected,
            "top1_adapted_nids_fused_margin": float(
                selected["adapted_nids_fused_score"]
                - candidates[1]["adapted_nids_fused_score"]
            ),
            "refined_mask_bbox_xyxy": refined_box,
            "refined_mask_bbox_target_metrics_evaluation_only": refined,
            "ranked_candidates": candidates,
            "query_sam_receipt": query_mask_receipts[key],
        }

    del descriptor_model, weights
    gc.collect()
    torch.cuda.empty_cache()
    gate_met = (
        opportunities == int(protocol["decision_gate"]["required_opportunity_queries"])
        and successes == int(protocol["decision_gate"]["required_refined_queries"])
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_D15_HELDOUT_WEIGHT_ADAPTER_NIDS_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": nids.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": nids.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_HELDOUT_WEIGHT_ADAPTER_NIDS_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_HELDOUT_WEIGHT_ADAPTER_NIDS_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(receipts),
            "opportunity_queries": opportunities,
            "refined_iou_gate_queries": successes,
            "minimum_refined_iou": min(refined_ious),
            "mean_refined_iou": float(np.mean(refined_ious)),
        },
        "adapter_receipt": {
            "training_keys": training_keys,
            "labels": [int(value) for value in label_tensor.tolist()],
            "training_descriptor_sha256": {
                key: hashlib.sha256(training_descriptors[key].tobytes()).hexdigest()
                for key in training_keys
            },
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "losses": losses,
            "adapter_tensor_sha256": _adapter_hash(adapter),
            "holdout_family": protocol["source"]["holdout_family"],
        },
        "query_receipts": receipts,
        "memory_receipts": {
            "target": {
                key: hashlib.sha256(value[0].tobytes()).hexdigest()
                for key, value in target_memory.items()
            },
            "training_sam": training_mask_receipts,
            "holdout_sam": memory_mask_receipts,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "adapter_training_device": "cpu",
            "proposal": proposal_runtime,
            "sam_model_type": sam_model_type,
            "grounding_dino_calls": proposal_runtime["grounding_dino_calls"],
            "sam2_image_calls": len(training_masks) + len(memory_masks) + len(query_masks),
            "dinov2_representation_calls": representation_calls,
            "empty_mask_candidates_skipped": empty_mask_candidates,
        },
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    nids.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
