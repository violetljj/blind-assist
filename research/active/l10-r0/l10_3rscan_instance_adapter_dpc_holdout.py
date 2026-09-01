#!/usr/bin/env python3
"""Evaluate the frozen residual instance adapter on DPC proposal failures."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_instance_head_descriptor_materialize as feature  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-instance-adapter-dpc-holdout-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-instance-adapter-dpc-holdout-result-v1"


def _encode(images: list[Image.Image], boxes: list[list[float]], protocol: dict[str, Any], model: Any, torch: Any) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for image, box in zip(images, boxes, strict=True):
            crop, _ = feature._crop(image, box, float(protocol["descriptor"]["crop_expansion"]))
            tensor, _ = feature._tensor(crop, int(protocol["descriptor"]["input_size"]), torch)
            descriptor = model.forward_features(tensor.unsqueeze(0).to(protocol["runtime"]["device"]))["x_norm_clstoken"].float()
            values.append(torch.nn.functional.normalize(descriptor, dim=1)[0].cpu().numpy())
    return np.stack(values).astype(np.float32)


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    inputs = {}
    for key in ("cohort", "intermediate", "adapter_result"):
        row = protocol[key]
        path = HERE / row["path"]
        pixel.require(pixel.sha256(path) == row["sha256"], f"{key.upper()}_HASH")
        inputs[key] = pixel.load_json(path)
        pixel.require(inputs[key]["schema"] == row["required_schema"], f"{key.upper()}_SCHEMA")
    pixel.require(inputs["adapter_result"]["conclusion"] == protocol["adapter_result"]["required_conclusion"], "ADAPTER_CONCLUSION")

    artifact_root = ROOT / protocol["source"]["artifact_root"]
    model_path = ROOT / protocol["model"]["path"]
    pixel.require(pixel.sha256(model_path) == protocol["model"]["sha256"], "MODEL_HASH")
    adapter_path = artifact_root / inputs["adapter_result"]["artifact"]["path"]
    pixel.require(adapter_path.stat().st_size == int(inputs["adapter_result"]["artifact"]["bytes"]), "ADAPTER_BYTES")
    pixel.require(pixel.sha256(adapter_path) == inputs["adapter_result"]["artifact"]["sha256"], "ADAPTER_HASH")
    cohort = inputs["cohort"]
    intermediate = inputs["intermediate"]
    for manifest_key, receipt in cohort["source_manifest"].items():
        path = artifact_root / receipt["path"]
        pixel.require(path.stat().st_size == int(receipt["bytes"]), f"ZIP_BYTES:{manifest_key}")
        pixel.require(pixel.sha256(path) == receipt["sha256"], f"ZIP_HASH:{manifest_key}")

    import torch
    from romatch.models.transformer import vit_large

    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]),
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
    ).eval()
    model.load_state_dict(weights)
    model = model.to(protocol["runtime"]["device"])

    adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    adapter_spec = inputs["adapter_result"]["adapter"]

    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = int(adapter_spec["dimension"])
            rank = int(adapter_spec["rank"])
            self.norm = torch.nn.LayerNorm(dimension)
            self.down = torch.nn.Linear(dimension, rank, bias=False)
            self.up = torch.nn.Linear(rank, dimension, bias=False)
            self.scale = float(adapter_spec["residual_scale"])

        def forward(self, value: Any) -> Any:
            residual = self.up(torch.nn.functional.gelu(self.down(self.norm(value))))
            return torch.nn.functional.normalize(value + self.scale * residual, dim=-1)

    adapter = Adapter().eval()
    adapter.load_state_dict(adapter_payload["state_dict"])

    images: dict[str, Image.Image] = {}
    rgb_receipts = {}
    for key, row in cohort["images"].items():
        zip_receipt = cohort["source_manifest"][f"{row['scan_id']}/sequence.zip"]
        with zipfile.ZipFile(artifact_root / zip_receipt["path"]) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            images[key] = opened.convert("RGB")
        pixel.require(list(images[key].size) == row["color_size"], f"RGB_SIZE:{key}")
        rgb_receipts[key] = {
            "scan_id": row["scan_id"],
            "zip_member": row["zip_member"],
            "rgb_sha256": hashlib.sha256(payload).hexdigest(),
        }

    reference_keys = list(protocol["evaluation"]["reference_keys"])
    reference_descriptors = _encode(
        [images[key] for key in reference_keys],
        [cohort["images"][key]["bbox_xyxy"] for key in reference_keys],
        protocol,
        model,
        torch,
    )
    with torch.inference_mode():
        adapted_references = adapter(torch.from_numpy(reference_descriptors)).numpy()

    episodes = []
    for query_key in protocol["evaluation"]["query_keys"]:
        candidates = list(intermediate["query_receipts"][query_key]["ranked_candidates"])
        query_image = images[query_key]
        query_descriptors = _encode(
            [query_image] * len(candidates),
            [row["box_xyxy"] for row in candidates],
            protocol,
            model,
            torch,
        )
        with torch.inference_mode():
            adapted_queries = adapter(torch.from_numpy(query_descriptors)).numpy()
        baseline_scores = np.max(query_descriptors @ reference_descriptors.T, axis=1)
        adapted_scores = np.max(adapted_queries @ adapted_references.T, axis=1)
        truth_ious = np.asarray([float(row["target_metrics_evaluation_only"]["iou"]) for row in candidates])
        correct_index = int(np.argmax(truth_ious))

        def ranked(scores: np.ndarray) -> list[int]:
            return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))

        original_order = list(range(len(candidates)))
        baseline_order = ranked(baseline_scores)
        adapted_order = ranked(adapted_scores)
        original_rank = original_order.index(correct_index) + 1
        baseline_rank = baseline_order.index(correct_index) + 1
        adapted_rank = adapted_order.index(correct_index) + 1
        episodes.append(
            {
                "query_key": query_key,
                "candidate_count": len(candidates),
                "reachable_correct_candidate_index": correct_index,
                "reachable_correct_iou_evaluation_only": float(truth_ious[correct_index]),
                "original_fused_rank": original_rank,
                "frozen_dino_rank": baseline_rank,
                "adapted_rank": adapted_rank,
                "frozen_dino_top1_iou_evaluation_only": float(truth_ious[baseline_order[0]]),
                "adapted_top1_iou_evaluation_only": float(truth_ious[adapted_order[0]]),
                "frozen_dino_recall_at_3": baseline_rank <= 3,
                "adapted_recall_at_3": adapted_rank <= 3,
                "correct_candidate_frozen_score": float(baseline_scores[correct_index]),
                "correct_candidate_adapted_score": float(adapted_scores[correct_index]),
                "frozen_top1_score_margin_over_correct": float(baseline_scores[baseline_order[0]] - baseline_scores[correct_index]),
                "adapted_top1_score_margin_over_correct": float(adapted_scores[adapted_order[0]] - adapted_scores[correct_index]),
                "ranking": [
                    {
                        "candidate_index": index,
                        "truth_iou_evaluation_only": float(truth_ious[index]),
                        "frozen_dino_score": float(baseline_scores[index]),
                        "adapted_score": float(adapted_scores[index]),
                    }
                    for index in range(len(candidates))
                ],
            }
        )

    incremental_improvements = sum(row["adapted_rank"] < row["frozen_dino_rank"] for row in episodes)
    incremental_regressions = sum(row["adapted_rank"] > row["frozen_dino_rank"] for row in episodes)
    gate_met = (
        all(row["adapted_rank"] == 1 for row in episodes)
        and min(row["adapted_top1_iou_evaluation_only"] for row in episodes) >= float(protocol["gate"]["minimum_top1_iou"])
        and incremental_improvements >= int(protocol["gate"]["minimum_incremental_rank_improvements"])
        and incremental_regressions == 0
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "TARGET_FAMILY_DISJOINT_CONSUMED_DPC_PROPOSAL_HOLDOUT_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "cohort": protocol["cohort"],
        "intermediate": protocol["intermediate"],
        "adapter_result": protocol["adapter_result"],
        "descriptor": protocol["descriptor"],
        "rgb_receipts": rgb_receipts,
        "episodes": episodes,
        "metrics": {
            "query_count": len(episodes),
            "original_fused_top1": sum(row["original_fused_rank"] == 1 for row in episodes),
            "frozen_dino_top1": sum(row["frozen_dino_rank"] == 1 for row in episodes),
            "adapted_top1": sum(row["adapted_rank"] == 1 for row in episodes),
            "incremental_rank_improvements": incremental_improvements,
            "incremental_rank_regressions": incremental_regressions,
            "minimum_adapted_top1_iou": min(row["adapted_top1_iou_evaluation_only"] for row in episodes),
            "mean_adapted_top1_iou": float(np.mean([row["adapted_top1_iou_evaluation_only"] for row in episodes])),
        },
        "gate": {**protocol["gate"], "met": gate_met},
        "runtime": {
            "device": torch.cuda.get_device_name(torch.device(protocol["runtime"]["device"])),
            "rgb_members_opened": len(rgb_receipts),
            "feature_calls": len(reference_keys) + sum(row["candidate_count"] for row in episodes),
            "adapter_training_steps": 0,
        },
        "conclusion": (
            "L10_3RSCAN_INSTANCE_ADAPTER_DPC_HOLDOUT_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_INSTANCE_ADAPTER_DPC_HOLDOUT_DEVELOPMENT_GATE_NOT_MET"
        ),
        "next_action": protocol["next_action"] if gate_met else protocol["fallback_action"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    del model, weights
    torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
