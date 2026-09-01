#!/usr/bin/env python3
"""Train a small physical-door instance head on frozen DINOv2 descriptors."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-instance-head-train-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-instance-head-train-result-v1"


def _groups(manifest: dict[str, Any], descriptors: Any, torch: Any) -> dict[str, dict[str, Any]]:
    index = {str(sample_id): position for position, sample_id in enumerate(descriptors["sample_ids"].tolist())}
    matrix = torch.from_numpy(descriptors["descriptors"].astype(np.float32))
    sample_by_id = {row["sample_id"]: row for row in manifest["samples"]}
    groups: dict[str, dict[str, Any]] = {}
    for identity in manifest["identities"]:
        positives = []
        hard = None
        ordinary = None
        for sample_id in identity["sample_ids"]:
            sample = sample_by_id[sample_id]
            value = matrix[index[sample_id]]
            if sample["role"] == "positive":
                positives.append(value)
            elif sample["role"] == "same_scene_hard_negative":
                hard = value
            elif sample["role"] == "ordinary_negative":
                ordinary = value
        pixel.require(len(positives) >= 3 and hard is not None and ordinary is not None, f"IDENTITY_SAMPLES:{identity['identity_key']}")
        groups[identity["identity_key"]] = {
            "split": identity["split"],
            "positives": torch.stack(positives),
            "hard": hard,
            "ordinary": ordinary,
        }
    return groups


def _project_groups(groups: dict[str, dict[str, Any]], projector: Any | None, torch: Any) -> dict[str, dict[str, Any]]:
    projected = {}
    for key, row in groups.items():
        if projector is None:
            positives = torch.nn.functional.normalize(row["positives"], dim=1)
            hard = torch.nn.functional.normalize(row["hard"], dim=0)
            ordinary = torch.nn.functional.normalize(row["ordinary"], dim=0)
        else:
            positives = projector(row["positives"])
            hard = projector(row["hard"].unsqueeze(0))[0]
            ordinary = projector(row["ordinary"].unsqueeze(0))[0]
        projected[key] = {"split": row["split"], "positives": positives, "hard": hard, "ordinary": ordinary}
    return projected


def _metrics(groups: dict[str, dict[str, Any]], split: str, torch: Any) -> dict[str, Any]:
    positive_hard_gaps: list[float] = []
    hard_ordinary_gaps: list[float] = []
    hierarchy: list[bool] = []
    ranking: list[bool] = []
    identity_rows = []
    for identity_key in sorted(groups):
        row = groups[identity_key]
        if row["split"] != split:
            continue
        positives = row["positives"]
        hard = row["hard"]
        ordinary = row["ordinary"]
        local_ph = []
        local_ho = []
        local_ranking = []
        for anchor_index in range(len(positives)):
            anchor = positives[anchor_index]
            candidate_positive = torch.cat((positives[:anchor_index], positives[anchor_index + 1 :]), dim=0)
            positive_score = float(torch.max(candidate_positive @ anchor))
            hard_score = float(hard @ anchor)
            ordinary_score = float(ordinary @ anchor)
            ph_gap = positive_score - hard_score
            ho_gap = hard_score - ordinary_score
            local_ph.append(ph_gap)
            local_ho.append(ho_gap)
            local_ranking.append(positive_score > max(hard_score, ordinary_score))
            positive_hard_gaps.append(ph_gap)
            hard_ordinary_gaps.append(ho_gap)
            hierarchy.append(ph_gap > 0.0 and ho_gap > 0.0)
            ranking.append(local_ranking[-1])
        identity_rows.append(
            {
                "identity_key": identity_key,
                "positive_anchor_count": len(positives),
                "ranking_accuracy": float(np.mean(local_ranking)),
                "mean_positive_hard_gap": float(np.mean(local_ph)),
                "mean_hard_ordinary_gap": float(np.mean(local_ho)),
            }
        )
    pixel.require(positive_hard_gaps, f"EMPTY_METRIC_SPLIT:{split}")
    return {
        "identity_count": len(identity_rows),
        "anchor_count": len(positive_hard_gaps),
        "ranking_accuracy": float(np.mean(ranking)),
        "positive_hard_accuracy": float(np.mean(np.asarray(positive_hard_gaps) > 0.0)),
        "hierarchy_accuracy": float(np.mean(hierarchy)),
        "mean_positive_hard_gap": float(np.mean(positive_hard_gaps)),
        "mean_hard_ordinary_gap": float(np.mean(hard_ordinary_gaps)),
        "identities": identity_rows,
    }


def _loss(groups: dict[str, dict[str, Any]], projector: Any, protocol: dict[str, Any], torch: Any) -> Any:
    losses = []
    margin_ph = float(protocol["loss"]["positive_hard_margin"])
    margin_ho = float(protocol["loss"]["hard_ordinary_margin"])
    for identity_key in sorted(groups):
        row = groups[identity_key]
        if row["split"] != "train":
            continue
        positives = projector(row["positives"])
        hard = projector(row["hard"].unsqueeze(0))[0]
        ordinary = projector(row["ordinary"].unsqueeze(0))[0]
        for anchor_index in range(len(positives)):
            anchor = positives[anchor_index]
            positive_candidates = torch.cat((positives[:anchor_index], positives[anchor_index + 1 :]), dim=0)
            positive_score = torch.min(positive_candidates @ anchor)
            hard_score = hard @ anchor
            ordinary_score = ordinary @ anchor
            losses.append(torch.relu(margin_ph - (positive_score - hard_score)))
            losses.append(float(protocol["loss"]["hard_ordinary_weight"]) * torch.relu(margin_ho - (hard_score - ordinary_score)))
    return torch.stack(losses).mean()


def _atomic_torch_save(path: Path, payload: dict[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(protocol_path: Path, head_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("manifest", "descriptor_result"):
        row = protocol[key]
        path = HERE / row["path"]
        pixel.require(pixel.sha256(path) == row["sha256"], f"{key.upper()}_HASH")
        pixel.require(pixel.load_json(path)["schema"] == row["required_schema"], f"{key.upper()}_SCHEMA")
    manifest = pixel.load_json(HERE / protocol["manifest"]["path"])
    descriptor_result = pixel.load_json(HERE / protocol["descriptor_result"]["path"])
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    descriptor_path = artifact_root / descriptor_result["artifact"]["path"]
    pixel.require(descriptor_path.stat().st_size == int(descriptor_result["artifact"]["bytes"]), "DESCRIPTOR_BYTES")
    pixel.require(pixel.sha256(descriptor_path) == descriptor_result["artifact"]["sha256"], "DESCRIPTOR_HASH")

    import torch

    seed = int(protocol["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    descriptors = np.load(descriptor_path, allow_pickle=False)
    groups = _groups(manifest, descriptors, torch)

    class Projector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.LayerNorm(int(protocol["head"]["input_dimension"])),
                torch.nn.Linear(int(protocol["head"]["input_dimension"]), int(protocol["head"]["hidden_dimension"])),
                torch.nn.GELU(),
                torch.nn.Linear(int(protocol["head"]["hidden_dimension"]), int(protocol["head"]["output_dimension"])),
            )

        def forward(self, value: Any) -> Any:
            return torch.nn.functional.normalize(self.network(value), dim=-1)

    baseline = {
        split: _metrics(_project_groups(groups, None, torch), split, torch)
        for split in ("train", "validation")
    }
    projector = Projector()
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    history = []
    best_key = None
    best_epoch = None
    best_state = None
    best_validation = None
    for epoch in range(1, int(protocol["training"]["epochs"]) + 1):
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(groups, projector, protocol, torch)
        pixel.require(torch.isfinite(loss).item(), f"NONFINITE_LOSS:{epoch}")
        loss.backward()
        optimizer.step()
        if epoch % int(protocol["training"]["validation_interval_epochs"]) != 0:
            continue
        projector.eval()
        with torch.inference_mode():
            projected = _project_groups(groups, projector, torch)
            validation = _metrics(projected, "validation", torch)
            training = _metrics(projected, "train", torch)
        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "validation": {key: value for key, value in validation.items() if key != "identities"},
            "training": {key: value for key, value in training.items() if key != "identities"},
        }
        history.append(row)
        selection_key = (
            validation["ranking_accuracy"],
            validation["positive_hard_accuracy"],
            validation["hierarchy_accuracy"],
            validation["mean_positive_hard_gap"],
            validation["mean_hard_ordinary_gap"],
            -epoch,
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch
            best_state = deepcopy(projector.state_dict())
            best_validation = validation
    pixel.require(best_state is not None and best_epoch is not None and best_validation is not None, "NO_BEST_EPOCH")
    projector.load_state_dict(best_state)
    projector.eval()
    with torch.inference_mode():
        final_projected = _project_groups(groups, projector, torch)
        projected_metrics = {
            split: _metrics(final_projected, split, torch) for split in ("train", "validation")
        }
    gate = protocol["gate"]
    gate_met = (
        projected_metrics["validation"]["ranking_accuracy"]
        >= baseline["validation"]["ranking_accuracy"] + float(gate["minimum_ranking_accuracy_gain"])
        and projected_metrics["validation"]["mean_positive_hard_gap"]
        > baseline["validation"]["mean_positive_hard_gap"]
        and projected_metrics["validation"]["mean_hard_ordinary_gap"]
        > baseline["validation"]["mean_hard_ordinary_gap"]
    )
    artifact_payload = {
        "schema": "blindassist-l10-3rscan-instance-head-weights-v1",
        "protocol_sha256": pixel.sha256(protocol_path),
        "best_epoch": best_epoch,
        "architecture": protocol["head"],
        "state_dict": best_state,
    }
    _atomic_torch_save(head_path, artifact_payload, torch)
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "TARGET_DISJOINT_TRAIN_VALIDATION_INSTANCE_HEAD_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "manifest": protocol["manifest"],
        "descriptor_result": protocol["descriptor_result"],
        "head": protocol["head"],
        "loss": protocol["loss"],
        "training": protocol["training"],
        "selection": protocol["selection"],
        "baseline": baseline,
        "best_epoch": best_epoch,
        "projected": projected_metrics,
        "validation_delta": {
            "ranking_accuracy": projected_metrics["validation"]["ranking_accuracy"] - baseline["validation"]["ranking_accuracy"],
            "positive_hard_accuracy": projected_metrics["validation"]["positive_hard_accuracy"] - baseline["validation"]["positive_hard_accuracy"],
            "hierarchy_accuracy": projected_metrics["validation"]["hierarchy_accuracy"] - baseline["validation"]["hierarchy_accuracy"],
            "mean_positive_hard_gap": projected_metrics["validation"]["mean_positive_hard_gap"] - baseline["validation"]["mean_positive_hard_gap"],
            "mean_hard_ordinary_gap": projected_metrics["validation"]["mean_hard_ordinary_gap"] - baseline["validation"]["mean_hard_ordinary_gap"],
        },
        "history": history,
        "artifact": {
            "path": head_path.resolve().relative_to(artifact_root.resolve()).as_posix(),
            "bytes": head_path.stat().st_size,
            "sha256": pixel.sha256(head_path),
        },
        "gate": {**gate, "met": gate_met},
        "heldout_rgb_members_opened": 0,
        "heldout_score_comparisons": 0,
        "conclusion": (
            "L10_3RSCAN_INSTANCE_HEAD_TRAIN_VALIDATION_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_INSTANCE_HEAD_TRAIN_VALIDATION_DEVELOPMENT_GATE_NOT_MET"
        ),
        "next_action": protocol["next_action"] if gate_met else protocol["fallback_action"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.head.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
