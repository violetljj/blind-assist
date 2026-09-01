#!/usr/bin/env python3
"""Train an identity-preserving residual metric adapter on frozen descriptors."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_instance_head_train as base  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-instance-residual-adapter-train-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-instance-residual-adapter-train-result-v1"


def _preservation_loss(groups: dict[str, dict[str, Any]], adapter: Any, torch: Any) -> Any:
    rows = []
    for identity_key in sorted(groups):
        row = groups[identity_key]
        if row["split"] != "train":
            continue
        values = torch.cat((row["positives"], row["hard"].unsqueeze(0), row["ordinary"].unsqueeze(0)), dim=0)
        adapted = adapter(values)
        original = torch.nn.functional.normalize(values, dim=1)
        rows.append(1.0 - torch.sum(adapted * original, dim=1))
    return torch.cat(rows).mean()


def _perturbation(groups: dict[str, dict[str, Any]], adapter: Any, split: str, torch: Any) -> dict[str, float]:
    cosine = []
    for identity_key in sorted(groups):
        row = groups[identity_key]
        if row["split"] != split:
            continue
        values = torch.cat((row["positives"], row["hard"].unsqueeze(0), row["ordinary"].unsqueeze(0)), dim=0)
        adapted = adapter(values)
        original = torch.nn.functional.normalize(values, dim=1)
        cosine.extend(torch.sum(adapted * original, dim=1).tolist())
    return {
        "mean_cosine_to_frozen": float(np.mean(cosine)),
        "minimum_cosine_to_frozen": float(np.min(cosine)),
        "maximum_one_minus_cosine": float(np.max(1.0 - np.asarray(cosine))),
    }


def run(protocol_path: Path, adapter_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    dependency = protocol["dependency"]
    pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], "DEPENDENCY_HASH")
    for key in ("manifest", "descriptor_result", "predecessor"):
        row = protocol[key]
        path = HERE / row["path"]
        pixel.require(pixel.sha256(path) == row["sha256"], f"{key.upper()}_HASH")
        payload = pixel.load_json(path)
        pixel.require(payload["schema"] == row["required_schema"], f"{key.upper()}_SCHEMA")
    predecessor = pixel.load_json(HERE / protocol["predecessor"]["path"])
    pixel.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
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
    groups = base._groups(manifest, descriptors, torch)

    class ResidualAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimension = int(protocol["adapter"]["dimension"])
            rank = int(protocol["adapter"]["rank"])
            self.norm = torch.nn.LayerNorm(dimension)
            self.down = torch.nn.Linear(dimension, rank, bias=False)
            self.up = torch.nn.Linear(rank, dimension, bias=False)
            torch.nn.init.zeros_(self.up.weight)
            self.scale = float(protocol["adapter"]["residual_scale"])

        def forward(self, value: Any) -> Any:
            residual = self.up(torch.nn.functional.gelu(self.down(self.norm(value))))
            return torch.nn.functional.normalize(value + self.scale * residual, dim=-1)

    baseline_projected = base._project_groups(groups, None, torch)
    baseline = {split: base._metrics(baseline_projected, split, torch) for split in ("train", "validation")}
    adapter = ResidualAdapter()
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    best_epoch = 0
    best_state = deepcopy(adapter.state_dict())
    best_validation = baseline["validation"]
    best_key = (
        best_validation["ranking_accuracy"],
        best_validation["positive_hard_accuracy"],
        best_validation["hierarchy_accuracy"],
        best_validation["mean_positive_hard_gap"],
        best_validation["mean_hard_ordinary_gap"],
        0,
    )
    history = [
        {
            "epoch": 0,
            "loss": None,
            "ranking_loss": None,
            "preservation_loss": 0.0,
            "validation": {key: value for key, value in baseline["validation"].items() if key != "identities"},
        }
    ]
    for epoch in range(1, int(protocol["training"]["epochs"]) + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        ranking_loss = base._loss(groups, adapter, protocol, torch)
        preservation_loss = _preservation_loss(groups, adapter, torch)
        loss = ranking_loss + float(protocol["loss"]["preservation_weight"]) * preservation_loss
        pixel.require(torch.isfinite(loss).item(), f"NONFINITE_LOSS:{epoch}")
        loss.backward()
        optimizer.step()
        if epoch % int(protocol["training"]["validation_interval_epochs"]) != 0:
            continue
        adapter.eval()
        with torch.inference_mode():
            projected = base._project_groups(groups, adapter, torch)
            validation = base._metrics(projected, "validation", torch)
            training = base._metrics(projected, "train", torch)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "ranking_loss": float(ranking_loss.detach()),
                "preservation_loss": float(preservation_loss.detach()),
                "validation": {key: value for key, value in validation.items() if key != "identities"},
                "training": {key: value for key, value in training.items() if key != "identities"},
            }
        )
        key = (
            validation["ranking_accuracy"],
            validation["positive_hard_accuracy"],
            validation["hierarchy_accuracy"],
            validation["mean_positive_hard_gap"],
            validation["mean_hard_ordinary_gap"],
            -epoch,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = deepcopy(adapter.state_dict())
            best_validation = validation
    adapter.load_state_dict(best_state)
    adapter.eval()
    with torch.inference_mode():
        projected_groups = base._project_groups(groups, adapter, torch)
        projected = {split: base._metrics(projected_groups, split, torch) for split in ("train", "validation")}
        perturbation = {split: _perturbation(groups, adapter, split, torch) for split in ("train", "validation")}
    gate = protocol["gate"]
    gate_met = (
        projected["validation"]["ranking_accuracy"] >= baseline["validation"]["ranking_accuracy"]
        and projected["validation"]["mean_positive_hard_gap"] >= baseline["validation"]["mean_positive_hard_gap"]
        and projected["validation"]["mean_hard_ordinary_gap"] > baseline["validation"]["mean_hard_ordinary_gap"]
        and perturbation["validation"]["maximum_one_minus_cosine"] <= float(gate["maximum_validation_one_minus_cosine"])
    )
    payload = {
        "schema": "blindassist-l10-3rscan-instance-residual-adapter-weights-v1",
        "protocol_sha256": pixel.sha256(protocol_path),
        "best_epoch": best_epoch,
        "architecture": protocol["adapter"],
        "state_dict": best_state,
    }
    base._atomic_torch_save(adapter_path, payload, torch)
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "TARGET_DISJOINT_IDENTITY_PRESERVING_RESIDUAL_ADAPTER_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "predecessor": protocol["predecessor"],
        "manifest": protocol["manifest"],
        "descriptor_result": protocol["descriptor_result"],
        "literature_motivation": protocol["literature_motivation"],
        "adapter": protocol["adapter"],
        "loss": protocol["loss"],
        "training": protocol["training"],
        "selection": protocol["selection"],
        "baseline": baseline,
        "best_epoch": best_epoch,
        "projected": projected,
        "perturbation": perturbation,
        "validation_delta": {
            "ranking_accuracy": projected["validation"]["ranking_accuracy"] - baseline["validation"]["ranking_accuracy"],
            "positive_hard_accuracy": projected["validation"]["positive_hard_accuracy"] - baseline["validation"]["positive_hard_accuracy"],
            "hierarchy_accuracy": projected["validation"]["hierarchy_accuracy"] - baseline["validation"]["hierarchy_accuracy"],
            "mean_positive_hard_gap": projected["validation"]["mean_positive_hard_gap"] - baseline["validation"]["mean_positive_hard_gap"],
            "mean_hard_ordinary_gap": projected["validation"]["mean_hard_ordinary_gap"] - baseline["validation"]["mean_hard_ordinary_gap"],
        },
        "history": history,
        "artifact": {
            "path": adapter_path.resolve().relative_to(artifact_root.resolve()).as_posix(),
            "bytes": adapter_path.stat().st_size,
            "sha256": pixel.sha256(adapter_path),
        },
        "gate": {**gate, "met": gate_met},
        "heldout_rgb_members_opened": 0,
        "heldout_score_comparisons": 0,
        "conclusion": (
            "L10_3RSCAN_INSTANCE_RESIDUAL_ADAPTER_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_INSTANCE_RESIDUAL_ADAPTER_DEVELOPMENT_GATE_NOT_MET"
        ),
        "next_action": protocol["next_action"] if gate_met else protocol["fallback_action"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.adapter.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
