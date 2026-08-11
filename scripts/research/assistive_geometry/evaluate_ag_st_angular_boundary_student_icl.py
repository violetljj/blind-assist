#!/usr/bin/env python3
"""Evaluate the frozen R16 angular boundary specialist on checkpoint-unseen ICL exact geometry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from download_b0_arkitscenes_assets import require, sha256_file
from train_ag_st_source_boundary_student import (
    DEFAULT_ANGULAR_LABEL_RESULT,
    DEFAULT_BINDING,
    DEFAULT_MOBILENET,
    TarImageReader,
    _load_bound_rgb,
    _load_target,
    _source_metrics,
    bind_label_package,
    build_decoder,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-angular-boundary-student-r1/boundary-decoder.pt"
)
DEFAULT_R15_BASELINE = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-unified-factor-student-depthart-multisource-r0/icl-exact-boundary.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-angular-boundary-student-r1/icl-exact-boundary.json"
)


def checkpoint_parent_sources(split: dict[str, list[tuple[str, str]]]) -> set[str]:
    return {str(source) for values in split.values() for source, _ in values}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torchvision.models import mobilenet_v3_small

    started = time.monotonic()
    for path in (
        args.binding,
        args.angular_label_result,
        args.checkpoint,
        args.mobilenet_checkpoint,
        args.r15_baseline,
    ):
        require(path.is_file(), f"input missing: {path}")
    require(not args.output.exists(), f"output exists: {args.output}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(checkpoint.get("schema") == "blindassist_ag_st_source_boundary_decoder_v1", "checkpoint schema drift")
    require(checkpoint.get("target_mode") == "angular", "checkpoint is not angular-supervised")
    require(
        checkpoint["mobilenet_checkpoint_sha256"] == sha256_file(args.mobilenet_checkpoint),
        "MobileNet checkpoint digest drift",
    )
    require(
        checkpoint["label_contract"]["result_sha256"] == sha256_file(args.angular_label_result),
        "angular label package digest drift",
    )
    training_sources = checkpoint_parent_sources(checkpoint["split"])
    require("icl_exact" not in training_sources, "ICL was consumed by checkpoint selection or training")

    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    require(binding.get("status") == "SOURCE_NATIVE_BOUNDARY_RGB_BINDING_PASS", "RGB binding invalid")
    descriptors, label_contract = bind_label_package(
        list(binding["frames"]),
        sources=("icl_exact",),
        target_mode="angular",
        label_result=args.angular_label_result,
    )
    require(len(descriptors) == 12, "ICL frame count drift")
    require(len({str(row["parent_id"]) for row in descriptors}) == 1, "ICL parent count drift")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_model = mobilenet_v3_small(weights=None)
    backbone_model.load_state_dict(
        torch.load(args.mobilenet_checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    backbone = backbone_model.features.eval().to(device)
    decoder = build_decoder().to(device).eval()
    decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    targets: dict[int, dict[str, np.ndarray]] = {}
    predictions: dict[int, np.ndarray] = {}
    tar_reader = TarImageReader()
    try:
        with torch.no_grad():
            for index, row in enumerate(descriptors):
                rgb = _load_bound_rgb(row, tar_reader)
                target = _load_target(row, "angular")
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(
                    device=device, dtype=torch.float32
                ) / 255.0
                value = ((tensor - mean) / std)[None]
                captured = []
                for layer_index, layer in enumerate(backbone):
                    value = layer(value)
                    if layer_index in (1, 3, 8, 12):
                        captured.append(value)
                require(len(captured) == 4, "MobileNet feature capture drift")
                logits = decoder(tuple(captured), target["probability"].shape)
                predictions[index] = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)
                targets[index] = target
    finally:
        tar_reader.close()

    threshold = float(checkpoint["selected_threshold"])
    metrics = _source_metrics(predictions, descriptors, targets, threshold)["by_source"]["icl_exact"]
    baseline_document = json.loads(args.r15_baseline.read_text(encoding="utf-8"))
    baseline = baseline_document["overall"]["student_probability"]
    gates = {
        "icl_excluded_from_checkpoint_fit_selection_canary": "icl_exact" not in training_sources,
        "frame_count_eq_12_and_parent_count_eq_1": len(descriptors) == 12
        and len({str(row["parent_id"]) for row in descriptors}) == 1,
        "threshold_frozen_from_internal_selection": threshold == float(checkpoint["selected_threshold"]),
        "average_precision_above_constant_prevalence": float(metrics["average_precision"])
        > float(metrics["prevalence"]),
        "f1_within_2px_exceeds_r15": float(metrics["f1_within_2px"])
        > float(baseline["f1_within_tolerance"]),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_angular_boundary_student_icl_result_v1",
        "status": "ANGULAR_BOUNDARY_ICL_EXTERNAL_PASS" if passed else "ANGULAR_BOUNDARY_ICL_EXTERNAL_FAIL",
        "question": "Does the frozen ARKit/TUM angular-boundary specialist improve localization on checkpoint-unseen ICL exact geometry without ICL threshold selection?",
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "binding": str(args.binding.resolve()),
            "binding_sha256": sha256_file(args.binding),
            "angular_label_contract": label_contract,
            "r15_baseline": str(args.r15_baseline.resolve()),
            "r15_baseline_sha256": sha256_file(args.r15_baseline),
        },
        "protocol": {
            "training_sources": sorted(training_sources),
            "evaluation_source": "icl_exact",
            "parent_count": 1,
            "frame_count": 12,
            "checkpoint_or_threshold_selection_on_icl": False,
            "selected_threshold": threshold,
            "target": "source-exact boundary core; angular soft field is not used as evaluation truth",
        },
        "metrics": metrics,
        "r15_reference": {
            "average_precision": float(baseline["student_average_precision"]),
            "f1_within_2px": float(baseline["f1_within_tolerance"]),
            "threshold": float(baseline["threshold"]),
        },
        "gates": gates,
        "execution": {
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
        },
        "decision": {
            "angular_boundary_localization_replication_supported": passed,
            "teacher_filled_boundary_authorized": False,
            "formal_f1_authority_changed": False,
        },
        "claim_boundary": "Checkpoint-unseen one-parent synthetic-exact boundary diagnostic. No complete truth, task utility, real-world, formal F1, safety, deployment, or product claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--angular-label-result", type=Path, default=DEFAULT_ANGULAR_LABEL_RESULT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mobilenet-checkpoint", type=Path, default=DEFAULT_MOBILENET)
    parser.add_argument("--r15-baseline", type=Path, default=DEFAULT_R15_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "metrics": result["metrics"], "gates": result["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
