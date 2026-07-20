#!/usr/bin/env python3
"""Run the frozen r820 route-local plus global-background OFAT transfer probe."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_synthetic_mask_teacher_probe as teacher_probe
import run_public_video_route_conditioned_real_transfer_probe as r819
import run_public_video_route_conditioned_synthetic_probe as synthetic_probe
import sanpo_depth_anything_linear_probe as depth_probe


SCHEMA = "blindassist_public_video_route_background_real_transfer_probe_v1"


def combine_features(route_x: np.ndarray, global_x: np.ndarray) -> np.ndarray:
    route = np.asarray(route_x, dtype=np.float64)
    global_values = np.asarray(global_x, dtype=np.float64)
    if route.ndim != 2 or global_values.ndim != 2 or len(route) != len(global_values):
        raise ValueError("route and global event matrices must align")
    combined = np.concatenate([route, global_values], axis=1)
    if combined.shape[1] != 20 or not np.isfinite(combined).all():
        raise ValueError("r820 combined feature must be finite and 20-dimensional")
    return combined


def extract_matrices(args: argparse.Namespace, r819_contract: dict[str, Any]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[int], dict[str, Any]
]:
    bound = r819_contract["bound_inputs"]
    paths = {
        "synthetic_build_receipt_sha256": args.dataset / "build_receipt.json",
        "synthetic_manual_review_sha256": args.dataset / "qa" / "manual_review.json",
        "checkpoint_sha256": args.checkpoint,
        "r816_report_sha256": Path(bound["r816_report_path"]),
        "route_oracle_sha256": Path(bound["route_oracle_path"]),
        "r790_feature_contract_sha256": Path(bound["r790_feature_contract_path"]),
        "r790_baseline_report_sha256": Path(bound["r790_baseline_report_path"]),
    }
    for key, path in paths.items():
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"r819 bound input hash mismatch: {path}")
    oracle = common.load_json(Path(bound["route_oracle_path"]))
    feature_contract = common.load_json(Path(bound["r790_feature_contract_path"]))
    sources, verified_reports = r819.load_sources(feature_contract)
    generation, _route_examples, patch_records = synthetic_probe.load_dataset_records(args.dataset)
    synthetic_ids = {row["parent_source_id"] for row in patch_records}
    real_ids = {str(event["parent_source_id"]) for event in oracle["events"]}
    if synthetic_ids & real_ids:
        raise ValueError("synthetic and real parent sources overlap")

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, args.encoder)
    model.eval()
    synthetic_paths = sorted({str((args.dataset / row["image_path"]).resolve()) for row in generation})
    with torch.no_grad():
        synthetic_maps = {
            path: teacher_probe.extract_dino_map(model, path, input_size=args.input_size, layer_index=args.layer_index)
            for path in synthetic_paths
        }
        teacher = synthetic_probe.fit_distance_teacher(
            patch_records, synthetic_maps, ridge=args.teacher_ridge, sigma_patches=args.distance_sigma_patches,
        )
        frame_keys = sorted({
            (str(event["parent_source_id"]), int(frame["timestamp_ms"]))
            for event in oracle["events"] for frame in event["frames"]
            if int(frame.get("valid_anchor_count", 0)) == 3
        })
        score_maps: dict[tuple[str, int], np.ndarray] = {}
        for source_id in sorted(real_ids):
            source = sources[source_id]
            video = Path(source["local_video_path"])
            if common.sha256_file(video) != source["video_sha256"]:
                raise ValueError(f"real video hash mismatch: {source_id}")
            timestamps = [timestamp for sid, timestamp in frame_keys if sid == source_id]
            frames = r819.decode_at(video, timestamps)
            for timestamp, frame in zip(timestamps, frames):
                feature_map = r819.extract_frame_map(
                    model, frame, input_size=args.input_size, layer_index=args.layer_index,
                )
                score_maps[(source_id, timestamp)] = synthetic_probe.distance_score_map(feature_map, teacher)
    global_x, route_x, labels, sources_array, event_ids, frame_counts = r819.build_event_matrices(oracle, score_maps)
    metadata = {
        "teacher_coefficient_sha256": teacher["coefficient_sha256"],
        "verified_feature_reports": verified_reports,
        "synthetic_parent_source_ids": sorted(synthetic_ids),
        "real_parent_source_ids": sorted(real_ids),
    }
    return global_x, route_x, labels, sources_array, event_ids, frame_counts, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.dataset, args.src_root, args.checkpoint, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError("refusing to overwrite r820 output")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    r819_contract_path = Path(bound["r819_contract_path"])
    r819_report_path = Path(bound["r819_report_path"])
    if common.sha256_file(r819_contract_path) != bound["r819_contract_sha256"]:
        raise ValueError("r819 contract hash mismatch")
    if common.sha256_file(r819_report_path) != bound["r819_report_sha256"]:
        raise ValueError("r819 report hash mismatch")
    r819_contract = common.load_json(r819_contract_path)
    r819_report = common.load_json(r819_report_path)
    global_x, route_x, labels, sources, event_ids, frame_counts, metadata = extract_matrices(args, r819_contract)
    combined_x = combine_features(route_x, global_x)
    first = r819.source_loso_predictions(combined_x, labels, sources, args.head_ridge)
    second = r819.source_loso_predictions(combined_x, labels, sources, args.head_ridge)
    predictions, folds = first
    repeat_exact = predictions.tolist() == second[0].tolist() and folds == second[1]
    metrics = common.binary_metrics(labels, predictions)
    old = r819_report["evaluation"]["route_conditioned_readout"]["metrics"]
    gate = contract["gate"]
    checks = {
        "balanced_accuracy": metrics["balanced_accuracy"] >= float(gate["balanced_accuracy_at_least"]),
        "intervention_recall": metrics["candidate_alert_recall"] >= float(gate["intervention_recall_at_least"]),
        "context_recall": metrics["candidate_no_alert_recall"] >= float(gate["context_recall_at_least"]),
        "balanced_accuracy_gain_over_r819": metrics["balanced_accuracy"] - old["balanced_accuracy"] >= float(gate["balanced_accuracy_gain_over_r819_at_least"]),
        "context_recall_gain_over_r819": metrics["candidate_no_alert_recall"] - old["candidate_no_alert_recall"] >= float(gate["context_recall_gain_over_r819_at_least"]),
        "repeat_exact": repeat_exact,
    }
    rows = [{
        "event_id": event_id, "parent_source_id": str(source),
        "reference_intervention_required": bool(label), "valid_route_frame_count": int(frame_count),
        "r819_route_prediction": int(old_prediction), "r820_route_background_prediction": int(prediction),
    } for event_id, source, label, frame_count, old_prediction, prediction in zip(
        event_ids, sources, labels, frame_counts,
        r819_report["evaluation"]["route_conditioned_readout"]["predictions"], predictions,
    )]
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), **metadata},
        "single_change": contract["single_change"],
        "feature_dimension": int(combined_x.shape[1]),
        "event_count": int(len(labels)), "source_count": int(len(set(sources.tolist()))),
        "evaluation": {"metrics": metrics, "predictions": predictions.tolist(), "folds": folds, "repeat_exact": repeat_exact},
        "event_predictions": rows,
        "comparison": {
            "r819_route_balanced_accuracy": old["balanced_accuracy"],
            "r820_route_background_balanced_accuracy": metrics["balanced_accuracy"],
            "balanced_accuracy_gain": metrics["balanced_accuracy"] - old["balanced_accuracy"],
            "r819_context_recall": old["candidate_no_alert_recall"],
            "r820_context_recall": metrics["candidate_no_alert_recall"],
            "context_recall_gain": metrics["candidate_no_alert_recall"] - old["candidate_no_alert_recall"],
        },
        "checks": checks, "route_background_gate_passed": bool(all(checks.values())),
        "evidence_limit": "Post-r819, pre-registered single-feature OFAT on real provisional source-heldout events. Future anchors remain an offline oracle proxy; no provider, causal-runtime, calibration, blind, Android, or production credit.",
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vits",), default="vits")
    parser.add_argument("--layer-index", type=int, default=11)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--teacher-ridge", type=float, default=10.0)
    parser.add_argument("--head-ridge", type=float, default=1.0)
    parser.add_argument("--distance-sigma-patches", type=float, default=1.5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, "passed": value["route_background_gate_passed"],
                      "comparison": value["comparison"],
                      "output_sha256": common.sha256_file(parsed.output)}, ensure_ascii=False))
