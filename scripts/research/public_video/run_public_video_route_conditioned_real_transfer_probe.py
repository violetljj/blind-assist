#!/usr/bin/env python3
"""Test whether the synthetic distance-field teacher transfers to real provisional events.

The dense teacher is fit only from the visually reviewed r814a synthetic
counterfactual pairs.  A deterministic ridge event head is evaluated with one
complete real parent source held out.  Future route anchors from r797a are an
offline explicit-route oracle proxy; obstacle-hit bits and reference transition
times are never model inputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_synthetic_mask_teacher_probe as teacher_probe
import run_public_video_route_conditioned_synthetic_probe as synthetic_probe
import sanpo_depth_anything_linear_probe as depth_probe


SCHEMA = "blindassist_public_video_route_conditioned_real_transfer_probe_v1"


def merge_source(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for field in ("source_id", "local_video_path", "video_sha256"):
        if target.get(field) != incoming.get(field):
            raise ValueError(f"duplicate source metadata mismatch: {incoming.get('source_id')} {field}")
    by_time = {int(row["timestamp_ms"]): row for row in target["samples"]}
    for sample in incoming["samples"]:
        timestamp = int(sample["timestamp_ms"])
        if timestamp in by_time and by_time[timestamp] != sample:
            raise ValueError(f"duplicate source sample mismatch: {incoming['source_id']} {timestamp}")
        by_time[timestamp] = sample
    target["samples"] = [by_time[key] for key in sorted(by_time)]


def decode_at(video: Path, timestamps: Sequence[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    frames: list[np.ndarray] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot decode {video} at {timestamp}ms")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def global_field_features(score_map: np.ndarray) -> np.ndarray:
    values = np.asarray(score_map, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("global field must be finite and two-dimensional")
    flat = values.ravel()
    yy, xx = np.mgrid[:values.shape[0], :values.shape[1]]
    weights = np.exp(np.clip(values - values.max(), -30.0, 0.0))
    total = max(float(weights.sum()), 1e-12)
    return np.asarray([
        float(flat.mean()), float(np.quantile(flat, 0.75)), float(np.quantile(flat, 0.90)),
        float(flat.max()), float(np.mean(flat > 0.0)),
        float((weights * xx).sum() / total / max(values.shape[1] - 1, 1)),
        float((weights * yy).sum() / total / max(values.shape[0] - 1, 1)),
    ], dtype=np.float64)


def build_event_matrices(
    oracle: dict[str, Any],
    score_maps: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    global_rows: list[np.ndarray] = []
    route_rows: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    event_ids: list[str] = []
    frame_counts: list[int] = []
    for event in oracle["events"]:
        source = str(event["parent_source_id"])
        global_frames: list[np.ndarray] = []
        route_frames: list[np.ndarray] = []
        for frame in event["frames"]:
            anchors = frame.get("anchors", [])
            if int(frame.get("valid_anchor_count", 0)) != 3 or len(anchors) != 3:
                continue
            timestamp = int(frame["timestamp_ms"])
            score_map = score_maps[(source, timestamp)]
            waypoints = [anchor["point_xy_norm"] for anchor in anchors]
            global_frames.append(global_field_features(score_map))
            route_frames.append(synthetic_probe.route_conditioned_risk_features(score_map, waypoints))
        if not route_frames:
            raise ValueError(f"event has no valid explicit-route frames: {event['item_id']}")
        global_rows.append(np.mean(np.stack(global_frames), axis=0))
        route_rows.append(np.mean(np.stack(route_frames), axis=0))
        labels.append(int(bool(event["reference_intervention_required"])))
        sources.append(source)
        event_ids.append(str(event["item_id"]))
        frame_counts.append(len(route_frames))
    return (
        np.stack(global_rows), np.stack(route_rows), np.asarray(labels, dtype=np.int64),
        np.asarray(sources, dtype=str), event_ids, frame_counts,
    )


def source_loso_predictions(
    features: np.ndarray, labels: np.ndarray, sources: np.ndarray, ridge: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.full(len(labels), -1, dtype=np.int64)
    folds: list[dict[str, Any]] = []
    for held_out in sorted(set(sources.tolist())):
        train = np.flatnonzero(sources != held_out)
        test = np.flatnonzero(sources == held_out)
        classes = sorted(set(labels[train].tolist()))
        if classes != [0, 1]:
            raise ValueError(f"training fold lacks a class: {held_out}")
        fitted = common.fit_episode_ridge(features[train], labels[train], ridge=ridge, class_balanced=True)
        logits = features[test] @ fitted["kernel"] + fitted["bias"]
        predictions[test] = np.argmax(logits, axis=1).astype(np.int64)
        folds.append({
            "held_out_parent_source_id": held_out,
            "train_event_count": int(len(train)),
            "test_event_count": int(len(test)),
            "train_classes": classes,
            "coefficient_sha256": fitted["coefficient_sha256"],
            "metrics": common.binary_metrics(labels[test], predictions[test]),
        })
    if np.any(predictions < 0):
        raise RuntimeError("source LOSO left events unscored")
    return predictions, folds


def evaluate(
    global_x: np.ndarray, route_x: np.ndarray, labels: np.ndarray, sources: np.ndarray, *, ridge: float,
) -> dict[str, Any]:
    global_predictions, global_folds = source_loso_predictions(global_x, labels, sources, ridge)
    route_predictions, route_folds = source_loso_predictions(route_x, labels, sources, ridge)
    return {
        "global_readout": {
            "metrics": common.binary_metrics(labels, global_predictions),
            "predictions": global_predictions.tolist(),
            "folds": global_folds,
        },
        "route_conditioned_readout": {
            "metrics": common.binary_metrics(labels, route_predictions),
            "predictions": route_predictions.tolist(),
            "folds": route_folds,
        },
    }


def extract_frame_map(model: Any, image: np.ndarray, *, input_size: int, layer_index: int) -> np.ndarray:
    tensor, _ = model.image2tensor(image, input_size=input_size)
    patch_height, patch_width = tensor.shape[-2] // 14, tensor.shape[-1] // 14
    outputs = model.pretrained.get_intermediate_layers(tensor, [layer_index], return_class_token=True)
    return depth_probe.tokens_to_feature_map(
        outputs[0][0], patch_height=patch_height, patch_width=patch_width,
    ).astype(np.float64)


def load_sources(feature_contract: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    sources: dict[str, dict[str, Any]] = {}
    verified: list[dict[str, str]] = []
    for key, binding in feature_contract["feature_reports"].items():
        path = Path(binding["path"])
        actual = common.sha256_file(path)
        if actual != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        report = common.load_json(path)
        verified.append({"key": key, "path": path.as_posix(), "sha256": actual})
        for incoming in report["sources"]:
            source_id = incoming["source_id"]
            if source_id in sources:
                merge_source(sources[source_id], incoming)
            else:
                sources[source_id] = {**incoming, "samples": list(incoming["samples"])}
    return sources, verified


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.dataset, args.src_root, args.checkpoint, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError("refusing to overwrite real-transfer output")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    paths = {
        "synthetic_build_receipt_sha256": args.dataset / "build_receipt.json",
        "synthetic_manual_review_sha256": args.dataset / "qa" / "manual_review.json",
        "checkpoint_sha256": args.checkpoint,
        "r816_report_sha256": Path(bound["r816_report_path"]),
        "route_oracle_sha256": Path(bound["route_oracle_path"]),
        "r790_feature_contract_sha256": Path(bound["r790_feature_contract_path"]),
        "r790_baseline_report_sha256": Path(bound["r790_baseline_report_path"]),
    }
    comparator = contract.get("bound_comparator")
    if comparator:
        paths["comparator_report_sha256"] = Path(comparator["path"])
    for key, path in paths.items():
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input hash mismatch: {path}")
    review = common.load_json(args.dataset / "qa" / "manual_review.json")
    if review.get("disposition") != "accepted_train_only_auxiliary":
        raise ValueError("synthetic dataset is not accepted train-only auxiliary data")
    oracle = common.load_json(Path(bound["route_oracle_path"]))
    if oracle.get("explicit_route_intent_interface_supported") is not True:
        raise ValueError("explicit-route oracle report did not pass its frozen gate")
    feature_contract = common.load_json(Path(bound["r790_feature_contract_path"]))
    sources, verified_feature_reports = load_sources(feature_contract)
    generation, _route_examples, patch_records = synthetic_probe.load_dataset_records(args.dataset)
    synthetic_parent_ids = sorted({row["parent_source_id"] for row in patch_records})
    real_parent_ids = sorted({str(event["parent_source_id"]) for event in oracle["events"]})
    overlap = sorted(set(synthetic_parent_ids) & set(real_parent_ids))
    if overlap:
        raise ValueError(f"synthetic/real parent-source overlap: {overlap}")

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, args.encoder)
    model.eval()
    synthetic_paths = sorted({str((args.dataset / row["image_path"]).resolve()) for row in generation})
    synthetic_maps: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for path in synthetic_paths:
            synthetic_maps[path] = teacher_probe.extract_dino_map(
                model, path, input_size=args.input_size, layer_index=args.layer_index,
            )
        teacher = synthetic_probe.fit_distance_teacher(
            patch_records, synthetic_maps, ridge=args.teacher_ridge,
            sigma_patches=args.distance_sigma_patches,
        )
        frame_keys = sorted({
            (str(event["parent_source_id"]), int(frame["timestamp_ms"]))
            for event in oracle["events"] for frame in event["frames"]
            if int(frame.get("valid_anchor_count", 0)) == 3
        })
        real_maps: dict[tuple[str, int], np.ndarray] = {}
        for source_id in sorted({key[0] for key in frame_keys}):
            if source_id not in sources:
                raise ValueError(f"missing real source binding: {source_id}")
            source = sources[source_id]
            video_path = Path(source["local_video_path"])
            if common.sha256_file(video_path) != source["video_sha256"]:
                raise ValueError(f"real video hash mismatch: {source_id}")
            timestamps = [timestamp for sid, timestamp in frame_keys if sid == source_id]
            frames = decode_at(video_path, timestamps)
            for timestamp, frame in zip(timestamps, frames):
                feature_map = extract_frame_map(
                    model, frame, input_size=args.input_size, layer_index=args.layer_index,
                )
                real_maps[(source_id, timestamp)] = synthetic_probe.distance_score_map(feature_map, teacher)

    global_x, route_x, labels, source_values, event_ids, frame_counts = build_event_matrices(oracle, real_maps)
    first = evaluate(global_x, route_x, labels, source_values, ridge=args.head_ridge)
    second = evaluate(global_x, route_x, labels, source_values, ridge=args.head_ridge)
    repeat_exact = first == second
    global_metrics = first["global_readout"]["metrics"]
    route_metrics = first["route_conditioned_readout"]["metrics"]
    gate = contract["gate"]
    checks = {
        "minimum_event_count": len(labels) >= int(gate["minimum_event_count"]),
        "minimum_source_count": len(set(source_values.tolist())) >= int(gate["minimum_source_count"]),
        "route_balanced_accuracy": route_metrics["balanced_accuracy"] >= float(gate["route_balanced_accuracy_at_least"]),
        "route_intervention_recall": route_metrics["candidate_alert_recall"] >= float(gate["route_intervention_recall_at_least"]),
        "route_context_recall": route_metrics["candidate_no_alert_recall"] >= float(gate["route_context_recall_at_least"]),
        "route_gain_over_global": route_metrics["balanced_accuracy"] - global_metrics["balanced_accuracy"] >= float(gate["route_gain_over_global_at_least"]),
        "repeat_exact": repeat_exact,
        "all_folds_finite_and_two_class": all(
            fold["train_classes"] == [0, 1] and np.isfinite(fold["metrics"]["balanced_accuracy"])
            for key in ("global_readout", "route_conditioned_readout") for fold in first[key]["folds"]
        ),
        "synthetic_real_parent_sources_disjoint": not overlap,
    }
    comparator_metrics = None
    if comparator:
        comparator_report = common.load_json(Path(comparator["path"]))
        comparator_metrics = comparator_report["evaluation"]["route_conditioned_readout"]["metrics"]
        gain_gate = contract["gain_gate"]
        checks.update({
            "balanced_accuracy_gain_over_comparator": route_metrics["balanced_accuracy"] - comparator_metrics["balanced_accuracy"] >= float(gain_gate["balanced_accuracy_gain_at_least"]),
            "context_recall_gain_over_comparator": route_metrics["candidate_no_alert_recall"] - comparator_metrics["candidate_no_alert_recall"] >= float(gain_gate["context_recall_gain_at_least"]),
            "intervention_recall_noninferior_to_comparator": route_metrics["candidate_alert_recall"] >= comparator_metrics["candidate_alert_recall"] - float(gain_gate["intervention_recall_noninferiority_margin"]),
        })
    predictions = [{
        "event_id": event_id,
        "parent_source_id": str(source_id),
        "reference_intervention_required": bool(label),
        "valid_route_frame_count": int(frame_count),
        "global_prediction": int(global_prediction),
        "route_conditioned_prediction": int(route_prediction),
    } for event_id, source_id, label, frame_count, global_prediction, route_prediction in zip(
        event_ids, source_values, labels, frame_counts,
        first["global_readout"]["predictions"], first["route_conditioned_readout"]["predictions"],
    )]
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "contract_sha256": common.sha256_file(args.contract),
            "verified_bound_inputs": {key: common.sha256_file(path) for key, path in paths.items()},
            "verified_feature_reports": verified_feature_reports,
        },
        "teacher": {
            "source": contract.get("teacher", {}).get("source", "accepted train-only synthetic exact pairs"),
            "target": "continuous bbox distance field",
            "distance_sigma_patches": args.distance_sigma_patches,
            "ridge": args.teacher_ridge,
            "coefficient_sha256": teacher["coefficient_sha256"],
            "synthetic_parent_source_ids": synthetic_parent_ids,
            "real_parent_source_ids": real_parent_ids,
            "parent_source_overlap": overlap,
            "real_frames_used_to_fit_teacher": 0,
        },
        "input_contract": {
            "future_route_anchors_used": True,
            "future_route_anchors_role": "offline explicit-route oracle proxy only",
            "obstacle_hit_in_input": False,
            "trace_intrusion_score_in_input": False,
            "reference_transition_timestamp_in_input": False,
            "event_aggregation": "mean of pre-registered per-frame feature vector",
            "split": "leave one complete real parent_source_id out",
        },
        "event_count": int(len(labels)),
        "source_count": int(len(set(source_values.tolist()))),
        "intervention_event_count": int((labels == 1).sum()),
        "context_event_count": int((labels == 0).sum()),
        "global_feature_dimension": int(global_x.shape[1]),
        "route_feature_dimension": int(route_x.shape[1]),
        "evaluation": {**first, "repeat_exact": repeat_exact},
        "event_predictions": predictions,
        "checks": checks,
        "real_transfer_gate_passed": bool(all(checks.values())),
        "comparison": {
            "r790_current_past_only_baseline": common.load_json(Path(bound["r790_baseline_report_path"]))["event_balanced_accuracy"],
            "global_distance_field_balanced_accuracy": global_metrics["balanced_accuracy"],
            "route_conditioned_distance_field_balanced_accuracy": route_metrics["balanced_accuracy"],
            "route_gain_over_global": route_metrics["balanced_accuracy"] - global_metrics["balanced_accuracy"],
            "bound_comparator_route_metrics": comparator_metrics,
        },
        "evidence_limit": "Real provisional source-heldout transfer diagnosis with future route anchors as an offline oracle proxy. It is not provider evaluation, causal runtime evidence, calibration, blind truth, or production-promotion evidence.",
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
    parser.add_argument("--layer-index", type=int, choices=range(12), default=11)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--teacher-ridge", type=float, default=10.0)
    parser.add_argument("--head-ridge", type=float, default=1.0)
    parser.add_argument("--distance-sigma-patches", type=float, default=1.5)
    args = parser.parse_args(argv)
    if args.input_size <= 0 or args.input_size % 14 or min(args.teacher_ridge, args.head_ridge, args.distance_sigma_patches) <= 0:
        parser.error("input size must be a positive multiple of 14 and ridge/sigma must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "real_transfer_gate_passed": value["real_transfer_gate_passed"],
        "comparison": value["comparison"],
        "output_sha256": common.sha256_file(parsed.output),
    }, ensure_ascii=False))
