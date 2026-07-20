#!/usr/bin/env python3
"""Extract the frozen r7.62 causal spatial cache for route-head training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_ego_route_distance_field_probe as spatial
import run_public_video_multi_horizon_route_field_probe as horizon_field
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_temporal_route_feature_cache_v1"


class PatchExtractor:
    def __init__(self, model_dir: Path) -> None:
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()

    def extract(self, images: list[np.ndarray], batch_size: int) -> np.ndarray:
        rows = []
        for start in range(0, len(images), batch_size):
            resized = [Image.fromarray(cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB))
                       for image in images[start:start + batch_size]]
            inputs = self.processor(images=resized, return_tensors="pt", do_resize=False, do_center_crop=False)
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state[:, 1:, :].cpu().numpy()
            rows.append(hidden.reshape(len(hidden), 16, 16, hidden.shape[-1]))
        return np.concatenate(rows)


def causal_flow_grid(current: np.ndarray, past_frames: list[np.ndarray], side: int) -> np.ndarray:
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    height, width = current_gray.shape
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    channels = []
    for past in past_frames:
        past_gray = cv2.cvtColor(past, cv2.COLOR_BGR2GRAY)
        flow = dis.calc(current_gray, past_gray, None)
        resized = cv2.resize(flow, (side, side), interpolation=cv2.INTER_AREA)
        channels.extend([resized[..., 0] / width, resized[..., 1] / height])
    return np.stack(channels, axis=-1)


def compose_feature_grid(projected_tokens: np.ndarray, image: np.ndarray, flow_grid: np.ndarray) -> np.ndarray:
    side = projected_tokens.shape[0]
    rgb = cv2.cvtColor(cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:side, 0:side]
    coordinates = np.stack([(xx + 0.5) / side, (yy + 0.5) / side], axis=-1)
    return np.concatenate([projected_tokens, rgb, flow_grid, coordinates], axis=-1).transpose(2, 0, 1).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.dataset_report, args.manifest, args.event_contract, args.model_dir, args.output_cache, args.output_report):
        mil.reject_independent_direction(path)
    if args.output_cache.exists() or args.output_report.exists() or Path(str(args.output_report) + ".sha256").exists():
        raise ValueError("refusing to overwrite temporal route feature outputs")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dataset_report = lifecycle.verify_json_sidecar(args.dataset_report)
    event_contract = json.loads(args.event_contract.read_text(encoding="utf-8"))
    bound = contract["bound_inputs"]
    for path, key in ((args.dataset_report, "dataset_report_sha256"), (args.manifest, "manifest_sha256"),
                      (args.event_contract, "event_pressure_contract_sha256"),
                      (args.model_dir / "pytorch_model.bin", "dinov2_weights_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input hash mismatch: {path}")
    if common.sha256_file(args.manifest) != dataset_report["manifest"]["sha256"]:
        raise ValueError("manifest differs from dataset report")
    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    reports = {}
    sources = {}
    for key, binding in event_contract["feature_reports"].items():
        path = Path(binding["path"])
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"feature report hash mismatch: {key}")
        report = lifecycle.verify_json_sidecar(path)
        reports[key] = report
        for source in report["sources"]:
            if source["source_id"] in sources:
                raise ValueError(f"duplicate source: {source['source_id']}")
            sources[source["source_id"]] = source
    train_by_key = {(row["source_id"], int(row["timestamp_ms"])): row for row in manifest}
    eval_rows = []
    for event in event_contract["events"]:
        source = sources[event["source_id"]]
        for sample in source["samples"]:
            timestamp = int(sample["timestamp_ms"])
            if int(event["window_ms"][0]) <= timestamp < int(event["window_ms"][1]):
                eval_rows.append({"event_id": event["event_id"], "source_id": event["source_id"], "label": int(event["label"]),
                                  "timestamp_ms": timestamp, "detections": sample.get("detections", [])})
    union_by_source: dict[str, set[int]] = {}
    for source_id, timestamp in train_by_key:
        union_by_source.setdefault(source_id, set()).add(timestamp)
    for row in eval_rows:
        union_by_source.setdefault(row["source_id"], set()).add(int(row["timestamp_ms"]))
    side = int(contract["input"]["grid_side"])
    past_horizons = list(map(int, contract["input"]["causal_past_flow"]["horizons_ms"]))
    projection_spec = contract["input"]["current_dinov2_patch_projection"]
    projection = spatial.fixed_projection(int(projection_spec["input_dimension"]), int(projection_spec["output_dimension"]), int(projection_spec["seed"]))
    extractor = PatchExtractor(args.model_dir)
    feature_by_key = {}
    for source_id in sorted(union_by_source):
        source = sources[source_id]
        timestamps = sorted(union_by_source[source_id])
        decode_times = sorted(set(timestamps + [timestamp - horizon for timestamp in timestamps for horizon in past_horizons]))
        if decode_times[0] < 0:
            raise ValueError(f"causal history precedes source start: {source_id}")
        decoded = route_width.decode_at(Path(source["local_video_path"]), decode_times)
        frames = dict(zip(decode_times, decoded))
        current_images = [frames[timestamp] for timestamp in timestamps]
        tokens = extractor.extract(current_images, args.batch_size) @ projection
        for timestamp, image, token_grid in zip(timestamps, current_images, tokens):
            past = [frames[timestamp - horizon] for horizon in past_horizons]
            flow = causal_flow_grid(image, past, side)
            feature_by_key[(source_id, timestamp)] = compose_feature_grid(token_grid, image, flow)
    target_horizons = list(map(int, contract["target"]["channels_ms"]))
    sigma = float(contract["target"]["sigma_patches"])
    train_x = np.stack([feature_by_key[(row["source_id"], int(row["timestamp_ms"]))] for row in manifest])
    train_y = np.stack([horizon_field.horizon_fields(row["future_route_anchors"], side, sigma, target_horizons) for row in manifest]).astype(np.float32)
    train_sources = np.asarray([row["source_id"] for row in manifest])
    train_timestamps = np.asarray([int(row["timestamp_ms"]) for row in manifest], dtype=np.int64)
    eval_x = np.stack([feature_by_key[(row["source_id"], int(row["timestamp_ms"]))] for row in eval_rows])
    eval_sources = np.asarray([row["source_id"] for row in eval_rows])
    eval_events = np.asarray([row["event_id"] for row in eval_rows])
    eval_labels = np.asarray([int(row["label"]) for row in eval_rows], dtype=np.int64)
    eval_timestamps = np.asarray([int(row["timestamp_ms"]) for row in eval_rows], dtype=np.int64)
    expansion = float(contract["event_readout"]["marker_expansion_object_heights"])
    eval_obstacles = np.stack([spatial.obstacle_grid_mask(row["detections"], side, expansion) for row in eval_rows])
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_cache, train_x=train_x, train_y=train_y, train_sources=train_sources,
                        train_timestamps=train_timestamps, eval_x=eval_x, eval_sources=eval_sources,
                        eval_events=eval_events, eval_labels=eval_labels, eval_timestamps=eval_timestamps,
                        eval_obstacles=eval_obstacles)
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), "manifest_sha256": common.sha256_file(args.manifest),
                   "dinov2_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin")},
        "cache": {"path": str(args.output_cache), "sha256": common.sha256_file(args.output_cache),
                  "train_x_shape": list(train_x.shape), "train_y_shape": list(train_y.shape),
                  "eval_x_shape": list(eval_x.shape), "eval_obstacles_shape": list(eval_obstacles.shape)},
        "train_source_count": len(set(train_sources.tolist())), "eval_event_count": len(set(eval_events.tolist())),
        "future_frames_in_input": False, "event_labels_in_training": False,
        "authorization": contract["authorization"],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output_report) + ".sha256").write_text(common.sha256_file(args.output_report) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--event-contract", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "train_shape": value["cache"]["train_x_shape"],
                      "eval_shape": value["cache"]["eval_x_shape"],
                      "output_sha256": common.sha256_file(args.output_report)}, ensure_ascii=False))
