#!/usr/bin/env python3
"""Materialize hash-bound Android replay input from the frozen multisource geometry receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    prereg = load_json(args.preregistration)
    freeze = load_json(args.freeze_manifest)
    index = load_json(args.freeze_index)
    prereg_sha = sha256_file(args.preregistration)
    freeze_sha = sha256_file(args.freeze_manifest)
    if prereg_sha != freeze["preregistration_sha256"] or prereg_sha != index["preregistration_sha256"]:
        raise ValueError("preregistration SHA mismatch")
    if freeze_sha != index["geometry_freeze_manifest_sha256"]:
        raise ValueError("geometry freeze manifest SHA mismatch")
    if index.get("risk_or_detector_outputs_read_by_materializer") is not False:
        raise ValueError("freeze index does not attest blind materialization")
    events = {row["event_id"]: row for row in prereg["held_out_events"]}
    freeze_rows = {row["event_id"]: row for row in freeze["sources"]}
    index_rows = {row["event_id"]: row for row in index["sources"]}
    if set(events) != set(freeze_rows) or set(events) != set(index_rows) or len(events) != 6:
        raise ValueError("six-source inventory mismatch")
    output_sources = []
    staging = []
    for event_id, event in events.items():
        frozen = freeze_rows[event_id]
        indexed = index_rows[event_id]
        for key, frozen_key in (("source_id", "source_id"), ("window_ms", "window_ms")):
            if event[key] != frozen[frozen_key] or event[key] != indexed[frozen_key]:
                raise ValueError(f"{event_id} {key} drift")
        if event["expected_class"] != frozen["expected_class_reporting_only"]:
            raise ValueError(f"{event_id} class drift")
        video = (repo / frozen["video_path"]).resolve()
        if sha256_file(video) != frozen["video_sha256"]:
            raise ValueError(f"{event_id} video SHA mismatch")
        receipt = (args.freeze_manifest.parent / event_id / "route_projection_receipt.json").resolve()
        receipt_sha = sha256_file(receipt)
        if receipt_sha != indexed["route_projection_receipt_sha256"]:
            raise ValueError(f"{event_id} route receipt SHA mismatch")
        receipt_json = load_json(receipt)
        exact = {
            "source_id": event["source_id"], "window_ms": event["window_ms"],
            "video_sha256": frozen["video_sha256"], "preregistration_sha256": prereg_sha,
            "geometry_freeze_manifest_sha256": freeze_sha,
            "route_polygon_xy_norm": frozen["route_polygon_xy_norm"],
            "projection_status": frozen["projection_status"],
        }
        if any(receipt_json.get(key) != value for key, value in exact.items()):
            raise ValueError(f"{event_id} route receipt content drift")
        remote_video = f"ustrf-crosscam-multisource-r1/videos/{event_id}{video.suffix.lower()}"
        remote_receipt = f"ustrf-crosscam-multisource-r1/receipts/{event_id}.json"
        output_sources.append({
            "event_id": event_id,
            "source_id": event["source_id"],
            "expected_class": event["expected_class"],
            "projection_status": frozen["projection_status"],
            "window_start_ms": event["window_ms"][0],
            "window_end_ms": event["window_ms"][1],
            "replay_step_ms": 500,
            "route_polygon_xy_norm": frozen["route_polygon_xy_norm"],
            "video_path": remote_video,
            "video_sha256": frozen["video_sha256"],
            "projection_receipt_path": remote_receipt,
            "projection_receipt_sha256": receipt_sha,
        })
        staging.append({"host_path": str(video), "device_relative_path": remote_video, "sha256": frozen["video_sha256"]})
        staging.append({"host_path": str(receipt), "device_relative_path": remote_receipt, "sha256": receipt_sha})
    android_input = {
        "schema": "blindassist_ustrf_crosscam_multisource_android_input_v1",
        "preregistration_sha256": prereg_sha,
        "geometry_freeze_manifest_sha256": freeze_sha,
        "freeze_receipt_index_sha256": sha256_file(args.freeze_index),
        "uncertainty_frame_ratios": [0.01, 0.02, 0.03],
        "threshold_fit": False,
        "parameter_search": False,
        "training_authorized": False,
        "production_model_replacement_authorized": False,
        "sources": output_sources,
    }
    output_path = args.output_dir / "android_multisource_input.json"
    write_json(output_path, android_input)
    staging.insert(0, {
        "host_path": str(output_path.resolve()),
        "device_relative_path": "ustrf-crosscam-multisource-r1/input.json",
        "sha256": sha256_file(output_path),
    })
    write_json(args.output_dir / "host_staging_receipt.json", {
        "schema": "blindassist_ustrf_crosscam_multisource_android_staging_v1",
        "input_sha256": sha256_file(output_path),
        "geometry_freeze_manifest_sha256": freeze_sha,
        "risk_or_detector_outputs_read": False,
        "files": staging,
    })
    print(json.dumps({"ok": True, "input": str(output_path), "sha256": sha256_file(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
