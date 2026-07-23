#!/usr/bin/env python3
"""Stage admitted holdout RGB for exact Android Canvas/TFLite detector export."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from contract import load_json, sha256_file, validate_prereg


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--truth-windows", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replacement", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("refusing to overwrite holdout detector device bundle")
    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    execution = config["candidate_execution_contract"]
    replacement = load_json(args.replacement) if args.replacement else None
    planned_output = (
        replacement["planned_outputs"]["detector_device_bundle"]
        if replacement
        else execution["planned_device_bundle_path"]
    )
    if args.output_dir.resolve() != (repo / planned_output).resolve():
        raise RuntimeError("device bundle path differs from preregistration")
    truth = load_json(args.truth_windows)
    if truth.get("selection_authority") is not True or truth.get("admitted_source_count") != 2:
        raise RuntimeError("truth windows lack two-source selection authority")
    if truth.get("candidate_outputs_executed") is not False or truth.get("app_detector_or_event_outputs_exposed") is not False:
        raise RuntimeError("truth windows are not candidate/App blind")
    state = load_json(args.state)
    if state.get("status") != "complete" or state.get("candidate_outputs_executed") is not False:
        raise RuntimeError("holdout materialization is incomplete or contaminated")
    if replacement:
        replacement_sha = sha256_file(args.replacement)
        if truth.get("replacement_preregistration_sha256") != replacement_sha:
            raise RuntimeError("replacement truth-window binding mismatch")
        if state.get("replacement_preregistration_sha256") != replacement_sha:
            raise RuntimeError("replacement materialization state binding mismatch")
    detector = execution["app_detector"]
    rows = []
    source_counts = {}
    for source in truth["sources"]:
        if not source["admission_pass"]:
            continue
        source_id = source["source_id"]
        source_count = 0
        sequence_root = args.dataset_root / source_id / "sequences"
        for sequence_dir in sorted(path for path in sequence_root.iterdir() if path.is_dir()):
            bundle = load_json(sequence_dir / "bundle.json")
            frames_path = sequence_dir / "frames.jsonl"
            if bundle.get("frames_sha256") != sha256_file(frames_path):
                raise RuntimeError(f"sequence frame binding mismatch: {sequence_dir}")
            for frame in (json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line):
                image = sequence_dir / frame["rgb_path"]
                if sha256_file(image) != frame["rgb_sha256"]:
                    raise RuntimeError(f"RGB hash mismatch: {image}")
                relative = Path("images") / source_id / sequence_dir.name / f"{frame['frame_id']}.png"
                link_or_copy(image, args.output_dir / relative)
                rows.append({
                    "source_name": source_id,
                    "source_id": source_id,
                    "sequence_id": sequence_dir.name,
                    "frame_id": frame["frame_id"],
                    "source_capture_timestamp_ns": frame["source_capture_timestamp_ns"],
                    "source_size": [int(bundle["camera_info"]["width"]), int(bundle["camera_info"]["height"])],
                    "image_path": relative.as_posix(),
                    "image_sha256": frame["rgb_sha256"],
                    "host_input_tensor_sha256": "0" * 64,
                    "host_raw_output_sha256": "0" * 64,
                    "host_raw_person_max_confidence": 0.0,
                })
                source_count += 1
        source_counts[source_id] = source_count
    rows.sort(key=lambda row: (row["source_id"], row["sequence_id"], int(row["frame_id"])))
    payload = {
        "schema": "blindassist_ustrf_detector_taxonomy_device_input_v1",
        "purpose": "route_target_evidence_closure_r1_exact_android_canvas_detector_input",
        "config_sha256": sha256_file(args.config),
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "truth_windows_sha256": sha256_file(args.truth_windows),
        "materialization_state_sha256": sha256_file(args.state),
        "frame_count": len(rows),
        "source_frame_counts": source_counts,
        "input_shape": detector["input_shape"],
        "output_shape": detector["output_shape"],
        "model_sha256": detector["model_sha256"],
        "labels_sha256": detector["labels_sha256"],
        "person_class_index": detector["person_class_index"],
        "confidence_threshold": detector["confidence_threshold"],
        "nms_iou_threshold": detector["nms_iou_threshold"],
        "host_hash_placeholders_are_not_parity_claims": True,
        "frames": rows,
    }
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"frame_count": len(rows), "manifest_sha256": sha256_file(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
