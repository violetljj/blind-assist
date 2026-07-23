#!/usr/bin/env python3
"""Run one frozen candidate-blind visual person proposal pass on the full holdout RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from contract import load_json, sha256_file, validate_prereg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--replacement", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite proposal output: {args.output}")
    repo = args.repo.resolve()
    config = validate_prereg(load_json(args.config), repo=repo)
    state = load_json(args.state)
    if state.get("status") != "complete" or state.get("candidate_outputs_executed") is not False:
        raise RuntimeError("holdout materialization must be complete and candidate blind")
    passes = config["sealed_holdout"]["holdout_truth_freeze_contract"]["all_person_presence"]["visual_passes"]
    matches = [row for row in passes if row["id"] == args.pass_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate holdout visual pass: {args.pass_id}")
    visual_pass = matches[0]
    replacement = load_json(args.replacement) if args.replacement else None
    if replacement is not None:
        if replacement["base_preregistration"]["sha256"] != sha256_file(args.config):
            raise RuntimeError("replacement base preregistration binding mismatch")
        planned_key = {
            "HOLDOUT_PERSON_PASS_B_YOLOV8N": "visual_pass_b",
            "HOLDOUT_PERSON_PASS_C_YOLO11X": "visual_pass_c",
        }[args.pass_id]
        planned_output = replacement["planned_outputs"][planned_key]
        selected_source_ids = [row["source_id"] for row in replacement["replacement_sources"]]
        if state.get("replacement_preregistration_sha256") != sha256_file(args.replacement):
            raise RuntimeError("replacement materialization state binding mismatch")
    else:
        planned_output = visual_pass["planned_output_path"]
        selected_source_ids = load_json(
            repo / config["sealed_holdout"]["content_qualification_receipt"]["path"]
        )["selected_source_ids"]
    if args.output.resolve() != (repo / planned_output).resolve():
        raise RuntimeError("proposal output path differs from preregistration")
    model_path = Path(visual_pass["model_path"])
    if sha256_file(model_path) != visual_pass["model_sha256"]:
        raise RuntimeError("visual person model hash mismatch")
    rows: list[dict[str, Any]] = []
    for source_id in selected_source_ids:
        sequence_root = args.dataset_root / source_id / "sequences"
        for sequence_dir in sorted(path for path in sequence_root.iterdir() if path.is_dir()):
            bundle_path = sequence_dir / "bundle.json"
            frames_path = sequence_dir / "frames.jsonl"
            bundle = load_json(bundle_path)
            if bundle.get("candidate_outputs_executed") is not False or bundle.get("frames_sha256") != sha256_file(frames_path):
                raise RuntimeError(f"unverified or non-blind bundle: {sequence_dir}")
            for frame in frames_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(frame)
                image_path = sequence_dir / row["rgb_path"]
                if sha256_file(image_path) != row["rgb_sha256"]:
                    raise RuntimeError(f"RGB hash mismatch: {image_path}")
                rows.append({
                    "source_id": source_id,
                    "sequence_id": sequence_dir.name,
                    "frame_id": row["frame_id"],
                    "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
                    "image_path": image_path,
                    "image_sha256": row["rgb_sha256"],
                })
    model = YOLO(str(model_path))
    annotations = []
    for offset in range(0, len(rows), args.batch):
        chunk = rows[offset : offset + args.batch]
        results = model.predict(
            source=[str(row["image_path"]) for row in chunk],
            imgsz=int(visual_pass["imgsz"]),
            conf=float(visual_pass["confidence"]),
            iou=float(visual_pass["iou"]),
            classes=[int(visual_pass["class_id"])],
            device=args.device,
            batch=args.batch,
            verbose=False,
        )
        for row, result in zip(chunk, results, strict=True):
            boxes = []
            if result.boxes is not None:
                for xyxy, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    strict=True,
                ):
                    boxes.append({
                        "bbox_xyxy": [round(float(value), 3) for value in xyxy],
                        "proposal_confidence": round(float(confidence), 6),
                    })
            annotations.append({
                "source_id": row["source_id"],
                "sequence_id": row["sequence_id"],
                "frame_id": row["frame_id"],
                "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
                "image_sha256": row["image_sha256"],
                "person_proposals": boxes,
            })
        if len(annotations) % 256 < args.batch:
            print(f"holdout_person_proposal_frames={len(annotations)}/{len(rows)}", flush=True)
    payload = {
        "schema": "blindassist_crowdbot_holdout_visual_person_proposals_r1",
        "authority": "candidate_blind_annotation_proposal_only_not_final_truth_or_promotion_credit",
        "candidate_outputs_executed": False,
        "app_detector_or_event_outputs_exposed": False,
        "config_sha256": sha256_file(args.config),
        "replacement_preregistration_sha256": sha256_file(args.replacement) if args.replacement else None,
        "materialization_state_sha256": sha256_file(args.state),
        "pass_id": args.pass_id,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "imgsz": int(visual_pass["imgsz"]),
        "confidence": float(visual_pass["confidence"]),
        "iou": float(visual_pass["iou"]),
        "class_id": int(visual_pass["class_id"]),
        "frame_count": len(annotations),
        "frames": annotations,
        "production_authority": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "HOLDOUT_VISUAL_PERSON_PROPOSALS_MATERIALIZED",
        "pass_id": args.pass_id,
        "frame_count": len(annotations),
        "output_sha256": sha256_file(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
