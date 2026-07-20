#!/usr/bin/env python3
"""Machine-redact a pending SANPO draft into an auxiliary-only RGB derivative.

This adapter does not copy source masks or infer any risk field.  It is a
conservative privacy prefilter for a public RGB draft: faces, detected plates,
and whole people/vehicles are blurred before a separate privacy audit.  The
result is never a risk/event label, canonical row, calibration item, or model
promotion input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

from research.public_video.machine_redact_public_rgb_candidate import (
    LpdYuNet,
    RedactionError,
    WholeObjectPrivacyDetector,
    candidate_regions,
    redact_regions,
    sha256_file,
)


def load_draft_rows(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.draft.jsonl"
    if not manifest.is_file():
        raise RedactionError(f"SANPO draft manifest is missing: {manifest}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RedactionError(f"draft manifest line {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise RedactionError(f"draft manifest line {line_number} must be an object")
        image_value = row.get("image_path")
        if (
            not isinstance(image_value, str)
            or not image_value
            or Path(image_value).is_absolute()
            or any("blind" in part.lower() for part in Path(image_value).parts)
        ):
            raise RedactionError(f"draft manifest line {line_number} has an unsafe image path")
        if row.get("status") != "pending_review":
            raise RedactionError(f"draft manifest line {line_number} must remain pending_review")
        if any(row.get(field) is not None for field in (
            "expected_should_alert", "expected_risk_level", "expected_approach_state",
            "expected_risk_direction", "expected_distance_band", "expected_approach_alert",
            "expected_time_to_alert_frames",
        )):
            raise RedactionError(f"draft manifest line {line_number} contains a non-null risk field")
        source = row.get("source")
        if not isinstance(source, dict) or source.get("official_split") != "train":
            raise RedactionError(f"draft manifest line {line_number} is not an official train-source candidate")
        image_path = (root / image_value).resolve()
        try:
            image_path.relative_to(root.resolve())
        except ValueError as error:
            raise RedactionError(f"draft manifest line {line_number} escapes the draft root") from error
        if not image_path.is_file():
            raise RedactionError(f"draft image is missing: {image_path}")
        rows.append({"id": row.get("id"), "image_path": image_path, "source_frame_index": row.get("source_frame_index")})
    if not rows:
        raise RedactionError("SANPO draft manifest contains no rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yunet-model", type=Path, required=True)
    parser.add_argument("--lpd-yunet-model", type=Path, required=True)
    parser.add_argument("--yolo-model", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise RedactionError(f"refusing to overwrite redaction output: {args.output_dir}")
        for label, path in (("YuNet", args.yunet_model), ("LPD_YuNet", args.lpd_yunet_model), ("YOLO", args.yolo_model)):
            if not path.is_file():
                raise RedactionError(f"{label} model is missing: {path}")
        rows = load_draft_rows(args.draft_root)
        face_detector = cv2.FaceDetectorYN.create(str(args.yunet_model), "", (320, 320), 0.85, 0.3, 5000)
        plate_detector = LpdYuNet(args.lpd_yunet_model)
        whole_object_detector = WholeObjectPrivacyDetector(args.yolo_model, args.output_dir / ".ultralytics")
        images_dir = args.output_dir / "images"
        images_dir.mkdir(parents=True)
        frames: list[dict[str, Any]] = []
        frames_with_regions = 0
        region_count = 0
        for index, row in enumerate(rows):
            image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise RedactionError(f"cannot decode draft image: {row['image_path']}")
            regions = candidate_regions(image, face_detector, plate_detector, whole_object_detector)
            output = images_dir / f"frame_{index:04d}.png"
            if not cv2.imwrite(str(output), redact_regions(image, regions)):
                raise RedactionError(f"cannot write redacted image: {output}")
            frames.append({
                "source_id": row["id"], "source_frame_index": row["source_frame_index"],
                "file_name": output.name, "sha256": sha256_file(output),
                "redaction_region_count": len(regions),
            })
            if regions:
                frames_with_regions += 1
                region_count += len(regions)
        source_manifest = args.draft_root / "manifest.draft.jsonl"
        receipt = {
            "format": "blindassist_sanpo_auxiliary_machine_redaction_v1",
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            "frame_count": len(frames),
            "frames_with_machine_redaction_regions": frames_with_regions,
            "machine_redaction_region_count": region_count,
            "detectors": [
                "opencv_yunet_face_detection_2023mar",
                "opencv_lpd_yunet_2023mar",
                "ultralytics_yolov8n_whole_person_vehicle_blur",
            ],
            "frames": frames,
            "source_mask_role": "auxiliary_pixel_geometry_only",
            "risk_or_event_truth_present": False,
            "privacy_audit_required": True,
            "training_execution_authorized": False,
            "production_model_replacement_authorized": False,
            "important_limit": "Machine redaction is a privacy prefilter, not proof that every face, plate, or identifying text was detected.",
        }
        (args.output_dir / "machine_redaction_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps({"ok": True, "frames": len(frames), "regions": region_count, "privacy_audit_required": True}, ensure_ascii=False))
    except (RedactionError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
