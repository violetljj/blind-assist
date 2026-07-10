from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


DIRECTIONS = {"NONE", "LEFT", "CENTER", "RIGHT"}
DISTANCES = {"FAR", "MID", "NEAR", "CRITICAL"}
RISK_LEVELS = {"NONE", "LOW", "MEDIUM", "HIGH"}
APPROACH_STATES = {"UNKNOWN", "STABLE", "APPROACHING", "RECEDING"}
ACCEPTED_REVIEW_STATUSES = {"accepted_manual_review", "accepted_ai_review"}


def parse_bool(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be true or false")


def parse_optional_int(value: str, field: str) -> int | None:
    if not value.strip():
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def issue_tags(value: str) -> list[str]:
    return [item for item in re.split(r"[\s,;|]+", value.strip()) if item]


def finalize_row(row: dict[str, Any], review: dict[str, str], allow_ai_review: bool = False) -> dict[str, Any]:
    review_status = review.get("review_status", "").strip()
    if review_status not in ACCEPTED_REVIEW_STATUSES:
        raise ValueError("review_status must be accepted_manual_review or accepted_ai_review")
    reviewer_type = review.get("reviewer_type", "").strip()
    reviewer_id = review.get("reviewer_id", "").strip()
    confidence_text = review.get("review_confidence", "").strip()
    review_count_text = review.get("independent_review_count", "").strip()
    if review_status == "accepted_ai_review":
        if not allow_ai_review:
            raise ValueError("accepted_ai_review requires explicit --allow-ai-review")
        if reviewer_type != "ai_assistant" or not reviewer_id:
            raise ValueError("AI review requires reviewer_type=ai_assistant and reviewer_id")
        confidence = float(confidence_text)
        review_count = int(review_count_text)
        if not 0.65 <= confidence <= 1.0:
            raise ValueError("AI review confidence must be between 0.65 and 1.0")
        if review_count < 2:
            raise ValueError("AI review requires at least two independent review passes")
    else:
        confidence = float(confidence_text) if confidence_text else None
        review_count = int(review_count_text) if review_count_text else 1
    blocking_tags = issue_tags(review.get("issue_tags", ""))
    if blocking_tags:
        raise ValueError(f"blocking issue_tags must be resolved: {','.join(blocking_tags)}")
    expected_object_review = review_status
    if row.get("objects") and review.get("objects_review_status", "").strip() != expected_object_review:
        raise ValueError(f"objects_review_status must be {expected_object_review} when detection GT objects are present")

    direction = review.get("expected_risk_direction", "").strip()
    distance = review.get("expected_distance_band", "").strip()
    risk_level = review.get("expected_risk_level", "").strip()
    approach = review.get("expected_approach_state", "").strip()
    if direction not in DIRECTIONS:
        raise ValueError(f"invalid expected_risk_direction: {direction}")
    if distance not in DISTANCES:
        raise ValueError(f"invalid expected_distance_band: {distance}")
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"invalid expected_risk_level: {risk_level}")
    if approach not in APPROACH_STATES:
        raise ValueError(f"invalid expected_approach_state: {approach}")

    primary_object_id = review.get("primary_object_id", "").strip() or None
    object_ids = {item.get("id") for item in row.get("objects", [])}
    if primary_object_id and primary_object_id not in object_ids:
        raise ValueError(f"primary_object_id does not reference detection GT objects: {primary_object_id}")
    source_primary_region_id = review.get("source_primary_region_id", "").strip() or None
    source_region_ids = {item.get("id") for item in row.get("source_regions", [])}
    if source_primary_region_id and source_primary_region_id not in source_region_ids:
        raise ValueError(f"source_primary_region_id does not reference source_regions: {source_primary_region_id}")

    finalized = dict(row)
    finalized.update({
        "primary_object_id": primary_object_id,
        "source_primary_region_id": source_primary_region_id,
        "expected_risk_direction": direction,
        "expected_distance_band": distance,
        "expected_should_alert": parse_bool(review.get("expected_should_alert", ""), "expected_should_alert"),
        "expected_risk_level": risk_level,
        "expected_approach_state": approach,
        "expected_approach_alert": parse_bool(review.get("expected_approach_alert", ""), "expected_approach_alert"),
        "expected_time_to_alert_frames": parse_optional_int(
            review.get("expected_time_to_alert_frames", ""),
            "expected_time_to_alert_frames",
        ),
        "objects_review_status": review.get("objects_review_status", "").strip(),
        "review_status": review_status,
        "review_provenance": {
            "reviewer_type": reviewer_type or "human",
            "reviewer_id": reviewer_id or None,
            "confidence": confidence,
            "independent_review_count": review_count,
            "policy": "multi_agent_consensus_v1" if review_status == "accepted_ai_review" else "manual_review",
        },
        "review_notes": review.get("review_notes", "").strip(),
        "status": "accepted",
    })
    return finalized


def validate_bbox(sample_id: str, item: dict[str, Any], width: int, height: int, kind: str) -> str | None:
    try:
        x1, y1, x2, y2 = [float(value) for value in item["bbox_xyxy"]]
    except (KeyError, TypeError, ValueError):
        return f"{sample_id}: invalid {kind} bbox for {item.get('id')}"
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return f"{sample_id}: out-of-range {kind} bbox for {item.get('id')}"
    return None


def validate_final(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    coco_labels: set[str],
) -> list[str]:
    errors: list[str] = []
    expected_positions: dict[str, list[int]] = {}
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    images_root = (dataset_root / "images" / "test").resolve()
    masks_root = (dataset_root / "source_masks" / "test").resolve()
    for row in rows:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen_ids or "/" in sample_id or "\\" in sample_id:
            errors.append(f"duplicate, missing, or unsafe id: {sample_id}")
        seen_ids.add(sample_id)
        if row.get("status") != "accepted" or row.get("review_status") not in ACCEPTED_REVIEW_STATUSES:
            errors.append(f"{sample_id}: not accepted")
        image_path = (dataset_root / str(row.get("image_path", ""))).resolve()
        if not image_path.is_relative_to(images_root) or not image_path.is_file():
            errors.append(f"{sample_id}: image_path must be an existing file under images/test")
            continue
        image_hash = sha256_file(image_path)
        if image_hash != row.get("source", {}).get("sha256"):
            errors.append(f"{sample_id}: image SHA256 differs from draft source.sha256")
        if image_hash in seen_hashes:
            errors.append(f"{sample_id}: duplicate image hash with {seen_hashes[image_hash]}")
        seen_hashes[image_hash] = sample_id
        with Image.open(image_path) as image:
            actual_size = image.size
        width, height = int(row.get("width", -1)), int(row.get("height", -1))
        if actual_size != (width, height):
            errors.append(f"{sample_id}: image dimensions differ from manifest")
        mask_path = (masks_root / f"{sample_id}.png").resolve()
        if not mask_path.is_relative_to(masks_root) or not mask_path.is_file():
            errors.append(f"{sample_id}: missing source mask")
        elif sha256_file(mask_path) != row.get("source", {}).get("mask_sha256"):
            errors.append(f"{sample_id}: mask SHA256 differs from draft source.mask_sha256")
        object_ids: set[str] = set()
        if row.get("objects") and row.get("source_annotation_quality") != "HUMAN_ANNOTATED":
            errors.append(f"{sample_id}: detection GT objects require HUMAN_ANNOTATED source frame")
        for item in row.get("objects", []):
            item_id = str(item.get("id", ""))
            if not item_id or item_id in object_ids:
                errors.append(f"{sample_id}: duplicate or missing object id {item_id}")
            object_ids.add(item_id)
            if item.get("class") not in coco_labels:
                errors.append(f"{sample_id}: non-COCO detection GT class {item.get('class')}")
            bbox_error = validate_bbox(sample_id, item, width, height, "object")
            if bbox_error:
                errors.append(bbox_error)
        region_ids: set[str] = set()
        for item in row.get("source_regions", []):
            item_id = str(item.get("id", ""))
            if not item_id or item_id in region_ids:
                errors.append(f"{sample_id}: duplicate or missing source region id {item_id}")
            region_ids.add(item_id)
            bbox_error = validate_bbox(sample_id, item, width, height, "source region")
            if bbox_error:
                errors.append(bbox_error)
        if row.get("primary_object_id") and row["primary_object_id"] not in object_ids:
            errors.append(f"{sample_id}: primary_object_id not found in detection GT objects")
        if row.get("source_primary_region_id") and row["source_primary_region_id"] not in region_ids:
            errors.append(f"{sample_id}: source_primary_region_id not found in source regions")
        if row.get("source", {}).get("official_split") not in {"train", "test"}:
            errors.append(f"{sample_id}: missing official SANPO split")
        expected_positions.setdefault(str(row.get("sequence_id")), []).append(int(row.get("frame_index", -1)))
    for sequence_id, positions in expected_positions.items():
        if positions != list(range(len(positions))):
            errors.append(f"{sequence_id}: frame_index must be contiguous from 0")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a manually reviewed SANPO draft into benchmark manifest.jsonl.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--review-csv", default=None)
    parser.add_argument("--allow-ai-review", action="store_true", help="Allow explicitly provenance-marked multi-pass AI review.")
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    draft_path = root / "manifest.draft.jsonl"
    review_path = Path(args.review_csv).resolve() if args.review_csv else root / "qa" / "manual_review_checklist.csv"
    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists():
        raise SystemExit("Dataset roots are immutable after canonical manifest.jsonl is published; build a new root")

    rows = load_jsonl(draft_path)
    spec = json.loads((root / "dataset_spec.json").read_text(encoding="utf-8"))
    if float(spec.get("sampling", {}).get("target_fps", -1)) != 10.0:
        raise SystemExit("dataset_spec target_fps must be exactly 10 for the current benchmark")
    project_root = Path(__file__).resolve().parents[1]
    coco_labels = {
        line.strip()
        for line in (project_root / "app" / "src" / "main" / "assets" / "coco_labels.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    with review_path.open("r", newline="", encoding="utf-8-sig") as handle:
        review_list = list(csv.DictReader(handle))
    reviews: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for item in review_list:
        review_id = item.get("id", "")
        if not review_id or review_id in reviews:
            errors.append(f"duplicate or missing review id: {review_id}")
        else:
            reviews[review_id] = item
    draft_ids = [str(row.get("id", "")) for row in rows]
    for extra_id in sorted(set(reviews) - set(draft_ids)):
        errors.append(f"unexpected review row: {extra_id}")

    finalized: list[dict[str, Any]] = []
    for row in rows:
        review = reviews.get(row["id"])
        if review is None:
            errors.append(f"{row['id']}: missing review row")
            continue
        try:
            finalized.append(finalize_row(row, review, allow_ai_review=args.allow_ai_review))
        except (TypeError, ValueError) as error:
            errors.append(f"{row['id']}: {error}")
    errors.extend(validate_final(finalized, root, coco_labels))
    report = {
        "ok": not errors and len(finalized) == len(rows),
        "draft_count": len(rows),
        "finalized_count": len(finalized),
        "draft_sha256": sha256_file(draft_path),
        "review_sha256": sha256_file(review_path),
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "errors": errors,
    }
    report_path = root / "qa" / "finalize_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["ok"]:
        print(f"finalize_ok=false report={report_path}")
        return 1
    temp_path = root / "manifest.jsonl.tmp"
    temp_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in finalized), encoding="utf-8")
    temp_path.replace(manifest_path)
    print(f"finalize_ok=true manifest={manifest_path} rows={len(finalized)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
