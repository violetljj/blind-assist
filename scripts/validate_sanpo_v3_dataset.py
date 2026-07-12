#!/usr/bin/env python3
"""Validate the v3 four-class segmentation dataset and its blind-test lock.

The validator is deliberately strict.  It accepts a training manifest containing
only train/dev rows and a separately stored blind manifest; callers must not
give a trainer a directory to crawl.  Each semantic PNG uses class IDs 0..3:
walkable, boundary_step_curb, obstacle, unknown_nonwalkable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SPLITS = {"train", "dev", "blind"}
SEMANTIC_CLASSES = ("walkable", "boundary_step_curb", "obstacle", "unknown_nonwalkable")
SCENE_BUCKETS = (
    "parallel_boundary",
    "step_curb",
    "center_obstacle",
    "lateral_pedestrian_or_ebike",
    "low_light",
    "tactile_paving_occupied",
)
EVENT_PHASES = {"APPROACHING", "ALERTED", "PASSED"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def required_string(row: dict[str, Any], field: str, sample_id: str, errors: list[str]) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        errors.append(f"{sample_id}: missing {field}")
    return value


def row_session_id(row: dict[str, Any]) -> str:
    return str(row.get("session_id") or row.get("source", {}).get("session_id") or "").strip()


def validate_rows(rows: list[dict[str, Any]], root: Path, expected_split: set[str]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    sequence_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    session_splits: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    seen_images: dict[str, str] = {}
    seen_masks: dict[str, str] = {}
    class_pixels: Counter[str] = Counter()
    for row in rows:
        sample_id = required_string(row, "id", "<row>", errors)
        if sample_id in seen_ids:
            errors.append(f"{sample_id}: duplicate id")
        seen_ids.add(sample_id)
        split = required_string(row, "split", sample_id, errors)
        if split not in expected_split or split not in SPLITS:
            errors.append(f"{sample_id}: split {split!r} is not allowed in this manifest")
        session_id = row_session_id(row)
        if not session_id:
            errors.append(f"{sample_id}: missing session_id (direct or source.session_id)")
        else:
            session_splits[session_id].add(split)
        sequence_id = required_string(row, "sequence_id", sample_id, errors)
        sequence_rows[sequence_id].append(row)
        bucket = required_string(row, "scene_bucket", sample_id, errors)
        if bucket not in SCENE_BUCKETS:
            errors.append(f"{sample_id}: unsupported scene_bucket {bucket!r}")
        if not required_string(row, "risk_event_id", sample_id, errors):
            pass
        if row.get("expected_event_phase") not in EVENT_PHASES:
            errors.append(f"{sample_id}: expected_event_phase must be one of {sorted(EVENT_PHASES)}")
        if not isinstance(row.get("expected_should_alert"), bool):
            errors.append(f"{sample_id}: expected_should_alert must be boolean")
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        for field in ("dataset", "license", "license_url", "privacy_review_status"):
            if not str(source.get(field, "")).strip():
                errors.append(f"{sample_id}: missing source.{field}")
        image_rel = required_string(row, "image_path", sample_id, errors)
        mask_rel = required_string(row, "semantic_mask_path", sample_id, errors)
        image = (root / image_rel).resolve()
        mask = (root / mask_rel).resolve()
        try:
            image.relative_to(root)
            mask.relative_to(root)
        except ValueError:
            errors.append(f"{sample_id}: image or semantic mask escapes dataset root")
            continue
        if not image.is_file() or not mask.is_file():
            errors.append(f"{sample_id}: missing image or semantic mask")
            continue
        image_sha = sha256_file(image)
        mask_sha = sha256_file(mask)
        if image_sha != row.get("image_sha256"):
            errors.append(f"{sample_id}: image SHA256 mismatch")
        if mask_sha != row.get("semantic_mask_sha256"):
            errors.append(f"{sample_id}: semantic mask SHA256 mismatch")
        if image_sha in seen_images:
            errors.append(f"{sample_id}: duplicate image with {seen_images[image_sha]}")
        seen_images[image_sha] = sample_id
        if mask_sha in seen_masks:
            errors.append(f"{sample_id}: duplicate semantic mask with {seen_masks[mask_sha]}")
        seen_masks[mask_sha] = sample_id
        with Image.open(image) as rgb, Image.open(mask) as semantic:
            if rgb.size != semantic.size:
                errors.append(f"{sample_id}: image/semantic mask dimensions differ")
            values = np.unique(np.asarray(semantic.convert("L"), dtype=np.uint8))
        unknown_values = [int(value) for value in values if int(value) >= len(SEMANTIC_CLASSES)]
        if unknown_values:
            errors.append(f"{sample_id}: semantic mask contains unsupported class IDs {unknown_values}")
        for value in values:
            if int(value) < len(SEMANTIC_CLASSES):
                class_pixels[SEMANTIC_CLASSES[int(value)]] += 1
    for session_id, splits in session_splits.items():
        if len(splits) != 1:
            errors.append(f"session {session_id}: split leakage across {sorted(splits)}")
    sequence_summary: list[dict[str, Any]] = []
    for sequence_id, items in sequence_rows.items():
        ordered = sorted(items, key=lambda item: int(item.get("frame_index", -1)))
        indexes = [int(item.get("frame_index", -1)) for item in ordered]
        first = ordered[0] if ordered else {}
        if indexes != list(range(len(indexes))):
            errors.append(f"{sequence_id}: frame_index must be contiguous from 0")
        invariant_fields = ("split", "scene_bucket")
        for field in invariant_fields:
            if any(item.get(field) != first.get(field) for item in items):
                errors.append(f"{sequence_id}: {field} must be constant inside a sequence")
        sessions = {row_session_id(item) for item in items}
        if len(sessions) != 1:
            errors.append(f"{sequence_id}: source session_id must be constant inside a sequence")
        sequence_summary.append({
            "sequence_id": sequence_id,
            "split": first.get("split"),
            "scene_bucket": first.get("scene_bucket"),
            "session_id": next(iter(sessions), ""),
            "frame_count": len(items),
        })
    return errors, {
        "row_count": len(rows),
        "sequence_count": len(sequence_rows),
        "sequences": sequence_summary,
        "class_presence_frame_count": dict(class_pixels),
    }


def validate_access_lock(root: Path, train_rows: list[dict[str, Any]], blind_rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    policy_path = root / "access_policy.json"
    if not policy_path.is_file():
        return ["missing access_policy.json"]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("training_manifest") != "training_manifest.jsonl":
        errors.append("access policy must name training_manifest.jsonl as the only trainer input")
    if policy.get("blind_manifest") != "blind_holdout/manifest.jsonl":
        errors.append("access policy must name blind_holdout/manifest.jsonl")
    if policy.get("blind_label_access") != "benchmark_only":
        errors.append("blind labels must be benchmark_only")
    forbidden = set(policy.get("forbidden_training_paths", []))
    if "blind_holdout" not in forbidden:
        errors.append("access policy must forbid blind_holdout for training")
    if any(row.get("split") == "blind" for row in train_rows):
        errors.append("training manifest contains blind metadata")
    if any(row.get("split") != "blind" for row in blind_rows):
        errors.append("blind holdout manifest contains non-blind metadata")
    training_ids = {str(row.get("id")) for row in train_rows}
    blind_ids = {str(row.get("id")) for row in blind_rows}
    if training_ids & blind_ids:
        errors.append("training and blind manifests overlap by sample id")
    session_splits: dict[str, set[str]] = defaultdict(set)
    for row in train_rows + blind_rows:
        session_id = row_session_id(row)
        if session_id:
            session_splits[session_id].add(str(row.get("split", "")))
    for session_id, splits in session_splits.items():
        if len(splits) != 1:
            errors.append(f"session {session_id}: split leakage across separated manifests {sorted(splits)}")
    return errors


def validate_v3_coverage(train: dict[str, Any], blind: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    train_sequences = train["sequences"]
    blind_sequences = blind["sequences"]
    if train["row_count"] != 300 or blind["row_count"] != 120:
        errors.append("v3 requires 300 train/dev frames plus 120 blind frames (420 total)")
    if len(train_sequences) != 6 or any(item["frame_count"] != 50 for item in train_sequences):
        errors.append("v3 requires six train/dev continuous sequences of exactly 50 frames")
    if len(blind_sequences) != 2 or any(item["frame_count"] != 60 for item in blind_sequences):
        errors.append("v3 requires two blind continuous sequences of exactly 60 frames")
    blind_sessions = {str(item["session_id"]) for item in blind_sequences}
    if len(blind_sessions) != 2 or "" in blind_sessions:
        errors.append("v3 requires the two blind sequences to use two distinct non-empty source sessions")
    seen_buckets = {str(item["scene_bucket"]) for item in train_sequences}
    missing_buckets = sorted(set(SCENE_BUCKETS) - seen_buckets)
    if missing_buckets:
        errors.append("v3 train/dev coverage missing scene buckets: " + ", ".join(missing_buckets))
    splits = {str(item["split"]) for item in train_sequences}
    if not {"train", "dev"}.issubset(splits):
        errors.append("v3 train/dev sequences must include both train and dev splits")
    missing_classes = [name for name in SEMANTIC_CLASSES if not train["class_presence_frame_count"].get(name)]
    if missing_classes:
        errors.append("v3 train/dev masks do not contain all four semantic classes: " + ", ".join(missing_classes))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-manifest", default="training_manifest.jsonl")
    parser.add_argument("--blind-manifest", default="blind_holdout/manifest.jsonl")
    parser.add_argument("--require-v3-coverage", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional report path; never place it in blind_holdout.")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    train_rows = load_jsonl(root / args.training_manifest)
    blind_rows = load_jsonl(root / args.blind_manifest)
    errors, train_summary = validate_rows(train_rows, root, {"train", "dev"})
    blind_errors, blind_summary = validate_rows(blind_rows, root, {"blind"})
    errors.extend(blind_errors)
    errors.extend(validate_access_lock(root, train_rows, blind_rows))
    if args.require_v3_coverage:
        errors.extend(validate_v3_coverage(train_summary, blind_summary))
    report = {
        "ok": not errors,
        "dataset_root": str(root),
        "training": train_summary,
        "blind_holdout": blind_summary,
        "errors": errors,
    }
    if args.report:
        report_path = args.report.resolve()
        if "blind_holdout" in report_path.parts:
            raise SystemExit("report must not be written inside blind_holdout")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
