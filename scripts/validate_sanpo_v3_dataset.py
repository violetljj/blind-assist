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
import math
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
LABEL_AUTHORITIES = {
    "source_ground_truth",
    "procedural_ground_truth",
    "teacher_consensus_pseudo_label",
}
PSEUDO_LABEL_MIN_IOU = 0.90
PSEUDO_LABEL_MIN_TEMPORAL_CONSISTENCY = 0.85
PROCEDURAL_PROVENANCE_SCHEMA = "blindassist_procedural_ground_truth_v1"
PROCEDURAL_GENERATORS = {"tactile_occupied_compositor_v1"}
EXPANDED_COVERAGE_FORMAT = "blindassist_sanpo_v4_coverage_policy_v1"


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


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_bound_provenance_file(
    root: Path, provenance: dict[str, Any], path_field: str, sha_field: str,
    sample_id: str, errors: list[str],
) -> None:
    relative = str(provenance.get(path_field, "")).strip()
    expected_sha = str(provenance.get(sha_field, "")).strip()
    if not relative or len(expected_sha) != 64:
        errors.append(f"{sample_id}: procedural_ground_truth requires {path_field} and SHA256-bound {sha_field}")
        return
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"{sample_id}: procedural provenance path escapes dataset root: {relative}")
        return
    if not path.is_file():
        errors.append(f"{sample_id}: missing procedural provenance file: {relative}")
    elif sha256_file(path) != expected_sha:
        errors.append(f"{sample_id}: procedural provenance SHA256 mismatch: {relative}")


def validate_label_authority(
    row: dict[str, Any], sample_id: str, split: str, root: Path | None = None,
) -> list[str]:
    """Validate label provenance without allowing pseudo labels into dev/blind."""
    errors: list[str] = []
    authority = str(row.get("label_authority", "")).strip()
    provenance = row.get("label_provenance") if isinstance(row.get("label_provenance"), dict) else {}
    if authority not in LABEL_AUTHORITIES:
        return [f"{sample_id}: label_authority must be one of {sorted(LABEL_AUTHORITIES)}"]
    if split in {"dev", "blind"} and authority == "teacher_consensus_pseudo_label":
        errors.append(f"{sample_id}: {split} forbids teacher/pseudo labels")
    if authority == "source_ground_truth":
        for field in ("source_mask_sha256", "mapped_mask_sha256", "mapping_sha256", "annotation_kind"):
            if not str(provenance.get(field, "")).strip():
                errors.append(f"{sample_id}: source_ground_truth missing label_provenance.{field}")
        for field in ("source_mask_sha256", "mapped_mask_sha256", "mapping_sha256"):
            if len(str(provenance.get(field, ""))) != 64:
                errors.append(f"{sample_id}: label_provenance.{field} must be SHA256")
        if provenance.get("mapped_mask_sha256") != row.get("semantic_mask_sha256"):
            errors.append(f"{sample_id}: mapped source-ground-truth mask is not bound to semantic mask")
    elif authority == "procedural_ground_truth":
        if provenance.get("schema") != PROCEDURAL_PROVENANCE_SCHEMA:
            errors.append(f"{sample_id}: procedural_ground_truth requires schema {PROCEDURAL_PROVENANCE_SCHEMA}")
        if provenance.get("generator_id") not in PROCEDURAL_GENERATORS:
            errors.append(f"{sample_id}: procedural generator is not allow-listed")
        if not isinstance(provenance.get("seed"), int) or isinstance(provenance.get("seed"), bool):
            errors.append(f"{sample_id}: procedural_ground_truth seed must be an integer")
        matrix = provenance.get("transform_matrix")
        valid_matrix = (
            isinstance(matrix, list) and len(matrix) == 3
            and all(isinstance(row_values, list) and len(row_values) == 3 for row_values in matrix)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    for row_values in matrix for value in row_values)
        )
        if not valid_matrix:
            errors.append(f"{sample_id}: procedural_ground_truth requires a finite 3x3 transform_matrix")
        elif provenance.get("transform_sha256") != _canonical_json_sha256(matrix):
            errors.append(f"{sample_id}: transform_matrix SHA256 mismatch")
        inputs = provenance.get("source_masks")
        if not isinstance(inputs, list) or len(inputs) != 2:
            errors.append(f"{sample_id}: procedural_ground_truth requires exactly two source GT masks")
        else:
            roles = {str(item.get("role", "")) for item in inputs if isinstance(item, dict)}
            if roles != {"tactile_ground_truth", "obstacle_ground_truth"}:
                errors.append(f"{sample_id}: procedural source masks require tactile and obstacle GT roles")
            source_ids = {str(item.get("source_id", "")).strip() for item in inputs if isinstance(item, dict)}
            if len(source_ids) != 2 or "" in source_ids:
                errors.append(f"{sample_id}: procedural source GT masks require two distinct attested source_ids")
            identities = {(str(item.get("path", "")), str(item.get("sha256", "")))
                          for item in inputs if isinstance(item, dict)}
            if len(identities) != 2:
                errors.append(f"{sample_id}: procedural source GT masks must be distinct")
            if root is not None:
                for index, item in enumerate(inputs):
                    if not isinstance(item, dict):
                        errors.append(f"{sample_id}: procedural source mask {index} must be an object")
                        continue
                    _validate_bound_provenance_file(
                        root, item, "path", "sha256", sample_id, errors,
                    )
        raw_assets = provenance.get("source_assets")
        required_raw_roles = {"guide_rgb", "guide_polygon", "sanpo_rgb", "sanpo_raw_mask"}
        if not isinstance(raw_assets, list) or len(raw_assets) != len(required_raw_roles):
            errors.append(f"{sample_id}: procedural_ground_truth requires four raw source assets")
        else:
            roles = {str(item.get("role", "")) for item in raw_assets if isinstance(item, dict)}
            if roles != required_raw_roles:
                errors.append(f"{sample_id}: procedural raw source roles must be {sorted(required_raw_roles)}")
            for index, item in enumerate(raw_assets):
                if not isinstance(item, dict):
                    errors.append(f"{sample_id}: procedural raw source asset {index} must be an object")
                    continue
                if not str(item.get("source_id", "")).strip():
                    errors.append(f"{sample_id}: procedural raw source asset {index} requires source_id")
                if root is not None:
                    _validate_bound_provenance_file(
                        root, item, "path", "sha256", sample_id, errors,
                    )
        if provenance.get("output_mask_sha256") != row.get("semantic_mask_sha256"):
            errors.append(f"{sample_id}: procedural output mask is not bound to semantic mask")
        if root is not None:
            _validate_bound_provenance_file(
                root, provenance, "generator_code_path", "generator_code_sha256", sample_id, errors,
            )
            _validate_bound_provenance_file(
                root, provenance, "generator_config_path", "generator_config_sha256", sample_id, errors,
            )
    else:
        teachers = provenance.get("teachers")
        if not isinstance(teachers, list) or len(teachers) != 2:
            errors.append(f"{sample_id}: teacher consensus requires exactly two teachers")
        else:
            identities: set[tuple[str, str]] = set()
            for index, teacher in enumerate(teachers):
                if not isinstance(teacher, dict):
                    errors.append(f"{sample_id}: teacher {index} must be an object")
                    continue
                model_id = str(teacher.get("model_id", "")).strip()
                weights_sha = str(teacher.get("weights_sha256", "")).strip()
                output_sha = str(teacher.get("output_sha256", "")).strip()
                if not model_id or len(weights_sha) != 64 or len(output_sha) != 64:
                    errors.append(f"{sample_id}: teacher {index} requires model_id and SHA256-bound weights/output")
                identities.add((model_id, weights_sha))
            if len(identities) != 2:
                errors.append(f"{sample_id}: teacher consensus requires two independent model identities")
        try:
            agreement_iou = float(provenance.get("agreement_iou", 0.0))
            temporal_consistency = float(provenance.get("temporal_consistency", 0.0))
        except (TypeError, ValueError):
            agreement_iou = temporal_consistency = -1.0
        if agreement_iou < PSEUDO_LABEL_MIN_IOU:
            errors.append(f"{sample_id}: teacher consensus IoU is below {PSEUDO_LABEL_MIN_IOU}")
        if temporal_consistency < PSEUDO_LABEL_MIN_TEMPORAL_CONSISTENCY:
            errors.append(f"{sample_id}: temporal consistency is below {PSEUDO_LABEL_MIN_TEMPORAL_CONSISTENCY}")
        if provenance.get("consensus_mask_sha256") != row.get("semantic_mask_sha256"):
            errors.append(f"{sample_id}: consensus mask SHA256 is not bound to semantic mask")
    return errors


def validate_rows(rows: list[dict[str, Any]], root: Path, expected_split: set[str]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    sequence_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    session_splits: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    seen_images: dict[str, str] = {}
    class_pixels: Counter[str] = Counter()
    semantic_mask_hash_counts: Counter[str] = Counter()
    raw_mask_hash_counts: Counter[str] = Counter()
    for row in rows:
        sample_id = required_string(row, "id", "<row>", errors)
        if sample_id in seen_ids:
            errors.append(f"{sample_id}: duplicate id")
        seen_ids.add(sample_id)
        split = required_string(row, "split", sample_id, errors)
        if split not in expected_split or split not in SPLITS:
            errors.append(f"{sample_id}: split {split!r} is not allowed in this manifest")
        errors.extend(validate_label_authority(row, sample_id, split, root))
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
        event_fields = ("risk_event_id", "expected_event_phase", "expected_should_alert")
        if row.get("benchmark_kind") == "semantic_segmentation_only":
            if any(row.get(field) not in (None, "") for field in event_fields):
                errors.append(f"{sample_id}: semantic_segmentation_only must not carry risk-event labels")
        else:
            required_string(row, "risk_event_id", sample_id, errors)
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
        # SANPO machine annotations can legitimately reuse one mask across
        # distinct RGB frames in a continuous session.  Duplicate samples are
        # therefore identified by RGB SHA, while raw-mask SHA is enforced by
        # the source inventory and forbidden from crossing target splits.
        semantic_mask_hash_counts[mask_sha] += 1
        provenance = row.get("label_provenance") if isinstance(row.get("label_provenance"), dict) else {}
        raw_mask_sha = str(provenance.get("source_mask_sha256", "")).strip()
        if raw_mask_sha:
            raw_mask_hash_counts[raw_mask_sha] += 1
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
            "official_split": str(
                (first.get("source") if isinstance(first.get("source"), dict) else {}).get(
                    "official_split", ""
                )
            ).strip(),
            "frame_count": len(items),
        })
    duplicate_semantic_rows = sum(count - 1 for count in semantic_mask_hash_counts.values() if count > 1)
    duplicate_raw_rows = sum(count - 1 for count in raw_mask_hash_counts.values() if count > 1)
    return errors, {
        "row_count": len(rows),
        "sequence_count": len(sequence_rows),
        "sequences": sequence_summary,
        "class_presence_frame_count": dict(class_pixels),
        "duplicate_mask_observation": {
            "semantic_duplicate_hash_count": sum(count > 1 for count in semantic_mask_hash_counts.values()),
            "semantic_duplicate_row_count": duplicate_semantic_rows,
            "semantic_duplicate_row_ratio": round(duplicate_semantic_rows / max(1, len(rows)), 6),
            "raw_duplicate_hash_count": sum(count > 1 for count in raw_mask_hash_counts.values()),
            "raw_duplicate_row_count": duplicate_raw_rows,
            "raw_duplicate_row_ratio": round(duplicate_raw_rows / max(1, len(rows)), 6),
            "policy": "observation_only_with_rgb_duplicate_rejection_and_raw_cross_split_rejection",
        },
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


def validate_expanded_coverage(
    train: dict[str, Any], blind: dict[str, Any], policy: dict[str, Any],
) -> list[str]:
    """Validate session-scaled coverage without weakening the blind lock."""
    errors: list[str] = []
    if policy.get("format") != EXPANDED_COVERAGE_FORMAT:
        return [f"coverage policy format must be {EXPANDED_COVERAGE_FORMAT}"]
    train_sequences = train["sequences"]
    blind_sequences = blind["sequences"]
    sequence_frame_count = int(policy.get("sequence_frame_count", 0))
    blind_frame_count = int(policy.get("blind_sequence_frame_count", 0))
    blind_sequence_count = int(policy.get("blind_sequence_count", 0))
    if sequence_frame_count <= 0 or any(
        int(item["frame_count"]) != sequence_frame_count for item in train_sequences
    ):
        errors.append(
            f"expanded coverage requires every train/dev sequence to contain exactly {sequence_frame_count} frames"
        )
    if len(blind_sequences) != blind_sequence_count or any(
        int(item["frame_count"]) != blind_frame_count for item in blind_sequences
    ):
        errors.append(
            f"expanded coverage requires exactly {blind_sequence_count} blind sequences of {blind_frame_count} frames"
        )
    sessions_by_split = {
        split: {str(item["session_id"]) for item in train_sequences if item["split"] == split}
        for split in ("train", "dev")
    }
    for split, field in (("train", "min_train_sessions"), ("dev", "min_dev_sessions")):
        minimum = int(policy.get(field, 0))
        if len(sessions_by_split[split]) < minimum:
            errors.append(
                f"expanded coverage requires at least {minimum} distinct {split} sessions, "
                f"got {len(sessions_by_split[split])}"
            )
    blind_sessions = {str(item["session_id"]) for item in blind_sequences}
    if len(blind_sessions) != blind_sequence_count or "" in blind_sessions:
        errors.append("expanded coverage requires distinct non-empty blind source sessions")

    required_scenes = policy.get("required_scene_sessions")
    if not isinstance(required_scenes, dict) or not required_scenes:
        errors.append("expanded coverage requires required_scene_sessions")
    else:
        for bucket, requirements in required_scenes.items():
            if bucket not in SCENE_BUCKETS or not isinstance(requirements, dict):
                errors.append(f"expanded coverage has unsupported scene requirement {bucket!r}")
                continue
            matching = [item for item in train_sequences if item["scene_bucket"] == bucket]
            for split in ("train", "dev"):
                minimum = int(requirements.get(split, 0))
                actual = len({str(item["session_id"]) for item in matching if item["split"] == split})
                if actual < minimum:
                    errors.append(
                        f"scene {bucket} requires at least {minimum} distinct {split} sessions, got {actual}"
                    )
            total_minimum = int(requirements.get("total", 0))
            actual_total = len({str(item["session_id"]) for item in matching})
            if actual_total < total_minimum:
                errors.append(
                    f"scene {bucket} requires at least {total_minimum} distinct total sessions, got {actual_total}"
                )

    official_split_policy = policy.get("official_split_by_target_split")
    if not isinstance(official_split_policy, dict):
        errors.append("expanded coverage requires official_split_by_target_split")
    else:
        for item in train_sequences + blind_sequences:
            target_split = str(item["split"])
            expected = str(official_split_policy.get(target_split, "")).strip()
            if not expected or item.get("official_split") != expected:
                errors.append(
                    f"{item['sequence_id']}: official split {item.get('official_split')!r} "
                    f"does not satisfy target split {target_split!r} -> {expected!r}"
                )
    missing_classes = [
        name for name in SEMANTIC_CLASSES if not train["class_presence_frame_count"].get(name)
    ]
    if missing_classes:
        errors.append(
            "expanded train/dev masks do not contain all four semantic classes: " + ", ".join(missing_classes)
        )
    return errors


def validate_v3_coverage(
    train: dict[str, Any], blind: dict[str, Any], coverage_policy: dict[str, Any] | None = None,
) -> list[str]:
    if coverage_policy is not None:
        return validate_expanded_coverage(train, blind, coverage_policy)
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
        coverage_policy = None
        recipe_path = root / "assembly_recipe.json"
        if recipe_path.is_file():
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            if isinstance(recipe.get("coverage_policy"), dict):
                coverage_policy = recipe["coverage_policy"]
        errors.extend(validate_v3_coverage(train_summary, blind_summary, coverage_policy))
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
