#!/usr/bin/env python3
"""Audit train-only synthetic counterfactual provenance and frozen response.

The audit verifies parent-source binding, descendant leakage rules, final image
hashes, pair structure, and threshold-free frozen SegFormer response.  Natural
language edits never create pixel truth; objects/masks must remain absent.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_silver_segformer_free_space_probe as clearance


SCHEMA = "blindassist_public_silver_synthetic_counterfactual_response_audit_v2"
RESPONSE_FIELDS = (
    "median_lower_nonwalkable_mean",
    "median_core_nonwalkable_mean",
    "median_path_lower_nonwalkable_mean",
    "median_path_offset_mean",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def validate_contract(
    dataset_root: Path,
    spec: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    parent_index: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    failures: list[str] = []
    provenance = spec.get("provenance_contract") or {}
    leakage = spec.get("leakage_contract") or {}
    if provenance.get("role") != "train_only_representation_augmentation":
        failures.append("dataset_not_train_only")
    if provenance.get("validation_or_test_use_authorized") is not False:
        failures.append("validation_or_test_not_closed")
    if provenance.get("pixel_mask_available") is not False or provenance.get("pixel_supervision_role") != "none":
        failures.append("unverified_pixel_supervision_present")
    if leakage.get("rice_street_external_pressure_is_trainable") is not False:
        failures.append("rice_external_pressure_not_isolated")
    counts = spec.get("counts") or {}
    expected_records = int(counts.get("accepted_total", 0))
    expected_pairs = int(counts.get("synthetic_risk", 0))
    minimum_family_sources = int(counts.get("minimum_parent_sources_per_obstruction_family", 0))
    if len(rows) != expected_records or expected_records <= 0:
        failures.append("unexpected_record_count")
    grouped: dict[str, list[dict[str, Any]]] = {}
    final_hashes: set[str] = set()
    for row in rows:
        if row.get("split") != "train" or row.get("status") != "accepted":
            failures.append(f"non_train_or_nonaccepted:{row.get('id')}")
        if row.get("objects"):
            failures.append(f"unverified_geometry:{row.get('id')}")
        image_path = (dataset_root / str(row.get("image_path", ""))).resolve()
        if not image_path.is_relative_to(dataset_root.resolve()) or not image_path.is_file():
            failures.append(f"missing_or_escaped_image:{row.get('id')}")
            continue
        attributes = row.get("attributes") or {}
        expected_hash = attributes.get("final_image_sha256")
        actual_hash = common.sha256_file(image_path)
        if expected_hash != actual_hash:
            failures.append(f"final_hash_mismatch:{row.get('id')}")
        if actual_hash in final_hashes:
            failures.append(f"duplicate_final_image:{row.get('id')}")
        final_hashes.add(actual_hash)
        pair_id = attributes.get("counterfactual_pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            failures.append(f"missing_pair_id:{row.get('id')}")
            continue
        grouped.setdefault(pair_id, []).append(row)
        source = row.get("source") or {}
        parent_hash = source.get("parent_frame_sha256")
        indexed = parent_index.get(str(parent_hash))
        if indexed is None:
            failures.append(f"unknown_parent_hash:{row.get('id')}")
        elif indexed != (source.get("parent_episode_id"), source.get("parent_source_id")):
            failures.append(f"parent_binding_mismatch:{row.get('id')}")
        if attributes.get("synthetic") is True and not attributes.get("generated_source_sha256"):
            failures.append(f"missing_generated_source_hash:{row.get('id')}")
    for pair_id, members in grouped.items():
        states = {(row.get("attributes") or {}).get("risk_state") for row in members}
        synthetic = {(row.get("attributes") or {}).get("synthetic") for row in members}
        parent_sources = {(row.get("source") or {}).get("parent_source_id") for row in members}
        if len(members) != 2 or states != {"clear", "risk"} or synthetic != {False, True} or len(parent_sources) != 1:
            failures.append(f"invalid_pair_structure:{pair_id}")
    if len(grouped) != expected_pairs or expected_pairs <= 0:
        failures.append("unexpected_pair_count")
    family_sources: dict[str, set[str]] = {}
    for row in rows:
        attributes = row.get("attributes") or {}
        if attributes.get("synthetic") is True:
            family = str(attributes.get("obstruction_family"))
            family_sources.setdefault(family, set()).add(str((row.get("source") or {}).get("parent_source_id")))
    if minimum_family_sources < 2:
        failures.append("weak_family_source_minimum")
    for family, sources in family_sources.items():
        if len(sources) < minimum_family_sources:
            failures.append(f"insufficient_family_sources:{family}")
    return {
        "passed": not failures,
        "failures": failures,
        "record_count": len(rows),
        "counterfactual_pair_count": len(grouped),
        "unique_final_image_count": len(final_hashes),
        "obstruction_family_parent_source_counts": {key: len(value) for key, value in sorted(family_sources.items())},
    }


def response_rows(
    dataset_root: Path,
    rows: Sequence[dict[str, Any]],
    teacher: clearance.FrozenTeacher,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["attributes"]["counterfactual_pair_id"], []).append(row)
    result: list[dict[str, Any]] = []
    for pair_id, members in sorted(grouped.items()):
        scores: dict[str, dict[str, float | int]] = {}
        ids: dict[str, str] = {}
        for row in members:
            state = row["attributes"]["risk_state"]
            image = cv2.imread(str(dataset_root / row["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode dataset image: {row['image_path']}")
            scores[state] = clearance.score(teacher.describe([image], batch_size=batch_size))
            ids[state] = row["id"]
        deltas = {field: float(scores["risk"][field]) - float(scores["clear"][field]) for field in RESPONSE_FIELDS}
        result.append({
            "counterfactual_pair_id": pair_id,
            "clear_id": ids["clear"],
            "risk_id": ids["risk"],
            "risk_minus_clear": deltas,
            "all_preregistered_channels_increase": all(value > 0.0 for value in deltas.values()),
        })
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.dataset_root, args.package_root, args.model_dir, args.model_review, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    dataset_root = args.dataset_root.resolve()
    spec = common.load_json(dataset_root / "dataset_spec.json")
    rows = load_jsonl(dataset_root / "manifest.jsonl")
    episodes, _ = common.load_episode_specs(args.package_root.resolve())
    parent_index = {
        frame["sha256"]: (episode["episode_id"], episode["source_id"])
        for episode in episodes
        for frame in episode["frames"]
    }
    provenance = validate_contract(dataset_root, spec, rows, parent_index)
    review = common.load_json(args.model_review.resolve())
    contact = review.get("contact_sheet") or {}
    contact_path = (dataset_root / str(contact.get("path", ""))).resolve()
    manifest_ids = {row["id"] for row in rows}
    reviewed_ids = {
        value
        for pair in review.get("pairs", [])
        if isinstance(pair, dict) and pair.get("accepted") is True
        for value in (pair.get("clear_id"), pair.get("risk_id"))
        if isinstance(value, str)
    }
    visual_review_gate = bool(
        review.get("schema") == "blindassist_synthetic_counterfactual_gpt_vlm_pair_review_v1"
        and review.get("all_pairs_reviewed") is True
        and review.get("all_pairs_accepted_for_train_only_pair_ranking") is True
        and review.get("validation_or_test_use_authorized") is False
        and review.get("pixel_or_mask_supervision_authorized") is False
        and reviewed_ids == manifest_ids
        and contact_path.is_relative_to(dataset_root)
        and contact_path.is_file()
        and contact.get("sha256") == common.sha256_file(contact_path)
    )
    teacher = clearance.FrozenTeacher(args.model_dir.resolve())
    responses = response_rows(dataset_root, rows, teacher, batch_size=args.batch_size)
    response_gate = bool(responses and all(row["all_preregistered_channels_increase"] for row in responses))
    short_run = bool(provenance["passed"] and visual_review_gate)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "inputs": {
            "dataset_spec_sha256": common.sha256_file(dataset_root / "dataset_spec.json"),
            "generation_records_sha256": common.sha256_file(dataset_root / "generation_records.jsonl"),
            "manifest_sha256": common.sha256_file(dataset_root / "manifest.jsonl"),
            "gpt_vlm_pair_review_sha256": common.sha256_file(args.model_review),
            "package_root": str(args.package_root.resolve()),
            "model_weights_sha256": common.sha256_file(args.model_dir / "pytorch_model.bin"),
        },
        "provenance_and_leakage_gate": provenance,
        "gpt_vlm_visual_review_gate": {
            "passed": visual_review_gate,
            "reviewed_manifest_id_count": len(reviewed_ids),
            "reviewed_pair_count": len(review.get("pairs", [])),
            "human_truth_claimed": False,
        },
        "frozen_response_contract": {
            "model": "nvidia/segformer-b2-finetuned-ade-512-512",
            "trainable_parameters": 0,
            "response_fields": list(RESPONSE_FIELDS),
            "acceptance": "risk_minus_clear strictly positive for every field and every pair",
            "threshold_fitted": False,
        },
        "counterfactual_responses": responses,
        "frozen_response_gate": {"passed": response_gate},
        "hard_counterexample_count": sum(not row["all_preregistered_channels_increase"] for row in responses),
        "parent_source_isolated_representation_short_run_authorized": short_run,
        "pixel_or_mask_supervision_authorized": False,
        "validation_or_test_use_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
        "evidence_limit": "Six train-only GPT/VLM-reviewed synthetic edits over six real parent sources. Frozen response is diagnostic, not an inclusion gate: hard nonresponses are retained to train the representation. Real Rice and source-isolated evaluation remain mandatory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-review", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "provenance_gate_passed": report["provenance_and_leakage_gate"]["passed"],
        "frozen_response_gate_passed": report["frozen_response_gate"]["passed"],
        "visual_review_gate_passed": report["gpt_vlm_visual_review_gate"]["passed"],
        "short_run_authorized": report["parent_source_isolated_representation_short_run_authorized"],
        "output_sha256": common.sha256_file(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
