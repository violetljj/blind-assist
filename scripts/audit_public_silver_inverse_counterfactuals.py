#!/usr/bin/env python3
"""Audit train-only inverse counterfactual pairs with licensed real risk ends."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_inverse_counterfactual_audit_v1"
DATASET_SCHEMA = "blindassist_public_silver_inverse_counterfactual_dataset_v1"
REVIEW_SCHEMA = "blindassist_gpt_vlm_inverse_counterfactual_pair_review_v1"
ALLOWED_LICENSES = {"CC0", "CC BY 4.0", "CC BY-SA 4.0"}


def verify_sidecar(path: Path) -> str:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    actual = common.sha256_file(path)
    if expected != actual:
        raise ValueError(f"sidecar mismatch: {path}")
    return actual


def validate_pair_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pair_id = row.get("attributes", {}).get("counterfactual_pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("missing counterfactual_pair_id")
        grouped.setdefault(pair_id, []).append(row)
    parent_sources: set[str] = set()
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"pair must contain two endpoints: {pair_id}")
        states = {row["attributes"].get("risk_state"): row for row in members}
        if set(states) != {"clear", "risk"}:
            raise ValueError(f"pair must contain one clear and one risk: {pair_id}")
        clear, risk = states["clear"], states["risk"]
        if clear.get("label") != 0 or clear["attributes"].get("synthetic") is not True or clear["attributes"].get("endpoint_role") != "gpt_removed_obstacle":
            raise ValueError(f"invalid synthetic clear endpoint: {pair_id}")
        if risk.get("label") != 1 or risk["attributes"].get("synthetic") is not False or risk["attributes"].get("endpoint_role") != "real_licensed_risk":
            raise ValueError(f"invalid real risk endpoint: {pair_id}")
        clear_source, risk_source = clear.get("source", {}), risk.get("source", {})
        if clear_source != risk_source:
            raise ValueError(f"pair source contracts differ: {pair_id}")
        source_id = clear_source.get("parent_source_id")
        if not isinstance(source_id, str) or source_id in parent_sources:
            raise ValueError(f"parent source must be unique per pair: {pair_id}")
        parent_sources.add(source_id)
        if clear_source.get("license") not in ALLOWED_LICENSES:
            raise ValueError(f"unaccepted license: {pair_id}")
        if clear_source.get("lineage_rule") != "hold_out_source_and_all_descendants":
            raise ValueError(f"missing descendant isolation: {pair_id}")
        if not str(clear_source.get("source_page_url", "")).startswith("https://commons.wikimedia.org/wiki/File:"):
            raise ValueError(f"invalid source page: {pair_id}")
    return {"pair_count": len(grouped), "parent_source_count": len(parent_sources), "pair_ids": sorted(grouped)}


def reconstruct_risk_derivative(original_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    source = Image.open(original_path).convert("RGB")
    source.thumbnail(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size)
    canvas.paste(source, ((target_size[0] - source.width) // 2, (target_size[1] - source.height) // 2))
    return np.asarray(canvas, dtype=np.int16)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.dataset_root, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    root = args.dataset_root.resolve()
    spec_path, manifest_path, review_path = root / "dataset_spec.json", root / "manifest.jsonl", root / "qa" / "gpt_vlm_pair_review.json"
    spec_sha, manifest_sha, review_sha = (verify_sidecar(path) for path in (spec_path, manifest_path, review_path))
    spec = common.load_json(spec_path)
    review = common.load_json(review_path)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if spec.get("schema") != DATASET_SCHEMA or spec.get("purpose") != "train_only_static_pair_ranking":
        raise ValueError("invalid dataset contract")
    if spec.get("object_annotations") != 0 or spec.get("pixel_masks") != 0:
        raise ValueError("inverse dataset must not claim object or pixel truth")
    auth = spec.get("authorizations", {})
    if auth.get("train_only_representation_probe") is not True or any(auth.get(key) is not False for key in ("validation", "test", "calibration", "blind", "real_source_count_credit", "android_runtime_change", "production_model_replacement")):
        raise ValueError("invalid authorization boundary")
    pair_contract = validate_pair_rows(rows)
    if pair_contract["pair_count"] != spec.get("accepted_pairs"):
        raise ValueError("spec pair count mismatch")
    if review.get("schema") != REVIEW_SCHEMA or review.get("dataset_acceptance") is not True:
        raise ValueError("model review did not accept dataset")
    reviews = {row.get("counterfactual_pair_id"): row for row in review.get("pairs", [])}
    if set(reviews) != set(pair_contract["pair_ids"]) or not all(row.get("accepted") is True for row in reviews.values()):
        raise ValueError("review coverage mismatch")

    image_audit: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        image_path = root / row["image_path"]
        if common.sha256_file(image_path) != row["sha256"]:
            raise ValueError(f"image hash mismatch: {row['image_id']}")
        with Image.open(image_path) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"image dimensions mismatch: {row['image_id']}")
        source = row["source"]
        original_path = root / source["original_file_path"]
        if common.sha256_file(original_path) != source["original_file_sha256"]:
            raise ValueError(f"original source hash mismatch: {row['image_id']}")
        grouped.setdefault(row["attributes"]["counterfactual_pair_id"], []).append(row)
        image_audit.append({"image_id": row["image_id"], "sha256_verified": True, "original_source_sha256_verified": True})

    derivative_audit: list[dict[str, Any]] = []
    for pair_id, members in sorted(grouped.items()):
        risk = next(row for row in members if row["attributes"]["risk_state"] == "risk")
        actual = np.asarray(Image.open(root / risk["image_path"]).convert("RGB"), dtype=np.int16)
        expected = reconstruct_risk_derivative(root / risk["source"]["original_file_path"], (risk["width"], risk["height"]))
        delta = np.abs(actual - expected)
        mean_error, p99_error = float(delta.mean()), float(np.percentile(delta, 99))
        if mean_error > 2.0 or p99_error > 12.0:
            raise ValueError(f"real risk endpoint is not a bounded resize derivative: {pair_id}")
        derivative_audit.append({"pair_id": pair_id, "mean_abs_decode_error": mean_error, "p99_abs_decode_error": p99_error, "bounded_resize_derivative": True})

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"dataset_spec_sha256": spec_sha, "manifest_sha256": manifest_sha, "model_review_sha256": review_sha},
        "pair_contract": pair_contract,
        "image_audit": image_audit,
        "real_risk_derivative_audit": derivative_audit,
        "provenance_gate_passed": True,
        "parent_source_isolated_representation_short_run_authorized": True,
        "five_seed_bootstrap_authorized": False,
        "evidence_limit": "Train-only GPT/VLM provisional inverse counterfactuals. No object geometry, evaluation truth, calibration, blind, Android or production authority.",
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False,
        "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({"ok": True, "pair_count": payload["pair_contract"]["pair_count"], "gate_passed": payload["provenance_gate_passed"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
