#!/usr/bin/env python3
"""Create a new v2 provisional-training attestation from immutable v1 inputs.

The command never edits a legacy source or silver manifest. It permits any
ordinary publicly downloadable, hash-bound, model-only v1 input and preserves
its privacy limitation flag; license metadata is recorded when available but
is not an isolated-research gate.
The resulting labels are trainable provisional supervision, not human truth,
calibration data, blind-evaluation truth, or production authorization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from validate_public_video_silver_labels import SilverLabelError, load_json, validate


class PromotionError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(*, legacy_silver_path: Path, legacy_source_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise PromotionError(f"refusing to overwrite output root: {output_root}")
    legacy_silver = load_json(legacy_silver_path)
    legacy_source = load_json(legacy_source_path)
    try:
        validate(legacy_silver, source_manifest_path=legacy_source_path)
    except SilverLabelError as error:
        raise PromotionError(f"legacy silver manifest is not valid v1 evidence: {error}") from error
    if legacy_silver.get("schema") != "blindassist_public_video_silver_labels_v1":
        raise PromotionError("only immutable v1 silver manifests can be promoted")
    source_info = legacy_source.get("source")
    license_name = source_info.get("license") if isinstance(source_info, dict) else None
    license_receipt: dict[str, Any] | None = None
    if not license_name:
        receipt_path = legacy_source_path.parent.parent / "public_candidate_receipt.json"
        if receipt_path.is_file():
            receipt = load_json(receipt_path)
            if receipt.get("source_id") == legacy_source.get("source_id") and receipt.get("expected_license") == "CC0 1.0":
                license_name = "CC0-1.0"
                license_receipt = {"path": str(receipt_path.resolve()), "sha256": sha256_file(receipt_path)}
    license_name = license_name or "unknown_recorded_nonblocking"
    if legacy_source.get("human_event_truth_present") is not False or legacy_source.get("privacy_audit_required") is not True:
        raise PromotionError("legacy source must preserve non-human truth and privacy-audit flags")
    image_root = (legacy_source_path.parent / "images").resolve()
    if not image_root.is_dir():
        raise PromotionError(f"legacy source image directory is missing: {image_root}")
    frames = legacy_source.get("frames")
    if not isinstance(frames, list) or not frames:
        raise PromotionError("legacy source contains no frame records")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("file_name"), str) or not isinstance(frame.get("sha256"), str):
            raise PromotionError(f"legacy source frame {index} lacks file_name or SHA256")
        image_path = (image_root / frame["file_name"]).resolve()
        if not image_path.is_relative_to(image_root) or not image_path.is_file():
            raise PromotionError(f"legacy source frame {index} is missing or escapes image_root")
        if sha256_file(image_path) != frame["sha256"]:
            raise PromotionError(f"legacy source frame {index} image SHA256 does not match")

    source_v2 = copy.deepcopy(legacy_source)
    source_v2.update({
        "format": "blindassist_public_rgb_timeline_source_manifest_v2",
        "provisional_training_authorized": True,
        "training_execution_authorized": True,
        "promotion": {
            "source_manifest_v1_sha256": sha256_file(legacy_source_path),
            "source_manifest_v1_path": str(legacy_source_path.resolve()),
            "image_root": str(image_root),
            "mode": "provisional_model_supervision",
            "research_use_basis": "ordinary_public_download",
            "important_limit": "Not human event truth, calibration data, blind-evaluation truth, or production authorization.",
        },
    })
    source_v2["source"] = {**(source_info if isinstance(source_info, dict) else {}), "license": license_name}
    if license_receipt is not None:
        source_v2["promotion"]["license_receipt"] = license_receipt
    output_root.mkdir(parents=True)
    source_v2_path = output_root / "source_manifest_v2.json"
    source_v2_path.write_text(json.dumps(source_v2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    silver_v2 = copy.deepcopy(legacy_silver)
    silver_v2.update({
        "schema": "blindassist_public_video_silver_labels_v2",
        "training_execution_authorized": True,
        "training_mode": "provisional_model_supervision",
    })
    silver_v2["source"]["source_manifest_sha256"] = sha256_file(source_v2_path)
    silver_v2["source"]["source_manifest_path"] = "source_manifest_v2.json"
    silver_v2_path = output_root / "silver_labels_v2.json"
    silver_v2_path.write_text(json.dumps(silver_v2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        validation = validate(silver_v2, source_manifest_path=source_v2_path)
    except SilverLabelError as error:
        raise PromotionError(f"generated v2 manifest failed validation: {error}") from error
    receipt = {
        "schema": "blindassist_public_silver_provisional_training_promotion_v1",
        "legacy_silver_manifest_sha256": sha256_file(legacy_silver_path),
        "legacy_silver_manifest_path": str(legacy_silver_path.resolve()),
        "legacy_source_manifest_sha256": sha256_file(legacy_source_path),
        "legacy_source_manifest_path": str(legacy_source_path.resolve()),
        "image_root": str(image_root),
        "source_manifest_v2_sha256": sha256_file(source_v2_path),
        "silver_labels_v2_sha256": sha256_file(silver_v2_path),
        "validation": validation,
        "human_event_truth_present": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    (output_root / "promotion_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-silver-manifest", type=Path, required=True)
    parser.add_argument("--legacy-source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = promote(
            legacy_silver_path=args.legacy_silver_manifest,
            legacy_source_path=args.legacy_source_manifest,
            output_root=args.output_root,
        )
    except (PromotionError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "output_root": str(args.output_root.resolve()), "training_execution_authorized": True, "silver_labels_v2_sha256": receipt["silver_labels_v2_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
