#!/usr/bin/env python3
"""Fail-closed audit for an acquired SANPO-Synthetic replay package.

It verifies source hashes and the exact shared SANPO-to-four-class taxonomy
mapping before a package may be considered for benchmark-only pretraining.
This is a data-intake check; it grants no model, event, calibration, device or
production authorization.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from build_public_v3_canonical_dataset import SANPO_MAP, sha256_file


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing replay manifest: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("replay manifest is empty")
    return rows


def safe_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("missing relative asset path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"asset path escapes replay root: {value}") from error
    if not path.is_file():
        raise ValueError(f"missing replay asset: {value}")
    return path


def audit(root: Path) -> dict[str, Any]:
    rows = load_rows(root / "manifest.replay.jsonl")
    errors: list[str] = []
    raw_ids: Counter[int] = Counter()
    mapped_ids: Counter[int] = Counter()
    source_indices: list[int] = []
    hash_fields = (
        ("image_path", "image_sha256"),
        ("source_mask_path", "source_mask_sha256"),
        ("source_depth_path", "source_depth_sha256"),
    )
    for index, row in enumerate(rows):
        sample = str(row.get("id", index))
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        if source.get("source_id") != "sanpo_synthetic_v0" or source.get("official_split") != "train":
            errors.append(f"{sample}: requires SANPO-Synthetic official train source")
        authorization = row.get("authorization") if isinstance(row.get("authorization"), dict) else {}
        for forbidden in ("real_finetune_or_eval", "human_event_truth", "calibration", "blind_evaluation", "android_runtime", "production_model_replacement"):
            if authorization.get(forbidden) is not False:
                errors.append(f"{sample}: {forbidden} must remain false")
        assets: dict[str, Path] = {}
        for path_field, hash_field in hash_fields:
            try:
                path = safe_file(root, row.get(path_field))
            except ValueError as error:
                errors.append(f"{sample}: {error}")
                continue
            assets[path_field] = path
            if sha256_file(path) != row.get(hash_field):
                errors.append(f"{sample}: {hash_field} mismatch")
        if "image_path" not in assets or "source_mask_path" not in assets:
            continue
        with Image.open(assets["image_path"]) as image, Image.open(assets["source_mask_path"]) as mask:
            if image.size != mask.size:
                errors.append(f"{sample}: RGB/mask dimensions differ")
            values = np.asarray(mask.convert("RGB"), dtype=np.uint8)[..., 0]
        unknown = sorted(set(int(value) for value in np.unique(values)) - set(SANPO_MAP))
        if unknown:
            errors.append(f"{sample}: unmapped SANPO class IDs {unknown}")
        else:
            raw_ids.update(int(value) for value in np.unique(values))
            mapped_ids.update(SANPO_MAP[int(value)] for value in np.unique(values))
        try:
            source_indices.append(int(row["source_frame_index"]))
        except (KeyError, TypeError, ValueError):
            errors.append(f"{sample}: missing numeric source_frame_index")
    contiguous = source_indices == sorted(source_indices) and len(set(source_indices)) == len(source_indices)
    if not contiguous:
        errors.append("source frame indices must be unique and increasing")
    four_class_coverage = set(mapped_ids) == {0, 1, 2, 3}
    if not four_class_coverage:
        errors.append(f"source window does not cover all four mapped classes: {sorted(mapped_ids)}")
    return {
        "schema": "blindassist_sanpo_synthetic_pretraining_intake_audit_v1",
        "ok": not errors,
        "frame_count": len(rows),
        "source_frame_range": [min(source_indices), max(source_indices)] if source_indices else None,
        "source_indices_unique_and_increasing": contiguous,
        "raw_class_ids": sorted(raw_ids),
        "mapped_four_class_ids": sorted(mapped_ids),
        "four_class_coverage": four_class_coverage,
        "all_assets_sha256_bound": not any("mismatch" in error or "missing replay asset" in error for error in errors),
        "authorization": "benchmark_only_pretraining_candidate",
        "production_authorized": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.replay_root.resolve()
        report_path = args.report.resolve()
        if report_path.exists():
            raise ValueError(f"refusing to overwrite report: {report_path}")
        report = audit(root)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": report["ok"], "frame_count": report["frame_count"], "errors": report["errors"]}, ensure_ascii=False))
        return 0 if report["ok"] else 1
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
