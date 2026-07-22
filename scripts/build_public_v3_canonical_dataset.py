#!/usr/bin/env python3
"""Assemble a SHA-bound public-source v3 dataset through allow-listed adapters.

The builder never accepts an already-canonical manifest.  It consumes a recipe
whose sequences point at downloaded source packages, remaps native pixel masks,
creates provenance/attestation, and publishes only after the total gate is green.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import prepare_sanpo_v3_dataset_views as views
import sanpo_training_gate as gate
import validate_sanpo_v3_dataset as validator


SANPO_MAP = {
    0: 3, 1: 0, 2: 1, 3: 0, 4: 2, 5: 0, 6: 0, 7: 3,
    8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 1,
    16: 2, 17: 0, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2,
    24: 2, 25: 2, 26: 2, 27: 3, 28: 2, 29: 3, 30: 3,
}
ADAPTER_MAPS = {"sanpo_v0": SANPO_MAP}
ASSET_INVENTORY_SCHEMA = "blindassist_source_asset_inventory_v1"
ASSEMBLY_RECIPE = "assembly_recipe.json"
ASSET_INVENTORY = "source_asset_inventory.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_source_sequence(
    source_rows: list[dict[str, Any]], native_session: str, expected_count: int,
    expected_official_split: str | None,
) -> list[dict[str, Any]]:
    """Select one contiguous, official-split-bound source session.

    Frame count belongs to the recipe instead of being inferred from the target
    split.  This keeps the canonical builder reusable as train/dev session
    coverage grows while retaining the fixed 60-frame blind contract.
    """
    if expected_count <= 0:
        raise ValueError("expected_frame_count must be positive")
    selected = [
        item for item in source_rows
        if str(item.get("source", {}).get("session_id") or item.get("session_id")) == native_session
    ]
    selected.sort(key=lambda item: int(item.get("frame_index", -1)))
    indexes = [int(item.get("frame_index", -1)) for item in selected]
    if len(selected) != expected_count or indexes != list(range(expected_count)):
        raise ValueError(
            f"{native_session}: requires contiguous {expected_count}-frame source sequence"
        )
    if expected_official_split:
        actual = {
            str(item.get("source", {}).get("official_split", "")).strip()
            for item in selected
        }
        if actual != {expected_official_split}:
            raise ValueError(
                f"{native_session}: official split {sorted(actual)!r} differs from recipe "
                f"{expected_official_split!r}"
            )
    return selected


def safe_source_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"source path escapes package root: {value}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def remap_mask(source: Path, output: Path, adapter_id: str) -> tuple[str, str]:
    mapping = ADAPTER_MAPS.get(adapter_id)
    if mapping is None:
        raise ValueError(f"adapter {adapter_id!r} has no full-mask mapper")
    with Image.open(source) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    native = array[..., 0]
    unknown = sorted(set(int(value) for value in np.unique(native)) - set(mapping))
    if unknown:
        raise ValueError(f"native mask has unmapped class IDs: {unknown}")
    target = np.full(native.shape, 255, dtype=np.uint8)
    for native_id, target_id in mapping.items():
        target[native == native_id] = target_id
    if np.any(target == 255):
        raise ValueError("mask mapping left unlabeled pixels")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(target, mode="L").save(output)
    return sha256_file(source), sha256_file(output)


def evidence_copy(source: Path, staging: Path, source_id: str, kind: str) -> tuple[str, str]:
    destination = staging / "source_evidence" / source_id / f"{kind}{source.suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.relative_to(staging).as_posix(), sha256_file(destination)


def copy_sha_bound_file(
    package_root: Path, staging: Path, source_value: str, expected_sha: str, destination: Path,
) -> tuple[str, str]:
    """Copy one package-local file after verifying its declared SHA256."""
    source = safe_source_path(package_root, source_value)
    actual = sha256_file(source)
    if len(expected_sha) != 64 or actual != expected_sha:
        raise ValueError(f"source SHA256 mismatch: {source_value}")
    output = staging / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return destination.as_posix(), actual


def copy_content_addressed(
    package_root: Path, staging: Path, source_value: str, expected_sha: str, source_id: str, role: str,
) -> tuple[str, str]:
    source = safe_source_path(package_root, source_value)
    actual = sha256_file(source)
    if len(expected_sha) != 64 or actual != expected_sha:
        raise ValueError(f"source SHA256 mismatch: {source_value}")
    suffix = source.suffix.lower() or ".bin"
    relative = Path("raw_evidence") / source_id / f"{role}_{actual}{suffix}"
    output = staging / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        shutil.copy2(source, output)
    elif sha256_file(output) != actual:
        raise ValueError(f"raw evidence collision: {relative.as_posix()}")
    return relative.as_posix(), actual


def copy_procedural_assets(
    package_root: Path, staging: Path, source_row: dict[str, Any], sample_id: str, split: str,
) -> tuple[str, str, dict[str, Any], str, str, list[dict[str, str]]]:
    """Import and path-rebind a fully reviewed procedural sample."""
    if source_row.get("label_authority") != "procedural_ground_truth":
        raise ValueError(f"{sample_id}: procedural adapter requires procedural_ground_truth")
    provenance = deepcopy(source_row.get("label_provenance"))
    if not isinstance(provenance, dict):
        raise ValueError(f"{sample_id}: procedural adapter requires label_provenance")

    image_source = safe_source_path(package_root, str(source_row["image_path"]))
    image_rel = Path("images") / split / f"{sample_id}{image_source.suffix.lower()}"
    image_path, image_sha = copy_sha_bound_file(
        package_root, staging, str(source_row["image_path"]), str(source_row.get("image_sha256", "")), image_rel,
    )
    mask_rel = Path("semantic_masks") / split / f"{sample_id}.png"
    mask_path, mask_sha = copy_sha_bound_file(
        package_root, staging, str(source_row["semantic_mask_path"]),
        str(source_row.get("semantic_mask_sha256", "")), mask_rel,
    )

    for key, sha_key, filename in (
        ("generator_code_path", "generator_code_sha256", "generator.py"),
        ("generator_config_path", "generator_config_sha256", "config.json"),
    ):
        rebound, _ = copy_sha_bound_file(
            package_root, staging, str(provenance.get(key, "")), str(provenance.get(sha_key, "")),
            Path("procedural_evidence") / sample_id / filename,
        )
        provenance[key] = rebound

    inputs = provenance.get("source_masks")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError(f"{sample_id}: procedural adapter requires exactly two source masks")
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ValueError(f"{sample_id}: procedural source mask {index} is not an object")
        suffix = safe_source_path(package_root, str(item.get("path", ""))).suffix or ".png"
        rebound, _ = copy_sha_bound_file(
            package_root, staging, str(item.get("path", "")), str(item.get("sha256", "")),
            Path("procedural_evidence") / sample_id / "source_masks" / f"{index}_{item.get('role', 'source')}{suffix}",
        )
        item["path"] = rebound
    raw_assets = provenance.get("source_assets")
    required_roles = {"guide_rgb", "guide_polygon", "sanpo_rgb", "sanpo_raw_mask"}
    if not isinstance(raw_assets, list) or {
        str(item.get("role", "")) for item in raw_assets if isinstance(item, dict)
    } != required_roles:
        raise ValueError(f"{sample_id}: procedural adapter requires all four raw source assets")
    rebound_assets: list[dict[str, str]] = []
    for item in raw_assets:
        if not isinstance(item, dict):
            raise ValueError(f"{sample_id}: procedural raw source asset is not an object")
        role = str(item.get("role", ""))
        source_id = str(item.get("source_id", ""))
        rebound, digest = copy_content_addressed(
            package_root, staging, str(item.get("path", "")), str(item.get("sha256", "")), source_id, role,
        )
        item["path"] = rebound
        rebound_item: dict[str, Any] = {
            "role": role, "source_id": source_id, "path": rebound, "sha256": digest,
        }
        if isinstance(item.get("remote_receipt"), dict):
            rebound_item["remote_receipt"] = deepcopy(item["remote_receipt"])
        rebound_assets.append(rebound_item)
    if provenance.get("output_mask_sha256") != mask_sha:
        raise ValueError(f"{sample_id}: procedural output mask SHA256 mismatch")
    return image_path, mask_path, provenance, image_sha, mask_sha, rebound_assets


def assemble(recipe_path: Path, output_root: Path, report_path: Path) -> dict[str, Any]:
    recipe = load_json(recipe_path.resolve())
    output = output_root.resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise ValueError("refusing to overwrite canonical output or stale staging root")
    staging.mkdir(parents=True)
    report: dict[str, Any] = {"schema": "blindassist_public_v3_assembly_v1", "ok": False, "errors": []}
    try:
        receipts = recipe.get("sources")
        sequences = recipe.get("sequences")
        if not isinstance(receipts, list) or not isinstance(sequences, list):
            raise ValueError("recipe requires sources and sequences lists")
        receipt_by_id = {str(item["source_id"]): item for item in receipts}
        recipe_destination = staging / ASSEMBLY_RECIPE
        shutil.copy2(recipe_path.resolve(), recipe_destination)
        attested_sources: list[dict[str, Any]] = []
        for source_id, receipt in receipt_by_id.items():
            adapter_id = str(receipt.get("adapter_id", ""))
            if adapter_id not in gate.ALLOWED_SOURCE_ADAPTERS:
                raise ValueError(f"{source_id}: adapter is not allow-listed")
            package_root = Path(receipt["package_root"]).resolve()
            bound: dict[str, str] = {}
            for kind in ("inventory",):
                bound[f"{kind}_path"], bound[f"{kind}_sha256"] = evidence_copy(
                    safe_source_path(package_root, str(receipt[f"{kind}_path"])), staging, source_id, kind
                )
            for kind in ("license_evidence", "privacy_evidence"):
                if receipt.get(f"{kind}_path"):
                    bound[f"{kind}_path"], bound[f"{kind}_sha256"] = evidence_copy(
                        safe_source_path(package_root, str(receipt[f"{kind}_path"])), staging, source_id, kind
                    )
            attested_sources.append({
                "source_id": receipt["source_id"],
                "adapter_id": receipt["adapter_id"],
                "dataset": receipt["dataset"],
                "dataset_version": receipt["dataset_version"],
                "license": receipt.get("license", "unknown_recorded_nonblocking"),
                "license_url": receipt.get("license_url", "unknown_recorded_nonblocking"),
                "privacy_review_status": receipt.get("privacy_review_status", "unknown_recorded"),
                "research_use_basis": "ordinary_public_download",
            } | bound)
        rows: list[dict[str, Any]] = []
        inventory_assets: list[dict[str, Any]] = []
        seen_sessions: set[str] = set()
        for sequence in sequences:
            source_id = str(sequence["source_id"])
            receipt = receipt_by_id[source_id]
            adapter_id = str(receipt["adapter_id"])
            package_root = Path(sequence.get("package_root", receipt["package_root"])).resolve()
            source_rows = load_jsonl(safe_source_path(package_root, str(sequence["manifest_path"])))
            native_session = str(sequence["native_session_id"])
            expected_count = int(sequence.get(
                "expected_frame_count", 60 if sequence["split"] == "blind" else 50,
            ))
            expected_official_split = str(sequence.get("official_split", "")).strip() or None
            selected = select_source_sequence(
                source_rows, native_session, expected_count, expected_official_split,
            )
            global_session = f"{source_id}:{native_session}"
            if global_session in seen_sessions:
                raise ValueError(f"duplicate native session in recipe: {global_session}")
            seen_sessions.add(global_session)
            sequence_id = f"{global_session}:{sequence['scene_bucket']}"
            mapping_sha = sha256_json(ADAPTER_MAPS[adapter_id]) if adapter_id in ADAPTER_MAPS else ""
            for index, source_row in enumerate(selected):
                sample_id = f"{source_id}_{native_session}_{index:06d}".replace("/", "_")
                if adapter_id == "procedural_tactile_v1":
                    image_value, mask_value, label_provenance, image_sha, mapped_mask_sha, raw_assets = copy_procedural_assets(
                        package_root, staging, source_row, sample_id, str(sequence["split"]),
                    )
                else:
                    image_source = safe_source_path(package_root, str(source_row["image_path"]))
                    source_mask_value = source_row.get("source_mask_path") or f"source_masks/test/{source_row['id']}.png"
                    mask_source = safe_source_path(package_root, str(source_mask_value))
                    image_rel = Path("images") / str(sequence["split"]) / f"{sample_id}{image_source.suffix.lower()}"
                    mask_rel = Path("semantic_masks") / str(sequence["split"]) / f"{sample_id}.png"
                    image_dest, mask_dest = staging / image_rel, staging / mask_rel
                    image_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_source, image_dest)
                    source_mask_sha, mapped_mask_sha = remap_mask(mask_source, mask_dest, adapter_id)
                    image_value, mask_value, image_sha = image_rel.as_posix(), mask_rel.as_posix(), sha256_file(image_dest)
                    raw_mask_value, _ = copy_content_addressed(
                        package_root, staging, str(source_mask_value), source_mask_sha, source_id, "sanpo_raw_mask",
                    )
                    raw_assets = [
                        {"role": "sanpo_rgb", "source_id": source_id, "path": image_value, "sha256": image_sha},
                        {"role": "sanpo_raw_mask", "source_id": source_id, "path": raw_mask_value, "sha256": source_mask_sha},
                    ]
                    label_provenance = {"annotation_kind": "pixel_panoptic_taxonomy_map",
                        "source_mask_sha256": source_mask_sha, "mapped_mask_sha256": mapped_mask_sha,
                        "mapping_sha256": mapping_sha,
                        "source_assets": deepcopy(raw_assets)}
                source_asset_ids: list[str] = []
                for asset_index, asset in enumerate(raw_assets):
                    entry_id = f"{sample_id}:{asset['role']}:{asset_index}"
                    source_asset_ids.append(entry_id)
                    inventory_item: dict[str, Any] = {
                        "entry_id": entry_id, "sample_id": sample_id, "source_id": asset["source_id"],
                        "session_id": global_session, "frame_index": index, "role": asset["role"],
                        "path": asset["path"], "sha256": asset["sha256"],
                    }
                    if isinstance(asset.get("remote_receipt"), dict):
                        inventory_item["remote_receipt"] = deepcopy(asset["remote_receipt"])
                    inventory_assets.append(inventory_item)
                source = {
                    "source_id": source_id,
                    "session_id": global_session,
                    "dataset": receipt["dataset"], "license": receipt["license"],
                    "license_url": receipt["license_url"], "privacy_review_status": receipt["privacy_review_status"],
                }
                if expected_official_split:
                    source["official_split"] = expected_official_split
                rows.append({
                    "id": sample_id, "split": sequence["split"], "session_id": global_session,
                    "sequence_id": sequence_id, "frame_index": index, "scene_bucket": sequence["scene_bucket"],
                    "benchmark_kind": "semantic_segmentation_only", "image_path": image_value,
                    "semantic_mask_path": mask_value, "image_sha256": image_sha,
                    "semantic_mask_sha256": mapped_mask_sha,
                    "label_authority": "procedural_ground_truth" if adapter_id == "procedural_tactile_v1" else "source_ground_truth",
                    "label_provenance": label_provenance,
                    "source_asset_ids": source_asset_ids,
                    "source": source,
                })
        manifest = staging / "reviewed-source-manifest.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        inventory_path = staging / ASSET_INVENTORY
        inventory_path.write_text(json.dumps({
            "schema": ASSET_INVENTORY_SCHEMA, "assets": inventory_assets,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "source_attestation.json").write_text(json.dumps({
            "schema": gate.ATTESTATION_SCHEMA,
            "recipe_path": ASSEMBLY_RECIPE, "recipe_sha256": sha256_file(recipe_destination),
            "asset_inventory_path": ASSET_INVENTORY, "asset_inventory_sha256": sha256_file(inventory_path),
            "sources": attested_sources,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        views.prepare_views(manifest, staging)
        prepublish_report = gate.run_gate(staging, staging / "qa" / "prepublish_gate_report.json")
        if prepublish_report["overall_status"] != "green":
            raise ValueError("total gate is red: " + "; ".join(prepublish_report["errors"][:8]))
        os.replace(staging, output)
        gate_report = gate.run_gate(output, output / "qa" / "training_gate_report.json")
        if gate_report["overall_status"] != "green":
            raise ValueError("final-root gate is red: " + "; ".join(gate_report["errors"][:8]))
        gate.consume_training_authorization(output, output / "qa" / "training_gate_report.json")
        report.update({"ok": True, "dataset_root": str(output), "row_count": len(rows), "gate_report_sha256": gate_report["report_sha256"]})
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["staging_root"] = str(staging)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256_file(report_path)
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(f"{report['report_sha256']}  {report_path.name}\n", encoding="ascii")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.recipe, args.dataset_root, args.report)
    print(json.dumps({"ok": report["ok"], "report_sha256": report["report_sha256"], "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
