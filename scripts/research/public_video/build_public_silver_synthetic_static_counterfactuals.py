#!/usr/bin/env python3
"""Build train-only controlled static counterfactual pairs from public RGB.

Each pair copies a hash-bound candidate-no-alert episode and deterministically
composites one isolated obstacle asset into the same frames with an increasing
perspective scale.  The resulting alpha mask and bounding box are exact
composition geometry.  These samples are synthetic/provisional augmentation
only: they are excluded when their parent real source is held out and never
count as calibration, blind-evaluation, or production evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
from validate_public_video_silver_labels import load_json, sha256_file, validate


SCHEMA = "blindassist_train_only_synthetic_static_counterfactuals_v1"
PROMPT_ID = "blindassist-controlled-static-counterfactual-composition-v1"
PROMPT_TEXT = """Create a train-only controlled counterfactual pair from the
same public RGB frames. Preserve the clear episode exactly. Insert one isolated
opaque static obstacle into the visible forward corridor with deterministic
alpha compositing and increasing perspective scale. Record exact masks and
boxes. Treat both labels as synthetic/provisional supervision, never human
truth, calibration data, blind truth, or production authorization."""

PAIR_SPECS: tuple[dict[str, Any], ...] = (
    {
        "slug": "chcne_barricade",
        "base_episode_id": "sanpo-chcne-static-furniture-passable-0177-0207",
        "asset_name": "barricade",
        "center_x": 0.49,
        "bottom_y": 0.95,
        "width_fractions": (0.16, 0.28, 0.44),
    },
    {
        "slug": "vcz_sand_pile",
        "base_episode_id": "sanpo-vcz-stroller-distant-clear-near-field-0397-0427",
        "asset_name": "sand_pile",
        "center_x": 0.52,
        "bottom_y": 0.95,
        "width_fractions": (0.18, 0.31, 0.49),
    },
    {
        "slug": "bangkok_barricade",
        "base_episode_id": "wikimedia-bangkok-driveway-clear-nearfield-3064-3070",
        "asset_name": "barricade",
        "center_x": 0.52,
        "bottom_y": 0.95,
        "width_fractions": (0.16, 0.29, 0.45),
    },
)


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this builder's scope: {path}")


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEXT.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def alpha_crop(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("obstacle asset must be a four-channel RGBA/BGRA image")
    coordinates = np.argwhere(image[:, :, 3] > 8)
    if coordinates.size == 0:
        raise ValueError("obstacle asset contains no opaque pixels")
    y1, x1 = coordinates.min(axis=0)
    y2, x2 = coordinates.max(axis=0) + 1
    return image[y1:y2, x1:x2]


def compose_obstacle(
    background: np.ndarray,
    obstacle: np.ndarray,
    *,
    width_fraction: float,
    center_x: float,
    bottom_y: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError("background must be a three-channel image")
    if not 0 < width_fraction < 1 or not 0 < center_x < 1 or not 0 < bottom_y <= 1:
        raise ValueError("composition fractions must be inside the image")
    cropped = alpha_crop(obstacle)
    height, width = background.shape[:2]
    target_width = max(2, int(round(width * width_fraction)))
    target_height = max(2, int(round(cropped.shape[0] * target_width / cropped.shape[1])))
    maximum_height = max(2, int(round(height * bottom_y)))
    if target_height > maximum_height:
        scale = maximum_height / target_height
        target_height = maximum_height
        target_width = max(2, int(round(target_width * scale)))
    resized = cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA if target_width < cropped.shape[1] else cv2.INTER_LINEAR,
    )
    x1 = int(round(center_x * width - target_width / 2))
    y1 = int(round(bottom_y * height - target_height))
    x1 = min(max(x1, 0), width - target_width)
    y1 = min(max(y1, 0), height - target_height)
    x2, y2 = x1 + target_width, y1 + target_height
    alpha = resized[:, :, 3].astype(np.float32) / 255.0
    foreground = resized[:, :, :3].astype(np.float32)
    output = background.copy()
    region = output[y1:y2, x1:x2].astype(np.float32)
    output[y1:y2, x1:x2] = np.clip(
        foreground * alpha[:, :, None] + region * (1.0 - alpha[:, :, None]),
        0,
        255,
    ).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = np.where(resized[:, :, 3] > 8, 255, 0).astype(np.uint8)
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        raise ValueError("resized obstacle produced an empty mask")
    mask_y1, mask_x1 = coordinates.min(axis=0)
    mask_y2, mask_x2 = coordinates.max(axis=0) + 1
    return output, mask, [int(mask_x1), int(mask_y1), int(mask_x2), int(mask_y2)]


def source_license_fields(parent_source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = parent_source.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("license"), str):
        raise ValueError("parent source has no reusable license metadata")
    review = parent_source.get("license_review")
    return dict(source), dict(review) if isinstance(review, dict) else None


def find_parent_episode(
    parent_root: Path,
    episode_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    episodes, _excluded = common.load_episode_specs(parent_root)
    matches = [episode for episode in episodes if episode["episode_id"] == episode_id]
    if len(matches) != 1:
        raise ValueError(f"expected one base episode, found {len(matches)}: {episode_id}")
    spec = matches[0]
    if spec["label"] != 0:
        raise ValueError(f"base episode must be candidate_no_alert: {episode_id}")
    silver = load_json(Path(spec["silver_path"]))
    parent_source = load_json(Path(spec["source_path"]))
    episode = next(row for row in silver["episodes"] if row["episode_id"] == episode_id)
    return spec, episode, silver, parent_source


def build_pair(
    *,
    output_root: Path,
    pair_spec: dict[str, Any],
    parent_spec: dict[str, Any],
    parent_episode: dict[str, Any],
    parent_source: dict[str, Any],
    obstacle_path: Path,
    copied_asset_path: Path,
) -> dict[str, Any]:
    slug = str(pair_spec["slug"])
    asset_name = str(pair_spec["asset_name"])
    frames = list(parent_spec["frames"])
    fractions = tuple(float(value) for value in pair_spec["width_fractions"])
    if len(frames) != len(fractions) or len(frames) < 2:
        raise ValueError(f"pair {slug} needs one width fraction per source frame")
    obstacle = cv2.imread(str(obstacle_path), cv2.IMREAD_UNCHANGED)
    if obstacle is None:
        raise ValueError(f"cannot decode obstacle asset: {obstacle_path}")

    image_root = output_root / "images" / "train"
    mask_root = output_root / "masks" / "train"
    package_root = output_root / "packages" / f"synthetic-static-{slug}-20260717"
    package_root.mkdir(parents=True)
    source_info, license_review = source_license_fields(parent_source)
    synthetic_source_id = f"synthetic_static_cf_{slug}"
    pair_id = f"synthetic-static-{slug}-clear-to-obstacle"
    frame_rows: list[dict[str, Any]] = []
    clear_hashes: list[str] = []
    alert_hashes: list[str] = []
    composition_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []

    for index, (parent_frame, width_fraction) in enumerate(zip(frames, fractions)):
        background_path = Path(parent_frame["path"])
        background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if background is None:
            raise ValueError(f"cannot decode parent frame: {background_path}")
        clear_name = f"{slug}_clear_{index:02d}.png"
        alert_name = f"{slug}_alert_{index:02d}.png"
        mask_name = f"{slug}_alert_{index:02d}.png"
        clear_path = image_root / clear_name
        alert_path = image_root / alert_name
        mask_path = mask_root / mask_name
        shutil.copy2(background_path, clear_path)
        composed, mask, bbox = compose_obstacle(
            background,
            obstacle,
            width_fraction=width_fraction,
            center_x=float(pair_spec["center_x"]),
            bottom_y=float(pair_spec["bottom_y"]),
        )
        if not cv2.imwrite(str(alert_path), composed) or not cv2.imwrite(str(mask_path), mask):
            raise ValueError(f"cannot write composed pair frame: {slug}:{index}")
        clear_hash = sha256_file(clear_path)
        alert_hash = sha256_file(alert_path)
        clear_hashes.append(clear_hash)
        alert_hashes.append(alert_hash)
        height, width = background.shape[:2]
        frame_rows.extend([
            {
                "frame_index": index,
                "file_name": clear_name,
                "sha256": clear_hash,
                "synthetic_variant": "clear_exact_copy",
                "parent_frame_sha256": parent_frame["sha256"],
            },
            {
                "frame_index": len(frames) + index,
                "file_name": alert_name,
                "sha256": alert_hash,
                "synthetic_variant": "static_obstacle_composite",
                "parent_frame_sha256": parent_frame["sha256"],
                "mask_path": str(mask_path.relative_to(output_root)).replace("\\", "/"),
                "mask_sha256": sha256_file(mask_path),
                "bbox_xyxy": bbox,
            },
        ])
        composition_rows.append({
            "pair_slug": slug,
            "frame_index": index,
            "parent_frame_sha256": parent_frame["sha256"],
            "asset_sha256": sha256_file(copied_asset_path),
            "width_fraction": width_fraction,
            "center_x": float(pair_spec["center_x"]),
            "bottom_y": float(pair_spec["bottom_y"]),
            "bbox_xyxy": bbox,
            "mask_path": str(mask_path.relative_to(output_root)).replace("\\", "/"),
            "mask_sha256": sha256_file(mask_path),
        })
        common_attributes = {
            "pair_id": pair_id,
            "parent_source_id": parent_spec["source_id"],
            "synthetic": True,
            "provisional": True,
        }
        generation_rows.extend([
            {
                "id": f"{slug}_clear_{index:02d}",
                "image_path": str(clear_path.relative_to(output_root)).replace("\\", "/"),
                "split": "train",
                "width": width,
                "height": height,
                "labels": [],
                "prompt": PROMPT_TEXT,
                "objects": [],
                "attributes": {**common_attributes, "variant": "clear_exact_copy"},
                "status": "accepted",
                "source": str(background_path),
            },
            {
                "id": f"{slug}_alert_{index:02d}",
                "image_path": str(alert_path.relative_to(output_root)).replace("\\", "/"),
                "split": "train",
                "width": width,
                "height": height,
                "labels": ["static_obstacle"],
                "prompt": PROMPT_TEXT,
                "objects": [{
                    "class": "static_obstacle",
                    "bbox_xyxy": bbox,
                    "bbox_source": "deterministic_alpha_composition",
                    "mask_path": str(mask_path.relative_to(output_root)).replace("\\", "/"),
                }],
                "attributes": {
                    **common_attributes,
                    "variant": "static_obstacle_composite",
                    "asset_name": asset_name,
                    "width_fraction": width_fraction,
                },
                "status": "accepted",
                "source": str(background_path),
            },
        ])

    source_manifest = {
        "format": "blindassist_public_rgb_timeline_source_manifest_v2",
        "source_id": synthetic_source_id,
        "source": {
            **source_info,
            "dataset": f"Controlled synthetic counterfactual over {source_info.get('dataset', 'public RGB')}",
        },
        "frame_count": len(frame_rows),
        "frames": frame_rows,
        "privacy_audit_required": True,
        "human_event_truth_present": False,
        "source_masks_or_geometry_used": True,
        "provisional_training_authorized": True,
        "training_execution_authorized": True,
        "production_model_replacement_authorized": False,
        "synthetic_counterfactual": {
            "train_only": True,
            "parent_source_id": parent_spec["source_id"],
            "parent_episode_id": parent_spec["episode_id"],
            "parent_source_manifest_sha256": parent_spec["source_sha256"],
            "parent_silver_manifest_sha256": parent_spec["silver_sha256"],
            "asset_name": asset_name,
            "asset_path": str(copied_asset_path.relative_to(output_root)).replace("\\", "/"),
            "asset_sha256": sha256_file(copied_asset_path),
            "composition_contract": "exact background copy plus deterministic alpha overlay; no generative background edits",
            "holdout_exclusion_contract": "exclude this pair whenever parent_source_id is the real held-out source",
        },
        "promotion": {
            "image_root": str(image_root.resolve()),
            "mode": "train_only_synthetic_provisional_augmentation",
            "important_limit": "Not human event truth, calibration data, blind-evaluation truth, production authorization, or standalone validation evidence.",
        },
    }
    if license_review is not None:
        source_manifest["license_review"] = license_review
    source_path = package_root / "source_manifest_v2.json"
    write_json(source_path, source_manifest)
    risk_mechanism = "static_corridor_narrowing"
    silver = {
        "schema": "blindassist_public_video_silver_labels_v2",
        "source": {
            "source_id": synthetic_source_id,
            "source_manifest_path": "source_manifest_v2.json",
            "source_manifest_sha256": sha256_file(source_path),
            "human_event_truth_present": False,
            "privacy_audit_required": True,
        },
        "labeler": {
            "provider": "openai+deterministic_compositor",
            "model": "gpt-image-2-obstacle-assets+alpha-compositor-v1",
            "prompt_id": PROMPT_ID,
            "prompt_sha256": prompt_sha256(),
            "review_mode": "multiframe_temporal",
        },
        "episodes": [
            {
                "episode_id": f"synthetic-{slug}-clear",
                "evidence_frame_sha256": clear_hashes,
                "silver_should_alert": "candidate_no_alert",
                "confidence": float(parent_episode["confidence"]),
                "counterfactual_pair_id": pair_id,
                "risk_profile": {
                    "risk_mechanism": risk_mechanism,
                    "primary_hazard_type": "controlled_empty_forward_corridor",
                    "corridor_relation": "exact_parent_clear_frames",
                    "lifecycle": "no_alert",
                    "counterfactual_pair_id": pair_id,
                },
                "negative_decision_quality": dict(parent_episode["negative_decision_quality"]),
                "uncertainty_reasons": [
                    "The clear variant exactly copies the parent provisional no-alert evidence.",
                    "The parent label remains machine silver rather than human event truth.",
                ],
            },
            {
                "episode_id": f"synthetic-{slug}-{asset_name}-approach",
                "evidence_frame_sha256": alert_hashes,
                "silver_should_alert": "candidate_alert",
                "confidence": 0.90,
                "counterfactual_pair_id": pair_id,
                "risk_profile": {
                    "risk_mechanism": risk_mechanism,
                    "primary_hazard_type": f"synthetic_{asset_name}",
                    "corridor_relation": "deterministically_inserted_into_forward_corridor",
                    "lifecycle": "approach_alertable",
                    "counterfactual_pair_id": pair_id,
                },
                "uncertainty_reasons": [
                    "Object appearance is AI-generated and does not establish real-world frequency or photometric representativeness.",
                    "Placement geometry is exact, but the alert remains synthetic provisional supervision.",
                ],
            },
        ],
        "training_execution_authorized": True,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
        "training_mode": "provisional_model_supervision",
    }
    silver_path = package_root / "silver_labels_v2.json"
    write_json(silver_path, silver)
    validation = validate(silver, source_manifest_path=source_path)
    write_json(package_root / "composition_records.json", {
        "schema": SCHEMA,
        "pair_id": pair_id,
        "records": composition_rows,
    })
    write_json(package_root / "promotion_receipt.json", {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_id": pair_id,
        "parent_source_id": parent_spec["source_id"],
        "parent_episode_id": parent_spec["episode_id"],
        "source_manifest_v2_sha256": sha256_file(source_path),
        "silver_labels_v2_sha256": sha256_file(silver_path),
        "validation": validation,
        "train_only": True,
        "human_event_truth_present": False,
        "independent_model_directions_used": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    })
    return {
        "slug": slug,
        "pair_id": pair_id,
        "parent_source_id": parent_spec["source_id"],
        "parent_episode_id": parent_spec["episode_id"],
        "asset_name": asset_name,
        "episode_count": 2,
        "frame_count": len(frame_rows),
        "generation_rows": generation_rows,
    }


def build(
    *,
    parent_root: Path,
    barricade_asset: Path,
    sand_pile_asset: Path,
    output_root: Path,
    pair_specs: Sequence[dict[str, Any]] = PAIR_SPECS,
) -> dict[str, Any]:
    parent_root = parent_root.resolve()
    barricade_asset = barricade_asset.resolve()
    sand_pile_asset = sand_pile_asset.resolve()
    output_root = output_root.resolve()
    for path in (parent_root, barricade_asset, sand_pile_asset, output_root):
        reject_independent_direction(path)
    if output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {output_root}")
    if not parent_root.is_dir() or not barricade_asset.is_file() or not sand_pile_asset.is_file():
        raise FileNotFoundError("parent root or synthetic obstacle asset is missing")

    output_root.mkdir(parents=True)
    try:
        (output_root / "images" / "train").mkdir(parents=True)
        (output_root / "masks" / "train").mkdir(parents=True)
        (output_root / "packages").mkdir()
        asset_root = output_root / "assets"
        asset_root.mkdir()
        copied_assets = {
            "barricade": asset_root / "barricade_rgba.png",
            "sand_pile": asset_root / "sand_pile_rgba.png",
        }
        shutil.copy2(barricade_asset, copied_assets["barricade"])
        shutil.copy2(sand_pile_asset, copied_assets["sand_pile"])
        source_assets = {
            "barricade": barricade_asset,
            "sand_pile": sand_pile_asset,
        }
        results: list[dict[str, Any]] = []
        generation_rows: list[dict[str, Any]] = []
        for pair_spec in pair_specs:
            parent_spec, parent_episode, _silver, parent_source = find_parent_episode(
                parent_root,
                str(pair_spec["base_episode_id"]),
            )
            asset_name = str(pair_spec["asset_name"])
            if asset_name not in source_assets:
                raise ValueError(f"unsupported asset_name: {asset_name}")
            result = build_pair(
                output_root=output_root,
                pair_spec=dict(pair_spec),
                parent_spec=parent_spec,
                parent_episode=parent_episode,
                parent_source=parent_source,
                obstacle_path=source_assets[asset_name],
                copied_asset_path=copied_assets[asset_name],
            )
            generation_rows.extend(result.pop("generation_rows"))
            results.append(result)

        dataset_spec = {
            "name": "blindassist_mainline_train_only_static_counterfactual_r8",
            "task": "controlled_counterfactual_episode_classification_with_auxiliary_detection_masks",
            "classes": [{"id": 0, "name": "static_obstacle"}],
            "scenes": ["licensed_public_first_person_navigation_rgb"],
            "attributes": {
                "variant": ["clear_exact_copy", "static_obstacle_composite"],
                "distance": ["far", "mid", "near"],
                "geometry_source": ["deterministic_alpha_composition"],
            },
            "negative_cases": ["exact_parent_clear_corridor"],
            "counts": {
                "pair_count": len(results),
                "image_count": len(generation_rows),
                "positive_image_count": sum(bool(row["objects"]) for row in generation_rows),
                "negative_image_count": sum(not row["objects"] for row in generation_rows),
            },
            "splits": {"train": 1.0, "val": 0.0, "test": 0.0},
            "image_style": "source-native public RGB with isolated photorealistic obstacle alpha composition",
            "output_resolution": "source-native; no resize of parent RGB",
            "annotation_target": "manifest.jsonl + YOLO + COCO + exact PNG alpha-derived masks",
            "intended_use": "train-only mainline representation diagnosis and augmentation",
            "exclusions": [
                "not validation data",
                "not calibration data",
                "not blind-evaluation data",
                "not production promotion evidence",
                "not human event truth",
            ],
        }
        write_json(output_root / "dataset_spec.json", dataset_spec)
        write_jsonl(output_root / "generation_records.jsonl", generation_rows)
        receipt = {
            "schema": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_root": str(parent_root),
            "output_root": str(output_root),
            "pair_count": len(results),
            "episode_count": 2 * len(results),
            "image_count": len(generation_rows),
            "pairs": results,
            "asset_sha256": {
                name: sha256_file(path) for name, path in copied_assets.items()
            },
            "holdout_exclusion_required": True,
            "independent_model_directions_used": False,
            "human_event_truth_present": False,
            "calibration_authorized": False,
            "blind_evaluation_authorized": False,
            "production_model_replacement_authorized": False,
        }
        receipt_path = output_root / "build_receipt.json"
        write_json(receipt_path, receipt)
        Path(str(receipt_path) + ".sha256").write_text(
            sha256_file(receipt_path) + "\n",
            encoding="ascii",
        )
        return receipt
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--barricade-asset", type=Path, required=True)
    parser.add_argument("--sand-pile-asset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = build(
            parent_root=args.parent_root,
            barricade_asset=args.barricade_asset,
            sand_pile_asset=args.sand_pile_asset,
            output_root=args.output_root,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **receipt}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
