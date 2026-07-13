#!/usr/bin/env python3
"""Build a deterministic tactile-paving-occupied sequence from two source GTs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


GENERATOR_ID = "tactile_occupied_compositor_v1"
PROVENANCE_SCHEMA = "blindassist_procedural_ground_truth_v1"
OBSTACLE_CLASS_IDS = {12, 20, 21}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tactile_mask(label_path: Path, size: tuple[int, int]) -> Image.Image:
    width, height = size
    output = Image.new("L", size, 0)
    draw = ImageDraw.Draw(output)
    found = False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7 or fields[0] != "0" or (len(fields) - 1) % 2:
            continue
        coordinates = [float(value) for value in fields[1:]]
        points = [(coordinates[i] * width, coordinates[i + 1] * height) for i in range(0, len(coordinates), 2)]
        draw.polygon(points, fill=255)
        found = True
    if not found or not np.any(np.asarray(output)):
        raise ValueError("GuideTWSI label contains no non-empty class-0 polygon")
    return output


def source_paths(row: dict, root: Path) -> tuple[Path, Path]:
    image = root / str(row["image_path"])
    mask_rel = row.get("source_mask_path") or f"source_masks/test/{row['id']}.png"
    return image, root / str(mask_rel)


def has_obstacle(mask_path: Path) -> bool:
    with Image.open(mask_path) as image:
        classes = np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, 0]
    return bool(np.isin(classes, list(OBSTACLE_CLASS_IDS)).any())


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def copy_raw_evidence(source: Path, output: Path, role: str) -> tuple[str, str]:
    """Copy an immutable generator input into the package and bind its SHA256."""
    digest = sha256_file(source)
    suffix = source.suffix.lower() or ".bin"
    destination = output / "procedural_evidence" / "raw_sources" / f"{role}_{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    elif sha256_file(destination) != digest:
        raise ValueError(f"raw evidence collision: {destination}")
    return relative(destination, output), digest


def crc32_file(path: Path) -> str:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return f"{checksum & 0xffffffff:08x}"


def guide_remote_receipt(path: Path, inventory_path: Path, explicit_member: str | None) -> dict:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    size, crc32 = path.stat().st_size, crc32_file(path)
    members = inventory.get("members") if isinstance(inventory.get("members"), list) else []
    matches = [
        item for item in members if isinstance(item, dict)
        and int(item.get("size", -1)) == size and str(item.get("crc32", "")).lower() == crc32
        and (explicit_member is None or str(item.get("path", "")) == explicit_member)
    ]
    if len(matches) != 1:
        raise ValueError(f"Guide raw asset must match exactly one remote inventory member: {path}")
    source = inventory.get("source") if isinstance(inventory.get("source"), dict) else {}
    archive = {key: source.get(key) for key in ("etag", "generation", "md5_base64")}
    if any(not str(value or "").strip() for value in archive.values()):
        raise ValueError("Guide remote inventory lacks archive etag/generation/md5_base64")
    return {
        "origin_member_path": str(matches[0]["path"]), "size": size, "crc32": crc32,
        "archive": archive,
    }


def build(args: argparse.Namespace) -> list[dict]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    guide_image = args.guide_image.resolve()
    guide_label = args.guide_label.resolve()
    sanpo_root = args.sanpo_root.resolve()
    with Image.open(guide_image) as opened:
        background = opened.convert("RGB")
    width, height = background.size
    tactile = tactile_mask(guide_label, background.size)
    tactile_bbox = tactile.getbbox()
    assert tactile_bbox

    rows = load_jsonl(args.sanpo_manifest)
    candidates = [row for row in rows if has_obstacle(source_paths(row, sanpo_root)[1])]
    if len(candidates) < args.frame_count:
        raise ValueError(f"only {len(candidates)} SANPO frames contain classes {sorted(OBSTACLE_CLASS_IDS)}")
    candidates = candidates[:args.frame_count]

    evidence = output / "procedural_evidence"
    tactile_copy = evidence / "tactile_ground_truth.png"
    code_copy = evidence / "generator.py"
    config_path = evidence / "config.json"
    evidence.mkdir(parents=True, exist_ok=True)
    tactile.save(tactile_copy)
    shutil.copy2(Path(__file__).resolve(), code_copy)
    guide_image_evidence = copy_raw_evidence(guide_image, output, "guide_rgb")
    guide_label_evidence = copy_raw_evidence(guide_label, output, "guide_polygon")
    guide_inventory = Path(args.guide_inventory).resolve()
    guide_image_receipt = guide_remote_receipt(
        guide_image, guide_inventory, getattr(args, "guide_image_member", None),
    )
    guide_label_receipt = guide_remote_receipt(
        guide_label, guide_inventory, getattr(args, "guide_label_member", None),
    )
    rng = random.Random(args.seed)
    tactile_left, tactile_top, tactile_right, tactile_bottom = tactile_bbox
    tactile_points = np.argwhere(np.asarray(tactile) > 0)
    tactile_points = tactile_points[np.argsort(tactile_points[:, 1])]
    start_x = tactile_left + rng.uniform(0.15, 0.28) * (tactile_right - tactile_left)
    end_x = tactile_left + rng.uniform(0.72, 0.85) * (tactile_right - tactile_left)
    y_center = tactile_top + rng.uniform(0.40, 0.62) * (tactile_bottom - tactile_top)
    scale = rng.uniform(0.28, 0.38) * min(width / 2208.0, height / 1242.0)
    config = {
        "schema": "blindassist_procedural_tactile_config_v1", "generator_id": GENERATOR_ID,
        "seed": args.seed, "split": args.split, "session_id": args.session_id,
        "scene_bucket": args.scene_bucket,
        "frame_count": args.frame_count, "obstacle_class_ids": sorted(OBSTACLE_CLASS_IDS),
        "motion": {"start_x": start_x, "end_x": end_x, "y_center": y_center, "scale": scale},
        "guide_image_sha256": sha256_file(guide_image), "guide_label_sha256": sha256_file(guide_label),
        "sanpo_manifest_sha256": sha256_file(args.sanpo_manifest),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    code_sha, config_sha, tactile_sha = map(sha256_file, (code_copy, config_path, tactile_copy))

    manifest_rows: list[dict] = []
    sequence_id = f"procedural_{args.session_id}"
    for index, source_row in enumerate(candidates):
        source_image_path, source_mask_path = source_paths(source_row, sanpo_root)
        sanpo_image_evidence = copy_raw_evidence(source_image_path, output, "sanpo_rgb")
        sanpo_mask_evidence = copy_raw_evidence(source_mask_path, output, "sanpo_raw_mask")
        with Image.open(source_image_path) as opened:
            source_rgb = opened.convert("RGB")
        with Image.open(source_mask_path) as opened:
            source_mask_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        obstacle = Image.fromarray((np.isin(source_mask_rgb[:, :, 0], list(OBSTACLE_CLASS_IDS)) * 255).astype(np.uint8), "L")
        bbox = obstacle.getbbox()
        if bbox is None:
            raise AssertionError("filtered obstacle frame became empty")
        obstacle = obstacle.crop(bbox)
        source_rgb = source_rgb.crop(bbox)
        target_w = max(12, int(obstacle.width * scale))
        target_h = max(12, int(obstacle.height * scale))
        obstacle = obstacle.resize((target_w, target_h), Image.Resampling.NEAREST)
        source_rgb = source_rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
        progress = index / max(1, args.frame_count - 1)
        ease = progress * progress * (3.0 - 2.0 * progress)
        target_index = int(round((0.15 + 0.70 * ease) * (len(tactile_points) - 1)))
        target_y, target_x = (int(value) for value in tactile_points[target_index])
        obstacle_points = np.argwhere(np.asarray(obstacle) > 0)
        obstacle_center_y, obstacle_center_x = obstacle_points.mean(axis=0)
        x = int(round(target_x - obstacle_center_x))
        y = int(round(target_y - obstacle_center_y))
        matrix = [[scale, 0.0, float(x - bbox[0] * scale)], [0.0, scale, float(y - bbox[1] * scale)], [0.0, 0.0, 1.0]]

        light_factor = 1.0
        if args.scene_bucket == "low_light":
            light_factor = 0.18 + 0.04 * (0.5 + 0.5 * math.sin(progress * math.pi * 2.0))
        composed = ImageEnhance.Brightness(background).enhance(light_factor)
        composed.paste(source_rgb, (x, y), obstacle)
        semantic_array = np.full((height, width), 3, dtype=np.uint8)
        semantic_array[np.asarray(tactile) > 0] = 0
        obstacle_canvas = Image.new("L", (width, height), 0)
        obstacle_canvas.paste(obstacle, (x, y))
        overlap_pixels = int(np.count_nonzero(
            (np.asarray(obstacle_canvas) > 0) & (np.asarray(tactile) > 0)
        ))
        if overlap_pixels == 0:
            raise ValueError(f"frame {index}: transformed obstacle does not occupy tactile paving")
        semantic_array[np.asarray(obstacle_canvas) > 0] = 2
        semantic = Image.fromarray(semantic_array, "L")

        sample_id = f"{sequence_id}_{index:06d}"
        image_path = output / "images" / args.split / f"{sample_id}.png"
        mask_path = output / "semantic_masks" / args.split / f"{sample_id}.png"
        obstacle_copy = evidence / "obstacle_ground_truth" / f"{sample_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        obstacle_copy.parent.mkdir(parents=True, exist_ok=True)
        composed.save(image_path)
        semantic.save(mask_path)
        shutil.copy2(source_mask_path, obstacle_copy)
        image_sha, mask_sha, obstacle_sha = map(sha256_file, (image_path, mask_path, obstacle_copy))
        provenance = {
            "schema": PROVENANCE_SCHEMA, "generator_id": GENERATOR_ID,
            "generator_code_path": relative(code_copy, output), "generator_code_sha256": code_sha,
            "generator_config_path": relative(config_path, output), "generator_config_sha256": config_sha,
            "seed": args.seed, "transform_matrix": matrix, "transform_sha256": canonical_sha(matrix),
            "tactile_obstacle_overlap_pixels": overlap_pixels,
            "photometric_brightness_factor": light_factor,
            "source_masks": [
                {"role": "tactile_ground_truth", "source_id": "guidetwsi_sdome_15k", "path": relative(tactile_copy, output), "sha256": tactile_sha},
                {"role": "obstacle_ground_truth", "source_id": "sanpo_real_v0", "path": relative(obstacle_copy, output), "sha256": obstacle_sha},
            ],
            "source_assets": [
                {"role": "guide_rgb", "source_id": "guidetwsi_sdome_15k", "path": guide_image_evidence[0], "sha256": guide_image_evidence[1], "remote_receipt": guide_image_receipt},
                {"role": "guide_polygon", "source_id": "guidetwsi_sdome_15k", "path": guide_label_evidence[0], "sha256": guide_label_evidence[1], "remote_receipt": guide_label_receipt},
                {"role": "sanpo_rgb", "source_id": "sanpo_real_v0", "path": sanpo_image_evidence[0], "sha256": sanpo_image_evidence[1]},
                {"role": "sanpo_raw_mask", "source_id": "sanpo_real_v0", "path": sanpo_mask_evidence[0], "sha256": sanpo_mask_evidence[1]},
            ],
            "output_mask_sha256": mask_sha,
        }
        manifest_rows.append({
            "id": sample_id, "split": args.split, "session_id": args.session_id,
            "sequence_id": sequence_id, "frame_index": index,
            "scene_bucket": args.scene_bucket, "benchmark_kind": "semantic_segmentation_only",
            "image_path": relative(image_path, output), "semantic_mask_path": relative(mask_path, output),
            "image_sha256": image_sha, "semantic_mask_sha256": mask_sha,
            "label_authority": "procedural_ground_truth", "label_provenance": provenance,
            "source": {
                "source_id": "guidetwsi_sanpo_procedural_v1", "dataset": "GuideTWSI SDome-15K + SANPO-Real v0",
                "license": "CC BY 4.0 (SANPO) and CC0 (GuideTWSI)",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "privacy_review_status": "automated_privacy_clear",
                "session_id": args.session_id,
            },
        })
    manifest = output / "reviewed-source-manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8")
    return manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-image", type=Path, required=True)
    parser.add_argument("--guide-label", type=Path, required=True)
    parser.add_argument("--guide-inventory", type=Path, required=True)
    parser.add_argument("--guide-image-member")
    parser.add_argument("--guide-label-member")
    parser.add_argument("--sanpo-manifest", type=Path, required=True)
    parser.add_argument("--sanpo-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "blind"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--scene-bucket", choices=("tactile_paving_occupied", "low_light"), default="tactile_paving_occupied")
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frame_count <= 0:
        parser.error("--frame-count must be positive")
    rows = build(args)
    print(json.dumps({"output": str(args.output.resolve()), "frames": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
