#!/usr/bin/env python3
"""Build the hash-bound, source-isolated R1.2d three-class training dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from r12d_contract import CLASSES, require, sha256_file, validate_matrix, write_json


ROADWORK_CLASS_TO_ID = {"Cone": 0, "Tubular Marker": 1}
VAL_CITY_PREFIXES = [
    "washington_dc", "new_york_city", "los_angeles", "san_francisco", "san_antonio",
    "jacksonville", "indianapolis", "philadelphia", "minneapolis",
    "charlotte", "chicago", "columbus", "boston", "denver", "detroit",
    "houston", "phoenix",
]


def source_id(file_name: str, split: str) -> str:
    stem = Path(file_name).stem.lower()
    if split == "train":
        for prefix in ("pittsburgh", "pgh04", "pgh03", "pgh02", "pgh01", "img"):
            if stem.startswith(prefix):
                return f"roadwork_pittsburgh_{prefix}"
        raise ValueError(f"unrecognized Pittsburgh training source: {file_name}")
    for prefix in VAL_CITY_PREFIXES:
        if stem.startswith(prefix):
            return f"roadwork_{prefix}"
    raise ValueError(f"unrecognized held-out city source: {file_name}")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def label_text(rows: list[tuple[int, float, float, float, float]]) -> str:
    return "".join(f"{c} {x:.9f} {y:.9f} {w:.9f} {h:.9f}\n" for c, x, y, w, h in rows)


def roadwork_rows(document: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    categories = {row["id"]: row["name"] for row in document["categories"]}
    target_ids = {cid for cid, name in categories.items() if name in ROADWORK_CLASS_TO_ID}
    images = {row["id"]: row for row in document["images"]}
    annotations: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in document["annotations"]:
        if row["category_id"] in target_ids:
            copied = dict(row)
            copied["target_class_id"] = ROADWORK_CLASS_TO_ID[categories[row["category_id"]]]
            annotations[row["image_id"]].append(copied)
    return images, annotations


def select_roadwork_image_ids(document: dict[str, Any], split: str, selection: dict[str, Any]) -> list[int]:
    images, annotations = roadwork_rows(document)
    rank = lambda image_id: hashlib.sha256(images[image_id]["file_name"].encode("utf-8")).hexdigest()
    target = [image_id for image_id in images if annotations.get(image_id)]
    negative = [image_id for image_id in images if not annotations.get(image_id)]
    if split == "train":
        chosen = sorted(target, key=rank)[: int(selection["train_target_images"])]
        chosen += sorted(negative, key=rank)[: int(selection["train_target_free_images"])]
        return sorted(chosen)
    chosen = []
    for source in sorted({source_id(row["file_name"], split) for row in images.values()}):
        source_target = [image_id for image_id in target if source_id(images[image_id]["file_name"], split) == source]
        source_negative = [image_id for image_id in negative if source_id(images[image_id]["file_name"], split) == source]
        chosen += sorted(source_target, key=rank)[: int(selection["validation_target_images_per_source"])]
        chosen += sorted(source_negative, key=rank)[: int(selection["validation_target_free_images_per_source"])]
    return sorted(chosen)


def emit_roadwork(
    *, split: str, annotation_path: Path, images_root: Path, output: Path,
    manifest: list[dict[str, Any]], image_lists: dict[str, list[str]], selection: dict[str, Any],
    normalization: dict[str, collections.Counter[str]],
) -> None:
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    images, annotations = roadwork_rows(document)
    for image_id in select_roadwork_image_ids(document, split, selection):
        image = images[image_id]
        source = images_root / image["file_name"]
        require(source.is_file(), f"ROADWork image missing: {source}")
        width, height = int(image["width"]), int(image["height"])
        rows: list[tuple[int, float, float, float, float]] = []
        geometry = []
        seen_labels: set[tuple[int, float, float, float, float]] = set()
        raw_target_box_count = 0
        for annotation in sorted(annotations.get(image_id, []), key=lambda row: row["id"]):
            x, y, w, h = map(float, annotation["bbox"])
            require(w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= width + 1 and y + h <= height + 1,
                    f"invalid ROADWork box: {annotation['id']}")
            raw_target_box_count += 1
            row = (annotation["target_class_id"], (x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
            key = (row[0], *(round(value, 9) for value in row[1:]))
            normalization[split]["raw_rows"] += 1
            if key in seen_labels:
                normalization[split]["exact_duplicate_rows_removed"] += 1
                normalization[split][f"class_{row[0]}_duplicates_removed"] += 1
                continue
            seen_labels.add(key)
            rows.append(row)
            normalization[split]["emitted_unique_rows"] += 1
            geometry.append({
                "annotation_id": annotation["id"], "class_id": annotation["target_class_id"],
                "bbox_xywh_px": [x, y, w, h], "area_fraction": w * h / (width * height),
            })
        if raw_target_box_count != len(rows):
            normalization[split]["affected_images"] += 1
        safe_name = f"roadwork_{image_id:07d}_{Path(image['file_name']).name}"
        destination = output / "images" / split / safe_name
        link_or_copy(source, destination)
        label_path = output / "labels" / split / f"{Path(safe_name).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label_text(rows), encoding="ascii")
        relative = destination.relative_to(output).as_posix()
        image_lists[split].append(relative)
        manifest.append({
            "sample_id": f"roadwork-{split}-{image_id}", "dataset_id": "cmu_roadwork_iccv2025",
            "split": split, "source_id": source_id(image["file_name"], split),
            "original_file_name": image["file_name"], "image_path": relative,
            "label_path": label_path.relative_to(output).as_posix(), "width": width, "height": height,
            "image_sha256": sha256_file(destination), "label_sha256": sha256_file(label_path),
            "target_box_count": len(rows), "raw_target_box_count": raw_target_box_count,
            "exact_duplicate_annotation_count": raw_target_box_count - len(rows), "geometry": geometry,
            "annotation_authority": "ROADWork manually verified instance annotation",
        })


def emit_bollards(
    *, root: Path, output: Path, manifest: list[dict[str, Any]], image_lists: dict[str, list[str]], repeat: int,
) -> dict[str, Any]:
    groups: dict[str, list[Path]] = collections.defaultdict(list)
    for image in sorted(root.glob("*.jpg")):
        groups[sha256_file(image)].append(image)
    require(len(groups) == 40, f"expected 40 unique bollard images after SHA dedup, got {len(groups)}")
    disagreements = []
    for image_sha in sorted(groups):
        variants = groups[image_sha]
        candidates = []
        for image in variants:
            label = image.with_suffix(".txt")
            require(label.is_file(), f"bollard label missing: {label}")
            rows = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            candidates.append((len(rows), image.name.lower(), image, label, rows))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        _, _, image, label, raw_rows = candidates[0]
        if len({(count, sha256_file(candidate_label)) for count, _, _, candidate_label, _ in candidates}) > 1:
            disagreements.append({
                "image_sha256": image_sha, "variant_count": len(candidates),
                "selected_label": label.name, "label_box_counts": [row[0] for row in candidates],
            })
        parsed = []
        geometry = []
        for raw in raw_rows:
            values = raw.split()
            require(len(values) == 5, f"invalid bollard label row: {raw}")
            class_id, x, y, w, h = map(float, values)
            require(class_id == 0 and 0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1,
                    f"invalid bollard geometry: {raw}")
            parsed.append((2, x, y, w, h))
            geometry.append({"class_id": 2, "bbox_xywh_norm": [x, y, w, h], "area_fraction": w * h})
        safe_name = f"mendeley_bollard_{image_sha[:16]}.jpg"
        destination = output / "images" / "train" / safe_name
        link_or_copy(image, destination)
        label_path = output / "labels" / "train" / f"{Path(safe_name).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label_text(parsed), encoding="ascii")
        relative = destination.relative_to(output).as_posix()
        image_lists["train"].extend([relative] * repeat)
        manifest.append({
            "sample_id": f"mendeley-bollard-{image_sha[:16]}", "dataset_id": "mendeley_3psr2g4s7r_v2",
            "split": "train", "source_id": "mendeley_stationary_bollard",
            "original_file_name": image.name, "image_path": relative,
            "label_path": label_path.relative_to(output).as_posix(),
            "image_sha256": image_sha, "label_sha256": sha256_file(label_path),
            "target_box_count": len(parsed), "geometry": geometry, "sampler_repeat": repeat,
            "annotation_authority": "Mendeley CC BY 4.0 human YOLO bounding boxes",
        })
    return {"raw_file_count": sum(len(rows) for rows in groups.values()), "unique_image_count": len(groups),
            "label_disagreement_group_count": len(disagreements), "label_selection": "maximum_box_count_then_filename",
            "disagreements": disagreements}


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    matrix_path = args.matrix.resolve()
    matrix = validate_matrix(matrix_path, repo)
    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite dataset: {output}")
    roadwork_annotations = args.roadwork_annotations.resolve()
    roadwork_images = args.roadwork_images.resolve()
    bollard_root = args.bollard_root.resolve()
    data = matrix["data"]
    roadwork = data["roadwork"]
    require(sha256_file(args.roadwork_archive.resolve()) == roadwork["archive_sha256"], "ROADWork archive hash mismatch")
    require(sha256_file(args.bollard_archive.resolve()) == data["mendeley_bollard"]["archive_sha256"], "bollard archive hash mismatch")
    train_annotation = roadwork_annotations / roadwork["train_annotation"]
    val_annotation = roadwork_annotations / roadwork["validation_annotation"]
    require(sha256_file(train_annotation) == roadwork["train_annotation_sha256"], "ROADWork train annotation drifted")
    require(sha256_file(val_annotation) == roadwork["validation_annotation_sha256"], "ROADWork val annotation drifted")
    manifest: list[dict[str, Any]] = []
    image_lists: dict[str, list[str]] = {"train": [], "val": []}
    normalization = {"train": collections.Counter(), "val": collections.Counter()}
    emit_roadwork(split="train", annotation_path=train_annotation, images_root=roadwork_images,
                  output=output, manifest=manifest, image_lists=image_lists, selection=roadwork["selection"],
                  normalization=normalization)
    emit_roadwork(split="val", annotation_path=val_annotation, images_root=roadwork_images,
                  output=output, manifest=manifest, image_lists=image_lists, selection=roadwork["selection"],
                  normalization=normalization)
    bollard_audit = emit_bollards(root=bollard_root, output=output, manifest=manifest, image_lists=image_lists,
                                  repeat=int(data["mendeley_bollard"]["train_repeat_factor"]))
    manifest.sort(key=lambda row: row["sample_id"])
    manifest_path = output / "training_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest), encoding="utf-8")
    for split in ("train", "val"):
        (output / f"{split}.txt").write_text("".join(str((output / path).resolve()) + "\n" for path in image_lists[split]), encoding="utf-8")
    (output / "data.yaml").write_text(
        f"path: {output.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n  0: traffic cone\n  1: delineator\n  2: bollard\n",
        encoding="utf-8",
    )
    split_sources = {split: sorted({row["source_id"] for row in manifest if row["split"] == split}) for split in ("train", "val")}
    roadwork_overlap = set(source for source in split_sources["train"] if source.startswith("roadwork_")) & set(split_sources["val"])
    require(not roadwork_overlap, f"ROADWork source leakage: {sorted(roadwork_overlap)}")
    class_boxes = collections.Counter()
    per_source = collections.Counter()
    for row in manifest:
        per_source[(row["split"], row["source_id"])] += 1
        for geometry in row["geometry"]:
            class_boxes[CLASSES[geometry["class_id"]]] += 1
    receipt = {
        "schema": "blindassist_ustrf_r12d_training_dataset_receipt_v1",
        "matrix_sha256": sha256_file(matrix_path), "training_manifest_sha256": sha256_file(manifest_path),
        "data_yaml_sha256": sha256_file(output / "data.yaml"), "classes": CLASSES,
        "unique_image_count": len(manifest), "effective_train_draw_count_per_epoch": len(image_lists["train"]),
        "unique_split_counts": collections.Counter(row["split"] for row in manifest),
        "class_box_counts": dict(class_boxes),
        "split_sources": split_sources,
        "per_source_image_counts": {f"{split}:{source}": count for (split, source), count in sorted(per_source.items())},
        "roadwork": {
            "archive_sha256": sha256_file(args.roadwork_archive.resolve()),
            "train_annotation_sha256": sha256_file(train_annotation),
            "validation_annotation_sha256": sha256_file(val_annotation),
            "license": roadwork["license"], "source_disjoint": True,
            "review_receipt": roadwork["annotations_review"], "exact_geometry": "COCO bbox_xywh_px",
            "label_normalization": {
                "rule": "deduplicate_identical_serialized_yolo_rows_after_9_decimal_normalization_keep_lowest_annotation_id",
                "reason": "Ultralytics otherwise performs the same implicit duplicate-row removal while building its label cache",
                "train": dict(normalization["train"]), "val": dict(normalization["val"]),
            },
        },
        "mendeley_bollard": {
            "archive_sha256": sha256_file(args.bollard_archive.resolve()),
            "license": data["mendeley_bollard"]["license"], "doi": data["mendeley_bollard"]["doi"],
            "exact_geometry": "YOLO bbox_xywh_norm", **bollard_audit,
        },
        "gates": {
            "archive_hashes_passed": True, "annotation_hashes_passed": True,
            "exact_geometry_passed": True, "source_disjoint_roadwork_validation_passed": True,
            "event_frames_in_training": False, "synthetic_or_provisional_labels_in_training": False,
            "three_class_coverage_passed": set(class_boxes) == set(CLASSES),
            "dataset_admission_passed": set(class_boxes) == set(CLASSES),
        },
        "authority": {
            "r12d_research_training_data_admitted": True, "human_event_truth_claimed": False,
            "r13_inventory_read_authorized": False, "production_model_replacement_authorized": False,
        },
    }
    write_json(output / "dataset_receipt.json", receipt)
    print("USTRF_R12D_DATASET_OK", len(manifest), len(image_lists["train"]), len(image_lists["val"]), dict(class_boxes))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--roadwork-archive", type=Path, required=True)
    parser.add_argument("--roadwork-annotations", type=Path, required=True)
    parser.add_argument("--roadwork-images", type=Path, required=True)
    parser.add_argument("--bollard-archive", type=Path, required=True)
    parser.add_argument("--bollard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
