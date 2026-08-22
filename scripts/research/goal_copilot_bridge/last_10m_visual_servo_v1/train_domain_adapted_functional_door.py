#!/usr/bin/env python3
"""Domain-adapt the functional-door detector with public TartanAir segmentation.

Only the previously designated ArchVizTinyHouseDay development environment is
used.  Exact ``door`` is the room-door positive; ``cupboard``/``drawer`` and
``fridge`` provide explicit furniture-door negatives.  No S2 environment is
read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_functional_door_detector import CLASS_NAMES, sha256_file


MIN_COMPONENT_PIXELS = 64
SYNTHETIC_CLASS_MAP = {"door": 0, "cupboard": 2, "drawer": 2, "fridge": 3}


def trajectory_split(name: str) -> str:
    bucket = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16) % 5
    return "val" if bucket == 0 else "train"


def components_to_yolo(segmentation: np.ndarray, label_id: int, class_id: int) -> list[str]:
    count, _, stats, _ = cv2.connectedComponentsWithStats((segmentation == label_id).astype(np.uint8), 8)
    height, width = segmentation.shape
    rows = []
    for component in range(1, count):
        x, y, box_width, box_height, pixels = [int(value) for value in stats[component]]
        if pixels < MIN_COMPONENT_PIXELS or box_width < 4 or box_height < 4:
            continue
        center_x = (x + box_width / 2.0) / width
        center_y = (y + box_height / 2.0) / height
        rows.append(f"{class_id} {center_x:.8f} {center_y:.8f} {box_width / width:.8f} {box_height / height:.8f}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tartanair-environment", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--doordetect-split", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite domain-adaptation output: {output}")
    synthetic_images = output / "synthetic" / "images"
    synthetic_labels = output / "synthetic" / "labels"
    synthetic_images.mkdir(parents=True)
    synthetic_labels.mkdir(parents=True)

    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))["name_map"]
    required = set(SYNTHETIC_CLASS_MAP)
    if not required.issubset(label_map):
        raise RuntimeError(f"Missing TartanAir labels: {sorted(required - set(label_map))}")
    split_paths: dict[str, list[Path]] = {"train": [], "val": []}
    instance_counts: dict[str, dict[int, int]] = {"train": {}, "val": {}}
    for trajectory in sorted((args.tartanair_environment / "Data_easy").glob("P*")):
        split = trajectory_split(trajectory.name)
        for seg_path in sorted((trajectory / "seg_lcam_front").glob("*_lcam_front_seg.png")):
            frame_id = seg_path.name.split("_")[0]
            image_path = trajectory / "image_lcam_front" / f"{frame_id}_lcam_front.png"
            if not image_path.is_file():
                raise RuntimeError(f"Missing synchronized image: {image_path}")
            segmentation = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
            if segmentation is None or segmentation.ndim != 2:
                raise RuntimeError(f"Invalid segmentation: {seg_path}")
            labels = []
            for name, class_id in SYNTHETIC_CLASS_MAP.items():
                rows = components_to_yolo(segmentation, int(label_map[name]), class_id)
                labels.extend(rows)
                instance_counts[split][class_id] = instance_counts[split].get(class_id, 0) + len(rows)
            target_stem = f"{trajectory.name}_{frame_id}"
            target_image = synthetic_images / f"{target_stem}.png"
            try:
                os.link(image_path.resolve(), target_image)
            except OSError:
                shutil.copy2(image_path, target_image)
            (synthetic_labels / f"{target_stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            split_paths[split].append(target_image.resolve())

    for split in ("train", "val"):
        original = [Path(line.strip()).resolve() for line in (args.doordetect_split / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        combined = original + split_paths[split]
        (output / f"{split}.txt").write_text("\n".join(path.as_posix() for path in combined) + "\n", encoding="utf-8")
    names_yaml = "\n".join(f"  {index}: {json.dumps(name)}" for index, name in enumerate(CLASS_NAMES))
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"train: {(output / 'train.txt').as_posix()}\nval: {(output / 'val.txt').as_posix()}\nnc: 4\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "blindassist.functional_door_domain_adaptation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "source_environment": "ArchVizTinyHouseDay",
        "excluded_environments": ["RetroOffice", "CountryHouse"],
        "s2_truth_or_evaluator_access": False,
        "synthetic_class_map": SYNTHETIC_CLASS_MAP,
        "minimum_component_pixels": MIN_COMPONENT_PIXELS,
        "split_rule": "trajectory sha256 modulo 5; bucket 0 is val",
        "synthetic_train_images": len(split_paths["train"]),
        "synthetic_val_images": len(split_paths["val"]),
        "synthetic_instance_counts": {split: {CLASS_NAMES[key]: value for key, value in sorted(counts.items())} for split, counts in instance_counts.items()},
        "weights_sha256": sha256_file(args.weights),
        "train_list_sha256": sha256_file(output / "train.txt"),
        "val_list_sha256": sha256_file(output / "val.txt"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    from ultralytics import YOLO

    YOLO(str(args.weights.resolve())).train(
        data=str(data_yaml), epochs=args.epochs, imgsz=640, batch=args.batch, workers=4,
        device=0, project=str(output), name="yolo11n_domain_adapted", exist_ok=False,
        seed=0, deterministic=True, pretrained=True, patience=10, plots=True, verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
