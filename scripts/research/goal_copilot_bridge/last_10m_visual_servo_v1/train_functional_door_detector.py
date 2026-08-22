#!/usr/bin/env python3
"""Train a room-door vs furniture-door detector from public DoorDetect labels.

This is a development-only provider component.  It never reads the TartanAir
S2 cohort, its truth, or any evaluator output.  The source dataset is split by
an immutable path hash before the first training call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CLASS_NAMES = ["door", "handle", "cabinet door", "refrigerator door"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VAL_MODULUS = 5
VAL_BUCKET = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_name(relative_image_path: str) -> str:
    value = int(hashlib.sha256(relative_image_path.encode("utf-8")).hexdigest(), 16)
    return "val" if value % VAL_MODULUS == VAL_BUCKET else "train"


def _yaml_path(path: Path) -> str:
    return path.resolve().as_posix()


def materialize(source: Path, output: Path, base_weights: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    images_dir = source / "images"
    labels_dir = source / "labels"
    images = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No images found under {images_dir}")

    split_paths: dict[str, list[Path]] = {"train": [], "val": []}
    class_counts: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}
    missing_labels: list[str] = []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            missing_labels.append(image.name)
            continue
        split = split_name(image.relative_to(source).as_posix())
        split_paths[split].append(image.resolve())
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                class_counts[split][int(fields[0])] += 1
    if missing_labels:
        raise RuntimeError(f"Missing labels for {len(missing_labels)} images: {missing_labels[:5]}")
    if not split_paths["train"] or not split_paths["val"]:
        raise RuntimeError("Deterministic split produced an empty partition")

    output.mkdir(parents=True, exist_ok=True)
    for split, paths in split_paths.items():
        (output / f"{split}.txt").write_text(
            "\n".join(_yaml_path(path) for path in paths) + "\n", encoding="utf-8"
        )
    data_yaml = output / "data.yaml"
    names_yaml = "\n".join(f"  {index}: {json.dumps(name)}" for index, name in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        f"train: {_yaml_path(output / 'train.txt')}\n"
        f"val: {_yaml_path(output / 'val.txt')}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )

    try:
        import subprocess

        source_commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        source_commit = "UNAVAILABLE"
    manifest = {
        "schema_version": "blindassist.functional_door_training.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "truth_isolation": "No TartanAir S2 input, truth, or evaluator output is read by this trainer.",
        "source": {
            "repository": "https://github.com/MiguelARD/DoorDetect-Dataset",
            "commit": source_commit,
            "image_count": len(images),
            "classes": CLASS_NAMES,
        },
        "split": {
            "algorithm": "sha256(relative_image_path) integer modulo 5; bucket 0 is val",
            "train_images": len(split_paths["train"]),
            "val_images": len(split_paths["val"]),
            "train_class_instances": {CLASS_NAMES[k]: v for k, v in sorted(class_counts["train"].items())},
            "val_class_instances": {CLASS_NAMES[k]: v for k, v in sorted(class_counts["val"].items())},
            "train_list_sha256": sha256_file(output / "train.txt"),
            "val_list_sha256": sha256_file(output / "val.txt"),
        },
        "provider": {
            "base_weights": str(base_weights.resolve()),
            "base_weights_sha256": sha256_file(base_weights),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    manifest = materialize(args.source, args.output, args.weights)
    print(json.dumps(manifest, indent=2))

    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    model.train(
        data=str((args.output / "data.yaml").resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.output.resolve()),
        name="yolo11n_room_vs_furniture_door",
        exist_ok=False,
        seed=0,
        deterministic=True,
        pretrained=True,
        patience=15,
        plots=True,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
