#!/usr/bin/env python3
"""Materialize an automatic public TartanAir door-segmentation train set.

Only development environments are accepted.  The remote ZIP files are read
through HTTP range requests so selected samples are fetched without downloading
multi-gigabyte archives in full.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


REPOSITORY = "theairlabcmu/tartanair2"
FORMAL_ENVIRONMENTS = frozenset({"RetroOffice", "CountryHouse", "AmericanDiner", "House"})
HARD_NEGATIVE_NAMES = frozenset({"cupboard", "drawer", "fridge"})
MIN_PIXELS = 64


def split_for_trajectory(environment: str, trajectory: str) -> str:
    digest = hashlib.sha256(f"{environment}/{trajectory}".encode()).digest()
    return "val" if digest[0] % 5 == 0 else "train"


def sample_key(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def binary_door_mask(segmentation: np.ndarray, door_id: int) -> np.ndarray:
    return (segmentation == door_id).astype(np.uint8)


def _trajectory(name: str) -> str:
    return next(part for part in Path(name).parts if part.startswith("P") and part[1:].isdigit())


def _image_member(seg_member: str) -> str:
    return seg_member.replace("/seg_lcam_front/", "/image_lcam_front/").replace("_lcam_front_seg.png", "_lcam_front.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", nargs="+", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-per-environment", type=int, default=300)
    parser.add_argument("--hard-negative-per-environment", type=int, default=150)
    parser.add_argument("--background-negative-per-environment", type=int, default=50)
    args = parser.parse_args()
    environments = list(dict.fromkeys(args.environments))
    _require(not (set(environments) & FORMAL_ENVIRONMENTS), "formal cohort environment requested for training")
    _require(not args.output.exists(), "semantic training materialization output already exists")

    from huggingface_hub import HfFileSystem

    output = args.output.resolve()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "masks" / split).mkdir(parents=True)
    fs = HfFileSystem()
    rows = []
    for environment in environments:
        label_path = args.label_root / environment / "seg_label_map.json"
        labels = json.loads(label_path.read_text(encoding="utf-8"))["name_map"]
        _require("door" in labels, f"exact door absent in {environment}")
        door_id = int(labels["door"])
        hard_ids = {int(labels[name]) for name in HARD_NEGATIVE_NAMES if name in labels}
        base = f"datasets/{REPOSITORY}/{environment}/Data_easy"
        with fs.open(f"{base}/seg_lcam_front.zip", "rb") as seg_file, fs.open(f"{base}/image_lcam_front.zip", "rb") as image_file:
            with zipfile.ZipFile(seg_file) as seg_zip, zipfile.ZipFile(image_file) as image_zip:
                pools: dict[str, list[tuple[str, bytes, str]]] = {"positive": [], "hard_negative": [], "background_negative": []}
                for member in seg_zip.namelist():
                    if not member.endswith("_lcam_front_seg.png"):
                        continue
                    raw = seg_zip.read(member)
                    segmentation = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                    _require(segmentation is not None and segmentation.ndim == 2, f"invalid public segmentation: {member}")
                    door_pixels = int((segmentation == door_id).sum())
                    hard_pixels = int(np.isin(segmentation, list(hard_ids)).sum()) if hard_ids else 0
                    kind = "positive" if door_pixels >= MIN_PIXELS else ("hard_negative" if door_pixels == 0 and hard_pixels >= MIN_PIXELS else ("background_negative" if door_pixels == 0 else ""))
                    if kind:
                        pools[kind].append((member, raw, split_for_trajectory(environment, _trajectory(member))))
                limits = {"positive": args.positive_per_environment, "hard_negative": args.hard_negative_per_environment, "background_negative": args.background_negative_per_environment}
                for kind, pool in pools.items():
                    selected = sorted(pool, key=lambda row: sample_key(row[0]))[: limits[kind]]
                    for member, seg_raw, split in selected:
                        image_member = _image_member(member)
                        image_raw = image_zip.read(image_member)
                        image = Image.open(io.BytesIO(image_raw)).convert("RGB")
                        segmentation = cv2.imdecode(np.frombuffer(seg_raw, np.uint8), cv2.IMREAD_UNCHANGED)
                        mask = Image.fromarray(binary_door_mask(segmentation, door_id))
                        stem = f"{environment}_{_trajectory(member)}_{Path(member).name.split('_')[0]}"
                        image_path = output / "images" / split / f"{stem}.png"
                        mask_path = output / "masks" / split / f"{stem}.png"
                        image.save(image_path)
                        mask.save(mask_path)
                        rows.append({"environment": environment, "source_member": member, "split": split, "kind": kind, "image_path": str(image_path), "mask_path": str(mask_path), "image_sha256": sha256(image_path), "mask_sha256": sha256(mask_path)})
        print(f"materialized {environment}: {sum(row['environment'] == environment for row in rows)} samples", flush=True)

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\nmasks_dir: masks\nnames:\n  0: background\n  1: door\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "blindassist_public_tartanair_door_semantic_training_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "source_environments": environments,
        "excluded_formal_environments": sorted(FORMAL_ENVIRONMENTS),
        "private_truth_access": False,
        "selection": {"minimum_pixels": MIN_PIXELS, "positive_per_environment": args.positive_per_environment, "hard_negative_per_environment": args.hard_negative_per_environment, "background_negative_per_environment": args.background_negative_per_environment, "ordering": "source member sha256 ascending", "split": "environment/trajectory sha256 first byte modulo 5; zero is validation"},
        "case_count": len(rows),
        "train_count": sum(row["split"] == "train" for row in rows),
        "val_count": sum(row["split"] == "val" for row in rows),
        "kind_counts": {kind: sum(row["kind"] == kind for row in rows) for kind in ("positive", "hard_negative", "background_negative")},
        "data_yaml_sha256": sha256(data_yaml),
        "cases": rows,
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("case_count", "train_count", "val_count", "kind_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
