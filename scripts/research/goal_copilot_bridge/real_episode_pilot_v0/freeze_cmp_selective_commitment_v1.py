"""Freeze disjoint CMP Development/Confirmation/Reserve cohorts before provider output."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA_VERSION = "cmp_selective_commitment_v1_fresh_split_v0"
SALT = "cmp-selective-commitment-v1-fresh"
SPLIT_SIZES = {"development": 32, "confirmation": 64}


class FreezeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_objects(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring("<root>" + path.read_text(encoding="utf-8", errors="replace") + "</root>")
    objects = []
    for node in root.findall("object"):
        points = node.find("points")
        objects.append({
            "label": int(node.findtext("label", "-1")),
            "labelname": node.findtext("labelname", "").strip().lower(),
            "x": [float(item.text) for item in points.findall("x")],
            "y": [float(item.text) for item in points.findall("y")],
        })
    return objects


def assign_splits(candidates: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted((dict(item) for item in candidates), key=lambda item: item["fresh_rank_sha256"])
    development_end = SPLIT_SIZES["development"]
    confirmation_end = development_end + SPLIT_SIZES["confirmation"]
    return {
        "development": ordered[:development_end],
        "confirmation": ordered[development_end:confirmation_end],
        "reserve": ordered[confirmation_end:],
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    consumed_path = args.consumed_roster.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FreezeError("fresh split output already exists")
    consumed_doc = json.loads(consumed_path.read_text(encoding="utf-8"))
    consumed_hashes = {item["rgb_sha256"] for item in consumed_doc["observations"]}
    if len(consumed_hashes) != 89 or consumed_doc.get("selected_count") != 89:
        raise FreezeError("consumed roster identity mismatch")

    eligible = []
    for split in ("base", "extended"):
        data_dir = dataset_root / split / split
        for rgb_path in sorted(data_dir.glob("*.jpg")):
            xml_path = rgb_path.with_suffix(".xml")
            label_path = rgb_path.with_suffix(".png")
            doors = [item for item in parse_objects(xml_path) if item["labelname"] == "door"]
            labels = np.asarray(Image.open(label_path))
            door_y, door_x = np.nonzero(labels == 4)
            if len(doors) != 1 or door_x.size == 0:
                continue
            rgb_sha256 = sha256_file(rgb_path)
            width, height = Image.open(rgb_path).size
            relative_rgb = rgb_path.relative_to(dataset_root).as_posix()
            eligible.append({
                "rgb_path": relative_rgb,
                "xml_path": xml_path.relative_to(dataset_root).as_posix(),
                "label_path": label_path.relative_to(dataset_root).as_posix(),
                "image_width": width,
                "image_height": height,
                "native_xml_door": doors[0],
                "native_mask_bbox_xyxy": [
                    int(door_x.min()), int(door_y.min()), int(door_x.max()) + 1, int(door_y.max()) + 1,
                ],
                "door_pixel_count": int(door_x.size),
                "rgb_sha256": rgb_sha256,
                "xml_sha256": sha256_file(xml_path),
                "label_sha256": sha256_file(label_path),
                "fresh_rank_sha256": hashlib.sha256(f"{SALT}|{relative_rgb}".encode("utf-8")).hexdigest(),
            })
    if len(eligible) != 211:
        raise FreezeError(f"eligible universe drift: {len(eligible)}")
    eligible_hashes = {item["rgb_sha256"] for item in eligible}
    if not consumed_hashes <= eligible_hashes:
        raise FreezeError("consumed roster is not a subset of the native eligible universe")
    fresh = [item for item in eligible if item["rgb_sha256"] not in consumed_hashes]
    if len(fresh) != 122:
        raise FreezeError(f"fresh denominator drift: {len(fresh)}")
    splits = assign_splits(fresh)
    if [len(splits[name]) for name in ("development", "confirmation", "reserve")] != [32, 64, 26]:
        raise FreezeError("fresh split sizes drift")
    assigned = [item["rgb_sha256"] for rows in splits.values() for item in rows]
    if len(assigned) != len(set(assigned)) or set(assigned) & consumed_hashes:
        raise FreezeError("fresh split overlap detected")

    result = {
        "schema_version": SCHEMA_VERSION,
        "selection_salt": SALT,
        "dataset_root": str(dataset_root),
        "source_archives": {
            "CMP_facade_DB_base.zip": sha256_file(dataset_root / "CMP_facade_DB_base.zip"),
            "CMP_facade_DB_extended.zip": sha256_file(dataset_root / "CMP_facade_DB_extended.zip"),
        },
        "consumed_roster_sha256": sha256_file(consumed_path),
        "eligible_universe_count": len(eligible),
        "consumed_count": len(consumed_hashes),
        "fresh_count": len(fresh),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "provider_calls": 0,
        "teacher_calls": 0,
        "truth_authority": "NATIVE_GT",
        "splits": splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"output": str(output), "sha256": sha256_file(output), **result["split_counts"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--consumed-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(freeze(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
