from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"
DEFAULT_OUTPUT_ROOT = ".downloads/detector-lab/datasets/coco100"
DEFAULT_SEED = 260527
DEFAULT_SAMPLE_COUNT = 100


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, retries: int) -> None:
    if target.is_file() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            print(f"download={url} target={target} attempt={attempt}")
            urlretrieve(url, tmp)
            tmp.replace(target)
            return
        except (OSError, URLError) as error:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            print(f"download_retry={attempt} error={type(error).__name__}: {error}")
            time.sleep(min(2 * attempt, 10))


def load_instances(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("annotations/instances_val2017.json") as handle:
            return json.load(handle)


def annotations_by_image(instances: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for annotation in instances["annotations"]:
        if annotation.get("iscrowd") == 1:
            continue
        image_id = int(annotation["image_id"])
        result.setdefault(image_id, []).append(annotation)
    return result


def select_images(instances: dict[str, Any], sample_count: int, seed: int) -> list[dict[str, Any]]:
    categories = {int(item["id"]): item["name"] for item in instances["categories"]}
    image_by_id = {int(item["id"]): item for item in instances["images"]}
    grouped_annotations = annotations_by_image(instances)

    candidates = [image_id for image_id in image_by_id if image_id in grouped_annotations]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected_ids = candidates[:sample_count]

    selected = []
    for image_id in selected_ids:
        image = image_by_id[image_id]
        annotations = grouped_annotations[image_id]
        category_names = sorted({categories[int(item["category_id"])] for item in annotations})
        selected.append(
            {
                "id": image_id,
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "annotation_count": len(annotations),
                "categories": category_names,
            }
        )
    return selected


def coco100_annotations(instances: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    categories = {int(item["id"]): item["name"] for item in instances["categories"]}
    grouped_annotations = annotations_by_image(instances)
    selected_ids = {int(image["id"]) for image in selected}

    images = []
    for image in selected:
        image_id = int(image["id"])
        annotations = []
        for annotation in grouped_annotations.get(image_id, []):
            category_id = int(annotation["category_id"])
            x, y, width, height = [float(part) for part in annotation["bbox"]]
            annotations.append(
                {
                    "id": int(annotation["id"]),
                    "category_id": category_id,
                    "category_name": categories[category_id],
                    "bbox_xywh": [x, y, width, height],
                    "bbox_xyxy": [x, y, x + width, y + height],
                    "area": float(annotation.get("area", width * height)),
                }
            )

        images.append(
            {
                "id": image_id,
                "file_name": image["file_name"],
                "relative_path": f"images/{image['file_name']}",
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": annotations,
            }
        )

    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "name": "coco100-val2017-fixed-sample-annotations",
        "source": "COCO val2017 instances",
        "sample_count": len(images),
        "selected_image_ids": sorted(selected_ids),
        "matching_policy": {
            "category": "same COCO category name",
            "iou_threshold": 0.5,
            "crowd_annotations": "excluded",
        },
        "images": images,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fixed COCO val2017 sample for detector validation.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_root = resolve_path(project_root, args.output_root)
    images_root = output_root / "images"
    annotations_zip = output_root / "annotations_trainval2017.zip"

    download(ANNOTATIONS_URL, annotations_zip, args.retries)
    instances = load_instances(annotations_zip)
    selected = select_images(instances, args.sample_count, args.seed)

    manifest_images = []
    for image in selected:
        file_name = image["file_name"]
        image_url = f"{IMAGE_BASE_URL}/{file_name}"
        image_path = images_root / file_name
        download(image_url, image_path, args.retries)
        manifest_images.append(
            {
                **image,
                "relative_path": f"images/{file_name}",
                "source_url": image_url,
                "sha256": sha256_file(image_path),
                "size_bytes": image_path.stat().st_size,
            }
        )

    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "name": "coco100-val2017-fixed-sample",
        "sample_count": len(manifest_images),
        "requested_count": args.sample_count,
        "seed": args.seed,
        "sources": {
            "annotations": ANNOTATIONS_URL,
            "images": IMAGE_BASE_URL,
            "dataset_page": "https://cocodataset.org/dataset/detection-2017.htm",
            "overview_page": "https://cocodataset.org/",
        },
        "annotations_zip_sha256": sha256_file(annotations_zip),
        "images": manifest_images,
    }
    write_json(output_root / "coco100_manifest.json", manifest)
    write_json(output_root / "coco100_annotations.json", coco100_annotations(instances, selected))
    (output_root / "images.txt").write_text(
        "".join(f"{item['relative_path']}\n" for item in manifest_images),
        encoding="utf-8",
    )
    print(f"manifest={output_root / 'coco100_manifest.json'}")
    print(f"annotations={output_root / 'coco100_annotations.json'}")
    print(f"image_count={len(manifest_images)}")


if __name__ == "__main__":
    main()
