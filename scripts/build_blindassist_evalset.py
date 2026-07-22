from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve

from PIL import Image, ImageStat


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"
DEFAULT_CACHE_ROOT = ".downloads/detector-lab/datasets/coco100"
DEFAULT_SEED = 260527
DEFAULT_SAMPLE_COUNT = 150

SCENE_TARGETS = [
    ("front_near_primary", 45),
    ("side_passing_target", 25),
    ("far_large_object", 20),
    ("near_small_obstacle", 25),
    ("low_light_or_occlusion", 20),
    ("corridor_or_outdoor_slow", 15),
]

ASSIST_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "chair",
    "potted plant",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "book",
    "sports ball",
    "umbrella",
}

SMALL_OBSTACLES = {"backpack", "handbag", "suitcase", "bottle", "cup", "book", "sports ball", "umbrella"}
LARGE_OBJECTS = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "bench", "chair"}
OUTDOOR_CUES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"}

LEFT_BOUNDARY = 0.35
RIGHT_BOUNDARY = 0.65
MID_BOTTOM_RATIO = 0.45
MID_AREA_RATIO = 0.06
CENTER_NEAR_BOTTOM_RATIO = 0.60
CENTER_NEAR_AREA_RATIO = 0.12
NEAR_BOTTOM_RATIO = 0.62
NEAR_AREA_RATIO = 0.14
CRITICAL_BOTTOM_RATIO = 0.72
CRITICAL_AREA_RATIO = 0.20


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_instances(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("annotations/instances_val2017.json") as handle:
            return json.load(handle)


def annotations_by_image(instances: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in instances["annotations"]:
        if int(annotation.get("iscrowd", 0)) == 1:
            continue
        grouped[int(annotation["image_id"])].append(annotation)
    return grouped


def bbox_xyxy(annotation: dict[str, Any]) -> list[float]:
    x, y, width, height = [float(part) for part in annotation["bbox"]]
    return [x, y, x + width, y + height]


def bbox_metrics(annotation: dict[str, Any], image: dict[str, Any]) -> dict[str, float]:
    x, y, width, height = [float(part) for part in annotation["bbox"]]
    image_width = float(image["width"])
    image_height = float(image["height"])
    center_x = (x + width / 2) / image_width
    center_y = (y + height / 2) / image_height
    bottom = (y + height) / image_height
    area_ratio = (width * height) / (image_width * image_height)
    return {
        "center_x": center_x,
        "center_y": center_y,
        "bottom": bottom,
        "area_ratio": area_ratio,
        "width_ratio": width / image_width,
        "height_ratio": height / image_height,
    }


def direction_for(center_x: float) -> str:
    if center_x < LEFT_BOUNDARY:
        return "LEFT"
    if center_x > RIGHT_BOUNDARY:
        return "RIGHT"
    return "CENTER"


def distance_band_for(bottom: float, area_ratio: float, direction: str) -> str:
    if direction == "CENTER" and (bottom >= CRITICAL_BOTTOM_RATIO or area_ratio >= CRITICAL_AREA_RATIO):
        return "CRITICAL"
    near_bottom = CENTER_NEAR_BOTTOM_RATIO if direction == "CENTER" else NEAR_BOTTOM_RATIO
    near_area = CENTER_NEAR_AREA_RATIO if direction == "CENTER" else NEAR_AREA_RATIO
    if bottom >= near_bottom or area_ratio >= near_area:
        return "NEAR"
    if bottom >= MID_BOTTOM_RATIO or area_ratio >= MID_AREA_RATIO:
        return "MID"
    return "FAR"


def risk_level_for(distance_band: str, direction: str, scene_bucket: str) -> str:
    if distance_band == "CRITICAL":
        return "HIGH"
    if distance_band == "NEAR":
        return "HIGH" if direction == "CENTER" else "MEDIUM"
    if distance_band == "MID":
        return "MEDIUM" if direction == "CENTER" and scene_bucket in {"front_near_primary", "corridor_or_outdoor_slow"} else "LOW"
    return "NONE"


def should_alert_for(distance_band: str, direction: str, scene_bucket: str) -> bool:
    if distance_band in {"CRITICAL", "NEAR"}:
        return True
    return direction == "CENTER" and distance_band == "MID" and scene_bucket in {"front_near_primary", "corridor_or_outdoor_slow"}


def assist_scenario_for(scene_bucket: str, labels: set[str]) -> str:
    if scene_bucket == "corridor_or_outdoor_slow":
        if labels & {"traffic light", "stop sign", "car", "bus", "truck", "bicycle", "motorcycle"}:
            return "OUTDOOR_SLOW"
        return "CORRIDOR"
    if scene_bucket == "side_passing_target":
        return "CROWDED" if "person" in labels else "GENERAL"
    if scene_bucket == "low_light_or_occlusion":
        return "INDOOR"
    return "GENERAL"


def overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = bbox_xyxy(a)
    bx1, by1, bx2, by2 = bbox_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return intersection / min(area_a, area_b)


def has_occlusion_candidate(annotations: list[dict[str, Any]]) -> bool:
    for index, left in enumerate(annotations):
        for right in annotations[index + 1:]:
            if overlap_ratio(left, right) >= 0.18:
                return True
    return False


def scene_bucket_for(label: str, metrics: dict[str, float], all_labels: set[str], annotations: list[dict[str, Any]]) -> str | None:
    direction = direction_for(metrics["center_x"])
    bottom = metrics["bottom"]
    area_ratio = metrics["area_ratio"]
    if label in LARGE_OBJECTS and direction == "CENTER" and (bottom >= 0.60 or area_ratio >= 0.12):
        return "front_near_primary"
    if label in LARGE_OBJECTS and direction in {"LEFT", "RIGHT"} and bottom >= 0.45:
        return "side_passing_target"
    if label in LARGE_OBJECTS and bottom < 0.56 and 0.018 <= area_ratio <= 0.10:
        return "far_large_object"
    if label in SMALL_OBSTACLES and (bottom >= 0.50 or area_ratio >= 0.018):
        return "near_small_obstacle"
    if has_occlusion_candidate(annotations) or len(all_labels) >= 5:
        return "low_light_or_occlusion"
    if all_labels & OUTDOOR_CUES:
        return "corridor_or_outdoor_slow"
    return None


def primary_annotation_for(
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    categories: dict[int, str],
) -> tuple[str | None, dict[str, Any] | None]:
    best: tuple[float, str, dict[str, Any]] | None = None
    labels = {categories[int(item["category_id"])] for item in annotations}
    for annotation in annotations:
        label = categories[int(annotation["category_id"])]
        if label not in ASSIST_CLASSES:
            continue
        metrics = bbox_metrics(annotation, image)
        bucket = scene_bucket_for(label, metrics, labels, annotations)
        if bucket is None:
            continue
        direction = direction_for(metrics["center_x"])
        distance = distance_band_for(metrics["bottom"], metrics["area_ratio"], direction)
        bucket_bonus = {
            "front_near_primary": 6.0,
            "side_passing_target": 5.0,
            "near_small_obstacle": 4.5,
            "low_light_or_occlusion": 4.0,
            "far_large_object": 3.5,
            "corridor_or_outdoor_slow": 3.0,
        }[bucket]
        distance_bonus = {"CRITICAL": 4.0, "NEAR": 3.0, "MID": 2.0, "FAR": 1.0}[distance]
        score = bucket_bonus + distance_bonus + metrics["area_ratio"] * 4.0 + (1.0 if direction == "CENTER" else 0.0)
        if best is None or score > best[0]:
            best = (score, bucket, annotation)
    if best is None:
        return None, None
    return best[1], best[2]


def select_images(instances: dict[str, Any], sample_count: int, seed: int) -> list[dict[str, Any]]:
    categories = {int(item["id"]): item["name"] for item in instances["categories"]}
    image_by_id = {int(item["id"]): item for item in instances["images"]}
    grouped = annotations_by_image(instances)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for image_id, annotations in grouped.items():
        image = image_by_id[image_id]
        bucket, primary = primary_annotation_for(image, annotations, categories)
        if bucket is None or primary is None:
            continue
        candidates[bucket].append({"image": image, "primary": primary, "annotations": annotations, "bucket": bucket})

    rng = random.Random(seed)
    for rows in candidates.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for bucket, target_count in SCENE_TARGETS:
        for item in candidates[bucket]:
            image_id = int(item["image"]["id"])
            if image_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(image_id)
            if sum(1 for row in selected if row["bucket"] == bucket) >= target_count:
                break

    if len(selected) < sample_count:
        leftovers = [
            item
            for rows in candidates.values()
            for item in rows
            if int(item["image"]["id"]) not in selected_ids
        ]
        leftovers.sort(
            key=lambda item: (
                item["bucket"],
                int(item["image"]["id"]),
            )
        )
        rng.shuffle(leftovers)
        for item in leftovers:
            image_id = int(item["image"]["id"])
            if image_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(image_id)
            if len(selected) >= sample_count:
                break

    if len(selected) < sample_count:
        raise SystemExit(f"Only selected {len(selected)} images; requested {sample_count}")
    return selected[:sample_count]


def source_image(cache_root: Path, file_name: str) -> Path:
    return cache_root / "images" / file_name


def ensure_source_image(cache_root: Path, file_name: str, retries: int) -> Path:
    path = source_image(cache_root, file_name)
    download(f"{IMAGE_BASE_URL}/{file_name}", path, retries)
    return path


def object_record(annotation: dict[str, Any], categories: dict[int, str], class_name_to_id: dict[str, int]) -> dict[str, Any]:
    label = categories[int(annotation["category_id"])]
    x, y, width, height = [float(part) for part in annotation["bbox"]]
    return {
        "id": f"coco_ann_{int(annotation['id'])}",
        "class": label,
        "class_id": class_name_to_id[label],
        "bbox_xyxy": [round(x, 3), round(y, 3), round(x + width, 3), round(y + height, 3)],
        "bbox_source": "coco2017_instances_val2017",
        "iscrowd": int(annotation.get("iscrowd", 0)),
        "area": round(float(annotation.get("area", width * height)), 3),
    }


def yolo_line(obj: dict[str, Any], width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(part) for part in obj["bbox_xyxy"]]
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    center_x = (x1 + box_width / 2) / width
    center_y = (y1 + box_height / 2) / height
    return f"{int(obj['class_id'])} {center_x:.6f} {center_y:.6f} {box_width / width:.6f} {box_height / height:.6f}"


def mean_luma(path: Path) -> float:
    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32))
        return float(ImageStat.Stat(gray).mean[0])


def build_spec(classes: list[dict[str, Any]], sample_count: int) -> dict[str, Any]:
    return {
        "name": "blindassist_real_walk_evalset",
        "task": "detection + assistive-risk-evaluation",
        "classes": classes,
        "scenes": [name for name, _ in SCENE_TARGETS],
        "attributes": {
            "expected_risk_direction": ["NONE", "LEFT", "CENTER", "RIGHT"],
            "expected_distance_band": ["FAR", "MID", "NEAR", "CRITICAL"],
            "expected_should_alert": [True, False],
            "expected_risk_level": ["NONE", "LOW", "MEDIUM", "HIGH"],
            "assist_scenario": ["GENERAL", "INDOOR", "CORRIDOR", "CROWDED", "OUTDOOR_SLOW"],
        },
        "counts": {
            "requested_total": sample_count,
            "scene_targets": {name: count for name, count in SCENE_TARGETS},
        },
        "splits": {"test": 1.0},
        "image_style": "real public dataset images; no synthetic generation",
        "output_resolution": "source image resolution",
        "annotation_target": "manifest.jsonl + YOLO labels + COCO instances_test.json",
        "privacy_policy": "source images are retained locally only and are not committed to Git",
        "source_policy": {
            "primary_source": "COCO 2017 val images and instance annotations",
            "dataset_page": "https://cocodataset.org/dataset/detection-2017.htm",
            "image_base_url": IMAGE_BASE_URL,
            "redistribution_policy": "internal local evaluation artifact; do not redistribute original images from this repo",
        },
    }


def write_source_notes(dataset_root: Path, sample_count: int) -> None:
    text = f"""# BlindAssist Evalset Sources

This local evaluation set contains {sample_count} real public-dataset images selected from COCO 2017 validation images.

- Dataset page: https://cocodataset.org/dataset/detection-2017.htm
- Image base URL: {IMAGE_BASE_URL}
- Annotation URL: {ANNOTATIONS_URL}
- Redistribution policy in this workspace: original images are local-only evaluation artifacts under `test-artifacts.local/datasets/` and must not be committed to Git.
- BlindAssist risk fields are project-specific annotations generated for review and algorithm evaluation; standard COCO exports intentionally exclude those fields.

Planned but not included in this first generated artifact: Open Images, LOCO, GND, LAVN. They remain good sources for a future GPT/Codex-curated expansion after license/source evidence review and download setup.
"""
    (dataset_root / "source_licenses.md").write_text(text, encoding="utf-8")


def validate_manifest(rows: list[dict[str, Any]], dataset_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    hashes: dict[str, str] = {}
    directions = {"NONE", "LEFT", "CENTER", "RIGHT"}
    distances = {"FAR", "MID", "NEAR", "CRITICAL"}
    levels = {"NONE", "LOW", "MEDIUM", "HIGH"}
    scenarios = {"GENERAL", "INDOOR", "CORRIDOR", "CROWDED", "OUTDOOR_SLOW"}
    for row in rows:
        image_path = dataset_root / row["image_path"]
        if not image_path.is_file():
            errors.append(f"{row['id']}: missing image {row['image_path']}")
            continue
        image_hash = sha256_file(image_path)
        if image_hash in hashes:
            errors.append(f"{row['id']}: duplicate image hash with {hashes[image_hash]}")
        hashes[image_hash] = row["id"]
        if row.get("expected_risk_direction") not in directions:
            errors.append(f"{row['id']}: invalid expected_risk_direction")
        if row.get("expected_distance_band") not in distances:
            errors.append(f"{row['id']}: invalid expected_distance_band")
        if not isinstance(row.get("expected_should_alert"), bool):
            errors.append(f"{row['id']}: expected_should_alert must be boolean")
        if row.get("expected_risk_level") not in levels:
            errors.append(f"{row['id']}: invalid expected_risk_level")
        if row.get("assist_scenario") not in scenarios:
            errors.append(f"{row['id']}: invalid assist_scenario")
        width, height = int(row["width"]), int(row["height"])
        for obj in row.get("objects", []):
            x1, y1, x2, y2 = [float(part) for part in obj["bbox_xyxy"]]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                errors.append(f"{row['id']}: invalid bbox for {obj.get('id')}")
    return {
        "ok": not errors,
        "errors": errors,
        "image_count": len(rows),
        "unique_hashes": len(hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local BlindAssist real-walking evaluation set from public COCO images.")
    parser.add_argument("--output-root", default=None, help="Dataset output root. Defaults to test-artifacts.local/datasets/blindassist-evalset-<timestamp>.")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT, help="COCO cache root with annotations/images.")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cache_root = resolve_path(project_root, args.cache_root)
    dataset_root = resolve_path(
        project_root,
        args.output_root or Path("test-artifacts.local") / "datasets" / f"blindassist-evalset-{now_stamp()}",
    )
    annotations_zip = cache_root / "annotations_trainval2017.zip"
    download(ANNOTATIONS_URL, annotations_zip, args.retries)
    instances = load_instances(annotations_zip)

    categories = {int(item["id"]): item["name"] for item in instances["categories"]}
    classes = [
        {"id": index, "name": item["name"], "supercategory": item.get("supercategory", "object")}
        for index, item in enumerate(instances["categories"])
    ]
    class_name_to_id = {item["name"]: index for index, item in enumerate(instances["categories"])}
    selected = select_images(instances, args.sample_count, args.seed)

    for relative in [
        "images/test",
        "labels_yolo/test",
        "annotations",
        "qa",
    ]:
        (dataset_root / relative).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()

    for index, item in enumerate(selected, 1):
        image = item["image"]
        source = ensure_source_image(cache_root, image["file_name"], args.retries)
        suffix = Path(image["file_name"]).suffix.lower() or ".jpg"
        sample_id = f"blindassist_eval_{index:06d}"
        dest_rel = Path("images/test") / f"{sample_id}{suffix}"
        dest = dataset_root / dest_rel
        shutil.copy2(source, dest)

        primary = item["primary"]
        primary_label = categories[int(primary["category_id"])]
        primary_metrics = bbox_metrics(primary, image)
        direction = direction_for(primary_metrics["center_x"])
        distance_band = distance_band_for(primary_metrics["bottom"], primary_metrics["area_ratio"], direction)
        risk_level = risk_level_for(distance_band, direction, item["bucket"])
        should_alert = should_alert_for(distance_band, direction, item["bucket"])
        image_labels = {categories[int(annotation["category_id"])] for annotation in item["annotations"]}
        assist_scenario = assist_scenario_for(item["bucket"], image_labels)
        luma = mean_luma(dest)

        objects = [
            object_record(annotation, categories, class_name_to_id)
            for annotation in item["annotations"]
            if categories[int(annotation["category_id"])] in class_name_to_id
        ]
        labels = sorted({obj["class"] for obj in objects})
        label_path = dataset_root / "labels_yolo/test" / f"{sample_id}.txt"
        label_path.write_text(
            "\n".join(yolo_line(obj, int(image["width"]), int(image["height"])) for obj in objects) + "\n",
            encoding="utf-8",
        )

        source_hash = sha256_file(dest)
        rationale = (
            f"{item['bucket']} selected from COCO object '{primary_label}', "
            f"direction={direction}, distance={distance_band}, should_alert={should_alert}; "
            "risk label is a BlindAssist prelabel for isolated GPT/Codex review, not a safety guarantee."
        )
        row = {
            "id": sample_id,
            "image_path": dest_rel.as_posix(),
            "split": "test",
            "width": int(image["width"]),
            "height": int(image["height"]),
            "labels": labels,
            "prompt": "real public COCO val2017 image selected for BlindAssist walking-risk evaluation",
            "objects": objects,
            "attributes": {
                "scene_bucket": item["bucket"],
                "source_luma_mean": round(luma, 3),
                "primary_label": primary_label,
                "object_count": len(objects),
                "possible_occlusion": has_occlusion_candidate(item["annotations"]),
            },
            "expected_risk_direction": direction,
            "expected_distance_band": distance_band,
            "expected_should_alert": should_alert,
            "expected_risk_level": risk_level,
            "assist_scenario": assist_scenario,
            "primary_object_id": f"coco_ann_{int(primary['id'])}",
            "risk_rationale": rationale,
            "review_status": "pending_model_review",
            "status": "pending_review",
            "source": {
                "dataset": "COCO2017 val",
                "license": "COCO dataset terms; annotations from COCO, images from source URLs",
                "original_url_or_id": f"{IMAGE_BASE_URL}/{image['file_name']}",
                "original_image_id": int(image["id"]),
                "original_file_name": image["file_name"],
                "sha256": source_hash,
                "redistribution_policy": "local_internal_eval_only_do_not_commit_original_image",
            },
        }
        rows.append(row)
        bucket_counts[item["bucket"]] += 1
        review_rows.append(
            {
                "id": sample_id,
                "image_path": dest_rel.as_posix(),
                "scene_bucket": item["bucket"],
                "primary_label": primary_label,
                "expected_risk_direction": direction,
                "expected_distance_band": distance_band,
                "expected_should_alert": str(should_alert).lower(),
                "expected_risk_level": risk_level,
                "assist_scenario": assist_scenario,
                "review_status": row["review_status"],
                "reviewer_type": "ai_model",
                "reviewer_id": "",
                "review_confidence": "",
                "independent_review_count": "",
                "review_notes": "",
            }
        )

    write_json(dataset_root / "dataset_spec.json", build_spec(classes, args.sample_count))
    (dataset_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (dataset_root / "generation_records.jsonl").write_text("", encoding="utf-8")
    write_source_notes(dataset_root, len(rows))

    with (dataset_root / "qa" / "model_review_checklist.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    validation = validate_manifest(rows, dataset_root)
    validation["scene_counts"] = dict(bucket_counts)
    validation["created_at"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    write_json(dataset_root / "qa" / "blindassist_manifest_validation.json", validation)
    if not validation["ok"]:
        raise SystemExit("BlindAssist manifest validation failed; see qa/blindassist_manifest_validation.json")

    print(f"dataset={dataset_root}")
    print(f"sample_count={len(rows)}")
    print(f"scene_counts={dict(bucket_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
