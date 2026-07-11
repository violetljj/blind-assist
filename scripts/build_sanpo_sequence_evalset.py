from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen, urlretrieve

import numpy as np
from PIL import Image, ImageDraw


DATASET_PAGE = "https://google-research-datasets.github.io/sanpo_dataset/"
DATASET_REPO = "https://github.com/google-research-datasets/sanpo_dataset"
LICENSE_NAME = "Creative Commons Attribution 4.0 International"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SANPO_CITATION = "Waghmare et al., SANPO: A Scene Understanding Accessibility and Human Navigation Dataset, WACV 2025, pp. 7855-7864"
GCS_BUCKET = "gresearch"
GCS_PREFIX = "sanpo_dataset/v0"
GCS_API = f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
GCS_MEDIA = f"https://storage.googleapis.com/{GCS_BUCKET}"
DEFAULT_SESSION_ID = "-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG"
DEFAULT_CAMERA = "camera_chest"
DEFAULT_LENS = "left"
DEFAULT_TARGET_FPS = 10.0
DEFAULT_MAX_FRAMES = 30

EXACT_COCO_MAPPINGS = {
    "pedestrian": "person",
    "traffic light": "traffic light",
}

REVIEW_ISSUE_TAGS = [
    "missing_target",
    "wrong_class",
    "extra_target",
    "bad_bbox",
    "duplicate_sample",
    "unsafe_or_sensitive",
    "low_quality",
    "risk_label_uncertain",
    "sequence_break",
]


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_base64_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64_encode(digest.digest())


def base64_encode(value: bytes) -> str:
    import base64

    return base64.b64encode(value).decode("ascii")


def verify_gcs_md5(path: Path, item: dict[str, Any]) -> None:
    expected = item.get("md5Hash")
    if expected and md5_base64_file(path) != expected:
        raise ValueError(f"GCS MD5 mismatch for {path}")


def fetch_json(url: str, retries: int = 3) -> Any:
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, URLError) as error:
            if attempt == retries:
                raise
            print(f"json_retry={attempt} url={url} error={type(error).__name__}: {error}")
            time.sleep(min(attempt * 2, 8))
    raise AssertionError("unreachable")


def fetch_text(url: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8-sig")
        except (OSError, URLError) as error:
            if attempt == retries:
                raise
            print(f"text_retry={attempt} url={url} error={type(error).__name__}: {error}")
            time.sleep(min(attempt * 2, 8))
    raise AssertionError("unreachable")


def list_gcs_objects(prefix: str, retries: int = 3) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        query = {"prefix": prefix, "maxResults": 1000}
        if page_token:
            query["pageToken"] = page_token
        payload = fetch_json(f"{GCS_API}?{urlencode(query)}", retries=retries)
        items.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            return items


def download(url: str, target: Path, retries: int = 3) -> None:
    if target.is_file() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            print(f"download={url} target={target} attempt={attempt}")
            urlretrieve(url, temp)
            temp.replace(target)
            return
        except (OSError, URLError) as error:
            if temp.exists():
                temp.unlink()
            if attempt == retries:
                raise
            print(f"download_retry={attempt} error={type(error).__name__}: {error}")
            time.sleep(min(attempt * 2, 8))


def get_gcs_object(object_name: str, retries: int = 3) -> dict[str, Any]:
    return fetch_json(f"{GCS_API}/{quote(object_name, safe='')}", retries=retries)


def media_url(object_name: str, generation: str | None = None) -> str:
    url = f"{GCS_MEDIA}/{quote(object_name, safe='/')}"
    return f"{url}?generation={generation}" if generation else url


def object_inventory(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item.get("name"),
        "generation": item.get("generation"),
        "metageneration": item.get("metageneration"),
        "size": int(item["size"]) if item.get("size") is not None else None,
        "md5_base64": item.get("md5Hash"),
        "crc32c_base64": item.get("crc32c"),
    }


def frame_number(object_name: str) -> int:
    return int(Path(object_name).stem)


def resample_indices(
    available_indices: Iterable[int],
    source_fps: float,
    target_fps: float,
    start_frame: int,
    max_frames: int,
) -> list[int]:
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if target_fps > source_fps:
        raise ValueError("target_fps cannot exceed source_fps without frame duplication")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    available = set(int(item) for item in available_indices if int(item) >= start_frame)
    selected: list[int] = []
    step = source_fps / target_fps
    sample_index = 0
    while len(selected) < max_frames:
        candidate = start_frame + int(math.floor(sample_index * step + 0.5))
        if candidate in available and (not selected or candidate != selected[-1]):
            selected.append(candidate)
        if candidate > (max(available) if available else start_frame):
            break
        sample_index += 1
    return selected


def parse_mask_regions(
    mask_path: Path,
    label_by_id: dict[int, str],
    mapped_classes: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapped_classes = mapped_classes or EXACT_COCO_MAPPINGS
    with Image.open(mask_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    regions: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for color, pixel_count in zip(colors, counts, strict=True):
        class_id, instance_high, instance_low = [int(value) for value in color]
        if class_id == 0:
            continue
        class_name = label_by_id.get(class_id, f"unknown_{class_id}")
        instance_id = instance_high * 256 + instance_low
        matches = np.all(rgb == color, axis=2)
        ys, xs = np.where(matches)
        if xs.size == 0:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        region_id = f"sanpo_{class_id}_{instance_id}"
        region = {
            "id": region_id,
            "class": class_name,
            "sanpo_class_id": class_id,
            "instance_id": instance_id,
            "bbox_xyxy": bbox,
            "pixel_count": int(pixel_count),
            "bbox_source": "sanpo_panoptic_or_semantic_mask",
        }
        regions.append(region)
        mapped = mapped_classes.get(class_name)
        if mapped:
            objects.append(
                {
                    "id": region_id,
                    "class": mapped,
                    "bbox_xyxy": bbox,
                    "bbox_source": "sanpo_exact_class_mapping_pending_human_review",
                    "source_class": class_name,
                    "source_class_id": class_id,
                }
            )
    return regions, objects


def draw_review_image(image_path: Path, output_path: Path, regions: list[dict[str, Any]]) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    priority = {"obstacle", "stairs", "curb", "inaccessible surface", "pedestrian", "vehicle", "pole"}
    for region in regions:
        if region["class"] not in priority:
            continue
        x1, y1, x2, y2 = region["bbox_xyxy"]
        color = (220, 38, 38) if region["class"] in {"obstacle", "stairs", "curb", "inaccessible surface"} else (37, 99, 235)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        draw.text((x1 + 4, max(0, y1 - 18)), region["class"], fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((960, 540))
    image.save(output_path, quality=88)


def build_preview(dataset_root: Path, rows: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for row in rows:
        rel = Path("qa") / "boxed" / f"{row['id']}.jpg"
        draw_review_image(dataset_root / row["image_path"], dataset_root / rel, row["source_regions"])
        classes = Counter(region["class"] for region in row["source_regions"])
        cards.append(
            "<article><img loading='lazy' src='../{}' alt='{}'><h2>{}</h2>"
            "<p>source frame {} · target frame {} · {}</p>"
            "<p><strong>Review:</strong> risk fields pending; confirm primary obstacle, direction, distance, alert and approach state.</p></article>".format(
                rel.as_posix(),
                html.escape(row["id"]),
                html.escape(row["id"]),
                row["source_frame_index"],
                row["frame_index"],
                html.escape(", ".join(f"{name}:{count}" for name, count in classes.most_common())),
            )
        )
    document = """<!doctype html><html lang='en'><head><meta charset='utf-8'><title>BlindAssist SANPO review</title>
<style>body{font:15px system-ui;background:#f6f5f1;color:#172033;margin:0;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}article{background:white;border:1px solid #d8dce5;border-radius:14px;padding:14px}img{width:100%;height:auto;border-radius:9px}h2{font-size:15px}p{color:#526075;line-height:1.45}</style></head><body><h1>BlindAssist SANPO pilot — pending human review</h1><main>""" + "\n".join(cards) + "</main></body></html>\n"
    (dataset_root / "qa" / "preview.html").write_text(document, encoding="utf-8")


def validate_rows(rows: list[dict[str, Any]], dataset_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    sequence_positions: dict[str, list[int]] = {}
    for row in rows:
        sample_id = row.get("id")
        if not sample_id or sample_id in seen_ids:
            errors.append(f"duplicate or missing id: {sample_id}")
        seen_ids.add(sample_id)
        path = dataset_root / str(row.get("image_path", ""))
        if not path.is_file():
            errors.append(f"{sample_id}: missing image {row.get('image_path')}")
            continue
        digest = sha256_file(path)
        if digest in seen_hashes:
            errors.append(f"{sample_id}: duplicate image hash with {seen_hashes[digest]}")
        seen_hashes[digest] = sample_id
        if row.get("status") != "pending_review":
            errors.append(f"{sample_id}: imported SANPO row must remain pending_review")
        if any(row.get(field) is not None for field in (
            "expected_risk_direction",
            "expected_distance_band",
            "expected_should_alert",
            "expected_risk_level",
            "expected_approach_state",
            "expected_approach_alert",
            "expected_time_to_alert_frames",
        )):
            errors.append(f"{sample_id}: unreviewed risk field is not null")
        source = row.get("source", {})
        for field in ("dataset", "license", "license_url", "original_url_or_id", "sha256", "redistribution_policy"):
            if not source.get(field):
                errors.append(f"{sample_id}: missing source.{field}")
        sequence_positions.setdefault(str(row.get("sequence_id")), []).append(int(row.get("frame_index", -1)))
    for sequence_id, positions in sequence_positions.items():
        if positions != list(range(len(positions))):
            errors.append(f"{sequence_id}: target frame_index must be contiguous from 0")
    return {
        "ok": not errors,
        "benchmark_ready": False,
        "reason": "BlindAssist risk and approach fields require human review before benchmark promotion.",
        "errors": errors,
        "image_count": len(rows),
        "unique_hashes": len(seen_hashes),
        "pending_review_count": sum(row.get("status") == "pending_review" for row in rows),
    }


def source_annotation_quality(frame_index: int, annotation_types: dict[str, str]) -> str:
    return str(annotation_types.get(str(frame_index), "UNKNOWN"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local BlindAssist continuous-sequence pilot from SANPO-Real.")
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--camera", choices=("camera_chest", "camera_head"), default=DEFAULT_CAMERA)
    parser.add_argument("--lens", choices=("left",), default=DEFAULT_LENS, help="SANPO segmentation is aligned to the left lens.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.output_root) if args.output_root else project_root / "test-artifacts.local" / "datasets" / f"blindassist-sanpo-pilot-{now_stamp()}"
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if (dataset_root / "manifest.jsonl").exists():
        raise SystemExit("Refusing to rebuild a dataset root that already contains canonical manifest.jsonl")
    if not math.isclose(args.target_fps, DEFAULT_TARGET_FPS, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit("Current BlindAssist benchmark uses a fixed 100ms time step; --target-fps must be exactly 10")
    for relative in ("images/test", "source_masks/test", "qa/boxed"):
        (dataset_root / relative).mkdir(parents=True, exist_ok=True)

    session_prefix = f"{GCS_PREFIX}/sanpo-real/{args.session_id}"
    description_name = f"{session_prefix}/description.json"
    labelmap_name = f"{GCS_PREFIX}/labelmap.json"
    annotation_type_name = f"{session_prefix}/{args.camera}/{args.lens}/frame_segmentation_annotation_type.json"
    description_object = get_gcs_object(description_name, args.retries)
    labelmap_object = get_gcs_object(labelmap_name, args.retries)
    annotation_type_object = get_gcs_object(annotation_type_name, args.retries)
    description_url = media_url(description_name, description_object.get("generation"))
    labelmap_url = media_url(labelmap_name, labelmap_object.get("generation"))
    annotation_type_url = media_url(annotation_type_name, annotation_type_object.get("generation"))
    description = fetch_json(description_url, args.retries)
    labelmap = fetch_json(labelmap_url, args.retries)
    annotation_types = fetch_json(annotation_type_url, args.retries)
    label_by_id = {int(value): key for key, value in labelmap.items()}

    camera_locations = list(description.get("session_camera_location", []))
    if args.camera not in camera_locations:
        raise SystemExit(f"Session {args.session_id} does not provide {args.camera}: {camera_locations}")
    camera_details = description.get("session_camera_details", [])
    camera_index = camera_locations.index(args.camera)
    source_fps = float(camera_details[camera_index]["fps"])
    dimensions = camera_details[camera_index][f"{args.lens}_camera_params"]

    official_split = None
    split_inventory: list[dict[str, Any]] = []
    for split_name in ("train", "test"):
        object_name = f"{GCS_PREFIX}/sanpo-real/splits/{split_name}_session_ids.txt"
        split_object = get_gcs_object(object_name, args.retries)
        split_inventory.append(object_inventory(split_object))
        session_ids = {
            line.strip()
            for line in fetch_text(media_url(object_name, split_object.get("generation")), args.retries).splitlines()
            if line.strip()
        }
        if args.session_id in session_ids:
            official_split = split_name
    if official_split is None:
        raise SystemExit(f"Session {args.session_id} is absent from official SANPO train/test split files")

    frame_prefix = f"{session_prefix}/{args.camera}/{args.lens}/video_frames/"
    mask_prefix = f"{session_prefix}/{args.camera}/{args.lens}/segmentation_masks/"
    frame_objects = list_gcs_objects(frame_prefix, args.retries)
    mask_objects = list_gcs_objects(mask_prefix, args.retries)
    frames = {frame_number(item["name"]): item for item in frame_objects if item["name"].endswith(".png")}
    masks = {frame_number(item["name"]): item for item in mask_objects if item["name"].endswith(".png")}
    selected = resample_indices(sorted(set(frames) & set(masks)), source_fps, args.target_fps, args.start_frame, args.max_frames)
    if len(selected) < args.max_frames:
        raise SystemExit(f"Only {len(selected)} aligned RGB/mask frames available; requested {args.max_frames}")

    sequence_id = f"sanpo_{args.session_id}_{args.camera}_{args.lens}_{args.start_frame:06d}_{int(args.target_fps)}fps"
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    region_counts: Counter[str] = Counter()
    for target_index, source_index in enumerate(selected):
        sample_id = f"{sequence_id}_{target_index:06d}"
        image_rel = Path("images/test") / f"{sample_id}.png"
        mask_rel = Path("source_masks/test") / f"{sample_id}.png"
        image_path = dataset_root / image_rel
        mask_path = dataset_root / mask_rel
        download(media_url(frames[source_index]["name"], frames[source_index].get("generation")), image_path, args.retries)
        download(media_url(masks[source_index]["name"], masks[source_index].get("generation")), mask_path, args.retries)
        verify_gcs_md5(image_path, frames[source_index])
        verify_gcs_md5(mask_path, masks[source_index])
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        with Image.open(mask_path) as mask:
            if mask.size != (actual_width, actual_height):
                raise ValueError(f"RGB/mask dimension mismatch for source frame {source_index}")
        if (actual_width, actual_height) != (int(dimensions["image_width"]), int(dimensions["image_height"])):
            raise ValueError(f"Description/image dimension mismatch for source frame {source_index}")
        regions, mapped_objects = parse_mask_regions(mask_path, label_by_id)
        annotation_quality = source_annotation_quality(source_index, annotation_types)
        objects = mapped_objects if annotation_quality == "HUMAN_ANNOTATED" else []
        region_counts.update(region["class"] for region in regions)
        row = {
            "id": sample_id,
            "image_path": image_rel.as_posix(),
            "split": "test",
            "width": actual_width,
            "height": actual_height,
            "labels": sorted({item["class"] for item in objects}),
            "objects": objects,
            "source_regions": regions,
            "sequence_id": sequence_id,
            "frame_index": target_index,
            "frame_timestamp_ms": int(round(target_index * 1000 / args.target_fps)),
            "source_frame_index": source_index,
            "source_timestamp_ms": int(round(source_index * 1000 / source_fps)),
            "source_annotation_quality": annotation_quality,
            "source_mapped_objects": mapped_objects,
            "assist_scenario": "OUTDOOR_SLOW",
            "expected_risk_direction": None,
            "expected_distance_band": None,
            "expected_should_alert": None,
            "expected_risk_level": None,
            "expected_approach_state": None,
            "expected_approach_alert": None,
            "expected_time_to_alert_frames": None,
            "primary_object_id": None,
            "review_status": "pending_model_review",
            "status": "pending_review",
            "source": {
                "dataset": "SANPO-Real v0",
                "official_split": official_split,
                "session_id": args.session_id,
                "camera": args.camera,
                "lens": args.lens,
                "license": LICENSE_NAME,
                "license_url": LICENSE_URL,
                "dataset_page": DATASET_PAGE,
                "original_url_or_id": media_url(frames[source_index]["name"], frames[source_index].get("generation")),
                "original_object_name": frames[source_index]["name"],
                "original_md5_base64": frames[source_index].get("md5Hash"),
                "original_generation": frames[source_index].get("generation"),
                "original_metageneration": frames[source_index].get("metageneration"),
                "original_size": int(frames[source_index]["size"]),
                "original_crc32c_base64": frames[source_index].get("crc32c"),
                "sha256": sha256_file(image_path),
                "mask_original_url": media_url(masks[source_index]["name"], masks[source_index].get("generation")),
                "mask_original_md5_base64": masks[source_index].get("md5Hash"),
                "mask_original_generation": masks[source_index].get("generation"),
                "mask_original_size": int(masks[source_index]["size"]),
                "mask_original_crc32c_base64": masks[source_index].get("crc32c"),
                "mask_sha256": sha256_file(mask_path),
                "redistribution_policy": "local_internal_eval_only_do_not_commit_original_frames",
                "consent_or_release_id": None,
                "privacy_note": "SANPO states that faces and license plates are blurred; reviewer must still flag residual PII.",
            },
        }
        rows.append(row)
        review_rows.append({
            "id": sample_id,
            "sequence_id": sequence_id,
            "frame_index": target_index,
            "source_frame_index": source_index,
            "source_annotation_quality": row["source_annotation_quality"],
            "primary_object_id": "",
            "source_primary_region_id": "",
            "expected_risk_direction": "",
            "expected_distance_band": "",
            "expected_should_alert": "",
            "expected_risk_level": "",
            "expected_approach_state": "",
            "expected_approach_alert": "",
            "expected_time_to_alert_frames": "",
            "review_status": "pending_model_review",
            "objects_review_status": "pending" if objects else "not_applicable",
            "reviewer_type": "model",
            "reviewer_id": "",
            "reviewer_model_version": "",
            "review_confidence": "",
            "independent_review_count": "",
            "issue_tags": "",
            "review_notes": "",
        })

    spec = {
        "name": "blindassist_sanpo_continuous_sequence_pilot",
        "task": "continuous-sequence assistive-risk evaluation",
        "source_type": "real public dataset; no synthetic generation",
        "source": {
            "dataset": "SANPO-Real v0",
            "dataset_page": DATASET_PAGE,
            "repository": DATASET_REPO,
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "session_id": args.session_id,
            "official_split": official_split,
            "camera": args.camera,
            "lens": args.lens,
        },
        "sampling": {
            "source_fps": source_fps,
            "target_fps": args.target_fps,
            "start_frame": args.start_frame,
            "selected_source_frames": selected,
            "frame_count": len(selected),
        },
        "source_inventory": {
            "description": object_inventory(description_object),
            "labelmap": object_inventory(labelmap_object),
            "annotation_types": object_inventory(annotation_type_object),
            "official_split_files": split_inventory,
            "selected_rgb": [object_inventory(frames[index]) for index in selected],
            "selected_masks": [object_inventory(masks[index]) for index in selected],
        },
        "output_resolution": f"{dimensions['image_width']}x{dimensions['image_height']}",
        "annotation_target": "candidate manifest + manual BlindAssist risk review; benchmark promotion only after all required fields are accepted",
        "risk_fields": [
            "expected_risk_direction",
            "expected_distance_band",
            "expected_should_alert",
            "expected_risk_level",
            "expected_approach_state",
            "expected_approach_alert",
            "expected_time_to_alert_frames",
        ],
        "negative_cases": ["no supported COCO target but path may still be blocked"],
        "review_issue_tags": REVIEW_ISSUE_TAGS,
        "privacy_policy": "Original frames and masks remain in ignored test-artifacts.local and are not committed to Git.",
    }
    write_json(dataset_root / "dataset_spec.json", spec)
    write_json(dataset_root / "source_session_description.json", description)
    write_json(dataset_root / "source_labelmap.json", labelmap)
    write_json(dataset_root / "source_annotation_types.json", annotation_types)
    (dataset_root / "manifest.draft.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (dataset_root / "source_licenses.md").write_text(
        f"""# SANPO source and license\n\n- Dataset: SANPO-Real v0\n- Dataset page: {DATASET_PAGE}\n- Repository: {DATASET_REPO}\n- License: [{LICENSE_NAME}]({LICENSE_URL})\n- Session: `{args.session_id}` / `{args.camera}` / `{args.lens}` / official split `{official_split}`\n- Local policy: original frames and masks are local-only evaluation artifacts and must not be committed to Git.\n- Attribution: {SANPO_CITATION}. Dataset page: {DATASET_PAGE}\n- Changes made by BlindAssist: selected one official session; used `{args.camera}/{args.lens}`; resampled {source_fps:g} FPS to {args.target_fps:g} FPS; converted SANPO masks to review regions; mapped only exact `pedestrian -> person` and `traffic light -> traffic light`; added project-specific draft risk fields and QA overlays.\n- Privacy: SANPO reports face and license-plate blurring; BlindAssist review must still reject or re-blur residual personally identifiable information.\n""",
        encoding="utf-8",
    )
    with (dataset_root / "qa" / "model_review_checklist.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)
    build_preview(dataset_root, rows)
    validation = validate_rows(rows, dataset_root)
    validation["source_region_frame_counts"] = dict(region_counts.most_common())
    validation["created_at"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    write_json(dataset_root / "qa" / "manifest_validation.json", validation)
    write_json(dataset_root / "qa" / "download_inventory.json", {
        "rgb_bytes": sum((dataset_root / row["image_path"]).stat().st_size for row in rows),
        "mask_bytes": sum((dataset_root / "source_masks/test" / f"{row['id']}.png").stat().st_size for row in rows),
        "frame_count": len(rows),
        "sequence_id": sequence_id,
    })
    print(f"dataset_root={dataset_root}")
    print(f"frames={len(rows)} source_fps={source_fps} target_fps={args.target_fps}")
    print(f"validation_ok={validation['ok']} benchmark_ready={validation['benchmark_ready']}")
    print(f"preview={dataset_root / 'qa' / 'preview.html'}")
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
