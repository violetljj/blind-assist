"""Freeze and materialize a strong-truth public referring-expression cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests


SCHEMA_VERSION = "groundbench_referent_89_roster_v0"
SELECTION_SALT = "blindassist-groundbench-outdoor-referent-89-v0"
SELECTED_COUNT = 89
SOURCE_REVISION = "010520d396f7b1775adc425e0b88fdc6fe95bb34"
BENCHMARK_SHA256 = "c111145c1ffc21a8821245755d0c7d8ef3218258d7e0d7ae2f36da8a9459ecf8"
IMAGE_MANIFEST_SHA256 = "224e8c984f8002cae96bc3bf2b9ce886ab59f4e53b18f0b8e7dd400be7ae7472"
ELIGIBLE_CATEGORY_GROUPS = frozenset({"vehicle", "outdoor/accessory"})
COCO_IMAGE_BASE = "https://images.cocodataset.org"


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


def expression_from_question(question: str) -> str:
    marker = "this sentence describes:"
    if marker not in question:
        raise FreezeError("unexpected GroundBench question template")
    expression = question.split(marker, 1)[1].strip()
    if not expression or "<image>" in expression:
        raise FreezeError("empty or malformed referring expression")
    return expression


def polygon_bbox(flat_polygon: Sequence[Any], width: int, height: int) -> list[float]:
    if len(flat_polygon) != 128:
        raise FreezeError("GroundBench 64-point target must contain 128 coordinates")
    values = [float(value) for value in flat_polygon]
    if not all(math.isfinite(value) for value in values):
        raise FreezeError("polygon contains non-finite coordinates")
    xs, ys = values[0::2], values[1::2]
    box = [min(xs), min(ys), max(xs), max(ys)]
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise FreezeError("polygon-derived bbox is outside the source image")
    return box


def select_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    eligible = []
    for raw in rows:
        annotations = raw["annotations"]
        if (
            annotations["category_group"] not in ELIGIBLE_CATEGORY_GROUPS
            or int(annotations["same_class_distractors"]) < 1
        ):
            continue
        identity = "|".join(str(value) for value in (
            annotations["dataset"], raw["image"], annotations["image_id"], annotations["ann_id"],
        ))
        eligible.append(dict(raw, selection_rank_sha256=hashlib.sha256(
            f"{SELECTION_SALT}|{identity}".encode("utf-8"),
        ).hexdigest()))
    ordered = sorted(eligible, key=lambda item: item["selection_rank_sha256"])
    return ordered[:SELECTED_COUNT], len(ordered)


def coco_transport_url(source_url: str) -> str:
    """Use COCO's working official HTTP endpoint without disabling TLS checks."""
    prefix = "https://images.cocodataset.org/"
    if not source_url.startswith(prefix):
        raise FreezeError("unexpected COCO image host")
    return "http://images.cocodataset.org/" + source_url[len(prefix):]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = args.benchmark.resolve()
    image_manifest = args.image_manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FreezeError("frozen roster already exists")
    if sha256_file(benchmark) != BENCHMARK_SHA256:
        raise FreezeError("GroundBench benchmark identity mismatch")
    if sha256_file(image_manifest) != IMAGE_MANIFEST_SHA256:
        raise FreezeError("GroundBench image manifest identity mismatch")

    rows = [json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1500:
        raise FreezeError("GroundBench row denominator drift")
    selected, eligible_count = select_rows(rows)
    if eligible_count != 353 or len(selected) != SELECTED_COUNT:
        raise FreezeError("GroundBench eligible or selected denominator drift")

    with image_manifest.open("r", encoding="utf-8", newline="") as stream:
        image_rows = {row["image"]: row for row in csv.DictReader(stream)}
    if len(image_rows) != 1500:
        raise FreezeError("GroundBench image manifest denominator drift")

    observations = []
    identities = set()
    for index, row in enumerate(selected, start=1):
        annotations = row["annotations"]
        identity = [annotations["dataset"], row["image"], annotations["image_id"], annotations["ann_id"]]
        identity_key = json.dumps(identity, separators=(",", ":"))
        if identity_key in identities or row["image"] not in image_rows:
            raise FreezeError("duplicate identity or missing image manifest row")
        identities.add(identity_key)
        image_row = image_rows[row["image"]]
        width, height = int(annotations["image_w"]), int(annotations["image_h"])
        target = row["conversations"][1]["value"]
        observations.append({
            "observation_id": f"groundbench-ref-{index:03d}",
            "source_identity": identity,
            "source_dataset": annotations["dataset"],
            "source_split": annotations["split"],
            "source_revision": SOURCE_REVISION,
            "source_image_path": row["image"],
            "source_image_url": f"{COCO_IMAGE_BASE}/{row['image']}",
            "rgb_path": f"pixels/{Path(row['image']).name}",
            "rgb_sha256": image_row["sha256"].lower(),
            "rgb_bytes": int(image_row["bytes"]),
            "image_width": width,
            "image_height": height,
            "goal_text": expression_from_question(row["conversations"][0]["value"]),
            "native_mask_bbox_xyxy": polygon_bbox(target, width, height),
            "native_exact_64_polygon_xy": [float(value) for value in target],
            "category_id": int(annotations["category_id"]),
            "category_name": annotations["category_name"],
            "category_group": annotations["category_group"],
            "same_class_distractors": int(annotations["same_class_distractors"]),
            "selection_rank_sha256": row["selection_rank_sha256"],
        })

    result = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repository": "Social-AI-2026/GroundBench",
            "revision": SOURCE_REVISION,
            "release_version": "1.0.0",
            "benchmark_sha256": BENCHMARK_SHA256,
            "image_manifest_sha256": IMAGE_MANIFEST_SHA256,
            "upstream_images": "COCO 2014 train2014",
            "upstream_referring_expressions": "RefCOCO/RefCOCO+/RefCOCOg",
        },
        "selection": {
            "salt": SELECTION_SALT,
            "eligible_category_groups": sorted(ELIGIBLE_CATEGORY_GROUPS),
            "minimum_same_class_distractors": 1,
            "eligible_count": eligible_count,
            "selected_count": len(observations),
        },
        "dataset_root": str(output.parent),
        "truth_authority": "PUBLIC_DATASET_DERIVED_GT_STRONG",
        "provider_calls": 0,
        "teacher_calls": 0,
        "pixels_downloaded_at_freeze": 0,
        "observations": observations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"output": str(output), "sha256": sha256_file(output), "eligible": eligible_count, "selected": len(observations)}


def download_pixels(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve()
    output = args.receipt.resolve()
    if output.exists():
        raise FreezeError("pixel receipt already exists")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    if roster.get("provider_calls") != 0 or roster.get("pixels_downloaded_at_freeze") != 0:
        raise FreezeError("roster is not a pre-provider metadata-only freeze")
    dataset_root = Path(roster["dataset_root"]).resolve()
    completed = []
    session = requests.Session()
    session.headers["User-Agent"] = "BlindAssist-noncommercial-research"
    for index, item in enumerate(roster["observations"], start=1):
        target = (dataset_root / item["rgb_path"]).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or sha256_file(target) != item["rgb_sha256"]:
            temporary = target.with_suffix(target.suffix + ".part")
            transport_url = coco_transport_url(item["source_image_url"])
            with session.get(transport_url, timeout=120, stream=True) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
            if temporary.stat().st_size != item["rgb_bytes"] or sha256_file(temporary) != item["rgb_sha256"]:
                raise FreezeError(f"COCO pixel identity mismatch: {item['observation_id']}")
            temporary.replace(target)
        completed.append({
            "observation_id": item["observation_id"], "rgb_path": str(target),
            "bytes": target.stat().st_size, "sha256": sha256_file(target),
            "transport_url": coco_transport_url(item["source_image_url"]),
        })
        print(f"pixels {index}/{len(roster['observations'])} {item['observation_id']}", flush=True)
    receipt = {
        "schema_version": "groundbench_referent_89_pixel_receipt_v0",
        "roster_sha256": sha256_file(roster_path),
        "downloaded_and_verified": len(completed),
        "provider_calls": 0,
        "teacher_calls": 0,
        "pixels": completed,
    }
    atomic_json(output, receipt)
    return {"receipt": str(output), "sha256": sha256_file(output), "verified": len(completed)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freezer = subparsers.add_parser("freeze")
    freezer.add_argument("--benchmark", type=Path, required=True)
    freezer.add_argument("--image-manifest", type=Path, required=True)
    freezer.add_argument("--output", type=Path, required=True)
    downloader = subparsers.add_parser("download")
    downloader.add_argument("--roster", type=Path, required=True)
    downloader.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze(args) if args.command == "freeze" else download_pixels(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
