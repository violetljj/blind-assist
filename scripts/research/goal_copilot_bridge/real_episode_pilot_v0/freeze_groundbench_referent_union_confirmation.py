"""Freeze the next untouched GroundBench referent cohort after the consumed first 89."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.freeze_groundbench_referent_89 import (
    BENCHMARK_SHA256,
    COCO_IMAGE_BASE,
    ELIGIBLE_CATEGORY_GROUPS,
    IMAGE_MANIFEST_SHA256,
    SELECTION_SALT,
    SOURCE_REVISION,
    FreezeError,
    atomic_json,
    expression_from_question,
    polygon_bbox,
    sha256_file,
)


SCHEMA_VERSION = "groundbench_referent_union_confirmation_roster_v0"
CONSUMED_COUNT = 89
CONFIRMATION_COUNT = 64


def ordered_eligible(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
    return sorted(eligible, key=lambda item: item["selection_rank_sha256"])


def source_identity(row: Mapping[str, Any]) -> list[Any]:
    annotations = row["annotations"]
    return [annotations["dataset"], row["image"], annotations["image_id"], annotations["ann_id"]]


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = args.benchmark.resolve()
    image_manifest = args.image_manifest.resolve()
    consumed_path = args.consumed_roster.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FreezeError("union Confirmation roster already exists")
    if sha256_file(benchmark) != BENCHMARK_SHA256 or sha256_file(image_manifest) != IMAGE_MANIFEST_SHA256:
        raise FreezeError("GroundBench source identity mismatch")
    rows = [json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    ordered = ordered_eligible(rows)
    if len(ordered) != 353:
        raise FreezeError("GroundBench eligible denominator drift")
    consumed = json.loads(consumed_path.read_text(encoding="utf-8"))
    consumed_identities = [item["source_identity"] for item in consumed["observations"]]
    if consumed_identities != [source_identity(row) for row in ordered[:CONSUMED_COUNT]]:
        raise FreezeError("consumed first-89 identity does not match the frozen ordering")

    with image_manifest.open("r", encoding="utf-8", newline="") as stream:
        image_rows = {row["image"]: row for row in csv.DictReader(stream)}
    selected = ordered[CONSUMED_COUNT:CONSUMED_COUNT + CONFIRMATION_COUNT]
    observations = []
    for offset, row in enumerate(selected, start=CONSUMED_COUNT + 1):
        annotations = row["annotations"]
        image_row = image_rows[row["image"]]
        width, height = int(annotations["image_w"]), int(annotations["image_h"])
        target = row["conversations"][1]["value"]
        observations.append({
            "observation_id": f"groundbench-ref-{offset:03d}",
            "source_identity": source_identity(row),
            "source_dataset": annotations["dataset"], "source_split": annotations["split"],
            "source_revision": SOURCE_REVISION, "source_image_path": row["image"],
            "source_image_url": f"{COCO_IMAGE_BASE}/{row['image']}",
            "rgb_path": f"pixels/{Path(row['image']).name}",
            "rgb_sha256": image_row["sha256"].lower(), "rgb_bytes": int(image_row["bytes"]),
            "image_width": width, "image_height": height,
            "goal_text": expression_from_question(row["conversations"][0]["value"]),
            "native_mask_bbox_xyxy": polygon_bbox(target, width, height),
            "native_exact_64_polygon_xy": [float(value) for value in target],
            "category_id": int(annotations["category_id"]), "category_name": annotations["category_name"],
            "category_group": annotations["category_group"],
            "same_class_distractors": int(annotations["same_class_distractors"]),
            "selection_rank_sha256": row["selection_rank_sha256"],
        })
    reserve = [source_identity(row) for row in ordered[CONSUMED_COUNT + CONFIRMATION_COUNT:]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": SOURCE_REVISION,
        "selection_salt": SELECTION_SALT,
        "consumed_roster_sha256": sha256_file(consumed_path),
        "eligible_count": len(ordered), "consumed_count": CONSUMED_COUNT,
        "confirmation_count": len(observations), "reserve_count": len(reserve),
        "dataset_root": str(output.parent), "truth_authority": "PUBLIC_DATASET_DERIVED_GT_STRONG",
        "provider_calls": 0, "teacher_calls": 0, "pixels_downloaded_at_freeze": 0,
        "observations": observations, "reserve_source_identities": reserve,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return {"output": str(output), "sha256": sha256_file(output), "confirmation": len(observations), "reserve": len(reserve)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--consumed-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
