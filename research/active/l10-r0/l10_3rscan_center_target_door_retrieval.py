#!/usr/bin/env python3
"""Freeze and replay truth-proposed stable-door retrieval on 3RScan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_V1 = "blindassist-l10-3rscan-center-target-door-retrieval-protocol-v1"
PROTOCOL_V2 = "blindassist-l10-3rscan-center-target-door-retrieval-protocol-v2"
COHORT_SCHEMAS = {
    PROTOCOL_V1: "blindassist-l10-3rscan-center-target-door-retrieval-cohort-v1",
    PROTOCOL_V2: "blindassist-l10-3rscan-center-target-door-retrieval-cohort-v2",
}
RESULT_SCHEMAS = {
    PROTOCOL_V1: "blindassist-l10-3rscan-center-target-door-retrieval-result-v1",
    PROTOCOL_V2: "blindassist-l10-3rscan-center-target-door-retrieval-result-v2",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_path(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"MISSING_{label}:{path}")
    require(sha256(path) == expected, f"SHA256_{label}:{path.name}")


def resolve_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = load_json(path)
    schema = raw.get("schema")
    require(schema in COHORT_SCHEMAS, "PROTOCOL_SCHEMA_MISMATCH")
    if schema == PROTOCOL_V1:
        return raw, raw
    base_path = HERE / raw["base_protocol_path"]
    verify_path(base_path, raw["base_protocol_sha256"], "BASE_PROTOCOL")
    base = load_json(base_path)
    require(base.get("schema") == PROTOCOL_V1, "BASE_PROTOCOL_SCHEMA")
    resolved = merge_dict(base, raw["overrides"])
    resolved["schema"] = schema
    return raw, resolved


def source_record(path: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(artifact_root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def consumed_triples(protocol: dict[str, Any]) -> tuple[set[tuple[str, str, int]], list[dict[str, Any]]]:
    triples: set[tuple[str, str, int]] = set()
    receipts: list[dict[str, Any]] = []
    for item in protocol["source"]["consumed_target_cohorts"]:
        path = HERE / item["path"]
        verify_path(path, item["sha256"], "CONSUMED_COHORT")
        cohort = load_json(path)
        found = 0
        for episode in cohort.get("episodes", []):
            if not all(key in episode for key in ("reference_scan_id", "rescan_id", "target_instance_id")):
                continue
            triples.add(
                (
                    str(episode["reference_scan_id"]),
                    str(episode["rescan_id"]),
                    int(episode["target_instance_id"]),
                )
            )
            found += 1
        receipts.append({"path": path.name, "sha256": item["sha256"], "target_triples": found})
    return triples, receipts


def verify_protocol_dependencies(protocol: dict[str, Any], artifact_root: Path) -> tuple[Path, Path]:
    for key, expected in (
        ("protocol", protocol["predecessor_lock"]["protocol_sha256"]),
        ("cohort", protocol["predecessor_lock"]["cohort_sha256"]),
        ("result", protocol["predecessor_lock"]["result_sha256"]),
        ("implementation", protocol["predecessor_lock"]["implementation_sha256"]),
    ):
        verify_path(HERE / protocol["predecessor_lock"][f"{key}_path"], expected, "PREDECESSOR")
    candidate_protocol = HERE / protocol["source"]["candidate_protocol_path"]
    verify_path(candidate_protocol, protocol["source"]["candidate_protocol_sha256"], "CANDIDATE_PROTOCOL")
    data_root = artifact_root / protocol["source"]["dataset_relative_path"]
    model_root = artifact_root / protocol["frozen_model"]["model_relative_path"]
    require(data_root.is_dir(), f"DATA_ROOT:{data_root}")
    require(model_root.is_dir(), f"MODEL_ROOT:{model_root}")
    verify_path(model_root / "model.safetensors", protocol["frozen_model"]["weights_sha256"], "MODEL")
    return data_root, model_root


def freeze(protocol_path: Path, artifact_root: Path, cohort_path: Path) -> None:
    raw_protocol, protocol = resolve_protocol(protocol_path)
    data_root, _ = verify_protocol_dependencies(protocol, artifact_root)
    excluded, exclusion_receipts = consumed_triples(protocol)
    candidate_protocol = load_json(HERE / protocol["source"]["candidate_protocol_path"])
    rows = extent.candidate_rows(candidate_protocol, data_root, require_geometry=True)
    rules = protocol["frozen_cohort"]["frame_rules"]
    cache: dict[tuple[str, int], tuple[dict[str, Any] | None, dict[str, int]]] = {}
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0}
    selected: list[dict[str, Any]] = []
    used_physical_targets: set[tuple[str, int]] = set()
    reference_counts: dict[str, int] = {}
    considered = 0
    maximum_per_reference = int(
        protocol["frozen_cohort"].get("maximum_targets_per_reference_scan", 1)
    )

    def selected_frame(scan_id: str, target_id: int) -> dict[str, Any] | None:
        key = (scan_id, target_id)
        if key not in cache:
            cache[key] = pixel.select_frame(data_root, scan_id, target_id, rules)
            for name, count in cache[key][1].items():
                opened[name] += count
        return cache[key][0]

    for row in rows:
        triple = (
            str(row["reference_scan_id"]),
            str(row["rescan_id"]),
            int(row["target_instance_id"]),
        )
        physical_target = (triple[0], triple[2])
        if (
            triple in excluded
            or physical_target in used_physical_targets
            or reference_counts.get(triple[0], 0) >= maximum_per_reference
        ):
            continue
        if not all((data_root / scan / "sequence.zip").is_file() for scan in triple[:2]):
            continue
        considered += 1
        reference = selected_frame(triple[0], triple[2])
        query = selected_frame(triple[1], triple[2])
        if reference is None or query is None:
            continue
        selected.append(
            {
                "episode_id": f"DR{len(selected) + 1:02d}",
                **row,
                "reference": reference,
                "query": query,
            }
        )
        used_physical_targets.add(physical_target)
        reference_counts[triple[0]] = reference_counts.get(triple[0], 0) + 1
        if len(selected) == int(protocol["frozen_cohort"]["physical_targets"]):
            break

    require(opened["rgb_members"] == 0, "RGB_OPENED_BEFORE_FREEZE")
    require(
        len(selected) == int(protocol["frozen_cohort"]["physical_targets"]),
        f"DOOR_RETRIEVAL_COHORT_NOT_EVALUABLE:{len(selected)}",
    )
    source_manifest: dict[str, dict[str, Any]] = {}
    for episode in selected:
        for scan_id in (episode["reference_scan_id"], episode["rescan_id"]):
            for name in ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip"):
                path = data_root / scan_id / name
                source_manifest[f"{scan_id}/{name}"] = source_record(path, artifact_root)
    for name in ("3RScan.json", "objects.json"):
        source_manifest[name] = source_record(data_root / name, artifact_root)

    images: dict[str, dict[str, Any]] = {}
    for episode in selected:
        for role, scan_key in (("reference", "reference_scan_id"), ("query", "rescan_id")):
            key = f"{episode['episode_id']}_{role}"
            frame = episode[role]
            images[key] = {
                "episode_id": episode["episode_id"],
                "role": role,
                "scan_id": episode[scan_key],
                "target_instance_id": episode["target_instance_id"],
                "target_label": episode["target_label"],
                "frame": frame["frame"],
                "color_size": frame["color_size"],
                "bbox_xyxy": frame["bbox_xyxy"],
                "image_margin_pixels": frame["image_margin_pixels"],
                "inside_vertex_fraction": frame["inside_vertex_fraction"],
                "depth_visible_ratio": frame["depth_visible_ratio"],
                "zip_member": f"frame-{int(frame['frame']):06d}.color.jpg",
            }
    anchors: list[dict[str, Any]] = []
    episode_ids = [episode["episode_id"] for episode in selected]
    for anchor_role, candidate_role in (("reference", "query"), ("query", "reference")):
        for episode_id in episode_ids:
            anchors.append(
                {
                    "anchor": f"{episode_id}_{anchor_role}",
                    "positive": f"{episode_id}_{candidate_role}",
                    "candidates": [f"{candidate}_{candidate_role}" for candidate in episode_ids],
                }
            )
    cohort = {
        "schema": COHORT_SCHEMAS[raw_protocol["schema"]],
        "authority": "FROZEN_PRE_RGB_TARGET_TRIPLE_DISJOINT_3RSCAN_DOOR_RETRIEVAL_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "entrypoint_sha256": sha256(Path(__file__).resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "selection": {
            "candidate_rows": len(rows),
            "consumed_target_triples": len(excluded),
            "candidate_rows_considered": considered,
            "unique_physical_target_rule": True,
            "maximum_targets_per_reference_scan": maximum_per_reference,
            "reference_scan_target_counts": dict(sorted(reference_counts.items())),
            "opened_members": opened,
            "frame_rules": rules,
            "exclusion_receipts": exclusion_receipts,
        },
        "source_manifest": dict(sorted(source_manifest.items())),
        "episodes": selected,
        "images": images,
        "anchors": anchors,
        "counts": {
            "physical_targets": 3,
            "directed_anchors": 6,
            "positive_pairs": 6,
            "negative_pairs": 12,
        },
        "proposal_rule": protocol["frozen_cohort"]["proposal_rule"],
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(cohort_path, cohort)


def square_proposal(image: Image.Image, bbox: list[float]) -> Image.Image:
    x0 = max(0, int(math.floor(bbox[0])))
    y0 = max(0, int(math.floor(bbox[1])))
    x1 = min(image.width, int(math.ceil(bbox[2])) + 1)
    y1 = min(image.height, int(math.ceil(bbox[3])) + 1)
    require(0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height, f"BBOX:{bbox}")
    crop = image.crop((x0, y0, x1, y1))
    side = max(crop.width, crop.height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas


def center_crop(image: Image.Image, scale: float) -> Image.Image:
    size = max(1, round(min(image.size) * scale))
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def encode(
    protocol: dict[str, Any],
    cohort: dict[str, Any],
    artifact_root: Path,
    crop_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    data_root = artifact_root / protocol["source"]["dataset_relative_path"]
    model_root = artifact_root / protocol["frozen_model"]["model_relative_path"]
    verify_path(model_root / "model.safetensors", protocol["frozen_model"]["weights_sha256"], "MODEL")
    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(model_root, local_files_only=True).eval().to("cpu")
    keys = sorted(cohort["images"])
    crops: list[Image.Image] = []
    receipts: dict[str, dict[str, Any]] = {}
    crop_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        row = cohort["images"][key]
        zip_path = data_root / row["scan_id"] / "sequence.zip"
        manifest = cohort["source_manifest"][f"{row['scan_id']}/sequence.zip"]
        require(zip_path.stat().st_size == int(manifest["bytes"]), f"ZIP_BYTES:{key}")
        require(sha256(zip_path) == manifest["sha256"], f"ZIP_SHA256:{key}")
        with zipfile.ZipFile(zip_path) as archive:
            payload = archive.read(row["zip_member"])
        image_sha = hashlib.sha256(payload).hexdigest()
        with Image.open(__import__("io").BytesIO(payload)) as source:
            image = source.convert("RGB")
        require(list(image.size) == row["color_size"], f"RGB_SIZE:{key}:{image.size}")
        proposal = square_proposal(image, row["bbox_xyxy"])
        crop_path = crop_dir / f"{key}.png"
        proposal.save(crop_path)
        receipts[key] = {
            "zip_member": row["zip_member"],
            "rgb_sha256": image_sha,
            "crop_path": str(crop_path.resolve()),
            "crop_sha256": sha256(crop_path),
            "crop_size": list(proposal.size),
        }
        crops.extend(center_crop(proposal, scale) for scale in (1.0, 0.5, 0.25))
    batch = processor(images=crops, return_tensors="pt")
    with torch.inference_mode():
        values = model(**batch).last_hidden_state[:, 0]
        values = torch.nn.functional.normalize(values, dim=1)
    array = values.cpu().numpy().astype(np.float32).reshape(len(keys), 3, -1)
    return {key: array[index] for index, key in enumerate(keys)}, receipts


def pair_scores(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    matrix = left @ right.T
    baseline = float(matrix[0, 0])
    local = np.concatenate((matrix[1:, :].ravel(), matrix[:, 1:].ravel()))
    return baseline, float(0.25 * baseline + 0.75 * local.max())


def retrieval(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    anchors = sorted({row["anchor"] for row in rows})
    correct = 0
    reciprocal: list[float] = []
    margins: list[float] = []
    details: list[dict[str, Any]] = []
    for anchor in anchors:
        ranked = sorted(
            (row for row in rows if row["anchor"] == anchor),
            key=lambda row: (-row[score_key], row["candidate"]),
        )
        rank = next(index for index, row in enumerate(ranked, 1) if row["label"] == 1)
        positive = next(row[score_key] for row in ranked if row["label"] == 1)
        negative = max(row[score_key] for row in ranked if row["label"] == 0)
        margin = float(positive - negative)
        correct += int(rank == 1)
        reciprocal.append(1.0 / rank)
        margins.append(margin)
        details.append(
            {
                "anchor": anchor,
                "positive_rank": rank,
                "positive_margin": round(margin, 6),
                "ranking": [row["candidate"] for row in ranked],
            }
        )
    return {
        "anchors": len(anchors),
        "top1_correct": correct,
        "top1_accuracy": round(correct / len(anchors), 6),
        "mrr": round(float(np.mean(reciprocal)), 6),
        "mean_positive_margin": round(float(np.mean(margins)), 6),
        "minimum_positive_margin": round(float(np.min(margins)), 6),
        "details": details,
    }


def replay(
    protocol_path: Path,
    cohort_path: Path,
    artifact_root: Path,
    crop_dir: Path,
    result_path: Path,
) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    raw_protocol, protocol = resolve_protocol(protocol_path)
    cohort = load_json(cohort_path)
    require(
        cohort.get("schema") == COHORT_SCHEMAS[raw_protocol["schema"]],
        "COHORT_SCHEMA_MISMATCH",
    )
    require(cohort["protocol_sha256"] == sha256(protocol_path), "COHORT_PROTOCOL_SHA256")
    require(cohort["entrypoint_sha256"] == sha256(Path(__file__).resolve()), "COHORT_ENTRYPOINT_SHA256")
    verify_protocol_dependencies(protocol, artifact_root)
    embeddings, rgb_receipts = encode(protocol, cohort, artifact_root, crop_dir)
    rows: list[dict[str, Any]] = []
    for group in cohort["anchors"]:
        for candidate in group["candidates"]:
            baseline, upgraded = pair_scores(embeddings[group["anchor"]], embeddings[candidate])
            rows.append(
                {
                    "anchor": group["anchor"],
                    "candidate": candidate,
                    "label": int(candidate == group["positive"]),
                    "baseline": baseline,
                    "upgraded": upgraded,
                }
            )
    labels = np.asarray([row["label"] for row in rows])
    baseline_scores = np.asarray([row["baseline"] for row in rows])
    upgraded_scores = np.asarray([row["upgraded"] for row in rows])
    baseline_retrieval = retrieval(rows, "baseline")
    upgraded_retrieval = retrieval(rows, "upgraded")
    baseline_auroc = float(roc_auc_score(labels, baseline_scores))
    upgraded_auroc = float(roc_auc_score(labels, upgraded_scores))
    gate_met = (
        upgraded_retrieval["top1_correct"] == int(cohort["counts"]["directed_anchors"])
        and upgraded_retrieval["minimum_positive_margin"] > 0.0
    )
    result = {
        "schema": RESULT_SCHEMAS[raw_protocol["schema"]],
        "authority": "CONSUMED_TARGET_TRIPLE_DISJOINT_3RSCAN_PROVIDER_TRANSFER_DEVELOPMENT_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__).resolve())},
        "conclusion": (
            "L10_3RSCAN_CENTER_TARGET_DOOR_RETRIEVAL_TRANSFER_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_CENTER_TARGET_DOOR_RETRIEVAL_TRANSFER_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "counts": cohort["counts"],
        "metrics": {
            "baseline": {
                "auroc": round(baseline_auroc, 6),
                "average_precision": round(float(average_precision_score(labels, baseline_scores)), 6),
                "retrieval": baseline_retrieval,
            },
            "upgraded": {
                "auroc": round(upgraded_auroc, 6),
                "average_precision": round(float(average_precision_score(labels, upgraded_scores)), 6),
                "retrieval": upgraded_retrieval,
            },
        },
        "deltas": {
            "auroc": round(upgraded_auroc - baseline_auroc, 6),
            "top1_accuracy": round(
                upgraded_retrieval["top1_accuracy"] - baseline_retrieval["top1_accuracy"], 6
            ),
            "mean_positive_margin": round(
                upgraded_retrieval["mean_positive_margin"]
                - baseline_retrieval["mean_positive_margin"],
                6,
            ),
        },
        "rgb_members_opened": len(rgb_receipts),
        "rgb_receipts": rgb_receipts,
        "decision_gate": protocol["decision_gate"],
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(result_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--artifact-root", type=Path, required=True)
    freeze_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--artifact-root", type=Path, required=True)
    replay_parser.add_argument("--crop-dir", type=Path, required=True)
    replay_parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.artifact_root, args.cohort)
    else:
        replay(args.protocol, args.cohort, args.artifact_root, args.crop_dir, args.result)


if __name__ == "__main__":
    main()
