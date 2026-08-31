#!/usr/bin/env python3
"""Transfer the frozen center-scale lock to truth-proposed SceneNN doors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_real_posed_portal_transfer as scene  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-center-target-door-retrieval-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-scenenn-center-target-door-retrieval-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-center-target-door-retrieval-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
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


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def validate_predecessors(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for section_name in ("predecessor_lock", "scene_source"):
        section = protocol[section_name]
        for key, value in section.items():
            if not key.endswith("_path"):
                continue
            hash_key = key[:-5] + "_sha256"
            if hash_key not in section:
                continue
            path = HERE / value
            require(path.is_file(), f"MISSING_PREDECESSOR:{path}")
            require(sha256(path) == section[hash_key], f"PREDECESSOR_SHA256:{path.name}")
            if path.suffix == ".json":
                loaded[path.name] = load_json(path)
    return loaded


def manifest_file(
    geometry_root: Path,
    source_manifest: dict[str, Any],
    relative_path: str,
) -> Path:
    path = geometry_root / relative_path
    require(path.is_file(), f"MISSING_GEOMETRY:{path}")
    frozen = source_manifest[relative_path]
    require(path.stat().st_size == int(frozen["bytes"]), f"GEOMETRY_BYTES:{relative_path}")
    require(sha256(path) == frozen["sha256"], f"GEOMETRY_SHA256:{relative_path}")
    return path


def freeze(
    protocol_path: Path,
    geometry_root: Path,
    rgb_root: Path,
    cohort_path: Path,
) -> None:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    loaded = validate_predecessors(protocol)
    source_cohort = loaded[protocol["scene_source"]["cohort_path"]]
    receipt = loaded[protocol["scene_source"]["rgbd_receipt_path"]]
    source_manifest = source_cohort["source"]["input_manifest"]
    intrinsic_path = manifest_file(
        geometry_root, source_manifest, "payload/intrinsic/asus.ini"
    )
    intrinsic = scene.parse_intrinsic(intrinsic_path)
    k = scene.intrinsic_matrix(intrinsic)
    receipt_by_scene = {row["scene_id"]: row for row in receipt["scenes"]}
    images: dict[str, dict[str, Any]] = {}
    source_files: dict[str, dict[str, Any]] = {}

    for episode in source_cohort["episodes"]:
        episode_id = episode["episode_id"]
        require(episode_id in protocol["frozen_cohort"]["episodes"], f"EPISODE:{episode_id}")
        scene_id = episode["scene_id"]
        target_id = int(episode["target_door_instance_id"])
        base = f"payload/{scene_id}"
        ply_path = manifest_file(geometry_root, source_manifest, f"{base}/{scene_id}.ply")
        xml_path = manifest_file(geometry_root, source_manifest, f"{base}/{scene_id}.xml")
        trajectory_path = manifest_file(geometry_root, source_manifest, f"{base}/trajectory.log")
        for path in (ply_path, xml_path, trajectory_path):
            relative = path.relative_to(geometry_root).as_posix()
            source_files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        labels_by_id = scene.parse_xml_labels(xml_path)
        require(target_id in labels_by_id, f"TARGET_LABEL:{scene_id}")
        require(labels_by_id[target_id]["text"].casefold() == "door", f"TARGET_NOT_DOOR:{scene_id}")
        xyz, labels = scene.read_vertices(ply_path)
        target_points = xyz[labels == target_id]
        require(len(target_points) >= 3, f"TARGET_POINTS:{scene_id}")
        poses = scene.parse_poses(trajectory_path)
        source_receipt = receipt_by_scene[scene_id]
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            pixels, _ = scene.project_world(target_points, poses[frame]["camera_to_world"], k)
            envelope = scene.envelope_from_pixels(
                pixels, int(intrinsic["width"]), int(intrinsic["height"])
            )
            require(envelope is not None, f"TARGET_ENVELOPE:{scene_id}:{role}")
            _, _, stats = envelope
            image_path = rgb_root / scene_id / "image" / f"frame.{frame:04d}.png"
            require(image_path.is_file(), f"MISSING_RGB:{image_path}")
            frozen_image = source_receipt["sealed"][role]
            require(sha256(image_path) == frozen_image["image_sha256"], f"RGB_SHA256:{scene_id}:{role}")
            key = f"{scene_id}_{role}"
            images[key] = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "role": role,
                "frame": frame,
                "target_door_instance_id": target_id,
                "bbox_xyxy": stats["bbox_xyxy"],
                "envelope_pixels": stats["pixels"],
                "rgb_path": image_path.relative_to(rgb_root).as_posix(),
                "rgb_sha256": frozen_image["image_sha256"],
            }

    scene_ids = sorted({row["scene_id"] for row in images.values()})
    require(len(scene_ids) == 3 and len(images) == 6, "FROZEN_IMAGE_COUNT")
    anchors: list[dict[str, Any]] = []
    for anchor_role, candidate_role in (("reference", "query"), ("query", "reference")):
        for source_scene in scene_ids:
            anchor = f"{source_scene}_{anchor_role}"
            anchors.append(
                {
                    "anchor": anchor,
                    "positive": f"{source_scene}_{candidate_role}",
                    "candidates": [f"{candidate_scene}_{candidate_role}" for candidate_scene in scene_ids],
                }
            )
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_CONSUMED_RGBD_TRUTH_PROPOSED_DOOR_RETRIEVAL_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "source_cohort_path": protocol["scene_source"]["cohort_path"],
        "source_cohort_sha256": protocol["scene_source"]["cohort_sha256"],
        "geometry_root": str(geometry_root.resolve()),
        "rgb_root": str(rgb_root.resolve()),
        "source_files": source_files,
        "images": images,
        "anchors": anchors,
        "counts": {"physical_doors": 3, "directed_anchors": 6, "positive_pairs": 6, "negative_pairs": 12},
        "proposal_rule": protocol["frozen_cohort"]["proposal_rule"],
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(cohort_path, cohort)


def square_proposal(image: Image.Image, bbox: list[int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    require(0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height, f"BBOX:{bbox}")
    crop = image.crop((x0, y0, x1, y1))
    side = max(crop.width, crop.height)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas


def center_crop(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    size = max(1, round(min(width, height) * scale))
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def encode(
    cohort: dict[str, Any],
    rgb_root: Path,
    model_root: Path,
    crop_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(model_root, local_files_only=True).eval().to("cpu")
    keys = sorted(cohort["images"])
    scales = (1.0, 0.5, 0.25)
    inputs: list[Image.Image] = []
    crop_receipts: dict[str, dict[str, Any]] = {}
    crop_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        row = cohort["images"][key]
        image_path = rgb_root / row["rgb_path"]
        require(sha256(image_path) == row["rgb_sha256"], f"RGB_SHA256:{key}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        proposal = square_proposal(image, row["bbox_xyxy"])
        proposal_path = crop_dir / f"{key}.png"
        proposal.save(proposal_path)
        crop_receipts[key] = {
            "path": str(proposal_path.resolve()),
            "width": proposal.width,
            "height": proposal.height,
            "sha256": sha256(proposal_path),
        }
        inputs.extend(center_crop(proposal, scale) for scale in scales)
    batch = processor(images=inputs, return_tensors="pt")
    with torch.inference_mode():
        values = model(**batch).last_hidden_state[:, 0]
        values = torch.nn.functional.normalize(values, dim=1)
    array = values.cpu().numpy().astype(np.float32).reshape(len(keys), 3, -1)
    return {key: array[index] for index, key in enumerate(keys)}, crop_receipts


def pair_scores(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    matrix = left @ right.T
    baseline = float(matrix[0, 0])
    local = np.concatenate((matrix[1:, :].ravel(), matrix[:, 1:].ravel()))
    upgraded = float(0.25 * baseline + 0.75 * local.max())
    return baseline, upgraded


def retrieval(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_anchor: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_anchor.setdefault(row["anchor"], []).append(row)
    correct = 0
    reciprocal: list[float] = []
    margins: list[float] = []
    details: list[dict[str, Any]] = []
    for anchor in sorted(by_anchor):
        ranked = sorted(by_anchor[anchor], key=lambda row: (-row[key], row["candidate"]))
        positive_rank = next(index for index, row in enumerate(ranked, 1) if row["label"] == 1)
        positive_score = next(row[key] for row in ranked if row["label"] == 1)
        negative_score = max(row[key] for row in ranked if row["label"] == 0)
        margin = positive_score - negative_score
        correct += int(positive_rank == 1)
        reciprocal.append(1.0 / positive_rank)
        margins.append(margin)
        details.append(
            {
                "anchor": anchor,
                "positive_rank": positive_rank,
                "positive_margin": round(float(margin), 6),
                "ranking": [row["candidate"] for row in ranked],
            }
        )
    return {
        "anchors": len(by_anchor),
        "top1_correct": correct,
        "top1_accuracy": round(correct / len(by_anchor), 6),
        "mrr": round(float(np.mean(reciprocal)), 6),
        "mean_positive_margin": round(float(np.mean(margins)), 6),
        "minimum_positive_margin": round(float(np.min(margins)), 6),
        "details": details,
    }


def replay(
    protocol_path: Path,
    cohort_path: Path,
    rgb_root: Path,
    model_root: Path,
    crop_dir: Path,
    result_path: Path,
) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    protocol = load_json(protocol_path)
    cohort = load_json(cohort_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "COHORT_PROTOCOL_SHA256")
    embeddings, crop_receipts = encode(cohort, rgb_root, model_root, crop_dir)
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
    gate_met = (
        upgraded_retrieval["top1_correct"] == 6
        and upgraded_retrieval["minimum_positive_margin"] > 0.0
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_RGBD_PROVIDER_DISJOINT_PORTAL_INSTANCE_TRANSFER_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "model": {
            "path": str(model_root.resolve()),
            "model_safetensors_sha256": sha256(model_root / "model.safetensors"),
            "device": "CPU",
        },
        "conclusion": "L10_SCENENN_CENTER_TARGET_DOOR_RETRIEVAL_TRANSFER_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_CENTER_TARGET_DOOR_RETRIEVAL_TRANSFER_DEVELOPMENT_GATE_NOT_MET",
        "gate_met": gate_met,
        "counts": cohort["counts"],
        "metrics": {
            "baseline": {
                "auroc": round(float(roc_auc_score(labels, baseline_scores)), 6),
                "average_precision": round(float(average_precision_score(labels, baseline_scores)), 6),
                "retrieval": baseline_retrieval,
            },
            "upgraded": {
                "auroc": round(float(roc_auc_score(labels, upgraded_scores)), 6),
                "average_precision": round(float(average_precision_score(labels, upgraded_scores)), 6),
                "retrieval": upgraded_retrieval,
            },
        },
        "deltas": {
            "auroc": round(float(roc_auc_score(labels, upgraded_scores) - roc_auc_score(labels, baseline_scores)), 6),
            "top1_accuracy": round(float(upgraded_retrieval["top1_accuracy"] - baseline_retrieval["top1_accuracy"]), 6),
            "mean_positive_margin": round(float(upgraded_retrieval["mean_positive_margin"] - baseline_retrieval["mean_positive_margin"]), 6),
        },
        "crop_receipts": crop_receipts,
        "decision_gate": protocol["decision_gate"],
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(result_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--geometry-root", type=Path, required=True)
    freeze_parser.add_argument("--rgb-root", type=Path, required=True)
    freeze_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--rgb-root", type=Path, required=True)
    replay_parser.add_argument("--model-root", type=Path, required=True)
    replay_parser.add_argument("--crop-dir", type=Path, required=True)
    replay_parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.geometry_root, args.rgb_root, args.cohort)
    else:
        replay(args.protocol, args.cohort, args.rgb_root, args.model_root, args.crop_dir, args.result)


if __name__ == "__main__":
    main()
