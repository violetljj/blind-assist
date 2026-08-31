#!/usr/bin/env python3
"""Freeze and replay one center-scale target lock on 3D Street View."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
PROTOCOL_SCHEMA = "blindassist-l10-3dstreetview-center-target-lock-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3dstreetview-center-target-lock-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3dstreetview-center-target-lock-result-v1"


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


def image_id(path: Path) -> tuple[str, str]:
    stem = path.stem
    require(stem.endswith(("_i", "_p")), f"UNEXPECTED_IMAGE_NAME:{path.name}")
    return stem[:-2], stem[-1]


def decodable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def is_test_group(group: str) -> bool:
    return hashlib.sha256(group.encode("utf-8")).digest()[0] % 4 == 0


def freeze(protocol_path: Path, dataset_root: Path, cohort_path: Path) -> None:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    source = protocol["source_snapshot"]
    archive = dataset_root / "test_matching.tar"
    pairs_path = dataset_root / "verpairs.txt"
    format_path = dataset_root / "format.txt"
    require(archive.stat().st_size == int(source["archive_prefix_bytes"]), "ARCHIVE_SIZE")
    require(sha256(archive) == source["archive_prefix_sha256"], "ARCHIVE_SHA256")
    require(sha256(pairs_path) == source["verpairs_sha256"], "VERPAIRS_SHA256")
    require(sha256(format_path) == source["format_sha256"], "FORMAT_SHA256")

    raw_root = dataset_root / "raw_prefix" / "verTest" / "data"
    images: dict[str, dict[str, str]] = {}
    rejected: list[str] = []
    for path in sorted(raw_root.glob("*.jpg")):
        identifier, kind = image_id(path)
        if not decodable(path):
            rejected.append(path.name)
            continue
        require(identifier not in images, f"DUPLICATE_ID:{identifier}")
        images[identifier] = {"kind": kind, "path": path.relative_to(dataset_root).as_posix()}
    require(
        len(images) == int(source["complete_decodable_jpegs_expected"]),
        f"DECODABLE_COUNT:{len(images)}",
    )

    pairs: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for line_number, line in enumerate(pairs_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        require(len(fields) == 5, f"PAIR_FIELDS:{line_number}")
        left, right, label_text, heading_text, pitch_text = fields
        if left not in images or right not in images:
            counts["unavailable"] += 1
            continue
        left_group, right_group = left.split("-", 1)[0], right.split("-", 1)[0]
        left_test, right_test = is_test_group(left_group), is_test_group(right_group)
        if left_test != right_test:
            counts["cross_partition"] += 1
            continue
        split = "test" if left_test else "train"
        label = int(label_text)
        pair_kind = "_".join(sorted((images[left]["kind"], images[right]["kind"])))
        pair_kind = {"i_i": "image_image", "i_p": "image_patch", "p_p": "patch_patch"}[pair_kind]
        record = {
            "line": line_number,
            "left": left,
            "right": right,
            "label": label,
            "relative_heading_degrees": float(heading_text),
            "relative_pitch_degrees": float(pitch_text),
            "left_group": left_group,
            "right_group": right_group,
            "split": split,
            "pair_kind": pair_kind,
        }
        pairs.append(record)
        counts[f"{split}_{'positive' if label else 'negative'}"] += 1
        counts[f"{split}_{pair_kind}"] += 1

    require(counts["train_positive"] > 0 and counts["train_negative"] > 0, "EMPTY_TRAIN_CLASS")
    require(counts["test_positive"] > 0 and counts["test_negative"] > 0, "EMPTY_TEST_CLASS")
    used_ids = sorted({row[key] for row in pairs for key in ("left", "right")})
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "EXPLORE_DEVELOPMENT_REAL_STREETVIEW_FROZEN_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "dataset_root": str(dataset_root.resolve()),
        "source": {
            "archive_prefix_sha256": sha256(archive),
            "verpairs_sha256": sha256(pairs_path),
            "format_sha256": sha256(format_path),
            "decodable_images": len(images),
            "rejected_images": rejected,
            "used_images": len(used_ids),
        },
        "counts": dict(sorted(counts.items())),
        "images": {identifier: images[identifier] for identifier in used_ids},
        "pairs": pairs,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(cohort_path, cohort)


def center_crop(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, round(width * scale))
    crop_height = max(1, round(height * scale))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def encode(
    cohort: dict[str, Any],
    dataset_root: Path,
    model_root: Path,
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(model_root, local_files_only=True).eval().to("cpu")
    identifiers = sorted(cohort["images"])
    embeddings: list[np.ndarray] = []
    scales = (1.0, 0.5, 0.25)
    for start in range(0, len(identifiers), batch_size):
        batch_ids = identifiers[start : start + batch_size]
        crops: list[Image.Image] = []
        for identifier in batch_ids:
            path = dataset_root / cohort["images"][identifier]["path"]
            with Image.open(path) as source:
                image = source.convert("RGB")
            crops.extend(center_crop(image, scale) for scale in scales)
        inputs = processor(images=crops, return_tensors="pt")
        with torch.inference_mode():
            values = model(**inputs).last_hidden_state[:, 0]
            values = torch.nn.functional.normalize(values, dim=1)
        embeddings.append(values.cpu().numpy().astype(np.float32).reshape(len(batch_ids), 3, -1))
    return identifiers, np.concatenate(embeddings, axis=0)


def save_cache(path: Path, ids: list[str], embeddings: np.ndarray, identity: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez(handle, ids=np.asarray(ids), embeddings=embeddings, identity=np.asarray(identity))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def score_pair(
    left: np.ndarray,
    right: np.ndarray,
    kind: str,
    left_kind: str,
    right_kind: str,
) -> tuple[float, float]:
    matrix = left @ right.T
    baseline = float(matrix[0, 0])
    if kind == "patch_patch":
        upgraded = baseline
    elif kind == "image_patch":
        if left_kind == "i" and right_kind == "p":
            upgraded = float(matrix[:, 0].max())
        elif left_kind == "p" and right_kind == "i":
            upgraded = float(matrix[0, :].max())
        else:
            raise ValueError(f"PAIR_KIND_MISMATCH:{left_kind}:{right_kind}")
    else:
        local = np.concatenate((matrix[1:, :].ravel(), matrix[:, 1:].ravel()))
        upgraded = float(0.25 * baseline + 0.75 * local.max())
    return baseline, upgraded


def threshold_from_train(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    index = int(np.argmax((true_positive + (1.0 - false_positive)) / 2.0))
    return float(thresholds[index])


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    predicted = scores >= threshold
    return {
        "auroc": round(float(roc_auc_score(labels, scores)), 6),
        "average_precision": round(float(average_precision_score(labels, scores)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predicted)), 6),
        "threshold_from_train": round(float(threshold), 6),
    }


def retrieval_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    anchors: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        anchors[row["left"]].append(row)
    eligible = [values for values in anchors.values() if any(v["label"] for v in values) and any(not v["label"] for v in values)]
    top1 = 0
    reciprocal_ranks: list[float] = []
    for values in eligible:
        ranked = sorted(values, key=lambda row: (-row[score_key], row["line"]))
        top1 += int(ranked[0]["label"] == 1)
        first_positive = next(index for index, row in enumerate(ranked, 1) if row["label"] == 1)
        reciprocal_ranks.append(1.0 / first_positive)
    return {
        "eligible_anchors": len(eligible),
        "top1_accuracy": round(top1 / len(eligible), 6) if eligible else None,
        "mrr": round(float(np.mean(reciprocal_ranks)), 6) if eligible else None,
    }


def replay(
    protocol_path: Path,
    dataset_root: Path,
    cohort_path: Path,
    model_root: Path,
    result_path: Path,
    cache_path: Path,
    batch_size: int,
) -> None:
    protocol = load_json(protocol_path)
    cohort = load_json(cohort_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "COHORT_PROTOCOL_SHA256")
    model_hash = sha256(model_root / "model.safetensors")
    cache_identity = hashlib.sha256(
        (sha256(cohort_path) + model_hash + "|1.0,0.5,0.25").encode("utf-8")
    ).hexdigest()
    if cache_path.is_file():
        cached = np.load(cache_path)
        require(str(cached["identity"]) == cache_identity, "CACHE_IDENTITY")
        identifiers = [str(value) for value in cached["ids"]]
        embeddings = cached["embeddings"]
    else:
        identifiers, embeddings = encode(cohort, dataset_root, model_root, batch_size)
        save_cache(cache_path, identifiers, embeddings, cache_identity)
    lookup = {identifier: embeddings[index] for index, identifier in enumerate(identifiers)}

    scored: list[dict[str, Any]] = []
    for pair in cohort["pairs"]:
        baseline, upgraded = score_pair(
            lookup[pair["left"]],
            lookup[pair["right"]],
            pair["pair_kind"],
            cohort["images"][pair["left"]]["kind"],
            cohort["images"][pair["right"]]["kind"],
        )
        scored.append({**pair, "baseline": baseline, "upgraded": upgraded})
    train = [row for row in scored if row["split"] == "train"]
    test = [row for row in scored if row["split"] == "test"]
    train_labels = np.asarray([row["label"] for row in train])
    test_labels = np.asarray([row["label"] for row in test])
    metrics: dict[str, Any] = {}
    for key in ("baseline", "upgraded"):
        train_scores = np.asarray([row[key] for row in train])
        test_scores = np.asarray([row[key] for row in test])
        threshold = threshold_from_train(train_labels, train_scores)
        metrics[key] = {
            "train": classification_metrics(train_labels, train_scores, threshold),
            "test": classification_metrics(test_labels, test_scores, threshold),
            "test_retrieval": retrieval_metrics(test, key),
        }
    subgroups: dict[str, Any] = {}
    for kind in ("image_image", "image_patch", "patch_patch"):
        rows = [row for row in test if row["pair_kind"] == kind]
        labels = np.asarray([row["label"] for row in rows])
        entry: dict[str, Any] = {"pairs": len(rows), "positive": int(labels.sum())}
        if len(set(labels.tolist())) == 2:
            from sklearn.metrics import roc_auc_score

            entry["baseline_auroc"] = round(float(roc_auc_score(labels, [r["baseline"] for r in rows])), 6)
            entry["upgraded_auroc"] = round(float(roc_auc_score(labels, [r["upgraded"] for r in rows])), 6)
        subgroups[kind] = entry

    baseline_auc = metrics["baseline"]["test"]["auroc"]
    upgraded_auc = metrics["upgraded"]["test"]["auroc"]
    baseline_top1 = metrics["baseline"]["test_retrieval"]["top1_accuracy"]
    upgraded_top1 = metrics["upgraded"]["test_retrieval"]["top1_accuracy"]
    auc_delta = upgraded_auc - baseline_auc
    top1_delta = (upgraded_top1 - baseline_top1) if baseline_top1 is not None else 0.0
    gate_met = (auc_delta >= 0.01 or top1_delta >= 0.03) and auc_delta >= -0.01 and top1_delta >= -0.01
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "EXPLORE_DEVELOPMENT_REAL_STREETVIEW_MECHANISM_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "model": {"path": str(model_root.resolve()), "model_safetensors_sha256": model_hash, "device": "CPU"},
        "conclusion": "L10_3DSTREETVIEW_CENTER_TARGET_LOCK_DEVELOPMENT_GATE_MET" if gate_met else "L10_3DSTREETVIEW_CENTER_TARGET_LOCK_DEVELOPMENT_GATE_NOT_MET",
        "gate_met": gate_met,
        "counts": cohort["counts"],
        "metrics": metrics,
        "deltas": {"test_auroc": round(auc_delta, 6), "test_retrieval_top1": round(top1_delta, 6)},
        "test_subgroups": subgroups,
        "decision_gate": protocol["evaluation"]["decision_gate"],
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(result_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--dataset-root", type=Path, required=True)
    freeze_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--dataset-root", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--model-root", type=Path, required=True)
    replay_parser.add_argument("--result", type=Path, required=True)
    replay_parser.add_argument("--cache", type=Path, required=True)
    replay_parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.dataset_root, args.cohort)
    else:
        replay(args.protocol, args.dataset_root, args.cohort, args.model_root, args.result, args.cache, args.batch_size)


if __name__ == "__main__":
    main()
