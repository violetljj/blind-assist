from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from research_backend import (  # noqa: E402
    BackendCandidate,
    Workload,
    runtime_capabilities,
    select_backend,
    torch_observation,
)


@dataclass(frozen=True)
class ImageRow:
    key: str
    target_id: str
    role: str
    path: Path
    sha256: str
    commons_file: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_hash(rows: list[tuple[str, int, str]]) -> str:
    payload = "".join(f"{name}\t{size}\t{digest}\n" for name, size, digest in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, dim=-1)


def _load_models(clip_path: Path, dino_path: Path, device: str) -> dict[str, Any]:
    return {
        "clip": CLIPModel.from_pretrained(clip_path, local_files_only=True).eval().to(device),
        "dino": AutoModel.from_pretrained(dino_path, local_files_only=True).eval().to(device),
    }


def _forward(
    models: dict[str, Any], clip_pixels: torch.Tensor, dino_pixels: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        clip_vision = models["clip"].vision_model(pixel_values=clip_pixels.to(device))
        clip = _normalized(models["clip"].visual_projection(clip_vision.pooler_output))
        dino = models["dino"](pixel_values=dino_pixels.to(device)).last_hidden_state
    return clip, dino


def _select_backend(
    clip_path: Path,
    dino_path: Path,
    representative: tuple[torch.Tensor, torch.Tensor],
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cpu_models = _load_models(clip_path, dino_path, "cpu")
    gpu_models = _load_models(clip_path, dino_path, "cuda") if torch.cuda.is_available() else None
    clip_pixels, dino_pixels = representative
    cpu = BackendCandidate(
        "named-poi-clip-dinov2-cpu",
        "cpu",
        lambda: _forward(cpu_models, clip_pixels, dino_pixels, "cpu"),
        lambda output: torch_observation(output=output),
    )
    gpu = None
    if gpu_models is not None:
        gpu = BackendCandidate(
            "named-poi-clip-dinov2-cuda",
            "cuda",
            lambda: _forward(gpu_models, clip_pixels, dino_pixels, "cuda"),
            lambda output: torch_observation(output=output),
            torch.cuda.synchronize,
        )
    backend = select_backend(
        Workload.MODEL_INFERENCE,
        cpu=cpu,
        gpu=gpu,
        cpu_reason="ACCELERATOR_UNAVAILABLE" if gpu is None else None,
        record_path=receipt_path,
        warmups=0,
        repeats=1,
        capabilities=runtime_capabilities(),
    )
    selected = gpu_models if backend["selected_device_type"] == "cuda" else cpu_models
    if selected is None:
        raise RuntimeError("SELECTED_MODEL_MISSING")
    if selected is gpu_models:
        del cpu_models
    else:
        del gpu_models
    gc.collect()
    return backend, selected


def _validate_and_load_roles(
    protocol: dict[str, Any], library_path: Path, data_root: Path
) -> tuple[list[ImageRow], dict[str, dict[str, Any]], dict[str, Any]]:
    if _sha256(library_path) != protocol["source"]["library_sha256"]:
        raise ValueError("SOURCE_LIBRARY_HASH_MISMATCH")
    library = json.loads(library_path.read_text(encoding="utf-8"))
    all_rows: list[tuple[str, int, str]] = []
    targets = {str(item["id"]): item for item in library["targets"]}
    for target in library["targets"]:
        for reference in target["reference_images"]:
            path = data_root / "images" / target["id"] / Path(reference["local_path"]).name
            digest = _sha256(path)
            if digest != reference["sha256"]:
                raise ValueError(f"IMAGE_HASH_MISMATCH:{target['id']}:{path.name}")
            all_rows.append((path.relative_to(data_root).as_posix(), path.stat().st_size, digest))
    all_rows.sort()
    observed_source = {
        "image_count": len(all_rows),
        "image_bytes": sum(item[1] for item in all_rows),
        "image_inventory_sha256": _inventory_hash(all_rows),
    }
    for key, value in observed_source.items():
        if value != protocol["source"][key]:
            raise ValueError(
                f"SOURCE_{key.upper()}_MISMATCH:expected={protocol['source'][key]}:observed={value}"
            )

    roles: list[ImageRow] = []
    for target_id, spec in protocol["role_freeze"]["targets"].items():
        target = targets[target_id]
        seen: set[int] = set()
        for role in ("reference", "calibration", "evaluation"):
            for source_index in spec[role]:
                index = int(source_index) - 1
                if index in seen:
                    raise ValueError(f"ROLE_OVERLAP:{target_id}:{source_index}")
                seen.add(index)
                reference = target["reference_images"][index]
                path = data_root / "images" / target_id / Path(reference["local_path"]).name
                roles.append(
                    ImageRow(
                        key=f"{target_id}:{source_index:02d}",
                        target_id=target_id,
                        role=role,
                        path=path,
                        sha256=str(reference["sha256"]),
                        commons_file=str(reference["commons_file"]),
                    )
                )
    return roles, targets, observed_source


def _encode_images(
    rows: list[ImageRow],
    models: dict[str, Any],
    clip_processor: Any,
    dino_processor: Any,
    device: str,
    patch_grid: int,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    encoded: dict[str, dict[str, np.ndarray]] = {}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = [Image.open(row.path).convert("RGB") for row in batch_rows]
        clip_pixels = clip_processor(images=images, return_tensors="pt")["pixel_values"]
        dino_pixels = dino_processor(images=images, return_tensors="pt")["pixel_values"]
        clip, dino_hidden = _forward(models, clip_pixels, dino_pixels, device)
        patch_count = dino_hidden.shape[1] - 1
        side = int(round(math.sqrt(patch_count)))
        if side * side != patch_count:
            raise ValueError(f"DINO_PATCH_GRID_NOT_SQUARE:{patch_count}")
        patch_map = dino_hidden[:, 1:].reshape(len(batch_rows), side, side, -1)
        patch_map = patch_map.permute(0, 3, 1, 2)
        patch_map = F.adaptive_avg_pool2d(patch_map, (patch_grid, patch_grid))
        patches = _normalized(patch_map.permute(0, 2, 3, 1).reshape(len(batch_rows), -1, patch_map.shape[1]))
        pooled = _normalized(dino_hidden[:, 1:].mean(dim=1))
        for row, clip_value, pooled_value, patch_value in zip(
            batch_rows, clip, pooled, patches, strict=True
        ):
            encoded[row.key] = {
                "clip": clip_value.detach().cpu().numpy().astype(np.float32),
                "dino": pooled_value.detach().cpu().numpy().astype(np.float32),
                "patches": patch_value.detach().cpu().numpy().astype(np.float32),
            }
    return encoded


def _encode_names(
    target_ids: list[str], targets: dict[str, dict[str, Any]], models: dict[str, Any], processor: Any, device: str
) -> np.ndarray:
    prompts = []
    for target_id in target_ids:
        english = str(targets[target_id]["names"][-1])
        prompts.append(f"a street-level photo of {english}")
    values = processor(text=prompts, padding=True, return_tensors="pt")
    values = {key: value.to(device) for key, value in values.items() if key in {"input_ids", "attention_mask"}}
    with torch.inference_mode():
        text = models["clip"].text_model(**values).pooler_output
        text = _normalized(models["clip"].text_projection(text))
    return text.detach().cpu().numpy().astype(np.float32)


def _patch_geometry(query: np.ndarray, reference: np.ndarray, grid: int) -> dict[str, float]:
    similarity = query @ reference.T
    q_to_r = similarity.argmax(axis=1)
    r_to_q = similarity.argmax(axis=0)
    mutual = [(q, int(r)) for q, r in enumerate(q_to_r) if r_to_q[int(r)] == q]
    mutual.sort(key=lambda pair: float(similarity[pair[0], pair[1]]), reverse=True)
    mutual = mutual[:32]
    if not mutual:
        return {"score": 0.0, "matches": 0, "inlier_ratio": 0.0, "mean_similarity": 0.0}
    scores = np.asarray([similarity[q, r] for q, r in mutual], dtype=np.float32)
    mean_similarity = float(scores.mean())
    inlier_ratio = 0.0
    if len(mutual) >= 4:
        denominator = max(grid - 1, 1)
        src = np.asarray([[(r % grid) / denominator, (r // grid) / denominator] for _, r in mutual], dtype=np.float32)
        dst = np.asarray([[(q % grid) / denominator, (q // grid) / denominator] for q, _ in mutual], dtype=np.float32)
        _, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=0.12,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if inliers is not None:
            inlier_ratio = float(inliers.mean())
    coverage = min(1.0, len(mutual) / 12.0)
    score = mean_similarity * (0.5 + 0.5 * inlier_ratio) * coverage
    return {
        "score": float(score),
        "matches": len(mutual),
        "inlier_ratio": inlier_ratio,
        "mean_similarity": mean_similarity,
    }


def _score_query(
    query: ImageRow,
    target_ids: list[str],
    references: dict[str, list[ImageRow]],
    encoded: dict[str, dict[str, np.ndarray]],
    name_vectors: np.ndarray,
    patch_grid: int,
) -> dict[str, Any]:
    query_features = encoded[query.key]
    arms = {"name_only": [], "global_reference": [], "facade_fingerprint": []}
    geometry_debug: dict[str, Any] = {}
    for target_index, target_id in enumerate(target_ids):
        name_score = float(query_features["clip"] @ name_vectors[target_index])
        clip_scores = []
        dino_scores = []
        geometry_scores = []
        ref_debug = []
        for reference in references[target_id]:
            ref_features = encoded[reference.key]
            clip_score = float(query_features["clip"] @ ref_features["clip"])
            dino_score = float(query_features["dino"] @ ref_features["dino"])
            geometry = _patch_geometry(query_features["patches"], ref_features["patches"], patch_grid)
            clip_scores.append(clip_score)
            dino_scores.append(dino_score)
            geometry_scores.append(geometry["score"])
            ref_debug.append({"reference": reference.key, "clip": clip_score, "dino": dino_score, **geometry})
        clip_ref = max(clip_scores)
        dino_ref = max(dino_scores)
        geometry_ref = max(geometry_scores)
        arms["name_only"].append(name_score)
        arms["global_reference"].append(0.55 * clip_ref + 0.45 * dino_ref)
        arms["facade_fingerprint"].append(0.35 * clip_ref + 0.25 * dino_ref + 0.40 * geometry_ref)
        geometry_debug[target_id] = ref_debug
    return {"query": query.key, "target": query.target_id, "scores": arms, "geometry": geometry_debug}


def _margin(scores: list[float], goal_index: int) -> float:
    others = [value for index, value in enumerate(scores) if index != goal_index]
    return float(scores[goal_index] - max(others))


def _calibrate_threshold(positives: list[float], negatives: list[float]) -> float:
    candidates = {0.0}
    values = sorted(set(positives + negatives))
    candidates.update(values)
    candidates.update((left + right) / 2.0 for left, right in zip(values, values[1:]))
    best = (-1.0, -float("inf"))
    for threshold in candidates:
        if threshold < 0.0:
            continue
        recall = sum(value >= threshold for value in positives) / len(positives)
        specificity = sum(value < threshold for value in negatives) / len(negatives)
        balanced = 0.5 * (recall + specificity)
        candidate = (balanced, threshold)
        if candidate > best:
            best = candidate
    return float(best[1])


def _evaluate(
    rows: list[dict[str, Any]], target_ids: list[str], thresholds: dict[str, float]
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for arm in ("name_only", "global_reference", "facade_fingerprint"):
        positives = []
        negatives = []
        retrieval_correct = 0
        details = []
        for row in rows:
            scores = row["scores"][arm]
            true_index = target_ids.index(row["target"])
            predicted_index = int(np.argmax(scores))
            retrieval_correct += int(predicted_index == true_index)
            positive_margin = _margin(scores, true_index)
            positives.append(positive_margin)
            negative_margins = [_margin(scores, index) for index in range(len(target_ids)) if index != true_index]
            negatives.extend(negative_margins)
            details.append(
                {
                    "query": row["query"],
                    "truth": row["target"],
                    "prediction": target_ids[predicted_index],
                    "positive_margin": positive_margin,
                    "confirmed": positive_margin >= thresholds[arm],
                }
            )
        threshold = thresholds[arm]
        positive_hits = sum(value >= threshold for value in positives)
        false_confirms = sum(value >= threshold for value in negatives)
        positive_recall = positive_hits / len(positives)
        specificity = 1.0 - false_confirms / len(negatives)
        metrics[arm] = {
            "threshold": threshold,
            "queries": len(rows),
            "wrong_goal_pairs": len(negatives),
            "top1_correct": retrieval_correct,
            "top1_accuracy": retrieval_correct / len(rows),
            "positive_confirmed": positive_hits,
            "positive_confirmation_recall": positive_recall,
            "wrong_goal_false_confirmations": false_confirms,
            "wrong_goal_specificity": specificity,
            "balanced_accuracy": 0.5 * (positive_recall + specificity),
            "details": details,
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("named_poi_facade_fingerprint_protocol_v1.json"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts.local/knowledge/named-poi-facade-v2")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-facade-fingerprint-v1")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.resolve().read_text(encoding="utf-8"))
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    library_path = data_root / "target_library.json"
    rows, targets, observed_source = _validate_and_load_roles(protocol, library_path, data_root)
    target_ids = list(protocol["role_freeze"]["targets"])
    references = {target_id: [row for row in rows if row.target_id == target_id and row.role == "reference"] for target_id in target_ids}
    calibration_rows = [row for row in rows if row.role == "calibration"]
    evaluation_rows = [row for row in rows if row.role == "evaluation"]

    clip_path = ROOT / protocol["models"]["clip"]["path"]
    dino_path = ROOT / protocol["models"]["dinov2"]["path"]
    if _sha256(clip_path / "pytorch_model.bin") != protocol["models"]["clip"]["weights_sha256"]:
        raise ValueError("CLIP_MODEL_HASH_MISMATCH")
    if _sha256(dino_path / "model.safetensors") != protocol["models"]["dinov2"]["weights_sha256"]:
        raise ValueError("DINO_MODEL_HASH_MISMATCH")
    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative_image = Image.open(rows[0].path).convert("RGB")
    representative = (
        clip_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
        dino_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
    )
    backend, models = _select_backend(clip_path, dino_path, representative, output_root / "backend_receipt.json")
    device = str(backend["selected_device_type"])
    patch_grid = int(protocol["models"]["dinov2"]["patch_grid"])
    encoded = _encode_images(rows, models, clip_processor, dino_processor, device, patch_grid, args.batch_size)
    name_vectors = _encode_names(target_ids, targets, models, clip_processor, device)

    calibration_scored = [_score_query(row, target_ids, references, encoded, name_vectors, patch_grid) for row in calibration_rows]
    thresholds: dict[str, float] = {}
    for arm in ("name_only", "global_reference", "facade_fingerprint"):
        positives = []
        negatives = []
        for row in calibration_scored:
            true_index = target_ids.index(row["target"])
            positives.append(_margin(row["scores"][arm], true_index))
            negatives.extend(_margin(row["scores"][arm], index) for index in range(len(target_ids)) if index != true_index)
        thresholds[arm] = _calibrate_threshold(positives, negatives)

    evaluation_scored = [_score_query(row, target_ids, references, encoded, name_vectors, patch_grid) for row in evaluation_rows]
    metrics = _evaluate(evaluation_scored, target_ids, thresholds)
    fingerprint = metrics["facade_fingerprint"]
    global_reference = metrics["global_reference"]
    retained = (
        fingerprint["top1_correct"] > global_reference["top1_correct"]
        and fingerprint["wrong_goal_false_confirmations"] <= global_reference["wrong_goal_false_confirmations"]
    )
    result = {
        "schema": "l10-named-poi-facade-fingerprint-result-v1",
        "protocol_sha256": _sha256(args.protocol.resolve()),
        "source": observed_source,
        "roles": {
            "targets": len(target_ids),
            "reference_images": len([row for row in rows if row.role == "reference"]),
            "calibration_queries": len(calibration_rows),
            "evaluation_queries": len(evaluation_rows),
            "evaluation_wrong_goal_pairs": len(evaluation_rows) * (len(target_ids) - 1),
        },
        "execution_backend": backend,
        "ocr_calls": 0,
        "thresholds": thresholds,
        "metrics": metrics,
        "stop_condition": {
            "retain_local_geometry": retained,
            "decision": "RETAIN_NON_OCR_LOCAL_GEOMETRY" if retained else "DO_NOT_TUNE_LOCALIZE_NEXT_INFORMATION_GAP",
        },
        "claim_scope": protocol["source"]["claim_scope"],
        "evaluation_scores": evaluation_scored,
    }
    output_path = output_root / "result.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_path), "metrics": metrics, "stop_condition": result["stop_condition"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
