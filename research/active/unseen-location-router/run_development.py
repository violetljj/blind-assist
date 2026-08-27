"""Train frozen fusion arms and evaluate held-out Development locations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


SCHEMA = "blindassist.unseen_location_router.development_result.v1"
MODALITIES = ("visual", "ocr", "geo")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FUSION = load_module("ulr_fusion_runtime", Path(__file__).with_name("fusion.py"))


@dataclass(frozen=True)
class FeatureRow:
    image_id: str
    split: str
    role: str
    location_id: str
    capture_group: str
    descriptor: np.ndarray
    blur_variance: float
    ocr_texts: tuple[str, ...]
    ocr_scores: tuple[float, ...]
    latitude: float | None
    longitude: float | None
    gps_accuracy_m: float | None
    illumination: str


@dataclass(frozen=True)
class Location:
    location_id: str
    split: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Example:
    image_id: str
    scores: np.ndarray
    available: np.ndarray
    quality: np.ndarray
    label: int
    illumination: str
    ocr_present: bool
    blur_variance: float
    gps_accuracy_m: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_meters(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius = 6_371_008.8
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    d_phi = math.radians(lat_b - lat_a)
    d_lambda = math.radians(lon_b - lon_a)
    value = math.sin(d_phi / 2) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def normalize_text(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in folded if character.isalnum())


def text_grams(texts: Iterable[str]) -> set[str]:
    value = "".join(normalize_text(text) for text in texts)
    if not value:
        return set()
    if len(value) == 1:
        return {value}
    return {value[index:index + 2] for index in range(len(value) - 1)}


def illumination_by_capture_group(texts_root: Path) -> dict[str, str]:
    module = load_module("ulr_build_manifest_for_times", Path(__file__).with_name("build_manifest.py"))
    result: dict[str, str] = {}
    for filename in ("Metadata-Images.xlsx", "Metadata-Videos.xlsx"):
        for row in module.read_first_xlsx_sheet(texts_root / filename):
            source = Path(str(row.get("Filename", ""))).stem.upper()
            timestamp = str(row.get("Creation Time", ""))
            try:
                hour = int(timestamp.split()[1].split(":")[0])
            except (IndexError, ValueError):
                continue
            result[f"field:{source}"] = "day" if 7 <= hour < 18 else "night"
    return result


def load_features(database: Path, texts_root: Path | None) -> tuple[list[FeatureRow], dict[str, str]]:
    connection = sqlite3.connect(database)
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    inferred_illumination = illumination_by_capture_group(texts_root) if texts_root else {}
    rows = []
    for row in connection.execute(
        """SELECT image_id, split, role, location_id, capture_group, descriptor, descriptor_dim,
        blur_variance, ocr_texts_json, ocr_scores_json, latitude, longitude,
        gps_accuracy_m, illumination FROM features"""
    ):
        descriptor = np.frombuffer(row[5], dtype=np.float32, count=int(row[6])).copy()
        illumination = row[13] if row[13] != "unknown" else inferred_illumination.get(row[4], "unknown")
        rows.append(FeatureRow(
            image_id=row[0], split=row[1], role=row[2], location_id=row[3], capture_group=row[4],
            descriptor=descriptor, blur_variance=float(row[7]), ocr_texts=tuple(json.loads(row[8])),
            ocr_scores=tuple(float(value) for value in json.loads(row[9])),
            latitude=float(row[10]) if row[10] is not None else None,
            longitude=float(row[11]) if row[11] is not None else None,
            gps_accuracy_m=float(row[12]) if row[12] is not None else None,
            illumination=illumination,
        ))
    connection.close()
    return rows, metadata


def load_locations(manifest: Path) -> list[Location]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return [Location(
        location_id=row["location_id"], split=row["split"],
        latitude=float(row["latitude"]), longitude=float(row["longitude"]),
    ) for row in payload["locations"] if row["split"] != "test"]


def gallery_profiles(rows: Sequence[FeatureRow]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[FeatureRow]] = defaultdict(list)
    for row in rows:
        if row.role == "gallery":
            grouped[row.location_id].append(row)
    profiles: dict[str, dict[str, Any]] = {}
    for location_id, items in grouped.items():
        descriptor = np.stack([item.descriptor for item in items]).mean(axis=0)
        descriptor /= max(np.linalg.norm(descriptor), 1e-12)
        texts = tuple(text for item in items for text in item.ocr_texts)
        profiles[location_id] = {
            "descriptor": descriptor,
            "text_grams": text_grams(texts),
            "text_chars": sum(len(normalize_text(text)) for text in texts),
            "count": len(items),
        }
    return profiles


def build_examples(
    rows: Sequence[FeatureRow], locations: Sequence[Location], *, split: str, candidate_count: int
) -> tuple[list[Example], dict[str, int]]:
    profiles = gallery_profiles([row for row in rows if row.split == split])
    candidates = [location for location in locations if location.split == split and location.location_id in profiles]
    examples: list[Example] = []
    counters = defaultdict(int)
    for query in rows:
        if query.split != split or query.role != "query":
            continue
        counters["query_total"] += 1
        if query.latitude is None or query.longitude is None:
            counters["missing_query_gps"] += 1
            continue
        ranked = sorted(candidates, key=lambda location: (
            haversine_meters(query.latitude, query.longitude, location.latitude, location.longitude),
            location.location_id,
        ))[:candidate_count]
        candidate_ids = [location.location_id for location in ranked]
        if query.location_id not in candidate_ids:
            counters["target_outside_candidate_set"] += 1
            continue
        query_grams = text_grams(query.ocr_texts)
        query_confidence = float(np.mean(query.ocr_scores)) if query.ocr_scores else 0.0
        query_chars = sum(len(normalize_text(text)) for text in query.ocr_texts)
        scores = np.zeros((candidate_count, len(MODALITIES)), dtype=np.float32)
        available = np.ones_like(scores, dtype=bool)
        quality = np.zeros((candidate_count, len(MODALITIES), 3), dtype=np.float32)
        for index, location in enumerate(ranked):
            profile = profiles[location.location_id]
            visual_score = float(np.dot(query.descriptor, profile["descriptor"]))
            gallery_grams = profile["text_grams"]
            ocr_available = bool(query_grams and gallery_grams)
            ocr_score = len(query_grams & gallery_grams) / len(query_grams | gallery_grams) if ocr_available else 0.0
            distance = haversine_meters(query.latitude, query.longitude, location.latitude, location.longitude)
            geo_score = math.exp(-distance / 75.0)
            scores[index] = (visual_score, ocr_score, geo_score)
            available[index] = (True, ocr_available, True)
            quality[index, 0] = (
                min(1.0, math.log1p(query.blur_variance) / 8.0),
                min(1.0, profile["count"] / 4.0),
                1.0,
            )
            quality[index, 1] = (
                query_confidence,
                min(1.0, query_chars / 20.0),
                min(1.0, profile["text_chars"] / 40.0),
            )
            accuracy = query.gps_accuracy_m if query.gps_accuracy_m is not None else 100.0
            quality[index, 2] = (
                math.exp(-accuracy / 30.0),
                1.0,
                candidate_count / 16.0,
            )
        examples.append(Example(
            image_id=query.image_id, scores=scores, available=available, quality=quality,
            label=candidate_ids.index(query.location_id), illumination=query.illumination,
            ocr_present=bool(query_grams), blur_variance=query.blur_variance,
            gps_accuracy_m=query.gps_accuracy_m if query.gps_accuracy_m is not None else 100.0,
        ))
        counters["evaluable"] += 1
    return examples, dict(counters)


def tensors(examples: Sequence[Example], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.tensor(np.stack([example.scores for example in examples]), device=device),
        torch.tensor(np.stack([example.available for example in examples]), device=device),
        torch.tensor(np.stack([example.quality for example in examples]), device=device),
        torch.tensor([example.label for example in examples], dtype=torch.long, device=device),
    )


def train_model(model: torch.nn.Module, fit: Sequence[Example], validation: Sequence[Example], *, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    fit_tensors = tensors(fit, device)
    validation_tensors = tensors(validation, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best_state = None
    best_accuracy = -1.0
    stale = 0
    for _epoch in range(60):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, FUSION.QualityConditionedEvidenceRouter):
            output = model(fit_tensors[0], fit_tensors[1], fit_tensors[2])
        else:
            output = model(fit_tensors[0], fit_tensors[1])
        loss = F.cross_entropy(output.candidate_scores, fit_tensors[3])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            if isinstance(model, FUSION.QualityConditionedEvidenceRouter):
                prediction = model(validation_tensors[0], validation_tensors[1], validation_tensors[2]).candidate_scores
            else:
                prediction = model(validation_tensors[0], validation_tensors[1]).candidate_scores
            accuracy = float((prediction.argmax(dim=1) == validation_tensors[3]).float().mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break
    if best_state is None:
        raise RuntimeError("no model state selected")
    model.load_state_dict(best_state)
    return model


def accuracy_from_logits(logits: Tensor, labels: Tensor, topk: int = 1) -> float:
    k = min(topk, logits.shape[1])
    correct = logits.topk(k, dim=1).indices.eq(labels[:, None]).any(dim=1)
    return float(correct.float().mean())


def evaluate_baselines(examples: Sequence[Example], device: torch.device) -> dict[str, dict[str, float | int]]:
    scores, available, _, labels = tensors(examples, device)
    result: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(MODALITIES):
        logits = scores[:, :, index].masked_fill(~available[:, :, index], torch.finfo(scores.dtype).min)
        row_available = available[:, :, index].any(dim=1)
        available_logits = logits[row_available]
        available_labels = labels[row_available]
        result[f"{name}_only"] = {
            "top1_all_queries_missing_counts_wrong": float(
                (logits.argmax(dim=1).eq(labels) & row_available).float().mean()
            ),
            "top1_when_available": accuracy_from_logits(available_logits, available_labels) if row_available.any() else None,
            "top5_when_available": accuracy_from_logits(available_logits, available_labels, 5) if row_available.any() else None,
            "query_count": len(examples),
            "queries_with_modality": int(row_available.sum()),
        }
    fixed = FUSION.fixed_equal_available_fusion(scores, available).candidate_scores
    result["fixed_equal_available_fusion"] = {
        "top1": accuracy_from_logits(fixed, labels),
        "top5": accuracy_from_logits(fixed, labels, 5),
        "query_count": len(examples),
        "queries_with_modality": len(examples),
    }
    return result


def evaluate_model(model: torch.nn.Module, examples: Sequence[Example], device: torch.device) -> tuple[dict[str, float | int], np.ndarray | None]:
    scores, available, quality, labels = tensors(examples, device)
    model.eval()
    with torch.inference_mode():
        if isinstance(model, FUSION.QualityConditionedEvidenceRouter):
            output = model(scores, available, quality)
            weights = output.modality_weights.cpu().numpy()
        else:
            output = model(scores, available)
            weights = None
    return {
        "top1": accuracy_from_logits(output.candidate_scores, labels),
        "top5": accuracy_from_logits(output.candidate_scores, labels, 5),
        "query_count": len(examples),
        "local_hard_negative_error": 1.0 - accuracy_from_logits(output.candidate_scores, labels),
    }, weights


def model_strata(model: torch.nn.Module, examples: Sequence[Example], device: torch.device) -> dict[str, Any]:
    blur_values = np.array([example.blur_variance for example in examples])
    median_blur = float(np.median(blur_values))
    groups = {
        "day": [example for example in examples if example.illumination == "day"],
        "night": [example for example in examples if example.illumination == "night"],
        "illumination_unknown": [example for example in examples if example.illumination == "unknown"],
        "ocr_present": [example for example in examples if example.ocr_present],
        "ocr_missing": [example for example in examples if not example.ocr_present],
        "lower_blur_variance_half": [example for example in examples if example.blur_variance <= median_blur],
        "higher_blur_variance_half": [example for example in examples if example.blur_variance > median_blur],
    }
    result: dict[str, Any] = {}
    for name, items in groups.items():
        result[name] = {"count": len(items), "top1": evaluate_model(model, items, device)[0]["top1"]} if items else {
            "count": 0, "top1": None
        }
    return result


def routing_diagnostics(examples: Sequence[Example], weights: np.ndarray) -> dict[str, Any]:
    target_weights = np.stack([weights[index, example.label] for index, example in enumerate(examples)])
    present = np.array([example.ocr_present for example in examples], dtype=bool)
    blur = np.array([example.blur_variance for example in examples])
    median_blur = float(np.median(blur))

    def mean_rows(mask: np.ndarray) -> dict[str, float] | None:
        if not mask.any():
            return None
        values = target_weights[mask].mean(axis=0)
        return {name: float(values[index]) for index, name in enumerate(MODALITIES)}

    return {
        "target_candidate_mean_weights": mean_rows(np.ones(len(examples), dtype=bool)),
        "ocr_present": {"count": int(present.sum()), "weights": mean_rows(present)},
        "ocr_missing": {"count": int((~present).sum()), "weights": mean_rows(~present)},
        "lower_blur_variance_half": {"count": int((blur <= median_blur).sum()), "weights": mean_rows(blur <= median_blur)},
        "higher_blur_variance_half": {"count": int((blur > median_blur).sum()), "weights": mean_rows(blur > median_blur)},
        "router_weight_variance": {
            name: float(target_weights[:, index].var()) for index, name in enumerate(MODALITIES)
        },
    }


def internal_train_split(examples: Sequence[Example]) -> tuple[list[Example], list[Example]]:
    fit, validation = [], []
    for example in examples:
        value = int.from_bytes(hashlib.sha256(f"ulr-internal|{example.image_id}".encode()).digest()[:8], "big")
        (validation if value % 5 == 0 else fit).append(example)
    if not fit or not validation:
        raise RuntimeError("internal train split is empty")
    return fit, validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--texts-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8, choices=(4, 8, 16))
    args = parser.parse_args()

    rows, feature_metadata = load_features(args.database, args.texts_root)
    if any(row.split == "test" for row in rows):
        raise RuntimeError("Development feature store must not contain test images")
    locations = load_locations(args.manifest)
    train_examples, train_coverage = build_examples(rows, locations, split="train", candidate_count=args.candidate_count)
    development_examples, development_coverage = build_examples(
        rows, locations, split="development", candidate_count=args.candidate_count
    )
    fit, internal_validation = internal_train_split(train_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baselines = evaluate_baselines(development_examples, device)
    seeds: dict[str, Any] = {}
    for seed in (1701, 2701):
        torch.manual_seed(seed)
        static = train_model(FUSION.StaticLearnedFusion(3).to(device), fit, internal_validation, seed=seed)
        torch.manual_seed(seed)
        router = train_model(FUSION.QualityConditionedEvidenceRouter(3, 3).to(device), fit, internal_validation, seed=seed)
        static_metrics, _ = evaluate_model(static, development_examples, device)
        router_metrics, router_weights = evaluate_model(router, development_examples, device)
        assert router_weights is not None
        gain = float(router_metrics["top1"]) - float(static_metrics["top1"])
        seeds[str(seed)] = {
            "static_learned_fusion": static_metrics,
            "quality_conditioned_dynamic_router": router_metrics,
            "absolute_top1_gain": gain,
            "static_strata": model_strata(static, development_examples, device),
            "router_strata": model_strata(router, development_examples, device),
            "routing": routing_diagnostics(development_examples, router_weights),
        }

    advances = all(
        row["absolute_top1_gain"] >= 0.08
        and row["quality_conditioned_dynamic_router"]["local_hard_negative_error"]
        <= row["static_learned_fusion"]["local_hard_negative_error"]
        for row in seeds.values()
    )
    result = {
        "schema": SCHEMA,
        "status": "ADVANCE_GATE_MET" if advances else "DEVELOPMENT_CANARY_GATE_NOT_MET",
        "scope": "bounded_real_image_mechanics_canary",
        "candidate_count": args.candidate_count,
        "train_query_count": len(train_examples),
        "internal_fit_count": len(fit),
        "internal_validation_count": len(internal_validation),
        "development_query_count": len(development_examples),
        "train_candidate_coverage": train_coverage,
        "development_candidate_coverage": development_coverage,
        "untrained_baselines": baselines,
        "seeds": seeds,
        "advance_gate": {
            "required_absolute_top1_gain_each_seed": 0.08,
            "local_hard_negative_error_must_not_increase": True,
            "met": advances,
        },
        "test_images_read": 0,
        "feature_manifest_sha256": feature_metadata["manifest_sha256"],
        "feature_database_sha256": sha256_file(args.database),
        "limitations": [
            "at most four gallery and eight query capture groups per train/development location",
            "OCR correctness has no independent transcription truth and is not reported",
            "unknown illumination labels prevent complete day/night stratification",
            "single district and non-POI location classes",
            "test split remains unopened",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
