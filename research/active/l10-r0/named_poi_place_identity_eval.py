"""Run the frozen PB2-A single-frame place-identity source ceiling."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))

import named_poi_facade_fingerprint as facade  # noqa: E402
from research_backend import (  # noqa: E402
    BackendCandidate,
    Workload,
    runtime_capabilities,
    select_backend,
    torch_observation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label}_HASH_MISMATCH:expected={expected}:observed={observed}")


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


class _InferenceModel(nn.Module):
    def __init__(self, backbone: nn.Module, aggregator: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(images))


class _FrozenDinoV2Backbone(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = images.shape
        tokens = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        token = tokens[:, 0]
        features = tokens[:, 1:].reshape(batch, height // 14, width // 14, 768)
        return features.permute(0, 3, 1, 2), token


def _purge_official_models_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]


def _load_salad(protocol: dict[str, Any], device: str) -> nn.Module:
    spec = protocol["arms"]["C1_SALAD"]
    repo = ROOT / "artifacts.local/models/salad-official"
    dino_repo = ROOT / spec["dinov2_torch_hub_dependency"]["local_path"]
    _purge_official_models_modules()
    sys.path.insert(0, str(repo))
    try:
        from models.aggregators.salad import SALAD

        dino = torch.hub.load(
            str(dino_repo), "dinov2_vitb14", source="local", pretrained=False, verbose=False
        )
        model = _InferenceModel(
            _FrozenDinoV2Backbone(dino),
            SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256),
        )
        state = torch.load(ROOT / spec["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.eval().to(device)
    finally:
        sys.path.remove(str(repo))


def _load_mixvpr(protocol: dict[str, Any], device: str) -> nn.Module:
    spec = protocol["arms"]["C2_MixVPR"]
    repo = ROOT / "artifacts.local/models/mixvpr-official"
    _purge_official_models_modules()
    sys.path.insert(0, str(repo))
    try:
        from models.aggregators.mixvpr import MixVPR
        from models.backbones.resnet import ResNet

        model = _InferenceModel(
            ResNet("resnet50", pretrained=False, layers_to_freeze=2, layers_to_crop=[4]),
            MixVPR(
                in_channels=1024,
                in_h=20,
                in_w=20,
                out_channels=1024,
                mix_depth=4,
                mlp_ratio=1,
                out_rows=4,
            ),
        )
        state = torch.load(ROOT / spec["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.eval().to(device)
    finally:
        sys.path.remove(str(repo))


def _forward(model: nn.Module, pixels: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        return F.normalize(model(pixels.to(device)), dim=-1)


def _select_specialized_backend(
    name: str,
    loader: Callable[[str], nn.Module],
    representative: torch.Tensor,
    receipt_path: Path,
) -> tuple[dict[str, Any], nn.Module]:
    cpu_model = loader("cpu")
    gpu_model = loader("cuda") if torch.cuda.is_available() else None
    cpu = BackendCandidate(
        f"{name}-cpu",
        "cpu",
        lambda: _forward(cpu_model, representative, "cpu"),
        lambda output: torch_observation(output=output),
    )
    gpu = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            f"{name}-cuda",
            "cuda",
            lambda: _forward(gpu_model, representative, "cuda"),
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
    selected = gpu_model if backend["selected_device_type"] == "cuda" else cpu_model
    if selected is None:
        raise RuntimeError(f"{name}_SELECTED_MODEL_MISSING")
    if selected is gpu_model:
        del cpu_model
    else:
        del gpu_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return backend, selected


def _encode_specialized(
    rows: list[dict[str, Any]],
    model: nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        pixels = torch.stack([transform(Image.open(row["path"]).convert("RGB")) for row in batch])
        descriptors = _forward(model, pixels, device).detach().cpu().numpy().astype(np.float32)
        for row, descriptor in zip(batch, descriptors, strict=True):
            result[row["key"]] = descriptor
    return result


def _threshold(positive: list[float], negative: list[float]) -> dict[str, float]:
    values = sorted(set(positive + negative))
    epsilon = max(1e-7, (max(values) - min(values)) * 1e-7)
    candidates = [max(values) + epsilon, min(values) - epsilon]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:], strict=False))
    best: tuple[tuple[float, float, float, float], dict[str, float]] | None = None
    for threshold in candidates:
        tpr = sum(score >= threshold for score in positive) / len(positive)
        fpr = sum(score >= threshold for score in negative) / len(negative)
        balanced = 0.5 * (tpr + 1.0 - fpr)
        row = {"threshold": threshold, "balanced_accuracy": balanced, "tpr": tpr, "fpr": fpr}
        key = (balanced, tpr, -fpr, threshold)
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return best[1]


def _zscore(values: dict[str, float]) -> dict[str, float]:
    vector = np.asarray(list(values.values()), dtype=np.float64)
    scale = float(vector.std())
    if scale < 1e-12:
        return {key: 0.0 for key in values}
    mean = float(vector.mean())
    return {key: (value - mean) / scale for key, value in values.items()}


def _score_descriptors(
    rows: list[dict[str, Any]], descriptors: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for split, split_rows in by_split.items():
        entity_ids = sorted({row["entity_id"] for row in split_rows})
        refs = {
            entity: [row["key"] for row in split_rows if row["entity_id"] == entity and row["role"] == "reference"]
            for entity in entity_ids
        }
        queries = [row for row in split_rows if row["role"] == "query"]
        scores[split] = {}
        for arm in ("B0_CLIP", "B1_DINOv2", "C1_SALAD", "C2_MixVPR"):
            arm_scores: dict[str, dict[str, float]] = {}
            for query in queries:
                vector = descriptors[arm][query["key"]]
                arm_scores[query["key"]] = {
                    entity: max(float(vector @ descriptors[arm][ref]) for ref in refs[entity])
                    for entity in entity_ids
                }
            scores[split][arm] = arm_scores
        scores[split]["B2_CLIP_DINO"] = {}
        for query in queries:
            clip = _zscore(scores[split]["B0_CLIP"][query["key"]])
            dino = _zscore(scores[split]["B1_DINOv2"][query["key"]])
            scores[split]["B2_CLIP_DINO"][query["key"]] = {
                entity: 0.5 * (clip[entity] + dino[entity]) for entity in entity_ids
            }
    return scores


def _metrics(
    queries: list[dict[str, Any]], arm_scores: dict[str, dict[str, float]], threshold: float
) -> dict[str, Any]:
    top1_hits = 0
    top3_hits = 0
    positive_accepts = 0
    wrong_accepts = 0
    wrong_total = 0
    entity_hits: dict[str, list[int]] = defaultdict(list)
    facets: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "top1": 0, "accepted": 0})
    records = []
    for query in queries:
        values = arm_scores[query["key"]]
        order = sorted(values, key=lambda entity: (-values[entity], entity))
        target = query["entity_id"]
        top1 = order[0] == target
        top3 = target in order[:3]
        accepted = values[target] >= threshold
        wrong = [entity for entity in order if entity != target]
        accepted_wrong = [entity for entity in wrong if values[entity] >= threshold]
        top1_hits += top1
        top3_hits += top3
        positive_accepts += accepted
        wrong_accepts += len(accepted_wrong)
        wrong_total += len(wrong)
        entity_hits[target].append(int(top1))
        facet = facets[query["facet"]]
        facet["count"] += 1
        facet["top1"] += top1
        facet["accepted"] += accepted
        records.append(
            {
                "key": query["key"],
                "target": target,
                "facet": query["facet"],
                "ranked_entities": order,
                "scores": values,
                "target_score": values[target],
                "top1": bool(top1),
                "top3": bool(top3),
                "positive_accepted": bool(accepted),
                "wrong_entities_accepted": accepted_wrong,
            }
        )
    count = len(queries)
    fpr = wrong_accepts / wrong_total
    return {
        "query_count": count,
        "recall_at_1": top1_hits / count,
        "recall_at_3": top3_hits / count,
        "entity_macro_recall_at_1": float(np.mean([np.mean(hits) for hits in entity_hits.values()])),
        "positive_acceptance": positive_accepts / count,
        "wrong_building_false_confirmation": fpr,
        "target_absent_rejection": 1.0 - fpr,
        "state_counts": {
            "positive_accept": positive_accepts,
            "positive_reject": count - positive_accepts,
            "wrong_accept": wrong_accepts,
            "wrong_reject": wrong_total - wrong_accepts,
        },
        "facets": {
            name: {
                "count": row["count"],
                "recall_at_1": row["top1"] / row["count"],
                "positive_acceptance": row["accepted"] / row["count"],
            }
            for name, row in sorted(facets.items())
        },
        "records": records,
    }


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metrics["recall_at_1"],
        metrics["positive_acceptance"],
        -metrics["wrong_building_false_confirmation"],
        metrics["recall_at_3"],
    )


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "records"}


def run(protocol_path: Path, output_root: Path, batch_size: int) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for label, source in protocol["sources"].items():
        _require_hash(ROOT / source["path"], source["sha256"], label.upper())
    manifest_path = ROOT / protocol["sources"]["dataset_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["inventory_sha256"] != protocol["sources"]["dataset_manifest"]["inventory_sha256"]:
        raise ValueError("DATASET_INVENTORY_HASH_MISMATCH")
    audit = json.loads((ROOT / protocol["sources"]["source_audit"]["path"]).read_text(encoding="utf-8"))
    if audit["counts"]["model_calls_before_freeze"] != 0:
        raise ValueError("SOURCE_AUDIT_MODEL_CALLS_NOT_ZERO")

    clip_path = ROOT / protocol["arms"]["B0_CLIP"]["source"]
    dino_path = ROOT / protocol["arms"]["B1_DINOv2"]["source"]
    _require_hash(clip_path / "pytorch_model.bin", protocol["arms"]["B0_CLIP"]["weights_sha256"], "CLIP")
    _require_hash(dino_path / "model.safetensors", protocol["arms"]["B1_DINOv2"]["weights_sha256"], "DINOV2")
    for arm in ("C1_SALAD", "C2_MixVPR"):
        _require_hash(ROOT / protocol["arms"][arm]["checkpoint"], protocol["arms"][arm]["checkpoint_sha256"], arm)
    if _git_head(ROOT / "artifacts.local/models/salad-official") != protocol["arms"]["C1_SALAD"]["repository_commit"]:
        raise ValueError("SALAD_REPOSITORY_COMMIT_MISMATCH")
    if _git_head(ROOT / "artifacts.local/models/mixvpr-official") != protocol["arms"]["C2_MixVPR"]["repository_commit"]:
        raise ValueError("MIXVPR_REPOSITORY_COMMIT_MISMATCH")
    dino_dependency = protocol["arms"]["C1_SALAD"]["dinov2_torch_hub_dependency"]
    if _git_head(ROOT / dino_dependency["local_path"]) != dino_dependency["repository_commit"]:
        raise ValueError("SALAD_DINOV2_DEPENDENCY_COMMIT_MISMATCH")

    rows: list[dict[str, Any]] = []
    for entity in manifest["entities"]:
        for index, image in enumerate(entity["references"], start=1):
            path = Path(image["local_path"])
            _require_hash(path, image["sha256"], f"REFERENCE:{entity['id']}:{index}")
            rows.append({"key": f"ref:{entity['split']}:{entity['id']}:{index:02d}", "entity_id": entity["id"], "split": entity["split"], "role": "reference", "facet": None, "path": path})
        for image in entity["queries"]:
            path = Path(image["local_path"])
            _require_hash(path, image["sha256"], f"QUERY:{image['key']}")
            rows.append({"key": f"query:{image['key']}", "entity_id": entity["id"], "split": entity["split"], "role": "query", "facet": image["facet"], "path": path})
    split_entities = {split: {row["entity_id"] for row in rows if row["split"] == split} for split in ("development", "test")}
    if split_entities["development"] & split_entities["test"]:
        raise ValueError("BUILDING_SPLIT_OVERLAP")

    output_root.mkdir(parents=True, exist_ok=True)
    development_rows = [row for row in rows if row["split"] == "development"]
    representative_images = [Image.open(row["path"]).convert("RGB") for row in development_rows[:8]]

    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative_baseline = (
        clip_processor(images=representative_images, return_tensors="pt")["pixel_values"],
        dino_processor(images=representative_images, return_tensors="pt")["pixel_values"],
    )
    baseline_backend, baseline_models = facade._select_backend(
        clip_path, dino_path, representative_baseline, output_root / "baseline_backend_receipt.json"
    )
    baseline_rows = [
        facade.ImageRow(row["key"], row["entity_id"], row["role"], row["path"], _sha256(row["path"]), "")
        for row in rows
    ]
    encoded_baseline = facade._encode_images(
        baseline_rows,
        baseline_models,
        clip_processor,
        dino_processor,
        baseline_backend["selected_device_type"],
        patch_grid=9,
        batch_size=batch_size,
    )
    descriptors: dict[str, dict[str, np.ndarray]] = {
        "B0_CLIP": {key: value["clip"] for key, value in encoded_baseline.items()},
        "B1_DINOv2": {key: value["dino"] for key, value in encoded_baseline.items()},
    }
    del baseline_models, encoded_baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    salad_transform = T.Compose([T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(mean, std)])
    mix_transform = T.Compose([T.Resize((320, 320), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean, std)])

    representative_salad = torch.stack([salad_transform(image) for image in representative_images])
    salad_backend, salad_model = _select_specialized_backend(
        "pb2a-salad", lambda device: _load_salad(protocol, device), representative_salad, output_root / "salad_backend_receipt.json"
    )
    descriptors["C1_SALAD"] = _encode_specialized(rows, salad_model, salad_transform, salad_backend["selected_device_type"], batch_size)
    salad_peak = int(torch.cuda.max_memory_allocated()) if salad_backend["selected_device_type"] == "cuda" else 0
    del salad_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    representative_mix = torch.stack([mix_transform(image) for image in representative_images])
    mix_backend, mix_model = _select_specialized_backend(
        "pb2a-mixvpr", lambda device: _load_mixvpr(protocol, device), representative_mix, output_root / "mixvpr_backend_receipt.json"
    )
    descriptors["C2_MixVPR"] = _encode_specialized(rows, mix_model, mix_transform, mix_backend["selected_device_type"], batch_size)
    mix_peak = int(torch.cuda.max_memory_allocated()) if mix_backend["selected_device_type"] == "cuda" else 0
    del mix_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores = _score_descriptors(rows, descriptors)
    arms = ["B0_CLIP", "B1_DINOv2", "B2_CLIP_DINO", "C1_SALAD", "C2_MixVPR"]
    queries = {split: [row for row in rows if row["split"] == split and row["role"] == "query"] for split in ("development", "test")}
    thresholds: dict[str, dict[str, float]] = {}
    development_metrics: dict[str, dict[str, Any]] = {}
    for arm in arms:
        positives = [scores["development"][arm][row["key"]][row["entity_id"]] for row in queries["development"]]
        negatives = [score for row in queries["development"] for entity, score in scores["development"][arm][row["key"]].items() if entity != row["entity_id"]]
        thresholds[arm] = _threshold(positives, negatives)
        development_metrics[arm] = _metrics(queries["development"], scores["development"][arm], thresholds[arm]["threshold"])

    baseline = max(("B0_CLIP", "B1_DINOv2", "B2_CLIP_DINO"), key=lambda arm: (_selection_key(development_metrics[arm]), arm))
    challenger = max(("C1_SALAD", "C2_MixVPR"), key=lambda arm: (_selection_key(development_metrics[arm]), arm))
    test_metrics = {arm: _metrics(queries["test"], scores["test"][arm], thresholds[arm]["threshold"]) for arm in arms}
    baseline_test = test_metrics[baseline]
    challenger_test = test_metrics[challenger]
    recall_gain = challenger_test["recall_at_1"] - baseline_test["recall_at_1"]
    acceptance_gain = challenger_test["positive_acceptance"] - baseline_test["positive_acceptance"]
    identity_gain = recall_gain >= 0.10 - 1e-12 or acceptance_gain >= 0.10 - 1e-12
    wrong_guardrail = challenger_test["wrong_building_false_confirmation"] <= baseline_test["wrong_building_false_confirmation"] + 0.02 + 1e-12
    positive_guardrail = challenger_test["positive_acceptance"] >= 0.25 - 1e-12 and challenger_test["positive_acceptance"] >= baseline_test["positive_acceptance"] - 0.05 - 1e-12
    passed = identity_gain and wrong_guardrail and positive_guardrail
    decision = "L10_PB2A_SPECIALIZED_VPR_IDENTITY_GATE_MET" if passed else "L10_PB2A_SPECIALIZED_VPR_IDENTITY_GATE_NOT_MET_STOP_SINGLE_FRAME_APPEARANCE_ONLY"

    result = {
        "schema": "l10-named-poi-place-identity-result-v1",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "inventory_sha256": manifest["inventory_sha256"],
        "selection": {"baseline": baseline, "challenger": challenger, "authority": "DEVELOPMENT_ONLY"},
        "thresholds": thresholds,
        "development": {arm: _compact(metrics) for arm, metrics in development_metrics.items()},
        "test": {arm: _compact(metrics) for arm, metrics in test_metrics.items()},
        "selected_comparison": {
            "baseline": baseline,
            "challenger": challenger,
            "recall_at_1_gain": recall_gain,
            "positive_acceptance_gain": acceptance_gain,
            "wrong_building_false_confirmation_gain": challenger_test["wrong_building_false_confirmation"] - baseline_test["wrong_building_false_confirmation"],
            "identity_gain_clause": identity_gain,
            "wrong_building_guardrail": wrong_guardrail,
            "positive_authority_guardrail": positive_guardrail,
        },
        "backend": {
            "baseline": baseline_backend,
            "salad": {**salad_backend, "full_run_peak_cuda_memory_bytes": salad_peak},
            "mixvpr": {**mix_backend, "full_run_peak_cuda_memory_bytes": mix_peak},
        },
        "decision": decision,
        "claim_boundary": protocol["claim_boundary"],
        "raw_records": {
            split: {arm: metrics["records"] for arm, metrics in ((arm, development_metrics[arm] if split == "development" else test_metrics[arm]) for arm in arms)}
            for split in ("development", "test")
        },
    }
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("named_poi_place_identity_protocol_v1.json"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-place-identity-v1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = run(args.protocol.resolve(), args.output_root.resolve(), args.batch_size)
    print(json.dumps({"selection": result["selection"], "selected_comparison": result["selected_comparison"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
