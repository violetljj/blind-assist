from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


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


ARMS = ("identity_only_ready", "clip_entrance_ready", "entity_bound_entrance_graph")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_hash(paths: list[Path], root: Path) -> str:
    rows = []
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        rows.append(f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\t{_sha256(path)}\n")
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _load_rows(protocol: dict[str, Any]) -> tuple[list[facade.ImageRow], list[facade.ImageRow], dict[str, Any], dict[str, Any]]:
    ref_path = ROOT / protocol["sources"]["reference_library"]["path"]
    fresh_path = ROOT / protocol["sources"]["fresh_multifacet_library"]["path"]
    if _sha256(ref_path) != protocol["sources"]["reference_library"]["sha256"]:
        raise ValueError("REFERENCE_LIBRARY_HASH_MISMATCH")
    if _sha256(fresh_path) != protocol["sources"]["fresh_multifacet_library"]["sha256"]:
        raise ValueError("FRESH_LIBRARY_HASH_MISMATCH")
    references = json.loads(ref_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    fresh_root = fresh_path.parent
    fresh_paths = [
        fresh_root / "images" / target["id"] / Path(row["local_path"]).name
        for target in fresh["targets"]
        for row in target["facets"]
    ]
    expected = protocol["sources"]["fresh_multifacet_library"]
    observed = {
        "image_count": len(fresh_paths),
        "image_bytes": sum(path.stat().st_size for path in fresh_paths),
        "inventory_sha256": _inventory_hash(fresh_paths, fresh_root),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise ValueError(f"FRESH_{key.upper()}_MISMATCH:expected={expected[key]}:observed={value}")

    refs_by_id = {str(row["id"]): row for row in references["targets"]}
    fresh_by_id = {str(row["id"]): row for row in fresh["targets"]}
    reference_rows = []
    query_rows = []
    metadata: dict[str, Any] = {}
    for target_id, roles in protocol["targets"].items():
        ref_target = refs_by_id[target_id]
        for index in roles["reference_indices"]:
            source = ref_target["reference_images"][int(index) - 1]
            path = ref_path.parent / "images" / target_id / Path(source["local_path"]).name
            if _sha256(path) != source["sha256"]:
                raise ValueError(f"REFERENCE_IMAGE_HASH_MISMATCH:{target_id}:{index}")
            reference_rows.append(
                facade.ImageRow(f"ref:{target_id}:{int(index):02d}", target_id, "reference", path, source["sha256"], source["commons_file"])
            )
        fresh_target = fresh_by_id[target_id]
        for role, indices_key in (("calibration", "calibration_indices"), ("evaluation", "evaluation_indices")):
            for index in roles[indices_key]:
                source = fresh_target["facets"][int(index) - 1]
                path = fresh_path.parent / "images" / target_id / Path(source["local_path"]).name
                if _sha256(path) != source["sha256"]:
                    raise ValueError(f"FRESH_IMAGE_HASH_MISMATCH:{target_id}:{index}")
                key = f"fresh:{target_id}:{int(index):02d}"
                query_rows.append(facade.ImageRow(key, target_id, role, path, source["sha256"], source["commons_file"]))
                human_labels = protocol.get("human_entrance_labels")
                if human_labels is not None:
                    target_labels = human_labels.get(target_id, {})
                    if str(index) not in target_labels:
                        raise ValueError(f"MISSING_HUMAN_ENTRANCE_LABEL:{target_id}:{index}")
                    entrance_view = bool(target_labels[str(index)])
                    entrance_authority = "HUMAN_FROZEN_BEFORE_MODEL_INFERENCE"
                else:
                    entrance_view = source["facet_hint"] == "entrance"
                    entrance_authority = "WEAK_COMMONS_FILENAME_METADATA"
                metadata[key] = {
                    "facet_hint": source["facet_hint"],
                    "entrance_view": entrance_view,
                    "entrance_authority": entrance_authority,
                    "facet_reasons": source["facet_reasons"],
                }
    keys = [row.key for row in reference_rows + query_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("ROLE_KEY_OVERLAP")
    return reference_rows, query_rows, refs_by_id, {"metadata": metadata, "source": observed}


def _grounder_forward(model: Any, inputs: Any) -> Any:
    with torch.inference_mode():
        return model(**inputs)


def _select_grounder(
    model_path: Path, processor: Any, image: Image.Image, prompt: str, receipt: Path
) -> tuple[dict[str, Any], Any]:
    cpu_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, local_files_only=True).eval().to("cpu")
    cpu_inputs = processor(images=[image], text=[prompt], return_tensors="pt", padding=True)
    gpu_model = None
    gpu_inputs = None
    if torch.cuda.is_available():
        gpu_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, local_files_only=True).eval().to("cuda")
        gpu_inputs = processor(images=[image], text=[prompt], return_tensors="pt", padding=True).to("cuda")
    cpu = BackendCandidate(
        "named-poi-entrance-grounding-dino-cpu",
        "cpu",
        lambda: _grounder_forward(cpu_model, cpu_inputs),
        lambda output: torch_observation(model=cpu_model, output=output),
    )
    gpu = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "named-poi-entrance-grounding-dino-cuda",
            "cuda",
            lambda: _grounder_forward(gpu_model, gpu_inputs),
            lambda output: torch_observation(model=gpu_model, output=output),
            torch.cuda.synchronize,
        )
    backend = select_backend(
        Workload.MODEL_INFERENCE,
        cpu=cpu,
        gpu=gpu,
        cpu_reason="ACCELERATOR_UNAVAILABLE" if gpu is None else None,
        record_path=receipt,
        warmups=0,
        repeats=1,
        capabilities=runtime_capabilities(),
    )
    selected = gpu_model if backend["selected_device_type"] == "cuda" else cpu_model
    if selected is None:
        raise RuntimeError("SELECTED_GROUNDER_MISSING")
    if selected is gpu_model:
        del cpu_model, cpu_inputs
    else:
        del gpu_model, gpu_inputs
    gc.collect()
    return backend, selected


def _encode_prompt(model: Any, processor: Any, text: str, device: str) -> np.ndarray:
    values = processor(text=[text], padding=True, return_tensors="pt")
    values = {key: value.to(device) for key, value in values.items() if key in {"input_ids", "attention_mask"}}
    with torch.inference_mode():
        pooled = model.text_model(**values).pooler_output
        vector = facade._normalized(model.text_projection(pooled))
    return vector[0].detach().cpu().numpy().astype(np.float32)


def _identity_scores(
    query: facade.ImageRow,
    target_ids: list[str],
    references: dict[str, list[facade.ImageRow]],
    encoded: dict[str, dict[str, np.ndarray]],
    patch_grid: int,
) -> list[float]:
    query_features = encoded[query.key]
    scores = []
    for target_id in target_ids:
        clip_values = []
        dino_values = []
        geometry_values = []
        for reference in references[target_id]:
            ref = encoded[reference.key]
            clip_values.append(float(query_features["clip"] @ ref["clip"]))
            dino_values.append(float(query_features["dino"] @ ref["dino"]))
            geometry_values.append(facade._patch_geometry(query_features["patches"], ref["patches"], patch_grid)["score"])
        scores.append(0.35 * max(clip_values) + 0.25 * max(dino_values) + 0.40 * max(geometry_values))
    return scores


def _proposal_evidence(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[float, list[dict[str, Any]]]:
    inputs = processor(images=[image], text=[prompt], return_tensors="pt", padding=True).to(device)
    outputs = _grounder_forward(model, inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    proposals = []
    image_area = float(image.width * image.height)
    for score, label, box in zip(result["scores"], result["text_labels"], result["boxes"], strict=True):
        x1, y1, x2, y2 = [float(value) for value in box.detach().cpu().tolist()]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        bottom = max(0.0, min(1.0, y2 / image.height))
        size_support = min(1.0, area / max(0.02 * image_area, 1.0))
        bound_score = float(score) * (0.5 + 0.5 * bottom) * size_support
        proposals.append({"label": str(label), "score": float(score), "box_xyxy": [x1, y1, x2, y2], "bound_score": bound_score})
    proposals.sort(key=lambda row: row["bound_score"], reverse=True)
    return (proposals[0]["bound_score"] if proposals else 0.0), proposals[:10]


def _best_threshold(positives: list[float], negatives: list[float], nonnegative: bool = False) -> float:
    values = sorted(set(positives + negatives))
    candidates = set(values)
    candidates.update((left + right) / 2.0 for left, right in zip(values, values[1:]))
    if nonnegative:
        candidates.add(0.0)
    best = (-1.0, -float("inf"))
    for threshold in candidates:
        if nonnegative and threshold < 0.0:
            continue
        recall = sum(value >= threshold for value in positives) / len(positives)
        specificity = sum(value < threshold for value in negatives) / len(negatives)
        candidate = (0.5 * (recall + specificity), threshold)
        if candidate > best:
            best = candidate
    return float(best[1])


def _metric(rows: list[dict[str, Any]], ready_key: str) -> dict[str, Any]:
    tp = sum(row["entrance_view"] and row[ready_key] and row["identity_correct"] for row in rows)
    fp = sum(row[ready_key] and (not row["entrance_view"] or not row["identity_correct"]) for row in rows)
    fn = sum(row["entrance_view"] and not (row[ready_key] and row["identity_correct"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_ready": tp, "false_ready": fp, "missed_ready": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("named_poi_multifacet_entrance_protocol_v1.json"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts.local/evidence/l10-r0/named-poi-multifacet-entrance-v1")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reference_rows, query_rows, _, source_info = _load_rows(protocol)
    target_ids = list(protocol["targets"])
    references = {target_id: [row for row in reference_rows if row.target_id == target_id] for target_id in target_ids}
    all_rows = reference_rows + query_rows

    clip_path = ROOT / protocol["models"]["clip"]["path"]
    dino_path = ROOT / protocol["models"]["dinov2"]["path"]
    grounder_path = ROOT / protocol["models"]["grounding_dino"]["path"]
    if _sha256(clip_path / "pytorch_model.bin") != protocol["models"]["clip"]["weights_sha256"]:
        raise ValueError("CLIP_HASH_MISMATCH")
    if _sha256(dino_path / "model.safetensors") != protocol["models"]["dinov2"]["weights_sha256"]:
        raise ValueError("DINO_HASH_MISMATCH")
    if _sha256(grounder_path / "model.safetensors") != protocol["models"]["grounding_dino"]["weights_sha256"]:
        raise ValueError("GROUNDER_HASH_MISMATCH")

    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = facade.AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative_image = Image.open(all_rows[0].path).convert("RGB")
    representative = (
        clip_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
        dino_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
    )
    encoder_backend, models = facade._select_backend(clip_path, dino_path, representative, output_root / "encoder_backend_receipt.json")
    encoder_device = str(encoder_backend["selected_device_type"])
    patch_grid = int(protocol["models"]["dinov2"]["patch_grid"])
    encoded = facade._encode_images(all_rows, models, clip_processor, dino_processor, encoder_device, patch_grid, args.batch_size)
    entrance_vector = _encode_prompt(models["clip"], clip_processor, protocol["entrance_graph"]["clip_prompt"], encoder_device)

    grounder_processor = AutoProcessor.from_pretrained(grounder_path, local_files_only=True)
    grounder_backend, grounder = _select_grounder(
        grounder_path,
        grounder_processor,
        representative_image,
        protocol["models"]["grounding_dino"]["prompt"],
        output_root / "grounder_backend_receipt.json",
    )
    grounder_device = str(grounder_backend["selected_device_type"])

    scored = []
    for row in query_rows:
        image = Image.open(row.path).convert("RGB")
        identity_scores = _identity_scores(row, target_ids, references, encoded, patch_grid)
        true_index = target_ids.index(row.target_id)
        identity_margin = facade._margin(identity_scores, true_index)
        predicted_index = int(np.argmax(identity_scores))
        clip_entrance = float(encoded[row.key]["clip"] @ entrance_vector)
        proposal_score, proposals = _proposal_evidence(
            grounder,
            grounder_processor,
            image,
            protocol["models"]["grounding_dino"]["prompt"],
            grounder_device,
            float(protocol["models"]["grounding_dino"]["box_threshold"]),
            float(protocol["models"]["grounding_dino"]["text_threshold"]),
        )
        scored.append(
            {
                "query": row.key,
                "role": row.role,
                "target": row.target_id,
                "prediction": target_ids[predicted_index],
                "identity_correct": predicted_index == true_index,
                "identity_margin": identity_margin,
                "wrong_goal_margins": [facade._margin(identity_scores, index) for index in range(len(target_ids)) if index != true_index],
                "entrance_view": bool(source_info["metadata"][row.key]["entrance_view"]),
                "facet_hint": source_info["metadata"][row.key]["facet_hint"],
                "clip_entrance_score": clip_entrance,
                "proposal_score": proposal_score,
                "graph_score": 0.55 * clip_entrance + 0.45 * proposal_score,
                "proposals": proposals,
            }
        )

    calibration = [row for row in scored if row["role"] == "calibration"]
    evaluation = [row for row in scored if row["role"] == "evaluation"]
    identity_threshold = _best_threshold(
        [row["identity_margin"] for row in calibration],
        [value for row in calibration for value in row["wrong_goal_margins"]],
        nonnegative=True,
    )
    clip_threshold = _best_threshold(
        [row["clip_entrance_score"] for row in calibration if row["entrance_view"]],
        [row["clip_entrance_score"] for row in calibration if not row["entrance_view"]],
    )
    graph_threshold = _best_threshold(
        [row["graph_score"] for row in calibration if row["entrance_view"]],
        [row["graph_score"] for row in calibration if not row["entrance_view"]],
    )
    for row in evaluation:
        identity_confirmed = row["identity_correct"] and row["identity_margin"] >= identity_threshold
        row["identity_confirmed"] = identity_confirmed
        row["identity_only_ready"] = identity_confirmed
        row["clip_entrance_ready"] = identity_confirmed and row["clip_entrance_score"] >= clip_threshold
        row["entity_bound_entrance_graph"] = identity_confirmed and row["graph_score"] >= graph_threshold

    metrics = {arm: _metric(evaluation, arm) for arm in ARMS}
    identity_metrics = {
        "queries": len(evaluation),
        "top1_correct": sum(row["identity_correct"] for row in evaluation),
        "top1_accuracy": sum(row["identity_correct"] for row in evaluation) / len(evaluation),
        "confirmed_correct": sum(row["identity_confirmed"] for row in evaluation),
        "confirmation_recall": sum(row["identity_confirmed"] for row in evaluation) / len(evaluation),
        "wrong_goal_false_confirmations": sum(value >= identity_threshold for row in evaluation for value in row["wrong_goal_margins"]),
    }
    baseline = metrics["identity_only_ready"]
    successor = metrics["entity_bound_entrance_graph"]
    retained = (
        successor["false_ready"] <= 0.5 * baseline["false_ready"]
        and successor["true_ready"] >= baseline["true_ready"]
    )
    result = {
        "schema": "l10-named-poi-multifacet-entrance-result-v1",
        "protocol_sha256": _sha256(protocol_path),
        "source": source_info["source"],
        "execution_backends": {"encoder": encoder_backend, "grounder": grounder_backend},
        "ocr_calls": 0,
        "roles": {
            "targets": len(target_ids),
            "reference_images": len(reference_rows),
            "calibration_queries": len(calibration),
            "evaluation_queries": len(evaluation),
            "evaluation_entrance_views": sum(row["entrance_view"] for row in evaluation),
            "evaluation_non_entrance_views": sum(not row["entrance_view"] for row in evaluation),
        },
        "thresholds": {"identity_margin": identity_threshold, "clip_entrance": clip_threshold, "entrance_graph": graph_threshold},
        "identity_metrics": identity_metrics,
        "joint_metrics": metrics,
        "stop_condition": {
            "retain_graph": retained,
            "decision": "RETAIN_ENTITY_BOUND_ENTRANCE_GRAPH" if retained else "DO_NOT_TUNE_CHANGE_ENTRANCE_INFORMATION_SOURCE",
        },
        "claim_scope": "Fresh public metadata-defined entrance-view gating only; no exact entrance box, traversability, public access, metric localization, navigation, arrival, product, user-benefit, or safety evidence.",
        "evaluation_rows": evaluation,
    }
    output_path = output_root / "result.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_path), "identity_metrics": identity_metrics, "joint_metrics": metrics, "stop_condition": result["stop_condition"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
