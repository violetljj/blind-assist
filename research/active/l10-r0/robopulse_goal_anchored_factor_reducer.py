from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from research_backend import (  # noqa: E402
    BackendCandidate,
    DeviceObservation,
    Workload,
    runtime_capabilities,
    select_backend,
    torch_observation,
)

LABELS = ("progress", "regression")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inference_rows(annotation_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    for source in payload:
        required = {"id", "task", "image", "image_dataset"}
        if not required.issubset(source):
            raise ValueError("MISSING_INFERENCE_FIELD")
        if not isinstance(source["image"], list) or len(source["image"]) != 8:
            raise ValueError("EXPECTED_EIGHT_IMAGES")
        rows.append({key: source[key] for key in sorted(required)})
    return rows


def _select_cohort(rows: list[dict[str, Any]], rows_per_source: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image_dataset"]), []).append(row)
    selected = []
    for source in sorted(grouped):
        ranked = sorted(
            grouped[source],
            key=lambda row: (hashlib.sha256(str(row["id"]).encode()).hexdigest(), str(row["id"])),
        )
        if len(ranked) < rows_per_source:
            raise ValueError(f"INSUFFICIENT_SOURCE_ROWS:{source}")
        selected.extend(ranked[:rows_per_source])
    return selected


def _label_token_ids(processor: Any, model_spec: dict[str, Any]) -> torch.Tensor:
    ids = []
    for label in LABELS:
        encoded = processor.tokenizer.encode(
            str(model_spec["label_tokens"][label]), add_special_tokens=False
        )
        if len(encoded) != 1:
            raise ValueError(f"LABEL_NOT_SINGLE_TOKEN:{label}:{encoded}")
        ids.append(int(encoded[0]))
    return torch.tensor(ids, dtype=torch.long)


def _build_inputs(
    processor: Any, images: list[Image.Image], prompt: str
) -> dict[str, torch.Tensor]:
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return dict(
        processor(text=[rendered], images=images, padding=True, return_tensors="pt")
    )


def _prepare_inputs(
    processor: Any,
    row: dict[str, Any],
    media_root: Path,
    algorithm: dict[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    images = [Image.open(media_root / relative).convert("RGB") for relative in row["image"]]
    prompts = {
        "baseline": str(algorithm["baseline_prompt"]).format(task=row["task"])
    }
    prompts.update(
        {
            name: str(algorithm["factor_prompt_template"]).format(
                task=row["task"], factor=description
            )
            for name, description in algorithm["factors"].items()
        }
    )
    try:
        return (
            {name: _build_inputs(processor, images, prompt) for name, prompt in prompts.items()},
            prompts,
        )
    finally:
        for image in images:
            image.close()


def _run_logits(
    model: Qwen2VLForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
    label_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    values = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**values, use_cache=False)
        logits = output.logits[0, -1]
        return logits.index_select(0, label_ids.to(logits.device)).detach()


def _observation(model: Qwen2VLForConditionalGeneration, output: torch.Tensor) -> DeviceObservation:
    return torch_observation(model=model, output=output)


def _select_backend(
    cpu_model: Qwen2VLForConditionalGeneration,
    gpu_model: Qwen2VLForConditionalGeneration | None,
    representative: dict[str, torch.Tensor],
    label_ids: torch.Tensor,
    receipt_path: Path,
) -> dict[str, Any]:
    cpu = BackendCandidate(
        "qwen2-vl-robopulse-torch-cpu",
        "cpu",
        lambda: _run_logits(cpu_model, representative, label_ids, "cpu"),
        lambda output: _observation(cpu_model, output),
    )
    gpu = None
    cpu_reason = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "qwen2-vl-robopulse-torch-cuda",
            "cuda",
            lambda: _run_logits(gpu_model, representative, label_ids, "cuda"),
            lambda output: _observation(gpu_model, output),
            torch.cuda.synchronize,
        )
    else:
        cpu_reason = "ACCELERATOR_UNAVAILABLE"
    return select_backend(
        Workload.MODEL_INFERENCE,
        cpu=cpu,
        gpu=gpu,
        cpu_reason=cpu_reason,
        record_path=receipt_path,
        warmups=0,
        repeats=1,
        capabilities=runtime_capabilities(),
    )


def _prediction(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    scores = {}
    for name, output in outputs.items():
        logp = torch.log_softmax(output.float(), dim=0)
        scores[name] = {
            "progress_log_probability": round(float(logp[0].item()), 8),
            "regression_log_probability": round(float(logp[1].item()), 8),
            "progress_margin": round(float((logp[0] - logp[1]).item()), 8),
        }
    baseline = "progress" if scores["baseline"]["progress_margin"] > 0 else "regression"
    factors = {key: value for key, value in scores.items() if key != "baseline"}
    positives = sum(value["progress_margin"] > 0 for value in factors.values())
    successor = (
        "progress"
        if factors["conflict_integrity"]["progress_margin"] > 0 and positives >= 3
        else "regression"
    )
    return {
        "baseline_prediction": baseline,
        "successor_prediction": successor,
        "positive_factor_count": positives,
        "scores": scores,
    }


def _truth(annotation_path: Path, selected_ids: set[str]) -> dict[str, str]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    truth = {}
    for row in payload:
        row_id = str(row["id"])
        if row_id not in selected_ids:
            continue
        value = str(row["conversations"][1]["value"])
        if value == "<score>+1</score>":
            truth[row_id] = "progress"
        elif value == "<score>-1</score>":
            truth[row_id] = "regression"
        else:
            raise ValueError(f"UNEXPECTED_TRUTH:{row_id}:{value}")
    return truth


def _balanced_accuracy(rows: list[dict[str, Any]], field: str) -> float:
    recalls = []
    for label in LABELS:
        members = [row for row in rows if row["truth"] == label]
        if not members:
            raise ValueError(f"MISSING_TRUTH_CLASS:{label}")
        recalls.append(sum(row[field] == label for row in members) / len(members))
    return sum(recalls) / len(recalls)


def _evaluate(
    protocol: dict[str, Any], provider: dict[str, Any], annotation_path: Path
) -> dict[str, Any]:
    predictions = provider["predictions"]
    truth = _truth(annotation_path, {row["id"] for row in predictions})
    rows = []
    for prediction in predictions:
        label = truth[prediction["id"]]
        rows.append(
            {
                "id": prediction["id"],
                "image_dataset": prediction["image_dataset"],
                "truth": label,
                "baseline_prediction": prediction["baseline_prediction"],
                "successor_prediction": prediction["successor_prediction"],
            }
        )
    baseline_balanced = _balanced_accuracy(rows, "baseline_prediction")
    successor_balanced = _balanced_accuracy(rows, "successor_prediction")
    gain = successor_balanced - baseline_balanced
    regressions = sum(
        row["baseline_prediction"] == row["truth"]
        and row["successor_prediction"] != row["truth"]
        for row in rows
    )
    gate = protocol["frozen_gate"]
    if len(rows) < int(gate["minimum_evaluable_rows"]):
        decision = protocol["decision_labels"]["not_evaluable"]
    elif (
        successor_balanced >= float(gate["minimum_successor_balanced_accuracy"])
        and gain >= float(gate["minimum_absolute_balanced_accuracy_gain"])
        and regressions <= int(gate["maximum_baseline_correct_to_successor_wrong"])
    ):
        decision = protocol["decision_labels"]["pass"]
    else:
        decision = protocol["decision_labels"]["fail"]
    factor_signs = {
        factor: sum(
            prediction["scores"][factor]["progress_margin"] > 0
            for prediction in predictions
        )
        for factor in protocol["frozen_algorithm"]["factors"]
    }
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "protocol_sha256": provider["protocol_sha256"],
        "provider_sha256": provider["provider_sha256"],
        "truth_loaded_after_provider_seal": True,
        "denominators": {
            "source_rows": int(protocol["source"]["expected_rows"]),
            "evaluable_rows": len(rows),
            "progress_truth": sum(row["truth"] == "progress" for row in rows),
            "regression_truth": sum(row["truth"] == "regression" for row in rows),
            "baseline_correct_to_successor_wrong": regressions,
        },
        "baseline": {
            "correct": sum(row["baseline_prediction"] == row["truth"] for row in rows),
            "accuracy": sum(row["baseline_prediction"] == row["truth"] for row in rows) / len(rows),
            "balanced_accuracy": baseline_balanced,
        },
        "successor": {
            "correct": sum(row["successor_prediction"] == row["truth"] for row in rows),
            "accuracy": sum(row["successor_prediction"] == row["truth"] for row in rows) / len(rows),
            "balanced_accuracy": successor_balanced,
        },
        "absolute_balanced_accuracy_gain": gain,
        "factor_progress_sign_counts": factor_signs,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }


def _load_model(
    model_spec: dict[str, Any], cache_dir: Path, device: str
) -> Qwen2VLForConditionalGeneration:
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_dir=cache_dir,
        torch_dtype=dtype,
    ).eval()
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--backend-receipt", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = protocol["source"]
    annotation_path = args.data_root / source["annotation_file"]
    archive_path = args.data_root / source["media_archive"]
    if annotation_path.stat().st_size != int(source["annotation_size_bytes"]):
        raise ValueError("ANNOTATION_SIZE_MISMATCH")
    if _sha256(annotation_path) != source["annotation_sha256"]:
        raise ValueError("ANNOTATION_HASH_MISMATCH")
    if archive_path.stat().st_size != int(source["media_archive_size_bytes"]):
        raise ValueError("MEDIA_ARCHIVE_SIZE_MISMATCH")
    if _sha256(archive_path) != source["media_archive_sha256"]:
        raise ValueError("MEDIA_ARCHIVE_HASH_MISMATCH")
    rows = _load_inference_rows(annotation_path)
    if len(rows) != int(source["expected_rows"]):
        raise ValueError("SOURCE_ROW_COUNT_MISMATCH")
    selected = _select_cohort(rows, int(protocol["cohort"]["rows_per_source"]))
    if len(selected) != int(protocol["cohort"]["expected_rows"]):
        raise ValueError("COHORT_SIZE_MISMATCH")
    media_root = args.data_root
    missing = [path for row in selected for path in row["image"] if not (media_root / path).is_file()]
    if missing:
        raise ValueError(f"MISSING_MEDIA:{len(missing)}:{missing[:8]}")

    model_spec = protocol["model"]
    processor = AutoProcessor.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_dir=args.model_cache,
        min_pixels=int(model_spec["minimum_pixels"]),
        max_pixels=int(model_spec["maximum_pixels"]),
    )
    label_ids = _label_token_ids(processor, model_spec)
    representative, _ = _prepare_inputs(
        processor, selected[0], media_root, protocol["frozen_algorithm"]
    )
    cpu_model = _load_model(model_spec, args.model_cache, "cpu")
    gpu_model = _load_model(model_spec, args.model_cache, "cuda") if torch.cuda.is_available() else None
    backend = _select_backend(
        cpu_model, gpu_model, representative["baseline"], label_ids, args.backend_receipt
    )
    selected_device = str(backend["selected_device_type"])
    selected_model = gpu_model if selected_device == "cuda" else cpu_model
    if selected_model is None:
        raise ValueError("SELECTED_MODEL_MISSING")

    predictions = []
    for index, row in enumerate(selected, start=1):
        inputs, _ = _prepare_inputs(processor, row, media_root, protocol["frozen_algorithm"])
        outputs = {
            name: _run_logits(selected_model, values, label_ids, selected_device)
            for name, values in inputs.items()
        }
        predictions.append(
            {
                "id": str(row["id"]),
                "task": str(row["task"]),
                "image_dataset": str(row["image_dataset"]),
                "image_sha256": [_sha256(media_root / path) for path in row["image"]],
                **_prediction(outputs),
            }
        )
        if index % 10 == 0 or index == len(selected):
            print(json.dumps({"completed": index, "total": len(selected)}), flush=True)

    provider = {
        "schema_version": 1,
        "provider": "L10-SC44-ROBOPULSE-GOAL-ANCHORED-FACTOR-PROVIDER",
        "protocol_sha256": _sha256(args.protocol),
        "source_admission_decision": protocol["decision_labels"]["source_admitted"],
        "annotation_sha256": _sha256(annotation_path),
        "media_archive_sha256": _sha256(archive_path),
        "execution_backend": backend,
        "model": {
            **model_spec,
            "actual_device": selected_device,
            "framework_version": torch.__version__,
            "python": platform.python_version(),
        },
        "truth_firewall": protocol["frozen_algorithm"]["truth_firewall"],
        "predictions": predictions,
    }
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider["provider_sha256"] = _sha256(args.provider_output)
    result = _evaluate(protocol, provider, annotation_path)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "baseline": result["baseline"],
                "successor": result["successor"],
                "absolute_balanced_accuracy_gain": result["absolute_balanced_accuracy_gain"],
                "factor_progress_sign_counts": result["factor_progress_sign_counts"],
                "backend": {
                    key: backend[key]
                    for key in ("selected_backend", "selected_device_type", "selection_reason")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
