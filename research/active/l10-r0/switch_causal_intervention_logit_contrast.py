from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import urllib.request
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

LABELS = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_once(url: str, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _load_rows_without_truth(annotation_path: Path) -> list[dict[str, Any]]:
    raw = annotation_path.read_text(encoding="utf-8")
    redacted, count = re.subn(
        r'"GT"\s*:\s*"[A-D]"\s*,|,\s*"GT"\s*:\s*"[A-D]"', "", raw
    )
    payload = json.loads(redacted)
    rows = payload.get("data")
    if not isinstance(rows, list) or count != len(rows):
        raise ValueError("TRUTH_REDACTION_COUNT_MISMATCH")
    required = {"id", "query", "query_img_path", "option_imgs_path"}
    for row in rows:
        if set(row) != required:
            raise ValueError(f"UNEXPECTED_REDACTED_SOURCE_SCHEMA:{sorted(row)}")
        if not isinstance(row["option_imgs_path"], list) or len(row["option_imgs_path"]) != 4:
            raise ValueError("EXPECTED_FOUR_CANDIDATE_IMAGES")
    return rows


def _extract_action(query: str) -> str | None:
    matches = re.findall(r'action\s+"([^"]+)"', query, flags=re.IGNORECASE)
    return matches[0].strip() if len(matches) == 1 and matches[0].strip() else None


def _build_prompt_inputs(
    processor: Any, images: list[Image.Image], prompt: str
) -> dict[str, torch.Tensor]:
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return dict(
        processor(
            text=[rendered],
            images=images,
            padding=True,
            return_tensors="pt",
        )
    )


def _prepare_inputs(
    processor: Any,
    row: dict[str, Any],
    media_root: Path,
    algorithm: dict[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], str]:
    action = _extract_action(str(row["query"]))
    if action is None:
        raise ValueError("ACTION_EXTRACTION_FAILED")
    images = [Image.open(media_root / row["query_img_path"]).convert("RGB")]
    images.extend(
        Image.open(media_root / path).convert("RGB") for path in row["option_imgs_path"]
    )
    prompts = {
        "plain": str(algorithm["plain_prompt"]).format(action=action),
        "causal": str(algorithm["causal_prompt"]).format(action=action),
        "no_action": str(algorithm["no_action_prompt"]).format(action=action),
    }
    return (
        {
            name: _build_prompt_inputs(processor, images, prompt)
            for name, prompt in prompts.items()
        },
        action,
    )


def _label_token_ids(processor: Any, model_spec: dict[str, Any]) -> torch.Tensor:
    ids = []
    for label in LABELS:
        token = str(model_spec["label_tokens"][label])
        encoded = processor.tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"LABEL_NOT_SINGLE_TOKEN:{label}:{encoded}")
        ids.append(int(encoded[0]))
    return torch.tensor(ids, dtype=torch.long)


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
        selected = logits.index_select(0, label_ids.to(logits.device))
    return selected


def _infer_triplet(
    model: Qwen2VLForConditionalGeneration,
    prepared: dict[str, dict[str, torch.Tensor]],
    label_ids: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    return {
        name: _run_logits(model, inputs, label_ids, device)
        for name, inputs in prepared.items()
    }


def _observation(
    model: Qwen2VLForConditionalGeneration, output: dict[str, torch.Tensor]
) -> DeviceObservation:
    return torch_observation(model=model, output=output)


def _select_backend(
    cpu_model: Qwen2VLForConditionalGeneration,
    gpu_model: Qwen2VLForConditionalGeneration | None,
    representative: dict[str, dict[str, torch.Tensor]],
    label_ids: torch.Tensor,
    receipt_path: Path,
) -> dict[str, Any]:
    cpu = BackendCandidate(
        "qwen2-vl-causal-torch-cpu",
        "cpu",
        lambda: _infer_triplet(cpu_model, representative, label_ids, "cpu"),
        lambda output: _observation(cpu_model, output),
    )
    gpu = None
    cpu_reason = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "qwen2-vl-causal-torch-cuda",
            "cuda",
            lambda: _infer_triplet(gpu_model, representative, label_ids, "cuda"),
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


def _prediction(output: dict[str, torch.Tensor]) -> dict[str, Any]:
    plain = torch.log_softmax(output["plain"].float(), dim=0)
    causal = torch.log_softmax(output["causal"].float(), dim=0)
    no_action = torch.log_softmax(output["no_action"].float(), dim=0)
    contrast = causal - no_action
    baseline_index = int(torch.argmax(plain).item())
    successor_index = int(torch.argmax(contrast).item())
    return {
        "baseline_prediction": LABELS[baseline_index],
        "successor_prediction": LABELS[successor_index],
        "scores": {
            label: {
                "plain_log_probability": round(float(plain[index].item()), 8),
                "causal_log_probability": round(float(causal[index].item()), 8),
                "no_action_log_probability": round(float(no_action[index].item()), 8),
                "intervention_logit_contrast": round(float(contrast[index].item()), 8),
            }
            for index, label in enumerate(LABELS)
        },
    }


def _evaluate(
    protocol: dict[str, Any], provider: dict[str, Any], annotation_path: Path
) -> dict[str, Any]:
    truth_rows = json.loads(annotation_path.read_text(encoding="utf-8"))["data"]
    truth = {str(row["id"]): str(row["GT"]) for row in truth_rows}
    rows = []
    for prediction in provider["predictions"]:
        label = truth[prediction["id"]]
        rows.append(
            {
                "id": prediction["id"],
                "truth": label,
                "baseline_prediction": prediction["baseline_prediction"],
                "successor_prediction": prediction["successor_prediction"],
                "baseline_correct": prediction["baseline_prediction"] == label,
                "successor_correct": prediction["successor_prediction"] == label,
            }
        )
    count = len(rows)
    baseline_correct = sum(row["baseline_correct"] for row in rows)
    successor_correct = sum(row["successor_correct"] for row in rows)
    regressions = sum(
        row["baseline_correct"] and not row["successor_correct"] for row in rows
    )
    baseline_accuracy = baseline_correct / count if count else 0.0
    successor_accuracy = successor_correct / count if count else 0.0
    gain = successor_accuracy - baseline_accuracy
    gate = protocol["frozen_gate"]
    if count < int(gate["minimum_evaluable_tasks"]):
        decision = protocol["decision_labels"]["not_evaluable"]
    elif (
        successor_accuracy >= float(gate["minimum_successor_accuracy"])
        and gain >= float(gate["minimum_absolute_accuracy_gain"])
        and regressions <= int(gate["maximum_baseline_correct_to_successor_wrong"])
    ):
        decision = protocol["decision_labels"]["pass"]
    else:
        decision = protocol["decision_labels"]["fail"]
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "protocol_sha256": provider["protocol_sha256"],
        "provider_sha256": provider["provider_sha256"],
        "truth_loaded_after_provider_seal": True,
        "denominators": {
            "tasks_expected": int(protocol["source"]["expected_task_count"]),
            "tasks_evaluable": count,
            "baseline_correct_to_successor_wrong": regressions,
        },
        "baseline": {"correct": baseline_correct, "accuracy": baseline_accuracy},
        "successor": {"correct": successor_correct, "accuracy": successor_accuracy},
        "absolute_accuracy_gain": gain,
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
    protocol_sha256 = _sha256(args.protocol)
    source = protocol["source"]
    task_path = str(source["task_path"])
    base_url = (
        "https://huggingface.co/datasets/BAAI-Agents/SWITCH-Basic-v1-open/resolve/"
        f"{source['revision']}/{task_path}"
    )
    media_root = args.data_root / task_path
    annotation_path = media_root / "vqa.json"
    _download_once(f"{base_url}/vqa.json", annotation_path)
    if annotation_path.stat().st_size != int(source["annotation_size_bytes"]):
        raise ValueError("ANNOTATION_SIZE_MISMATCH")
    if _sha256(annotation_path) != str(source["annotation_sha256"]):
        raise ValueError("ANNOTATION_HASH_MISMATCH")
    rows = _load_rows_without_truth(annotation_path)
    if len(rows) != int(source["expected_task_count"]):
        raise ValueError("TASK_COUNT_MISMATCH")
    for row in rows:
        for relative_path in [row["query_img_path"], *row["option_imgs_path"]]:
            _download_once(f"{base_url}/{relative_path}", media_root / relative_path)

    model_spec = protocol["model"]
    processor = AutoProcessor.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_dir=args.model_cache,
        min_pixels=int(model_spec["minimum_pixels"]),
        max_pixels=int(model_spec["maximum_pixels"]),
    )
    label_ids = _label_token_ids(processor, model_spec)
    prepared = [
        _prepare_inputs(processor, row, media_root, protocol["frozen_algorithm"])
        for row in rows
    ]
    cpu_model = _load_model(model_spec, args.model_cache, "cpu")
    gpu_model = None
    if torch.cuda.is_available():
        gpu_model = _load_model(model_spec, args.model_cache, "cuda")
    backend = _select_backend(
        cpu_model, gpu_model, prepared[0][0], label_ids, args.backend_receipt
    )
    selected_model = gpu_model if backend["selected_device_type"] == "cuda" else cpu_model
    selected_device = str(backend["selected_device_type"])
    predictions = []
    for row, (inputs, action) in zip(rows, prepared, strict=True):
        logits = _infer_triplet(selected_model, inputs, label_ids, selected_device)
        predictions.append(
            {
                "id": str(row["id"]),
                "action": action,
                "query_img_sha256": _sha256(media_root / row["query_img_path"]),
                "option_img_sha256": {
                    label: _sha256(media_root / path)
                    for label, path in zip(LABELS, row["option_imgs_path"], strict=True)
                },
                **_prediction(logits),
            }
        )
    provider = {
        "schema_version": 1,
        "provider": "L10-SC43-SWITCH-CAUSAL-INTERVENTION-LOGIT-CONTRAST-PROVIDER",
        "protocol_sha256": protocol_sha256,
        "source_admission_decision": protocol["decision_labels"]["source_admitted"],
        "annotation_sha256": _sha256(annotation_path),
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
                "absolute_accuracy_gain": result["absolute_accuracy_gain"],
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
