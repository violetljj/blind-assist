from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

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

LABELS = ("A", "B", "C")
APPROVED_INPUT_FIELDS = {"image_path", "object", "action_narration"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _download_once(url: str, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _clip_id(image_path: str) -> str:
    parts = Path(image_path).parts
    if len(parts) != 3 or parts[0] != "object-state-data":
        raise ValueError(f"UNEXPECTED_IMAGE_PATH:{image_path}")
    return parts[1]


def _load_public_inputs(split_path: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with split_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {
            "image_path",
            "caption_key",
            "caption_path",
            "qa_path",
            "conversation_path",
            "object",
            "action_narration",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("UNEXPECTED_SPLIT_SCHEMA")
        for raw in reader:
            row = {field: str(raw[field]).strip() for field in APPROVED_INPUT_FIELDS}
            groups[_clip_id(row["image_path"])].append(row)

    if len(groups) != int(protocol["source"]["public_benchmark_clip_count"]):
        raise ValueError("PUBLIC_BENCHMARK_CLIP_COUNT_MISMATCH")
    pattern = re.compile(str(protocol["source"]["eligible_action_regex"]), re.I)
    eligible: list[dict[str, Any]] = []
    for clip_id, rows in groups.items():
        objects = {row["object"] for row in rows}
        actions = {row["action_narration"] for row in rows}
        candidates = sorted(
            {
                row["image_path"]
                for row in rows
                if Path(row["image_path"]).name != "state_change.jpg"
            },
            key=lambda path: _key("SC42_OPTION_V1", clip_id, path),
        )
        if len(objects) != 1 or len(actions) != 1:
            raise ValueError(f"INCONSISTENT_PUBLIC_INPUTS:{clip_id}")
        action = next(iter(actions))
        if not pattern.search(action):
            continue
        if len(candidates) != int(protocol["source"]["candidate_frame_count"]):
            raise ValueError(f"CANDIDATE_FRAME_COUNT_MISMATCH:{clip_id}")
        eligible.append(
            {
                "clip_id": clip_id,
                "object": next(iter(objects)),
                "action_narration": action,
                "candidate_paths": candidates,
            }
        )
    if len(eligible) != int(protocol["source"]["eligible_clip_count"]):
        raise ValueError("ELIGIBLE_CLIP_COUNT_MISMATCH")
    eligible.sort(key=lambda row: _key("SC42_OSCAR_GLEA_V1", row["clip_id"]))
    return eligible[: int(protocol["source"]["expected_task_count"])]


def _action_class(action: str) -> str | None:
    normalized = action.casefold().strip()
    if normalized.startswith("open "):
        return "open"
    if normalized.startswith("close "):
        return "close"
    if normalized.startswith(("turn on ", "switch on ")):
        return "turn_on"
    if normalized.startswith(("turn off ", "switch off ")):
        return "turn_off"
    return None


def _prepare_inputs(
    processor: Any,
    row: dict[str, Any],
    media_root: Path,
    algorithm: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    action_class = _action_class(str(row["action_narration"]))
    if action_class is None:
        raise ValueError("ACTION_SCRIPT_FAILED")
    object_name = str(row["object"]).strip()
    if not object_name:
        raise ValueError("EMPTY_OBJECT")
    script = algorithm["state_scripts"][action_class]
    prompts = {
        "entity": str(algorithm["entity_prompt"]).format(object=object_name),
        "before": str(script["before"]).format(object=object_name),
        "after": str(script["after"]).format(object=object_name),
    }
    images = [Image.open(media_root / path).convert("RGB") for path in row["candidate_paths"]]
    image_inputs = processor(images=images, return_tensors="pt")
    text_inputs = processor(
        text=[prompts["entity"], prompts["before"], prompts["after"]],
        return_tensors="pt",
        padding=True,
    )
    return (
        {
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        },
        {"action_class": action_class, **prompts},
    )


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)


def _infer(
    model: CLIPModel, tensors: dict[str, torch.Tensor], device: str
) -> dict[str, torch.Tensor]:
    values = {key: value.to(device) for key, value in tensors.items()}
    with torch.inference_mode():
        vision = model.vision_model(pixel_values=values["pixel_values"])
        global_embeddings = _normalized(model.visual_projection(vision.pooler_output))
        patch_hidden = model.vision_model.post_layernorm(vision.last_hidden_state[:, 1:, :])
        patch_embeddings = _normalized(model.visual_projection(patch_hidden))
        text_embeddings = _normalized(
            model.text_projection(
                model.text_model(
                    input_ids=values["input_ids"],
                    attention_mask=values["attention_mask"],
                ).pooler_output
            )
        )
    return {"global": global_embeddings, "patches": patch_embeddings, "text": text_embeddings}


def _observation(model: CLIPModel, output: dict[str, torch.Tensor]) -> DeviceObservation:
    return torch_observation(model=model, output=output)


def _select_backend(
    cpu_model: CLIPModel,
    gpu_model: CLIPModel | None,
    representative: dict[str, torch.Tensor],
    receipt_path: Path,
) -> dict[str, Any]:
    cpu = BackendCandidate(
        "clip-patch-torch-cpu",
        "cpu",
        lambda: _infer(cpu_model, representative, "cpu"),
        lambda output: _observation(cpu_model, output),
    )
    gpu = None
    cpu_reason = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "clip-patch-torch-cuda",
            "cuda",
            lambda: _infer(gpu_model, representative, "cuda"),
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
        repeats=2,
        capabilities=runtime_capabilities(),
    )


def _argmax(values: torch.Tensor) -> int:
    return int(torch.argmax(values).item())


def _prediction(
    output: dict[str, torch.Tensor], algorithm: dict[str, Any]
) -> dict[str, Any]:
    global_embeddings = output["global"]
    patches = output["patches"]
    entity, before, after = output["text"]
    effect_axis = _normalized((after - before).unsqueeze(0))[0]
    baseline_scores = global_embeddings @ after
    global_effect = global_embeddings @ effect_axis
    entity_scores = patches @ entity
    patch_effect = patches @ effect_axis
    top_k = int(algorithm["patch_top_k"])
    entity_indices = torch.topk(entity_scores, k=top_k, dim=1).indices
    localized_effect = torch.gather(patch_effect, 1, entity_indices).mean(dim=1)
    baseline_index = _argmax(baseline_scores)
    successor_index = max(
        range(len(LABELS)),
        key=lambda index: (
            float(localized_effect[index].item()),
            float(global_effect[index].item()),
            -index,
        ),
    )
    return {
        "baseline_prediction": LABELS[baseline_index],
        "successor_prediction": LABELS[successor_index],
        "scores": {
            label: {
                "desired_state_similarity": round(float(baseline_scores[index].item()), 8),
                "localized_effect_projection": round(float(localized_effect[index].item()), 8),
                "global_effect_projection": round(float(global_effect[index].item()), 8),
            }
            for index, label in enumerate(LABELS)
        },
    }


def _load_truth(split_path: Path) -> dict[str, str]:
    truth: dict[str, str] = {}
    with split_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row["caption_key"]) == "Frame_3":
                truth[_clip_id(str(row["image_path"]))] = str(row["image_path"])
    return truth


def _evaluate(
    protocol: dict[str, Any], provider: dict[str, Any], split_path: Path, media_root: Path
) -> dict[str, Any]:
    truth_paths = _load_truth(split_path)
    rows = []
    for prediction in provider["predictions"]:
        truth_path = truth_paths[prediction["clip_id"]]
        truth_hash = _sha256(media_root / truth_path)
        truth_label = next(
            label
            for label, digest in prediction["candidate_sha256"].items()
            if digest == truth_hash
        )
        rows.append(
            {
                "clip_id": prediction["clip_id"],
                "action_class": prediction["action_class"],
                "truth": truth_label,
                "baseline_prediction": prediction["baseline_prediction"],
                "successor_prediction": prediction["successor_prediction"],
                "baseline_correct": prediction["baseline_prediction"] == truth_label,
                "successor_correct": prediction["successor_prediction"] == truth_label,
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
    by_action = {}
    for action_class in sorted({row["action_class"] for row in rows}):
        subset = [row for row in rows if row["action_class"] == action_class]
        by_action[action_class] = {
            "count": len(subset),
            "baseline_correct": sum(row["baseline_correct"] for row in subset),
            "successor_correct": sum(row["successor_correct"] for row in subset),
        }
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
        "by_action": by_action,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }


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
    revision = str(source["revision"])
    base_url = f"https://huggingface.co/datasets/ali-vosoughi/oscar-dataset/resolve/{revision}"
    split_path = args.data_root / str(source["split_path"])
    _download_once(f"{base_url}/{source['split_path']}", split_path)
    if split_path.stat().st_size != int(source["split_size_bytes"]):
        raise ValueError("SPLIT_SIZE_MISMATCH")
    if _sha256(split_path) != str(source["split_sha256"]):
        raise ValueError("SPLIT_HASH_MISMATCH")
    rows = _load_public_inputs(split_path, protocol)
    if len(rows) != int(source["expected_task_count"]):
        raise ValueError("TASK_COUNT_MISMATCH")
    media_root = args.data_root / "data"
    for row in rows:
        for relative_path in row["candidate_paths"]:
            _download_once(f"{base_url}/data/{relative_path}", media_root / relative_path)

    model_spec = protocol["model"]
    processor = AutoProcessor.from_pretrained(
        model_spec["model_id"], revision=model_spec["revision"], cache_dir=args.model_cache
    )
    cpu_model = CLIPModel.from_pretrained(
        model_spec["model_id"], revision=model_spec["revision"], cache_dir=args.model_cache
    ).eval()
    gpu_model = None
    if torch.cuda.is_available():
        gpu_model = CLIPModel.from_pretrained(
            model_spec["model_id"], revision=model_spec["revision"], cache_dir=args.model_cache
        ).eval().to("cuda")

    prepared = [
        _prepare_inputs(processor, row, media_root, protocol["frozen_algorithm"])
        for row in rows
    ]
    backend = _select_backend(cpu_model, gpu_model, prepared[0][0], args.backend_receipt)
    selected_model = gpu_model if backend["selected_device_type"] == "cuda" else cpu_model
    selected_device = str(backend["selected_device_type"])
    predictions = []
    for row, (tensors, prompts) in zip(rows, prepared, strict=True):
        output = _infer(selected_model, tensors, selected_device)
        candidate_hashes = {
            label: _sha256(media_root / path)
            for label, path in zip(LABELS, row["candidate_paths"], strict=True)
        }
        predictions.append(
            {
                "clip_id": row["clip_id"],
                "object": row["object"],
                "action_narration": row["action_narration"],
                "action_class": prompts["action_class"],
                "prompts": {
                    key: prompts[key] for key in ("entity", "before", "after")
                },
                "candidate_sha256": candidate_hashes,
                **_prediction(output, protocol["frozen_algorithm"]),
            }
        )
    provider = {
        "schema_version": 1,
        "provider": "L10-SC42-OSCAR-GOAL-LOCALIZED-EFFECT-AXIS-PROVIDER",
        "protocol_sha256": protocol_sha256,
        "source_admission_decision": protocol["decision_labels"]["source_admitted"],
        "split_sha256": _sha256(split_path),
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
    result = _evaluate(protocol, provider, split_path, media_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "denominators": result["denominators"],
                "baseline": result["baseline"],
                "successor": result["successor"],
                "absolute_accuracy_gain": result["absolute_accuracy_gain"],
                "by_action": result["by_action"],
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
