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

import cv2
import numpy as np
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
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _load_source_rows_without_truth(annotation_path: Path) -> list[dict[str, Any]]:
    raw = annotation_path.read_text(encoding="utf-8")
    redacted, count = re.subn(
        r'"GT"\s*:\s*"[A-D]"\s*,|,\s*"GT"\s*:\s*"[A-D]"', "", raw
    )
    payload = json.loads(redacted)
    rows = payload.get("data")
    if not isinstance(rows, list) or count != len(rows):
        raise ValueError("TRUTH_REDACTION_COUNT_MISMATCH")
    required = {"id", "query", "query_video_path", "option_imgs_path"}
    for row in rows:
        if set(row) != required:
            raise ValueError(f"UNEXPECTED_REDACTED_SOURCE_SCHEMA:{sorted(row)}")
        if not isinstance(row["option_imgs_path"], list) or len(row["option_imgs_path"]) != 4:
            raise ValueError("EXPECTED_FOUR_CANDIDATE_IMAGES")
    return rows


def _extract_goal(query: str) -> str | None:
    matches = re.findall(r'flow of\s+"([^"]+)"', query, flags=re.IGNORECASE)
    return matches[0].strip() if len(matches) == 1 and matches[0].strip() else None


def _sample_video(path: Path, count: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise ValueError("VIDEO_HAS_NO_FRAMES")
        indices = np.rint(np.linspace(0, frame_count - 1, count)).astype(np.int64)
        frames: list[Image.Image] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"VIDEO_FRAME_DECODE_FAILED:{index}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames
    finally:
        capture.release()


def _prepare_inputs(
    processor: Any,
    row: dict[str, Any],
    media_root: Path,
    algorithm: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], str]:
    goal = _extract_goal(str(row["query"]))
    if goal is None:
        raise ValueError("GOAL_EXTRACTION_FAILED")
    video_frames = _sample_video(
        media_root / row["query_video_path"], int(algorithm["video_frame_samples"])
    )
    candidate_images = [
        Image.open(media_root / path).convert("RGB") for path in row["option_imgs_path"]
    ]
    prompts = [
        str(algorithm["desired_prompt"]).format(goal=goal),
        str(algorithm["counterfactual_prompt"]).format(goal=goal),
    ]
    video = processor(images=video_frames, return_tensors="pt")
    candidates = processor(images=candidate_images, return_tensors="pt")
    text = processor(text=prompts, return_tensors="pt", padding=True)
    tensors = {
        "video_pixel_values": video["pixel_values"],
        "candidate_pixel_values": candidates["pixel_values"],
        "input_ids": text["input_ids"],
        "attention_mask": text["attention_mask"],
    }
    return tensors, goal


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)


def _infer(
    model: CLIPModel, tensors: dict[str, torch.Tensor], device: str
) -> dict[str, torch.Tensor]:
    values = {key: value.to(device) for key, value in tensors.items()}
    with torch.inference_mode():
        video = _normalized(
            model.visual_projection(
                model.vision_model(
                    pixel_values=values["video_pixel_values"]
                ).pooler_output
            )
        )
        candidates = _normalized(
            model.visual_projection(
                model.vision_model(
                    pixel_values=values["candidate_pixel_values"]
                ).pooler_output
            )
        )
        text = _normalized(
            model.text_projection(
                model.text_model(
                    input_ids=values["input_ids"],
                    attention_mask=values["attention_mask"],
                ).pooler_output
            )
        )
    return {"video": video, "candidates": candidates, "text": text}


def _observation(model: CLIPModel, output: dict[str, torch.Tensor]) -> DeviceObservation:
    return torch_observation(model=model, output=output)


def _select_backend(
    cpu_model: CLIPModel,
    gpu_model: CLIPModel | None,
    representative: dict[str, torch.Tensor],
    receipt_path: Path,
) -> dict[str, Any]:
    capabilities = runtime_capabilities()
    cpu = BackendCandidate(
        "clip-torch-cpu",
        "cpu",
        lambda: _infer(cpu_model, representative, "cpu"),
        lambda output: _observation(cpu_model, output),
    )
    gpu = None
    cpu_reason = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "clip-torch-cuda",
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
        capabilities=capabilities,
    )


def _prediction(
    output: dict[str, torch.Tensor], algorithm: dict[str, Any]
) -> dict[str, Any]:
    video = output["video"]
    candidates = output["candidates"]
    text = output["text"]
    desired = candidates @ text[0]
    counterfactual = candidates @ text[1]
    same_scene = torch.max(candidates @ video.T, dim=1).values
    successor = desired - counterfactual + same_scene
    labels = ("A", "B", "C", "D")
    baseline_index = int(torch.argmax(desired).item())
    successor_index = int(torch.argmax(successor).item())
    return {
        "baseline_prediction": labels[baseline_index],
        "successor_prediction": labels[successor_index],
        "scores": {
            label: {
                "desired_similarity": round(float(desired[index].item()), 8),
                "counterfactual_similarity": round(
                    float(counterfactual[index].item()), 8
                ),
                "same_scene_similarity": round(float(same_scene[index].item()), 8),
                "successor_score": round(float(successor[index].item()), 8),
            }
            for index, label in enumerate(labels)
        },
        "score_weights": algorithm["score_weights"],
    }


def _evaluate(
    protocol: dict[str, Any], provider: dict[str, Any], annotation_path: Path
) -> dict[str, Any]:
    truth_rows = json.loads(annotation_path.read_text(encoding="utf-8"))["data"]
    truth = {str(row["id"]): str(row["GT"]) for row in truth_rows}
    rows = []
    for row in provider["predictions"]:
        label = truth[row["id"]]
        rows.append(
            {
                "id": row["id"],
                "goal": row["goal"],
                "truth": label,
                "baseline_prediction": row["baseline_prediction"],
                "successor_prediction": row["successor_prediction"],
                "baseline_correct": row["baseline_prediction"] == label,
                "successor_correct": row["successor_prediction"] == label,
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
    gate = protocol["frozen_gate"]
    if count < int(gate["minimum_evaluable_tasks"]):
        decision = protocol["decision_labels"]["not_evaluable"]
    elif (
        successor_accuracy >= float(gate["minimum_successor_accuracy"])
        and successor_accuracy - baseline_accuracy
        >= float(gate["minimum_absolute_accuracy_gain"])
        and regressions <= int(gate["maximum_baseline_correct_to_successor_wrong"])
    ):
        decision = protocol["decision_labels"]["pass"]
    else:
        decision = protocol["decision_labels"]["fail"]
    return {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "decision": decision,
        "protocol_sha256": protocol["protocol_sha256"],
        "provider_sha256": provider["provider_sha256"],
        "truth_loaded_after_provider_seal": True,
        "denominators": {
            "tasks_expected": int(protocol["source"]["expected_task_count"]),
            "tasks_evaluable": count,
            "baseline_correct_to_successor_wrong": regressions,
        },
        "baseline": {"correct": baseline_correct, "accuracy": baseline_accuracy},
        "successor": {"correct": successor_correct, "accuracy": successor_accuracy},
        "absolute_accuracy_gain": successor_accuracy - baseline_accuracy,
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
    protocol["protocol_sha256"] = _sha256(args.protocol)
    source = protocol["source"]
    revision = source["revision"]
    task_path = source["task_path"]
    base_url = f"https://huggingface.co/datasets/BAAI-Agents/SWITCH-Basic-v1-open/resolve/{revision}/{task_path}"
    media_root = args.data_root / task_path
    annotation_path = media_root / "vqa.json"
    _download_once(f"{base_url}/vqa.json", annotation_path)
    if annotation_path.stat().st_size != int(source["annotation_size_bytes"]):
        raise ValueError("ANNOTATION_SIZE_MISMATCH")
    rows = _load_source_rows_without_truth(annotation_path)
    if len(rows) != int(source["expected_task_count"]):
        raise ValueError("TASK_COUNT_MISMATCH")
    for row in rows:
        for relative_path in [row["query_video_path"], *row["option_imgs_path"]]:
            _download_once(f"{base_url}/{relative_path}", media_root / relative_path)

    model_spec = protocol["model"]
    processor = AutoProcessor.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_dir=args.model_cache,
    )
    cpu_model = CLIPModel.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_dir=args.model_cache,
    ).eval()
    gpu_model = None
    if torch.cuda.is_available():
        gpu_model = CLIPModel.from_pretrained(
            model_spec["model_id"],
            revision=model_spec["revision"],
            cache_dir=args.model_cache,
        ).eval().to("cuda")

    prepared = [
        _prepare_inputs(processor, row, media_root, protocol["frozen_algorithm"])
        for row in rows
    ]
    backend = _select_backend(cpu_model, gpu_model, prepared[0][0], args.backend_receipt)
    selected_model = gpu_model if backend["selected_device_type"] == "cuda" else cpu_model
    selected_device = str(backend["selected_device_type"])
    predictions = []
    for row, (tensors, goal) in zip(rows, prepared, strict=True):
        output = _infer(selected_model, tensors, selected_device)
        predictions.append(
            {
                "id": str(row["id"]),
                "goal": goal,
                "query_video_path": row["query_video_path"],
                "option_imgs_path": row["option_imgs_path"],
                **_prediction(output, protocol["frozen_algorithm"]),
            }
        )
    provider = {
        "schema_version": 1,
        "provider": "L10-SC41-SWITCH-CAUSAL-TERMINAL-VERIFICATION-PROVIDER",
        "protocol_sha256": protocol["protocol_sha256"],
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
    args.provider_output.write_text(
        json.dumps(provider, indent=2) + "\n", encoding="utf-8"
    )
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
                    for key in (
                        "selected_backend",
                        "selected_device_type",
                        "selection_reason",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
