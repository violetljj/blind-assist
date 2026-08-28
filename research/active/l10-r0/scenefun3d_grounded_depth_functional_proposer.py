from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from scenefun3d_depth_functional_proposer import (
    _depth_decision,
    _depth_multiview_candidates,
    _depth_observation,
)
from scenefun3d_functional_handoff_ceiling import _load_parent_boxes
from scenefun3d_multiview_functional_proposer import (
    MODEL_CONFIDENCE,
    MODEL_IMAGE_SIZE,
    _prepare_frames,
    _sha256,
    _single_view_candidates,
    evaluate_provider,
)


GROUNDING_PROMPT = "drawer handle. cabinet knob."
GROUNDING_TEXT_LABELS = ("handle", "knob")
GROUNDING_BATCH = 2
GROUNDING_FRAME_DECIMATION = 3
GROUNDING_TEXT_THRESHOLD = 0.25


def _is_functional_label(label: str) -> bool:
    normalized = label.casefold()
    return any(token in normalized for token in GROUNDING_TEXT_LABELS)


def build_grounded_depth_provider(
    scene_dir: Path, video_id: str, model_path: Path
) -> dict[str, Any]:
    video_dir = scene_dir / video_id
    object_boxes_path = video_dir / f"{video_id}_3dod_annotation.json"
    depth_dir = video_dir / "lowres_depth"
    parents = _load_parent_boxes(object_boxes_path)
    parent_by_id = {parent.binding_id: parent for parent in parents}
    frames = _prepare_frames(scene_dir, video_id, parents)[::GROUNDING_FRAME_DECIMATION]
    frames = [
        frame for frame in frames if (depth_dir / frame.image_path.name).is_file()
    ]
    if not frames:
        raise RuntimeError("No RGB frame has matching depth, pose, and parent geometry")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path, local_files_only=True
    ).to(device).eval()

    observations = []
    raw_grounded_detections = 0
    functional_label_detections = 0
    started = time.perf_counter()
    for start in range(0, len(frames), GROUNDING_BATCH):
        chunk = frames[start : start + GROUNDING_BATCH]
        images = []
        for frame in chunk:
            with Image.open(frame.image_path) as image:
                images.append(image.convert("RGB"))
        text = [GROUNDING_PROMPT] * len(images)
        inputs = processor(
            images=images, text=text, return_tensors="pt", padding=True
        ).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=MODEL_CONFIDENCE,
            text_threshold=GROUNDING_TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width) for image in images],
        )
        for frame, result in zip(chunk, results):
            depth = cv2.imread(
                str(depth_dir / frame.image_path.name), cv2.IMREAD_UNCHANGED
            )
            if depth is None:
                continue
            raw_grounded_detections += len(result["boxes"])
            for box, confidence, label in zip(
                result["boxes"], result["scores"], result["text_labels"]
            ):
                if not _is_functional_label(str(label)):
                    continue
                functional_label_detections += 1
                observation = _depth_observation(
                    frame,
                    tuple(float(value) for value in box.detach().cpu().tolist()),
                    float(confidence.item()),
                    parent_by_id,
                    depth,
                )
                if observation is not None:
                    observations.append(observation)
    runtime_seconds = time.perf_counter() - started

    observations_by_parent = {
        parent.binding_id: [
            observation
            for observation in observations
            if observation.parent_binding_id == parent.binding_id
        ]
        for parent in parents
    }
    single_view = {
        parent.binding_id: _single_view_candidates(
            parent.binding_id, observations_by_parent[parent.binding_id]
        )
        for parent in parents
    }
    multiview = {
        parent.binding_id: _depth_multiview_candidates(
            parent.binding_id, observations_by_parent[parent.binding_id]
        )
        for parent in parents
    }
    return {
        "schema_version": 1,
        "provider": "L10-SC11-TASK-GROUNDED-PARENT-BOUND-RGBD-MULTIVIEW-PROPOSER",
        "truth_isolation": (
            "Functional annotations, description target IDs, affordance labels, and evaluator "
            "part points are not loaded until after this provider payload is written."
        ),
        "inputs": {
            "scene_dir": str(scene_dir),
            "visit_id": scene_dir.name,
            "video_id": video_id,
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path / "model.safetensors"),
            "object_boxes_sha256": _sha256(object_boxes_path),
        },
        "frozen_parameters": {
            "grounding_prompt": GROUNDING_PROMPT,
            "grounding_text_labels": list(GROUNDING_TEXT_LABELS),
            "grounding_batch": GROUNDING_BATCH,
            "grounding_frame_decimation_after_rgb_stride": GROUNDING_FRAME_DECIMATION,
            "effective_source_frame_stride": 18,
            "model_image_size_reference": MODEL_IMAGE_SIZE,
            "box_threshold": MODEL_CONFIDENCE,
            "text_threshold": GROUNDING_TEXT_THRESHOLD,
            "lift_and_multiview": "native_depth_backprojection_then_frozen_sc10_consensus",
        },
        "runtime": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "runtime_seconds": round(runtime_seconds, 3),
            "frames_inferred": len(frames),
            "raw_grounded_detections": raw_grounded_detections,
            "functional_label_detections": functional_label_detections,
            "depth_valid_parent_bound_observations": len(observations),
        },
        "parent_observation_counts": {
            parent.binding_id: len(observations_by_parent[parent.binding_id])
            for parent in parents
        },
        "arms": {"single_view": single_view, "multiview": multiview},
    }


def _grounded_decision(result: dict[str, Any]) -> str:
    inherited = _depth_decision(result)
    if inherited == "SC10_NATIVE_DEPTH_MULTIVIEW_FUNCTIONAL_PROPOSAL_DEVELOPMENT_SIGNAL":
        return "SC11_TASK_GROUNDED_RGBD_MULTIVIEW_FUNCTIONAL_PROPOSAL_DEVELOPMENT_SIGNAL"
    if inherited.startswith("SC10_NOT_EVALUABLE"):
        return inherited.replace("SC10_", "SC11_", 1)
    return "SC11_TASK_GROUNDED_RGBD_MULTIVIEW_FUNCTIONAL_PROPOSAL_GATE_NOT_MET"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene_dir = args.scene_dir.resolve()
    provider = build_grounded_depth_provider(
        scene_dir, args.video_id, args.model.resolve()
    )
    args.provider_output.parent.mkdir(parents=True, exist_ok=True)
    args.provider_output.write_text(json.dumps(provider, indent=2) + "\n", encoding="utf-8")
    provider_sha256 = _sha256(args.provider_output)

    result = evaluate_provider(scene_dir, args.video_id, provider, provider_sha256)
    result["experiment"] = (
        "L10-SC11-SCENEFUN3D-TASK-GROUNDED-PARENT-BOUND-RGBD-MULTIVIEW-FUNCTIONAL-PROPOSER"
    )
    result["decision"] = _grounded_decision(result)
    result["claim_layer"] = (
        "PRIVILEGED_PARENT_BOUND_TASK_GROUNDED_RGBD_FUNCTIONAL_PROPOSAL_DEVELOPMENT"
    )
    result["claim_boundary"] = (
        "This is one already-opened SceneFun3D Development scene, one fixed handle/knob "
        "prompt, privileged exact 3D parent boxes, and native ARKit depth. A positive result "
        "would establish a narrow task-grounded RGB-D functional proposal signal, not general "
        "open-vocabulary functionality, phone-camera transfer, reachability, orientation, "
        "arrival, completion, user benefit, or safety."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "runtime": result["provider_runtime"],
                "single_view_proposal": result["arms"]["single_view"]["proposal"],
                "multiview_proposal": result["arms"]["multiview"]["proposal"],
                "single_view_task": {
                    key: value
                    for key, value in result["arms"]["single_view"]["task"].items()
                    if key not in {"tasks", "not_evaluable"}
                },
                "multiview_task": {
                    key: value
                    for key, value in result["arms"]["multiview"]["task"].items()
                    if key not in {"tasks", "not_evaluable"}
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
