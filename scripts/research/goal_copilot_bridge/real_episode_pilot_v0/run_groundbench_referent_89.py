"""Run the unchanged Brain decision baseline on fresh strong referent truth."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import PIL
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0 import provider_adapter
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_facade_native_door_89 import (
    atomic_json,
    evaluate,
    run_brain,
    sha256_file,
)


SCHEMA_VERSION = "groundbench_referent_89_run_v0"
ROSTER_SCHEMA_VERSION = "groundbench_referent_89_roster_v0"
CLAIM_CEILING = "STATIC_COCO_OUTDOOR_REFERENT_SELECTION_ONLY_NO_APPROACH_CONTROL_RANGE_BEARING_ARRIVAL_LOST_PERSISTENCE_SAFETY_OR_PRODUCT_CLAIM"
PROMPT_MODE = "PER_OBSERVATION_PUBLIC_REFERRING_EXPRESSION_LOWERCASE_PERIOD_TERMINATED"


class RunError(RuntimeError):
    pass


def expression_prompt(expression: str) -> str:
    value = " ".join(expression.strip().lower().split())
    if not value:
        raise RunError("empty referring expression")
    return value.rstrip(".") + " ."


def run_expression_inference(
    model_dir: Path, metadata: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    weight_path = model_dir / dino.WEIGHTS_FILENAME
    if not weight_path.is_file() or sha256_file(weight_path) != dino.WEIGHTS_SHA256:
        raise RunError("pinned Grounding DINO weights are missing or changed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True,
    ).to(device).eval()
    outputs = []
    for index, item in enumerate(metadata, start=1):
        with Image.open(item["path"]) as source:
            image = source.convert("RGB")
        prompt = expression_prompt(str(item["goal_text"]))
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            raw = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            raw, inputs.input_ids, threshold=dino.BOX_THRESHOLD, text_threshold=dino.TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        proposals = []
        for box, score, label in zip(result["boxes"], result["scores"], labels):
            proposals.append({
                "bbox_xyxy": [round(float(value), 6) for value in box.detach().cpu().tolist()],
                "score": round(float(score.detach().cpu()), 12),
                "label": str(label),
            })
        kept = dino.deterministic_nms(proposals)
        outputs.append({
            "image_id": item["id"], "image_sha256": item["image_sha256"],
            "public_expression": item["goal_text"], "grounding_prompt": prompt,
            "raw_above_threshold_count": len(proposals), "proposals": kept,
        })
        print(f"inference {index}/{len(metadata)} image={item['id']} proposals={len(kept)}", flush=True)
    return outputs, {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "pillow": PIL.__version__,
        "cuda": str(torch.version.cuda), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "NONE",
    }


def build_episode(item: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    frame_id = item["observation_id"]
    candidates = []
    for rank, proposal in enumerate(proposals, start=1):
        x0, y0, x1, y1 = proposal["bbox_xyxy"]
        candidates.append({
            "candidate_id": f"gdino-{frame_id}-{rank:03d}",
            "region": {
                "frame_id": frame_id, "coordinate_space": "NORMALIZED_XYXY",
                "x_min": x0 / item["image_width"], "y_min": y0 / item["image_height"],
                "x_max": x1 / item["image_width"], "y_max": y1 / item["image_height"],
            },
            "category_label": proposal["label"], "proposal_score": proposal["score"], "provider_rank": rank,
        })
    return {
        "episode_id": item["observation_id"], "goal_text": item["goal_text"],
        "image_path": item["absolute_rgb_path"], "candidates": candidates,
        "evaluator_episode": {
            "goal_spec": {"target_name": item["goal_text"]},
            "observation_window": {"frame_ids": [frame_id], "start_timestamp_ms": index, "end_timestamp_ms": index},
        },
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = args.formal_root.resolve()
    if formal_root.exists():
        raise RunError("formal GroundBench root already exists")
    base_lock = provider_adapter.preflight_provider(codex_exe=args.codex_exe, model_dir=args.model_dir)
    lock = {
        "schema_version": "groundbench_referent_provider_lock_v0",
        "mechanical_base_provider_lock": base_lock,
        "proposal_adapter": {
            "prompt_mode": PROMPT_MODE,
            "box_threshold": dino.BOX_THRESHOLD, "text_threshold": dino.TEXT_THRESHOLD,
            "nms_iou_threshold": dino.NMS_IOU_THRESHOLD, "max_proposals_per_image": dino.MAX_PROPOSALS_PER_IMAGE,
            "authority": "VISUAL_PROPOSAL_ONLY",
        },
        "brain_policy_id": base_lock["brain_policy_id"],
        "claim_ceiling": CLAIM_CEILING,
    }
    formal_root.mkdir(parents=True)
    atomic_json(formal_root / "provider-lock.json", lock)
    return lock


def run(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve()
    if sha256_file(roster_path) != args.roster_sha256:
        raise RunError("frozen GroundBench roster hash mismatch")
    roster_doc = json.loads(roster_path.read_text(encoding="utf-8"))
    if (
        roster_doc.get("schema_version") != ROSTER_SCHEMA_VERSION
        or roster_doc.get("provider_calls") != 0
        or roster_doc.get("teacher_calls") != 0
        or len(roster_doc.get("observations", [])) != 89
    ):
        raise RunError("frozen GroundBench roster contract mismatch")
    formal_root = args.formal_root.resolve()
    lock_path = formal_root / "provider-lock.json"
    if not lock_path.is_file() or (formal_root / "run").exists():
        raise RunError("provider lock missing or formal run already exists")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    executable, model_dir = provider_adapter.verify_provider_lock(lock["mechanical_base_provider_lock"])
    if lock["proposal_adapter"]["prompt_mode"] != PROMPT_MODE:
        raise RunError("expression proposal adapter drift")
    run_dir = formal_root / "run"
    run_dir.mkdir()

    dataset_root = Path(roster_doc["dataset_root"]).resolve()
    public_rows, metadata = [], []
    for item in roster_doc["observations"]:
        rgb = (dataset_root / item["rgb_path"]).resolve()
        if not rgb.is_file() or rgb.stat().st_size != item["rgb_bytes"] or sha256_file(rgb) != item["rgb_sha256"]:
            raise RunError(f"frozen COCO pixel mismatch: {item['observation_id']}")
        public_rows.append(dict(item, absolute_rgb_path=str(rgb)))
        metadata.append({
            "id": item["observation_id"], "path": str(rgb), "image_sha256": item["rgb_sha256"],
            "goal_text": item["goal_text"],
        })
    proposals, runtime = run_expression_inference(model_dir, metadata)
    atomic_json(run_dir / "proposal-provider-output.json", {
        "prompt_mode": PROMPT_MODE, "runtime": runtime, "outputs": proposals,
    })
    episodes = [
        build_episode(item, result["proposals"], index)
        for index, (item, result) in enumerate(zip(public_rows, proposals), start=1)
    ]
    atomic_json(run_dir / "public-provider-input.json", {"episodes": episodes})
    decisions, receipts = run_brain(
        episodes=episodes, run_dir=run_dir / "brain", executable=executable,
        model=provider_adapter.CODEX_MODEL, reasoning_effort=provider_adapter.CODEX_REASONING_EFFORT,
        batch_size=args.batch_size,
    )
    evaluation = evaluate(roster_doc["observations"], episodes, decisions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "roster_sha256": args.roster_sha256,
        "provider_lock_sha256": sha256_file(lock_path),
        "truth_authority": roster_doc["truth_authority"],
        "proposal_prompt_mode": PROMPT_MODE,
        "proposal_observations": 89,
        "brain_provider_calls": len(receipts), "brain_provider_attempts": len(receipts),
        "provider_in_doubt": 0, "teacher_calls": 0, "retry_count": 0, "reruns": 0,
        "brain_batch_receipts": receipts, "raw_brain_decisions": decisions,
        "evaluation": evaluation, "claim_ceiling": CLAIM_CEILING,
    }
    report["report_sha256"] = materializer.content_sha256(report)
    atomic_json(run_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--formal-root", type=Path, required=True)
        child.add_argument("--model-dir", type=Path, required=True)
        child.add_argument("--codex-exe", type=Path, default=Path(r"E:\codex-tools\bin\codex.exe"))
        if command == "run":
            child.add_argument("--roster", type=Path, required=True)
            child.add_argument("--roster-sha256", required=True)
            child.add_argument("--batch-size", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    result = preflight(args) if args.command == "preflight" else run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
