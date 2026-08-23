"""Develop and confirm a CLIP candidate-level referent reranker on GroundBench."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import PIL
from PIL import Image, ImageEnhance

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0 import provider_adapter
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_facade_native_door_89 import (
    IOU_THRESHOLD,
    atomic_json,
    evaluate,
    iou,
    sha256_file,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_referent_89 import (
    CLAIM_CEILING,
    build_episode,
    run_expression_inference,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_referent_union_confirmation import (
    run_brain_stdin,
)


SCHEMA_VERSION = "groundbench_clip_candidate_verifier_v0"
DEVELOPMENT_SCHEMA_VERSION = "groundbench_clip_candidate_verifier_development_v0"
CONFIRMATION_ROSTER_SCHEMA_VERSION = "groundbench_referent_union_confirmation_roster_v0"
CLIP_WEIGHTS_SHA256 = "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
CLIP_CONFIG_SHA256 = "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770"
VARIANTS = (
    "CROP_100",
    "CROP_125",
    "FOCUS_CONTEXT_020",
    "DUAL_CROP125_FOCUS020",
)


class RunError(RuntimeError):
    pass


def expanded_box(box: Sequence[float], width: int, height: int, expansion: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(value) for value in box)
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    half_width, half_height = (x1 - x0) * expansion / 2, (y1 - y0) * expansion / 2
    left = max(0, min(width - 1, round(center_x - half_width)))
    top = max(0, min(height - 1, round(center_y - half_height)))
    right = max(left + 1, min(width, round(center_x + half_width)))
    bottom = max(top + 1, min(height, round(center_y + half_height)))
    return left, top, right, bottom


def focused_context(image: Image.Image, box: Sequence[float], outside_brightness: float = 0.2) -> Image.Image:
    pixel_box = expanded_box(box, image.width, image.height, 1.0)
    focused = ImageEnhance.Brightness(image).enhance(outside_brightness)
    focused.paste(image.crop(pixel_box), pixel_box)
    return focused


def _normalized_truth_box(item: Mapping[str, Any]) -> list[float]:
    width, height = float(item["image_width"]), float(item["image_height"])
    x0, y0, x1, y1 = item["native_mask_bbox_xyxy"]
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def _candidate_box_pixels(candidate: Mapping[str, Any], width: int, height: int) -> list[float]:
    region = candidate["region"]
    return [
        float(region["x_min"]) * width,
        float(region["y_min"]) * height,
        float(region["x_max"]) * width,
        float(region["y_max"]) * height,
    ]


def rerank_episode(episode: Mapping[str, Any], scores: Mapping[str, float]) -> dict[str, Any]:
    candidates = [dict(item) for item in episode["candidates"]]
    if set(scores) != {str(item["candidate_id"]) for item in candidates}:
        raise RunError("candidate score coverage mismatch")
    original_rank = {str(item["candidate_id"]): int(item["provider_rank"]) for item in candidates}
    ordered = sorted(candidates, key=lambda item: (-float(scores[str(item["candidate_id"])]), original_rank[str(item["candidate_id"])]))
    for rank, candidate in enumerate(ordered, start=1):
        candidate["provider_rank"] = rank
        candidate["candidate_verifier_score"] = round(float(scores[str(candidate["candidate_id"])]), 12)
    result = dict(episode)
    result["candidates"] = ordered
    return result


def ranking_evaluation(roster: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for truth, episode in zip(roster, episodes):
        target = _normalized_truth_box(truth)
        correct_ranks = []
        for candidate in episode["candidates"]:
            region = candidate["region"]
            box = [region[key] for key in ("x_min", "y_min", "x_max", "y_max")]
            if iou(box, target) >= IOU_THRESHOLD:
                correct_ranks.append(int(candidate["provider_rank"]))
        best_rank = min(correct_ranks, default=None)
        rows.append({"episode_id": episode["episode_id"], "correct_candidate_rank": best_rank})
    available = [row for row in rows if row["correct_candidate_rank"] is not None]
    return {
        "observation_count": len(rows),
        "proposal_availability": len(available),
        "recall_at_k": {
            str(k): sum(row["correct_candidate_rank"] is not None and row["correct_candidate_rank"] <= k for row in rows)
            for k in (1, 3, 5, 10)
        },
        "mean_reciprocal_rank_given_available": (
            sum(1.0 / row["correct_candidate_rank"] for row in available) / len(available) if available else None
        ),
        "rows": rows,
    }


def select_development_variant(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = metrics["PROVIDER_ORDER"]
    order = {name: index for index, name in enumerate(VARIANTS)}
    selected = max(
        VARIANTS,
        key=lambda name: (
            int(metrics[name]["recall_at_k"]["1"]),
            float(metrics[name]["mean_reciprocal_rank_given_available"] or 0.0),
            -order[name],
        ),
    )
    promising = int(metrics[selected]["recall_at_k"]["1"]) > int(baseline["recall_at_k"]["1"])
    return {
        "selected_variant": selected,
        "terminal": "CLIP_CANDIDATE_VERIFIER_DEVELOPMENT_PROMISING" if promising else "CLIP_CANDIDATE_VERIFIER_DEVELOPMENT_NOT_PROMISING",
        "confirmation_authorized": promising,
        "selection_rule": "MAX_RECALL_AT_1_THEN_MRR_THEN_FIXED_VARIANT_ORDER",
        "baseline_recall_at_1": baseline["recall_at_k"]["1"],
        "selected_recall_at_1": metrics[selected]["recall_at_k"]["1"],
    }


class ClipScorer:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if sha256_file(model_dir / "pytorch_model.bin") != CLIP_WEIGHTS_SHA256:
            raise RunError("CLIP weights missing or changed")
        if sha256_file(model_dir / "config.json") != CLIP_CONFIG_SHA256:
            raise RunError("CLIP config missing or changed")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()

    def score_episode(self, image: Image.Image, expression: str, episode: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        torch = self.torch
        text_inputs = self.processor(text=[expression], return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        component_scores: dict[str, dict[str, float]] = {}
        for candidate in episode["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            box = _candidate_box_pixels(candidate, image.width, image.height)
            views = [
                image.crop(expanded_box(box, image.width, image.height, 1.0)),
                image.crop(expanded_box(box, image.width, image.height, 1.25)),
                focused_context(image, box),
            ]
            inputs = self.processor(images=views, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarities = (image_features @ text_features.T).squeeze(1).detach().cpu().tolist()
            exact, expanded, focus = (float(value) for value in similarities)
            component_scores[candidate_id] = {
                "CROP_100": exact,
                "CROP_125": expanded,
                "FOCUS_CONTEXT_020": focus,
                "DUAL_CROP125_FOCUS020": (expanded + focus) / 2.0,
            }
        return {
            variant: {candidate_id: values[variant] for candidate_id, values in component_scores.items()}
            for variant in VARIANTS
        }


def _runtime(scorer: ClipScorer) -> dict[str, str]:
    import torch
    import transformers

    return {
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "pillow": PIL.__version__, "cuda": str(torch.version.cuda), "device": scorer.device,
        "gpu": torch.cuda.get_device_name(0) if scorer.device == "cuda" else "NONE",
    }


def _load_public_rows(roster_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    roster_doc = json.loads(roster_path.read_text(encoding="utf-8"))
    dataset_root = Path(roster_doc["dataset_root"]).resolve()
    public_rows = []
    for item in roster_doc["observations"]:
        rgb = (dataset_root / item["rgb_path"]).resolve()
        if not rgb.is_file() or rgb.stat().st_size != item["rgb_bytes"] or sha256_file(rgb) != item["rgb_sha256"]:
            raise RunError(f"frozen COCO pixel mismatch: {item['observation_id']}")
        public_rows.append(dict(item, absolute_rgb_path=str(rgb)))
    return roster_doc, public_rows


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise RunError("Development output already exists")
    roster_path, proposal_path = args.roster.resolve(), args.proposal_output.resolve()
    roster_doc, public_rows = _load_public_rows(roster_path)
    proposal_doc = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposals = proposal_doc["outputs"]
    if len(public_rows) != 89 or len(proposals) != 89:
        raise RunError("Development denominator mismatch")
    episodes = [
        build_episode(item, result["proposals"], index)
        for index, (item, result) in enumerate(zip(public_rows, proposals), start=1)
    ]
    scorer = ClipScorer(args.clip_model_dir.resolve())
    variants: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    score_rows = []
    for index, (item, episode) in enumerate(zip(public_rows, episodes), start=1):
        with Image.open(item["absolute_rgb_path"]) as opened:
            image = opened.convert("RGB")
        scores = scorer.score_episode(image, item["goal_text"], episode)
        for name in VARIANTS:
            variants[name].append(rerank_episode(episode, scores[name]))
        score_rows.append({"episode_id": episode["episode_id"], "scores": scores})
        print(f"clip-development {index}/{len(episodes)} {episode['episode_id']}", flush=True)
    metrics = {"PROVIDER_ORDER": ranking_evaluation(roster_doc["observations"], episodes)}
    metrics.update({name: ranking_evaluation(roster_doc["observations"], value) for name, value in variants.items()})
    selection = select_development_variant(metrics)
    report = {
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "role": "CONSUMED_DEVELOPMENT_ONLY",
        "roster_sha256": sha256_file(roster_path), "proposal_output_sha256": sha256_file(proposal_path),
        "clip_weights_sha256": CLIP_WEIGHTS_SHA256, "clip_config_sha256": CLIP_CONFIG_SHA256,
        "variants_frozen_before_scoring": list(VARIANTS), "runtime": _runtime(scorer),
        "provider_calls_added": 0, "teacher_calls": 0, "metrics": metrics, "selection": selection,
        "score_rows": score_rows, "claim_ceiling": CLAIM_CEILING,
    }
    report["report_sha256"] = materializer.content_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = args.formal_root.resolve()
    if formal_root.exists():
        raise RunError("formal CLIP Confirmation root already exists")
    development_path = args.development_report.resolve()
    development = json.loads(development_path.read_text(encoding="utf-8"))
    selection = development.get("selection", {})
    if not selection.get("confirmation_authorized") or selection.get("selected_variant") not in VARIANTS:
        raise RunError("Development did not authorize Confirmation")
    clip_dir = args.clip_model_dir.resolve()
    if sha256_file(clip_dir / "pytorch_model.bin") != CLIP_WEIGHTS_SHA256:
        raise RunError("CLIP weights missing or changed")
    base_lock = provider_adapter.preflight_provider(codex_exe=args.codex_exe, model_dir=args.dino_model_dir)
    lock = {
        "schema_version": "groundbench_clip_candidate_verifier_provider_lock_v0",
        "mechanical_base_provider_lock": base_lock,
        "development_report_sha256": sha256_file(development_path),
        "development_report_content_sha256": development["report_sha256"],
        "selected_variant": selection["selected_variant"],
        "clip_model_dir": str(clip_dir), "clip_weights_sha256": CLIP_WEIGHTS_SHA256,
        "arms": {"V0": "PUBLIC_EXPRESSION_PROVIDER_ORDER", "V1": "SAME_CANDIDATES_CLIP_RERANKED"},
        "success_rule": "verifier_rank1_v1 > verifier_rank1_v0 AND correct_v1 > correct_v0 AND wrong_all_v1 <= wrong_all_v0",
        "retry_count": 0, "reruns": 0, "teacher_calls": 0, "claim_ceiling": CLAIM_CEILING,
    }
    formal_root.mkdir(parents=True)
    atomic_json(formal_root / "provider-lock.json", lock)
    return lock


def paired_verdict(v0_rank: Mapping[str, Any], v1_rank: Mapping[str, Any], v0: Mapping[str, Any], v1: Mapping[str, Any]) -> dict[str, Any]:
    rank0, rank1 = int(v0_rank["recall_at_k"]["1"]), int(v1_rank["recall_at_k"]["1"])
    correct0 = int(v0["outcome_counts"].get("CORRECT_GROUNDING", 0))
    correct1 = int(v1["outcome_counts"].get("CORRECT_GROUNDING", 0))
    wrong0 = int(v0["wrong_confident_guidance_all_observations"]["numerator"])
    wrong1 = int(v1["wrong_confident_guidance_all_observations"]["numerator"])
    supported = rank1 > rank0 and correct1 > correct0 and wrong1 <= wrong0
    return {
        "verdict": "CLIP_CANDIDATE_VERIFIER_SUPPORTED" if supported else "CLIP_CANDIDATE_VERIFIER_NOT_SUPPORTED",
        "verifier_rank1_v0": rank0, "verifier_rank1_v1": rank1,
        "correct_grounding_v0": correct0, "correct_grounding_v1": correct1,
        "wrong_confident_guidance_all_v0": wrong0, "wrong_confident_guidance_all_v1": wrong1,
        "success_rule": "verifier_rank1_v1 > verifier_rank1_v0 AND correct_v1 > correct_v0 AND wrong_all_v1 <= wrong_all_v0",
    }


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve()
    if sha256_file(roster_path) != args.roster_sha256:
        raise RunError("CLIP Confirmation roster hash mismatch")
    roster_doc, public_rows = _load_public_rows(roster_path)
    if roster_doc.get("schema_version") != CONFIRMATION_ROSTER_SCHEMA_VERSION or len(public_rows) != 64:
        raise RunError("CLIP Confirmation roster contract mismatch")
    formal_root, run_dir = args.formal_root.resolve(), args.formal_root.resolve() / "run"
    lock_path = formal_root / "provider-lock.json"
    if not lock_path.is_file() or run_dir.exists():
        raise RunError("provider lock missing or Confirmation already consumed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    executable, dino_model_dir = provider_adapter.verify_provider_lock(lock["mechanical_base_provider_lock"])
    selected_variant = lock["selected_variant"]
    run_dir.mkdir()
    metadata = [
        {"id": item["observation_id"], "path": item["absolute_rgb_path"], "image_sha256": item["rgb_sha256"], "goal_text": item["goal_text"]}
        for item in public_rows
    ]
    proposal_outputs, dino_runtime = run_expression_inference(dino_model_dir, metadata)
    v0_episodes = [
        build_episode(item, output["proposals"], index)
        for index, (item, output) in enumerate(zip(public_rows, proposal_outputs), start=1)
    ]
    scorer = ClipScorer(Path(lock["clip_model_dir"]).resolve())
    v1_episodes, score_rows = [], []
    for index, (item, episode) in enumerate(zip(public_rows, v0_episodes), start=1):
        with Image.open(item["absolute_rgb_path"]) as opened:
            image = opened.convert("RGB")
        scores = scorer.score_episode(image, item["goal_text"], episode)[selected_variant]
        v1_episodes.append(rerank_episode(episode, scores))
        score_rows.append({"episode_id": episode["episode_id"], "scores": scores})
        print(f"clip-confirmation {index}/{len(v0_episodes)} {episode['episode_id']}", flush=True)
    atomic_json(run_dir / "proposal-provider-output.json", {"runtime": dino_runtime, "outputs": proposal_outputs})
    atomic_json(run_dir / "candidate-verifier-output.json", {"runtime": _runtime(scorer), "selected_variant": selected_variant, "rows": score_rows})
    arms = {"V0": v0_episodes, "V1": v1_episodes}
    atomic_json(run_dir / "public-provider-input-v0.json", {"episodes": v0_episodes})
    atomic_json(run_dir / "public-provider-input-v1.json", {"episodes": v1_episodes})
    decisions, receipts, evaluations = {}, {}, {}
    for arm in ("V0", "V1"):
        decisions[arm], receipts[arm] = run_brain_stdin(
            episodes=arms[arm], run_dir=run_dir / f"brain-{arm.lower()}", executable=executable,
            model=provider_adapter.CODEX_MODEL, reasoning_effort=provider_adapter.CODEX_REASONING_EFFORT,
            batch_size=args.batch_size,
        )
        evaluations[arm] = evaluate(roster_doc["observations"], arms[arm], decisions[arm])
    ranking = {arm: ranking_evaluation(roster_doc["observations"], arms[arm]) for arm in ("V0", "V1")}
    report = {
        "schema_version": SCHEMA_VERSION, "roster_sha256": args.roster_sha256,
        "provider_lock_sha256": sha256_file(lock_path), "truth_authority": roster_doc["truth_authority"],
        "selected_variant": selected_variant, "same_candidate_sets": True,
        "brain_provider_calls": {arm: len(receipts[arm]) for arm in ("V0", "V1")},
        "provider_in_doubt": 0, "teacher_calls": 0, "retry_count": 0, "reruns": 0,
        "brain_batch_receipts": receipts, "raw_brain_decisions": decisions,
        "ranking_evaluations": ranking, "evaluations": evaluations,
        "paired_verdict": paired_verdict(ranking["V0"], ranking["V1"], evaluations["V0"], evaluations["V1"]),
        "claim_ceiling": CLAIM_CEILING,
    }
    report["report_sha256"] = materializer.content_sha256(report)
    atomic_json(run_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    development = subparsers.add_parser("development")
    development.add_argument("--roster", type=Path, required=True)
    development.add_argument("--proposal-output", type=Path, required=True)
    development.add_argument("--clip-model-dir", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--formal-root", type=Path, required=True)
    preflight_parser.add_argument("--development-report", type=Path, required=True)
    preflight_parser.add_argument("--clip-model-dir", type=Path, required=True)
    preflight_parser.add_argument("--dino-model-dir", type=Path, required=True)
    preflight_parser.add_argument("--codex-exe", type=Path, default=Path(r"E:\codex-tools\bin\codex.exe"))
    confirmation = subparsers.add_parser("run")
    confirmation.add_argument("--formal-root", type=Path, required=True)
    confirmation.add_argument("--roster", type=Path, required=True)
    confirmation.add_argument("--roster-sha256", required=True)
    confirmation.add_argument("--batch-size", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    if args.command == "development":
        result = run_development(args)
    elif args.command == "preflight":
        result = preflight(args)
    else:
        result = run_confirmation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
