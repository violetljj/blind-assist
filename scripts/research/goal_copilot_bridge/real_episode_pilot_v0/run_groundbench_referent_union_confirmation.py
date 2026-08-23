"""Paired untouched confirmation of a public-domain parent-proposal union."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import PIL
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0 import provider_adapter
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_silver_b_brain_baseline as brain
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_facade_native_door_89 import (
    atomic_json,
    evaluate,
    sha256_file,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_referent_89 import (
    CLAIM_CEILING,
    build_episode,
    expression_prompt,
)


SCHEMA_VERSION = "groundbench_referent_union_confirmation_v0"
ROSTER_SCHEMA_VERSION = "groundbench_referent_union_confirmation_roster_v0"
DOMAIN_LEXICON = (
    "airplane", "backpack", "bench", "bicycle", "boat", "bus", "car", "fire hydrant", "handbag", "kite",
    "motorcycle", "parking meter", "skis", "snowboard", "skateboard", "stop sign", "suitcase", "surfboard",
    "tennis racket", "tie", "traffic light", "train", "truck", "umbrella",
)
DOMAIN_LEXICON_PROMPT = " . ".join(DOMAIN_LEXICON) + " ."
UNION_MAX_CANDIDATES = 10


class RunError(RuntimeError):
    pass


def brain_command(
    executable: Path, schema_path: Path, raw_path: Path, rendered: Sequence[Path], model: str, reasoning_effort: str,
) -> list[str]:
    command = [
        str(executable), "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
        "--json", "--color", "never", "--sandbox", "read-only", "--model", model,
        "-c", f'model_reasoning_effort="{reasoning_effort}"', "--output-schema", str(schema_path),
        "--output-last-message", str(raw_path),
    ]
    for path in rendered:
        command.extend(["--image", str(path)])
    return command


def run_brain_stdin(
    *, episodes: Sequence[Mapping[str, Any]], run_dir: Path, executable: Path, model: str,
    reasoning_effort: str, batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schema_path = run_dir / "brain-output-schema.json"
    atomic_json(schema_path, brain._schema(brain.POLICY_ID))
    aliases = [(f"case-{index:03d}", episode) for index, episode in enumerate(episodes, start=1)]
    rendered = {}
    for case_id, episode in aliases:
        path = run_dir / "brain-inputs" / f"{case_id}.jpg"
        brain._render_input(episode, case_id, path)
        rendered[case_id] = path
    decisions, receipts = [], []
    for offset in range(0, len(aliases), batch_size):
        batch = aliases[offset:offset + batch_size]
        batch_id = f"batch-{offset // batch_size + 1:03d}"
        batch_dir = run_dir / "batches" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        prompt = brain._prompt(batch, brain.POLICY_ID)
        (batch_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        raw_path = batch_dir / "last-message.json"
        command = brain_command(
            executable, schema_path, raw_path, [rendered[case_id] for case_id, _ in batch], model, reasoning_effort,
        )
        if len(subprocess.list2cmdline(command)) >= 16_000 or prompt in command:
            raise RunError("stdin transport preflight failed")
        atomic_json(batch_dir / "dispatch.json", {
            "status": "DISPATCH_STARTED", "attempt": 1, "transport": "STDIN",
            "prompt_bytes": len(prompt.encode("utf-8")), "case_ids": [case_id for case_id, _ in batch],
        })
        result = subprocess.run(
            command, cwd=batch_dir, shell=False, capture_output=True, text=True, input=prompt,
            encoding="utf-8", errors="replace", timeout=900,
        )
        (batch_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (batch_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0 or not raw_path.is_file():
            atomic_json(batch_dir / "completion.json", {"status": "IN_DOUBT", "returncode": result.returncode})
            raise RunError(f"{batch_id} provider call is in_doubt; no retry permitted")
        try:
            parsed = json.loads(raw_path.read_text(encoding="utf-8"))
            validated = brain._validate_raw(parsed, batch, brain.POLICY_ID)
        except (json.JSONDecodeError, brain.BrainRunError) as error:
            atomic_json(batch_dir / "completion.json", {"status": "INVALID_OUTPUT", "error": str(error)})
            raise RunError(f"{batch_id} invalid provider output; no retry permitted") from error
        decisions.extend(validated)
        receipt = {
            "status": "RUN_SUCCESS", "attempt": 1, "transport": "STDIN", "episode_count": len(batch),
            "response_sha256": materializer.content_sha256(parsed),
        }
        atomic_json(batch_dir / "completion.json", receipt)
        receipts.append({"batch_id": batch_id, **receipt})
        print(f"{batch_id} complete {len(batch)} transport=STDIN", flush=True)
    return decisions, receipts


def paired_verdict(v0: Mapping[str, Any], v1: Mapping[str, Any]) -> dict[str, Any]:
    v0_proposal = int(v0["proposal_availability"]["numerator"])
    v1_proposal = int(v1["proposal_availability"]["numerator"])
    v0_correct = int(v0["outcome_counts"].get("CORRECT_GROUNDING", 0))
    v1_correct = int(v1["outcome_counts"].get("CORRECT_GROUNDING", 0))
    v0_wrong = int(v0["wrong_confident_guidance_all_observations"]["numerator"])
    v1_wrong = int(v1["wrong_confident_guidance_all_observations"]["numerator"])
    supported = v1_proposal > v0_proposal and v1_correct > v0_correct and v1_wrong <= v0_wrong
    return {
        "verdict": "DOMAIN_LEXICON_PROPOSAL_UNION_SUPPORTED" if supported else "DOMAIN_LEXICON_PROPOSAL_UNION_NOT_SUPPORTED",
        "proposal_availability_v0": v0_proposal, "proposal_availability_v1": v1_proposal,
        "correct_grounding_v0": v0_correct, "correct_grounding_v1": v1_correct,
        "wrong_confident_guidance_all_v0": v0_wrong, "wrong_confident_guidance_all_v1": v1_wrong,
        "success_rule": "proposal_v1 > proposal_v0 AND correct_v1 > correct_v0 AND wrong_all_v1 <= wrong_all_v0",
    }


def _infer(processor: Any, model: Any, image: Image.Image, prompt: str, device: str, source: str) -> list[dict[str, Any]]:
    import torch

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        raw = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        raw, inputs.input_ids, threshold=dino.BOX_THRESHOLD, text_threshold=dino.TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]
    labels = result.get("text_labels", result.get("labels", []))
    return [{
        "bbox_xyxy": [round(float(value), 6) for value in box.detach().cpu().tolist()],
        "score": round(float(score.detach().cpu()), 12), "label": str(label), "proposal_source": source,
    } for box, score, label in zip(result["boxes"], result["scores"], labels)]


def run_dual_inference(model_dir: Path, metadata: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
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
        with Image.open(item["path"]) as opened:
            image = opened.convert("RGB")
        expression = _infer(processor, model, image, expression_prompt(item["goal_text"]), device, "PUBLIC_EXPRESSION")
        lexicon = _infer(processor, model, image, DOMAIN_LEXICON_PROMPT, device, "FIXED_PUBLIC_DOMAIN_LEXICON")
        v0 = dino.deterministic_nms(expression)
        v1 = dino.deterministic_nms(expression + lexicon)[:UNION_MAX_CANDIDATES]
        outputs.append({
            "image_id": item["id"], "image_sha256": item["image_sha256"],
            "public_expression": item["goal_text"], "expression_prompt": expression_prompt(item["goal_text"]),
            "domain_lexicon_prompt": DOMAIN_LEXICON_PROMPT,
            "v0_expression_proposals": v0, "v1_union_proposals": v1,
            "expression_raw_count": len(expression), "lexicon_raw_count": len(lexicon),
        })
        print(f"inference {index}/{len(metadata)} image={item['id']} v0={len(v0)} v1={len(v1)}", flush=True)
    return outputs, {
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "pillow": PIL.__version__, "cuda": str(torch.version.cuda), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "NONE",
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = args.formal_root.resolve()
    if formal_root.exists():
        raise RunError("formal union Confirmation root already exists")
    base_lock = provider_adapter.preflight_provider(codex_exe=args.codex_exe, model_dir=args.model_dir)
    lock = {
        "schema_version": "groundbench_referent_union_provider_lock_v0",
        "mechanical_base_provider_lock": base_lock,
        "arms": {
            "V0": "PUBLIC_EXPRESSION_PROPOSALS",
            "V1": "PUBLIC_EXPRESSION_PLUS_FIXED_DOMAIN_LEXICON_PROPOSAL_UNION_TOP10",
        },
        "domain_lexicon": DOMAIN_LEXICON,
        "domain_lexicon_prompt": DOMAIN_LEXICON_PROMPT,
        "union_max_candidates": UNION_MAX_CANDIDATES,
        "union_budget_selection": "MINIMUM_K_WITH_DEVELOPMENT_UNION_RECALL_ABOVE_V0_K10_55_VS_54",
        "arm_order": ["V0", "V1"],
        "success_rule": "proposal_v1 > proposal_v0 AND correct_v1 > correct_v0 AND wrong_all_v1 <= wrong_all_v0",
        "retry_count": 0, "reruns": 0, "teacher_calls": 0,
        "claim_ceiling": CLAIM_CEILING,
    }
    formal_root.mkdir(parents=True)
    atomic_json(formal_root / "provider-lock.json", lock)
    return lock


def run(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve()
    if sha256_file(roster_path) != args.roster_sha256:
        raise RunError("union Confirmation roster hash mismatch")
    roster_doc = json.loads(roster_path.read_text(encoding="utf-8"))
    if (
        roster_doc.get("schema_version") != ROSTER_SCHEMA_VERSION
        or len(roster_doc.get("observations", [])) != 64
        or roster_doc.get("provider_calls") != 0
    ):
        raise RunError("union Confirmation roster contract mismatch")
    formal_root = args.formal_root.resolve()
    lock_path = formal_root / "provider-lock.json"
    run_dir = formal_root / "run"
    if not lock_path.is_file() or run_dir.exists():
        raise RunError("provider lock missing or union Confirmation already consumed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    executable, model_dir = provider_adapter.verify_provider_lock(lock["mechanical_base_provider_lock"])
    if (
        tuple(lock["domain_lexicon"]) != DOMAIN_LEXICON
        or lock["union_max_candidates"] != UNION_MAX_CANDIDATES
        or lock["arm_order"] != ["V0", "V1"]
    ):
        raise RunError("frozen union policy drift")
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
    provider_outputs, runtime = run_dual_inference(model_dir, metadata)
    atomic_json(run_dir / "proposal-provider-output.json", {"runtime": runtime, "outputs": provider_outputs})
    episodes_by_arm = {
        "V0": [build_episode(item, output["v0_expression_proposals"], index) for index, (item, output) in enumerate(zip(public_rows, provider_outputs), start=1)],
        "V1": [build_episode(item, output["v1_union_proposals"], index) for index, (item, output) in enumerate(zip(public_rows, provider_outputs), start=1)],
    }
    atomic_json(run_dir / "public-provider-input-v0.json", {"episodes": episodes_by_arm["V0"]})
    atomic_json(run_dir / "public-provider-input-v1.json", {"episodes": episodes_by_arm["V1"]})
    decisions, receipts, evaluations = {}, {}, {}
    for arm in ("V0", "V1"):
        decisions[arm], receipts[arm] = run_brain_stdin(
            episodes=episodes_by_arm[arm], run_dir=run_dir / f"brain-{arm.lower()}", executable=executable,
            model=provider_adapter.CODEX_MODEL, reasoning_effort=provider_adapter.CODEX_REASONING_EFFORT,
            batch_size=args.batch_size,
        )
        evaluations[arm] = evaluate(roster_doc["observations"], episodes_by_arm[arm], decisions[arm])
    report = {
        "schema_version": SCHEMA_VERSION, "roster_sha256": args.roster_sha256,
        "provider_lock_sha256": sha256_file(lock_path), "truth_authority": roster_doc["truth_authority"],
        "proposal_passes_per_observation": {"V0": 1, "V1": 2},
        "v1_union_max_candidates": UNION_MAX_CANDIDATES,
        "brain_provider_calls": {arm: len(receipts[arm]) for arm in ("V0", "V1")},
        "provider_in_doubt": 0, "teacher_calls": 0, "retry_count": 0, "reruns": 0,
        "brain_batch_receipts": receipts, "raw_brain_decisions": decisions,
        "evaluations": evaluations, "paired_verdict": paired_verdict(evaluations["V0"], evaluations["V1"]),
        "claim_ceiling": CLAIM_CEILING,
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
