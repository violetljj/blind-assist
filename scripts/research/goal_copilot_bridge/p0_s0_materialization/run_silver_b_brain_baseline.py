"""Run one existing multimodal Codex Brain on a reviewed Silver-B cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from scripts.research.goal_copilot_bridge.p0_grounding import p0_evaluator
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


POLICY_ID = "P0-SILVER-B-SINGLE-BRAIN-BASELINE-V1"
ACTIONS = {"SELECT", "AMBIGUOUS", "ABSTAIN"}


class BrainRunError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BrainRunError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _render_input(episode: Mapping[str, Any], model_case_id: str, output_path: Path) -> None:
    with Image.open(episode["image_path"]) as opened:
        image = opened.convert("RGB")
    header_height = 112
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font, font = _font(24), _font(18)
    draw.text((8, 7), model_case_id, fill=(0, 0, 0), font=font)
    goal = str(episode["goal_text"])
    draw.text((8, 37), goal[:105], fill=(0, 0, 0), font=title_font)
    draw.text((8, 76), "Blue boxes are proposal candidates; labels are candidate rank numbers.", fill=(0, 0, 0), font=font)
    frame_id = str(episode["evaluator_episode"]["observation_window"]["frame_ids"][0])
    for candidate in episode["candidates"]:
        region = candidate["region"]
        _require(region["frame_id"] == frame_id, "candidate frame mismatch")
        box = [
            round(float(region["x_min"]) * image.width),
            round(float(region["y_min"]) * image.height) + header_height,
            round(float(region["x_max"]) * image.width),
            round(float(region["y_max"]) * image.height) + header_height,
        ]
        rank = str(candidate["provider_rank"])
        draw.rectangle(box, outline=(0, 110, 255), width=4)
        draw.rectangle((box[0], box[1], box[0] + 12 + 12 * len(rank), box[1] + 25), fill=(255, 255, 255))
        draw.text((box[0] + 3, box[1] + 2), rank, fill=(0, 70, 180), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95)


def _schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["episode_id", "action", "selected_candidate_ids", "confidence", "rationale"],
                    "properties": {
                        "episode_id": {"type": "string"},
                        "action": {"type": "string", "enum": sorted(ACTIONS)},
                        "selected_candidate_ids": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"},
                    },
                },
            }
        },
    }


def _prompt(batch: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    episode_specs = []
    for model_case_id, episode in batch:
        episode_specs.append({
            "episode_id": model_case_id,
            "goal": episode["goal_text"],
            "candidates": [
                {
                    "candidate_id": item["candidate_id"],
                    "rank_label_in_image": item["provider_rank"],
                    "category": item["category_label"],
                    "normalized_xyxy": [item["region"][key] for key in ("x_min", "y_min", "x_max", "y_max")],
                }
                for item in episode["candidates"]
            ],
        })
    return (
        "You are the single existing VLM Brain baseline for conditioned entrance-candidate selection. "
        "Use only the attached score-neutral images and the episode specifications below; do not use tools, files, web, or outside knowledge. "
        "Each image header identifies its episode and goal. Blue numbered boxes are Grounding DINO proposals, not truth. "
        "For each episode independently: choose SELECT only when one or more listed candidates reliably satisfy the natural-language goal; "
        "use AMBIGUOUS when the referring expression does not determine a referent; use ABSTAIN when the target or a reliable matching candidate is not visible. "
        "For SELECT, selected_candidate_ids must contain the chosen IDs in preference order. It may contain multiple IDs only when the goal genuinely permits multiple visible entrances. "
        "For AMBIGUOUS or ABSTAIN, selected_candidate_ids must be empty. Return exactly one decision per requested episode and no extra episodes.\n\n"
        + json.dumps(episode_specs, ensure_ascii=False, indent=2)
    )


def _validate_raw(raw: Mapping[str, Any], batch: Sequence[tuple[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    decisions = raw.get("decisions")
    _require(isinstance(decisions, list), "Brain decisions missing")
    expected = {model_case_id: episode for model_case_id, episode in batch}
    _require({str(item.get("episode_id")) for item in decisions} == set(expected), "Brain episode set mismatch")
    result = []
    for item in decisions:
        model_case_id = str(item["episode_id"])
        episode = expected[model_case_id]
        action = str(item.get("action"))
        selected = [str(value) for value in item.get("selected_candidate_ids", [])]
        valid_ids = {str(candidate["candidate_id"]) for candidate in episode["candidates"]}
        _require(action in ACTIONS, "unknown Brain action")
        _require(len(selected) == len(set(selected)) and set(selected) <= valid_ids, "Brain selected unknown or duplicate candidate")
        _require(bool(selected) == (action == "SELECT"), "Brain action/selection mismatch")
        result.append(dict(
            item,
            episode_id=str(episode["episode_id"]),
            model_case_id=model_case_id,
            action=action,
            selected_candidate_ids=selected,
        ))
    return result


def _evidence(provider_id: str, evidence_id: str, evidence_type: str, frame_id: str, timestamp: int, region: Mapping[str, Any], confidence: float, target_name: str | None) -> dict[str, Any]:
    return {
        "provider_id": provider_id,
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "source_frame_id": frame_id,
        "source_timestamp_ms": timestamp,
        "region_in_source_frame": dict(region),
        "confidence": confidence,
        "validity": "VALID",
        "expiry_timestamp_ms": timestamp + 60_000,
        "identity_claim": {"target_name": target_name, "relation": "entrance_of" if target_name else "none"},
        "provenance": {
            "implementation_id": provider_id,
            "config_id": POLICY_ID,
            "source_kind": "VLM_PROVIDER" if provider_id == "codex-brain" else "RGB_PROVIDER",
        },
    }


def _frozen_output(episode: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    frame_id = str(episode["evaluator_episode"]["observation_window"]["frame_ids"][0])
    timestamp = int(episode["evaluator_episode"]["observation_window"]["start_timestamp_ms"])
    selected_ids = list(raw["selected_candidate_ids"])
    selected_id = selected_ids[0] if selected_ids else None
    target_name = str(episode["evaluator_episode"]["goal_spec"]["target_name"])
    evidences = []
    candidates = []
    for item in episode["candidates"]:
        candidate_id = str(item["candidate_id"])
        structure_id = f"structure--{candidate_id}"
        candidate_evidence = [structure_id]
        evidences.append(_evidence(
            "grounding-dino-tiny", structure_id, "ENTRANCE_STRUCTURE", frame_id, timestamp,
            item["region"], float(item["proposal_score"]), None,
        ))
        provider_ids = ["grounding-dino-tiny"]
        identity = None
        if candidate_id == selected_id:
            identity_id = f"brain-identity--{candidate_id}"
            evidences.append(_evidence(
                "codex-brain", identity_id, "OPEN_VOCAB", frame_id, timestamp,
                item["region"], float(raw["confidence"]), target_name,
            ))
            candidate_evidence.append(identity_id)
            provider_ids.append("codex-brain")
            identity = target_name
        candidates.append({
            "candidate_id": candidate_id,
            "region": dict(item["region"]),
            "category_label": str(item["category_label"]),
            "identity_hypothesis": identity,
            "confidence": float(item["proposal_score"]),
            "provider_rank": int(item["provider_rank"]),
            "provider_ids": provider_ids,
            "evidence_ids": candidate_evidence,
        })
    provider_order = [str(item["candidate_id"]) for item in sorted(episode["candidates"], key=lambda value: int(value["provider_rank"]))]
    ranked = selected_ids + [value for value in provider_order if value not in selected_ids]
    if raw["action"] == "SELECT":
        selected = next(item for item in candidates if item["candidate_id"] == selected_id)
        supporting = list(selected["evidence_ids"])
        decision = {
            "status": "GROUNDED",
            "selected_candidate_id": selected_id,
            "ranked_candidate_ids": ranked,
            "source_frame_id": frame_id,
            "decision_timestamp_ms": timestamp + 1,
            "spatial_region": dict(selected["region"]),
            "goal_identity_support": "SUPPORTED",
            "spatial_support": "SUPPORTED",
            "confidence": float(raw["confidence"]),
            "supporting_evidence_ids": supporting,
            "competing_candidate_ids": [value for value in ranked if value != selected_id],
            "abstention_reason": None,
            "persistence_handoff_token": {
                "handoff_id": f"handoff--{episode['episode_id']}",
                "candidate_id": selected_id,
                "source_frame_id": frame_id,
                "spatial_region": dict(selected["region"]),
                "evidence_ids": supporting,
            },
        }
    else:
        status = "AMBIGUOUS" if raw["action"] == "AMBIGUOUS" else "ABSTAIN_NO_RELIABLE_EVIDENCE"
        decision = {
            "status": status,
            "selected_candidate_id": None,
            "ranked_candidate_ids": ranked,
            "source_frame_id": None,
            "decision_timestamp_ms": timestamp + 1,
            "spatial_region": None,
            "goal_identity_support": "NOT_EVALUABLE",
            "spatial_support": "NOT_EVALUABLE",
            "confidence": None,
            "supporting_evidence_ids": [],
            "competing_candidate_ids": ranked,
            "abstention_reason": "AMBIGUOUS_CANDIDATES" if raw["action"] == "AMBIGUOUS" else "NO_CANDIDATE",
            "persistence_handoff_token": None,
        }
    structure_ids = [item["evidence_id"] for item in evidences if item["provider_id"] == "grounding-dino-tiny"]
    brain_ids = [item["evidence_id"] for item in evidences if item["provider_id"] == "codex-brain"]
    return {
        "schema_version": 1,
        "episode_id": str(episode["episode_id"]),
        "provider_runs": [
            {"provider_id": "grounding-dino-tiny", "status": "RUN_SUCCESS", "source_frame_ids": [frame_id], "evidence_ids": structure_ids, "candidate_ids": provider_order, "failure_reason": None},
            {"provider_id": "codex-brain", "status": "RUN_SUCCESS", "source_frame_ids": [frame_id], "evidence_ids": brain_ids, "candidate_ids": selected_ids, "failure_reason": None},
        ],
        "evidence": evidences,
        "candidates": candidates,
        "decision": decision,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    _require(cohort.get("claim_ceiling") == "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY", "input cohort exceeds allowed claim ceiling")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    schema_path = run_dir / "brain-output-schema.json"
    materializer.write_json(schema_path, _schema())
    executable = args.codex_exe.resolve()
    provider = {
        "executable": str(executable),
        "executable_sha256": _sha256_file(executable),
        "cli_version": args.cli_version,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "authentication": "CHATGPT_LOGIN_PREFLIGHT_CONFIRMED",
    }
    materializer.write_json(run_dir / "provider-receipt.json", provider)
    episodes = cohort["episodes"]
    aliased_episodes = [(f"case-{index:03d}", episode) for index, episode in enumerate(episodes, start=1)]
    input_dir = run_dir / "brain-inputs"
    image_paths = {}
    for model_case_id, episode in aliased_episodes:
        path = input_dir / f"{model_case_id}.jpg"
        _render_input(episode, model_case_id, path)
        image_paths[model_case_id] = path

    all_raw: list[dict[str, Any]] = []
    batch_receipts = []
    for offset in range(0, len(aliased_episodes), args.batch_size):
        batch = aliased_episodes[offset:offset + args.batch_size]
        batch_id = f"batch-{offset // args.batch_size + 1:03d}"
        batch_dir = run_dir / "batches" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        prompt = _prompt(batch)
        (batch_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        completed = False
        errors = []
        for attempt in range(1, 3):
            raw_path = batch_dir / f"attempt-{attempt}-last-message.json"
            command = [
                str(executable), "exec", "--ephemeral", "--ignore-rules", "--json", "--color", "never", "--sandbox", "read-only",
                "--model", args.model, "-c", f'model_reasoning_effort="{args.reasoning_effort}"',
                "--output-schema", str(schema_path), "--output-last-message", str(raw_path),
            ]
            for model_case_id, _episode in batch:
                command.extend(["--image", str(image_paths[model_case_id])])
            command.extend(["--", prompt])
            started = {"batch_id": batch_id, "attempt": attempt, "model_case_ids": [item[0] for item in batch], "status": "DISPATCH_STARTED"}
            materializer.write_json(batch_dir / f"attempt-{attempt}-dispatch.json", started)
            command.insert(2, "--skip-git-repo-check")
            result = subprocess.run(command, cwd=run_dir, shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
            (batch_dir / f"attempt-{attempt}-stdout.txt").write_text(result.stdout, encoding="utf-8")
            (batch_dir / f"attempt-{attempt}-stderr.txt").write_text(result.stderr, encoding="utf-8")
            try:
                _require(result.returncode == 0 and raw_path.is_file(), f"Codex exit {result.returncode}")
                parsed = json.loads(raw_path.read_text(encoding="utf-8"))
                decisions = _validate_raw(parsed, batch)
                all_raw.extend(decisions)
                batch_receipts.append({"batch_id": batch_id, "attempt": attempt, "status": "RUN_SUCCESS", "episode_count": len(batch), "response_sha256": materializer.content_sha256(parsed)})
                completed = True
                print(f"{batch_id} complete episodes={len(batch)} attempt={attempt}", flush=True)
                break
            except (BrainRunError, json.JSONDecodeError) as error:
                errors.append(str(error))
        _require(completed, f"{batch_id} failed twice: {errors}")

    raw_by_id = {str(item["episode_id"]): item for item in all_raw}
    case_map = [{"model_case_id": model_case_id, "episode_id": str(episode["episode_id"])} for model_case_id, episode in aliased_episodes]
    materializer.write_json(run_dir / "post_run_evaluator_case_map.json", {"created_after_all_model_calls": True, "cases": case_map})
    outputs = []
    cases = []
    for episode in episodes:
        raw = raw_by_id[str(episode["episode_id"])]
        output = _frozen_output(episode, raw)
        p0_evaluator.validate_output(output, episode["evaluator_episode"])
        outputs.append(output)
        cases.append((episode["evaluator_episode"], output))
    evaluation = p0_evaluator.evaluate_batch(cases)
    raw_actions = {action: sum(item["action"] == action for item in all_raw) for action in sorted(ACTIONS)}
    report = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "cohort_report_sha256": cohort["report_sha256"],
        "provider": provider,
        "batch_receipts": batch_receipts,
        "raw_action_counts": raw_actions,
        "raw_decisions": all_raw,
        "frozen_evaluator": evaluation,
        "allowed_analysis": "CONDITIONED_CANDIDATE_SELECTION_AND_WEAK_GROUNDING_MECHANICS",
        "claim_ceiling": "SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY",
    }
    report["report_sha256"] = materializer.content_sha256(report)
    materializer.write_json(run_dir / "brain-baseline-report.json", report)
    materializer.write_json(run_dir / "frozen-system-outputs.json", {"outputs": outputs})
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--codex-exe", required=True, type=Path)
    parser.add_argument("--cli-version", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--batch-size", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({
        "episode_count": report["frozen_evaluator"]["aggregate"]["episode_count"],
        "raw_action_counts": report["raw_action_counts"],
        "mechanics_aggregate": report["frozen_evaluator"]["aggregate"],
        "claim_ceiling": report["claim_ceiling"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
