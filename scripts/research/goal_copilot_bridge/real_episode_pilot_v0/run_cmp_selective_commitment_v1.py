"""Calibrate and confirm a deterministic selective-commitment gate on fresh CMP truth."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0 import provider_adapter
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_grounding_dino_s0_r1 as dino
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_facade_native_door_89 import (
    CLAIM_CEILING,
    RunError,
    atomic_json,
    build_episode,
    evaluate,
    run_brain,
    sha256_file,
)


SCHEMA_VERSION = "cmp_selective_commitment_v1_v0"
POLICY_GRID = (
    {"policy_id": "CONF_075", "minimum_brain_confidence": 0.75, "require_provider_rank_1": False},
    {"policy_id": "CONF_085", "minimum_brain_confidence": 0.85, "require_provider_rank_1": False},
    {"policy_id": "CONF_095", "minimum_brain_confidence": 0.95, "require_provider_rank_1": False},
    {"policy_id": "CONF_075_RANK1", "minimum_brain_confidence": 0.75, "require_provider_rank_1": True},
    {"policy_id": "CONF_085_RANK1", "minimum_brain_confidence": 0.85, "require_provider_rank_1": True},
    {"policy_id": "CONF_095_RANK1", "minimum_brain_confidence": 0.95, "require_provider_rank_1": True},
)
MIN_DEVELOPMENT_COMMITMENTS = 8
MAX_DEVELOPMENT_WRONG_ALL = 1
MIN_DEVELOPMENT_COMMITMENT_ACCURACY = 0.90
MIN_CONFIRMATION_CORRECT_RETENTION = 0.80


def gate_decisions(
    decisions: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]], policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    episode_by_id = {episode["episode_id"]: episode for episode in episodes}
    gated = []
    for raw in decisions:
        item = copy.deepcopy(dict(raw))
        if item["action"] != "SELECT":
            gated.append(item)
            continue
        episode = episode_by_id[item["episode_id"]]
        selected_id = item["selected_candidate_ids"][0]
        candidates = {candidate["candidate_id"]: candidate for candidate in episode["candidates"]}
        if selected_id not in candidates:
            raise RunError("selected candidate missing from public episode")
        rank = int(candidates[selected_id]["provider_rank"])
        confidence_ok = float(item["confidence"]) >= float(policy["minimum_brain_confidence"])
        rank_ok = not policy["require_provider_rank_1"] or rank == 1
        if not (confidence_ok and rank_ok):
            item["action"] = "CONTESTED"
            item["selected_candidate_ids"] = []
            item["selective_commitment_gate"] = {
                "policy_id": policy["policy_id"],
                "original_action": "SELECT",
                "brain_confidence": float(raw["confidence"]),
                "selected_provider_rank": rank,
                "confidence_ok": confidence_ok,
                "rank_ok": rank_ok,
            }
        gated.append(item)
    return gated


def policy_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    action_counts = evaluation["brain_action_counts"]
    commitments = int(action_counts.get("SELECT", 0))
    correct = int(evaluation["outcome_counts"].get("CORRECT_GROUNDING", 0))
    wrong = int(evaluation["wrong_confident_guidance_all_observations"]["numerator"])
    precision = correct / commitments if commitments else None
    return {
        "commitments": commitments,
        "correct_grounding": correct,
        "wrong_confident_guidance_all": wrong,
        "commitment_accuracy": precision,
        "eligible": (
            commitments >= MIN_DEVELOPMENT_COMMITMENTS
            and wrong <= MAX_DEVELOPMENT_WRONG_ALL
            and precision is not None
            and precision >= MIN_DEVELOPMENT_COMMITMENT_ACCURACY
        ),
    }


def select_policy(
    decisions: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]], roster: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = []
    for policy in POLICY_GRID:
        gated = gate_decisions(decisions, episodes, policy)
        evaluation = evaluate(roster, episodes, gated)
        candidates.append({"policy": dict(policy), "metrics": policy_metrics(evaluation)})
    eligible = [item for item in candidates if item["metrics"]["eligible"]]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda item: (
                -item["metrics"]["correct_grounding"],
                item["metrics"]["wrong_confident_guidance_all"],
                -item["metrics"]["commitments"],
                item["policy"]["policy_id"],
            ),
        )[0]
        selection_mode = "PREDECLARED_SAFETY_GATES_PASSED"
    else:
        fallback = [item for item in candidates if item["metrics"]["commitments"] >= MIN_DEVELOPMENT_COMMITMENTS]
        if not fallback:
            raise RunError("no policy meets the minimum Development commitment denominator")
        selected = sorted(
            fallback,
            key=lambda item: (
                -(item["metrics"]["correct_grounding"] - 3 * item["metrics"]["wrong_confident_guidance_all"]),
                -(item["metrics"]["commitment_accuracy"] or 0.0),
                -item["metrics"]["correct_grounding"],
                item["policy"]["policy_id"],
            ),
        )[0]
        selection_mode = "PREDECLARED_FALLBACK_UTILITY"
    return {
        "selection_mode": selection_mode,
        "selection_requirements": {
            "minimum_commitments": MIN_DEVELOPMENT_COMMITMENTS,
            "maximum_wrong_confident_guidance_all": MAX_DEVELOPMENT_WRONG_ALL,
            "minimum_commitment_accuracy": MIN_DEVELOPMENT_COMMITMENT_ACCURACY,
            "fallback_utility": "correct_grounding - 3 * wrong_confident_guidance_all",
        },
        "policy_grid": candidates,
        "selected_policy": selected["policy"],
    }


def load_frozen_split(roster_path: Path, roster_sha256: str, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256_file(roster_path) != roster_sha256:
        raise RunError("fresh split roster hash mismatch")
    document = json.loads(roster_path.read_text(encoding="utf-8"))
    if document.get("provider_calls") != 0 or document.get("teacher_calls") != 0:
        raise RunError("fresh split roster was not frozen before providers")
    rows = document["splits"][split]
    expected = {"development": 32, "confirmation": 64}[split]
    if len(rows) != expected:
        raise RunError(f"{split} denominator mismatch")
    return document, rows


def provider_preflight(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = args.formal_root.resolve()
    if formal_root.exists():
        raise RunError("formal selective-commitment root already exists")
    lock = provider_adapter.preflight_provider(codex_exe=args.codex_exe, model_dir=args.model_dir)
    formal_root.mkdir(parents=True)
    atomic_json(formal_root / "provider-lock.json", lock)
    atomic_json(formal_root / "policy-grid-freeze.json", {
        "schema_version": SCHEMA_VERSION,
        "policy_grid": POLICY_GRID,
        "development_selection_requirements": {
            "minimum_commitments": MIN_DEVELOPMENT_COMMITMENTS,
            "maximum_wrong_confident_guidance_all": MAX_DEVELOPMENT_WRONG_ALL,
            "minimum_commitment_accuracy": MIN_DEVELOPMENT_COMMITMENT_ACCURACY,
            "fallback_utility": "correct_grounding - 3 * wrong_confident_guidance_all",
        },
        "confirmation_success_rule": {
            "wrong_confident_guidance": "V1 < V0",
            "correct_grounding_retention": MIN_CONFIRMATION_CORRECT_RETENTION,
            "commitment_accuracy": "V1 >= V0",
        },
        "provider_retry_count": 0,
        "reruns": 0,
    })
    return lock


def run_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    roster_path = args.roster.resolve()
    roster_doc, roster = load_frozen_split(roster_path, args.roster_sha256, split)
    formal_root = args.formal_root.resolve()
    lock_path = formal_root / "provider-lock.json"
    if not lock_path.is_file():
        raise RunError("provider preflight receipt is missing")
    provider_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    executable, model_dir = provider_adapter.verify_provider_lock(provider_lock)
    split_dir = formal_root / split
    if split_dir.exists():
        raise RunError(f"formal {split} directory already exists; refusing replay")
    split_dir.mkdir()

    public_rows = []
    metadata = []
    dataset_root = Path(roster_doc["dataset_root"]).resolve()
    for item in roster:
        rgb = (dataset_root / item["rgb_path"]).resolve()
        if sha256_file(rgb) != item["rgb_sha256"]:
            raise RunError("fresh RGB hash mismatch")
        public_rows.append(dict(item, absolute_rgb_path=str(rgb)))
        metadata.append({"id": Path(item["rgb_path"]).stem, "path": str(rgb), "image_sha256": item["rgb_sha256"]})
    proposals, runtime = dino.run_inference(model_dir, metadata)
    atomic_json(split_dir / "proposal-provider-output.json", {"runtime": runtime, "outputs": proposals})
    episodes = [
        build_episode(item, result["proposals"], index)
        for index, (item, result) in enumerate(zip(public_rows, proposals), start=1)
    ]
    atomic_json(split_dir / "public-provider-input.json", {"episodes": episodes})
    decisions, receipts = run_brain(
        episodes=episodes, run_dir=split_dir / "brain", executable=executable,
        model=provider_adapter.CODEX_MODEL, reasoning_effort=provider_adapter.CODEX_REASONING_EFFORT,
        batch_size=args.batch_size,
    )
    evaluation = evaluate(roster, episodes, decisions)
    report = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "roster_sha256": args.roster_sha256,
        "provider_lock_sha256": sha256_file(lock_path),
        "provider_calls": len(receipts),
        "provider_attempts": len(receipts),
        "provider_in_doubt": 0,
        "teacher_calls": 0,
        "reruns": 0,
        "brain_batch_receipts": receipts,
        "raw_brain_decisions": decisions,
        "v0_evaluation": evaluation,
        "claim_ceiling": CLAIM_CEILING,
    }
    report["report_sha256"] = materializer.content_sha256(report)
    atomic_json(split_dir / "split-report.json", report)
    return {"report": report, "episodes": episodes, "roster": roster}


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    result = run_split(args, "development")
    selection = select_policy(result["report"]["raw_brain_decisions"], result["episodes"], result["roster"])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "development_report_sha256": result["report"]["report_sha256"],
        "provider_calls_added_by_selection": 0,
        "selected_before_confirmation_provider_calls": True,
        **selection,
    }
    receipt["receipt_sha256"] = materializer.content_sha256(receipt)
    output = args.formal_root.resolve() / "selected-policy.json"
    if output.exists():
        raise RunError("selected-policy receipt already exists")
    atomic_json(output, receipt)
    return receipt


def confirmation_verdict(v0: Mapping[str, Any], v1: Mapping[str, Any]) -> dict[str, Any]:
    v0_wrong = int(v0["wrong_confident_guidance_all_observations"]["numerator"])
    v1_wrong = int(v1["wrong_confident_guidance_all_observations"]["numerator"])
    v0_correct = int(v0["outcome_counts"].get("CORRECT_GROUNDING", 0))
    v1_correct = int(v1["outcome_counts"].get("CORRECT_GROUNDING", 0))
    retention = v1_correct / v0_correct if v0_correct else None
    v0_precision = v0["commitment_accuracy"]["value"]
    v1_precision = v1["commitment_accuracy"]["value"]
    supported = (
        v1_wrong < v0_wrong
        and retention is not None
        and retention >= MIN_CONFIRMATION_CORRECT_RETENTION
        and v1_precision is not None
        and v0_precision is not None
        and v1_precision >= v0_precision
    )
    return {
        "verdict": "SELECTIVE_COMMITMENT_SUPPORTED" if supported else "SELECTIVE_COMMITMENT_NOT_SUPPORTED",
        "wrong_confident_guidance_v0": v0_wrong,
        "wrong_confident_guidance_v1": v1_wrong,
        "correct_grounding_v0": v0_correct,
        "correct_grounding_v1": v1_correct,
        "correct_grounding_retention": retention,
        "commitment_accuracy_v0": v0_precision,
        "commitment_accuracy_v1": v1_precision,
    }


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    policy_path = args.formal_root.resolve() / "selected-policy.json"
    if not policy_path.is_file() or sha256_file(policy_path) != args.policy_sha256:
        raise RunError("Development-selected policy receipt hash mismatch")
    policy_receipt = json.loads(policy_path.read_text(encoding="utf-8"))
    result = run_split(args, "confirmation")
    gated = gate_decisions(
        result["report"]["raw_brain_decisions"], result["episodes"], policy_receipt["selected_policy"],
    )
    v1_evaluation = evaluate(result["roster"], result["episodes"], gated)
    final = {
        "schema_version": SCHEMA_VERSION,
        "confirmation_report_sha256": result["report"]["report_sha256"],
        "selected_policy_file_sha256": args.policy_sha256,
        "selected_policy": policy_receipt["selected_policy"],
        "v0_evaluation": result["report"]["v0_evaluation"],
        "v1_evaluation": v1_evaluation,
        "paired_verdict": confirmation_verdict(result["report"]["v0_evaluation"], v1_evaluation),
        "provider_calls_added_by_v1": 0,
        "teacher_calls": 0,
        "reruns": 0,
        "claim_ceiling": CLAIM_CEILING,
    }
    final["report_sha256"] = materializer.content_sha256(final)
    atomic_json(args.formal_root.resolve() / "confirmation-result.json", final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "development", "confirmation"):
        child = subparsers.add_parser(command)
        child.add_argument("--formal-root", type=Path, required=True)
        child.add_argument("--codex-exe", type=Path, default=Path(r"E:\codex-tools\bin\codex.exe"))
        child.add_argument("--model-dir", type=Path, required=True)
        if command != "preflight":
            child.add_argument("--roster", type=Path, required=True)
            child.add_argument("--roster-sha256", required=True)
            child.add_argument("--batch-size", type=int, choices=range(1, 9), default=8)
        if command == "confirmation":
            child.add_argument("--policy-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = provider_preflight(args)
    elif args.command == "development":
        result = run_development(args)
    else:
        result = run_confirmation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
