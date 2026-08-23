"""Screen a scene-plus-candidate-zoom representation on consumed GroundBench data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0 import provider_adapter
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_silver_b_brain_baseline as brain
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_facade_native_door_89 import (
    atomic_json,
    evaluate,
    run_brain,
    sha256_file,
)


SCHEMA_VERSION = "groundbench_candidate_zoom_development_v0"
CLAIM_CEILING = "CONSUMED_STATIC_GROUNDBENCH_DEVELOPMENT_ONLY_NO_CONFIRMATION_APPROACH_CONTROL_OR_PRODUCT_CLAIM"


class RunError(RuntimeError):
    pass


def development_decision(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    baseline_correct = int(baseline["given_usable_proposal"]["correct_grounding"])
    candidate_correct = int(candidate["given_usable_proposal"]["correct_grounding"])
    baseline_wrong = int(baseline["wrong_confident_guidance_all_observations"]["numerator"])
    candidate_wrong = int(candidate["wrong_confident_guidance_all_observations"]["numerator"])
    promising = candidate_correct > baseline_correct and candidate_wrong <= baseline_wrong
    return {
        "terminal": (
            "CANDIDATE_ZOOM_REPRESENTATION_DEVELOPMENT_PROMISING"
            if promising
            else "CANDIDATE_ZOOM_REPRESENTATION_DEVELOPMENT_NOT_PROMISING"
        ),
        "confirmation_authorized": promising,
        "rule": "CORRECT_GROUNDING_STRICTLY_INCREASES_AND_WRONG_CONFIDENT_GUIDANCE_DOES_NOT_INCREASE",
        "baseline_correct_grounding": baseline_correct,
        "candidate_correct_grounding": candidate_correct,
        "baseline_wrong_confident_guidance": baseline_wrong,
        "candidate_wrong_confident_guidance": candidate_wrong,
    }


def _load_verified(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise RunError(f"{label} missing or SHA-256 mismatch")
    return json.loads(resolved.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RunError("development output already exists")
    roster_doc = _load_verified(args.roster, args.roster_sha256, "consumed roster")
    public_doc = _load_verified(args.public_input, args.public_input_sha256, "public provider input")
    baseline_report = _load_verified(args.baseline_report, args.baseline_report_sha256, "baseline report")
    provider_lock = _load_verified(
        args.provider_lock, str(baseline_report["provider_lock_sha256"]), "provider lock"
    )
    if baseline_report.get("roster_sha256") != args.roster_sha256:
        raise RunError("baseline report does not bind the supplied roster")
    roster = list(roster_doc.get("observations", []))
    episodes = list(public_doc.get("episodes", []))
    if len(roster) != 89 or len(episodes) != 89 or baseline_report["evaluation"]["observation_count"] != 89:
        raise RunError("expected the consumed 89-observation GroundBench development source")
    executable, _ = provider_adapter.verify_provider_lock(provider_lock["mechanical_base_provider_lock"])

    usable_ids = {
        str(row["episode_id"])
        for row in baseline_report["evaluation"]["observations"]
        if row.get("correct_candidate_ids")
    }
    selected = [
        (truth, episode)
        for truth, episode in zip(roster, episodes)
        if str(episode["episode_id"]) in usable_ids
    ]
    if len(selected) != 77:
        raise RunError("consumed usable-proposal denominator drifted")
    selected_roster = [truth for truth, _ in selected]
    selected_episodes = [episode for _, episode in selected]
    baseline_decisions = [
        decision
        for decision in baseline_report["raw_brain_decisions"]
        if str(decision["episode_id"]) in usable_ids
    ]
    baseline_evaluation = evaluate(selected_roster, selected_episodes, baseline_decisions)
    implementation = {
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "representation_path": str(Path(brain.__file__).resolve()),
        "representation_sha256": sha256_file(Path(brain.__file__).resolve()),
    }

    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "experiment-manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "data_role": "CONSUMED_DEVELOPMENT",
        "roster_sha256": args.roster_sha256,
        "public_input_sha256": args.public_input_sha256,
        "baseline_report_sha256": args.baseline_report_sha256,
        "provider_lock_sha256": baseline_report["provider_lock_sha256"],
        "representation_id": brain.CANDIDATE_ZOOM_REPRESENTATION_ID,
        "implementation": implementation,
        "selected_observations": len(selected_episodes),
        "selection_rule": "BASELINE_USABLE_PROPOSAL_ONLY",
        "teacher_calls": 0,
        "claim_ceiling": CLAIM_CEILING,
    })
    decisions, receipts = run_brain(
        episodes=selected_episodes,
        run_dir=output_dir / "brain",
        executable=executable,
        model=provider_adapter.CODEX_MODEL,
        reasoning_effort=provider_adapter.CODEX_REASONING_EFFORT,
        batch_size=args.batch_size,
        render_input=brain._render_input_with_candidate_zoom,
        prompt_builder=brain._prompt_with_candidate_zoom,
    )
    candidate_evaluation = evaluate(selected_roster, selected_episodes, decisions)
    disposition = development_decision(baseline_evaluation, candidate_evaluation)
    result = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "CONSUMED_DEVELOPMENT",
        "representation_id": brain.CANDIDATE_ZOOM_REPRESENTATION_ID,
        "implementation": implementation,
        "observation_count": len(selected_episodes),
        "provider_calls": len(receipts),
        "provider_attempts": len(receipts),
        "provider_in_doubt": 0,
        "teacher_calls": 0,
        "baseline_evaluation": baseline_evaluation,
        "candidate_evaluation": candidate_evaluation,
        "raw_candidate_decisions": decisions,
        "provider_receipts": receipts,
        "disposition": disposition,
        "claim_ceiling": CLAIM_CEILING,
    }
    result["content_sha256"] = materializer.content_sha256(result)
    atomic_json(output_dir / "result.json", result)
    print(json.dumps({
        "terminal": disposition["terminal"],
        "observation_count": len(selected_episodes),
        "baseline_correct": disposition["baseline_correct_grounding"],
        "candidate_correct": disposition["candidate_correct_grounding"],
        "baseline_wrong": disposition["baseline_wrong_confident_guidance"],
        "candidate_wrong": disposition["candidate_wrong_confident_guidance"],
        "content_sha256": result["content_sha256"],
    }, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--public-input", type=Path, required=True)
    parser.add_argument("--public-input-sha256", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-report-sha256", required=True)
    parser.add_argument("--provider-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=8)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
