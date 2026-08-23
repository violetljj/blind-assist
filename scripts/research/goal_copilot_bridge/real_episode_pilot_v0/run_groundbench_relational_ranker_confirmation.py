"""Run paired fresh Confirmation for the frozen relational candidate ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_regrounding_v0 import provider_adapter
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_facade_native_door_89 import (
    atomic_json,
    evaluate,
    sha256_file,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_clip_candidate_verifier import (
    ClipScorer,
    ranking_evaluation,
    rerank_episode,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_referent_89 import (
    CLAIM_CEILING,
    build_episode,
    run_expression_inference,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_groundbench_referent_union_confirmation import (
    run_brain_stdin,
)
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.train_groundbench_relational_candidate_ranker import (
    FEATURE_NAMES,
    candidate_features,
    model_scores,
)


SCHEMA_VERSION = "groundbench_relational_candidate_ranker_confirmation_v0"
ROSTER_SCHEMA_VERSION = "groundbench_referent_union_confirmation_roster_v0"


class RunError(RuntimeError):
    pass


def paired_verdict(
    v0_rank: Mapping[str, Any], v1_rank: Mapping[str, Any],
    v0: Mapping[str, Any], v1: Mapping[str, Any],
) -> dict[str, Any]:
    rank0, rank1 = int(v0_rank["recall_at_k"]["1"]), int(v1_rank["recall_at_k"]["1"])
    correct0 = int(v0["outcome_counts"].get("CORRECT_GROUNDING", 0))
    correct1 = int(v1["outcome_counts"].get("CORRECT_GROUNDING", 0))
    wrong0 = int(v0["wrong_confident_guidance_all_observations"]["numerator"])
    wrong1 = int(v1["wrong_confident_guidance_all_observations"]["numerator"])
    supported = rank1 > rank0 and correct1 > correct0 and wrong1 <= wrong0
    return {
        "verdict": "RELATIONAL_CANDIDATE_RANKER_SUPPORTED" if supported else "RELATIONAL_CANDIDATE_RANKER_NOT_SUPPORTED",
        "rank1_v0": rank0, "rank1_v1": rank1,
        "correct_grounding_v0": correct0, "correct_grounding_v1": correct1,
        "wrong_confident_guidance_all_v0": wrong0, "wrong_confident_guidance_all_v1": wrong1,
        "success_rule": "rank1_v1 > rank1_v0 AND correct_v1 > correct_v0 AND wrong_all_v1 <= wrong_all_v0",
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    formal_root = args.formal_root.resolve()
    if formal_root.exists():
        raise RunError("formal relational-ranker Confirmation root already exists")
    development_dir = args.development_run.resolve()
    report_path = development_dir / "development-report.json"
    model_path = development_dir / "relational-ranker.joblib"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not report.get("gate", {}).get("confirmation_authorized")
        or report.get("feature_names") != list(FEATURE_NAMES)
        or report.get("model", {}).get("model_sha256") != sha256_file(model_path)
    ):
        raise RunError("Development report does not authorize this frozen model")
    base_lock = provider_adapter.preflight_provider(codex_exe=args.codex_exe, model_dir=args.dino_model_dir)
    clip_dir = args.clip_model_dir.resolve()
    ClipScorer(clip_dir)
    lock = {
        "schema_version": "groundbench_relational_candidate_ranker_provider_lock_v0",
        "mechanical_base_provider_lock": base_lock,
        "development_report_file_sha256": sha256_file(report_path),
        "development_report_content_sha256": report["report_sha256"],
        "model_path": str(model_path), "model_sha256": sha256_file(model_path),
        "feature_names": list(FEATURE_NAMES), "clip_model_dir": str(clip_dir),
        "arms": {"V0": "PUBLIC_EXPRESSION_PROVIDER_ORDER", "V1": "SAME_CANDIDATES_FROZEN_RELATIONAL_RERANK"},
        "success_rule": "rank1_v1 > rank1_v0 AND correct_v1 > correct_v0 AND wrong_all_v1 <= wrong_all_v0",
        "retry_count": 0, "reruns": 0, "teacher_calls": 0, "claim_ceiling": CLAIM_CEILING,
    }
    formal_root.mkdir(parents=True)
    atomic_json(formal_root / "provider-lock.json", lock)
    return lock


def run(args: argparse.Namespace) -> dict[str, Any]:
    roster_path = args.roster.resolve()
    if sha256_file(roster_path) != args.roster_sha256:
        raise RunError("relational-ranker Confirmation roster hash mismatch")
    roster_doc = json.loads(roster_path.read_text(encoding="utf-8"))
    if (
        roster_doc.get("schema_version") != ROSTER_SCHEMA_VERSION
        or len(roster_doc.get("observations", [])) != 64
        or roster_doc.get("provider_calls") != 0
        or roster_doc.get("teacher_calls") != 0
    ):
        raise RunError("relational-ranker Confirmation roster contract mismatch")
    formal_root, run_dir = args.formal_root.resolve(), args.formal_root.resolve() / "run"
    lock_path = formal_root / "provider-lock.json"
    if not lock_path.is_file() or run_dir.exists():
        raise RunError("provider lock missing or Confirmation already consumed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    executable, dino_model_dir = provider_adapter.verify_provider_lock(lock["mechanical_base_provider_lock"])
    model_path = Path(lock["model_path"]).resolve()
    if sha256_file(model_path) != lock["model_sha256"] or lock["feature_names"] != list(FEATURE_NAMES):
        raise RunError("frozen relational model drift")
    model = joblib.load(model_path)
    dataset_root = Path(roster_doc["dataset_root"]).resolve()
    public_rows, metadata = [], []
    for item in roster_doc["observations"]:
        rgb = (dataset_root / item["rgb_path"]).resolve()
        if not rgb.is_file() or rgb.stat().st_size != item["rgb_bytes"] or sha256_file(rgb) != item["rgb_sha256"]:
            raise RunError(f"frozen COCO pixel mismatch: {item['observation_id']}")
        public = dict(item, absolute_rgb_path=str(rgb))
        public_rows.append(public)
        metadata.append({
            "id": item["observation_id"], "path": str(rgb), "image_sha256": item["rgb_sha256"],
            "goal_text": item["goal_text"],
        })
    run_dir.mkdir()
    proposal_outputs, dino_runtime = run_expression_inference(dino_model_dir, metadata)
    v0_episodes = [
        build_episode(item, output["proposals"], index)
        for index, (item, output) in enumerate(zip(public_rows, proposal_outputs), start=1)
    ]
    scorer = ClipScorer(Path(lock["clip_model_dir"]).resolve())
    v1_episodes, feature_rows = [], []
    for index, (item, episode) in enumerate(zip(public_rows, v0_episodes), start=1):
        with Image.open(item["absolute_rgb_path"]) as opened:
            image = opened.convert("RGB")
        clip_scores = scorer.score_episode(image, item["goal_text"], episode)
        features = {
            str(candidate["candidate_id"]): candidate_features(candidate, item["goal_text"], clip_scores)
            for candidate in episode["candidates"]
        }
        scores = model_scores(model, features)
        v1_episodes.append(rerank_episode(episode, scores))
        feature_rows.append({"episode_id": episode["episode_id"], "features": features, "ranker_scores": scores})
        print(f"relational-confirmation {index}/{len(v0_episodes)} {episode['episode_id']}", flush=True)
    atomic_json(run_dir / "proposal-provider-output.json", {"runtime": dino_runtime, "outputs": proposal_outputs})
    atomic_json(run_dir / "relational-ranker-output.json", {"feature_names": list(FEATURE_NAMES), "rows": feature_rows})
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
        "same_candidate_sets": True, "model_sha256": lock["model_sha256"],
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
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--formal-root", type=Path, required=True)
    preflight_parser.add_argument("--development-run", type=Path, required=True)
    preflight_parser.add_argument("--clip-model-dir", type=Path, required=True)
    preflight_parser.add_argument("--dino-model-dir", type=Path, required=True)
    preflight_parser.add_argument("--codex-exe", type=Path, default=Path(r"E:\codex-tools\bin\codex.exe"))
    confirmation = subparsers.add_parser("run")
    confirmation.add_argument("--formal-root", type=Path, required=True)
    confirmation.add_argument("--roster", type=Path, required=True)
    confirmation.add_argument("--roster-sha256", required=True)
    confirmation.add_argument("--batch-size", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    result = preflight(args) if args.command == "preflight" else run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
