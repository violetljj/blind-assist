"""Train and gate one fixed relational candidate ranker on consumed GroundBench truth."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.real_episode_pilot_v0.run_cmp_facade_native_door_89 import (
    IOU_THRESHOLD,
    atomic_json,
    iou,
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
)


SCHEMA_VERSION = "groundbench_relational_candidate_ranker_development_v0"
FEATURE_NAMES = (
    "provider_score", "reciprocal_provider_rank",
    "clip_crop_100", "clip_crop_125", "clip_focus_context_020", "clip_dual",
    "center_x", "center_y", "width", "height", "area", "log_aspect",
    "left_affinity", "right_affinity", "top_affinity", "bottom_affinity", "center_affinity",
    "large_affinity", "small_affinity", "foreground_affinity", "background_affinity",
)
TRAIN_END_POSITION = 217
HOLDOUT_START_POSITION = 218
HOLDOUT_END_POSITION = 281


class TrainError(RuntimeError):
    pass


def expression_signals(expression: str) -> dict[str, float]:
    text = " " + " ".join(expression.lower().split()) + " "
    patterns = {
        "left": r"\b(left|leftmost)\b",
        "right": r"\b(right|rightmost)\b",
        "top": r"\b(top|topmost|upper|highest|above)\b",
        "bottom": r"\b(bottom|bottommost|lower|lowest|below)\b",
        "center": r"\b(center|centre|central|middle)\b",
        "large": r"\b(large|larger|largest|big|bigger|biggest)\b",
        "small": r"\b(small|smaller|smallest|tiny|tiniest)\b",
        "foreground": r"\b(front|foreground|closest|nearest|near)\b",
        "background": r"\b(back|background|farthest|furthest)\b",
    }
    return {name: float(bool(re.search(pattern, text))) for name, pattern in patterns.items()}


def candidate_features(
    candidate: Mapping[str, Any], expression: str, clip_scores: Mapping[str, Mapping[str, float]],
) -> list[float]:
    candidate_id = str(candidate["candidate_id"])
    region = candidate["region"]
    x0, y0, x1, y1 = (float(region[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    width, height = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    center_x, center_y, area = (x0 + x1) / 2, (y0 + y1) / 2, width * height
    signals = expression_signals(expression)
    center_affinity = max(0.0, 1.0 - 2.0 * math.hypot(center_x - 0.5, center_y - 0.5) / math.sqrt(0.5))
    values = [
        float(candidate["proposal_score"]), 1.0 / int(candidate["provider_rank"]),
        float(clip_scores["CROP_100"][candidate_id]),
        float(clip_scores["CROP_125"][candidate_id]),
        float(clip_scores["FOCUS_CONTEXT_020"][candidate_id]),
        float(clip_scores["DUAL_CROP125_FOCUS020"][candidate_id]),
        center_x, center_y, width, height, area, math.log(width / height),
        signals["left"] * (1.0 - center_x), signals["right"] * center_x,
        signals["top"] * (1.0 - center_y), signals["bottom"] * center_y,
        signals["center"] * center_affinity,
        signals["large"] * area, signals["small"] * (1.0 - area),
        signals["foreground"] * area, signals["background"] * (1.0 - area),
    ]
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise TrainError("invalid relational feature vector")
    return values


def _normalized_truth_box(item: Mapping[str, Any]) -> list[float]:
    width, height = float(item["image_width"]), float(item["image_height"])
    x0, y0, x1, y1 = item["native_mask_bbox_xyxy"]
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def correct_candidate_ids(truth: Mapping[str, Any], episode: Mapping[str, Any]) -> set[str]:
    target = _normalized_truth_box(truth)
    result = set()
    for candidate in episode["candidates"]:
        region = candidate["region"]
        box = [region[key] for key in ("x_min", "y_min", "x_max", "y_max")]
        if iou(box, target) >= IOU_THRESHOLD:
            result.add(str(candidate["candidate_id"]))
    return result


def fit_pairwise_ranker(rows: Sequence[Mapping[str, Any]]) -> tuple[Pipeline, int]:
    examples, labels = [], []
    for row in rows:
        correct = [value for candidate_id, value in row["features"].items() if candidate_id in row["correct_candidate_ids"]]
        wrong = [value for candidate_id, value in row["features"].items() if candidate_id not in row["correct_candidate_ids"]]
        for positive in correct:
            for negative in wrong:
                difference = np.asarray(positive, dtype=np.float64) - np.asarray(negative, dtype=np.float64)
                examples.extend([difference, -difference])
                labels.extend([1, 0])
    if not examples or len(set(labels)) != 2:
        raise TrainError("insufficient pairwise training examples")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ranker", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=0)),
    ])
    model.fit(np.asarray(examples), np.asarray(labels))
    return model, len(examples)


def model_scores(model: Pipeline, feature_rows: Mapping[str, Sequence[float]]) -> dict[str, float]:
    return {
        candidate_id: float(model.decision_function(np.asarray([features], dtype=np.float64))[0])
        for candidate_id, features in feature_rows.items()
    }


def development_gate(provider_metrics: Mapping[str, Any], ranker_metrics: Mapping[str, Any]) -> dict[str, Any]:
    provider_rank1 = int(provider_metrics["recall_at_k"]["1"])
    ranker_rank1 = int(ranker_metrics["recall_at_k"]["1"])
    provider_mrr = float(provider_metrics["mean_reciprocal_rank_given_available"] or 0.0)
    ranker_mrr = float(ranker_metrics["mean_reciprocal_rank_given_available"] or 0.0)
    promising = ranker_rank1 > provider_rank1 and ranker_mrr >= provider_mrr
    return {
        "terminal": "RELATIONAL_CANDIDATE_RANKER_DEVELOPMENT_PROMISING" if promising else "RELATIONAL_CANDIDATE_RANKER_DEVELOPMENT_NOT_PROMISING",
        "confirmation_authorized": promising,
        "success_rule": "holdout_rank1_v1 > holdout_rank1_v0 AND holdout_mrr_v1 >= holdout_mrr_v0",
        "provider_rank1": provider_rank1, "ranker_rank1": ranker_rank1,
        "provider_mrr": provider_mrr, "ranker_mrr": ranker_mrr,
    }


def _proposal_list(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "proposals" in output:
        return list(output["proposals"])
    if "v0_expression_proposals" in output:
        return list(output["v0_expression_proposals"])
    raise TrainError("unknown GroundBench proposal output schema")


def _load_source(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path]:
    roster_path = root / "frozen-roster-private-truth.json"
    proposal_path = root / "formal" / "run" / "proposal-provider-output.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    proposal_doc = json.loads(proposal_path.read_text(encoding="utf-8"))
    outputs = proposal_doc["outputs"]
    if len(roster["observations"]) != len(outputs):
        raise TrainError("source roster/proposal denominator mismatch")
    dataset_root = Path(roster["dataset_root"]).resolve()
    public_rows, episodes = [], []
    for index, (truth, output) in enumerate(zip(roster["observations"], outputs), start=1):
        rgb = (dataset_root / truth["rgb_path"]).resolve()
        if not rgb.is_file() or sha256_file(rgb) != truth["rgb_sha256"]:
            raise TrainError(f"source pixel mismatch: {truth['observation_id']}")
        public = dict(truth, absolute_rgb_path=str(rgb))
        public_rows.append(public)
        episodes.append(build_episode(public, _proposal_list(output), index))
    return roster, public_rows, episodes, proposal_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise TrainError("relational ranker output already exists")
    roots = [path.resolve() for path in args.source_root]
    if len(roots) != 4:
        raise TrainError("exactly four ordered consumed source roots are required")
    scorer = ClipScorer(args.clip_model_dir.resolve())
    cached = json.loads(args.cached_clip_report.resolve().read_text(encoding="utf-8"))
    cached_scores = {row["episode_id"]: row["scores"] for row in cached["score_rows"]}
    all_truth, all_episodes, feature_rows, source_receipts = [], [], [], []
    for root_index, root in enumerate(roots):
        roster, public_rows, episodes, proposal_path = _load_source(root)
        source_receipts.append({
            "root": str(root),
            "roster_sha256": sha256_file(root / "frozen-roster-private-truth.json"),
            "proposal_output_sha256": sha256_file(proposal_path),
            "observation_count": len(episodes),
        })
        for item, episode in zip(public_rows, episodes):
            if root_index == 0:
                if episode["episode_id"] not in cached_scores:
                    raise TrainError("cached CLIP score coverage mismatch")
                scores = cached_scores[episode["episode_id"]]
            else:
                with Image.open(item["absolute_rgb_path"]) as opened:
                    image = opened.convert("RGB")
                scores = scorer.score_episode(image, item["goal_text"], episode)
            features = {
                str(candidate["candidate_id"]): candidate_features(candidate, item["goal_text"], scores)
                for candidate in episode["candidates"]
            }
            feature_rows.append({
                "episode_id": episode["episode_id"], "features": features,
                "correct_candidate_ids": sorted(correct_candidate_ids(item, episode)),
            })
            all_truth.append(item)
            all_episodes.append(episode)
            print(f"relational-features {len(all_episodes)}/281 {episode['episode_id']}", flush=True)
    positions = [int(item["observation_id"].rsplit("-", 1)[1]) for item in all_truth]
    if positions != list(range(1, HOLDOUT_END_POSITION + 1)):
        raise TrainError("consumed GroundBench positions are not contiguous 1..281")
    training_rows = feature_rows[:TRAIN_END_POSITION]
    model, pairwise_example_count = fit_pairwise_ranker(training_rows)
    output_dir.mkdir(parents=True)
    model_path = output_dir / "relational-ranker.joblib"
    joblib.dump(model, model_path)
    holdout_truth = all_truth[HOLDOUT_START_POSITION - 1:HOLDOUT_END_POSITION]
    holdout_v0 = all_episodes[HOLDOUT_START_POSITION - 1:HOLDOUT_END_POSITION]
    holdout_features = feature_rows[HOLDOUT_START_POSITION - 1:HOLDOUT_END_POSITION]
    holdout_v1 = [
        rerank_episode(episode, model_scores(model, row["features"]))
        for episode, row in zip(holdout_v0, holdout_features)
    ]
    provider_metrics = ranking_evaluation(holdout_truth, holdout_v0)
    ranker_metrics = ranking_evaluation(holdout_truth, holdout_v1)
    gate = development_gate(provider_metrics, ranker_metrics)
    report = {
        "schema_version": SCHEMA_VERSION, "role": "CONSUMED_DEVELOPMENT_WITH_HELD_OUT_BLOCK",
        "source_receipts": source_receipts,
        "partitions": {"train_positions": [1, TRAIN_END_POSITION], "holdout_positions": [HOLDOUT_START_POSITION, HOLDOUT_END_POSITION]},
        "feature_names": list(FEATURE_NAMES),
        "model": {"type": "PAIRWISE_LOGISTIC_REGRESSION", "C": 1.0, "solver": "liblinear", "random_state": 0,
                  "pairwise_example_count": pairwise_example_count, "model_sha256": sha256_file(model_path)},
        "provider_calls_added": 0, "teacher_calls": 0, "retry_count": 0, "reruns": 0,
        "holdout": {"provider_order": provider_metrics, "relational_ranker": ranker_metrics},
        "gate": gate, "claim_ceiling": CLAIM_CEILING,
    }
    report["report_sha256"] = materializer.content_sha256(report)
    atomic_json(output_dir / "development-report.json", report)
    atomic_json(output_dir / "feature-ledger.json", {
        "schema_version": "groundbench_relational_candidate_feature_ledger_v0",
        "feature_names": list(FEATURE_NAMES), "rows": feature_rows,
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--cached-clip-report", type=Path, required=True)
    parser.add_argument("--clip-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
