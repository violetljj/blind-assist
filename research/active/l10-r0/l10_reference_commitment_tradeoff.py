"""Paired analysis of already-consumed SEVN commitment outputs; no model access."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ARMS = ("FIXED_SWEEP", "TRIGGERED_ACTIVE", "TRIGGERED_VERIFIED")
OUTCOMES = ("CORRECT", "WRONG", "UNKNOWN")


def analyze(result: dict) -> dict:
    episodes = result["episodes"]
    if not episodes or len({row["episode_id"] for row in episodes}) != len(episodes):
        raise ValueError("Need a nonempty set of unique paired episodes")
    totals = {arm: {**dict.fromkeys(OUTCOMES, 0), "online_views": 0} for arm in ARMS}
    rows = []
    attribution = Counter()
    for episode in episodes:
        arms = {name: {**entry, "outcome": entry["outcome"].removesuffix("_BINDING")}
                for name, entry in episode["arms"].items()}
        for arm in ARMS:
            outcome = arms[arm]["outcome"]
            views = arms[arm]["extra_observation_count"]
            if outcome not in OUTCOMES or type(views) is not int or views < 0:
                raise ValueError(f"Invalid paired outcome or view count: {episode['episode_id']}/{arm}")
            totals[arm][outcome] += 1
            totals[arm]["online_views"] += views
        targets = [entry for entry in episode.get("diagnostic_controls", [])
                   if entry["kind"] == "TARGET_ORACLE"]
        if len(targets) > 1:
            raise ValueError("Ambiguous target oracle; do not silently select one")
        target_supported = targets[0]["verifier"]["accepted"] if targets else None
        lost_correct = (arms["TRIGGERED_ACTIVE"]["outcome"] == "CORRECT"
                        and arms["TRIGGERED_VERIFIED"]["outcome"] != "CORRECT")
        loss_kind = None
        if lost_correct:
            loss_kind = (
                "target_box_supported_but_runtime_commit_rejected" if target_supported is True
                else "target_box_not_supported" if target_supported is False
                else "target_support_not_evaluable"
            )
            attribution[loss_kind] += 1
        rows.append({
            "episode_id": episode["episode_id"],
            "outcomes": {arm: arms[arm]["outcome"] for arm in ARMS},
            "online_views_saved_by_triggering": arms["FIXED_SWEEP"]["extra_observation_count"]
                                                - arms["TRIGGERED_ACTIVE"]["extra_observation_count"],
            "target_oracle_supported": target_supported,
            "lost_correct_attribution": loss_kind,
            "runtime_reference_reasons": [ref["reason"] for ref in
                                           (episode.get("verification") or {}).get("references", [])],
        })
    for counts in totals.values():
        commits = counts["CORRECT"] + counts["WRONG"]
        counts["commit_precision"] = counts["CORRECT"] / commits if commits else None
    baseline = totals["TRIGGERED_ACTIVE"]
    retained = sum(row["outcomes"]["TRIGGERED_ACTIVE"] == "CORRECT"
                   and row["outcomes"]["TRIGGERED_VERIFIED"] == "CORRECT" for row in rows)
    wrong_removed = sum(row["outcomes"]["TRIGGERED_ACTIVE"] == "WRONG"
                        and row["outcomes"]["TRIGGERED_VERIFIED"] != "WRONG" for row in rows)
    saved = totals["FIXED_SWEEP"]["online_views"] - baseline["online_views"]
    fixed_cost = totals["FIXED_SWEEP"]["online_views"]
    return {
        "analysis_kind": "CONSUMED_PAIRED_DIAGNOSTIC",
        "original_decision": result["decision"],
        "episode_count": len(episodes), "arms": totals,
        "correct_retention": retained / baseline["CORRECT"] if baseline["CORRECT"] else None,
        "wrong_commit_reduction": wrong_removed / baseline["WRONG"] if baseline["WRONG"] else None,
        "wrong_commit_reduction_status": "EVALUABLE" if baseline["WRONG"] else "NOT_EVALUABLE_NO_BASELINE_ERRORS",
        "online_views_saved_by_triggering": saved,
        "online_view_reduction_fraction": saved / fixed_cost if fixed_cost else None,
        "separate_reference_setup_views": result["reference_setup_views"],
        "lost_correct_attribution": dict(attribution), "paired_episodes": rows,
        "claim_boundary": "Existing episode outputs only. No new predictions, tuning, fresh confirmation, "
                          "causal proof from oracle labels, or measured workflow-speed improvement.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_bytes()
    analysis = analyze(json.loads(source))
    analysis["input_sha256"] = hashlib.sha256(source).hexdigest()
    artifact_root = Path(__file__).resolve().parents[3] / "artifacts.local"
    args.output.resolve().relative_to(artifact_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if args.input.read_bytes() != source:
        raise RuntimeError("Input changed during read-only analysis")
    print(json.dumps({key: analysis[key] for key in (
        "episode_count", "correct_retention", "wrong_commit_reduction_status",
        "online_views_saved_by_triggering", "lost_correct_attribution")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
