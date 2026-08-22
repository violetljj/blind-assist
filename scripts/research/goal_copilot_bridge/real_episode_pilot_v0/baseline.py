"""Run the frozen provider-output adapter through selective-guidance V0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.research.goal_copilot_bridge.selective_guidance_v0.contract import (
    CandidateCardinality,
    CurrentFrameObservation,
    OutputToken,
    RangeBucket,
    decide,
)


def _range_bucket(range_m: float | None, config: dict) -> RangeBucket:
    if range_m is None:
        return RangeBucket.UNKNOWN
    if range_m <= float(config["handoff_near_max_m"]):
        return RangeBucket.NEAR
    if range_m >= float(config["far_min_m"]):
        return RangeBucket.FAR
    return RangeBucket.APPROACHING


def run_baseline(public: dict, provider: dict, config: dict) -> dict:
    if public.get("schema_version") != "blindassist_real_episode_public_manifest_v0":
        raise ValueError("public manifest schema mismatch")
    if config.get("schema_version") != "blindassist_real_episode_baseline_config_v0":
        raise ValueError("baseline config schema mismatch")
    if config.get("identity_persistence") is not False or config.get("model_or_threshold_search") is not False:
        raise ValueError("baseline config violates successor boundary")
    provider_rows = {row["observation_id"]: row for row in provider.get("observations", [])}
    predictions = []
    for episode in public["episodes"]:
        for frame in episode["observations"]:
            row = provider_rows.get(frame["observation_id"], {"candidates": []})
            candidates = sorted(row.get("candidates", []), key=lambda item: int(item["rank"]))[: int(config["maximum_proposals"])]
            selected = candidates[0] if candidates else None
            x = float(selected["x_center_fraction"]) if selected else 0.5
            direction = OutputToken.GUIDE_LEFT if x < float(config["bearing_left_max_fraction"]) else (
                OutputToken.GUIDE_RIGHT if x > float(config["bearing_right_min_fraction"]) else OutputToken.GUIDE_STRAIGHT
            )
            range_bucket = _range_bucket(float(selected["range_m"]) if selected and selected.get("range_m") is not None else None, config)
            obs = CurrentFrameObservation(
                goal_contract=episode["goal_contract"],
                frame_id=frame["observation_id"],
                observed_at_ms=int(frame["timestamp_ms"]),
                decision_at_ms=int(frame["timestamp_ms"]) + int(round(float(row.get("latency_ms", 0.0)))),
                visible_candidate_ids=tuple(item["candidate_id"] for item in candidates),
                selected_referent=selected["candidate_id"] if selected else None,
                cardinality=CandidateCardinality(row.get("candidate_cardinality", "AMBIGUOUS")),
                target_visible=row.get("target_visible"),
                selection_authorized=bool(row.get("selection_authorized", False)),
                requested_direction=direction if selected else None,
                range_bucket=range_bucket,
                range_uncertainty=row.get("range_uncertainty"),
                evidence_ttl_ms=max(1_000, int(round(float(row.get("latency_ms", 0.0)))) + 1),
                stop_for_safety=bool(row.get("stop_for_safety", False)),
                handoff_ready=range_bucket is RangeBucket.NEAR and bool(row.get("selection_authorized", False)),
                latency_ms=float(row.get("latency_ms", 0.0)),
            )
            decision = decide(obs)
            predictions.append({
                "episode_id": episode["episode_id"],
                "observation_id": frame["observation_id"],
                "candidate_ids": [item["candidate_id"] for item in candidates],
                "selected_referent": decision.selected_referent,
                "decision_state": decision.status.value,
                "command": decision.command.value if decision.command else None,
                "range_bucket": decision.range_bucket.value,
                "confident_spoken_guidance": decision.command in {OutputToken.GUIDE_LEFT, OutputToken.GUIDE_RIGHT, OutputToken.GUIDE_STRAIGHT},
                "latency_ms": obs.latency_ms,
            })
    return {
        "schema_version": "blindassist_real_episode_baseline_prediction_v0",
        "private_truth_access": False,
        "provider_identity": config["provider_identity"],
        "model_or_threshold_search": False,
        "identity_persistence": False,
        "predictions": predictions,
        "claim_ceiling": config["claim_ceiling"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--provider-observations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("baseline output already exists")
    payload = run_baseline(
        json.loads(args.public_manifest.read_text(encoding="utf-8")),
        json.loads(args.provider_observations.read_text(encoding="utf-8")),
        json.loads(args.config.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
