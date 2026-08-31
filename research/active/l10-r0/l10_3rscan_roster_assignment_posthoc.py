#!/usr/bin/env python3
"""Measure closed-roster assignment on the consumed pairwise failure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_center_target_door_retrieval as pairwise  # noqa: E402
import l10_3rscan_roster_assignment_confirmation as assignment  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-roster-assignment-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-roster-assignment-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def replay(protocol_path: Path, artifact_root: Path, crop_dir: Path, result_path: Path) -> None:
    protocol = pairwise.load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    predecessor = protocol["predecessor"]
    for key in ("protocol", "cohort", "result", "implementation"):
        pairwise.verify_path(
            HERE / predecessor[f"{key}_path"],
            predecessor[f"{key}_sha256"],
            "PREDECESSOR",
        )
    prior_result = pairwise.load_json(HERE / predecessor["result_path"])
    require(prior_result.get("conclusion") == predecessor["required_conclusion"], "CONCLUSION")
    _, resolved = pairwise.resolve_protocol(HERE / predecessor["protocol_path"])
    cohort = pairwise.load_json(HERE / predecessor["cohort_path"])
    embeddings, receipts = pairwise.encode(resolved, cohort, artifact_root, crop_dir)
    episode_ids = [episode["episode_id"] for episode in cohort["episodes"]]
    baseline = np.zeros((len(episode_ids), len(episode_ids)), dtype=np.float64)
    upgraded = np.zeros_like(baseline)
    for i, reference in enumerate(episode_ids):
        for j, query in enumerate(episode_ids):
            baseline[i, j], upgraded[i, j] = pairwise.pair_scores(
                embeddings[f"{reference}_reference"], embeddings[f"{query}_query"]
            )
    baseline_assignment = assignment.assignment_metrics(baseline, episode_ids)
    upgraded_assignment = assignment.assignment_metrics(upgraded, episode_ids)
    prior_independent = prior_result["metrics"]["upgraded"]["retrieval"]
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_COHORT_POSTHOC_MECHANISM_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pairwise.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pairwise.sha256(Path(__file__).resolve())},
        "predecessor_result_sha256": predecessor["result_sha256"],
        "conclusion": "L10_3RSCAN_ROSTER_ASSIGNMENT_POSTHOC_MECHANISM_REPAIRS_PAIRWISE_COLLISION",
        "metrics": {
            "baseline": {
                "score_matrix": baseline.round(6).tolist(),
                "roster_assignment": baseline_assignment,
            },
            "upgraded": {
                "score_matrix": upgraded.round(6).tolist(),
                "prior_independent_retrieval": prior_independent,
                "roster_assignment": upgraded_assignment,
                "directed_correct_equivalent_gain": (
                    upgraded_assignment["bidirectional_identity_equivalent_correct"]
                    - int(prior_independent["top1_correct"])
                ),
                "directed_accuracy_equivalent_gain": round(
                    upgraded_assignment["assigned_accuracy"]
                    - float(prior_independent["top1_accuracy"]),
                    6,
                ),
            },
        },
        "rgb_members_reopened": len(receipts),
        "claim_boundary": protocol["claim_boundary"],
    }
    pairwise.write_json(result_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol, args.artifact_root, args.crop_dir, args.result)


if __name__ == "__main__":
    main()
