#!/usr/bin/env python3
"""Recompose frozen global-geometry and active-query receipts without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROTOCOL_SCHEMA = "blindassist-l10-3rscan-complementary-corroboration-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-complementary-corroboration-posthoc-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def replay(protocol_path: Path, output_path: Path) -> None:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    require(sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    inputs: dict[str, dict[str, Any]] = {}
    for name, row in protocol["inputs"].items():
        path = HERE / row["path"]
        require(sha256(path) == row["sha256"], f"INPUT_HASH:{name}")
        value = load_json(path)
        require(value["conclusion"] == row["required_conclusion"], f"INPUT_CONCLUSION:{name}")
        inputs[name] = value

    global_result = inputs["global_epipolar"]
    active_result = inputs["active_query"]
    require(set(global_result["decisions"]) == set(active_result["decisions"]), "PAIR_SET")
    minimum_majority = float(protocol["decision_rule"]["minimum_directional_active_query_fraction"])
    decisions: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    for pair_id, global_decision in global_result["decisions"].items():
        active_decision = active_result["decisions"][pair_id]
        require(global_decision["label"] == active_decision["label"], f"PAIR_LABEL:{pair_id}")
        query_receipt = active_result["query_consistency_receipts"][pair_id]
        forward = query_receipt["primary_to_active_query"]
        reverse = query_receipt["active_to_primary_query"]
        forward_fraction = float(forward["paired_cycle_fraction"]) if forward is not None else 0.0
        reverse_fraction = float(reverse["paired_cycle_fraction"]) if reverse is not None else 0.0
        active_majority = bool(
            query_receipt["both_reference_query_branches_supported"]
            and forward_fraction >= minimum_majority
            and reverse_fraction >= minimum_majority
        )
        global_support = bool(global_decision["global_epipolar_support"])
        local_commit = bool(global_decision["predecessor_bilateral_mask_paired_commit"])
        corroborated = global_support or active_majority
        commit = local_commit and corroborated
        receipts[pair_id] = {
            "local_bilateral_commit": local_commit,
            "global_epipolar_support": global_support,
            "active_query_forward_paired_fraction": forward_fraction,
            "active_query_reverse_paired_fraction": reverse_fraction,
            "active_query_bidirectional_majority_support": active_majority,
            "complementary_corroboration": corroborated,
        }
        decisions[pair_id] = {
            **{key: value for key, value in global_decision.items() if key not in {"commit", "correct"}},
            "active_query_bidirectional_majority_support": active_majority,
            "complementary_corroboration": corroborated,
            "commit": commit,
            "correct": commit if global_decision["label"] == "target_present" else not commit,
        }

    positives = [row for row in decisions.values() if row["label"] == "target_present"]
    negatives = [row for row in decisions.values() if row["label"] == "target_absent"]
    positive_commits = sum(bool(row["commit"]) for row in positives)
    false_commits = sum(bool(row["commit"]) for row in negatives)
    gate = protocol["decision_gate"]
    gate_met = bool(
        len(positives) == int(gate["required_positive_pairs"])
        and len(negatives) == int(gate["required_target_absent_pairs"])
        and positive_commits >= int(gate["minimum_positive_commits"])
        and false_commits <= int(gate["maximum_target_absent_false_commits"])
    )
    write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_ZERO_MODEL_POSTHOC_COMPLEMENTARY_CORROBORATION_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "inputs": {
                name: {"path": row["path"], "sha256": row["sha256"]}
                for name, row in protocol["inputs"].items()
            },
            "conclusion": (
                "L10_3RSCAN_COMPLEMENTARY_CORROBORATION_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met else "L10_3RSCAN_COMPLEMENTARY_CORROBORATION_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "positive_pairs": len(positives),
                "positive_commits": positive_commits,
                "target_absent_pairs": len(negatives),
                "target_absent_false_commits": false_commits,
                "committed_precision": (
                    positive_commits / (positive_commits + false_commits)
                    if positive_commits + false_commits else 0.0
                ),
            },
            "decisions": decisions,
            "corroboration_receipts": receipts,
            "runtime": {"model_calls": 0, "receipt_recompositions": len(decisions)},
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
