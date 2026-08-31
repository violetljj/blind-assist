#!/usr/bin/env python3
"""Recompose consumed L10 receipts into a selective corroboration cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROTOCOL_SCHEMA = "blindassist-l10-3rscan-selective-corroboration-cascade-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-selective-corroboration-cascade-posthoc-result-v1"


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

    old_local = inputs["consumed_sibling_local"]
    old_complementary = inputs["consumed_sibling_complementary"]
    new_partial = inputs["consumed_physical_target_partial_confirmation"]
    require(set(old_local["decisions"]) == set(old_complementary["decisions"]), "OLD_PAIR_SET")
    minimum_direct_fraction = float(protocol["decision_rule"]["minimum_direct_local_cycle_fraction"])

    panels: dict[str, dict[str, Any]] = {
        "consumed_sibling_panel": {},
        "consumed_physical_target_panel": {},
    }
    for pair_id, local_decision in old_local["decisions"].items():
        complementary_decision = old_complementary["decisions"][pair_id]
        require(local_decision["label"] == complementary_decision["label"], f"OLD_LABEL:{pair_id}")
        panels["consumed_sibling_panel"][pair_id] = {
            "label": local_decision["label"],
            "local_commit": bool(complementary_decision["predecessor_bilateral_mask_paired_commit"]),
            "primary_local_cycle_fraction": float(old_local["prompt_receipts"][pair_id]["all_cycle_fraction"]),
            "complementary_corroboration": bool(complementary_decision["complementary_corroboration"]),
        }

    for pair_id, partial_decision in new_partial["decisions"].items():
        panels["consumed_physical_target_panel"][pair_id] = {
            "label": partial_decision["label"],
            "local_commit": bool(partial_decision["local_bilateral_commit"]),
            "primary_local_cycle_fraction": float(
                new_partial["prompt_receipts"][pair_id]["primary"]["all_cycle_fraction"]
            ),
            "complementary_corroboration": bool(
                partial_decision["global_epipolar_support"]
                or partial_decision["active_query_bidirectional_majority_support"]
            ),
        }

    decisions: dict[str, Any] = {}
    panel_metrics: dict[str, Any] = {}
    for panel_name, rows in panels.items():
        for pair_id, row in rows.items():
            strong_local = bool(
                row["local_commit"]
                and row["primary_local_cycle_fraction"] >= minimum_direct_fraction
            )
            request_corroboration = bool(row["local_commit"] and not strong_local)
            commit = bool(
                row["local_commit"]
                and (strong_local or (request_corroboration and row["complementary_corroboration"]))
            )
            decisions[pair_id] = {
                "id": pair_id,
                "panel": panel_name,
                **row,
                "strong_local_direct_exit": strong_local,
                "corroboration_requested": request_corroboration,
                "commit": commit,
                "correct": commit if row["label"] == "target_present" else not commit,
            }

        panel_rows = [row for row in decisions.values() if row["panel"] == panel_name]
        positives = [row for row in panel_rows if row["label"] == "target_present"]
        negatives = [row for row in panel_rows if row["label"] == "target_absent"]
        panel_metrics[panel_name] = {
            "positive_pairs": len(positives),
            "positive_commits": sum(row["commit"] for row in positives),
            "target_absent_pairs": len(negatives),
            "target_absent_false_commits": sum(row["commit"] for row in negatives),
            "corroboration_requests": sum(row["corroboration_requested"] for row in panel_rows),
        }

    positives = [row for row in decisions.values() if row["label"] == "target_present"]
    negatives = [row for row in decisions.values() if row["label"] == "target_absent"]
    positive_commits = sum(row["commit"] for row in positives)
    false_commits = sum(row["commit"] for row in negatives)
    corroboration_requests = sum(row["corroboration_requested"] for row in decisions.values())
    gate = protocol["decision_gate"]
    gate_met = bool(
        len(positives) == int(gate["required_positive_pairs"])
        and len(negatives) == int(gate["required_target_absent_pairs"])
        and positive_commits >= int(gate["minimum_positive_commits"])
        and false_commits <= int(gate["maximum_target_absent_false_commits"])
        and all(
            panel_metrics[name]["positive_commits"] >= int(requirement["minimum_positive_commits"])
            and panel_metrics[name]["target_absent_false_commits"]
            <= int(requirement["maximum_target_absent_false_commits"])
            for name, requirement in gate["per_panel"].items()
        )
    )
    committed = positive_commits + false_commits
    write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "CONSUMED_ZERO_MODEL_POSTHOC_SELECTIVE_CASCADE_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
            "inputs": {
                name: {"path": row["path"], "sha256": row["sha256"]}
                for name, row in protocol["inputs"].items()
            },
            "conclusion": (
                "L10_3RSCAN_SELECTIVE_CORROBORATION_CASCADE_POSTHOC_DEVELOPMENT_GATE_MET"
                if gate_met else "L10_3RSCAN_SELECTIVE_CORROBORATION_CASCADE_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
            ),
            "gate_met": gate_met,
            "metrics": {
                "positive_pairs": len(positives),
                "positive_commits": positive_commits,
                "positive_recall": positive_commits / len(positives) if positives else 0.0,
                "target_absent_pairs": len(negatives),
                "target_absent_false_commits": false_commits,
                "committed_precision": positive_commits / committed if committed else 0.0,
                "corroboration_requests": corroboration_requests,
                "corroboration_request_fraction": corroboration_requests / len(decisions),
                "counterfactual_universal_corroboration_requests": len(decisions),
                "counterfactual_extra_branch_avoidance_fraction": 1.0 - corroboration_requests / len(decisions),
                "panels": panel_metrics,
            },
            "decisions": decisions,
            "runtime": {"model_calls": 0, "receipt_recompositions": len(decisions)},
            "literature_motivation": protocol["literature_motivation"],
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
