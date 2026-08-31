#!/usr/bin/env python3
"""Run a frozen local carrier and adjudicate the fixed selective direct exit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cycle_component_open_set_posthoc as local_carrier  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-selective-cascade-confirmation-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-selective-cascade-confirmation-result-v1"


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
    inner_path = HERE / protocol["local_carrier"]["protocol_path"]
    require(sha256(inner_path) == protocol["local_carrier"]["protocol_sha256"], "LOCAL_PROTOCOL_HASH")
    cohort_path = HERE / protocol["source"]["cohort_path"]
    require(sha256(cohort_path) == protocol["source"]["cohort_sha256"], "COHORT_HASH")

    temporary_root = ROOT / "artifacts.local" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".json", dir=temporary_root, delete=False) as stream:
        local_result_path = Path(stream.name)
    try:
        local_carrier.replay(inner_path, local_result_path)
        local_result = load_json(local_result_path)
    finally:
        local_result_path.unlink(missing_ok=True)

    require(
        local_result["conclusion"] in protocol["local_carrier"]["allowed_conclusions"],
        "LOCAL_CONCLUSION",
    )
    threshold = float(protocol["decision_rule"]["minimum_direct_local_cycle_fraction"])
    decisions: dict[str, Any] = {}
    for pair_id, local in local_result["decisions"].items():
        local_commit = bool(local["commit"])
        cycle_fraction = float(local["reference_cycle_fraction"])
        direct_exit = bool(local_commit and cycle_fraction >= threshold)
        request = bool(local_commit and not direct_exit)
        commit = direct_exit
        decisions[pair_id] = {
            **{key: value for key, value in local.items() if key not in {"commit", "correct"}},
            "local_commit": local_commit,
            "strong_local_direct_exit": direct_exit,
            "corroboration_requested": request,
            "corroboration_status": (
                "NOT_REQUESTED_DIRECT_EXIT" if direct_exit
                else "NOT_REQUESTED_LOCAL_NON_COMMIT" if not local_commit
                else "REQUESTED_BUT_NOT_AVAILABLE_IN_FROZEN_SOURCE"
            ),
            "commit": commit,
            "correct": commit if local["label"] == "target_present" else not commit,
        }

    positives = [row for row in decisions.values() if row["label"] == "target_present"]
    negatives = [row for row in decisions.values() if row["label"] == "target_absent"]
    positive_commits = sum(row["commit"] for row in positives)
    false_commits = sum(row["commit"] for row in negatives)
    requests = sum(row["corroboration_requested"] for row in decisions.values())
    gate = protocol["decision_gate"]
    gate_met = bool(
        len(positives) == int(gate["required_positive_pairs"])
        and len(negatives) == int(gate["required_target_absent_pairs"])
        and positive_commits >= int(gate["minimum_positive_commits"])
        and false_commits <= int(gate["maximum_target_absent_false_commits"])
        and requests <= int(gate["maximum_unserved_corroboration_requests"])
    )
    committed = positive_commits + false_commits
    write_json(output_path, {
        "schema": RESULT_SCHEMA,
        "authority": "FRESH_ONE_TARGET_SCAN_FAMILY_SELECTIVE_CASCADE_CHALLENGE_RESULT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": sha256(Path(__file__))},
        "source": {"cohort_path": cohort_path.name, "cohort_sha256": sha256(cohort_path)},
        "local_carrier": {
            "protocol_path": inner_path.name,
            "protocol_sha256": sha256(inner_path),
            "conclusion": local_result["conclusion"],
            "decisions": local_result["decisions"],
            "prompt_receipts": local_result["prompt_receipts"],
            "reference_support_receipts": local_result["reference_support_receipts"],
            "query_support_receipts": local_result["query_support_receipts"],
        },
        "conclusion": (
            "L10_3RSCAN_SELECTIVE_CORROBORATION_CASCADE_SCAN_FAMILY_CHALLENGE_GATE_MET"
            if gate_met else "L10_3RSCAN_SELECTIVE_CORROBORATION_CASCADE_SCAN_FAMILY_CHALLENGE_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "positive_pairs": len(positives),
            "positive_commits": positive_commits,
            "target_absent_pairs": len(negatives),
            "target_absent_false_commits": false_commits,
            "committed_precision": positive_commits / committed if committed else 0.0,
            "corroboration_requests": requests,
        },
        "decisions": decisions,
        "runtime": {
            **local_result["runtime"],
            "selective_adjudication_model_calls": 0,
            "measured_early_exit_runtime": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
