"""Create-once analysis for a complete evaluable B5-A run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .fresh_benchmark import load_fresh_benchmark
from .protocol_b5a import (
    ARMS,
    GENERATIONS_PER_TRAJECTORY,
    PAIRED_IDENTITIES,
    PROTOCOL_ID,
    build_protocol_manifest,
    canonical_manifest_sha256,
)
from .run_b4a import _sha256
from .analyze_b4a import EPSILON, _events, _trajectory


def analyze(repo_root: Path, run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "execution_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if frozen != build_protocol_manifest(repo_root):
        raise RuntimeError("protocol does not match frozen B5-A implementation")
    if manifest.get("terminal") != "B5A_EXECUTION_COMPLETE":
        raise RuntimeError("run is not complete and evaluable")
    if manifest.get("protocol_manifest_sha256") != canonical_manifest_sha256(frozen):
        raise RuntimeError("execution manifest protocol identity mismatch")
    events_path = run_dir / "events.jsonl"
    events = _events(events_path)
    benchmark = load_fresh_benchmark()
    qualified = {
        row["instance_id"]: row
        for row in json.loads(
            (repo_root / frozen["fresh_harder_cohort"]["certificate_path"]).read_text(encoding="utf-8")
        )["instances"]
    }
    instances = {row["instance_id"]: row for row in benchmark["instances"]}
    expected = len(PAIRED_IDENTITIES) * len(ARMS) * GENERATIONS_PER_TRAJECTORY
    completions = [row for row in events if row.get("kind") == "completion"]
    if len(completions) != expected:
        raise RuntimeError("completion count differs from frozen budget")
    if any(row.get("protocol_id") != PROTOCOL_ID for row in events):
        raise RuntimeError("event protocol identity mismatch")
    trajectories: list[dict[str, Any]] = []
    for pair in PAIRED_IDENTITIES:
        instance_id = str(pair["instance_id"])
        instance = dict(instances[instance_id])
        instance["qualified_global_score"] = qualified[instance_id]["global_score"]
        identity = int(pair["paired_identity"])
        for arm in ARMS:
            selected = [
                row
                for row in events
                if row.get("instance_id") == instance_id
                and row.get("paired_identity") == identity
                and row.get("arm") == arm
            ]
            trajectories.append(_trajectory(selected, instance, arm, identity))
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trajectories:
        grouped[(row["instance_id"], row["paired_identity"])][row["arm"]] = row
    pairs = []
    wins = losses = ties = 0
    for (instance_id, identity), arms in sorted(grouped.items()):
        control = arms["structured_control"]
        balanced = arms["structured_balanced"]
        delta = balanced["normalized_progress"] - control["normalized_progress"]
        if delta > EPSILON:
            disposition = "balanced_win"
            wins += 1
        elif delta < -EPSILON:
            disposition = "balanced_loss"
            losses += 1
        else:
            disposition = "tie"
            ties += 1
        pairs.append(
            {
                "instance_id": instance_id,
                "paired_identity": identity,
                "control_normalized_progress": control["normalized_progress"],
                "balanced_normalized_progress": balanced["normalized_progress"],
                "paired_normalized_delta": delta,
                "disposition": disposition,
            }
        )
    deltas = [row["paired_normalized_delta"] for row in pairs]
    control_rows = [row for row in trajectories if row["arm"] == "structured_control"]
    balanced_rows = [row for row in trajectories if row["arm"] == "structured_balanced"]
    control_global = sum(row["global_optimum_reached"] for row in control_rows)
    balanced_global = sum(row["global_optimum_reached"] for row in balanced_rows)
    control_unsafe = sum(row["unsafe_count"] for row in control_rows)
    balanced_unsafe = sum(row["unsafe_count"] for row in balanced_rows)
    control_invalid = sum(row["semantic_invalid_count"] for row in control_rows)
    balanced_invalid = sum(row["semantic_invalid_count"] for row in balanced_rows)
    integrity = all(row["operator_integrity"] for row in balanced_rows)
    matched_cost = len(control_rows) == len(balanced_rows) == 9 and expected == 144
    replicated = (
        statistics.median(deltas) > EPSILON
        and wins >= 6
        and losses == 0
        and balanced_global >= control_global
        and balanced_unsafe <= control_unsafe
        and balanced_invalid <= control_invalid
        and matched_cost
        and integrity
    )
    return {
        "schema": "l10m_b5a_result_v1",
        "protocol_id": PROTOCOL_ID,
        "run_id": run_dir.name,
        "model_calls": expected,
        "terminal": "B5A_EVALUABLE_COMPLETE",
        "scientific_verdict": (
            "B5A_GENERALIZATION_REPLICATED_ADMITTED_L10M_SEARCH_OPERATOR"
            if replicated
            else "B5A_GENERALIZATION_NOT_REPLICATED"
        ),
        "primary": {
            "median_paired_normalized_progress_delta": statistics.median(deltas),
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "control_global_optimum_reach": control_global,
            "balanced_global_optimum_reach": balanced_global,
            "operator_integrity": integrity,
            "replication_rule_passed": replicated,
        },
        "safety_and_validity": {
            "control_unsafe_count": control_unsafe,
            "balanced_unsafe_count": balanced_unsafe,
            "control_semantic_invalid_count": control_invalid,
            "balanced_semantic_invalid_count": balanced_invalid,
        },
        "cost": {
            "control_model_calls": 72,
            "balanced_model_calls": 72,
            "matched_model_call_cost": matched_cost,
            "monetary_cost": "not_available_from_ChatGPT_authenticated_provider_receipts",
        },
        "paired_results": pairs,
        "trajectories": trajectories,
        "source_sha256": {
            "events.jsonl": _sha256(events_path),
            "execution_manifest.json": _sha256(run_dir / "execution_manifest.json"),
            "protocol.json": _sha256(protocol_path),
        },
        "global_optimum_boundary": frozen["global_optimum_boundary"],
        "claim_ceiling": frozen["claim_ceiling"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite create-once result: {args.output}")
    result = analyze(args.repo_root.resolve(), args.run_dir.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "terminal": result["terminal"], "verdict": result["scientific_verdict"]}))


if __name__ == "__main__":
    main()
