"""Create-once GC2-A protocol freeze and zero-model characterization runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_2a.evaluator import evaluate_condition
from scripts.research.goal_copilot_2a.noise import CORRUPTIONS, SEVERITIES, condition_names

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
GC1 = HERE.parent / "goal_copilot_bridge" / "pilot"
BASELINE = GC1 / "initial_policy.py"
WINNER = HERE / "frozen_gc1_winner.py"
SCENARIOS = GC1 / "dev_scenarios.json"
PROTOCOL = HERE / "protocol.json"
EXPECTED_WINNER = "24d4e57374dd99363700ae881d18db536e48ec5f79f39e95c5b873e96edbc3a1"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(payload))


def freeze(output_root: Path) -> Path:
    if output_root.exists():
        raise FileExistsError(f"GC2-A root already exists: {output_root}")
    if sha256(WINNER) != EXPECTED_WINNER:
        raise RuntimeError("frozen GC1 winner digest mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["model_call_budget"] != 0:
        raise RuntimeError("GC2-A must remain zero-model")
    output_root.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "GOAL_COPILOT_2A_PROTOCOL_FROZEN_ZERO_MODEL_RUN_AUTHORIZED",
        "source_commit": git_head(),
        "protocol_sha256": sha256(PROTOCOL),
        "baseline_sha256": sha256(BASELINE),
        "winner_sha256": sha256(WINNER),
        "scenario_sha256": sha256(SCENARIOS),
        "conditions": list(condition_names()),
        "model_call_budget": 0,
        "scenario_evidence_role": protocol["inputs"]["scenario_evidence_role"],
        "gc2b_admission": protocol["gc2b_admission"],
        "gc1_status": protocol["gc1_status"],
        "resume_authorized": False,
    }
    digest = hashlib.sha256(canonical(payload)).hexdigest()
    payload["protocol_seal_digest"] = digest
    path = output_root / "formal_protocol_seal.json"
    write_once(path, payload)
    return path


def _first_failure(cells: dict[str, Any], corruption: str, clean_completion: int) -> str | None:
    for severity in SEVERITIES:
        metrics = cells[f"{corruption}_{severity}"]["metrics"]
        if (
            metrics["completion_count"] < clean_completion
            or metrics["unsafe_guidance"] > 0
            or metrics["premature_completion"] > 0
            or metrics["eligible_reacquisition_rate"] < 0.8
        ):
            return severity
    return None


def run(output_root: Path) -> dict[str, Any]:
    seal_path = output_root / "formal_protocol_seal.json"
    if not seal_path.is_file():
        raise RuntimeError("GC2-A protocol seal missing")
    seal = json.loads(seal_path.read_text())
    if seal["source_commit"] != git_head() or seal["winner_sha256"] != sha256(WINNER):
        raise RuntimeError("GC2-A frozen input drift")
    if (output_root / "formal_closeout.json").exists():
        raise FileExistsError("GC2-A is already closed")

    policies = {"GC1_BASELINE": BASELINE, "GC1_WINNER": WINNER}
    matrix: dict[str, dict[str, Any]] = {}
    for policy_name, policy_path in policies.items():
        matrix[policy_name] = {
            condition: evaluate_condition(policy_path, SCENARIOS, condition)
            for condition in condition_names()
        }
    winner_cells = matrix["GC1_WINNER"]
    baseline_cells = matrix["GC1_BASELINE"]
    primary = winner_cells["COMBINED_MODERATE"]["metrics"]
    threshold = json.loads(PROTOCOL.read_text())["gc2b_admission"]
    reasons = []
    if primary["completion_rate"] < threshold["winner_completion_rate_below"]:
        reasons.append("COMBINED_MODERATE_COMPLETION_BELOW_0_8")
    if primary["unsafe_guidance"] > threshold["or_unsafe_guidance_above"]:
        reasons.append("COMBINED_MODERATE_UNSAFE_GUIDANCE")
    if primary["premature_completion"] > threshold["or_premature_completion_above"]:
        reasons.append("COMBINED_MODERATE_PREMATURE_COMPLETION")
    if primary["eligible_reacquisition_rate"] < threshold["or_eligible_reacquisition_rate_below"]:
        reasons.append("COMBINED_MODERATE_REACQUISITION_COLLAPSE")
    admitted = bool(reasons)
    first_failures = {
        corruption: _first_failure(winner_cells, corruption, winner_cells["CLEAN"]["metrics"]["completion_count"])
        for corruption in CORRUPTIONS
    }
    summary = {
        "schema_version": 1,
        "protocol_id": "GOAL-COPILOT-2A",
        "model_calls": 0,
        "clean": {
            name: matrix[name]["CLEAN"]["metrics"] for name in policies
        },
        "combined": {
            severity: {
                name: matrix[name][f"COMBINED_{severity}"]["metrics"] for name in policies
            }
            for severity in SEVERITIES
        },
        "winner_first_failure_by_corruption": first_failures,
        "gc2b_admission": admitted,
        "gc2b_admission_reasons": reasons,
        "next_authorized_route": (
            "GOAL-COPILOT-2B_NOISE_ROBUST_SKY_SEARCH_PROTOCOL_DESIGN"
            if admitted else "GOAL-COPILOT-3_RECORDED_RGB_EVIDENCE_PROTOCOL_DESIGN"
        ),
        "gc2b_model_calls_authorized": False,
        "gc1_model_search_authorized": False,
        "claim_ceiling": "consumed_dev_deterministic_perception_corruption_characterization_only",
        "resume_authorized": False,
    }
    write_once(output_root / "condition_matrix.json", matrix)
    write_once(output_root / "analysis.json", summary)
    replay = {
        "status": "PASS" if matrix == {
            name: {condition: evaluate_condition(path, SCENARIOS, condition) for condition in condition_names()}
            for name, path in policies.items()
        } else "FAIL",
        "model_calls": 0,
    }
    if replay["status"] != "PASS":
        raise RuntimeError("GC2-A deterministic replay failed")
    write_once(output_root / "replay_receipt.json", replay)
    closeout = {
        **summary,
        "status": "GOAL_COPILOT_2A_COMPLETE",
        "protocol_seal_digest": seal["protocol_seal_digest"],
        "source_commit": seal["source_commit"],
        "replay_status": "PASS",
    }
    write_once(output_root / "formal_closeout.json", closeout)
    return closeout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output_root.resolve()) if args.command == "freeze" else run(args.output_root.resolve())
    print(result if isinstance(result, Path) else json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
