"""BlindAssist-only dev selection, winner lock, and optional sealed fresh adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from evaluator import DEV_SCENARIOS, evaluate_payload, evaluate_scenarios
from fresh_crypto import unseal

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "initial_policy.py"
FRESH_ENVELOPE = HERE / "fresh_scenarios.enc.json"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"create-once receipt differs: {path}")
        return
    path.write_bytes(data)


def _selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
    metrics = item["assessment"]["metrics"]
    return (
        -metrics["minimum_family_completion_rate"],
        -metrics["completion_count"],
        -metrics["reacquisition_success"],
        -metrics["normalized_progress_total"],
        metrics["timeouts"],
        metrics["actions_on_completed_scenarios"],
        metrics["candidate_complexity_ast_nodes"],
        metrics["candidate_digest"],
    )


def _family_improvements(baseline: dict[str, Any], winner: dict[str, Any]) -> dict[str, int]:
    base = baseline["metrics"]["family_completion_counts"]
    win = winner["metrics"]["family_completion_counts"]
    return {family: win[family] - base[family] for family in sorted(base)}


def _paired_regressions(baseline: dict[str, Any], winner: dict[str, Any]) -> list[str]:
    base = {item["scenario_id"]: item for item in baseline["outcomes"]}
    win = {item["scenario_id"]: item for item in winner["outcomes"]}
    return [key for key in sorted(base) if base[key]["goal_completion"] and not win[key]["goal_completion"]]


def adjudicate(
    replicate_dirs: list[Path], output_root: Path, fresh_key_hex: str | None = None
) -> dict[str, Any]:
    baseline_dev = evaluate_scenarios(BASELINE, DEV_SCENARIOS)
    candidates: dict[str, dict[str, Any]] = {}
    replicate_summaries: list[dict[str, Any]] = []
    for replicate_dir in replicate_dirs:
        manifest = json.loads((replicate_dir / "candidate_manifest.json").read_text())
        local: list[dict[str, Any]] = []
        for entry in manifest["candidates"]:
            path = replicate_dir / entry["solution_file"]
            assessment = evaluate_scenarios(path, DEV_SCENARIOS)
            digest = assessment["metrics"]["candidate_digest"]
            if digest != entry["candidate_digest"]:
                raise RuntimeError(f"candidate digest mismatch: {path}")
            item = {
                "candidate_digest": digest,
                "candidate_bundle_digest": entry["candidate_bundle_digest"],
                "source_replicate": manifest["replicate_id"],
                "source_file": str(path.resolve()),
                "assessment": assessment,
            }
            local.append(item)
            candidates.setdefault(digest, item)
        eligible_local = [item for item in local if item["assessment"]["metrics"]["hard_gate_pass"]]
        best_local = sorted(eligible_local, key=_selection_key)[0] if eligible_local else None
        replicate_summaries.append({
            "replicate_id": manifest["replicate_id"],
            "generation_attempts": manifest["generation_attempts"],
            "candidate_count": len(local),
            "unique_candidate_count": len({item["candidate_digest"] for item in local}),
            "best_candidate_digest": best_local["candidate_digest"] if best_local else None,
            "best_completion_count": best_local["assessment"]["metrics"]["completion_count"] if best_local else None,
            "baseline_beating_valid_candidate": any(
                item["assessment"]["metrics"]["hard_gate_pass"]
                and item["assessment"]["metrics"]["completion_count"] > baseline_dev["metrics"]["completion_count"]
                for item in local
            ),
        })

    eligible = [item for item in candidates.values() if item["assessment"]["metrics"]["hard_gate_pass"]]
    winner = sorted(eligible, key=_selection_key)[0] if eligible else None
    dev_improvements = (
        _family_improvements(baseline_dev, winner["assessment"]) if winner else {}
    )
    fresh_authorized = bool(
        winner
        and winner["assessment"]["metrics"]["completion_count"] > baseline_dev["metrics"]["completion_count"]
        and sum(value > 0 for value in dev_improvements.values()) >= 2
    )
    dev_analysis = {
        "baseline": baseline_dev,
        "replicates": replicate_summaries,
        "candidate_count": sum(item["candidate_count"] for item in replicate_summaries),
        "unique_candidate_count": len(candidates),
        "hard_gate_valid_candidate_count": len(eligible),
        "hard_gate_failure_count": len(candidates) - len(eligible),
        "winner_dev": winner["assessment"] if winner else None,
        "dev_family_improvements": dev_improvements,
        "fresh_execution_authorized": fresh_authorized,
    }
    _write_once(output_root / "dev_analysis.json", dev_analysis)

    selection = {
        "status": "LOCK_WINNER" if winner else "NO_ELIGIBLE_WINNER",
        "candidate_digest": winner["candidate_digest"] if winner else None,
        "candidate_bundle_digest": winner["candidate_bundle_digest"] if winner else None,
        "selection_rule": [
            "minimum_family_completion_rate_desc", "completion_count_desc",
            "reacquisition_success_desc", "normalized_progress_total_desc",
            "timeouts_asc", "actions_on_completed_scenarios_asc",
            "candidate_complexity_ast_nodes_asc", "candidate_digest_lexical_asc",
        ],
        "selection_input_digests": sorted(candidates),
    }
    _write_once(output_root / "winner_selection.json", selection)
    if winner:
        winner_dir = output_root / "locked_winner" / winner["candidate_digest"]
        winner_dir.mkdir(parents=True, exist_ok=True)
        destination = winner_dir / "policy.py"
        source = Path(winner["source_file"])
        if not destination.exists():
            shutil.copy2(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != winner["candidate_digest"]:
            raise RuntimeError("locked winner copy mismatch")

    if not fresh_authorized:
        fresh_analysis = {
            "status": "NOT_EXECUTED",
            "baseline": None,
            "winner": None,
            "completion_delta": None,
            "family_improvements": None,
            "regressions": None,
            "pass": False,
        }
        verdict = "GOAL_COPILOT_1_SKY_SEARCH_VALUE_NOT_ESTABLISHED_ON_DEVELOPMENT"
    else:
        if not fresh_key_hex:
            raise RuntimeError("fresh admission passed but the in-memory unseal key is unavailable")
        fresh_payload = unseal(json.loads(FRESH_ENVELOPE.read_text()), fresh_key_hex)
        baseline_fresh = evaluate_payload(BASELINE, fresh_payload)
        winner_fresh = evaluate_payload(Path(winner["source_file"]), fresh_payload)
        improvements = _family_improvements(baseline_fresh, winner_fresh)
        regressions = _paired_regressions(baseline_fresh, winner_fresh)
        wm = winner_fresh["metrics"]
        bm = baseline_fresh["metrics"]
        passed = bool(
            wm["hard_gate_pass"]
            and wm["unsafe_guidance"] == 0
            and wm["premature_completion"] == 0
            and wm["completion_count"] >= bm["completion_count"] + 2
            and sum(value > 0 for value in improvements.values()) >= 2
            and not regressions
            and min(wm["family_completion_counts"].values()) >= 1
        )
        fresh_analysis = {
            "status": "EXECUTED",
            "baseline": baseline_fresh,
            "winner": winner_fresh,
            "completion_delta": wm["completion_count"] - bm["completion_count"],
            "family_improvements": improvements,
            "regressions": regressions,
            "pass": passed,
        }
        if wm["unsafe_guidance"] > 0:
            verdict = "GOAL_COPILOT_1_SKY_SEARCH_VALUE_NOT_ESTABLISHED_SAFETY_GATE_FAILED"
        elif wm["premature_completion"] > 0:
            verdict = "GOAL_COPILOT_1_SKY_SEARCH_VALUE_NOT_ESTABLISHED_COMPLETION_INTEGRITY_FAILED"
        elif passed:
            verdict = "GOAL_COPILOT_1_SKY_SEARCH_SIGNAL_ESTABLISHED_ON_SEALED_PILOT"
        else:
            verdict = "GOAL_COPILOT_1_SKY_SEARCH_VALUE_NOT_ESTABLISHED_ON_SEALED_PILOT"
    _write_once(output_root / "fresh_analysis.json", fresh_analysis)
    result = {
        "dev_analysis": dev_analysis,
        "winner_selection": selection,
        "fresh_analysis": fresh_analysis,
        "final_verdict": verdict,
        "resume_authorized": False,
        "claim_ceiling": "small_deterministic_symbolic_closed_loop_pilot_no_real_vision_or_superiority_claim",
    }
    _write_once(output_root / "adjudication_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fresh-key-hex")
    args = parser.parse_args()
    print(json.dumps(adjudicate(args.replicate, args.output_root, args.fresh_key_hex), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
