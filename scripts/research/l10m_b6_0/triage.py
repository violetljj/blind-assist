"""Two-phase, zero-model-call L10M B6-0 reachability triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable

from scripts.research.l10m_b1.evaluator import evaluate_spec
from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, PolicySpec, all_specs, canonical_spec, parse_raw, parse_structured
from scripts.research.l10m_b4.fresh_benchmark import evaluate_fresh_instance, load_fresh_benchmark
from scripts.research.l10m_b4.hard_benchmark import evaluate_instance, legal_neighbors, load_benchmark


PROTOCOL = "L10M-B6-0-REACHABILITY-HYPOTHESIS-TRIAGE-V1"
EPS = 1e-12
SUPPORTED = {
    "L10M-B1-STRUCTURED-SEARCHABILITY-MATCHED-V2-FRESH-SUCCESSOR": ("B1_V2_FRESH_SUCCESSOR", "CONFIRMATORY_SET", "hidden"),
    "L10M-B3-A-BALANCED-EXPLORATION-CAUSAL-TEST-V1": ("B3_A_BALANCED_EXPLORATION", "CONFIRMATORY_SET", "hidden"),
    "L10M-B4-A-HARDER-COHORT-PAIRED-SEARCH-V2-ABSOLUTE-WORKDIR": ("B4_A_HARDER_COHORT", "CONFIRMATORY_SET", "hard"),
    "L10M-B5-A-FRESH-GENERALIZATION-REPLICATION-V1": ("B5_A_FRESH_REPLICATION", "HYPOTHESIS_GENERATING_ONLY", "fresh"),
}
EXPECTED_TERMINALS = {
    "B1_V2_FRESH_SUCCESSOR": "B1_EVALUABLE_COMPLETE",
    "B3_A_BALANCED_EXPLORATION": "B3A_EVALUABLE_COMPLETE",
    "B4_A_HARDER_COHORT": "B4A_EVALUABLE_COMPLETE",
    "B5_A_FRESH_REPLICATION": "B5A_EVALUABLE_COMPLETE",
}


class ProtocolIntegrityError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_create_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def git_head(repo_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_context(kind: str) -> tuple[dict[str, dict[str, Any]], Callable[[PolicySpec, dict[str, Any]], dict[str, object]]]:
    if kind == "hidden":
        return {"hidden": {}}, lambda spec, _: evaluate_spec(spec)
    if kind == "hard":
        payload = load_benchmark()
        return {row["instance_id"]: row for row in payload["instances"]}, evaluate_instance
    if kind == "fresh":
        payload = load_fresh_benchmark()
        return {row["instance_id"]: row for row in payload["instances"]}, evaluate_fresh_instance
    raise ValueError(kind)


def strict_distance(start: PolicySpec, scores: dict[PolicySpec, float], optimum: float) -> int | None:
    queue: deque[tuple[PolicySpec, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        state, distance = queue.popleft()
        if abs(scores[state] - optimum) <= EPS:
            return distance
        for neighbor in legal_neighbors(state):
            if neighbor not in seen and scores[neighbor] > scores[state] + EPS:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def reachability(distance: int | None) -> float:
    return 0.0 if distance is None else 1.0 / (1.0 + distance)


def trajectory_specs(manifest: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    protocol_id = str(manifest.get("protocol_id"))
    if protocol_id.startswith("L10M-B1-"):
        seeds = manifest.get("seeds", sorted({row.get("seed") for row in events if row.get("seed") is not None}))
        return [{"key": {"seed": int(seed), "arm": arm}, "instance_id": "hidden"} for seed in seeds for arm in ("raw", "structured")]
    if protocol_id == "L10M-B3-A-BALANCED-EXPLORATION-CAUSAL-TEST-V1":
        return [{"key": {"seed": int(seed), "arm": arm}, "instance_id": "hidden"} for seed in manifest["seeds"] for arm in manifest["arms"]]
    if protocol_id in SUPPORTED:
        return [{"key": {"instance_id": row["instance_id"], "paired_identity": int(row["paired_identity"]), "arm": arm}, "instance_id": row["instance_id"]} for row in manifest["paired_identities"] for arm in manifest["arms"]]
    return []


def matches(event: dict[str, Any], key: dict[str, Any]) -> bool:
    return all(event.get(name) == value for name, value in key.items())


def reconstruct(spec: dict[str, Any], events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    contexts, evaluator = score_context(kind)
    instance = contexts[spec["instance_id"]]
    scores = {state: float(evaluator(state, instance)["behavioral_score"]) for state in all_specs()}
    optimum = max(scores.values())
    initial = scores[INITIAL_SPEC]
    gap = optimum - initial
    if gap <= EPS:
        raise ValueError("OUTCOME_NOT_RECONSTRUCTIBLE")
    rows = sorted([row for row in events if row.get("kind") == "completion" and matches(row, spec["key"])], key=lambda row: int(row["generation"]))
    if [int(row["generation"]) for row in rows] != list(range(1, 9)):
        raise ValueError("MISSING_STEP_RECEIPT")
    incumbent, incumbent_score = INITIAL_SPEC, initial
    steps: list[dict[str, Any]] = []
    parser = parse_raw if spec["key"]["arm"] == "raw" else parse_structured
    for row in rows:
        if row.get("returncode") != 0 or row.get("transport_runtime_failure"):
            raise ValueError("CONFLICTING_RECEIPTS")
        candidate = parser(row["candidate_output"])
        score = scores[candidate]
        if row.get("behavioral_score") is None or abs(float(row["behavioral_score"]) - score) > EPS:
            raise ValueError("NON_DETERMINISTIC_REPLAY")
        strict = score > incumbent_score + EPS
        if "strict_improvement" in row and bool(row["strict_improvement"]) != strict:
            raise ValueError("CONFLICTING_RECEIPTS")
        before = incumbent
        before_score = incumbent_score
        if strict:
            incumbent, incumbent_score = candidate, score
        steps.append({
            "generation": int(row["generation"]), "candidate": canonical_spec(candidate),
            "candidate_score": score, "incumbent_before": canonical_spec(before),
            "incumbent_after": canonical_spec(incumbent), "score_before": before_score,
            "score_after": incumbent_score, "strict_improvement": strict,
            "reachability_before": reachability(strict_distance(before, scores, optimum)),
            "reachability_after": reachability(strict_distance(incumbent, scores, optimum)),
        })
    primary = next((row for row in steps[:-1] if len(legal_neighbors(parse_structured(_render_structured_json(row["incumbent_before"])))) >= 2), None)
    if primary is None:
        raise ValueError("NO_EVALUABLE_BRANCH_DECISION")
    terminal_progress = (incumbent_score - initial) / gap
    after_progress = (primary["score_after"] - initial) / gap
    return {
        "primary_generation": primary["generation"], "immediate_score": primary["candidate_score"],
        "reachability_before": primary["reachability_before"], "reachability_after": primary["reachability_after"],
        "reachability_delta": primary["reachability_after"] - primary["reachability_before"],
        "normalized_progress_after": after_progress, "terminal_best_normalized_progress": terminal_progress,
        "future_progress_gain": terminal_progress - after_progress,
    }


def _render_structured_json(canonical: str) -> str:
    raw = json.loads(canonical)
    return json.dumps({
        "progress_contract": {"mode": "POSITIVE_PROGRESS|CONFIRMED_NO_PROGRESS|UNKNOWN_PROGRESS", "mutable": False},
        "stuck_response": {"on_confirmed_stuck": raw["stuck_response"]},
        "recovery_transition": {"while_active": raw["recovery_transition_action"]},
        "action_selection": {"turn_threshold": raw["action_selection_turn_threshold"]},
        "fallback": {"min_quality": raw["fallback_min_quality"], "action": raw["fallback_action"]},
    })


def discover_execution_runs(repo_root: Path) -> list[Path]:
    base = repo_root / "artifacts.local" / "evidence"
    return sorted(path.parent for path in base.glob("l10m_*/**/runs/*/execution_manifest.json"))


def inventory(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = repo_root.resolve()
    protocol_path = repo_root / "scripts/research/l10m_b6_0/protocol.json"
    discovered, eligible, hypothesis, excluded = [], [], [], []
    audit_rows = []
    source_hashes: dict[str, str] = {}
    for run_dir in discover_execution_runs(repo_root):
        relative = run_dir.relative_to(repo_root).as_posix()
        manifest_path, events_path = run_dir / "execution_manifest.json", run_dir / "events.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol_id = str(manifest.get("protocol_id"))
        run_id = str(manifest.get("run_id", run_dir.name))
        discovered.append({"run_id": run_id, "path": relative, "protocol_id": protocol_id})
        source_hashes[manifest_path.relative_to(repo_root).as_posix()] = sha256(manifest_path)
        if events_path.exists():
            source_hashes[events_path.relative_to(repo_root).as_posix()] = sha256(events_path)
        if protocol_id not in SUPPORTED:
            excluded.append({"run_id": run_id, "path": relative, "reason": "TRANSITION_NOT_RECONSTRUCTIBLE"})
            continue
        cohort, authority, kind = SUPPORTED[protocol_id]
        result_path = run_dir / "result.json"
        if not result_path.exists():
            excluded.append({"run_id": run_id, "path": relative, "reason": "OUTCOME_NOT_RECONSTRUCTIBLE"})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        source_hashes[result_path.relative_to(repo_root).as_posix()] = sha256(result_path)
        if result.get("terminal") != EXPECTED_TERMINALS[cohort]:
            excluded.append({"run_id": run_id, "path": relative, "reason": "CONFLICTING_RECEIPTS"})
            continue
        events = load_jsonl(events_path)
        for item in trajectory_specs(manifest, events):
            key = item["key"]
            trajectory_id = cohort + ":" + ":".join(f"{name}={key[name]}" for name in sorted(key))
            try:
                first = reconstruct(item, events, kind)
                second = reconstruct(item, events, kind)
                if first != second:
                    raise ValueError("NON_DETERMINISTIC_REPLAY")
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                excluded.append({"trajectory_id": trajectory_id, "run_id": run_id, "reason": "TRANSITION_NOT_RECONSTRUCTIBLE", "detail": type(error).__name__})
                continue
            except ValueError as error:
                reason = str(error) if str(error).isupper() else "TRANSITION_NOT_RECONSTRUCTIBLE"
                excluded.append({"trajectory_id": trajectory_id, "run_id": run_id, "reason": reason})
                continue
            row = {"trajectory_id": trajectory_id, "run_id": run_id, "cohort": cohort, "policy": key["arm"], "authority": authority, "key": key, "instance_id": item["instance_id"], "benchmark_kind": kind}
            (hypothesis if authority == "HYPOTHESIS_GENERATING_ONLY" else eligible).append(row)
            audit_rows.append({"trajectory_id": trajectory_id, "deterministic_replay": "PASS", "primary_decision_reconstructible": True})
    manifest = {
        "protocol": PROTOCOL, "repo_head": git_head(repo_root), "generated_before_outcome_analysis": True,
        "hypothesis_seed_authority": "EXPLORATORY_HYPOTHESIS_SEED_ONLY",
        "hypothesis_generating_runs": hypothesis, "confirmatory_runs": eligible, "excluded_runs": excluded,
        "all_discovered_runs": discovered,
        "cohorts": dict(Counter(row["cohort"] for row in eligible + hypothesis)),
        "policies": dict(Counter(row["policy"] for row in eligible + hypothesis)),
        "reconstruction_definition": {"complete_generations": list(range(1, 9)), "selection": "semantic-valid safe strict score improvement", "deterministic_replay_required": True},
        "immediate_score_definition": "recorded candidate behavioral_score verified against exhaustive deterministic evaluator",
        "reachability_definition": "B5-C legal-neighbor strictly-increasing shortest path to any global optimum; rank encoding unreachable=0 else 1/(1+steps)",
        "outcome_definition": "terminal_best_normalized_progress-normalized_progress_after_primary_decision",
        "source_sha256": source_hashes,
    }
    audit = {"protocol": PROTOCOL, "rows": audit_rows, "exclusion_reason_counts": dict(Counter(row["reason"] for row in excluded))}
    return manifest, audit


def average_ranks(values: Iterable[float]) -> list[float]:
    values = list(values); order = sorted(range(len(values)), key=values.__getitem__); ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]: ranks[index] = rank
        start = end
    return ranks


def residual(values: list[float], control: list[float]) -> list[float]:
    mean_x, mean_y = statistics.mean(control), statistics.mean(values)
    denom = sum((x - mean_x) ** 2 for x in control)
    slope = 0.0 if denom <= EPS else sum((x - mean_x) * (y - mean_y) for x, y in zip(control, values, strict=True)) / denom
    return [y - (mean_y + slope * (x - mean_x)) for x, y in zip(control, values, strict=True)]


def pearson(left: list[float], right: list[float]) -> float | None:
    ml, mr = statistics.mean(left), statistics.mean(right)
    dl, dr = [x - ml for x in left], [x - mr for x in right]
    denom = math.sqrt(sum(x*x for x in dl) * sum(x*x for x in dr))
    return None if denom <= EPS else sum(x*y for x, y in zip(dl, dr, strict=True)) / denom


def partial(rows: list[dict[str, Any]]) -> float | None:
    i = average_ranks(row["I"] for row in rows); r = average_ranks(row["R"] for row in rows); y = average_ranks(row["Y"] for row in rows)
    return pearson(residual(r, i), residual(y, i))


def solve_ols(x: list[list[float]], y: list[float]) -> list[float]:
    p = len(x[0]); a = [[sum(row[i]*row[j] for row in x) for j in range(p)] + [sum(row[i]*target for row, target in zip(x, y, strict=True))] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda row: abs(a[row][col])); a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) <= EPS: a[col][col] += 1e-9
        scale = a[col][col]; a[col] = [v/scale for v in a[col]]
        for row in range(p):
            if row == col: continue
            factor = a[row][col]; a[row] = [v-factor*w for v, w in zip(a[row], a[col], strict=True)]
    return [a[i][-1] for i in range(p)]


def cohort_rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for cohort in sorted({row["cohort"] for row in rows}):
        group = [row for row in rows if row["cohort"] == cohort]
        ranks = {name: average_ranks(row[name] for row in group) for name in ("I", "R", "Y")}
        for index, row in enumerate(group): output.append({**row, **{name: ranks[name][index] for name in ranks}})
    return output


def loco(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policies = sorted({row["policy"] for row in rows}); ranked = cohort_rank_rows(rows); folds = []
    for held in sorted({row["cohort"] for row in rows}):
        train, test = [r for r in ranked if r["cohort"] != held], [r for r in ranked if r["cohort"] == held]
        def vector(row: dict[str, Any], augmented: bool) -> list[float]:
            base = [1.0, row["I"]] + [1.0 if row["policy"] == policy else 0.0 for policy in policies[1:]]
            return base + ([row["R"]] if augmented else [])
        beta0, beta1 = solve_ols([vector(r, False) for r in train], [r["Y"] for r in train]), solve_ols([vector(r, True) for r in train], [r["Y"] for r in train])
        errors0 = [abs(sum(a*b for a,b in zip(vector(r,False),beta0,strict=True))-r["Y"]) for r in test]
        errors1 = [abs(sum(a*b for a,b in zip(vector(r,True),beta1,strict=True))-r["Y"]) for r in test]
        mae0, mae1 = statistics.mean(errors0), statistics.mean(errors1)
        folds.append({"held_out_cohort": held, "n": len(test), "baseline_MAE": mae0, "augmented_MAE": mae1, "MAE_improvement": mae0-mae1, "relative_MAE_improvement": 0.0 if mae0 <= EPS else (mae0-mae1)/mae0, "baseline_absolute_error_sum": sum(errors0), "augmented_absolute_error_sum": sum(errors1)})
    base_sum, aug_sum = sum(f["baseline_absolute_error_sum"] for f in folds), sum(f["augmented_absolute_error_sum"] for f in folds)
    return {"folds": folds, "baseline_heldout_MAE": base_sum/len(rows), "augmented_heldout_MAE": aug_sum/len(rows), "relative_MAE_improvement": 0.0 if base_sum <= EPS else (base_sum-aug_sum)/base_sum, "median_per_cohort_MAE_improvement": statistics.median(f["MAE_improvement"] for f in folds)}


def analyze(repo_root: Path, manifest_path: Path, digest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = digest_path.read_text(encoding="ascii").strip().split()[0]
    if sha256(manifest_path) != expected: raise ProtocolIntegrityError("sealed manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or not manifest.get("generated_before_outcome_analysis"): raise ProtocolIntegrityError("invalid sealed manifest")
    source_ok = all(sha256(repo_root / path) == digest for path, digest in manifest["source_sha256"].items())
    records = []
    for row in manifest["confirmatory_runs"] + manifest["hypothesis_generating_runs"]:
        run_dir = next(repo_root / item["path"] for item in manifest["all_discovered_runs"] if item["run_id"] == row["run_id"])
        events = load_jsonl(run_dir / "events.jsonl")
        rebuilt = reconstruct({"key": row["key"], "instance_id": row["instance_id"]}, events, row["benchmark_kind"])
        records.append({**row, "I": rebuilt["immediate_score"], "R": rebuilt["reachability_after"], "Y": rebuilt["future_progress_gain"], **rebuilt})
    confirmatory = [row for row in records if row["authority"] == "CONFIRMATORY_SET"]
    cohorts = {}
    for name in sorted({row["cohort"] for row in confirmatory}):
        group = [row for row in confirmatory if row["cohort"] == name]
        cohorts[name] = {"n": len(group), "partial_spearman_R_Y_given_I": partial(group) if len(group) >= 6 else None}
    loco_result = loco(confirmatory) if len(cohorts) >= 2 else {"folds": [], "relative_MAE_improvement": None, "median_per_cohort_MAE_improvement": None}
    for fold in loco_result["folds"]:
        cohorts[fold["held_out_cohort"]].update({"baseline_MAE": fold["baseline_MAE"], "augmented_MAE": fold["augmented_MAE"], "MAE_improvement": fold["MAE_improvement"]})
    policy_metrics = {}
    for policy in sorted({row["policy"] for row in confirmatory}):
        group = [row for row in confirmatory if row["policy"] == policy]; cohort_count = len({row["cohort"] for row in group})
        ranked = cohort_rank_rows(group)
        policy_metrics[policy] = {"n": len(group), "cohort_count": cohort_count, "evaluable": len(group) >= 6 and cohort_count >= 2, "partial_association": partial(ranked) if len(group) >= 6 and cohort_count >= 2 else None}
    evaluable_cohorts = [row for row in cohorts.values() if row["n"] >= 6]
    evaluable_policies = [row for row in policy_metrics.values() if row["evaluable"]]
    minimum = len(evaluable_cohorts) >= 2 and len(evaluable_policies) >= 2
    incremental = loco_result.get("relative_MAE_improvement") is not None and loco_result["relative_MAE_improvement"] >= .02 and loco_result["median_per_cohort_MAE_improvement"] > 0
    rhos = [row["partial_spearman_R_Y_given_I"] for row in evaluable_cohorts]
    cohort_gate = bool(minimum and all(value is not None for value in rhos) and sum(value > 0 for value in rhos)/len(rhos) >= .75 and statistics.median(rhos) >= .20 and all(value > -.20 for value in rhos))
    policy_gate = bool(len(evaluable_policies) >= 2 and all(row["partial_association"] is not None and row["partial_association"] > 0 for row in evaluable_policies))
    integrity = source_ok
    passed = minimum and incremental and cohort_gate and policy_gate and integrity
    reasons = []
    if not minimum: reasons.append("INSUFFICIENT_INDEPENDENT_HISTORICAL_EVIDENCE")
    if not incremental: reasons.append("NO_INCREMENTAL_INFORMATION_BEYOND_IMMEDIATE_SCORE")
    if not cohort_gate: reasons.append("CROSS_COHORT_INCONSISTENCY")
    if any(value is not None and value <= -.20 for value in rhos): reasons.append("STRONG_COHORT_REVERSAL")
    if not policy_gate: reasons.append("CROSS_POLICY_INCONSISTENCY")
    if not integrity: reasons.append("PROTOCOL_INTEGRITY_FAILURE")
    verdict = "REACHABILITY_HYPOTHESIS_SUPPORTED_FOR_DEVELOPMENT" if passed else "REACHABILITY_HYPOTHESIS_NOT_SUPPORTED"
    return {
        "protocol": PROTOCOL, "verdict": verdict, "reasons": reasons,
        "experimental_model_calls": 0, "fresh_tasks_consumed": 0, "historical_runs_modified": 0,
        "hypothesis_seed_authority": "EXPLORATORY_HYPOTHESIS_SEED_ONLY",
        "eligibility": {"all_discovered_execution_runs": len(manifest["all_discovered_runs"]), "confirmatory_trajectories": len(manifest["confirmatory_runs"]), "hypothesis_generating_trajectories": len(manifest["hypothesis_generating_runs"]), "excluded": len(manifest["excluded_runs"])},
        "minimum_evidence_gate": "PASS" if minimum else "FAIL", "incremental_information_gate": "PASS" if incremental else "FAIL",
        "cross_cohort_consistency_gate": "PASS" if cohort_gate else "FAIL", "cross_policy_consistency_gate": "PASS" if policy_gate else "FAIL", "protocol_integrity_gate": "PASS" if integrity else "FAIL",
        "cohort_metrics": cohorts, "policy_metrics": policy_metrics, "loco_metrics": loco_result,
        "pooled_result_authority": "SECONDARY_DESCRIPTIVE_ONLY", "cohort_reversal_occurred": any(value is not None and value <= -.20 for value in rhos),
        "l10m_exploration_policy_route": "OPEN_FOR_B6_A_DEVELOPMENT_ONLY" if passed else "CLOSED",
        "b6a_authorized": passed, "rps_development_authorized": passed, "fresh_cohort_authorized": False,
        "operator_admission_authorized": False, "balanced_v2_authorized": False, "conditional_balanced_authorized": False,
    }, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory"); inv.add_argument("--repo-root", type=Path, required=True); inv.add_argument("--output-dir", type=Path, required=True); inv.add_argument("--tracked-manifest", type=Path, required=True)
    ana = sub.add_parser("analyze"); ana.add_argument("--repo-root", type=Path, required=True); ana.add_argument("--manifest", type=Path, required=True); ana.add_argument("--digest", type=Path, required=True); ana.add_argument("--output-dir", type=Path, required=True); ana.add_argument("--tracked-result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        manifest, audit = inventory(args.repo_root); payload = canonical_bytes(manifest); digest = hashlib.sha256(payload).hexdigest()
        write_create_once(args.output_dir / "eligibility_manifest.json", payload); write_create_once(args.output_dir / "eligibility_manifest.sha256", (digest + "  eligibility_manifest.json\n").encode("ascii")); write_create_once(args.output_dir / "reconstruction_audit.json", canonical_bytes(audit)); write_create_once(args.tracked_manifest, payload); write_create_once(args.tracked_manifest.with_suffix(".sha256"), (digest + "  eligibility_manifest.json\n").encode("ascii"))
        print(json.dumps({"phase": "INVENTORY_ELIGIBILITY", "manifest_sha256": digest, "confirmatory": len(manifest["confirmatory_runs"]), "hypothesis_generating": len(manifest["hypothesis_generating_runs"]), "excluded": len(manifest["excluded_runs"])}))
    else:
        result, records = analyze(args.repo_root.resolve(), args.manifest.resolve(), args.digest.resolve()); payload = canonical_bytes(result)
        write_create_once(args.output_dir / "primary_decision_records.jsonl", b"".join(canonical_bytes(row) for row in records)); write_create_once(args.output_dir / "cohort_metrics.json", canonical_bytes(result["cohort_metrics"])); write_create_once(args.output_dir / "policy_metrics.json", canonical_bytes(result["policy_metrics"])); write_create_once(args.output_dir / "loco_metrics.json", canonical_bytes(result["loco_metrics"])); write_create_once(args.output_dir / "secondary_descriptive.json", canonical_bytes({"authority": "SECONDARY_DESCRIPTIVE_ONLY", "analyses": []})); write_create_once(args.output_dir / "result.json", payload); write_create_once(args.tracked_result, payload)
        print(json.dumps({"phase": "SEALED_ANALYSIS", "verdict": result["verdict"], "result_sha256": hashlib.sha256(payload).hexdigest(), "experimental_model_calls": 0}))


if __name__ == "__main__":
    main()
