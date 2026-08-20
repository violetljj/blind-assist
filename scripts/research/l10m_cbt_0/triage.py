"""Two-phase, zero-model-call candidate bottleneck analysis for consumed L10M runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, PolicySpec, all_specs, canonical_spec, parse_raw, parse_structured
from scripts.research.l10m_b6_0.triage import EPS, canonical_bytes, load_jsonl, score_context, sha256, write_create_once


PROTOCOL = "L10M-CBT-0-CANDIDATE-BOTTLENECK-TRIAGE-V1"
B6_MANIFEST_SHA256 = "f415f1f61c24fe3718350374e9180a56a41145f355de7035fe9030a50b615fbe"
TARGET_COHORTS = ("B4_A_HARDER_COHORT", "B5_A_FRESH_REPLICATION")


class ProtocolIntegrityError(RuntimeError):
    pass


def git_head(repo_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def spec_from_canonical(value: str) -> PolicySpec:
    spec = PolicySpec(**json.loads(value)); spec.validate(); return spec


def _event_key_matches(event: dict[str, Any], key: dict[str, Any]) -> bool:
    return all(event.get(name) == value for name, value in key.items())


def _canonical_candidate(event: dict[str, Any], arm: str) -> str | None:
    if event.get("admitted_canonical"):
        return str(event["admitted_canonical"])
    output = event.get("candidate_output")
    if not output:
        return None
    parser = parse_raw if arm == "raw" else parse_structured
    return canonical_spec(parser(output))


def candidate_sources(event: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    admitted = _canonical_candidate(event, arm)
    model = event.get("model_proposal_canonical")
    disposition = str(event.get("operator_disposition") or "")
    rows: list[dict[str, Any]] = []
    if model:
        legal = disposition not in {"COVERAGE_PROJECTION"}
        rows.append({"canonical": str(model), "source": "MODEL_PROPOSAL", "legally_selectable": legal, "exclusion_reason": None if legal else "FORBIDDEN_BY_FROZEN_OPERATOR"})
    if admitted and admitted != model:
        rows.append({"canonical": admitted, "source": "OPERATOR_MATERIALIZED", "legally_selectable": True, "exclusion_reason": None})
    if admitted and not rows:
        rows.append({"canonical": admitted, "source": "RECORDED_ADMITTED_CANDIDATE", "legally_selectable": True, "exclusion_reason": None})
    unique: dict[tuple[str, bool], dict[str, Any]] = {}
    for row in rows:
        unique[(row["canonical"], row["legally_selectable"])] = row
    return list(unique.values())


def inventory(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = repo_root.resolve()
    b6_path = repo_root / "scripts/research/l10m_b6_0/eligibility_manifest.json"
    if sha256(b6_path) != B6_MANIFEST_SHA256:
        raise ProtocolIntegrityError("B6 eligibility identity changed")
    b6 = json.loads(b6_path.read_text(encoding="utf-8"))
    run_paths = {row["run_id"]: row["path"] for row in b6["all_discovered_runs"]}
    trajectory_rows = b6["confirmatory_runs"] + b6["hypothesis_generating_runs"]
    decisions, excluded, source_hashes = [], [], {b6_path.relative_to(repo_root).as_posix(): sha256(b6_path)}
    for trajectory in trajectory_rows:
        run_dir = repo_root / run_paths[trajectory["run_id"]]
        events_path = run_dir / "events.jsonl"
        source_hashes[events_path.relative_to(repo_root).as_posix()] = sha256(events_path)
        events = sorted([row for row in load_jsonl(events_path) if row.get("kind") == "completion" and _event_key_matches(row, trajectory["key"])], key=lambda row: int(row["generation"]))
        if [int(row["generation"]) for row in events] != list(range(1, 9)):
            excluded.append({"trajectory_id": trajectory["trajectory_id"], "reason": "MISSING_STEP_RECEIPT"}); continue
        for event in events:
            try:
                sources = candidate_sources(event, trajectory["policy"])
                for source in sources: spec_from_canonical(source["canonical"])
            except Exception as error:
                excluded.append({"trajectory_id": trajectory["trajectory_id"], "generation": int(event["generation"]), "reason": "CANDIDATE_NOT_RECONSTRUCTIBLE", "detail": type(error).__name__}); continue
            legal = sorted({row["canonical"] for row in sources if row["legally_selectable"]})
            chosen = _canonical_candidate(event, trajectory["policy"])
            decisions.append({
                "decision_id": f"{trajectory['trajectory_id']}:generation={int(event['generation'])}",
                "trajectory_id": trajectory["trajectory_id"], "run_id": trajectory["run_id"],
                "cohort": trajectory["cohort"], "policy": trajectory["policy"], "key": trajectory["key"],
                "instance_id": trajectory["instance_id"], "benchmark_kind": trajectory["benchmark_kind"],
                "generation": int(event["generation"]), "candidate_sources": sources,
                "legal_nonincumbent_candidates": legal, "chosen_candidate": chosen,
                "selection_identifiable": len(legal) >= 2,
                "cross_arm_candidates_included": False, "future_candidates_included": False,
            })
    manifest = {
        "protocol": PROTOCOL, "repo_head": git_head(repo_root), "generated_before_metric_analysis": True,
        "source_b6_eligibility_sha256": B6_MANIFEST_SHA256,
        "eligible_trajectories": sorted({row["trajectory_id"] for row in decisions}),
        "eligible_decisions": decisions, "excluded": excluded,
        "structural_counts": {
            "trajectories": len({row["trajectory_id"] for row in decisions}), "decisions": len(decisions),
            "selection_identifiable_decisions": sum(row["selection_identifiable"] for row in decisions),
            "decisions_by_cohort": dict(Counter(row["cohort"] for row in decisions)),
            "decisions_by_policy": dict(Counter(row["policy"] for row in decisions)),
        },
        "candidate_set_definition": "same arm, same decision, already materialized, legal under frozen arm operator; never paired-arm or future candidates",
        "source_sha256": source_hashes,
    }
    audit = {"protocol": PROTOCOL, "excluded_reason_counts": dict(Counter(row["reason"] for row in excluded)), "candidate_cardinality": dict(Counter(len(row["legal_nonincumbent_candidates"]) for row in decisions))}
    return manifest, audit


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else statistics.mean(values)


def analyze(repo_root: Path, manifest_path: Path, digest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = digest_path.read_text(encoding="ascii").strip().split()[0]
    if sha256(manifest_path) != expected:
        raise ProtocolIntegrityError("sealed manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or not manifest.get("generated_before_metric_analysis"):
        raise ProtocolIntegrityError("invalid sealed manifest")
    source_ok = all(sha256(repo_root / path) == digest for path, digest in manifest["source_sha256"].items())
    if not source_ok:
        return {"protocol": PROTOCOL, "verdict": "CBT0_NOT_EVALUABLE_PROTOCOL_INTEGRITY", "protocol_integrity_gate": "FAIL", "experimental_model_calls": 0, "fresh_tasks_consumed": 0, "historical_runs_modified": 0}, []
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in manifest["eligible_decisions"]: by_trajectory[decision["trajectory_id"]].append(decision)
    records: list[dict[str, Any]] = []
    transmission: list[dict[str, Any]] = []
    for trajectory_id, decisions in sorted(by_trajectory.items()):
        decisions.sort(key=lambda row: row["generation"])
        contexts, evaluator = score_context(decisions[0]["benchmark_kind"]); instance = contexts[decisions[0]["instance_id"]]
        scores = {spec: float(evaluator(spec, instance)["behavioral_score"]) for spec in all_specs()}
        initial_score, optimum = scores[INITIAL_SPEC], max(scores.values()); gap = optimum - initial_score
        incumbent, incumbent_score = INITIAL_SPEC, initial_score
        trajectory_records = []
        for decision in decisions:
            legal_specs = [spec_from_canonical(value) for value in decision["legal_nonincumbent_candidates"]]
            observed_specs = sorted({spec_from_canonical(row["canonical"]) for row in decision["candidate_sources"]}, key=canonical_spec)
            chosen = None if decision["chosen_candidate"] is None else spec_from_canonical(decision["chosen_candidate"])
            best_legal_score = max([incumbent_score] + [scores[spec] for spec in legal_specs])
            best_observed_score = max([incumbent_score] + [scores[spec] for spec in observed_specs])
            chosen_score = None if chosen is None else scores[chosen]
            identifiable = len(set(legal_specs)) >= 2 and chosen is not None
            best_nonincumbent_score = None if not legal_specs else max(scores[spec] for spec in legal_specs)
            selection_regret = None if not identifiable else max(0.0, float(best_nonincumbent_score) - float(chosen_score))
            strict = chosen_score is not None and chosen_score > incumbent_score + EPS
            retained_before = incumbent
            if strict:
                incumbent, incumbent_score = chosen, float(chosen_score)
            retention_regret = 0.0 if chosen_score is None else max(0.0, float(chosen_score) - incumbent_score)
            record = {
                **{key: decision[key] for key in ("decision_id", "trajectory_id", "cohort", "policy", "generation")},
                "legal_nonincumbent_candidate_count": len(set(legal_specs)), "observed_candidate_count": len(set(observed_specs)),
                "incumbent_score": scores[retained_before], "best_legal_available_score": best_legal_score,
                "best_observed_score": best_observed_score, "chosen_candidate_score": chosen_score,
                "retained_successor_score": incumbent_score, "availability_gap": best_legal_score - scores[retained_before],
                "normalized_availability_gap": (best_legal_score - scores[retained_before]) / gap,
                "substantive_candidate_available": (best_legal_score - scores[retained_before]) / gap >= .10 - EPS,
                "global_optimum_candidate_available": any(abs(scores[spec] - optimum) <= EPS for spec in legal_specs),
                "selection_identifiable": identifiable, "selection_regret": selection_regret,
                "normalized_selection_regret": None if selection_regret is None else selection_regret / gap,
                "strictly_improving_candidate_retained": strict, "retention_regret": retention_regret,
                "retained_candidate_became_next_parent": strict and decision["generation"] < 8,
            }
            records.append(record); trajectory_records.append(record)
        for index, row in enumerate(trajectory_records):
            if not row["retained_candidate_became_next_parent"]: continue
            later = trajectory_records[index + 1:]
            transmission.append({"decision_id": row["decision_id"], "cohort": row["cohort"], "policy": row["policy"], "descendant_generated": bool(later), "productive_descendant": any(item["strictly_improving_candidate_retained"] for item in later)})
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        trajectories = sorted({row["trajectory_id"] for row in group})
        return {
            "decisions": len(group), "trajectories": len(trajectories),
            "positive_availability_rate": sum(row["availability_gap"] > EPS for row in group) / len(group),
            "substantive_availability_rate": sum(row["substantive_candidate_available"] for row in group) / len(group),
            "global_optimum_candidate_decision_rate": sum(row["global_optimum_candidate_available"] for row in group) / len(group),
            "global_optimum_candidate_trajectory_rate": sum(any(row["trajectory_id"] == trajectory and row["global_optimum_candidate_available"] for row in group) for trajectory in trajectories) / len(trajectories),
            "selection_identifiable_decisions": sum(row["selection_identifiable"] for row in group),
            "mean_selection_regret": _mean(group, "selection_regret"),
            "strict_retention_count": sum(row["strictly_improving_candidate_retained"] for row in group),
            "positive_retention_regret_rate": sum(row["retention_regret"] > EPS for row in group) / len(group),
        }
    cohort_metrics = {name: summarize([row for row in records if row["cohort"] == name]) for name in sorted({row["cohort"] for row in records})}
    policy_metrics = {name: summarize([row for row in records if row["policy"] == name]) for name in sorted({row["policy"] for row in records})}
    identifiable = [row for row in records if row["selection_identifiable"]]
    ident_cohorts, ident_policies = {row["cohort"] for row in identifiable}, {row["policy"] for row in identifiable}
    ident_gate = len(identifiable) >= 20 and len(ident_cohorts) >= 2 and len(ident_policies) >= 2
    positive_regrets = [row["normalized_selection_regret"] for row in identifiable if row["selection_regret"] is not None and row["selection_regret"] > EPS]
    selection_failure = bool(ident_gate and len(positive_regrets) / len(identifiable) >= .25 and statistics.median(positive_regrets) >= .10)
    retention_failure = sum(row["retention_regret"] > EPS for row in records) / len(records) >= .10
    target_ceiling = all(cohort_metrics[name]["global_optimum_candidate_trajectory_rate"] <= .10 for name in TARGET_COHORTS)
    transmission_metrics = {
        "retained_parents_with_future_generation": len(transmission),
        "descendant_generation_rate": 0.0 if not transmission else sum(row["descendant_generated"] for row in transmission) / len(transmission),
        "productive_descendant_rate": 0.0 if not transmission else sum(row["productive_descendant"] for row in transmission) / len(transmission),
    }
    if not ident_gate: verdict = "SELECTION_BOTTLENECK_NOT_IDENTIFIABLE_FROM_HISTORICAL_LOGS"
    elif selection_failure: verdict = "GOOD_CANDIDATES_EXIST_BUT_SELECTION_FAILS"
    elif retention_failure: verdict = "RETENTION_OR_TRANSMISSION_BOTTLENECK"
    elif target_ceiling: verdict = "CANDIDATE_AVAILABILITY_CEILING"
    else: verdict = "NO_DOMINANT_CANDIDATE_BOTTLENECK_IDENTIFIED"
    return {
        "protocol": PROTOCOL, "verdict": verdict, "protocol_integrity_gate": "PASS",
        "experimental_model_calls": 0, "fresh_tasks_consumed": 0, "historical_runs_modified": 0,
        "selection_identifiability_gate": "PASS" if ident_gate else "FAIL",
        "selection_identifiable_decisions": len(identifiable), "selection_identifiable_cohorts": len(ident_cohorts), "selection_identifiable_policies": len(ident_policies),
        "selection_failure_gate": "PASS" if selection_failure else "FAIL_OR_NOT_EVALUABLE",
        "retention_failure_gate": "PASS" if retention_failure else "FAIL",
        "hard_cohort_candidate_availability_ceiling": target_ceiling,
        "completion_availability_finding": "HARD_COHORT_CANDIDATE_AVAILABILITY_CEILING_OBSERVED" if target_ceiling else "HARD_COHORT_GLOBAL_CANDIDATES_OBSERVED",
        "cohort_metrics": cohort_metrics, "policy_metrics": policy_metrics, "transmission_metrics": transmission_metrics,
        "bottleneck_map": {
            "generation_operator_representation": "SUPPORTED_AS_DEVELOPMENT_ROUTING_DIAGNOSTIC" if target_ceiling else "NOT_ESTABLISHED",
            "ranking_credit_assignment": "NOT_IDENTIFIABLE" if not ident_gate else "SUPPORTED" if selection_failure else "NOT_SUPPORTED",
            "retention_population_mechanics": "SUPPORTED" if retention_failure else "NOT_SUPPORTED",
            "parent_descendant_transmission": "SECONDARY_DESCRIPTIVE_ONLY",
        },
        "next_route": "GENERATION_OPERATOR_REPRESENTATION_DIAGNOSTIC_ONLY" if target_ceiling else "NO_NEW_ROUTE_IDENTIFIED",
        "ranking_development_authorized": False, "generation_model_experiment_authorized": False,
        "fresh_cohort_authorized": False, "operator_admission_authorized": False,
        "l10m_exploration_policy_route": "CLOSED", "b6a_authorized": False, "rps_authorized": False, "balanced_v2_authorized": False,
        "claim_ceiling": "consumed historical development routing only; selection non-identifiability is not selection success, and candidate availability is not a causal generation-quality result",
    }, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory"); inv.add_argument("--repo-root", type=Path, required=True); inv.add_argument("--output-dir", type=Path, required=True); inv.add_argument("--tracked-manifest", type=Path, required=True)
    ana = sub.add_parser("analyze"); ana.add_argument("--repo-root", type=Path, required=True); ana.add_argument("--manifest", type=Path, required=True); ana.add_argument("--digest", type=Path, required=True); ana.add_argument("--output-dir", type=Path, required=True); ana.add_argument("--tracked-result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        manifest, audit = inventory(args.repo_root); payload = canonical_bytes(manifest); digest = hashlib.sha256(payload).hexdigest()
        for path, content in ((args.output_dir/"eligibility_manifest.json", payload), (args.output_dir/"eligibility_manifest.sha256", (digest+"  eligibility_manifest.json\n").encode("ascii")), (args.output_dir/"candidate_set_audit.json", canonical_bytes(audit)), (args.tracked_manifest, payload), (args.tracked_manifest.with_suffix(".sha256"), (digest+"  eligibility_manifest.json\n").encode("ascii"))): write_create_once(path, content)
        print(json.dumps({"phase": "CANDIDATE_SET_ELIGIBILITY", "manifest_sha256": digest, **manifest["structural_counts"], "excluded": len(manifest["excluded"])}))
    else:
        result, records = analyze(args.repo_root.resolve(), args.manifest.resolve(), args.digest.resolve()); payload = canonical_bytes(result)
        for path, content in ((args.output_dir/"decision_records.jsonl", b"".join(canonical_bytes(row) for row in records)), (args.output_dir/"cohort_metrics.json", canonical_bytes(result.get("cohort_metrics", {}))), (args.output_dir/"policy_metrics.json", canonical_bytes(result.get("policy_metrics", {}))), (args.output_dir/"transmission_metrics.json", canonical_bytes(result.get("transmission_metrics", {}))), (args.output_dir/"result.json", payload), (args.tracked_result, payload)): write_create_once(path, content)
        print(json.dumps({"phase": "SEALED_BOTTLENECK_ANALYSIS", "verdict": result["verdict"], "result_sha256": hashlib.sha256(payload).hexdigest(), "experimental_model_calls": 0}))


if __name__ == "__main__": main()
