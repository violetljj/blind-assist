"""Zero-call exhaustive GOR-0 ceiling decomposition for L10M."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.research.l10m_b1.policy_space import INITIAL_SPEC, PolicySpec, all_specs, canonical_spec
from scripts.research.l10m_b4.hard_benchmark import legal_neighbors
from scripts.research.l10m_b6_0.triage import EPS, canonical_bytes, score_context, sha256, write_create_once
from scripts.research.l10m_cbt_0.triage import spec_from_canonical


PROTOCOL = "L10M-GOR-0-GENERATION-OPERATOR-REPRESENTATION-CEILING-DECOMPOSITION-V1"
CBT_MANIFEST_SHA256 = "ddb1f825263e0a1cc02f996b5691b7ef4dbeeaeda3c05789285fde5cbb8c5b00"
TARGET_COHORTS = ("B4_A_HARDER_COHORT", "B5_A_FRESH_REPLICATION")
FIELDS = tuple(asdict(INITIAL_SPEC))


class ProtocolIntegrityError(RuntimeError):
    pass


def git_head(repo_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def changed_fields(before: PolicySpec, after: PolicySpec) -> list[str]:
    left, right = asdict(before), asdict(after)
    return [field for field in FIELDS if left[field] != right[field]]


def reverse_distances(scores: dict[PolicySpec, float], *, strict: bool) -> dict[PolicySpec, int]:
    optimum = max(scores.values())
    roots = [state for state, score in scores.items() if abs(score-optimum) <= EPS]
    distance = {state: 0 for state in roots}; queue: deque[PolicySpec] = deque(roots)
    while queue:
        target = queue.popleft()
        for predecessor in legal_neighbors(target):
            if predecessor in distance: continue
            if strict and scores[target] <= scores[predecessor] + EPS: continue
            distance[predecessor] = distance[target] + 1; queue.append(predecessor)
    return distance


def landscape_key(kind: str, instance_id: str) -> str:
    return f"{kind}:{instance_id}"


def build_landscape(kind: str, instance_id: str) -> tuple[dict[str, Any], dict[PolicySpec, float], dict[PolicySpec, int], dict[PolicySpec, int]]:
    contexts, evaluator = score_context(kind); instance = contexts[instance_id]
    scores = {state: float(evaluator(state, instance)["behavioral_score"]) for state in all_specs()}
    operator_distance = reverse_distances(scores, strict=False); strict_distance = reverse_distances(scores, strict=True)
    optimum = max(scores.values())
    score_payload = {canonical_spec(state): scores[state] for state in sorted(scores, key=canonical_spec)}
    summary = {
        "landscape": landscape_key(kind, instance_id), "benchmark_kind": kind, "instance_id": instance_id,
        "representation_state_count": len(scores), "completion_state_count": sum(abs(score-optimum) <= EPS for score in scores.values()),
        "global_optimum_score": optimum, "score_landscape_sha256": hashlib.sha256(canonical_bytes(score_payload)).hexdigest(),
        "operator_reachable_state_count": len(operator_distance), "strict_reachable_state_count": len(strict_distance),
        "initial_operator_distance": operator_distance.get(INITIAL_SPEC), "initial_strict_admissible_distance": strict_distance.get(INITIAL_SPEC),
    }
    return summary, scores, operator_distance, strict_distance


def source_hashes(repo_root: Path, cbt: dict[str, Any]) -> dict[str, str]:
    paths = [
        "scripts/research/l10m_cbt_0/eligibility_manifest.json", "scripts/research/l10m_b1/policy_space.py",
        "scripts/research/l10m_b1/evaluator.py", "scripts/research/l10m_b1/hidden_cohort_v1.json",
        "scripts/research/l10m_b3a/exploration.py", "scripts/research/l10m_b4/hard_benchmark.py",
        "scripts/research/l10m_b4/hard_benchmark_v1.json", "scripts/research/l10m_b4/fresh_benchmark.py",
        "scripts/research/l10m_b4/fresh_benchmark_v1.json",
    ]
    paths.extend(cbt["source_sha256"])
    return {path: sha256(repo_root/path) for path in sorted(set(paths))}


def build_oracle_manifest(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(); cbt_path = repo_root/"scripts/research/l10m_cbt_0/eligibility_manifest.json"
    if sha256(cbt_path) != CBT_MANIFEST_SHA256: raise ProtocolIntegrityError("CBT candidate-set manifest changed")
    cbt = json.loads(cbt_path.read_text(encoding="utf-8"))
    landscapes = sorted({(row["benchmark_kind"], row["instance_id"]) for row in cbt["eligible_decisions"]})
    summaries = {}
    for kind, instance_id in landscapes:
        summary, _, _, _ = build_landscape(kind, instance_id); summaries[summary["landscape"]] = summary
    return {
        "protocol": PROTOCOL, "repo_head": git_head(repo_root), "generated_before_trajectory_analysis": True,
        "source_cbt_manifest_sha256": CBT_MANIFEST_SHA256,
        "representation_definition": "exhaustive 162-state frozen PolicySpec product",
        "operator_definition": "legal one-field adjacent numeric or categorical-destination moves",
        "strict_acceptance_definition": "each retained edge has frozen score strictly greater than parent",
        "landscapes": summaries,
        "eligible_trajectory_count": len(cbt["eligible_trajectories"]), "eligible_decision_count": len(cbt["eligible_decisions"]),
        "source_sha256": source_hashes(repo_root, cbt),
    }


def classify(*, representation_supported: bool, operator_supported: bool, target_trajectories: int, target_completion_candidates: int) -> str:
    if not representation_supported: return "REPRESENTATION_EXPRESSIBILITY_CEILING"
    if not operator_supported: return "OPERATOR_SUPPORT_CEILING"
    if target_trajectories > 0 and target_completion_candidates == 0: return "GENERATION_COVERAGE_CEILING"
    return "CEILING_SOURCE_NOT_IDENTIFIABLE"


def oracle_fields(state: PolicySpec, scores: dict[PolicySpec, float], distances: dict[PolicySpec, int]) -> list[str]:
    current = distances.get(state)
    if current is None or current == 0: return []
    fields = []
    for neighbor in legal_neighbors(state):
        if scores[neighbor] > scores[state] + EPS and distances.get(neighbor) == current-1:
            fields.extend(changed_fields(state, neighbor))
    return sorted(set(fields))


def analyze(repo_root: Path, manifest_path: Path, digest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = digest_path.read_text(encoding="ascii").strip().split()[0]
    if sha256(manifest_path) != expected: raise ProtocolIntegrityError("sealed oracle manifest digest mismatch")
    oracle = json.loads(manifest_path.read_text(encoding="utf-8"))
    if oracle.get("protocol") != PROTOCOL or not oracle.get("generated_before_trajectory_analysis"): raise ProtocolIntegrityError("invalid oracle manifest")
    if not all(sha256(repo_root/path) == digest for path, digest in oracle["source_sha256"].items()):
        return {"protocol": PROTOCOL, "verdict": "CEILING_SOURCE_NOT_IDENTIFIABLE", "protocol_integrity_gate": "FAIL", "experimental_model_calls": 0, "fresh_tasks_consumed": 0}, []
    cbt = json.loads((repo_root/"scripts/research/l10m_cbt_0/eligibility_manifest.json").read_text(encoding="utf-8"))
    decisions_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cbt["eligible_decisions"]: decisions_by_trajectory[row["trajectory_id"]].append(row)
    cache = {}
    for summary in oracle["landscapes"].values():
        built, scores, operator_distance, strict_distance = build_landscape(summary["benchmark_kind"], summary["instance_id"])
        if built != summary: raise ProtocolIntegrityError("oracle landscape reconstruction mismatch")
        cache[summary["landscape"]] = (scores, operator_distance, strict_distance)
    records, trajectories = [], []
    easy_success_fields: Counter[str] = Counter(); hard_required_fields: Counter[str] = Counter(); hard_missing_fields: Counter[str] = Counter()
    for trajectory_id, decisions in sorted(decisions_by_trajectory.items()):
        decisions.sort(key=lambda row: row["generation"]); first = decisions[0]
        key = landscape_key(first["benchmark_kind"], first["instance_id"]); scores, operator_distance, strict_distance = cache[key]
        incumbent = INITIAL_SPEC; incumbent_score = scores[incumbent]; retained_history = []
        initial_distance = strict_distance.get(incumbent); retained_sequence = [initial_distance]
        reachability_losses = 0; completion_candidate = False
        for decision in decisions:
            candidate = None if decision["chosen_candidate"] is None else spec_from_canonical(decision["chosen_candidate"])
            before, before_score = incumbent, incumbent_score; before_distance = strict_distance.get(before)
            candidate_score = None if candidate is None else scores[candidate]; candidate_distance = None if candidate is None else strict_distance.get(candidate)
            fields = [] if candidate is None else changed_fields(before, candidate)
            required = oracle_fields(before, scores, strict_distance)
            strict = candidate_score is not None and candidate_score > before_score + EPS
            directed = bool(strict and before_distance is not None and candidate_distance is not None and candidate_distance < before_distance)
            exact = bool(directed and candidate in legal_neighbors(before) and candidate_distance == before_distance-1)
            if candidate_distance == 0: completion_candidate = True
            if strict:
                incumbent, incumbent_score = candidate, float(candidate_score); retained_history.append((decision["generation"], fields, strict_distance.get(incumbent)))
            after_distance = strict_distance.get(incumbent); retained_sequence.append(after_distance)
            if before_distance is not None and after_distance is None: reachability_losses += 1
            if decision["cohort"] in TARGET_COHORTS:
                hard_required_fields.update(required)
                if required and not set(required).intersection(fields): hard_missing_fields.update(required)
            record = {
                "decision_id": decision["decision_id"], "trajectory_id": trajectory_id, "cohort": decision["cohort"], "policy": decision["policy"], "generation": decision["generation"],
                "landscape": key, "parent_strict_distance": before_distance, "candidate_strict_distance": candidate_distance,
                "retained_strict_distance": after_distance, "candidate_operator_distance": None if candidate is None else operator_distance.get(candidate),
                "candidate_changed_fields": fields, "candidate_edit_arity": len(fields), "oracle_next_edit_fields": required,
                "strictly_retained": strict, "completion_directed": directed, "exact_oracle_edge": exact,
                "completion_candidate": candidate_distance == 0, "strict_reachability_lost": before_distance is not None and after_distance is None,
            }
            records.append(record)
        if completion_candidate and first["cohort"] in {"B1_V2_FRESH_SUCCESSOR", "B3_A_BALANCED_EXPLORATION"}:
            first_zero = next((index for index, (_, _, distance) in enumerate(retained_history) if distance == 0), None)
            if first_zero is not None:
                for _, fields, _ in retained_history[max(0, first_zero-2):first_zero+1]: easy_success_fields.update(fields)
        candidate_distances = [row["candidate_strict_distance"] for row in records if row["trajectory_id"] == trajectory_id and row["candidate_strict_distance"] is not None]
        trajectories.append({
            "trajectory_id": trajectory_id, "cohort": first["cohort"], "policy": first["policy"], "landscape": key,
            "initial_strict_distance": initial_distance, "retained_distance_sequence": retained_sequence,
            "minimum_generated_candidate_strict_distance": None if not candidate_distances else min(candidate_distances),
            "completion_candidate_generated": completion_candidate, "terminal_strict_distance": strict_distance.get(incumbent),
            "reachability_loss_count": reachability_losses,
        })
    cohort_metrics = {}
    for cohort in sorted({row["cohort"] for row in trajectories}):
        tr = [row for row in trajectories if row["cohort"] == cohort]; dr = [row for row in records if row["cohort"] == cohort]
        direction_opportunities = [row for row in dr if row["parent_strict_distance"] not in (None, 0)]
        cohort_metrics[cohort] = {
            "trajectories": len(tr), "completion_candidate_trajectories": sum(row["completion_candidate_generated"] for row in tr),
            "completion_candidate_trajectory_rate": sum(row["completion_candidate_generated"] for row in tr)/len(tr),
            "initial_strict_distances": sorted({row["initial_strict_distance"] for row in tr}),
            "minimum_generated_candidate_distance_distribution": dict(Counter(str(row["minimum_generated_candidate_strict_distance"]) for row in tr)),
            "terminal_distance_distribution": dict(Counter(str(row["terminal_strict_distance"]) for row in tr)),
            "completion_directed_candidates": sum(row["completion_directed"] for row in dr),
            "exact_oracle_edges": sum(row["exact_oracle_edge"] for row in dr),
            "oracle_direction_opportunities": len(direction_opportunities),
            "completion_directed_rate": 0.0 if not direction_opportunities else sum(row["completion_directed"] for row in direction_opportunities)/len(direction_opportunities),
            "reachability_loss_trajectories": sum(row["reachability_loss_count"] > 0 for row in tr),
            "retained_distance_sequences": {row["trajectory_id"]: row["retained_distance_sequence"] for row in tr},
        }
    target_landscapes = [summary for summary in oracle["landscapes"].values() if summary["benchmark_kind"] in {"hard", "fresh"}]
    representation_supported = all(summary["completion_state_count"] > 0 for summary in target_landscapes)
    grammar_supported = all(summary["initial_operator_distance"] is not None for summary in target_landscapes)
    strict_supported = all(summary["initial_strict_admissible_distance"] is not None for summary in target_landscapes)
    operator_supported = grammar_supported and strict_supported
    target_trajectories = [row for row in trajectories if row["cohort"] in TARGET_COHORTS]
    target_completion = sum(row["completion_candidate_generated"] for row in target_trajectories)
    verdict = classify(representation_supported=representation_supported, operator_supported=operator_supported, target_trajectories=len(target_trajectories), target_completion_candidates=target_completion)
    subreason = None if operator_supported else "OPERATOR_GRAMMAR_GAP" if not grammar_supported else "STRICT_ACCEPTANCE_BARRIER"
    taxonomy = {
        "easy_success_last_three_edit_fields": dict(easy_success_fields),
        "hard_oracle_required_fields": dict(hard_required_fields),
        "hard_oracle_fields_missed_by_generated_edit": dict(hard_missing_fields),
        "coupled_multi_field_required_by_oracle": False if strict_supported else None,
        "interpretation": "field occurrence counts over deterministic oracle opportunities; not independent samples or causal effects",
    }
    return {
        "protocol": PROTOCOL, "verdict": verdict, "verdict_subreason": subreason, "protocol_integrity_gate": "PASS",
        "experimental_model_calls": 0, "fresh_tasks_consumed": 0, "historical_runs_modified": 0,
        "representation_gate": "PASS" if representation_supported else "FAIL", "operator_grammar_gate": "PASS" if grammar_supported else "FAIL",
        "strict_acceptance_path_gate": "PASS" if strict_supported else "FAIL", "generation_coverage_gate": "FAIL" if verdict == "GENERATION_COVERAGE_CEILING" else "NOT_APPLICABLE_OR_PASS",
        "target_trajectories": len(target_trajectories), "target_completion_candidate_trajectories": target_completion,
        "landscape_metrics": oracle["landscapes"], "cohort_metrics": cohort_metrics, "structural_taxonomy": taxonomy,
        "next_route": "GENERATION_MECHANISM_DEVELOPMENT_QUESTION_ONLY" if verdict == "GENERATION_COVERAGE_CEILING" else "REPRESENTATION_SEARCH_SPACE_QUESTION_ONLY" if verdict == "REPRESENTATION_EXPRESSIBILITY_CEILING" else "OPERATOR_GRAMMAR_OR_ACCEPTANCE_QUESTION_ONLY" if verdict == "OPERATOR_SUPPORT_CEILING" else "OBSERVABILITY_FIRST",
        "generation_experiment_authorized": False, "new_operator_authorized": False, "fresh_cohort_authorized": False, "operator_admission_authorized": False,
        "exploration_policy_route": "CLOSED", "ranking_route": "NOT_IDENTIFIABLE", "retention_bottleneck": "NOT_SUPPORTED",
        "claim_ceiling": "frozen historical generation setup coverage only; not intrinsic model incapacity, causal mechanism, generalization, admission, device, user, or safety evidence",
    }, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("oracle"); inv.add_argument("--repo-root", type=Path, required=True); inv.add_argument("--output-dir", type=Path, required=True); inv.add_argument("--tracked-manifest", type=Path, required=True)
    ana = sub.add_parser("analyze"); ana.add_argument("--repo-root", type=Path, required=True); ana.add_argument("--manifest", type=Path, required=True); ana.add_argument("--digest", type=Path, required=True); ana.add_argument("--output-dir", type=Path, required=True); ana.add_argument("--tracked-result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "oracle":
        manifest = build_oracle_manifest(args.repo_root); payload = canonical_bytes(manifest); digest = hashlib.sha256(payload).hexdigest()
        for path, content in ((args.output_dir/"oracle_manifest.json", payload), (args.output_dir/"oracle_manifest.sha256", (digest+"  oracle_manifest.json\n").encode("ascii")), (args.tracked_manifest, payload), (args.tracked_manifest.with_suffix(".sha256"), (digest+"  oracle_manifest.json\n").encode("ascii"))): write_create_once(path, content)
        print(json.dumps({"phase": "GRAPH_ORACLE_MANIFEST", "manifest_sha256": digest, "landscapes": len(manifest["landscapes"]), "states_per_landscape": 162, "trajectory_analysis_run": False}))
    else:
        result, records = analyze(args.repo_root.resolve(), args.manifest.resolve(), args.digest.resolve()); payload = canonical_bytes(result)
        for path, content in ((args.output_dir/"decision_distance_records.jsonl", b"".join(canonical_bytes(row) for row in records)), (args.output_dir/"cohort_distance_metrics.json", canonical_bytes(result.get("cohort_metrics", {}))), (args.output_dir/"structural_taxonomy.json", canonical_bytes(result.get("structural_taxonomy", {}))), (args.output_dir/"result.json", payload), (args.tracked_result, payload)): write_create_once(path, content)
        print(json.dumps({"phase": "SEALED_TRAJECTORY_DECOMPOSITION", "verdict": result["verdict"], "result_sha256": hashlib.sha256(payload).hexdigest(), "experimental_model_calls": 0}))


if __name__ == "__main__": main()
