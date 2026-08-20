"""Zero-model-call, read-only B4-A/B5-A Balanced effect heterogeneity autopsy."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

from scripts.research.l10m_b1.policy_space import (
    INITIAL_SPEC,
    PolicySpec,
    all_specs,
    canonical_spec,
    parse_structured,
)

from .fresh_benchmark import evaluate_fresh_instance, load_fresh_benchmark
from .hard_benchmark import evaluate_instance, legal_neighbors, load_benchmark


ANALYSIS_ID = "L10M-B5-C-BALANCED-EFFECT-HETEROGENEITY-AUTOPSY-V1"
EPSILON = 1e-12
SUBSTANTIVE_NORMALIZED_GAIN = 0.10

INPUTS = {
    "b4": {
        "stage": "B4-A",
        "result": "artifacts.local/evidence/l10m_b4/b4a_v2/runs/b4av2-20260820T133016-815ed378/result.json",
        "result_sha256": "50102673579283c1ab4552c3827eb98d297e0e5b19c22dfdf28042b2280a1370",
        "events": "artifacts.local/evidence/l10m_b4/b4a_v2/runs/b4av2-20260820T133016-815ed378/events.jsonl",
        "events_sha256": "6f1d4b7b40a7e9c763c7d072e75b42379c970199b3c8434b800ae2db972a3103",
        "manifest": "artifacts.local/evidence/l10m_b4/b4a_v2/runs/b4av2-20260820T133016-815ed378/execution_manifest.json",
        "manifest_sha256": "5696a8fd86f74872a6a1384e0d72e647a31bbefad45c99bf1fc266a9d592dbba",
        "certificate": "artifacts.local/evidence/l10m_b4/hard_benchmark_v1/certificate.json",
        "certificate_sha256": "7f2cf3a1fb4db8534e5af3839c264dc377be48db63538d7c85c023aabf3c2696",
        "expected_terminal": "B4A_EVALUABLE_COMPLETE",
        "benchmark_loader": load_benchmark,
        "evaluator": evaluate_instance,
    },
    "b5": {
        "stage": "B5-A",
        "result": "artifacts.local/evidence/l10m_b5/b5a/runs/b5a-20260820T142630-82025cc9/result.json",
        "result_sha256": "97f54954aeb6ae22a15420b94e0ce20f88d44e7d3115bb638469ca4bf1d69e9c",
        "events": "artifacts.local/evidence/l10m_b5/b5a/runs/b5a-20260820T142630-82025cc9/events.jsonl",
        "events_sha256": "dbc5bf284aa27ce75ede836dc9fea142473014adc8021795a35b7ed630bb995a",
        "manifest": "artifacts.local/evidence/l10m_b5/b5a/runs/b5a-20260820T142630-82025cc9/execution_manifest.json",
        "manifest_sha256": "84466ac0f8c6cb78ed84ca0ae418396630c4281ead946a79d40c8897f0f81078",
        "certificate": "artifacts.local/evidence/l10m_b5/fresh_harder_v1/certificate.json",
        "certificate_sha256": "22be26089adaaa7d3302ea7f965b7373d642ac26033430046e89b8d828a9b446",
        "expected_terminal": "B5A_EVALUABLE_COMPLETE",
        "benchmark_loader": load_fresh_benchmark,
        "evaluator": evaluate_fresh_instance,
    },
}

DOMAIN_FEATURES = (
    "fine_turn",
    "wide_forward",
    "turn_pressure_delta",
    "lost_total",
    "quality_total",
    "recovery_total",
    "directional_demand_abs_diff",
    "initial_score",
    "global_score",
    "initial_to_global_gap",
    "global_optimum_count",
    "local_maximum_count",
    "shortest_strict_steps_to_global_optimum",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _score_landscape(
    instance: dict[str, Any], evaluator: Callable[[PolicySpec, dict[str, Any]], dict[str, object]]
) -> dict[PolicySpec, float]:
    return {
        spec: float(evaluator(spec, instance)["behavioral_score"])
        for spec in all_specs()
    }


def shortest_strict_steps(
    start: PolicySpec, scores: dict[PolicySpec, float], global_score: float
) -> int | None:
    queue: deque[tuple[PolicySpec, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        current, distance = queue.popleft()
        if abs(scores[current] - global_score) <= EPSILON:
            return distance
        for neighbor in legal_neighbors(current):
            if neighbor in seen or scores[neighbor] <= scores[current] + EPSILON:
                continue
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def _trailing_false(values: list[bool]) -> int:
    count = 0
    for value in reversed(values):
        if value:
            break
        count += 1
    return count


def _trajectory(
    rows: list[dict[str, Any]],
    instance: dict[str, Any],
    evaluator: Callable[[PolicySpec, dict[str, Any]], dict[str, object]],
    global_score: float,
) -> dict[str, Any]:
    completions = sorted(
        [row for row in rows if row.get("kind") == "completion"],
        key=lambda row: int(row["generation"]),
    )
    if [int(row["generation"]) for row in completions] != list(range(1, 9)):
        raise RuntimeError("trajectory is not an exact eight-generation completion")
    scores = _score_landscape(instance, evaluator)
    incumbent = INITIAL_SPEC
    incumbent_score = scores[incumbent]
    initial_score = incumbent_score
    gap = global_score - initial_score
    seen_candidates: set[str] = set()
    step_rows: list[dict[str, Any]] = []
    strict_flags: list[bool] = []
    coverage_count = coverage_no_improvement = 0
    first_strict = first_substantive = None
    for event in completions:
        if event.get("returncode") != 0 or event.get("transport_runtime_failure"):
            raise RuntimeError("runtime failure found in evaluable sealed trajectory")
        candidate = parse_structured(event["candidate_output"])
        candidate_key = canonical_spec(candidate)
        candidate_score = scores[candidate]
        if abs(candidate_score - float(event["behavioral_score"])) > EPSILON:
            raise RuntimeError("event score differs from exhaustive landscape")
        before_score = incumbent_score
        strict = bool(event["strict_improvement"])
        recomputed_strict = candidate_score > before_score + EPSILON
        if strict != recomputed_strict:
            raise RuntimeError("strict-improvement flag differs from recomputation")
        repeated_candidate = candidate_key in seen_candidates
        seen_candidates.add(candidate_key)
        disposition = str(event.get("operator_disposition") or "")
        if disposition == "COVERAGE_PROJECTION":
            coverage_count += 1
            coverage_no_improvement += int(not strict)
        if strict:
            incumbent = candidate
            incumbent_score = candidate_score
            if first_strict is None:
                first_strict = int(event["generation"])
            if first_substantive is None and (candidate_score - before_score) / gap >= SUBSTANTIVE_NORMALIZED_GAIN - EPSILON:
                first_substantive = int(event["generation"])
        strict_flags.append(strict)
        step_rows.append(
            {
                "generation": int(event["generation"]),
                "candidate_canonical": candidate_key,
                "candidate_score": candidate_score,
                "incumbent_score_before": before_score,
                "incumbent_score_after": incumbent_score,
                "strict_improvement": strict,
                "normalized_improvement_gain": (incumbent_score - before_score) / gap,
                "repeated_candidate": repeated_candidate,
                "operator_disposition": disposition,
                "operator_move_token": event.get("operator_move_token"),
            }
        )
    nonimproving_generations = [index for index, strict in enumerate(strict_flags) if not strict]
    recoverable_nonimprovements = sum(
        any(strict_flags[later] for later in range(index + 1, len(strict_flags)))
        for index in nonimproving_generations
    )
    recovery_rate = (
        None
        if not nonimproving_generations
        else recoverable_nonimprovements / len(nonimproving_generations)
    )
    terminal_distance = shortest_strict_steps(incumbent, scores, global_score)
    terminal_local_maximum = all(
        scores[neighbor] <= incumbent_score + EPSILON for neighbor in legal_neighbors(incumbent)
    )
    return {
        "initial_score": initial_score,
        "final_score": incumbent_score,
        "normalized_progress": (incumbent_score - initial_score) / gap,
        "normalized_optimum_residual": (global_score - incumbent_score) / gap,
        "first_strict_improvement_generation": first_strict,
        "first_substantive_improvement_generation": first_substantive,
        "strict_improvement_count": sum(strict_flags),
        "no_improvement_count": len(strict_flags) - sum(strict_flags),
        "best_so_far_improvement_count": sum(strict_flags),
        "unique_candidate_count": len(seen_candidates),
        "repeated_candidate_count": sum(row["repeated_candidate"] for row in step_rows),
        "repeated_incumbent_generation_count": len(strict_flags) - sum(strict_flags),
        "dead_end_tail_generation_count": _trailing_false(strict_flags),
        "post_no_improvement_productive_return_rate": recovery_rate,
        "coverage_projection_count": coverage_count,
        "coverage_projection_no_improvement_count": coverage_no_improvement,
        "terminal_strict_steps_to_global_optimum": terminal_distance,
        "terminal_local_maximum": terminal_local_maximum,
        "terminal_canonical": canonical_spec(incumbent),
        "steps": step_rows,
    }


def _landscape_features(instance: dict[str, Any], certificate: dict[str, Any]) -> dict[str, float | int]:
    motifs = instance["motifs"]
    left = sum(value for key, value in motifs.items() if key.endswith("_left"))
    right = sum(value for key, value in motifs.items() if key.endswith("_right"))
    return {
        "fine_turn": motifs["fine_turn"],
        "wide_forward": motifs["wide_forward"],
        "turn_pressure_delta": motifs["fine_turn"] - motifs["wide_forward"],
        "lost_total": motifs["lost_left"] + motifs["lost_right"],
        "quality_total": sum(value for key, value in motifs.items() if key.startswith("quality_")),
        "recovery_total": motifs["recovery_left"] + motifs["recovery_right"],
        "directional_demand_abs_diff": abs(left - right),
        "initial_score": float(certificate["initial_score"]),
        "global_score": float(certificate["global_score"]),
        "initial_to_global_gap": float(certificate["global_score"] - certificate["initial_score"]),
        "global_optimum_count": int(certificate["global_optimum_count"]),
        "local_maximum_count": int(certificate["local_maximum_count"]),
        "shortest_strict_steps_to_global_optimum": int(certificate["shortest_strict_steps_to_global_optimum"]),
    }


def _first_divergence(control: dict[str, Any], balanced: dict[str, Any]) -> dict[str, Any]:
    for control_step, balanced_step in zip(control["steps"], balanced["steps"], strict=True):
        if control_step["candidate_canonical"] == balanced_step["candidate_canonical"]:
            continue
        generation = int(control_step["generation"])
        window_end = min(8, generation + 1)
        control_window = max(
            step["incumbent_score_after"]
            for step in control["steps"]
            if generation <= step["generation"] <= window_end
        )
        balanced_window = max(
            step["incumbent_score_after"]
            for step in balanced["steps"]
            if generation <= step["generation"] <= window_end
        )
        return {
            "generation": generation,
            "control_candidate_score": control_step["candidate_score"],
            "balanced_candidate_score": balanced_step["candidate_score"],
            "balanced_candidate_advantage_raw": balanced_step["candidate_score"] - control_step["candidate_score"],
            "control_strict_improvement": control_step["strict_improvement"],
            "balanced_strict_improvement": balanced_step["strict_improvement"],
            "balanced_operator_disposition": balanced_step["operator_disposition"],
            "two_generation_best_score_advantage_raw": balanced_window - control_window,
        }
    return {"generation": None}


def _domain_hypotheses(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    for feature in DOMAIN_FEATURES:
        values = sorted({float(row["features"][feature]) for row in instance_rows})
        thresholds = [(left + right) / 2 for left, right in zip(values, values[1:])]
        for threshold in thresholds:
            low = [row for row in instance_rows if float(row["features"][feature]) <= threshold]
            high = [row for row in instance_rows if float(row["features"][feature]) > threshold]
            if len(low) < 2 or len(high) < 2:
                continue
            if {row["stage"] for row in low} != {"B4-A", "B5-A"}:
                continue
            if {row["stage"] for row in high} != {"B4-A", "B5-A"}:
                continue
            for favorable_name, favorable, adverse_name, adverse in (
                ("low", low, "high", high),
                ("high", high, "low", low),
            ):
                if all(row["median_paired_delta"] > 0.05 for row in favorable) and all(
                    row["median_paired_delta"] <= 0.0 for row in adverse
                ):
                    admitted.append(
                        {
                            "feature": feature,
                            "threshold": threshold,
                            "favorable_side": favorable_name,
                            "adverse_side": adverse_name,
                            "favorable_instances": [row["instance_id"] for row in favorable],
                            "adverse_instances": [row["instance_id"] for row in adverse],
                            "status": "RETROSPECTIVE_CONDITIONAL_DOMAIN_HYPOTHESIS_ONLY",
                        }
                    )
    return admitted


def _signature_summary(pairs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    losses = [row for row in pairs if row["disposition"] == "balanced_loss"]
    nonlosses = [row for row in pairs if row["disposition"] != "balanced_loss"]
    loss_hits = [row for row in losses if row["mechanism_signatures"][key]]
    nonloss_hits = [row for row in nonlosses if row["mechanism_signatures"][key]]
    loss_rate = 0.0 if not losses else len(loss_hits) / len(losses)
    nonloss_rate = 0.0 if not nonlosses else len(nonloss_hits) / len(nonlosses)
    qualified = (
        len(loss_hits) >= 3
        and len({row["instance_id"] for row in loss_hits}) >= 2
        and loss_rate >= 0.75
        and nonloss_rate <= 0.5 * loss_rate
    )
    return {
        "signature": key,
        "loss_hits": len(loss_hits),
        "loss_total": len(losses),
        "loss_rate": loss_rate,
        "loss_instance_count": len({row["instance_id"] for row in loss_hits}),
        "nonloss_hits": len(nonloss_hits),
        "nonloss_total": len(nonlosses),
        "nonloss_rate": nonloss_rate,
        "qualified": qualified,
    }


def classify_terminal(
    domain_hypotheses: list[dict[str, Any]], mechanism_summaries: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    if domain_hypotheses:
        return "OBSERVABLE_CONDITIONAL_DOMAIN_HYPOTHESIS_IDENTIFIED", [
            row["feature"] for row in domain_hypotheses
        ]
    qualified = [row["signature"] for row in mechanism_summaries if row["qualified"]]
    if qualified:
        return "BALANCED_V2_MECHANISM_HYPOTHESIS_IDENTIFIED", qualified
    return "NO_REPRODUCIBLE_HETEROGENEITY_EXPLANATION_CLOSE_OPERATOR_ADMISSION_ROUTE", []


def analyze(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    stage_payloads: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    model_calls = 0
    for stage_key, config in INPUTS.items():
        for field in ("result", "events", "manifest", "certificate"):
            path = repo_root / str(config[field])
            observed = _sha256(path)
            expected = str(config[f"{field}_sha256"])
            if observed != expected:
                raise RuntimeError(f"{stage_key} {field} identity mismatch")
            source_hashes[str(config[field])] = observed
        result = json.loads((repo_root / str(config["result"])).read_text(encoding="utf-8"))
        if result.get("terminal") != config["expected_terminal"] or result.get("model_calls") != 144:
            raise RuntimeError(f"{stage_key} result is not the expected complete terminal")
        events = _load_jsonl(repo_root / str(config["events"]))
        completions = [row for row in events if row.get("kind") == "completion"]
        if len(completions) != 144:
            raise RuntimeError(f"{stage_key} completion count differs from 144")
        benchmark = config["benchmark_loader"]()
        certificate = json.loads((repo_root / str(config["certificate"])).read_text(encoding="utf-8"))
        stage_payloads[stage_key] = {
            "config": config,
            "result": result,
            "events": events,
            "instances": {row["instance_id"]: row for row in benchmark["instances"]},
            "certificates": {row["instance_id"]: row for row in certificate["instances"]},
        }

    trajectory_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    for stage_key, payload in stage_payloads.items():
        config = payload["config"]
        grouped_events: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for event in payload["events"]:
            if event.get("kind") == "completion":
                grouped_events[(str(event["instance_id"]), int(event["paired_identity"]), str(event["arm"]))].append(event)
        stage_trajectories: dict[tuple[str, int, str], dict[str, Any]] = {}
        for key, rows in grouped_events.items():
            instance_id, identity, arm = key
            trajectory = _trajectory(
                rows,
                payload["instances"][instance_id],
                config["evaluator"],
                float(payload["certificates"][instance_id]["global_score"]),
            )
            trajectory.update(
                {"stage": config["stage"], "instance_id": instance_id, "paired_identity": identity, "arm": arm}
            )
            stage_trajectories[key] = trajectory
            trajectory_rows.append(trajectory)
        instance_pair_deltas: dict[str, list[float]] = defaultdict(list)
        pair_keys = sorted({(instance_id, identity) for instance_id, identity, _ in stage_trajectories})
        for instance_id, identity in pair_keys:
            control = stage_trajectories[(instance_id, identity, "structured_control")]
            balanced = stage_trajectories[(instance_id, identity, "structured_balanced")]
            delta = balanced["normalized_progress"] - control["normalized_progress"]
            disposition = "balanced_win" if delta > EPSILON else "balanced_loss" if delta < -EPSILON else "tie"
            horizon_waste = (
                balanced["no_improvement_count"] >= control["no_improvement_count"] + 2
                and balanced["dead_end_tail_generation_count"] >= control["dead_end_tail_generation_count"]
                and delta < -EPSILON
            )
            balanced_rate = balanced["post_no_improvement_productive_return_rate"]
            control_rate = control["post_no_improvement_productive_return_rate"]
            coverage_nonproductive = (
                balanced["coverage_projection_count"] >= 6
                and balanced["coverage_projection_no_improvement_count"] >= 5
                and balanced_rate is not None
                and control_rate is not None
                and balanced_rate + 0.25 <= control_rate
            )
            pair = {
                "stage": config["stage"],
                "instance_id": instance_id,
                "paired_identity": identity,
                "disposition": disposition,
                "paired_normalized_progress_delta": delta,
                "first_divergence": _first_divergence(control, balanced),
                "control": {key: value for key, value in control.items() if key != "steps"},
                "balanced": {key: value for key, value in balanced.items() if key != "steps"},
                "mechanism_signatures": {
                    "finite_horizon_exploration_waste": horizon_waste,
                    "nonproductive_coverage_projection": coverage_nonproductive,
                },
            }
            pair_rows.append(pair)
            instance_pair_deltas[instance_id].append(delta)
        for instance_id, deltas in sorted(instance_pair_deltas.items()):
            instance_rows.append(
                {
                    "stage": config["stage"],
                    "instance_id": instance_id,
                    "median_paired_delta": statistics.median(deltas),
                    "mean_paired_delta": statistics.mean(deltas),
                    "wins": sum(value > EPSILON for value in deltas),
                    "ties": sum(abs(value) <= EPSILON for value in deltas),
                    "losses": sum(value < -EPSILON for value in deltas),
                    "features": _landscape_features(
                        payload["instances"][instance_id], payload["certificates"][instance_id]
                    ),
                }
            )

    domain_hypotheses = _domain_hypotheses(instance_rows)
    mechanism_summaries = [
        _signature_summary(pair_rows, "finite_horizon_exploration_waste"),
        _signature_summary(pair_rows, "nonproductive_coverage_projection"),
    ]
    terminal, terminal_basis = classify_terminal(domain_hypotheses, mechanism_summaries)
    stage_means: dict[str, dict[str, float]] = {}
    for stage in ("B4-A", "B5-A"):
        rows = [row for row in trajectory_rows if row["stage"] == stage]
        stage_means[stage] = {
            arm: statistics.mean(
                row["normalized_progress"] for row in rows if row["arm"] == arm
            )
            for arm in ("structured_control", "structured_balanced")
        }
        stage_means[stage]["balanced_minus_control"] = (
            stage_means[stage]["structured_balanced"] - stage_means[stage]["structured_control"]
        )
    balanced_shift = stage_means["B5-A"]["structured_balanced"] - stage_means["B4-A"]["structured_balanced"]
    control_shift = stage_means["B5-A"]["structured_control"] - stage_means["B4-A"]["structured_control"]
    return {
        "schema": "l10m_b5c_balanced_effect_heterogeneity_autopsy_v1",
        "analysis_id": ANALYSIS_ID,
        "mode": "CONSUMED_EVIDENCE_READ_ONLY_ZERO_MODEL_CALL_AUTOPSY",
        "model_calls": model_calls,
        "terminal": terminal,
        "terminal_basis": terminal_basis,
        "input_role": "B4-A and B5-A are consumed sealed evidence; no fresh or confirmation authority",
        "domain_analysis": {
            "tested_feature_allowlist": list(DOMAIN_FEATURES),
            "qualification_rule": "single-feature threshold; at least two instances per side; each side spans B4-A and B5-A; every favorable instance median delta > 0.05 and every adverse instance median delta <= 0",
            "qualified_hypotheses": domain_hypotheses,
            "claim_ceiling": "retrospective conditional-domain hypothesis only; any rule requires a separately frozen fresh validation",
        },
        "mechanism_analysis": {
            "qualification_rule": "signature in at least 3 loss pairs across at least 2 instances, loss prevalence >= 0.75, and non-loss prevalence <= half the loss prevalence",
            "signature_summaries": mechanism_summaries,
            "claim_ceiling": "retrospective Balanced V2 mechanism hypothesis only; no tuning or admission on consumed B5-A",
        },
        "cross_cohort_attribution": {
            "stage_mean_normalized_progress": stage_means,
            "balanced_progress_shift_b5_minus_b4": balanced_shift,
            "control_progress_shift_b5_minus_b4": control_shift,
            "paired_advantage_shift_b5_minus_b4": balanced_shift - control_shift,
            "interpretation_boundary": "descriptive decomposition across different cohorts; not a causal estimate of cohort identity",
        },
        "instance_rows": instance_rows,
        "paired_divergence_and_trajectory_rows": pair_rows,
        "source_sha256": source_hashes,
        "limitations": [
            "six unique landscapes and three provider identities per landscape are insufficient to establish a stable domain without fresh validation",
            "paired identities are outcome-blind session labels, not controlled provider RNG seeds",
            "trajectory features are post-treatment and cannot define a pre-dispatch applicability rule",
            "failure to qualify a mechanism does not prove stochasticity",
        ],
        "next_boundary": (
            "freeze a conditional-domain fresh validation; do not admit the operator"
            if terminal == "OBSERVABLE_CONDITIONAL_DOMAIN_HYPOTHESIS_IDENTIFIED"
            else "return to Development for a versioned Balanced V2; do not tune on B5-A"
            if terminal == "BALANCED_V2_MECHANISM_HYPOTHESIS_IDENTIFIED"
            else "close the Balanced operator-admission route; retain B4-A as a local phenomenon and spend no new fresh budget"
        ),
        "claim_ceiling": "mechanism autopsy over consumed finite synthetic B4-A/B5-A evidence only; not fresh evidence, operator admission, general model superiority, complete search, device, user, safety-effect, or production evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite create-once autopsy result: {args.output}")
    result = analyze(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "terminal": result["terminal"], "model_calls": 0}))


if __name__ == "__main__":
    main()
