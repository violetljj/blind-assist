"""Outcome-free paired-layout planning calculation for the CNH route design.

Only Python's standard library is used. This does not open dataset/model outputs.
Normal planning approximations are NOT actual bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import NormalDist


NORMAL = NormalDist()
Z95 = NORMAL.inv_cdf(0.975)
DESIGNS = {"old80": (24, 20), "reallocate80": (32, 32),
           "double_priority124": (48, 40), "priority64_each": (64, 64)}


def estimate(n: float, delta: float, discordance: float) -> dict:
    """D = candidate event hit - baseline event hit, D in {-1,0,1}.

    Two positive events within a layout are assumed perfectly correlated,
    hence each layout supplies one independent observation. Layouts and the
    two obstacle families are independent in this planning model.
    """
    variance = discordance - delta * delta
    se = math.sqrt(variance / n)
    ci_threshold = Z95 * se
    return {
        "layouts": n,
        "effective_independent_units": n,
        "se_delta": se,
        "approx_95ci_halfwidth": ci_threshold,
        "prob_ci_lower_gt_zero": NORMAL.cdf((delta - ci_threshold) / se),
        "prob_observed_delta_ge_010_and_ci_lower_gt_zero": NORMAL.cdf(
            (delta - max(0.10, ci_threshold)) / se),
    }


def required_layouts(delta: float, discordance: float, target: float) -> dict:
    variance = discordance - delta * delta
    zpower = NORMAL.inv_cdf(target)
    ci_n = math.ceil(variance * (Z95 + zpower) ** 2 / delta ** 2)
    if delta <= 0.10:
        joint_n = None  # P(estimated delta >= true delta) approaches 0.5.
    else:
        floor_n = math.ceil(variance * zpower ** 2 / (delta - 0.10) ** 2)
        joint_n = max(ci_n, floor_n)
    return {"per_family_target_probability": target,
            "layouts_per_family_ci_only": ci_n,
            "layouts_per_family_ci_and_point_floor": joint_n}


def monte_carlo(nhead: int, nrod: int, delta: float, q: float,
                rng: random.Random, repetitions: int = 10000) -> dict:
    """Check normal planning with discrete paired differences; no CI fitting.

    Keep the planning variance known to separate normal-approximation error
    from a different actual estimator. Each layout's two events share D.
    """
    plus, minus = (q + delta) / 2, (q - delta) / 2
    lower_cutoff = Z95 * math.sqrt((q - delta * delta) * (1 / nhead + 1 / nrod) / 4)
    passed_ci = passed_joint = 0
    for _ in range(repetitions):
        totals = []
        for n in (nhead, nrod):
            total = 0
            for _ in range(n):
                u = rng.random()
                total += 1 if u < plus else (-1 if u < plus + minus else 0)
            totals.append(total / n)
        mean = sum(totals) / 2
        if mean > lower_cutoff:
            passed_ci += 1
            if mean + 1e-12 >= 0.10:
                passed_joint += 1
    p_ci, p_joint = passed_ci / repetitions, passed_joint / repetitions
    return {"repetitions": repetitions, "prob_macro_ci_lower_gt_zero": p_ci,
            "prob_macro_point_floor_and_ci": p_joint,
            "joint_probability_mc_se": math.sqrt(p_joint * (1 - p_joint) / repetitions),
            "note": "Discrete D simulation using known planning variance; does not simulate actual paired-layout bootstrap or other gates."}


def calculate() -> dict:
    rows = []
    rng = random.Random(20260924)
    for delta in (0.10, 0.15):
        for q in (0.20, 0.35):
            assert abs(delta) <= q <= 1
            for name, (nhead, nrod) in DESIGNS.items():
                head, rod = estimate(nhead, delta, q), estimate(nrod, delta, q)
                # Equal-family macro effect, not a post-hoc pooled event score.
                # Var((mean_head+mean_rod)/2)=variance*(1/nhead+1/nrod)/4.
                pooled = estimate(4 / (1 / nhead + 1 / nrod), delta, q)
                pooled.pop("layouts")
                rows.append({
                    "design": name, "true_delta": delta,
                    "discordance_probability": q,
                    "p_candidate_only": (q + delta) / 2,
                    "p_baseline_only": (q - delta) / 2,
                    "head": head, "rod": rod,
                    "balanced_two_family_macro_effect": pooled,
                    "monte_carlo_check": monte_carlo(nhead, nrod, delta, q, rng),
                    "diagnostic_not_gate_prob_both_families_ci_positive":
                        head["prob_ci_lower_gt_zero"] * rod["prob_ci_lower_gt_zero"],
                    "diagnostic_not_gate_prob_both_families_point_and_ci":
                        head["prob_observed_delta_ge_010_and_ci_lower_gt_zero"] *
                        rod["prob_observed_delta_ge_010_and_ci_lower_gt_zero"],
                })
    sizes = []
    for delta in (0.10, 0.15):
        for q in (0.20, 0.35):
            required = required_layouts(delta, q, 0.80)
            joint = required["layouts_per_family_ci_and_point_floor"]
            sizes.append({
                "true_delta": delta, "discordance_probability": q,
                "macro_target_probability": 0.80,
                "balanced_layouts_per_family_ci_only":
                    math.ceil(required["layouts_per_family_ci_only"] / 2),
                "balanced_layouts_per_family_ci_and_point_floor":
                    None if joint is None else math.ceil(joint / 2),
            })
    return {
        "schema_version": 1,
        "monte_carlo_seed": 20260924,
        "evidence_type": "synthetic_outcome_free_analytical_and_monte_carlo_design_only",
        "backend": "CPU", "backend_reason": "TASK_NOT_GPU_SUITABLE",
        "method": "normal approximation with true planning variance q-delta^2; two-sided 95% CI",
        "primary_estimand": "equal-weight head/rod family macro paired recall difference; no per-family significance gate",
        "assumptions": [
            "Each layout supplies at most two positive events; assume both exist and their paired differences have ICC=1.",
            "Paired per-event difference D has P(+1)=(q+delta)/2 and P(-1)=(q-delta)/2.",
            "Layouts and obstacle families are independent; different sites may violate this.",
            "One fixed trained model and one fixed condition; no seed, validation-threshold, or sensor-parameter variability is simulated.",
            "Actual evaluation must use the specified paired layout bootstrap; these approximate intervals are not its measured operating characteristics.",
            "Same false-alert budget and event eligibility are assumed fixed; no real outcomes, datasets, or model outputs were read.",
            "Other outcome gates and simultaneous stress conditions can only reduce these acceptance probabilities.",
            "The primary rule uses the balanced macro effect only; simultaneous per-family probabilities are diagnostics, not plan requirements.",
            "At true macro delta=0.10 the point-estimate floor alone has asymptotic probability 0.5, so collecting more layouts cannot achieve 80% acceptance at the exact floor.",
        ],
        "rows": rows,
        "sample_size_approximations": sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = calculate()
    # Mechanical identities, not experimental evidence or test-data checks.
    for row in report["rows"]:
        for family in ("head", "rod"):
            r = row[family]
            assert 0 <= r["prob_observed_delta_ge_010_and_ci_lower_gt_zero"] <= r["prob_ci_lower_gt_zero"] <= 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['rows'])} analytical scenarios to {args.output}")
    for row in report["rows"]:
        macro = row["balanced_two_family_macro_effect"]
        print(f"d={row['true_delta']:.2f} q={row['discordance_probability']:.2f} "
              f"{row['design']}: "
              f"macro CI={macro['prob_ci_lower_gt_zero']:.3f}; "
              f"macro CI+floor={macro['prob_observed_delta_ge_010_and_ci_lower_gt_zero']:.3f}; "
              f"MC={row['monte_carlo_check']['prob_macro_point_floor_and_ci']:.3f}")
    print(json.dumps(report["sample_size_approximations"], indent=2))


if __name__ == "__main__":
    main()
