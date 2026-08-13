from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ag_factorwise_no_regret_oracle_parent_gate_canary import (  # noqa: E402
    derive_terminal,
    evaluate_lane_gate,
)


def parent(
    parent_id: str,
    domain: str,
    *,
    selected_coverage: float,
    selected_mae: float,
    selected_bad: float,
    oracle_coverage: float = 0.5,
) -> dict:
    base = {"mae_m": 1.0, "bad_gt_0_10_m_fraction": 0.5}
    return {
        "parent_id": parent_id,
        "domain": domain,
        "base": base,
        "expert": {"mae_m": 1.2, "bad_gt_0_10_m_fraction": 0.6},
        "selected": {
            "mae_m": selected_mae,
            "bad_gt_0_10_m_fraction": selected_bad,
        },
        "oracle": {"mae_m": 0.8, "bad_gt_0_10_m_fraction": 0.4},
        "selected_coverage_fraction": selected_coverage,
        "oracle_coverage_fraction": oracle_coverage,
    }


def summary(rows: list[dict]) -> dict:
    def values(selected: list[dict]) -> dict:
        result = {
            lane: {
                metric: sum(float(row[lane][metric]) for row in selected)
                / len(selected)
                for metric in ("mae_m", "bad_gt_0_10_m_fraction")
            }
            for lane in ("base", "expert", "selected", "oracle")
        }
        result["selected_coverage_fraction"] = sum(
            float(row["selected_coverage_fraction"]) for row in selected
        ) / len(selected)
        result["oracle_coverage_fraction"] = sum(
            float(row["oracle_coverage_fraction"]) for row in selected
        ) / len(selected)
        return result

    domains = sorted({row["domain"] for row in rows})
    return {
        "parent_macro": values(rows),
        "by_domain": {
            domain: values([row for row in rows if row["domain"] == domain])
            for domain in domains
        },
        "per_parent": rows,
    }


class FactorwiseNoRegretCanaryTest(unittest.TestCase):
    def test_selector_pass_requires_nonzero_safe_coverage_across_parents(self) -> None:
        rows = [
            parent(
                "a",
                "ARKIT",
                selected_coverage=0.4,
                selected_mae=0.8,
                selected_bad=0.4,
            ),
            parent(
                "b",
                "BONN",
                selected_coverage=0.2,
                selected_mae=1.0,
                selected_bad=0.5,
            ),
        ]
        gate = evaluate_lane_gate(summary(rows), "selected")
        self.assertTrue(gate["pass"])
        self.assertEqual(2, gate["nonzero_parent_count"])
        self.assertEqual(0, gate["harmful_parent_count"])

    def test_one_harmed_parent_rejects_macro_improvement(self) -> None:
        rows = [
            parent(
                "helpful",
                "ARKIT",
                selected_coverage=1.0,
                selected_mae=0.2,
                selected_bad=0.1,
            ),
            parent(
                "harmed",
                "BONN",
                selected_coverage=1.0,
                selected_mae=1.01,
                selected_bad=0.5,
            ),
        ]
        gate = evaluate_lane_gate(summary(rows), "selected")
        self.assertFalse(gate["pass"])
        self.assertEqual(1, gate["harmful_parent_count"])
        self.assertEqual("harmed", gate["harmful_parents"][0]["parent_id"])

    def test_zero_coverage_fallback_is_not_mechanism_success(self) -> None:
        rows = [
            parent(
                "a",
                "ARKIT",
                selected_coverage=0.0,
                selected_mae=1.0,
                selected_bad=0.5,
            ),
            parent(
                "b",
                "BONN",
                selected_coverage=0.0,
                selected_mae=1.0,
                selected_bad=0.5,
            ),
        ]
        gate = evaluate_lane_gate(summary(rows), "selected")
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["checks"]["minimum_macro_coverage"])

    def test_terminal_routes_oracle_headroom_without_selector_pass_to_lcb(self) -> None:
        oracle = {"pass": True}
        selector = {"pass": False}
        terminal, successor = derive_terminal(oracle, selector)
        self.assertEqual(
            "AG_FACTORWISE_NO_REGRET_ORACLE_HEADROOM_SELECTOR_GATE_FAIL",
            terminal,
        )
        self.assertEqual(
            "TRAIN_ONE_SIDED_ADVANTAGE_LCB_ROUTER_ON_FIT_ONLY",
            successor,
        )


if __name__ == "__main__":
    unittest.main()
