#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

from scripts.research.taro_o0r_factor_headroom_runtime.statistics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    StatisticsError,
    DIAGNOSTIC_ARMS,
    evaluate_factor_diagnostics,
    evaluate_factor_headroom,
    holm_bonferroni,
    proper_interval_score,
)


MODE = "VALUE_ONLY_COMMON_SUPPORT"
BASELINE = "NONE"
CANDIDATE = "SCALE_SUPPORT_BOUNDARY"


def _arm_row(
    *,
    parent: str,
    query: str,
    arm: str,
    truth_value: float | None,
    truth_state: str,
    width: float | None,
    predicted_state: str,
    known: bool = True,
    strata: dict[str, object] | None = None,
) -> dict[str, object]:
    if known:
        assert truth_value is not None and width is not None
        value = truth_value
        lower = truth_value - width / 2.0
        upper = truth_value + width / 2.0
    else:
        value = None
        lower = None
        upper = None
    return {
        "parent_id": parent,
        "frame_id": f"{parent}:frame",
        "query_id": query,
        "arm": arm,
        "mode": MODE,
        "truth_value_m": truth_value,
        "truth_state": truth_state,
        "truth_known": truth_value is not None,
        "value_m": value,
        "interval_lower_m": lower,
        "interval_upper_m": upper,
        "state": predicted_state,
        "known": known,
        "strata": dict(strata or {}),
    }


def _paired_rows(
    *,
    parent_widths: dict[str, tuple[float, float]] | None = None,
    candidate_unknown: bool = False,
    all_clear_truth: bool = False,
) -> list[dict[str, object]]:
    widths = parent_widths or {
        "p0": (0.60, 0.10),
        "p1": (0.60, 0.10),
        "p2": (0.60, 0.10),
        "p3": (0.60, 0.10),
    }
    rows: list[dict[str, object]] = []
    for index, (parent, (baseline_width, candidate_width)) in enumerate(widths.items()):
        orientation = "portrait" if index < len(widths) / 2 else "landscape"
        cases = [("q0", 0.0, "CLEAR_OBSERVED" if all_clear_truth else "OCCUPIED_OBSERVED")]
        if parent_widths is None:
            cases.append(("q1", 1.0, "CLEAR_OBSERVED"))
        for query, truth_value, truth_state in cases:
            strata = {"orientation": orientation, "distance_band": "near" if query == "q0" else "far"}
            rows.append(
                _arm_row(
                    parent=parent,
                    query=query,
                    arm=BASELINE,
                    truth_value=truth_value,
                    truth_state=truth_state,
                    width=baseline_width,
                    predicted_state=truth_state,
                    strata=strata,
                )
            )
            rows.append(
                _arm_row(
                    parent=parent,
                    query=query,
                    arm=CANDIDATE,
                    truth_value=truth_value,
                    truth_state=truth_state,
                    width=None if candidate_unknown else candidate_width,
                    predicted_state="UNKNOWN" if candidate_unknown else truth_state,
                    known=not candidate_unknown,
                    strata=strata,
                )
            )
    return rows


class ProperIntervalScoreTest(unittest.TestCase):
    def test_score_is_width_inside_and_penalizes_miss(self) -> None:
        self.assertAlmostEqual(proper_interval_score(1.0, 0.8, 1.2), 0.4)
        self.assertAlmostEqual(proper_interval_score(0.5, 0.8, 1.2), 12.4)

    def test_invalid_interval_fails_closed(self) -> None:
        with self.assertRaisesRegex(StatisticsError, "INTERVAL_ORDER_INVALID"):
            proper_interval_score(1.0, 1.2, 0.8)


class FactorHeadroomStatisticsTest(unittest.TestCase):
    def test_known_numeric_truth_may_have_unknown_tristate(self) -> None:
        rows = _paired_rows()
        for row in rows:
            if row["query_id"] == "q1":
                row["truth_state"] = "UNKNOWN"
        result = evaluate_factor_headroom(rows, bootstrap_replicates=50)
        self.assertEqual(result["counts"]["truth_known_queries"], 8)
        self.assertEqual(result["counts"]["truth_not_known_queries_excluded"], 0)
        self.assertAlmostEqual(result["primary"]["parent_macro_paired_improvement_m"], 0.5)

    def test_primary_parent_macro_and_guardrails_pass(self) -> None:
        result = evaluate_factor_headroom(
            _paired_rows(),
            bootstrap_replicates=400,
            critical_strata=("orientation", "distance_band"),
        )
        self.assertTrue(result["gates"]["passed"])
        self.assertAlmostEqual(result["primary"]["parent_macro_paired_improvement_m"], 0.5)
        self.assertAlmostEqual(result["primary"]["bootstrap_lcb_m"], 0.5)
        self.assertEqual(result["primary"]["favorable_parent_fraction"], 1.0)
        self.assertEqual(result["guardrails"]["false_clear_difference"]["bootstrap_ucb"], 0.0)
        self.assertEqual(result["guardrails"]["known_coverage_difference"]["bootstrap_lcb"], 0.0)
        self.assertFalse(result["guardrails"]["all_unknown"]["candidate_all_unknown"])
        self.assertEqual(result["unknown_policy"]["arm_unknown_as_negative"], False)

    def test_bootstrap_is_deterministic(self) -> None:
        rows = _paired_rows()
        first = evaluate_factor_headroom(rows, bootstrap_replicates=257)
        second = evaluate_factor_headroom(rows, bootstrap_replicates=257)
        self.assertEqual(first["primary"]["bootstrap"], second["primary"]["bootstrap"])
        self.assertEqual(first["configuration"]["bootstrap_seed"], DEFAULT_BOOTSTRAP_SEED)

    def test_all_unknown_is_not_converted_to_negative_or_zero_loss(self) -> None:
        result = evaluate_factor_headroom(
            _paired_rows(candidate_unknown=True),
            bootstrap_replicates=50,
        )
        self.assertFalse(result["gates"]["passed"])
        self.assertFalse(result["gates"]["denominators_defined"])
        self.assertFalse(result["gates"]["all_unknown_forbidden"])
        self.assertTrue(result["guardrails"]["all_unknown"]["candidate_all_unknown"])
        self.assertIsNone(result["primary"]["parent_macro_paired_improvement_m"])
        self.assertIn("PRIMARY_DENOMINATOR_UNDEFINED", result["failure_codes"])
        for parent in result["parents"].values():
            self.assertEqual(parent["candidate_known_coverage"], 0.0)
            self.assertIsNone(parent["candidate_false_clear_rate"])

    def test_numeric_known_but_tristate_unknown_is_not_a_negative_class(self) -> None:
        rows = _paired_rows()
        for row in rows:
            if row["arm"] == CANDIDATE:
                row["state"] = "UNKNOWN"
        result = evaluate_factor_headroom(rows, bootstrap_replicates=50)
        self.assertEqual(result["guardrails"]["known_coverage_difference"]["parent_macro_difference"], 0.0)
        self.assertTrue(result["guardrails"]["all_unknown"]["candidate_all_unknown"])
        self.assertEqual(result["guardrails"]["all_unknown"]["candidate_classified_queries"], 0)
        self.assertFalse(result["gates"]["all_unknown_forbidden"])
        self.assertFalse(result["guardrails"]["false_clear_difference"]["defined"])
        for parent in result["parents"].values():
            self.assertEqual(parent["candidate_known_coverage"], 1.0)
            self.assertEqual(parent["paired_classified_occupied_queries"], 0)
            self.assertIsNone(parent["candidate_false_clear_rate"])

    def test_missing_paired_arm_row_is_invalid(self) -> None:
        rows = _paired_rows()
        rows.pop()
        with self.assertRaisesRegex(StatisticsError, "ARM_PAIR_IDENTITY_MISMATCH"):
            evaluate_factor_headroom(rows, bootstrap_replicates=20)

    def test_false_clear_undefined_denominator_fails_gate(self) -> None:
        result = evaluate_factor_headroom(
            _paired_rows(all_clear_truth=True),
            bootstrap_replicates=50,
        )
        self.assertFalse(result["gates"]["denominators_defined"])
        self.assertFalse(result["gates"]["false_clear_difference_ucb_noninferior"])
        self.assertIn("FALSE_CLEAR_DENOMINATOR_UNDEFINED", result["failure_codes"])

    def test_single_parent_driver_is_detected(self) -> None:
        rows = _paired_rows(
            parent_widths={
                "p0": (1.00, 0.10),
                "p1": (0.20, 0.30),
                "p2": (0.20, 0.30),
                "p3": (0.20, 0.30),
            }
        )
        result = evaluate_factor_headroom(rows, bootstrap_replicates=200)
        self.assertEqual(result["guardrails"]["single_parent_driver"]["driver_parent_ids"], ["p0"])
        self.assertFalse(result["gates"]["single_parent_driver_forbidden"])

    def test_critical_stratum_reversal_is_detected(self) -> None:
        rows = _paired_rows(
            parent_widths={
                "p0": (0.80, 0.10),
                "p1": (0.80, 0.10),
                "p2": (0.20, 0.30),
                "p3": (0.20, 0.30),
            }
        )
        result = evaluate_factor_headroom(
            rows,
            bootstrap_replicates=200,
            critical_strata=("orientation",),
        )
        reversals = result["guardrails"]["critical_strata"]["reversals"]
        self.assertEqual([(row["stratum"], row["level"]) for row in reversals], [("orientation", "landscape")])
        self.assertFalse(result["gates"]["critical_stratum_reversal_forbidden"])

    def test_requested_missing_stratum_fails_closed(self) -> None:
        result = evaluate_factor_headroom(
            _paired_rows(),
            bootstrap_replicates=30,
            critical_strata=("not_bound",),
        )
        self.assertFalse(result["guardrails"]["critical_strata"]["defined"])
        self.assertIn("CRITICAL_STRATUM_DENOMINATOR_UNDEFINED", result["failure_codes"])

    def test_predeclared_structurally_absent_level_is_not_fabricated(self) -> None:
        rows = _paired_rows()
        for row in rows:
            row["strata"]["orientation"] = "landscape"
        strict = evaluate_factor_headroom(rows, bootstrap_replicates=30, critical_strata=("orientation",))
        self.assertFalse(strict["guardrails"]["critical_strata"]["defined"])
        bounded = evaluate_factor_headroom(
            rows,
            bootstrap_replicates=30,
            critical_strata=("orientation",),
            structurally_not_applicable_strata={"orientation": "fixed landscape source contract"},
        )
        self.assertTrue(bounded["guardrails"]["critical_strata"]["defined"])
        self.assertTrue(bounded["guardrails"]["critical_strata"]["rows"][0]["structurally_not_applicable"])

    def test_formal_bootstrap_default_remains_twenty_thousand(self) -> None:
        parameter = inspect.signature(evaluate_factor_headroom).parameters["bootstrap_replicates"]
        self.assertEqual(parameter.default, 20_000)
        self.assertEqual(DEFAULT_BOOTSTRAP_REPLICATES, 20_000)


class HolmCorrectionTest(unittest.TestCase):
    def test_seven_factor_diagnostics_use_exact_parent_sign_flip_then_holm(self) -> None:
        rows: list[dict[str, object]] = []
        base_rows = _paired_rows()
        baseline = [row for row in base_rows if row["arm"] == BASELINE]
        candidate = [row for row in base_rows if row["arm"] == CANDIDATE]
        rows.extend(baseline)
        for arm in DIAGNOSTIC_ARMS:
            for row in candidate:
                clone = dict(row)
                clone["arm"] = arm
                rows.append(clone)
        result = evaluate_factor_diagnostics(rows)
        self.assertEqual(result["holm"]["family_size"], 7)
        self.assertEqual(result["arms"], list(DIAGNOSTIC_ARMS))
        self.assertTrue(all(result["contrasts"][arm]["defined"] for arm in DIAGNOSTIC_ARMS))

    def test_seven_diagnostic_holm_step_down(self) -> None:
        result = holm_bonferroni(
            {
                "d1": 0.001,
                "d2": 0.006,
                "d3": 0.020,
                "d4": 0.040,
                "d5": 0.200,
                "d6": 0.500,
                "d7": 0.900,
            }
        )
        self.assertEqual(result["family_size"], 7)
        self.assertTrue(result["rejected"]["d1"])
        self.assertTrue(result["rejected"]["d2"])
        self.assertFalse(result["rejected"]["d3"])
        self.assertTrue(all(not result["rejected"][name] for name in ("d4", "d5", "d6", "d7")))
        self.assertAlmostEqual(result["adjusted_p_values"]["d1"], 0.007)
        self.assertAlmostEqual(result["adjusted_p_values"]["d2"], 0.036)

    def test_holm_rejects_wrong_family_size(self) -> None:
        with self.assertRaisesRegex(StatisticsError, "HOLM_FAMILY_SIZE_INVALID"):
            holm_bonferroni({f"d{i}": 0.1 for i in range(6)})


if __name__ == "__main__":
    unittest.main()
