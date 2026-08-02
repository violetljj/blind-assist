import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from evaluate_stage_c_d6_sanpo_real_veto_calibration import (
        CANDIDATE_FEATURES,
        fit_oof,
        paired_direction,
        stable_fold_assignments,
    )
except ModuleNotFoundError as error:
    if error.name == "sklearn":
        raise unittest.SkipTest(
            "Calibration tests require the sklearn validation venv"
        ) from error
    raise


def row(
    session: str,
    phase: str,
    target: float,
    candidate: float,
) -> dict:
    value = {
        "source_session_id": session,
        "phase": phase,
        "false_alert_target": target,
        "comparator_mean": 0.5,
        "comparator_p95": 0.5,
        "comparator_max": 0.5,
        "known_mean": 0.8,
        "known_p95": 0.9,
        "log1p_eligible_cell_count": 3.0,
        "near_fraction": 0.5,
        "body_fraction": 0.5,
        "direction_2_fraction": 0.5,
        "distance_mean_normalized": 0.5,
        "candidate_mean": candidate,
        "candidate_p95": candidate,
        "candidate_max": candidate,
    }
    return value


class SanpoRealVetoCalibrationTest(unittest.TestCase):
    def test_grouped_oof_keeps_positive_phases_together(self):
        rows = []
        for index in range(10):
            session = f"positive-{index}"
            rows.append(
                row(session, "positive_alertable", 0.0, 0.1)
            )
            rows.append(row(session, "positive_passed", 1.0, 0.9))
        for index in range(10):
            rows.append(
                row(
                    f"negative-{index}",
                    "negative_event",
                    1.0,
                    0.8,
                )
            )

        assignments = stable_fold_assignments(rows)
        result = fit_oof(rows, CANDIDATE_FEATURES, assignments)
        pairs = paired_direction(rows, result["probability"])

        self.assertEqual(len(assignments), 20)
        self.assertEqual(result["metrics"]["auroc"], 1.0)
        self.assertEqual(pairs["pair_count"], 10)
        self.assertEqual(pairs["passed_score_higher_count"], 10)
        self.assertEqual(set(assignments.values()), set(range(5)))
        for fold in range(5):
            self.assertGreaterEqual(
                sum(value == fold for value in assignments.values()),
                4,
            )


if __name__ == "__main__":
    unittest.main()
