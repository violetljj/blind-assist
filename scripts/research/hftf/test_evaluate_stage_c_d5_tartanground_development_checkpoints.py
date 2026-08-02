import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_stage_c_d5_tartanground_development_checkpoints import (
    comparisons,
    parse_models,
)


def metrics(value):
    return {
        "risk_future_body_head": {
            "f1": value,
            "precision": value,
            "recall": value,
            "false_positive_rate": 1.0 - value,
            "risk_score_mae": 1.0 - value,
        },
        "future_body_head_macro_f1": value,
    }


class TartanGroundCheckpointEvaluationTest(unittest.TestCase):
    def test_parse_models_rejects_duplicate_names(self):
        with self.assertRaises(ValueError):
            parse_models(
                [
                    ["same", "single", "a.pt"],
                    ["same", "history", "b.pt"],
                ]
            )

    def test_comparison_counts_environment_wins_and_losses(self):
        models = {
            "single": {
                "overall": metrics(0.5),
                "by_environment": {
                    "A": metrics(0.5),
                    "B": metrics(0.5),
                },
            },
            "history": {
                "overall": metrics(0.6),
                "by_environment": {
                    "A": metrics(0.7),
                    "B": metrics(0.4),
                },
            },
        }

        result = comparisons(models, "single")["history"]

        self.assertAlmostEqual(
            result["aggregate"]["future_body_head_macro_f1"],
            0.1,
        )
        self.assertEqual(result["environment_macro_f1_wins"], 1)
        self.assertEqual(result["environment_macro_f1_losses"], 1)
        self.assertAlmostEqual(
            result["environment_macro_f1_worst_delta"],
            -0.1,
        )


if __name__ == "__main__":
    unittest.main()
