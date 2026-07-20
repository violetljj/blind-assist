import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_public_silver_mechanism_routed_expert_probe as probe


class MechanismRoutedExpertProbeTest(unittest.TestCase):
    def test_nested_evaluation_excludes_held_out_source(self):
        # Each of three sources contributes a balanced pair for each mechanism.
        features = []
        alerts = []
        mechanisms = []
        episode_ids = []
        source_ids = []
        for source_index in range(3):
            for mechanism in (0, 1):
                for alert in (0, 1):
                    # Mechanism is x0 sign; alert is x1 sign inside each mechanism.
                    features.append([(-2.0 if mechanism == 0 else 2.0) + source_index * 0.01, -2.0 if alert == 0 else 2.0])
                    alerts.append(alert)
                    mechanisms.append(mechanism)
                    episode_ids.append(f"s{source_index}-m{mechanism}-a{alert}")
                    source_ids.append(f"s{source_index}")
        result = probe.nested_source_evaluation(
            np.asarray(features), np.asarray(alerts), np.asarray(mechanisms), episode_ids, source_ids, ridge=1.0
        )
        self.assertEqual(1.0, result["router_metrics"]["balanced_accuracy"])
        self.assertEqual(1.0, result["routed_expert_metrics"]["balanced_accuracy"])
        for fold in result["folds"]:
            self.assertTrue(all(item.startswith(fold["held_out_source_id"]) for item in fold["held_out_episode_ids"]))

    def test_gate_requires_routed_improvement_not_oracle_only(self):
        metrics = lambda score: {
            "balanced_accuracy": score,
            "candidate_no_alert_recall": score,
            "candidate_alert_recall": score,
        }
        result = {
            "repeat_exact": True,
            "unified_alert_head": {"metrics": metrics(0.6)},
            "mechanism_routed": {
                "router_metrics": metrics(0.5),
                "oracle_expert_metrics": metrics(0.9),
                "routed_expert_metrics": metrics(0.6),
                "per_mechanism": {
                    name: {"oracle_expert_metrics": metrics(0.9)} for name in probe.MECHANISMS
                },
            },
        }
        gate = probe.gate(result, minimum_balanced_accuracy=0.7, minimum_class_recall=0.5)
        self.assertTrue(gate["oracle_upper_bound_passed"])
        self.assertFalse(gate["observable_routed_gate_passed"])

    def test_predict_uses_two_class_logits(self):
        fitted = {"kernel": np.asarray([[-1.0, 1.0]]), "bias": np.zeros(2)}
        self.assertEqual([0, 1], probe.predict(fitted, np.asarray([[-1.0], [1.0]])).tolist())


if __name__ == "__main__":
    unittest.main()
