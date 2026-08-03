import json
import unittest
from pathlib import Path

from evaluate_corridor_conditioned_future_field_a2_1 import FEATURE_NAMES


class FrozenCorridorConditionedFutureFieldA21Test(unittest.TestCase):
    def test_frozen_model_dimensions_and_order(self) -> None:
        path = Path(__file__).with_name(
            "CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_FROZEN_MODEL.json"
        )
        model = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(model["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(len(model["feature_mean"]), len(FEATURE_NAMES))
        self.assertEqual(len(model["feature_scale"]), len(FEATURE_NAMES))
        self.assertEqual(
            len(model["weights_intercept_then_features"]), len(FEATURE_NAMES) + 1
        )


if __name__ == "__main__":
    unittest.main()
