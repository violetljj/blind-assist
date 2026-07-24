from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import causal_route_intrusion_signal_r0 as core


def measurement(frame: int, radial: float, lateral: float, height: float) -> dict:
    return {
        "frame_id": float(frame),
        "timestamp_ns": float(frame * 100_000_000),
        "timestamp_s": frame * 0.1,
        "radial": radial,
        "lateral": lateral,
        "log_height": height,
    }


class CausalRouteIntrusionSignalR0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config_path = (
            cls.repo / "configs/ustrf_causal_route_intrusion_signal_r0.json"
        )
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def test_frozen_contract_and_parent_stop_terminal_pass(self) -> None:
        result = core.load_and_verify_config(self.repo, self.config_path)
        self.assertEqual(
            result["signal"]["name"],
            "causal_route_relative_intrusion_trend_2_of_3",
        )

    def test_policy_authority_cannot_be_opened(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["claim_boundary"]["successor_policy"] = True
        temporary = self.repo / "artifacts.local" / "test-signal-config-mutation.json"
        core.atomic_write_json(temporary, changed)
        try:
            with self.assertRaisesRegex(core.SignalContractError, "authority_opened"):
                core.load_and_verify_config(self.repo, temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def test_two_of_three_trends_are_positive(self) -> None:
        rows = [
            measurement(i, 1.0 - i * 0.1, 0.8 - i * 0.05, 0.1)
            for i in range(5)
        ]
        result = core.signal_components(rows)
        self.assertTrue(result["signal_positive"])
        self.assertEqual(result["positive_component_count"], 2)

    def test_one_of_three_is_not_positive(self) -> None:
        rows = [
            measurement(i, 1.0 - i * 0.1, 0.4 + i * 0.05, 0.2 - i * 0.02)
            for i in range(5)
        ]
        result = core.signal_components(rows)
        self.assertFalse(result["signal_positive"])
        self.assertEqual(result["positive_component_count"], 1)

    def test_timestamp_translation_is_invariant(self) -> None:
        rows = [
            measurement(i, 1.0 - i * 0.1, 0.8 - i * 0.02, 0.1 + i * 0.03)
            for i in range(5)
        ]
        shifted = copy.deepcopy(rows)
        for row in shifted:
            row["timestamp_s"] += 10_000.0
        left = core.signal_components(rows)
        right = core.signal_components(shifted)
        self.assertEqual(left["signal_positive"], right["signal_positive"])
        self.assertAlmostEqual(
            left["route_relative_radial_distance_slope_per_s"],
            right["route_relative_radial_distance_slope_per_s"],
            places=10,
        )

    def test_future_change_does_not_change_prefix(self) -> None:
        prefix = [
            measurement(i, 1.0 - i * 0.1, 0.8 - i * 0.02, 0.1 + i * 0.03)
            for i in range(5)
        ]
        before = core.signal_components(prefix)
        future = measurement(5, 100.0, 100.0, -100.0)
        self.assertEqual(before, core.signal_components((prefix + [future])[:5]))

    def test_half_open_poisson_zero_event_bound(self) -> None:
        upper = core._poisson_upper_mean(0, 0.95)
        self.assertAlmostEqual(upper, -__import__("math").log(0.05), places=10)

    def test_invalid_history_length_fails(self) -> None:
        with self.assertRaisesRegex(core.SignalContractError, "history_not_five"):
            core.signal_components(
                [measurement(i, 1.0, 1.0, 1.0) for i in range(4)]
            )


if __name__ == "__main__":
    unittest.main()
