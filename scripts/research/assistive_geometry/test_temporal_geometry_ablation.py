from __future__ import annotations

import copy
import unittest

import torch

from scripts.research.assistive_geometry.temporal_geometry_ablation import (
    CANDIDATES,
    PARAMETER_BUDGET,
    build_temporal_candidate,
    candidate_receipt,
    encode_geometry_state,
    validate_output_contract,
)


def fixture(batch: int = 2, steps: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(19)
    return {
        "clearance_m": torch.rand(batch, steps, 3) * 2.0,
        "clearance_valid": torch.ones(batch, steps, 3, dtype=torch.bool),
        "occupancy_probability": torch.rand(batch, steps, 3, 3),
        "task_confidence": torch.rand(batch, steps, 3, 3),
        "state_known": torch.ones(batch, steps, 3, 3, dtype=torch.bool),
        "ground_support": torch.rand(batch, steps, 3),
        "captured_at_s": torch.arange(steps, dtype=torch.float32).repeat(batch, 1) * 0.1,
    }


class TemporalGeometryAblationTests(unittest.TestCase):
    def test_unknown_occupancy_payload_is_neutralized(self) -> None:
        left = fixture(batch=1, steps=3)
        left["state_known"][:, 1, 0, 0] = False
        right = copy.deepcopy(left)
        left["occupancy_probability"][:, 1, 0, 0] = 0.0
        right["occupancy_probability"][:, 1, 0, 0] = 1.0
        self.assertTrue(torch.equal(encode_geometry_state(left), encode_geometry_state(right)))

    def test_non_monotonic_timestamp_fails_closed(self) -> None:
        state = fixture(batch=1, steps=3)
        state["captured_at_s"][0, 2] = state["captured_at_s"][0, 1]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            encode_geometry_state(state)

    def test_all_candidates_share_contract_and_budget(self) -> None:
        state = fixture()
        for name in CANDIDATES:
            model = build_temporal_candidate(name)
            output = model(state)
            validate_output_contract(output, batch=2, steps=8)
            receipt = candidate_receipt(name, model)
            self.assertLessEqual(receipt["parameter_count"], PARAMETER_BUDGET)
            self.assertFalse(receipt["predicts_final_tristate"])

    def test_all_candidates_are_causal(self) -> None:
        base = fixture(batch=1, steps=8)
        changed = copy.deepcopy(base)
        changed["clearance_m"][:, 4:] = 0.0
        changed["occupancy_probability"][:, 4:] = 1.0
        for name in CANDIDATES:
            torch.manual_seed(23)
            model = build_temporal_candidate(name).eval()
            with torch.no_grad():
                prefix = model(base)
                perturbed = model(changed)
            for key in prefix:
                self.assertTrue(torch.allclose(prefix[key][:, :4], perturbed[key][:, :4], atol=1e-6, rtol=0.0), name)


if __name__ == "__main__":
    unittest.main()
