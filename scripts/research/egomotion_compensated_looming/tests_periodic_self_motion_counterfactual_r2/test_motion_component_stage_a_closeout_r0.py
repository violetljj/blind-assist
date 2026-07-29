from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_motion_component_stage_a_closeout_independent_r0 as closeout,
)


ROOT = Path(__file__).resolve().parents[4]
PATHS = closeout._default_paths(ROOT)


class MotionComponentStageACloseoutTests(unittest.TestCase):
    def _build(self) -> tuple[dict, dict]:
        return closeout.validate_and_build(
            ROOT,
            **{
                key: value
                for key, value in PATHS.items()
                if key not in {"closeout_path", "decision_path"}
            },
        )

    def test_valid_stage_a_closes_with_contract_only_stage_b(self) -> None:
        receipt, decision = self._build()
        self.assertEqual(receipt["terminal"], "VALID / STAGE_A_COMPLETE")
        self.assertEqual(
            receipt["replicated_components"],
            ["ROTATION_MINUS_STATIC", "TRANSLATION_MINUS_ROTATION"],
        )
        self.assertEqual(
            receipt["unstable_components"], ["FULL_MINUS_MAX_SINGLE"]
        )
        self.assertEqual(
            decision["decision"],
            "ACTIVATE_STAGE_B_CONTRACT_PREPARATION_ONLY",
        )
        self.assertFalse(decision["stage_b_execution_authorized"])
        self.assertFalse(
            decision["formal_execution_authorized_by_this_decision"]
        )

    def test_contrasts_are_recomputed_from_arm_metrics(self) -> None:
        analysis = closeout.load_json(PATHS["stage_2_analysis_path"])
        forged = copy.deepcopy(analysis)
        forged["routing_direction_summary"]["FULL_MINUS_MAX_SINGLE"][
            "positive_count"
        ] = 4
        with self.assertRaisesRegex(
            closeout.InvalidStageACloseout,
            "STAGE_2_FULL_MINUS_MAX_SINGLE_positive_count",
        ):
            closeout.recompute_contrasts(forged, 2)

    def test_stage_2_receipt_mutation_fails_closed(self) -> None:
        receipt = closeout.load_json(PATHS["stage_2_receipt_path"])
        receipt["terminal"] = "VALID / STAGE_2_ROUTING_COMPLETE_FORGED"
        with tempfile.TemporaryDirectory() as directory:
            forged_path = Path(directory) / "stage2_receipt.json"
            forged_path.write_text(
                json.dumps(receipt, ensure_ascii=False), encoding="utf-8"
            )
            arguments = {
                key: value
                for key, value in PATHS.items()
                if key not in {"closeout_path", "decision_path"}
            }
            arguments["stage_2_receipt_path"] = forged_path
            with self.assertRaisesRegex(
                closeout.InvalidStageACloseout,
                "STAGE_2_RECEIPT_TERMINAL",
            ):
                closeout.validate_and_build(ROOT, **arguments)

    def test_analysis_controls_reject_pair_pooled_inference(self) -> None:
        analysis = closeout.load_json(PATHS["stage_1_analysis_path"])
        analysis["analysis_controls"]["pair_pooled_inference"] = True
        with self.assertRaisesRegex(
            closeout.InvalidStageACloseout,
            "STAGE_1_ANALYSIS_ANALYSIS_CONTROLS",
        ):
            closeout.recompute_contrasts(analysis, 1)


if __name__ == "__main__":
    unittest.main()
