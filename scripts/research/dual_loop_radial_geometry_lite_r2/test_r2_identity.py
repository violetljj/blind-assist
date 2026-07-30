from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
GEOMETRY = importlib.import_module("radial_geometry")
EVALUATOR = importlib.import_module("evaluate_replay")
PRODUCER = importlib.import_module("run_replay")


class R2IdentityTest(unittest.TestCase):
    def test_r2_changes_identity_not_scientific_parameters(self) -> None:
        gray = np.zeros((32, 32), dtype=np.uint8)
        current = GEOMETRY.FrameObservation(
            source_frame_id="f0",
            captured_at_ns=0,
            target_id="track-000",
            track_epoch="track-000:epoch-1",
            region="CENTER",
            roi_xywh_normalized=(0.5, 0.5, 0.5, 0.5),
            gray=gray,
            history_reset=True,
        )
        rows = GEOMETRY.evaluate_pair(None, current)
        self.assertEqual(
            {row["protocol_id"] for row in rows},
            {GEOMETRY.PROTOCOL_ID},
        )
        self.assertEqual(
            {row["implementation_id"] for row in rows},
            {GEOMETRY.IMPLEMENTATION_ID},
        )
        self.assertEqual(
            GEOMETRY.PARAMETER_SHA256,
            GEOMETRY._R1.PARAMETER_SHA256,
        )
        self.assertEqual(
            [row["abstention_reason"] for row in rows],
            ["INSUFFICIENT_HISTORY", "INSUFFICIENT_HISTORY"],
        )

    def test_wrapped_producer_and_evaluator_use_r2_identity(self) -> None:
        self.assertIs(PRODUCER._R1.evaluate_pair, GEOMETRY.evaluate_pair)
        self.assertEqual(
            EVALUATOR._R1.PROTOCOL_ID,
            GEOMETRY.PROTOCOL_ID,
        )
        self.assertEqual(
            EVALUATOR.SCIENTIFIC_GATE_CONTRACT_SHA256,
            EVALUATOR._R1.SCIENTIFIC_GATE_CONTRACT_SHA256,
        )

    def test_producer_rejects_r1_execution_evidence_before_open(self) -> None:
        r1_root = Path(
            "artifacts.local/evidence/dual-loop/"
            "target-track-causal-radial-geometry-lite-r1"
        )
        forbidden_replay = (
            r1_root / "alias" / ".." / "run-r1" / "producer_output.jsonl"
        )
        forbidden_images = (
            r1_root / "alias" / ".." / "run-r1" / "producer-images"
        )
        with mock.patch.object(
            PRODUCER,
            "_ORIGINAL_R1_RUN",
            side_effect=AssertionError("delegated before firewall"),
        ):
            with self.assertRaisesRegex(ValueError, "forbidden input"):
                PRODUCER.run(
                    forbidden_replay,
                    Path("images"),
                    Path("r2-output.jsonl"),
                    "0" * 64,
                )
            with self.assertRaisesRegex(ValueError, "forbidden input"):
                PRODUCER.run(
                    Path("replay_input.jsonl"),
                    forbidden_images,
                    Path("r2-output.jsonl"),
                    "0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
