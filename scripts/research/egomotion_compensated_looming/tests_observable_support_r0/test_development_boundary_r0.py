from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    PROTOCOL_SHA256,
    enumerate_trials,
    sha256_file,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0.receipt import (
    DESIGN_LOCK_SHA256,
    RECEIPT_SCHEMA_VERSION,
)
from scripts.research.egomotion_compensated_looming.run_observable_support_recovery_development_r0 import (
    DEVELOPMENT_SEEDS,
    OUTPUT_ROOT,
    development_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_ROOT = Path(__file__).resolve().parents[1]


class DevelopmentBoundaryR0Test(unittest.TestCase):
    def test_design_and_protocol_hashes_are_exact(self) -> None:
        design = (
            REPO_ROOT
            / "docs"
            / "research"
            / "rcle"
            / "RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_LOCK_2026-07-26.json"
        )
        self.assertEqual(sha256_file(design), DESIGN_LOCK_SHA256)
        self.assertEqual(
            PROTOCOL_SHA256,
            "d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502",
        )

    def test_development_inventory_is_exact_and_complete(self) -> None:
        protocol = development_protocol()
        trials = enumerate_trials(protocol)
        self.assertEqual(DEVELOPMENT_SEEDS, tuple(range(2000, 2020)))
        self.assertEqual(len(trials), 2520)
        self.assertEqual(len({trial.trial_id for trial in trials}), 2520)
        self.assertEqual(sum(trial.split == "clean" for trial in trials), 1680)
        self.assertEqual(sum(trial.split == "stress" for trial in trials), 840)
        self.assertEqual({trial.seed for trial in trials}, set(DEVELOPMENT_SEEDS))

    def test_output_location_and_receipt_schema_are_fixed(self) -> None:
        expected = (
            REPO_ROOT
            / "artifacts.local"
            / "evidence"
            / "rcle_observable_support_recovery_r0"
            / "development_gate_r0"
        )
        self.assertEqual(OUTPUT_ROOT, expected)
        schema_path = (
            MODULE_ROOT
            / "schemas"
            / "rcle_observable_support_recovery_development_receipt_r0.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            RECEIPT_SCHEMA_VERSION,
        )

    def test_candidate_source_has_static_oracle_firewall(self) -> None:
        candidate_root = MODULE_ROOT / "rcle_observable_support_r0"
        forbidden = (
            "occlusion_" + "masks",
            "_apply_" + "degradation",
            "generator_" + "occlusion",
        )
        for path in candidate_root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{token} leaked into {path}")

    def test_runner_has_no_seed_or_output_override(self) -> None:
        runner = (
            MODULE_ROOT / "run_observable_support_recovery_development_r0.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--seed", runner)
        self.assertNotIn("--output-root", runner)
        self.assertNotIn("--dataset-root", runner)
        self.assertNotIn("range(3000", runner)


if __name__ == "__main__":
    unittest.main()
