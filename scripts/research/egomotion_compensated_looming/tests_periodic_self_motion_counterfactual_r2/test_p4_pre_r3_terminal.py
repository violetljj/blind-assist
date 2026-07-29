from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    p4_pre_r3_terminal_receipt_r0 as producer,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_p4_pre_r3_terminal_independent_r0 as validator,
)


class PreR3TerminalTests(unittest.TestCase):
    def test_validator_rejects_nonzero_r3_arm_count(self):
        with tempfile.TemporaryDirectory() as literal:
            root = Path(literal)
            activation = root / "activation.json"
            manipulation = root / "producer.json"
            independent = root / "independent.json"
            result = root / "result.json"
            activation.write_text(
                json.dumps(
                    {
                        "protocol_id": producer.PROTOCOL_ID,
                        "formal_execution_authorized": True,
                        "p4_activated": True,
                    }
                ),
                encoding="utf-8",
            )
            manipulation.write_text(
                json.dumps(
                    {
                        "terminal": producer.TERMINAL,
                        "r3_imported_or_executed": False,
                        "algorithm_output_read": False,
                        "subgroups": [
                            {
                                "blur_subgroup_pass": True,
                                "low_texture_subgroup_pass": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            independent.write_text(
                json.dumps(
                    {
                        "validation": "VALID",
                        "terminal": producer.TERMINAL,
                        "receipt_sha256": producer.sha256_file(manipulation),
                    }
                ),
                encoding="utf-8",
            )
            value = {
                "scientific_terminal": producer.TERMINAL,
                "execution_state": "COMPLETE_PRE_R3_TERMINAL",
                "analysis_performed": False,
                "formal_r3_execution": {
                    "arms_started": 1,
                    "arms_completed": 0,
                    "main_pair_core_calls": 0,
                    "guard_pair_core_calls": 0,
                },
                "manipulation": {
                    "failed_subgroups": [
                        {
                            "blur_subgroup_pass": True,
                            "low_texture_subgroup_pass": False,
                        }
                    ]
                },
                "bindings": {
                    "activation_lock_sha256": producer.sha256_file(activation),
                    "manipulation_producer_receipt_sha256": producer.sha256_file(
                        manipulation
                    ),
                    "manipulation_independent_receipt_sha256": producer.sha256_file(
                        independent
                    ),
                },
            }
            result.write_text(json.dumps(value), encoding="utf-8")
            receipt = validator.validate(
                activation, manipulation, independent, result, root
            )
            self.assertFalse(receipt["validated"])
            self.assertIn("RESULT_R3_COUNTS", receipt["errors"])

    def test_reporting_modules_do_not_import_r3_or_runner(self):
        for module in (producer, validator):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn("rgb_algorithm_development_canary", source)
            self.assertNotIn("p4_formal_runner_r0", source)
