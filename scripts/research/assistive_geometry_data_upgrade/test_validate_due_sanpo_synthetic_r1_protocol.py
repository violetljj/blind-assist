from __future__ import annotations

import copy
import unittest

from scripts.research.assistive_geometry_data_upgrade import validate_due_sanpo_synthetic_r1_protocol as validator


class SanpoSyntheticR1ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = validator.load_json(validator.PROTOCOL_PATH)

    def test_protocol_lock_validates(self) -> None:
        result = validator.validate_protocol(self.protocol)
        self.assertEqual("VALID", result["status"])

    def test_session_drift_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["locked_source"]["session_id"] = "different"
        with self.assertRaisesRegex(validator.ProtocolError, "locked source drift"):
            validator.validate_protocol(mutated)

    def test_frame_body_access_is_not_authorized(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["execution_authority"]["frame_body_download_or_open"] = True
        with self.assertRaisesRegex(validator.ProtocolError, "execution authority drift"):
            validator.validate_protocol(mutated)

    def test_pose_row_order_cannot_become_frame_binding(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["capability_claim_contract"]["pose_transform_materialized"]["pose_row_order_is_frame_binding"] = True
        with self.assertRaisesRegex(validator.ProtocolError, "pose claim drift"):
            validator.validate_protocol(mutated)

    def test_derived_cadence_cannot_become_explicit_timestamp(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["capability_claim_contract"]["explicit_timestamp_materialized"]["status"] = "PRESENT"
        with self.assertRaisesRegex(validator.ProtocolError, "timestamp claim drift"):
            validator.validate_protocol(mutated)

    def test_depth_cannot_become_support_truth(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["capability_claim_contract"]["oracle_support_factor"]["depth_is_not_support_truth"] = False
        with self.assertRaisesRegex(validator.ProtocolError, "support claim drift"):
            validator.validate_protocol(mutated)

    def test_panoptic_inventory_cannot_become_validated_boundary(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["capability_claim_contract"]["r2_obstacle_boundary_truth_materialized"]["validated_for_claim"] = True
        with self.assertRaisesRegex(validator.ProtocolError, "boundary claim drift"):
            validator.validate_protocol(mutated)

    def test_single_parent_cannot_pass_f1_gate(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["capability_claim_contract"]["parent_gate"]["r2_f1_parent_gate_pass"] = True
        with self.assertRaisesRegex(validator.ProtocolError, "parent gate drift"):
            validator.validate_protocol(mutated)

    def test_training_authority_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["execution_authority"]["model_or_training"] = True
        with self.assertRaisesRegex(validator.ProtocolError, "execution authority drift"):
            validator.validate_protocol(mutated)

    def test_output_root_escape_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["output_contract"]["owned_root"] = "outside"
        with self.assertRaisesRegex(validator.ProtocolError, "output contract drift"):
            validator.validate_protocol(mutated)

    def test_successor_drift_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["unique_successor"] = "OPEN_ALL_PAYLOAD"
        with self.assertRaisesRegex(validator.ProtocolError, "successor drift"):
            validator.validate_protocol(mutated)

    def test_pre_execution_observed_count_rejected(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["source_object_contract"]["observed_counts"]["aligned_indices"] = 25
        with self.assertRaisesRegex(validator.ProtocolError, "pre-execution observed count drift"):
            validator.validate_protocol(mutated)

    def test_unknown_or_void_cannot_be_negative(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["capability_claim_contract"]["r2_obstacle_boundary_truth_materialized"]["unknown_or_void_is_negative"] = True
        with self.assertRaisesRegex(validator.ProtocolError, "boundary claim drift"):
            validator.validate_protocol(mutated)


if __name__ == "__main__":
    unittest.main()
