import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.activate_depthart_task_preserving_d1_quality import validate


class ActivateD1QualityTest(unittest.TestCase):
    def test_exact_contract_validates_and_authority_drift_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asset = root / "asset.bin"
            asset.write_bytes(b"frozen")
            digest = hashlib.sha256(b"frozen").hexdigest().upper()
            sessions = [{"visit_id": f"p{i}", "video_id": f"s{i}", "frame_count": 300,
                         "frame_stems_sha256": "A" * 64} for i in range(8)]
            protocol = {
                "schema": "blindassist_depthart_task_preserving_d1_quality_protocol_v1",
                "protocol_id": "DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN",
                "status": "FROZEN_PRE_OUTCOME_EXPLICIT_ACTIVATION_REQUIRED",
                "owning_protocol_id": "DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_SCREEN",
                "strict_g4d_terminal_immutable": "CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED",
                "bindings": {"asset": {"path": "asset.bin", "bytes": 6, "sha256": digest}},
                "cohort": {"ordered_sessions": sessions},
                "execution": {"chunk_size_frames": 50, "chunks_per_session": 6, "total_chunks": 48},
                "metric_semantics": {"known_coverage_denominator": "TRUTH_KNOWN_CELLS"},
                "authority": {"r2_cohort_access": False, "performance_before_quality_pass": False},
            }
            self.assertEqual(validate(protocol, root)["frames"], 2400)
            drift = copy.deepcopy(protocol)
            drift["authority"]["r2_cohort_access"] = True
            with self.assertRaisesRegex(ValueError, "firewall"):
                validate(drift, root)


if __name__ == "__main__":
    unittest.main()
