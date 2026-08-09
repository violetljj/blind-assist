import tempfile
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.audit_b0_data_capability import audit


def session(source_id: str, *, aligned: bool = True, readable: bool = True) -> dict:
    return {
        "source_id": source_id,
        "dataset": "Fixture",
        "split": "unspecified",
        "session_kind": "media_session",
        "counts": {"rgb_count": 2, "depth_count": 2, "pose_count": 0},
        "decodability": {"status": "all_profiled_readable" if readable else "corrupt"},
        "corrupt_frames": [],
        "hash_errors": [],
        "rgb_mask_depth_pose_alignment": {
            "status": "aligned_by_frame_key" if aligned else "partial_or_misaligned_by_frame_key"
        },
        "role_flags": {
            "is_consumed": None,
            "is_burned": None,
            "is_fresh": None,
            "is_reserved": None,
        },
    }


class AuditB0DataCapabilityTest(unittest.TestCase):
    def test_structural_profile_does_not_promote_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.json"
            ledger_path.write_text("{}", encoding="utf-8")
            result = audit(
                {
                    "schema_version": "dataset-master-ledger-v1",
                    "generated_at": "fixture",
                    "sessions": [session("A"), session("B", aligned=False), session("C", readable=False)],
                },
                ledger_path,
            )
        self.assertEqual(result["profile"]["rgb_depth_sessions"], 3)
        self.assertEqual(result["profile"]["structurally_eligible_rgb_depth_sessions"], 1)
        self.assertEqual(result["status"], "STRUCTURAL_CANDIDATES_FOUND_ROSTER_ADMISSION_NOT_AUTHORIZED")
        self.assertIn("license scope for Assistive Geometry B0", result["structural_eligibility_rule"]["does_not_prove"])

    def test_wrong_schema_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = Path(directory) / "ledger.json"
            ledger_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                audit({"schema_version": "wrong", "sessions": []}, ledger_path)


if __name__ == "__main__":
    unittest.main()
