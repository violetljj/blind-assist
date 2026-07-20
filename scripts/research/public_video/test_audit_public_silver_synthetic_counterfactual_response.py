import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_silver_synthetic_counterfactual_response as audit


class SyntheticCounterfactualResponseAuditTest(unittest.TestCase):
    def test_contract_rejects_unverified_geometry(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            image = root / "x.png"
            image.write_bytes(b"x")
            rows = [{
                "id": "x", "image_path": "x.png", "split": "train", "status": "accepted",
                "objects": [{"class": "fake"}],
                "attributes": {"final_image_sha256": audit.common.sha256_file(image), "counterfactual_pair_id": "p", "synthetic": False},
                "source": {"parent_frame_sha256": "h", "parent_episode_id": "e", "parent_source_id": "s"},
            }]
            spec = {
                "provenance_contract": {"role": "train_only_representation_augmentation", "validation_or_test_use_authorized": False, "pixel_mask_available": False, "pixel_supervision_role": "none"},
                "leakage_contract": {"rice_street_external_pressure_is_trainable": False},
            }
            result = audit.validate_contract(root, spec, rows, {"h": ("e", "s")})
            self.assertFalse(result["passed"])
            self.assertIn("unverified_geometry:x", result["failures"])

    def test_response_fields_are_fixed_and_interpretable(self):
        self.assertEqual(4, len(audit.RESPONSE_FIELDS))
        self.assertIn("median_core_nonwalkable_mean", audit.RESPONSE_FIELDS)
        self.assertIn("median_path_offset_mean", audit.RESPONSE_FIELDS)


if __name__ == "__main__":
    unittest.main()
