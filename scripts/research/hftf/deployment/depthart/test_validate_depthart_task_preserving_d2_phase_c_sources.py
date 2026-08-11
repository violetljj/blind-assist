import copy
import unittest

from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d2_phase_c_sources import validate_manifest_shape


def fixture() -> tuple[dict, list[dict]]:
    roles = [
        {
            "role": "D2_TRAIN" if index < 4 else "D2_DEVELOPMENT_SEALED",
            "role_order": index % 4 + 1,
            "phase_a_order": index + 1,
            "pool_order": index + 2,
            "visit_id": f"v{index}",
            "video_id": f"s{index}",
            "selected_frame_stems": [f"s{index}_{frame}" for frame in range(300)],
        }
        for index in range(8)
    ]
    manifest = {
        "schema": "blindassist_depthart_task_preserving_d2_phase_c_source_manifest_v1",
        "terminal": "D2_PHASE_C_SOURCE_MATERIALIZATION_PASS_EXACT_EIGHT_SEALED",
        "identity_count": 8,
        "train_identity_count": 4,
        "development_sealed_identity_count": 4,
        "source_asset_count": 32,
        "extracted_file_count": 9600,
        "exact_total_body_bytes": 5281655713,
        "roles": copy.deepcopy(roles),
        "image_decode": False,
        "truth_derivation": False,
        "model_output_read": False,
        "training_executed": False,
        "development_outcome_opened": False,
        "r2_cohort_access": "NONE",
    }
    return manifest, roles


class D2PhaseCSourceValidatorTest(unittest.TestCase):
    def test_valid_shape_passes(self) -> None:
        manifest, roles = fixture()
        validate_manifest_shape(manifest, roles)

    def test_role_change_fails(self) -> None:
        manifest, roles = fixture()
        manifest["roles"][0]["video_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "role binding drift"):
            validate_manifest_shape(manifest, roles)

    def test_decode_or_outcome_fails(self) -> None:
        manifest, roles = fixture()
        manifest["image_decode"] = True
        with self.assertRaisesRegex(ValueError, "decode drift"):
            validate_manifest_shape(manifest, roles)


if __name__ == "__main__":
    unittest.main()
