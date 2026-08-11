import copy
import unittest

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2r1 import CHECKPOINT_SCHEMA
from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d2r1_result import validate_manifest


def fixture() -> tuple[dict, dict]:
    thresholds = {
        "minimum_truth_known_cells_per_identity": 1,
        "minimum_truth_clear_cells_per_identity": 1,
        "minimum_truth_occupied_cells_per_identity": 1,
        "minimum_truth_known_cells_per_band_horizon": 1,
        "minimum_valid_band_clearances_per_identity": 1,
    }
    videos = []
    selected = []
    for index in range(16):
        identity = {"pool_order": index + 1, "visit_id": f"v{index}", "video_id": f"s{index}", "fold": "Training"}
        selected.append(identity)
        videos.append(
            {
                "schema": CHECKPOINT_SCHEMA,
                "phase_a_order": index + 1,
                **identity,
                "qualified": True,
                "qualification_failures": [],
                "selected_frame_stems": [f"s{index}_{frame / 10:.1f}" for frame in range(300)],
                "truth_support": {
                    "known_cells": 2,
                    "clear_cells": 1,
                    "occupied_cells": 1,
                    "valid_band_clearances": 1,
                    "known_by_grid": {f"{band}@{horizon:.1f}m": 1 for band in ("left", "center", "right") for horizon in (1.0, 1.5, 2.0)},
                },
                "windows_tested": 1,
                "decoded_frame_count": 300,
                "per_frame_truth_rows_saved": False,
                "rgb_read": False,
                "model_output_read": False,
                "r2_cohort_access": "NONE",
            }
        )
    from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d2r1 import role_assignments
    manifest = {
        "schema": "blindassist_depthart_task_preserving_d2r1_manifest_v1",
        "terminal": "D2R1_SOURCE_SUPPORT_PASS_4_TRAIN_4_DEVELOPMENT_ROLES_LOCKED",
        "identity_count": 16,
        "qualified_identity_count": 16,
        "videos": videos,
        "truth_support_thresholds": thresholds,
        "role_assignments": role_assignments(videos),
        "per_frame_truth_rows_saved": False,
        "rgb_read": False,
        "model_output_read": False,
        "r2_cohort_access": "NONE",
    }
    return manifest, {"selected_phase_b": selected}


class D2R1ResultValidatorTest(unittest.TestCase):
    def test_valid_fixture_passes(self) -> None:
        manifest, phase_a = fixture()
        self.assertEqual(8, validate_manifest(manifest, phase_a)["role_count"])

    def test_partial_or_changed_role_fails(self) -> None:
        manifest, phase_a = fixture()
        manifest = copy.deepcopy(manifest)
        manifest["role_assignments"] = manifest["role_assignments"][:7]
        with self.assertRaisesRegex(ValueError, "role assignment drift"):
            validate_manifest(manifest, phase_a)

    def test_outcome_access_fails(self) -> None:
        manifest, phase_a = fixture()
        manifest = copy.deepcopy(manifest)
        manifest["model_output_read"] = True
        with self.assertRaisesRegex(ValueError, "outcome access drift"):
            validate_manifest(manifest, phase_a)


if __name__ == "__main__":
    unittest.main()
