from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_stage_c_f0_1_sanpo_authority import _validate_authority


def _report(source_id: str, acquired: dict[str, str]) -> dict:
    frames = list(range(0, 50, 2))
    return {
        "schema": "blindassist_hftf_sanpo_pose_geometry_authority_r0",
        "terminal": "HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED",
        "evaluation_mode": "frozen_canonical_replication",
        "claim_ceiling": "SOURCE_SPECIFIC_GEOMETRY_PROXY_ONLY",
        "source_session_ids": [source_id],
        "manifest_frame_count": 25,
        "mainline_changed": False,
        "default_app_changed": False,
        "allowed_next_step": "H0_2_COHORT_AGGREGATION",
        "input_hashes": {
            "dataset_spec_sha256": acquired["dataset_spec_sha256"],
            "manifest_sha256": acquired["manifest_sha256"],
            "camera_poses_sha256": acquired["camera_poses_sha256"],
        },
        "official_loader_authority": {
            "ok": True,
            "expected_markers_present": True,
            "clean_tracked_tree": True,
        },
        "source_pose_authority": {
            "ok": True,
            "gcs_description_authenticated": True,
            "gcs_camera_poses_authenticated": True,
            "binding_count": 25,
            "bindings": [
                {
                    "source_frame_index": frame,
                    "raw_pose_row_index": frame,
                    "tracking_state": "TrackingState.READY",
                    "ok": True,
                }
                for frame in frames
            ],
        },
        "transform_direction_canary": {
            "ok": True,
            "frozen_canonical_replication_admitted": True,
            "frozen_canonical_rank": 1,
            "admitted_semantics": (
                "p_world = R_xyzw @ p_opencv_camera + camera_translation_m"
            ),
            "frozen_canonical_hypothesis": {
                "orientation_hypothesis": "R",
                "camera_basis_rows": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "best": {"coverage": 0.7, "median_relative_depth_error": 0.001},
        },
        "ground_and_body_proxy_canary": {
            "ok": True,
            "frame_count_with_ground": 25,
            "local_ground_plane_frame_count": 25,
            "vertical_axis": "+Z",
            "standard_body_proxy_frame_admitted_for_h1": True,
            "physical_camera_to_body_calibration_admitted": False,
            "chosen_axis": {"median_axis_alignment": 0.999},
        },
        "capability_decisions": {
            "standard_body_proxy_for_h1_geometry_mechanics": "ELIGIBLE",
            "physical_camera_to_person_calibration": "NOT_EVALUABLE",
            "student_or_event_effect": "NOT_EVALUABLE",
        },
    }


class StageCF01SanpoAuthorityAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "role": "heldout",
            "official_split": "test",
            "session_id": "a" * 64,
            "selected_source_frames": list(range(0, 50, 2)),
        }
        self.acquired = {
            "dataset_spec_sha256": "spec",
            "manifest_sha256": "manifest",
            "camera_poses_sha256": "poses",
        }

    def test_exact_authority_contract_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            path.write_text(
                json.dumps(_report("a" * 64, self.acquired)),
                encoding="utf-8",
            )
            result = _validate_authority(self.source, self.acquired, path)
            self.assertTrue(result["ok"])
            self.assertEqual([], result["errors"])

    def test_hash_substitution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            report = _report("a" * 64, self.acquired)
            report["input_hashes"]["manifest_sha256"] = "substitute"
            path.write_text(json.dumps(report), encoding="utf-8")
            result = _validate_authority(self.source, self.acquired, path)
            self.assertFalse(result["ok"])
            self.assertIn(
                "acquisition_hash_binding_mismatch", result["errors"]
            )

    def test_noncausal_pose_index_substitution_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            report = _report("a" * 64, self.acquired)
            report["source_pose_authority"]["bindings"][2][
                "raw_pose_row_index"
            ] = 3
            path.write_text(json.dumps(report), encoding="utf-8")
            result = _validate_authority(self.source, self.acquired, path)
            self.assertFalse(result["ok"])
            self.assertIn("source_pose_authority_failed", result["errors"])

    def test_physical_calibration_claim_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authority.json"
            report = _report("a" * 64, self.acquired)
            report["ground_and_body_proxy_canary"][
                "physical_camera_to_body_calibration_admitted"
            ] = True
            path.write_text(json.dumps(report), encoding="utf-8")
            result = _validate_authority(self.source, self.acquired, path)
            self.assertFalse(result["ok"])
            self.assertIn("ground_proxy_authority_failed", result["errors"])


if __name__ == "__main__":
    unittest.main()
