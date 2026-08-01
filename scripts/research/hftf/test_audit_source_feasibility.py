from __future__ import annotations

import csv
import gzip
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_source_feasibility import _require_artifacts_output, audit_replay


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceFeasibilityAuditTest(unittest.TestCase):
    def _make_replay(
        self,
        root: Path,
        *,
        frame_count: int = 1,
        valid_pose_binding: bool = False,
        invalid_pose_values: bool = False,
        valid_body_contract: bool = False,
        placeholder_body_contract: bool = False,
        separate_sequences: bool = False,
        corrupt_image_hash: bool = False,
        truncated_png: bool = False,
        missing_group_identity: bool = False,
        absurd_body_geometry: bool = False,
        duplicate_asset_bindings: bool = False,
        string_false_qa: bool = False,
        invalid_metric_scalar_types: bool = False,
        boolean_intrinsics: bool = False,
        tamper_qa: bool = False,
    ) -> Path:
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir()
        (root / "depth").mkdir()
        (root / "source_metadata").mkdir()
        (root / "qa").mkdir()

        pose_path = root / "source_metadata" / "camera_poses.csv"
        pose_columns = [
            "tracking_state",
            "pos_x",
            "pos_y",
            "pos_z",
            "q_x",
            "q_y",
            "q_z",
            "q_w",
        ]
        with pose_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=pose_columns)
            writer.writeheader()
            for index in range(frame_count):
                writer.writerow(
                    {
                        "tracking_state": "TrackingState.READY",
                        "pos_x": index,
                        "pos_y": 0,
                        "pos_z": 0,
                        "q_x": 0,
                        "q_y": 0,
                        "q_z": 0,
                        "q_w": 1,
                    }
                )

        rows: list[dict[str, object]] = []
        depth_summaries: list[dict[str, object]] = []
        for index in range(frame_count):
            image = root / "images" / f"{index:06d}.png"
            mask = root / "masks" / f"{index:06d}.png"
            depth = root / "depth" / f"{index:06d}.float16.gz"
            Image.new("RGB", (2, 2), (0, 0, 0)).save(image)
            Image.new("L", (2, 2), 0).save(mask)
            if truncated_png and index == 0:
                image.write_bytes(
                    b"\x89PNG\r\n\x1a\n"
                    + struct.pack(">I", 13)
                    + b"IHDR"
                    + struct.pack(">II", 2, 2)
                )
            with gzip.open(depth, "wb") as handle:
                handle.write(struct.pack("<6e", 2, 2, 1, 1, 1, 1))

            session_id = f"session-{index}" if separate_sequences else "session-a"
            sequence_id = f"sequence-{index}" if separate_sequences else "sequence-a"
            if missing_group_identity and index == 0:
                session_id = ""
            row = {
                "id": f"fixture-{index}",
                "image_path": f"images/{index:06d}.png",
                "image_sha256": (
                    "0" * 64
                    if corrupt_image_hash and index == 0
                    else _sha256(image)
                ),
                "source_mask_path": f"masks/{index:06d}.png",
                "source_mask_sha256": _sha256(mask),
                "source_depth_path": f"depth/{index:06d}.float16.gz",
                "source_depth_sha256": _sha256(depth),
                "width": 2,
                "height": 2,
                "session_id": session_id,
                "sequence_id": sequence_id,
                "frame_index": 0 if separate_sequences else index,
                "source_frame_index": index,
                "source_timestamp_ms": index * 1000,
                "source_annotation_quality": "SYNTHETIC",
                "event_truth": None,
                "source": {
                    "dataset": "SANPO-Synthetic v0",
                    "official_split": "train",
                    "session_id": session_id,
                },
                "modalities": {
                    "camera_poses": {
                        "path": "source_metadata/camera_poses.csv",
                        "sha256": _sha256(pose_path),
                    }
                },
                "authorization": {"human_event_truth": False},
            }
            rows.append(row)
            depth_summaries.append(
                {
                    "path": depth.name,
                    "finite_positive_fraction": 1.0,
                }
            )

        if duplicate_asset_bindings and frame_count >= 2:
            for key in (
                "image_path",
                "image_sha256",
                "source_mask_path",
                "source_mask_sha256",
                "source_depth_path",
                "source_depth_sha256",
            ):
                rows[1][key] = rows[0][key]
            depth_summaries[1]["path"] = depth_summaries[0]["path"]

        manifest_path = root / "manifest.replay.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        spec: dict[str, object] = {
            "schema": "blindassist_sanpo_synthetic_replay_v1",
            "camera": {
                "fx": True if boolean_intrinsics else 100.0,
                "fy": 100.0,
                "cx": 1.0,
                "cy": 1.0,
                "image_width": 2,
                "image_height": 2,
            },
        }
        if valid_body_contract:
            spec["hftf_body_frame_contract"] = {
                "schema": "blindassist_hftf_body_frame_v1",
                "camera_frame": "camera_chest_left",
                "body_frame": "pedestrian_body",
                "transform_direction": "body_from_camera",
                "position_unit": "meter",
                "quaternion_order": "xyzw",
                "axis_convention": "x_forward_y_left_z_up",
                "camera_to_body": {
                    "translation_m": [
                        0.0,
                        0.0,
                        99.0 if absurd_body_geometry else 1.4,
                    ],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "ground_reference": {
                    "kind": "camera_height_along_body_z",
                    "camera_height_m": 1.4,
                },
                "source_authority": "fixture-authority",
                "verification_method": "fixture-recompute",
            }
        if placeholder_body_contract:
            spec["body_frame"] = {
                "camera_to_body_transform": {"kind": "fixture"},
                "camera_height_m": 1.4,
            }
        if valid_pose_binding or invalid_pose_values:
            binding_path = (
                root / "source_metadata" / "hftf_pose_binding.jsonl"
            )
            binding_rows: list[dict[str, object]] = []
            for index, manifest_row in enumerate(rows):
                invalid = invalid_pose_values and index == 0
                binding_rows.append(
                    {
                        "manifest_id": manifest_row["id"],
                        "session_id": manifest_row["session_id"],
                        "sequence_id": manifest_row["sequence_id"],
                        "source_frame_index": manifest_row[
                            "source_frame_index"
                        ],
                        "source_timestamp_ms": manifest_row[
                            "source_timestamp_ms"
                        ],
                        "raw_pose_row_index": index,
                        "tracking_state": (
                            "TrackingState.LOST"
                            if invalid
                            else "TrackingState.READY"
                        ),
                        "position_m": (
                            [None, 0.0, 0.0]
                            if invalid
                            else [float(index), 0.0, 0.0]
                        ),
                        "quaternion_xyzw": (
                            [0.0, 0.0, 0.0, 0.0]
                            if invalid
                            else [0.0, 0.0, 0.0, 1.0]
                        ),
                    }
                )
            binding_path.write_text(
                "".join(json.dumps(row) + "\n" for row in binding_rows),
                encoding="utf-8",
            )
            spec["hftf_pose_binding"] = {
                "schema": "blindassist_hftf_pose_binding_v1",
                "path": "source_metadata/hftf_pose_binding.jsonl",
                "sha256": _sha256(binding_path),
                "transform_direction": "world_from_camera",
                "position_unit": "meter",
                "time_unit": "millisecond",
                "quaternion_order": "xyzw",
                "camera_frame": "camera_chest_left",
                "world_frame": "sanpo_world",
                "admitted_tracking_states": ["TrackingState.READY"],
                "source_authority": "fixture-authority",
                "binding_method": "fixture-explicit-index",
            }
        (root / "dataset_spec.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )

        metric = {
            "schema": "blindassist_sanpo_synthetic_metric_replay_audit_v1",
            "ok": not tamper_qa,
            "frame_count": (
                999
                if tamper_qa
                else True
                if invalid_metric_scalar_types
                else frame_count
            ),
            "metric_depth_source_integrity": (
                "false" if string_false_qa else True
            ),
            "depth_summaries": depth_summaries,
        }
        if invalid_metric_scalar_types:
            for summary in depth_summaries:
                summary["finite_positive_fraction"] = True
        validation = {
            "ok": "false" if string_false_qa else not tamper_qa,
            "frame_count": 999 if tamper_qa else frame_count,
            "all_rgb_mask_dimensions_match": (
                "false" if string_false_qa else True
            ),
            "required_modalities_hash_bound": (
                "false" if string_false_qa else True
            ),
            "all_frames_official_train_split": (
                "false" if string_false_qa else True
            ),
            "imu_status": "absent_in_fixture",
            "production_authorized": False,
        }
        (root / "qa" / "metric_replay_audit_fixture.json").write_text(
            json.dumps(metric), encoding="utf-8"
        )
        (root / "qa" / "replay_validation.json").write_text(
            json.dumps(validation), encoding="utf-8"
        )
        return root

    def test_current_shape_allows_only_static_projection_canary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(self._make_replay(Path(temp_dir)))

        self.assertEqual(report["terminal"], "HFTF_H0_SOURCE_FEASIBILITY_PARTIAL")
        self.assertEqual(
            report["capability_decisions"][
                "static_metric_geometry_projection_canary"
            ],
            "ELIGIBLE",
        )
        self.assertEqual(
            report["capability_decisions"][
                "multi_height_human_envelope_teacher_canary"
            ],
            "NOT_EVALUABLE",
        )
        self.assertIn(
            "source_specific_pose_time_mapping_verifier_required",
            report["blockers"],
        )

    def test_exact_contracts_are_structural_only_and_do_not_self_admit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir),
                    frame_count=2,
                    valid_pose_binding=True,
                    valid_body_contract=True,
                )
            )

        self.assertEqual(
            report["terminal"],
            "HFTF_H0_SOURCE_FEASIBILITY_PARTIAL",
        )
        self.assertTrue(
            report["checks"]["hftf_pose_binding_contract"]["ok"]
        )
        self.assertFalse(
            report["checks"]["hftf_pose_binding_contract"][
                "trusted_source_mapping_admitted"
            ]
        )
        self.assertTrue(
            report["checks"]["body_frame_contract"]["ok"]
        )
        self.assertFalse(
            report["checks"]["body_frame_contract"][
                "trusted_calibration_admitted"
            ]
        )
        self.assertTrue(
            report["checks"][
                "future_mechanics_structure_ready_but_not_admitted"
            ]
        )
        self.assertEqual(
            report["capability_decisions"][
                "multi_height_human_envelope_teacher_canary"
            ],
            "NOT_EVALUABLE",
        )
        self.assertEqual(
            report["capability_decisions"][
                "short_horizon_future_teacher_canary"
            ],
            "NOT_EVALUABLE",
        )
        self.assertEqual(
            report["capability_decisions"]["student_effect_evaluation"],
            "NOT_EVALUABLE",
        )
        self.assertEqual(
            report["allowed_next_step"],
            "static_metric_geometry_projection_canary_only",
        )

    def test_invalid_pose_values_block_future_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir),
                    frame_count=2,
                    invalid_pose_values=True,
                    valid_body_contract=True,
                )
            )

        self.assertFalse(
            report["checks"]["hftf_pose_binding_contract"]["ok"]
        )
        self.assertEqual(
            report["capability_decisions"][
                "short_horizon_future_teacher_canary"
            ],
            "NOT_EVALUABLE",
        )
        self.assertEqual(
            report["allowed_next_step"],
            "static_metric_geometry_projection_canary_only",
        )

    def test_placeholder_body_contract_does_not_admit_multi_height(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir),
                    frame_count=2,
                    valid_pose_binding=True,
                    placeholder_body_contract=True,
                )
            )

        self.assertFalse(report["checks"]["body_frame_contract"]["ok"])
        self.assertEqual(
            report["capability_decisions"][
                "multi_height_human_envelope_teacher_canary"
            ],
            "NOT_EVALUABLE",
        )

    def test_two_single_frame_sequences_do_not_create_a_future_span(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir),
                    frame_count=2,
                    valid_pose_binding=True,
                    valid_body_contract=True,
                    separate_sequences=True,
                )
            )

        self.assertEqual(
            report["source"]["maximum_single_sequence_span_ms"], 0
        )
        self.assertEqual(
            report["capability_decisions"][
                "short_horizon_future_teacher_canary"
            ],
            "NOT_EVALUABLE",
        )

    def test_self_reported_effect_contract_cannot_upgrade_h0(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_replay(
                Path(temp_dir),
                frame_count=2,
                valid_pose_binding=True,
                valid_body_contract=True,
            )
            spec_path = root / "dataset_spec.json"
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["hftf_evaluation_contract"] = {
                "positive_parent_event_count": 999,
                "negative_parent_event_count": 999,
                "parent_session_split": True,
            }
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            report = audit_replay(root)

        self.assertEqual(
            report["terminal"],
            "HFTF_H0_SOURCE_FEASIBILITY_PARTIAL",
        )
        self.assertEqual(
            report["capability_decisions"]["student_effect_evaluation"],
            "NOT_EVALUABLE",
        )

    def test_hash_mismatch_fails_source_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir), corrupt_image_hash=True
                )
            )

        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )
        self.assertFalse(report["checks"]["bound_files"]["ok"])

    def test_camera_dimensions_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self._make_replay(Path(temp_dir))
            spec_path = root / "dataset_spec.json"
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["camera"]["image_width"] = 3
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            report = audit_replay(root)

        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )
        self.assertFalse(report["checks"]["camera_dimensions_match_manifest"])

    def test_qa_ok_and_frame_count_are_not_trusted_when_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(Path(temp_dir), tamper_qa=True)
            )

        self.assertFalse(report["checks"]["metric_qa"]["ok"])
        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )

    def test_string_false_qa_fields_do_not_pass_truthiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(Path(temp_dir), string_false_qa=True)
            )

        self.assertFalse(report["checks"]["metric_qa"]["ok"])
        self.assertFalse(
            report["checks"]["metric_qa"][
                "replay_validation_shape_valid"
            ]
        )
        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )

    def test_boolean_metric_scalars_are_not_numeric_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir), invalid_metric_scalar_types=True
                )
            )

        qa = report["checks"]["metric_qa"]
        self.assertFalse(qa["metric_ok_and_frame_count_valid"])
        self.assertFalse(qa["finite_positive_depth"])
        self.assertFalse(qa["ok"])
        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )

    def test_boolean_intrinsics_are_not_numeric_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(Path(temp_dir), boolean_intrinsics=True)
            )

        self.assertFalse(report["checks"]["intrinsics_valid"])
        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )

    def test_duplicate_asset_bindings_cannot_inflate_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir),
                    frame_count=2,
                    duplicate_asset_bindings=True,
                )
            )

        bound_files = report["checks"]["bound_files"]
        self.assertFalse(bound_files["ok"])
        self.assertTrue(bound_files["duplicate_canonical_paths"])
        self.assertTrue(
            bound_files["duplicate_observation_hash_triplets"]
        )
        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )

    def test_missing_session_identity_cannot_be_silently_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir), missing_group_identity=True
                )
            )

        self.assertFalse(
            report["checks"][
                "all_manifest_rows_have_nonempty_session_and_sequence"
            ]
        )
        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )

    def test_truncated_png_fails_full_decode_even_when_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(Path(temp_dir), truncated_png=True)
            )

        self.assertFalse(report["checks"]["bound_files"]["ok"])
        self.assertTrue(
            report["checks"]["bound_files"]["png_shape_mismatches"]
        )
        self.assertEqual(
            report["terminal"], "HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE"
        )

    def test_absurd_body_geometry_fails_structural_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_replay(
                self._make_replay(
                    Path(temp_dir),
                    valid_body_contract=True,
                    absurd_body_geometry=True,
                )
            )

        body = report["checks"]["body_frame_contract"]
        self.assertFalse(body["geometry_internal_consistency_valid"])
        self.assertFalse(body["ok"])
        self.assertEqual(
            report["capability_decisions"][
                "multi_height_human_envelope_teacher_canary"
            ],
            "NOT_EVALUABLE",
        )

    def test_cli_output_scope_rejects_non_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                _require_artifacts_output(Path(temp_dir) / "report.json")


if __name__ == "__main__":
    unittest.main()
