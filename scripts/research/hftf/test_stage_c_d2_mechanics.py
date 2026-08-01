from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_stage_c_d2_transport_effect as evaluator  # noqa: E402
import preprocess_stage_c_d2_future_blind as preprocessor  # noqa: E402
from stage_c_d2_mechanics_common import (  # noqa: E402
    ANCHORS,
    arrays_from_arm,
    compute_field,
    field_parameters,
    nullable_field,
    predicted_bases,
    validate_acquired_source_binding,
    validate_cross_parent_bindings,
)


class StageCD2MechanicsTest(unittest.TestCase):
    def test_frame_index_rejects_embedded_pose_content(self) -> None:
        frames = [
            {
                "normalized_index": index,
                "source_frame_index": index * 4,
                "manifest_id": str(index),
                "pose_slice": {"path": "pose", "sha256": "a" * 64},
                "depth": {"path": "depth", "sha256": "b" * 64},
                "mask": {"path": "mask", "sha256": "c" * 64},
            }
            for index in range(13)
        ]
        self.assertEqual(
            list(preprocessor._frame_map({"frames": frames})),
            list(range(13)),
        )
        frames[8]["position_m"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "embeds pose"):
            preprocessor._frame_map({"frames": frames})

    def test_acquired_camera_must_match_qualified_parent(self) -> None:
        frames = [
            {
                "normalized_index": index,
                "source_frame_index": index,
            }
            for index in range(13)
        ]
        camera = {
            "fx": 100.0,
            "fy": 101.0,
            "cx": 50.0,
            "cy": 51.0,
            "image_width": 100,
            "image_height": 102,
        }
        qualified = {
            "selected_source_frames": list(range(13)),
            "camera": camera,
        }
        source = {"frames": frames, "camera": dict(camera)}
        validate_acquired_source_binding(qualified, source)
        source["camera"]["fx"] = 99.0
        with self.assertRaisesRegex(ValueError, "camera binding"):
            validate_acquired_source_binding(qualified, source)

    def test_cross_parent_hashes_reject_mixed_valid_parents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata_path = root / "metadata.json"
            mechanics_path = root / "mechanics.json"
            metadata_path.write_text("metadata\n", encoding="utf-8")
            mechanics_path.write_text("mechanics\n", encoding="utf-8")

            def digest(path: Path) -> str:
                import hashlib

                return hashlib.sha256(path.read_bytes()).hexdigest()

            clarification = {
                "parents": {
                    "metadata_qualification_result": {
                        "sha256": digest(metadata_path)
                    }
                }
            }
            g0 = {
                "parents": {
                    "swept_envelope_mechanics": {
                        "sha256": digest(mechanics_path)
                    }
                }
            }
            validate_cross_parent_bindings(
                clarification,
                metadata_path,
                g0,
                mechanics_path,
            )
            clarification["parents"]["metadata_qualification_result"][
                "sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(ValueError, "cross-parent"):
                validate_cross_parent_bindings(
                    clarification,
                    metadata_path,
                    g0,
                    mechanics_path,
                )

    def test_d2_1_yaw_sign_and_tangent_prediction(self) -> None:
        history = {
            "position_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        current = {
            "position_m": [0.4, 0.0, 0.0],
            "quaternion_xyzw": [
                0.0,
                math.sin(math.pi / 4.0),
                0.0,
                math.cos(math.pi / 4.0),
            ],
        }
        plane = {
            "camera_ground_projection_m": [0.4, 0.0, 0.0],
            "normal_toward_camera": [0.0, 1.0, 0.0],
        }
        _, predicted, receipt = predicted_bases(history, current, plane)
        self.assertAlmostEqual(
            float(receipt["yaw_delta_rad"]),
            math.pi / 2.0,
        )
        np.testing.assert_allclose(
            predicted[0.4][0],
            [0.8, 0.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            predicted[0.4][1],
            [0.0, 0.0, -1.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            predicted[0.8][1],
            [-1.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_all_points_participate_in_second_order_clearance(self) -> None:
        parameters = {
            "theta_edges": np.asarray([-0.1, 0.1]),
            "distance_edges": np.asarray([0.0, 2.0]),
            "height_bands": [(0.35, 1.35), (1.35, 2.05)],
            "widths": np.asarray([0.4, 0.3]),
            "order_statistic": 2,
            "final_edge_atol_m": 1e-12,
            "final_edge_rtol": 0.0,
            "clip_min_m": -0.5,
            "clip_max_m": 1.0,
        }
        basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        outside = np.asarray(
            [[0.5, 0.5], [0.41, 0.42], [0.5, 0.5]],
            dtype=np.float64,
        )
        clearance = compute_field(outside, basis, parameters)
        self.assertGreater(float(clearance[0, 0, 0]), 0.0)
        self.assertLess(float(clearance[0, 0, 0]), 0.03)
        one_point = compute_field(outside[:, :1], basis, parameters)
        self.assertEqual(float(one_point[0, 0, 0]), 1.0)

    def test_unknown_is_null_and_cannot_become_numeric(self) -> None:
        known = np.zeros((2, 6, 6), dtype=bool)
        known[0, 0, 0] = True
        clearance = np.full((2, 6, 6), 0.25)
        arm = {
            "probe_pass_counts": np.where(known, 5, 0).tolist(),
            "known": known.tolist(),
            "clearance_m": nullable_field(known, clearance),
        }
        parsed_known, parsed_clearance = arrays_from_arm(arm)
        self.assertTrue(parsed_known[0, 0, 0])
        self.assertTrue(np.isnan(parsed_clearance[1, 0, 0]))
        arm["clearance_m"][1][0][0] = 1.0
        with self.assertRaisesRegex(ValueError, "UNKNOWN cell"):
            arrays_from_arm(arm)

    def test_synthetic_preprocessor_repeat_is_byte_deterministic(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        g0 = json.loads(
            (
                repo_root
                / "docs/research/hftf/"
                "HFTF_STAGE_C_SIGNED_CLEARANCE_CURRENT_BRIDGE_G0_"
                "2026-08-01.json"
            ).read_text(encoding="utf-8")
        )
        mechanics = json.loads(
            (
                repo_root
                / "docs/research/hftf/"
                "HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_"
                "CANARY_D0_2026-08-01.json"
            ).read_text(encoding="utf-8")
        )
        parameters = field_parameters(g0, mechanics)
        frames = [
            {
                "normalized_index": index,
                "source_frame_index": index,
                "manifest_id": str(index),
                "pose_slice": {"path": "pose", "sha256": "a" * 64},
                "depth": {"path": "depth", "sha256": "b" * 64},
                "mask": {"path": "mask", "sha256": "c" * 64},
            }
            for index in range(13)
        ]
        source = {
            "session_id": "synthetic-determinism-canary",
            "camera": {
                "image_width": 128,
                "image_height": 128,
                "fx": 100.0,
                "fy": 100.0,
                "cx": 64.0,
                "cy": 64.0,
            },
            "frames": frames,
        }
        depth = np.full((128, 128), np.nan, dtype=np.float32)
        for row in range(65, 128):
            depth[row, :] = 150.0 / (row - 64)
        semantic = np.ones((128, 128), dtype=np.uint16)
        history = {
            "position_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        current = {
            "position_m": [0.1, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        first, first_points = preprocessor.preprocess_anchor(
            source,
            2,
            history,
            current,
            depth,
            semantic,
            parameters,
        )
        second, second_points = preprocessor.preprocess_anchor(
            source,
            2,
            history,
            current,
            depth,
            semantic,
            parameters,
        )
        first_json = (
            json.dumps(first, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        second_json = (
            json.dumps(second, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        first_npy = io.BytesIO()
        second_npy = io.BytesIO()
        np.save(
            first_npy,
            first_points.astype("<f8", copy=False),
            allow_pickle=False,
        )
        np.save(
            second_npy,
            second_points.astype("<f8", copy=False),
            allow_pickle=False,
        )
        self.assertEqual(first_json, second_json)
        self.assertEqual(first_npy.getvalue(), second_npy.getvalue())

    def test_preprocessor_attempt_is_fsynced_before_first_input_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract_path = root / "contract.json"
            contract_path.write_text("{}\n", encoding="utf-8")
            output_root = root / "predictions"
            frames = [
                {
                    "normalized_index": index,
                    "source_frame_index": index,
                    "manifest_id": str(index),
                    "pose_slice": {"path": "pose", "sha256": "a" * 64},
                    "depth": {"path": "depth", "sha256": "b" * 64},
                    "mask": {"path": "mask", "sha256": "c" * 64},
                }
                for index in range(13)
            ]
            context = {
                "contract": {
                    "authorization": {
                        "future_blind_preprocessor_execution_authorized": True,
                        "future_truth_open_authorized_before_completion": False,
                    },
                    "canonical_artifacts": {
                        "future_blind_prediction_root": str(output_root)
                    },
                },
                "g0": {},
                "mechanics": {},
                "source_index_path": root / "index.json",
                "sources": [
                    {
                        "session_id": "source",
                        "camera": {},
                        "frames": frames,
                    }
                ],
            }
            events: list[str] = []

            def fake_fsync(_descriptor: int) -> None:
                events.append("attempt_fsynced")

            def first_input_read(*_args: object) -> dict[str, object]:
                self.assertTrue((output_root / "attempt.json").is_file())
                self.assertEqual(["attempt_fsynced"], events)
                events.append("first_pose_or_media_read")
                raise RuntimeError("stop after ordering assertion")

            with (
                mock.patch.object(
                    preprocessor, "load_context", return_value=context
                ),
                mock.patch.object(
                    preprocessor, "field_parameters", return_value={}
                ),
                mock.patch.object(
                    preprocessor, "_load_pose", side_effect=first_input_read
                ),
                mock.patch.object(
                    preprocessor.os, "fsync", side_effect=fake_fsync
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "ordering assertion"
                ):
                    preprocessor.run(contract_path, output_root)
            attempt = json.loads(
                (output_root / "attempt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                preprocessor.ATTEMPT_STATUS,
                attempt["status"],
            )
            self.assertEqual(
                ["attempt_fsynced", "first_pose_or_media_read"],
                events,
            )

    def test_truth_join_receipt_is_fsynced_before_first_future_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract_path = root / "contract.json"
            contract_path.write_text("{}\n", encoding="utf-8")
            output_path = root / "effect.json"
            truth_receipt = root / "truth-join-once.json"
            prediction_root = root / "predictions"
            prediction_root.mkdir()
            (prediction_root / "completion.json").write_text(
                "{}\n", encoding="utf-8"
            )
            source_id = "source"
            frames = [
                {
                    "normalized_index": index,
                    "source_frame_index": index,
                    "manifest_id": str(index),
                    "pose_slice": {"path": "pose", "sha256": "a" * 64},
                    "depth": {"path": "depth", "sha256": "b" * 64},
                    "mask": {"path": "mask", "sha256": "c" * 64},
                }
                for index in range(13)
            ]
            context = {
                "contract": {
                    "authorization": {
                        "truth_effect_execution_after_completion_authorized": True,
                        "second_truth_join_authorized": False,
                    },
                    "canonical_artifacts": {
                        "effect_result": str(output_path),
                        "truth_join_once_receipt": str(truth_receipt),
                        "future_blind_prediction_root": str(prediction_root),
                        "truth_effect_failure": str(
                            root / "failure.json"
                        ),
                    },
                },
                "g0": {},
                "mechanics": {},
                "source_index_path": root / "index.json",
                "sources": [
                    {
                        "session_id": source_id,
                        "camera": {},
                        "frames": frames,
                    }
                ],
            }
            predictions = {
                (source_id, 2): {
                    "horizons": [
                        {"horizon_s": 0.4},
                        {"horizon_s": 0.8},
                    ]
                }
            }
            events: list[str] = []

            def fake_fsync(_descriptor: int) -> None:
                events.append("truth_receipt_fsynced")

            def first_future_read(*_args: object) -> dict[str, object]:
                self.assertTrue(truth_receipt.is_file())
                self.assertEqual(["truth_receipt_fsynced"], events)
                events.append("first_future_pose_or_media_read")
                raise RuntimeError("stop after truth ordering assertion")

            with (
                mock.patch.object(
                    evaluator, "load_context", return_value=context
                ),
                mock.patch.object(
                    evaluator,
                    "_load_completion",
                    return_value=({}, predictions),
                ),
                mock.patch.object(
                    evaluator, "field_parameters", return_value={}
                ),
                mock.patch.object(
                    evaluator, "_load_pose", side_effect=first_future_read
                ),
                mock.patch.object(
                    preprocessor.os, "fsync", side_effect=fake_fsync
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "truth ordering assertion"
                ):
                    evaluator.run(contract_path, output_path)
            receipt = json.loads(
                truth_receipt.read_text(encoding="utf-8")
            )
            self.assertEqual(
                "D2_TRUTH_JOIN_STARTED_NO_SECOND_JOIN_AUTHORIZED",
                receipt["status"],
            )
            self.assertEqual(
                [
                    "truth_receipt_fsynced",
                    "first_future_pose_or_media_read",
                ],
                events,
            )

    def test_supported_terminal_only_authorizes_contract_freeze(
        self,
    ) -> None:
        authorization = evaluator.result_authorization(
            evaluator.SUPPORTED
        )
        self.assertTrue(
            authorization["freeze_rgb_student_contract_authorized"]
        )
        for key in (
            "rgb_student_training_authorized",
            "rgb_student_execution_authorized",
            "reserved_official_test_open_authorized",
            "research_mainline_changed",
            "default_app_changed",
            "android_changed",
            "production_authorized",
            "safety_claim_authorized",
        ):
            self.assertFalse(authorization[key])

    def test_prior_pretruth_failure_blocks_second_future_join(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract_path = root / "contract.json"
            contract_path.write_text("{}\n", encoding="utf-8")
            failure_path = root / "failure.json"
            failure_path.write_text(
                json.dumps(
                    {
                        "schema": evaluator.FAILURE_SCHEMA,
                        "terminal": (
                            evaluator.PRETRUTH_FAILURE_TERMINAL
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            context = {
                "contract": {
                    "authorization": {
                        "truth_effect_execution_after_completion_authorized": True,
                        "second_truth_join_authorized": False,
                    },
                    "canonical_artifacts": {
                        "truth_effect_failure": str(failure_path)
                    },
                }
            }
            with (
                mock.patch.object(
                    evaluator, "load_context", return_value=context
                ),
                mock.patch.object(evaluator, "_load_completion") as completion,
                mock.patch.object(evaluator, "_load_pose") as pose,
                mock.patch.object(evaluator, "_load_current") as current,
            ):
                with self.assertRaisesRegex(
                    FileExistsError, "failure already sealed"
                ):
                    evaluator.run(contract_path, root / "result.json")
            completion.assert_not_called()
            pose.assert_not_called()
            current.assert_not_called()

    def test_effect_summary_supports_only_when_all_gates_pass(self) -> None:
        source_ids = [f"source-{index}" for index in range(6)]
        observations = []
        known = np.ones((2, 6, 6), dtype=bool)
        for source_id in source_ids:
            for anchor in ANCHORS:
                for horizon in (0.4, 0.8):
                    truth = np.full((2, 6, 6), 0.2)
                    truth[:, 0, 0] = -0.2
                    persistence = truth + 0.1
                    persistence[:, 0, 0] = 0.3
                    observations.append(
                        {
                            "session_id": source_id,
                            "anchor_normalized_index": anchor,
                            "horizon_s": horizon,
                            "truth_known": known,
                            "truth_clearance": truth,
                            "arms": {
                                evaluator.PERSISTENCE: (
                                    known,
                                    persistence,
                                ),
                                evaluator.ADVECTED: (known, truth.copy()),
                            },
                            "unknown_to_safe_violations": 0,
                        }
                    )
        result = evaluator.summarize_observations(
            source_ids,
            observations,
        )
        self.assertTrue(result["opportunity_adequate"])
        self.assertTrue(result["all_effect_gates_passed"])
        self.assertEqual(
            sum(row["passed"] for row in result["opportunity_strata"]),
            24,
        )
        self.assertGreaterEqual(result["relative_mae_reduction"], 0.1)
        self.assertGreaterEqual(result["absolute_mae_reduction_m"], 0.03)

    def test_opportunity_failure_stops_before_effect_metrics(self) -> None:
        source_ids = [f"source-{index}" for index in range(6)]
        unknown = np.zeros((2, 6, 6), dtype=bool)
        nan = np.full((2, 6, 6), np.nan)
        observations = [
            {
                "session_id": source_id,
                "horizon_s": horizon,
                "truth_known": unknown,
                "truth_clearance": nan,
                "arms": {
                    evaluator.PERSISTENCE: (unknown, nan),
                    evaluator.ADVECTED: (unknown, nan),
                },
                "unknown_to_safe_violations": 0,
            }
            for source_id in source_ids
            for _anchor in ANCHORS
            for horizon in (0.4, 0.8)
        ]
        result = evaluator.summarize_observations(
            source_ids,
            observations,
        )
        self.assertFalse(result["opportunity_adequate"])
        self.assertNotIn("effect_gates", result)
        self.assertNotIn("six_source_macro_mae_m", result)


if __name__ == "__main__":
    unittest.main()
