from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_stage_c_d3_q0_sealed_effect as evaluator  # noqa: E402
import preprocess_stage_c_d3_q0_selected_future_blind as preprocessor  # noqa: E402
import run_stage_c_d3_q0_next_slot as runner  # noqa: E402
import stage_c_d3_q0_common as common  # noqa: E402


CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "research"
    / "hftf"
    / "HFTF_STAGE_C_D3_Q0_1_SCHEMA_REPAIR_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json"
)


def receipt(name: str, frame_index: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "name": name,
        "generation": "11",
        "size": 7,
        "md5_base64": "YWJj",
    }
    if frame_index is not None:
        value["frame_index"] = frame_index
    return value


def source() -> dict[str, object]:
    session_id = "a" * 64
    frames = list(range(13))
    return {
        "session_id": session_id,
        "d3_roster_slot_index": 2,
        "metadata_eligible_rank": 2,
        "description_object": {"session": "synthetic"},
        "source_fps": 5.0,
        "selected_source_frames": frames,
        "camera_pose_object_receipt": receipt("pose.csv"),
        "camera": {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 50.0,
            "cy": 50.0,
            "image_width": 100,
            "image_height": 100,
        },
        "media_object_listing_receipts": {
            modality: {
                "required_frame_receipts": [
                    receipt(f"{modality}/{index}", index)
                    for index in frames
                ]
            }
            for modality in ("rgb", "mask", "depth")
        },
    }


def observation(
    anchor: int,
    horizon: float,
    *,
    unknown_violations: int = 0,
) -> dict[str, object]:
    known = np.ones((2, 6, 6), dtype=bool)
    counts = np.full((2, 6, 6), 9, dtype=int)
    truth = np.ones((2, 6, 6), dtype=float)
    truth[:, 0, 0] = -0.1
    return {
        "anchor_normalized_index": anchor,
        "horizon_s": horizon,
        "future_normalized_index": (
            anchor + (2 if horizon == 0.4 else 4)
        ),
        "predicted_basis": {},
        "support": {
            arm: {
                "probe_pass_counts": counts.tolist(),
                "known": known.tolist(),
            }
            for arm in runner.ARM_NAMES
        },
        "truth": {
            "probe_pass_counts": counts.tolist(),
            "known": known.tolist(),
            "signed_clearance_m": truth.tolist(),
        },
        "unknown_to_safe_violations": unknown_violations,
    }


class StageCD3Q0PipelineTest(unittest.TestCase):
    def test_download_plan_is_pose_one_depth_mask_2_through_12_rgb_zero(
        self,
    ) -> None:
        plan = runner.planned_downloads(source())
        self.assertEqual(sum(row["kind"] == "pose" for row in plan), 1)
        self.assertEqual(sum(row["kind"] == "depth" for row in plan), 11)
        self.assertEqual(sum(row["kind"] == "mask" for row in plan), 11)
        self.assertEqual(sum(row["kind"] == "rgb" for row in plan), 0)
        self.assertEqual(
            [
                row["normalized_index"]
                for row in plan
                if row["kind"] == "depth"
            ],
            list(range(2, 13)),
        )

    def test_qualification_four_strata_exact_denominator_and_pass(
        self,
    ) -> None:
        observations = [
            observation(anchor, horizon)
            for anchor in range(2, 9)
            for horizon in (0.4, 0.8)
        ]
        summary = runner.summarize_qualification(observations)
        self.assertTrue(summary["qualified"])
        self.assertEqual(len(summary["strata"]), 4)
        for row in summary["strata"]:
            self.assertEqual(row["denominator"], 252)
            self.assertEqual(row["common_known_count"], 252)
            self.assertEqual(row["known_risk_count"], 7)
            self.assertEqual(row["known_safe_count"], 245)
            self.assertTrue(row["passed"])

    def test_unknown_violation_counted_once_and_fails(self) -> None:
        observations = [
            observation(
                anchor,
                horizon,
                unknown_violations=(
                    1 if (anchor, horizon) == (2, 0.4) else 0
                ),
            )
            for anchor in range(2, 9)
            for horizon in (0.4, 0.8)
        ]
        summary = runner.summarize_qualification(observations)
        self.assertEqual(summary["unknown_to_safe_violations"], 1)
        self.assertFalse(summary["qualified"])

    def test_payload_is_published_before_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            layout = {
                "sealed_payload": root / "sealed.json",
                "selector": root / "selector.json",
            }
            events: list[str] = []
            payload = {"schema": "sealed"}
            builder_receipts: list[str] = []

            def writer(path: Path, value: dict[str, object]) -> None:
                del value
                events.append(path.name)

            def builder(digest: str) -> dict[str, object]:
                builder_receipts.append(digest)
                return {"schema": "selector"}

            with mock.patch.object(
                runner,
                "sha256",
                side_effect=AssertionError(
                    "selector must not reopen sealed payload"
                ),
            ):
                runner.publish_payload_then_selector(
                    payload,
                    builder,
                    layout,
                    writer=writer,
                )
            self.assertEqual(events, ["sealed.json", "selector.json"])
            self.assertEqual(
                builder_receipts,
                [common.durable_json_sha256(payload)],
            )

    def test_selector_builder_removes_only_duplicate_top_level_attempt(
        self,
    ) -> None:
        src = source()
        summary = runner.summarize_qualification(
            [
                observation(anchor, horizon)
                for anchor in range(2, 9)
                for horizon in (0.4, 0.8)
            ]
        )
        context = {
            "contract_path": Path("contract.json"),
            "roster_sha256": "c" * 64,
        }
        layout = {
            "attempt": Path("attempt.json"),
            "content_index": Path("content_index.json"),
        }
        with mock.patch.object(runner, "sha256", return_value="f" * 64):
            selector = runner._selector(
                src,
                context,
                layout,
                {"synthetic": True},
                "e" * 64,
                summary,
            )
        self.assertNotIn("slot_attempt_sha256", selector)
        self.assertEqual(
            "f" * 64,
            selector["source_authority_and_content_hashes"][
                "slot_attempt_sha256"
            ],
        )
        common.validate_selector(
            selector,
            src,
            "f" * 64,
            "c" * 64,
        )

    def test_qualifier_computes_truth_field_14_times_and_no_arm_field(
        self,
    ) -> None:
        src = source()
        poses = {
            index: {
                "position_m": [0.0, 0.0, float(index)],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
            for index in range(13)
        }
        media = {
            index: (
                np.ones((8, 8), dtype=float),
                np.ones((8, 8), dtype=np.uint8),
            )
            for index in range(2, 13)
        }
        basis = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )
        field_calls: list[int] = []

        def fake_field(*_args: object) -> np.ndarray:
            field_calls.append(1)
            value = np.ones((2, 6, 6), dtype=float)
            value[:, 0, 0] = -0.1
            return value

        with (
            mock.patch.object(
                runner,
                "load_slot_content",
                return_value=(poses, media),
            ),
            mock.patch.object(
                runner,
                "field_parameters",
                return_value={},
            ),
            mock.patch.object(
                runner,
                "fit_current_plane",
                return_value={},
            ),
            mock.patch.object(
                runner,
                "predicted_bases",
                return_value=(
                    basis,
                    {0.4: basis, 0.8: basis},
                    {"forbidden_motion": True},
                ),
            ),
            mock.patch.object(
                runner,
                "compute_known",
                return_value=(
                    np.full((2, 6, 6), 9, dtype=int),
                    np.ones((2, 6, 6), dtype=bool),
                ),
            ),
            mock.patch.object(
                runner,
                "compute_points",
                return_value=np.zeros((3, 1), dtype=float),
            ),
            mock.patch.object(
                runner,
                "compute_field",
                side_effect=fake_field,
            ),
        ):
            payload, summary = runner.qualify_content(
                src,
                {"camera": src["camera"]},
                g0={},
                mechanics={},
            )
        self.assertEqual(len(field_calls), 14)
        self.assertTrue(summary["qualified"])
        self.assertEqual(payload["observation_count"], 14)
        for row in payload["observations"]:
            for arm in runner.ARM_NAMES:
                self.assertNotIn("clearance_m", row["support"][arm])
            self.assertNotIn("motion", row)

    def test_preprocessor_index_rejects_embedded_pose(self) -> None:
        index = {
            "frames": [
                {
                    "normalized_index": value,
                    "source_frame_index": value,
                    "manifest_id": str(value),
                }
                for value in range(13)
            ]
        }
        self.assertEqual(
            list(preprocessor._frame_map(index)),
            list(range(13)),
        )
        index["frames"][4]["position_m"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "embeds pose"):
            preprocessor._frame_map(index)

    def test_preprocessor_reads_only_history_current_not_future_only(
        self,
    ) -> None:
        index = {
            "camera": source()["camera"],
            "frames": [
                {
                    "normalized_index": value,
                    "source_frame_index": value,
                    "manifest_id": str(value),
                }
                for value in range(13)
            ],
        }
        pose_reads: list[int] = []
        media_reads: list[int] = []

        def fake_pose(
            frame: dict[str, object],
            _root: Path,
        ) -> dict[str, object]:
            pose_reads.append(int(frame["normalized_index"]))
            return {}

        def fake_current(
            frame: dict[str, object],
            _camera: dict[str, object],
            _root: Path,
        ) -> tuple[np.ndarray, np.ndarray]:
            media_reads.append(int(frame["normalized_index"]))
            return np.zeros((1, 1)), np.zeros((1, 1))

        with (
            mock.patch.object(
                preprocessor,
                "_load_pose",
                side_effect=fake_pose,
            ),
            mock.patch.object(
                preprocessor,
                "_load_current",
                side_effect=fake_current,
            ),
        ):
            preprocessor.load_future_blind_inputs(index, Path("."))
        self.assertEqual(pose_reads, list(range(0, 9)))
        self.assertEqual(media_reads, list(range(2, 9)))
        self.assertFalse(set(range(9, 13)) & set(pose_reads))
        self.assertFalse(set(range(9, 13)) & set(media_reads))

    def test_prediction_support_must_exactly_match_qualifier(self) -> None:
        known = np.ones((2, 6, 6), dtype=bool)
        counts = np.full((2, 6, 6), 9, dtype=int)
        clearance = np.ones((2, 6, 6), dtype=float).tolist()
        prediction = {
            "known": known.tolist(),
            "probe_pass_counts": counts.tolist(),
            "clearance_m": clearance,
        }
        sealed = {
            "known": known.tolist(),
            "probe_pass_counts": counts.tolist(),
        }
        evaluator._support_matches_prediction(sealed, prediction)
        sealed["probe_pass_counts"][0][0][0] = 8
        with self.assertRaisesRegex(ValueError, "support mismatch"):
            evaluator._support_matches_prediction(sealed, prediction)

    def test_unknown_prediction_clearance_cannot_be_numeric(self) -> None:
        known = np.ones((2, 6, 6), dtype=bool)
        known[0, 0, 0] = False
        counts = np.full((2, 6, 6), 9, dtype=int)
        counts[0, 0, 0] = 0
        prediction = {
            "known": known.tolist(),
            "probe_pass_counts": counts.tolist(),
            "clearance_m": np.ones((2, 6, 6), dtype=float).tolist(),
        }
        sealed = {
            "known": known.tolist(),
            "probe_pass_counts": counts.tolist(),
        }
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            evaluator._support_matches_prediction(sealed, prediction)

    def test_only_supported_terminal_can_freeze_student_contract(self) -> None:
        supported = evaluator.result_authorization(evaluator.SUPPORTED)
        stopped = evaluator.result_authorization(evaluator.STOP)
        self.assertTrue(
            supported["freeze_rgb_student_contract_authorized"]
        )
        self.assertFalse(
            stopped["freeze_rgb_student_contract_authorized"]
        )
        self.assertFalse(
            supported["rgb_student_protocol_execution_authorized"]
        )
        self.assertFalse(supported["safety_claim_authorized"])

    def test_pretruth_failure_is_durable_and_closes_effect_rerun(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = root / "contract.json"
            selection = root / "selection.json"
            completion = root / "formal" / "predictions" / "completion.json"
            common.write_json_exclusive_fsync(contract, {"contract": True})
            common.write_json_exclusive_fsync(selection, {"selection": True})
            common.write_json_exclusive_fsync(
                completion,
                {"completion": True},
            )
            paths = evaluator._formal_paths(root)
            context = {"contract_path": contract}
            error = evaluator.EffectError("pretruth mismatch")
            evaluator._freeze_effect_failure(
                paths,
                context,
                selection,
                error,
                evaluator.PRETRUTH_FAILURE,
            )
            failure = common.load_json(paths["failure"])
            frozen_hash = common.sha256(paths["failure"])
            self.assertEqual(
                failure["terminal"],
                evaluator.PRETRUTH_FAILURE,
            )
            self.assertFalse(failure["sealed_payload_open_started"])
            self.assertFalse(failure["effect_rerun_authorized"])
            self.assertFalse(
                failure["second_sealed_payload_open_authorized"]
            )
            evaluator._freeze_effect_failure(
                paths,
                context,
                selection,
                evaluator.EffectError("second call"),
                evaluator.PRETRUTH_FAILURE,
            )
            self.assertEqual(frozen_hash, common.sha256(paths["failure"]))

    def test_orphan_preprocessor_attempt_freezes_without_input_reopen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "screening"
            contract = Path(temp) / "contract.json"
            common.write_json_exclusive_fsync(contract, {"contract": True})
            paths = preprocessor._prediction_paths(root)
            common.write_json_exclusive_fsync(
                paths["attempt"],
                {"orphan_attempt": True},
            )
            context = {"root": root, "contract_path": contract}
            with (
                mock.patch.object(
                    preprocessor,
                    "validate_execution_contract",
                    return_value=context,
                ),
                mock.patch.object(
                    preprocessor,
                    "SCREENING_ROOT_RELATIVE",
                    root.resolve(),
                ),
                mock.patch.object(
                    preprocessor,
                    "validate_selection",
                    side_effect=AssertionError(
                        "recovery must not reopen selection"
                    ),
                ),
                self.assertRaisesRegex(
                    preprocessor.PreprocessorError,
                    "prior preprocessor attempt",
                ),
            ):
                preprocessor.run_preprocessor(contract)
            failure = common.load_json(paths["failure"])
            self.assertEqual(
                failure["terminal"],
                preprocessor.FAILURE_TERMINAL,
            )
            self.assertIsNone(failure["selection_sha256"])
            self.assertFalse(failure["preprocessor_rerun_authorized"])

    def test_first_q0_1_call_only_initializes_control_plane(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "screening"
            contract = Path(temp) / "contract.json"
            common.write_json_exclusive_fsync(contract, {"contract": True})
            paths = common.aggregate_paths(root)
            slots = [
                {
                    "session_id": f"{index:064x}",
                    "d3_roster_slot_index": index,
                }
                for index in range(1, 41)
            ]
            authority = {
                "q0_protocol_sha256": "1" * 64,
                "metadata_roster_sha256": "c" * 64,
                "q0_execution_contract_sha256": "2" * 64,
                "q0_invalid_result_sha256": "3" * 64,
                "q0_screening_invalid_sha256": "4" * 64,
            }
            context = {
                "root": root,
                "contract_path": contract,
                "roster_sha256": "c" * 64,
                "slots": slots,
                "retries": 3,
                "carry_forward_authority": authority,
            }
            with (
                mock.patch.object(
                    runner,
                    "validate_execution_contract",
                    return_value=context,
                ),
                mock.patch.object(
                    runner,
                    "SCREENING_ROOT_RELATIVE",
                    root.resolve(),
                ),
                mock.patch.object(
                    runner,
                    "materialize_content",
                    side_effect=AssertionError("media must stay closed"),
                ),
                mock.patch.object(
                    runner,
                    "qualify_content",
                    side_effect=AssertionError("truth must stay closed"),
                ),
            ):
                result = runner.run_next_slot(contract, retries=3)
            self.assertTrue(result["screening_initialized"])
            self.assertFalse(result["media_pose_support_truth_opened"])
            self.assertEqual(1, result["consumed_slot_count"])
            self.assertEqual(0, result["newly_opened_slot_count"])
            self.assertEqual(2, result["next_slot_index"])
            self.assertTrue(paths["screening_attempt"].is_file())
            carry = common.slot_layout(root, slots[0])["carry_forward"]
            self.assertTrue(carry.is_file())
            receipt = common.load_json(carry)
            self.assertFalse(receipt["sealed_payload_read"])
            self.assertFalse(receipt["invalid_selector_read"])
            self.assertFalse(receipt["outcome_fields_imported"])
            self.assertEqual(
                set(common.slot_layout(root, slots[0])["slot_root"].iterdir()),
                {carry},
            )

    def test_second_q0_1_call_targets_only_original_slot_2(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "screening"
            contract = Path(temp) / "contract.json"
            common.write_json_exclusive_fsync(contract, {"contract": True})
            slots = [
                {
                    "session_id": f"{index:064x}",
                    "d3_roster_slot_index": index,
                }
                for index in range(1, 41)
            ]
            authority = {
                "q0_protocol_sha256": "1" * 64,
                "metadata_roster_sha256": "c" * 64,
                "q0_execution_contract_sha256": "2" * 64,
                "q0_invalid_result_sha256": "3" * 64,
                "q0_screening_invalid_sha256": "4" * 64,
            }
            context = {
                "root": root,
                "contract_path": contract,
                "roster_sha256": "c" * 64,
                "slots": slots,
                "retries": 3,
                "carry_forward_authority": authority,
                "g0": {},
                "mechanics": {},
            }
            runner.initialize_q0_1_control_plane(context)
            opened: list[int] = []

            def stop_at_media(
                selected: dict[str, object],
                _layout: dict[str, Path],
                _retries: int,
            ) -> dict[str, object]:
                opened.append(int(selected["d3_roster_slot_index"]))
                raise runner.SlotExecutionError("synthetic stop")

            with (
                mock.patch.object(
                    runner,
                    "validate_execution_contract",
                    return_value=context,
                ),
                mock.patch.object(
                    runner,
                    "SCREENING_ROOT_RELATIVE",
                    root.resolve(),
                ),
                mock.patch.object(
                    runner,
                    "materialize_content",
                    side_effect=stop_at_media,
                ),
                self.assertRaisesRegex(
                    runner.SlotExecutionError, "synthetic stop"
                ),
            ):
                runner.run_next_slot(contract, retries=3)
            self.assertEqual([2], opened)
            self.assertFalse(
                common.slot_layout(root, slots[0])["attempt"].exists()
            )
            self.assertTrue(
                common.slot_layout(root, slots[1])["attempt"].is_file()
            )

    def test_orphan_effect_attempt_freezes_without_truth_reopen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "screening"
            contract = Path(temp) / "contract.json"
            common.write_json_exclusive_fsync(contract, {"contract": True})
            paths = evaluator._formal_paths(root)
            common.write_json_exclusive_fsync(
                paths["attempt"],
                {"orphan_attempt": True},
            )
            context = {"root": root, "contract_path": contract}
            with (
                mock.patch.object(
                    evaluator,
                    "validate_execution_contract",
                    return_value=context,
                ),
                mock.patch.object(
                    evaluator,
                    "SCREENING_ROOT_RELATIVE",
                    root.resolve(),
                ),
                mock.patch.object(
                    evaluator,
                    "_prepare_effect_inputs",
                    side_effect=AssertionError(
                        "recovery must not reopen predictions"
                    ),
                ),
                mock.patch.object(
                    evaluator,
                    "_load_payload_once",
                    side_effect=AssertionError(
                        "recovery must not reopen sealed payload"
                    ),
                ),
                self.assertRaisesRegex(
                    evaluator.EffectError,
                    "prior effect attempt",
                ),
            ):
                evaluator.run_evaluator(contract)
            failure = common.load_json(paths["failure"])
            self.assertEqual(
                failure["terminal"],
                evaluator.PRETRUTH_FAILURE,
            )
            self.assertFalse(failure["sealed_payload_open_started"])
            self.assertFalse(failure["effect_rerun_authorized"])

    def test_frozen_execution_contract_validates_all_bound_receipts(
        self,
    ) -> None:
        context = common.validate_execution_contract(
            CONTRACT_PATH,
            runner.IMPLEMENTATION_KEY,
            Path(runner.__file__),
            verify_git=False,
        )
        self.assertEqual(len(context["slots"]), 40)
        self.assertEqual(context["retries"], 3)
        self.assertEqual(
            context["g0"]["schema"],
            common.G0_SCHEMA,
        )
        self.assertEqual(
            context["mechanics"]["schema"],
            common.MECHANICS_SCHEMA,
        )


if __name__ == "__main__":
    unittest.main()
