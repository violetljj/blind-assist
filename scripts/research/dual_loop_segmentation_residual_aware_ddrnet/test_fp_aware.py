from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .contract import validate_config_contract
from .evaluate import build_frame_row, compare_seed, pack_ids, unpack_ids
from .train import FpAwareSampler, build_hard_negative_masks
from .validate import validate


@dataclass
class Record:
    session_id: str


class FpAwareTrainingTests(unittest.TestCase):
    def test_hard_negative_definition_excludes_true_hazard(self) -> None:
        prediction = np.asarray([[[1, 2], [1, 0]]], dtype=np.uint8)
        truth = np.asarray([[[0, 2], [3, 0]]], dtype=np.uint8)
        actual = build_hard_negative_masks(prediction, truth)
        expected = np.asarray([[[True, False], [True, False]]])
        np.testing.assert_array_equal(actual, expected)

    def test_unguided_branch_keeps_full_frame_and_uses_fp_weights(self) -> None:
        images = np.stack(
            [
                np.full((8, 8, 3), 11, dtype=np.float32),
                np.full((8, 8, 3), 22, dtype=np.float32),
            ]
        )
        masks = np.zeros((2, 8, 8), dtype=np.int64)
        hard = np.zeros((2, 8, 8), dtype=bool)
        hard[0, 0, 0] = True
        hard[1, :, :] = True
        sampler = FpAwareSampler(
            images,
            masks,
            [Record("s"), Record("s")],
            hard,
            batch_size=200,
            seed=7,
            guided_crop_fraction=0.0,
            boundary_guided_probability=0.65,
            crop_min_fraction=0.55,
            crop_max_fraction=0.85,
            horizontal_flip_probability=0.0,
        )
        batch_images, batch_masks = sampler.next_batch()
        self.assertEqual((200, 8, 8, 3), batch_images.shape)
        self.assertEqual((200, 8, 8), batch_masks.shape)
        selected_high_weight = int(np.count_nonzero(batch_images[:, 0, 0, 0] == 22))
        self.assertGreater(selected_high_weight, 180)
        self.assertEqual({"fp_weighted_full_frame": 200}, sampler.receipt()["branch_draws"])
        self.assertEqual("unchanged_full_frame", sampler.receipt()["unguided_transform"])

    def test_pool_coverage_fails_closed(self) -> None:
        images = np.zeros((1, 4, 4, 3), dtype=np.float32)
        masks = np.zeros((1, 4, 4), dtype=np.int64)
        sampler = FpAwareSampler(
            images,
            masks,
            [Record("s")],
            np.zeros((1, 4, 4), dtype=bool),
            batch_size=1,
            seed=1,
            guided_crop_fraction=0.7,
            boundary_guided_probability=0.65,
            crop_min_fraction=0.55,
            crop_max_fraction=0.85,
            horizontal_flip_probability=0.5,
        )
        with self.assertRaisesRegex(RuntimeError, "zero candidates"):
            sampler.validate_pool_coverage()

    def test_prediction_pack_round_trip(self) -> None:
        ids = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256) % 4
        np.testing.assert_array_equal(ids.astype(np.uint8), unpack_ids(pack_ids(ids)))

    def test_class_guard_does_not_credit_hazard_class_swap(self) -> None:
        truth = np.zeros((256, 256), dtype=np.uint8)
        prediction = np.zeros((256, 256), dtype=np.uint8)
        truth[4, 4] = 1
        prediction[4, 4] = 2
        row = build_frame_row(
            manifest={
                "id": "x",
                "role": "consumed_old_blind",
                "source_id": "s",
                "session_id": "s",
                "frame_id": 1,
                "image_sha256": "i",
                "canonical_mask_sha256": "m",
            },
            trace={"detections": []},
            source_size=(256, 256),
            truth_ids=truth,
            predicted_ids=prediction,
            seed=1,
            arm="FP_AWARE_CANDIDATE",
            checkpoint_sha256="c",
        )
        self.assertEqual(0, row["metrics"]["boundary_candidate"]["tp"])
        self.assertEqual(1, row["metrics"]["boundary_candidate"]["fn"])
        self.assertEqual(1, row["metrics"]["obstacle_candidate"]["fp"])

    def test_config_contract_rejects_seed_sampling_and_gate_drift(self) -> None:
        base = {
            "protocol_id": "DUAL_LOOP_SEGMENTATION_FP_AWARE_DDRNET_R0",
            "candidate_id": "FP_WEIGHTED_UNGUIDED_FULL_FRAME",
            "stage": "DEVELOPMENT",
            "training": {
                "optimizer": "Adam",
                "optimizer_steps": 1200,
                "head_warmup_steps": 100,
                "eval_every_steps": 50,
                "batch_size": 12,
                "seeds": [20260711, 20260712, 20260713],
                "head_learning_rate": 0.0003,
                "finetune_learning_rate": 0.00005,
                "finetune_final_lr_ratio": 0.1,
                "freeze_batchnorm_statistics": True,
                "sampling": {
                    "session_selection": "uniform",
                    "positive_guided_fraction": 0.7,
                    "fp_weighted_full_frame_fraction": 0.3,
                    "boundary_probability_within_positive_branch": 0.65,
                    "crop_min_fraction": 0.55,
                    "crop_max_fraction": 0.85,
                    "horizontal_flip_probability": 0.5,
                    "zero_session_fp_weight": "NOT_EVALUABLE_BEFORE_TRAINING",
                    "full_frame_transform_unchanged": True,
                },
                "loss": {
                    "weighted_cross_entropy": 0.5,
                    "weighted_soft_dice": 0.4,
                    "weighted_focal": 0.1,
                    "focal_gamma": 2.0,
                    "maximum_class_weight": 4.0,
                    "class_weight_source": "train_only",
                },
            },
            "evaluation": {
                "roles": ["consumed_old_blind", "r1_consumed_fresh"],
                "role_counts": {"consumed_old_blind": 120, "r1_consumed_fresh": 200},
                "batch_size": 12,
                "connectivity": 8,
                "component_hit_rule": "positive_intersection",
                "same_seed_pairing": True,
                "all_three_seeds_required": True,
                "best_seed_selection_forbidden": True,
                "gates": {
                    "min_fp_pixel_reduction": 0.3,
                    "min_overall_recall_retention": 0.9,
                    "min_session_recall_retention": 0.8,
                    "min_boundary_recall_retention": 0.8,
                    "min_obstacle_recall_retention": 0.8,
                    "min_delta_recall_C_minus_A": 0.05,
                    "max_delta_false_positive_area_fraction_C_minus_A": 0.05,
                    "min_candidate_component_recall": 0.5,
                    "max_false_activation_components_per_frame": 3.0,
                },
            },
            "terminals": {
                "supported": "FP_WEIGHTED_SAMPLING_SUPPORTED_DEVELOPMENT_ONLY",
                "not_supported": "FP_WEIGHTED_SAMPLING_NOT_SUPPORTED",
                "not_evaluable": "FP_WEIGHTED_SAMPLING_NOT_EVALUABLE",
            },
        }
        validate_config_contract(base)
        for path, value in (
            (("training", "seeds"), [20260711]),
            (("training", "sampling", "positive_guided_fraction"), 0.6),
            (("evaluation", "gates", "min_fp_pixel_reduction"), 0.1),
        ):
            changed = deepcopy(base)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                validate_config_contract(changed)

    def test_validator_fails_closed_and_refuses_receipt_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "validation.json"
            receipt = validate(
                repo_root=root,
                config_path=Path("missing-config.json"),
                result_path=Path("missing-result.json"),
                output_path=output,
                device="cpu",
            )
            self.assertEqual("INVALID", receipt["status"])
            self.assertEqual("FP_WEIGHTED_SAMPLING_NOT_EVALUABLE", receipt["terminal"])
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                validate(
                    repo_root=root,
                    config_path=Path("missing-config.json"),
                    result_path=Path("missing-result.json"),
                    output_path=output,
                    device="cpu",
                )

    def test_decision_requires_all_relative_and_absolute_gates(self) -> None:
        def aggregate(tp: int, fp: int, truth: int = 100) -> dict:
            return {
                "candidate": {"tp": tp, "fp": fp},
                "boundary_candidate": {"tp": tp // 2},
                "obstacle_candidate": {"tp": tp // 2},
                "components": {
                    "false_activation_component_count": fp,
                    "component_recall": 0.8,
                    "false_activation_components_per_frame": 2.0,
                },
                "delta_recall_C_minus_A": 0.10,
                "delta_false_positive_area_fraction_C_minus_A": 0.04,
            }

        baseline = {
            "overall": aggregate(100, 100),
            "sessions": {"a": {"candidate": {"tp": 50}}, "b": {"candidate": {"tp": 50}}},
        }
        candidate = {
            "overall": aggregate(95, 60),
            "sessions": {"a": {"candidate": {"tp": 45}}, "b": {"candidate": {"tp": 50}}},
        }
        gates = {
            "min_fp_pixel_reduction": 0.30,
            "min_overall_recall_retention": 0.90,
            "min_session_recall_retention": 0.80,
            "min_boundary_recall_retention": 0.80,
            "min_obstacle_recall_retention": 0.80,
            "min_delta_recall_C_minus_A": 0.05,
            "max_delta_false_positive_area_fraction_C_minus_A": 0.05,
            "min_candidate_component_recall": 0.50,
            "max_false_activation_components_per_frame": 3.0,
        }
        result = compare_seed(seed=1, baseline=baseline, candidate=candidate, gates=gates)
        self.assertTrue(result["all_nine_gates_passed"])
        candidate["overall"]["components"]["false_activation_components_per_frame"] = 3.1
        result = compare_seed(seed=1, baseline=baseline, candidate=candidate, gates=gates)
        self.assertFalse(result["all_nine_gates_passed"])


if __name__ == "__main__":
    unittest.main()
