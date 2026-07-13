from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanpo_segmentation_model
import train_sanpo_segmentation_keras_torch as protocol


def metrics(mean_iou: float, boundary_iou: float, pixel_accuracy: float = 0.5) -> dict:
    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "per_class": {
            "boundary_step_curb": {"iou": boundary_iou},
            "unknown_nonwalkable": {"iou": mean_iou / 2.0},
        },
    }


class SanpoTrainingProtocolTest(unittest.TestCase):
    def test_default_protocol_uses_small_batch_fixed_steps_and_three_seeds(self) -> None:
        args = protocol.parse_args(["--dataset-root", "fixture"])
        self.assertEqual(12, args.batch_size)
        self.assertEqual(1200, args.optimizer_steps)
        self.assertEqual(protocol.DEFAULT_SEEDS, args.seeds)
        self.assertTrue(args.two_stage)
        self.assertEqual(100, args.head_warmup_steps)
        self.assertEqual(5e-5, args.finetune_learning_rate)
        self.assertTrue(args.freeze_backbone_batchnorm)
        self.assertEqual(0.75, args.backbone_alpha)
        self.assertEqual(96, args.decoder_channels)
        self.assertEqual("session_balanced_guided", args.sampler_strategy)
        self.assertEqual(0.75, args.unknown_rich_quantile)

    def test_single_seed_audit_and_alpha_ablation_are_explicit(self) -> None:
        args = protocol.parse_args([
            "--dataset-root", "fixture", "--seed", "7", "--backbone-alpha", "1.0",
        ])
        self.assertEqual((7,), args.seeds)
        self.assertEqual(1.0, args.backbone_alpha)
        self.assertEqual(8, args.detail_output_stride)
        self.assertEqual(32, args.semantic_output_stride)

    def test_p0_head_only_seed_pairs_separate_initialization_and_sampler(self) -> None:
        args = protocol.parse_args([
            "--dataset-root", "fixture", "--head-only", "--optimizer-steps", "100",
            "--minimum-optimizer-steps", "100", "--head-warmup-steps", "100",
            "--seed-pairs", "11:11,12:11,13:11,11:12,11:13",
        ])
        self.assertTrue(args.head_only)
        self.assertEqual(((11, 11), (12, 11), (13, 11), (11, 12), (11, 13)), args.seed_pairs)
        self.assertEqual((11, 12, 13), args.seeds)
        with self.assertRaises(SystemExit):
            protocol.parse_args([
                "--dataset-root", "fixture", "--seed", "1", "--seed-pairs", "1:2",
            ])

    def test_input_size_probes_and_single_stage_fallback_are_explicit(self) -> None:
        for input_size in protocol.ALLOWED_INPUT_SIZES:
            args = protocol.parse_args([
                "--dataset-root", "fixture", "--input-size", str(input_size), "--no-two-stage",
            ])
            self.assertEqual(input_size, args.input_size)
            self.assertFalse(args.two_stage)
        with self.assertRaises(SystemExit):
            protocol.parse_args([
                "--dataset-root", "fixture", "--optimizer-steps", "100",
                "--minimum-optimizer-steps", "50", "--head-warmup-steps", "100",
            ])

    def test_loss_weights_are_bounded_despite_extreme_boundary_imbalance(self) -> None:
        counts = {
            "walkable": 1_000_000,
            "boundary_step_curb": 1,
            "obstacle": 10_000,
            "unknown_nonwalkable": 500_000,
        }
        weights = protocol.class_loss_weights(counts, maximum=4.0)
        self.assertEqual((4,), weights.shape)
        self.assertLessEqual(float(weights.max()), 4.0)
        self.assertGreaterEqual(float(weights.min()), 0.3499)
        self.assertGreater(float(weights[protocol.BOUNDARY_CLASS_ID]), float(weights[0]))

    def test_selection_score_cannot_hide_boundary_collapse(self) -> None:
        healthy = metrics(0.40, 0.30)
        collapsed = metrics(0.80, 0.0)
        self.assertGreater(protocol.selection_score(healthy), 0.0)
        self.assertEqual(0.0, protocol.selection_score(collapsed))
        self.assertGreater(protocol.checkpoint_key(healthy), protocol.checkpoint_key(collapsed))

    def test_multi_seed_summary_reports_dispersion_and_worst_seed(self) -> None:
        runs = [
            {"selection_score": 0.20, "dev_mask_metrics": metrics(0.30, 0.15, 0.50)},
            {"selection_score": 0.30, "dev_mask_metrics": metrics(0.40, 0.25, 0.60)},
            {"selection_score": 0.10, "dev_mask_metrics": metrics(0.20, 0.05, 0.40)},
        ]
        summary = protocol.aggregate_seed_metrics(runs)
        self.assertEqual(3, summary["seed_count"])
        self.assertAlmostEqual(0.30, summary["mean_iou"]["mean"])
        self.assertAlmostEqual(0.20, summary["mean_iou"]["minimum"])
        self.assertGreater(summary["mean_iou"]["std"], 0.0)

    def test_p0_factor_summary_keeps_init_and_sampler_comparisons_separate(self) -> None:
        runs = []
        for model_seed, sampler_seed, mean_iou, boundary_iou in (
            (11, 11, 0.30, 0.20), (12, 11, 0.40, 0.30), (13, 11, 0.20, 0.10),
            (11, 12, 0.35, 0.25), (11, 13, 0.25, 0.15),
        ):
            run_metrics = metrics(mean_iou, boundary_iou)
            runs.append({
                "model_seed": model_seed,
                "sampler_seed": sampler_seed,
                "selection_score": protocol.selection_score(run_metrics),
                "dev_mask_metrics": run_metrics,
            })
        summary = protocol.factor_variation_summary(runs)
        init = summary["model_initialization_effect_with_sampler_fixed"]
        sampler = summary["sampler_effect_with_model_initialization_fixed"]
        self.assertEqual(1, len(init))
        self.assertEqual(11, init[0]["sampler_seed"])
        self.assertEqual([11, 12, 13], init[0]["model_seeds"])
        self.assertEqual(1, len(sampler))
        self.assertEqual(11, sampler[0]["model_seed"])
        self.assertEqual([11, 12, 13], sampler[0]["sampler_seeds"])

    def test_composite_loss_is_finite_on_backend_neutral_keras_ops(self) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow/Keras smoke environment is not installed")
        args = SimpleNamespace(ce_weight=0.5, dice_weight=0.4, focal_weight=0.1, focal_gamma=2.0)
        loss = protocol.build_composite_loss(tf.keras, np.ones(4, dtype=np.float32), args)
        labels = np.array([[[0, 1], [2, 3]]], dtype=np.int32)
        logits = np.zeros((1, 2, 2, 4), dtype=np.float32)
        value = float(loss(labels, logits).numpy())
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_sampler_balances_sessions_and_guides_rare_class_crops(self) -> None:
        images = np.zeros((4, 16, 16, 3), dtype=np.float32)
        masks = np.zeros((4, 16, 16), dtype=np.int64)
        masks[:, 6:10, 6:10] = protocol.BOUNDARY_CLASS_ID
        records = [
            SimpleNamespace(session_id="many") for _ in range(3)
        ] + [SimpleNamespace(session_id="single")]
        sampler = protocol.SessionBalancedCropSampler(
            images,
            masks,
            records,
            batch_size=8,
            seed=17,
            guided_crop_fraction=1.0,
            crop_min_fraction=0.5,
            crop_max_fraction=0.5,
            horizontal_flip_probability=0.0,
        )
        for _ in range(100):
            batch_images, batch_masks = sampler.next_batch()
            self.assertEqual((8, 16, 16, 3), batch_images.shape)
            self.assertEqual((8, 16, 16), batch_masks.shape)
        report = sampler.report()
        total = sum(report["session_draws"].values())
        many_share = report["session_draws"]["many"] / total
        self.assertGreater(many_share, 0.43)
        self.assertLess(many_share, 0.57)
        self.assertGreater(report["guided_crop_hits"]["boundary_step_curb"], 0)

    def test_deterministic_quota_sampler_closes_exact_25_percent_with_batch_six(self) -> None:
        images = np.stack([
            np.full((16, 16, 3), index, dtype=np.uint8) for index in range(8)
        ])
        masks = np.zeros((8, 16, 16), dtype=np.uint8)
        masks[0, 4:12, 4:12] = protocol.BOUNDARY_CLASS_ID
        masks[1, 4:12, 4:12] = protocol.OBSTACLE_CLASS_ID
        masks[2, :, :] = 3
        masks[3, :12, :] = 3
        masks[4, 2:14, 2:14] = protocol.BOUNDARY_CLASS_ID
        masks[5, 2:14, 2:14] = protocol.OBSTACLE_CLASS_ID
        records = [SimpleNamespace(session_id=f"session-{index % 4}") for index in range(8)]

        def make(seed: int) -> protocol.DeterministicQuotaSampler:
            return protocol.DeterministicQuotaSampler(
                images,
                masks,
                records,
                batch_size=6,
                seed=seed,
                crop_min_fraction=0.5,
                crop_max_fraction=0.75,
                horizontal_flip_probability=0.5,
                unknown_rich_quantile=0.75,
            )

        first = make(17)
        second = make(17)
        prefix = make(17)
        prefix.next_batch()
        prefix_report = prefix.report()
        self.assertFalse(prefix_report["schedule_cycle_complete"])
        self.assertEqual(1, prefix_report["maximum_quota_draw_imbalance"])
        prefix.next_batch()
        prefix_report = prefix.report()
        self.assertTrue(prefix_report["schedule_cycle_complete"])
        self.assertEqual({name: 3 for name in protocol.QUOTA_NAMES}, prefix_report["quota_draws"])
        for _ in range(100):
            first_images, first_masks = first.next_batch()
            second_images, second_masks = second.next_batch()
            np.testing.assert_array_equal(first_images, second_images)
            np.testing.assert_array_equal(first_masks, second_masks)
        report = first.report()
        self.assertEqual({name: 150 for name in protocol.QUOTA_NAMES}, report["quota_draws"])
        self.assertEqual({name: 0.25 for name in protocol.QUOTA_NAMES}, report["quota_realized_share"])
        self.assertEqual(150, report["full_frame_draws"]["hard_negative"])
        self.assertEqual(150, report["full_frame_draws"]["unknown_full_frame"])
        self.assertEqual(150, report["guided_crop_hits"]["boundary_step_curb"])
        self.assertEqual(150, report["guided_crop_hits"]["obstacle"])
        different = make(18)
        for _ in range(100):
            different.next_batch()
        self.assertNotEqual(report["sample_trace_sha256"], different.report()["sample_trace_sha256"])

    def test_deterministic_quota_sampler_fails_closed_when_a_quota_has_no_candidates(self) -> None:
        images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[0, 2:6, 2:6] = protocol.OBSTACLE_CLASS_ID
        masks[1, :, :] = 3
        records = [SimpleNamespace(session_id="a"), SimpleNamespace(session_id="b")]
        with self.assertRaisesRegex(ValueError, "boundary"):
            protocol.DeterministicQuotaSampler(
                images,
                masks,
                records,
                batch_size=4,
                seed=1,
                crop_min_fraction=0.5,
                crop_max_fraction=0.75,
                horizontal_flip_probability=0.0,
            )

    def test_weight_path_and_model_hyperparameter_contracts_fail_closed(self) -> None:
        path = protocol.seeded_weight_path(Path("candidate.weights.h5"), 9)
        self.assertEqual("candidate.seed-9.weights.h5", path.name)
        stage_path = protocol.stage_weight_path(Path("candidate.weights.h5"), 9, "head_warmup")
        self.assertEqual("candidate.seed-9.stage-head-warmup.weights.h5", stage_path.name)
        paired = protocol.seed_pair_weight_path(Path("candidate.weights.h5"), 9, 10)
        self.assertEqual("candidate.model-seed-9.sampler-seed-10.weights.h5", paired.name)
        self.assertAlmostEqual(5e-5, protocol.cosine_decay_value(5e-5, 0.1, 0, 100))
        self.assertAlmostEqual(5e-6, protocol.cosine_decay_value(5e-5, 0.1, 100, 100))
        with self.assertRaisesRegex(ValueError, "input_size"):
            sanpo_segmentation_model.build_mobilenetv3_lraspp(None, 320)
        with self.assertRaisesRegex(ValueError, "backbone_alpha"):
            sanpo_segmentation_model.build_mobilenetv3_lraspp(None, 256, backbone_alpha=0.5)
        with self.assertRaisesRegex(ValueError, "decoder_channels"):
            sanpo_segmentation_model.build_mobilenetv3_lraspp(None, 256, decoder_channels=0)
        with self.assertRaisesRegex(ValueError, "detail_output_stride"):
            sanpo_segmentation_model.build_mobilenetv3_lraspp(None, 256, detail_output_stride=16)
        with self.assertRaisesRegex(ValueError, "semantic_output_stride"):
            sanpo_segmentation_model.build_mobilenetv3_lraspp(None, 256, semantic_output_stride=8)

    def test_lraspp_context_branch_is_a_sigmoid_gate_without_pooled_batchnorm(self) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow/Keras smoke environment is not installed")
        model = sanpo_segmentation_model.build_mobilenetv3_lraspp(
            tf.keras, 256, backbone_weights=None,
        )
        layer_names = {layer.name for layer in model.layers}
        self.assertIn("lraspp_context_sigmoid", layer_names)
        self.assertNotIn("lraspp_context_bn", layer_names)
        self.assertNotIn("lraspp_context_relu", layer_names)
        sigmoid = model.get_layer("lraspp_context_sigmoid")
        self.assertEqual("sigmoid", sigmoid.activation.__name__)

    def test_paper_endpoint_probe_keeps_deep_semantics_at_os16_and_detail_at_os4(self) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow/Keras smoke environment is not installed")
        model = sanpo_segmentation_model.build_mobilenetv3_lraspp(
            tf.keras,
            256,
            backbone_weights=None,
            detail_output_stride=4,
            semantic_output_stride=16,
        )
        self.assertEqual((None, 256, 256, 4), tuple(model.output.shape))
        contract = model.sanpo_feature_contract
        self.assertEqual(4, contract["detail_output_stride"])
        self.assertEqual("expanded_conv_project_bn", contract["detail_endpoint"])
        self.assertEqual(16, contract["semantic_output_stride"])
        self.assertEqual("deepest_backbone_output", contract["semantic_endpoint"])

    def test_two_stage_trainability_freezes_backbone_and_keeps_head_live(self) -> None:
        class Layer:
            def __init__(self, name: str) -> None:
                self.name = name
                self.trainable = True

        class BatchNormalization(Layer):
            pass

        fake_keras = SimpleNamespace(layers=SimpleNamespace(BatchNormalization=BatchNormalization))
        backbone = Layer("Conv")
        batchnorm = BatchNormalization("expanded_conv_bn")
        head = Layer("lraspp_high_project")
        logits = Layer("semantic_logits")
        model = SimpleNamespace(layers=[backbone, batchnorm, head, logits])
        warmup = protocol.configure_trainable_layers(
            model, fake_keras, backbone_trainable=False, keep_batchnorm_frozen=True,
        )
        self.assertFalse(backbone.trainable)
        self.assertFalse(batchnorm.trainable)
        self.assertTrue(head.trainable)
        self.assertTrue(logits.trainable)
        self.assertEqual(2, warmup["trainable_layer_count"])
        finetune = protocol.configure_trainable_layers(
            model, fake_keras, backbone_trainable=True, keep_batchnorm_frozen=True,
        )
        self.assertTrue(backbone.trainable)
        self.assertFalse(batchnorm.trainable)
        self.assertEqual(1, finetune["frozen_batchnorm_count"])


if __name__ == "__main__":
    unittest.main()
