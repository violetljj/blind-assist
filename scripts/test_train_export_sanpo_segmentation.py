from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_export_sanpo_segmentation as candidate


class SegmentationCandidateToolTest(unittest.TestCase):
    def write_record(self, root: Path, sample_id: str, split: str, session: str, class_id: int = 0) -> dict:
        image = root / "images" / f"{sample_id}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "black").save(image)
        masks = {}
        for index, name in enumerate(candidate.CLASS_NAMES):
            path = root / "masks" / f"{sample_id}_{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((8, 8), 255 if index == class_id else 0, dtype=np.uint8)
            Image.fromarray(pixels, mode="L").save(path)
            masks[name] = str(path.relative_to(root)).replace("\\", "/")
        return {
            "id": sample_id,
            "segmentation_split": split,
            "source_session_id": session,
            "label_authority": "source_ground_truth",
            "image_path": str(image.relative_to(root)).replace("\\", "/"),
            "semantic_mask_paths": masks,
            "scene_bucket": "parallel_curb",
        }

    def write_manifest(self, root: Path, rows: list[dict]) -> Path:
        path = root / "train_dev.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_load_records_rejects_blind_labels_without_opening_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = self.write_record(root, "train", "train", "session_train")
            dev = self.write_record(root, "dev", "dev", "session_dev")
            blind = self.write_record(root, "blind", "blind_holdout", "session_blind")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                candidate.load_records(self.write_manifest(root, [train, dev, blind]))

    def test_load_records_rejects_session_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = self.write_record(root, "train", "train", "one_session")
            dev = self.write_record(root, "dev", "dev", "one_session")
            with self.assertRaisesRegex(ValueError, "session leakage"):
                candidate.load_records(self.write_manifest(root, [train, dev]))

    def test_dev_accepts_only_fully_bound_procedural_labels_and_rejects_pseudo(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = self.write_record(root, "train", "train", "session_train")
            dev = self.write_record(root, "dev", "dev", "session_dev", 2)
            semantic = root / "masks" / "dev_semantic.png"
            Image.fromarray(np.full((8, 8), 2, dtype=np.uint8), mode="L").save(semantic)
            dev.pop("semantic_mask_paths")
            dev["semantic_mask_path"] = semantic.relative_to(root).as_posix()
            dev["semantic_mask_sha256"] = candidate.training_gate.sha256_file(semantic)
            evidence = root / "procedural_evidence"
            evidence.mkdir()
            tactile = evidence / "tactile.png"
            obstacle = evidence / "obstacle.png"
            code = evidence / "generator.py"
            config = evidence / "config.json"
            Image.new("L", (8, 8), 1).save(tactile)
            Image.new("L", (8, 8), 2).save(obstacle)
            code.write_text("# tactile_occupied_compositor_v1\n", encoding="utf-8")
            config.write_text('{"version":1}\n', encoding="utf-8")
            matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            dev["label_authority"] = "procedural_ground_truth"
            dev["label_provenance"] = {
                "schema": candidate.training_gate.validator.PROCEDURAL_PROVENANCE_SCHEMA,
                "generator_id": "tactile_occupied_compositor_v1",
                "generator_code_path": code.relative_to(root).as_posix(),
                "generator_code_sha256": candidate.training_gate.sha256_file(code),
                "generator_config_path": config.relative_to(root).as_posix(),
                "generator_config_sha256": candidate.training_gate.sha256_file(config),
                "seed": 913,
                "transform_matrix": matrix,
                "transform_sha256": candidate.training_gate.validator._canonical_json_sha256(matrix),
                "source_masks": [
                    {"role": "tactile_ground_truth", "source_id": "guidetwsi_fixture", "path": tactile.relative_to(root).as_posix(), "sha256": candidate.training_gate.sha256_file(tactile)},
                    {"role": "obstacle_ground_truth", "source_id": "sanpo_fixture", "path": obstacle.relative_to(root).as_posix(), "sha256": candidate.training_gate.sha256_file(obstacle)},
                ],
                "source_assets": [
                    {"role": "guide_rgb", "source_id": "guidetwsi_fixture", "path": tactile.relative_to(root).as_posix(), "sha256": candidate.training_gate.sha256_file(tactile)},
                    {"role": "guide_polygon", "source_id": "guidetwsi_fixture", "path": config.relative_to(root).as_posix(), "sha256": candidate.training_gate.sha256_file(config)},
                    {"role": "sanpo_rgb", "source_id": "sanpo_fixture", "path": dev["image_path"], "sha256": candidate.training_gate.sha256_file(root / dev["image_path"])},
                    {"role": "sanpo_raw_mask", "source_id": "sanpo_fixture", "path": obstacle.relative_to(root).as_posix(), "sha256": candidate.training_gate.sha256_file(obstacle)},
                ],
                "output_mask_sha256": dev["semantic_mask_sha256"],
            }
            records = candidate.load_records(self.write_manifest(root, [train, dev]))
            self.assertEqual("procedural_ground_truth", records[1].label_authority)
            dev["label_provenance"]["generator_config_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "invalid procedural dev label"):
                candidate.load_records(self.write_manifest(root, [train, dev]))
            dev["label_authority"] = "teacher_consensus_pseudo_label"
            with self.assertRaisesRegex(ValueError, "forbids teacher/pseudo"):
                candidate.load_records(self.write_manifest(root, [train, dev]))

    def test_mask_contract_requires_full_non_overlapping_four_class_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = self.write_record(root, "train", "train", "session_train")
            record = candidate.load_records(self.write_manifest(root, [row, self.write_record(root, "dev", "dev", "session_dev", 2)]))[0]
            mask = candidate.validate_binary_masks(record)
            self.assertEqual((8, 8), mask.shape)
            self.assertTrue(np.all(mask == 0))
            overlap = record.masks["obstacle"]
            Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(overlap)
            with self.assertRaisesRegex(ValueError, "overlap"):
                candidate.validate_binary_masks(record)

    def test_confusion_metrics_are_reported_per_class(self) -> None:
        metrics = candidate.confusion_and_metrics(
            [np.array([[0, 1], [2, 3]], dtype=np.uint8)],
            [np.array([[0, 1], [3, 3]], dtype=np.uint8)],
        )
        self.assertEqual(set(candidate.CLASS_NAMES), set(metrics["per_class"]))
        self.assertAlmostEqual(1.0, metrics["per_class"]["walkable"]["iou"])
        self.assertLess(metrics["per_class"]["obstacle"]["iou"], 1.0)

    def test_canonical_single_id_semantic_mask_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = self.write_record(root, "train", "train", "session_train")
            dev = self.write_record(root, "dev", "dev", "session_dev", 2)
            id_mask = root / "masks" / "train_semantic.png"
            Image.fromarray(np.full((8, 8), 3, dtype=np.uint8), mode="L").save(id_mask)
            train.pop("semantic_mask_paths")
            train["semantic_mask_path"] = str(id_mask.relative_to(root)).replace("\\", "/")
            record = candidate.load_records(self.write_manifest(root, [train, dev]))[0]
            self.assertEqual({}, record.masks)
            self.assertEqual(3, int(candidate.validate_binary_masks(record)[0, 0]))

    def test_production_asset_output_is_rejected_before_tensorflow_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = candidate.parse_args(["--dataset-root", str(root), "--output", "app/src/main/assets/not_allowed.tflite"])
            with self.assertRaisesRegex(ValueError, "production app assets"):
                candidate.run(args)

    def test_export_only_requires_imported_weights(self) -> None:
        with self.assertRaises(SystemExit):
            candidate.parse_args(["--dataset-root", "fixture", "--export-only"])

    def test_imported_weights_require_backend_equivalence_report(self) -> None:
        with self.assertRaises(SystemExit):
            candidate.parse_args([
                "--dataset-root", "fixture", "--import-weights", "candidate.weights.h5", "--export-only",
            ])

    def test_authoritative_model_definition_is_delegated(self) -> None:
        self.assertIsNotNone(candidate.sanpo_segmentation_model.build_mobilenetv3_lraspp)

    def test_model_config_cli_is_explicit_and_validated(self) -> None:
        args = candidate.parse_args([
            "--dataset-root", "fixture",
            "--backbone-alpha", "1.0",
            "--decoder-channels", "128",
            "--input-size", "512",
        ])
        self.assertEqual(1.0, args.backbone_alpha)
        self.assertEqual(128, args.decoder_channels)
        self.assertEqual(512, args.input_size)
        with self.assertRaises(SystemExit):
            candidate.parse_args([
                "--dataset-root", "fixture",
                "--backbone-alpha", "0.5",
            ])
        with self.assertRaises(SystemExit):
            candidate.parse_args([
                "--dataset-root", "fixture",
                "--decoder-channels", "0",
            ])
        with self.assertRaises(SystemExit):
            candidate.parse_args([
                "--dataset-root", "fixture",
                "--input-size", "320",
            ])

    def test_tensorflow_builds_and_exports_full_int8_contract(self) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow export environment is not installed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = self.write_record(root, "train", "train", "session_train", 0)
            dev = self.write_record(root, "dev", "dev", "session_dev", 2)
            records = candidate.load_records(self.write_manifest(root, [train, dev]))
            model = candidate.build_mobilenetv3_lraspp(tf, candidate.INPUT_SIZE)
            self.assertEqual((None, 256, 256, 4), model.output_shape)
            output = root / "candidate_int8.tflite"
            candidate.export_full_int8(tf, model, [records[0]], output, candidate.INPUT_SIZE, 1)
            contract = candidate.validate_int8_tflite(tf, output)
            self.assertEqual("int8", contract["input"]["dtype"])
            self.assertEqual("int8", contract["output"]["dtype"])
            with self.assertRaisesRegex(AssertionError, "shape mismatch"):
                candidate.validate_int8_tflite(tf, output, input_size=512)


if __name__ == "__main__":
    unittest.main()
