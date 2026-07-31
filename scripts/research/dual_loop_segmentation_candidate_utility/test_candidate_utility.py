from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .component_metrics import component_metrics, component_records, connected_components, pixel_metrics
from .evaluate_candidate_utility import (
    CandidateUtilityInputError,
    _json_default,
    load_manifest,
)
from ..dual_loop_segmentation_complementarity.produce_host_trace import (
    CONFIDENCE_THRESHOLD,
    MODEL_INPUT_SIZE,
    NMS_IOU_THRESHOLD,
    SCHEMA_VERSION as HOST_TRACE_SCHEMA_VERSION,
    HostTraceInputError,
    finalize_existing_trace,
    load_manifest as load_host_manifest,
    sha256_file as host_sha256_file,
)
from .temporal_metrics import summarize_temporal, warp_mask


class CandidateUtilityMetricTests(unittest.TestCase):
    def test_pixel_metrics_and_eight_connected_components_are_deterministic(self) -> None:
        predicted = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        truth = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        metrics = pixel_metrics(predicted, truth)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["fn"], 0)
        components = connected_components(predicted)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].bbox, (0, 0, 3, 3))

    def test_candidate_component_records_keep_truth_and_box_distance(self) -> None:
        candidate = np.zeros((8, 8), dtype=bool)
        candidate[2:4, 5:7] = True
        truth = np.zeros((8, 8), dtype=bool)
        truth[2:4, 5:7] = True
        yolo = np.zeros((8, 8), dtype=bool)
        yolo[2:4, 1:3] = True
        confidence = np.full((8, 8), 0.8, dtype=np.float32)
        margin = np.full((8, 8), 0.2, dtype=np.float32)
        rows = component_records(
            {"obstacle": candidate},
            truth,
            yolo,
            confidence,
            margin,
            source_id="source",
            frame_id=4,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["truth_intersects"])
        self.assertEqual(rows[0]["truth_intersection_pixels"], 4)
        self.assertGreater(rows[0]["nearest_yolo_box_distance_pixels"], 0)
        self.assertEqual(component_metrics(candidate, truth)["component_recall"], 1.0)

    def test_motion_warp_and_temporal_lifecycle(self) -> None:
        previous = np.zeros((8, 8), dtype=bool)
        previous[2:4, 2:4] = True
        current = np.zeros((8, 8), dtype=bool)
        current[2:4, 3:5] = True
        raw = summarize_temporal([previous, current], frame_ids=[0, 1])
        warped = summarize_temporal(
            [previous, current],
            frame_ids=[0, 1],
            motion_warps=[[[1, 0, 1], [0, 1, 0]]],
        )
        self.assertLess(raw["raw_adjacent_iou"]["median"], warped["motion_warped_adjacent_iou"]["median"])
        self.assertTrue(warped["motion_warp_available"])
        shifted = warp_mask(previous, [[1, 0, 1], [0, 1, 0]])
        self.assertTrue(np.array_equal(shifted, current))

        empty = np.zeros((8, 8), dtype=bool)
        lifecycle = summarize_temporal([empty, previous, empty], frame_ids=[0, 1, 2])
        self.assertEqual(lifecycle["candidate_birth_count"], 1)
        self.assertEqual(lifecycle["candidate_death_count"], 1)
        self.assertEqual(lifecycle["persistence_frames"]["median"], 1.0)
        self.assertEqual(lifecycle["flicker_track_count"], 1)


class CandidateUtilityInputRegressionTests(unittest.TestCase):
    def _write_image(self, path: Path, color: tuple[int, int, int]) -> str:
        Image.new("RGB", (4, 4), color).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest

    def _row(self, image_path: Path, image_sha: str, *, frame_id: int, timestamp: int) -> dict[str, object]:
        return {
            "schema_version": "fixture",
            "source_id": "source",
            "frame_id": frame_id,
            "source_capture_timestamp_ns": timestamp,
            "image_path": image_path.name,
            "image_sha256": image_sha,
            "width": 4,
            "height": 4,
        }

    def test_manifest_duplicate_time_descending_and_sha_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "frame.png"
            image_sha = self._write_image(image, (10, 20, 30))

            duplicate_manifest = root / "duplicate.jsonl"
            duplicate_row = self._row(image, image_sha, frame_id=0, timestamp=1)
            duplicate_manifest.write_text(
                json.dumps(duplicate_row) + "\n" + json.dumps(duplicate_row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(CandidateUtilityInputError):
                load_manifest(duplicate_manifest, dataset_root=root, split=None, require_truth=False)

            descending_manifest = root / "descending.jsonl"
            first = self._row(image, image_sha, frame_id=0, timestamp=2)
            second = self._row(image, image_sha, frame_id=1, timestamp=1)
            descending_manifest.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(CandidateUtilityInputError):
                load_manifest(descending_manifest, dataset_root=root, split=None, require_truth=False)

            bad_sha_manifest = root / "bad-sha.jsonl"
            bad = self._row(image, "0" * 64, frame_id=0, timestamp=1)
            bad_sha_manifest.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaises(CandidateUtilityInputError):
                load_manifest(bad_sha_manifest, dataset_root=root, split=None, require_truth=False)

    def test_numpy_scalar_json_receipt_serialization_is_supported(self) -> None:
        encoded = json.dumps(
            {"count": np.int64(7), "fraction": np.float64(0.25), "array": np.array([1, 2])},
            default=_json_default,
        )
        self.assertEqual(json.loads(encoded), {"count": 7, "fraction": 0.25, "array": [1, 2]})

    def test_host_manifest_timestamp_order_is_per_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "frame.png"
            image_sha = self._write_image(image, (1, 2, 3))
            rows = []
            for source_id in ("source-a", "source-b"):
                for frame_id in range(2):
                    row = self._row(image, image_sha, frame_id=frame_id, timestamp=frame_id)
                    row["source_id"] = source_id
                    rows.append(row)
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            observations = load_host_manifest(manifest)
            self.assertEqual(len(observations), 4)

    def test_host_decoder_contract_matches_kotlin_fixed_tensor_contract(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        detector_source = (
            repo_root / "core" / "vision" / "src" / "main" / "java" / "com" / "linnan"
            / "blindassist" / "vision" / "TfliteYoloDetector.kt"
        ).read_text(encoding="utf-8")
        preprocessor_source = (
            repo_root / "core" / "vision" / "src" / "main" / "java" / "com" / "linnan"
            / "blindassist" / "vision" / "ImagePreprocessor.kt"
        ).read_text(encoding="utf-8")
        self.assertIn(f"const val INPUT_SIZE = {MODEL_INPUT_SIZE}", detector_source)
        self.assertIn(f"const val CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD}f", detector_source)
        self.assertIn(f"const val IOU_THRESHOLD = {NMS_IOU_THRESHOLD}f", detector_source)
        self.assertIn("inputTensor.dataType() == DataType.FLOAT32", detector_source)
        self.assertIn("outputTensor.dataType() == DataType.FLOAT32", detector_source)
        self.assertIn("putFloat(((pixel shr 16) and 0xFF) / 255f)", preprocessor_source)
        self.assertIn("putFloat(((pixel shr 8) and 0xFF) / 255f)", preprocessor_source)
        self.assertIn("putFloat((pixel and 0xFF) / 255f)", preprocessor_source)

    def _write_host_fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        image = root / "frame.png"
        image_sha = self._write_image(image, (40, 50, 60))
        manifest = root / "manifest.jsonl"
        manifest_row = {
            "schema_version": "fixture",
            "source_id": "source",
            "frame_id": 0,
            "source_capture_timestamp_ns": 0,
            "image_path": image.name,
            "image_sha256": image_sha,
            "width": 4,
            "height": 4,
        }
        manifest.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
        model = root / "model.tflite"
        model.write_bytes(b"model")
        labels = root / "labels.txt"
        labels.write_text("person\n", encoding="utf-8")
        output = root / "trace.jsonl"
        receipt = root / "receipt.json"
        progress = root / "progress.json"
        model_sha = host_sha256_file(model)
        labels_sha = host_sha256_file(labels)
        output_row = {
            "schema_version": HOST_TRACE_SCHEMA_VERSION,
            "source_id": "source",
            "frame_id": 0,
            "source_capture_timestamp_ns": 0,
            "image_sha256": image_sha,
            "detector_model_sha256": model_sha,
            "detector_labels_sha256": labels_sha,
            "input_size": MODEL_INPUT_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "detection_count": 0,
            "detections": [],
        }
        output.write_text(json.dumps(output_row) + "\n", encoding="utf-8")
        return manifest, model, labels, output, receipt

    def test_finalize_existing_rejects_missing_multiple_and_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, model, labels, output, receipt = self._write_host_fixture(root)
            progress = root / "progress.json"
            with self.assertRaises(HostTraceInputError):
                finalize_existing_trace(
                    repo_root=root,
                    manifest_path=manifest,
                    model_path=model,
                    labels_path=labels,
                    output_path=root / "missing.jsonl",
                    receipt_path=receipt,
                    progress_path=progress,
                    threads=1,
                )

            receipt.write_text("{}", encoding="utf-8")
            with self.assertRaises(HostTraceInputError):
                finalize_existing_trace(
                    repo_root=root,
                    manifest_path=manifest,
                    model_path=model,
                    labels_path=labels,
                    output_path=output,
                    receipt_path=receipt,
                    progress_path=progress,
                    threads=1,
                )

            receipt.unlink()
            row = json.loads(output.read_text(encoding="utf-8"))
            row["source_id"] = "wrong-source"
            output.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(HostTraceInputError):
                finalize_existing_trace(
                    repo_root=root,
                    manifest_path=manifest,
                    model_path=model,
                    labels_path=labels,
                    output_path=output,
                    receipt_path=receipt,
                    progress_path=progress,
                    threads=1,
                )


if __name__ == "__main__":
    unittest.main()
