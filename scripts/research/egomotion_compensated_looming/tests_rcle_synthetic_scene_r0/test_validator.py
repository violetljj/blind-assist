"""Tests for the isolated RCLE synthetic-scene validator."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_DIR = Path(__file__).resolve().parents[1] / "rcle_synthetic_scene_r0"
sys.path.insert(0, str(MODULE_DIR))
import validator  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidatorTest(unittest.TestCase):
    def make_dataset(self, root: Path) -> None:
        spec = {
            "rendering": {
                "width": 4,
                "height": 3,
                "intrinsics": {
                    "fx_pixels": 2.0,
                    "fy_pixels": 2.0,
                    "cx_pixels": 1.5,
                    "cy_pixels": 1.0,
                },
            }
        }
        (root / "dataset_spec.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
        spec_hash = sha(root / "dataset_spec.json")
        frames = []
        all_hashes = []
        for index in range(2):
            rgb_rel = f"frames/rgb_{index}.png"
            depth_rel = f"frames/depth_{index}.npy"
            (root / "frames").mkdir(exist_ok=True)
            Image.new("RGB", (4, 3), (10 + index, 20, 30)).save(root / rgb_rel)
            np.save(root / depth_rel, np.full((3, 4), 2.0 + index, np.float32))
            rgb_hash = sha(root / rgb_rel)
            depth_hash = sha(root / depth_rel)
            all_hashes.extend((rgb_hash, depth_hash))
            frames.append(
                {
                    "scene_id": "scene_dev",
                    "split": "development",
                    "sequence_id": "scene_dev/positive",
                    "motion_id": "positive",
                    "frame_index": index,
                    "timestamp_s": index / 10.0,
                    "spec_sha256": spec_hash,
                    "rgb_path": rgb_rel,
                    "rgb_sha256": rgb_hash,
                    "depth_npy_path": depth_rel,
                    "depth_npy_sha256": depth_hash,
                    "K": [[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
                    "T_world_camera": np.eye(4).tolist(),
                    "T_camera_world": np.eye(4).tolist(),
                }
            )
        scenes = [
            {
                "scene_id": "scene_dev",
                "split": "development",
                "sequence_id": "scene_dev/positive",
                "frame_count": 2,
                "pair_count": 1,
                "sequence_content_sha256": hashlib.sha256(
                    "".join(all_hashes).encode("ascii")
                ).hexdigest(),
                "spec_sha256": spec_hash,
            }
        ]
        (root / "scene_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in scenes), encoding="utf-8"
        )
        (root / "frame_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in frames), encoding="utf-8"
        )
        receipt = {
            "files": {
                name: {"sha256": sha(root / name)}
                for name in (
                    "dataset_spec.json",
                    "scene_manifest.jsonl",
                    "frame_manifest.jsonl",
                )
            }
        }
        (root / "receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

    def update_receipt_hash(self, root: Path, name: str) -> None:
        path = root / "receipt.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["files"][name]["sha256"] = sha(root / name)
        path.write_text(json.dumps(receipt), encoding="utf-8")

    def test_valid_dataset_passes_without_algorithm_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            report = validator.validate_dataset(root)
        self.assertEqual("PASS", report["status"])
        self.assertEqual(
            {"scenes": 1, "frames": 2, "sequences": 1}, report["counts"]
        )
        self.assertFalse(report["algorithm_results_present"])

    def test_frame_hash_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            Image.new("RGB", (4, 3), (255, 0, 0)).save(root / "frames/rgb_0.png")
            report = validator.validate_dataset(root)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("RGB hash mismatch", report["errors"][0])

    def test_scene_split_leakage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            manifest = root / "scene_manifest.jsonl"
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            rows.append(
                {
                    **rows[0],
                    "split": "sealed",
                    "sequence_id": "scene_dev/sealed_copy",
                }
            )
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.update_receipt_hash(root, manifest.name)
            report = validator.validate_dataset(root)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("scene leakage", report["errors"][0])

    def test_escaping_asset_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            manifest = root / "frame_manifest.jsonl"
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["rgb_path"] = "../outside.png"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.update_receipt_hash(root, manifest.name)
            report = validator.validate_dataset(root)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("escapes dataset root", report["errors"][0])

    def test_nonpositive_depth_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            depth = root / "frames/depth_0.npy"
            np.save(depth, np.zeros((3, 4), np.float32))
            manifest = root / "frame_manifest.jsonl"
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["depth_npy_sha256"] = sha(depth)
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.update_receipt_hash(root, manifest.name)
            report = validator.validate_dataset(root)
        self.assertEqual("FAIL", report["status"])
        self.assertIn("finite and positive", report["errors"][0])

    def test_pose_inverse_and_timestamp_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            manifest = root / "frame_manifest.jsonl"
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["T_camera_world"][0][3] = 1.0
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.update_receipt_hash(root, manifest.name)
            pose_report = validator.validate_dataset(root)
            self.assertIn("not inverses", pose_report["errors"][0])

            self.make_dataset(root)
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            rows[1]["timestamp_s"] = rows[0]["timestamp_s"]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.update_receipt_hash(root, manifest.name)
            timestamp_report = validator.validate_dataset(root)
        self.assertEqual("FAIL", pose_report["status"])
        self.assertEqual("FAIL", timestamp_report["status"])
        self.assertIn("strictly increasing", timestamp_report["errors"][0])


if __name__ == "__main__":
    unittest.main()
