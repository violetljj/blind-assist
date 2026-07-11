from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPTS = Path(__file__).resolve().parent


def module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


freeze = module("freeze_sanpo_v3_regression")
validator = module("validate_sanpo_v3_dataset")
views = module("prepare_sanpo_v3_dataset_views")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class SanpoV3DatasetControlsTest(unittest.TestCase):
    def make_v3_rows(self, root: Path) -> list[dict]:
        buckets = list(validator.SCENE_BUCKETS)
        sequences = [("train", bucket, 50) for bucket in buckets[:3]]
        sequences += [("dev", bucket, 50) for bucket in buckets[3:]]
        sequences += [("blind", "parallel_boundary", 60), ("blind", "low_light", 60)]
        rows: list[dict] = []
        counter = 0
        for seq_index, (split, bucket, frame_count) in enumerate(sequences):
            sequence_id = f"sequence_{seq_index}"
            for frame_index in range(frame_count):
                sample_id = f"sample_{counter:04d}"
                image_rel = Path("images") / f"{sample_id}.png"
                mask_rel = Path("semantic_masks") / f"{sample_id}.png"
                image = root / image_rel
                mask = root / mask_rel
                image.parent.mkdir(parents=True, exist_ok=True)
                mask.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (5, 5), (counter % 256, counter // 256, seq_index)).save(image)
                # A unique 5x5 base-4 pattern; aggregate coverage includes all classes.
                values = []
                value = counter
                for _ in range(25):
                    values.append(value % 4)
                    value //= 4
                Image.frombytes("L", (5, 5), bytes(values)).save(mask)
                rows.append({
                    "id": sample_id,
                    "split": split,
                    "session_id": f"session_{seq_index}",
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "scene_bucket": bucket,
                    "risk_event_id": f"event_{seq_index}",
                    "expected_event_phase": "ALERTED" if frame_index == 2 else "APPROACHING",
                    "expected_should_alert": frame_index >= 2,
                    "image_path": image_rel.as_posix(),
                    "semantic_mask_path": mask_rel.as_posix(),
                    "image_sha256": digest(image),
                    "semantic_mask_sha256": digest(mask),
                    "source": {
                        "dataset": "SANPO-Real" if seq_index < 6 else "authorized_local_capture",
                        "license": "CC BY 4.0" if seq_index < 6 else "authorized_local_capture",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/" if seq_index < 6 else "local-consent-record",
                        "privacy_review_status": "approved_no_identifiable_subjects",
                    },
                })
                counter += 1
        return rows

    def test_v3_coverage_and_blind_lock_accepts_separated_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            source = root / "reviewed.jsonl"
            write_jsonl(source, rows)
            original_argv = sys.argv
            try:
                sys.argv = ["prepare", "--source-manifest", str(source), "--dataset-root", str(root)]
                self.assertEqual(0, views.main())
            finally:
                sys.argv = original_argv
            train_rows = validator.load_jsonl(root / "training_manifest.jsonl")
            blind_rows = validator.load_jsonl(root / "blind_holdout" / "manifest.jsonl")
            errors, train_summary = validator.validate_rows(train_rows, root, {"train", "dev"})
            blind_errors, blind_summary = validator.validate_rows(blind_rows, root, {"blind"})
            self.assertEqual([], errors + blind_errors)
            self.assertEqual([], validator.validate_access_lock(root, train_rows, blind_rows))
            self.assertEqual([], validator.validate_v3_coverage(train_summary, blind_summary))

    def test_blind_metadata_and_session_leakage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = self.make_v3_rows(root)
            source = root / "reviewed.jsonl"
            write_jsonl(source, rows)
            original_argv = sys.argv
            try:
                sys.argv = ["prepare", "--source-manifest", str(source), "--dataset-root", str(root)]
                views.main()
            finally:
                sys.argv = original_argv
            train = validator.load_jsonl(root / "training_manifest.jsonl")
            blind = validator.load_jsonl(root / "blind_holdout" / "manifest.jsonl")
            train.append(blind[0])
            errors = validator.validate_access_lock(root, train, blind)
            self.assertTrue(any("blind metadata" in error or "overlap" in error for error in errors))
            blind[0]["session_id"] = train[0]["session_id"]
            self.assertTrue(any("session" in error and "leakage" in error for error in validator.validate_access_lock(root, train[:-1], blind)))
            errors, _ = validator.validate_rows(train, root, {"train", "dev"})
            blind_errors, _ = validator.validate_rows(blind, root, {"blind"})
            self.assertTrue(errors or blind_errors)

    def test_90_frame_regression_lock_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for directory in (source / "images" / "test", source / "source_masks" / "test"):
                directory.mkdir(parents=True, exist_ok=True)
            rows: list[dict] = []
            for index in range(90):
                sample_id = f"frame_{index:03d}"
                image = source / "images" / "test" / f"{sample_id}.png"
                mask = source / "source_masks" / "test" / f"{sample_id}.png"
                Image.new("RGB", (2, 2), (index, 0, 0)).save(image)
                Image.new("L", (2, 2), index % 4).save(mask)
                rows.append({"id": sample_id, "image_path": f"images/test/{sample_id}.png"})
            write_jsonl(source / "manifest.jsonl", rows)
            config = root / "benchmark-config.json"
            config.write_text('{"runs_per_frame":3}\n', encoding="utf-8")
            output = root / "sanpo-v3-regression-90f"
            result = subprocess.run([
                sys.executable, str(SCRIPTS / "freeze_sanpo_v3_regression.py"),
                "--source-root", str(source), "--output-root", str(output),
                "--benchmark-config", str(config), "--device-report", "device-report-20260711",
            ], text=True, capture_output=True)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertEqual([], freeze.verify_lock(output))
            Image.new("RGB", (2, 2), "white").save(output / "images" / "test" / "frame_000.png")
            self.assertTrue(any("SHA256 differs" in error for error in freeze.verify_lock(output)))


if __name__ == "__main__":
    unittest.main()
