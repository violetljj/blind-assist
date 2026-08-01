"""Regression tests for the manifest-driven model matrix runner."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.model_matrix.run_model_matrix import (
    OUTPUT_KEYS,
    TRACE_SCHEMA_VERSION,
    load_dataset_frames,
    run_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ModelMatrixRunnerTest(unittest.TestCase):
    def test_fixture_oracle_outputs_and_resume_prefix(self) -> None:
        temp_parent = REPO_ROOT / "artifacts.local" / "tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="model-matrix-test-", dir=str(temp_parent)
        ) as temp_name:
            root = Path(temp_name)
            output_root = root / "output"
            mask_path = root / "oracle-mask.bin"
            mask_path.write_bytes(b"synthetic-class-id-mask")
            mask_sha256 = hashlib.sha256(mask_path.read_bytes()).hexdigest()

            frames = []
            for index in range(3):
                frames.append(
                    {
                        "source_id": "synthetic-camera",
                        "sequence_id": "synthetic-sequence",
                        "frame_id": f"frame-{index}",
                        "frame_index": index,
                        "source_frame_index": index + 100,
                        "timestamp_ms": index * 100,
                        "secret_truth": "must-not-reach-ordinary-adapter",
                        "oracle_mask_path": str(mask_path),
                        "oracle_mask_sha256": mask_sha256,
                        "fixture_output": {
                            "detections": [{"class_id": 1, "score": 0.9}],
                            "risk_output": {
                                "status": "present",
                                "raw_level": "SAFE",
                                "stable_level": "SAFE",
                                "active": False,
                            },
                            "known": "KNOWN",
                            "clearance": 1.25,
                            "latency_ms": {
                                "preprocess": 0.1,
                                "inference": 1.2,
                                "postprocess": 0.2,
                                "total": 1.5,
                            },
                        },
                    }
                )

            dataset = {
                "dataset_id": "synthetic_dataset",
                "root": str(root),
                "format": "json_frames",
                "truth_fields": [
                    "secret_truth",
                    "oracle_mask_path",
                    "oracle_mask_sha256",
                ],
                "frames": frames,
            }
            model_registry = {
                "schema_version": "blindassist.model_matrix.model_registry.v1",
                "models": [
                    {
                        "model_id": "fixture_model",
                        "family": "fixture",
                        "evidence_role": "development_canary",
                        "adapter": {"kind": "fixture"},
                        "config": {"contract": "fixture-v1"},
                    },
                    {
                        "model_id": "oracle_model",
                        "family": "oracle_reference",
                        "evidence_role": "oracle_reference",
                        "adapter": {"kind": "truth_mask"},
                        "config": {"drives_alerts": False},
                    },
                ],
            }
            dataset_registry = {
                "schema_version": "blindassist.model_matrix.dataset_registry.v1",
                "datasets": [dataset],
            }
            manifest = {
                "schema_version": "blindassist.model_matrix.manifest.v1",
                "run_id": "SYNTHETIC_MODEL_MATRIX_TEST",
                "model_registry": str(root / "models.json"),
                "dataset_registry": str(root / "datasets.json"),
                "trace_schema": str(MODULE_ROOT / "trace_schema.json"),
                "output_root": str(output_root),
                "default_resolution": {"width": 64, "height": 36},
                "jobs": [
                    {
                        "job_id": "fixture_job",
                        "model_id": "fixture_model",
                        "dataset_id": "synthetic_dataset",
                        "mode": "run",
                    },
                    {
                        "job_id": "oracle_job",
                        "model_id": "oracle_model",
                        "dataset_id": "synthetic_dataset",
                        "mode": "run",
                    },
                ],
            }
            _write_json(root / "models.json", model_registry)
            _write_json(root / "datasets.json", dataset_registry)
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, manifest)

            first = run_matrix(
                manifest_path=manifest_path,
                repo_root=REPO_ROOT,
                output_override=str(output_root),
                max_frames=2,
            )
            self.assertEqual(first["status"], "COMPLETE")
            self.assertEqual(first["job_count"], 2)

            fixture_trace = output_root / "jobs" / "fixture_job" / "trace.jsonl"
            oracle_trace = output_root / "jobs" / "oracle_job" / "trace.jsonl"
            fixture_rows = _read_jsonl(fixture_trace)
            oracle_rows = _read_jsonl(oracle_trace)
            self.assertEqual(len(fixture_rows), 2)
            self.assertEqual(len(oracle_rows), 2)

            fixture_row = fixture_rows[0]
            self.assertEqual(fixture_row["schema_version"], TRACE_SCHEMA_VERSION)
            self.assertEqual(fixture_row["status"], "OK")
            self.assertEqual(fixture_row["known_status"], "KNOWN")
            self.assertEqual(set(fixture_row["outputs"]), set(OUTPUT_KEYS))
            self.assertEqual(fixture_row["outputs"]["detections"]["count"], 1)
            self.assertEqual(fixture_row["outputs"]["risk_output"]["status"], "present")
            self.assertEqual(fixture_row["clearance_status"], "present")
            self.assertRegex(fixture_row["model_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(fixture_row["config_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(fixture_row["truth_fields_read"], [])

            oracle_row = oracle_rows[0]
            self.assertEqual(oracle_row["known_status"], "KNOWN")
            self.assertEqual(oracle_row["outputs"]["mask"]["status"], "present")
            self.assertEqual(
                oracle_row["truth_fields_read"],
                ["oracle_mask_path", "oracle_mask_sha256"],
            )
            self.assertEqual(
                oracle_row["adapter_metadata"]["evidence_role"], "oracle_reference"
            )
            self.assertFalse(oracle_row["adapter_metadata"]["drives_alerts"])

            loaded_frame = load_dataset_frames(REPO_ROOT, dataset)[0]
            ordinary_input = loaded_frame.public_input(set(dataset["truth_fields"]))
            self.assertNotIn("secret_truth", ordinary_input)
            self.assertNotIn("oracle_mask_path", ordinary_input)

            resumed = run_matrix(
                manifest_path=manifest_path,
                repo_root=REPO_ROOT,
                output_override=str(output_root),
                max_frames=3,
            )
            self.assertEqual(resumed["status"], "COMPLETE")
            self.assertEqual(
                resumed["jobs"][0]["completed_frame_count"],
                3,
            )
            self.assertEqual(len(_read_jsonl(fixture_trace)), 3)
            self.assertEqual(len(_read_jsonl(oracle_trace)), 3)

            reused = run_matrix(
                manifest_path=manifest_path,
                repo_root=REPO_ROOT,
                output_override=str(output_root),
                max_frames=3,
            )
            self.assertEqual(reused["status"], "COMPLETE")
            self.assertTrue(all(job["reused_existing_trace"] for job in reused["jobs"]))
            self.assertEqual(len(_read_jsonl(fixture_trace)), 3)
            self.assertEqual(len(_read_jsonl(oracle_trace)), 3)


if __name__ == "__main__":
    unittest.main()
