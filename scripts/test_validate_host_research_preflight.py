from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_host_research_preflight.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_payload(script: Path, execution_class: str = "long") -> dict:
    payload = {
        "schema_version": "blindassist.host_research_preflight.v1",
        "task_id": "TEST-HOST-LONG-R0",
        "execution_class": execution_class,
        "implementation": {
            "script": script.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(script),
        },
        "workload": {
            "class": "cpu_data_parallel",
            "real_data_mechanics_match": True,
            "input_identity": "fixture-sha256:" + ("a" * 64),
        },
        "pilot": {
            "representative_units": 10,
            "wall_seconds": 2.0,
            "projected_full_units": 100,
            "projected_full_wall_seconds": 20.0,
            "maximum_expected_wall_seconds": 30.0,
            "same_access_mechanics": True,
            "output_equivalence": "PASS",
            "progress_samples": 2,
        },
        "scheduler": {
            "backend": "cpu_process_pool",
            "workers": 8,
            "reason": "8 workers beat the serial pilot without memory pressure",
            "comparison_performed": True,
            "scientific_parameters_unchanged": True,
            "estimated_gib_per_worker": 0.25,
            "reserve_memory_gib": 2.5,
            "requires_ac_power": False,
            "inject_workers": True,
        },
        "progress": {
            "path": "artifacts.local/evidence/test/progress.json",
            "fields": [
                "phase",
                "completed_units",
                "total_units",
                "throughput",
                "eta_seconds",
                "last_progress_at",
                "status",
            ],
            "update_interval_seconds": 10,
            "verified_in_pilot": True,
        },
        "terminal": {
            "success_path": "artifacts.local/evidence/test/result.json",
            "failure_path": "artifacts.local/evidence/test/failure.json",
        },
    }
    if execution_class == "formal":
        payload["formal"] = {
            "one_shot": True,
            "claim_created_by_runner_only": True,
            "claim_path": "artifacts.local/evidence/test/run.claim.json",
            "output_path": "artifacts.local/evidence/test/result.json",
            "failure_receipt_path": (
                "artifacts.local/evidence/test/failure.json"
            ),
            "activation_authority": "TEST_ACTIVATION_LOCK",
        }
    return payload


class HostResearchPreflightTest(unittest.TestCase):
    def _run(self, payload: dict) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--repo-root",
                    str(REPO_ROOT),
                    "--receipt",
                    str(receipt),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        return completed.returncode, json.loads(completed.stdout)

    def test_valid_long_receipt_passes(self) -> None:
        code, result = self._run(_valid_payload(VALIDATOR))
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "QUALIFIED")
        self.assertEqual(result["errors"], [])

    def test_formal_receipt_requires_formal_contract(self) -> None:
        code, result = self._run(
            _valid_payload(VALIDATOR, execution_class="formal")
            | {"formal": {}}
        )
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "PERFORMANCE_NOT_QUALIFIED")
        self.assertTrue(
            any("formal.one_shot" in error for error in result["errors"])
        )

    def test_rejects_black_box_progress(self) -> None:
        payload = _valid_payload(VALIDATOR)
        payload["pilot"]["progress_samples"] = 1
        payload["progress"]["verified_in_pilot"] = False
        code, result = self._run(payload)
        self.assertEqual(code, 2)
        self.assertTrue(
            any("progress_samples" in error for error in result["errors"])
        )
        self.assertTrue(
            any("verified_in_pilot" in error for error in result["errors"])
        )

    def test_rejects_unbounded_projection(self) -> None:
        payload = _valid_payload(VALIDATOR)
        payload["pilot"]["maximum_expected_wall_seconds"] = 200.0
        code, result = self._run(payload)
        self.assertEqual(code, 2)
        self.assertTrue(
            any("exceeds 4x" in error for error in result["errors"])
        )

    def test_rejects_script_hash_drift(self) -> None:
        payload = _valid_payload(VALIDATOR)
        payload["implementation"]["sha256"] = "0" * 64
        code, result = self._run(payload)
        self.assertEqual(code, 2)
        self.assertTrue(
            any("does not match" in error for error in result["errors"])
        )

    def test_cuda_requires_vram_budget(self) -> None:
        payload = _valid_payload(VALIDATOR)
        payload["scheduler"]["backend"] = "cuda"
        code, result = self._run(payload)
        self.assertEqual(code, 2)
        self.assertTrue(
            any("minimum_free_vram_gib" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
