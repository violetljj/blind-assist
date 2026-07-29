from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    p4_formal_runner_r0 as formal,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(formal.canonical_bytes(value))


class P4FormalRunnerTests(unittest.TestCase):
    def test_resource_state_machine_preserves_8_6_4_gib_gates(self) -> None:
        self.assertEqual(
            formal.resource_action(
                8 * formal.GIB, in_flight=0, paging_positive_streak=0
            ),
            "REFILL",
        )
        self.assertEqual(
            formal.resource_action(
                7 * formal.GIB, in_flight=2, paging_positive_streak=0
            ),
            "WAIT_NO_REFILL",
        )
        self.assertEqual(
            formal.resource_action(
                7 * formal.GIB, in_flight=0, paging_positive_streak=0
            ),
            "PAUSE_RESUMABLE",
        )
        self.assertEqual(
            formal.resource_action(
                5 * formal.GIB, in_flight=2, paging_positive_streak=0
            ),
            "DRAIN",
        )
        self.assertEqual(
            formal.resource_action(
                3 * formal.GIB, in_flight=2, paging_positive_streak=0
            ),
            "STOP_RESUMABLE",
        )
        self.assertEqual(
            formal.resource_action(
                10 * formal.GIB, in_flight=2, paging_positive_streak=2
            ),
            "STOP_RESUMABLE",
        )

    def test_claim_exact_resume_and_drift_fail_closed(self) -> None:
        authority = {
            "activation_sha256": "a" * 64,
            "identity_manifest_sha256": "b" * 64,
            "manipulation_receipt_sha256": "c" * 64,
            "manipulation_producer_receipt_sha256": "f" * 64,
            "trajectory_manifest_sha256": "d" * 64,
            "runner_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = formal.ensure_claim(output, "run-r0", authority)
            second = formal.ensure_claim(output, "run-r0", authority)
            self.assertEqual(first, second)
            corrupted = dict(first)
            corrupted["activation_sha256"] = "e" * 64
            write_json(output / "claim.json", corrupted)
            with self.assertRaisesRegex(
                formal.InvalidFormalEvidence, "FORMAL_CLAIM_DRIFT"
            ):
                formal.ensure_claim(output, "run-r0", authority)

    def test_incomplete_staging_is_quarantined_and_restarted(self) -> None:
        identity = {
            "sequence_id": "FORMAL_ADVIO_13_MAIN_00__STATIC_CAMERA__CLEAN"
        }
        authority = {
            "activation_sha256": "a" * 64,
            "identity_manifest_sha256": "b" * 64,
            "manipulation_receipt_sha256": "c" * 64,
            "manipulation_producer_receipt_sha256": "f" * 64,
            "trajectory_manifest_sha256": "d" * 64,
            "runner_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            staging = (
                output
                / "staging"
                / f"{identity['sequence_id']}.attempt.tmp"
            )
            staging.mkdir(parents=True)
            (staging / "partial.bin").write_bytes(b"not reusable")
            completed, pending = formal.recover_state(
                output, [identity], authority, "run-r0"
            )
            self.assertEqual(completed, [])
            self.assertEqual(pending, [identity])
            interruption = next(
                (output / "quarantine").glob(
                    "*.incomplete/interruption.json"
                )
            )
            value = json.loads(interruption.read_text(encoding="utf-8"))
            self.assertEqual(
                value["disposition"], "RESTART_FROM_FRAME_ZERO"
            )
            self.assertFalse(value["outcome_values_emitted"])

    def test_completed_arm_corruption_is_invalid_not_recomputed(self) -> None:
        identity = {
            "sequence_id": "FORMAL_ADVIO_13_MAIN_00__STATIC_CAMERA__CLEAN",
            "cluster_id": "ADVIO_13__MAIN_00",
            "block": "ADVIO_13",
            "ordinal": 0,
            "role": "MAIN_FACTORIAL",
            "arm": "STATIC_CAMERA__CLEAN",
            "numeric_seed_uint64": 1,
        }
        authority = {
            "activation_sha256": "a" * 64,
            "identity_manifest_sha256": "b" * 64,
            "manipulation_receipt_sha256": "c" * 64,
            "manipulation_producer_receipt_sha256": "f" * 64,
            "trajectory_manifest_sha256": "d" * 64,
            "runner_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arm = output / "arms" / identity["sequence_id"]
            arm.mkdir(parents=True)
            write_json(
                arm / "receipt.json",
                {
                    "schema": (
                        "rcle.periodic_self_motion_counterfactual."
                        "p4_arm_receipt.v1"
                    ),
                    "protocol_id": formal.PROTOCOL_ID,
                    "run_id": "run-r0",
                    **authority,
                    **identity,
                    "frame_count": formal.FRAME_COUNT,
                    "pair_count": formal.PAIR_COUNT,
                    "terminal": "ARM_COMPLETE",
                    "rgb_frames_retained": False,
                    "sequence16_android_realtime": False,
                    "pair_ledger_path": (
                        f"arms/{identity['sequence_id']}/pair_ledger.jsonl"
                    ),
                    "pair_ledger_sha256": "0" * 64,
                    "frame_manifest_path": (
                        f"arms/{identity['sequence_id']}/frame_manifest.jsonl"
                    ),
                    "frame_manifest_sha256": "0" * 64,
                    "reduced_metrics_path": (
                        f"arms/{identity['sequence_id']}/reduced_metrics.json"
                    ),
                    "reduced_metrics_sha256": "0" * 64,
                },
            )
            with self.assertRaisesRegex(
                formal.InvalidFormalEvidence, "ARM_ARTIFACT_DRIFT"
            ):
                formal.recover_state(
                    output, [identity], authority, "run-r0"
                )

    def test_authority_is_validated_before_numeric_import_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("activation.json", "manipulation.json", "identity.json", "trajectory.json"):
                write_json(root / name, {})
            task = {
                "activation_path": str(root / "activation.json"),
                "manipulation_path": str(root / "manipulation.json"),
                "identity_path": str(root / "identity.json"),
                "trajectory_path": str(root / "trajectory.json"),
                "output_dir": str(root / "output"),
                "run_id": "run-r0",
                "identity": {},
                "attempt_id": "attempt",
            }
            with self.assertRaisesRegex(
                formal.InvalidFormalEvidence, "ACTIVATION_PROTOCOL"
            ):
                formal._evaluate_arm(task)
            self.assertFalse((root / "output").exists())

    def test_progress_and_telemetry_have_no_outcome_fields(self) -> None:
        resource = {
            "available_ram_bytes": 9 * formal.GIB,
            "resident_memory_bytes": 1,
            "read_bytes": 2,
            "write_bytes": 3,
            "swap_in_total": 0,
            "swap_out_total": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            formal._write_progress(
                output, 12, 8, 1.0, resource, "RUNNING"
            )
            payload = json.loads(
                (output / "progress.json").read_text(encoding="utf-8")
            )
            serialized = json.dumps(payload).lower()
            self.assertNotIn("trigger", serialized)
            self.assertNotIn("response", serialized)
            self.assertFalse(payload["outcome_fields_present"])
            self.assertEqual(payload["completed_pairs"], 12 * 601)
            formal._append_telemetry(
                output,
                resource=resource,
                completed_count=12,
                in_flight=8,
                action="REFILL",
                started=1.0,
            )
            telemetry = json.loads(
                (output / "telemetry.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            telemetry_text = json.dumps(telemetry).lower()
            self.assertNotIn("trigger", telemetry_text)
            self.assertNotIn("response", telemetry_text)
            self.assertFalse(telemetry["outcome_fields_present"])

    def test_atomic_paths_never_include_rgb_payload(self) -> None:
        staging, final = formal._worker_paths(
            Path("evidence"), "FORMAL_X", "attempt"
        )
        self.assertTrue(str(staging).endswith("FORMAL_X.attempt.tmp"))
        self.assertTrue(str(final).endswith("arms\\FORMAL_X") or str(final).endswith("arms/FORMAL_X"))
        self.assertNotIn("rgb", str(staging).lower())


if __name__ == "__main__":
    unittest.main()
