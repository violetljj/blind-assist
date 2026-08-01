from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.riskseg_r0_pidnet_preflight.validate_device_preflight import (
    RUN_ROLE_TECHNICAL_PREFLIGHT,
    RUN_ROLE_TRAINED_FINAL,
    validate,
)


MODEL_SHA256 = "a" * 64


def receipt(*, run_role: str, cache_before: list[dict]) -> dict:
    status = (
        "QNN_HTP_FORMAL_SUSTAINED_TRAINED_FINAL_PASS"
        if run_role == RUN_ROLE_TRAINED_FINAL
        else "QNN_HTP_FORMAL_SUSTAINED_PREFLIGHT_PASS"
    )
    return {
        "status": status,
        "run_role": run_role,
        "formal_sustained_run": True,
        "duration_observed_ms": 600_001,
        "failure_count": 0,
        "gates": {
            "failure_count_zero": True,
            "total_p95_at_most_100_ms": True,
            "degradation_at_most_1_20x": True,
            "no_severe_thermal": True,
            "qnn_cached_context_created": True,
            "strict_int8_tensor_contract": True,
            "argmax_in_0_to_3": True,
        },
        "device": {"model": "SM-S9280", "soc_model": "SM8650"},
        "delegate": {
            "backend": "QNN_HTP",
            "precision": "HTP_PRECISION_QUANTIZED",
            "capability": "HTP_RUNTIME_QUANTIZED",
        },
        "model": {
            "sha256": MODEL_SHA256,
            "class_order": [
                "walkable",
                "blocking_obstacle",
                "boundary_level_change",
                "unknown_nonwalkable",
            ],
        },
        "qnn_cached_context": {"before": cache_before},
        "sample_count": 100,
        "timing_ms": {
            "total": {"p95": 75.0},
            "initial_window_p95": 75.0,
            "final_window_p95": 76.0,
            "final_over_initial_p95_ratio": 1.01,
            "inference": {"p95": 5.0},
        },
        "thermal": {"maximum_status": 0},
    }


def full_delegation(nodes: int) -> str:
    return (
        "TfLiteQnnDelegate delegate: "
        f"{nodes} nodes delegated out of {nodes} nodes with 1 partitions."
    )


class ValidateDevicePreflightTest(unittest.TestCase):
    def run_validation(
        self, value: dict, logcat: str, *, run_role: str
    ) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "receipt.json"
            logcat_path = root / "logcat.txt"
            receipt_path.write_text(json.dumps(value), encoding="utf-8")
            logcat_path.write_text(logcat, encoding="utf-8")
            return validate(
                receipt_path,
                logcat_path,
                expected_model_sha256=MODEL_SHA256,
                expected_run_role=run_role,
            )

    def test_clean_trained_cache_accepts_save_then_restore_and_dynamic_nodes(self) -> None:
        logcat = "\n".join(
            [
                "caching in SAVE MODE.",
                full_delegation(173),
                "caching in RESTORE MODE.",
                full_delegation(173),
            ]
        )
        result = self.run_validation(
            receipt(run_role=RUN_ROLE_TRAINED_FINAL, cache_before=[]),
            logcat,
            run_role=RUN_ROLE_TRAINED_FINAL,
        )
        self.assertEqual("PIDNET_S_TRAINED_FINAL_DEVICE_PASS", result["status"])
        self.assertEqual(2, result["full_delegation_marker_count"])
        self.assertEqual("CLEAN_CACHE_SAVE_THEN_RESTORE", result["expected_cache_log_lifecycle"])

    def test_preexisting_preflight_cache_keeps_two_restore_contract(self) -> None:
        logcat = "\n".join(
            [
                "caching in RESTORE MODE.",
                full_delegation(163),
                "caching in RESTORE MODE.",
                full_delegation(163),
            ]
        )
        result = self.run_validation(
            receipt(
                run_role=RUN_ROLE_TECHNICAL_PREFLIGHT,
                cache_before=[{"relative_path": "context.bin"}],
            ),
            logcat,
            run_role=RUN_ROLE_TECHNICAL_PREFLIGHT,
        )
        self.assertEqual("PIDNET_S_TECHNICAL_PREFLIGHT_PASS", result["status"])
        self.assertEqual(
            "PREEXISTING_CACHE_RESTORED_TWICE",
            result["expected_cache_log_lifecycle"],
        )

    def test_partial_delegation_still_fails(self) -> None:
        logcat = "\n".join(
            [
                "caching in SAVE MODE.",
                "TfLiteQnnDelegate delegate: 172 nodes delegated out of 173 nodes "
                "with 1 partitions.",
                "caching in RESTORE MODE.",
                "TfLiteQnnDelegate delegate: 172 nodes delegated out of 173 nodes "
                "with 1 partitions.",
            ]
        )
        result = self.run_validation(
            receipt(run_role=RUN_ROLE_TRAINED_FINAL, cache_before=[]),
            logcat,
            run_role=RUN_ROLE_TRAINED_FINAL,
        )
        self.assertEqual("PIDNET_S_TRAINED_FINAL_DEVICE_INVALID", result["status"])
        self.assertFalse(result["checks"]["full_graph_delegated_twice"])


if __name__ == "__main__":
    unittest.main()
