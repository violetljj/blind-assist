from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_backend import (
    BackendCandidate,
    BackendSelectionError,
    DeviceObservation,
    Workload,
    select_backend,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def candidate(
    name: str,
    expected: str,
    actual: str,
    seconds: float,
    clock: FakeClock,
) -> BackendCandidate:
    def run() -> DeviceObservation:
        clock.advance(seconds)
        return DeviceObservation(actual, f"{actual}-device", "fake")

    return BackendCandidate(name, expected, run, lambda output: output)


class ResearchBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.capabilities = {"test": True}

    def test_gpu_first_workload_selects_faster_gpu_and_records_device(self) -> None:
        clock = FakeClock()
        record = select_backend(
            Workload.POINT_CLOUD_MATCHING,
            cpu=candidate("numpy", "cpu", "cpu", 4.0, clock),
            gpu=candidate("torch-cuda", "cuda", "cuda:0", 1.0, clock),
            warmups=0,
            repeats=2,
            timer=clock.now,
            capabilities=self.capabilities,
        )
        self.assertEqual("torch-cuda", record["selected_backend"])
        self.assertEqual("cuda", record["selected_device_type"])
        self.assertFalse(record["cpu_fallback"])

    def test_measured_faster_cpu_has_explicit_reason(self) -> None:
        clock = FakeClock()
        record = select_backend(
            Workload.MODEL_INFERENCE,
            cpu=candidate("cpu", "cpu", "cpu", 1.0, clock),
            gpu=candidate("cuda", "cuda", "cuda", 2.0, clock),
            warmups=0,
            repeats=2,
            timer=clock.now,
            capabilities=self.capabilities,
        )
        self.assertEqual("CPU_FASTER_MEASURED", record["selection_reason"])
        self.assertTrue(record["cpu_fallback"])
        self.assertEqual(2, len(record["benchmarks"]))

    def test_gpu_capable_workload_cannot_silently_omit_gpu(self) -> None:
        clock = FakeClock()
        with self.assertRaisesRegex(
            BackendSelectionError, "GPU_FIRST_WORKLOAD_REQUIRES"
        ):
            select_backend(
                Workload.BATCH_TENSOR,
                cpu=candidate("cpu", "cpu", "cpu", 1.0, clock),
                timer=clock.now,
                capabilities=self.capabilities,
            )

    def test_explicit_unavailable_accelerator_fallback_is_recorded(self) -> None:
        clock = FakeClock()
        record = select_backend(
            Workload.BATCH_TENSOR,
            cpu=candidate("cpu", "cpu", "cpu", 1.0, clock),
            cpu_reason="ACCELERATOR_UNAVAILABLE",
            timer=clock.now,
            capabilities=self.capabilities,
        )
        self.assertEqual("ACCELERATOR_UNAVAILABLE", record["selection_reason"])
        self.assertTrue(record["cpu_fallback"])

    def test_declared_cuda_candidate_cannot_report_cpu(self) -> None:
        clock = FakeClock()
        with self.assertRaisesRegex(BackendSelectionError, "SILENT_DEVICE_FALLBACK"):
            select_backend(
                Workload.MODEL_INFERENCE,
                cpu=candidate("cpu", "cpu", "cpu", 2.0, clock),
                gpu=candidate("cuda", "cuda", "cpu", 1.0, clock),
                warmups=0,
                repeats=1,
                timer=clock.now,
                capabilities=self.capabilities,
            )

    def test_cuda_onnx_candidate_cannot_expose_only_cpu_provider(self) -> None:
        clock = FakeClock()

        def run() -> DeviceObservation:
            clock.advance(1.0)
            return DeviceObservation(
                "cuda",
                "declared-cuda-device",
                "onnxruntime",
                ("CPUExecutionProvider",),
            )

        gpu = BackendCandidate("onnx-cuda", "cuda", run, lambda output: output)
        with self.assertRaisesRegex(
            BackendSelectionError, "SILENT_PROVIDER_FALLBACK"
        ):
            select_backend(
                Workload.MODEL_INFERENCE,
                cpu=candidate("cpu", "cpu", "cpu", 2.0, clock),
                gpu=gpu,
                warmups=0,
                repeats=1,
                timer=clock.now,
                capabilities=self.capabilities,
            )

    def test_cpu_task_class_does_not_require_gpu_benchmark(self) -> None:
        clock = FakeClock()
        record = select_backend(
            Workload.JSON,
            cpu=candidate("stdlib-json", "cpu", "cpu", 1.0, clock),
            timer=clock.now,
            capabilities=self.capabilities,
        )
        self.assertEqual("CPU_TASK_CLASS_JSON", record["selection_reason"])
        self.assertFalse(record["cpu_fallback"])

    def test_record_is_persisted_atomically(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.json"
            select_backend(
                Workload.IO_ETL,
                cpu=candidate("stdlib", "cpu", "cpu", 1.0, clock),
                record_path=path,
                timer=clock.now,
                capabilities=self.capabilities,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("blindassist-execution-backend-v1", payload["schema"])
            self.assertEqual([], list(path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
