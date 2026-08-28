"""Measured and auditable backend selection for BlindAssist experiments.

The selector deliberately does not benchmark a synthetic CUDA kernel.  Callers
provide small, equivalent probes from the real workload so the decision reflects
model inference, tensor work, or point-cloud matching that the experiment will
actually execute.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA = "blindassist-execution-backend-v1"


class BackendSelectionError(RuntimeError):
    """The backend contract could not produce an honest launch decision."""


class Workload(str, Enum):
    MODEL_INFERENCE = "model-inference"
    BATCH_TENSOR = "batch-tensor"
    POINT_CLOUD_MATCHING = "point-cloud-matching"
    IO_ETL = "io-etl"
    ARCHIVE = "archive"
    JSON = "json"
    SCALAR_SCORING = "scalar-scoring"


GPU_FIRST_WORKLOADS = frozenset(
    {
        Workload.MODEL_INFERENCE,
        Workload.BATCH_TENSOR,
        Workload.POINT_CLOUD_MATCHING,
    }
)
CPU_WORKLOADS = frozenset(
    {
        Workload.IO_ETL,
        Workload.ARCHIVE,
        Workload.JSON,
        Workload.SCALAR_SCORING,
    }
)
CPU_REASON_CODES = frozenset(
    {
        "CPU_FASTER_MEASURED",
        "TASK_NOT_GPU_SUITABLE",
        "ACCELERATOR_UNAVAILABLE",
        "GPU_BACKEND_UNAVAILABLE",
        "FROZEN_PROTOCOL_CPU_ONLY",
    }
)


@dataclass(frozen=True)
class DeviceObservation:
    device_type: str
    device_name: str
    framework: str
    providers: tuple[str, ...] = ()

    def normalized_type(self) -> str:
        value = self.device_type.casefold()
        return "cuda" if value.startswith("cuda") else value


@dataclass(frozen=True)
class BackendCandidate:
    name: str
    expected_device_type: str
    run_probe: Callable[[], Any]
    observe: Callable[[Any], DeviceObservation]
    synchronize: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    expected_device_type: str
    actual_device_type: str
    actual_device_name: str
    framework: str
    providers: tuple[str, ...]
    samples_seconds: tuple[float, ...]
    median_seconds: float


def runtime_capabilities() -> dict[str, Any]:
    capabilities: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": {"available": False},
        "onnxruntime": {"available": False},
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        capabilities["torch"] = {
            "available": True,
            "version": str(torch.__version__),
            "cuda_available": cuda_available,
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    except (ImportError, OSError, RuntimeError) as error:
        capabilities["torch"] = {
            "available": False,
            "error_type": type(error).__name__,
        }
    try:
        import onnxruntime as ort

        capabilities["onnxruntime"] = {
            "available": True,
            "version": str(ort.__version__),
            "available_providers": list(ort.get_available_providers()),
        }
    except (ImportError, OSError, RuntimeError) as error:
        capabilities["onnxruntime"] = {
            "available": False,
            "error_type": type(error).__name__,
        }
    return capabilities


def _validate_observation(
    candidate: BackendCandidate, observation: DeviceObservation
) -> None:
    expected = candidate.expected_device_type.casefold()
    actual = observation.normalized_type()
    if expected.startswith("cuda"):
        expected = "cuda"
    if actual != expected:
        raise BackendSelectionError(
            "SILENT_DEVICE_FALLBACK: "
            f"backend={candidate.name} expected={expected} actual={actual} "
            f"device={observation.device_name}"
        )
    if expected == "cuda" and "CPUExecutionProvider" in observation.providers:
        if "CUDAExecutionProvider" not in observation.providers:
            raise BackendSelectionError(
                "SILENT_PROVIDER_FALLBACK: CUDA candidate exposes only CPUExecutionProvider"
            )


def benchmark(
    candidate: BackendCandidate,
    *,
    warmups: int = 1,
    repeats: int = 3,
    timer: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be >= 0 and repeats must be >= 1")
    observation: DeviceObservation | None = None
    for _ in range(warmups):
        output = candidate.run_probe()
        candidate.synchronize()
        observation = candidate.observe(output)
        _validate_observation(candidate, observation)
    samples = []
    for _ in range(repeats):
        started = timer()
        output = candidate.run_probe()
        candidate.synchronize()
        elapsed = timer() - started
        observation = candidate.observe(output)
        _validate_observation(candidate, observation)
        samples.append(float(elapsed))
    assert observation is not None
    return BenchmarkResult(
        backend=candidate.name,
        expected_device_type=candidate.expected_device_type,
        actual_device_type=observation.normalized_type(),
        actual_device_name=observation.device_name,
        framework=observation.framework,
        providers=observation.providers,
        samples_seconds=tuple(samples),
        median_seconds=float(statistics.median(samples)),
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def select_backend(
    workload: Workload | str,
    *,
    cpu: BackendCandidate,
    gpu: BackendCandidate | None = None,
    cpu_reason: str | None = None,
    record_path: Path | None = None,
    warmups: int = 1,
    repeats: int = 3,
    timer: Callable[[], float] = time.perf_counter,
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workload = Workload(workload)
    measured: list[BenchmarkResult] = []

    if workload in CPU_WORKLOADS:
        cpu_result = benchmark(cpu, warmups=0, repeats=1, timer=timer)
        measured.append(cpu_result)
        selected = cpu_result
        reason = f"CPU_TASK_CLASS_{workload.value.upper().replace('-', '_')}"
        fallback = False
    elif workload in GPU_FIRST_WORKLOADS:
        if gpu is None:
            if cpu_reason not in CPU_REASON_CODES - {"CPU_FASTER_MEASURED"}:
                raise BackendSelectionError(
                    "GPU_FIRST_WORKLOAD_REQUIRES_GPU_PROBE_OR_EXPLICIT_CPU_REASON"
                )
            cpu_result = benchmark(cpu, warmups=0, repeats=1, timer=timer)
            measured.append(cpu_result)
            selected = cpu_result
            reason = str(cpu_reason)
            fallback = True
        else:
            cpu_result = benchmark(
                cpu, warmups=warmups, repeats=repeats, timer=timer
            )
            gpu_result = benchmark(
                gpu, warmups=warmups, repeats=repeats, timer=timer
            )
            measured.extend((cpu_result, gpu_result))
            if gpu_result.median_seconds <= cpu_result.median_seconds:
                selected = gpu_result
                reason = "GPU_FASTER_OR_EQUAL_MEASURED"
                fallback = False
            else:
                selected = cpu_result
                reason = "CPU_FASTER_MEASURED"
                fallback = True
    else:  # pragma: no cover - exhaustive Enum guard
        raise BackendSelectionError(f"UNKNOWN_WORKLOAD:{workload}")

    record = {
        "schema": SCHEMA,
        "workload": workload.value,
        "selection_status": "SELECTED",
        "selected_backend": selected.backend,
        "selected_device_type": selected.actual_device_type,
        "selected_device_name": selected.actual_device_name,
        "selected_framework": selected.framework,
        "selected_providers": list(selected.providers),
        "selection_reason": reason,
        "cpu_fallback": fallback,
        "benchmarks": [asdict(item) for item in measured],
        "runtime_capabilities": dict(capabilities or runtime_capabilities()),
    }
    if selected.actual_device_type == "cpu" and workload in GPU_FIRST_WORKLOADS:
        if reason not in CPU_REASON_CODES:
            raise BackendSelectionError("CPU_SELECTION_MISSING_EXPLICIT_REASON")
    if record_path is not None:
        _atomic_write_json(record_path, record)
    print(json.dumps({"execution_backend": record}, ensure_ascii=False), flush=True)
    return record


def torch_observation(
    *, model: Any | None = None, output: Any | None = None
) -> DeviceObservation:
    import torch

    devices: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            devices.add(str(value.device))
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    if model is not None:
        try:
            devices.update(str(parameter.device) for parameter in model.parameters())
        except (AttributeError, TypeError):
            pass
    visit(output)
    if not devices:
        raise BackendSelectionError("TORCH_ACTUAL_DEVICE_NOT_OBSERVABLE")
    normalized = {"cuda" if value.startswith("cuda") else value for value in devices}
    if len(normalized) != 1:
        raise BackendSelectionError(f"TORCH_MIXED_DEVICE_EXECUTION:{sorted(devices)}")
    device_type = next(iter(normalized))
    device_name = (
        torch.cuda.get_device_name(0) if device_type == "cuda" else platform.processor() or "CPU"
    )
    return DeviceObservation(device_type, device_name, f"torch-{torch.__version__}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("capabilities",))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"schema": SCHEMA, "runtime_capabilities": runtime_capabilities()}
    if args.output:
        _atomic_write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
