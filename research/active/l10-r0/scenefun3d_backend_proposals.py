from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from research_backend import (  # noqa: E402
    BackendCandidate,
    DeviceObservation,
    Workload,
    runtime_capabilities,
    select_backend,
    torch_observation,
)
from scenefun3d_functional_handoff_ceiling import (  # noqa: E402
    PARENT_CONTAINMENT_MARGIN_M,
    PARENT_CONTAINMENT_MIN_FRACTION,
    FunctionalProposal,
    _load_json,
    _load_parent_boxes,
    _load_ply_xyz,
)
from scenefun3d_functional_set_integrity import _build_proposals  # noqa: E402


@dataclass(frozen=True)
class FunctionalCenter:
    candidate_id: str
    label: str
    center: np.ndarray


def _torch_build_proposals(
    paths: dict[str, Path], device: str = "cuda"
) -> tuple[dict[str, FunctionalProposal], int, Any]:
    import torch

    xyz = _load_ply_xyz(paths["laser_scan"])
    transform = torch.as_tensor(
        np.load(paths["transform"]), dtype=torch.float64, device=device
    )
    parents = _load_parent_boxes(paths["object_boxes"])
    centers = torch.as_tensor(
        np.stack([row.center for row in parents]), dtype=torch.float64, device=device
    )
    lengths = torch.as_tensor(
        np.stack([row.lengths for row in parents]), dtype=torch.float64, device=device
    )
    axes = torch.as_tensor(
        np.stack([row.axes for row in parents]), dtype=torch.float64, device=device
    )
    proposals: dict[str, FunctionalProposal] = {}
    unmatched = 0
    sentinel = torch.empty(1, device=device)
    for annotation in _load_json(paths["annotations"])["annotations"]:
        if annotation["label"] == "exclude":
            continue
        source = torch.as_tensor(
            np.asarray(xyz[np.asarray(annotation["indices"], dtype=np.int64)]),
            dtype=torch.float64,
            device=device,
        )
        homogeneous = torch.cat(
            (source, torch.ones((len(source), 1), dtype=source.dtype, device=device)),
            dim=1,
        )
        points = (homogeneous @ transform.T)[:, :3]
        point_center = points.mean(dim=0)
        differences = points[:, None, :] - centers[None, :, :]
        local = torch.einsum("nbi,bji->nbj", differences, axes)
        inside = torch.all(
            torch.abs(local)
            <= lengths[None, :, :] / 2.0 + PARENT_CONTAINMENT_MARGIN_M,
            dim=2,
        )
        coverage = inside.to(torch.float64).mean(dim=0).cpu().numpy()
        distances = torch.linalg.vector_norm(
            point_center[None, :] - centers, dim=1
        ).cpu().numpy()
        ranked = sorted(
            range(len(parents)),
            key=lambda index: (
                float(coverage[index]),
                -float(distances[index]),
                parents[index].binding_id,
            ),
            reverse=True,
        )
        if not ranked or coverage[ranked[0]] < PARENT_CONTAINMENT_MIN_FRACTION:
            unmatched += 1
            continue
        index = ranked[0]
        points_numpy = points.cpu().numpy()
        proposals[annotation["annot_id"]] = FunctionalProposal(
            candidate_id=annotation["annot_id"],
            points=points_numpy,
            center=points_numpy.mean(axis=0),
            parent=parents[index],
            parent_coverage=float(coverage[index]),
        )
    return proposals, unmatched, sentinel


def _assert_equivalent(
    cpu_value: tuple[dict[str, FunctionalProposal], int],
    gpu_value: tuple[dict[str, FunctionalProposal], int, Any],
) -> None:
    cpu_proposals, cpu_unmatched = cpu_value
    gpu_proposals, gpu_unmatched, _ = gpu_value
    if cpu_unmatched != gpu_unmatched or set(cpu_proposals) != set(gpu_proposals):
        raise ValueError("CPU_GPU_PROPOSAL_SET_MISMATCH")
    for candidate_id, cpu in cpu_proposals.items():
        gpu = gpu_proposals[candidate_id]
        if cpu.parent.binding_id != gpu.parent.binding_id:
            raise ValueError(f"CPU_GPU_PARENT_MISMATCH:{candidate_id}")
        if not np.allclose(cpu.center, gpu.center, rtol=0.0, atol=1e-9):
            raise ValueError(f"CPU_GPU_CENTER_MISMATCH:{candidate_id}")


class MeasuredProposalBuilder:
    def __init__(
        self,
        representative_paths: dict[str, Path],
        receipt_path: Path,
    ) -> None:
        self.representative_paths = representative_paths
        self.receipt_path = receipt_path
        self._last_cpu: tuple[dict[str, FunctionalProposal], int] | None = None
        self._last_gpu: tuple[dict[str, FunctionalProposal], int, Any] | None = None
        self.selected_backend = ""

    def select(self) -> dict[str, Any]:
        capabilities = runtime_capabilities()

        def run_cpu() -> tuple[dict[str, FunctionalProposal], int]:
            self._last_cpu = _build_proposals(self.representative_paths)
            return self._last_cpu

        cpu = BackendCandidate(
            "numpy-cpu",
            "cpu",
            run_cpu,
            lambda _: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"
            ),
        )
        gpu: BackendCandidate | None = None
        cpu_reason: str | None = None
        torch_capability = capabilities.get("torch", {})
        if torch_capability.get("available") and torch_capability.get("cuda_available"):
            import torch

            def run_gpu() -> tuple[dict[str, FunctionalProposal], int, Any]:
                self._last_gpu = _torch_build_proposals(self.representative_paths)
                return self._last_gpu

            gpu = BackendCandidate(
                "torch-cuda",
                "cuda",
                run_gpu,
                lambda output: torch_observation(output=output[2]),
                torch.cuda.synchronize,
            )
        else:
            cpu_reason = "ACCELERATOR_UNAVAILABLE"
        record = select_backend(
            Workload.POINT_CLOUD_MATCHING,
            cpu=cpu,
            gpu=gpu,
            cpu_reason=cpu_reason,
            record_path=self.receipt_path,
            warmups=0,
            repeats=2,
            capabilities=capabilities,
        )
        if self._last_gpu is not None and self._last_cpu is not None:
            _assert_equivalent(self._last_cpu, self._last_gpu)
        self.selected_backend = str(record["selected_backend"])
        return record

    def build(
        self, paths: dict[str, Path]
    ) -> tuple[dict[str, FunctionalProposal], int]:
        if not self.selected_backend:
            raise RuntimeError("BACKEND_NOT_SELECTED")
        if paths == self.representative_paths:
            if self.selected_backend == "torch-cuda" and self._last_gpu is not None:
                return self._last_gpu[0], self._last_gpu[1]
            if self.selected_backend == "numpy-cpu" and self._last_cpu is not None:
                return self._last_cpu
        if self.selected_backend == "torch-cuda":
            proposals, unmatched, _ = _torch_build_proposals(paths)
            return proposals, unmatched
        return _build_proposals(paths)


def _numpy_functional_centers(paths: dict[str, Path]) -> dict[str, FunctionalCenter]:
    xyz = _load_ply_xyz(paths["laser_scan"])
    transform = np.load(paths["transform"])
    centers: dict[str, FunctionalCenter] = {}
    for annotation in _load_json(paths["annotations"])["annotations"]:
        if annotation["label"] == "exclude":
            continue
        points = np.asarray(xyz[np.asarray(annotation["indices"], dtype=np.int64)])
        homogeneous = np.column_stack((points, np.ones(len(points))))
        transformed = (homogeneous @ transform.T)[:, :3]
        centers[annotation["annot_id"]] = FunctionalCenter(
            annotation["annot_id"], annotation["label"], transformed.mean(axis=0)
        )
    return centers


def _torch_functional_centers(
    paths: dict[str, Path], device: str = "cuda"
) -> tuple[dict[str, FunctionalCenter], Any]:
    import torch

    xyz = _load_ply_xyz(paths["laser_scan"])
    transform = torch.as_tensor(
        np.load(paths["transform"]), dtype=torch.float64, device=device
    )
    centers: dict[str, FunctionalCenter] = {}
    sentinel = torch.empty(1, device=device)
    for annotation in _load_json(paths["annotations"])["annotations"]:
        if annotation["label"] == "exclude":
            continue
        points = torch.as_tensor(
            np.asarray(xyz[np.asarray(annotation["indices"], dtype=np.int64)]),
            dtype=torch.float64,
            device=device,
        )
        homogeneous = torch.cat(
            (points, torch.ones((len(points), 1), dtype=points.dtype, device=device)),
            dim=1,
        )
        center = ((homogeneous @ transform.T)[:, :3]).mean(dim=0).cpu().numpy()
        centers[annotation["annot_id"]] = FunctionalCenter(
            annotation["annot_id"], annotation["label"], center
        )
    return centers, sentinel


class MeasuredFunctionalCenterBuilder:
    def __init__(self, representative_paths: dict[str, Path], receipt_path: Path) -> None:
        self.representative_paths = representative_paths
        self.receipt_path = receipt_path
        self._last_cpu: dict[str, FunctionalCenter] | None = None
        self._last_gpu: tuple[dict[str, FunctionalCenter], Any] | None = None
        self.selected_backend = ""

    def select(self) -> dict[str, Any]:
        capabilities = runtime_capabilities()

        def run_cpu() -> dict[str, FunctionalCenter]:
            self._last_cpu = _numpy_functional_centers(self.representative_paths)
            return self._last_cpu

        cpu = BackendCandidate(
            "numpy-cpu",
            "cpu",
            run_cpu,
            lambda _: DeviceObservation(
                "cpu", platform.processor() or "CPU", f"numpy-{np.__version__}"
            ),
        )
        gpu: BackendCandidate | None = None
        cpu_reason: str | None = None
        torch_capability = capabilities.get("torch", {})
        if torch_capability.get("available") and torch_capability.get("cuda_available"):
            import torch

            def run_gpu() -> tuple[dict[str, FunctionalCenter], Any]:
                self._last_gpu = _torch_functional_centers(self.representative_paths)
                return self._last_gpu

            gpu = BackendCandidate(
                "torch-cuda",
                "cuda",
                run_gpu,
                lambda output: torch_observation(output=output[1]),
                torch.cuda.synchronize,
            )
        else:
            cpu_reason = "ACCELERATOR_UNAVAILABLE"
        record = select_backend(
            Workload.POINT_CLOUD_MATCHING,
            cpu=cpu,
            gpu=gpu,
            cpu_reason=cpu_reason,
            record_path=self.receipt_path,
            warmups=0,
            repeats=2,
            capabilities=capabilities,
        )
        if self._last_cpu is not None and self._last_gpu is not None:
            gpu_centers = self._last_gpu[0]
            if set(self._last_cpu) != set(gpu_centers):
                raise ValueError("CPU_GPU_FUNCTIONAL_CENTER_SET_MISMATCH")
            for candidate_id, cpu_center in self._last_cpu.items():
                if not np.allclose(
                    cpu_center.center,
                    gpu_centers[candidate_id].center,
                    rtol=0.0,
                    atol=1e-9,
                ):
                    raise ValueError(f"CPU_GPU_FUNCTIONAL_CENTER_MISMATCH:{candidate_id}")
        self.selected_backend = str(record["selected_backend"])
        return record

    def build(self, paths: dict[str, Path]) -> dict[str, FunctionalCenter]:
        if not self.selected_backend:
            raise RuntimeError("BACKEND_NOT_SELECTED")
        if paths == self.representative_paths:
            if self.selected_backend == "torch-cuda" and self._last_gpu is not None:
                return self._last_gpu[0]
            if self.selected_backend == "numpy-cpu" and self._last_cpu is not None:
                return self._last_cpu
        if self.selected_backend == "torch-cuda":
            return _torch_functional_centers(paths)[0]
        return _numpy_functional_centers(paths)
