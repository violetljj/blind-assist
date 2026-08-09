"""Training-only DepthART SelectiveScan adapter with an explicit autograd boundary."""

from __future__ import annotations

import importlib
from typing import Any


def install_depthart_training_scan(tvimblock: Any) -> tuple[Any, dict[str, Any]]:
    """Install the eager custom Function instead of the inference torch.library op.

    The deployment package intentionally exposes a registered operator for export
    and inference. PyTorch warns when that outer operator is used for training
    because it has no Autograd-key registration, even though its CUDA body calls
    a custom Function. Training therefore bypasses only the outer dispatcher and
    calls the package's explicit ``torch.autograd.Function`` path directly.
    """
    if not hasattr(tvimblock, "cross_selective_scan"):
        raise TypeError("DepthART tvimblock does not expose cross_selective_scan")
    ops = importlib.import_module("depthart_selective_scan.ops")
    cross_scan = importlib.import_module("depthart_selective_scan.cross_scan")
    eager = getattr(ops, "_cuda_impl", None)
    extension = getattr(ops, "_C", None)
    if eager is None or extension is None:
        raise RuntimeError("DepthART training SelectiveScan eager CUDA backend is unavailable")
    cross_scan.selective_scan = eager
    previous = tvimblock.cross_selective_scan
    tvimblock.cross_selective_scan = cross_scan.cross_selective_scan
    return previous, {
        "package_module": str(ops.__file__),
        "dispatch_boundary": "depthart_selective_scan.ops._cuda_impl",
        "autograd_boundary": "depthart_selective_scan.ops._SelectiveScanAutograd",
        "outer_torch_library_operator_used_for_training": False,
    }
