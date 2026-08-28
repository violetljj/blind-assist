"""Fail-closed smoke probe for the shared DTR/L10 research GPU runtime."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("research-dtr-r0", "research-l10-r0"), required=True)
    args = parser.parse_args()

    import cupy as cp
    import cv2
    import numpy as np
    import scipy
    import sklearn
    import torch
    import transformers
    import ultralytics
    from numba import cuda
    from rapidocr import RapidOCR  # noqa: F401
    from rosbags.highlevel import AnyReader  # noqa: F401

    torch_input = torch.arange(4096, dtype=torch.float32, device="cuda").reshape(64, 64)
    torch_value = float((torch_input @ torch_input.T)[0, 0].item())

    cupy_input = cp.arange(4096, dtype=cp.float32)
    cupy_value = float(cp.sum(cupy_input).get())

    @cuda.jit
    def add_one(values):
        index = cuda.grid(1)
        if index < values.size:
            values[index] += 1

    host_values = np.zeros(65536, dtype=np.float32)
    device_values = cuda.to_device(host_values)
    threads = 256
    blocks = (host_values.size + threads - 1) // threads
    add_one[blocks, threads](device_values)
    cuda.synchronize()
    numba_value = float(device_values.copy_to_host().sum())

    if torch_value != 85344.0:
        raise RuntimeError(f"torch_cuda_probe_mismatch:{torch_value}")
    if cupy_value != 8386560.0:
        raise RuntimeError(f"cupy_cuda_probe_mismatch:{cupy_value}")
    if numba_value != float(host_values.size):
        raise RuntimeError(f"numba_cuda_probe_mismatch:{numba_value}")

    print(
        json.dumps(
            {
                "schema": "blindassist-shared-research-gpu-v1",
                "profile": args.profile,
                "status": "PASS",
                "actual_device": torch.cuda.get_device_name(0),
                "torch": {
                    "version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "actual_tensor_device": str(torch_input.device),
                },
                "cupy": {
                    "version": cp.__version__,
                    "actual_device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                },
                "numba": {
                    "version": __import__("numba").__version__,
                    "cuda_available": cuda.is_available(),
                    "kernel_sum": numba_value,
                },
                "route_dependencies": {
                    "numpy": np.__version__,
                    "opencv": cv2.__version__,
                    "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__,
                    "rosbags": "imported",
                    "rapidocr": "imported",
                    "ultralytics": ultralytics.__version__,
                    "transformers": transformers.__version__,
                },
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
