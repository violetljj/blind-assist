"""Fail-closed local runtime canary for the pinned ABotN 3DGS renderer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import requests

from .audit_abotn_poibench_truth_source import (
    REPOSITORY_ID,
    REPOSITORY_REVISION,
    SCHEMA as SOURCE_AUDIT_SCHEMA,
)


SCHEMA = "blindassist_abotn_render_runtime_audit_v0"
OFFICIAL_MINIMUM_VRAM_MIB = 24 * 1024


def classify_runtime(
    *,
    host_os: str,
    gpu_count: int,
    maximum_gpu_vram_mib: int,
    torch_cuda_available: bool,
    cuda_compiler_available: bool,
) -> dict[str, Any]:
    failures = []
    if host_os.lower() != "linux":
        failures.append("HOST_OS_NOT_LINUX")
    if gpu_count < 1:
        failures.append("NO_NVIDIA_GPU")
    if maximum_gpu_vram_mib < OFFICIAL_MINIMUM_VRAM_MIB:
        failures.append("VRAM_BELOW_OFFICIAL_24GB_MINIMUM")
    if not torch_cuda_available:
        failures.append("PYTORCH_CUDA_UNAVAILABLE")
    if not cuda_compiler_available:
        failures.append("CUDA_COMPILER_UNAVAILABLE")
    if "VRAM_BELOW_OFFICIAL_24GB_MINIMUM" in failures:
        terminal = "NOT_EVALUABLE_LOCAL_RENDER_RUNTIME_VRAM_BELOW_OFFICIAL_MINIMUM"
    elif failures:
        terminal = "NOT_EVALUABLE_LOCAL_RENDER_RUNTIME_REQUIREMENTS_UNMET"
    else:
        terminal = "LOCAL_RENDER_RUNTIME_PREFLIGHT_PASS"
    return {"terminal": terminal, "failures": failures, "eligible": not failures}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _gpu_rows() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            rows.append({
                "name": parts[0],
                "memory_total_mib": int(parts[1]),
                "memory_free_mib": int(parts[2]),
                "driver_version": parts[3],
            })
    return rows


def run_audit(source_audit_path: Path, output: Path) -> dict[str, Any]:
    source = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != SOURCE_AUDIT_SCHEMA:
        raise ValueError("unexpected source audit schema")
    if source.get("classification", {}).get("overall") != "ARRIVAL_TRUTH_ONLY_INTERNAL_RESEARCH_CANDIDATE":
        raise ValueError("source audit does not authorize a render feasibility canary")

    renderer_url = (
        f"https://raw.githubusercontent.com/{REPOSITORY_ID}/{REPOSITORY_REVISION}"
        "/render_server/README.md"
    )
    response = requests.get(renderer_url, timeout=60)
    response.raise_for_status()
    renderer_readme = response.content
    if b">= 24 GB per GPU" not in renderer_readme or b"Linux (Ubuntu 20.04+)" not in renderer_readme:
        raise ValueError("pinned renderer requirements changed unexpectedly")

    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        torch_version = str(torch.__version__)
        torch_cuda_version = str(torch.version.cuda)
        torch_cuda_available = bool(torch.cuda.is_available())
        cuda_home = str(CUDA_HOME) if CUDA_HOME else None
    except Exception as exc:  # pragma: no cover - host-dependent diagnostic
        torch_version = None
        torch_cuda_version = None
        torch_cuda_available = False
        cuda_home = None
        torch_error = repr(exc)
    else:
        torch_error = None

    gpu_rows = _gpu_rows()
    maximum_vram = max((row["memory_total_mib"] for row in gpu_rows), default=0)
    classification = classify_runtime(
        host_os=platform.system(),
        gpu_count=len(gpu_rows),
        maximum_gpu_vram_mib=maximum_vram,
        torch_cuda_available=torch_cuda_available,
        cuda_compiler_available=bool(cuda_home),
    )
    result = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "REVERSIBLE_EXPLORATION_CANARY_LITE",
        "source_audit": {
            "path": str(source_audit_path.resolve()),
            "sha256": hashlib.sha256(source_audit_path.read_bytes()).hexdigest(),
        },
        "official_renderer": {
            "repository": REPOSITORY_ID,
            "revision": REPOSITORY_REVISION,
            "requirements_url": renderer_url,
            "requirements_sha256": hashlib.sha256(renderer_readme).hexdigest(),
            "host_os": "Linux (Ubuntu 20.04+)",
            "minimum_vram_mib": OFFICIAL_MINIMUM_VRAM_MIB,
            "cuda_extensions_required": True,
        },
        "local_runtime": {
            "host_os": platform.system(),
            "host_release": platform.release(),
            "gpus": gpu_rows,
            "torch_version": torch_version,
            "torch_cuda_version": torch_cuda_version,
            "torch_cuda_available": torch_cuda_available,
            "cuda_home": cuda_home,
            "torch_error": torch_error,
        },
        "classification": classification,
        "scene_payloads_downloaded": 0,
        "render_calls": 0,
        "teacher_calls": 0,
        "provider_calls": 0,
        "claim_ceiling": "LOCAL_RUNTIME_FEASIBILITY_ONLY",
        "next_action": (
            "REQUIRE_VALIDATED_LINUX_CUDA_HOST_WITH_AT_LEAST_24GB_VRAM_OR_SELECT_ALTERNATE_PIXEL_TRUTH_SOURCE"
            if not classification["eligible"]
            else "DOWNLOAD_SMALLEST_SCENE_AND_RENDER_ONE_FROZEN_INITIAL_VIEW"
        ),
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.source_audit.resolve(), args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "terminal": result["classification"]["terminal"],
        "failures": result["classification"]["failures"],
        "next_action": result["next_action"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
