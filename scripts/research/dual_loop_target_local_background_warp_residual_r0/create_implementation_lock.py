"""Create a hash-bound B Development implementation lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import tempfile

import cv2
import numpy as np

from . import common


def source_files(module_dir: Path) -> list[Path]:
    return sorted(module_dir.glob("*.py"))


def build_lock(repo_root: Path) -> dict[str, object]:
    module_dir = Path(__file__).resolve().parent
    return {
        "schema": "blindassist.target_local_warp_residual_implementation_lock.v1",
        "status": "VALID",
        "stage": "DEVELOPMENT",
        "protocol_id": common.PROTOCOL_ID,
        "implementation_id": common.IMPLEMENTATION_ID,
        "contract_sha256": common.contract_sha256(repo_root),
        "source_files": {path.name: common.sha256_file(path) for path in source_files(module_dir)},
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "ring_config_ids": list(common.RING_CONFIGS),
        "model_ids": [common.SHI_TOMASI_MODEL_ID, common.SIMILARITY_MODEL_ID],
        "truth_read_by_producer": False,
        "android_or_runtime": False,
    }


def write_exclusive(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args(argv)
    lock = build_lock(args.repo_root or Path(__file__).resolve().parents[3])
    write_exclusive(args.output, lock)
    print(json.dumps({"status": lock["status"], "implementation_id": lock["implementation_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
