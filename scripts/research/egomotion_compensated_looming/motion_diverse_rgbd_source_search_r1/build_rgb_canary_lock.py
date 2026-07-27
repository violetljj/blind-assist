"""Freeze RGB runner, unchanged core, protocol, and local inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FILES = [
    "scripts/research/egomotion_compensated_looming/motion_diverse_rgbd_source_search_r1/run_rgb_canary.py",
    "scripts/research/egomotion_compensated_looming/rgb_algorithm_development_canary_cid_sims_r0/producer.py",
    "scripts/research/egomotion_compensated_looming/configs/phase_a_synthetic_signal_audit_r0.json",
    "scripts/research/egomotion_compensated_looming/rcle_minimal/rotation_compensation.py",
    "scripts/research/egomotion_compensated_looming/rcle_minimal_r1/local_expansion.py",
    "scripts/research/egomotion_compensated_looming/rcle_minimal_r1/sparse_flow.py",
    "scripts/research/egomotion_compensated_looming/rcle_observable_support_r0/__init__.py",
    "scripts/research/egomotion_compensated_looming/rcle_observable_support_r0/evaluation.py",
    "scripts/research/egomotion_compensated_looming/real_data_geometry_canary_r0/producer.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--adapter-amendment", type=Path, required=True)
    parser.add_argument("--rgb-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.rgb_implementation_lock.v1",
        "status": "FROZEN_BEFORE_RGB_ALGORITHM_OUTCOME",
        "files": [{"path": path, "sha256": sha(repo / path)} for path in FILES],
        "local_inputs": {
            "adapter_amendment_sha256": sha(args.adapter_amendment.resolve()),
            "canary_sha256": sha(args.canary.resolve()),
            "rgb_manifest_sha256": sha(args.rgb_manifest.resolve()),
        },
        "workers": 8,
        "algorithm_changed": False,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(os.fspath(args.output.resolve()), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"rgb_implementation_lock_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
