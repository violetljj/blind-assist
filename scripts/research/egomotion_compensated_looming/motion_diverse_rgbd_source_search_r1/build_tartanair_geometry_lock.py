"""Bind the TartanAir geometry implementation and local inputs before execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FILES = [
    "scripts/research/egomotion_compensated_looming/motion_diverse_rgbd_source_search_r1/run_tartanair_geometry.py",
    "scripts/research/egomotion_compensated_looming/motion_diverse_rgbd_geometry_admission_r0/run_eth3d_geometry.py",
    "scripts/research/egomotion_compensated_looming/motion_diverse_rgbd_geometry_admission_r0/template.py",
    "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/producer.py",
    "scripts/research/egomotion_compensated_looming/pb_h1_role_proxy/geometry.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--extract-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.tartanair_geometry_lock.v1",
        "status": "FROZEN_BEFORE_TARTANAIR_GEOMETRY",
        "files": [{"path": path, "sha256": sha(repo / path)} for path in FILES],
        "local_inputs": {
            "amendment_sha256": sha(args.amendment.resolve()),
            "extract_manifest_sha256": sha(args.extract_manifest.resolve()),
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
    print(json.dumps({"implementation_lock_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
