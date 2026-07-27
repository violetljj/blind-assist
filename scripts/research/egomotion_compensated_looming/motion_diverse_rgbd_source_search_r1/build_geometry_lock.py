"""Build a hash-bound implementation lock for a frozen geometry batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--pose-queue", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--depth-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    paths = [
        "scripts/research/egomotion_compensated_looming/motion_diverse_rgbd_source_search_r1/run_geometry_batch.py",
        "scripts/research/egomotion_compensated_looming/run_motion_diverse_geometry_batch_r1.py",
        "scripts/research/egomotion_compensated_looming/motion_diverse_rgbd_geometry_admission_r0/run_eth3d_geometry.py",
        "scripts/research/egomotion_compensated_looming/motion_diverse_rgbd_geometry_admission_r0/template.py",
        "scripts/research/egomotion_compensated_looming/real_positive_approach_role_admission_r2_cid_sims/producer.py",
        "scripts/fetch_remote_zip_members.py",
        str(args.contract.resolve().relative_to(repo)).replace("\\", "/"),
    ]
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.implementation_lock.v1",
        "protocol_id": batch["protocol_id"],
        "batch_id": batch["batch_id"],
        "status": "FROZEN_BEFORE_DEPTH_GEOMETRY",
        "files": [{"path": path, "sha256": sha(repo / path)} for path in paths],
        "local_inputs": {
            "pose_queue_sha256": sha(args.pose_queue.resolve()),
            "geometry_batch_sha256": sha(args.batch.resolve()),
            "depth_inventory_sha256": sha(args.depth_inventory.resolve()),
        },
        "execution": {
            "workers": 8,
            "rgb_bytes_allowed": False,
            "algorithm_change_allowed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"batch_id": batch["batch_id"], "lock_sha256": sha(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
