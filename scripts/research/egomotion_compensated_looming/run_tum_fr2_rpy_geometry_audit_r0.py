from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from egomotion_compensated_looming.tum_fr2_rpy_geometry_audit.audit import (
    run_audit,
    sha256_file,
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    source_base = (
        repo_root
        / "artifacts.local/datasets/rcle_tum_fr2_rpy_source_native_r0"
    )
    root = source_base / "rgbd_dataset_freiburg2_rpy"
    archive = source_base / "rgbd_dataset_freiburg2_rpy.tgz"
    output = (
        repo_root
        / "artifacts.local/evidence/rcle_tum_fr2_rpy_geometry_audit_r0"
    )
    output.mkdir(parents=True, exist_ok=True)
    result = run_audit(root, archive)
    result_bytes = canonical_bytes(result)
    result_path = output / "result.json"
    result_path.write_bytes(result_bytes)
    contract = (
        repo_root
        / "docs/research/rcle/"
        "RCLE_TUM_FR2_RPY_SOURCE_NATIVE_GEOMETRY_AUDIT_R0_CONTRACT_2026-07-26.json"
    )
    amendment = (
        repo_root
        / "docs/research/rcle/"
        "RCLE_TUM_FR2_RPY_SOURCE_NATIVE_GEOMETRY_AUDIT_R0_IMPLEMENTATION_AMENDMENT_2026-07-26.json"
    )
    implementation = (
        repo_root
        / "scripts/research/egomotion_compensated_looming/"
        "tum_fr2_rpy_geometry_audit/audit.py"
    )
    runner = Path(__file__).resolve()
    pb_h1 = (
        repo_root
        / "scripts/research/egomotion_compensated_looming/"
        "pb_h1_role_proxy/geometry.py"
    )
    receipt = {
        "schema_version": "rcle.tum_fr2_rpy.source_native_geometry_audit.receipt.v1",
        "result_sha256": sha256(result_bytes).hexdigest(),
        "contract_sha256": sha256_file(contract),
        "implementation_amendment_sha256": sha256_file(amendment),
        "implementation_sha256": sha256_file(implementation),
        "runner_sha256": sha256_file(runner),
        "pb_h1_geometry_sha256": sha256_file(pb_h1),
        "archive_sha256": result["archive"]["sha256"],
        "groundtruth_sha256": result["groundtruth_sha256"],
        "terminal": result["terminal"],
    }
    (output / "receipt.json").write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "result": str(result_path),
                "terminal": result["terminal"],
                "result_sha256": receipt["result_sha256"],
                "windows": len(result["windows"]),
                "evaluable_windows": result["coverage"][
                    "evaluable_window_count"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
