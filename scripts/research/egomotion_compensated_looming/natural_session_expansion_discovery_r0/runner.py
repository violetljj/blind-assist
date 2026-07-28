from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner as r3,
)


PROTOCOL_ID = "RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0"
PAIR_COUNT = 601
DISCOVERY_SESSIONS = {
    13: "ADVIO_OFFICE01_SEQUENCE13_IPHONE",
    14: "ADVIO_OFFICE02_SEQUENCE14_IPHONE",
    15: "ADVIO_OFFICE03_SEQUENCE15_IPHONE",
    17: "ADVIO_OFFICE05_SEQUENCE17_IPHONE",
}
SEALED_SESSION = 16
REQUIRED_INPUTS = (
    "iphone/frames.mov",
    "iphone/frames.csv",
    "ground-truth/pose.csv",
)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_receipt(source_root: Path, receipt_path: Path, session: int) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("session_number") != session
        or receipt.get("sealed_session_touched") is not False
    ):
        raise ValueError("SOURCE_RECEIPT_SESSION_OR_FIREWALL_MISMATCH")
    members = receipt.get("members", {})
    if sorted(members) != sorted(REQUIRED_INPUTS):
        raise ValueError("SOURCE_RECEIPT_MEMBER_SET_MISMATCH")
    for relative in REQUIRED_INPUTS:
        path = source_root / relative
        if (
            not path.is_file()
            or sha256_file(path) != members[relative].get("sha256")
        ):
            raise ValueError(f"SOURCE_MEMBER_IDENTITY_MISMATCH:{relative}")


def run(
    session: int,
    source_root: Path,
    receipt_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if session == SEALED_SESSION:
        raise PermissionError("SEALED_UNSEEN_SESSION_ACCESS_FORBIDDEN")
    if session not in DISCOVERY_SESSIONS:
        raise ValueError("SESSION_NOT_IN_FROZEN_DISCOVERY_SET")
    source_root = source_root.resolve()
    verify_receipt(source_root, receipt_path.resolve(), session)
    source_id = DISCOVERY_SESSIONS[session]
    prior_source_id = r3.SOURCE_ID
    r3.SOURCE_ID = source_id
    try:
        result = r3.run(
            source_root,
            output_dir.resolve(),
            max_pairs=PAIR_COUNT,
            start_frame=0,
            progress_every=25,
            resize_scale=0.5,
            quaternion_component_order="wxyz",
            distortion_correction=True,
            pose_to_camera_rotation=r3.T_CAM_IMU_ROTATION,
            evidence_context={
                "protocol_id": PROTOCOL_ID,
                "research_track": (
                    "DEVELOPMENT_DIAGNOSTIC"
                    if session == 15
                    else "CAPABILITY_DISCOVERY"
                ),
                "outcome_access_state": (
                    "TUNED_ON" if session == 15 else "OUTPUT_INSPECTED"
                ),
                "stage": (
                    "DEVELOPMENT_DIAGNOSTIC"
                    if session == 15
                    else "CAPABILITY_DISCOVERY"
                ),
                "session_number": session,
                "selection_freeze": (
                    "RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0_"
                    "CONTRACT_2026-07-28.json"
                ),
                "implementation_version": (
                    "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
                ),
                "r3_core_sha256": sha256_file(Path(r3.__file__).resolve()),
                "metadata_identity_generalization_only": True,
                "algorithm_adjustment": False,
                "threshold_changed": False,
                "three_pair_rule_changed": False,
                "single_process_continuous_state_required": True,
                "pair_pooling_authorized": False,
                "forbidden_metrics": ["AUROC", "F1"],
                "sealed_session_accessed": False,
            },
        )
    finally:
        r3.SOURCE_ID = prior_source_id
    execution = result["execution"]
    if (
        execution["candidate_pair_count"] != PAIR_COUNT
        or execution["threshold_per_s"] != 0.01
        or execution["required_consecutive_pairs"] != 3
        or execution["single_process_pair_state_continuous"] is not True
        or execution["support_manager_baseline_pair_count"] != 1
        or not (10.0 <= execution["duration_s"] <= 30.0)
    ):
        raise ValueError("FROZEN_EXECUTION_CONTRACT_MISMATCH")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.session,
        args.source_root,
        args.source_receipt,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "session": args.session,
                "duration_s": result["execution"]["duration_s"],
                "candidate_pair_count": result["execution"][
                    "candidate_pair_count"
                ],
                "evaluable_pair_fraction": result["execution"][
                    "evaluable_pair_fraction"
                ],
                "runtime_s": result["execution"]["runtime_s"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
