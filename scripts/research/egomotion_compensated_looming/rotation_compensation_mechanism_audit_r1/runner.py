from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner as discovery,
)


PROTOCOL_ID = "RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1"
EXPECTED_INPUTS = {
    "iphone/frames.mov": (
        "5fddda2af443aaa35ab560e87d1d5f87"
        "ba800726bf5530459d4d1f39af5cb620"
    ),
    "iphone/frames.csv": (
        "cffa6c6d9e453b0bfd4ed7bca33bf549"
        "52840edc1c1039eec3b7bb2911a59ce9"
    ),
    "ground-truth/pose.csv": (
        "c410785639e813c35d2c14ad62a7864d"
        "b1eb19a39ceb7da513fa2a5e852ad643"
    ),
}


def implementation_hashes() -> dict[str, str]:
    paths = {
        "mechanism_runner": Path(__file__).resolve(),
        "discovery_engine": Path(discovery.__file__).resolve(),
    }
    return {
        name: discovery.sha256_file(path) for name, path in paths.items()
    }


def verify_source(source_root: Path) -> None:
    for relative, expected in EXPECTED_INPUTS.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if discovery.sha256_file(path) != expected:
            raise ValueError(f"SOURCE_HASH_MISMATCH:{relative}")


def run_arm(
    source_root: Path,
    output_dir: Path,
    arm: str,
    *,
    pair_count: int = 600,
) -> dict[str, Any]:
    if arm not in {"raw", "undistorted"}:
        raise ValueError(f"UNKNOWN_AB_ARM:{arm}")
    source_root = source_root.resolve()
    verify_source(source_root)
    context = {
        "protocol_id": PROTOCOL_ID,
        "research_track": "DEVELOPMENT_DIAGNOSTIC",
        "outcome_access_state": "TUNED_ON",
        "stage": "DEVELOPMENT_DIAGNOSTIC",
        "ab_arm": arm,
        "implementation_version": "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3",
        "implementation_hashes": implementation_hashes(),
        "selection_freeze": (
            "RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1_"
            "CONTRACT_2026-07-28.json"
        ),
        "threshold_changed": False,
        "three_pair_rule_changed": False,
        "single_process_continuous_state_required": True,
        "runtime_pilot": pair_count != 600,
    }
    result = discovery.run(
        source_root,
        output_dir.resolve(),
        max_pairs=pair_count,
        start_frame=0,
        progress_every=25,
        resize_scale=0.5,
        quaternion_component_order="wxyz",
        distortion_correction=arm == "undistorted",
        pose_to_camera_rotation=discovery.T_CAM_IMU_ROTATION,
        evidence_context=context,
    )
    if result["execution"]["candidate_pair_count"] != pair_count:
        raise ValueError("FROZEN_PAIR_COUNT_MISMATCH")
    if (
        result["execution"]["support_manager_baseline_pair_count"] != 1
    ):
        raise ValueError("CONTINUOUS_STATE_BASELINE_COUNT_MISMATCH")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("raw", "undistorted"), required=True
    )
    parser.add_argument("--pilot-pairs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pair_count = args.pilot_pairs if args.pilot_pairs is not None else 600
    if not (1 <= pair_count <= 600):
        raise ValueError("PILOT_PAIR_COUNT_OUT_OF_RANGE")
    result = run_arm(
        args.source_root,
        args.output_dir,
        args.arm,
        pair_count=pair_count,
    )
    print(
        {
            "protocol_id": PROTOCOL_ID,
            "arm": args.arm,
            "pairs": result["execution"]["candidate_pair_count"],
            "runtime_s": result["execution"]["runtime_s"],
            "summary_sha256": discovery.sha256_file(
                args.output_dir.resolve() / "summary.json"
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
