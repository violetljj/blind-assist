#!/usr/bin/env python3
"""Freeze twelve current-recipe-unseen sitting_rpy factor labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

import materialize_ag_r2_tum_real_fresh_confirmation_labels as base  # noqa: E402


PARENT_ID = "rgbd_dataset_freiburg3_sitting_rpy"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "2AC397FFF9E21CBFAD707D549BC07B83D27FE02F59F3678072E9A7BBA684A67E"
)
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "artifacts.local/datasets/ag-r2-final-sitting-rpy-source-r0"
    / PARENT_ID
)
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "artifacts.local/downloads/quality-gated-clearance-fusion-r0-1-tum"
    / f"{PARENT_ID}.tgz"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-sitting-rpy-final-confirmation-labels-r0"
)
EXTRA_CURRENT_RECIPE_RECEIPTS = {
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-multisource-metric-depth-student-r0/result.json": "F0703357B0F25C7ABF209EE53DE9B04E588BEDE3629C1B1F5273D9E31D41BFF3",
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-metric-scale-residual-bank-r2/result.json": "68188B00FB4951771443B4706526F6BAAED1AA85E77C4AE4D73958600AB4C0E5",
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-hybrid-factor-student-to-ag-seam-r10/result.json": "5077213EA3B5B0CF0186E755BF1D0FDDF8C25DBB65D415E3A05263A6780A9720",
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-session-metric-scale-anchor-calibration-r0/result.json": "12F6AF0D80A3CBF25204D507FE8065B49AAFC6332DD75C57D97D8816D76D1A06",
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-walking-xyz-session-anchored-consumed-seam-r0/result.json": "D1688247CC6ABCCA9941FED34DCF5ADA7C5B22226FD6834C511C613CD8003377",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for name in ("source_root", "source_archive", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main() -> int:
    result = base.run(
        parse_args(),
        expected_source_archive_sha256=EXPECTED_SOURCE_ARCHIVE_SHA256,
        parent_id=PARENT_ID,
        role="CURRENT_RECIPE_UNSEEN_REAL_CONFIRMATION",
        orientation="LANDSCAPE_IDENTITY",
        selection_token="AG_R2_TUM_FR3_SITTING_RPY_FINAL_CONFIRMATION_R0",
        dataset_name="TUM RGB-D fr3 sitting_rpy",
        globally_unopened_claim=False,
        extra_checkpoint_receipts=EXTRA_CURRENT_RECIPE_RECEIPTS,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "frame_count": result["frame_count"],
                "coverage": result["coverage"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
