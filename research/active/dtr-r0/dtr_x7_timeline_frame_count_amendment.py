"""Seal the X7 timeline/source-support frame-count correction.

The frozen X7 workers wrote complete ledgers for all 4,811 cohort timeline
frames.  The original assembler incorrectly compared that total with X3's
4,787 optimized source frames.  The difference is exactly four causal
five-scan warm-up frames per sequence, all fail-closed with zero cells.  This
amendment changes no candidate cell, velocity, prediction, scorer, or gate.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from dtr_c1_global_obb_cohort_admission import require, sha256_file, write_json


REPO = Path(__file__).resolve().parents[3]
X7_SCRIPT = REPO / "research/active/dtr-r0/dtr_x7_full_static_world_anchor_replay.py"
EXPECTED_X7_SCRIPT_SHA256 = "c9b5a58eafc732b692d2892adc5dbbb17a20e2b1240f86f114c48d63ec3e3f8c"
EXPECTED_FREEZE_SHA256 = "57faf3768e95cf7b6f06357de89ef5e85249e54b36fde6d8388167dc0c429aa6"
LEDGER_SCHEMA = "blindassist-dtr-x7-full-static-world-anchor-ledger-v1"
MATERIALIZATION_SCHEMA = "blindassist-dtr-x7-full-materialization-v1"
SEQUENCES = (
    "huang-2-2019-01-25_0",
    "huang-basement-2019-01-25_0",
    "huang-lane-2019-02-12_0",
    "memorial-court-2019-03-16_0",
    "meyer-green-2019-03-16_0",
    "tressider-2019-03-16_1",
)
TIMELINE_FRAMES = 4811
SOURCE_SUPPORTED_FRAMES = 4787
WARMUP_FRAMES_PER_SEQUENCE = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / "artifacts.local/evidence/dtr-x7/full-static-world-anchor-replay-20260829",
    )
    parser.add_argument(
        "--x3-root",
        type=Path,
        default=REPO / "artifacts.local/evidence/dtr-x3/full-lag-floxel-replay-mp",
    )
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    x3_root = args.x3_root.resolve(strict=True)
    freeze = (root / "freeze.json").resolve(strict=True)
    require(sha256_file(X7_SCRIPT) == EXPECTED_X7_SCRIPT_SHA256, "x7_script_drift")
    require(sha256_file(freeze) == EXPECTED_FREEZE_SHA256, "x7_freeze_drift")

    x3_materialization_path = (x3_root / "materialization.json").resolve(strict=True)
    x3_materialization = json.loads(x3_materialization_path.read_text(encoding="utf-8"))
    require(x3_materialization.get("status") == "COMPLETE", "x3_materialization_status")
    require(x3_materialization.get("truth_blind") is True, "x3_materialization_truth")
    require(
        int(x3_materialization.get("optimized_frames", -1)) == SOURCE_SUPPORTED_FRAMES,
        "x3_supported_frame_count",
    )

    sequence_rows = []
    total_frames = total_input = total_removed = 0
    for sequence in SEQUENCES:
        base = root / "sequences" / sequence
        ledger = (base / "lag-floxel.npz").resolve(strict=True)
        manifest_path = (base / "lag-floxel.json").resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("schema") == LEDGER_SCHEMA, f"x7_schema:{sequence}")
        require(manifest.get("truth_blind") is True, f"x7_truth:{sequence}")
        require(manifest.get("ledger_sha256") == sha256_file(ledger), f"x7_hash:{sequence}")
        with np.load(ledger, allow_pickle=False) as values:
            frames = values["frames"]
            offsets = values["offsets"]
            require(len(frames) == int(manifest["frames"]), f"x7_frames:{sequence}")
            require(frames[:WARMUP_FRAMES_PER_SEQUENCE].tolist() == [0, 1, 2, 3], f"x7_warmup_frames:{sequence}")
            require(
                np.diff(offsets[: WARMUP_FRAMES_PER_SEQUENCE + 1]).tolist() == [0, 0, 0, 0],
                f"x7_warmup_not_empty:{sequence}",
            )
        diagnostics = manifest["diagnostics"]
        total_frames += int(manifest["frames"])
        total_input += int(diagnostics["input_cells"])
        total_removed += int(diagnostics["static_cells_removed"])
        sequence_rows.append(
            {
                "sequence": sequence,
                "frames": int(manifest["frames"]),
                "ledger_sha256": sha256_file(ledger),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )

    fail_closed_warmup_frames = TIMELINE_FRAMES - SOURCE_SUPPORTED_FRAMES
    require(total_frames == TIMELINE_FRAMES, f"x7_timeline_frames:{total_frames}")
    require(
        fail_closed_warmup_frames == WARMUP_FRAMES_PER_SEQUENCE * len(SEQUENCES),
        "x7_warmup_total",
    )
    amendment_script_sha = sha256_file(Path(__file__).resolve())
    amendment_path = root / "timeline-frame-count-amendment.json"
    materialization_path = root / "materialization.json"
    materialization = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "COMPLETE",
        "truth_blind": True,
        "sequences": len(sequence_rows),
        "frames": total_frames,
        "source_supported_frames": SOURCE_SUPPORTED_FRAMES,
        "fail_closed_warmup_frames": fail_closed_warmup_frames,
        "input_cells": total_input,
        "static_cells_removed": total_removed,
        "retained_cells": total_input - total_removed,
        "backend": {"python": platform.python_version(), "raw_lidar_decode": "CPU"},
        "freeze": str(freeze),
        "freeze_sha256": sha256_file(freeze),
        "sequence_manifests": sequence_rows,
        "amendment": {
            "reason": "TIMELINE_FRAMES_WERE_COMPARED_WITH_SOURCE_SUPPORTED_FRAMES",
            "script": str(Path(__file__).resolve()),
            "script_sha256": amendment_script_sha,
            "receipt": str(amendment_path),
        },
    }
    write_json(materialization_path, materialization)
    amendment = {
        "schema": "blindassist-dtr-x7-timeline-frame-count-amendment-v1",
        "status": "SEALED",
        "changes_algorithm_outputs": False,
        "timeline_frames": total_frames,
        "source_supported_frames": SOURCE_SUPPORTED_FRAMES,
        "fail_closed_warmup_frames": fail_closed_warmup_frames,
        "x7_script_sha256": sha256_file(X7_SCRIPT),
        "freeze_sha256": sha256_file(freeze),
        "x3_materialization_sha256": sha256_file(x3_materialization_path),
        "materialization_sha256": sha256_file(materialization_path),
        "sequence_outputs": sequence_rows,
    }
    write_json(amendment_path, amendment)
    print(json.dumps(amendment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
