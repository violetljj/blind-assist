"""Add all sealed M1-PD residual cells to the C28/C29 causal authority trace.

This is a truth-blind representation materializer.  It does not score labels or
select an authority rule.  M1-PD cells already encode reciprocal, ego-compensated
raw-point direct velocity; C30 exposes them alongside the C28 ray states so a
downstream policy can search local/temporal authority structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import dtr_c27_persistent_point_support as c27
from jrdb_rgb_bridge import require, sha256_file, write_json


REPO = Path(__file__).resolve().parents[3]
SCHEMA = "blindassist-dtr-c30-truth-blind-raw-residual-authority-trace-v1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.c29_trace.resolve(strict=True)
    c25_path = args.c25_predictions.resolve(strict=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    c25 = json.loads(c25_path.read_text(encoding="utf-8"))
    require(
        source.get("schema") == "blindassist-dtr-c29-truth-blind-authority-trace-v1"
        and source.get("truth_blind") is True,
        "c30_source_trace_contract",
    )
    c25_rows = {str(row["sequence"]): row for row in c25["sequences"]}
    output_rows = []
    raw_cells = 0
    for sequence_row in source["sequences"]:
        sequence = str(sequence_row["sequence"])
        pd_source = c25_rows[sequence]["sources"]["ledgers"]["M1_PD_GLOBAL"]
        pd = c27._load_arrays(
            Path(pd_source["ledger"]),
            Path(pd_source["manifest"]),
            {
                "frames", "offsets", "forward_m", "left_m",
                "velocity_forward_mps", "velocity_left_mps",
                "source_point_count", "flow_support",
            },
        )
        trace_frames = sequence_row["frames"]
        require(
            [int(row["frame"]) for row in trace_frames]
            == [int(value) for value in pd["frames"]],
            f"c30_frame_drift:{sequence}",
        )
        enriched_frames = []
        for index, frame_row in enumerate(trace_frames):
            rows = [dict(row) for row in frame_row["rows"]]
            start, stop = int(pd["offsets"][index]), int(pd["offsets"][index + 1])
            for point_index in range(start, stop):
                support = min(
                    1.0,
                    float(pd["source_point_count"][point_index]) / 3.0,
                    float(pd["flow_support"][point_index]),
                )
                rows.append(
                    {
                        "lineage_id": -(point_index + 1),
                        "status": "RAW_PD_RESIDUAL",
                        "visibility": "HIT",
                        "emitted": False,
                        "age_s": 0.0,
                        "height_voxels": 1,
                        "seed_confidence": support,
                        "source_point_count": int(pd["source_point_count"][point_index]),
                        "flow_support": float(pd["flow_support"][point_index]),
                        "q": support,
                        "h": 1.0,
                        "w": support,
                        "dp_m": 0.0,
                        "dv_mps": 0.0,
                        "forward_m": float(pd["forward_m"][point_index]),
                        "left_m": float(pd["left_m"][point_index]),
                        "velocity_forward_mps": float(pd["velocity_forward_mps"][point_index]),
                        "velocity_left_mps": float(pd["velocity_left_mps"][point_index]),
                    }
                )
            raw_cells += stop - start
            enriched_frames.append(
                {
                    "frame": int(frame_row["frame"]),
                    "frame_time_s": float(frame_row["frame_time_s"]),
                    "rows": rows,
                }
            )
        output_rows.append(
            {
                "sequence": sequence,
                "frames": enriched_frames,
                "status_counts": {
                    **sequence_row["status_counts"],
                    "RAW_PD_RESIDUAL": int(pd["offsets"][-1]),
                },
            }
        )
    result = {
        "schema": SCHEMA,
        "truth_blind": True,
        "prediction_boundary": "C28 causal lineage/ray features plus every sealed M1-PD reciprocal raw-point residual; no labels, sequence identity, frame id, or future truth passed to policy",
        "sequences": output_rows,
        "diagnostics": {"raw_pd_residual_cells": raw_cells},
        "source": {
            "c29_trace_sha256": sha256_file(source_path),
            "c25_predictions_sha256": sha256_file(c25_path),
        },
    }
    write_json(args.output.resolve(), result)
    return result


def parse_args() -> argparse.Namespace:
    c28 = REPO / "artifacts.local" / "evidence" / "dtr-c28" / "visibility-conditioned-point-memory"
    c25 = REPO / "artifacts.local" / "evidence" / "dtr-c25" / "fresh-point-flow-confirmation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--c29-trace", type=Path, default=c28 / "authority-trace.json")
    parser.add_argument("--c25-predictions", type=Path, default=c25 / "predictions.json")
    parser.add_argument("--output", type=Path, default=REPO / "artifacts.local" / "evidence" / "dtr-c30" / "raw-residual-authority-trace.json")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"schema": result["schema"], **result["diagnostics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
