#!/usr/bin/env python3
"""Materialize the fixed 120-frame DA V2 model-variant validation roster."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_dav2_model_variant_gate_r0 import sha256_file
from evaluate_metric3d_clearance_field_a0 import _depth_lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.dense_manifest.read_text(encoding="utf-8"))
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("the frozen source manifest must contain exactly 120 rows")
    source_root = args.source_root.resolve()
    lookups: dict[str, dict[str, Path]] = {}
    output_rows = []
    for index, row in enumerate(rows):
        old_path = Path(str(row["frame_path"]))
        sequence_root = old_path.parent.parent.name
        sequence_path = source_root / sequence_root
        rgb_path = sequence_path / "rgb" / old_path.name
        if sequence_root not in lookups:
            lookups[sequence_root] = _depth_lookup(sequence_path)
        depth_path = lookups[sequence_root].get(str(rgb_path.resolve()))
        if depth_path is None:
            raise ValueError(f"no registered depth for {rgb_path}")
        output_rows.append(
            {
                "index": index,
                "frame_id": f"{sequence_root}:{rgb_path.stem}",
                "sequence_root": sequence_root,
                "sequence_id": str(row["sequence_id"]),
                "timestamp": float(row["timestamp"]),
                "rgb_path": f"rgb/{rgb_path.name}",
                "depth_path": f"depth/{depth_path.name}",
                "rgb_sha256": sha256_file(rgb_path),
                "depth_sha256": sha256_file(depth_path),
                "intrinsics_fx_fy_cx_cy": row["intrinsics_fx_fy_cx_cy"],
            }
        )
    sequence_counts: dict[str, int] = {}
    for row in output_rows:
        sequence = str(row["sequence_id"])
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
    if sorted(sequence_counts.values()) != [30, 30, 30, 30]:
        raise ValueError("expected four fixed 30-frame windows")
    output = {
        "schema": "blindassist_dav2_model_variant_gate_r0_roster",
        "data_role": "consumed_development_engineering_regression_only",
        "source_manifest_sha256": sha256_file(args.dense_manifest),
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
