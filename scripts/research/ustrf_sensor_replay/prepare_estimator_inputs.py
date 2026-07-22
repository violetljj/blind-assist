from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import read_json, safe_file, sha256, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite output")
    args.output.mkdir(parents=True)
    sources = []
    for source_dir in sorted(args.bundle_root.resolve().iterdir()):
        bundle_path = source_dir / "bundle.json"
        if not bundle_path.is_file():
            continue
        bundle = read_json(bundle_path)
        root = Path(bundle["source_root"]).resolve()
        ledger_path = source_dir / "frames.jsonl"
        rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
        target = args.output / f"{bundle['source']['source_id']}.jsonl"
        sanitized = []
        for row in rows:
            rgb = safe_file(root, row["rgb_path"])
            depth = safe_file(root, row["depth_path"])
            sanitized.append({
                "frame_id": row["frame_id"],
                "timestamp_s": row["rgb_timestamp_s"],
                "rgb_path": str(rgb),
                "depth_path": str(depth),
                "rgb_sha256": row["rgb_sha256"],
                "depth_sha256": row["depth_sha256"],
                "depth_encoding": row["depth_encoding"],
                "depth_scale": row["depth_scale"],
                "intrinsics_fx_fy_cx_cy": row["intrinsics_fx_fy_cx_cy"],
            })
        target.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in sanitized), encoding="utf-8")
        sources.append({
            "source_id": bundle["source"]["source_id"],
            "sanitized_ledger": target.name,
            "sanitized_ledger_sha256": sha256(target),
            "source_frames_sha256": sha256(ledger_path),
            "forbidden_fields_absent": all("camera_to_world" not in row and "pose_timestamp_s" not in row for row in sanitized),
            "frame_count": len(sanitized),
        })
    write_json(args.output / "estimator_inputs.json", {
        "schema": "blindassist_ustrf_sensor_replay_r3_estimator_inputs_v1",
        "sources": sources,
        "ground_truth_pose_fields_exposed": False,
        "production_authority": False,
    })
    print(json.dumps({"sources": len(sources), "ground_truth_pose_fields_exposed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
