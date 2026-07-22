from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from run_host_coverage import SOURCE_BUNDLES, read_json, sha256


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host-ledger", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite device bundle: {args.output_dir}")
    config = read_json(args.config)
    ledgers = [read_json(path) for path in args.host_ledger]
    if sum(int(row["frame_count"]) for row in ledgers) != config["parent"]["frame_count"]:
        raise ValueError("host ledger frame count mismatch")
    manifest_rows: list[dict] = []
    for ledger in ledgers:
        source_name = ledger["source_name"]
        bundle_root = Path("artifacts.local/evidence/ustrf-sensor-replay-r3/source-replacement-lilocbench-v1") / SOURCE_BUNDLES[source_name]
        bundle = read_json(bundle_root / "bundle.json")
        frame_rows = {
            row["frame_id"]: row
            for row in map(json.loads, (bundle_root / "frames.jsonl").read_text(encoding="utf-8").splitlines())
        }
        for row in ledger["frames"]:
            frame = frame_rows[row["frame_id"]]
            source_image = Path(bundle["source_root"]) / frame["rgb_path"]
            relative = Path("images") / source_name / f"{row['frame_id']}.png"
            if sha256(source_image) != row["image_sha256"]:
                raise ValueError(f"host ledger image hash mismatch: {source_name}/{row['frame_id']}")
            link_or_copy(source_image, args.output_dir / relative)
            manifest_rows.append({
                "source_name": source_name,
                "source_id": ledger["source_id"],
                "frame_id": row["frame_id"],
                "image_path": relative.as_posix(),
                "image_sha256": row["image_sha256"],
                "host_input_tensor_sha256": row["input_tensor_sha256"],
                "host_raw_output_sha256": row["raw_output_sha256"],
                "host_raw_person_max_confidence": row["raw_person_max_confidence"],
            })
    manifest_rows.sort(key=lambda row: (row["source_name"], int(row["frame_id"])))
    payload = {
        "schema": "blindassist_ustrf_detector_taxonomy_device_input_v1",
        "config_sha256": sha256(args.config),
        "frame_count": len(manifest_rows),
        "input_shape": config["detector"]["input_shape"],
        "output_shape": config["detector"]["output_shape"],
        "model_sha256": config["detector"]["model_sha256"],
        "labels_sha256": config["detector"]["labels_sha256"],
        "person_class_index": config["detector"]["person_class_index"],
        "confidence_threshold": config["detector"]["confidence_threshold"],
        "nms_iou_threshold": config["detector"]["nms_iou_threshold"],
        "frames": manifest_rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps({"frame_count": len(manifest_rows), "manifest_sha256": sha256(args.output_dir / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
