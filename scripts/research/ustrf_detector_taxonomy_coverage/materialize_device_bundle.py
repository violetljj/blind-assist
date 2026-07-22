from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from run_host_coverage import SOURCE_BUNDLES, read_json, selected_frame_ids, sha256


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def validate_host_ledgers(config_path: Path, ledger_paths: list[Path]) -> tuple[dict, list[dict]]:
    config = read_json(config_path)
    config_sha = sha256(config_path)
    windows_path = Path(config["parent"]["windows_path"])
    if sha256(windows_path) != config["parent"]["windows_sha256"]:
        raise ValueError("window hash mismatch")
    windows = read_json(windows_path)
    ledgers = [read_json(path) for path in ledger_paths]
    if len(ledgers) != len(SOURCE_BUNDLES):
        raise ValueError("host ledger source count mismatch")
    seen_sources: set[str] = set()
    for ledger in ledgers:
        if ledger.get("schema") != "blindassist_ustrf_detector_taxonomy_host_ledger_v1":
            raise ValueError("unexpected host ledger schema")
        source_name = ledger["source_name"]
        if source_name not in SOURCE_BUNDLES or source_name in seen_sources:
            raise ValueError("missing, duplicate, or unexpected host ledger source")
        seen_sources.add(source_name)
        if ledger["config_sha256"] != config_sha or ledger["windows_sha256"] != config["parent"]["windows_sha256"]:
            raise ValueError("host ledger config/window binding mismatch")
        if ledger["model_sha256"] != config["detector"]["model_sha256"] or ledger["labels_sha256"] != config["detector"]["labels_sha256"]:
            raise ValueError("host ledger model/labels binding mismatch")
        if ledger["input_shape"] != config["detector"]["input_shape"] or ledger["output_shape"] != config["detector"]["output_shape"]:
            raise ValueError("host ledger tensor contract mismatch")
        if ledger["confidence_threshold"] != config["detector"]["confidence_threshold"] or ledger["nms_iou_threshold"] != config["detector"]["nms_iou_threshold"]:
            raise ValueError("host ledger threshold/NMS drift")
        expected_count = config["parent"]["source_frame_counts"].get(ledger["source_id"])
        rows = ledger["frames"]
        frame_ids = [row["frame_id"] for row in rows]
        expected_ids = selected_frame_ids(windows, ledger["source_id"])
        if expected_count is None or ledger["frame_count"] != expected_count or len(rows) != expected_count:
            raise ValueError("host ledger per-source frame count mismatch")
        if len(set(frame_ids)) != len(frame_ids) or frame_ids != expected_ids:
            raise ValueError("host ledger frame identity/order mismatch")
    if seen_sources != set(SOURCE_BUNDLES):
        raise ValueError("incomplete host ledger sources")
    if sum(int(row["frame_count"]) for row in ledgers) != config["parent"]["frame_count"]:
        raise ValueError("host ledger total frame count mismatch")
    return config, ledgers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host-ledger", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite device bundle: {args.output_dir}")
    config, ledgers = validate_host_ledgers(args.config, args.host_ledger)
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
        "windows_sha256": config["parent"]["windows_sha256"],
        "frame_count": len(manifest_rows),
        "source_frame_counts": config["parent"]["source_frame_counts"],
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
