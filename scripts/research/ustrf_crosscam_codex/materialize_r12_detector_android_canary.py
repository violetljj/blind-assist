#!/usr/bin/env python3
"""Materialize seen R1.1 target frames for the R1.2 Android parser canary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from contract import sha256_file


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    ledger_path = args.ledger.resolve()
    source_manifest_path = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "canary_manifest.json"
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite canary directory: {output_dir}")
    ledger = load(ledger_path)
    source_manifest = load(source_manifest_path)
    if ledger.get("diagnostic_set_role") != "seen_diagnostic_not_held_out":
        raise ValueError("canary requires seen R1.1 ledger")
    if source_manifest.get("diagnostic_set_role") != "seen_diagnostic_not_held_out":
        raise ValueError("canary requires seen R1.1 source manifest")

    source_by_event = {item["event_id"]: item for item in source_manifest["sources"]}
    if set(source_by_event) != {item["event_id"] for item in ledger["events"]}:
        raise ValueError("ledger/source event mismatch")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for event in ledger["events"]:
        event_id = event["event_id"]
        source = source_by_event[event_id]
        video_path = (repo_root / source["video_path"]).resolve()
        if sha256_file(video_path) != source["video_sha256"]:
            raise ValueError(f"source video hash mismatch: {event_id}")
        target = event["target_instance"]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open source: {event_id}")
        try:
            for frame in target["frames"]:
                if frame.get("visibility") != "visible":
                    continue
                capture.set(cv2.CAP_PROP_POS_MSEC, float(frame["timestamp_ms"]))
                ok, image = capture.read()
                if not ok or image is None:
                    raise ValueError(f"cannot decode {event_id}/{frame['frame_id']}")
                relative = Path("images") / f"{event_id}_{frame['frame_id']}.png"
                destination = output_dir / relative
                if not cv2.imwrite(str(destination), image):
                    raise ValueError(f"cannot write {destination}")
                rows.append(
                    {
                        "event_id": event_id,
                        "frame_id": frame["frame_id"],
                        "timestamp_ms": frame["timestamp_ms"],
                        "image_asset": f"ustrf_r12_detector/{relative.as_posix()}",
                        "image_sha256": sha256_file(destination),
                        "decoded_width": int(image.shape[1]),
                        "decoded_height": int(image.shape[0]),
                        "target_bbox_xyxy_norm": frame["bbox_xyxy_norm"],
                        "detector_label_allowlist": target["detector_label_allowlist"],
                    }
                )
        finally:
            capture.release()

    payload = {
        "schema": "blindassist_ustrf_r12_android_parser_canary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "seen_r11_diagnostic_parser_canary_not_r12_held_out",
        "source_ledger_sha256": sha256_file(ledger_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "frames": rows,
        "authority": {
            "r12_sources_read": False,
            "threshold_fit": False,
            "training_performed": False,
            "production_model_replacement_authorized": False,
        },
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(manifest_path) + ".sha256").write_text(sha256_file(manifest_path) + "\n", encoding="ascii")
    print("USTRF_R12_ANDROID_CANARY_OK", len(rows), sha256_file(manifest_path))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
