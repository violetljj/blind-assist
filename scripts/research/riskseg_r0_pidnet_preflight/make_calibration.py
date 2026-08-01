from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.research.riskseg_r0_pidnet_preflight.modeling import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    sha256_file,
)


def _jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(row)
    return rows


def _evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"need {count} rows, found {len(rows)}")
    indices = np.linspace(0, len(rows) - 1, num=count, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-session", type=int, default=8)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if args.per_session <= 0:
        raise ValueError("--per-session must be positive")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _jsonl(manifest):
        if row.get("role") == "train":
            grouped[str(row["source_session_id"])].append(row)
    if not grouped:
        raise ValueError("manifest has no train rows")

    selected: list[dict] = []
    for session_id in sorted(grouped):
        session_rows = sorted(
            grouped[session_id],
            key=lambda row: (int(row["source_frame_id"]), str(row["id"])),
        )
        selected.extend(_evenly_spaced(session_rows, args.per_session))

    calibration = np.empty(
        (len(selected), INPUT_HEIGHT, INPUT_WIDTH, 3),
        dtype=np.float32,
    )
    selected_receipt: list[dict] = []
    for index, row in enumerate(selected):
        image_path = (repo_root / str(row["source_image_path"])).resolve()
        if sha256_file(image_path) != row["source_image_sha256"]:
            raise ValueError(f"source image hash mismatch: {image_path}")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").resize(
                (INPUT_WIDTH, INPUT_HEIGHT),
                resample=Image.Resampling.BILINEAR,
            )
            array = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
        calibration[index] = array
        selected_receipt.append(
            {
                "id": row["id"],
                "source_session_id": row["source_session_id"],
                "source_frame_id": row["source_frame_id"],
                "source_image_path": row["source_image_path"],
                "source_image_sha256": row["source_image_sha256"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "calibration_nhwc_0_1.npy"
    np.save(npy_path, calibration, allow_pickle=False)
    receipt = {
        "schema_version": "blindassist.riskseg_r0.pidnet_calibration.v1",
        "protocol_id": "RISKSEG_R0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "train_only_quantization_calibration",
        "selection": "sorted_session_then_evenly_spaced_frames",
        "per_session": args.per_session,
        "session_count": len(grouped),
        "sample_count": len(selected),
        "manifest_path": str(manifest.relative_to(repo_root)).replace("\\", "/"),
        "manifest_sha256": sha256_file(manifest),
        "array_path": npy_path.name,
        "array_sha256": sha256_file(npy_path),
        "array_shape": list(calibration.shape),
        "array_dtype": str(calibration.dtype),
        "array_range": [float(calibration.min()), float(calibration.max())],
        "normalization_deferred_to_converter": True,
        "samples": selected_receipt,
    }
    receipt_path = output_dir / "calibration_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
