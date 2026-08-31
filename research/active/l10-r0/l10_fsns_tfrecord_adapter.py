#!/usr/bin/env python3
"""Materialize one official FSNS TFRecord shard without changing its truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tensorflow as tf


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_int(feature: object) -> int:
    values = feature.int64_list.value
    if not values:
        raise ValueError("missing int64 scalar")
    return int(values[0])


def scalar_bytes(feature: object) -> bytes:
    values = feature.bytes_list.value
    if not values:
        raise ValueError("missing bytes scalar")
    return bytes(values[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    input_path, output_dir = args.input.resolve(), args.output_dir.resolve()
    manifest_path = output_dir / "manifest.jsonl"
    receipt_path = output_dir / "receipt.json"
    if manifest_path.exists() or receipt_path.exists():
        raise FileExistsError(f"adapter output already exists: {output_dir}")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, raw in enumerate(tf.data.TFRecordDataset([str(input_path)])):
        example = tf.train.Example.FromString(bytes(raw.numpy()))
        features = example.features.feature
        encoded = scalar_bytes(features["image/encoded"])
        image_format = scalar_bytes(features["image/format"]).decode("ascii").lower()
        if image_format != "png":
            raise ValueError(f"unexpected image format at {index}: {image_format}")
        width = scalar_int(features["image/width"])
        original_width = scalar_int(features["image/orig_width"])
        height = scalar_int(features["image/height"])
        if width != 600 or height != 150 or original_width % 150:
            raise ValueError(f"unexpected FSNS geometry at {index}: {width}x{height}/{original_width}")
        image_path = images_dir / f"{index:06d}.png"
        image_path.write_bytes(encoded)
        rows.append(
            {
                "schema": "blindassist-l10-fsns-example-v1",
                "index": index,
                "image": image_path.relative_to(output_dir).as_posix(),
                "image_sha256": hashlib.sha256(encoded).hexdigest(),
                "image_bytes": len(encoded),
                "width": width,
                "height": height,
                "original_width": original_width,
                "view_count": original_width // 150,
                "text": scalar_bytes(features["image/text"]).decode("utf-8"),
            }
        )
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    receipt = {
        "schema": "blindassist-l10-fsns-adapter-receipt-v1",
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "examples": len(rows),
        "unique_texts": len({row["text"] for row in rows}),
        "view_count_histogram": {
            str(count): sum(row["view_count"] == count for row in rows)
            for count in sorted({row["view_count"] for row in rows})
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "truth_boundary": "FSNS image/text is the normalized canonical street name for the multi-view sign sample; it is not a per-tile text box, facade, portal, access, arrival, or safety label.",
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
