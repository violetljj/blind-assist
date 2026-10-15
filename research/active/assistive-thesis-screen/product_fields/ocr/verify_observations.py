"""Check observation provenance, row count, and native-coordinate integrity."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_ocr import sha, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from PIL import Image, ImageOps

    inputs = json.loads((args.input / "image_receipts.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    terminal = json.loads((args.output / "terminal.json").read_text(encoding="utf-8"))
    assert len(inputs) == len(rows) == 96
    assert len({r["image"] for r in rows}) == 96
    assert terminal["status"] == "complete" and terminal["completed"] == 96
    assert sha(args.output / "predictions.jsonl") == terminal["predictions_sha256"]
    boxes = 0
    unavailable_dimensions = []
    coordinate_images = []
    for receipt, row in zip(inputs, rows):
        assert receipt["image"] == row["image"]
        assert receipt["sha256"] == row["source_sha256"]
        source = args.input / "images" / row["image"]
        assert sha(source) == row["source_sha256"]
        with Image.open(source) as stored:
            raw_w, raw_h = stored.size
            orientation = stored.getexif().get(274, 1)
            w, h = ImageOps.exif_transpose(stored).size
        coordinate_images.append({"image": row["image"], "stored_raster_width": raw_w,
                                  "stored_raster_height": raw_h, "exif_orientation": orientation,
                                  "observation_width": w, "observation_height": h})
        if "height" in row and "width" in row:
            assert (h, w) == (row["height"], row["width"])
        else:
            assert row["status"] in ["no_text", "error", "upstream_error"]
            unavailable_dimensions.append({"image": row["image"], "width": w, "height": h})
        for line in row["lines"]:
            assert isinstance(line["text"], str)
            confidence = line["confidence"]
            assert math.isfinite(confidence) and 0 <= confidence <= 1
            assert len(line["box"]) == 4
            for x, y in line["box"]:
                assert math.isfinite(x) and math.isfinite(y)
                assert 0 <= x <= w and 0 <= y <= h
            assert line["accepted_at_default"] == (bool(line["text"].strip()) and confidence >= 0.5)
            boxes += 1
    result = {"status": "passed", "images": len(rows), "boxes": boxes,
              "input_bytes_verified": sum(r["bytes"] for r in inputs),
              "checks": ["exact 96-row order and unique images", "source and output SHA256", "native image dimensions", "finite in-bounds quadrilaterals", "finite recognition scores in [0,1]", "explicit default score acceptance"],
              "source_dimensions_for_no_text_or_failed_rows": unavailable_dimensions}
    write_json(args.output / "validation.json", result)
    write_json(args.output / "coordinate_contract.json", {
        "coordinate_space": "EXIF-transposed full-resolution image pixels, not JPEG stored raster",
        "load_policy": "RapidOCR path input uses PIL.ImageOps.exif_transpose before ndarray conversion; supplemental OpenCV imread also honors EXIF orientation",
        "required_consumer_policy": "Apply PIL.ImageOps.exif_transpose to source image before crop, dimension, or spatial feature calculations; do not apply raw-PIL coordinates to these boxes",
        "manifest_clarification": "The original manifest phrase original image pixel coordinates refers to the loaded EXIF-oriented image. This sidecar makes that implicit loader behavior explicit without altering observation outputs.",
        "nonidentity_exif_images": sum(x["exif_orientation"] != 1 for x in coordinate_images),
        "images": coordinate_images,
    })
    print(json.dumps(result))


if __name__ == "__main__":
    main()
