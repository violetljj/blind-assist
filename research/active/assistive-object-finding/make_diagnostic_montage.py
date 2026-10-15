"""Independent direct-source alignment and visual audit of cached RVOS masks.

This deliberately does not import the replay adapter or its RLE/metric code.
CPU only: TASK_NOT_GPU_SUITABLE. No model execution or new downloads.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def decode(rle, expected_shape):
    if rle is None:
        return np.zeros(expected_shape, dtype=bool)
    assert tuple(rle["size"]) == expected_shape
    encoded = rle["counts"]
    if isinstance(encoded, list):
        runs = encoded
    else:
        runs, cursor = [], 0
        while cursor < len(encoded):
            value, shift = 0, 0
            while True:
                digit = ord(encoded[cursor]) - 48
                cursor += 1
                value |= (digit & 31) << shift
                shift += 5
                if not digit & 32:
                    if digit & 16:
                        value |= -1 << shift
                    break
            if len(runs) >= 3:
                value += runs[-2]
            runs.append(value)
    assert sum(runs) == np.prod(expected_shape) and min(runs) >= 0
    flat = np.zeros(np.prod(expected_shape), dtype=bool)
    cursor = 0
    for index, run in enumerate(runs):
        if index & 1:
            flat[cursor:cursor + run] = True
        cursor += run
    return flat.reshape(expected_shape, order="F")


def bbox(mask):
    yy, xx = np.nonzero(mask)
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1] if len(xx) else None


def iou(a, b):
    if a is None or b is None:
        return 0.0
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = [json.loads(line) for line in (root / "data/samples.jsonl").read_text().splitlines()]
    metadata = json.loads((root / "data/mevis_meta_expressions_v2.json").read_bytes())["videos"]
    gt_source = json.loads((root / "data/mevis_mask_dict.json").read_bytes())
    source_bytes = (root / "official_streamingrvos/results.json").read_bytes()
    source = json.loads(source_bytes)
    checked, missing, unique_queries = [], [], set()
    for row in rows:
        vid, qid = row["source_id"], row["query_id"]
        item = source.get(vid, {}).get(qid)
        if item is None:
            missing.append((vid, qid))
            continue
        assert item["video_id"] == vid and str(item["exp_id"]) == qid
        assert metadata[vid]["expressions"][qid]["exp"] == row["query"]
        assert item["exp"] in (f"<image>\\nPlease segment {row['query']}.",
                                f"<image>\nPlease segment {row['query']}.")
        assert item["frames"] == metadata[vid]["frames"]
        assert len(item["frames"]) == len(set(item["frames"])) == len(item["prediction_masks"])
        frame = Path(row["image_path"]).stem
        assert int(frame) == row["frame_idx"]
        source_index = item["frames"].index(frame)
        gt_index = metadata[vid]["frames"].index(frame)
        assert source_index == gt_index
        shape = row["height"], row["width"]
        target_ids = metadata[vid]["expressions"][qid]["anno_id"]
        assert len(target_ids) == 1 and str(target_ids[0]) == row["annotation_id"]
        gt = decode(gt_source[str(target_ids[0])][gt_index], shape)
        pred = decode(item["prediction_masks"][source_index], shape)
        gb, pb = bbox(gt), bbox(pred)
        assert gb == row["gt_bbox_xyxy"] and bool(gt.any()) == row["gt_present"]
        row = dict(row)
        row.update(pred_bbox=pb, pred_pixels=int(pred.sum()), box_iou=iou(gb, pb),
                   source_index=source_index, gt_index=gt_index)
        checked.append(row)
        unique_queries.add((vid, qid))
    selected, used = [], set()
    groups = (("absent_false_output", lambda r: not r["gt_present"] and r["pred_pixels"] > 0,
               lambda r: -r["pred_pixels"]),
              ("visible_wrong_or_missed", lambda r: r["gt_present"] and r["box_iou"] < .5,
               lambda r: (r["box_iou"], -r["pred_pixels"])),
              ("visible_correct", lambda r: r["gt_present"] and r["box_iou"] >= .5,
               lambda r: -r["box_iou"]))
    for group, include, sort_key in groups:
        count = 0
        for row in sorted(filter(include, checked), key=sort_key):
            if row["source_id"] in used:
                continue
            row = dict(row, diagnostic_group=group)
            selected.append(row)
            used.add(row["source_id"])
            count += 1
            if count == 4:
                break
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        big = ImageFont.truetype("arialbd.ttf", 17)
    except OSError:
        font = ImageFont.load_default(size=15)
        big = ImageFont.load_default(size=17)
    canvas = Image.new("RGB", (1440, 4 * 360 + 40), "#ececec")
    ImageDraw.Draw(canvas).text((8, 8), "GT = green | official StreamingRVOS-1B = magenta | masks and boxes in original image coordinates", font=big, fill="black")
    for index, row in enumerate(selected):
        vid, qid = row["source_id"], row["query_id"]
        item = source[vid][qid]
        shape = row["height"], row["width"]
        gt = decode(gt_source[row["annotation_id"]][row["gt_index"]], shape)
        pred = decode(item["prediction_masks"][row["source_index"]], shape)
        with Image.open(row["image_path"]) as im:
            assert im.size == (shape[1], shape[0])
            rgb = np.asarray(im.convert("RGB")).astype(np.float32)
        rgb[gt] = .55 * rgb[gt] + .45 * np.array([0, 255, 0])
        rgb[pred] = .55 * rgb[pred] + .45 * np.array([255, 0, 255])
        image = Image.fromarray(rgb.astype(np.uint8))
        image.thumbnail((480, 250))
        x, y = index % 3 * 480, index // 3 * 360 + 40
        canvas.paste(image, (x, y))
        drawer = ImageDraw.Draw(canvas)
        for box, color in ((row["gt_bbox_xyxy"], "#00ff00"), (row["pred_bbox"], "#ff00ff")):
            if box is not None:
                sx, sy = image.width / shape[1], image.height / shape[0]
                drawer.rectangle((x + box[0] * sx, y + box[1] * sy,
                                  x + box[2] * sx, y + box[3] * sy), outline=color, width=3)
        label = f"{row['diagnostic_group']} | {vid}/q{qid}/f{row['frame_idx']}"
        drawer.text((x + 4, y + 253), label, font=font, fill="black")
        drawer.text((x + 4, y + 272), f"GT {'present' if row['gt_present'] else 'ABSENT'}; pred {row['pred_pixels']} px; bbox IoU {row['box_iou']:.3f}", font=font, fill="black")
        drawer.text((x + 4, y + 292), "\n".join(textwrap.wrap(row["query"], width=56)), font=font, fill="black")
    out = root / "official_streamingrvos"
    canvas.save(out / "diagnostic_streaming_montage.jpg", quality=93)
    result = {"source_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "independent_adapter_imports": False, "matched_rows": len(checked),
              "missing_rows": len(missing), "exact_query_text_matches": len(unique_queries),
              "source_frame_order_matches_metadata": True, "mask_dimensions_match_images": True,
              "native_gt_recomputed_boxes_match_manifest": True,
              "visible": sum(r["gt_present"] for r in checked),
              "visible_bbox_success_iou05": sum(r["gt_present"] and r["box_iou"] >= .5 for r in checked),
              "absent": sum(not r["gt_present"] for r in checked),
              "absent_nonempty_prediction": sum(not r["gt_present"] and r["pred_pixels"] > 0 for r in checked),
              "selected_diagnostic_groups": dict(Counter(r["diagnostic_group"] for r in selected)),
              "selected_rows": selected,
              "limits": "Source cache replay; no local model runtime, no confidence scores; enriched Development windows."}
    (out / "independent_mapping_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "selected_rows"}, indent=2))


if __name__ == "__main__":
    main()
