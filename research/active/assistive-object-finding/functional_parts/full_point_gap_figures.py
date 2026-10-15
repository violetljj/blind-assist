"""Pillow-only figures for completed, consumed IIT validation diagnostics.

No Torch, model execution or logits are imported/read. GT is used only for the
illustration overlay and crop, after deterministic examples were selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

BASE = "original_softmax32"  # full_point_gap_diagnostics.py frozen definition
STABLE = "stable_log_odds64"
BLUE = (28, 93, 211)
RED = (219, 54, 54)
GREEN = (0, 149, 76)
CYAN = (0, 235, 242)
INK = (31, 42, 57)
MUTED = (92, 106, 124)
BG = (246, 248, 251)
WHITE = (255, 255, 255)
FONT_PATHS = (Path("C:/Windows/Fonts/arial.ttf"),
              Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))


def font(size):
    for path in FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def receipt(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def text(draw, xy, value, size=24, color=INK, anchor=None):
    draw.text(xy, str(value), font=font(size), fill=color, anchor=anchor)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def original_inputs(root):
    preferred = root / "diagnostics/dino_point_gap/inputs/analysis"
    analysis = preferred if preferred.is_dir() else root / "analysis"
    models, inputs = {}, {}
    for name in ("segformer", "dino"):
        per_image = analysis / name / "per_image.jsonl"
        summary = analysis / name / "summary.json"
        values = list(rows(per_image))
        if any(value.get("split") not in ("val", "validation") for value in values):
            raise ValueError("Only consumed validation per_image rows may be plotted")
        models[name] = {value["source_id"]: value for value in values}
        if len(values) != len(models[name]):
            raise ValueError("Duplicate source IDs")
        saved = read_json(summary)["overall"]
        if len(values) != saved["images"]:
            raise ValueError("Image denominator differs from completed summary")
        points = [p for value in values for p in value["points"]]
        if (len(points) != saved["point_queries"] or sum(p["hit"] for p in points) != saved["point_hits"]):
            raise ValueError("Point denominator/hits differ from completed summary")
        inputs[name] = {"per_image": receipt(per_image), "summary": receipt(summary)}
    if models["segformer"].keys() != models["dino"].keys():
        raise ValueError("Model validation image sets differ")
    return models, inputs


def paired_counts(models):
    buckets = defaultdict(lambda: {"queries": 0, "segformer": 0, "dino": 0})
    keys = set()
    for sid, seg in models["segformer"].items():
        dino = models["dino"][sid]
        if seg["image_source"] != dino["image_source"]:
            raise ValueError("Source labels differ")
        sp = {p["class_id"]: p for p in seg["points"]}
        dp = {p["class_id"]: p for p in dino["points"]}
        if sp.keys() != dp.keys():
            raise ValueError("Query sets differ")
        for class_id, point in sp.items():
            other = dp[class_id]
            if (point["small_region"] != other["small_region"] or
                    point["target_pixels"] != other["target_pixels"]):
                raise ValueError("Point stratum differs")
            keys.add((sid, class_id))
            bucket = buckets[seg["image_source"], bool(point["small_region"])]
            bucket["queries"] += 1
            bucket["segformer"] += int(point["hit"])
            bucket["dino"] += int(other["hit"])
    return buckets, keys


def bar_figure(buckets, out):
    image = Image.new("RGB", (2200, 1180), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (65, 35), "Point localization: source and target-area mixture", 43)
    text(draw, (65, 100), "Consumed IIT validation | original float32 softmax spatial argmax | fixed 0-100% axis", 25, MUTED)
    text(draw, (65, 143), "Native semantic-class union, not physical instance / safe-contact truth", 24, MUTED)
    left, right, top, bottom = 130, 2125, 220, 750
    for value in range(0, 101, 20):
        y = bottom - value / 100 * (bottom - top)
        draw.line((left, y, right, y), fill=(215, 222, 232), width=2)
        text(draw, (left - 18, y), f"{value}%", 24, MUTED, "rm")
    groups = [("captured_filename_heuristic", True, "Captured filename", "Small target <=1%"),
              ("ImageNet_filename", True, "ImageNet filename", "Small target <=1%"),
              ("captured_filename_heuristic", False, "Captured filename", "Target >1%"),
              ("ImageNet_filename", False, "ImageNet filename", "Target >1%")]
    table = []
    for index, (source, small, label, scale) in enumerate(groups):
        bucket = buckets[source, small]
        count = bucket["queries"]
        if count == 0:
            raise ValueError("Missing expected source-area stratum")
        center = left + (index + 0.5) * (right - left) / 4
        for offset, model, color in [(-89, "segformer", BLUE), (89, "dino", RED)]:
            x = center + offset
            hit = bucket[model]
            height = hit / count * (bottom - top)
            draw.rounded_rectangle((x - 69, bottom - height, x + 69, bottom), radius=8, fill=color)
            text(draw, (x, bottom - height - 45), f"{hit}/{count}", 23, color, "mm")
            text(draw, (x, bottom - height - 17), f"{100 * hit / count:.2f}%", 23, color, "mm")
            table.append({"source": source, "small_region": small, "model": model, "hits": hit, "queries": count})
        text(draw, (center, bottom + 34), label, 27, INK, "mm")
        text(draw, (center, bottom + 71), scale, 25, MUTED, "mm")
    draw.rounded_rectangle((65, 865, 2135, 1090), radius=18, fill=WHITE, outline=(219, 225, 233), width=2)
    total_small = sum(buckets[source, True]["queries"] for source in ("captured_filename_heuristic", "ImageNet_filename"))
    capture = buckets["captured_filename_heuristic", True]
    natural = buckets["ImageNet_filename", True]
    seg_small = capture["segformer"] + natural["segformer"]
    dino_small = capture["dino"] + natural["dino"]
    text(draw, (92, 887), "The overall small-target score is a weighted mixture", 30)
    text(draw, (92, 935), f"Captured: {capture['queries']}/{total_small} = {100*capture['queries']/total_small:.2f}% of small queries; "
         f"ImageNet: {natural['queries']}/{total_small} = {100*natural['queries']/total_small:.2f}%", 27)
    text(draw, (92, 982), f"Overall small: SegFormer {seg_small}/{total_small} ({100*seg_small/total_small:.2f}%)   |   "
         f"DINO {dino_small}/{total_small} ({100*dino_small/total_small:.2f}%)", 27)
    text(draw, (92, 1030), "Filename source is heuristic; captured and multiple-object subsets overlap. These are not independent causal effects.", 23, MUTED)
    draw.rectangle((80, 1123, 109, 1152), fill=BLUE)
    text(draw, (121, 1122), "SegFormer-B0", 25)
    draw.rectangle((430, 1123, 459, 1152), fill=RED)
    text(draw, (471, 1122), "DINO-S/16", 25)
    path = out / "01_source_area_point_hit.png"
    image.save(path)
    return path, table


def marker(draw, xy, kind, color, size=9):
    x, y = xy
    if kind == "S":
        draw.ellipse((x-size, y-size, x+size, y+size), fill=color, outline=WHITE, width=3)
    elif kind == "D":
        for width, fill in [(9, WHITE), (4, color)]:
            draw.line((x-size, y-size, x+size, y+size), fill=fill, width=width)
            draw.line((x-size, y+size, x+size, y-size), fill=fill, width=width)
    else:
        draw.polygon([(x, y-size-2), (x+size+2, y), (x, y+size+2), (x-size-2, y)], fill=color, outline=WHITE, width=3)


def panel(canvas, rgb, target, crop, box, points):
    x0, y0, x1, y1 = crop
    width, height = x1-x0, y1-y0
    rgb = rgb.crop(crop).convert("RGBA")
    binary = target.crop(crop)
    alpha = binary.point(lambda value: 56 if value else 0)
    overlay = Image.new("RGBA", rgb.size, CYAN + (0,))
    overlay.putalpha(alpha)
    rgb = Image.alpha_composite(rgb, overlay)
    border = binary.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    outline = Image.new("RGBA", rgb.size, CYAN + (0,))
    outline.putalpha(border)
    rgb = Image.alpha_composite(rgb, outline).convert("RGB")
    bx, by, bw, bh = box
    resized = ImageOps.contain(rgb, (bw, bh), Image.Resampling.LANCZOS)
    ox, oy = bx+(bw-resized.width)//2, by+(bh-resized.height)//2
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((bx, by, bx+bw, by+bh), fill=(230, 234, 240))
    canvas.paste(resized, (ox, oy))
    visibility = {}
    for kind, point, color in points:
        visible = point is not None and x0 <= point[0] < x1 and y0 <= point[1] < y1
        visibility[kind] = visible
        if visible:
            # Pixel centers remain centered under the display-only resampling.
            px = ox + (point[0]-x0+0.5)*resized.width/width-0.5
            py = oy + (point[1]-y0+0.5)*resized.height/height-0.5
            marker(draw, (px, py), kind, color)
    return visibility


def example_pages(root, examples, models, diagnostic_rows, metadata, out):
    pages, example_receipts = [], []
    for page in range(math.ceil(len(examples)/4)):
        canvas = Image.new("RGB", (2200, 1760), BG)
        draw = ImageDraw.Draw(canvas)
        text(draw, (45, 28), f"Selected point-localization examples ({page+1}/{math.ceil(len(examples)/4)})", 39)
        text(draw, (45, 82), "Illustrative selected extremes, not a random audit. GT cyan overlay / outline is for display only.", 25, MUTED)
        text(draw, (45, 122), "Blue circle: Seg original | Red cross: DINO original | Green diamond: DINO stable log-odds64", 25, MUTED)
        for local, example in enumerate(examples[page*4:page*4+4]):
            index = page*4+local
            sid, class_id = example["source_id"], int(example["class_id"])
            meta = metadata[sid]
            if meta.get("split") not in ("val", "validation"):
                raise ValueError("Example is not existing validation")
            image_path = root / "data/iit/images" / PureWindowsPath(meta["image"]).name
            mask_path = root / "data/iit/masks" / PureWindowsPath(meta["mask"]).name
            hashes = {"image": receipt(image_path), "mask": receipt(mask_path)}
            for name in ("image", "mask"):
                expected = meta.get(name+"_sha256")
                if expected and expected != hashes[name]["sha256"]:
                    raise ValueError(f"Frozen {name} SHA mismatch: {sid}")
            with Image.open(image_path) as value:
                rgb = value.convert("RGB")
            with Image.open(mask_path) as value:
                if value.mode not in ("L", "P"):
                    raise ValueError("Expected native byte-valued L/P mask")
                label = Image.frombytes("L", value.size, value.tobytes())
            if rgb.size != label.size or rgb.size != (meta["width"], meta["height"]):
                raise ValueError("RGB, mask and metadata dimensions disagree")
            target = label.point(lambda value: 255 if value == class_id else 0)
            bbox = target.getbbox()
            if bbox is None:
                raise ValueError("Selected positive class has no target pixels")
            diag = diagnostic_rows[sid, class_id]
            if diag.get("native_shape") != [rgb.height, rgb.width]:
                raise ValueError("Diagnostic shape mismatch")
            selected_points = {}
            for model in ("segformer", "dino"):
                saved = next(p for p in models[model][sid]["points"] if p["class_id"] == class_id)
                actual = diag["models"][model]["methods"][BASE]
                if list(saved["point_xy"]) != list(actual["point_xy"]) or saved["hit"] != actual["hit"]:
                    raise ValueError("Diagnostic original point differs from saved baseline")
                if (example[model[:3]+"_hit"] if model == "segformer" else example["dino_hit"]) != saved["hit"]:
                    raise ValueError("Example selection no longer matches baseline hits")
                selected_points[model] = actual
            stable = diag["models"]["dino"]["methods"][STABLE]
            points = [("S", selected_points["segformer"]["point_xy"], BLUE),
                      ("D", selected_points["dino"]["point_xy"], RED),
                      ("L", stable["point_xy"], GREEN)]
            for _, xy, _ in points:
                if xy is not None and (len(xy) != 2 or not (0 <= xy[0] < rgb.width and 0 <= xy[1] < rgb.height)):
                    raise ValueError("Point outside native image")
            pad = max(16, round(0.22*max(bbox[2]-bbox[0], bbox[3]-bbox[1])))
            crop = (max(0, bbox[0]-pad), max(0, bbox[1]-pad), min(rgb.width, bbox[2]+pad), min(rgb.height, bbox[3]+pad))
            x = 35 + (local % 2)*1080
            y = 190 + (local // 2)*750
            draw.rounded_rectangle((x, y, x+1050, y+720), radius=14, fill=WHITE, outline=(216, 223, 232), width=2)
            text(draw, (x+18, y+16), f"{index+1}. {sid} | {example['class_name']}", 24)
            text(draw, (x+18, y+52), f"Native {rgb.width}x{rgb.height}; target {100*diag['target_fraction']:.3f}%; "
                 f"{'DINO rescue' if example['event']=='dino_win' else 'DINO loss'}", 22, MUTED)
            text(draw, (x+18, y+91), "Whole RGB + target GT", 22)
            text(draw, (x+546, y+91), "Target GT crop (display only)", 22)
            panel(canvas, rgb, target, (0, 0, rgb.width, rgb.height), (x+15, y+122, 500, 390), points)
            visible = panel(canvas, rgb, target, crop, (x+535, y+122, 500, 390), points)
            draw = ImageDraw.Draw(canvas)
            for line, model, color in [(0, "segformer", BLUE), (1, "dino", RED)]:
                value = selected_points[model]
                name = "Seg original" if model == "segformer" else "DINO original"
                iou = example["seg_class_IoU" if model == "segformer" else "dino_class_IoU"]
                text(draw, (x+18, y+530+line*34), f"{name}: {value['point_xy']}, {'HIT' if value['hit'] else 'MISS'}; class IoU {100*iou:.2f}%", 22, color)
            text(draw, (x+18, y+598), f"DINO stable64: {stable['point_xy']}, {'HIT' if stable['hit'] else 'MISS'} (same logits, T=1 ranking)", 22, GREEN)
            outside = [kind for kind, shown in visible.items() if not shown]
            note = "Outside target crop: " + ", ".join(outside) if outside else "All three points are inside the target crop."
            text(draw, (x+18, y+636), note, 20, MUTED)
            text(draw, (x+18, y+670), "IoU uses argmax segmentation; a spatial score peak can hit even when class TP=0.", 19, MUTED)
            example_receipts.append({"selection": example, "inputs": hashes, "native_size": list(rgb.size),
                                     "GT_display_crop_xyxy": list(crop), "zoom_point_visibility": visible,
                                     "points": {"segformer_original": selected_points["segformer"],
                                                "dino_original": selected_points["dino"], "dino_stable64": stable}})
        text(draw, (45, 1713), "GT defines only the explanatory cyan region/crop; no new inference, training, selection rule or test access.", 23, MUTED)
        path = out / f"{page+2:02d}_selected_examples.png"
        canvas.save(path)
        pages.append(path)
    return pages, example_receipts


def run(args):
    started = time.monotonic()
    root = args.root.resolve(strict=True)
    diag = args.diag_out.resolve(strict=True)
    terminal_path = diag / "terminal.json"
    summary_path = diag / "summary.json"
    # Do not touch per_query, RGB or GT for an incomplete diagnostic run.
    terminal = read_json(terminal_path)
    diagnostic_summary = read_json(summary_path)
    if terminal.get("status") != "COMPLETE" or diagnostic_summary.get("status") != "COMPLETE":
        raise ValueError("Completed full diagnostic terminal and summary are required")
    if (terminal.get("completed_images") != terminal.get("frozen_images") or
            not diagnostic_summary.get("all_processed_original_points_and_confusions_exactly_reproduced")):
        raise ValueError("Diagnostic did not complete the frozen set / exact baseline reproduction")
    selection = read_json(args.examples)
    examples = selection["examples"]
    if len(examples) != 8 or len({(e["source_id"], e["class_id"]) for e in examples}) != len(examples):
        raise ValueError("Expected eight unique, previously selected examples")
    models, inputs = original_inputs(root)
    buckets, expected_keys = paired_counts(models)
    selected_keys = {(e["source_id"], int(e["class_id"])) for e in examples}
    per_query_path = diag / "per_query.jsonl"
    selected = {}
    seen = set()
    for value in rows(per_query_path):
        key = (value["source_id"], int(value["class_id"]))
        if key in seen:
            raise ValueError("Duplicate diagnostic query")
        seen.add(key)
        if key in selected_keys:
            selected[key] = value
    if seen != expected_keys or len(seen) != terminal["positive_queries"] or selected.keys() != selected_keys:
        raise ValueError("Full diagnostic query denominator/set mismatch")
    metadata_path = root / "metadata/iit_training_manifest.jsonl"
    metadata = {r["source_id"]: r for r in rows(metadata_path) if r["source_id"] in {key[0] for key in selected_keys}}
    if len(metadata) != len({key[0] for key in selected_keys}):
        raise ValueError("Missing example metadata")
    out = args.out.resolve()
    if root not in out.parents:
        raise ValueError("Output must be a new directory under task --root")
    out.mkdir(parents=True, exist_ok=False)
    bar, counts = bar_figure(buckets, out)
    pages, examples_manifest = example_pages(root, examples, models, selected, metadata, out)
    manifest = {"status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.monotonic()-started, "backend": "Pillow + Python stdlib; no Torch / models / logits",
                "scope": "posthoc consumed native semantic-union validation; no test; not physical role or safety evidence",
                "inputs": {"script": receipt(__file__), "selection": receipt(args.examples), "metadata": receipt(metadata_path),
                           "original": inputs, "diagnostic_terminal": receipt(terminal_path),
                           "diagnostic_summary": receipt(summary_path), "diagnostic_per_query": receipt(per_query_path)},
                "selection_explanation": {key: selection.get(key) for key in ("selection_scope", "tie_break", "caveat")},
                "plot_contract": {"bar_axis_percent": [0, 100], "bar_metric": BASE,
                                  "stable_method": STABLE, "GT_usage": "display overlay and target crop only; examples fixed in prior selection JSON",
                                  "point_display_transform": "pixel center affine mapping; no point recomputation"},
                "bar_counts": counts, "examples": examples_manifest,
                "outputs": [receipt(path) for path in [bar]+pages]}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "out": str(out), "images": len(pages)+1,
                      "wall_seconds": time.monotonic()-started}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--diag-out", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
