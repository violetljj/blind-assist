#!/usr/bin/env python3
"""Independent CPU audit and automatically selected Development case figures.

This does not import the evaluator, change methods, select a head, or tune gates.
Geometry is checked before any ground truth or scored records are opened.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(2)
M1 = "M1_base_argmax"
M4 = "M4_base_flip_mean"
M5 = "M5_base_center_crop_mean"
M6 = "M6_base_peak_crop_mean"
COLORS = {M1: (255, 65, 75), M5: (15, 230, 240), M6: (255, 225, 30)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def keyed(records):
    result = {r["sample_id"]: r for r in records}
    assert len(result) == len(records), "Duplicate sample IDs"
    return result


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def array(path):
    value = np.load(path, allow_pickle=False).squeeze()
    assert value.shape == (14, 14) and np.isfinite(value).all(), str(path)
    return value.astype(np.float32, copy=False)


def resize_minmax(raw, size):
    value = cv2.resize(np.asarray(raw, np.float32), tuple(size), interpolation=cv2.INTER_LINEAR)
    return (value - value.min()) / (value.max() - value.min() + 1e-10)


def mass(value):
    value = np.asarray(value, np.float64)
    return value / value.sum() if value.sum() > 0 else np.zeros_like(value)


def peak(value):
    index = int(value.argmax())
    return [float(index % value.shape[1]) + .5, float(index // value.shape[1]) + .5]


def expected_box(w, h, center=None):
    cw, ch = max(1, round(w / 2)), max(1, round(h / 2))
    if center is None:
        x, y = (w - cw) // 2, (h - ch) // 2
    else:
        x = max(0, min(w - cw, round(center[0] - cw / 2)))
        y = max(0, min(h - ch, round(center[1] - ch / 2)))
    return [x, y, x + cw, y + ch]


def released_head_consistency(heads):
    """Check saved-head algebra, without independently rerunning CLIP/DINO."""
    raw, ree, rem = [array(heads[name]) for name in ("raw", "ree", "rem")]
    valid = raw != 0
    inferred_c = np.divide(ree.astype(np.float64), raw, out=np.zeros_like(raw, dtype=np.float64), where=valid)
    assert (inferred_c[valid] >= -1e-6).all() and (inferred_c[valid] <= 1 + 1e-6).all()
    local_mean = np.empty((14, 14), np.float64)
    for y in range(14):
        for x in range(14):
            local_mean[y, x] = raw[max(0, y - 1):y + 2, max(0, x - 1):x + 2].astype(np.float64).mean()
    expected_rem = inferred_c * local_mean
    assert np.allclose(expected_rem[valid], rem[valid], atol=2e-5, rtol=2e-5)
    return float(np.abs(expected_rem[valid] - rem[valid]).max()), int((~valid).sum())


def geometry_audit(root, out, additional_path=None):
    # Deliberately no manifest/GT or evaluator reads in this function.
    observation_path = root / "data/observations.jsonl"
    base_path = root / "inference/base_predictions.jsonl"
    add_path = additional_path or root / "additional_complete/predictions.jsonl"
    seal_path = root / "analysis_head/sealed_head.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["selected_head"] == "ree" and not seal["evaluation_used"]
    assert seal["base_predictions_sha256"] == digest(base_path)
    obs, base, additional = map(keyed, map(rows, (observation_path, base_path, add_path)))
    assert obs.keys() == base.keys() == additional.keys(), "Incomplete or mismatched identities"
    assert len(obs) == 280
    allowed = {"sample_id", "image_path", "source_id", "image_sha256", "action", "query_action"}
    assert all(set(r) <= allowed for r in obs.values()), "Unexpected inference observation fields"
    runner = Path(__file__).with_name("run_baseline.py")
    tree = ast.parse(runner.read_text(encoding="utf-8-sig"))
    observer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "observe")
    observer_source = ast.get_source_segment(runner.read_text(encoding="utf-8-sig"), observer)
    # The actual observer explicitly projects public image/action fields.
    expected_projection = '("sample_id", "image_path", "source_id", "image_sha256")'
    assert expected_projection in observer_source
    checks = {"identity_pairs": len(obs), "head_ree": 0, "retained_base_equal": 0,
              "fixed_crop_matches": 0, "peak_crop_matches_without_GT": 0,
              "inverse_flip_exact_all_heads": 0, "source_dimensions": 0,
              "four_view_released_head_algebra": 0}
    head_errors, head_zero_cells = [], 0
    for sid in sorted(obs):
        b, a, o = base[sid], additional[sid], obs[sid]
        w, h = a["width"], a["height"]
        assert a["primary_head"] == "ree"
        assert a["action"] == o["action"] == b["action"]
        assert a["image_path"] == o["image_path"] == b["image_path"]
        checks["head_ree"] += 1
        with Image.open(o["image_path"]) as image:
            assert image.size == (w, h)
        checks["source_dimensions"] += 1
        for head in ("raw", "ree", "rem"):
            assert np.array_equal(array(b["base_heads"][head]), array(a["base_heads"][head]))
            assert np.array_equal(array(a["heads"]["flip"][head]), array(a["heads"]["flip_raw"][head])[:, ::-1])
        checks["retained_base_equal"] += 1
        checks["inverse_flip_exact_all_heads"] += 1
        for view in ("base", "fixed_crop", "peak_crop", "flip_raw"):
            error, zeros = released_head_consistency(a["heads"][view])
            head_errors.append(error)
            head_zero_cells += zeros
            checks["four_view_released_head_algebra"] += 1
        assert a["flip_map_is_inverse_restored"] is True
        assert a["flip_map_path"] == a["heads"]["flip"]["ree"]
        assert a["base_map_path"] == a["base_heads"]["ree"]
        assert a["fixed_crop_map_path"] == a["heads"]["fixed_crop"]["ree"]
        assert a["peak_crop_map_path"] == a["heads"]["peak_crop"]["ree"]
        assert a["fixed_crop_bbox_xyxy"] == expected_box(w, h)
        checks["fixed_crop_matches"] += 1
        raw = array(a["base_map_path"])
        if raw.max() == raw.min():
            assert a["peak_crop_fallback"] is not None
            point = [w / 2, h / 2]
            box = expected_box(w, h)
        else:
            point = peak(cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR))
            box = expected_box(w, h, point)
        assert np.allclose(a["peak_original_xy"], point, atol=0, rtol=0)
        assert a["peak_crop_bbox_xyxy"] == box
        checks["peak_crop_matches_without_GT"] += 1
    receipt = {"status": "PASS", "checks": checks, "selected_head": "ree",
               "ground_truth_opened": False, "scored_records_opened": False,
               "inference_observation_keys": sorted(set().union(*(set(r) for r in obs.values()))),
               "runner_observe_source": observer_source,
               "runner_sha256": digest(runner), "observations_sha256": digest(observation_path),
               "base_predictions_sha256": digest(base_path), "additional_predictions_sha256": digest(add_path),
               "additional_predictions_path": str(add_path),
               "sealed_head_sha256": digest(seal_path),
               "head_algebra": {"views": len(head_errors), "max_absolute_rem_consistency_error": max(head_errors),
                                 "uninferable_zero_raw_cells": head_zero_cells,
                                 "formula": "C=ree/raw; rem=C*valid-neighbor-3x3mean(raw)",
                                 "limit": "Saved-head consistency only; C inferred, not independently recomputed from CLIP"},
               "scope": "Stored-output geometry and actual observation schema; no rerun of model internals"}
    dump(out / "geometry_audit.json", receipt)
    return base, additional


def maps_for(record):
    size = (record["width"], record["height"])
    base = mass(resize_minmax(array(record["base_map_path"]), size))
    flip = mass(resize_minmax(array(record["flip_map_path"]), size))
    result = {M1: base, M4: mass(base / 2 + flip / 2)}
    crop_receipts = {}
    for method, prefix in ((M5, "fixed_crop"), (M6, "peak_crop")):
        x0, y0, x1, y1 = record[prefix + "_bbox_xyxy"]
        local = mass(resize_minmax(array(record[prefix + "_map_path"]), (x1 - x0, y1 - y0)))
        projected = np.zeros_like(base)
        projected[y0:y1, x0:x1] = local
        assert projected[:y0].sum() == 0 and projected[y1:].sum() == 0
        assert projected[:, :x0].sum() == 0 and projected[:, x1:].sum() == 0
        result[method] = mass((base + projected) / 2)
        crop_receipts[method] = {"crop_mass": float(projected.sum()),
                                 "base_mass": float(base.sum()), "fused_mass": float(result[method].sum()),
                                 "outside_crop_zero": True}
    return result, crop_receipts


def native_metrics(prediction, gt_raw):
    prediction = resize_minmax(prediction, (224, 224))
    target = cv2.resize(gt_raw.astype(np.float64) / 255, (224, 224), interpolation=cv2.INTER_LINEAR)
    p, q = prediction / (prediction.sum() + 1e-12), target / (target.sum() + 1e-12)
    kld = float((q * np.log(q / (p + 1e-12) + 1e-12)).sum())
    sim = float(np.minimum(p, q).sum())
    # Author cal_nss divides both incoming maps by 255 again. This check follows
    # that exact order independently, including the fixation >.1 convention.
    saliency, fixation_input = prediction / 255, target / 255
    fixation = (fixation_input - fixation_input.min()) / (fixation_input.max() - fixation_input.min() + 1e-12) > .1
    nss = None if saliency.std() == 0 or not fixation.any() else float(
        (((saliency - saliency.mean()) / saliency.std()) * fixation).sum() / (fixation.astype(float) + 1e-12).sum())
    return {"KLD": kld, "SIM": sim, "NSS": nss}


def select_cases(scored):
    indexed = {(r["sample_id"], r["method"]): r for r in scored if r["phase"] == "evaluation"}
    agd = sorted({sid for sid, method in indexed if indexed[sid, method]["dataset"] == "AGD20K"})
    iit = sorted({sid for sid, method in indexed if indexed[sid, method]["dataset"] != "AGD20K"})
    chosen = []
    used = set()

    def take(ordered, count, reason):
        for sid in ordered:
            if sid not in used:
                chosen.append({"sample_id": sid, "selection_reason": reason})
                used.add(sid)
                count -= 1
                if count == 0:
                    break

    proxy = lambda sid, method: indexed[sid, method]["GT_relative_peak_proxy"]
    take(sorted(agd, key=lambda sid: (proxy(sid, M1), sid)), 2, "AGD lowest M1 GT-relative-peak proxy; ties sample_id")
    change = lambda sid: proxy(sid, M6) - proxy(sid, M5)
    take(sorted((s for s in agd if change(s) > 0), key=lambda sid: (-change(sid), sid)), 2,
         "AGD largest positive M6-minus-M5 proxy; already-selected skipped")
    take(sorted((s for s in agd if change(s) < 0), key=lambda sid: (change(sid), sid)), 2,
         "AGD most negative M6-minus-M5 proxy; already-selected skipped")
    take([s for s in iit if indexed[s, M1]["native_empty"]], 1, "IIT native union5/9 empty; sample_id order")
    take([s for s in iit if not indexed[s, M1]["native_empty"] and not indexed[s, M1]["native_union_hit"]], 1,
         "IIT nonempty M1 miss; sample_id order")
    # There may be no empty IID-native examples in the fixed evaluation split.
    if len(chosen) < 8:
        take([s for s in iit if not indexed[s, M1]["native_union_hit"]], 8 - len(chosen),
             "IIT additional M1 miss; sample_id order; earlier category unavailable")
    return chosen[:8], indexed


def raw_diagnostic(base_record):
    raw = array(base_record["base_heads"]["raw"])
    ree = array(base_record["base_heads"]["ree"])
    result = {"raw_min": float(raw.min()), "raw_max": float(raw.max()),
              "raw_all_strictly_negative": bool((raw < 0).all()),
              "raw_negative_cells": int((raw < 0).sum()), "cells": raw.size,
              "ree_min": float(ree.min()), "ree_max": float(ree.max())}
    if (raw != 0).all():
        clip = ree.astype(np.float64) / raw
        y, x = np.unravel_index(ree.argmax(), ree.shape)
        result.update({"inferred_C_equals_ree_div_raw": True,
                       "inferred_C_min": float(clip.min()), "inferred_C_max": float(clip.max()),
                       "C_at_ree_raw_grid_argmax": float(clip[y, x]),
                       "C_at_ree_raw_grid_argmax_percentile_le": float((clip <= clip[y, x]).mean()),
                       "ree_grid_argmax_xy": [int(x), int(y)]})
    return result


def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def panel(image, title, note, point=None, bbox=None, color=(255, 255, 255)):
    width, height = 350, 300
    result = Image.new("RGB", (width, height), (20, 25, 31))
    draw = ImageDraw.Draw(result)
    draw.text((9, 7), title, font=font(16), fill="white")
    view = image.copy().convert("RGB")
    ratio = min((width - 16) / view.width, 218 / view.height)
    view = view.resize((round(view.width * ratio), round(view.height * ratio)), Image.Resampling.BILINEAR)
    left, top = (width - view.width) // 2, 33 + (218 - view.height) // 2
    result.paste(view, (left, top))
    if bbox:
        coords = [left + bbox[0] * ratio, top + bbox[1] * ratio, left + bbox[2] * ratio, top + bbox[3] * ratio]
        draw.rectangle(coords, outline=color, width=2)
    if point:
        x, y = left + point[0] * ratio, top + point[1] * ratio
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="black", width=1)
        draw.line((x - 9, y, x + 9, y), fill=color, width=2)
        draw.line((x, y - 9, x, y + 9), fill=color, width=2)
    draw.text((9, 258), note[:48], font=font(12), fill=color)
    return result


def heat_overlay(image, values):
    norm = np.asarray(values, np.float64)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-12)
    colors = cv2.cvtColor(cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8(.55 * np.asarray(image) + .45 * colors))


def case_figure(sample, additional, case_records, probabilities, out, number):
    with Image.open(sample["image_path"]) as file:
        original = file.convert("RGB")
    gt = np.asarray(Image.open(sample["gt_path"]))
    is_agd = sample["dataset"] == "AGD20K"
    if is_agd:
        truth = heat_overlay(original, gt)
        gt_note = "Continuous soft GT; not a functional mask"
    else:
        pixels = np.asarray(original).copy()
        pixels[gt == 5] = .35 * pixels[gt == 5] + .65 * np.array([60, 245, 70])
        pixels[gt == 9] = .35 * pixels[gt == 9] + .65 * np.array([230, 65, 235])
        truth = Image.fromarray(pixels)
        gt_note = "Native 5=grasp green; 9=wrap magenta"
    panels = [panel(original, "Original / action: " + sample["action"], sample.get("object", "")),
              panel(truth, "Ground truth (evaluation only)", gt_note)]
    for method in (M1, M5, M6):
        record = case_records[method]
        if is_agd:
            note = "GT peak %.3f | KLD %.3f" % (record["GT_relative_peak_proxy"], record["KLD"])
        else:
            note = "native hit %s | point label %s" % (record["native_union_hit"], record["native_point_label"])
        box = additional["fixed_crop_bbox_xyxy"] if method == M5 else additional["peak_crop_bbox_xyxy"] if method == M6 else None
        panels.append(panel(heat_overlay(original, probabilities[method]), method.split("_")[0] + " / predicted heatmap",
                            note, peak(probabilities[method]), box, COLORS[method]))
    canvas = Image.new("RGB", (1750, 345), (20, 25, 31))
    draw = ImageDraw.Draw(canvas)
    draw.text((9, 5), "%02d  %s" % (number, sample["sample_id"]), fill="white", font=font(16))
    for i, tile in enumerate(panels):
        canvas.paste(tile, (i * 350, 35))
    path = out / ("%02d_%s.jpg" % (number, sample["sample_id"]))
    canvas.save(path, quality=90)
    return path, canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--additional-predictions", type=Path)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    root = args.root.resolve()
    out = args.output_dir or root / "review/cases"
    out.mkdir(parents=True, exist_ok=True)
    base, additional = geometry_audit(root, out, args.additional_predictions)
    if args.geometry_only:
        print(json.dumps({"status": "PASS", "geometry_images": len(base), "ground_truth_opened": False}))
        return
    # Truth and scored outputs enter only after the independent geometry receipt.
    samples = keyed(rows(root / "data/samples.jsonl"))
    scored_path = root / "analysis/per_image_method_metrics.jsonl"
    scored = rows(scored_path)
    selected, indexed = select_cases(scored)
    checks, figures = [], []
    for number, case in enumerate(selected, 1):
        sid = case["sample_id"]
        sample, extra = samples[sid], additional[sid]
        probabilities, projection = maps_for(extra)
        gt = np.asarray(Image.open(sample["gt_path"]))
        case_records = {method: indexed[sid, method] for method in (M1, M4, M5, M6)}
        independent = {}
        for method in (M1, M4, M5, M6):
            record = case_records[method]
            point = peak(probabilities[method])
            assert np.array_equal(point, record["point_xy_pixel_centers"]), (sid, method, point)
            x, y = int(point[0] - .5), int(point[1] - .5)
            measured = {"point_xy_pixel_centers": point}
            if sample["dataset"] == "AGD20K":
                prediction = array(extra["base_map_path"]) if method == M1 else probabilities[method]
                measured.update(native_metrics(prediction, gt))
                measured["GT_relative_peak_proxy"] = float(gt[y, x]) / float(gt.max())
                for name in ("KLD", "SIM", "NSS", "GT_relative_peak_proxy"):
                    if measured[name] is None or record[name] is None:
                        assert measured[name] is None and record[name] is None
                    else:
                        assert abs(measured[name] - record[name]) < 1e-5, (sid, method, name, measured[name], record[name])
            else:
                measured.update(native_point_label=int(gt[y, x]), native_union_hit=bool(gt[y, x] in (5, 9)))
                assert measured["native_point_label"] == record["native_point_label"]
                assert measured["native_union_hit"] == record["native_union_hit"]
            independent[method] = measured
        gt_crop_mass = {}
        if sample["dataset"] == "AGD20K":
            for prefix in ("fixed_crop", "peak_crop"):
                x0, y0, x1, y1 = extra[prefix + "_bbox_xyxy"]
                gt_crop_mass[prefix] = float(gt[y0:y1, x0:x1].sum() / gt.sum()) if gt.sum() else None
        image_path, figure = case_figure(sample, extra, case_records, probabilities, out, number)
        figures.append(figure)
        checks.append({**case, "dataset": sample["dataset"], "action": sample["action"],
                       "object": sample.get("object"), "figure": image_path.name,
                       "independent_recomputation": independent, "projection": projection,
                       "posthoc_GT_soft_mass_in_ROI_diagnostic_only": gt_crop_mass,
                       "raw_CAM": raw_diagnostic(base[sid]),
                       "fixed_crop_bbox_xyxy": extra["fixed_crop_bbox_xyxy"],
                       "peak_crop_bbox_xyxy": extra["peak_crop_bbox_xyxy"]})
    gate_records = rows(root / "analysis/per_image_gate_decisions.jsonl")
    gates = {r["sample_id"]: r for r in gate_records if r["gate"] == "G1_M1_entropy"}
    iit_diagnostics = []
    for (sid, method), record in sorted(indexed.items()):
        if method != M1 or record["dataset"] == "AGD20K":
            continue
        iit_diagnostics.append({"sample_id": sid, "native_union_hit": record["native_union_hit"],
                                "native_empty": record["native_empty"],
                                "G1_accepted": gates[sid]["accepted"],
                                "G1_confidence": gates[sid]["confidence"],
                                **raw_diagnostic(base[sid])})
    accepted_wrong = [r for r in iit_diagnostics if r["G1_accepted"] and not r["native_union_hit"]]
    report = {"status": "PASS", "geometry_receipt": "geometry_audit.json", "selected_head": "ree",
              "case_selection": "Evaluation extremes selected automatically after scoring; diagnostic only, not prevalence evidence",
              "case_count": len(checks), "cases": checks,
              "independently_checked_image_method_rows": len(checks) * 4,
              "IIT_eval_raw_diagnostics": iit_diagnostics,
              "IIT_eval_G1_accepted_misses": len(accepted_wrong),
              "IIT_eval_images": len(iit_diagnostics),
              "IIT_eval_all_negative_raw": sum(r["raw_all_strictly_negative"] for r in iit_diagnostics),
              "IIT_eval_G1_accepted": sum(r["G1_accepted"] for r in iit_diagnostics),
              "IIT_eval_G1_accepted_misses_all_negative_raw": sum(r["raw_all_strictly_negative"] for r in accepted_wrong),
              "IIT_eval_G1_accepted_misses_C_argmax_at_grid_min": sum(abs(r.get("C_at_ree_raw_grid_argmax", math.inf) - r.get("inferred_C_min", -math.inf)) < 1e-7 for r in accepted_wrong),
              "script_sha256": digest(__file__), "scored_records_sha256": digest(scored_path),
              "wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu,
              "gpu_seconds": 0, "download_bytes": 0}
    dump(out / "casebook_audit.json", report)
    montage = Image.new("RGB", (1750, 345 * len(figures)), (20, 25, 31))
    for i, figure in enumerate(figures):
        montage.paste(figure, (0, 345 * i))
    montage.save(out / "casebook_montage.jpg", quality=88)
    print(json.dumps({key: report[key] for key in ("status", "case_count", "independently_checked_image_method_rows", "IIT_eval_G1_accepted_misses", "IIT_eval_G1_accepted_misses_all_negative_raw", "wall_seconds", "cpu_seconds")}))


if __name__ == "__main__":
    main()
