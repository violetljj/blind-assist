"""Build observation-only field evidence; no reference text or quality labels.

Inputs are original image receipts, raw pixels, OCR observations and model
predictions. All lexical searches use the predicted field text. A matched OCR
line is evidence for a string on the image, never proof of object ownership.
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
from PIL import Image, ImageOps


FIELDS = ("product", "brand", "variety")
LOCAL_MATCH_MIN = 0.8  # Fixed generic edit similarity, not fitted to outcomes.
FIELD_KEYS = (
    "field_exact_support", "field_fuzzy_support", "field_conf_support",
    "field_weakest_support", "field_text_height", "field_spatial_compactness",
    "field_box_area", "field_local_sharpness", "field_local_contrast",
    "field_boundary_contact",
)
BASE_KEYS = ("is_product", "is_brand", "is_variety", "output_empty", "output_length")
PIXEL_KEYS = tuple(f"pixel_{i:02d}" for i in range(16))
OCR_KEYS = ("ocr_conf_mean", "ocr_conf_median", "ocr_conf_min", "ocr_box_count", "ocr_char_count")
CONF_KEYS = ("vlm_field_mean_logprob", "vlm_sequence_mean_logprob")
GROUPS = {"baseline": list(BASE_KEYS), "global_pixel": list(PIXEL_KEYS),
          "global_ocr": list(OCR_KEYS), "vlm_conf": list(CONF_KEYS),
          "field_evidence": list(FIELD_KEYS)}


def load_rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Expected an array or JSONL: {path}")
    return data


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def tokens(text):
    """Unicode case/compatibility folding, punctuation splitting; no aliases."""
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", str(text)).casefold(), flags=re.UNICODE)


def index_rows(rows, name):
    result = {}
    for row in rows:
        key = row.get("image")
        if not isinstance(key, str) or Path(key).name != key:
            raise ValueError(f"{name}: image must be a basename")
        if key in result:
            raise ValueError(f"{name}: duplicate image {key}")
        result[key] = row
    return result


def adapt_prediction(row, field):
    value = row.get("fields", {}).get(field)
    if isinstance(value, str):
        return value, {}
    if isinstance(value, dict):
        return str(value.get("text") or ""), value
    return "", {}


def mean_logprob(source):
    for key in ("mean_token_logprob", "mean_logprob", "sequence_mean_logprob", "token_logprob_mean", "avg_logprob"):
        value = finite(source.get(key))
        if value is not None:
            return value
    values = source.get("token_logprobs")
    if isinstance(values, list):
        values = [finite(v.get("logprob") if isinstance(v, dict) else v) for v in values]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    return None


def adapt_ocr(row, width, height):
    """Only native-image coordinates are accepted; no inferred rescaling."""
    if row is None:
        return [], "missing_ocr", []
    dimensions_ok = row.get("width") == width and row.get("height") == height
    if row.get("status") == "no_text" and not row.get("lines") and row.get("width") is None and row.get("height") is None:
        # The native dimensions still come from original pixels/coordinate
        # sidecar; no line geometry exists to reinterpret in this case.
        dimensions_ok = True
    warnings = [] if dimensions_ok else ["ocr_dimensions_mismatch_geometry_disabled"]
    lines = []
    for item in row.get("lines", []):
        text = str(item.get("text") or "")
        if not text:
            continue
        confidence = finite(item.get("confidence"))
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("OCR confidence outside [0,1]")
        polygon = item.get("box", item.get("polygon"))
        native = None
        if dimensions_ok and polygon is not None:
            p = np.asarray(polygon, dtype=float)
            if p.ndim == 2 and p.shape[0] >= 3 and p.shape[1] == 2 and np.isfinite(p).all():
                if p[:, 0].min() >= -1 and p[:, 1].min() >= -1 and p[:, 0].max() <= width + 1 and p[:, 1].max() <= height + 1:
                    native = np.stack([np.clip(p[:, 0], 0, width), np.clip(p[:, 1], 0, height)], axis=-1)
                else:
                    warnings.append("outside_native_coordinates_geometry_disabled")
            else:
                warnings.append("invalid_polygon_geometry_disabled")
        lines.append({"text": text, "tokens": tokens(text), "confidence": confidence,
                      "polygon": native, "accepted_at_default": item.get("accepted_at_default")})
    return lines, str(row.get("status", "provided")), sorted(set(warnings))


def whole_ocr(lines, status):
    conf = [line["confidence"] for line in lines if line["confidence"] is not None]
    result = dict.fromkeys(OCR_KEYS)
    if conf:
        result.update(ocr_conf_mean=float(np.mean(conf)), ocr_conf_median=float(np.median(conf)), ocr_conf_min=min(conf))
    if status in ("ok", "no_text", "provided"):
        result.update(ocr_box_count=len(lines), ocr_char_count=sum(len(line["text"]) for line in lines))
    return result


def union_rectangle_area(rectangles):
    """Exact union area of axis-aligned native bounding rectangles."""
    xs = sorted({x for r in rectangles for x in (r[0], r[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left)
        covered = 0.0
        end = -float("inf")
        for bottom, top in intervals:
            covered += max(0.0, top - max(bottom, end))
            end = max(end, top)
        area += (right - left) * covered
    return area


def local_geometry(gray, lines, selected):
    result = dict.fromkeys(FIELD_KEYS[4:])
    height, width = gray.shape
    rectangles = []
    crops = []
    boundary = []
    for idx in selected:
        polygon = lines[idx]["polygon"]
        if polygon is None:
            continue
        x0, y0 = np.min(polygon, axis=0)
        x1, y1 = np.max(polygon, axis=0)
        if x1 <= x0 or y1 <= y0:
            continue
        rectangles.append((float(x0), float(y0), float(x1), float(y1)))
        # No resizing: local clarity is native-pixel luminance evidence.
        crop = gray[int(math.floor(y0)):int(math.ceil(y1)), int(math.floor(x0)):int(math.ceil(x1))]
        if crop.size:
            crops.append(crop)
        boundary.append(float(x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1))
    if rectangles:
        area = union_rectangle_area(rectangles)
        enclosing = (max(r[2] for r in rectangles) - min(r[0] for r in rectangles)) * (max(r[3] for r in rectangles) - min(r[1] for r in rectangles))
        result.update(field_text_height=float(np.median([(r[3] - r[1]) / height for r in rectangles])),
                      field_spatial_compactness=area / enclosing,
                      field_box_area=area / (width * height), field_boundary_contact=float(np.mean(boundary)))
    if crops:
        sharpness = []
        for crop in crops:
            if min(crop.shape) >= 3:
                lap = crop[:-2, 1:-1] + crop[2:, 1:-1] + crop[1:-1, :-2] + crop[1:-1, 2:] - 4 * crop[1:-1, 1:-1]
                sharpness.append(float(lap.var()))
        result["field_local_sharpness"] = math.log1p(float(np.mean(sharpness))) if sharpness else None
        result["field_local_contrast"] = float(np.mean([crop.std() for crop in crops]))
    return result, rectangles


def field_evidence(prediction, lines, gray, ocr_status):
    pred = tokens(prediction)
    values = dict.fromkeys(FIELD_KEYS)
    evidence = {"prediction_tokens": pred, "matches": [], "matched_boxes": [],
                "association_claim": "image-level lexical support only; object ownership unknown"}
    if not pred or ocr_status not in ("ok", "no_text", "provided"):
        evidence["state"] = "empty_prediction" if not pred else "unavailable_ocr"
        return values, evidence
    candidates = [(word, i) for i, line in enumerate(lines) for word in line["tokens"]]
    exact, fuzzy, weighted, selected = [], [], [], set()
    for word in pred:
        matches = [(SequenceMatcher(None, word, other, autojunk=False).ratio(), i, other) for other, i in candidates]
        if matches:
            similarity, idx, other = max(matches, key=lambda m: (m[0], lines[m[1]]["confidence"] or 0.0, -m[1]))
            confidence = lines[idx]["confidence"]
            exact.append(float(word == other))
            fuzzy.append(similarity)
            weighted.append(similarity * confidence if confidence is not None else None)
            local_match = similarity >= LOCAL_MATCH_MIN
            if local_match:
                selected.add(idx)
            evidence["matches"].append({"prediction_token": word, "ocr_token": other, "line_index": idx,
                                        "similarity": similarity, "confidence": confidence, "local_match": local_match})
        else:
            exact.append(0.0)
            fuzzy.append(0.0)
            weighted.append(0.0)
    values.update(field_exact_support=float(np.mean(exact)), field_fuzzy_support=float(np.mean(fuzzy)),
                  field_conf_support=float(np.mean(weighted)) if all(v is not None for v in weighted) else None,
                  field_weakest_support=min(fuzzy))
    geometry, rectangles = local_geometry(gray, lines, sorted(selected))
    values.update(geometry)
    evidence["state"] = "observed_text" if lines else "no_ocr_text"
    evidence["matched_boxes"] = [{"line_index": i, "text": lines[i]["text"],
                                   "confidence": lines[i]["confidence"],
                                   "box": lines[i]["polygon"].tolist() if lines[i]["polygon"] is not None else None}
                                  for i in sorted(selected)]
    evidence["native_bounding_rectangles"] = rectangles
    return values, evidence


def token_f1(left, right):
    from collections import Counter
    a, b = Counter(tokens(left)), Counter(tokens(right))
    if not a or not b:
        return None  # Empty agreement is not evidence of a correct reading.
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values()))


def run(args):
    started = time.perf_counter()
    receipts = load_rows(args.receipts)
    receipt_index = index_rows(receipts, "image receipts")
    ocr = index_rows(load_rows(args.ocr), "OCR")
    predictions = index_rows(load_rows(args.predictions), "predictions")
    second = index_rows(load_rows(args.second_predictions), "second predictions") if args.second_predictions else None
    contract = None
    if args.coordinate_contract:
        contract_data = json.loads(args.coordinate_contract.read_text(encoding="utf-8-sig"))
        contract = index_rows(contract_data["images"], "OCR coordinate contract")
    extra = (set(ocr) | set(predictions) | (set(second) if second is not None else set())) - set(receipt_index)
    if extra:
        raise ValueError(f"Unexpected image identifiers: {sorted(extra)}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing observations: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, warnings = [], []
    for index, source in enumerate(receipts):
        if time.perf_counter() - started > args.wall_budget_s:
            raise TimeoutError("Evidence CPU wall budget reached")
        name = source["image"]
        image_path = args.images / name
        actual_sha = sha(image_path)
        if source.get("sha256") and source["sha256"] != actual_sha:
            raise ValueError(f"Original image checksum changed: {name}")
        if ocr.get(name, {}).get("source_sha256") not in (None, actual_sha):
            raise ValueError(f"OCR source checksum mismatch: {name}")
        with Image.open(image_path) as im:
            stored_width, stored_height = im.size
            exif_orientation = im.getexif().get(274, 1)
            # RapidOCR's path loader uses ImageOps.exif_transpose. This exact,
            # non-resampling orientation preserves native pixel correspondence.
            gray = np.asarray(ImageOps.exif_transpose(im).convert("L"), dtype=np.float32) / 255.0
        height, width = gray.shape
        if contract is not None:
            expected = contract.get(name)
            observed = {"stored_raster_width": stored_width, "stored_raster_height": stored_height,
                        "exif_orientation": exif_orientation, "observation_width": width, "observation_height": height}
            if expected is None or any(expected.get(key) != value for key, value in observed.items()):
                raise ValueError(f"OCR coordinate sidecar disagrees with actual pixels/EXIF: {name}")
        lines, status, image_warnings = adapt_ocr(ocr.get(name), width, height)
        warnings.extend({"image": name, "warning": warning} for warning in image_warnings)
        global_ocr = whole_ocr(lines, status)
        raw_pixel = source.get("features")
        if not isinstance(raw_pixel, list) or len(raw_pixel) != 16:
            raise ValueError(f"Expected unchanged old 16 pixel features: {name}")
        prediction = predictions.get(name, {})
        if prediction.get("source_sha256") not in (None, actual_sha):
            raise ValueError(f"Prediction source checksum mismatch: {name}")
        if second is not None and second.get(name, {}).get("source_sha256") not in (None, actual_sha):
            raise ValueError(f"Second prediction source checksum mismatch: {name}")
        for field in FIELDS:
            text, field_meta = adapt_prediction(prediction, field)
            feature = {"is_product": int(field == "product"), "is_brand": int(field == "brand"),
                       "is_variety": int(field == "variety"), "output_empty": int(not text.strip()),
                       "output_length": len(tokens(text))}
            feature.update(dict(zip(PIXEL_KEYS, map(finite, raw_pixel))))
            feature.update(global_ocr)
            feature.update(vlm_field_mean_logprob=mean_logprob(field_meta), vlm_sequence_mean_logprob=mean_logprob(prediction))
            local, evidence = field_evidence(text, lines, gray, status)
            feature.update(local)
            row_groups = dict(GROUPS)
            if second is not None:
                other, _ = adapt_prediction(second.get(name, {}), field)
                feature["second_pass_token_f1"] = token_f1(text, other)
                row_groups["second_agreement"] = ["second_pass_token_f1"]
                evidence["second_prediction"] = other
            rows.append({"image": name, "index": index, "model": args.model, "field": field,
                         "prediction": text, "features": feature, "groups": row_groups,
                         "prediction_status": prediction.get("status", "provided" if prediction else "missing_prediction"),
                         "ocr_status": status, "source_sha256": actual_sha, "width": width, "height": height,
                         "coordinate_space": "exif_transposed_native_pixels",
                         "stored_width": stored_width, "stored_height": stored_height, "exif_orientation": exif_orientation,
                         "evidence": evidence})
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(args.output)
    inputs = {"receipts": args.receipts, "ocr": args.ocr, "predictions": args.predictions}
    if args.second_predictions:
        inputs["second_predictions"] = args.second_predictions
    if args.coordinate_contract:
        inputs["coordinate_contract"] = args.coordinate_contract
    report = {"script_sha256": sha(Path(__file__)), "python": sys.executable,
              "input_files": {k: {"path": str(p), "sha256": sha(p)} for k, p in inputs.items()},
              "output": str(args.output), "output_sha256": sha(args.output), "images": len(receipts),
              "field_rows": len(rows), "feature_groups": GROUPS, "field_feature_count": len(FIELD_KEYS),
              "model": args.model, "wall_s": time.perf_counter() - started,
              "local_match_min": LOCAL_MATCH_MIN, "ocr_lines_policy": "all nonempty observed lines including low confidence",
              "null_policy": "missing evidence remains null; train-only imputation is downstream",
              "coordinate_policy": "EXIF-transposed full-resolution native pixels matching RapidOCR path loader; dimension mismatch disables geometry; no rescaling",
              "truth_access": "no evaluator truth or human quality annotations read", "warnings": warnings,
              "coordinate_contract_checked_images": len(receipts) if contract is not None else 0,
              "missing_predictions": sorted(set(receipt_index) - set(predictions)),
              "missing_ocr": sorted(set(receipt_index) - set(ocr))}
    args.output.with_suffix(".receipt.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("images", "field_rows", "field_feature_count", "wall_s", "missing_predictions", "missing_ocr")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--second-predictions", type=Path)
    parser.add_argument("--coordinate-contract", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-budget-s", type=float, default=240)
    run(parser.parse_args())
