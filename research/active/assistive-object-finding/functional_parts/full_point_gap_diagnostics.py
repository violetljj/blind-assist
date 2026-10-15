"""CPU-only, posthoc point-selection diagnostics on the frozen IIT val exports.

This does not execute a model or update training. Original float32 softmax points
must exactly reproduce full_evaluate_baselines.py before any comparisons count.
All alternative point selectors use predictions only; GT is used afterward for
scoring and explicitly marked oracle diagnostics. Reusing this consumed val set
does not create independent confirmation or a new generalization experiment.
"""
from __future__ import annotations

import os

# Set before any numerical-library import. Never call any torch.cuda API here.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import collections
import ctypes
import datetime
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import sys
import time
import traceback

import numpy as np


MODELS = ("segformer", "dino")
BASE = "original_softmax32"
METHODS = (BASE, "stable_log_odds64", "temperature2_softmax32",
           "stable_peak_predmask", "largest_component_edt")
CLASS_NAMES = {0: "background", 1: "contain", 2: "cut", 3: "display",
               4: "engine", 5: "grasp", 6: "hit", 7: "pound", 8: "support",
               9: "wrap-grasp"}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def indexed(rows, label):
    result = {}
    for row in rows:
        sid = str(row.get("source_id", ""))
        if not sid or sid in result:
            raise ValueError(f"Missing or duplicate source_id in {label}: {sid!r}")
        result[sid] = row
    return result


def basename(value):
    # Frozen records retain the primary Windows paths when run on the worker.
    return PureWindowsPath(str(value).replace("/", "\\")).name


def memory_info():
    """Current/peak resident bytes and available host memory; read-only OS APIs."""
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in
                ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                 "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                              wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Status)]
        kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        status = Status()
        status.dwLength = ctypes.sizeof(status)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                         counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"working_set_bytes": int(counters.WorkingSetSize),
                "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "available_host_bytes": int(status.ullAvailPhys),
                "total_host_bytes": int(status.ullTotalPhys)}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    current = None
    statm = Path("/proc/self/statm")
    if statm.exists():
        current = int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    return {"working_set_bytes": current, "peak_working_set_bytes": int(peak),
            "available_host_bytes": available}


def lower_priority():
    if os.name != "nt":
        os.nice(5)
        return {"priority": "nice+5"}
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetPriorityClass.restype = wintypes.BOOL
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel.GetPriorityClass.restype = wintypes.DWORD
    process = kernel.GetCurrentProcess()
    if not kernel.SetPriorityClass(process, 0x00004000):
        raise ctypes.WinError(ctypes.get_last_error())
    actual = int(kernel.GetPriorityClass(process))
    if actual != 0x00004000:
        raise RuntimeError(f"BelowNormal priority did not apply: {actual}")
    return {"priority": "BelowNormal", "priority_class": actual}


def softmax32(logits, temperature=1.0):
    # Match the original NumPy evaluator, including dtype and in-place divide.
    value = logits.copy() if temperature == 1.0 else logits / np.float32(temperature)
    value = value - value.max(0, keepdims=True)
    value = np.exp(value)
    value /= value.sum(0, keepdims=True)
    assert value.dtype == np.float32
    return value


def stable_log_odds64(logits, class_id):
    """z_c - logsumexp(z_others), explicitly excluding self before reduction.

    This is a strictly monotonic transform of exact T=1 class probability. It
    changes no mathematical spatial ranking. Unlike subtracting exp(z_c) from
    an all-class sum, it retains differences when the dominant probability has
    rounded to 1.0. Only native HxW arrays are widened, not the full CxHxW grid.
    """
    maximum = np.full(logits.shape[1:], -np.inf, dtype=np.float64)
    for other in range(logits.shape[0]):
        if other != class_id:
            np.maximum(maximum, logits[other], out=maximum)
    total = np.zeros(logits.shape[1:], dtype=np.float64)
    for other in range(logits.shape[0]):
        if other != class_id:
            total += np.exp(logits[other].astype(np.float64) - maximum)
    return logits[class_id].astype(np.float64) - maximum - np.log(total)


def peak_xy(score, constraint=None):
    if constraint is None:
        index = int(score.argmax())
    else:
        valid = np.flatnonzero(constraint)
        if valid.size == 0:
            return None
        index = int(valid[int(score.ravel()[valid].argmax())])
    y, x = np.unravel_index(index, score.shape)
    return [int(x), int(y)]


def largest_component_point(mask):
    from scipy import ndimage
    if not mask.any():
        return None, {"predicted_components": 0, "largest_component_pixels": 0}
    # Fixed 4-connectivity, largest-area component, first label on ties. The
    # one-pixel zero pad gives a defined image-boundary distance even for all-1.
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = int(sizes.argmax())
    component = labels == selected
    del labels
    distance = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
    point = peak_xy(distance)
    assert point is not None and component[point[1], point[0]]
    return point, {"predicted_components": int(count),
                   "largest_component_pixels": int(sizes[selected]),
                   "largest_component_inradius_pixels": float(distance.max())}


def score_point(point, gt, prediction, class_id):
    if point is None:
        return {"point_xy": None, "native_label_at_point": None,
                "predicted_class_at_point": None, "hit": False,
                "empty_prediction_miss": True}
    x, y = point
    actual = int(gt[y, x])
    return {"point_xy": point, "native_label_at_point": actual,
            "predicted_class_at_point": int(prediction[y, x]),
            "hit": bool(actual == class_id) if actual != 255 else None,
            "empty_prediction_miss": False}


def reconstruct_native(array, shape, torch, functional):
    if array.dtype != np.float32 or array.ndim != 3 or array.shape[0] != 10:
        raise ValueError(f"Expected frozen float32 10xHxW logits, got {array.dtype} {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite frozen logits")
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    if tuple(array.shape[1:]) != tuple(shape):
        tensor = functional.interpolate(tensor, size=shape, mode="bilinear", align_corners=False)
    return tensor[0].numpy()


def grid_center_counts(gt, grid_shape):
    gh, gw = grid_shape
    # Nearest native label to each grid-centre location under align_corners=False.
    ys = np.clip(np.floor((np.arange(gh) + .5) * gt.shape[0] / gh).astype(int),
                 0, gt.shape[0] - 1)
    xs = np.clip(np.floor((np.arange(gw) + .5) * gt.shape[1] / gw).astype(int),
                 0, gt.shape[1] - 1)
    return np.bincount(gt[np.ix_(ys, xs)].ravel(), minlength=256)


def analyze_model(array, gt, old, include_raw, torch, functional):
    from scipy import ndimage
    logits = reconstruct_native(array, gt.shape, torch, functional)
    prediction = logits.argmax(0).astype(np.uint8)
    classes = [int(c) for c in np.unique(gt) if c not in (0, 255)]
    old_points = {int(point["class_id"]): point for point in old["points"]}
    if set(classes) != set(old_points):
        raise AssertionError("Old per_image query classes differ from native GT")
    original = softmax32(logits)
    result = {}
    for c in classes:
        score = original[c]
        point = score_point(peak_xy(score), gt, prediction, c)
        reference = old_points[c]
        for key in ("point_xy", "hit", "native_label_at_point"):
            if point[key] != reference[key]:
                raise AssertionError(f"Original evaluator mismatch class {c} field {key}: "
                                     f"{point[key]!r} != {reference[key]!r}")
        maximum = float(score.max())
        result[c] = {"methods": {BASE: point},
                     "original_peak_probability_float32": maximum,
                     "original_peak_equal_one": maximum == 1.0,
                     "original_maximum_tie_pixels": int(np.count_nonzero(score == maximum))}
    del original
    softened = softmax32(logits, 2.0)
    for c in classes:
        result[c]["methods"]["temperature2_softmax32"] = score_point(
            peak_xy(softened[c]), gt, prediction, c)
    del softened
    center_counts = grid_center_counts(gt, array.shape[1:])
    valid = gt != 255
    cm = np.bincount((gt[valid].astype(np.int64) * 10 + prediction[valid]).ravel(),
                     minlength=100).reshape(10, 10)
    if not np.array_equal(cm, np.asarray(old["confusion"])):
        raise AssertionError("Reconstructed native confusion differs from old per_image")
    for c in classes:
        values = result[c]
        target = gt == c
        predicted_class = prediction == c
        stable = stable_log_odds64(logits, c)
        stable_point = peak_xy(stable)
        values["methods"]["stable_log_odds64"] = score_point(stable_point, gt, prediction, c)
        values["methods"]["stable_peak_predmask"] = score_point(
            peak_xy(stable, predicted_class), gt, prediction, c)
        interior, components = largest_component_point(predicted_class)
        values["methods"]["largest_component_edt"] = score_point(interior, gt, prediction, c)
        if include_raw:
            values["methods"]["raw_logit_diagnostic"] = score_point(
                peak_xy(logits[c]), gt, prediction, c)
        inside = float(stable[target].max())
        outside = float(stable[~target].max()) if (~target).any() else None
        values["GT_oracle_diagnostics"] = {
            "stable_max_inside": inside, "stable_max_outside": outside,
            "stable_inside_minus_outside": inside - outside if outside is not None else None,
            "output_grid_centres_on_target": int(center_counts[c]),
            "output_grid_total_centres": int(array.shape[1] * array.shape[2]),
            "grid_coverage_scope": "resolution proxy only; not an upper bound on sub-grid localization"}
        point = values["methods"][BASE]["point_xy"]
        if values["methods"][BASE]["hit"] is False:
            nearest = ndimage.distance_transform_edt(~target)
            distance = float(nearest[point[1], point[0]])
            del nearest
        else:
            distance = 0.0 if values["methods"][BASE]["hit"] else None
        tp = int(cm[c, c])
        fn = int(cm[c].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        values.update({"class_TP_pixels": tp, "class_FN_pixels": fn, "class_FP_pixels": fp,
                       "class_IoU": tp / (tp + fn + fp),
                       "predicted_class_pixels": int(predicted_class.sum()),
                       "original_error_nearest_GT_pixels": distance,
                       "original_error_nearest_GT_diagonal_fraction":
                           distance / math.hypot(*gt.shape) if distance is not None else None,
                       **components})
        del stable
    return result, cm


def new_bucket(methods):
    return {"queries": 0, "models": {model: {
        "methods": {method: collections.Counter() for method in methods},
        "conditions": collections.Counter(), "diagnostics": collections.Counter()}
        for model in MODELS}, "dino_vs_segformer": {
            method: collections.Counter() for method in methods}}


def update_bucket(bucket, query, methods):
    bucket["queries"] += 1
    for model in MODELS:
        row = query["models"][model]
        target = bucket["models"][model]
        original = row["methods"][BASE]
        conditions = target["conditions"]
        conditions["original_wrong_background"] += original["native_label_at_point"] == 0
        conditions["original_wrong_other_foreground"] += original["native_label_at_point"] not in (
            None, 0, 255, query["class_id"])
        conditions["original_miss"] += original["hit"] is False
        conditions["original_miss_but_class_IoU_at_least_0.5"] += (
            original["hit"] is False and row["class_IoU"] >= .5)
        conditions["original_miss_but_class_predicted_somewhere"] += (
            original["hit"] is False and row["predicted_class_pixels"] > 0)
        conditions["no_predicted_class_pixels"] += row["predicted_class_pixels"] == 0
        conditions["original_point_predicted_as_other_class"] += (
            original["predicted_class_at_point"] != query["class_id"])
        conditions["original_peak_equal_one"] += row["original_peak_equal_one"]
        conditions["original_peak_tied"] += row["original_maximum_tie_pixels"] > 1
        conditions["peak_equal_one_and_tied"] += (
            row["original_peak_equal_one"] and row["original_maximum_tie_pixels"] > 1)
        conditions["original_miss_peak_equal_one_and_tied"] += (
            original["hit"] is False and row["original_peak_equal_one"] and
            row["original_maximum_tie_pixels"] > 1)
        conditions["zero_GT_output_grid_centres"] += (
            row["GT_oracle_diagnostics"]["output_grid_centres_on_target"] == 0)
        distance = row["original_error_nearest_GT_pixels"]
        if original["hit"] is False and distance is not None:
            target["diagnostics"]["miss_distance_sum_pixels"] += distance
            target["diagnostics"]["miss_distance_count"] += 1
            for cutoff in (2, 4, 8, 16, 32, 64):
                conditions[f"original_miss_distance_at_most_{cutoff}px"] += distance <= cutoff
        target["diagnostics"]["class_IoU_sum"] += row["class_IoU"]
        target["diagnostics"]["original_peak_tie_pixels_sum"] += row["original_maximum_tie_pixels"]
        for method in methods:
            item = row["methods"][method]
            counts = target["methods"][method]
            counts["hits"] += item["hit"] is True
            counts["misses"] += item["hit"] is False
            counts["not_evaluable_ignore"] += item["hit"] is None
            counts["empty_prediction_misses"] += item["empty_prediction_miss"]
            counts["changed_xy_vs_original"] += item["point_xy"] != original["point_xy"]
            counts["rescued_vs_original"] += original["hit"] is False and item["hit"] is True
            counts["harmed_vs_original"] += original["hit"] is True and item["hit"] is False
            counts["wrong_background"] += item["hit"] is False and item["native_label_at_point"] == 0
            counts["wrong_other_foreground"] += item["hit"] is False and item["native_label_at_point"] not in (
                None, 0, 255, query["class_id"])
    for method in methods:
        seg = query["models"]["segformer"]["methods"][method]["hit"]
        dino = query["models"]["dino"]["methods"][method]["hit"]
        counts = bucket["dino_vs_segformer"][method]
        counts["both_hit"] += seg is True and dino is True
        counts["both_miss"] += seg is False and dino is False
        counts["dino_only_hit"] += seg is False and dino is True
        counts["segformer_only_hit"] += seg is True and dino is False
        counts["ignore_in_either"] += seg is None or dino is None


def finish_bucket(bucket):
    for model in MODELS:
        for counts in bucket["models"][model]["methods"].values():
            denominator = counts["hits"] + counts["misses"]
            counts["evaluable_queries"] = denominator
            counts["hit_rate"] = counts["hits"] / denominator if denominator else None
            counts["net_hits_vs_original"] = counts["rescued_vs_original"] - counts["harmed_vs_original"]
        diagnostics = bucket["models"][model]["diagnostics"]
        diagnostics["mean_positive_query_class_IoU"] = diagnostics["class_IoU_sum"] / bucket["queries"]
        count = diagnostics["miss_distance_count"]
        diagnostics["mean_original_miss_distance_pixels"] = (
            diagnostics["miss_distance_sum_pixels"] / count if count else None)
    for counts in bucket["dino_vs_segformer"].values():
        counts["net_dino_hits"] = counts["dino_only_hit"] - counts["segformer_only_hit"]
    return bucket


def partition_keys(query):
    small = "small" if query["small_region"] else "not_small"
    cls = str(query["class_id"])
    # Existing bootstrap_group is an exact-RGB-SHA identity, so it is retained
    # per query rather than allocating thousands of redundant summary buckets.
    # Source strata use the original captured/ImageNet filename heuristic;
    # neither field establishes a physical capture/scene/instance identity.
    source = query["image_source"]
    objects = query["annotated_object_count"]
    object_kind = "unknown" if objects is None else "multiple" if objects > 1 else str(objects)
    yield "overall"
    yield "class/" + cls
    yield "small/" + small
    yield "source/" + source
    yield "class_x_small/" + cls + "/" + small
    yield "source_x_small/" + source + "/" + small
    yield "class_x_source_x_small/" + cls + "/" + source + "/" + small
    yield "annotated_object_count/" + object_kind
    if query["models"]["dino"]["class_IoU"] > query["models"]["segformer"]["class_IoU"]:
        yield "query_class_IoU_comparison/dino_higher"
    else:
        yield "query_class_IoU_comparison/dino_not_higher"


def run(args):
    started = time.monotonic()
    cpu_started = time.process_time()
    root = args.root.resolve(strict=True)
    out = args.out.resolve()
    if root not in out.parents:
        raise ValueError("Output must be an exclusive new directory inside --root")
    out.mkdir(parents=True, exist_ok=False)
    base = {"started_utc": utc_now(), "pid": os.getpid(), "root": str(root),
            "out": str(out), "max_seconds": args.max_seconds, "limit": args.limit,
            "backend": "FROZEN_PROTOCOL_CPU_ONLY", "cuda_visible_devices": "",
            "numerical_threads": 1, "model_execution": False,
            "scope": "posthoc diagnostics on consumed native semantic-union val; not independent confirmation"}
    completed = 0
    queries = 0
    peak_working_set = 0
    minimum_free = None
    try:
        base.update(lower_priority())
        inputs = {"script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)}}
        metadata_path = root / "metadata/iit_training_manifest.jsonl"
        if not metadata_path.exists():
            metadata_path = root / "data/iit/records.jsonl"
        inputs["GT_metadata"] = {"path": str(metadata_path), "sha256": sha256(metadata_path)}
        metadata = indexed(read_rows(metadata_path), "GT metadata")
        predictions, previous, previous_summaries = {}, {}, {}
        for model in MODELS:
            paths = {"predictions": root / "exports" / model / "predictions.jsonl",
                     "per_image": root / "analysis" / model / "per_image.jsonl",
                     "summary": root / "analysis" / model / "summary.json"}
            inputs[model] = {key: {"path": str(path), "sha256": sha256(path)}
                             for key, path in paths.items()}
            predictions[model] = indexed(read_rows(paths["predictions"]), model + " predictions")
            previous[model] = indexed(read_rows(paths["per_image"]), model + " per_image")
            previous_summaries[model] = json.loads(paths["summary"].read_text(encoding="utf-8-sig"))
            if set(predictions[model]) != set(previous[model]):
                raise AssertionError(f"Prediction/per_image source IDs differ: {model}")
        ids = sorted(predictions["segformer"])
        if set(ids) != set(predictions["dino"]):
            raise AssertionError("Model source sets differ")
        if len(ids) != args.expected_images:
            raise AssertionError(f"Expected {args.expected_images} frozen val rows, got {len(ids)}")
        for model in MODELS:
            if previous_summaries[model]["overall"]["images"] != len(ids):
                raise AssertionError("Summary image denominator differs from predictions")
        for sid in ids:
            for model in MODELS:
                row = predictions[model][sid]
                if row.get("split") not in ("val", "validation"):
                    raise ValueError("Only existing validation exports may be diagnosed")
                if row.get("native_class_ids") != list(range(10)):
                    raise ValueError("Unexpected native class mapping")
                if row.get("supervision_scope") != "full_image":
                    raise ValueError("Only full-image supervision is comparable here")
            if sid not in metadata or metadata[sid].get("split") not in ("val", "validation"):
                raise AssertionError(f"Missing native validation metadata: {sid}")
        json_write(out / "inputs.json", {**base, "inputs": inputs, "images_in_frozen_set": len(ids)})
        memory_before_torch = memory_info()
        if (memory_before_torch["available_host_bytes"] is not None and
                memory_before_torch["available_host_bytes"] < args.min_free_mib * 1024**2):
            raise RuntimeError("Insufficient host headroom before CPU-only Torch interpolation")
        # Lazy import only in a real worker run. --self-test never imports Torch.
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        base["torch_version"] = torch.__version__
        base["numpy_version"] = np.__version__
        base["memory_before_torch"] = memory_before_torch
        methods = METHODS + (("raw_logit_diagnostic",) if args.include_raw_logit else ())
        buckets = {}
        grids = {model: collections.Counter() for model in MODELS}
        total_confusion = {model: np.zeros((10, 10), dtype=np.int64) for model in MODELS}
        image_seconds = []
        stop_reason = None
        target_images = min(len(ids), args.limit) if args.limit is not None else len(ids)
        with (out / "per_query.jsonl").open("x", encoding="utf-8") as output:
            for sid in ids[:target_images]:
                memory = memory_info()
                available = memory["available_host_bytes"]
                peak_working_set = max(peak_working_set, memory["peak_working_set_bytes"])
                if available is not None:
                    minimum_free = available if minimum_free is None else min(minimum_free, available)
                if time.monotonic() - started >= args.max_seconds:
                    stop_reason = "MAX_SECONDS"
                    break
                if available is not None and available < args.min_free_mib * 1024**2:
                    stop_reason = "HOST_HEADROOM"
                    break
                image_started = time.monotonic()
                meta = metadata[sid]
                mask_name = basename(meta.get("mask") or sid + ".png")
                if mask_name != sid + ".png":
                    raise AssertionError("Unexpected mask basename; refuse path guessing")
                gt_path = root / "data/iit/masks" / mask_name
                if meta.get("mask_sha256") and sha256(gt_path) != meta["mask_sha256"]:
                    raise AssertionError(f"Native mask SHA changed: {sid}")
                with Image.open(gt_path) as image:
                    gt = np.asarray(image).copy()
                if gt.ndim != 2 or not np.isin(gt, list(range(10)) + [255]).all():
                    raise ValueError(f"Invalid native GT: {sid}")
                if not (gt != 255).any():
                    raise ValueError("No valid GT pixels")
                model_values, logit_receipts = {}, {}
                for model in MODELS:
                    row = predictions[model][sid]
                    if basename(row["gt_path"]) != mask_name:
                        raise AssertionError("Model GT and native metadata disagree")
                    logit_path = root / "exports" / model / "logits" / basename(row["prediction_path"])
                    array = np.load(logit_path, allow_pickle=False)
                    grid = tuple(int(v) for v in array.shape)
                    grids[model][str(grid)] += 1
                    logit_receipts[model] = {"path": str(logit_path), "sha256": sha256(logit_path),
                                             "shape": list(grid), "dtype": str(array.dtype)}
                    model_values[model], cm = analyze_model(
                        array, gt, previous[model][sid], args.include_raw_logit, torch, functional)
                    total_confusion[model] += cm
                    del array
                for c in sorted(model_values["segformer"]):
                    area = int(np.count_nonzero(gt == c))
                    valid_pixels = int(np.count_nonzero(gt != 255))
                    small = area / valid_pixels <= .01
                    for model in MODELS:
                        old_point = next(p for p in previous[model][sid]["points"] if p["class_id"] == c)
                        if (old_point["target_pixels"] != area or old_point["small_region"] != small):
                            raise AssertionError("Area/stratum differs from original evaluator")
                    grouping = previous["segformer"][sid]
                    other_grouping = previous["dino"][sid]
                    for key in ("bootstrap_group", "source_group_known", "image_source"):
                        if grouping[key] != other_grouping[key]:
                            raise AssertionError("Original model source grouping differs")
                    source_group = str(grouping["bootstrap_group"])
                    query = {"source_id": sid, "class_id": c, "class_name": CLASS_NAMES[c],
                             "target_pixels": area, "valid_pixels": valid_pixels,
                             "target_fraction": area / valid_pixels, "small_region": small,
                             "source_group": source_group,
                             "source_group_known": bool(grouping["source_group_known"]),
                             "image_source": str(grouping["image_source"]),
                             "source_group_scope": "original evaluator bootstrap identity; not verified physical source group",
                             "source_group_status": meta.get("source_group_status", "unknown"),
                             "annotated_object_count": len(meta["objects"]) if isinstance(meta.get("objects"), list) else None,
                             "object_categories": [o.get("category") for o in meta.get("objects", [])],
                             "native_shape": list(gt.shape), "logit_inputs": logit_receipts,
                             "models": {model: model_values[model][c] for model in MODELS}}
                    for key in partition_keys(query):
                        if key not in buckets:
                            buckets[key] = new_bucket(methods)
                        update_bucket(buckets[key], query, methods)
                    output.write(json.dumps(query, ensure_ascii=False, allow_nan=False) + "\n")
                    queries += 1
                output.flush()
                completed += 1
                image_seconds.append(time.monotonic() - image_started)
                memory = memory_info()
                peak_working_set = max(peak_working_set, memory["peak_working_set_bytes"])
                progress = {**base, "status": "RUNNING", "updated_utc": utc_now(),
                            "completed_images": completed, "target_images": target_images,
                            "frozen_images": len(ids), "positive_queries": queries,
                            "last_source_id": sid, "wall_seconds": time.monotonic() - started,
                            "cpu_seconds": time.process_time() - cpu_started,
                            "mean_image_seconds": float(np.mean(image_seconds)),
                            "eta_seconds_for_frozen_set": float(np.mean(image_seconds)) * (len(ids) - completed),
                            "memory": memory, "peak_working_set_bytes": peak_working_set,
                            "minimum_available_host_bytes": minimum_free}
                json_write(out / "progress.json", progress)
                if completed % 25 == 0 or completed == target_images:
                    print(json.dumps({k: progress[k] for k in (
                        "status", "completed_images", "positive_queries", "wall_seconds",
                        "mean_image_seconds", "peak_working_set_bytes")}), flush=True)
        status = "COMPLETE" if completed == len(ids) else "LIMIT_REACHED"
        if stop_reason is None and status != "COMPLETE":
            stop_reason = "EXPLICIT_IMAGE_LIMIT"
        if status == "COMPLETE":
            for model in MODELS:
                old_summary = previous_summaries[model]["overall"]
                original = buckets["overall"]["models"][model]["methods"][BASE]
                if original["hits"] != old_summary["point_hits"]:
                    raise AssertionError("Full-set original hit total differs from saved baseline")
                if original["hits"] + original["misses"] != old_summary["point_queries"]:
                    raise AssertionError("Full-set query denominator differs from saved baseline")
                if not np.array_equal(total_confusion[model], old_summary["confusion_GT_rows_prediction_columns"]):
                    raise AssertionError("Full-set confusion differs from saved baseline")
        final_memory = memory_info()
        summary = {**base, "status": status, "stop_reason": stop_reason,
                   "completed_utc": utc_now(), "completed_images": completed,
                   "frozen_images": len(ids), "positive_queries": queries,
                   "all_processed_original_points_and_confusions_exactly_reproduced": True,
                   "wall_seconds": time.monotonic() - started,
                   "cpu_seconds": time.process_time() - cpu_started,
                   "image_seconds": {"mean": float(np.mean(image_seconds)) if image_seconds else None,
                                     "median": float(np.median(image_seconds)) if image_seconds else None,
                                     "maximum": max(image_seconds) if image_seconds else None},
                   "peak_working_set_bytes": max(peak_working_set, final_memory["peak_working_set_bytes"]),
                   "minimum_available_host_bytes": minimum_free, "final_memory": final_memory,
                   "inputs": inputs, "actual_export_logit_shapes": grids,
                   "class_names": CLASS_NAMES, "methods": list(methods),
                   "source_strata_definition": "original per_image.image_source: captured/ImageNet filename heuristic; not physical source groups",
                   "method_definitions": {
                       BASE: "original bilinear float32 native resize then NumPy softmax32 spatial argmax",
                       "stable_log_odds64": "z_c-logsumexp(z_other), self explicitly excluded, float64; exact T1 softmax ranking in real arithmetic",
                       "temperature2_softmax32": "fixed T=2 float32 class softmax spatial argmax; no target calibration",
                       "stable_peak_predmask": "stable log-odds peak constrained to predicted argmax class; empty is miss",
                       "largest_component_edt": "4-connected largest predicted class component then maximum padded Euclidean interior distance; empty is miss",
                       "raw_logit_diagnostic": "optional raw class logit peak; not a calibrated strong baseline"},
                   "selection_truth_contract": "selectors never use GT; GT supplies query presence and afterward scoring/oracle margin/grid proxy",
                   "interpretation_limits": [
                       "Consumed same-source validation diagnostic; not fresh confirmation or cross-source generalization.",
                       "Semantic union masks do not establish instance-exclusive parts, physical roles, safe contact, or BLV benefit.",
                       "Stable log-odds is mathematically T1-equivalent; any difference isolates floating-point ranking/tie effects, not a new model.",
                       "GT grid-centre coverage is a resolution proxy, not a hard upper bound on localization.",
                       "No learned method, architecture change, extra model inference, target tuning, or CUDA initialization."],
                   "original_overall_metrics": {model: previous_summaries[model]["overall"] for model in MODELS},
                   "reconstructed_confusion": {model: value.tolist() for model, value in total_confusion.items()},
                   "partitions": {key: finish_bucket(value) for key, value in sorted(buckets.items())}}
        json_write(out / "summary.json", summary)
        json_write(out / "terminal.json", {key: summary[key] for key in (
            "status", "stop_reason", "started_utc", "completed_utc", "pid", "completed_images",
            "frozen_images", "positive_queries", "wall_seconds", "cpu_seconds", "peak_working_set_bytes")})
        print(json.dumps({"status": status, "images": completed, "queries": queries,
                          "wall_seconds": summary["wall_seconds"]}), flush=True)
    except BaseException as error:
        failure = {**base, "status": "FAILED", "completed_utc": utc_now(),
                   "completed_images": completed, "positive_queries": queries,
                   "wall_seconds": time.monotonic() - started,
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()}
        json_write(out / "terminal.json", failure)
        raise


def self_test():
    # Synthetic arrays only: no Torch import, model, GPU, or dataset access.
    logits = np.array([[[0., 0.]], [[20., 30.]], [[0., 0.]]], dtype=np.float32)
    probability = softmax32(logits)
    assert np.all(probability[1] == 1.)
    assert peak_xy(probability[1]) == [0, 0]
    stable = stable_log_odds64(logits, 1)
    assert stable[0, 1] > stable[0, 0]
    assert peak_xy(stable) == [1, 0]
    # Monotonic ranking agrees away from precision saturation.
    moderate = np.array([[[1., 2., -1.]], [[2., 1., 1.]], [[0., 0., .5]]], dtype=np.float32)
    for c in range(3):
        assert peak_xy(softmax32(moderate)[c]) == peak_xy(stable_log_odds64(moderate, c))
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:6, 2:7] = True
    mask[0, 8] = True
    point, info = largest_component_point(mask)
    assert point == [4, 3] and mask[point[1], point[0]]
    assert info["predicted_components"] == 2 and info["largest_component_pixels"] == 25
    empty, info = largest_component_point(np.zeros((2, 2), dtype=bool))
    assert empty is None and info["predicted_components"] == 0
    assert peak_xy(np.ones((2, 2)), np.zeros((2, 2), dtype=bool)) is None
    gt = np.array([[0, 1]], dtype=np.uint8)
    prediction = np.array([[1, 1]], dtype=np.uint8)
    original = score_point(peak_xy(probability[1]), gt, prediction, 1)
    assert original["point_xy"] == [0, 0] and original["hit"] is False
    assert score_point(None, gt, prediction, 1)["hit"] is False
    assert "torch" not in sys.modules
    print("PASS: float32 saturation/ties, stable excluded-self ranking, moderate exact-order "
          "agreement, component interior, empty miss, no GT-conditioned original point; Torch not imported")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-images", type=int, default=1757)
    parser.add_argument("--min-free-mib", type=int, default=2048)
    parser.add_argument("--include-raw-logit", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None or args.out is None:
        parser.error("--root and --out are required outside --self-test")
    if args.max_seconds <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--max-seconds and --limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
