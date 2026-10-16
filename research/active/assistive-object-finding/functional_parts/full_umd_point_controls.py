"""Fixed CPU-only UMD point controls on the complete, frozen IIT exports.

No model is imported or executed. CPU Torch is used only for the exact original
float32 native-size interpolation. CUDA is hidden before numerical imports.
Each original per-image confusion matrix and point must reproduce exactly.
Selectors never receive GT; GT is used afterwards for scoring and the explicitly
labelled three-candidate oracle. Atomic image chunks permit bounded recovery.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import collections
import io
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

# Reuse the already-published IIT rules, not a newly tuned UMD variant. This
# module has no eager Torch/model import. Its source SHA is part of the seal.
import full_point_gap_diagnostics as rules

MODELS = ("segformer", "dino")
METHODS = rules.METHODS
BASE = rules.BASE
POOL = ("stable_log_odds64", "temperature2_softmax32", "largest_component_edt")
SHARED = (1, 2, 5, 7, 8, 9)
UMD_TO_IIT = {0: 0, 1: 5, 2: 2, 3: 255, 4: 1, 5: 7, 6: 8, 7: 9}
ORACLE = "fixed_three_candidate_GT_oracle_EVAL_ONLY"
ALL_METHODS = (*METHODS, ORACLE)
_STOP = None


def stop_signal(number, _frame):
    global _STOP
    _STOP = "SIGNAL_" + str(number)


def read_jsonl(path):
    return rules.read_rows(Path(path))


def source_id(row):
    return str(row.get("source_id", ""))


def index_rows(rows, name):
    return rules.indexed(rows, name)


def remap_path(value, root, marker):
    """Keep valid paths; otherwise relocate the exact declared payload suffix."""
    path = Path(value)
    if path.is_file():
        return path
    normalized = str(value).replace("\\", "/")
    suffix = "/" + marker + "/"
    if suffix not in normalized:
        raise FileNotFoundError(f"Cannot relocate missing declared input: {value}")
    relocated = root / marker / normalized.rsplit(suffix, 1)[1]
    if not relocated.is_file():
        raise FileNotFoundError(relocated)
    return relocated


def read_gt(path, space):
    from scipy.io import loadmat
    if path.suffix.lower() != ".mat" or space != "umd_native_0_7":
        raise ValueError("This fixed run accepts the released UMD native MAT only")
    payload = path.read_bytes()
    raw = np.asarray(loadmat(io.BytesIO(payload), variable_names=["gt_label"])["gt_label"])
    if raw.ndim != 2 or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError("gt_label is not a finite 2-D integer array")
    if not np.isin(raw, (-1, *UMD_TO_IIT)).all():
        raise ValueError("Unexpected native label; never silently map to background")
    gt = np.full(raw.shape, 255, dtype=np.uint8)
    for old, new in UMD_TO_IIT.items():
        gt[raw == old] = new
    reasons = {"native_void_minus1": raw == -1, "unsupported_scoop3": raw == 3}
    return gt, reasons, rules.hashlib.sha256(payload).hexdigest()


def score_point(point, gt, prediction, class_id, reasons):
    result = rules.score_point(point, gt, prediction, class_id)
    result["ignore_reason"] = None
    if result["hit"] is None:
        x, y = point
        found = [name for name, mask in reasons.items() if mask[y, x]]
        if len(found) != 1:
            raise AssertionError("Every unknown point must have one native ignore reason")
        result["ignore_reason"] = found[0]
    return result


def selectors(logits, prediction, classes):
    """Prediction-only function: receives neither mask truth nor validity mask."""
    result = {c: {} for c in classes}
    original = rules.softmax32(logits)
    for c in classes:
        score = original[c]
        maximum = float(score.max())
        result[c][BASE] = rules.peak_xy(score)
        result[c]["original_peak_probability_float32"] = maximum
        result[c]["original_peak_equal_one"] = maximum == 1.0
        result[c]["original_maximum_tie_pixels"] = int(np.count_nonzero(score == maximum))
    del original
    softened = rules.softmax32(logits, 2.0)
    for c in classes:
        result[c]["temperature2_softmax32"] = rules.peak_xy(softened[c])
    del softened
    for c in classes:
        mask = prediction == c
        stable = rules.stable_log_odds64(logits, c)
        result[c]["stable_log_odds64"] = rules.peak_xy(stable)
        result[c]["stable_peak_predmask"] = rules.peak_xy(stable, mask)
        interior, components = rules.largest_component_point(mask)
        result[c]["largest_component_edt"] = interior
        result[c]["predicted_class_pixels_including_ignored_truth"] = int(mask.sum())
        result[c].update(components)
        del stable, mask
    return result


def oracle_status(points):
    # Unknown is not a miss. If no known-good point exists and an ignored point
    # is present, the candidate pool's success status remains unknown.
    values = [points[name]["hit"] for name in POOL]
    return True if any(v is True for v in values) else None if any(v is None for v in values) else False


def analyze_model(array, gt, reasons, old, torch, functional):
    logits = rules.reconstruct_native(array, gt.shape, torch, functional)
    prediction = logits.argmax(0).astype(np.uint8)
    valid = gt != 255
    valid_pixels = int(valid.sum())
    if not valid_pixels:
        raise ValueError("Unexpected entirely ignored source frame")
    cm = np.bincount((gt[valid].astype(np.int64) * 10 + prediction[valid]).ravel(),
                     minlength=100).reshape(10, 10)
    if not np.array_equal(cm, np.asarray(old["confusion"], dtype=np.int64)):
        raise AssertionError("Original native confusion matrix does not exactly reproduce")
    if old["valid_pixels"] != valid_pixels or old["ignored_pixels"] != int((~valid).sum()):
        raise AssertionError("Original valid/ignore denominators differ")
    old_points = {int(row["class_id"]): row for row in old["points"]}
    classes = tuple(int(c) for c in np.unique(gt) if c not in (0, 255))
    if set(classes) != set(old_points) or set(classes) - set(SHARED):
        raise AssertionError("Original positive semantic queries differ")
    selected = selectors(logits, prediction, classes)
    queries = {}
    for c in classes:
        selected_c = selected[c]
        points = {method: score_point(selected_c[method], gt, prediction, c, reasons)
                  for method in METHODS}
        reference = old_points[c]
        for key in ("point_xy", "native_label_at_point", "hit"):
            if reference[key] != points[BASE][key]:
                raise AssertionError(f"Original point mismatch class={c} field={key}")
        target_pixels = int(cm[c].sum())
        target_fraction = target_pixels / valid_pixels
        if (target_pixels != reference["target_pixels"] or
                target_fraction != reference["target_fraction"] or
                (target_fraction <= .01) != reference["small_region"]):
            raise AssertionError("Original target-area denominator differs")
        tp = int(cm[c, c])
        predicted_valid = int(cm[:, c].sum())
        union = target_pixels + predicted_valid - tp
        iou = tp / union
        support = ("no_valid_predicted_class" if predicted_valid == 0 else
                   "predicted_but_zero_GT_intersection" if tp == 0 else
                   "TP_exists_IoU_below_0.5" if iou < .5 else "TP_exists_IoU_at_least_0.5")
        pool_hit = oracle_status(points)
        points[ORACLE] = {"hit": pool_hit, "point_xy": None,
                          "native_label_at_point": None,
                          "predicted_class_at_point": None,
                          "empty_prediction_miss": False,
                          "ignore_reason": "candidate_pool_contains_unknown" if pool_hit is None else None,
                          "GT_used_for_oracle_only": True}
        queries[c] = {"methods": points, "class_TP_pixels": tp,
                      "class_GT_pixels": target_pixels,
                      "class_predicted_valid_pixels": predicted_valid,
                      "class_IoU": iou, "segmentation_support": support,
                      "target_pixels": target_pixels, "target_fraction": target_fraction,
                      "small_region": target_fraction <= .01,
                      "pool_distinct_nonempty_candidates": len({tuple(selected_c[m])
                          for m in POOL if selected_c[m] is not None}),
                      **{k: v for k, v in selected_c.items() if k not in METHODS}}
    # Every unknown must also reproduce the original explicit reason split.
    expected_ignored = {name: int(mask.sum()) for name, mask in reasons.items()}
    expected_points = {name: sum(q["methods"][BASE]["ignore_reason"] == name
                                 for q in queries.values()) for name in reasons}
    if expected_ignored != old["ignored_pixels_by_reason"]:
        raise AssertionError("Original ignored-pixel reason counts differ")
    if expected_points != old["ignored_point_queries_by_reason"]:
        raise AssertionError("Original ignored-point reason counts differ")
    return queries, cm


def partition_names(query):
    small = "small" if query["small_region"] else "not_small"
    c = str(query["class_id"])
    return ("overall", "class/" + c, "area/" + small, "source/" + query["source_kind"],
            "class_area/" + c + "/" + small,
            "source_area/" + query["source_kind"] + "/" + small)


def describe_counts(count):
    hits, misses, unknown, empty = (int(v) for v in count)
    total = hits + misses + unknown
    evaluable = hits + misses
    return {"positive_queries": total, "hits": hits, "misses": misses, "unknown_ignore": unknown,
            "empty_prediction_misses": empty, "evaluable_queries": evaluable,
            "hit_rate_evaluable": hits / evaluable if evaluable else None,
            "hits_over_all_positive_conservative": hits / total if total else None}


def intervals(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {"CI95_percentile": np.percentile(values, [2.5, 97.5]).tolist() if len(values) else None,
            "valid_draws": len(values)}


def rate_arrays(array):
    # shape ...,4: hit, miss, unknown, empty; empty is a miss subset.
    hit = array[..., 0]
    total = array[..., :3].sum(-1)
    evaluable = array[..., :2].sum(-1)
    def divide(a, b):
        return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
    return {"hits_over_all_positive_conservative": divide(hit, total),
            "hit_rate_evaluable": divide(hit, evaluable)}


def summarize(images, draws, seed, expected_groups):
    groups = sorted({image["bootstrap_group"] for image in images})
    if len(groups) != expected_groups:
        raise AssertionError(f"Expected {expected_groups} source groups, got {len(groups)}")
    group_index = {name: i for i, name in enumerate(groups)}
    grouped = {}
    common_with_base = {}
    common_cross_model = {}
    conditions = collections.defaultdict(collections.Counter)
    paired_counts = collections.defaultdict(collections.Counter)
    query_total = 0
    cm_total = {m: np.zeros((10, 10), dtype=np.int64) for m in MODELS}
    for image in images:
        for model in MODELS:
            cm_total[model] += np.asarray(image["models"][model]["confusion"], dtype=np.int64)
        for query in image["queries"]:
            query_total += 1
            partitions = partition_names(query)
            gi = group_index[image["bootstrap_group"]]
            for part in partitions:
                if part not in grouped:
                    grouped[part] = np.zeros((len(groups), len(MODELS), len(ALL_METHODS), 4), dtype=np.int64)
                    common_with_base[part] = np.zeros((len(groups), len(MODELS), len(ALL_METHODS), 3), dtype=np.int64)
                    common_cross_model[part] = np.zeros((len(groups), len(ALL_METHODS), 3), dtype=np.int64)
                for mi, model in enumerate(MODELS):
                    row = query["models"][model]
                    base = row["methods"][BASE]
                    for pi, method in enumerate(ALL_METHODS):
                        point = row["methods"][method]
                        hit = point["hit"]
                        grouped[part][gi, mi, pi, 0 if hit is True else 1 if hit is False else 2] += 1
                        grouped[part][gi, mi, pi, 3] += point["empty_prediction_miss"]
                        if hit is not None and base["hit"] is not None:
                            common_with_base[part][gi, mi, pi] += (hit is True, base["hit"] is True, 1)
                        counter = conditions[(part, model, method)]
                        for state, truth in (("hit", True), ("miss", False), ("unknown", None)):
                            if hit is truth:
                                counter[state + "/segmentation_support/" + row["segmentation_support"]] += 1
                        counter["changed_xy_vs_original"] += point["point_xy"] != base["point_xy"] if method != ORACLE else 0
                        counter["rescued_known_miss_vs_original"] += base["hit"] is False and hit is True
                        counter["harmed_known_hit_vs_original"] += base["hit"] is True and hit is False
                        counter["gain_conservative_vs_original"] += base["hit"] is not True and hit is True
                        counter["loss_conservative_vs_original"] += base["hit"] is True and hit is not True
                        counter["wrong_background"] += hit is False and point["native_label_at_point"] == 0
                        counter["wrong_other_foreground"] += (hit is False and point["native_label_at_point"]
                            not in (None, 0, 255, query["class_id"]))
                        if point["ignore_reason"] is not None:
                            counter["unknown_reason/" + point["ignore_reason"]] += 1
                        counter["original_peak_equal_one"] += row["original_peak_equal_one"]
                        counter["original_peak_tied"] += row["original_maximum_tie_pixels"] > 1
                for pi, method in enumerate(ALL_METHODS):
                    seg = query["models"]["segformer"]["methods"][method]["hit"]
                    dino = query["models"]["dino"]["methods"][method]["hit"]
                    counts = paired_counts[(part, method)]
                    counts["dino_hit_seg_not_hit_conservative"] += dino is True and seg is not True
                    counts["seg_hit_dino_not_hit_conservative"] += seg is True and dino is not True
                    counts["both_known_hit"] += dino is True and seg is True
                    counts["both_known_miss"] += dino is False and seg is False
                    counts["unknown_in_either"] += dino is None or seg is None
                    if dino is not None and seg is not None:
                        common_cross_model[part][gi, pi] += (dino is True, seg is True, 1)
    rng = np.random.default_rng(seed)
    # The same complete 108-source draw is used for every method/partition.
    weights = np.zeros((draws, len(groups)), dtype=np.float64)
    for b in range(draws):
        weights[b] = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
    result = {}
    for part, counts in sorted(grouped.items()):
        total = counts.sum(0)
        boot = (weights @ counts.reshape(len(groups), -1)).reshape(draws, len(MODELS), len(ALL_METHODS), 4)
        rates = rate_arrays(boot)
        common_base_counts = common_with_base[part]
        common_cross_counts = common_cross_model[part]
        common_base_total = common_base_counts.sum(0)
        common_cross_total = common_cross_counts.sum(0)
        boot_base_common = (weights @ common_base_counts.reshape(len(groups), -1)).reshape(draws, len(MODELS), len(ALL_METHODS), 3)
        boot_cross_common = (weights @ common_cross_counts.reshape(len(groups), -1)).reshape(draws, len(ALL_METHODS), 3)
        def common_known_delta(values):
            numerator = values[..., 0] - values[..., 1]
            return np.divide(numerator, values[..., 2], out=np.full(numerator.shape, np.nan),
                             where=values[..., 2] > 0)
        common_base_deltas = common_known_delta(boot_base_common)
        common_cross_deltas = common_known_delta(boot_cross_common)
        models = {}
        for mi, model in enumerate(MODELS):
            models[model] = {}
            for pi, method in enumerate(ALL_METHODS):
                models[model][method] = {**describe_counts(total[mi, pi]),
                    "conditions": dict(conditions[(part, model, method)]),
                    "bootstrap": {metric: intervals(value[:, mi, pi]) for metric, value in rates.items()},
                    "paired_delta_vs_original": {metric: intervals(value[:, mi, pi] - value[:, mi, 0])
                                                   for metric, value in rates.items()},
                    "paired_common_known_vs_original": {
                        "method_hits": int(common_base_total[mi, pi, 0]),
                        "original_hits": int(common_base_total[mi, pi, 1]),
                        "common_evaluable_queries": int(common_base_total[mi, pi, 2]),
                        "bootstrap_hit_rate_delta": intervals(common_base_deltas[:, mi, pi])}}
            # Descriptive best fixed method is marked posthoc; no output is selected from GT.
            fixed_counts = total[mi, :len(METHODS), 0]
            best = int(fixed_counts.argmax())
            pool_positions = [ALL_METHODS.index(method) for method in POOL]
            best_pool = max(pool_positions, key=lambda position: int(total[mi, position, 0]))
            models[model]["candidate_pool_headroom_descriptive"] = {
                "pool_definition": list(POOL), "best_fixed_method_posthoc_descriptive": METHODS[best],
                "best_fixed_hits": int(fixed_counts[best]), "pool_oracle_hits": int(total[mi, -1, 0]),
                "best_fixed_pool_member_posthoc_descriptive": ALL_METHODS[best_pool],
                "best_fixed_pool_member_hits": int(total[mi, best_pool, 0]),
                "oracle_minus_best_fixed_pool_member_hits": int(total[mi, -1, 0] - total[mi, best_pool, 0]),
                "note": "Oracle counts no known-good candidate as unknown if any candidate is ignored. This is not a learned selector result."}
        result[part] = {"models": models, "dino_minus_segformer": {
            method: {"paired_counts": dict(paired_counts[(part, method)]),
                     "paired_common_known": {
                         "dino_hits": int(common_cross_total[pi, 0]),
                         "segformer_hits": int(common_cross_total[pi, 1]),
                         "common_evaluable_queries": int(common_cross_total[pi, 2]),
                         "bootstrap_hit_rate_delta": intervals(common_cross_deltas[:, pi])},
                     "bootstrap_delta": {metric: intervals(value[:, 1, pi] - value[:, 0, pi])
                                         for metric, value in rates.items()}}
            for pi, method in enumerate(ALL_METHODS)}}
    return {"positive_queries": query_total, "source_groups": groups,
            "bootstrap": {"seed": seed, "draws": draws, "clusters": len(groups),
                          "unit": "author tool/scene source group; no physical instance independence assumed",
                          "same_group_draws_for_all_models_methods_and_strata": True},
            "reconstructed_confusion": {m: cm.tolist() for m, cm in cm_total.items()},
            "partitions": result}


def load_chunks(out, expected_ids):
    accepted = []
    for path in sorted((out / "chunks").glob("*.jsonl")):
        payload = path.read_bytes()
        if not payload.endswith(b"\n"):
            raise ValueError(f"Published atomic chunk is truncated: {path}")
        for line in payload.splitlines():
            row = json.loads(line)
            index = len(accepted)
            if row["index"] != index or row["source_id"] != expected_ids[index]:
                raise ValueError("Resume chunks are not an exact frozen source-order prefix")
            accepted.append(row)
    return accepted


def write_chunk(out, pending):
    if not pending:
        return
    path = out / "chunks" / f"{pending[0]['index']:07d}-{pending[-1]['index']:07d}.jsonl"
    if path.exists():
        raise FileExistsError("Never overwrite a completed image chunk")
    tmp = path.with_suffix(".jsonl." + str(time.time_ns()) + ".inprogress")
    # Atomic replacement publishes only a complete final-newline JSONL. Interrupted
    # .inprogress material is preserved and ignored, never silently truncated.
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        for row in pending:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def run(args):
    started = time.monotonic()
    cpu_start = time.process_time()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks").mkdir(exist_ok=True)
    inputs = {"records": root / "data/umd/shared_label_records.jsonl"}
    for model in MODELS:
        inputs[model + "_predictions"] = root / "transfer" / model / "export/predictions.jsonl"
        inputs[model + "_old_per_image"] = root / "transfer" / model / "evaluation/per_image.jsonl"
        inputs[model + "_old_summary"] = root / "transfer" / model / "evaluation/summary.json"
    input_shas = {name: rules.sha256(path) for name, path in inputs.items()}
    source_shas = {"this_script": rules.sha256(__file__), "IIT_selector_rules": rules.sha256(rules.__file__)}
    seal = {"input_sha256": input_shas, "source_sha256": source_shas,
            "expected_images": args.expected_images, "expected_queries": args.expected_queries,
            "expected_groups": args.expected_groups, "seed": args.seed, "draws": args.bootstrap,
            "methods": list(METHODS), "fixed_pool": list(POOL), "maximum_total_wall_seconds": args.max_seconds,
            "limit_smoke_only": args.limit, "one_cpu_thread": True}
    seal_path = out / "input_seal.json"
    if seal_path.exists():
        if json.loads(seal_path.read_text(encoding="utf-8")) != seal:
            raise ValueError("Resume requires identical code, source inputs, rules and budget")
    else:
        rules.json_write(seal_path, seal)
    lock = out / "owner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_utc": rules.utc_now(),
                                 "script": str(Path(__file__).resolve())}))
    pending = []
    completed = []
    old_wall = old_cpu = 0.
    stop_reason = None
    try:
        for previous in (out / "progress.json", out / "terminal.json"):
            if previous.exists():
                status = json.loads(previous.read_text(encoding="utf-8"))
                old_wall = max(old_wall, status.get("cumulative_wall_seconds", 0.))
                old_cpu = max(old_cpu, status.get("cumulative_cpu_seconds", 0.))
        attempts = out / "attempts"
        attempts.mkdir(exist_ok=True)
        previous_terminal = out / "terminal.json"
        if previous_terminal.exists():
            archive = attempts / (str(time.time_ns()) + "_prior_terminal.json")
            archive.write_bytes(previous_terminal.read_bytes())
        # Import only CPU tensor/interpolation functionality, never trainer,
        # exporter, models, torch.cuda APIs, checkpoints or GPU model assets.
        import torch
        import torch.nn.functional as functional
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        priority = rules.lower_priority()
        records_list = read_jsonl(inputs["records"])
        records = index_rows(records_list, "UMD records")
        ids = [source_id(row) for row in records_list]
        if len(ids) != args.expected_images:
            raise ValueError(f"Expected {args.expected_images} frames, got {len(ids)}")
        old = {m: index_rows(read_jsonl(inputs[m + "_old_per_image"]), m) for m in MODELS}
        predictions = {m: index_rows(read_jsonl(inputs[m + "_predictions"]), m) for m in MODELS}
        for model in MODELS:
            if set(ids) != set(old[model]) or set(ids) != set(predictions[model]):
                raise ValueError("Records, original evaluation and exports must cover the same complete images")
        completed = load_chunks(out, ids)
        def progress(state):
            value = {"status": state, "pid": os.getpid(), "completed_images": len(completed),
                "total_images": len(ids), "positive_queries": sum(len(x["queries"]) for x in completed),
                "cumulative_wall_seconds": old_wall + time.monotonic() - started,
                "cumulative_cpu_seconds": old_cpu + time.process_time() - cpu_start,
                "memory": rules.memory_info(), "priority": priority, "updated_utc": rules.utc_now(),
                "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
                "cuda_visibility": os.environ["CUDA_VISIBLE_DEVICES"], "GPU_model_inference": False}
            rules.json_write(out / "progress.json", value)
            return value
        progress("RUNNING")
        for index in range(len(completed), len(ids)):
            free = rules.memory_info().get("available_host_bytes")
            if free is not None and free < args.min_free_mib * 1024**2:
                stop_reason = "MINIMUM_FREE_HOST_MEMORY"
                break
            if _STOP or old_wall + time.monotonic() - started >= args.max_seconds:
                stop_reason = _STOP or "MAXIMUM_CUMULATIVE_WALL_BUDGET"
                break
            if args.limit is not None and index >= args.limit:
                stop_reason = "SMOKE_LIMIT_NOT_FULL_RESULT"
                break
            sid = ids[index]
            meta = records[sid]
            group = str(meta["source_group"])
            if meta.get("manual_annotation_status") != "AUTHOR_SELECTED_MANUAL":
                raise ValueError("Only author-selected manual frames accepted")
            pred_rows = [predictions[m][sid] for m in MODELS]
            for row in pred_rows:
                if (row.get("native_class_ids") != list(range(10)) or
                        row.get("split") != "external_transfer" or row.get("source_group") != group or
                        {int(k): int(v) for k, v in row["umd_to_iit"].items()} != UMD_TO_IIT):
                    raise ValueError("Frozen output vocabulary, source identity or mapping changed")
            gt_path = remap_path(pred_rows[0]["gt_path"], root, "data")
            if remap_path(pred_rows[1]["gt_path"], root, "data") != gt_path:
                raise ValueError("Models do not share the identical original GT path")
            gt, reasons, gt_sha = read_gt(gt_path, pred_rows[0].get("gt_label_space"))
            image = {"index": index, "source_id": sid, "bootstrap_group": group,
                     "source_kind": meta["image_source"], "gt_file_sha256": gt_sha,
                     "models": {}, "queries": []}
            by_model = {}
            for model, row in zip(MODELS, pred_rows):
                logit_path = remap_path(row["prediction_path"], root, "transfer")
                payload = logit_path.read_bytes()
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                query_values, cm = analyze_model(array, gt, reasons, old[model][sid], torch, functional)
                image["models"][model] = {"confusion": cm.tolist(),
                    "prediction_file_sha256": rules.hashlib.sha256(payload).hexdigest(),
                    "logit_shape": list(array.shape), "original_reproduction": "EXACT_ALL_CM_CELLS_AND_POINTS"}
                by_model[model] = query_values
                del array, payload
            if set(by_model[MODELS[0]]) != set(by_model[MODELS[1]]):
                raise AssertionError("Model positive queries differ")
            for c in sorted(by_model[MODELS[0]]):
                reference = by_model[MODELS[0]][c]
                image["queries"].append({"class_id": c, "small_region": reference["small_region"],
                    "source_kind": image["source_kind"],
                    "models": {m: by_model[m][c] for m in MODELS}})
            pending.append(image)
            if len(pending) >= args.chunk_images:
                write_chunk(out, pending)
                completed.extend(pending)
                pending.clear()
                progress("RUNNING")
        write_chunk(out, pending)
        completed.extend(pending)
        pending.clear()
        all_done = len(completed) == len(ids) and args.limit is None
        status = "COMPLETE" if all_done else "STOPPED"
        if all_done:
            summary = summarize(completed, args.bootstrap, args.seed, args.expected_groups)
            if summary["positive_queries"] != args.expected_queries:
                raise AssertionError("Complete positive-query denominator differs")
            for model in MODELS:
                reference = json.loads(inputs[model + "_old_summary"].read_text(encoding="utf-8"))
                cm_old = reference["shared6_foreground_mIoU_primary"]["confusion_GT_rows_prediction_columns"]
                if summary["reconstructed_confusion"][model] != cm_old:
                    raise AssertionError("Full reconstructed confusion differs from original summary")
                count = summary["partitions"]["overall"]["models"][model][BASE]
                if (count["hits"] != reference["shared6_foreground_mIoU_primary"]["point_hits"] or
                        count["unknown_ignore"] != reference["point_queries_landing_on_any_ignore"]):
                    raise AssertionError("Full original point/ignore denominator differs")
            summary.update({"status": status, "images": len(completed), "input_seal": seal,
                "selectors_use_GT": False, "GT_oracle_EVAL_ONLY": ORACLE,
                "small_region_definition": "semantic-class union <=1% of native valid-image area, not physical part/instance truth",
                "limits": ["Consumed cross-source diagnostic; no fresh confirmation or architecture improvement claim.",
                    "All ten original model channels retained; native void -1 and scoop3 remain separately ignored.",
                    "Stable T1 ranking changes floating-point computation, not mathematical probability order.",
                    "Oracle uses truth for evaluation only; no GT-selected point is a deployable output.",
                    "No correct argmax segmentation pixel is not a hard bound on lower-ranked scores.",
                    "Source clusters are author tools/scenes, not 9939 independent physical instances."]})
            rules.json_write(out / "summary.json", summary)
        final = progress(status)
        final.update({"stop_reason": stop_reason, "original_reproduction": "EXACT_FOR_EVERY_COMPLETED_IMAGE",
                      "source_sha256": source_shas, "input_sha256": input_shas})
        if all_done:
            final["summary_sha256"] = rules.sha256(out / "summary.json")
        rules.json_write(out / "terminal.json", final)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        # Preserve completed chunks and the failing attempt; root verifies owner
        # identity before any restart. Unpublished .inprogress chunks are evidence.
        failure = {"status": "FAILED", "pid": os.getpid(), "completed_images": len(completed),
                   "cumulative_wall_seconds": old_wall + time.monotonic() - started,
                   "cumulative_cpu_seconds": old_cpu + time.process_time() - cpu_start,
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(), "updated_utc": rules.utc_now(),
                   "source_sha256": source_shas, "input_sha256": input_shas}
        rules.json_write(out / "terminal.json", failure)
        raise
    finally:
        # Only the lock created by this invocation is released.
        lock.unlink()


def self_test():
    rules.self_test()
    gt = np.array([[0, 5, 255, 255]], dtype=np.uint8)
    prediction = np.array([[5, 5, 5, 5]], dtype=np.uint8)
    reasons = {"native_void_minus1": np.array([[False, False, True, False]]),
               "unsupported_scoop3": np.array([[False, False, False, True]])}
    assert score_point([2, 0], gt, prediction, 5, reasons)["ignore_reason"] == "native_void_minus1"
    assert score_point([3, 0], gt, prediction, 5, reasons)["hit"] is None
    assert score_point(None, gt, prediction, 5, reasons)["hit"] is False
    points = {m: {"hit": False} for m in POOL}
    assert oracle_status(points) is False
    points[POOL[0]]["hit"] = None
    assert oracle_status(points) is None
    points[POOL[1]]["hit"] = True
    assert oracle_status(points) is True
    # A truth-blind high-confidence point is allowed to land on ignored GT.
    logits = np.zeros((10, 1, 4), dtype=np.float32)
    logits[5] = [1, 2, 30, 3]
    chosen = selectors(logits, logits.argmax(0), (5,))[5]
    assert chosen["stable_log_odds64"] == [2, 0]
    assert score_point(chosen["stable_log_odds64"], gt, prediction, 5, reasons)["hit"] is None
    counter = describe_counts([2, 1, 1, 1])
    assert counter["hits_over_all_positive_conservative"] == .5
    assert counter["hit_rate_evaluable"] == 2 / 3
    rates = rate_arrays(np.array([[2, 1, 1, 1]], dtype=np.float64))
    assert rates["hits_over_all_positive_conservative"][0] == .5
    # Exercise the same clustered aggregation and paired/common-known code on
    # two tiny source groups, without dataset pixels or Torch.
    images = []
    for gi in range(2):
        query = {"class_id": 5, "small_region": gi == 0, "source_kind": "tool", "models": {}}
        for model in MODELS:
            methods = {}
            for method in ALL_METHODS:
                hit = not (model == "dino" and gi == 0 and method == BASE)
                methods[method] = {"hit": hit, "point_xy": [0, 0],
                    "native_label_at_point": 5 if hit else 0, "empty_prediction_miss": False,
                    "ignore_reason": None}
            query["models"][model] = {"methods": methods, "segmentation_support": "TP_exists_IoU_below_0.5",
                "original_peak_equal_one": False, "original_maximum_tie_pixels": 1}
        images.append({"bootstrap_group": "g" + str(gi), "models": {
            m: {"confusion": np.eye(10, dtype=np.int64).tolist()} for m in MODELS}, "queries": [query]})
    synthetic = summarize(images, 20, 20261015, 2)
    assert synthetic["positive_queries"] == 2
    overall = synthetic["partitions"]["overall"]
    dino = overall["models"]["dino"]
    assert dino[BASE]["hits"] == 1 and dino["stable_log_odds64"]["hits"] == 2
    assert dino["stable_log_odds64"]["paired_common_known_vs_original"]["common_evaluable_queries"] == 2
    assert overall["dino_minus_segformer"][BASE]["paired_common_known"]["segformer_hits"] == 2
    assert dino["candidate_pool_headroom_descriptive"]["oracle_minus_best_fixed_pool_member_hits"] == 0
    assert "torch" not in sys.modules
    print(json.dumps({"status": "PASS", "synthetic_only": True,
        "contracts": ["published IIT rules", "separate void/scoop unknown", "empty miss",
                      "unknown-preserving candidate oracle", "no GT-conditioned selector",
                      "all-positive versus evaluable denominator", "cluster bootstrap and common-known pairing"],
        "torch_imported": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-seconds", type=float, default=10800.)
    parser.add_argument("--min-free-mib", type=int, default=1536)
    parser.add_argument("--chunk-images", type=int, default=10)
    parser.add_argument("--expected-images", type=int, default=9939)
    parser.add_argument("--expected-queries", type=int, default=19370)
    parser.add_argument("--expected-groups", type=int, default=108)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--limit", type=int, help="Smoke only; always STOPPED, never complete metrics")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None or args.out is None:
        parser.error("--root and --out required")
    if min(args.max_seconds, args.chunk_images, args.bootstrap) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Budgets/counts must be positive")
    signal.signal(signal.SIGINT, stop_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_signal)
    run(args)


if __name__ == "__main__":
    main()
