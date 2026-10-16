#!/usr/bin/env python3
"""Complete six-arm joint-point campaign report from numerical receipts only.

No Torch, model, image, GT mask, checkpoint or logits are read. All six training
and all 24 export/score pairs must be COMPLETE before numerical conclusions are
produced. Reconstruct score totals, preserve UNKNOWN, and bootstrap paired
image/class queries with shared IIT RGB / UMD author-source cluster draws.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ARMS = ("plain", "plain_quality", "detail", "detail_quality", "candidate_quality", "joint")
FAMILIES = ("point", "budget")
DATASETS = ("iit", "umd")
EXPECTED = {"iit": (1757, 3821, 1754), "umd": (9939, 19370, 108)}
SHARED = (1, 2, 5, 7, 8, 9)
COUNT_FIELDS = ("queries", "hits", "evaluable", "unknown", "small_queries", "small_hits", "small_evaluable", "small_unknown")
COMPARISONS = (("plain_quality", "plain"), ("detail", "plain"),
               ("detail_quality", "plain_quality"), ("candidate_quality", "plain_quality"),
               ("joint", "plain_quality"), ("joint", "detail_quality"), ("joint", "candidate_quality"))
CLASSES = {1: "contain", 2: "cut", 3: "display", 4: "engine", 5: "grasp", 6: "hit",
           7: "pound", 8: "support", 9: "wrap-grasp"}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def preflight(root):
    states, missing = {}, []
    for arm in ARMS:
        paths = [root / "runs" / arm / "terminal.json"]
        paths.extend(root / "evaluation" / family / arm / dataset / stage / "terminal.json"
                     for family in FAMILIES for dataset in DATASETS for stage in ("export", "score"))
        for path in paths:
            relative = path.relative_to(root).as_posix()
            state = read(path).get("status", "MISSING_STATUS") if path.is_file() else "MISSING"
            states[relative] = state
            if state != "COMPLETE":
                missing.append({"path": relative, "status": state})
    return states, missing


def verify_outputs(directory, inputs):
    terminal_path = directory / "terminal.json"
    terminal = read(terminal_path)
    inputs[terminal_path] = sha(terminal_path)
    for filename, expected in terminal.get("output_sha256", {}).items():
        path = directory / filename
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Completed output SHA mismatch: {path}")
        inputs[path] = expected
    return terminal


def segmentation(cm, ids):
    union = cm.sum(0) + cm.sum(1) - cm.diagonal()
    iou = np.divide(cm.diagonal(), union, out=np.full(10, np.nan), where=union > 0)
    values = iou[list(ids)]
    return {"mIoU": float(np.nanmean(values)) if np.isfinite(values).any() else None,
            "per_class_iou": {str(i): float(v) if np.isfinite(v) else None for i, v in enumerate(iou)},
            "confusion_GT_rows_prediction_columns": cm.tolist()}


def point_summary(count):
    value = {field: int(number) for field, number in zip(COUNT_FIELDS, count)}
    for prefix in ("", "small_"):
        value[prefix + "hit_rate_evaluable"] = (value[prefix + "hits"] / value[prefix + "evaluable"]
            if value[prefix + "evaluable"] else None)
        value[prefix + "hits_over_all_positive_queries"] = (value[prefix + "hits"] / value[prefix + "queries"]
            if value[prefix + "queries"] else None)
    return value


def assert_match(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(label)
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if abs(actual - expected) > 1e-12:
            raise AssertionError(f"{label}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(label)


def load_evaluation(directory, dataset, basis, inputs):
    score_terminal = verify_outputs(directory / "score", inputs)
    export_terminal = verify_outputs(directory / "export", inputs)
    summary_path = directory / "score/summary.json"
    summary = read(summary_path)
    inputs[summary_path] = sha(summary_path)
    config_path = directory / "export/config.json"
    export_config = read(config_path)
    inputs[config_path] = sha(config_path)
    score_config_path = directory / "score/config.json"
    score_config = read(score_config_path)
    inputs[score_config_path] = sha(score_config_path)
    if summary.get("status") != "COMPLETE" or summary["dataset"] != dataset:
        raise ValueError("Score dataset/status differs from the declared matrix cell")
    if (score_config["dataset"] != dataset or export_config["dataset"] != dataset or
            score_config["manifest_sha256"] != inputs[directory / "export/predictions.jsonl"] or
            summary.get("selectors_read_GT") is not False or
            summary.get("training_or_checkpoint_selection_on_UMD") is not False):
        raise ValueError("Export-to-score lineage or evaluator-only UMD contract differs")
    methods = sorted(summary["methods"])
    main = "quality" if "quality" in methods else "stable"
    if main not in methods or "stable" not in methods:
        raise ValueError("Missing shared stable selector")
    classes = tuple(SHARED if dataset == "umd" else range(1, 10))
    if summary["primary_foreground_ids"] != list(classes):
        raise ValueError("Primary class vocabulary changed")
    records_path = directory / "score/per_image.jsonl"
    if records_path not in inputs:
        inputs[records_path] = sha(records_path)
    raw_rows = list(rows(records_path))
    expected_images, expected_queries, expected_groups = EXPECTED[dataset]
    if len(raw_rows) != expected_images or len({r["source_id"] for r in raw_rows}) != expected_images:
        raise ValueError(f"Complete unique-image denominator differs: {records_path}")
    raw_rows.sort(key=lambda r: r["source_id"])
    groups = sorted({r["bootstrap_group"] for r in raw_rows})
    if len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} source groups, got {len(groups)}")
    if dataset == "iit" and any(not g.startswith("rgb_sha256:") for g in groups):
        raise ValueError("IIT bootstrap must use exact RGB SHA groups")
    group_index = {g: i for i, g in enumerate(groups)}
    if basis is None:
        basis = {"groups": groups, "images": {}, "queries": [], "query_group": [],
                 "small": [], "class_ids": [], "source_kind": []}
        for r in raw_rows:
            cm = np.asarray(r["confusion"], dtype=np.int64)
            positive = sorted(r["points"][main], key=lambda p: p["class_id"])
            basis["images"][r["source_id"]] = (r["gt_sha256"], r["bootstrap_group"], cm.sum(1).tolist(),
                r["valid_pixels"], r["ignored_pixels_by_reason"])
            for p in positive:
                if p["class_id"] in classes:
                    basis["queries"].append((r["source_id"], p["class_id"], p["target_pixels"], p["target_fraction"], p["small_region"]))
                    basis["query_group"].append(group_index[r["bootstrap_group"]])
                    basis["small"].append(p["small_region"])
                    basis["class_ids"].append(p["class_id"])
                    basis["source_kind"].append("clutter" if "/clutter/" in r["bootstrap_group"] else "tool" if dataset == "umd" else str(r.get("image_source")))
        if len(basis["queries"]) != expected_queries:
            raise ValueError("Full positive image/class query denominator differs")
        for key, dtype in (("query_group", np.int32), ("small", bool), ("class_ids", np.int8)):
            basis[key] = np.asarray(basis[key], dtype=dtype)
    elif groups != basis["groups"]:
        raise ValueError("Comparison must retain identical RGB/tool/scene groups")
    status = {method: np.full(expected_queries, -2, dtype=np.int8) for method in methods}
    group_cm = np.zeros((len(groups), 10, 10), dtype=np.int64)
    counts = np.zeros((len(groups), len(methods), 10, 8), dtype=np.int64)
    conditions = {method: Counter() for method in methods}
    ignored_pixels = Counter()
    cursor = 0
    for r in raw_rows:
        sid = r["source_id"]
        cm = np.asarray(r["confusion"], dtype=np.int64)
        actual_image = (r["gt_sha256"], r["bootstrap_group"], cm.sum(1).tolist(), r["valid_pixels"], r["ignored_pixels_by_reason"])
        if sid not in basis["images"] or actual_image != basis["images"][sid]:
            raise ValueError("Paired arms differ in image/GT identity or denominators")
        if cm.shape != (10, 10) or int(cm.sum()) != r["valid_pixels"]:
            raise ValueError("Malformed native confusion matrix")
        gi = group_index[r["bootstrap_group"]]
        group_cm[gi] += cm
        ignored_pixels.update(r["ignored_pixels_by_reason"])
        per_method = {method: sorted(r["points"][method], key=lambda p: p["class_id"]) for method in methods}
        reference = per_method[main]
        if any([p["class_id"] for p in ps] != [p["class_id"] for p in reference] for ps in per_method.values()):
            raise ValueError("Selectors must have identical native positive-class query sets")
        for local, ref in enumerate(reference):
            cls = ref["class_id"]
            if cls not in classes:
                raise ValueError("Unexpected positive target outside primary label space")
            expected_query = basis["queries"][cursor]
            for mi, method in enumerate(methods):
                p = per_method[method][local]
                actual_query = (sid, cls, p["target_pixels"], p["target_fraction"], p["small_region"])
                if actual_query != expected_query or not (p["hit"] is None or type(p["hit"]) is bool):
                    raise ValueError("Query alignment or tri-state status changed")
                hit = p["hit"]
                status[method][cursor] = 1 if hit is True else 0 if hit is False else -1
                values = (1, hit is True, hit is not None, hit is None)
                counts[gi, mi, cls, :4] += values
                if p["small_region"]:
                    counts[gi, mi, cls, 4:] += values
                conditions[method]["category/" + p["category"]] += 1
                if p["ignore_reason"]:
                    conditions[method]["unknown_reason/" + p["ignore_reason"]] += 1
                if hit is False:
                    conditions[method]["miss_support/" + p["segmentation_support"]] += 1
                    if p["small_region"]:
                        conditions[method]["small_miss_support/" + p["segmentation_support"]] += 1
            cursor += 1
    if cursor != expected_queries or any((a == -2).any() for a in status.values()):
        raise ValueError("Unfilled complete-query matrix")
    cm = group_cm.sum(0)
    reconstructed = segmentation(cm, classes)
    for key in ("mIoU", "per_class_iou", "confusion_GT_rows_prediction_columns"):
        if key == "per_class_iou":
            for cls, value in reconstructed[key].items():
                assert_match(value, summary["segmentation"][key][cls], "per-class IoU")
        else:
            assert_match(reconstructed[key], summary["segmentation"][key], key)
    method_stats = {}
    for mi, method in enumerate(methods):
        total = counts[:, mi].sum(0)
        primary = point_summary(total[list(classes)].sum(0))
        per_class = {str(c): point_summary(total[c]) for c in classes}
        for key, value in primary.items():
            assert_match(value, summary["methods"][method]["primary"][key], "primary " + method + "/" + key)
        for cls, stats in per_class.items():
            for key, value in stats.items():
                assert_match(value, summary["methods"][method]["per_class"][cls][key], "per-class points")
        method_stats[method] = {"primary": primary, "per_class": per_class, "errors": dict(conditions[method])}
    if dict(ignored_pixels) != summary["ignored_pixels_by_reason"]:
        raise AssertionError("Native void/scoop pixel counts differ")
    return {"dataset": dataset, "main_method": main, "images": expected_images, "positive_queries": expected_queries,
            "bootstrap_groups": len(groups), "segmentation": reconstructed,
            "shared6_segmentation": segmentation(cm, SHARED), "methods": method_stats,
            "ignored_pixels_by_reason": dict(ignored_pixels), "export_terminal": export_terminal,
            "export_config": export_config, "score_terminal": score_terminal,
            "status_arrays": status, "group_cm": group_cm, "group_counts": counts,
            "score_seconds": score_terminal.get("wall_seconds"), "export_seconds": export_terminal.get("wall_seconds")}, basis


def percentile(array):
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {"CI95_percentile": np.percentile(array, [2.5, 97.5]).tolist() if len(array) else None,
            "valid_draws": len(array)}


def divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def cm_miou(cms, classes):
    tp = np.diagonal(cms, axis1=-2, axis2=-1)
    union = cms.sum(-2) + cms.sum(-1) - tp
    iou = divide(tp, union)
    selected = iou[:, list(classes)]
    return np.nanmean(selected, axis=1)


def grouped_metrics(cell, weights, dataset):
    classes = SHARED if dataset == "umd" else tuple(range(1, 10))
    gcm = cell["group_cm"]
    boot_cm = (weights @ gcm.reshape(len(gcm), -1)).reshape(len(weights), 10, 10)
    metrics = {"mIoU": cm_miou(boot_cm, classes), "shared6_mIoU": cm_miou(boot_cm, SHARED)}
    all_counts = cell["group_counts"]
    methods = sorted(cell["methods"])
    boot_count = (weights @ all_counts.reshape(len(gcm), -1)).reshape(len(weights), len(methods), 10, 8)
    for mi, method in enumerate(methods):
        for name, cls in [("overall", list(classes)), *[("class/" + str(c), [c]) for c in classes]]:
            count = boot_count[:, mi, cls].sum(1)
            prefix = method + "/" + name + "/"
            metrics[prefix + "hits_over_all_positive_queries"] = divide(count[:, 1], count[:, 0])
            metrics[prefix + "hit_rate_evaluable"] = divide(count[:, 1], count[:, 2])
            metrics[prefix + "small_hits_over_all_positive_queries"] = divide(count[:, 5], count[:, 4])
    return metrics


def paired_comparison(a, b, basis, weights):
    sa, sb = a["status_arrays"][a["main_method"]], b["status_arrays"][b["main_method"]]
    grouped = basis["query_group"]
    group_count = len(basis["groups"])
    strata = {"overall": np.ones(len(sa), dtype=bool), "small": basis["small"], "not_small": ~basis["small"]}
    strata.update({"class/" + str(c): basis["class_ids"] == c for c in sorted(set(basis["class_ids"].tolist()))})
    if a["dataset"] == "umd":
        strata.update({"source/" + k: np.asarray(basis["source_kind"]) == k for k in ("tool", "clutter")})
    result = {}
    for name, selected in strata.items():
        ga, gb = sa[selected], sb[selected]
        gi = grouped[selected]
        common = (ga >= 0) & (gb >= 0)
        counts = np.zeros((group_count, 6), dtype=np.float64)
        for ci, value in enumerate((ga == 1, gb == 1, np.ones(len(ga), dtype=bool),
                                    (ga == 1) & common, (gb == 1) & common, common)):
            counts[:, ci] = np.bincount(gi, weights=value, minlength=group_count)
        boot = weights @ counts
        total = counts.sum(0)
        result[name] = {"positive_queries": len(ga), "a_hits": int(total[0]), "b_hits": int(total[1]),
            "a_unknown": int((ga < 0).sum()), "b_unknown": int((gb < 0).sum()),
            "conservative_net_hits": int(total[0] - total[1]),
            "conservative_hit_rate_delta": float((total[0] - total[1]) / total[2]) if total[2] else None,
            "conservative_bootstrap_delta": percentile(divide(boot[:, 0] - boot[:, 1], boot[:, 2])),
            "paired_common_evaluable": {"queries": int(total[5]), "a_hits": int(total[3]), "b_hits": int(total[4]),
                "hit_rate_delta": float((total[3] - total[4]) / total[5]) if total[5] else None,
                "bootstrap_delta": percentile(divide(boot[:, 3] - boot[:, 4], boot[:, 5]))},
            "transitions": {"rescued_known_miss": int(((ga == 1) & (gb == 0)).sum()),
                "harmed_known_hit": int(((ga == 0) & (gb == 1)).sum()),
                "unknown_in_either": int(((ga < 0) | (gb < 0)).sum()),
                "both_hit": int(((ga == 1) & (gb == 1)).sum()), "both_known_miss": int(((ga == 0) & (gb == 0)).sum())}}
    return result


def training_data(root, inputs):
    result = {}
    shared = None
    invariant_keys = ("records_sha256", "pretrained_sha256", "cache_index_sha256", "source_sha256", "model_source_sha256",
        "size", "input_size", "seed", "init_seed", "effective_batch", "lr", "schedule_epochs", "val_every", "compute_budget_seconds")
    for arm in ARMS:
        directory = root / "runs" / arm
        paths = [directory / name for name in ("terminal.json", "config.json", "progress.jsonl")]
        for path in paths:
            inputs[path] = sha(path)
        terminal, config = read(paths[0]), read(paths[1])
        if config["arm"] != arm or terminal["arm"] != arm or terminal["test_access"] is not False:
            raise ValueError("Training identity/test-access contract differs")
        current_shared = {k: config.get(k) for k in invariant_keys}
        if shared is None:
            shared = current_shared
        elif shared != current_shared:
            raise ValueError("Same-data/initialization/schedule/compute-budget invariants differ across arms")
        if config["train_rows"] != 4417 or config["cleanval_rows"] != 1757:
            raise ValueError("Full IIT4417/1757 training denominator differs")
        if terminal["images_exposed"] != terminal["expected_images_exposed"] or terminal["images_exposed"] != terminal["epochs"] * 4417:
            raise ValueError("Training did not complete its declared full exposure")
        curves = [r for r in rows(paths[2]) if r.get("stage") in ("validation", "epoch")]
        validations = [r for r in curves if r["stage"] == "validation"]
        if not validations or max(r["epoch"] for r in curves) != terminal["epochs"]:
            raise ValueError("Complete training requires its final epoch/validation record")
        eligible = [r for r in validations if r.get("budget_checkpoint_eligible")]
        latest = max(eligible, key=lambda r: r["epoch"]) if eligible else None
        budget = terminal.get("compute_budget_seconds")
        result[arm] = {"terminal": terminal, "config": config, "curves": curves,
            "latest_budget_eligible_boundary": latest,
            "last_budget_boundary_undershoot_seconds": budget - latest["training_update_wall_seconds"] if budget is not None and latest else None,
            "cost_scope": "update wall excludes validation and checkpoint I/O; terminal process wall is this attempt only after a resume",
            "selected_checkpoint_costs": {}}
    if len({r["terminal"]["images_exposed"] for r in result.values()}) != 1:
        raise ValueError("Equal-exposure training arms did not use equal full exposure")
    return result, shared


def selected_cost(training, export_terminal, family):
    epoch, updates = export_terminal["checkpoint_epoch"], export_terminal["checkpoint_updates"]
    matches = [r for r in training["curves"] if r.get("epoch") == epoch and r.get("updates") == updates and r["stage"] == "validation"]
    selected_key = training["terminal"]["best_budget_key" if family == "budget" else "best_key"]
    matches = [r for r in matches if [r["point_hit"][r["selection_method"]], r["miou_foreground"]] == selected_key]
    if not matches:
        raise ValueError("Exported checkpoint does not match the recorded selected validation key/boundary")
    costs = [r["training_update_wall_seconds"] for r in matches]
    if family == "budget" and not all(r.get("budget_checkpoint_eligible") for r in matches):
        raise ValueError("Budget checkpoint was selected outside the shared wall-budget ceiling")
    budget = training["terminal"].get("compute_budget_seconds")
    return {"epoch": epoch, "updates": updates, "images_exposed": matches[-1]["images_exposed"],
        "training_update_wall_seconds": costs[0] if len(set(costs)) == 1 else None,
        "matching_history_cost_range_seconds": [min(costs), max(costs)],
        "matching_validation_records": len(matches),
        "budget_seconds": budget,
        "selected_boundary_undershoot_seconds": budget - max(costs) if family == "budget" and budget is not None else None,
        "last_eligible_boundary_undershoot_seconds": training["last_budget_boundary_undershoot_seconds"] if family == "budget" else None}


def strip_cell(cell):
    return {k: v for k, v in cell.items() if k not in ("status_arrays", "group_cm", "group_counts", "bootstrap_arrays")}


def markdown(summary):
    lines = ["# Full joint functional-part campaign", "", "All six training arms and 24 export/score cells completed. Numerical receipts were reconstructed; no model, RGB, GT mask or logits were reopened.", "",
        "This is a single-training-seed exploratory ablation on already consumed IIT/UMD data. Cluster intervals describe source variation for these trained models, not training-seed uncertainty or independent confirmation.", ""]
    for family in FAMILIES:
        lines += ["## " + ("Equal-exposure training, IIT point-selected checkpoint" if family == "point" else "Shared update-wall ceiling, IIT budget-selected checkpoint"), "",
            "| Arm | IIT mIoU | IIT primary hits / 3821 | IIT small hits / queries | UMD shared6 mIoU | UMD primary hits / 19370 | UMD small hits / queries | Selected update wall s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            iit, umd = (summary["matrix"][family][dataset][arm] for dataset in DATASETS)
            pi, pu = (cell["methods"][cell["main_method"]]["primary"] for cell in (iit, umd))
            cost = summary["training"][arm]["selected_checkpoint_costs"][family]
            seconds = cost["training_update_wall_seconds"]
            lines.append(f"| {arm} ({iit['main_method']}) | {100*iit['segmentation']['mIoU']:.3f}% | {pi['hits']}/3821 | {pi['small_hits']}/{pi['small_queries']} | {100*umd['segmentation']['mIoU']:.3f}% | {pu['hits']}/19370 | {pu['small_hits']}/{pu['small_queries']} | {seconds:.3f} |" if seconds is not None else
                         f"| {arm} ({iit['main_method']}) | {100*iit['segmentation']['mIoU']:.3f}% | {pi['hits']}/3821 | {pi['small_hits']}/{pi['small_queries']} | {100*umd['segmentation']['mIoU']:.3f}% | {pu['hits']}/19370 | {pu['small_hits']}/{pu['small_queries']} | AMBIGUOUS; see JSON |")
        lines += ["", "UNKNOWN landing counts and evaluable-only rates remain separate in summary.json. Primary selector is quality when available, otherwise stable. All original ten predicted classes remain; UMD native -1 and scoop3 are ignored separately.", "",
            "| UMD comparison A minus B | Net hits / 19370 | Delta pp [95% source-cluster interval] | Small net hits |", "|---|---:|---:|---:|"]
        for a, b in COMPARISONS:
            paired = summary["comparisons"][family]["umd"][a + "_minus_" + b]["point_paired"]
            whole, small = paired["overall"], paired["small"]
            ci = whole["conservative_bootstrap_delta"]["CI95_percentile"]
            lines.append(f"| {a} − {b} | {whole['conservative_net_hits']:+d} | {100*whole['conservative_hit_rate_delta']:+.3f} [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] | {small['conservative_net_hits']:+d} |")
        if family == "budget":
            lines += ["", "The budget is a common upper bound on accumulated update wall, sampled only at scheduled validation boundaries. Selected epochs/exposure and actual seconds can differ by arm. This is not exact equal-seconds or equal-FLOPs training; per-arm selected and last-eligible undershoot are explicit in JSON.", ""]
    lines += ["## Full cost and errors", "", "| Arm | Final images exposed | Update wall s | CUDA event span s | Process wall this attempt s | AMP skipped | Peak allocated GiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, value in summary["training"].items():
        t = value["terminal"]
        lines.append(f"| {arm} | {t['images_exposed']} | {t['training_update_wall_seconds']:.3f} | {t.get('training_cuda_event_seconds',0):.3f} | {t['wall_seconds']:.3f} | {t.get('amp_skipped_updates',0)} | {t.get('peak_allocated_bytes',0)/2**30:.3f} |")
    lines += ["", "Full per-class, small-region, fixed-selector, native foreground-support errors, numerical cost curves and paired common-evaluable comparisons are in summary.json. These are semantic class unions, not physical instance-exclusive parts or task-role truth. No absence/rejection or blind-user safety endpoint is measured.", "",
        "CUDA event spans include idle intervals between stream events and are not a sum of kernel execution time. Update wall excludes validation/checkpoint I/O. Resume process wall is not a cumulative whole-run wall. No unsupported total-GPU-hours or FLOPs claim is inferred.", "",
        "There are multiple exploratory comparisons and one training seed per arm. Intervals are not multiplicity-adjusted. UMD was used only for evaluation; checkpoint selection uses the recorded IIT validation boundaries.", ""]
    return "\n".join(lines)


def run(args):
    begin, cpu_begin = time.monotonic(), time.process_time()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    states, missing = preflight(root)
    if missing:
        result = {"status": "NOT_COMPLETE", "checked_terminals": len(states), "incomplete": missing,
                  "generated_utc": utc(), "source_sha256": sha(__file__), "performance_conclusions": "NOT_PRODUCED"}
        write(output / "terminal.json", result)
        write(output / "summary.json", result)
        (output / "REPORT.md").write_text("# NOT COMPLETE\n\nNo performance conclusion was produced. " + str(len(missing)) + " required terminal records are missing or not COMPLETE. See terminal.json.\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        return 2
    inputs = {}
    training, invariants = training_data(root, inputs)
    matrix = {f: {d: {} for d in DATASETS} for f in FAMILIES}
    comparisons = {f: {d: {} for d in DATASETS} for f in FAMILIES}
    for dataset in DATASETS:
        basis = None
        cells = {}
        for family in FAMILIES:
            for arm in ARMS:
                directory = root / "evaluation" / family / arm / dataset
                cell, basis = load_evaluation(directory, dataset, basis, inputs)
                if cell["export_terminal"]["arm"] != arm:
                    raise ValueError("Exported arm differs from declared matrix location")
                train_config = training[arm]["config"]
                export_config = cell["export_config"]
                train_terminal_path = root / "runs" / arm / "terminal.json"
                if (export_config["selection"] != family or
                        export_config["training_terminal_sha256"] != inputs[train_terminal_path] or
                        export_config["trainer_source_sha256"] != train_config["source_sha256"] or
                        export_config["model_source_sha256"] != train_config["model_source_sha256"] or
                        (dataset == "iit" and export_config["records_sha256"] != train_config["records_sha256"])):
                    raise ValueError("Selected checkpoint/source/training-terminal lineage differs")
                checkpoint_cost = selected_cost(training[arm], cell["export_terminal"], family)
                checkpoint_cost["checkpoint_sha256"] = export_config["checkpoint_sha256"]
                existing = training[arm]["selected_checkpoint_costs"].get(family)
                if existing is not None and existing != checkpoint_cost:
                    raise ValueError("IIT and UMD were exported from different selected checkpoint cursors")
                training[arm]["selected_checkpoint_costs"][family] = checkpoint_cost
                cells[(family, arm)] = cell
        if len({cell["export_config"]["records_sha256"] for cell in cells.values()}) != 1:
            raise ValueError("A dataset's observation manifest must be identical across arms and checkpoint families")
        count = len(basis["groups"])
        rng = np.random.default_rng(args.seed)
        weights = np.zeros((args.bootstrap, count), dtype=np.float64)
        for draw in range(args.bootstrap):
            weights[draw] = np.bincount(rng.integers(0, count, count), minlength=count)
        for family in FAMILIES:
            for arm in ARMS:
                cell = cells[(family, arm)]
                metrics = grouped_metrics(cell, weights, dataset)
                cell["bootstrap_arrays"] = metrics
                cell["bootstrap"] = {"draws": args.bootstrap, "seed": args.seed, "groups": count,
                    "intervals": {key: percentile(value) for key, value in metrics.items()}}
                matrix[family][dataset][arm] = strip_cell(cell)
            for a, b in COMPARISONS:
                ca, cb = cells[(family, a)], cells[(family, b)]
                main_a, main_b = ca["main_method"], cb["main_method"]
                metrics_a, metrics_b = ca["bootstrap_arrays"], cb["bootstrap_arrays"]
                paired = paired_comparison(ca, cb, basis, weights)
                comparisons[family][dataset][a + "_minus_" + b] = {
                    "a_main_method": main_a, "b_main_method": main_b, "point_paired": paired,
                    "segmentation_mIoU_delta": ca["segmentation"]["mIoU"] - cb["segmentation"]["mIoU"],
                    "segmentation_mIoU_bootstrap_delta": percentile(metrics_a["mIoU"] - metrics_b["mIoU"]),
                    "shared6_mIoU_bootstrap_delta": percentile(metrics_a["shared6_mIoU"] - metrics_b["shared6_mIoU"]),
                    "fixed_stable_paired": paired_comparison(dict(ca, main_method="stable"), dict(cb, main_method="stable"), basis, weights)}
        del cells, weights, basis
    # Budget and equal-exposure are reporting families from the SAME six runs,
    # not twelve independently trained models. Do not double-count training cost.
    summary = {"status": "COMPLETE", "generated_utc": utc(), "source_sha256": sha(__file__),
        "required_complete_terminals": len(states), "training_runs": 6, "score_cells": 24,
        "seed_count_per_training_arm": 1, "matrix": matrix, "comparisons": comparisons,
        "training": training, "shared_training_invariants": invariants,
        "bootstrap": {"draws": args.bootstrap, "seed": args.seed, "same_cluster_draws_across_all_arms_and_checkpoint_families": True,
            "iit_grouping": "1754 exact RGB SHA groups", "umd_grouping": "108 author tool/scene groups",
            "training_seed_uncertainty": "NOT_MEASURED", "multiplicity_adjusted": False},
        "denominators": {d: dict(images=EXPECTED[d][0], positive_queries=EXPECTED[d][1], source_groups=EXPECTED[d][2]) for d in DATASETS},
        "inputs_sha256": {str(path.relative_to(root)): value for path, value in inputs.items()},
        "CPU_only": True, "model_GT_RGB_logit_or_checkpoint_read": False,
        "wall_seconds": time.monotonic() - begin, "cpu_seconds": time.process_time() - cpu_begin}
    write(output / "summary.json", summary)
    (output / "REPORT.md").write_text(markdown(summary), encoding="utf-8")
    if "torch" in sys.modules:
        raise AssertionError("Report must not import Torch")
    terminal = {"status": "COMPLETE", "required_complete_terminals": len(states), "score_cells": 24,
        "wall_seconds": time.monotonic() - begin, "cpu_seconds": time.process_time() - cpu_begin,
        "source_sha256": sha(__file__), "output_sha256": {name: sha(output / name) for name in ("summary.json", "REPORT.md")}}
    write(output / "terminal.json", terminal)
    print(json.dumps(terminal), flush=True)
    return 0


def selftest():
    a = {"dataset": "umd", "main_method": "quality", "status_arrays": {"quality": np.array([1, -1, 0, 1], dtype=np.int8)}}
    b = {"dataset": "umd", "main_method": "stable", "status_arrays": {"stable": np.array([0, 1, -1, 1], dtype=np.int8)}}
    basis = {"groups": ["a", "b"], "query_group": np.array([0, 0, 1, 1]),
        "small": np.array([True, True, False, False]), "class_ids": np.array([5, 5, 2, 2]),
        "source_kind": ["tool", "tool", "clutter", "clutter"]}
    weights = np.array([[1., 1.], [2., 0.], [0., 2.]])
    result = paired_comparison(a, b, basis, weights)["overall"]
    assert result["positive_queries"] == 4 and result["conservative_net_hits"] == 0
    assert result["a_unknown"] == result["b_unknown"] == 1
    assert result["paired_common_evaluable"]["queries"] == 2
    assert result["paired_common_evaluable"]["a_hits"] == 2
    assert result["paired_common_evaluable"]["b_hits"] == 1
    assert point_summary([4, 2, 3, 1, 2, 1, 1, 1])["small_hit_rate_evaluable"] == 1
    cm = np.eye(10, dtype=np.int64)
    assert segmentation(cm, SHARED)["mIoU"] == 1
    assert cm_miou(np.stack([cm, cm]), SHARED).tolist() == [1., 1.]
    assert "torch" not in sys.modules
    print(json.dumps({"status": "PASS", "synthetic_only": True,
        "checks": ["tri-state preservation", "paired same image/class", "common-evaluable denominator", "shared source draws", "segmentation sufficient statistics"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
        return
    if args.root is None or args.output is None or args.bootstrap < 1:
        parser.error("Positive --bootstrap and --root/--output required")
    try:
        result = run(args)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / "terminal.json", {"status": "FAILED", "error": str(error),
            "error_type": type(error).__name__, "traceback": traceback.format_exc(), "generated_utc": utc()})
        raise
    raise SystemExit(result)


if __name__ == "__main__":
    main()
