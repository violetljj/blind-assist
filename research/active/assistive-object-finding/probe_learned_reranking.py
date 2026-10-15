#!/usr/bin/env python3
"""One fixed Development source-grouped candidate reranking probe, CPU only.

Three GroupKFold folds; no tuning or held-out-label candidate selection.
Training labels: IoU >= .5 positive, <= .2 negative, intermediate ignored.
Presence and eligible pool remain original DINO max >= .25. This is a small
cross-fitted feasibility diagnostic, not independent confirmation or novelty.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analyze_pilot import (evaluate_outputs, iou, load_manifest, paired_outputs,
                           read_jsonl, row_key, top_box)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")


def feature_matrix(observation):
    """Observable allowlist only. Does not accept or inspect manifest labels."""
    candidates = observation["candidates"]
    with np.load(observation["feature_cache"]) as cache:
        boxes = cache["boxes"].astype(np.float64)
        scores = cache["dino_scores"].astype(np.float64)
        crops = cache["crop_embeddings"].astype(np.float64)
        query_emb = cache["query_embedding"].astype(np.float64).reshape(-1)
    if len(candidates) != len(scores):
        raise ValueError("Feature cache/candidate count mismatch")
    if len(scores) and (not np.allclose(boxes, [c["bbox_xyxy"] for c in candidates])
                        or not np.allclose(scores, [c["score"] for c in candidates])):
        raise ValueError("Feature cache order mismatch")
    eligible = np.flatnonzero(scores >= .25)
    names = ["dino_logit", "clip_cosine", "cx", "cy", "width", "height",
             "area", "log_aspect", "dino_relative_mean", "clip_relative_mean",
             "dino_relative_best", "clip_relative_best", "dino_rank", "clip_rank",
             "relative_x", "relative_y", "relative_area", "max_other_iou",
             "query_lr_x", "query_tb_y", "query_size_area", "query_lr_rank",
             "query_size_rank", "pool_count_log"]
    if not len(eligible):
        return eligible, np.empty((0, 2)), np.empty((0, len(names))), names
    b, s = boxes[eligible], scores[eligible]
    c = crops[eligible] @ query_emb
    logit = np.log(np.clip(s, 1e-6, 1-1e-6) / np.clip(1-s, 1e-6, 1))
    independent = np.column_stack([logit, c])
    width, height = float(observation["width"]), float(observation["height"])
    cx, cy = (b[:, 0]+b[:, 2])/(2*width), (b[:, 1]+b[:, 3])/(2*height)
    bw, bh = (b[:, 2]-b[:, 0])/width, (b[:, 3]-b[:, 1])/height
    area = bw*bh
    n = len(s)
    # Stable ordinal ranks computed from observations only.
    def rank(v):
        order = np.argsort(v, kind="stable")
        out = np.empty(len(v), dtype=float)
        out[order] = np.arange(len(v)) / max(1, len(v)-1)
        return out
    other_iou = np.zeros(n)
    for j in range(n):
        other_iou[j] = max((iou(b[j].tolist(), b[k].tolist()) for k in range(n) if k != j), default=0.)
    words = set(re.findall(r"[a-z]+", str(observation["query"]).lower()))
    lr = int(bool(words & {"right", "rightmost"})) - int(bool(words & {"left", "leftmost"}))
    tb = int(bool(words & {"bottom", "bottommost", "lower"})) - int(bool(words & {"top", "topmost", "upper"}))
    size = int(bool(words & {"large", "larger", "largest", "big", "bigger", "biggest"})) - int(bool(words & {"small", "smaller", "smallest", "little", "tiny"}))
    comparative = np.column_stack([
        logit, c, cx, cy, bw, bh, area, np.log(np.maximum(bw, 1e-9)/np.maximum(bh, 1e-9)),
        s-s.mean(), c-c.mean(), s-s.max(), c-c.max(), rank(s), rank(c),
        cx-cx.mean(), cy-cy.mean(), area-area.mean(), other_iou,
        lr*(cx-cx.mean()), tb*(cy-cy.mean()), size*(area-area.mean()),
        lr*(rank(cx)-.5), size*(rank(area)-.5), np.full(n, np.log1p(n))])
    if not np.isfinite(comparative).all():
        raise ValueError("Nonfinite observation feature")
    return eligible, independent, comparative, names


def training_arrays(indices, rows, observations, features, variant):
    # Only training-row labels are passed to this helper by the fold loop.
    data, labels, weights = [], [], []
    source_frames = Counter(rows[i]["source_id"] for i in indices)
    for idx in indices:
        row = rows[idx]
        eligible, independent, comparative, _ = features[idx]
        x = independent if variant == "independent" else comparative
        obs = observations[row_key(row)]
        overlaps = np.asarray([iou(obs["candidates"][int(j)]["bbox_xyxy"], row["gt_bbox_xyxy"])
                               if row["gt_present"] else 0. for j in eligible])
        valid = (overlaps >= .5) | (overlaps <= .2)
        y = (overlaps[valid] >= .5).astype(int)
        if not len(y):
            continue
        w = np.empty(len(y))
        classes = np.unique(y)
        for label in classes:
            selected = y == label
            w[selected] = 1. / (len(classes)*selected.sum()*source_frames[row["source_id"]])
        data.append(x[valid]); labels.append(y); weights.append(w)
    X, y, w = np.concatenate(data), np.concatenate(labels), np.concatenate(weights)
    w *= len(w)/w.sum()
    return X, y, w


def prediction_row(observation, eligible, decision, method, fold):
    out = dict(observation)
    candidates = [dict(c) for c in observation["candidates"]]
    for c in candidates:
        c["ranking_score"] = -100.+float(c["score"])
    for j, value in zip(eligible, decision):
        candidates[int(j)]["ranking_score"] = 100.+float(value)
    candidates.sort(key=lambda c: (-c["ranking_score"], -c["score"], tuple(c["bbox_xyxy"])))
    out.update(candidates=candidates, selection_method=method, oof_fold=fold,
               score_semantics="model_confidence",
               ranking_semantics="original_DINO_presence_unmodified; learned_ranking_only",
               baseline_grounding_elapsed_s=out.get("elapsed_s"), elapsed_s=None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    root = args.root.resolve()
    dest = root / "learned_rank_probe"
    dest.mkdir(exist_ok=False)
    manifest, predictions = root/"data/samples.jsonl", root/"inference/base_predictions.jsonl"
    config = {"label": "Development source-grouped cross-fitting, not independent confirmation",
              "split": "GroupKFold(n_splits=3), group=source_id, shuffle=False",
              "model": "StandardScaler + LogisticRegression(C=1,L2,solver=lbfgs,max_iter=200,random_state=0)",
              "train_labels": "IoU>=0.5 positive, <=0.2 negative; intermediate ignored",
              "sample_weight": "each labeled frame total 1/full-training-source-frame-count; equal observed classes within frame; global normalized mean1; ignored-only frames contribute zero",
              "eligible_pool": "original DINO score>=0.25; no pool expansion",
              "presence": "original max DINO score>=0.25, unchanged",
              "new_download_bytes": 0, "gpu_seconds": 0,
              "script_sha256": digest(__file__), "manifest_sha256": digest(manifest),
              "baseline_sha256": digest(predictions)}
    write_json(dest/"config.json", config)
    rows = load_manifest(manifest)
    observations = {row_key(r):r for r in read_jsonl(predictions)}
    if set(map(row_key, rows)) != set(observations):
        raise ValueError("Manifest/prediction coverage mismatch")
    # Explicit observation allowlist for feature construction, excluding labels/source.
    features = [feature_matrix({k:observations[row_key(r)][k] for k in
                 ("candidates", "feature_cache", "query", "width", "height")}) for r in rows]
    models = {"dino": [None]*len(rows), "independent": [None]*len(rows),
              "comparative": [None]*len(rows)}
    folds, fits = [], []
    groups = np.asarray([r["source_id"] for r in rows])
    fit_start = time.monotonic()
    for fold, (train, test) in enumerate(GroupKFold(n_splits=3).split(np.zeros(len(rows)), groups=groups)):
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Source leak")
        folds.append({"fold":fold,"train_sources":sorted(set(groups[train])),
                      "test_sources":sorted(set(groups[test])), "train_frames":len(train),
                      "test_frames":len(test)})
        for idx in test:
            models["dino"][idx] = {**observations[row_key(rows[idx])], "oof_fold":fold}
        for variant in ("independent", "comparative"):
            X, y, weights = training_arrays(train, rows, observations, features, variant)
            if len(np.unique(y)) != 2:
                raise ValueError("Training fold lacks both classes")
            scaler = StandardScaler().fit(X, sample_weight=weights)
            clf = LogisticRegression(C=1.,solver="lbfgs", max_iter=200,random_state=0)
            clf.fit(scaler.transform(X),y,sample_weight=weights)
            fits.append({"fold":fold, "model":variant, "candidate_samples":len(y),
                         "positive_samples":int(y.sum()), "iterations":clf.n_iter_.tolist(),
                         "feature_names":features[0][3][:2] if variant=="independent" else features[0][3],
                         "scaler_mean":scaler.mean_.tolist(), "scaler_scale":scaler.scale_.tolist(),
                         "coefficient":clf.coef_.tolist(), "intercept":clf.intercept_.tolist()})
            for idx in test:
                eligible, independent, comparative, _ = features[idx]
                test_features = independent if variant == "independent" else comparative
                decision = clf.decision_function(scaler.transform(test_features)) if len(eligible) else []
                models[variant][idx] = prediction_row(observations[row_key(rows[idx])], eligible,
                                                      decision, "OOF_L2_logistic_"+variant, fold)
            if time.monotonic()-fit_start > 120:
                raise TimeoutError("CPU fitting budget 120s exhausted; no retry/tuning")
    outputs = {}
    for name, values in models.items():
        if any(v is None for v in values):
            raise ValueError("Incomplete OOF")
        with (dest/(name+"_predictions.jsonl")).open("w",encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
        outputs[name] = {row_key(r):top_box(v["candidates"],.25) for r,v in zip(rows,values)}
    summary = {"label":config["label"],"frames":len(rows),"sources":len(set(groups)),
               "metrics":{n:evaluate_outputs(rows,o,.5) for n,o in outputs.items()},
               "paired_vs_dino":{n:paired_outputs(rows,outputs["dino"],o,.5) for n,o in outputs.items() if n!="dino"},
               "comparative_vs_independent":paired_outputs(rows,outputs["independent"],outputs["comparative"],.5)}
    present = [r for r in rows if r["gt_present"]]
    eligible_oracle = sum(any(c["score"]>=.25 and iou(c["bbox_xyxy"],r["gt_bbox_xyxy"])>=.5
                             for c in observations[row_key(r)]["candidates"]) for r in present)
    full_oracle = sum(any(iou(c["bbox_xyxy"],r["gt_bbox_xyxy"])>=.5
                         for c in observations[row_key(r)]["candidates"]) for r in present)
    summary["evaluator_only_coverage"]={"visible_frames":len(present),"eligible_pool_oracle":eligible_oracle,
                                         "all_cached_pool_oracle":full_oracle,"pool_expansion_tested":False}
    summary["by_source"] = {}
    for source in sorted(set(groups)):
        subset = [r for r in rows if r["source_id"]==source]
        summary["by_source"][source] = {"metrics":{n:evaluate_outputs(subset,o,.5) for n,o in outputs.items()},
            "paired_vs_dino":{n:paired_outputs(subset,outputs["dino"],o,.5) for n,o in outputs.items() if n!="dino"}}
    summary["by_fold"] = []
    for fold in folds:
        subset = [r for r in rows if r["source_id"] in fold["test_sources"]]
        summary["by_fold"].append({**fold,"metrics":{n:evaluate_outputs(subset,o,.5) for n,o in outputs.items()},
            "paired_vs_dino":{n:paired_outputs(subset,outputs["dino"],o,.5) for n,o in outputs.items() if n!="dino"}})
    summary["elapsed_seconds"]=time.monotonic()-start
    summary["fit_seconds"]=time.monotonic()-fit_start
    write_json(dest/"summary.json",summary)
    write_json(dest/"folds_and_models.json",{"folds":folds,"fits":fits})
    write_json(dest/"receipt.json",{"config_sha256":digest(dest/"config.json"),
        "outputs":{p.name:digest(p) for p in dest.glob("*.jsonl")},
        "script_sha256":digest(__file__),"status":"COMPLETE","new_download_bytes":0,"gpu_seconds":0})
    print(json.dumps({k:v for k,v in summary.items() if k not in ("by_source","by_fold")},ensure_ascii=False))


if __name__ == "__main__":
    main()
