"""Train observation-only error predictors, seal scores, then evaluate separately.

The fit command cannot open dev-eval annotations. All fits use the unchanged
64-image training group. Models optimize strict lexical-coverage errors; relaxed
labels are a sensitivity analysis with exactly the same sealed scores.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "evaluation"))
from score_fields import read_records, score_text, ranking_metrics

CS = (0.1, 1.0, 10.0)
SEED = 20261015


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def model_groups(groups):
    base = groups["baseline"]
    global_keys = base + groups["global_pixel"] + groups["global_ocr"]
    all_baselines = global_keys + groups["vlm_conf"]
    result = {
        "field_length": base,
        "global_quality": global_keys,
        "global_plus_vlm_conf": all_baselines,
        "field_text_only": groups["field_evidence"][:4],
        "global_plus_vlm_and_text": all_baselines + groups["field_evidence"][:4],
        "all_plus_field_evidence": all_baselines + groups["field_evidence"],
    }
    if "second_agreement" in groups:
        result["all_plus_field_and_agreement"] = result["all_plus_field_evidence"] + groups["second_agreement"]
    return result


def matrix(rows, keys):
    return np.asarray([[np.nan if r["features"].get(k) is None else r["features"][k] for k in keys] for r in rows], dtype=float)


def pipeline(c):
    # Same dimension-preserving imputation and regularization across every arm.
    # An all-missing training feature remains zero, never fit on eval pixels.
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=SEED))


def train_one(x, y, indices, c):
    if len(np.unique(y[indices])) < 2:
        return {"constant": float(np.mean(y[indices]))}
    return pipeline(c).fit(x[indices], y[indices])


def predict(model, x):
    return np.full(len(x), model["constant"]) if isinstance(model, dict) else model.predict_proba(x)[:, 1]


def fit(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("Preserve the original fitted models/scores; choose a fresh output directory")
    rows = read_records(args.features)
    truth = read_records(args.train_truth)
    assert len(truth) == 64 and all(r["split"] == "train" for r in truth)
    truth_index = {r["image"]: r for r in truth}
    assert len(truth_index) == 64
    assert len(rows) == 288 and len({(r["image"], r["field"]) for r in rows}) == 288
    assert {r["field"] for r in rows} == {"product", "brand", "variety"}
    assert len({r["image"] for r in rows}) == 96
    assert all(r["prediction_status"] != "missing_prediction" for r in rows)
    train, targets = [], []
    for r in rows:
        if r["image"] in truth_index:
            scored = score_text(r["prediction"], truth_index[r["image"]]["fields"][r["field"]])
            if scored["status"] == "EVALUABLE":
                train.append(r)
                targets.append(scored["strict_error"])
    y = np.array(targets)
    image_groups = np.array([r["image"] for r in train])
    folds = list(GroupKFold(n_splits=4).split(np.zeros(len(y)), y, image_groups))
    for a, b in folds:
        assert not set(image_groups[a]) & set(image_groups[b])
    arms = model_groups(rows[0]["groups"])
    assert all(r["groups"] == rows[0]["groups"] for r in rows)
    fitted, training, oof_rows = {}, {}, []
    scored_rows = [{k: r[k] for k in ("image", "field", "prediction", "prediction_status")} | {"risks": {}} for r in rows]
    for arm, keys in arms.items():
        if time.perf_counter() - started > 300:
            raise TimeoutError("Fit integration CPU wall budget reached")
        x = matrix(train, keys)
        candidates, predictions = [], {}
        for c in CS:
            oof = np.empty(len(y))
            for a, b in folds:
                oof[b] = predict(train_one(x, y, a, c), x[b])
            loss = log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7), labels=[0, 1])
            candidates.append({"C": c, "group_cv_log_loss": float(loss)})
            predictions[c] = oof
        best = min(candidates, key=lambda r: (r["group_cv_log_loss"], r["C"]))
        c = best["C"]
        oof = predictions[c]
        # Inclusive threshold may accept more than 80% when there are ties.
        threshold = float(np.sort(oof)[math.ceil(0.8 * len(oof)) - 1])
        model = train_one(x, y, np.arange(len(y)), c)
        fitted[arm] = {"model": model, "keys": keys, "threshold": threshold}
        risk = predict(model, matrix(rows, keys))
        for r, score in zip(scored_rows, risk):
            r["risks"][arm] = float(score)
        training[arm] = {"feature_count": len(keys), "features": keys, "selected_C": c, "candidates": candidates,
                         "train_oof_80pct_threshold": threshold, "train_oof_actual_coverage": float(np.mean(oof <= threshold)),
                         "all_missing_train_features": [k for j, k in enumerate(keys) if np.isnan(x[:, j]).all()]}
        for row, label, score in zip(train, y, oof):
            oof_rows.append({"image": row["image"], "field": row["field"], "arm": arm,
                             "strict_error": int(label), "risk": float(score)})
    args.out.mkdir(parents=True)
    (args.out / "models.pkl").write_bytes(pickle.dumps(fitted))
    write_rows(args.out / "sealed_risks.jsonl", scored_rows)
    write_rows(args.out / "train_oof.jsonl", oof_rows)
    write_json(args.out / "fit_receipt.json", {
        "features_sha256": sha(args.features), "training_truth_sha256": sha(args.train_truth),
        "script_sha256": sha(__file__),
        "scoring_source_sha256": {name: sha(Path(__file__).parent / "evaluation" / name) for name in ("score_fields.py", "build_manifest.py")},
        "train_image_ids": sorted(truth_index), "sealed_risks_sha256": sha(args.out / "sealed_risks.jsonl"),
        "models_sha256": sha(args.out / "models.pkl"), "train_images": len(truth), "train_evaluable_fields": len(y),
        "train_errors": int(y.sum()), "candidate_images": 96, "candidate_fields": len(rows),
        "fit_wall_s": time.perf_counter() - started, "cv": "four image-group held folds; minimum OOF log loss, smallest C on tie",
        "threshold_note": "Chosen using training OOF scores after training-only C selection; not an unbiased training performance estimate",
        "dev_eval_truth_opened": False, "arms": training,
        "folds": [{"train_images": sorted(set(image_groups[a])), "validation_images": sorted(set(image_groups[b]))} for a, b in folds],
    })
    print(json.dumps({"phase": "fit", "train_fields": len(y), "arms": list(arms), "wall_s": time.perf_counter() - started}))


def metrics(rows, arm, label):
    y = [r[label] for r in rows]
    risk = [r["risks"][arm] for r in rows]
    keys = [r["image"] + ":" + r["field"] for r in rows]
    result = ranking_metrics(y, risk, keys)
    n = len(rows)
    if not n:
        return result
    order = sorted(range(n), key=lambda i: (risk[i], hashlib.sha256(keys[i].encode()).hexdigest()))
    coverage = {}
    for fraction in (0.6, 0.8):
        k = math.ceil(n * fraction)
        edge = risk[order[k - 1]]
        lower = [i for i in range(n) if risk[i] < edge]
        tied = [i for i in range(n) if risk[i] == edge]
        expected = sum(y[i] for i in lower) + (k - len(lower)) * sum(y[i] for i in tied) / len(tied)
        coverage[str(fraction)] = {"accepted_fields": k, "candidate_fields": n, "actual_coverage": k / n,
                                   "errors_hash_tiebreak": sum(y[i] for i in order[:k]),
                                   "errors_expected_random_ties": expected, "risk_expected_random_ties": expected / k}
    result["matched_coverage"] = coverage
    return result


def evaluate(args):
    started = time.perf_counter()
    receipt = json.loads((args.fitted / "fit_receipt.json").read_text(encoding="utf-8"))
    assert sha(args.fitted / "sealed_risks.jsonl") == receipt["sealed_risks_sha256"]
    assert sha(args.fitted / "models.pkl") == receipt["models_sha256"]
    # Only now can the separate evaluator open held-out annotations.
    truth = read_records(args.eval_truth)
    assert len(truth) == 32 and all(r["split"] == "dev_eval" for r in truth)
    target = {r["image"]: r for r in truth}
    train_images = set(receipt["train_image_ids"])
    assert not train_images & set(target)
    predictions = read_records(args.fitted / "sealed_risks.jsonl")
    rows = []
    for r in predictions:
        if r["image"] in target:
            score = score_text(r["prediction"], target[r["image"]]["fields"][r["field"]])
            rows.append(r | score | {"required": target[r["image"]]["fields"][r["field"]]["required"]})
    assert len(rows) == 96
    valid = [r for r in rows if r["status"] == "EVALUABLE"]
    arms = list(receipt["arms"])
    results = {}
    for field in ("all", "product", "brand", "variety"):
        selected = [r for r in valid if field == "all" or r["field"] == field]
        result = {"evaluable_fields": len(selected), "empty_outputs": sum(r["prediction_empty"] for r in selected), "arms": {}}
        for arm in arms:
            result["arms"][arm] = {label: metrics(selected, arm, label + "_error") for label in ("strict", "relaxed")}
            threshold = receipt["arms"][arm]["train_oof_80pct_threshold"]
            accepted = [r for r in selected if r["risks"][arm] <= threshold]
            result["arms"][arm]["train_threshold_eval"] = {"threshold": threshold, "accepted_fields": len(accepted),
                "candidate_fields": len(selected), "coverage": len(accepted) / len(selected) if selected else None,
                "strict_errors": sum(r["strict_error"] for r in accepted), "relaxed_errors": sum(r["relaxed_error"] for r in accepted)}
        results[field] = result
    # Paired image-cluster bootstrap; all same-image fields travel together.
    # Descriptive Development uncertainty, never a model-selection procedure.
    by_image = defaultdict(list)
    for r in valid:
        by_image[r["image"]].append(r)
    images = sorted(target)
    rng = np.random.default_rng(SEED)
    pairs = [("all_plus_field_evidence", "global_plus_vlm_conf"), ("all_plus_field_evidence", "field_length"),
             ("all_plus_field_evidence", "global_plus_vlm_and_text")]
    if "all_plus_field_and_agreement" in arms:
        pairs.append(("all_plus_field_and_agreement", "all_plus_field_evidence"))
    bootstrap = {}
    for left, right in pairs:
        differences = defaultdict(list)
        for _ in range(1000):
            sample = [r for image in rng.choice(images, size=len(images), replace=True) for r in by_image[image]]
            a, b = metrics(sample, left, "strict_error"), metrics(sample, right, "strict_error")
            for name in ("auroc_error_positive", "ap_error_positive", "aurc_tie_expected"):
                if a.get(name) is not None and b.get(name) is not None:
                    differences[name].append(a[name] - b[name])
            differences["risk_at_80pct"].append(a["matched_coverage"]["0.8"]["risk_expected_random_ties"] - b["matched_coverage"]["0.8"]["risk_expected_random_ties"])
        bootstrap[left + "_minus_" + right] = {name: {"percentile_95_interval": np.quantile(values, [0.025, 0.975]).tolist(), "valid_resamples": len(values)} for name, values in differences.items()}
    if args.out.exists():
        raise FileExistsError("Preserve original evaluation output")
    args.out.mkdir(parents=True)
    write_rows(args.out / "eval_fields.jsonl", rows)
    write_json(args.out / "metrics.json", {"truth_sha256": sha(args.eval_truth), "sealed_risks_sha256": receipt["sealed_risks_sha256"],
        "eval_images": len(truth), "evaluable_fields": len(valid), "not_evaluable_fields": len(rows) - len(valid),
        "label": "automatic required-span lexical coverage; not semantic or hallucination correctness", "positive_class": "error",
        "groups": results, "paired_image_cluster_bootstrap_1000": bootstrap, "seed": SEED,
        "wall_s": time.perf_counter() - started})
    print(json.dumps({"phase": "evaluate", "eval_fields": len(valid), "wall_s": time.perf_counter() - started}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("fit")
    a.add_argument("--features", type=Path, required=True)
    a.add_argument("--train-truth", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = commands.add_parser("evaluate")
    b.add_argument("--fitted", type=Path, required=True)
    b.add_argument("--eval-truth", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (fit if args.command == "fit" else evaluate)(args)
