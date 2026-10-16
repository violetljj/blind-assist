"""Independent, training-only ORBIT cached-score controls (Development replay).

Preferred input is head/train.meta.npz; only S/full/valid/y/user/video are read.
Alternatively pass orbit_dump's train_patch/patch directory; only S/full/y/user
are read from each video NPZ. No X, P, boxes, masks, checkpoint, or GPU is used.

Two fixed strengths of query-patch competition are selected separately in each
held-out-user fold, using only the other three folds' video-macro accuracy.
This is cross-fitted selection, not fresh confirmation or fully nested research.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Set before NumPy import; respect a stricter existing numeric thread limit.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    try:
        _limit = min(4, max(1, int(os.environ.get(_key, "4"))))
    except ValueError:
        _limit = 4
    os.environ[_key] = str(_limit)

import numpy as np


COMBOS = (("competition_0p5_top8", 0.5), ("competition_1p0_top8", 1.0))
SELECTED = "competition_crossfit_top8"
BASELINES = ("top8", "top8_fuse")
BOOTSTRAP_SEED = 20261016


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_receipt(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def zscore(scores):
    """Population SD over this video's valid object slots, as in orbit_temporal."""
    scores = np.asarray(scores, dtype=np.float32)
    return ((scores - scores.mean(axis=1, keepdims=True)) /
            (scores.std(axis=1, keepdims=True) + 1e-6))


def top_mean(scores, k):
    if scores.ndim != 3 or not 1 <= k <= scores.shape[1]:
        raise ValueError("top-k must fit the spatial-token axis")
    return np.partition(scores, scores.shape[1] - k, axis=1)[:, -k:, :].mean(axis=1)


def competing_margin(scores):
    """C[n,t,o] = S[n,t,o] - max_{j != o} S[n,t,j], including exact ties."""
    if scores.shape[2] < 2:
        raise ValueError("competition requires at least two enrolled objects")
    # A loop over <= 13 object slots avoids a [N,T,O,O] expansion.
    other = np.empty_like(scores)
    for obj in range(scores.shape[2]):
        left = scores[:, :, :obj].max(axis=2) if obj else None
        right = scores[:, :, obj + 1:].max(axis=2) if obj + 1 < scores.shape[2] else None
        other[:, :, obj] = right if left is None else left if right is None else np.maximum(left, right)
    return scores - other


def score_arms(scores, full, video):
    """All fixed arms share exactly the same cached similarity tensor."""
    scores = np.asarray(scores, dtype=np.float32)
    full = np.asarray(full, dtype=np.float32)
    if scores.ndim != 3 or full.shape != (scores.shape[0], scores.shape[2]):
        raise ValueError(f"bad score/full shapes in {video}: {scores.shape}, {full.shape}")
    if scores.shape[0] == 0 or scores.shape[1] < 16:
        raise ValueError(f"empty video or fewer than 16 spatial tokens: {video}")
    if not np.isfinite(scores).all() or not np.isfinite(full).all():
        raise ValueError(f"non-finite cached scores in {video}")
    partitioned = np.partition(scores, (scores.shape[1] - 16, scores.shape[1] - 8), axis=1)
    top8 = partitioned[:, -8:, :].mean(axis=1)
    arms = {"full": full, "top8": top8, "top8_fuse": zscore(top8) + zscore(full),
            "maxpool": scores.max(axis=1), "top16": partitioned[:, -16:, :].mean(axis=1)}
    # Same eight locations for every object in each query, sampled without
    # replacement. Per-video derived seeds are stable if input order changes.
    for seed in (0, 1, 2):
        token = f"random8:{seed}:{video}".encode("utf-8")
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
        locations = np.stack([rng.choice(scores.shape[1], size=8, replace=False)
                              for _ in range(len(scores))])
        arms[f"random8_s{seed}"] = scores[np.arange(len(scores))[:, None], locations].mean(axis=1)
    margin = competing_margin(scores)
    arms["margin_top8"] = top_mean(margin, 8)
    for name, strength in COMBOS:
        arms[name] = top_mean(scores + strength * margin, 8)
    return arms


def video_record(video, user, scores, full, labels, query_ids):
    arms = score_arms(scores, full, video)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (len(scores),) or labels.min() < 0 or labels.max() >= scores.shape[2]:
        raise ValueError(f"invalid object labels in {video}")
    return {"video": str(video), "user": str(user), "n_objects": int(scores.shape[2]),
            "query_ids": list(query_ids), "y": labels,
            "predictions": {name: values.argmax(axis=1).astype(np.int64)
                            for name, values in arms.items()}}


def read_packed(path):
    receipt = source_receipt(path)
    # NpzFile loads only requested arrays, so neither the stored mask nor P/X
    # is decoded. S stays in its compact on-disk dtype until a video is sliced.
    with np.load(path, allow_pickle=False) as cache:
        scores = cache["S"]
        full = cache["full"]
        valid = cache["valid"].astype(bool)
        labels = cache["y"]
        users = cache["user"].astype(str)
        videos = cache["video"].astype(str)
    n = len(scores)
    if scores.ndim != 3 or full.shape != (n, scores.shape[2]) or valid.shape != full.shape:
        raise ValueError("invalid packed score/full/valid shapes")
    if any(np.asarray(a).shape != (n,) for a in (labels, users, videos)):
        raise ValueError("packed y/user/video must be one-dimensional arrays of N rows")
    records = []
    for video in sorted(set(videos)):
        idx = np.flatnonzero(videos == video)
        if len(set(users[idx])) != 1 or not np.all(valid[idx] == valid[idx[0]]):
            raise ValueError(f"mixed users or valid object slots within {video}")
        slots = np.flatnonzero(valid[idx[0]])
        mapping = {int(old): new for new, old in enumerate(slots)}
        if any(int(y) not in mapping for y in labels[idx]):
            raise ValueError(f"target label is a padded slot in {video}")
        y = np.array([mapping[int(old)] for old in labels[idx]], dtype=np.int64)
        ids = [f"{video}::cached_row_{j}" for j in range(len(idx))]
        rec = video_record(video, users[idx[0]], scores[idx][:, :, slots], full[idx][:, slots], y, ids)
        rec["source_rows"] = idx.tolist()
        rec["object_slots"] = slots.tolist()
        records.append(rec)
    return records, {"mode": "packed_meta", "arrays_read": ["S", "full", "valid", "y", "user", "video"],
                     "sources": [receipt]}


def read_patch_dir(path):
    path = Path(path)
    files = sorted(p for p in path.glob("*.npz") if not p.name.startswith("_"))
    if not files:
        raise ValueError(f"no per-video patch NPZ files in {path}")
    records, receipts = [], []
    for source in files:
        receipts.append(source_receipt(source))
        with np.load(source, allow_pickle=False) as cache:
            scores, full = cache["S"], cache["full"]
            user, target = str(cache["user"].item()), int(cache["y"].item())
        ids = [f"{source.stem}::cached_row_{j}" for j in range(len(scores))]
        records.append(video_record(source.stem, user, scores, full,
                                    np.full(len(scores), target, dtype=np.int64), ids))
    return records, {"mode": "patch_dir", "arrays_read": ["S", "full", "y", "user"], "sources": receipts}


def hit(record, arm):
    return record["predictions"][arm] == record["y"]


def add_crossfit(records):
    users = sorted({r["user"] for r in records})
    shuffled = users.copy()
    np.random.default_rng(0).shuffle(shuffled)
    fold_of = {user: i % 4 for i, user in enumerate(shuffled)}
    fold_rows = []
    for fold in range(4):
        train = [r for r in records if fold_of[r["user"]] != fold]
        held = [r for r in records if fold_of[r["user"]] == fold]
        if not train or not held:
            raise ValueError("cross-fitting requires at least one user in each of four folds")
        choice_scores = {name: float(np.mean([hit(r, name).mean() for r in train])) for name, _ in COMBOS}
        # COMBOS is ordered by smaller strength, so exact ties choose 0.5.
        chosen = max((name for name, _ in COMBOS), key=lambda name: choice_scores[name])
        for record in held:
            record["predictions"][SELECTED] = record["predictions"][chosen].copy()
            record["selected_arm"] = chosen
            record["fold"] = fold
        fold_rows.append({"heldout_fold": fold, "training_users": sorted({r["user"] for r in train}),
                          "heldout_users": sorted({r["user"] for r in held}),
                          "training_videos": len(train), "heldout_videos": len(held),
                          "selection_metric": "video_macro_accuracy_fraction",
                          "training_candidate_scores": choice_scores, "selected_arm": chosen,
                          "selected_strength": dict(COMBOS)[chosen]})
    return users, fold_of, fold_rows


def interval(values):
    lo, hi = np.quantile(values, [0.025, 0.975])
    return [float(lo), float(hi)]


def analyze(records, bootstrap=3000):
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    users, fold_of, folds = add_crossfit(records)
    arms = list(records[0]["predictions"])
    if any(set(r["predictions"]) != set(arms) for r in records):
        raise ValueError("inconsistent score arms between videos")
    query_ids = [qid for record in records for qid in record["query_ids"]]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate query IDs")
    by_user = {user: [r for r in records if r["user"] == user] for user in users}
    nvideo = np.array([len(by_user[u]) for u in users], dtype=np.float64)
    nquery = np.array([sum(len(r["y"]) for r in by_user[u]) for u in users], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(users), np.full(len(users), 1 / len(users)), size=bootstrap)
    summaries, boot_values = {}, {}
    for arm in arms:
        video_sums = np.array([sum(hit(r, arm).mean() for r in by_user[u]) for u in users])
        correct = np.array([sum(int(hit(r, arm).sum()) for r in by_user[u]) for u in users])
        user_means = video_sums / nvideo
        boot = {"video_macro": (weights @ video_sums) / (weights @ nvideo),
                "user_macro": (weights @ user_means) / len(users),
                "query_micro": (weights @ correct) / (weights @ nquery)}
        boot_values[arm] = boot
        summaries[arm] = {
            "correct_queries": int(correct.sum()), "queries": int(nquery.sum()),
            "query_micro_accuracy": float(correct.sum() / nquery.sum()),
            "video_macro_accuracy": float(video_sums.sum() / nvideo.sum()),
            "user_macro_accuracy": float(user_means.mean()),
            "user_macro_query_accuracy": float(np.mean(correct / nquery)),
            "user_cluster_bootstrap_95ci": {name: interval(value) for name, value in boot.items()},
            "per_user": {user: {"correct_queries": int(correct[i]), "queries": int(nquery[i]),
                                "videos": int(nvideo[i]), "query_accuracy": float(correct[i] / nquery[i]),
                                "video_macro_accuracy": float(user_means[i])}
                         for i, user in enumerate(users)}}
    comparisons = {}
    for baseline in BASELINES:
        comparisons[baseline] = {}
        for arm in arms:
            if arm == baseline:
                continue
            saved, harmed, per_user = [], [], {}
            both_correct = both_wrong = 0
            for user in users:
                usaved = uharmed = 0
                for record in by_user[user]:
                    new, old = hit(record, arm), hit(record, baseline)
                    for i in np.flatnonzero(new & ~old):
                        saved.append(record["query_ids"][i])
                        usaved += 1
                    for i in np.flatnonzero(old & ~new):
                        harmed.append(record["query_ids"][i])
                        uharmed += 1
                    both_correct += int((new & old).sum())
                    both_wrong += int((~new & ~old).sum())
                per_user[user] = {"saved": usaved, "harmed": uharmed, "net": usaved - uharmed}
            delta_metrics = {
                "video_macro": summaries[arm]["video_macro_accuracy"] - summaries[baseline]["video_macro_accuracy"],
                "user_macro": summaries[arm]["user_macro_accuracy"] - summaries[baseline]["user_macro_accuracy"],
                "query_micro": summaries[arm]["query_micro_accuracy"] - summaries[baseline]["query_micro_accuracy"]}
            comparisons[baseline][arm] = {
                "saved": len(saved), "harmed": len(harmed), "net": len(saved) - len(harmed),
                "both_correct": both_correct, "both_wrong": both_wrong, "queries": len(query_ids),
                "delta_accuracy_fraction": delta_metrics,
                "paired_user_bootstrap_95ci_delta": {
                    metric: interval(boot_values[arm][metric] - boot_values[baseline][metric])
                    for metric in boot_values[arm]},
                "per_user": per_user, "saved_query_ids": saved, "harmed_query_ids": harmed}
    per_video = []
    for record in records:
        per_video.append({
            "video": record["video"], "user": record["user"], "fold": record["fold"],
            "n_objects": record["n_objects"], "queries": len(record["y"]),
            "selected_arm": record["selected_arm"],
            "source_rows": record.get("source_rows"), "object_slots": record.get("object_slots"),
            "query_ids": record["query_ids"], "y": record["y"].tolist(),
            "predictions": {arm: record["predictions"][arm].tolist() for arm in arms},
            "correct_queries": {arm: int(hit(record, arm).sum()) for arm in arms},
            "accuracy_fraction": {arm: float(hit(record, arm).mean()) for arm in arms}})
    return {"n_users": len(users), "n_videos": len(records), "n_queries": len(query_ids),
            "users": users, "fold_of_user": fold_of, "fold_selection": folds,
            "arms": summaries, "comparisons_vs": comparisons, "per_video": per_video,
            "bootstrap": {"replicates": bootstrap, "seed": BOOTSTRAP_SEED, "unit": "user",
                          "method": "paired percentile cluster bootstrap; all videos/queries of a sampled user travel together",
                          "selection_refit_within_bootstrap": False,
                          "limit": "conditional intervals for these cross-fitted predictions, not training/selection uncertainty"}}


def self_test():
    # Sixteen patch tokens, two objects: raw top8 prefers object0 (0.9 vs
    # 0.89), but object0's strong patches also match object1 (0.89). Object1's
    # other eight patches match it alone, so competitive top8 prefers object1.
    scores = np.array([[[0.9, 0.89]] * 8 + [[0.0, 0.8]] * 8], dtype=np.float32)
    margin = competing_margin(scores)
    np.testing.assert_allclose(margin[0, 0], [0.01, -0.01], atol=1e-6)
    np.testing.assert_allclose(margin[0, -1], [-0.8, 0.8], atol=1e-6)
    arms = score_arms(scores, np.array([[0.0, 1.0]]), "fixture")
    assert int(arms["top8"].argmax(axis=1)[0]) == 0
    assert int(arms["margin_top8"].argmax(axis=1)[0]) == 1
    assert all(int(arms[name].argmax(axis=1)[0]) == 1 for name, _ in COMBOS)
    np.testing.assert_allclose(arms["top16"], [[0.45, 0.845]], atol=1e-6)
    np.testing.assert_array_equal(competing_margin(np.ones((1, 16, 3), dtype=np.float32)), 0.0)
    again = score_arms(scores, np.array([[0.0, 1.0]]), "fixture")
    for seed in (0, 1, 2):
        np.testing.assert_array_equal(arms[f"random8_s{seed}"], again[f"random8_s{seed}"])
    records = [video_record(f"fixture_v{i}", f"fixture_u{i}", scores,
                            np.array([[0.0, 1.0]]), np.array([1]), [f"q{i}"])
               for i in range(4)]
    report = analyze(records, bootstrap=20)
    assert report["n_queries"] == 4 and report["arms"]["top8"]["correct_queries"] == 0
    assert report["arms"][SELECTED]["correct_queries"] == 4
    comparison = report["comparisons_vs"]["top8"][SELECTED]
    assert comparison["saved"] == 4 and comparison["harmed"] == 0
    assert comparison["paired_user_bootstrap_95ci_delta"]["video_macro"] == [1.0, 1.0]
    for fold in report["fold_selection"]:
        assert not set(fold["training_users"]) & set(fold["heldout_users"])
        assert fold["selected_strength"] == 0.5
    print("SELF_TEST_PASS: hand-calculated margin/top-k, deterministic random8, four-fold selection, paired counts/bootstrap")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-meta", type=Path, help="preferred compact train.meta.npz; overrides --patch-dir")
    parser.add_argument("--patch-dir", type=Path, help="alternative per-video train_patch/patch directory")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.out is None or (args.packed_meta is None and args.patch_dir is None):
        parser.error("--out and either --packed-meta or --patch-dir are required")
    start = time.perf_counter()
    records, provenance = read_packed(args.packed_meta) if args.packed_meta else read_patch_dir(args.patch_dir)
    actual_users = len({r["user"] for r in records})
    if actual_users != 44:
        raise ValueError(f"expected 44 ORBIT training users, found {actual_users}; refusing another cohort")
    report = analyze(records, args.bootstrap)
    report.update({
        "schema": "orbit_independent_score_controls_v1", "status": "COMPLETE",
        "evidence_identity": "Development replay of previously extracted ORBIT training-user score caches",
        "provenance": {**provenance, "script": source_receipt(__file__)},
        "configuration": {
            "competition_formula": "C=S-max(other enrolled objects); score_o=mean(top8_t(S+alpha*C))",
            "fixed_strengths": [0.5, 1.0], "pure_margin_control": "mean(top8_t(C))",
            "fixed_pool_controls": ["maxpool", "top8", "top16", "random8_s0", "random8_s1", "random8_s2"],
            "random8": "eight uniform spatial locations without replacement per query; shared across object slots",
            "random8_seed": "first 8 SHA256 bytes of random8:<seed>:<video>, interpreted little-endian",
            "fusion_formula": "zscore(top8)+zscore(full), population SD over valid object slots, epsilon=1e-6",
            "tie_break": "numpy argmax chooses first valid object slot; selected alpha ties choose 0.5",
            "folds": 4, "fold_seed": 0, "fold_assignment": "sort users, NumPy default_rng(0).shuffle, enumerate modulo 4",
            "selection": "select alpha only on other three folds by video-macro accuracy; no baseline fallback",
            "accuracy_unit": "fraction, not percent; multiply differences by 100 for percentage points",
            "user_macro_definition": "mean over users of that user's video-macro accuracy",
            "user_macro_query_definition": "mean over users of correct queries / cached queries",
            "thread_environment": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
        "limits": [
            "Only cached sampled valid frames are evaluated; this is not the full official 50-task evaluation.",
            "All users were already available for method development; cross-fitting does not make this fresh confirmation.",
            "No test-user data, boxes, masks, X, P, or checkpoint is accessed by this script.",
            "Random8 matches input similarities and pool cardinality, not the computational cost of top-k search.",
            "Cached score replay excludes feature extraction and memory construction cost; runtime is not deployment latency.",
            "All fixed arms are retained even if the new competition mechanisms lose."],
        "wall_seconds": time.perf_counter() - start})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves earlier failed/negative result records.
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    for arm in ("top8", "top8_fuse", "margin_top8", *[name for name, _ in COMBOS], SELECTED):
        value = report["arms"][arm]
        print(f"{arm}: video={100*value['video_macro_accuracy']:.4f}% "
              f"user={100*value['user_macro_accuracy']:.4f}% "
              f"queries={value['correct_queries']}/{value['queries']}")
    print(f"COMPLETE: {args.out.resolve()}")


if __name__ == "__main__":
    main()
