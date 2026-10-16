"""CPU-only paired analysis of run_holdout.py user JSON receipts.

Read-only inputs: --run-dir/run.json, input_manifest.json, user_<sha16>.json.
Only NumPy and the standard library are used. No model, image, GPU, or network
access occurs. The run's fixed schema was agreed with its producer before this
implementation. --self-test uses small hand-computable in-memory fixtures.

Primary decision (fixed before results): fold 0, video-macro accuracy difference
negative_only - combined; paired user-cluster percentile CI lower bound >-0.5
percentage points, AND sum of users' post-first build medians decreases >=20%.
Users are sampling clusters; this does not turn video weighting into user
weighting. User-macro and query-micro estimates/intervals are reported separately.
The other 33 users are Development. All 44 are reused training users, not fresh
independent data. A partial/incomplete run can never receive a passing decision.

  python analyze_holdout.py --run-dir RUN --out NEW.json --bootstrap 5000
  python analyze_holdout.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np


ARMS = ("combined", "negative_only")
COMPONENTS = ("selection_s", "clustering_s", "build_total_s")
SEED = 20261016
MARGIN_PP = 0.5
MIN_BUILD_REDUCTION = 0.20
EXPECTED_USERS = 44


def read_json(path):
    data = path.read_bytes()
    return json.loads(data.decode("utf-8-sig")), {
        "file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def finite_nonnegative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def distribution(values):
    if not values:
        return None
    return {"n": len(values), "min": float(min(values)), "p25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "p75": float(np.quantile(values, .75)), "max": float(max(values))}


def saving_fraction(reference, candidate):
    return 1.0 - candidate / reference if reference is not None and reference > 0 and candidate is not None else None


def manifest_users_and_checks(manifest, expected):
    """Check the producer's actual users/objects/context/queries manifest schema."""
    indexed, issues, cache_rows = {}, [], []
    expected_by_user = {e["user"]: e for e in expected}
    for user in manifest.get("users", []):
        try:
            name = str(user["user"])
            if name in indexed:
                raise ValueError("duplicate user")
            objects, queries = user["objects"], user["queries"]
            names = [str(obj["object"]) for obj in objects]
            if len(names) < 2 or len(names) != len(set(names)):
                raise ValueError("duplicate/insufficient objects")
            if any(not isinstance(obj["context"], list) or not obj["context"] for obj in objects):
                raise ValueError("missing object context frame list")
            if len({q["video"] for q in queries}) != len(queries):
                raise ValueError("duplicate query videos")
            for query in queries:
                if str(query["object"]) not in names or len(query["frames"]) != len(query["cache_rows"]):
                    raise ValueError("query target or frame/cache-row alignment invalid")
                if not all(isinstance(r, int) and not isinstance(r, bool) and 0 <= r < 11064
                           for r in query["cache_rows"]):
                    raise ValueError("query cache row outside fixed [0,11064) input")
                cache_rows.extend(query["cache_rows"])
            summary = {"user": name, "fold": int(user["fold"]), "n_objects": len(names),
                       "context_frames": sum(len(obj["context"]) for obj in objects),
                       "query_frames": sum(len(q["frames"]) for q in queries), "n_videos": len(queries)}
            if name not in expected_by_user or any(summary[k] != expected_by_user[name].get(k) for k in summary):
                issues.append(f"manifest vs run.expected_users identity/count/fold differs: {name}")
            indexed[name] = user
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"invalid manifest user: {error}")
    if set(indexed) != set(expected_by_user):
        issues.append("manifest user IDs differ from run.expected_users")
    if len(cache_rows) != len(set(cache_rows)):
        issues.append("manifest repeats query cache rows")
    if len(expected) == EXPECTED_USERS and set(cache_rows) != set(range(11064)):
        issues.append("manifest query cache rows do not exactly cover [0,11064)")
    return indexed, issues


def identity_checks(raw, manifest_user):
    """Exact enrolled-object order and query/target/frame/cache-row identity."""
    if manifest_user is None:
        return ["user missing from validated input manifest"]
    issues = []
    try:
        objects = [str(obj["object"]) for obj in manifest_user["objects"]]
        if raw.get("user") != str(manifest_user["user"]) or raw.get("fold") != int(manifest_user["fold"]):
            issues.append("manifest user/fold identity differs")
        if raw.get("objects") != objects:
            issues.append("manifest enrolled object names/order differ")
        produced, declared = raw.get("predictions", []), manifest_user["queries"]
        if not isinstance(produced, list) or [q.get("video") for q in produced] != [q["video"] for q in declared]:
            return issues + ["manifest query-video identities/order differ"]
        for actual, query in zip(produced, declared):
            for key in ("frames", "cache_rows"):
                if actual.get(key) != query[key]:
                    issues.append(f"manifest {key} exact identity differs: {query['video']}")
            expected_target = objects.index(str(query["object"]))
            if actual.get("target_index") != expected_target:
                issues.append(f"manifest scorer target identity differs: {query['video']}")
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"unable to verify manifest identity: {error}")
    return issues


def decision_status(counts_complete, primary_complete, timing_complete, integrity_complete, ni_met, speed_met):
    if not counts_complete or not primary_complete or not timing_complete:
        return "partial_not_decidable"
    if not integrity_complete:
        return "not_established_integrity_or_consistency_failure"
    if ni_met is True and speed_met is True:
        return "predeclared_development_criteria_met"
    return "not_established"


def timing_user(raw):
    """Recompute three-repeat timing; retain original FE cost even on cache hit."""
    issues, arms = [], {}
    repeats = raw.get("repeats", [])
    if not isinstance(repeats, list):
        repeats = []
    for arm in ARMS:
        rows = [r for r in repeats if isinstance(r, dict) and r.get("arm") == arm]
        ids = [r.get("repeat") for r in rows]
        valid = (len(rows) == 3 and all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
                 and sorted(ids) == [0, 1, 2]
                 and all(finite_nonnegative(r.get(k)) for r in rows for k in COMPONENTS))
        if not valid:
            issues.append(f"{arm}: missing/invalid three-repeat timing")
            arms[arm] = None
            continue
        rows = sorted(rows, key=lambda row: row["repeat"])
        arms[arm] = {
            "first_build_s": {key: float(rows[0][key]) for key in COMPONENTS},
            "post_first_median_s": {key: float(statistics.median(r[key] for r in rows[1:])) for key in COMPONENTS},
            "repeat_build_total_s": [float(r["build_total_s"]) for r in rows],
        }
    original_fe = raw.get("extract_once_s")
    actual_fe = raw.get("actual_extract_this_invocation_s")
    original_fe = float(original_fe) if finite_nonnegative(original_fe) else None
    actual_fe = float(actual_fe) if finite_nonnegative(actual_fe) else None
    if original_fe is None:
        issues.append("original context feature extraction elapsed is unknown")
    for arm in ARMS:
        if arms[arm] is not None:
            for phase in ("first_build_s", "post_first_median_s"):
                arms[arm]["original_FE_plus_" + phase + "_decomposition_estimate_s"] = (
                    original_fe + arms[arm][phase]["build_total_s"] if original_fe is not None else None)
    both = all(arms[a] is not None for a in ARMS)
    warm_reduction = (saving_fraction(arms["combined"]["post_first_median_s"]["build_total_s"],
                                     arms["negative_only"]["post_first_median_s"]["build_total_s"])
                      if both else None)
    first_reduction = (saving_fraction(arms["combined"]["first_build_s"]["build_total_s"],
                                      arms["negative_only"]["first_build_s"]["build_total_s"])
                       if both else None)
    return {"status": "complete" if both else "partial_unknown", "issues": issues, "arms": arms,
            "original_context_FE_s": original_fe, "actual_context_FE_this_invocation_s": actual_fe,
            "context_cache": raw.get("context_cache"), "warm_build_reduction_pct":
            (100 * warm_reduction if warm_reduction is not None else None),
            "first_build_reduction_pct": 100 * first_reduction if first_reduction is not None else None,
            "user_wall_s": raw.get("user_wall_s"), "user_gpu_wall_s": raw.get("user_gpu_wall_s")}


def normalize_user(raw, expected, filename, manifest_user=None):
    issues, videos, rows_seen, query_ids = [], [], set(), set()
    user = raw.get("user")
    if user != expected["user"]:
        issues.append("result user differs from expected user")
    wanted_filename = "user_" + hashlib.sha256(str(user).encode()).hexdigest()[:16] + ".json"
    if filename != wanted_filename:
        issues.append("user receipt filename hash does not match user")
    if raw.get("status") != "complete":
        issues.append("producer user status is not complete")
    if raw.get("fold") != expected.get("fold"):
        issues.append("user fold differs from fixed expected fold")
    objects = raw.get("objects", [])
    n_objects = len(objects) if isinstance(objects, list) else 0
    if n_objects < 2 or raw.get("n_objects") != n_objects or n_objects != expected.get("n_objects"):
        issues.append("object counts differ or are missing")
    video_ids = set()
    prediction_records = raw.get("predictions", [])
    if not isinstance(prediction_records, list):
        issues.append("predictions is not a list")
        prediction_records = []
    for rec in prediction_records:
        try:
            video, frames, target = rec["video"], rec["frames"], rec["target_index"]
            cache_rows = rec["cache_rows"]
            if not isinstance(video, str) or video in video_ids:
                raise ValueError("missing/duplicate video identifier")
            if (not isinstance(frames, list) or not frames or not all(isinstance(f, str) for f in frames)
                    or len(set(frames)) != len(frames)):
                raise ValueError("empty/duplicate/invalid query frame IDs")
            if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < n_objects:
                raise ValueError("invalid scorer-only target index")
            if (not isinstance(cache_rows, list) or len(cache_rows) != len(frames)
                    or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in cache_rows)
                    or len(set(cache_rows)) != len(cache_rows) or rows_seen.intersection(cache_rows)):
                raise ValueError("invalid/duplicate cache row identifiers")
            values = {}
            for arm in ARMS:
                pred = rec["predictions"][arm]
                if (len(pred) != len(frames) or not all(isinstance(p, int) and not isinstance(p, bool)
                                                       and 0 <= p < n_objects for p in pred)):
                    raise ValueError(f"invalid predictions for {arm}")
                values[arm] = np.asarray(pred, dtype=np.int64) == target
            combined, negative = values["combined"], values["negative_only"]
            correct = {arm: int(values[arm].sum()) for arm in ARMS}
            saved, harmed = int((negative & ~combined).sum()), int((combined & ~negative).sum())
            videos.append({"user": user, "fold": expected.get("fold"), "n_objects": n_objects, "video": video,
                           "queries": len(frames), "correct": correct, "saved": saved, "harmed": harmed,
                           "both_correct": int((negative & combined).sum()), "both_wrong": int((~negative & ~combined).sum()),
                           "query_ids_sha256": hashlib.sha256(json.dumps(frames, separators=(",", ":")).encode()).hexdigest()})
            video_ids.add(video); rows_seen.update(cache_rows); query_ids.update(frames)
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"invalid video: {error}")
    actual = {"n_videos": len(videos), "query_frames": sum(v["queries"] for v in videos),
              "context_frames": raw.get("context_frames"), "n_objects": n_objects}
    for key in ("n_videos", "query_frames", "context_frames", "n_objects"):
        if actual[key] != expected.get(key):
            issues.append(f"{key}: actual {actual[key]} != expected {expected.get(key)}")
    if raw.get("query_frames") != actual["query_frames"]:
        issues.append("producer query_frames differs from counted predictions")
    per_user_accuracy = {
        arm: {"correct": sum(v["correct"][arm] for v in videos),
              "query_denominator": actual["query_frames"],
              "query_accuracy_pct": (100 * sum(v["correct"][arm] for v in videos) / actual["query_frames"]
                                     if actual["query_frames"] else None),
              "video_macro_accuracy_pct": (100 * float(np.mean([v["correct"][arm] / v["queries"] for v in videos]))
                                            if videos else None)}
        for arm in ARMS
    }
    consistency_issues = [f"{key} is not True" for key in
                          ("same_others", "all_repeat_centers_close", "consistency_pass")
                          if raw.get(key) is not True]
    if raw.get("consistency_status") != "pass":
        consistency_issues.append("producer user consistency_status is not pass")
    return {"user": user, "fold": expected.get("fold"), "n_objects": n_objects,
            "counts_complete": not issues, "issues": issues, "counts": actual, "videos": videos,
            "identity_issues": identity_checks(raw, manifest_user), "consistency_issues": consistency_issues,
            "accuracy": per_user_accuracy,
            "paired_counts": {"saved": sum(v["saved"] for v in videos), "harmed": sum(v["harmed"] for v in videos),
                              "query_denominator": actual["query_frames"]},
            "cache_rows": sorted(rows_seen), "timing": timing_user(raw),
            "config_sha256": raw.get("config_sha256"), "input_hashes": raw.get("input_hashes"),
            "source_hashes": raw.get("source_hashes"), "same_others": raw.get("same_others"),
            "all_repeat_centers_close": raw.get("all_repeat_centers_close"),
            "producer_consistency_pass": raw.get("consistency_pass"),
            "producer_consistency_status": raw.get("consistency_status")}


def bootstrap(users, n_bootstrap, draw_cache):
    """Each row is a whole user; all video/query counts travel with that user."""
    if not users:
        return {key: {"difference_pp": None, "ci95_percentile_pp": None}
                for key in ("query_micro", "video_macro", "user_macro_video")}
    names = tuple(u["user"] for u in users)
    qn = np.asarray([sum(v["queries"] for v in u["videos"]) for u in users], dtype=np.int64)
    vn = np.asarray([len(u["videos"]) for u in users], dtype=np.int64)
    qdiff = np.asarray([sum(v["saved"] - v["harmed"] for v in u["videos"]) for u in users], dtype=float)
    vdiff = np.asarray([sum((v["saved"] - v["harmed"]) / v["queries"] for v in u["videos"])
                        for u in users], dtype=float)
    umean = vdiff / vn
    points = {"query_micro": qdiff.sum() / qn.sum(), "video_macro": vdiff.sum() / vn.sum(),
              "user_macro_video": umean.mean()}
    result = {key: {"difference_pp": 100 * float(value), "ci95_percentile_pp": None}
              for key, value in points.items()}
    if len(users) >= 2:
        key = (n_bootstrap, names)
        if key not in draw_cache:
            draw_cache[key] = np.random.default_rng(SEED).integers(0, len(users), size=(n_bootstrap, len(users)))
        draws = draw_cache[key]
        samples = {"query_micro": qdiff[draws].sum(1) / qn[draws].sum(1),
                   "video_macro": vdiff[draws].sum(1) / vn[draws].sum(1),
                   "user_macro_video": umean[draws].mean(1)}
        for key, values in samples.items():
            result[key]["ci95_percentile_pp"] = (100 * np.quantile(values, [.025, .975])).tolist()
    return result


def aggregate_timing(users, expected_count):
    complete = [u for u in users if u["timing"]["status"] == "complete"]
    all_timed = len(complete) == expected_count
    arms = {}
    for arm in ARMS:
        sums = {phase: {key: sum(u["timing"]["arms"][arm][phase][key] for u in complete)
                        for key in COMPONENTS} for phase in ("first_build_s", "post_first_median_s")}
        arms[arm] = {"sum_available_users_s": sums,
                     "sum_all_expected_users_s": sums if all_timed else None}
    reduction = None
    if complete:
        reduction = saving_fraction(arms["combined"]["sum_available_users_s"]["post_first_median_s"]["build_total_s"],
                                    arms["negative_only"]["sum_available_users_s"]["post_first_median_s"]["build_total_s"])
    fe_users = [u for u in complete if u["timing"]["original_context_FE_s"] is not None]
    fe_sum = sum(u["timing"]["original_context_FE_s"] for u in fe_users)
    estimates = {}
    for arm in ARMS:
        for phase in ("first_build_s", "post_first_median_s"):
            estimates[arm + "_" + phase] = (
                fe_sum + sum(u["timing"]["arms"][arm][phase]["build_total_s"] for u in fe_users)
                if len(fe_users) == expected_count else None)
    cost_reductions = [u["timing"]["warm_build_reduction_pct"] for u in complete
                       if u["timing"]["warm_build_reduction_pct"] is not None]
    actual_fe_known = [u["timing"]["actual_context_FE_this_invocation_s"] for u in users
                       if u["timing"]["actual_context_FE_this_invocation_s"] is not None]
    return {"status": "complete" if all_timed else "partial_unknown", "n_timed_users": len(complete),
            "expected_users": expected_count, "arms": arms,
            "warm_build_reduction_pct_available_subset": 100 * reduction if reduction is not None else None,
            "warm_build_reduction_pct_all_expected_users": 100 * reduction if reduction is not None and all_timed else None,
            "per_user_warm_reduction_pct_distribution": distribution(cost_reductions),
            "n_users_slower": sum(v < 0 for v in cost_reductions),
            "n_users_at_least_20pct_build_reduction": sum(v >= 20 for v in cost_reductions),
            "original_context_FE_s_sum_known": fe_sum, "n_users_original_FE_known": len(fe_users),
            "actual_context_FE_this_invocation_s_sum_known": sum(actual_fe_known),
            "n_users_actual_FE_known": len(actual_fe_known),
            "original_FE_plus_build_decomposition_estimate_s": estimates,
            "original_FE_plus_warm_build_reduction_pct_estimate": (
                100 * saving_fraction(estimates["combined_post_first_median_s"], estimates["negative_only_post_first_median_s"])
                if estimates["combined_post_first_median_s"] is not None
                and estimates["combined_post_first_median_s"] > 0 else None),
            "context_cache_all_hit_users": sum(u["timing"].get("context_cache", {}).get("all_hit") is True
                                                for u in users if isinstance(u["timing"].get("context_cache"), dict)),
            "interpretation": "Sum of per-user medians of repeats 1,2; FE/build sums are decompositions, not measured end-to-end latency"}


def summarize(users, expected, n_bootstrap, draw_cache):
    users = sorted((u for u in users if u["videos"]), key=lambda u: u["user"])
    videos = [v for u in users for v in u["videos"]]
    queries = sum(v["queries"] for v in videos)
    correct = {arm: sum(v["correct"][arm] for v in videos) for arm in ARMS}
    accuracy = {}
    for arm in ARMS:
        accuracy[arm] = {
            "correct": correct[arm], "query_denominator": queries,
            "query_accuracy_pct": 100 * correct[arm] / queries if queries else None,
            "video_macro_accuracy_pct": 100 * float(np.mean([v["correct"][arm] / v["queries"] for v in videos])) if videos else None,
            "user_macro_video_accuracy_pct": 100 * float(np.mean([
                np.mean([v["correct"][arm] / v["queries"] for v in u["videos"]]) for u in users])) if users else None,
        }
    complete = (len(users) == len(expected) and all(u["counts_complete"] for u in users))
    return {"status": "complete" if complete else "partial_available_subset", "expected_users": len(expected),
            "n_users": len(users), "n_videos": len(videos), "n_queries": queries,
            "users": [u["user"] for u in users], "accuracy": accuracy,
            "paired": {"candidate": "negative_only", "reference": "combined",
                       "saved": sum(v["saved"] for v in videos), "harmed": sum(v["harmed"] for v in videos),
                       "net_correct": correct["negative_only"] - correct["combined"],
                       "both_correct": sum(v["both_correct"] for v in videos),
                       "both_wrong": sum(v["both_wrong"] for v in videos),
                       "estimands": bootstrap(users, n_bootstrap, draw_cache)},
            "point_margin_query_count_interpretation": {
                "query_denominator": queries, "half_percentage_point_queries_real": .005 * queries,
                "largest_integer_net_query_loss_within_point_margin": math.floor(.005 * queries + 1e-12),
                "note": "Query-micro intuition only; not equivalent to the primary video-macro cluster-CI decision"},
            "timing": aggregate_timing(users, len(expected))}


def self_test():
    def video(n, combined, negative, saved, harmed):
        return {"queries": n, "correct": {"combined": combined, "negative_only": negative},
                "saved": saved, "harmed": harmed, "both_correct": combined - harmed,
                "both_wrong": n - combined - saved}
    users = [{"user": "A", "videos": [video(2, 0, 2, 2, 0), video(2, 0, 2, 2, 0)]},
             {"user": "B", "videos": [video(4, 4, 0, 0, 4)]}]
    got = bootstrap(users, 5000, {})
    assert got["query_micro"]["difference_pp"] == 0
    assert abs(got["video_macro"]["difference_pp"] - 100 / 3) < 1e-12
    assert got["user_macro_video"]["difference_pp"] == 0
    assert got["video_macro"]["ci95_percentile_pp"] == [-100.0, 100.0]
    repeats = [{"arm": a, "repeat": i, "selection_s": value / 2,
                "clustering_s": value / 4, "build_total_s": value}
               for a, vals in (("combined", [30, 12, 8]), ("negative_only", [20, 7, 5]))
               for i, value in enumerate(vals)]
    timed = timing_user({"repeats": repeats, "extract_once_s": 10,
                         "actual_extract_this_invocation_s": 0, "context_cache": {"all_hit": True}})
    assert timed["arms"]["combined"]["post_first_median_s"]["build_total_s"] == 10
    assert timed["arms"]["negative_only"]["post_first_median_s"]["build_total_s"] == 6
    assert abs(timed["warm_build_reduction_pct"] - 40) < 1e-12
    assert timed["arms"]["negative_only"]["original_FE_plus_post_first_median_s_decomposition_estimate_s"] == 16
    assert math.floor(.005 * 11064) == 55
    assert not (-.5 > -MARGIN_PP) and (-.499 > -MARGIN_PP)
    assert decision_status(True, True, True, False, True, True) == "not_established_integrity_or_consistency_failure"
    assert decision_status(False, True, True, True, True, True) == "partial_not_decidable"
    assert decision_status(True, True, True, True, True, True) == "predeclared_development_criteria_met"
    manifest_user = {"user": "A", "fold": 0, "objects": [{"object": "one", "context": ["c1"]},
                     {"object": "two", "context": ["c2"]}],
                     "queries": [{"video": "v", "object": "one", "frames": ["q"], "cache_rows": [0]}]}
    raw = {"user": "A", "fold": 0, "objects": ["one", "two"],
           "predictions": [{"video": "v", "frames": ["q"], "cache_rows": [0], "target_index": 0}]}
    assert identity_checks(raw, manifest_user) == []
    for key, wrong in (("target_index", 1), ("frames", ["changed"]), ("cache_rows", [1])):
        bad = json.loads(json.dumps(raw)); bad["predictions"][0][key] = wrong
        assert identity_checks(bad, manifest_user)
    bad = json.loads(json.dumps(raw)); bad["objects"].reverse()
    assert identity_checks(bad, manifest_user)
    print("PASS: paired counts/weighting, user bootstrap, warm medians, FE accounting, strict NI, consistency exclusion, manifest identity")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.run_dir is None or args.out is None or args.bootstrap < 1:
        parser.error("--run-dir, --out and a positive --bootstrap are required")
    run_dir, out = args.run_dir.resolve(), args.out.resolve()
    if out.exists():
        parser.error("--out already exists; choose a new result path")
    if out == run_dir / "run.json" or out.name.startswith("user_") or out == run_dir / "input_manifest.json":
        parser.error("--out must not replace or resemble producer input receipts")
    started, wall, cpu = datetime.now(timezone.utc).isoformat(), time.perf_counter(), time.process_time()
    run, run_receipt = read_json(run_dir / "run.json")
    receipts, issues, integrity_issues = [run_receipt], [], []
    expected = run.get("expected_users", run.get("config", {}).get("expected_users", []))
    if not isinstance(expected, list) or not all(isinstance(e, dict) and isinstance(e.get("user"), str) for e in expected):
        parser.error("run.json lacks the fixed expected_users schema")
    expected_by_user = {e["user"]: e for e in expected}
    if len(expected) != EXPECTED_USERS or len(expected_by_user) != EXPECTED_USERS:
        issues.append("expected-user list is not 44 unique users")
    for e in expected:
        for key in ("fold", "n_objects", "context_frames", "query_frames", "n_videos"):
            value = e.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                issues.append(f"expected user {e['user']} has missing/invalid {key}")
    shuffled = sorted(expected_by_user)
    np.random.default_rng(0).shuffle(shuffled)
    recomputed_fold = {u: i % 4 for i, u in enumerate(shuffled)}
    if any(e.get("fold") != recomputed_fold[e["user"]] for e in expected):
        issues.append("manifest fold assignment differs from sorted users/default_rng(0)/modulo4")
    config = run.get("config", {})
    if config.get("expected_users") != expected:
        integrity_issues.append("config.expected_users differs from root expected_users")
    computed_config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                                   ensure_ascii=False).encode()).hexdigest()
    if computed_config_sha != run.get("config_sha256"):
        integrity_issues.append("run config SHA does not match the exact fixed config")
    manifest_path = run_dir / "input_manifest.json"
    manifest_by_user = {}
    manifest_identity_issues = []
    if manifest_path.is_file():
        manifest, receipt = read_json(manifest_path); receipts.append(receipt)
        manifest_by_user, manifest_identity_issues = manifest_users_and_checks(manifest, expected)
        if receipt["sha256"] != config.get("input_hashes", {}).get("manifest_sha256"):
            manifest_identity_issues.append("input_manifest bytes SHA differs from fixed config input hash")
        declared_query_sha = manifest.get("query_cache", {}).get("features_sha256")
        if (not isinstance(declared_query_sha, str)
                or declared_query_sha.lower() != config.get("input_hashes", {}).get("query_features_sha256")):
            manifest_identity_issues.append("manifest query feature SHA differs from producer's actual-file hash receipt")
    else:
        manifest_identity_issues.append("input_manifest.json missing")
    integrity_issues.extend(manifest_identity_issues)
    users, seen = [], set()
    for path in sorted(run_dir.glob("user_*.json")):
        raw, receipt = read_json(path); receipts.append(receipt)
        user = raw.get("user")
        if user not in expected_by_user or user in seen:
            issues.append(f"unexpected/duplicate user receipt: {path.name}"); continue
        seen.add(user)
        normalized = normalize_user(raw, expected_by_user[user], path.name, manifest_by_user.get(user))
        if raw.get("config_sha256") != run.get("config_sha256"):
            normalized["identity_issues"].append("config SHA differs from run.json")
        for key in ("source_hashes", "input_hashes"):
            if not isinstance(config.get(key), dict) or not config[key] or raw.get(key) != config[key]:
                normalized["identity_issues"].append(f"{key} differs from or is missing in fixed run config")
        users.append(normalized)
    all_rows = [row for user in users for row in user["cache_rows"]]
    if len(all_rows) != len(set(all_rows)):
        issues.append("cache row IDs repeat across users")
    missing_users = sorted(set(expected_by_user) - seen)
    if missing_users:
        issues.append("some expected users are missing")
    producer_complete = run.get("all_counts_complete") is True
    counted_complete = (producer_complete and not issues and len(users) == EXPECTED_USERS
                        and all(u["counts_complete"] for u in users))
    producer_consistency_issues = [f"producer {key} is not True" for key in
                                   ("same_others", "all_repeat_centers_close", "consistency_pass", "conditional_pass_eligible")
                                   if run.get(key) is not True]
    if run.get("status") != "complete":
        producer_consistency_issues.append(f"producer status is {run.get('status')!r}, not complete")
    consistency_complete = (len(users) == EXPECTED_USERS and not producer_consistency_issues
                            and all(not u["consistency_issues"] for u in users))
    identity_complete = (len(users) == EXPECTED_USERS and not integrity_issues
                         and all(not u["identity_issues"] for u in users))
    integrity_complete = consistency_complete and identity_complete
    draw_cache = {}
    scopes = {}
    definitions = {"primary_fold0": lambda fold: fold == 0,
                   "development_folds1to3": lambda fold: fold in (1, 2, 3), "all44_development": lambda fold: True}
    definitions.update({f"fold{f}": (lambda fold, f=f: fold == f) for f in range(4)})
    for name, predicate in definitions.items():
        scopes[name] = summarize([u for u in users if predicate(u["fold"])],
                                 [e for e in expected if predicate(e.get("fold"))], args.bootstrap, draw_cache)
    groups = {"objects_le4": lambda n: n <= 4, "objects_5to7": lambda n: 5 <= n <= 7,
              "objects_ge8": lambda n: n >= 8}
    object_groups = {name: summarize([u for u in users if predicate(u["n_objects"])],
                                     [e for e in expected if isinstance(e.get("n_objects"), int)
                                      and predicate(e["n_objects"])], args.bootstrap, draw_cache)
                     for name, predicate in groups.items()}
    primary = scopes["primary_fold0"]
    ci = primary["paired"]["estimands"]["video_macro"]["ci95_percentile_pp"]
    ni_met = ci[0] > -MARGIN_PP if ci is not None else None
    saving = primary["timing"]["warm_build_reduction_pct_all_expected_users"]
    speed_met = saving >= 100 * MIN_BUILD_REDUCTION if saving is not None else None
    chosen_decision = decision_status(counted_complete, primary["status"] == "complete",
                                      primary["timing"]["status"] == "complete", integrity_complete, ni_met, speed_met)
    report = {
        "schema": "negative_train_holdout_analysis_v1", "started_at_utc": started,
        "status": ("partial" if not counted_complete else
                   "complete" if integrity_complete else "counts_complete_integrity_not_established"),
        "all_counts_complete": counted_complete,
        "producer_all_counts_complete": producer_complete, "issues": issues, "missing_users": missing_users,
        "integrity": {"identity_complete": identity_complete, "consistency_complete": consistency_complete,
                      "conditional_pass_eligible": counted_complete and integrity_complete,
                      "identity_issues": integrity_issues, "producer_consistency_issues": producer_consistency_issues,
                      "manifest_identity_issues": manifest_identity_issues,
                      "per_user_identity_issues": {u["user"]: u["identity_issues"] for u in users if u["identity_issues"]},
                      "per_user_consistency_issues": {u["user"]: u["consistency_issues"] for u in users if u["consistency_issues"]},
                      "query_feature_hash_check_scope": "compare declared/producer SHA receipts only; analyzer does not reread feature tensors"},
        "run_dir": str(run_dir), "config_sha256": run.get("config_sha256"),
        "fixed_config": run.get("config"), "input_receipts": receipts,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_users": expected, "recomputed_seed0_fold_of_user": recomputed_fold,
        "counts": {"n_users": len(users), "n_videos": sum(len(u["videos"]) for u in users),
                   "n_queries": sum(v["queries"] for u in users for v in u["videos"]),
                   "unique_cache_rows": len(set(all_rows)), "expected_users": EXPECTED_USERS},
        "decision": {"status": chosen_decision, "primary_scope": "primary_fold0",
                     "primary_accuracy_estimand": "video_macro difference; users are bootstrap clusters",
                     "noninferiority_margin_pp": MARGIN_PP, "strict_ci_lower_bound_rule": "lower > -0.5 pp",
                     "ni_criterion_met": ni_met, "build_reduction_required_pct": 100 * MIN_BUILD_REDUCTION,
                     "build_criterion_met": speed_met,
                     "integrity_and_consistency_eligible": integrity_complete,
                     "interpretation": "Partial data, mismatched manifest identity or failed/unknown consistency cannot pass even when both numerical criteria are met"},
        "scopes": scopes, "object_count_groups_all44_development": object_groups,
        "per_user": [{k: v for k, v in u.items() if k != "cache_rows"} for u in sorted(users, key=lambda u: u["user"])],
        "bootstrap": {"n": args.bootstrap, "seed": SEED, "unit": "paired whole-user clusters",
                      "interval": "percentile 2.5%,97.5%; no multiplicity correction",
                      "draws": "shared across estimands within a fixed user set; all paired videos/queries travel together"},
        "units": {"accuracy": "percent", "differences_and_CI": "percentage_points", "timing": "seconds"},
        "limits": [
            "All 44 ORBIT training users and their query caches were already reusable Development; no fresh independent test is claimed.",
            "The preselected parameters came from the previous six validation users; this script does not fit or select a method.",
            "Query accuracy, video-macro accuracy and user-macro video accuracy have different denominators and are not interchangeable.",
            "Half-point query-loss counts are only an intuition for query micro, not the primary video-macro noninferiority criterion.",
            "Warm cost is the sum of each user's median of repeat 1 and 2, not a median of the pooled build list.",
            "Cached context features do not make their original extraction free: original FE plus build is a decomposition estimate, not end-to-end timing.",
            "Missing receipts, fields or counters remain partial/unknown; no missing prediction is turned into a classification error.",
            "Counts can be complete while consistency or identity fails; descriptive scores then do not establish the registered decision.",
        ],
        "producer_invocations": run.get("invocations"),
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                        "processes": 1, "numpy_thread_environment": 1, "gpu_used_by_analyzer": False},
    }
    report["cost"] = {"wall_s_before_write": time.perf_counter() - wall, "cpu_s_before_write": time.process_time() - cpu,
                      "input_json_bytes": sum(r["bytes"] for r in receipts), "new_inference_frames": 0, "new_image_reads": 0}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(f"{report['status']}: {len(users)}/44 users; {report['counts']['n_queries']} queries; {chosen_decision}")
    for name in ("primary_fold0", "all44_development"):
        result = scopes[name]; paired = result["paired"]; estimate = paired["estimands"]["video_macro"]
        print(f"{name}: {result['n_users']} users/{result['n_queries']} queries; saved/harmed "
              f"{paired['saved']}/{paired['harmed']}; video delta {estimate['difference_pp']} pp, "
              f"CI {estimate['ci95_percentile_pp']}; warm build reduction "
              f"{result['timing']['warm_build_reduction_pct_all_expected_users']}%")
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
