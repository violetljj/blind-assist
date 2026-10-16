"""Independent CPU analysis of the fixed cheap-own-consistency wave.

Reads only JSON receipts under --run-dir. No images, features, Torch, inference,
network or previous analyzer imports. The fixed 33-user screen selects the
highest exact video-macro accuracy among own2048/4096/8192, with smaller K on an
exact tie. The selection is checked against frozen choice.json before reporting
the 11-user fold-0 comparison. All 44 users remain reused Development.

Primary accuracy: arithmetic mean of the three separately built per-video
accuracies, then video macro; users are the paired bootstrap clusters. This is
neither voting nor ensembling, and three repeats are not three independent users.
Repeat 0 and individual repeat results are supplementary. Primary criteria stay
CI lower > -0.5 pp and >=20% reduction in sum of user warm-build medians.
Center allclose differences are diagnostics, not an eligibility gate here.

  python analyze_approx.py --run-dir RUN --out NEW.json --bootstrap 5000
  python analyze_approx.py --self-test
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np


KS = (2048, 4096, 8192)
SEED = 20261016
MARGIN_PP = 0.5
SAVING_PCT = 20.0
ARMS = ("combined", "candidate")
COMPONENTS = ("selection_s", "clustering_s", "build_total_s")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def load_json(path):
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8-sig")), {
        "file": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def user_filename(user):
    return "user_" + hashlib.sha256(user.encode()).hexdigest()[:16] + ".json"


def manifest_index(manifest):
    indexed, issues, rows = {}, [], []
    for user in manifest.get("users", []):
        try:
            name = str(user["user"])
            objects = [str(obj["object"]) for obj in user["objects"]]
            if name in indexed or len(objects) < 2 or len(set(objects)) != len(objects):
                raise ValueError("duplicate user/objects or insufficient objects")
            if user["fold"] not in (0, 1, 2, 3):
                raise ValueError("invalid fixed fold")
            if len({q["video"] for q in user["queries"]}) != len(user["queries"]):
                raise ValueError("duplicate videos")
            for query in user["queries"]:
                if (str(query["object"]) not in objects or not query["frames"]
                        or len(query["frames"]) != len(query["cache_rows"])):
                    raise ValueError("invalid query identity")
                if not all(isinstance(r, int) and not isinstance(r, bool) and 0 <= r < 11064 for r in query["cache_rows"]):
                    raise ValueError("cache row outside registered input")
                rows.extend(query["cache_rows"])
            indexed[name] = user
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"manifest user invalid: {error}")
    if len(indexed) != 44 or len(rows) != 11064 or set(rows) != set(range(11064)):
        issues.append("manifest must contain 44 users and exactly one copy of each of 11064 query rows")
    shuffled = sorted(indexed)
    np.random.default_rng(0).shuffle(shuffled)
    fold_of = {u: i % 4 for i, u in enumerate(shuffled)}
    if any(user["fold"] != fold_of[name] for name, user in indexed.items()):
        issues.append("manifest folds differ from the registered seed-0 four-fold assignment")
    return indexed, issues


def expected_summary(user):
    return {"user": str(user["user"]), "fold": user["fold"], "n_objects": len(user["objects"]),
            "context_frames": sum(len(obj["context"]) for obj in user["objects"]),
            "query_frames": sum(len(q["frames"]) for q in user["queries"]), "n_videos": len(user["queries"])}


def check_user_header(raw, declared):
    issues = []
    expected = expected_summary(declared)
    for key, value in expected.items():
        if raw.get(key) != value:
            issues.append(f"{key} differs from manifest")
    if raw.get("objects") != [str(obj["object"]) for obj in declared["objects"]]:
        issues.append("enrolled object names/order differ from manifest")
    if raw.get("status") != "complete":
        issues.append("producer user status is not complete")
    return issues


def paired_rows(predictions, declared, reference, candidate):
    """Only this scorer reads targets; cached predictions are already finished."""
    issues, result = [], []
    objects = [str(obj["object"]) for obj in declared["objects"]]
    if not isinstance(predictions, list) or [p.get("video") for p in predictions] != [q["video"] for q in declared["queries"]]:
        return [], ["query video identities/order differ from manifest"]
    for produced, query in zip(predictions, declared["queries"]):
        try:
            n, target = len(query["frames"]), objects.index(str(query["object"]))
            for key in ("frames", "cache_rows"):
                if produced.get(key) != query[key]:
                    raise ValueError(f"{key} differs from manifest")
            if produced.get("target_index") != target:
                raise ValueError("scorer target differs from manifest")
            cp, npred = produced["predictions"][reference], produced["predictions"][candidate]
            if (len(cp) != n or len(npred) != n or not all(isinstance(p, int) and not isinstance(p, bool)
                                                         and 0 <= p < len(objects) for p in cp + npred)):
                raise ValueError("prediction length/type/object range invalid")
            cc, nc = sum(p == target for p in cp), sum(p == target for p in npred)
            saved = sum(c != target and v == target for c, v in zip(cp, npred))
            harmed = sum(c == target and v != target for c, v in zip(cp, npred))
            result.append({"video": query["video"], "n": n, "c": cc, "nc": nc,
                           "saved": saved, "harmed": harmed, "replicates": 1,
                           "both_correct": sum(c == target and v == target for c, v in zip(cp, npred)),
                           "both_wrong": sum(c != target and v != target for c, v in zip(cp, npred))})
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{query['video']}: {error}")
    return result, issues


def combine_repeat_users(repeat_users):
    """Average each video's three correctness counts, never vote on predictions."""
    merged = []
    maps = [{u["user"]: u for u in rows} for rows in repeat_users]
    for name in sorted(set.intersection(*(set(m) for m in maps)) if maps else set()):
        sources = [m[name] for m in maps]
        ids = [[v["video"] for v in u["videos"]] for u in sources]
        if not all(v == ids[0] for v in ids):
            continue
        videos = []
        for observations in zip(*(u["videos"] for u in sources)):
            if len({v["n"] for v in observations}) != 1:
                continue
            item = {"video": observations[0]["video"], "n": observations[0]["n"], "replicates": len(observations)}
            for key in ("c", "nc", "saved", "harmed", "both_correct", "both_wrong"):
                item[key] = statistics.mean(v[key] for v in observations)
                item[key + "_total_repeats"] = sum(v[key] for v in observations)
            videos.append(item)
        merged.append({"user": name, "fold": sources[0]["fold"], "n_objects": sources[0]["n_objects"],
                       "videos": videos, "source": "three-build accuracy mean"})
    return merged


def bootstrap(users, draws_count, cache):
    names = tuple(u["user"] for u in users)
    if not users:
        return {key: {"difference_pp": None, "ci95_percentile_pp": None}
                for key in ("query_micro", "video_macro", "user_macro_video")}
    qn = np.asarray([sum(v["n"] for v in u["videos"]) for u in users], dtype=float)
    vn = np.asarray([len(u["videos"]) for u in users], dtype=float)
    qd = np.asarray([sum(v["nc"] - v["c"] for v in u["videos"]) for u in users], dtype=float)
    vd = np.asarray([sum((v["nc"] - v["c"]) / v["n"] for v in u["videos"]) for u in users], dtype=float)
    ud = vd / vn
    points = {"query_micro": qd.sum() / qn.sum(), "video_macro": vd.sum() / vn.sum(), "user_macro_video": ud.mean()}
    result = {key: {"difference_pp": 100 * float(value), "ci95_percentile_pp": None} for key, value in points.items()}
    if len(users) >= 2 and draws_count:
        cache_key = (draws_count, names)
        if cache_key not in cache:
            cache[cache_key] = np.random.default_rng(SEED).integers(0, len(users), size=(draws_count, len(users)))
        indices = cache[cache_key]
        values = {"query_micro": qd[indices].sum(1) / qn[indices].sum(1),
                  "video_macro": vd[indices].sum(1) / vn[indices].sum(1), "user_macro_video": ud[indices].mean(1)}
        for key, scores in values.items():
            result[key]["ci95_percentile_pp"] = (100 * np.quantile(scores, [.025, .975])).tolist()
    return result


def summarize(users, expected_count, draws_count, cache):
    users = sorted([u for u in users if u["videos"]], key=lambda u: u["user"])
    videos = [v for u in users for v in u["videos"]]
    nq = sum(v["n"] for v in videos)
    accuracy = {}
    for arm, key in (("combined", "c"), ("candidate", "nc")):
        correct = sum(v[key] for v in videos)
        accuracy[arm] = {
            "mean_correct_per_build": correct, "unique_query_denominator": nq,
            "query_accuracy_pct": 100 * correct / nq if nq else None,
            "video_macro_accuracy_pct": 100 * statistics.mean(v[key] / v["n"] for v in videos) if videos else None,
            "user_macro_video_accuracy_pct": (100 * statistics.mean(statistics.mean(v[key] / v["n"] for v in u["videos"])
                                              for u in users) if users else None),
            "actual_correct_decisions_across_available_builds": sum(v.get(key + "_total_repeats", v[key]) for v in videos),
            "actual_decision_count_across_available_builds": sum(v["n"] * v["replicates"] for v in videos),
        }
    paired = {"saved_mean_per_build": sum(v["saved"] for v in videos),
              "harmed_mean_per_build": sum(v["harmed"] for v in videos),
              "saved_total_repeat_decisions": sum(v.get("saved_total_repeats", v["saved"]) for v in videos),
              "harmed_total_repeat_decisions": sum(v.get("harmed_total_repeats", v["harmed"]) for v in videos),
              "estimands": bootstrap(users, draws_count, cache)}
    user_details = []
    for u in users:
        n = sum(v["n"] for v in u["videos"])
        delta = 100 * statistics.mean((v["nc"] - v["c"]) / v["n"] for v in u["videos"])
        user_details.append({"user": u["user"], "fold": u["fold"], "n_objects": u["n_objects"], "n_videos": len(u["videos"]),
                             "unique_queries": n, "video_difference_pp": delta,
                             "combined_mean_correct": sum(v["c"] for v in u["videos"]),
                             "candidate_mean_correct": sum(v["nc"] for v in u["videos"]),
                             "saved_mean": sum(v["saved"] for v in u["videos"]), "harmed_mean": sum(v["harmed"] for v in u["videos"])})
    return {"status": "complete" if len(users) == expected_count else "partial_available_subset", "expected_users": expected_count,
            "n_users": len(users), "n_videos": len(videos), "unique_queries": nq, "accuracy": accuracy,
            "paired": paired, "per_user": user_details,
            "positive_negative_tied_users": [sum(u["video_difference_pp"] > 1e-10 for u in user_details),
                                             sum(u["video_difference_pp"] < -1e-10 for u in user_details),
                                             sum(abs(u["video_difference_pp"]) <= 1e-10 for u in user_details)]}


def exact_screen_accuracy(users):
    videos = [v for u in users for v in u["videos"]]
    return sum((Fraction(v["nc"], v["n"]) for v in videos), Fraction(0)) / len(videos) if videos else None


def choose_k(scores):
    return max(KS, key=lambda k: (scores[k], -k))


def warm_timing(raw):
    arms, issues = {}, []
    rows = raw.get("repeats", [])
    if not isinstance(rows, list):
        return {"status": "unknown", "issues": ["timing repeats is not a list"], "arms": {}}
    for arm in ARMS:
        values = [r for r in rows if isinstance(r, dict) and r.get("arm") == arm]
        ids = [r.get("repeat") for r in values]
        if (len(values) != 3 or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
                or sorted(ids) != [0, 1, 2] or not all(nonnegative(r.get(k)) for r in values for k in COMPONENTS)):
            issues.append(f"{arm}: missing/invalid three standalone timing repeats"); continue
        values.sort(key=lambda r: r["repeat"])
        arms[arm] = {"first_s": {k: float(values[0][k]) for k in COMPONENTS},
                     "warm_median_s": {k: float(statistics.median(r[k] for r in values[1:])) for k in COMPONENTS},
                     "repeat_build_total_s": [float(r["build_total_s"]) for r in values]}
    complete = len(arms) == 2
    return {"status": "complete" if complete else "unknown", "issues": issues, "arms": arms}


def aggregate_timing(timings, expected_count):
    known = {u: t for u, t in timings.items() if t["status"] == "complete"}
    result = {"status": "complete" if len(known) == expected_count else "partial", "n_users": len(known),
              "expected_users": expected_count, "arms": {}, "warm_build_reduction_pct": None, "per_user": []}
    for arm in ARMS:
        result["arms"][arm] = {phase: {k: sum(t["arms"][arm][phase][k] for t in known.values()) for k in COMPONENTS}
                                for phase in ("first_s", "warm_median_s")}
    c = result["arms"]["combined"]["warm_median_s"]["build_total_s"]
    n = result["arms"]["candidate"]["warm_median_s"]["build_total_s"]
    if c > 0:
        result["warm_build_reduction_pct_available_users"] = 100 * (1 - n / c)
        if len(known) == expected_count:
            result["warm_build_reduction_pct"] = 100 * (1 - n / c)
    for u, t in sorted(known.items()):
        c = t["arms"]["combined"]["warm_median_s"]["build_total_s"]
        n = t["arms"]["candidate"]["warm_median_s"]["build_total_s"]
        result["per_user"].append({"user": u, "warm_build_reduction_pct": 100 * (1 - n / c) if c > 0 else None, **t})
    result["interpretation"] = "11 fresh held-out users only; sum user medians of repeats 1,2. No all44 fresh timing exists."
    return result


def repeat_prediction_diagnostics(raw, declared):
    rows = raw.get("predictions_by_repeat", [])
    if not isinstance(rows, list) or len(rows) != 3:
        return {"status": "unknown", "issues": ["three prediction repeats missing"]}
    rows = sorted(rows, key=lambda r: r["repeat"])
    if [r["repeat"] for r in rows] != [0, 1, 2]:
        return {"status": "unknown", "issues": ["prediction repeat IDs differ from 0,1,2"]}
    changes = []
    for left, right in itertools.combinations(rows, 2):
        counts = {a: 0 for a in ARMS}; denominator = 0
        for x, y in zip(left["predictions"], right["predictions"]):
            denominator += len(x["frames"])
            for arm in ARMS:
                counts[arm] += sum(p != q for p, q in zip(x["predictions"][arm], y["predictions"][arm]))
        changes.append({"reference_repeat": left["repeat"], "candidate_repeat": right["repeat"],
                        "query_denominator_per_arm": denominator, "different_argmax": counts})
    return {"status": "complete", "user": str(declared["user"]), "pairwise_prediction_changes": changes,
            "center_diagnostics_only": {k: raw[k] for k in raw if "center" in k or "allclose" in k},
            "center_allclose_used_as_gate": False}


def classify_decision(counts_complete, identity_complete, choice_complete, timing_complete, ci, reduction):
    if not counts_complete or not timing_complete:
        return "partial_not_decidable"
    if not identity_complete or not choice_complete:
        return "not_established_integrity_failure"
    if ci is not None and ci[0] > -MARGIN_PP and reduction is not None and reduction >= SAVING_PCT:
        return "predeclared_development_criteria_met"
    return "not_established"


def self_test():
    def video(n, c, nc, saved, harmed):
        return {"video": "v", "n": n, "c": c, "nc": nc, "saved": saved, "harmed": harmed,
                "both_correct": c - harmed, "both_wrong": n - c - saved, "replicates": 1}
    users = [{"user": "A", "videos": [video(2, 0, 2, 2, 0), video(2, 0, 2, 2, 0)]},
             {"user": "B", "videos": [video(4, 4, 0, 0, 4)]}]
    got = bootstrap(users, 5000, {})
    assert got["query_micro"]["difference_pp"] == got["user_macro_video"]["difference_pp"] == 0
    assert abs(got["video_macro"]["difference_pp"] - 100 / 3) < 1e-12
    assert got["video_macro"]["ci95_percentile_pp"] == [-100.0, 100.0]
    repeat_users = [[{"user": "A", "fold": 0, "n_objects": 2, "videos": [v]}]
                    for v in (video(2, 2, 0, 0, 2), video(2, 0, 2, 2, 0), video(2, 2, 2, 0, 0))]
    mean_user = combine_repeat_users(repeat_users)[0]
    mean_video = mean_user["videos"][0]
    assert mean_video["c"] == mean_video["nc"] == 4 / 3
    assert mean_video["c_total_repeats"] == mean_video["nc_total_repeats"] == 4
    assert mean_video["replicates"] == 3
    assert choose_k({2048: Fraction(1, 3), 4096: Fraction(2, 6), 8192: Fraction(3, 9)}) == 2048
    assert choose_k({2048: Fraction(1, 3), 4096: Fraction(1, 2), 8192: Fraction(2, 3)}) == 8192
    declared = {"user": "A", "fold": 0, "objects": [{"object": "one"}, {"object": "two"}],
                "queries": [{"video": "v", "object": "one", "frames": ["q0", "q1"], "cache_rows": [0, 1]}]}
    pred = [{"video": "v", "frames": ["q0", "q1"], "cache_rows": [0, 1], "target_index": 0,
             "predictions": {"combined": [0, 1], "candidate": [0, 0]}}]
    rows, issues = paired_rows(pred, declared, "combined", "candidate")
    assert not issues and rows[0]["saved"] == 1 and rows[0]["harmed"] == 0
    bad = json.loads(json.dumps(pred)); bad[0]["frames"][0] = "wrong"
    assert paired_rows(bad, declared, "combined", "candidate")[1]
    assert classify_decision(True, False, True, True, [0.0, 1.0], 25) == "not_established_integrity_failure"
    assert classify_decision(True, True, True, True, [-.5, 1.0], 25) == "not_established"
    assert classify_decision(True, True, True, True, [-.499, 1.0], 20) == "predeclared_development_criteria_met"
    print("PASS: three-repeat accuracy mean, user-cluster weighting, exact screen/tie rule, identity, strict NI and cost criteria")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference-dir", type=Path, help="Optional relocated old user/run JSON directory; original byte SHAs must match")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.run_dir is None or args.out is None or args.bootstrap < 1:
        parser.error("--run-dir, a new --out, and a positive --bootstrap are required")
    root, out = args.run_dir.resolve(), args.out.resolve()
    if out.exists() or out in (root / "run.json", root / "choice.json", root / "input_manifest.json"):
        parser.error("--out must be a new analysis JSON, not an input receipt")
    if out.parent in (root / "screen", root / "holdout") or out.name.startswith("user_"):
        parser.error("--out must not resemble a producer user receipt")
    started, wall, cpu = datetime.now(timezone.utc).isoformat(), time.perf_counter(), time.process_time()
    receipts, identity_issues, count_issues = [], [], []
    read_cache = {}

    def read(path):
        path = Path(path).resolve()
        if path not in read_cache:
            value, receipt = load_json(path)
            read_cache[path] = (value, receipt)
            receipts.append(receipt)
        return read_cache[path]

    run, _ = read(root / "run.json")
    manifest, manifest_receipt = read(root / "input_manifest.json")
    config = run.get("config", {})
    declared, found_issues = manifest_index(manifest)
    identity_issues.extend(found_issues)
    if json_hash(config) != run.get("config_sha256"):
        identity_issues.append("fixed config bytes/content SHA does not match config_sha256")
    inputs, sources = config.get("input_hashes", {}), config.get("source_hashes", {})
    for key, value in (("candidate_k", list(KS)), ("holdout_repeats", 3), ("screen_repeats", 1),
                       ("keep", .15), ("parts", 32), ("query_topk", 8), ("center_allclose_is_gate", False)):
        if config.get(key) != value:
            identity_issues.append(f"fixed wave config differs: {key}")
    if inputs.get("manifest_sha256") != manifest_receipt["sha256"]:
        identity_issues.append("input manifest byte SHA differs from fixed config")
    if manifest.get("query_cache", {}).get("features_sha256") != inputs.get("query_features_sha256"):
        identity_issues.append("manifest query-feature SHA differs from producer actual-file SHA receipt")
    runner_path = Path(__file__).with_name("run_approx.py")
    if runner_path.is_file():
        actual_runner_sha = hashlib.sha256(runner_path.read_bytes()).hexdigest()
        if actual_runner_sha != sources.get("runner_sha256"):
            identity_issues.append("adjacent run_approx.py actual source SHA differs from fixed run source")
    else:
        actual_runner_sha = None
        identity_issues.append("run_approx.py source missing; actual new-runner SHA cannot be checked")
    expected = run.get("expected_users", {})
    if expected != config.get("expected_users"):
        identity_issues.append("root/config expected-user lists differ")
    expected_ids = {}
    for phase, size in (("screen", 33), ("holdout", 11)):
        entries = expected.get(phase, [])
        expected_ids[phase] = {e["user"] for e in entries}
        if len(entries) != size or len(expected_ids[phase]) != size:
            count_issues.append(f"{phase} expected-user list does not contain {size} distinct users")
        for entry in entries:
            name = entry["user"]
            if name not in declared or entry != expected_summary(declared[name]):
                identity_issues.append(f"expected-user header differs from manifest: {name}")
            elif (phase == "holdout") != (declared[name]["fold"] == 0):
                identity_issues.append(f"user is in wrong fixed phase: {name}")
    if (expected_ids["screen"] & expected_ids["holdout"] or
            expected_ids["screen"] | expected_ids["holdout"] != set(declared)):
        identity_issues.append("screen/holdout do not partition the 44 manifest users")

    def reference_for(raw, user):
        desc = raw.get("original_reference_receipt", {})
        fixed = config.get("original_reference_receipts", {}).get(user, {})
        if any(desc.get(k) != fixed.get(k) for k in ("path", "sha256")) or not fixed:
            identity_issues.append(f"original reference receipt differs from frozen config: {user}")
        path = (args.reference_dir / user_filename(user) if args.reference_dir is not None
                else Path(str(desc.get("path", ""))))
        embedded = desc.get("predictions")
        if not path.is_file():
            identity_issues.append(f"original reference bytes unavailable: {user}; use --reference-dir after relocation")
            return embedded if isinstance(embedded, list) else None
        original, receipt = read(path)
        if receipt["sha256"] != desc.get("sha256") or original.get("predictions") != embedded:
            identity_issues.append(f"embedded historical predictions/original byte SHA mismatch: {user}")
        if original.get("config_sha256") != inputs.get("prior_config_sha256"):
            identity_issues.append(f"historical user config SHA mismatch: {user}")
        old_inputs = original.get("input_hashes", {})
        if any(old_inputs.get(k) != inputs.get(k) for k in ("manifest_sha256", "query_features_sha256")):
            identity_issues.append(f"historical user query/manifest inputs changed: {user}")
        if original.get("source_hashes", {}).get("runner_sha256") != sources.get("prior_runner_sha256"):
            identity_issues.append(f"historical runner source SHA mismatch: {user}")
        prior_run_path = path.parent / "run.json"
        if prior_run_path.is_file():
            prior, prior_receipt = read(prior_run_path)
            if prior_receipt["sha256"] != inputs.get("prior_run_sha256") or prior.get("config_sha256") != inputs.get("prior_config_sha256"):
                identity_issues.append(f"original run bytes/config SHA mismatch: {user}")
        else:
            identity_issues.append(f"original run.json unavailable: {user}")
        return original.get("predictions")

    def header_checks(raw, user, phase):
        found = check_user_header(raw, declared[user])
        if raw.get("phase") != phase:
            found.append("producer phase differs")
        for key, value in (("config_sha256", run.get("config_sha256")), ("input_hashes", inputs), ("source_hashes", sources)):
            if raw.get(key) != value:
                found.append(f"{key} differs from fixed run")
        if raw.get("actual_extract_this_invocation_s") != 0.0:
            found.append("no-new-feature-extraction receipt missing or nonzero")
        identity_issues.extend(f"{phase}/{user}: {issue}" for issue in found)

    def compare_history(predictions, history, arm, user, repeat):
        if history is None:
            identity_issues.append(f"historical predictions missing: {user}")
            return None
        if [v.get("video") for v in predictions] != [v.get("video") for v in history]:
            identity_issues.append(f"historical video order differs: {user}")
            return None
        count, different = 0, 0
        for produced, old in zip(predictions, history):
            for key in ("frames", "cache_rows", "target_index"):
                if produced.get(key) != old.get(key):
                    identity_issues.append(f"historical {key} identity differs: {user}/{old.get('video')}")
            p = produced.get("predictions", {}).get(arm, [])
            q = old.get("predictions", {}).get("combined", [])
            if len(p) != len(q):
                identity_issues.append(f"historical prediction length differs: {user}"); return None
            count += len(p); different += sum(a != b for a, b in zip(p, q))
        return {"user": user, "repeat": repeat, "query_denominator": count, "different_argmax": different}

    screen = {k: [] for k in KS}
    screen_receipts, screen_history, screen_raw = {}, [], {}
    for user in sorted(expected_ids["screen"]):
        path = root / "screen" / user_filename(user)
        if not path.is_file():
            count_issues.append(f"screen user missing: {user}"); continue
        raw, receipt = read(path); screen_receipts[user] = receipt; screen_raw[user] = raw
        header_checks(raw, user, "screen")
        history = reference_for(raw, user)
        history_comparison = compare_history(raw.get("predictions", []), history, "combined_reference", user, 0)
        screen_history.append(history_comparison)
        if history_comparison is None or history_comparison["different_argmax"] != 0:
            identity_issues.append(f"screen historical baseline differs from original predictions: {user}")
        if raw.get("others_match_original") is not True or raw.get("own_prefix_identity") is not True:
            identity_issues.append(f"screen other-pool/prefix identity not true: {user}")
        for k in KS:
            rows, found = paired_rows(raw.get("predictions"), declared[user], "combined_reference", f"own{k}")
            identity_issues.extend(f"screen/{user}/own{k}: {issue}" for issue in found)
            screen[k].append({"user": user, "fold": declared[user]["fold"], "n_objects": len(declared[user]["objects"]),
                              "videos": rows, "source": "new shared-other screen vs verified historical combined"})
            if len(rows) != len(declared[user]["queries"]):
                count_issues.append(f"screen prediction counts incomplete: {user}/own{k}")
    screen_table = {str(k): summarize(screen[k], 33, 0, {}) for k in KS}
    exact_scores = {k: exact_screen_accuracy(screen[k]) for k in KS}
    for k, value in exact_scores.items():
        screen_table[str(k)]["exact_video_accuracy_ratio"] = str(value) if value is not None else None
    independent_k = choose_k(exact_scores) if all(value is not None for value in exact_scores.values()) else None
    choice_path = root / "choice.json"
    choice, choice_receipt = (read(choice_path) if choice_path.is_file() else ({}, None))
    choice_issues = []
    if choice_receipt is None:
        choice_issues.append("choice.json missing")
    else:
        if choice_receipt["sha256"] != run.get("choice_sha256"):
            choice_issues.append("choice byte SHA differs from root freeze receipt")
        if (choice.get("chosen_k") != independent_k or run.get("chosen_k") != independent_k
                or choice.get("config_sha256") != run.get("config_sha256")
                or choice.get("manifest_sha256") != manifest_receipt["sha256"]):
            choice_issues.append("frozen choice differs from exact independent screen selection/config/input")
        if choice.get("screen_users") != 33 or choice.get("screen_videos") != sum(len(u["videos"]) for u in screen[KS[0]]):
            choice_issues.append("frozen screen denominators differ")
        frozen_receipts = choice.get("screen_receipts", [])
        if {x.get("user") for x in frozen_receipts} != expected_ids["screen"]:
            choice_issues.append("frozen screen user receipt set differs")
        for frozen in frozen_receipts:
            user = frozen.get("user")
            if user not in screen_receipts or frozen.get("sha256") != screen_receipts[user]["sha256"]:
                choice_issues.append(f"frozen screen byte SHA differs: {user}")
        for k in KS:
            score = choice.get("candidate_scores", {}).get(str(k), {})
            try:
                if Fraction(score["exact_ratio"]) != exact_scores[k] or abs(score["video_macro_accuracy"] - float(exact_scores[k])) > 1e-12:
                    choice_issues.append(f"frozen candidate score differs: {k}")
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                choice_issues.append(f"frozen candidate score unknown: {k}")
    if any(screen_table[str(k)]["n_users"] != 33 for k in KS):
        choice_issues.append("screen incomplete; chosen K is only provisional")

    repeat_users = [[], [], []]
    timings, diagnostics, holdout_history, holdout_raw = {}, [], [], {}
    for user in sorted(expected_ids["holdout"]):
        path = root / "holdout" / user_filename(user)
        if not path.is_file():
            count_issues.append(f"holdout user missing: {user}"); continue
        raw, _ = read(path); holdout_raw[user] = raw; header_checks(raw, user, "holdout")
        history = reference_for(raw, user)
        if raw.get("chosen_k") != independent_k or choice_receipt is None or raw.get("choice_sha256") != choice_receipt["sha256"]:
            choice_issues.append(f"holdout user uses unknown/different frozen choice: {user}")
        if raw.get("same_others") is not True or raw.get("own_prefix_identity") is not True:
            identity_issues.append(f"holdout other-pool/own-prefix identity not true: {user}")
        prediction_rows = raw.get("predictions_by_repeat", [])
        if not isinstance(prediction_rows, list) or sorted(r.get("repeat", -1) for r in prediction_rows) != [0, 1, 2]:
            count_issues.append(f"holdout prediction repeats incomplete: {user}"); continue
        for rep in sorted(prediction_rows, key=lambda row: row["repeat"]):
            idx = rep["repeat"]
            rows, found = paired_rows(rep.get("predictions"), declared[user], "combined", "candidate")
            identity_issues.extend(f"holdout/{user}/repeat{idx}: {issue}" for issue in found)
            if len(rows) != len(declared[user]["queries"]):
                count_issues.append(f"holdout query counts incomplete: {user}/repeat{idx}")
            repeat_users[idx].append({"user": user, "fold": 0, "n_objects": len(declared[user]["objects"]),
                                      "videos": rows, "source": f"fresh standalone repeat{idx}"})
            holdout_history.append(compare_history(rep.get("predictions", []), history, "combined", user, idx))
        timings[user] = warm_timing(raw)
        timings[user]["original_context_FE_s"] = raw.get("extract_once_s") if nonnegative(raw.get("extract_once_s")) else None
        timings[user]["actual_context_FE_this_invocation_s"] = raw.get("actual_extract_this_invocation_s")
        diag = repeat_prediction_diagnostics(raw, declared[user])
        diagnostics.append(diag)
        if diag["status"] != "complete":
            count_issues.append(f"holdout repeat prediction diagnostics incomplete: {user}")
    mean_users = combine_repeat_users(repeat_users)
    cache = {}
    primary = summarize(mean_users, 11, args.bootstrap, cache)
    if primary["n_users"] != 11 or primary["unique_queries"] != 3192:
        count_issues.append("primary denominator differs from fixed 11 users/3192 distinct queries")
    repeat_summaries = {str(i): summarize(repeat_users[i], 11, 0, {}) for i in range(3)}
    chosen_screen = screen.get(independent_k, [])
    all_users = chosen_screen + mean_users
    all44 = summarize(all_users, 44, args.bootstrap, cache)
    folds = {str(f): summarize([u for u in all_users if u["fold"] == f], 11, args.bootstrap, cache) for f in range(4)}
    group_predicates = {"objects_le4": lambda n: n <= 4, "objects_5to7": lambda n: 5 <= n <= 7, "objects_ge8": lambda n: n >= 8}
    groups = {name: summarize([u for u in all_users if pred(u["n_objects"])],
                              sum(pred(len(u["objects"])) for u in declared.values()), args.bootstrap, cache)
              for name, pred in group_predicates.items()}
    timing = aggregate_timing(timings, 11)
    if len(timings) == 11 and all(nonnegative(t.get("original_context_FE_s")) for t in timings.values()):
        original_fe = sum(t["original_context_FE_s"] for t in timings.values())
        timing["original_context_FE_s_sum"] = original_fe
        timing["original_FE_plus_warm_build_decomposition_estimate_s"] = {
            arm: original_fe + timing["arms"][arm]["warm_median_s"]["build_total_s"] for arm in ARMS}
    else:
        timing["original_context_FE_s_sum"] = None
        timing["original_FE_plus_warm_build_decomposition_estimate_s"] = None
    timing["new_feature_extraction_cost_s_known"] = sum(t["actual_context_FE_this_invocation_s"] for t in timings.values()
                                                       if nonnegative(t.get("actual_context_FE_this_invocation_s")))

    invocations = run.get("invocations", [])
    charges = [i.get("budget_charge_s") for i in invocations]
    charged_sum = sum(charges) if charges and all(nonnegative(c) for c in charges) else None
    reported_charge = run.get("gpu_budget_charged_total_s")
    if charged_sum is None or not nonnegative(reported_charge) or abs(charged_sum - reported_charge) > 1e-6:
        identity_issues.append("complete screen+holdout GPU budget charge is unknown or inconsistent")
    if config.get("gpu_budget_s") != 3600 or (charged_sum is not None and charged_sum > 3600):
        identity_issues.append("registered 3600-second GPU budget differs or was exceeded")
    if {i.get("phase") for i in invocations if i.get("status") == "complete"} != {"screen", "holdout"}:
        identity_issues.append("completed cost receipts for both screen and holdout phases are missing")
    for phase in ("screen", "holdout"):
        phase_row = run.get("phases", {}).get(phase, {})
        if phase_row.get("all_counts_complete") is not True or set(phase_row.get("completed_users", [])) != expected_ids[phase]:
            count_issues.append(f"producer {phase} counts/user completion incomplete")
    counts_complete = (run.get("all_counts_complete") is True and not count_issues
                       and primary["n_users"] == 11 and all44["n_users"] == 44 and all44["unique_queries"] == 11064)
    if run.get("status") != "complete" or run.get("conditional_eligible") is not True:
        identity_issues.append("producer complete/conditional eligibility is not true")
    identity_complete, choice_complete = not identity_issues, not choice_issues
    ci = primary["paired"]["estimands"]["video_macro"]["ci95_percentile_pp"]
    reduction = timing["warm_build_reduction_pct"]
    chosen_decision = classify_decision(counts_complete, identity_complete, choice_complete, timing["status"] == "complete", ci, reduction)
    ranges = {}
    for arm in ARMS:
        values = [repeat_summaries[str(i)]["accuracy"][arm]["mean_correct_per_build"] for i in range(3)]
        ranges[arm] = {"correct_count_by_repeat": values, "correct_count_min_max": [min(values), max(values)],
                       "unique_query_denominator_per_repeat": primary["unique_queries"]}
    differences = [repeat_summaries[str(i)]["paired"]["estimands"]["video_macro"]["difference_pp"] for i in range(3)]
    report = {
        "schema": "cheap_consistency_analysis_v1", "started_at_utc": started,
        "status": "complete" if counts_complete and identity_complete and choice_complete else "partial_or_integrity_not_established",
        "all_counts_complete": counts_complete, "count_issues": count_issues,
        "integrity": {"identity_complete": identity_complete, "identity_issues": identity_issues,
                      "choice_complete": choice_complete, "choice_issues": choice_issues,
                      "actual_new_runner_sha256": actual_runner_sha,
                      "query_tensor_hash_scope": "Producer actual-byte hash receipts compared; analyzer reads JSON only",
                      "center_allclose_used_as_gate": False},
        "config_sha256": run.get("config_sha256"), "fixed_config": config,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "input_receipts": receipts,
        "selection": {"independently_selected_k": independent_k, "frozen_choice": choice,
                      "choice_sha256": choice_receipt["sha256"] if choice_receipt else None,
                      "table_33_screen_users": screen_table,
                      "rule": "highest exact Fraction video-macro accuracy; exact tie takes smaller K; no holdout selection"},
        "primary_fold0_three_build_accuracy_mean": primary, "supplementary_single_repeats": repeat_summaries,
        "repeat_prediction_robustness": {"per_user": diagnostics, "correct_count_ranges": ranges,
                                         "same_repeat_video_difference_pp": differences,
                                         "same_repeat_video_difference_pp_min_max": ([min(differences), max(differences)]
                                             if all(v is not None for v in differences) else None),
                                         "role": "diagnostic only; no best-repeat selection, voting or posthoc threshold repair"},
        "historical_baseline_check": {"screen": screen_history, "fresh_holdout_combined_by_repeat": holdout_history,
                                      "note": "Holdout numerical mismatches are disclosed; fresh matched controls define the comparison"},
        "all44_selected_development_supplement": all44, "per_fold": folds,
        "object_count_groups_all44_development": groups, "fresh_holdout_timing": timing,
        "decision": {"status": chosen_decision, "primary": "fold0 video macro of per-video three-build accuracy means",
                     "margin_pp": MARGIN_PP, "strict_accuracy_rule": "paired user-cluster 95% CI lower > -0.5pp",
                     "warm_build_reduction_required_pct": SAVING_PCT,
                     "accuracy_condition_met": ci[0] > -MARGIN_PP if ci is not None else None,
                     "cost_condition_met": reduction >= SAVING_PCT if reduction is not None else None,
                     "center_allclose_gate": False},
        "bootstrap": {"n": args.bootstrap, "seed": SEED, "unit": "paired whole-user clusters",
                      "interval": "percentile 2.5%,97.5%; repeats averaged before resampling; no multiplicity correction"},
        "producer_cost": {"registered_gpu_budget_s": 3600, "sum_invocation_budget_charge_s": charged_sum,
                          "reported_gpu_charge_s": reported_charge, "invocations": invocations,
                          "scope": "entire new screen and holdout wave, including repeated builds and cache queries"},
        "limits": [
            "All 44 users are reused training Development; selection-biased all44 results are supplementary, not fresh confirmation.",
            "Screen 33 uses verified historical combined predictions and shared other scores; it provides no standalone speed claim.",
            "Only 11 held-out users have new three-repeat standalone timing; no all44 fresh speed result exists.",
            "Three-build mean correctness is fractional per build; actual repeated-decision counts have a different denominator.",
            "All44 mixes single-build screen accuracies with three-build holdout means; each distinct video still has equal weight.",
            "Historical context feature extraction is retained as a decomposition cost; cached execution is not end-to-end personalization timing.",
            "Top6 own-pool truncation uses min(K,N) from the frozen pool prefix and is not claimed to be an unbiased approximation.",
            "Center tolerance differences are diagnostics in this new wave; old-wave failure and its original gate remain unchanged.",
            "Missing or mismatched input identity, frozen choice, costs, counts or source evidence cannot produce a passing decision.",
        ],
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                        "processes": 1, "numpy_threads": 1, "gpu_used_by_analyzer": False},
    }
    report["analyzer_cost"] = {"wall_s_before_write": time.perf_counter() - wall, "cpu_s_before_write": time.process_time() - cpu,
                               "input_json_bytes": sum(r["bytes"] for r in receipts), "new_inference": 0, "new_image_reads": 0}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    print(f"{report['status']}; K={independent_k}; {chosen_decision}")
    print(f"Primary: {primary['n_users']} users/{primary['unique_queries']} distinct queries; "
          f"video delta {primary['paired']['estimands']['video_macro']['difference_pp']}pp, CI {ci}; "
          f"warm build reduction {reduction}%")
    print(f"Single-repeat video deltas: {differences}; complete wave GPU charge: {charged_sum}s /3600s")
    print(f"Identity/choice/count issues: {len(identity_issues)}/{len(choice_issues)}/{len(count_issues)}; JSON: {out}")


if __name__ == "__main__":
    main()
