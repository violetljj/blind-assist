"""Fixed own-pool approximation: 33-user screen, then frozen 11-user holdout.

All inputs are existing consumed training-user caches; no feature extraction.
Screen shares other-similarity work across K=2048/4096/8192 and is NOT a
standalone performance measurement. Select pooled-video accuracy, exact ties
favor smaller K. Holdout compares the chosen K with the original K<=30000
using three independently constructed memories per path and predictions from
every repeat. Primary accuracy is each video's mean accuracy over three builds,
not voting or selecting a repeat. Approximate top6-dropmax is not unbiased.
Original source, main results, tolerances, and their failed gates stay unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
import types

sys.dont_write_bytecode = True
SUPPORT_SHA = "6966d515a6ed4a0e2e5056c667ec5cc82306db12dac599541b710619f51979b5"
BENCHMARK_SHA = "4f85b862305ab53738ac9ea3355895c543eec621c2595065bb30306530b71720"
HOLDOUT_SHA = "ef2ac78a74a6f93db0ecc1c583943914caba83a1cde87dcf91952da881e61b8c"
KS = (2048, 4096, 8192)
ARMS = ("combined", "candidate")
GPU_BUDGET_S = 3600.0
GIB = 2 ** 30


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def key(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def metrics(predictions):
    arms = tuple(predictions[0]["predictions"])
    result = {}
    for arm in arms:
        counts, n_frames, fractions = 0, 0, []
        for video in predictions:
            values = video["predictions"][arm]
            if len(values) != len(video["frames"]):
                raise ValueError("frame/prediction denominator mismatch")
            correct = sum(p == video["target_index"] for p in values)
            counts += correct
            n_frames += len(values)
            fractions.append(Fraction(correct, len(values)))
        exact = sum(fractions, Fraction(0)) / len(fractions)
        result[arm] = {"correct": counts, "frames": n_frames, "videos": len(fractions),
                       "frame_micro_accuracy": counts / n_frames,
                       "video_macro_accuracy": float(exact), "video_macro_exact_ratio": str(exact)}
    return result


def compare(candidate, reference, arms):
    if len(candidate) != len(reference):
        raise ValueError("video denominator mismatch")
    result = {a: {"frames": 0, "different_argmax": 0, "saved": 0, "harmed": 0} for a in arms}
    for current, prior in zip(candidate, reference):
        for field in ("video", "frames", "cache_rows", "target_index"):
            if current[field] != prior[field]:
                raise ValueError("prediction identity changed")
        target = current["target_index"]
        for arm in arms:
            a, b = current["predictions"][arm], prior["predictions"][arm]
            if len(a) != len(b) or len(a) != len(current["frames"]):
                raise ValueError("prediction lengths changed")
            result[arm]["frames"] += len(a)
            result[arm]["different_argmax"] += sum(x != y for x, y in zip(a, b))
            result[arm]["saved"] += sum(x == target and y != target for x, y in zip(a, b))
            result[arm]["harmed"] += sum(x != target and y == target for x, y in zip(a, b))
    return result


def original_comparison(predictions, original, arms):
    reference = [{**v, "predictions": {a: v["predictions"]["combined"] for a in arms}} for v in original]
    return compare(predictions, reference, arms)


def infer_named(H, runner, user, cache, memories):
    # Frozen classifier bytecode, with an isolated copy of its globals so its
    # arm-name list can name3screen heads. The original module is never mutated.
    # Its matmul/max32/top8 and final target attachment execute unchanged.
    namespace = dict(H.infer_cached.__globals__)
    namespace["ARMS"] = tuple(memories)
    function = types.FunctionType(H.infer_cached.__code__, namespace, "infer_cached_named")
    return function(runner, user, cache, memories)


def load_context(runner, user, reference, args, prior, S):
    flat, receipts = [], []
    features = reference["context_cache"]["features"]
    if [f["object"] for f in features] != reference["objects"]:
        raise RuntimeError("cached enrollment order changed")
    root = (args.prior_out / "features").resolve()
    for obj, feature in zip(user["objects"], features):
        runner.guard()
        path = Path(feature["path"]).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("feature path escapes prior owned cache")
        receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (receipt["status"] != "complete" or receipt["config_sha256"] != prior["config_sha256"] or
                receipt["runtime_sha256"] != S.json_hash(prior["runtime"]) or
                receipt["user"] != user["user"] or receipt["object"] != obj["object"] or
                receipt["frames_sha256"] != S.json_hash(obj["context"])):
            raise RuntimeError("cached context provenance mismatch")
        actual = sha256(path)
        if actual != receipt["feature_file_sha256"] or actual != feature["feature_file_sha256"]:
            raise RuntimeError("actual context feature SHA mismatch")
        array = runner.np.load(path, mmap_mode="c", allow_pickle=False)
        if array.dtype != runner.np.float32 or array.shape != (len(obj["context"]), 256, 384):
            raise RuntimeError("context feature dtype/shape changed")
        flat.append(array.reshape(-1, 384))
        receipts.append({"object": obj["object"], "path": str(path), "sha256": actual,
                         "receipt_sha256": sha256(path.with_suffix(".json"))})
    return flat, receipts


def sampling(runner, flat, obj):
    t, np = runner.torch, runner.np
    others_idx = [j for j in range(len(flat)) if j != obj]
    g = t.Generator(device="cpu").manual_seed(obj)
    picks = t.randperm(sum(len(flat[j]) for j in others_idx), generator=g)[:30000].numpy()
    # The full randperm is drawn AFTER others, before taking any prefix. Never
    # randperm(K); candidates are nested prefixes of the original own pool.
    own_full = t.randperm(len(flat[obj]), generator=g)[:30000].numpy()
    sampled, offset = [], 0
    for j in others_idx:
        indices = picks[(picks >= offset) & (picks < offset + len(flat[j]))] - offset
        sampled.append(flat[j][indices])
        offset += len(flat[j])
    return t.from_numpy(np.concatenate(sampled)).cuda(), picks, own_full


def expected_prefixes(runner, flat, chosen_k):
    others, full, prefix = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    rows = []
    t = runner.torch
    for obj, source in enumerate(flat):
        runner.guard()
        g = t.Generator(device="cpu").manual_seed(obj)
        total = sum(len(q) for i, q in enumerate(flat) if i != obj)
        picks = t.randperm(total, generator=g)[:30000].numpy()
        own = t.randperm(len(source), generator=g)[:30000].numpy()
        chosen = own[:chosen_k]
        others.update(picks.tobytes()); full.update(own.tobytes()); prefix.update(chosen.tobytes())
        rows.append({"object_index": obj, "n_patches": len(source), "full_own_count": len(own),
                     "candidate_own_count": len(chosen), "candidate_is_original_pool_prefix": True})
    return {"sampled_others_sha256": others.hexdigest(), "full_own_sha256": full.hexdigest(),
            "candidate_own_sha256": prefix.hexdigest(), "objects": rows,
            "baseline_own_proof": "reconstructed from frozen N.build_one source sampling order, not an instrumented baseline pool dump"}


def screen_build(runner, flat, N):
    t, np = runner.torch, runner.np
    started = N.sync_clock(runner)
    memories = {f"own{k}": [] for k in KS}
    timing = {f"own{k}": {"own_rank_s": 0.0, "clustering_s": 0.0} for k in KS}
    other_hash, own_full_hash = hashlib.sha256(), hashlib.sha256()
    own_hashes = {k: hashlib.sha256() for k in KS}
    retained_hashes = {k: hashlib.sha256() for k in KS}
    shared_seconds, rows = 0.0, []
    for obj, source in enumerate(flat):
        runner.guard()
        stage = N.sync_clock(runner)
        p = t.from_numpy(source).cuda()
        others, picks, own_full = sampling(runner, flat, obj)
        other_scores = np.empty(len(p), np.float32)
        for offset in range(0, len(p), runner.args.chunk):
            runner.guard()
            block = p[offset:offset + runner.args.chunk]
            other_scores[offset:offset + len(block)] = (block @ others.T).amax(1).cpu().numpy()
        del others
        shared_seconds += N.sync_clock(runner) - stage
        other_hash.update(picks.tobytes()); own_full_hash.update(own_full.tobytes())
        objects = {"object_index": obj, "n_patches": len(p), "original_own_pool": len(own_full), "candidates": {}}
        for k in KS:
            runner.guard()
            stage = N.sync_clock(runner)
            own_indices = own_full[:k]
            own = p[t.from_numpy(own_indices).to(p.device)]
            scores = np.empty(len(p), np.float32)
            for offset in range(0, len(p), runner.args.chunk):
                runner.guard()
                block = p[offset:offset + runner.args.chunk]
                scores[offset:offset + len(block)] = (block @ own.T).topk(6, dim=1).values[:, 1:].mean(1).cpu().numpy()
            keep = max(32, math.ceil(.15 * len(p)))
            selected = np.sort(np.argsort(-(scores - other_scores), kind="stable")[:keep])
            kept = p[t.from_numpy(selected).to(p.device)]
            del own, scores
            selected_at = N.sync_clock(runner)
            center = runner.kmeans(kept)
            clustered_at = N.sync_clock(runner)
            memories[f"own{k}"].append(center)
            timing[f"own{k}"]["own_rank_s"] += selected_at - stage
            timing[f"own{k}"]["clustering_s"] += clustered_at - selected_at
            own_hashes[k].update(own_indices.tobytes()); retained_hashes[k].update(selected.tobytes())
            objects["candidates"][str(k)] = {"actual_own_count": len(own_indices), "retained": keep,
                                           "is_original_own_prefix": True}
            del kept
        rows.append(objects)
        del p, block, other_scores
    memory = {a: t.stack(values) for a, values in memories.items()}
    return memory, {"screen_total_wall_s": N.sync_clock(runner) - started,
                    "shared_other_and_sampling_s": shared_seconds, "per_candidate": timing,
                    "sampled_others_sha256": other_hash.hexdigest(), "full_own_sha256": own_full_hash.hexdigest(),
                    "candidate_own_sha256": {str(k): own_hashes[k].hexdigest() for k in KS},
                    "retained_patch_sha256": {str(k): retained_hashes[k].hexdigest() for k in KS},
                    "objects": rows, "cost_scope": "shared-other candidate screening; not standalone speed"}


def candidate_build(runner, flat, chosen_k, N):
    t, np = runner.torch, runner.np
    runner.guard(); t.cuda.reset_peak_memory_stats()
    before = runner.resource()
    started = N.sync_clock(runner)
    selection_s = clustering_s = 0.0
    memories, rows = [], []
    other_hash, own_hash, full_hash, selected_hash = [hashlib.sha256() for _ in range(4)]
    for obj, source in enumerate(flat):
        runner.guard()
        stage = N.sync_clock(runner)
        p = t.from_numpy(source).cuda()
        others, picks, own_full = sampling(runner, flat, obj)
        own_indices = own_full[:chosen_k]
        own = p[t.from_numpy(own_indices).to(p.device)]
        own_scores, other_scores = np.empty(len(p), np.float32), np.empty(len(p), np.float32)
        for offset in range(0, len(p), runner.args.chunk):
            runner.guard()
            block = p[offset:offset + runner.args.chunk]
            own_scores[offset:offset + len(block)] = (block @ own.T).topk(6, dim=1).values[:, 1:].mean(1).cpu().numpy()
            other_scores[offset:offset + len(block)] = (block @ others.T).amax(1).cpu().numpy()
        keep = max(32, math.ceil(.15 * len(p)))
        selected = np.sort(np.argsort(-(own_scores - other_scores), kind="stable")[:keep])
        kept = p[t.from_numpy(selected).to(p.device)]
        del own, others, own_scores, other_scores
        selected_at = N.sync_clock(runner)
        center = runner.kmeans(kept)
        clustered_at = N.sync_clock(runner)
        memories.append(center)
        selection_s += selected_at - stage; clustering_s += clustered_at - selected_at
        other_hash.update(picks.tobytes()); own_hash.update(own_indices.tobytes())
        full_hash.update(own_full.tobytes()); selected_hash.update(selected.tobytes())
        rows.append({"object_index": obj, "support_patches": len(p), "sampled_others": len(picks),
                     "sampled_own": len(own_indices), "original_own_pool": len(own_full),
                     "candidate_is_original_pool_prefix": True, "retained": keep,
                     "selection_s": selected_at - stage, "clustering_s": clustered_at - selected_at})
        del p, kept, block
    memory = t.stack(memories)
    total = N.sync_clock(runner) - started
    return memory, {"arm": "candidate", "own_k": chosen_k, "selection_s": selection_s,
                    "clustering_s": clustering_s, "build_total_s": total,
                    "other_overhead_s": total - selection_s - clustering_s,
                    "resources_before": before, "resources_after": runner.resource(),
                    "sampled_others_sha256": other_hash.hexdigest(), "sampled_own_sha256": own_hash.hexdigest(),
                    "full_own_sha256": full_hash.hexdigest(), "selected_patch_indices_sha256": selected_hash.hexdigest(), "objects": rows}


def base_record(user, reference, config_sha, sources, inputs, original_receipt):
    return {"schema": 1, "status": "complete", "config_sha256": config_sha, "user": user["user"],
            "fold": user["fold"], "objects": reference["objects"], "n_objects": reference["n_objects"],
            "context_frames": reference["context_frames"], "query_frames": reference["query_frames"],
            "n_videos": reference["n_videos"], "source_hashes": sources, "input_hashes": inputs,
            "extract_once_s": reference["extract_once_s"], "actual_extract_this_invocation_s": 0.0,
            "original_reference_receipt": {**original_receipt, "predictions": reference["predictions"]}}


def screen_user(runner, user, reference, args, prior, S, N, H, cache, common):
    started = N.sync_clock(runner)
    runner.progress(user["user"], "approx_screen_start", started, fold=user["fold"])
    flat, receipts = load_context(runner, user, reference, args, prior, S)
    memories, build = screen_build(runner, flat, N)
    query_started = N.sync_clock(runner)
    predictions = infer_named(H, runner, user, cache, memories)
    query_seconds = N.sync_clock(runner) - query_started
    reference_videos = reference["predictions"]
    differences = original_comparison(predictions, reference_videos, tuple(memories))
    for video, original in zip(predictions, reference_videos):
        video["predictions"]["combined_reference"] = original["predictions"]["combined"]
    original_others = {r["sampled_others_sha256"] for r in reference["repeats"]}
    record = {**common, "phase": "screen", "predictions": predictions, "metrics": metrics(predictions),
              "builds": build, "context_feature_receipts": receipts, "history_comparison": differences,
              "others_match_original": build["sampled_others_sha256"] in original_others,
              "own_prefix_identity": True, "cost_scope": "candidate selection, shared other work; no standalone speed claim",
              "query_cache_read_and_all_heads_s": query_seconds, "resources": runner.resource(),
              "user_wall_s": time.perf_counter() - started, "user_gpu_wall_s": time.perf_counter() - started}
    del flat, memories
    runner.torch.cuda.empty_cache()
    runner.progress(user["user"], "approx_screen_complete", started, fold=user["fold"])
    return record


def holdout_user(runner, user, reference, args, prior, S, N, H, cache, common, choice, choice_sha, ordinal):
    started = N.sync_clock(runner)
    runner.progress(user["user"], "approx_holdout_start", started, fold=0)
    flat, receipts = load_context(runner, user, reference, args, prior, S)
    chosen_k = choice["chosen_k"]
    prefixes = expected_prefixes(runner, flat, chosen_k)
    centers = {a: [] for a in ARMS}
    repeats, prediction_rows = [], []
    for repeat in range(3):
        order = ARMS if (repeat + ordinal) % 2 == 0 else tuple(reversed(ARMS))
        memories = {}
        for position, arm in enumerate(order):
            memory, row = (N.build_one(runner, flat, "combined") if arm == "combined"
                           else candidate_build(runner, flat, chosen_k, N))
            row.update({"arm": arm, "repeat": repeat, "position": position, "order": order,
                        "centers_sha256": N.tensor_fingerprint(memory),
                        "matching_other_fingerprint": row["sampled_others_sha256"] == prefixes["sampled_others_sha256"]})
            if arm == "candidate":
                row["own_prefix_identity"] = (row["sampled_own_sha256"] == prefixes["candidate_own_sha256"] and
                                               row["full_own_sha256"] == prefixes["full_own_sha256"])
            else:
                row["own_prefix_identity"] = True
                row["own_prefix_evidence"] = "original full-pool sampling reconstructed from immutable source"
            centers[arm].append(memory.cpu().clone())
            memories[arm] = memory
            repeats.append(row)
        query_started = N.sync_clock(runner)
        predictions = infer_named(H, runner, user, cache, memories)
        prediction_rows.append({"repeat": repeat, "predictions": predictions, "metrics": metrics(predictions),
                                "query_cache_read_and_two_heads_s": N.sync_clock(runner) - query_started,
                                "history_comparison": original_comparison(predictions, reference["predictions"], ARMS)})
        del memories, memory
        runner.progress(user["user"], "approx_holdout_repeat_complete", started, repeat=repeat)
    center_pairs, prediction_pairs = [], []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        row = {"reference_repeat": first, "candidate_repeat": second, "arms": {}}
        for arm in ARMS:
            a, b = centers[arm][second], centers[arm][first]
            row["arms"][arm] = {"allclose": bool(runner.torch.allclose(a, b, rtol=1e-5, atol=1e-6)),
                                 "rtol": 1e-5, "atol": 1e-6, "max_abs_difference": float((a - b).abs().max())}
        center_pairs.append(row)
        prediction_pairs.append({"reference_repeat": first, "candidate_repeat": second,
            "arms": compare(prediction_rows[second]["predictions"], prediction_rows[first]["predictions"], ARMS)})
    mean_video = []
    for index, video in enumerate(prediction_rows[0]["predictions"]):
        values = {arm: statistics.mean(sum(p == video["target_index"] for p in r["predictions"][index]["predictions"][arm]) / len(video["frames"])
                                       for r in prediction_rows) for arm in ARMS}
        mean_video.append({"video": video["video"], "frames": len(video["frames"]), "accuracy": values})
    timing = {}
    for arm in ARMS:
        rows = [r for r in repeats if r["arm"] == arm]
        timing[arm] = {"first_build_s": {k: rows[0][k] for k in ("selection_s", "clustering_s", "build_total_s")},
                       "post_first_median_s": {k: statistics.median(r[k] for r in rows[1:]) for k in ("selection_s", "clustering_s", "build_total_s")}}
    record = {**common, "phase": "holdout", "chosen_k": chosen_k, "choice_sha256": choice_sha,
              "context_feature_receipts": receipts, "predictions_by_repeat": prediction_rows,
              "repeats": repeats, "timing": timing, "own_prefix_proof": prefixes,
              "same_others": all(r["matching_other_fingerprint"] for r in repeats),
              "own_prefix_identity": all(r["own_prefix_identity"] for r in repeats),
              "center_pairwise": center_pairs, "prediction_pairwise": prediction_pairs,
              "per_video_three_build_mean": mean_video,
              "primary_user_video_macro": {a: statistics.mean(v["accuracy"][a] for v in mean_video) for a in ARMS},
              "center_allclose_is_gate": False, "resources": runner.resource(),
              "user_wall_s": time.perf_counter() - started, "user_gpu_wall_s": time.perf_counter() - started}
    del flat, centers
    runner.torch.cuda.empty_cache()
    runner.progress(user["user"], "approx_holdout_complete", started)
    return record


def choose(screen_records):
    scores = {}
    for k in KS:
        arm = f"own{k}"
        values = [Fraction(sum(p == v["target_index"] for p in v["predictions"][arm]), len(v["frames"]))
                  for user in screen_records for v in user["predictions"]]
        scores[k] = sum(values, Fraction(0)) / len(values)
    chosen = max(KS, key=lambda k: (scores[k], -k))
    return chosen, {str(k): {"video_macro_accuracy": float(value), "exact_ratio": str(value)} for k, value in scores.items()}


def complete_valid(record, expected, config_sha, phase):
    if record.get("status") != "complete" or record.get("config_sha256") != config_sha or record.get("phase") != phase:
        return False
    return all(record.get(k) == expected[k] for k in ("user", "fold", "n_objects", "context_frames", "query_frames", "n_videos"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior-out", "query-features", "frozen-source-dir", "prior-run-source", "out"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--phase", choices=("screen", "holdout"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--cap-gb", type=float, default=2.5)
    parser.add_argument("--repeats", type=int, default=3, help="holdout only; screen builds each K once")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if any(getattr(args, k) is None for k in ("prior_out", "query_features", "frozen_source_dir", "prior_run_source", "out", "phase")):
        parser.error("all input/output paths and --phase are required")
    if args.repeats != 3 or not 1 <= args.batch <= 32 or not 1 <= args.chunk <= 2048 or not 0 < args.cap_gb <= 2.5:
        parser.error("fixed repeats3; batch<=32, chunk<=2048, cap<=2.5GiB, all positive")
    if args.out.resolve().is_relative_to(args.prior_out.resolve()):
        raise RuntimeError("new output must be outside prior immutable output")
    import psutil
    if psutil.virtual_memory().available < 2 * GIB:
        raise MemoryError("available RAM below2GiB; immediate stop")
    started = time.perf_counter()
    prior_path = args.prior_out / "run.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if not prior["all_counts_complete"] or not prior["status"].startswith("complete"):
        raise RuntimeError("prior44-user computation is not complete")
    sources = {"runner_sha256": sha256(__file__), "prior_runner_sha256": sha256(args.prior_run_source),
               "support_controls_sha256": sha256(args.frozen_source_dir / "support_controls.py"),
               "negative_memory_benchmark_sha256": sha256(args.frozen_source_dir / "negative_memory_benchmark.py")}
    for name, expected_sha, prior_name in (("prior_runner_sha256", HOLDOUT_SHA, "runner_sha256"),
            ("support_controls_sha256", SUPPORT_SHA, "support_controls_sha256"),
            ("negative_memory_benchmark_sha256", BENCHMARK_SHA, "negative_memory_benchmark_sha256")):
        if sources[name] != expected_sha or sources[name] != prior["config"]["source_hashes"][prior_name]:
            raise RuntimeError("frozen source SHA mismatch")
    for field, actual in (("batch", args.batch), ("matrix_chunk", args.chunk), ("cap_gib", args.cap_gb)):
        if actual != prior["config"][field]:
            raise RuntimeError("new path must preserve prior batching/precision/cap")
    payload = (args.prior_out / "input_manifest.json").read_bytes()
    manifest = json.loads(payload)
    query_sha = sha256(args.query_features)
    inputs = {"query_features_sha256": query_sha, "manifest_sha256": hashlib.sha256(payload).hexdigest(),
              "prior_run_sha256": sha256(prior_path), "prior_config_sha256": prior["config_sha256"]}
    if (query_sha != prior["config"]["input_hashes"]["query_features_sha256"] or
            query_sha != manifest["query_cache"]["features_sha256"].lower() or
            inputs["manifest_sha256"] != prior["config"]["input_hashes"]["manifest_sha256"]):
        raise RuntimeError("actual query/manifest SHA mismatch")
    sys.path.insert(0, str(args.frozen_source_dir.resolve()))
    S, N = importlib.import_module("support_controls"), importlib.import_module("negative_memory_benchmark")
    spec = importlib.util.spec_from_file_location("frozen_prior_holdout_approx", args.prior_run_source)
    H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    users = {}
    for user in manifest["users"]:
        user["user"], user["fold"] = str(user["user"]), int(user["fold"])
        for obj in user["objects"]:
            obj["object"] = str(obj["object"])
        users[user["user"]] = user
    expected_map = {e["user"]: e for e in prior["expected_users"]}
    references, original_receipts = {}, {}
    for user_id in users:
        path = args.prior_out / ("user_" + key(user_id) + ".json")
        reference = json.loads(path.read_text(encoding="utf-8"))
        if not H.completed_valid(reference, expected_map[user_id], prior["config_sha256"]):
            raise RuntimeError("prior user completion/count receipt mismatch")
        if reference["source_hashes"] != prior["config"]["source_hashes"] or reference["input_hashes"] != prior["config"]["input_hashes"]:
            raise RuntimeError("prior user input/source identity mismatch")
        references[user_id] = reference
        original_receipts[user_id] = {"path": str(path), "sha256": sha256(path)}
    expected = {"screen": [e for e in prior["expected_users"] if e["fold"] in (1, 2, 3)],
                "holdout": [e for e in prior["expected_users"] if e["fold"] == 0]}
    if len(expected["screen"]) != 33 or len(expected["holdout"]) != 11:
        raise RuntimeError("registered split must be33screen/11holdout users")
    config = {"schema": 1, "source_hashes": sources, "input_hashes": inputs,
              "original_reference_receipts": original_receipts, "expected_users": expected,
              "candidate_k": KS, "screen_repeats": 1, "holdout_repeats": 3,
              "batch": args.batch, "chunk": args.chunk, "cap_gib": args.cap_gb,
              "own_pool": "min(K,N) prefix of full randperm(N), after others draw, CPU seed=object ordinal",
              "score": "own top6 drop maximum mean minus same othermax; top6 approximation is not unbiased",
              "keep": .15, "parts": 32, "query_topk": 8, "kmeans_iters": 15, "kmeans_seed": 0,
              "screen_choice": "highest pooled-video accuracy; exact rational ties smallerK",
              "holdout_primary": "per-video arithmetic mean of3build accuracies, then user-cluster inference; no voting/best-repeat selection",
              "center_allclose_is_gate": False, "tolerance": {"rtol": 1e-5, "atol": 1e-6},
              "classifier_binding": "frozen H.infer_cached code object with copied globals; only ARMS replaced by current head names; original module unchanged; targets attached after all predictions",
              "gpu_budget_s": GPU_BUDGET_S, "feature_extraction": "none; priorFP32context+existingFP16query castFP32",
              "query_metadata_validation": "metadata file not reread; actual query/context cache byte SHAs verified"}
    config_sha = json_hash(config)
    args.out.mkdir(parents=True, exist_ok=True)
    root_path, choice_path = args.out / "run.json", args.out / "choice.json"
    if root_path.exists():
        run = json.loads(root_path.read_text(encoding="utf-8"))
        if run["config_sha256"] != config_sha or sha256(args.out / "input_manifest.json") != inputs["manifest_sha256"]:
            raise RuntimeError("resume configuration/source/input identity changed")
    else:
        if any(args.out.iterdir()):
            raise RuntimeError("nonempty output lacks its own config receipt")
        run = {"schema": 1, "config_sha256": config_sha, "config": config, "expected_users": expected,
               "all_counts_complete": False, "conditional_eligible": False, "invocations": [],
               "phases": {p: {"status": "not_started", "completed_users": [], "all_counts_complete": False} for p in ("screen", "holdout")}}
        (args.out / "input_manifest.json").write_bytes(payload)
    choice = None; choice_sha = None
    if args.phase == "holdout":
        if not choice_path.exists() or not run.get("choice_sha256"):
            raise RuntimeError("screen choice must be frozen before any holdout candidate prediction")
        choice_sha = sha256(choice_path)
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        if choice_sha != run["choice_sha256"] or choice["config_sha256"] != config_sha:
            raise RuntimeError("frozen screen choice/config identity mismatch")
        for receipt in choice["screen_receipts"]:
            if sha256(args.out / receipt["path"]) != receipt["sha256"]:
                raise RuntimeError("frozen screen receipt changed after choice")
    now = datetime.now(timezone.utc)
    for old in run["invocations"]:
        if old.get("status") == "running":
            old.update({"status": "interrupted_without_final_receipt",
                        "budget_charge_s": min(GPU_BUDGET_S, max(0.0, (now - datetime.fromisoformat(old["started_utc"])).total_seconds())),
                        "budget_charge_kind": "conservative launch-to-resume wall bound"})
    used = sum(v.get("budget_charge_s", v.get("gpu_job_wall_s", 0.0)) for v in run["invocations"])
    invocation = {"phase": args.phase, "status": "running", "started_utc": utc_now(), "prior_gpu_budget_charged_s": used}
    run["invocations"].append(invocation)
    run["phases"][args.phase]["status"] = "running"
    S.save_json(root_path, run)
    runner = None; attempt = None; completed = []
    try:
        import numpy as np
        import torch
        import timm
        cache = np.load(args.query_features, mmap_mode="r", allow_pickle=False)
        if cache.dtype != np.float16 or cache.shape != (11064, 256, 384):
            raise RuntimeError("fixed query cache dtype/shape changed")

        class CachedRunner(S.Runner):
            def __init__(self):
                self.np, self.torch, self.psutil, self.args = np, torch, psutil, args
                self.deadline = time.perf_counter() + GPU_BUDGET_S - used - 3
                self.guard()
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable")
                self.gpu_started = time.perf_counter()
                torch.set_num_threads(4); torch.cuda.set_device(0)
                total = torch.cuda.get_device_properties(0).total_memory
                torch.cuda.set_per_process_memory_fraction(args.cap_gb * GIB / total, device=0)
                torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
                self.info = {"torch": str(torch.__version__), "numpy": np.__version__, "timm": timm.__version__,
                             "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": total}
                if any(value != prior["runtime"][name] for name, value in self.info.items()):
                    raise RuntimeError("prior runtime changed")

            def guard(self):
                super().guard()
                if time.perf_counter() >= self.deadline:
                    raise RuntimeError("cumulative3600sGPU budget reached;3s reserved for release")

        phase_dir = args.out / args.phase
        phase_dir.mkdir(exist_ok=True)
        ordered = sorted(expected[args.phase], key=lambda e: (e["fold"], e["user"]))
        for ordinal, expectation in enumerate(ordered):
            user_id = expectation["user"]; user = users[user_id]
            path = phase_dir / ("user_" + key(user_id) + ".json")
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if not complete_valid(record, expectation, config_sha, args.phase):
                    raise RuntimeError("existing user receipt/config/counts mismatch")
                if args.phase == "holdout" and record["choice_sha256"] != choice_sha:
                    raise RuntimeError("completed holdout uses a different frozen choice")
                print(json.dumps({"user": user_id, "phase": args.phase, "stage": "resume_skip_complete"}), flush=True)
            else:
                if runner is None:
                    if used >= GPU_BUDGET_S - 3:
                        raise RuntimeError("no remaining GPU budget")
                    attempt = time.perf_counter(); runner = CachedRunner()
                    run["runtime"] = runner.info
                    S.save_json(root_path, run)
                common = base_record(user, references[user_id], config_sha, sources, inputs, original_receipts[user_id])
                record = (screen_user(runner, user, references[user_id], args, prior, S, N, H, cache, common)
                          if args.phase == "screen" else
                          holdout_user(runner, user, references[user_id], args, prior, S, N, H, cache, common, choice, choice_sha, ordinal))
                S.save_json(path, record)
            completed.append(record)
            phase = run["phases"][args.phase]
            phase["completed_users"] = [r["user"] for r in completed]
            phase["all_counts_complete"] = len(completed) == len(ordered)
            if runner is not None:
                invocation["gpu_elapsed_checkpoint_s"] = time.perf_counter() - runner.gpu_started
            S.save_json(root_path, run)
        if args.phase == "screen":
            if not all(r["others_match_original"] and r["own_prefix_identity"] for r in completed):
                raise RuntimeError("screen sampling identity check failed; no choice may be frozen")
            chosen_k, scores = choose(completed)
            receipts = [{"user": r["user"], "path": str(Path("screen") / ("user_" + key(r["user"]) + ".json")),
                         "sha256": sha256(phase_dir / ("user_" + key(r["user"]) + ".json"))} for r in completed]
            frozen = {"schema": 1, "chosen_k": chosen_k, "config_sha256": config_sha,
                      "manifest_sha256": inputs["manifest_sha256"], "candidate_scores": scores,
                      "screen_users": len(completed), "screen_videos": sum(r["n_videos"] for r in completed),
                      "screen_receipts": receipts, "selection": config["screen_choice"]}
            if choice_path.exists():
                if json.loads(choice_path.read_text(encoding="utf-8")) != frozen:
                    raise RuntimeError("existing frozen choice differs; never overwrite")
            else:
                S.save_json(choice_path, frozen)
            run["choice_sha256"] = sha256(choice_path); run["chosen_k"] = chosen_k
        run["phases"][args.phase]["status"] = "complete"
        run["all_counts_complete"] = all(p["all_counts_complete"] for p in run["phases"].values())
        if args.phase == "holdout":
            run["conditional_eligible"] = (run["all_counts_complete"] and
                all(r["same_others"] and r["own_prefix_identity"] for r in completed))
        run["status"] = "complete" if run["all_counts_complete"] else "screen_complete_holdout_unopened"
        invocation["status"] = "complete"
    except BaseException as exc:
        invocation.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        run["status"] = "failed"; run["phases"][args.phase]["status"] = "failed"
        raise
    finally:
        if runner is not None:
            runner.torch.cuda.synchronize()
            invocation["gpu_job_wall_s"] = time.perf_counter() - runner.gpu_started
            invocation["resources_at_finish"] = runner.resource()
            runner.torch.cuda.empty_cache()
        else:
            invocation["gpu_job_wall_s"] = time.perf_counter() - attempt if attempt is not None else 0.0
        invocation["budget_charge_s"] = invocation["gpu_job_wall_s"]
        invocation["actual_worker_invocation_wall_s"] = time.perf_counter() - started
        invocation["finished_utc"] = utc_now()
        run["gpu_budget_charged_total_s"] = used + invocation["budget_charge_s"]
        S.save_json(root_path, run)


def self_test():
    predictions = [{"video": "v", "frames": ["a", "b", "c"], "cache_rows": [0, 1, 2], "target_index": 0,
                    "predictions": {"own2048": [0, 1, 0], "own4096": [0, 0, 1], "own8192": [1, 1, 0]}}]
    selected, scores = choose([{"predictions": predictions}])
    assert selected == 2048 and scores["2048"]["exact_ratio"] == "2/3"
    scored = metrics(predictions)
    assert scored["own8192"]["correct"] == 1 and scored["own4096"]["frames"] == 3
    changed = compare(predictions, predictions, tuple(predictions[0]["predictions"]))
    assert all(row["different_argmax"] == row["saved"] == row["harmed"] == 0 for row in changed.values())
    print("stdlib fixture PASS: exact 2/3tie selectsK2048; denominators and identical argmax checks pass")


if __name__ == "__main__":
    main()
