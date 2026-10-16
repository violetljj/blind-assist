"""Fixed negative-only versus combined memory on 44 consumed training users.

Run development folds 1/2/3 first and the already assigned fold 0 last. No
parameter selection. Context uses fresh frozen Runner fp32 normalized features;
queries use the existing float16 train.X.npy, cast to fp32 without renormalizing.
This paired cache replay does not claim exact reproduction of older scores.
Query identity is materialized only after predictions, in the descriptive scorer.
No query images, boxes, masks, online downloads, or additional model are read.

Owned output contains run.json, an exact input_manifest.json copy, user_<hash>.json
receipts, and worker-only features/<userhash>/<objecthash>.npy plus cache receipts.
Complete users skip all model work. Complete object caches have SHA-checked bytes
and matching inputs/model/runtime/source/config; incomplete caches are rebuildable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True

FROZEN_SUPPORT_SHA = "6966d515a6ed4a0e2e5056c667ec5cc82306db12dac599541b710619f51979b5"
FROZEN_BENCHMARK_SHA = "4f85b862305ab53738ac9ea3355895c543eec621c2595065bb30306530b71720"
ARMS = ("combined", "negative_only")
BUDGET_GPU_S = 7200.0
GIB = 2 ** 30


class BudgetStop(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def user_key(user):
    return hashlib.sha256(str(user).encode("utf-8")).hexdigest()[:16]


def score_predictions(predictions):
    """Pure scorer, suitable for a hand-calculated fixture without NumPy/CUDA."""
    total = sum(len(v["frames"]) for v in predictions)
    if not total:
        raise ValueError("no query frames")
    metrics = {}
    saved = harmed = both_correct = both_wrong = 0
    for arm in ARMS:
        counts, acc = [], []
        for video in predictions:
            values = video["predictions"][arm]
            if len(values) != len(video["frames"]):
                raise ValueError("prediction length mismatch")
            correct = sum(p == video["target_index"] for p in values)
            counts.append(correct)
            acc.append(correct / len(values))
        metrics[arm] = {"correct": sum(counts), "frames": total,
                        "frame_micro_accuracy": sum(counts) / total,
                        "video_macro_accuracy": statistics.mean(acc)}
    for video in predictions:
        y = video["target_index"]
        for neg, combined in zip(video["predictions"]["negative_only"], video["predictions"]["combined"]):
            a, b = neg == y, combined == y
            saved += int(a and not b)
            harmed += int(b and not a)
            both_correct += int(a and b)
            both_wrong += int(not a and not b)
    return {"metrics": metrics, "negative_minus_combined": {"saved": saved, "harmed": harmed,
            "both_correct": both_correct, "both_wrong": both_wrong, "frames": total,
            "video_macro_difference_pp": 100 * (metrics["negative_only"]["video_macro_accuracy"] -
                                                  metrics["combined"]["video_macro_accuracy"])}}


def read_inputs(args):
    payload = args.manifest.read_bytes()
    manifest = json.loads(payload)
    users = manifest["users"]
    if len(users) != 44 or len({str(u["user"]) for u in users}) != 44:
        raise ValueError("this registered replay requires exactly 44 distinct training users")
    names, expected, seen_rows = set(), [], set()
    for user in users:
        user["user"] = str(user["user"])
        user["fold"] = int(user["fold"])
        if user["fold"] not in (0, 1, 2, 3):
            raise ValueError("fold must be 0,1,2,3")
        objects = user["objects"]
        ids = [str(o["object"]) for o in objects]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("each user needs >=2 distinct enrolled objects")
        for obj in objects:
            obj["object"] = str(obj["object"])
            if not obj["context"]:
                raise ValueError("empty context object")
            names.update(obj["context"])
        queries = user["queries"]
        if not queries or len({q["video"] for q in queries}) != len(queries):
            raise ValueError("empty or duplicate query video")
        for query in queries:
            rows = query["cache_rows"]
            if not rows or len(rows) != len(query["frames"]):
                raise ValueError("query cache_rows/frames mismatch")
            if any(not isinstance(r, int) or r < 0 or r >= 11064 for r in rows):
                raise ValueError("cache row outside fixed [0,11064) input")
            if len(rows) != len(set(rows)) or any(row in seen_rows for row in rows):
                raise ValueError("duplicate query feature rows")
            seen_rows.update(rows)
        expected.append({"user": user["user"], "fold": user["fold"], "n_objects": len(objects),
                         "context_frames": sum(len(o["context"]) for o in objects),
                         "query_frames": sum(len(q["frames"]) for q in queries), "n_videos": len(queries)})
    if {u["fold"] for u in users} != {0, 1, 2, 3}:
        raise ValueError("all four fixed folds must be present")
    metadata = []
    root = args.frames.resolve()
    for name in sorted(names):
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError("context frame names must be relative paths")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("context frame escapes --frames")
        info = path.stat()
        metadata.append({"name": name, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns})
    hashes = {"manifest_sha256": hashlib.sha256(payload).hexdigest(),
              "checkpoint_sha256": sha256(args.checkpoint), "query_features_sha256": sha256(args.query_features),
              "context_metadata_sha256": json_hash(metadata)}
    declared_query_sha = manifest.get("query_cache", {}).get("features_sha256")
    if not isinstance(declared_query_sha, str) or declared_query_sha.lower() != hashes["query_features_sha256"]:
        raise RuntimeError("actual query feature SHA does not match manifest.query_cache.features_sha256")
    return users, expected, hashes, metadata, payload


def synchronize(runner):
    runner.torch.cuda.synchronize()
    return time.perf_counter()


def load_context(runner, user, output, config_sha, runtime_sha, S):
    """Only this feature path is cached; scientific screening remains unchanged."""
    np = runner.np
    directory = output / "features" / user_key(user["user"])
    directory.mkdir(parents=True, exist_ok=True)
    flat, receipts = [], []
    started = time.perf_counter()
    hits = 0
    extracted_now = 0.0
    for obj in user["objects"]:
        runner.guard()
        key = user_key(obj["object"])
        path, receipt_path = directory / (key + ".npy"), directory / (key + ".json")
        identity = {"schema": 1, "config_sha256": config_sha, "runtime_sha256": runtime_sha,
                    "user": user["user"], "object": obj["object"],
                    "frames_sha256": json_hash(obj["context"]), "shape": [len(obj["context"]), 256, 384],
                    "dtype": "float32"}
        meta = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
        if meta is not None and any(meta.get(k) != v for k, v in identity.items()):
            raise RuntimeError("context cache configuration/owner mismatch")
        cache_hit = meta is not None and meta.get("status") == "complete"
        if cache_hit:
            if not path.exists() or sha256(path) != meta["feature_file_sha256"]:
                raise RuntimeError("context cache bytes do not match their saved SHA")
            hits += 1
        else:
            # This fixed generated path belongs to this run. An incomplete
            # feature buffer is rebuildable; complete cache bytes never rewrite.
            S.save_json(receipt_path, {**identity, "status": "building"})
            storage = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=tuple(identity["shape"]))
            feature_started = synchronize(runner)
            for offset, cls, patches in runner.batches(obj["context"]):
                storage[offset:offset + len(patches)] = patches.cpu().numpy()
                del cls, patches
            extract_s = synchronize(runner) - feature_started
            finalize = time.perf_counter()
            storage.flush()
            del storage
            feature_sha = sha256(path)
            meta = {**identity, "status": "complete", "feature_file_sha256": feature_sha,
                    "extract_once_s": extract_s, "cache_flush_and_hash_s": time.perf_counter() - finalize,
                    "created_utc": utc_now()}
            S.save_json(receipt_path, meta)
            extracted_now += extract_s
        # Copy-on-write mapping exposes a writable NumPy view without writing
        # cache files; frozen build_one only copies these arrays to CUDA.
        array = np.load(path, mmap_mode="c", allow_pickle=False)
        if array.shape != tuple(identity["shape"]) or array.dtype != np.float32:
            raise RuntimeError("context cache dtype/shape mismatch")
        flat.append(array.reshape(-1, 384))
        receipts.append({"object": obj["object"], "path": str(path), "cache_hit": cache_hit,
                         "feature_file_sha256": meta["feature_file_sha256"],
                         "extract_once_s": meta["extract_once_s"], "cache_flush_and_hash_s": meta["cache_flush_and_hash_s"]})
    return flat, {"all_hit": hits == len(user["objects"]), "objects_hit": hits,
                  "objects_extracted": len(user["objects"]) - hits,
                  "load_and_extract_wall_s": time.perf_counter() - started,
                  "extract_once_s": sum(r["extract_once_s"] for r in receipts),
                  "actual_extract_this_invocation_s": extracted_now,
                  "feature_set_sha256": json_hash([{k: r[k] for k in ("object", "feature_file_sha256")} for r in receipts]),
                  "features": receipts}


def infer_cached(runner, user, cache, memories):
    """Cache-row lookup and classifier receive no target labels."""
    t, np = runner.torch, runner.np
    result = []
    for query in user["queries"]:
        predictions = {arm: [] for arm in ARMS}
        for offset in range(0, len(query["cache_rows"]), runner.args.batch):
            runner.guard()
            rows = query["cache_rows"][offset:offset + runner.args.batch]
            batch = np.asarray(cache[rows], dtype=np.float16)
            p = t.from_numpy(batch).cuda().float()
            if not bool(t.isfinite(p).all()):
                raise FloatingPointError("non-finite cached query features")
            for arm, memory in memories.items():
                sim = p @ memory.reshape(-1, 384).T
                values = sim.reshape(len(p), 256, len(user["objects"]), 32).amax(3)
                scores = values.topk(8, dim=1).values.mean(1)
                predictions[arm].extend(scores.argmax(1).cpu().tolist())
                del sim, values, scores
            del batch, p
        result.append({"video": query["video"], "frames": query["frames"], "cache_rows": query["cache_rows"],
                       "predictions": predictions})
    # Only now, after all this user's predictions are fixed, expose targets to
    # the scorer/output. Context enrollment identities are legitimate inputs.
    objects = [o["object"] for o in user["objects"]]
    for record, query in zip(result, user["queries"]):
        record["target_index"] = objects.index(str(query["object"]))
    return result


def run_user(runner, user, query_cache, args, config_sha, sources, inputs, S, N, ordinal):
    started = synchronize(runner)
    gpu_started = started
    runner.torch.cuda.reset_peak_memory_stats()
    runner.progress(user["user"], "holdout_user_start", started, fold=user["fold"])
    flat, context = load_context(runner, user, args.out, config_sha, json_hash(runner.info), S)
    context_resources = runner.resource()
    first, repeats = {}, []
    samples = None
    for repeat in range(args.repeats):
        order = ARMS if (repeat + ordinal) % 2 == 0 else tuple(reversed(ARMS))
        for position, arm in enumerate(order):
            runner.guard()
            memory, row = N.build_one(runner, flat, arm)
            row.update({"repeat": repeat, "position": position, "order": order,
                        "centers_sha256": N.tensor_fingerprint(memory)})
            if samples is None:
                samples = row["sampled_others_sha256"]
            row["same_others_all_paths_repeats"] = row["sampled_others_sha256"] == samples
            cpu = memory.cpu().clone()
            if arm not in first:
                first[arm] = cpu
                row["repeat_center_consistency"] = {"first": True, "allclose": True, "max_abs_difference": 0.0}
            else:
                previous = first[arm]
                row["repeat_center_consistency"] = {"first": False,
                    "allclose": bool(runner.torch.allclose(cpu, previous, rtol=1e-5, atol=1e-6)),
                    "rtol": 1e-5, "atol": 1e-6, "max_abs_difference": float((cpu - previous).abs().max())}
            repeats.append(row)
            del memory, cpu
        runner.progress(user["user"], "holdout_repeat_complete", started, fold=user["fold"], repeat=repeat)
    del flat
    query_started = synchronize(runner)
    memories = {arm: first[arm].cuda() for arm in ARMS}
    predictions = infer_cached(runner, user, query_cache, memories)
    query_seconds = synchronize(runner) - query_started
    del memories, first
    resources = runner.resource()
    runner.torch.cuda.empty_cache()
    record = {"schema": 1, "status": "complete", "config_sha256": config_sha,
              "user": user["user"], "fold": user["fold"], "objects": [o["object"] for o in user["objects"]],
              "n_objects": len(user["objects"]), "context_frames": sum(len(o["context"]) for o in user["objects"]),
              "query_frames": sum(len(v["frames"]) for v in predictions), "n_videos": len(predictions),
              "predictions": predictions, "timing": N.timing_summary(repeats, context["extract_once_s"]),
              "repeats": repeats, "extract_once_s": context["extract_once_s"],
              "actual_extract_this_invocation_s": context["actual_extract_this_invocation_s"],
              "query_cache_read_and_two_arm_scoring_s": query_seconds, "context_cache": context,
              "same_others": all(r["same_others_all_paths_repeats"] for r in repeats),
              "all_repeat_centers_close": all(r["repeat_center_consistency"]["allclose"] for r in repeats),
              "source_hashes": sources, "input_hashes": inputs, "runtime_sha256": json_hash(runner.info),
              "resources": {"after_context": context_resources, "after_build_and_query": resources},
              "user_wall_s": time.perf_counter() - started,
              "user_gpu_wall_s": time.perf_counter() - gpu_started, "finished_utc": utc_now(),
              "representation_note": "fresh normalized FP32 context; existing FP16 query cache cast FP32 without renormalization"}
    record.update(score_predictions(predictions))
    record["consistency_pass"] = record["same_others"] and record["all_repeat_centers_close"]
    record["consistency_status"] = "pass" if record["consistency_pass"] else "failed_not_eligible_for_conditional_pass"
    runner.progress(user["user"], "holdout_user_complete", started, fold=user["fold"],
                    correct={a: record["metrics"][a]["correct"] for a in ARMS},
                    query_frames=record["query_frames"], context_cache_all_hit=context["all_hit"])
    return record


def completed_valid(record, expected, config_sha):
    if record.get("status") != "complete" or record.get("config_sha256") != config_sha:
        return False
    for key in ("user", "fold", "n_objects", "context_frames", "query_frames", "n_videos"):
        if record.get(key) != expected[key]:
            return False
    return sum(len(p["frames"]) for p in record["predictions"]) == expected["query_frames"]


def self_test():
    predictions = [{"video": "a", "frames": ["a0", "a1"], "cache_rows": [0, 1], "target_index": 0,
                    "predictions": {"combined": [0, 1], "negative_only": [0, 0]}},
                   {"video": "b", "frames": ["b0"], "cache_rows": [2], "target_index": 1,
                    "predictions": {"combined": [1], "negative_only": [0]}}]
    result = score_predictions(predictions)
    assert result["metrics"]["combined"]["correct"] == result["metrics"]["negative_only"]["correct"] == 2
    assert result["metrics"]["combined"]["video_macro_accuracy"] == .75
    assert result["metrics"]["negative_only"]["video_macro_accuracy"] == .5
    assert result["negative_minus_combined"]["saved"] == result["negative_minus_combined"]["harmed"] == 1
    assert result["negative_minus_combined"]["video_macro_difference_pp"] == -25
    print("stdlib scorer fixture PASS: 2 videos/3 frames; micro 2/3 both; macro .75 vs .50; save1/harm1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "frames", "query-features", "frozen-source-dir", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cap-gb", type=float, default=2.5)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, name) is None for name in ("manifest", "frames", "query_features", "frozen_source_dir", "checkpoint", "out")):
        parser.error("all input/output paths are required")
    if args.repeats != 3 or not 1 <= args.batch <= 32 or not 0 < args.cap_gb <= 2.5 or not 1 <= args.chunk <= 2048:
        parser.error("fixed repeats3; require batch<=32, cap-gb<=2.5, chunk<=2048, positive")
    import psutil
    if psutil.virtual_memory().available < 2 * GIB:
        raise MemoryError("available host RAM below2GiB; immediate stop before input hashing")
    invocation_started = time.perf_counter()
    sources = {"runner_sha256": sha256(__file__),
               "support_controls_sha256": sha256(args.frozen_source_dir / "support_controls.py"),
               "negative_memory_benchmark_sha256": sha256(args.frozen_source_dir / "negative_memory_benchmark.py")}
    if sources["support_controls_sha256"] != FROZEN_SUPPORT_SHA or sources["negative_memory_benchmark_sha256"] != FROZEN_BENCHMARK_SHA:
        raise RuntimeError("immutable dependency source SHA mismatch")
    sys.path.insert(0, str(args.frozen_source_dir.resolve()))
    S = importlib.import_module("support_controls")
    N = importlib.import_module("negative_memory_benchmark")
    users, expected, inputs, metadata, payload = read_inputs(args)
    config = {"schema": 1, "source_hashes": sources, "input_hashes": inputs, "expected_users": expected,
              "batch": args.batch, "cap_gib": args.cap_gb, "matrix_chunk": args.chunk, "repeats": args.repeats,
              "keep": .15, "parts": 32, "query_topk": 8, "kmeans_seed": 0, "kmeans_iters": 15,
              "sample_limit": 30000, "sample_seed": "object ordinal; frozen CPU torch.Generator order",
              "query_shape": [11064, 256, 384], "query_dtype": "float16 cache -> float32, no renormalization",
              "query_feature_identity": "actual file SHA must equal manifest.query_cache.features_sha256",
              "query_metadata_validation": "metadata file not supplied to this runner; its bytes/SHA are not independently validated",
              "context_dtype": "fresh AMP-float16 backbone -> normalized float32 cache",
              "context_identity": "parent-verified transferred frame package; startup size/mtime metadata+manifest SHA; feature cache bytes SHA",
              "fold_execution_order": [1, 2, 3, 0], "selection": "fixed arms; no tuning or selection on folds",
              "consistency_policy": "false same_others or all_repeat_centers_close prevents any conditional-pass interpretation",
              "gpu_budget_s": BUDGET_GPU_S, "center_tolerance": {"rtol": 1e-5, "atol": 1e-6}}
    config_sha = json_hash(config)
    args.out.mkdir(parents=True, exist_ok=True)
    root_path = args.out / "run.json"
    if root_path.exists():
        run = json.loads(root_path.read_text(encoding="utf-8"))
        if run["config_sha256"] != config_sha or sha256(args.out / "input_manifest.json") != inputs["manifest_sha256"]:
            raise RuntimeError("resume inputs/config/source changed; choose a new owned output")
    else:
        if any(args.out.iterdir()):
            raise RuntimeError("nonempty output lacks its ownership/config receipt")
        run = {"schema": 1, "config_sha256": config_sha, "config": config, "expected_users": expected,
               "completed_users": [], "all_counts_complete": False, "status": "starting", "invocations": [],
               "consistency_pass": False, "conditional_pass_eligible": False,
               "context_metadata": metadata}
        (args.out / "input_manifest.json").write_bytes(payload)
    # A prior unclosed process has no exact final CUDA lifetime. Conservatively
    # charge wall time since its launch, capped by the entire registered budget.
    now = datetime.now(timezone.utc)
    for old in run["invocations"]:
        if old.get("status") == "running":
            charged = min(BUDGET_GPU_S, max(0.0, (now - datetime.fromisoformat(old["started_utc"])).total_seconds()))
            old.update({"status": "interrupted_without_final_receipt", "budget_charge_s": charged,
                        "budget_charge_kind": "conservative launch-to-resume wall bound, not measured GPU use"})
    used_gpu = sum(float(v.get("budget_charge_s", v.get("gpu_job_wall_s", 0.0))) for v in run["invocations"])
    invocation = {"started_utc": utc_now(), "status": "running", "prior_gpu_budget_charged_s": used_gpu}
    run["invocations"].append(invocation)
    run["status"] = "running"
    S.save_json(root_path, run)
    runner = None
    gpu_attempt_started = None
    completed = []
    try:
        import numpy as np
        import psutil
        if psutil.virtual_memory().available < 2 * GIB:
            raise MemoryError("available host RAM below2GiB; immediate stop")
        query_cache = np.load(args.query_features, mmap_mode="r", allow_pickle=False)
        if query_cache.dtype != np.float16 or query_cache.shape != (11064, 256, 384):
            raise ValueError("query features must be float16 [11064,256,384]")
        expected_map = {e["user"]: e for e in expected}
        ranks = {fold: index for index, fold in enumerate((1, 2, 3, 0))}
        ordered = sorted(users, key=lambda u: (ranks[u["fold"]], u["user"]))
        # Ensure model/cache runtime checks and inherited chunk/iteration guards
        # also enforce the cumulative budget, reserving3s for release/receipts.
        class GuardedRunner(S.Runner):
            def __init__(self, runner_args, remaining):
                self.deadline = time.perf_counter() + remaining - 3.0
                super().__init__(runner_args)

            def guard(self):
                super().guard()
                if time.perf_counter() >= self.deadline:
                    raise BudgetStop("cumulative7200s GPU-job budget reached;3s reserved for release")

        for ordinal, user in enumerate(ordered):
            path = args.out / ("user_" + user_key(user["user"]) + ".json")
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if not completed_valid(record, expected_map[user["user"]], config_sha):
                    raise RuntimeError("completed user receipt is invalid; preserved for investigation")
                completed.append(record)
                print(json.dumps({"user": user["user"], "fold": user["fold"], "stage": "resume_skip_complete"}), flush=True)
            else:
                if runner is None:
                    if used_gpu >= BUDGET_GPU_S - 3:
                        raise BudgetStop("no registered GPU budget remains")
                    gpu_attempt_started = time.perf_counter()
                    runner = GuardedRunner(args, BUDGET_GPU_S - used_gpu)
                    if "runtime" in run and json_hash(run["runtime"]) != json_hash(runner.info):
                        raise RuntimeError("runtime changed; context cache reuse rejected")
                    run["runtime"] = runner.info
                    S.save_json(root_path, run)
                record = run_user(runner, user, query_cache, args, config_sha, sources, inputs, S, N, ordinal)
                S.save_json(path, record)
                completed.append(record)
            run["completed_users"] = [r["user"] for r in completed]
            run["all_counts_complete"] = len(completed) == len(expected)
            if runner is not None:
                invocation["gpu_elapsed_checkpoint_s"] = time.perf_counter() - runner.gpu_started
            S.save_json(root_path, run)
        run["all_counts_complete"] = len(completed) == 44
        run["all_repeat_centers_close"] = all(r["all_repeat_centers_close"] for r in completed)
        run["same_others"] = all(r["same_others"] for r in completed)
        run["consistency_pass"] = run["all_repeat_centers_close"] and run["same_others"]
        run["conditional_pass_eligible"] = run["all_counts_complete"] and run["consistency_pass"]
        run["status"] = "complete" if run["consistency_pass"] else "complete_consistency_failed_not_eligible"
        invocation["status"] = "complete"
    except BaseException as exc:
        run["status"] = "budget_stop" if isinstance(exc, BudgetStop) else "failed"
        invocation.update({"status": run["status"], "error_type": type(exc).__name__, "error": str(exc),
                           "traceback": traceback.format_exc()})
        raise
    finally:
        if runner is not None:
            runner.torch.cuda.synchronize()
            invocation["gpu_job_wall_s"] = time.perf_counter() - runner.gpu_started
            invocation["resources_at_finish"] = runner.resource()
            del runner.model
            runner.torch.cuda.empty_cache()
        else:
            invocation["gpu_job_wall_s"] = (time.perf_counter() - gpu_attempt_started
                                             if gpu_attempt_started is not None else 0.0)
            if gpu_attempt_started is not None:
                invocation["gpu_wall_kind"] = "conservative constructor-attempt wall; initialization failed"
        invocation["budget_charge_s"] = invocation["gpu_job_wall_s"]
        invocation["actual_worker_invocation_wall_s"] = time.perf_counter() - invocation_started
        invocation["finished_utc"] = utc_now()
        run["gpu_budget_charged_total_s"] = used_gpu + invocation["budget_charge_s"]
        S.save_json(root_path, run)


if __name__ == "__main__":
    main()
