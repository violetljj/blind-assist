"""Diagnose classification effects of failed repeat-centroid consistency checks.

Runs only users whose completed main receipt failed same_others or centroid
allclose. Exactly three fixed repeats, two independent memory paths each, using
SHA-verified cached context features and the same query cache. No backbone or
context/query image is loaded. Frozen kmeans/build_one/infer_cached are reused.
No voting, tolerance change, parameter search, main-output write, or rescue of
the original prespecified consistency gate. Timing is diagnostic job cost, not
a production performance comparison (the unused backbone is absent).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True

SUPPORT_SHA = "6966d515a6ed4a0e2e5056c667ec5cc82306db12dac599541b710619f51979b5"
BENCHMARK_SHA = "4f85b862305ab53738ac9ea3355895c543eec621c2595065bb30306530b71720"
HOLDOUT_SHA = "ef2ac78a74a6f93db0ecc1c583943914caba83a1cde87dcf91952da881e61b8c"
ARMS = ("combined", "negative_only")
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


def comparison(candidate, reference):
    """Scorer-only comparison: changed argmax includes wrong-to-wrong changes."""
    totals = {a: {"frames": 0, "different_argmax": 0, "saved": 0, "harmed": 0,
                  "both_correct": 0, "both_wrong": 0} for a in ARMS}
    videos = []
    if len(candidate) != len(reference):
        raise ValueError("video denominator changed")
    for current, prior in zip(candidate, reference):
        for key in ("video", "frames", "cache_rows", "target_index"):
            if current[key] != prior[key]:
                raise ValueError(f"prediction identity mismatch: {key}")
        per_video = {"video": current["video"], "arms": {}}
        n, target = len(current["frames"]), current["target_index"]
        for arm in ARMS:
            a, b = current["predictions"][arm], prior["predictions"][arm]
            if len(a) != n or len(b) != n:
                raise ValueError("prediction/frame denominator mismatch")
            counts = {"frames": n, "different_argmax": sum(x != y for x, y in zip(a, b)),
                      "saved": sum(x == target and y != target for x, y in zip(a, b)),
                      "harmed": sum(x != target and y == target for x, y in zip(a, b)),
                      "both_correct": sum(x == target and y == target for x, y in zip(a, b)),
                      "both_wrong": sum(x != target and y != target for x, y in zip(a, b))}
            per_video["arms"][arm] = counts
            for key, value in counts.items():
                totals[arm][key] += value
        videos.append(per_video)
    return {"direction": "candidate minus reference; saved/harmed use evaluator identity only",
            "arms": totals, "videos": videos}


def cached_context(runner, user, reference, run_out, config_sha, runtime_sha, S):
    features = reference["context_cache"]["features"]
    if [f["object"] for f in features] != reference["objects"]:
        raise RuntimeError("reference feature/object order mismatch")
    flat, receipts = [], []
    root = (run_out / "features").resolve()
    for obj, feature in zip(user["objects"], features):
        runner.guard()
        path = Path(feature["path"]).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("context feature path escapes original owned cache")
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (meta["status"] != "complete" or meta["config_sha256"] != config_sha or
                meta["runtime_sha256"] != runtime_sha or meta["user"] != user["user"] or
                meta["object"] != str(obj["object"]) or meta["frames_sha256"] != S.json_hash(obj["context"])):
            raise RuntimeError("context cache provenance mismatch")
        actual = sha256(path)
        if actual != meta["feature_file_sha256"] or actual != feature["feature_file_sha256"]:
            raise RuntimeError("actual context feature SHA mismatch")
        array = runner.np.load(path, mmap_mode="c", allow_pickle=False)
        if array.dtype != runner.np.float32 or array.shape != (len(obj["context"]), 256, 384):
            raise RuntimeError("cached context geometry mismatch")
        flat.append(array.reshape(-1, 384))
        receipts.append({"object": str(obj["object"]), "path": str(path), "sha256": actual,
                         "cache_receipt_sha256": sha256(path.with_suffix(".json"))})
    return flat, receipts


def diagnose_user(runner, user, reference, query_cache, args, main_run, S, N, H, ordinal):
    started = N.sync_clock(runner)
    runner.progress(user["user"], "repeat_diagnostic_start", started, fold=user["fold"])
    flat, cache_receipts = cached_context(runner, user, reference, args.run_out,
                                        main_run["config_sha256"], S.json_hash(main_run["runtime"]), S)
    all_centers = {a: [] for a in ARMS}
    repeats = []
    source_repeats = reference["repeats"]
    for repeat in range(3):
        order = ARMS if (repeat + ordinal) % 2 == 0 else tuple(reversed(ARMS))
        memories, builds = {}, {}
        for position, arm in enumerate(order):
            memory, row = N.build_one(runner, flat, arm)
            cpu = memory.cpu().clone()
            all_centers[arm].append(cpu)
            center_sha = N.tensor_fingerprint(memory)
            original_rows = [r for r in source_repeats if r["arm"] == arm]
            row.update({"repeat": repeat, "position": position, "order": order, "centers_sha256": center_sha,
                        "matching_original_center_hash_repeats": [r["repeat"] for r in original_rows if r["centers_sha256"] == center_sha],
                        "original_others_hashes": sorted({r["sampled_others_sha256"] for r in original_rows}),
                        "original_selection_hashes": sorted({r["selected_patch_indices_sha256"] for r in original_rows})})
            row["others_matches_original"] = row["sampled_others_sha256"] in row["original_others_hashes"]
            row["selection_matches_original"] = row["selected_patch_indices_sha256"] in row["original_selection_hashes"]
            memories[arm] = memory
            builds[arm] = row
            del cpu
        prediction_started = N.sync_clock(runner)
        predictions = H.infer_cached(runner, user, query_cache, memories)
        prediction_wall = N.sync_clock(runner) - prediction_started
        versus_original = comparison(predictions, reference["predictions"])
        score = H.score_predictions(predictions)
        repeats.append({"repeat": repeat, "order": order, "builds": builds,
                        "query_read_and_two_arm_predict_s": prediction_wall,
                        "predictions": predictions, "versus_original": versus_original,
                        "scoring": score, "resources": runner.resource()})
        del memories, memory
        runner.progress(user["user"], "repeat_diagnostic_prediction_complete", started, repeat=repeat,
                        changed_argmax_vs_original={a: versus_original["arms"][a]["different_argmax"] for a in ARMS})
    center_pairs, prediction_pairs = [], []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        pair = {"reference_repeat": first, "candidate_repeat": second, "arms": {}}
        for arm in ARMS:
            a, b = all_centers[arm][second], all_centers[arm][first]
            pair["arms"][arm] = {"allclose": bool(runner.torch.allclose(a, b, rtol=1e-5, atol=1e-6)),
                                  "rtol": 1e-5, "atol": 1e-6, "max_abs_difference": float((a - b).abs().max()),
                                  "hash_equal": N.tensor_fingerprint(a) == N.tensor_fingerprint(b)}
        center_pairs.append(pair)
        prediction_pairs.append({"reference_repeat": first, "candidate_repeat": second,
                                 **comparison(repeats[second]["predictions"], repeats[first]["predictions"])})
    fingerprints = {a: {"others_hashes": sorted({r["builds"][a]["sampled_others_sha256"] for r in repeats}),
                         "selection_hashes": sorted({r["builds"][a]["selected_patch_indices_sha256"] for r in repeats})} for a in ARMS}
    for value in fingerprints.values():
        value["same_others_across_repeats"] = len(value["others_hashes"]) == 1
        value["same_selection_across_repeats"] = len(value["selection_hashes"]) == 1
    result = {"user": user["user"], "fold": user["fold"], "n_objects": reference["n_objects"],
              "context_frames": reference["context_frames"], "query_frames": reference["query_frames"],
              "original_consistency": {"same_others": reference["same_others"],
                  "all_repeat_centers_close": reference["all_repeat_centers_close"],
                  "repeat_center_consistency": [{"arm": r["arm"], "repeat": r["repeat"],
                        "consistency": r["repeat_center_consistency"]} for r in source_repeats]},
              "context_cache_receipts": cache_receipts, "repeats": repeats, "center_pairwise": center_pairs,
              "prediction_pairwise": prediction_pairs, "fingerprints": fingerprints,
              "any_changed_argmax_vs_original": {a: any(r["versus_original"]["arms"][a]["different_argmax"] for r in repeats) for a in ARMS},
              "any_changed_argmax_between_repeats": {a: any(r["arms"][a]["different_argmax"] for r in prediction_pairs) for a in ARMS},
              "original_gate_status_unchanged": True, "actual_user_diagnostic_wall_s": N.sync_clock(runner) - started}
    del flat, all_centers
    runner.torch.cuda.empty_cache()
    return result


def self_test():
    original = [{"video": "v", "frames": ["0", "1", "2", "3"], "cache_rows": [0, 1, 2, 3],
                 "target_index": 0, "predictions": {"combined": [0, 1, 1, 0], "negative_only": [0, 1, 1, 0]}}]
    candidate = [{**original[0], "predictions": {a: [1, 0, 2, 0] for a in ARMS}}]
    result = comparison(candidate, original)
    for arm in ARMS:
        assert result["arms"][arm] == {"frames": 4, "different_argmax": 3, "saved": 1,
                                      "harmed": 1, "both_correct": 1, "both_wrong": 1}
    print("stdlib fixture PASS: argmax changes3/4 include wrong-to-wrong1; save1/harm1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-out", "query-features", "frozen-source-dir", "run-source", "out"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--remaining-gpu-s", type=float, help="controller's remaining budget, <=7200s")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cap-gb", type=float, default=2.5)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, k) is None for k in ("run_out", "query_features", "frozen_source_dir", "run_source", "out", "remaining_gpu_s")):
        parser.error("all paths and --remaining-gpu-s are required")
    if args.repeats != 3 or not 0 < args.remaining_gpu_s <= 7200:
        parser.error("fixed repeats3 and 0<remaining-gpu-s<=7200")
    if args.out.exists() or args.out.resolve().is_relative_to(args.run_out.resolve()):
        raise RuntimeError("choose a new diagnostic JSON outside the immutable main output")
    import psutil
    if psutil.virtual_memory().available < 2 * GIB:
        raise MemoryError("host RAM below2GiB; immediate stop")
    started = time.perf_counter()
    main_path = args.run_out / "run.json"
    main_run = json.loads(main_path.read_text(encoding="utf-8"))
    if not main_run["all_counts_complete"] or not main_run["status"].startswith("complete"):
        raise RuntimeError("main run has not finished all users; wait for its final receipt")
    config = main_run["config"]
    for key, actual in (("batch", args.batch), ("cap_gib", args.cap_gb), ("matrix_chunk", args.chunk)):
        if actual != config[key]:
            raise RuntimeError(f"diagnostic must preserve original {key}")
    sources = {"diagnostic_sha256": sha256(__file__), "runner_sha256": sha256(args.run_source),
               "support_controls_sha256": sha256(args.frozen_source_dir / "support_controls.py"),
               "negative_memory_benchmark_sha256": sha256(args.frozen_source_dir / "negative_memory_benchmark.py")}
    for key, expected in (("runner_sha256", HOLDOUT_SHA), ("support_controls_sha256", SUPPORT_SHA),
                          ("negative_memory_benchmark_sha256", BENCHMARK_SHA)):
        if sources[key] != expected or sources[key] != config["source_hashes"][key]:
            raise RuntimeError(f"immutable source mismatch: {key}")
    query_sha = sha256(args.query_features)
    manifest_path = args.run_out / "input_manifest.json"
    manifest_sha = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (query_sha != config["input_hashes"]["query_features_sha256"] or
            query_sha != manifest["query_cache"]["features_sha256"].lower() or
            manifest_sha != config["input_hashes"]["manifest_sha256"]):
        raise RuntimeError("actual query/manifest bytes differ from original inputs")
    sys.path.insert(0, str(args.frozen_source_dir.resolve()))
    S, N = importlib.import_module("support_controls"), importlib.import_module("negative_memory_benchmark")
    spec = importlib.util.spec_from_file_location("frozen_holdout_for_diagnostic", args.run_source)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    expected = {u["user"]: u for u in main_run["expected_users"]}
    manifest_users = {str(u["user"]): u for u in manifest["users"]}
    targets, reference_receipts = [], []
    for user_id in main_run["completed_users"]:
        path = args.run_out / ("user_" + H.user_key(user_id) + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not H.completed_valid(record, expected[user_id], main_run["config_sha256"]):
            raise RuntimeError("invalid original user completion receipt")
        if record["source_hashes"] != config["source_hashes"] or record["input_hashes"] != config["input_hashes"]:
            raise RuntimeError("original user source/input identity mismatch")
        if not record["same_others"] or not record["all_repeat_centers_close"]:
            user = manifest_users[user_id]
            user["user"], user["fold"] = str(user["user"]), int(user["fold"])
            for obj in user["objects"]:
                obj["object"] = str(obj["object"])
            targets.append((user, record))
            reference_receipts.append({"user": user_id, "path": str(path), "sha256": sha256(path)})
    available = min(args.remaining_gpu_s, max(0.0, 7200 - main_run["gpu_budget_charged_total_s"]))
    output = {"status": "running", "diagnostic_only": True, "original_gate_status_unchanged": True,
              "main_run_path": str(main_path), "main_run_sha256": sha256(main_path),
              "main_config_sha256": main_run["config_sha256"], "source_hashes": sources,
              "input_hashes": {"query_features_sha256": query_sha, "manifest_sha256": manifest_sha},
              "target_users": [u["user"] for u, _ in targets], "reference_receipts": reference_receipts,
              "gpu_budget_remaining_from_controller_s": args.remaining_gpu_s, "effective_available_gpu_s": available,
              "fixed_protocol": {"repeats": 3, "parts": 32, "keep": .15, "query_topk": 8,
                                 "rtol": 1e-5, "atol": 1e-6, "no_backbone_loaded": True,
                                 "no_new_feature_extraction": True, "no_voting_or_gate_repair": True}, "users": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    S.save_json(args.out, output)
    runner = None
    gpu_attempt = None
    try:
        if targets:
            import numpy as np
            import torch
            import timm
            cache = np.load(args.query_features, mmap_mode="r", allow_pickle=False)
            if cache.dtype != np.float16 or cache.shape != (11064, 256, 384):
                raise RuntimeError("query cache shape/dtype changed")
            if available <= 3:
                raise RuntimeError("insufficient remaining registered GPU budget")

            class CachedRunner(S.Runner):
                def __init__(self):
                    self.np, self.torch, self.psutil, self.args = np, torch, psutil, args
                    self.deadline = time.perf_counter() + available - 3
                    self.guard()
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA unavailable")
                    self.gpu_started = time.perf_counter()
                    torch.set_num_threads(4)
                    torch.cuda.set_device(0)
                    total = torch.cuda.get_device_properties(0).total_memory
                    torch.cuda.set_per_process_memory_fraction(args.cap_gb * GIB / total, device=0)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    self.info = {"torch": str(torch.__version__), "numpy": np.__version__, "timm": timm.__version__,
                                 "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                                 "gpu_total_bytes": total}
                    for key, value in self.info.items():
                        if value != main_run["runtime"][key]:
                            raise RuntimeError(f"runtime mismatch: {key}")

                def guard(self):
                    super().guard()
                    if time.perf_counter() >= self.deadline:
                        raise RuntimeError("remaining GPU budget reached;3s reserved for release")

            gpu_attempt = time.perf_counter()
            runner = CachedRunner()
            output["runtime"] = runner.info
            for ordinal, (user, reference) in enumerate(targets):
                record = diagnose_user(runner, user, reference, cache, args, main_run, S, N, H, ordinal)
                output["users"].append(record)
                S.save_json(args.out, output)
        output["status"] = "complete_diagnostic_only"
        output["any_argmax_difference_vs_original"] = {
            a: any(u["any_changed_argmax_vs_original"][a] for u in output["users"]) for a in ARMS}
        output["any_argmax_difference_between_repeats"] = {
            a: any(u["any_changed_argmax_between_repeats"][a] for u in output["users"]) for a in ARMS}
    except BaseException as exc:
        output.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc()})
        raise
    finally:
        if runner is not None:
            runner.torch.cuda.synchronize()
            output["gpu_job_wall_s"] = time.perf_counter() - runner.gpu_started
            output["resources_at_finish"] = runner.resource()
            runner.torch.cuda.empty_cache()
        else:
            output["gpu_job_wall_s"] = time.perf_counter() - gpu_attempt if gpu_attempt is not None else 0.0
        output["actual_diagnostic_invocation_wall_s"] = time.perf_counter() - started
        S.save_json(args.out, output)


if __name__ == "__main__":
    main()
