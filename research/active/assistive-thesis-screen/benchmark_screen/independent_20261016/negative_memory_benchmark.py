"""Measure removing own-consistency from frozen ORBIT support construction.

Performance-only replay of the same six consumed validation users. The two real
paths independently construct only their own memory, sharing context features
but neither sampled pools, scores, ranking, nor clustering work. Three repeats
alternate order; context is extracted once, not once per repeat. Query frames
are existing manifest inputs and are extracted once jointly for both memories.
Truth is accessed only after predictions, for descriptive scoring. No search or
noninferiority claim. support_controls.py is an immutable dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import support_controls as S

FROZEN_SOURCE_SHA256 = "6966d515a6ed4a0e2e5056c667ec5cc82306db12dac599541b710619f51979b5"
ARMS = ("combined", "negative_only")


def sync_clock(runner):
    runner.torch.cuda.synchronize()
    return time.perf_counter()


def tensor_fingerprint(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def build_one(runner, flat, arm):
    """A real standalone memory path; no reuse of the other arm's computation."""
    t, np = runner.torch, runner.np
    runner.guard()
    t.cuda.reset_peak_memory_stats()
    before = runner.resource()
    started = sync_clock(runner)
    selection_s = clustering_s = 0.0
    centers, object_records = [], []
    sizes = [len(p) for p in flat]
    sample_hash = hashlib.sha256()
    selected_hash = hashlib.sha256()
    for obj, source in enumerate(flat):
        runner.guard()
        stage = sync_clock(runner)
        p = t.from_numpy(source).cuda()
        # Identical generator sequence to the frozen module. Both paths draw
        # others first; skipping the subsequent own draw cannot change others.
        g = t.Generator(device="cpu").manual_seed(obj)
        others_idx = [j for j in range(len(flat)) if j != obj]
        total = sum(sizes[j] for j in others_idx)
        picks = t.randperm(total, generator=g)[:30000].numpy()
        sampled, offset = [], 0
        for j in others_idx:
            indices = picks[(picks >= offset) & (picks < offset + sizes[j])] - offset
            sampled.append(flat[j][indices])
            offset += sizes[j]
        others = t.from_numpy(np.concatenate(sampled)).cuda()
        del sampled
        own = None
        if arm == "combined":
            own_pick = t.randperm(len(p), generator=g)[:30000].to(p.device)
            own = p[own_pick]
            own_scores = np.empty(len(p), np.float32)
            del own_pick
        other_scores = np.empty(len(p), np.float32)
        for start in range(0, len(p), runner.args.chunk):
            runner.guard()
            block = p[start:start + runner.args.chunk]
            if own is not None:
                own_scores[start:start + len(block)] = (block @ own.T).topk(6, dim=1).values[:, 1:].mean(1).cpu().numpy()
            other_scores[start:start + len(block)] = (block @ others.T).amax(1).cpu().numpy()
        score = own_scores - other_scores if arm == "combined" else -other_scores
        keep = max(32, math.ceil(.15 * len(p)))
        selected = np.sort(np.argsort(-score, kind="stable")[:keep])
        kept = p[t.from_numpy(selected).to(p.device)]
        del own, others, score, other_scores
        if arm == "combined":
            del own_scores
        selected_at = sync_clock(runner)
        selection_s += selected_at - stage
        center = runner.kmeans(kept)
        clustered_at = sync_clock(runner)
        clustering_s += clustered_at - selected_at
        centers.append(center)
        # Diagnostic hashing is outside component timers, inside real total wall.
        sample_hash.update(picks.tobytes())
        selected_hash.update(selected.tobytes())
        object_records.append({"object_index": obj, "support_patches": len(p), "retained": keep,
                               "sampled_others": len(picks), "sampled_own": min(30000, len(p)) if arm == "combined" else 0,
                               "selection_s": selected_at - stage, "clustering_s": clustered_at - selected_at})
        del p, kept, block
    memory = t.stack(centers)
    elapsed = sync_clock(runner) - started
    return memory, {"arm": arm, "selection_s": selection_s, "clustering_s": clustering_s,
                    "build_total_s": elapsed, "other_overhead_s": elapsed - selection_s - clustering_s,
                    "resources_before": before, "resources_after": runner.resource(),
                    "sampled_others_sha256": sample_hash.hexdigest(), "selected_patch_indices_sha256": selected_hash.hexdigest(),
                    "objects": object_records}


def timing_summary(repeats, extract_s):
    result = {}
    for arm in ARMS:
        rows = [r for r in repeats if r["arm"] == arm]
        first = {key: rows[0][key] for key in ("selection_s", "clustering_s", "build_total_s")}
        warm = {key: statistics.median(r[key] for r in rows[1:])
                for key in ("selection_s", "clustering_s", "build_total_s")}
        result[arm] = {"first_build_s": first, "post_first_median_s": warm,
                       "support_extract_once_s": extract_s,
                       "extract_plus_first_build_estimate_s": extract_s + first["build_total_s"],
                       "extract_plus_post_first_build_estimate_s": extract_s + warm["build_total_s"]}
    return result


def verify_queries(runner, user, memories, reference):
    """One shared query extraction per video; no truth enters classification."""
    t = runner.torch
    expected = {v["video"]: v for v in reference["predictions"]}
    records = []
    started = sync_clock(runner)
    for query in user["queries"]:
        predictions = {a: [] for a in ARMS}
        for _, cls, patches in runner.batches(query["frames"]):
            for arm, memory in memories.items():
                sim = patches @ memory.reshape(-1, 384).T
                per_object = sim.reshape(len(patches), 256, len(user["objects"]), 32).amax(3)
                score = per_object.topk(8, dim=1).values.mean(1)
                predictions[arm].extend(score.argmax(1).cpu().tolist())
                del sim, per_object, score
            del cls, patches
        baseline = expected[query["video"]]
        if query["frames"] != baseline["frames"]:
            raise RuntimeError("reference query order changed")
        # Only scorer/diagnostic fields below use target identity.
        target = baseline["target_index"]
        records.append({"video": query["video"], "frames": query["frames"], "target_index": target,
                        "predictions": predictions,
                        "reference_mismatches": {a: sum(x != y for x, y in zip(predictions[a], baseline["predictions"][a])) for a in ARMS},
                        "correct": {a: sum(p == target for p in predictions[a]) for a in ARMS}})
    return {"query_extract_and_two_arm_scoring_s": sync_clock(runner) - started,
            "n_frames": sum(len(v["frames"]) for v in records), "n_videos": len(records),
            "mismatches": {a: sum(v["reference_mismatches"][a] for v in records) for a in ARMS},
            "correct": {a: sum(v["correct"][a] for v in records) for a in ARMS}, "videos": records}


def benchmark_user(runner, user, reference, ordinal, repeats):
    t, np = runner.torch, runner.np
    started = sync_clock(runner)
    runner.progress(user["user"], "benchmark_start", started)
    context_started = sync_clock(runner)
    flat, n_context = [], 0
    for obj in user["objects"]:
        names = obj["context"]
        runner.guard()
        storage = np.empty((len(names), 256, 384), np.float32)
        for offset, cls, patches in runner.batches(names):
            storage[offset:offset + len(patches)] = patches.cpu().numpy()
            del cls, patches
        flat.append(storage.reshape(-1, 384))
        n_context += len(names)
    del storage
    extract_s = sync_clock(runner) - context_started
    extraction_resources = runner.resource()
    first_centers = {}
    repeat_records = []
    sample_digest = None
    for repeat in range(repeats):
        order = ARMS if (repeat + ordinal) % 2 == 0 else tuple(reversed(ARMS))
        for position, arm in enumerate(order):
            memory, row = build_one(runner, flat, arm)
            row.update({"repeat": repeat, "position": position, "order": order,
                        "centers_sha256": tensor_fingerprint(memory)})
            if sample_digest is None:
                sample_digest = row["sampled_others_sha256"]
            row["same_others_all_paths_repeats"] = row["sampled_others_sha256"] == sample_digest
            cpu = memory.cpu().clone()
            if arm not in first_centers:
                first_centers[arm] = cpu
                row["repeat_center_consistency"] = {"first": True, "allclose": True, "max_abs_difference": 0.0}
            else:
                previous = first_centers[arm]
                row["repeat_center_consistency"] = {
                    "first": False, "allclose": bool(t.allclose(cpu, previous, rtol=1e-5, atol=1e-6)),
                    "rtol": 1e-5, "atol": 1e-6,
                    "max_abs_difference": float((cpu - previous).abs().max())}
            repeat_records.append(row)
            del memory, cpu
            runner.progress(user["user"], "build_complete", started, arm=arm, repeat=repeat,
                            selection_s=row["selection_s"], clustering_s=row["clustering_s"],
                            build_total_s=row["build_total_s"], consistency=row["repeat_center_consistency"])
    del flat
    memories = {a: first_centers[a].cuda() for a in ARMS}
    verification = verify_queries(runner, user, memories, reference)
    del memories, first_centers
    t.cuda.empty_cache()
    result = {"user": user["user"], "context_frames": n_context, "n_objects": len(user["objects"]),
              "extract_once_s": extract_s, "extract_resources": extraction_resources, "repeats": repeat_records,
              "timing": timing_summary(repeat_records, extract_s), "verification": verification,
              "all_repeat_centers_close": all(r["repeat_center_consistency"]["allclose"] for r in repeat_records),
              "all_paths_same_others": all(r["same_others_all_paths_repeats"] for r in repeat_records),
              "user_actual_wall_s": sync_clock(runner) - started}
    runner.progress(user["user"], "benchmark_complete", started, query_mismatches=verification["mismatches"])
    return result


def aggregate(users):
    result = {"n_users": len(users), "n_objects": sum(u["n_objects"] for u in users),
              "n_context_frames_extracted_once": sum(u["context_frames"] for u in users),
              "n_query_frames_extracted_once": sum(u["verification"]["n_frames"] for u in users),
              "actual_shared_context_extract_s": sum(u["extract_once_s"] for u in users),
              "actual_query_extract_and_verification_s": sum(u["verification"]["query_extract_and_two_arm_scoring_s"] for u in users),
              "actual_all_repeated_memory_builds_s": sum(r["build_total_s"] for u in users for r in u["repeats"]),
              "all_repeat_centers_close": all(u["all_repeat_centers_close"] for u in users),
              "all_paths_same_others": all(u["all_paths_same_others"] for u in users),
              "reference_prediction_mismatches": {a: sum(u["verification"]["mismatches"][a] for u in users) for a in ARMS},
              "descriptive_correct_frames": {a: sum(u["verification"]["correct"][a] for u in users) for a in ARMS},
              "arms": {}}
    for arm in ARMS:
        result["arms"][arm] = {
            "sum_first_build_s": {k: sum(u["timing"][arm]["first_build_s"][k] for u in users)
                                  for k in ("selection_s", "clustering_s", "build_total_s")},
            "sum_user_post_first_medians_s": {k: sum(u["timing"][arm]["post_first_median_s"][k] for u in users)
                                             for k in ("selection_s", "clustering_s", "build_total_s")},
            "extract_plus_first_build_estimate_s": sum(u["timing"][arm]["extract_plus_first_build_estimate_s"] for u in users),
            "extract_plus_post_first_build_estimate_s": sum(u["timing"][arm]["extract_plus_post_first_build_estimate_s"] for u in users)}
    combined, negative = result["arms"]["combined"], result["arms"]["negative_only"]
    result["warm_build_speedup_ratio"] = (combined["sum_user_post_first_medians_s"]["build_total_s"] /
                                             negative["sum_user_post_first_medians_s"]["build_total_s"])
    result["warm_extract_plus_build_speedup_estimate_ratio"] = (combined["extract_plus_post_first_build_estimate_s"] /
                                                                 negative["extract_plus_post_first_build_estimate_s"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "frames", "checkpoint", "reference-out", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cap-gb", type=float, default=2.5)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats != 3 or not 1 <= args.batch <= 32 or not 0 < args.cap_gb <= 2.5 or not 1 <= args.chunk <= 2048:
        parser.error("require exactly3 repeats, batch<=32, cap<=2.5GiB, chunk<=2048")
    if args.out.exists():
        raise RuntimeError("output already exists; choose a new output filename")
    started = time.perf_counter()
    frozen = S.sha256(S.__file__)
    if frozen != FROZEN_SOURCE_SHA256:
        raise RuntimeError("frozen support_controls.py source changed")
    reference_run = json.loads((args.reference_out / "run.json").read_text(encoding="utf-8"))
    users, inputs = S.manifest_inputs(args.manifest, args.frames)
    expected = reference_run["config"]
    checks = {"source_sha256": frozen, "manifest_sha256": S.sha256(args.manifest),
              "checkpoint_sha256": S.sha256(args.checkpoint), "frames_sha256": inputs["ordered_frames_sha256"],
              "batch": args.batch, "cap_gib": args.cap_gb, "matrix_chunk": args.chunk}
    for key, value in checks.items():
        if expected[key] != value:
            raise RuntimeError(f"reference configuration/input mismatch: {key}")
    references, reference_receipts = {}, []
    for user in users:
        path = args.reference_out / ("user_" + hashlib.sha256(user["user"].encode()).hexdigest()[:16] + ".json")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["config_sha256"] != reference_run["config_sha256"]:
            raise RuntimeError("reference user configuration mismatch")
        references[user["user"]] = rec
        reference_receipts.append({"path": str(path), "sha256": S.sha256(path)})
    output = {"status": "running", "development_only": True, "purpose": "performance-only; no noninferiority claim",
              "source_sha256": S.sha256(__file__), "frozen_support_source_sha256": frozen,
              "input_checks": checks, "reference_config_sha256": reference_run["config_sha256"],
              "reference_files": reference_receipts, "configuration": {
                  "repeats": args.repeats, "first_repeat_reported_separately": True,
                  "post_first_summary": "median of repeats1and2, then sum users",
                  "order": "alternate arms by repeat+user ordinal parity",
                  "backbone_forward": "context once; query once jointly; no extraction per repeat",
                  "selection_timing": "H2D features/pool sampling/matmul/CPU ranking/gather, CUDA synchronized",
                  "clustering_timing": "frozen Runner.kmeans, CUDA synchronized",
                  "total_timing": "actual standalone build wall including diagnostics and overhead",
                  "end_to_end_estimates": "one shared measured feature-extraction cost plus each path build; excludes model startup/query",
                  "repeat_center_tolerance": {"rtol": 1e-5, "atol": 1e-6}}, "users": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    S.save_json(args.out, output)
    runner = None
    try:
        runner = S.Runner(args)
        if S.json_hash(runner.info) != S.json_hash(reference_run["runtime"]):
            raise RuntimeError("reference runtime changed")
        output["runtime"] = runner.info
        for ordinal, user in enumerate(users):
            record = benchmark_user(runner, user, references[user["user"]], ordinal, args.repeats)
            output["users"].append(record)
            output["summary"] = aggregate(output["users"])
            S.save_json(args.out, output)
        summary = output["summary"]
        stable = summary["all_repeat_centers_close"] and summary["all_paths_same_others"]
        matched = all(v == 0 for v in summary["reference_prediction_mismatches"].values())
        output["status"] = "complete_consistency_pass" if stable and matched else "complete_consistency_difference"
    except BaseException as exc:
        output.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc()})
        raise
    finally:
        if runner is not None:
            runner.torch.cuda.synchronize()
            output["gpu_job_wall_s"] = time.perf_counter() - runner.gpu_started
            output["resources_at_finish"] = runner.resource()
            del runner.model
            runner.torch.cuda.empty_cache()
        else:
            output["gpu_job_wall_s"] = 0.0
        output["actual_worker_invocation_wall_s"] = time.perf_counter() - started
        S.save_json(args.out, output)


if __name__ == "__main__":
    main()
