"""Independent ORBIT support-selection controls; consumed validation, EXPLORE.

The manifest is the only selection input. No boxes/masks or query truth enter
feature extraction, support screening, clustering, or classification. Query
object identity is used only by score_records after predictions are complete.
All patch arms share one extraction, 32 prototypes and top-8 query pooling.

Semantics checked against orbit_eval.forward/kmeans and orbit_dump.part_memory_from:
AMP-fp16 backbone -> normalized fp32 CLS/patches; own top-6 drops its largest
match, NOT all same-frame matches; other/own pools are sampled to <=30000.
Unlike the author's quantile threshold, ranked arms retain exactly ceil(15%*N)
(at least 32), breaking ties by original patch order. K-means uses the same
seed-0 random-row initialization rule and 15 spherical updates for every arm.
This is an independent implementation, not a bitwise reproduction.

Run --users u0 for a canary, then omit --users with the SAME manifest/output to
resume all users without recomputing completed users. --users is an execution
subset, never part of the scientific configuration. Input bytes are rehashed.
No pretrained=True, network request, annotation read, or persistent feature dump.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    try:
        os.environ[_key] = str(min(4, max(1, int(os.environ.get(_key, "4")))))
    except ValueError:
        os.environ[_key] = "4"

ARMS = ("wholeframeCLS", "unscreened32", "random_keep15_s0", "random_keep15_s1",
        "random_keep15_s2", "own_only", "negative_only", "combined")
PATCH_ARMS = ARMS[1:]
GIB = 2 ** 30


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] * (hi - index) + values[hi] * (index - lo) if hi != lo else values[lo]


def paired_bootstrap(deltas):
    rng = random.Random(20261016)
    values = [statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000)]
    return {"difference_pp": 100 * statistics.mean(deltas),
            "user_paired_95ci_pp": [100 * percentile(values, .025), 100 * percentile(values, .975)],
            "resamples": 2000, "n_users": len(deltas), "seed": 20261016}


def score_records(records):
    """Pure scorer; each query label is consumed here, after inference only."""
    user_videos = {}
    arm_totals = {arm: {"correct": 0, "frames": 0} for arm in ARMS}
    videos = []
    for rec in records:
        n = len(rec["frames"])
        target = rec["target_index"]
        accuracy = {}
        for arm in ARMS:
            pred = rec["predictions"][arm]
            if len(pred) != n:
                raise ValueError("prediction length mismatch")
            correct = sum(p == target for p in pred)
            accuracy[arm] = correct / n
            arm_totals[arm]["correct"] += correct
            arm_totals[arm]["frames"] += n
        user_videos.setdefault(rec["user"], []).append(accuracy)
        videos.append({"user": rec["user"], "video": rec["video"], "frames": n,
                       "correct": {a: round(accuracy[a] * n) for a in ARMS},
                       "accuracy": accuracy})
    users = {u: {a: statistics.mean(v[a] for v in vs) for a in ARMS}
             for u, vs in sorted(user_videos.items())}
    metrics = {a: {**arm_totals[a], "frame_micro_accuracy": arm_totals[a]["correct"] / arm_totals[a]["frames"],
                   "video_macro_accuracy": statistics.mean(v["accuracy"][a] for v in videos),
                   "user_macro_accuracy": statistics.mean(u[a] for u in users.values())} for a in ARMS}
    comparisons = {}
    for reference in ("combined", "random_keep15_s0", "random_keep15_s1", "random_keep15_s2"):
        for candidate in ARMS:
            if candidate == reference:
                continue
            saved = harmed = both_correct = both_wrong = 0
            by_user = {}
            for rec in records:
                counts = by_user.setdefault(rec["user"], {"saved": 0, "harmed": 0, "frames": 0})
                for a, b in zip(rec["predictions"][candidate], rec["predictions"][reference]):
                    ca, cb = a == rec["target_index"], b == rec["target_index"]
                    saved += ca and not cb
                    harmed += cb and not ca
                    both_correct += ca and cb
                    both_wrong += not ca and not cb
                    counts["saved"] += int(ca and not cb)
                    counts["harmed"] += int(cb and not ca)
                    counts["frames"] += 1
            comparisons[f"{candidate}_minus_{reference}"] = {
                "candidate": candidate, "reference": reference, "saved": saved, "harmed": harmed,
                "both_correct": both_correct, "both_wrong": both_wrong,
                "frames": saved + harmed + both_correct + both_wrong, "per_user_counts": by_user,
                **paired_bootstrap([values[candidate] - values[reference] for values in users.values()])}
    return {"n_users": len(users), "n_videos": len(videos), "n_query_frames": sum(v["frames"] for v in videos),
            "metrics": metrics, "per_user_video_macro": users, "per_video": videos,
            "comparisons": comparisons, "bootstrap_unit": "user; macro over videos within user"}


def manifest_inputs(path, frames_root):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    users = manifest["users"]
    if not users or len({str(u["user"]) for u in users}) != len(users):
        raise ValueError("manifest requires unique nonempty users")
    names = set()
    for u in users:
        u["user"] = str(u["user"])
        objects = [str(o["object"]) for o in u["objects"]]
        if len(objects) < 2 or len(set(objects)) != len(objects):
            raise ValueError("each user requires >=2 unique enrolled objects")
        if len({q["video"] for q in u["queries"]}) != len(u["queries"]) or not u["queries"]:
            raise ValueError("query videos must be nonempty and unique per user")
        for obj in u["objects"]:
            obj["object"] = str(obj["object"])
            if not obj["context"]:
                raise ValueError("empty support object")
            names.update(obj["context"])
        for query in u["queries"]:
            if not query["frames"]:
                raise ValueError("empty query video")
            names.update(query["frames"])
    digest = hashlib.sha256()
    sources = []
    frames_root = frames_root.resolve()
    for name in sorted(names):
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError("frame names must be relative paths")
        path = (frames_root / name).resolve()
        if not path.is_relative_to(frames_root):
            raise ValueError("frame escapes --frames")
        entry = {"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
        sources.append(entry)
    return users, {"n_unique_files": len(sources), "bytes": sum(s["bytes"] for s in sources),
                   "ordered_frames_sha256": digest.hexdigest(), "frames": sources}


class Runner:
    def __init__(self, args):
        import numpy as np
        import psutil
        import torch
        import timm
        from safetensors.torch import load_file
        from timm.models.vision_transformer import checkpoint_filter_fn
        self.np, self.torch, self.psutil = np, torch, psutil
        self.args = args
        self.guard()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU or network fallback")
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        self.gpu_started = time.perf_counter()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.cap_gb * GIB / total, 1.0), device=0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False,
                                       img_size=224, num_classes=0)
        weights = checkpoint_filter_fn(load_file(str(args.checkpoint), device="cpu"), self.model)
        self.model.load_state_dict(weights, strict=True)
        del weights
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"], device="cuda").reshape(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device="cuda").reshape(1, 3, 1, 1)
        self.model = self.model.eval().cuda().to(memory_format=torch.channels_last)
        self.info = {"torch": torch.__version__, "timm": timm.__version__, "numpy": np.__version__,
                     "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                     "gpu_total_bytes": total, "mean": cfg["mean"], "std": cfg["std"],
                     "checkpoint_filter": "timm checkpoint_filter_fn (resamples pretrained position embedding)",
                     "same_frame_exclusion": False, "backbone_amp": "float16", "feature_storage": "float32",
                     "similarity_dtype": "float32; TF32 disabled", "preprocess": "PIL RGB; no resize; require 224x224"}

    def guard(self):
        available = self.psutil.virtual_memory().available
        if available < 2 * GIB:
            raise MemoryError(f"host RAM {available / GIB:.3f} GiB < 2 GiB; immediate stop, no wait")

    def resource(self):
        t = self.torch
        return {"allocated_gib": t.cuda.memory_allocated() / GIB, "reserved_gib": t.cuda.memory_reserved() / GIB,
                "peak_allocated_gib": t.cuda.max_memory_allocated() / GIB,
                "peak_reserved_gib": t.cuda.max_memory_reserved() / GIB,
                "rss_gib": self.psutil.Process().memory_info().rss / GIB,
                "available_ram_gib": self.psutil.virtual_memory().available / GIB}

    def progress(self, user, stage, started, **details):
        self.torch.cuda.synchronize()
        self.guard()
        print(json.dumps({"user": user, "stage": stage, "elapsed_s": time.perf_counter() - started,
                          **self.resource(), **details}), flush=True)

    def batches(self, names):
        from PIL import Image
        t, np = self.torch, self.np
        for start in range(0, len(names), self.args.batch):
            self.guard()
            batch = names[start:start + self.args.batch]
            images = []
            for name in batch:
                with Image.open(self.args.frames / name) as image:
                    if image.size != (224, 224):
                        raise ValueError(f"require original 224x224 frames, got {image.size}: {name}")
                    images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
            x = t.from_numpy(np.stack(images)).permute(0, 3, 1, 2).cuda().float().div_(255)
            x = ((x - self.mean) / self.std).contiguous(memory_format=t.channels_last)
            with t.inference_mode(), t.autocast("cuda", dtype=t.float16):
                tokens = self.model.forward_features(x)
            tokens = t.nn.functional.normalize(tokens.float(), dim=-1)
            if not bool(t.isfinite(tokens).all()):
                raise FloatingPointError("non-finite backbone features")
            prefix = self.model.num_prefix_tokens
            if tokens.shape != (len(batch), 257, 384) or prefix != 1:
                raise ValueError(f"unexpected feature geometry: {tokens.shape}, prefix {prefix}")
            cls = tokens[:, 0]
            patches = tokens[:, prefix:]
            del tokens, x, images
            yield start, cls, patches

    def kmeans(self, patches):
        t = self.torch
        g = t.Generator(device="cpu").manual_seed(0)
        indices = t.randperm(len(patches), generator=g)[:32].to(patches.device)
        centers = patches[indices].clone()
        if len(centers) != 32:
            raise ValueError("fewer than 32 support patches")
        for _ in range(15):
            self.guard()
            sums = t.zeros_like(centers)
            counts = t.zeros(32, device="cuda", dtype=t.long)
            for start in range(0, len(patches), 4096):
                block = patches[start:start + 4096]
                labels = (block @ centers.T).argmax(1)
                sums.index_add_(0, labels, block)
                counts += t.bincount(labels, minlength=32)
            nonempty = counts > 0
            centers[nonempty] = t.nn.functional.normalize(sums[nonempty] / counts[nonempty, None], dim=1)
        return centers

    def memories(self, flat):
        t, np = self.torch, self.np
        memories = {a: [] for a in PATCH_ARMS}
        diagnostics = []
        sizes = [len(p) for p in flat]
        for obj, source in enumerate(flat):
            self.guard()
            p = t.from_numpy(source).cuda()
            g = t.Generator(device="cpu").manual_seed(obj)
            others_idx = [j for j in range(len(flat)) if j != obj]
            total = sum(sizes[j] for j in others_idx)
            picks = t.randperm(total, generator=g)[:30000].numpy()
            sampled = []
            offset = 0
            for j in others_idx:
                indices = picks[(picks >= offset) & (picks < offset + sizes[j])] - offset
                sampled.append(flat[j][indices])
                offset += sizes[j]
            others = t.from_numpy(np.concatenate(sampled)).cuda()
            del sampled
            own_pick = t.randperm(len(p), generator=g)[:30000].to(p.device)
            own = p[own_pick]
            own_scores = np.empty(len(p), np.float32)
            other_scores = np.empty(len(p), np.float32)
            for start in range(0, len(p), self.args.chunk):
                self.guard()
                block = p[start:start + self.args.chunk]
                # top6-drop1 is a heuristic; same-frame neighbors remain eligible.
                own_scores[start:start + len(block)] = (block @ own.T).topk(6, dim=1).values[:, 1:].mean(1).cpu().numpy()
                other_scores[start:start + len(block)] = (block @ others.T).amax(1).cpu().numpy()
            del own, others, own_pick
            keep = max(32, math.ceil(.15 * len(p)))
            selection = {"unscreened32": None}
            for seed in (0, 1, 2):
                selection[f"random_keep15_s{seed}"] = np.random.default_rng(seed).choice(len(p), keep, replace=False)
            for arm, scores in (("own_only", own_scores), ("negative_only", -other_scores),
                                ("combined", own_scores - other_scores)):
                selection[arm] = np.argsort(-scores, kind="stable")[:keep]
            for arm, indices in selection.items():
                self.guard()
                # Original ordering gives every arm the same seed-0 row initialization rule.
                kept = p if indices is None else p[t.from_numpy(np.sort(indices)).to(p.device)]
                memories[arm].append(self.kmeans(kept))
                if indices is not None:
                    del kept
            diagnostics.append({"object_index": obj, "support_patches": len(p), "retained_15pct": keep,
                                "actual_retained_fraction": keep / len(p), "sampled_own": min(30000, len(p)),
                                "sampled_other": min(30000, total), "prototypes_per_arm": 32,
                                "own_only_vs_combined_overlap": int(np.intersect1d(selection["own_only"], selection["combined"]).size),
                                "negative_only_vs_combined_overlap": int(np.intersect1d(selection["negative_only"], selection["combined"]).size)})
            del p
        return {a: t.stack(memories[a]) for a in PATCH_ARMS}, diagnostics

    def infer_video(self, names, whole, memories):
        """Classifier sees frames only, never their target identities."""
        t = self.torch
        predictions = {a: [] for a in ARMS}
        n_objects = len(whole)
        for _, cls, patches in self.batches(names):
            self.guard()
            c, p = cls, patches
            predictions["wholeframeCLS"].extend((c @ whole.T).argmax(1).cpu().tolist())
            for arm, centers in memories.items():
                sim = (p @ centers.reshape(n_objects * 32, 384).T)
                per_obj = sim.reshape(len(p), 256, n_objects, 32).amax(3)
                scores = per_obj.topk(8, dim=1).values.mean(1)
                predictions[arm].extend(scores.argmax(1).cpu().tolist())
                del sim, per_obj, scores
            del c, p
        return predictions

    def user(self, user, config_hash):
        t, np = self.torch, self.np
        started = time.perf_counter()
        t.cuda.reset_peak_memory_stats()
        self.progress(user["user"], "start", started)
        flat, means = [], []
        n_context = 0
        support_started = time.perf_counter()
        for obj in user["objects"]:
            names = obj["context"]
            self.guard()
            storage = np.empty((len(names), 256, 384), dtype=np.float32)
            total = np.zeros(384, dtype=np.float32)
            for start, cls, patches in self.batches(names):
                storage[start:start + len(cls)] = patches.cpu().numpy()
                total += cls.cpu().numpy().sum(0)
            flat.append(storage.reshape(-1, 384))
            means.append(total / len(names))
            n_context += len(names)
        support_seconds = time.perf_counter() - support_started
        whole = t.nn.functional.normalize(t.from_numpy(np.stack(means)).cuda(), dim=1)
        memory_started = time.perf_counter()
        memories, diagnostic = self.memories(flat)
        t.cuda.synchronize()
        memory_seconds = time.perf_counter() - memory_started
        del flat, storage, means
        self.progress(user["user"], "support_complete", started, context_frames=n_context,
                      objects=len(user["objects"]), support_extract_s=support_seconds, memory_build_s=memory_seconds)
        records = []
        objects = [o["object"] for o in user["objects"]]
        query_started = time.perf_counter()
        for index, query in enumerate(user["queries"]):
            prediction = self.infer_video(query["frames"], whole, memories)
            # Scorer-only label materialization occurs after this video's predictions.
            records.append({"user": user["user"], "video": query["video"], "frames": query["frames"],
                            "target_index": objects.index(str(query["object"])), "predictions": prediction})
            self.progress(user["user"], "video_complete", started, video=query["video"],
                          completed_videos=index + 1, total_videos=len(user["queries"]))
        t.cuda.synchronize()
        result = {"config_sha256": config_hash, "user": user["user"], "objects": objects,
                  "context_frames": n_context, "memory_diagnostics": diagnostic, "predictions": records,
                  "timing_s": {"support_extract": support_seconds, "memory_build": memory_seconds,
                               "query_extract_and_all_arms": time.perf_counter() - query_started,
                               "user_wall": time.perf_counter() - started}, "resources": self.resource()}
        del whole, memories
        t.cuda.empty_cache()
        self.progress(user["user"], "complete", started)
        return result


def self_test():
    # Two users, three videos: video-macro and user-macro intentionally differ.
    records = []
    for user, video, values in (("a", "a0", [0, 0]), ("a", "a1", [1]), ("b", "b0", [1, 1])):
        predictions = {a: list(values) for a in ARMS}
        predictions["combined"] = [0] * len(values)
        records.append({"user": user, "video": video, "frames": ["x"] * len(values),
                        "target_index": 0, "predictions": predictions})
    result = score_records(records)
    metric = result["metrics"]["wholeframeCLS"]
    assert metric["correct"] == 2 and metric["frames"] == 5
    assert abs(metric["video_macro_accuracy"] - 1 / 3) < 1e-12
    assert metric["user_macro_accuracy"] == .25
    comparison = result["comparisons"]["wholeframeCLS_minus_combined"]
    assert comparison["saved"] == 0 and comparison["harmed"] == 3
    assert comparison["difference_pp"] == -75
    print("stdlib scorer fixture PASS: 2 users / 3 videos / 5 frames; no GPU imports")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cap-gb", type=float, default=2.5)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--users", default="", help="comma-separated execution subset; safe to omit on resume")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ("manifest", "frames", "checkpoint", "out")):
        parser.error("--manifest --frames --checkpoint --out are required")
    if not 1 <= args.batch <= 32 or not 0 < args.cap_gb <= 2.5 or not 1 <= args.chunk <= 2048:
        parser.error("require batch<=32, cap-gb<=2.5, chunk<=2048 (all positive)")
    started = time.perf_counter()
    users, frame_receipt = manifest_inputs(args.manifest, args.frames)
    selected = set(args.users.split(",")) if args.users else {u["user"] for u in users}
    if not selected.issubset({u["user"] for u in users}):
        parser.error("--users contains unknown user")
    config = {"schema": 1, "manifest_sha256": sha256(args.manifest), "source_sha256": sha256(__file__),
              "checkpoint_sha256": sha256(args.checkpoint), "frames_sha256": frame_receipt["ordered_frames_sha256"],
              "frames_root": str(args.frames.resolve()), "architecture": "vit_small_patch14_dinov2.lvd142m",
              "batch": args.batch, "cap_gib": args.cap_gb, "matrix_chunk": args.chunk, "arms": ARMS,
              "parts": 32, "pool_topk": 8, "kmeans_iters": 15, "kmeans_seed": 0, "keep_fraction": .15,
              "sample_limit": 30000, "sample_seed": "object ordinal; CPU torch.Generator",
              "same_frame_exclusion": False, "selection": "exact ceil15%, min32, stable ties by patch index",
              "users": [u["user"] for u in users], "input_unique_frames": frame_receipt["n_unique_files"]}
    config_hash = json_hash(config)
    args.out.mkdir(parents=True, exist_ok=True)
    run_path = args.out / "run.json"
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["config_sha256"] != config_hash:
            raise RuntimeError("output already belongs to different configuration/source/input bytes; use a new --out")
    else:
        if any(args.out.iterdir()):
            raise RuntimeError("refuse nonempty output without matching run.json")
        run = {"config_sha256": config_hash, "config": config, "invocations": []}
        save_json(run_path, run)
        save_json(args.out / "inputs.json", {"manifest": str(args.manifest.resolve()),
                  "checkpoint": str(args.checkpoint.resolve()), **frame_receipt})
    invocation = {"selected_users": sorted(selected), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "status": "running"}
    run["invocations"].append(invocation)
    save_json(run_path, run)
    runner = None
    completed = []
    try:
        for user in users:
            target = args.out / ("user_" + hashlib.sha256(user["user"].encode()).hexdigest()[:16] + ".json")
            if target.exists():
                record = json.loads(target.read_text(encoding="utf-8"))
                if record["config_sha256"] != config_hash or record["user"] != user["user"]:
                    raise RuntimeError("invalid completed-user receipt")
                completed.append(record)
                print(json.dumps({"user": user["user"], "stage": "resume_skip_completed"}), flush=True)
                continue
            if user["user"] not in selected:
                continue
            if runner is None:
                runner = Runner(args)
                if "runtime" in run and json_hash(run["runtime"]) != json_hash(runner.info):
                    raise RuntimeError("runtime changed since completed-user receipts; use a new --out")
                run["runtime"] = runner.info
                save_json(run_path, run)
            record = runner.user(user, config_hash)
            save_json(target, record)
            completed.append(record)
        if completed:
            summary = score_records([v for u in completed for v in u["predictions"]])
            summary.update({"config_sha256": config_hash, "expected_users": len(users),
                            "complete": len(completed) == len(users), "development_only": True,
                            "n_context_frames": sum(u["context_frames"] for u in completed),
                            "actual_user_wall_s": sum(u["timing_s"]["user_wall"] for u in completed)})
            save_json(args.out / "summary.json", summary)
        invocation["status"] = "complete" if len(completed) == len(users) else "partial_execution_subset"
    except BaseException as exc:
        invocation.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                           "traceback": traceback.format_exc()})
        raise
    finally:
        if runner is not None:
            runner.torch.cuda.synchronize()
            invocation["resources"] = runner.resource()
            invocation["gpu_job_wall_s"] = time.perf_counter() - runner.gpu_started
            del runner.model
            runner.torch.cuda.empty_cache()
        else:
            invocation["gpu_job_wall_s"] = 0
        invocation["actual_invocation_wall_s"] = time.perf_counter() - started
        invocation["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_json(run_path, run)


if __name__ == "__main__":
    main()
