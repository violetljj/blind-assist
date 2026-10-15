#!/usr/bin/env python3
"""CUDA-only, resumable Grounding DINO + CLIP candidate Development pilot.

The inference adapter selects only image/query/id/timestamp fields from the input
manifest. Ground truth is never passed to either model or candidate selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def read_outputs(path):
    output = {}
    if not path.exists():
        return output
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                key = (str(row["sequence_id"]), int(row["frame_idx"]))
                if key in output:
                    raise ValueError(f"Duplicate output key {key} in {path}")
                output[key] = row
    return output


def observations(path, limit):
    rows, seen = [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            # Explicit allowlist: no evaluator field enters inference records.
            row = {name: raw.get(name) for name in
                   ("sequence_id", "frame_idx", "image_path", "query", "timestamp_s")}
            row["sequence_id"], row["frame_idx"] = str(row["sequence_id"]), int(row["frame_idx"])
            key = (row["sequence_id"], row["frame_idx"])
            if key in seen or not row["query"] or not row["image_path"]:
                raise ValueError(f"Invalid observation {key}")
            seen.add(key)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def pooled(output):
    # Transformers 5 returns projected features in pooler_output; older builds
    # return a Tensor. Preserve support for the inspected installed API.
    return output if hasattr(output, "norm") else output.pooler_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=1650)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "writer.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    stage, completed, total = "startup", 0, 0
    prior_elapsed, stopped = 0.0, threading.Event()
    config_hash = None
    base_path, clip_path = root / "base_predictions.jsonl", root / "clip_predictions.jsonl"
    progress_path, terminal_path = root / "progress.json", root / "terminal.json"

    def elapsed():
        return prior_elapsed + time.monotonic() - started

    def progress():
        return {"pid": os.getpid(), "stage": stage, "completed": completed,
                "total": total, "elapsed_s": elapsed(), "config_sha256": config_hash,
                "last_activity_unix_s": time.time(), "eta_s": None}

    def event(kind, **kwargs):
        append_json(root / "events.jsonl", {"event": kind, "elapsed_s": elapsed(), **kwargs})

    def watchdog():
        while not stopped.wait(0.5):
            if elapsed() >= args.max_seconds:
                state = {**progress(), "status": "TIME_BUDGET_EXHAUSTED",
                         "limit_s": args.max_seconds, "partial_resumable": True}
                atomic_json(terminal_path, state)
                print(json.dumps(state), flush=True)
                # Preserve valid completed JSONL rows; release process-owned lock.
                lock.unlink(missing_ok=True)
                os._exit(124)

    try:
        config = {"manifest_sha256": sha256(args.manifest), "script_sha256": sha256(__file__),
                  "model_dir": str(args.model_dir.resolve()), "clip_dir": str(args.clip_dir.resolve()),
                  "model_revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
                  "clip_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                  "input_size": {"shortest_edge": 800, "longest_edge": 1333},
                  "oom_only_fallback": {"shortest_edge": 640, "longest_edge": 1067},
                  "candidate_threshold": 0.05, "text_threshold": 0.25,
                  "presence_threshold": 0.25, "nms_iou": 0.5, "candidate_topk": 30,
                  "clip_rerank_eligible": "DINO score >= 0.25; fixed pool and presence",
                  "clip_crop": "tight xyxy PIL crop, no GT/context expansion",
                  "limit": args.limit, "precision": "float16 CUDA autocast",
                  "max_seconds": args.max_seconds}
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        config_path = root / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Config drift: use a new output directory, preserve prior run")
        atomic_json(config_path, config)
        if progress_path.exists():
            old = json.loads(progress_path.read_text())
            prior_elapsed = float(old.get("elapsed_s", 0))
        if prior_elapsed >= args.max_seconds:
            raise TimeoutError("Saved cumulative budget is already exhausted")
        threading.Thread(target=watchdog, daemon=True).start()
        rows = observations(args.manifest, args.limit)
        total = len(rows)
        base_done, clip_done = read_outputs(base_path), read_outputs(clip_path)
        completed = len(base_done.keys() & clip_done.keys())
        stage = "imports"
        atomic_json(progress_path, progress())
        import numpy as np
        import torch
        import transformers
        from PIL import Image
        from torchvision.ops import nms
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPModel

        torch.set_num_threads(args.threads)
        if not torch.cuda.is_available():
            raise RuntimeError("ACCELERATOR_UNAVAILABLE: this run requires actual CUDA")
        device = torch.device("cuda")
        torch.manual_seed(0)
        stage = "model_loading"
        event("loading_models")
        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.model_dir, local_files_only=True, dtype=torch.float16).to(device).eval()
        clip_processor = AutoProcessor.from_pretrained(args.clip_dir, local_files_only=True)
        clip = CLIPModel.from_pretrained(
            args.clip_dir, local_files_only=True, dtype=torch.float16).to(device).eval()
        runtime = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                   "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0), "backend": "CUDA",
                   "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "pid": os.getpid(), "config_sha256": config_hash}
        atomic_json(root / "runtime.json", runtime)
        event("models_loaded", **runtime)
        feature_dir = root / "features"
        feature_dir.mkdir(exist_ok=True)
        query_cache = {}
        active_size = dict(config["input_size"])
        if (root / "resolution_fallback.json").exists():
            active_size = json.loads((root / "resolution_fallback.json").read_text())["size"]

        def grounding(image, query, size):
            # Official full-description input convention: lowercase, final period.
            text = query.strip().lower()
            if not text.endswith("."):
                text += "."
            inputs = processor(images=image, text=text, return_tensors="pt", size=size).to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=0.05, text_threshold=0.25,
                target_sizes=[(image.height, image.width)])[0]
            boxes, scores = result["boxes"].float(), result["scores"].float()
            labels = result.get("text_labels", result.get("labels", [""] * len(boxes)))
            valid = torch.isfinite(boxes).all(1) & torch.isfinite(scores)
            boxes[:, 0::2].clamp_(0, image.width)
            boxes[:, 1::2].clamp_(0, image.height)
            valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            ids = torch.where(valid)[0]
            kept = nms(boxes[ids], scores[ids], 0.5)[:30]
            ids = ids[kept].cpu().tolist()
            return [{"bbox_xyxy": boxes[i].cpu().tolist(), "score": float(scores[i]),
                     "text_label": str(labels[i])} for i in ids], len(boxes) - int(valid.sum())

        def rerank(image, query, candidates):
            if query not in query_cache:
                inputs = clip_processor(text=[query], return_tensors="pt", padding=True,
                                        truncation=True).to(device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    features = pooled(clip.get_text_features(**inputs)).float()
                    features /= features.norm(dim=-1, keepdim=True)
                query_cache[query] = features
            text_features = query_cache[query]
            if not candidates:
                return [], np.empty((0, text_features.shape[-1]), dtype=np.float16), text_features
            crops = [image.crop(tuple(c["bbox_xyxy"])) for c in candidates]
            inputs = clip_processor(images=crops, return_tensors="pt").to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                image_features = pooled(clip.get_image_features(**inputs)).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarities = (image_features @ text_features.T).squeeze(1).cpu().tolist()
            ranked = []
            for candidate, cosine in zip(candidates, similarities):
                # Eligible range [1,3] is always before ineligible [-3,-1].
                rank = float(cosine) + (2 if candidate["score"] >= 0.25 else -2)
                ranked.append({**candidate, "clip_cosine": float(cosine), "ranking_score": rank})
            ranked.sort(key=lambda c: (-c["ranking_score"], -c["score"], tuple(c["bbox_xyxy"])))
            return ranked, image_features.cpu().numpy().astype(np.float16), text_features

        first_probe = not (root / "first_frame_probe.json").exists()
        stage = "inference"
        for ordinal, row in enumerate(rows):
            key = (row["sequence_id"], row["frame_idx"])
            if key in base_done and key in clip_done:
                continue
            if elapsed() >= args.max_seconds - 1:
                raise TimeoutError("Cumulative wall budget exhausted before next frame")
            atomic_json(progress_path, {**progress(), "current_key": list(key)})
            tick = time.monotonic()
            image = Image.open(row["image_path"]).convert("RGB")
            if first_probe:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            dino_tick = time.monotonic()
            try:
                candidates, invalid_count = grounding(image, row["query"], active_size)
            except torch.cuda.OutOfMemoryError:
                if active_size != config["input_size"]:
                    raise
                event("cuda_oom", key=list(key), resolution=active_size)
                torch.cuda.empty_cache()
                active_size = dict(config["oom_only_fallback"])
                atomic_json(root / "resolution_fallback.json", {"size": active_size, "key": list(key)})
                candidates, invalid_count = grounding(image, row["query"], active_size)
            torch.cuda.synchronize()
            dino_seconds = time.monotonic() - dino_tick
            clip_tick = time.monotonic()
            reranked, crop_features, text_features = rerank(image, row["query"], candidates)
            torch.cuda.synchronize()
            clip_seconds = time.monotonic() - clip_tick
            presence = max((c["score"] for c in candidates), default=0.0)
            feature_name = hashlib.sha256((key[0] + "\0" + str(key[1])).encode()).hexdigest()[:24] + ".npz"
            np.savez_compressed(feature_dir / feature_name,
                                boxes=np.asarray([c["bbox_xyxy"] for c in candidates], dtype=np.float32).reshape(-1, 4),
                                dino_scores=np.asarray([c["score"] for c in candidates], dtype=np.float32),
                                crop_embeddings=crop_features,
                                query_embedding=text_features.cpu().numpy().astype(np.float16))
            common = {**row, "width": image.width, "height": image.height,
                      "presence_score": presence, "model": "IDEA-Research/grounding-dino-base",
                      "resolution": active_size, "invalid_boxes_dropped": invalid_count,
                      "feature_cache": str((feature_dir / feature_name).resolve()),
                      "config_sha256": config_hash}
            base = {**common, "candidates": candidates, "elapsed_s": dino_seconds,
                    "selection_method": "max_grounding_dino_score"}
            ranked = {**common, "candidates": reranked, "elapsed_s": dino_seconds + clip_seconds,
                      "clip_elapsed_s": clip_seconds, "selection_method": "clip_crop_text_cosine_eligible_dino_ge_025"}
            if key not in base_done:
                append_json(base_path, base)
            if key not in clip_done:
                append_json(clip_path, ranked)
            base_done[key], clip_done[key] = base, ranked
            completed = len(base_done.keys() & clip_done.keys())
            if first_probe:
                probe = {"key": list(key), "dino_s": dino_seconds, "clip_s": clip_seconds,
                         "total_frame_s": time.monotonic() - tick, "candidate_count": len(candidates),
                         "resolution": active_size, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "backend": "CUDA"}
                atomic_json(root / "first_frame_probe.json", probe)
                event("first_frame_probe", **probe)
                first_probe = False
            atomic_json(progress_path, progress())
            if completed % 20 == 0 or completed == 1 or completed == total:
                print(json.dumps({**progress(), "last_dino_s": dino_seconds, "last_clip_s": clip_seconds}), flush=True)
        stage = "finalizing"
        hashes = {str(path.relative_to(root)): sha256(path) for path in
                  (base_path, clip_path, root / "config.json", root / "runtime.json") if path.exists()}
        atomic_json(root / "hash_receipt.json", {"config_sha256": config_hash, "files": hashes,
                    "features_count": len(list(feature_dir.glob("*.npz"))),
                    "source_script_sha256": sha256(__file__), "manifest_sha256": sha256(args.manifest)})
        stage = "complete"
        atomic_json(progress_path, progress())
        result = {**progress(), "status": "SUCCESS", "partial_resumable": True,
                  "method_limits": ["Pretrained accessible grounding baseline, not full RVOS SOTA",
                                    "Tight-crop CLIP heuristic is not a trained novel method",
                                    "No GT read by inference; no FPS inferred from frame indices"]}
        atomic_json(terminal_path, result)
        print(json.dumps(result), flush=True)
        return 0
    except BaseException as exc:
        stage = "failed"
        details = {**progress(), "status": "TIME_BUDGET_EXHAUSTED" if isinstance(exc, TimeoutError) else "FAILED",
                   "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                   "partial_resumable": True}
        append_json(root / "errors.jsonl", details)
        atomic_json(progress_path, progress())
        atomic_json(terminal_path, details)
        print(json.dumps(details), flush=True)
        return 1
    finally:
        stopped.set()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
