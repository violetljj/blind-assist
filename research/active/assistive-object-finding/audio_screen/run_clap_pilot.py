#!/usr/bin/env python3
"""Frozen-prompt CLAP inference on WAV observations; ground truth stays outside.

Observations: sample_id,audio_path,source_id,variant,offset_s,duration_s.
Prompts: {classes:[{class_id: string,label: string,prompts:[string,...]}]}.
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


REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_json(path, data):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def observations(path, limit):
    rows, seen = [], set()
    for raw in read_rows(path):
        # Explicit allowlist prevents labels or known/unknown partitions entering inference.
        row = {key: raw.get(key) for key in ("sample_id", "audio_path", "source_id", "variant",
                                            "offset_s", "duration_s")}
        row["sample_id"] = str(row["sample_id"])
        if row["sample_id"] in seen or not row["audio_path"]:
            raise ValueError("Duplicate id or missing audio_path")
        seen.add(row["sample_id"])
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def expand_views(rows):
    from scipy.io import wavfile
    expanded = []
    for row in rows:
        rate, values = wavfile.read(row["audio_path"], mmap=True)
        offset = float(row.get("offset_s") or 0)
        available = len(values) / rate - offset
        duration = min(available, float(row["duration_s"])) if row.get("duration_s") is not None else available
        expanded.append({**row, "view": "whole", "window_idx": None,
                         "unit_id": row["sample_id"] + "::whole", "view_start_s": 0.0})
        if duration + 1e-6 >= 5.0:
            for index, start in enumerate((0.0, 1.25, 2.5)):
                expanded.append({**row, "offset_s": offset + start, "duration_s": 2.5,
                                 "view": "window", "window_idx": index, "view_start_s": start,
                                 "unit_id": row["sample_id"] + "::window" + str(index)})
    return expanded


def read_audio(row):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    sample_rate, values = wavfile.read(row["audio_path"])
    channels = 1 if values.ndim == 1 else values.shape[1]
    if values.dtype.kind == "u":
        middle = float(2 ** (values.dtype.itemsize * 8 - 1))
        audio = (values.astype(np.float32) - middle) / middle
    elif values.dtype.kind == "i":
        audio = values.astype(np.float32) / float(2 ** (values.dtype.itemsize * 8 - 1))
    elif values.dtype.kind == "f":
        audio = values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV dtype {values.dtype}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    offset = float(row.get("offset_s") or 0)
    if offset < 0:
        raise ValueError("offset_s must be nonnegative")
    begin = round(offset * sample_rate)
    duration = row.get("duration_s")
    end = begin + round(float(duration) * sample_rate) if duration is not None else len(audio)
    audio = audio[begin:end]
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty or nonfinite waveform")
    if sample_rate != 48000:
        divisor = math.gcd(int(sample_rate), 48000)
        audio = resample_poly(audio, 48000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    # Deterministic head selection prevents the official random truncation branch.
    audio = audio[:480000]
    details = {"original_rate_hz": int(sample_rate), "original_channels": channels,
               "input_dtype": str(values.dtype), "selected_samples_48khz": len(audio)}
    details["peak_absolute"] = float(np.abs(audio).max())
    details["rms"] = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return np.asarray(audio, dtype=np.float32), details


def pool(value):
    return value if hasattr(value, "norm") else value.pooler_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=840)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "writer.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    stage, count, total, prior_elapsed = "startup", 0, 0, 0.0
    stopped = threading.Event()
    config_hash = None
    terminal = root / "terminal.json"
    prediction_path = root / "predictions.jsonl"
    window_path = root / "window_predictions.jsonl"

    def elapsed():
        return prior_elapsed + time.perf_counter() - started

    def progress():
        return {"pid": os.getpid(), "stage": stage, "completed": count, "total": total,
                "elapsed_s": elapsed(), "last_activity_unix_s": time.time(), "config_sha256": config_hash}

    def watchdog():
        while not stopped.wait(0.5):
            if elapsed() >= args.max_seconds:
                result = {**progress(), "status": "TIME_BUDGET_EXHAUSTED", "partial_resumable": True}
                write_json(terminal, result)
                lock.unlink(missing_ok=True)
                print(json.dumps(result), flush=True)
                os._exit(124)

    try:
        config = {"model": "laion/clap-htsat-unfused", "revision": REVISION,
                  "model_dir": str(args.model_dir.resolve()), "manifest_sha256": sha256(args.manifest),
                  "prompts_sha256": sha256(args.prompts), "script_sha256": sha256(__file__),
                  "batch_size": args.batch_size, "limit": args.limit, "max_seconds": args.max_seconds,
                  "requested_device": args.device, "dtype": "float32", "seed": 0,
                  "preprocessing": "PCM dtype scaling; stereo mean; scipy polyphase 48k; head<=10s; official repeatpad",
                  "amplitude": "No peak/RMS normalization, clipping, denoising, or silence deletion",
                  "main_score": "First fixed template cosine per class; primary top1 excludes diagnostic_only classes",
                  "ensemble_score": "Mean L2-normalized prompt embeddings then class L2 normalization",
                  "views": "Whole plus 2.5s windows at 0,1.25,2.5s for clips >=5s; shorter use whole only",
                  "cosine": "L2-normalized audio embedding dot class embedding; no softmax",
                  "timing_clock": "perf_counter", "timing_resolution_s": time.get_clock_info("perf_counter").resolution}
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        config_path = root / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Config drift: preserve run and use a new output directory")
        write_json(config_path, config)
        for saved in (root / "progress.json", terminal):
            if saved.exists():
                prior_elapsed = max(prior_elapsed, float(json.loads(saved.read_text()).get("elapsed_s", 0)))
        if prior_elapsed >= args.max_seconds:
            raise TimeoutError("Cumulative saved budget exhausted")
        threading.Thread(target=watchdog, daemon=True).start()
        samples = observations(args.manifest, args.limit)
        rows = expand_views(samples)
        total = len(rows)
        classes = json.loads(args.prompts.read_text(encoding="utf-8-sig"))["classes"]
        ids = [str(c["class_id"]) for c in classes]
        primary_indices = [i for i, c in enumerate(classes) if not c.get("diagnostic_only", False)]
        if len(ids) != len(set(ids)) or not ids or any(not c["prompts"] for c in classes):
            raise ValueError("Class ids must be unique with nonempty fixed prompts")
        previous = []
        for saved in (prediction_path, window_path):
            if saved.exists():
                previous.extend(read_rows(saved))
        existing = {str(row["unit_id"]): row for row in previous}
        if len(existing) != len(previous):
            raise ValueError("Duplicate saved inference unit")
        count = len(existing)
        stage = "imports"
        write_json(root / "progress.json", progress())
        import numpy as np
        import scipy
        import torch
        import transformers
        from transformers import ClapModel, ClapProcessor
        torch.set_num_threads(4)
        torch.manual_seed(0)
        np.random.seed(0)
        available = torch.cuda.is_available()
        memory_before = torch.cuda.mem_get_info() if available else None
        cuda_suitable = available and memory_before[0] >= 3_000_000_000
        if args.device == "cuda" and not cuda_suitable:
            raise RuntimeError(f"Requested CUDA cannot reserve 3GB free VRAM: {memory_before}")
        if args.device != "cpu" and not cuda_suitable:
            raise RuntimeError(f"CUDA_VRAM_BELOW_3GB: stop without automatic CPU fallback, free/total={memory_before}")
        use_cuda = args.device != "cpu"
        device = torch.device("cuda" if use_cuda else "cpu")
        reason = "CUDA_AVAILABLE_WITH_3GB_FREE" if use_cuda else "ACCELERATOR_UNAVAILABLE"
        if use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
        stage = "model_loading"
        processor = ClapProcessor.from_pretrained(args.model_dir, local_files_only=True)
        model = ClapModel.from_pretrained(args.model_dir, local_files_only=True,
                                          dtype=torch.float32).to(device).eval()
        runtime = {"backend": str(device), "placement_reason": reason,
                   "cuda_free_total_before_bytes": list(memory_before) if memory_before else None,
                   "device": torch.cuda.get_device_name() if use_cuda else platform.processor(),
                   "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                   "scipy": scipy.__version__, "numpy": np.__version__, "parameters": sum(p.numel() for p in model.parameters()),
                   "dtype": "float32", "config_sha256": config_hash}
        write_json(root / "runtime.json", runtime)
        stage = "text_features"
        flat_prompts = [prompt for c in classes for prompt in c["prompts"]]
        tokens = processor.tokenizer(flat_prompts, return_tensors="pt", padding=True,
                                     truncation=True, max_length=512).to(device)
        tick = time.perf_counter()
        with torch.inference_mode():
            prompt_features = pool(model.get_text_features(**tokens)).float()
            prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
            class_features, offset = [], 0
            for c in classes:
                vector = prompt_features[offset:offset + len(c["prompts"])].mean(0)
                class_features.append(vector / vector.norm())
                offset += len(c["prompts"])
            class_features = torch.stack(class_features)
            first_indices, offset = [], 0
            for c in classes:
                first_indices.append(offset)
                offset += len(c["prompts"])
            first_features = prompt_features[first_indices]
        if use_cuda:
            torch.cuda.synchronize()
        text_seconds = time.perf_counter() - tick
        np.savez_compressed(root / "text_embeddings.npz", class_ids=np.asarray(ids),
                            class_embeddings=class_features.cpu().numpy(),
                            prompts=np.asarray(flat_prompts), prompt_embeddings=prompt_features.cpu().numpy())
        feature_dir = root / "embeddings"
        feature_dir.mkdir(exist_ok=True)
        todo = [row for row in rows if row["unit_id"] not in existing]
        stage = "audio_inference"
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            tick = time.perf_counter()
            loaded = [read_audio(row) for row in batch]
            load_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            features = processor.feature_extractor([item[0] for item in loaded], sampling_rate=48000,
                        padding="repeatpad", truncation="rand_trunc", return_tensors="pt").to(device)
            preprocess_seconds = time.perf_counter() - tick
            if use_cuda:
                torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode():
                audio_features = pool(model.get_audio_features(**features)).float()
                audio_features /= audio_features.norm(dim=-1, keepdim=True)
                first_scores = audio_features @ first_features.T
                ensemble_scores = audio_features @ class_features.T
                prompt_scores = audio_features @ prompt_features.T
            if use_cuda:
                torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - tick
            embeddings, cosine = audio_features.cpu().numpy(), first_scores.cpu().numpy()
            ensemble, per_prompt = ensemble_scores.cpu().numpy(), prompt_scores.cpu().numpy()
            if not np.isfinite(embeddings).all() or not np.isfinite(cosine).all():
                raise ValueError("Nonfinite embeddings or class cosine")
            for index, row in enumerate(batch):
                feature_path = feature_dir / (hashlib.sha256(row["unit_id"].encode()).hexdigest()[:24] + ".npy")
                np.save(feature_path, embeddings[index])
                ranking = np.asarray(primary_indices)[np.argsort(-cosine[index, primary_indices], kind="stable")]
                prompt_cosines, offset = {}, 0
                for c in classes:
                    prompt_cosines[str(c["class_id"])] = per_prompt[index, offset:offset + len(c["prompts"])].astype(float).tolist()
                    offset += len(c["prompts"])
                result = {**row, "cosine_scores": dict(zip(ids, cosine[index].astype(float).tolist())),
                          "ensemble_cosine_scores": dict(zip(ids, ensemble[index].astype(float).tolist())),
                          "prompt_cosine_scores": prompt_cosines,
                          "top1_class_id": ids[int(ranking[0])], "top1_cosine": float(cosine[index, ranking[0]]),
                          "top1_margin": float(cosine[index, ranking[0]] - cosine[index, ranking[1]]) if len(ids) > 1 else None,
                          "audio_embedding_path": str(feature_path), "preprocessing": loaded[index][1],
                          "batch_size": len(batch), "batch_audio_io_s": load_seconds,
                          "batch_feature_extraction_s": preprocess_seconds, "batch_model_forward_s": forward_seconds,
                          "elapsed_s": (load_seconds + preprocess_seconds + forward_seconds) / len(batch),
                          "backend": str(device), "config_sha256": config_hash}
                append_json(prediction_path if row["view"] == "whole" else window_path, result)
                existing[row["unit_id"]] = result
            count = len(existing)
            write_json(root / "progress.json", progress())
            if not (root / "first_batch_probe.json").exists():
                write_json(root / "first_batch_probe.json", {**progress(), "batch_size": len(batch),
                           "audio_io_s": load_seconds, "feature_extraction_s": preprocess_seconds,
                           "forward_s": forward_seconds, "text_features_s": text_seconds,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated() if use_cuda else None,
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved() if use_cuda else None})
            if count <= args.batch_size or count % 20 < args.batch_size or count == total:
                print(json.dumps(progress()), flush=True)
        stage = "complete"
        receipt = {**progress(), "status": "SUCCESS", "partial_resumable": True,
                   "sample_count": len(samples), "class_count": len(classes), "prompt_count": len(flat_prompts),
                   "hashes": {path.name: sha256(path) for path in
                              (prediction_path, window_path, config_path, root / "runtime.json", root / "text_embeddings.npz") if path.exists()},
                   "embedding_files": len(list(feature_dir.glob("*.npy")))}
        write_json(root / "progress.json", progress())
        write_json(terminal, receipt)
        print(json.dumps(receipt), flush=True)
        return 0
    except BaseException as exc:
        stage = "failed"
        details = {**progress(), "status": "FAILED", "error": str(exc),
                   "error_type": type(exc).__name__, "traceback": traceback.format_exc()}
        append_json(root / "errors.jsonl", details)
        write_json(root / "progress.json", progress())
        write_json(terminal, details)
        print(json.dumps(details), flush=True)
        return 1
    finally:
        stopped.set()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
