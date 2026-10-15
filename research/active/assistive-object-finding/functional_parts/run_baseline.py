#!/usr/bin/env python3
"""Actual SelectiveCL inference, four fixed image views, no evaluator truth.

The original vendor test_forward and all three raw heads are retained. The only
constructor adapters skip redundant pretrained downloads already in the full
checkpoint and isolate a training-only kmeans import. No feature/math changes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import types

HEADS = ("raw", "ree", "rem")
ACTIONS = ['beat', 'boxing', 'brush_with', 'carry', 'catch', 'cut', 'cut_with', 'drag',
           'drink_with', 'eat', 'hit', 'hold', 'jump', 'kick', 'lie_on', 'lift', 'look_out',
           'open', 'pack', 'peel', 'pick_up', 'pour', 'push', 'ride', 'sip', 'sit_on', 'stick',
           'stir', 'swing', 'take_photo', 'talk_on', 'text_on', 'throw', 'type_on', 'wash', 'write']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers can briefly deny atomic replacement of the destination.
    # Preserve the payload and retry only this observed sharing violation.
    for attempt, delay in enumerate((0.0, 0.02, 0.05, 0.1, 0.2, 0.4)):
        if delay:
            time.sleep(delay)
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def observe(path, limit):
    result, seen = [], set()
    for raw in rows(path):
        action = raw.get("action", raw.get("query_action"))
        # No GT paths, pixel labels, native boxes or evaluation partition enters.
        row = {name: raw.get(name) for name in ("sample_id", "image_path", "source_id", "image_sha256")}
        row["action"] = str(action)
        row["sample_id"] = str(row["sample_id"])
        if row["sample_id"] in seen or action not in ACTIONS or not row["image_path"]:
            raise ValueError(f"Invalid observation or Seen action: {row}")
        seen.add(row["sample_id"])
        result.append(row)
        if limit and len(result) >= limit:
            break
    return result


def load_official(vendor, weights, output):
    import torch
    sys.path.insert(0, str(vendor))
    adapters = []
    if importlib.util.find_spec("fast_pytorch_kmeans") is None:
        package = types.ModuleType("fast_pytorch_kmeans")

        class TrainingKMeansUnavailable:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Training-only KMeans must not be called by inference")

        package.KMeans = TrainingKMeansUnavailable
        sys.modules["fast_pytorch_kmeans"] = package
        adapters.append("training-only missing KMeans import isolated; execution raises")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not all(isinstance(k, str) for k in checkpoint):
        raise ValueError("Expected official plain state_dict")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
        adapters.append("removed uniform DDP module prefix")
    summary = {"keys": len(checkpoint), "tensor_bytes": sum(v.numel() * v.element_size() for v in checkpoint.values()),
               "clip_keys": sum(k.startswith("net.") for k in checkpoint),
               "dino_keys": sum(k.startswith("vit_model.") for k in checkpoint),
               "checkpoint_sha256": sha256(weights)}
    if not summary["clip_keys"] or not summary["dino_keys"]:
        raise ValueError("Checkpoint lacks complete CLIP/DINO; no implicit pretrained download")
    import models.open_clip as open_clip
    from models.open_clip.model import build_model_from_openai_state_dict
    import models.dino.utils as dino_utils
    net_state = {k[4:]: v for k, v in checkpoint.items() if k.startswith("net.")}

    def create_from_checkpoint(name, pretrained=None, **kwargs):
        if name != "ViT-B/16" or pretrained != "openai":
            raise ValueError("Unexpected official CLIP constructor")
        # This is the same builder used by official openai.load_openai_model.
        # float() mirrors factory.create_model default precision='fp32'.
        return build_model_from_openai_state_dict(dict(net_state)).float()

    open_clip.create_model = create_from_checkpoint
    dino_utils.load_pretrained_weights = lambda *args, **kwargs: None
    adapters.extend(["CLIP original OpenAI-state builder from checkpoint; no download",
                     "DINO redundant pretrained initialization skipped before verified complete current-model state load"])
    locate = importlib.import_module("models.locate")
    arguments = types.SimpleNamespace(alpha=0.6, gamma1=0.6, gamma2=0.6,
                                     cont_temperature=0.5, divide="Seen")
    model = locate.Net(aff_classes=36, args=arguments)
    incompatible = model.load_state_dict(checkpoint, strict=False)
    # The released checkpoint has a historical aff_fc unused by released code.
    # Official test.py also uses strict=False. All current model keys must load.
    if incompatible.missing_keys or set(incompatible.unexpected_keys) - {"aff_fc.weight", "aff_fc.bias"}:
        raise ValueError(f"Current-model weights incomplete or unexpected: {incompatible}")
    model = model.cuda().eval()
    summary.update({"strict_missing_keys": list(incompatible.missing_keys),
                    "strict_unexpected_keys": list(incompatible.unexpected_keys),
                    "all_current_model_keys_loaded": True,
                    "load_policy": "Official strict=False; require no missing and whitelist only unused historical aff_fc",
                    "parameters": sum(p.numel() for p in model.parameters()), "adapters": adapters,
                    "test_forward_module": str(Path(locate.__file__).resolve()),
                    "test_forward_source_sha256": sha256(locate.__file__), "dtype": "float32"})
    save(output / "model_loading_receipt.json", summary)
    return model


def crop_box(width, height, center=None):
    cw, ch = max(1, round(width * 0.5)), max(1, round(height * 0.5))
    if center is None:
        x, y = (width - cw) // 2, (height - ch) // 2
    else:
        x = max(0, min(width - cw, round(center[0] - cw / 2)))
        y = max(0, min(height - ch, round(center[1] - ch / 2)))
    return [int(x), int(y), int(x + cw), int(y + ch)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-head", choices=HEADS)
    parser.add_argument("--sealed-head", type=Path)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--views", choices=("base", "all", "additional"), default="base")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=1400)
    args = parser.parse_args()
    started, prior_elapsed = time.perf_counter(), 0.0
    root, output = args.root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "writer.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    stage, completed, total, config_hash = "startup", 0, 0, None
    stopped = threading.Event()
    predictions_path = output / ("base_predictions.jsonl" if args.views == "base" else "predictions.jsonl")
    terminal = output / "terminal.json"

    def progress():
        return {"pid": os.getpid(), "stage": stage, "completed": completed, "total": total,
                "elapsed_s": prior_elapsed + time.perf_counter() - started,
                "last_activity_unix_s": time.time(), "config_sha256": config_hash}

    def watchdog():
        while not stopped.wait(0.5):
            if progress()["elapsed_s"] >= args.max_seconds:
                result = {**progress(), "status": "TIME_BUDGET_EXHAUSTED", "partial_resumable": True}
                save(terminal, result)
                lock.unlink(missing_ok=True)
                print(json.dumps(result), flush=True)
                os._exit(124)

    try:
        primary_head = args.primary_head
        if args.sealed_head:
            sealed = json.loads(args.sealed_head.read_text(encoding="utf-8-sig"))
            primary_head = sealed["selected_head"]
            if primary_head not in HEADS:
                raise ValueError("Invalid sealed head")
        if args.views != "base" and primary_head is None:
            raise ValueError("Additional views require a sealed/explicit primary head")
        if args.views == "additional" and args.base_predictions is None:
            raise ValueError("Additional view stage requires preserved base predictions")
        config = {"manifest_sha256": sha256(args.manifest), "script_sha256": sha256(__file__),
                  "source_receipt_sha256": sha256(root / "source_receipt.json"),
                  "model_receipt_sha256": sha256(root / "model_receipt.json"),
                  "source_commit": "5422ed50ba34c2fb5f368fddf4b46cf5c9b597ed",
                  "primary_head": primary_head, "all_heads": HEADS, "views": args.views,
                  "sealed_head_sha256": sha256(args.sealed_head) if args.sealed_head else None,
                  "base_predictions_sha256": sha256(args.base_predictions) if args.base_predictions else None,
                  "limit": args.limit, "max_seconds": args.max_seconds, "dtype": "float32",
                  "preprocessing": "Official torchvision RGB Resize((224,224)) bilinear; ToTensor; ImageNet mean/std",
                  "crop": "round(.5W),round(.5H); integer half-open boxes; primary raw14 cv2.resize to original then argmax pixel center; flat/nonfinite fallback center",
                  "flip": "PIL horizontal flip before original preprocessing; stored primary map inverse-flipped",
                  "normalization": "No normalization/fusion in inference; evaluation applies official resize->minmax->L1",
                  "peak_tie": "numpy argmax, first row-major cell", "seed": 0,
                  "timing_resolution_s": time.get_clock_info("perf_counter").resolution}
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        config_path = output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
            raise ValueError("Config drift; preserve run and use a new output directory")
        save(config_path, config)
        for path in (output / "progress.json", terminal):
            if path.exists():
                prior_elapsed = max(prior_elapsed, float(json.loads(path.read_text()).get("elapsed_s", 0)))
        if prior_elapsed >= args.max_seconds:
            raise TimeoutError("Saved cumulative budget exhausted")
        threading.Thread(target=watchdog, daemon=True).start()
        observations = observe(args.manifest, args.limit)
        total = len(observations)
        base_cache = {row["sample_id"]: row for row in rows(args.base_predictions)} if args.base_predictions else {}
        previous = rows(predictions_path) if predictions_path.exists() else []
        existing = {row["sample_id"]: row for row in previous}
        if len(existing) != len(previous):
            raise ValueError("Duplicate saved prediction id")
        completed = len(existing)
        stage = "imports"
        import numpy as np
        import cv2
        import torch
        import torchvision
        from torchvision import transforms
        from PIL import Image, ImageOps
        torch.set_num_threads(4)
        torch.manual_seed(0)
        np.random.seed(0)
        if not torch.cuda.is_available():
            raise RuntimeError("ACCELERATOR_UNAVAILABLE; stop without fallback")
        free, capacity = torch.cuda.mem_get_info()
        if free < 3_000_000_000:
            raise RuntimeError(f"CUDA free VRAM below3GB: {free}; stop without preempting other task")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        os.environ["TORCH_HOME"] = str(root / "torch-cache")
        os.environ["HF_HOME"] = str(root / "hf-cache")
        os.environ["HF_HUB_OFFLINE"] = "1"
        stage = "model_loading"
        model = load_official(root / "vendor" / "SelectiveCL", root / "models" / "AGD20K-Seen.pth", output)
        runtime = {"backend": "CUDA", "device": torch.cuda.get_device_name(), "free_total_before_bytes": [free, capacity],
                   "torch": torch.__version__, "torchvision": torchvision.__version__, "python": sys.version,
                   "dtype": "float32", "config_sha256": config_hash}
        save(output / "runtime.json", runtime)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        map_root = output / "maps"
        map_root.mkdir(exist_ok=True)

        def predict(image, action):
            tick = time.perf_counter()
            tensor = transform(image).unsqueeze(0).cuda()
            label = torch.tensor([ACTIONS.index(action)], dtype=torch.long, device="cuda")
            torch.cuda.synchronize()
            preprocess_s = time.perf_counter() - tick
            tick = time.perf_counter()
            with torch.inference_mode():
                maps = model.test_forward(tensor, label)
            torch.cuda.synchronize()
            forward_s = time.perf_counter() - tick
            result = {name: value[0].detach().float().cpu().numpy() for name, value in zip(HEADS, maps)}
            if any(value.shape != (14, 14) for value in result.values()):
                raise ValueError("Unexpected raw model map shape")
            return result, {"preprocess_s": preprocess_s, "forward_s": forward_s,
                            "actual_view_s": preprocess_s + forward_s}

        stage = "inference"
        for row in observations:
            if row["sample_id"] in existing:
                continue
            save(output / "progress.json", {**progress(), "current_sample": row["sample_id"]})
            tick = time.perf_counter()
            image = Image.open(row["image_path"]).convert("RGB")
            width, height = image.size
            image_hash = sha256(row["image_path"])
            if row["image_sha256"] is not None and row["image_sha256"] != image_hash:
                raise ValueError("Observation image hash mismatch")
            io_s = time.perf_counter() - tick
            all_maps, times = {}, {}
            if args.views == "additional":
                cached = base_cache[row["sample_id"]]
                if cached["image_sha256"] != image_hash or cached["action"] != row["action"]:
                    raise ValueError("Preserved base image/query identity mismatch")
                all_maps["base"] = {head: np.load(path) for head, path in cached["base_heads"].items()}
                times["base"] = {**cached["elapsed"]["base"], "source": "preserved_actual_base_inference"}
            else:
                all_maps["base"], times["base"] = predict(image, row["action"])
            primary = all_maps["base"][primary_head] if primary_head else None
            center_box = crop_box(width, height)
            if primary is None:
                peak_xy, peak_box, crop_fallback = None, None, "HEAD_NOT_SELECTED_IN_BASE_STAGE"
            elif not np.isfinite(primary).all() or float(primary.max()) == float(primary.min()):
                peak_xy, peak_box = [width / 2, height / 2], center_box
                crop_fallback = "NONFINITE_PRIMARY" if not np.isfinite(primary).all() else "FLAT_PRIMARY"
            else:
                full_primary = cv2.resize(primary, (width, height), interpolation=cv2.INTER_LINEAR)
                iy, ix = np.unravel_index(int(np.argmax(full_primary)), full_primary.shape)
                peak_xy = [float(ix) + 0.5, float(iy) + 0.5]
                peak_box, crop_fallback = crop_box(width, height, peak_xy), None
            if args.views != "base":
                all_maps["fixed_crop"], times["fixed_crop"] = predict(image.crop(tuple(center_box)), row["action"])
                all_maps["peak_crop"], times["peak_crop"] = predict(image.crop(tuple(peak_box)), row["action"])
                flip_maps, times["flip"] = predict(ImageOps.mirror(image), row["action"])
                all_maps["flip_raw"] = flip_maps
                all_maps["flip"] = {name: value[:, ::-1].copy() for name, value in flip_maps.items()}
            identifier = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:24]
            paths = {}
            for view, head_maps in all_maps.items():
                if view == "base" and args.views == "additional":
                    paths[view] = base_cache[row["sample_id"]]["base_heads"]
                    continue
                paths[view] = {}
                for head, raw in head_maps.items():
                    path = map_root / f"{identifier}_{view}_{head}.npy"
                    np.save(path, raw)
                    paths[view][head] = str(path)
            record = {**row, "image_sha256": image_hash, "width": width, "height": height,
                      "primary_head": primary_head, "heads": paths, "base_heads": paths["base"],
                      "base_map_path": paths["base"].get(primary_head),
                      "fixed_crop_map_path": paths.get("fixed_crop", {}).get(primary_head),
                      "peak_crop_map_path": paths.get("peak_crop", {}).get(primary_head),
                      "flip_map_path": paths.get("flip", {}).get(primary_head),
                      "flip_raw_map_path": paths.get("flip_raw", {}).get(primary_head),
                      "flip_map_is_inverse_restored": True,
                      "fixed_crop_bbox_xyxy": center_box, "peak_crop_bbox_xyxy": peak_box,
                      "peak_original_xy": peak_xy,
                      "peak_crop_fallback": crop_fallback,
                      "raw_head_finite": {view: {head: bool(np.isfinite(raw).all()) for head, raw in head_maps.items()}
                                          for view, head_maps in all_maps.items()},
                      "coordinate_contract": "raw grid cell centers; resize to actual original/crop pixel rectangle; flip_map already inverse flipped",
                      "elapsed": times, "image_io_hash_s": io_s,
                      "model_calls": 1 if args.views == "base" else 3 if args.views == "additional" else 4,
                      "config_sha256": config_hash}
            append(predictions_path, record)
            existing[row["sample_id"]] = record
            completed = len(existing)
            save(output / "progress.json", progress())
            if not (output / "first_image_probe.json").exists():
                save(output / "first_image_probe.json", {**progress(), "times": times,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            if completed == 1 or completed % 20 == 0 or completed == total:
                print(json.dumps(progress()), flush=True)
        stage = "complete"
        hashes = {p.name: sha256(p) for p in (predictions_path, output / "runtime.json", output / "config.json",
                                             output / "model_loading_receipt.json") if p.exists()}
        result = {**progress(), "status": "SUCCESS", "partial_resumable": True, "hashes": hashes,
                  "raw_map_files": len(list(map_root.glob("*.npy"))),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        save(output / "progress.json", progress())
        save(terminal, result)
        print(json.dumps(result), flush=True)
        return 0
    except BaseException as exc:
        stage = "failed"
        details = {**progress(), "status": "FAILED", "error": str(exc), "error_type": type(exc).__name__,
                   "traceback": traceback.format_exc()}
        append(output / "errors.jsonl", details)
        save(output / "progress.json", progress())
        save(terminal, details)
        print(json.dumps(details), flush=True)
        return 1
    finally:
        stopped.set()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
