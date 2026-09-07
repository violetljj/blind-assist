"""Frozen NF-G8 CUDA training. Reads model RGB and train/val targets only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.research_backend import torch_observation

from whisker_model import (ARMS, HISTORY, IMAGE_SIZE, SUPPORT_SIZE, TARGET_ORDER,
                           WhiskerModel, masked_bce, support_bce, repeat_history)

CONFIG = dict(schema="nf-g8-whisker-training-v1", seeds=[17, 29, 43], epochs=40,
    optimizer="AdamW", learning_rate=.001, weight_decay=.0001, batch_size=64,
    image_size=list(IMAGE_SIZE), support_size=list(SUPPORT_SIZE), history=HISTORY,
    target_order=list(TARGET_ORDER), training_loss="masked_BCE_targets + 0.25_class_balanced_masked_BCE_support",
    support_class_balance="mean known positive pixels and mean known negative pixels, then average present classes; omit absent classes; all UNKNOWN zero",
    checkpoint_selection="minimum_validation_masked_target_BCE_first_tie",
    inputs="RGB only; fixed mounting/calibration validated but not model features",
    repeated_history="replace past RGB with current RGB only; timestamps and metadata unchanged",
    support_weight=.25, outcome_tuning=False, device_required="cuda",
    near_support_branch="current RGB only, never motion gated",
    ordinary_video="learned Conv3D across three RGB frames",
    bio_video="signed local Hassenstein-Reichardt products, learned excitatory/inhibitory convolutions",
    capacity_claim="same spatial backbone and heads; temporal parameter counts differ and are reported")


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(capture, paths):
    """Normalize junction aliases before making portable capture-relative keys."""
    root = capture.resolve()
    return {path.resolve().relative_to(root).as_posix(): sha(path) for path in paths}


def load_inputs(capture):
    model_dir = (capture / "model").resolve()
    dataset_path = model_dir / "dataset.json"
    label_path = capture / "training" / "labels.json"
    support_path = capture / "training" / "support.npz"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    allowed_frame_fields = {"sample_index", "rgb_path", "time_s", "clip_id", "frame_in_clip"}
    for frame in dataset["frames"]:
        if set(frame) != allowed_frame_fields:
            raise ValueError("RGB frame fields must exactly match the permitted deployment contract; extra metadata prohibited")
    labels = json.loads(label_path.read_text(encoding="utf-8-sig"))["targets"]
    calibration = dataset["calibration"]
    if (calibration["width"], calibration["height"]) != (640, 360) or abs(float(calibration["horizontal_fov_degrees"])-100) > 1e-6:
        raise ValueError("Frozen calibration requires 640x360 horizontal FOV100")
    frames = {int(f["sample_index"]): f for f in dataset["frames"]}
    samples = dataset["samples"]
    if len(frames) != len(dataset["frames"]) or not samples or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate frames or empty/duplicate samples")
    group_splits, clip_splits, frame_splits = {}, {}, {}
    for sample in samples:
        split = sample["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        for table, key in ((group_splits, sample["group_id"]), (clip_splits, sample["clip_id"])):
            if key in table and table[key] != split:
                raise ValueError("Group or clip crosses splits")
            table[key] = split
        history = [frames[int(i)] for i in sample["frame_indices"]]
        if len(history) != HISTORY or any(f["clip_id"] != sample["clip_id"] for f in history):
            raise ValueError("Require three frames from the sample clip")
        if any(float(b["time_s"]) <= float(a["time_s"]) or int(b["frame_in_clip"]) != int(a["frame_in_clip"])+1 for a, b in zip(history, history[1:])):
            raise ValueError("History must be chronological and contiguous")
        for index in map(int, sample["frame_indices"]):
            if index in frame_splits and frame_splits[index] != split:
                raise ValueError("Shared frame across splits")
            frame_splits[index] = split
    if set(group_splits.values()) != {"train", "val", "test"}:
        raise ValueError("All three group-separated splits required")
    expected = {s["sample_id"] for s in samples if s["split"] != "test"}
    if set(labels) != expected:
        raise ValueError("Labels must contain exactly train+val IDs; test labels prohibited")
    for target in labels.values():
        if np.asarray(target).shape != (4,) or not np.isin(target, [-1, 0, 1]).all():
            raise ValueError("Four targets must be binary or -1 UNKNOWN")
    for split in ("train", "val"):
        if not any(np.any(np.asarray(labels[s["sample_id"]]) >= 0) for s in samples if s["split"] == split):
            raise ValueError("Partition contains no evaluable targets")
    with np.load(support_path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("Support must contain exactly train+val IDs")
        support = {key: archive[key].copy() for key in archive.files}
    for value in support.values():
        if value.shape != (2, *SUPPORT_SIZE) or value.dtype != np.uint8 or not np.isin(value, [0, 1]).all():
            raise ValueError("Support must be uint8 binary 2x18x32")
    paths = {}
    for index, frame in frames.items():
        rel = Path(frame["rgb_path"])
        path = (model_dir / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(model_dir) or path.suffix.lower() != ".png":
            raise ValueError("RGB path must be a contained model PNG")
        paths[index] = path
    return dataset, labels, support, paths, [dataset_path, label_path, support_path]


def decode_images(paths):
    images, hashes = {}, {}
    timing = dict(image_decode_seconds=0., image_resize_preprocessing_seconds=0., image_hash_seconds=0.)
    for index, path in paths.items():
        started = time.perf_counter()
        hashes[str(index)] = sha(path)
        timing["image_hash_seconds"] += time.perf_counter()-started
        started = time.perf_counter()
        with Image.open(path) as image:
            if image.size != (640, 360):
                raise ValueError("Unexpected source image resolution")
            raw = image.convert("RGB")
            raw.load()
        timing["image_decode_seconds"] += time.perf_counter()-started
        started = time.perf_counter()
        resized = raw.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BOX)
        images[index] = np.asarray(resized, dtype=np.uint8).copy()
        timing["image_resize_preprocessing_seconds"] += time.perf_counter()-started
    return images, hashes, timing


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(indices):
    for start in range(0, len(indices), CONFIG["batch_size"]):
        yield indices[start:start+CONFIG["batch_size"]]


def rgb_batch(selected, samples, images, timing):
    torch.cuda.synchronize()
    started = time.perf_counter()
    array = np.stack([[images[int(i)] for i in samples[n]["frame_indices"]] for n in selected])
    rgb = torch.from_numpy(array).to("cuda").permute(0, 1, 4, 2, 3).float().div_(255.)
    torch.cuda.synchronize()
    timing["batch_assembly_transfer_normalization_seconds"] += time.perf_counter()-started
    return rgb


@torch.inference_mode()
def inference(model, selected_indices, samples, images, timing, repeated=False, support_indices=None):
    model.eval()
    values, maps = [], []
    support_indices = set(support_indices or [])
    for selected in batches(selected_indices):
        rgb = rgb_batch(selected, samples, images, timing)
        if repeated:
            rgb = repeat_history(rgb)
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits, support = model(rgb)
        torch.cuda.synchronize()
        timing["inference_forward_seconds"] += time.perf_counter()-started
        timing["inference_examples"] += len(selected)
        timing["inference_batches"] += 1
        values.append(logits.sigmoid().cpu())
        keep = [n for n, index in enumerate(selected) if index in support_indices]
        if keep:
            maps.append(support[keep].sigmoid().cpu())
    return torch.cat(values), (torch.cat(maps).numpy() if maps else None)


def train_one(arm, seed, samples, images, targets, support_targets, splits, output, timing):
    seed_all(seed)
    model = WhiskerModel(arm).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    checkpoint = output / f"{arm}_seed{seed}.pt"
    best, best_epoch, history = float("inf"), -1, []
    started_training = time.perf_counter()
    for epoch in range(CONFIG["epochs"]):
        model.train()
        order = np.random.permutation(splits["train"]).tolist()
        for selected in batches(order):
            rgb = rgb_batch(selected, samples, images, timing)
            target, mask = targets[selected].cuda(), support_targets[selected].cuda()
            optimizer.zero_grad(set_to_none=True)
            logits, support = model(rgb)
            loss = masked_bce(logits, target) + CONFIG["support_weight"]*support_bce(support, mask, target[:, :2])
            loss.backward()
            optimizer.step()
        model.eval()
        numerator, denominator = 0., 0
        with torch.inference_mode():
            for selected in batches(splits["val"]):
                logits, _ = model(rgb_batch(selected, samples, images, timing))
                target = targets[selected].cuda()
                known = int((target >= 0).sum().item())
                numerator += float(masked_bce(logits, target).item()) * known
                denominator += known
        val = numerator/denominator
        if not np.isfinite(val):
            raise RuntimeError("Non-finite validation loss")
        history.append(val)
        if val < best:
            best, best_epoch = val, epoch+1
            torch.save(model.state_dict(), checkpoint)
        progress = dict(stage="training", arm=arm, seed=seed, epoch=epoch+1,
                        epochs=CONFIG["epochs"], validation_masked_bce=val, best_epoch=best_epoch,
                        fit_index=ARMS.index(arm)*len(CONFIG['seeds'])+CONFIG['seeds'].index(seed)+1,
                        total_fits=len(ARMS)*len(CONFIG['seeds']),
                        elapsed_this_fit_s=time.perf_counter()-started_training,
                        estimated_remaining_this_fit_s=(time.perf_counter()-started_training)/(epoch+1)*(CONFIG['epochs']-epoch-1))
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    torch.cuda.synchronize()
    duration = time.perf_counter()-started_training
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    result, maps = {}, {}
    for mode in ("normal", "repeated_history"):
        probabilities, support = inference(model, list(range(len(samples))), samples, images, timing,
            repeated=mode == "repeated_history", support_indices=splits["test"])
        result[mode], maps[mode] = probabilities.numpy(), support
    return result, maps, dict(arm=arm, seed=seed, best_epoch=best_epoch,
        best_validation_masked_bce=best, validation_history=history, training_validation_wall_seconds=duration,
        checkpoint_sha256=sha(checkpoint), parameters=sum(p.numel() for p in model.parameters()),
        actual_parameter_device=str(next(model.parameters()).device),
        actual_backend_observation=asdict(torch_observation(model=model)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refuse overwrite: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is prohibited")
    verification = json.loads((args.capture / "verification.json").read_text(encoding="utf-8-sig"))
    if verification.get("status") != "PASS":
        raise ValueError("Capture verification must be PASS before training")
    dataset, labels, support, paths, input_paths = load_inputs(args.capture)
    input_paths.append(args.capture / "verification.json")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    parameter_counts = {arm: sum(p.numel() for p in WhiskerModel(arm).parameters()) for arm in ARMS}
    write_json(args.output / "config.json", dict(CONFIG, parameter_counts=parameter_counts))
    timing = dict(batch_assembly_transfer_normalization_seconds=0., inference_forward_seconds=0.,
                  inference_examples=0, inference_batches=0)
    try:
        images, hashes, decode_timing = decode_images(paths)
        timing.update(decode_timing)
        samples = dataset["samples"]
        splits = {split: [i for i, sample in enumerate(samples) if sample["split"] == split] for split in ("train", "val", "test")}
        targets = torch.tensor([labels.get(s["sample_id"], [-1]*4) for s in samples], dtype=torch.float32)
        support_targets = torch.from_numpy(np.stack([support.get(s["sample_id"], np.zeros((2, *SUPPORT_SIZE), np.uint8)) for s in samples])).float()
        predictions = dict(schema="nf-g8-whisker-predictions-v1", target_order=list(TARGET_ORDER),
                           sample_ids=[s["sample_id"] for s in samples], arms={})
        map_arrays = {"test_sample_ids": np.asarray([samples[i]["sample_id"] for i in splits["test"]])}
        records = []
        for arm in ARMS:
            runs, run_maps = [], []
            predictions["arms"][arm] = {"seeds": {}}
            for seed in CONFIG["seeds"]:
                result, maps, record = train_one(arm, seed, samples, images, targets, support_targets, splits, args.output, timing)
                records.append(record)
                runs.append(result)
                run_maps.append(maps)
                predictions["arms"][arm]["seeds"][str(seed)] = {mode: value.tolist() for mode, value in result.items()}
                for mode, value in maps.items():
                    map_arrays[f"{arm}__seed{seed}__{mode}"] = value.astype(np.float16)
            predictions["arms"][arm]["ensemble"] = {mode: np.mean([run[mode] for run in runs], axis=0).tolist() for mode in ("normal", "repeated_history")}
            for mode in ("normal", "repeated_history"):
                map_arrays[f"{arm}__ensemble__{mode}"] = np.mean([run[mode] for run in run_maps], axis=0).astype(np.float16)
            write_json(args.output / "predictions.json", predictions)
        np.savez_compressed(args.output / "support_predictions.npz", **map_arrays)
        write_json(args.output / "receipt.json", dict(schema="nf-g8-whisker-receipt-v1", status="complete",
            actual_backend="cuda", actual_device=torch.cuda.get_device_name(), torch_version=torch.__version__,
            cuda_version=torch.version.cuda, timing=timing, total_wall_seconds=time.perf_counter()-started,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), records=records,
            input_sha256=input_hashes(args.capture, input_paths), rgb_sha256=hashes,
            label_boundary="train+val target/support IDs validated exactly; no evaluator reads",
            timing_scope="decode and resize cached once; batch preparation excludes forward; training wall includes validation, preprocessing and checkpoints; inference is batched desktop CUDA, not phone latency"))
        write_json(args.output / "progress.json", dict(stage="complete"))
    except BaseException as error:
        write_json(args.output / "receipt.json", dict(status="failed", error=repr(error), timing=timing,
            total_wall_seconds=time.perf_counter()-started, actual_backend="cuda", actual_device=torch.cuda.get_device_name()))
        raise


if __name__ == "__main__":
    main()
