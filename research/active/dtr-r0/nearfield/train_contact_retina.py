"""Frozen NF-G7 CUDA training, validation selection, and label-blind inference.

Reads ONLY capture/model/dataset.json, its contained RGB paths, and
capture/training/labels.json (train/val targets). No test label/evaluator/spec
reader exists here. Feature extraction may process test inputs but never fits
normalizers or chooses architecture/hyperparameters from their distribution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tools.research_backend import torch_observation, runtime_capabilities
from contact_retina_model import (ARMS, CHANNELS, DEPTHS_M, HISTORY, IMAGE_SIZE,
    STRUCTURE_SIZE, ContactRetina, arm_features, prepare_features)

CONFIG = dict(schema="nf-g7-contact-retina-training-v1", seeds=[17, 29, 43],
    epochs=100, optimizer="AdamW", learning_rate=.001, weight_decay=.0001,
    loss="unweighted_binary_cross_entropy_cumulative_probabilities",
    checkpoint_selection="minimum_validation_BCE_first_tie", image_size=list(IMAGE_SIZE),
    structure_size=list(STRUCTURE_SIZE), depth_planes_m=list(DEPTHS_M), history=HISTORY,
    channels=CHANNELS, target_order=["BODY", "HEAD"], horizons_s=[1, 2, 3],
    single_frame_metadata="same_observed_past_8_relative_ego_poses_and_speeds_as_video",
    pose_claim="PRIVILEGED_SYNTHETIC_METRIC_POSE_NOT_IMU_ESTIMATE",
    arm_trainable_capacity="identical CNN, motion encoder, and prediction head",
    compute_claim="structure adds fixed GPU plane sweep; compute is NOT identical",
    repeated_history="copy all past RGB/camera/wearer poses from current; retain speed/time",
    device_required="cuda", outcome_tuning=False)


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                   encoding="utf-8")
    tmp.replace(path)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_inputs(capture: Path):
    model_dir = (capture / "model").resolve()
    dataset_path = model_dir / "dataset.json"
    label_path = capture / "training" / "labels.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    labels = json.loads(label_path.read_text(encoding="utf-8-sig"))["targets"]
    calibration = dataset["calibration"]
    if calibration.get("pose_translation_unit", "metres") not in ("metres", "meters", "m"):
        raise ValueError("Only metre-valued UE poses accepted")
    if (calibration["width"], calibration["height"]) != (640, 360):
        raise ValueError("Frozen capture resolution is 640x360")
    frames = {int(f["sample_index"]): f for f in dataset["frames"]}
    if len(frames) != len(dataset["frames"]):
        raise ValueError("Duplicate frame indices")
    samples = dataset["samples"]
    ids = [s["sample_id"] for s in samples]
    if len(ids) != len(set(ids)) or not samples:
        raise ValueError("Empty or duplicate sample IDs")
    group_splits, clip_splits, frame_splits = {}, {}, {}
    for sample in samples:
        split = sample["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        for table, key in ((group_splits, sample["group_id"]), (clip_splits, sample["clip_id"])):
            if key in table and table[key] != split:
                raise ValueError("Scene/appearance group or clip crosses splits")
            table[key] = split
        history = [frames[int(i)] for i in sample["frame_indices"]]
        if len(history) != HISTORY or len({f["clip_id"] for f in history}) != 1:
            raise ValueError("History must contain eight frames from one clip")
        if history[0]["clip_id"] != sample["clip_id"]:
            raise ValueError("Sample/observation clip mismatch")
        times = [float(f["time_s"]) for f in history]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("History is not strictly chronological")
        if any(int(b["frame_in_clip"]) != int(a["frame_in_clip"])+1
               for a, b in zip(history, history[1:])):
            raise ValueError("History contains a temporal gap")
        for index in sample["frame_indices"]:
            if index in frame_splits and frame_splits[index] != split:
                raise ValueError("Shared frame across splits")
            frame_splits[index] = split
    expected_labels = {s["sample_id"] for s in samples if s["split"] != "test"}
    if set(labels) != expected_labels:
        raise ValueError("Training labels must contain exactly train+val IDs, no test targets")
    for target in labels.values():
        a = np.asarray(target)
        if a.shape != (2, 3) or not np.isin(a, [0, 1]).all() or (np.diff(a, axis=-1) < 0).any():
            raise ValueError("Targets must be binary cumulative BODY/HEAD x three horizons")
    if set(group_splits.values()) != {"train", "val", "test"}:
        raise ValueError("All three group-separated partitions are required")
    paths = {}
    for index, frame in frames.items():
        rel = Path(frame["rgb_path"])
        path = (model_dir / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(model_dir) or path.suffix.lower() != ".png":
            raise ValueError("RGB path must be a contained model PNG")
        paths[index] = path
    return dataset, labels, frames, paths, dataset_path, label_path


def cache_features(dataset, frames, paths, output):
    start = time.perf_counter()
    images = {}
    image_hashes = {}
    for index, path in paths.items():
        image_hashes[str(index)] = {"rgb_path": frames[index]["rgb_path"], "sha256": sha(path)}
        with Image.open(path) as image:
            if image.size != (640, 360):
                raise ValueError(f"Unexpected image dimensions: {path}")
            # BOX area resampling keeps average energy of subpixel thin traces.
            small = image.convert("RGB").resize((STRUCTURE_SIZE[1], STRUCTURE_SIZE[0]), Image.Resampling.BOX)
            images[index] = np.asarray(small, dtype=np.uint8).copy()
    io_seconds = time.perf_counter() - start
    visual, motion, repeated_motion, times = [], [], [], []
    hfov = float(dataset["calibration"]["horizontal_fov_degrees"])
    samples = dataset["samples"]
    for n, sample in enumerate(samples):
        indices = [int(i) for i in sample["frame_indices"]]
        history = [frames[i] for i in indices]
        torch.cuda.synchronize()
        started = time.perf_counter()
        rgb = torch.as_tensor(np.stack([images[i] for i in indices]), device="cuda")
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.
        value, meta, repeat_meta = prepare_features(rgb, history, hfov)
        observation = torch_observation(output=(value, meta, repeat_meta))
        if observation.normalized_type() != "cuda":
            raise RuntimeError("Feature extraction silently left CUDA")
        visual.append(value.cpu())
        motion.append(meta.cpu())
        repeated_motion.append(repeat_meta.cpu())
        torch.cuda.synchronize()
        times.append(time.perf_counter()-started)
        if (n+1) % 24 == 0 or n+1 == len(samples):
            progress = dict(stage="feature_cache", completed=n+1, total=len(samples),
                            elapsed_s=time.perf_counter()-start, eta_s=(len(samples)-n-1)*statistics.mean(times))
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    return (torch.stack(visual), torch.stack(motion), torch.stack(repeated_motion),
            dict(io_decode_hash_seconds=io_seconds,
                 fixed_preprocess_seconds_total=sum(times),
                 fixed_preprocess_ms_p50=statistics.median(times)*1000,
                 fixed_preprocess_ms_p95=float(np.percentile(times, 95))*1000,
                 preprocessing_claim="full structure pipeline per sample including GPU transfer and cache copy; all seeds reuse",
                 actual_observation=asdict(observation), rgb_hashes=image_hashes))


def batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start+batch_size]


def select_batch_size(visual, motion, targets, train_indices):
    """Brief real forward/backward workload probe; no optimizer update or tuning."""
    records = []
    for size in (16, 32):
        seed_all(17)
        probe = ContactRetina().cuda().train()
        indices = train_indices[:size]
        x, m, y = visual[indices].cuda(), motion[indices].cuda(), targets[indices].cuda()
        elapsed = []
        for iteration in range(4):
            probe.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            scores = probe(x, m)
            F.binary_cross_entropy(scores, y).backward()
            torch.cuda.synchronize()
            if iteration:
                elapsed.append(time.perf_counter()-started)
        record = dict(batch_size=size, actual_examples=len(indices), samples_s=elapsed,
                      examples_per_second=len(indices)/statistics.median(elapsed),
                      observation=asdict(torch_observation(model=probe, output=scores)))
        records.append(record)
        del probe, x, m, y, scores
    selected = max(records, key=lambda row: row["examples_per_second"])["batch_size"]
    return selected, records


@torch.inference_mode()
def predict(model, visual, motion, indices, batch_size, arm, repeated=False):
    model.eval()
    result = []
    for selected in batches(indices, batch_size):
        value = arm_features(visual[selected].cuda(), arm, repeated)
        result.append(model(value, motion[selected].cuda()).cpu())
    return torch.cat(result)


def train_one(arm, seed, visual, motion, repeated_motion, targets, samples,
              split_indices, batch_size, output):
    seed_all(seed)
    model = ContactRetina().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
    best_loss, best_epoch, best_state = float("inf"), 0, None
    logs = []
    started = time.perf_counter()
    for epoch in range(1, CONFIG["epochs"]+1):
        model.train()
        permutation = torch.randperm(len(split_indices["train"]))
        order = split_indices["train"][permutation]
        loss_sum = 0.
        for selected in batches(order, batch_size):
            optimizer.zero_grad(set_to_none=True)
            value = arm_features(visual[selected].cuda(), arm)
            score = model(value, motion[selected].cuda())
            loss = F.binary_cross_entropy(score, targets[selected].cuda())
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
        val_scores = predict(model, visual, motion, split_indices["val"], batch_size, arm)
        val_loss = float(F.binary_cross_entropy(val_scores, targets[split_indices["val"]]))
        if not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss; frozen run stops")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log = dict(epoch=epoch, train_bce=loss_sum/len(order), val_bce=val_loss)
        logs.append(log)
        if epoch % 10 == 0 or epoch == 1:
            progress = dict(stage="training", arm=arm, seed=seed, epoch=epoch,
                            total_epochs=CONFIG["epochs"], best_epoch=best_epoch,
                            best_val_bce=best_loss, elapsed_s=time.perf_counter()-started)
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    model.load_state_dict(best_state)
    checkpoint = output / f"{arm}-seed{seed}.pt"
    torch.save(dict(state_dict=best_state, arm=arm, seed=seed, best_epoch=best_epoch,
                    best_val_bce=best_loss, config=CONFIG), checkpoint)
    write_json(output / f"{arm}-seed{seed}-history.json", logs)
    all_indices = torch.arange(len(samples))
    scores = predict(model, visual, motion, all_indices, batch_size, arm)
    rows = [dict(arm=arm, seed=seed, sample_id=s["sample_id"], split=s["split"],
                 scores=score.tolist()) for s, score in zip(samples, scores)]
    if arm != "single_frame":
        ablation = predict(model, visual, repeated_motion, all_indices, batch_size, arm, True)
        rows.extend(dict(arm=arm+"_repeated_history", trained_arm=arm, seed=seed,
                         sample_id=s["sample_id"], split=s["split"], scores=score.tolist())
                    for s, score in zip(samples, ablation))
    # Same precomputed-input batch measures model-only CUDA inference fairly.
    model.eval()
    ids = split_indices["val"][:batch_size]
    x, m = arm_features(visual[ids].cuda(), arm), motion[ids].cuda()
    infer_times = []
    with torch.inference_mode():
        for iteration in range(6):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            score = model(x, m)
            torch.cuda.synchronize()
            if iteration:
                infer_times.append(time.perf_counter()-tick)
    receipt = dict(arm=arm, seed=seed, best_epoch=best_epoch, best_val_bce=best_loss,
        epochs_run=CONFIG["epochs"], trainable_parameters=sum(p.numel() for p in model.parameters()),
        train_elapsed_seconds=time.perf_counter()-started, checkpoint=checkpoint.name,
        checkpoint_sha256=sha(checkpoint), model_observation=asdict(torch_observation(model=model, output=score)),
        model_only_batch_size=len(ids), model_only_batch_latency_ms_p50=statistics.median(infer_times)*1000,
        model_only_ms_per_example=statistics.median(infer_times)*1000/len(ids),
        model_only_timing_excludes="disk IO, RGB transfers, fixed spatial/structure preprocessing",
        repeated_history_refit=False)
    del model, optimizer, x, m, score
    return rows, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    artifact_root = (ROOT / "artifacts.local").resolve()
    if not output.is_relative_to(artifact_root) or output == artifact_root:
        raise ValueError("Output must be a fresh directory under canonical artifacts.local")
    if output.exists():
        raise FileExistsError("Fresh output required; this runner never overwrites or resumes a frozen run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED: no silent CPU fallback")
    output.mkdir(parents=True)
    start = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dataset, labels, frames, paths, dataset_path, label_path = load_inputs(capture)
        samples = dataset["samples"]
        sources = [Path(__file__), Path(__file__).with_name("contact_retina_model.py"),
                   Path(__file__).with_name("sparse_structure.py"), ROOT / "tools/research_backend.py"]
        sources_hash = {str(p.relative_to(ROOT)): sha(p) for p in sources}
        config_hash = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
        split_indices = {name: torch.tensor([i for i, s in enumerate(samples) if s["split"] == name])
                         for name in ("train", "val", "test")}
        # No test targets exist here. NaNs ensure accidental supervised use fails.
        targets = torch.full((len(samples), 2, 3), float("nan"))
        for i, sample in enumerate(samples):
            if sample["split"] != "test":
                targets[i] = torch.tensor(labels[sample["sample_id"]], dtype=torch.float32)
        receipt = dict(config=CONFIG, config_sha256=config_hash, input_dataset_sha256=sha(dataset_path),
                       train_val_labels_sha256=sha(label_path), sources_sha256=sources_hash,
                       runtime=runtime_capabilities(), capture=str(capture), output=str(output),
                       split_samples={k: len(v) for k, v in split_indices.items()},
                       test_labels_read=False, dataset_identifiers_used_as_features=False)
        write_json(output / "launch.json", receipt)
        visual, motion, repeated_motion, preprocessing = cache_features(dataset, frames, paths, output)
        receipt["preprocessing"] = preprocessing
        batch_size, probes = select_batch_size(visual, motion, targets, split_indices["train"])
        receipt.update(batch_size=batch_size, batch_size_workload_probes=probes)
        write_json(output / "launch.json", receipt)
        all_rows, models = [], []
        for arm in ARMS:
            for seed in CONFIG["seeds"]:
                rows, model = train_one(arm, seed, visual, motion, repeated_motion, targets,
                                       samples, split_indices, batch_size, output)
                all_rows.extend(rows)
                models.append(model)
                write_json(output / "predictions.partial.json", dict(rows=all_rows, models=models,
                           complete=False, config_sha256=config_hash))
        if any(sha(p) != sources_hash[str(p.relative_to(ROOT))] for p in sources):
            raise RuntimeError("Source changed during frozen training; result cannot be finalized")
        receipt.update(models=models, elapsed_seconds=time.perf_counter()-start, status="COMPLETE")
        write_json(output / "predictions.json", dict(schema="nf-g7-contact-predictions-v1", rows=all_rows,
                   target_order=CONFIG["target_order"], horizons_s=CONFIG["horizons_s"],
                   config_sha256=config_hash, complete=True))
        write_json(output / "receipt.json", receipt)
        write_json(output / "terminal.json", dict(status="COMPLETE", elapsed_seconds=time.perf_counter()-start,
                   models=len(models), prediction_rows=len(all_rows), test_labels_read=False))
        print(json.dumps(dict(status="COMPLETE", output=str(output), models=len(models))), flush=True)
    except BaseException as error:
        write_json(output / "terminal.json", dict(status="FAILED", error_type=type(error).__name__,
                   error=str(error), elapsed_seconds=time.perf_counter()-start,
                   resume_supported=False, completed_models_remain_for_diagnosis=True))
        raise


if __name__ == "__main__":
    main()
