"""NF-G9-A frozen G8 RGB inference; no fitting, labels, evaluator or thresholds."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from PIL import Image
import torch

from whisker_model import ARMS, HISTORY, IMAGE_SIZE, SUPPORT_SIZE, TARGET_ORDER, WhiskerModel
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.research_backend import torch_observation

SEEDS = (17, 29, 43)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load_model_inputs(capture):
    capture = capture.resolve()
    if read(capture / "verification.json").get("status") != "PASS":
        raise ValueError("Capture verification must PASS before inference")
    model_root = (capture / "model").resolve()
    dataset = read(model_root / "dataset.json")
    calibration = dataset["calibration"]
    if (calibration["width"], calibration["height"]) != (640, 360) or float(calibration["horizontal_fov_degrees"]) != 100.:
        raise ValueError("Frozen calibration requires 640x360 HFOV100")
    frame_fields = {"sample_index", "rgb_path", "time_s", "clip_id", "frame_in_clip"}
    sample_fields = {"sample_id", "clip_id", "group_id", "split", "frame_indices"}
    frames, paths = {}, {}
    for frame in dataset["frames"]:
        if set(frame) != frame_fields:
            raise ValueError("Unexpected frame metadata; pose/speed inputs prohibited")
        index = int(frame["sample_index"])
        if index in frames or not np.isfinite(float(frame["time_s"])):
            raise ValueError("Duplicate frame index or nonfinite time")
        rel = Path(frame["rgb_path"])
        path = (model_root / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(model_root) or path.suffix.lower() != ".png":
            raise ValueError("RGB path must be contained model PNG")
        frames[index], paths[index] = frame, path
    samples = dataset["samples"]
    if not samples or len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Empty or duplicate samples")
    for sample in samples:
        if set(sample) != sample_fields or sample["split"] != "test":
            raise ValueError("Require exact sanitized sample fields and all-test split")
        history = [frames[int(i)] for i in sample["frame_indices"]]
        if len(history) != HISTORY or any(f["clip_id"] != sample["clip_id"] for f in history):
            raise ValueError("Require three frames from sample clip")
        if any(float(b["time_s"]) <= float(a["time_s"]) or int(b["frame_in_clip"]) != int(a["frame_in_clip"])+1 for a, b in zip(history, history[1:])):
            raise ValueError("Require chronological contiguous history")
    provenance = {p.relative_to(capture).as_posix(): sha(p) for p in (model_root / "dataset.json", capture / "verification.json")}
    return dataset, paths, provenance


def checkpoint_inputs(learned):
    config = read(learned / "config.json")
    receipt = read(learned / "receipt.json")
    if receipt.get("status") != "complete" or receipt.get("actual_backend") != "cuda":
        raise ValueError("Require completed G8 CUDA source receipt")
    if config["seeds"] != list(SEEDS) or config["target_order"] != list(TARGET_ORDER) or config["image_size"] != list(IMAGE_SIZE):
        raise ValueError("Frozen G8 configuration mismatch")
    records = {(r["arm"], r["seed"]): r for r in receipt["records"]}
    if set(records) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("Source must contain nine G8 fits")
    paths, hashes = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            path = learned / f"{arm}_seed{seed}.pt"
            digest = sha(path)
            if digest != records[(arm, seed)]["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint hash mismatch")
            paths[(arm, seed)], hashes[path.name] = path, digest
    # Architecture source is pinned to the recovered G8 execution evidence.
    expected_model = receipt.get("recovery_evidence", {}).get("source_before_path_fix_sha256", {}).get("whisker_model.py")
    if expected_model and sha(Path(__file__).with_name("whisker_model.py")) != expected_model:
        raise ValueError("G8 architecture source changed since source fit")
    return paths, dict(checkpoint_sha256=hashes, config_sha256=sha(learned / "config.json"), receipt_sha256=sha(learned / "receipt.json"))


@torch.inference_mode()
def run(capture, learned, output):
    if output.exists():
        raise FileExistsError(f"Refuse overwrite: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, CPU fallback prohibited")
    dataset, paths, input_hashes = load_model_inputs(capture)
    checkpoints, source = checkpoint_inputs(learned)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    timing = dict(image_decode_seconds=0., image_resize_preprocessing_seconds=0.,
        rgb_hash_seconds=0., batch_assembly_transfer_normalization_seconds=0., inference_forward_seconds=0.)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        images, hashes = {}, {}
        for index, path in paths.items():
            tick = time.perf_counter()
            hashes[str(index)] = sha(path)
            timing["rgb_hash_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            with Image.open(path) as image:
                if image.size != (640, 360):
                    raise ValueError("Unexpected RGB resolution")
                raw = image.convert("RGB")
                raw.load()
            timing["image_decode_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            images[index] = np.asarray(raw.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BOX), dtype=np.uint8).copy()
            timing["image_resize_preprocessing_seconds"] += time.perf_counter()-tick
        samples = dataset["samples"]
        predictions = dict(schema="nf-g8-whisker-predictions-v1", target_order=list(TARGET_ORDER),
            sample_ids=[s["sample_id"] for s in samples], arms={})
        maps = {"test_sample_ids": np.asarray(predictions["sample_ids"])}
        records = []
        for arm in ARMS:
            scores, supports = [], []
            predictions["arms"][arm] = dict(seeds={})
            for seed in SEEDS:
                model = WhiskerModel(arm).cuda().eval()
                model.load_state_dict(torch.load(checkpoints[(arm, seed)], map_location="cuda", weights_only=True))
                score_parts, support_parts = [], []
                for offset in range(0, len(samples), 64):
                    selected = samples[offset:offset+64]
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    array = np.stack([[images[int(i)] for i in s["frame_indices"]] for s in selected])
                    rgb = torch.from_numpy(array).to("cuda").permute(0, 1, 4, 2, 3).float().div_(255.)
                    torch.cuda.synchronize()
                    timing["batch_assembly_transfer_normalization_seconds"] += time.perf_counter()-tick
                    tick = time.perf_counter()
                    logits, support = model(rgb)
                    values, support_values = logits.sigmoid().cpu().numpy(), support.sigmoid().cpu().numpy()
                    torch.cuda.synchronize()
                    timing["inference_forward_seconds"] += time.perf_counter()-tick
                    if not np.isfinite(values).all() or not np.isfinite(support_values).all():
                        raise RuntimeError("Nonfinite inference output")
                    score_parts.append(values)
                    support_parts.append(support_values)
                values, support_values = np.concatenate(score_parts), np.concatenate(support_parts)
                scores.append(values)
                supports.append(support_values)
                predictions["arms"][arm]["seeds"][str(seed)] = dict(normal=values.tolist())
                maps[f"{arm}__seed{seed}__normal"] = support_values.astype(np.float16)
                records.append(dict(arm=arm, seed=seed, samples=len(samples),
                    actual_backend_observation=asdict(torch_observation(model=model)), checkpoint_sha256=source["checkpoint_sha256"][checkpoints[(arm, seed)].name]))
                del model
            predictions["arms"][arm]["ensemble"] = dict(normal=np.mean(scores, axis=0).tolist())
            maps[f"{arm}__ensemble__normal"] = np.mean(supports, axis=0).astype(np.float16)
        write(output / "predictions.json", predictions)
        np.savez_compressed(output / "support_predictions.npz", **maps)
        write(output / "receipt.json", dict(schema="nf-g9-a-frozen-grounding-inference-v1", status="complete",
            actual_backend="cuda", actual_device=torch.cuda.get_device_name(), torch_version=torch.__version__, cuda_version=torch.version.cuda,
            input_sha256=input_hashes, rgb_sha256=hashes, learned_source=source, records=records,
            source_sha256={name: sha(Path(__file__).with_name(name)) for name in ("grounding_predict.py", "whisker_model.py")},
            timing=timing, total_wall_seconds=time.perf_counter()-started, frame_count=len(paths), sample_count=len(samples),
            batch_size=64, fit_count=0, inference_examples=len(samples)*9, target_order=list(TARGET_ORDER),
            support_shape=[2, *SUPPORT_SIZE], temporal_claim="none; static counterfactual source, normal predictions only",
            boundary="RGB only; no fitting, threshold selection, labels or evaluator reads; metadata validated but not fed to model",
            timing_scope="cached decode/resize once, batched CUDA forward including sigmoid and CPU output; not batch1 camera or phone latency"))
    except BaseException as error:
        write(output / "receipt.json", dict(status="failed", error=repr(error), timing=timing,
            actual_backend="cuda", actual_device=torch.cuda.get_device_name(), total_wall_seconds=time.perf_counter()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--learned-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.capture, args.learned_source, args.output)


if __name__ == "__main__":
    main()
