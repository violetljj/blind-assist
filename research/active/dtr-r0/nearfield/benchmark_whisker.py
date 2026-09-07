"""NF-G8 frozen-checkpoint, offline three-PNG-window CUDA latency benchmark.

Reads model inputs and learned artifacts only. No target or evaluator reader.
"""
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

from whisker_model import ARMS, HISTORY, IMAGE_SIZE, WhiskerModel

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.research_backend import torch_observation


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(capture, learned):
    """Pin model dataset, RGB payloads, configuration and all nine checkpoints."""
    receipt = json.loads((learned / "receipt.json").read_text(encoding="utf-8-sig"))
    config = json.loads((learned / "config.json").read_text(encoding="utf-8-sig"))
    if receipt.get("status") != "complete" or receipt.get("actual_backend") != "cuda":
        raise ValueError("Require completed CUDA training receipt")
    if config.get("seeds") != [17, 29, 43] or config.get("image_size") != list(IMAGE_SIZE):
        raise ValueError("Unexpected frozen model configuration")
    model_dir = (capture / "model").resolve()
    dataset_path = model_dir / "dataset.json"
    expected = {key.replace("\\", "/"): value for key, value in receipt["input_sha256"].items()}
    if sha(dataset_path) != expected.get("model/dataset.json"):
        raise ValueError("Model dataset does not match training receipt")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    frames = {}
    permitted = {"sample_index", "rgb_path", "time_s", "clip_id", "frame_in_clip"}
    for frame in dataset["frames"]:
        if set(frame) != permitted:
            raise ValueError("Unexpected frame metadata")
        index = int(frame["sample_index"])
        rel = Path(frame["rgb_path"])
        path = (model_dir / rel).resolve()
        if rel.is_absolute() or not path.is_relative_to(model_dir) or path.suffix.lower() != ".png":
            raise ValueError("RGB must be a contained model PNG")
        if index in frames or sha(path) != receipt["rgb_sha256"].get(str(index)):
            raise ValueError("Duplicate frame or RGB hash mismatch")
        frames[index] = path
    if set(map(str, frames)) != set(receipt["rgb_sha256"]):
        raise ValueError("RGB inventory differs from training receipt")
    test = [sample for sample in dataset["samples"] if sample["split"] == "test"]
    if len(test) < 50 or any(len(s["frame_indices"]) != HISTORY for s in test):
        raise ValueError("Require at least 50 fixed test windows with three RGB frames")
    records = {(r["arm"], r["seed"]): r for r in receipt["records"]}
    checkpoints = {}
    for arm in ARMS:
        for seed in config["seeds"]:
            path = learned / f"{arm}_seed{seed}.pt"
            if sha(path) != records[(arm, seed)]["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            checkpoints[(arm, seed)] = path
    return frames, test[:50], checkpoints, dict(
        dataset_sha256=sha(dataset_path), training_receipt_sha256=sha(learned / "receipt.json"),
        config_sha256=sha(learned / "config.json"),
        checkpoint_sha256={path.name: sha(path) for path in checkpoints.values()},
        verified_rgb_files=len(frames))


@torch.inference_mode()
def window(models, sample, frames):
    """Batch1 with fresh image decode every call; no decoded-image cache."""
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    started = total_start
    raw = []
    for index in sample["frame_indices"]:
        with Image.open(frames[int(index)]) as source:
            if source.size != (640, 360):
                raise ValueError("Unexpected PNG resolution")
            image = source.convert("RGB")
            image.load()
            raw.append(image)
    decode = time.perf_counter() - started
    started = time.perf_counter()
    array = np.stack([np.asarray(image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]),
        Image.Resampling.BOX), dtype=np.uint8) for image in raw])
    host = torch.from_numpy(array)[None]
    preprocessing = time.perf_counter() - started
    torch.cuda.synchronize()
    started = time.perf_counter()
    rgb = host.to("cuda").permute(0, 1, 4, 2, 3).float().div_(255.)
    torch.cuda.synchronize()
    transfer = time.perf_counter() - started
    started = time.perf_counter()
    # Normal training inference emits both classification and support heads.
    probabilities = torch.stack([model(rgb)[0].sigmoid() for model in models]).mean(0)
    scores = probabilities.cpu().numpy().copy()
    torch.cuda.synchronize()
    forward = time.perf_counter() - started
    total = time.perf_counter() - total_start
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite model scores")
    return dict(decode=decode, preprocessing=preprocessing,
                transfer_normalization=transfer, ensemble_forward_cpu_scores=forward, total=total), scores[0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--learned", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refuse overwrite: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frames, samples, checkpoints, provenance = validate_inputs(args.capture, args.learned)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    results = {}
    for arm in ARMS:
        models = []
        for seed in (17, 29, 43):
            model = WhiskerModel(arm).cuda().eval()
            model.load_state_dict(torch.load(checkpoints[(arm, seed)], map_location="cuda", weights_only=True))
            models.append(model)
        for sample in samples[:10]:
            window(models, sample, frames)
        rows, scores = [], []
        for sample in samples:
            timing, value = window(models, sample, frames)
            rows.append(timing)
            scores.append(value)
        results[arm] = dict(examples=50, ensemble_size=3, batch_size=1,
            milliseconds={key: dict(p50=float(np.percentile([r[key]*1000 for r in rows], 50)),
                                    p95=float(np.percentile([r[key]*1000 for r in rows], 95))) for key in rows[0]},
            per_window_milliseconds=[{key: value*1000 for key, value in row.items()} for row in rows],
            scores=scores, actual_backend_observation=asdict(torch_observation(model=models[0])))
        del model, models
        torch.cuda.empty_cache()
        print(json.dumps(dict(arm=arm, milliseconds=results[arm]["milliseconds"])), flush=True)
    result = dict(schema="nf-g8-whisker-offline-latency-v1", status="complete",
        device=torch.cuda.get_device_name(), actual_backend="cuda", torch_version=torch.__version__,
        cuda_version=torch.version.cuda, provenance=provenance, warmup_windows_per_arm=10,
        selection="first 50 test samples in unchanged dataset order; no labels read",
        sample_ids=[s["sample_id"] for s in samples], arms=results,
        scope="offline PNG window path, not real camera or phone or full alert latency; includes cold three-frame decode and redundant image work for single_frame for comparison",
        cold_definition="decoded-image cache absent; OS disk/page cache is not flushed; ten warmups precede measurement",
        timing_definition="decode includes PNG read/RGB conversion; preprocessing includes BOX resize/stack; transfer includes H2D and normalization; ensemble forward includes three models, sigmoid average and CPU scores; total includes all stage overhead; integrity hashing and checkpoint load excluded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
