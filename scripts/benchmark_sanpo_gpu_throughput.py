#!/usr/bin/env python3
"""Measure SANPO model GPU throughput without reading any dataset assets."""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch

import sanpo_segmentation_model


def measure(batch_size: int, steps: int, warmup: int, jit_compile: bool) -> dict[str, object]:
    keras.backend.clear_session()
    torch.cuda.empty_cache()
    model = sanpo_segmentation_model.build_mobilenetv3_lraspp(keras, 256, 4, None)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=jit_compile,
    )
    images = torch.rand((batch_size, 256, 256, 3), device="cuda") * 255.0
    masks = torch.randint(0, 4, (batch_size, 256, 256), device="cuda")
    for _ in range(warmup):
        model.train_on_batch(images, masks)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        model.train_on_batch(images, masks)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "steps": steps,
        "seconds": seconds,
        "images_per_second": batch_size * steps / seconds,
        "milliseconds_per_step": seconds * 1000.0 / steps,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "jit_compile": jit_compile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="16,32,64,96")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--jit-compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    keras.mixed_precision.set_global_policy("mixed_float16")
    torch.set_float32_matmul_precision("high")
    results = []
    for value in args.batches.split(","):
        batch_size = int(value)
        try:
            results.append(measure(batch_size, args.steps, args.warmup, args.jit_compile))
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": batch_size, "error": "cuda_out_of_memory"})
    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "model": "MobileNetV3Small(alpha=0.75)+LR-ASPP",
        "parameters": sanpo_segmentation_model.build_mobilenetv3_lraspp(keras, 256, 4, None).count_params(),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
