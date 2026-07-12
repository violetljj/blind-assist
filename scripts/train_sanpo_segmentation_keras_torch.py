#!/usr/bin/env python3
"""Train SANPO candidate with Keras 3 on the native-Windows PyTorch CUDA backend."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import sanpo_training_gate as training_gate
import train_export_sanpo_segmentation as shared


DEFAULT_WEIGHTS = "test-artifacts.local/segmentation-candidate/torch/mobilenetv3_lraspp.weights.h5"
DEFAULT_REPORT = "test-artifacts.local/segmentation-candidate/torch/training_report.json"


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = shared.project_root()
    dataset_root = resolve(root, args.dataset_root).resolve()
    manifest = dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST
    report_path = resolve(root, args.report).resolve()
    gate_path = resolve(dataset_root, args.training_gate_report).resolve()
    gate_report = training_gate.consume_training_authorization(dataset_root, gate_path)

    records = shared.load_records(manifest)
    train_records = shared.records_by_split(records, "train")
    dev_records = shared.records_by_split(records, "dev")
    for record in records:
        shared.validate_binary_masks(record)

    # Keras selects its backend at import time. Fail closed if the parent process
    # imported another backend or configured a conflicting value.
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import torch

    if keras.backend.backend() != "torch":
        raise RuntimeError(f"Expected KERAS_BACKEND=torch, got {keras.backend.backend()!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable; refusing to silently train on CPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    keras.utils.set_random_seed(args.seed)

    def preload(values: Sequence[shared.Record]) -> tuple[np.ndarray, np.ndarray]:
        examples = [shared.load_example(record, args.input_size) for record in values]
        images, masks = zip(*examples)
        return np.stack(images).astype(np.float32), np.stack(masks).astype(np.int64)

    train_images, train_masks = preload(train_records)
    dev_images, dev_masks = preload(dev_records)
    keras.mixed_precision.set_global_policy("mixed_float16")
    torch.set_float32_matmul_precision("high")

    model = shared.sanpo_segmentation_model.build_mobilenetv3_lraspp(
        keras, args.input_size, len(shared.CLASS_NAMES), backbone_weights="imagenet"
    )
    pixel_counts = shared.class_pixel_counts(train_records)
    counts = np.asarray([pixel_counts[name] for name in shared.CLASS_NAMES], dtype=np.float64)
    frequencies = counts / max(1.0, counts.sum())
    raw_weights = 1.0 / np.sqrt(np.maximum(frequencies, 1e-12))
    class_weights = np.clip(raw_weights / raw_weights.mean(), 0.25, 8.0).astype(np.float32)

    class WeightedSparseCrossentropy(keras.losses.Loss):
        def call(self, y_true: Any, y_pred: Any) -> Any:
            losses = keras.ops.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
            weights = keras.ops.take(
                keras.ops.convert_to_tensor(class_weights), keras.ops.cast(y_true, "int32")
            )
            return losses * weights

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=WeightedSparseCrossentropy(name="inverse_sqrt_frequency_crossentropy"),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy")],
        jit_compile=args.jit_compile,
    )
    fit_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    history = model.fit(
        train_images,
        train_masks,
        validation_data=(dev_images, dev_masks),
        batch_size=args.batch_size,
        shuffle=True,
        epochs=args.epochs,
        verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, min_delta=1e-4, restore_best_weights=True,
        )],
    )
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - fit_started
    dev_logits = model.predict(dev_images, batch_size=args.batch_size, verbose=0)
    dev_metrics = shared.confusion_and_metrics(
        np.argmax(dev_logits, axis=-1), dev_masks,
    )
    weights = resolve(root, args.weights).resolve()
    if not weights.name.endswith(".weights.h5"):
        raise ValueError("Keras weight output must end with .weights.h5")
    weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights)
    report = {
        "schema_version": 1,
        "candidate": "MobileNetV3Small(alpha=0.75)+LR-ASPP",
        "benchmark_only": True,
        "backend": "keras3_torch",
        "device": str(torch.cuda.get_device_name(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mixed_precision_policy": keras.mixed_precision.global_policy().name,
        "data_pipeline": "preloaded_numpy",
        "gpu_performance": {
            "jit_compile": args.jit_compile,
            "fit_seconds": fit_seconds,
            "training_images_per_second": (
                len(train_records) * len(history.history.get("loss", [])) / max(fit_seconds, 1e-9)
            ),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "manifest": str(manifest),
        "manifest_sha256": shared.sha256_file(manifest),
        "training_gate_report": str(gate_path),
        "training_gate_report_sha256": gate_report["report_sha256"],
        "blind_holdout_access": "not_accessed_by_trainer",
        "record_counts": {"train": len(train_records), "dev": len(dev_records)},
        "class_pixel_counts": pixel_counts,
        "class_loss_weights": {name: float(class_weights[index]) for index, name in enumerate(shared.CLASS_NAMES)},
        "training": {"requested_epochs": args.epochs, "completed_epochs": len(history.history.get("loss", [])), "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed, "early_stopping": {"monitor": "val_loss", "patience": 5, "restore_best_weights": True}, "history": history.history},
        "dev_mask_metrics": dev_metrics,
        "weights": str(weights),
        "weights_sha256": shared.sha256_file(weights),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    shared.write_json(report_path, report)
    Path(str(report_path) + ".sha256").write_text(shared.sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the gated SANPO candidate with Keras 3 + PyTorch CUDA.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--training-gate-report", default="qa/training_gate_report.json")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--input-size", type=int, default=shared.INPUT_SIZE, choices=[shared.INPUT_SIZE])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="GPU batch size; 64 is the measured safe default for the local 8 GB RTX 5060.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--jit-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile through Keras. Benchmark both modes; tiny runs may be faster without it.",
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    return args


def main() -> None:
    report = run(parse_args())
    print(f"weights={report['weights']}")
    print(f"weights_sha256={report['weights_sha256']}")
    print(f"device={report['device']}")


if __name__ == "__main__":
    main()
