#!/usr/bin/env python3
"""Train/export the benchmark-only candidate only after the total v3 gate is green.

There is intentionally no ``--manifest`` option.  This entrypoint accepts a
dataset root only, runs the hash-attested total gate itself, and then opens only
that root's canonical ``training_manifest.jsonl``.  It never receives a blind
directory or manifest, so an ordinary JSONL cannot bypass the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

import sanpo_training_gate as training_gate
import sanpo_segmentation_model


CLASS_NAMES = ("walkable", "boundary_step_curb", "obstacle", "unknown_nonwalkable")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
INPUT_SIZE = 256
DEFAULT_OUTPUT = "device-benchmark/benchmark-assets.local/segmentation/mobilenetv3_lraspp_int8_256.tflite"
DEFAULT_REPORT = "test-artifacts.local/segmentation-candidate/latest"
ALLOWED_SPLITS = {"train", "dev"}
FORBIDDEN_SPLITS = {"blind", "blind_holdout", "holdout", "test"}


@dataclass(frozen=True)
class Record:
    sample_id: str
    split: str
    session_id: str
    image_path: Path
    masks: dict[str, Path]
    semantic_mask_path: Path | None
    scene_bucket: str | None
    label_authority: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: every manifest row must be an object")
            yield value


def split_for(row: dict[str, Any]) -> str:
    value = row.get("segmentation_split", row.get("split"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{row.get('id', '<unknown>')}: missing segmentation_split")
    return value.strip().lower()


def session_for(row: dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    candidates = (row.get("source_session_id"), row.get("session_id"), source.get("session_id"), row.get("sequence_id"))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"{row.get('id', '<unknown>')}: missing source session ID")


def masks_for(row: dict[str, Any], manifest_dir: Path) -> tuple[dict[str, Path], Path | None]:
    indexed_mask = row.get("semantic_mask_path")
    if isinstance(indexed_mask, str) and indexed_mask.strip():
        path = resolve(manifest_dir, indexed_mask).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{row.get('id', '<unknown>')}: semantic_mask_path is missing: {path}")
        return {}, path
    masks = row.get("semantic_mask_paths", row.get("semantic_masks"))
    if not isinstance(masks, dict):
        raise ValueError(
            f"{row.get('id', '<unknown>')}: semantic_mask_paths must map all four classes to binary PNG masks"
        )
    unknown = sorted(set(masks) - set(CLASS_NAMES))
    missing = [name for name in CLASS_NAMES if not isinstance(masks.get(name), str) or not masks[name].strip()]
    if unknown or missing:
        raise ValueError(f"{row.get('id', '<unknown>')}: mask classes unknown={unknown}, missing={missing}")
    resolved = {name: resolve(manifest_dir, masks[name]).resolve() for name in CLASS_NAMES}
    for name, path in resolved.items():
        if not path.is_file():
            raise FileNotFoundError(f"{row.get('id', '<unknown>')}: {name} mask is missing: {path}")
    return resolved, None


def load_records(manifest: Path) -> list[Record]:
    """Read only the explicit train/dev manifest. Blind rows are a hard failure."""
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest}")
    records: list[Record] = []
    sessions: dict[str, str] = {}
    for row in json_lines(manifest):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Every training manifest row requires a non-empty id")
        split = split_for(row)
        if split in FORBIDDEN_SPLITS:
            raise ValueError(
                f"{sample_id}: split={split!r} is forbidden here; pass a train/dev-only manifest and keep blind labels separate"
            )
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"{sample_id}: split must be one of {sorted(ALLOWED_SPLITS)}, got {split!r}")
        label_authority = str(row.get("label_authority", "")).strip()
        if label_authority not in training_gate.validator.LABEL_AUTHORITIES:
            raise ValueError(f"{sample_id}: missing or unsupported label_authority")
        if split == "dev":
            if label_authority == "teacher_consensus_pseudo_label":
                raise ValueError(f"{sample_id}: dev forbids teacher/pseudo labels")
            if label_authority == "procedural_ground_truth":
                authority_errors = training_gate.validator.validate_label_authority(
                    row, sample_id, split, manifest.parent,
                )
                if authority_errors:
                    raise ValueError(f"{sample_id}: invalid procedural dev label: {'; '.join(authority_errors)}")
        session_id = session_for(row)
        earlier = sessions.setdefault(session_id, split)
        if earlier != split:
            raise ValueError(f"source session leakage: {session_id!r} appears in both {earlier!r} and {split!r}")
        image_value = row.get("image_path")
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError(f"{sample_id}: missing image_path")
        image_path = resolve(manifest.parent, image_value).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{sample_id}: image is missing: {image_path}")
        masks, semantic_mask_path = masks_for(row, manifest.parent)
        records.append(
            Record(
                sample_id=sample_id,
                split=split,
                session_id=session_id,
                image_path=image_path,
                masks=masks,
                semantic_mask_path=semantic_mask_path,
                scene_bucket=row.get("scene_bucket") if isinstance(row.get("scene_bucket"), str) else None,
                label_authority=label_authority,
            )
        )
    if not records:
        raise ValueError("Training manifest contains no records")
    if not any(item.split == "train" for item in records) or not any(item.split == "dev" for item in records):
        raise ValueError("Training manifest must contain both train and dev records")
    return records


def validate_binary_masks(record: Record) -> np.ndarray:
    """Read the canonical ID mask or combine four binary masks into class IDs."""
    with Image.open(record.image_path) as image:
        width, height = image.size
    if record.semantic_mask_path is not None:
        with Image.open(record.semantic_mask_path) as mask:
            if mask.size != (width, height):
                raise ValueError(
                    f"{record.sample_id}: semantic mask size {mask.size} != image {(width, height)}"
                )
            target = np.asarray(mask.convert("L"), dtype=np.uint8)
        if target.size == 0 or int(target.min()) < 0 or int(target.max()) >= len(CLASS_NAMES):
            raise ValueError(
                f"{record.sample_id}: semantic_mask_path IDs must be in 0..{len(CLASS_NAMES) - 1}"
            )
        return target
    channels: list[np.ndarray] = []
    for name in CLASS_NAMES:
        with Image.open(record.masks[name]) as mask:
            if mask.size != (width, height):
                raise ValueError(f"{record.sample_id}: {name} mask size {mask.size} != image {(width, height)}")
            values = np.asarray(mask.convert("L"))
        channels.append(values > 0)
    stacked = np.stack(channels, axis=-1)
    owners = stacked.sum(axis=-1)
    if np.any(owners > 1):
        raise ValueError(f"{record.sample_id}: semantic masks overlap; each pixel must have one class")
    if np.any(owners == 0):
        raise ValueError(f"{record.sample_id}: semantic masks leave unlabeled pixels; use unknown_nonwalkable explicitly")
    return np.argmax(stacked, axis=-1).astype(np.uint8)


def load_example(record: Record, input_size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(record.image_path) as image:
        rgb = np.asarray(image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR), dtype=np.float32)
    target = validate_binary_masks(record)
    target_image = Image.fromarray(target, mode="L").resize((input_size, input_size), Image.Resampling.NEAREST)
    return rgb, np.asarray(target_image, dtype=np.uint8)


def records_by_split(records: Sequence[Record], split: str) -> list[Record]:
    return [record for record in records if record.split == split]


def scene_coverage(records: Sequence[Record]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        if record.scene_bucket:
            result[record.scene_bucket] = result.get(record.scene_bucket, 0) + 1
    return dict(sorted(result.items()))


def class_pixel_counts(records: Sequence[Record]) -> dict[str, int]:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for record in records:
        target = validate_binary_masks(record)
        counts += np.bincount(target.reshape(-1), minlength=len(CLASS_NAMES))
    return {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)}


def make_dataset(tf: Any, records: Sequence[Record], input_size: int, batch_size: int, shuffle: bool, seed: int) -> Any:
    def generator() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        order = list(records)
        if shuffle:
            random.Random(seed).shuffle(order)
        for record in order:
            yield load_example(record, input_size)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(input_size, input_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(input_size, input_size), dtype=tf.uint8),
        ),
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_mobilenetv3_lraspp(tf: Any, input_size: int, num_classes: int = len(CLASS_NAMES)) -> Any:
    """Compatibility wrapper around the backend-neutral authoritative graph."""
    return sanpo_segmentation_model.build_mobilenetv3_lraspp(tf.keras, input_size, num_classes)


def confusion_and_metrics(predictions: Iterable[np.ndarray], targets: Iterable[np.ndarray]) -> dict[str, Any]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        predicted = np.asarray(prediction, dtype=np.int64).reshape(-1)
        expected = np.asarray(target, dtype=np.int64).reshape(-1)
        if predicted.shape != expected.shape:
            raise ValueError("prediction/target shape mismatch")
        valid = (expected >= 0) & (expected < len(CLASS_NAMES)) & (predicted >= 0) & (predicted < len(CLASS_NAMES))
        np.add.at(confusion, (expected[valid], predicted[valid]), 1)
    per_class: dict[str, dict[str, float | int]] = {}
    ious: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        denominator = tp + fp + fn
        iou = float(tp / denominator) if denominator else 0.0
        per_class[name] = {"iou": iou, "true_pixels": int(confusion[index, :].sum()), "predicted_pixels": int(confusion[:, index].sum())}
        ious.append(iou)
    return {"confusion_matrix": confusion.tolist(), "per_class": per_class, "mean_iou": float(np.mean(ious)), "pixel_accuracy": float(np.trace(confusion) / max(1, confusion.sum()))}


def evaluate_model(model: Any, records: Sequence[Record], input_size: int, batch_size: int) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        images, labels = zip(*(load_example(record, input_size) for record in chunk))
        logits = model.predict(np.stack(images), verbose=0)
        predictions.extend(np.argmax(logits, axis=-1))
        targets.extend(labels)
    return confusion_and_metrics(predictions, targets)


def representative_data(records: Sequence[Record], input_size: int, limit: int) -> Iterator[list[np.ndarray]]:
    for record in list(records)[: max(1, min(limit, len(records)))]:
        image, _ = load_example(record, input_size)
        yield [np.expand_dims(image, axis=0).astype(np.float32)]


def export_full_int8(tf: Any, model: Any, train_records: Sequence[Record], output: Path, input_size: int, representative_limit: int) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data(train_records, input_size, representative_limit)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(model_bytes)


def validate_int8_tflite(tf: Any, model_path: Path, input_size: int = INPUT_SIZE) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise AssertionError(f"Expected one input and one output, got {len(inputs)} inputs / {len(outputs)} outputs")
    input_detail, output_detail = inputs[0], outputs[0]
    expected_input = [1, input_size, input_size, 3]
    expected_output = [1, input_size, input_size, len(CLASS_NAMES)]
    if list(input_detail["shape"]) != expected_input or list(output_detail["shape"]) != expected_output:
        raise AssertionError(f"TFLite shape mismatch: input={input_detail['shape']}, output={output_detail['shape']}")
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise AssertionError(f"TFLite must be full INT8: input={input_detail['dtype']}, output={output_detail['dtype']}")
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    if input_scale <= 0 or output_scale <= 0:
        raise AssertionError(f"TFLite tensors are not quantized: input={input_detail['quantization']} output={output_detail['quantization']}")
    return {
        "input": {"shape": expected_input, "dtype": "int8", "quantization": [float(input_scale), int(input_zero)]},
        "output": {"shape": expected_output, "dtype": "int8", "quantization": [float(output_scale), int(output_zero)]},
        "size_bytes": model_path.stat().st_size,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_determinism(tf: Any, seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = project_root()
    dataset_root = resolve(root, args.dataset_root).resolve()
    manifest = dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST
    output = resolve(root, args.output)
    report_dir = resolve(root, args.report_dir)
    gate_report_path = resolve(dataset_root, args.training_gate_report).resolve()
    if output.resolve().is_relative_to((root / "app" / "src" / "main" / "assets").resolve()):
        raise ValueError("Refusing production app assets: this candidate is benchmark-only")
    gate_report = training_gate.consume_training_authorization(dataset_root, gate_report_path)
    records = load_records(manifest)
    train_records = records_by_split(records, "train")
    dev_records = records_by_split(records, "dev")
    for record in records:
        validate_binary_masks(record)
    import tensorflow as tf

    set_determinism(tf, args.seed)
    model = build_mobilenetv3_lraspp(tf, args.input_size)
    if args.import_weights:
        weights_path = resolve(root, args.import_weights).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Imported Keras weights not found: {weights_path}")
        model.load_weights(weights_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy")],
    )
    if args.export_only:
        history_payload: dict[str, list[Any]] = {}
    else:
        train_data = make_dataset(tf, train_records, args.input_size, args.batch_size, shuffle=True, seed=args.seed)
        dev_data = make_dataset(tf, dev_records, args.input_size, args.batch_size, shuffle=False, seed=args.seed)
        history_payload = model.fit(train_data, validation_data=dev_data, epochs=args.epochs, verbose=2).history
    dev_metrics = evaluate_model(model, dev_records, args.input_size, args.batch_size)
    export_full_int8(tf, model, train_records, output, args.input_size, args.representative_samples)
    tflite_contract = validate_int8_tflite(tf, output, args.input_size)
    report = {
        "schema_version": 1,
        "candidate": "MobileNetV3Small(alpha=0.75)+LR-ASPP",
        "benchmark_only": True,
        "promotion": "do_not_replace_default_model",
        "classes": list(CLASS_NAMES),
        "input_contract": "NHWC 256x256 RGB, int8 TFLite; dequantize then preserve 0..255 RGB values",
        "output_contract": "NHWC 256x256x4 int8 logits ordered as classes; argmax yields semantic class ID",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "training_gate_report": str(gate_report_path),
        "training_gate_report_sha256": gate_report["report_sha256"],
        "blind_holdout_access": "not_accessed_by_trainer: preflight is the only component permitted to inspect the benchmark-only holdout",
        "record_counts": {"train": len(train_records), "dev": len(dev_records)},
        "session_counts": {"train": len({item.session_id for item in train_records}), "dev": len({item.session_id for item in dev_records})},
        "scene_coverage": scene_coverage(records),
        "class_pixel_counts": class_pixel_counts(records),
        "training": {"epochs": 0 if args.export_only else args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed, "history": history_payload, "import_weights": args.import_weights, "export_only": args.export_only},
        "dev_mask_metrics": dev_metrics,
        "tflite_contract": tflite_contract,
        "output": str(output),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(report_dir / "segmentation_candidate_report.json", report)
    shutil.copy2(output, report_dir / output.name)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/export a benchmark-only full-INT8 four-class MobileNetV3 + LR-ASPP candidate.")
    parser.add_argument("--dataset-root", required=True, help="Canonical v3 dataset root; the entrypoint accepts no arbitrary manifest.")
    parser.add_argument("--training-gate-report", default="qa/training_gate_report.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Benchmark-only TFLite output; production app assets are rejected.")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE, choices=[INPUT_SIZE])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--representative-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--import-weights", help="Backend-neutral Keras .weights.h5 produced by the torch trainer.")
    parser.add_argument("--export-only", action="store_true", help="Skip fitting; requires --import-weights, then evaluate and export full INT8.")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.representative_samples <= 0:
        parser.error("epochs, batch-size, learning-rate and representative-samples must be positive")
    if args.export_only and not args.import_weights:
        parser.error("--export-only requires --import-weights")
    return args


def main() -> None:
    report = run(parse_args())
    print(f"candidate_tflite={report['output']}")
    print(f"dev_mean_iou={report['dev_mask_metrics']['mean_iou']:.4f}")
    print("promotion=do_not_replace_default_model")


if __name__ == "__main__":
    main()
