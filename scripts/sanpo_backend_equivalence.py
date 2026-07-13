#!/usr/bin/env python3
"""Fail-closed Keras Torch/TensorFlow numerical-equivalence gate for SANPO.

The orchestrator creates deterministic synthetic RGB tensors and never opens a
dataset manifest or blind asset.  Each backend runs in a separate interpreter,
loads the same weights into the shared model graph, and writes logits for an
exact comparison.  The resulting report is consumed by the TensorFlow export
entrypoint before it imports TensorFlow or writes a TFLite model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import sanpo_segmentation_model


REPORT_SCHEMA = "blindassist_sanpo_backend_equivalence_v3"
INPUT_SIZE = 256
ALLOWED_INPUT_SIZES = (256, 384, 512)
NUM_CLASSES = 4
DEFAULT_BACKBONE_ALPHA = 0.75
DEFAULT_DECODER_CHANNELS = 96
DEFAULT_DETAIL_OUTPUT_STRIDE = 8
DEFAULT_SEMANTIC_OUTPUT_STRIDE = 32
ALLOWED_BACKBONE_ALPHAS = (0.75, 1.0)
FIXED_INPUT_SEED = 20260713
FIXED_INPUT_COUNT = 4
MAX_ABS_THRESHOLD = 1e-4
ARGMAX_AGREEMENT_THRESHOLD = 0.9998
TORCH_EXECUTION_CONTRACT = {
    "nvidia_tf32_override": "0",
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_config(
    backbone_alpha: float,
    decoder_channels: int,
    input_size: int = INPUT_SIZE,
    detail_output_stride: int = DEFAULT_DETAIL_OUTPUT_STRIDE,
    semantic_output_stride: int = DEFAULT_SEMANTIC_OUTPUT_STRIDE,
) -> dict[str, Any]:
    if backbone_alpha not in ALLOWED_BACKBONE_ALPHAS:
        raise ValueError(f"backbone_alpha must be one of {ALLOWED_BACKBONE_ALPHAS}")
    if decoder_channels <= 0:
        raise ValueError("decoder_channels must be positive")
    if input_size not in ALLOWED_INPUT_SIZES:
        raise ValueError(f"input_size must be one of {ALLOWED_INPUT_SIZES}")
    if detail_output_stride not in (4, 8):
        raise ValueError("detail_output_stride must be one of: 4, 8")
    if semantic_output_stride not in (16, 32):
        raise ValueError("semantic_output_stride must be one of: 16, 32")
    return {
        "architecture": "MobileNetV3Small+LR-ASPP",
        "architecture_revision": sanpo_segmentation_model.ARCHITECTURE_REVISION,
        "input_size": int(input_size),
        "num_classes": NUM_CLASSES,
        "backbone_alpha": float(backbone_alpha),
        "decoder_channels": int(decoder_channels),
        "detail_output_stride": int(detail_output_stride),
        "semantic_output_stride": int(semantic_output_stride),
    }


def model_config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fixed_inputs(input_size: int = INPUT_SIZE) -> np.ndarray:
    """Return deterministic 0..255 RGB inputs independent of any dataset."""
    if input_size not in ALLOWED_INPUT_SIZES:
        raise ValueError(f"input_size must be one of {ALLOWED_INPUT_SIZES}")
    rng = np.random.default_rng(FIXED_INPUT_SEED)
    values = rng.integers(
        0, 256, size=(FIXED_INPUT_COUNT, input_size, input_size, 3), dtype=np.uint8,
    ).astype(np.float32)
    ramp = np.linspace(0.0, 255.0, input_size, dtype=np.float32)
    values[0, :, :, 0] = ramp[None, :]
    values[0, :, :, 1] = ramp[:, None]
    values[0, :, :, 2] = 127.0
    values[1].fill(0.0)
    values[2].fill(255.0)
    return values


def run_worker(
    backend: str,
    weights: Path,
    inputs: Path,
    output: Path,
    backbone_alpha: float,
    decoder_channels: int,
    input_size: int,
    detail_output_stride: int,
    semantic_output_stride: int,
) -> None:
    os.environ["KERAS_BACKEND"] = backend
    if backend == "torch":
        # Also set the vendor override before importing torch.  The
        # orchestrator sets it before process launch; this is a fail-safe for
        # direct worker invocation.
        os.environ["NVIDIA_TF32_OVERRIDE"] = TORCH_EXECUTION_CONTRACT["nvidia_tf32_override"]
        import keras
        import torch

        if keras.backend.backend() != "torch":
            raise RuntimeError(f"Expected Keras torch backend, got {keras.backend.backend()!r}")
        # The equivalence gate compares float32 weights against TensorFlow's
        # float32 export graph.  CuDNN enables TF32 convolution by default on
        # supported GPUs; trained MobileNetV3 kernels amplify that reduced
        # mantissa enough to create false cross-backend failures.  Keep this
        # worker on the exact float32 path regardless of the trainer's GPU.
        torch.set_float32_matmul_precision(TORCH_EXECUTION_CONTRACT["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = TORCH_EXECUTION_CONTRACT["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = TORCH_EXECUTION_CONTRACT["cudnn_allow_tf32"]
        keras.mixed_precision.set_global_policy("float32")
        model = sanpo_segmentation_model.build_mobilenetv3_lraspp(
            keras, input_size, NUM_CLASSES, backbone_weights=None,
            backbone_alpha=backbone_alpha, decoder_channels=decoder_channels,
            detail_output_stride=detail_output_stride,
            semantic_output_stride=semantic_output_stride,
        )
        model.load_weights(weights)
        logits = model.predict(np.load(inputs), batch_size=FIXED_INPUT_COUNT, verbose=0)
    elif backend == "tensorflow":
        import tensorflow as tf

        tf.keras.mixed_precision.set_global_policy("float32")
        model = sanpo_segmentation_model.build_mobilenetv3_lraspp(
            tf.keras, input_size, NUM_CLASSES, backbone_weights=None,
            backbone_alpha=backbone_alpha, decoder_channels=decoder_channels,
            detail_output_stride=detail_output_stride,
            semantic_output_stride=semantic_output_stride,
        )
        model.load_weights(weights)
        logits = model.predict(np.load(inputs), batch_size=FIXED_INPUT_COUNT, verbose=0)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    np.save(output, np.asarray(logits, dtype=np.float32), allow_pickle=False)


def worker_command(
    python: Path,
    backend: str,
    weights: Path,
    inputs: Path,
    output: Path,
    backbone_alpha: float,
    decoder_channels: int,
    input_size: int = INPUT_SIZE,
    detail_output_stride: int = DEFAULT_DETAIL_OUTPUT_STRIDE,
    semantic_output_stride: int = DEFAULT_SEMANTIC_OUTPUT_STRIDE,
) -> list[str]:
    return [
        str(python), str(Path(__file__).resolve()), "--worker-backend", backend,
        "--weights", str(weights), "--fixed-input", str(inputs), "--worker-output", str(output),
        "--backbone-alpha", str(backbone_alpha), "--decoder-channels", str(decoder_channels),
        "--input-size", str(input_size),
        "--detail-output-stride", str(detail_output_stride),
        "--semantic-output-stride", str(semantic_output_stride),
    ]


def compare_logits(torch_logits: np.ndarray, tensorflow_logits: np.ndarray) -> dict[str, Any]:
    if torch_logits.shape != tensorflow_logits.shape:
        raise ValueError(
            f"backend output shape mismatch: torch={torch_logits.shape} tensorflow={tensorflow_logits.shape}"
        )
    difference = np.abs(torch_logits.astype(np.float64) - tensorflow_logits.astype(np.float64))
    max_abs = float(difference.max(initial=0.0))
    mean_abs = float(difference.mean())
    torch_argmax = np.argmax(torch_logits, axis=-1)
    tensorflow_argmax = np.argmax(tensorflow_logits, axis=-1)
    agreement = float(np.mean(torch_argmax == tensorflow_argmax))
    passed = max_abs <= MAX_ABS_THRESHOLD and agreement >= ARGMAX_AGREEMENT_THRESHOLD
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "argmax_agreement": agreement,
        "passed": passed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    report_path = args.report.resolve()
    expected_model_config = model_config(
        args.backbone_alpha, args.decoder_channels, args.input_size,
        args.detail_output_stride, args.semantic_output_stride,
    )
    with tempfile.TemporaryDirectory(prefix="sanpo-backend-equivalence-") as temp:
        root = Path(temp)
        inputs_path = root / "fixed-inputs.npy"
        torch_output = root / "torch-logits.npy"
        tensorflow_output = root / "tensorflow-logits.npy"
        np.save(inputs_path, fixed_inputs(args.input_size), allow_pickle=False)
        for python, backend, output in (
            (args.torch_python.resolve(), "torch", torch_output),
            (args.tensorflow_python.resolve(), "tensorflow", tensorflow_output),
        ):
            worker_env = os.environ.copy()
            if backend == "torch":
                worker_env["NVIDIA_TF32_OVERRIDE"] = TORCH_EXECUTION_CONTRACT["nvidia_tf32_override"]
            subprocess.run(
                worker_command(
                    python, backend, weights, inputs_path, output,
                    args.backbone_alpha, args.decoder_channels, args.input_size,
                    args.detail_output_stride, args.semantic_output_stride,
                ),
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                env=worker_env,
            )
        metrics = compare_logits(
            np.load(torch_output, allow_pickle=False),
            np.load(tensorflow_output, allow_pickle=False),
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "green" if metrics["passed"] else "red",
            "export_authorized": bool(metrics["passed"]),
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "model_config": expected_model_config,
            "model_config_sha256": model_config_sha256(expected_model_config),
            "model_definition": str(Path(sanpo_segmentation_model.__file__).resolve()),
            "model_definition_sha256": sha256_file(Path(sanpo_segmentation_model.__file__).resolve()),
            "equivalence_tool_sha256": sha256_file(Path(__file__).resolve()),
            "fixed_input": {
                "kind": "deterministic_synthetic_rgb_0_255",
                "seed": FIXED_INPUT_SEED,
                "count": FIXED_INPUT_COUNT,
                "shape": [FIXED_INPUT_COUNT, args.input_size, args.input_size, 3],
                "sha256": sha256_file(inputs_path),
                "dataset_access": "none",
                "blind_holdout_access": "not_accessed",
            },
            "thresholds": {
                "max_abs_lte": MAX_ABS_THRESHOLD,
                "argmax_agreement_gte": ARGMAX_AGREEMENT_THRESHOLD,
            },
            "metrics": metrics,
            "interpreters": {
                "torch": str(args.torch_python.resolve()),
                "tensorflow": str(args.tensorflow_python.resolve()),
            },
            "torch_execution_contract": TORCH_EXECUTION_CONTRACT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    write_json(report_path, report)
    digest = sha256_file(report_path)
    report_path.with_suffix(report_path.suffix + ".sha256").write_text(
        f"{digest}  {report_path.name}\n", encoding="ascii",
    )
    report["report_sha256"] = digest
    return report


def consume_equivalence_authorization(
    weights: Path,
    report_path: Path,
    *,
    backbone_alpha: float = DEFAULT_BACKBONE_ALPHA,
    decoder_channels: int = DEFAULT_DECODER_CHANNELS,
    input_size: int = INPUT_SIZE,
    detail_output_stride: int = DEFAULT_DETAIL_OUTPUT_STRIDE,
    semantic_output_stride: int = DEFAULT_SEMANTIC_OUTPUT_STRIDE,
) -> dict[str, Any]:
    """Verify a preregistered green report before TensorFlow export."""
    weights = weights.resolve()
    report_path = report_path.resolve()
    sidecar = report_path.with_suffix(report_path.suffix + ".sha256")
    if not report_path.is_file() or not sidecar.is_file():
        raise ValueError("backend-equivalence report and SHA256 sidecar are required")
    digest = sha256_file(report_path)
    if sidecar.read_text(encoding="ascii").strip().split()[0] != digest:
        raise ValueError("backend-equivalence report SHA256 sidecar mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("backend-equivalence report schema mismatch")
    expected_model_config = model_config(
        backbone_alpha, decoder_channels, input_size,
        detail_output_stride, semantic_output_stride,
    )
    if report.get("model_config") != expected_model_config:
        raise ValueError("backend-equivalence report model config differs from the requested export config")
    if report.get("model_config_sha256") != model_config_sha256(expected_model_config):
        raise ValueError("backend-equivalence report model config hash mismatch")
    expected_thresholds = {
        "max_abs_lte": MAX_ABS_THRESHOLD,
        "argmax_agreement_gte": ARGMAX_AGREEMENT_THRESHOLD,
    }
    if report.get("thresholds") != expected_thresholds:
        raise ValueError("backend-equivalence thresholds differ from preregistered values")
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    max_abs = float(metrics.get("max_abs", float("inf")))
    argmax_agreement = float(metrics.get("argmax_agreement", -1.0))
    if (
        report.get("status") != "green"
        or report.get("export_authorized") is not True
        or metrics.get("passed") is not True
        or not math.isfinite(max_abs)
        or not math.isfinite(argmax_agreement)
        or max_abs > MAX_ABS_THRESHOLD
        or argmax_agreement < ARGMAX_AGREEMENT_THRESHOLD
    ):
        raise ValueError("backend-equivalence report does not authorize export")
    if not weights.is_file() or report.get("weights_sha256") != sha256_file(weights):
        raise ValueError("backend-equivalence report is bound to different weights")
    model_path = Path(sanpo_segmentation_model.__file__).resolve()
    if report.get("model_definition_sha256") != sha256_file(model_path):
        raise ValueError("backend-equivalence report is bound to a different model definition")
    if report.get("equivalence_tool_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("backend-equivalence report is bound to a different equivalence tool")
    if report.get("torch_execution_contract") != TORCH_EXECUTION_CONTRACT:
        raise ValueError("backend-equivalence report lacks the exact-float32 torch execution contract")
    fixed = report.get("fixed_input") if isinstance(report.get("fixed_input"), dict) else {}
    expected_fixed = {
        "kind": "deterministic_synthetic_rgb_0_255",
        "seed": FIXED_INPUT_SEED,
        "count": FIXED_INPUT_COUNT,
        "shape": [FIXED_INPUT_COUNT, input_size, input_size, 3],
        "dataset_access": "none",
        "blind_holdout_access": "not_accessed",
    }
    if any(fixed.get(key) != value for key, value in expected_fixed.items()):
        raise ValueError("backend-equivalence report does not prove dataset isolation")
    if len(str(fixed.get("sha256", ""))) != 64:
        raise ValueError("backend-equivalence report lacks the fixed-input hash")
    verified = dict(report)
    verified["report_sha256"] = digest
    return verified


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--torch-python", type=Path)
    parser.add_argument("--tensorflow-python", type=Path)
    parser.add_argument("--worker-backend", choices=("torch", "tensorflow"), help=argparse.SUPPRESS)
    parser.add_argument("--fixed-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--backbone-alpha", type=float, choices=ALLOWED_BACKBONE_ALPHAS,
        default=DEFAULT_BACKBONE_ALPHA,
    )
    parser.add_argument("--decoder-channels", type=int, default=DEFAULT_DECODER_CHANNELS)
    parser.add_argument("--input-size", type=int, choices=ALLOWED_INPUT_SIZES, default=INPUT_SIZE)
    parser.add_argument("--detail-output-stride", type=int, choices=(4, 8), default=DEFAULT_DETAIL_OUTPUT_STRIDE)
    parser.add_argument("--semantic-output-stride", type=int, choices=(16, 32), default=DEFAULT_SEMANTIC_OUTPUT_STRIDE)
    args = parser.parse_args(argv)
    if args.worker_backend:
        if args.fixed_input is None or args.worker_output is None:
            parser.error("worker mode requires --fixed-input and --worker-output")
    elif args.report is None or args.torch_python is None or args.tensorflow_python is None:
        parser.error("orchestrator mode requires --report, --torch-python and --tensorflow-python")
    if args.decoder_channels <= 0:
        parser.error("--decoder-channels must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.worker_backend:
        run_worker(
            args.worker_backend,
            args.weights.resolve(),
            args.fixed_input.resolve(),
            args.worker_output.resolve(),
            args.backbone_alpha,
            args.decoder_channels,
            args.input_size,
            args.detail_output_stride,
            args.semantic_output_stride,
        )
        return 0
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "max_abs": report["metrics"]["max_abs"],
        "argmax_agreement": report["metrics"]["argmax_agreement"],
        "model_config": report["model_config"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False))
    return 0 if report["export_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
