"""Fixed 11-frame synthetic RGB/native-depth reference diagnostic, without fitting."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from near_field import Camera, NearFieldEncoder, quantile_baseline
from research_backend import torch_observation


ARMS = ("quantile_stride4", "quantile_dense", "near_field")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Sensor path escapes the model directory")
    return path


def _finite(value):
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    return value


def _distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "p50": float(np.quantile(values, .50)) if values.size else None,
            "p95": float(np.quantile(values, .95)) if values.size else None,
            "mean": float(values.mean()) if values.size else None}


def _metric_class(metric_source: Path):
    package = metric_source / "depth_anything_v2"
    dpt = package / "dpt.py"
    if not dpt.is_file() or not (metric_source.name == "metric_depth"):
        raise ValueError("metric_source must be the existing metric_depth directory")
    for name, module in tuple(sys.modules.items()):
        if name == "depth_anything_v2" or name.startswith("depth_anything_v2."):
            location = getattr(module, "__file__", None)
            if location is not None and not Path(location).resolve().is_relative_to(metric_source):
                raise RuntimeError("Conflicting depth_anything_v2 source already imported: " + name)
            if any(not Path(path).resolve().is_relative_to(metric_source)
                   for path in getattr(module, "__path__", ())):
                raise RuntimeError("Conflicting depth_anything_v2 package search path: " + name)
    sys.path.insert(0, str(metric_source))
    try:
        module = importlib.import_module("depth_anything_v2.dpt")
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != dpt.resolve():
        raise RuntimeError("Metric model import resolved to a different source")
    files = {str(path.relative_to(metric_source)).replace("\\", "/"): _sha(path)
             for path in sorted(package.rglob("*.py"))}
    return module.DepthAnythingV2, {
        "directory": str(metric_source), "files_sha256": files,
        "closure_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }


def _decode(root: Path, row: dict, calibration: dict) -> dict:
    rgb_path = _contained(root, row["rgb_path"])
    depth_path = _contained(root, row["depth_path"])
    rgb_bytes, native_bytes = rgb_path.read_bytes(), depth_path.read_bytes()
    started = time.perf_counter()
    bgr = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb_decode_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    native = np.load(io.BytesIO(native_bytes), allow_pickle=False)
    native_decode_ms = (time.perf_counter() - started) * 1000
    shape = (int(calibration["height"]), int(calibration["width"]))
    if bgr is None or bgr.shape != (*shape, 3) or native.shape != shape or native.dtype != np.float32:
        raise ValueError("RGB/native raster violates source calibration or float32 depth contract")
    cam, wearer = row["camera_transform"], row["wearer_transform"]
    if not np.isclose(cam["yaw"], wearer["yaw"], atol=1e-8, rtol=0):
        raise ValueError("Camera/wearer yaw must agree for this prototype")
    if not np.isclose(cam["roll"], 0, atol=1e-8) or not np.isclose(wearer["roll"], 0, atol=1e-8):
        raise ValueError("Nonzero roll is outside the bounded camera contract")
    camera = Camera(width=shape[1], height=shape[0], hfov_deg=float(calibration["horizontal_fov_degrees"]),
                    camera_height_m=float(cam["z"] - wearer["z"]), pitch_deg=float(cam["pitch"]))
    return {"source": row, "bgr": bgr, "native": native, "camera": camera,
            "rgb_sha256": hashlib.sha256(rgb_bytes).hexdigest(),
            "native_sha256": hashlib.sha256(native_bytes).hexdigest(),
            "rgb_decode_ms": rgb_decode_ms, "native_npy_decode_ms": native_decode_ms}


def _postprocess(depth: np.ndarray, camera: Camera, encoder: NearFieldEncoder, arm: str, device):
    synchronize = arm == "near_field" and torch.device(device).type == "cuda"
    if synchronize:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if arm == "near_field":
        evidence = encoder.encode(depth)
        alerts = evidence.alerts()
        result = {"alerts": alerts, "summary": evidence.summary(), "token_bytes": evidence.token_bytes,
                  "device": str(encoder.device)}
    else:
        distance = quantile_baseline(depth, camera, stride=4 if arm == "quantile_stride4" else 1)
        alerts = np.isfinite(distance) & (distance <= 3.0)
        result = {"alerts": alerts, "distance_m": distance, "token_bytes": int(distance.nbytes), "device": "cpu",
                  "state_limit": "Quantile baselines return distances only; no independently validated clear state."}
    if synchronize:
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - started) * 1000


def _agreement(predicted, reference) -> dict:
    pred, ref = np.asarray(predicted, dtype=bool), np.asarray(reference, dtype=bool)
    return {"tp": int((pred & ref).sum()), "fp": int((pred & ~ref).sum()),
            "fn": int((~pred & ref).sum()), "tn": int((~pred & ~ref).sum())}


def _preview(first, last, output: Path) -> str | None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    canvas = Image.new("RGB", (960, 424), "white")
    draw = ImageDraw.Draw(canvas)
    for row, sample in enumerate((first, last)):
        bgr, native, predicted, index = sample
        tiles = [Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))]
        for depth in (native, predicted):
            valid = np.isfinite(depth) & (depth > 0)
            normalized = np.clip(np.nan_to_num(depth, nan=0, posinf=12, neginf=0) / 12, 0, 1)
            color = np.stack((1 - normalized, 1 - abs(2 * normalized - 1), normalized), axis=-1)
            color[~valid] = 0
            tiles.append(Image.fromarray(np.uint8(np.clip(color * 255, 0, 255))))
        for col, (tile, label) in enumerate(zip(tiles, ("RGB", "native depth reference", "RGB metric prediction"))):
            y = row * 202
            draw.text((col * 320 + 5, y + 3), f"frame {index}: {label}", fill="black")
            canvas.paste(tile.resize((320, 180)), (col * 320, y + 20))
    draw.text((5, 407), "Depth colors use fixed 0-12 m scale; black invalid. Synthetic source, no obstacle ground truth.", fill="black")
    path = output / "rgb_depth_first_last.png"
    with path.open("xb") as stream:
        canvas.save(stream, format="PNG")
    return str(path)


def run_replay(*, model_dir: Path, metric_source: Path, weights: Path, output: Path, device: str) -> dict:
    """Run all eleven source frames once after one first-frame warmup.

    Same three postprocessors see native and independently RGB-predicted depth.
    Native decisions are per-arm references, never independent obstacle truth.
    device selects the representation backend; metric inference stays on CUDA:0.
    """
    model_dir, metric_source = Path(model_dir).resolve(strict=True), Path(metric_source).resolve(strict=True)
    weights, output = Path(weights).resolve(strict=True), Path(output).resolve()
    if output == model_dir or output.is_relative_to(model_dir):
        raise ValueError("Outputs must not modify the source model directory")
    representation_device = torch.device(device)
    selected = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded GPU-first replay requires an available CUDA device; no silent CPU fallback")
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != "ue-sample-segment-v1":
        raise ValueError("Only the disclosed ue-sample-segment-v1 source is admitted")
    episodes = manifest["episodes"]
    if len(episodes) != 1 or len(episodes[0]["frames"]) != 11:
        raise ValueError("The bounded source must contain exactly one episode and all 11 frames")
    rows = episodes[0]["frames"]
    if [row["sample_index"] for row in rows] != list(range(11)):
        raise ValueError("Frames must preserve the complete ordered 0..10 source prefix")
    # Only these sensor fields are consumed; no plan, labels, actor or evaluator files are opened.
    decoded = [_decode(model_dir, row, manifest["calibration"]) for row in rows]
    model_class, source_identity = _metric_class(metric_source)
    weight_hash = _sha(weights)
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predicted_depth"
    prediction_dir.mkdir(exist_ok=False)
    report_rows, error_samples, inference_ms, inference_peaks = [], [], [], []
    post_ms = {branch: {arm: [] for arm in ARMS} for branch in ("native", "rgb")}
    totals = {arm: dict(tp=0, fp=0, fn=0, tn=0) for arm in ARMS}
    model, encoders = None, {}
    first_preview = last_preview = None
    try:
        with torch.cuda.device(selected), torch.inference_mode():
            allocation_before_model = int(torch.cuda.memory_allocated(selected))
            model = model_class(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            del state
            model.to(selected).eval()
            observation = asdict(torch_observation(model=model))
            actual_device = str(next(model.parameters()).device)
            if observation["device_type"] != "cuda":
                raise RuntimeError("Actual model parameters are not on CUDA")
            model_resident_delta = int(torch.cuda.memory_allocated(selected)) - allocation_before_model
            for sample in decoded:
                camera = sample["camera"]
                if camera not in encoders:
                    encoders[camera] = NearFieldEncoder(camera, device=str(representation_device))
            # Source infer_image automatically chooses CUDA; the context above fixes its device index.
            warmup = model.infer_image(decoded[0]["bgr"], 518)
            torch.cuda.synchronize(selected)
            for depth in (decoded[0]["native"], warmup):
                for arm in ARMS:
                    _postprocess(depth, decoded[0]["camera"], encoders[decoded[0]["camera"]], arm, representation_device)
            del warmup
            for sample in decoded:
                row, camera, native = sample["source"], sample["camera"], sample["native"]
                torch.cuda.synchronize(selected)
                torch.cuda.reset_peak_memory_stats(selected)
                started = time.perf_counter()
                predicted = np.asarray(model.infer_image(sample["bgr"], 518), dtype=np.float32)
                torch.cuda.synchronize(selected)
                elapsed_ms = (time.perf_counter() - started) * 1000
                if predicted.shape != native.shape:
                    raise ValueError("Metric model did not return source-resolution depth")
                inference_ms.append(elapsed_ms)
                inference_peaks.append(int(torch.cuda.max_memory_allocated(selected)))
                target = prediction_dir / f"{row['sample_index']:04d}.npy"
                with target.open("xb") as stream:
                    np.save(stream, predicted, allow_pickle=False)
                native_valid = np.isfinite(native) & (native > .08) & (native < 12)
                compared = native_valid & np.isfinite(predicted)
                errors = np.abs(predicted[compared].astype(np.float64) - native[compared])
                error_samples.append(errors)
                frame = {"sample_index": int(row["sample_index"]), "time_s": float(row["time_s"]),
                         "rgb_sha256": sample["rgb_sha256"], "native_depth_sha256": sample["native_sha256"],
                         "predicted_path": str(target), "predicted_sha256": _sha(target),
                         "camera": asdict(camera), "rgb_decode_ms_excluded": sample["rgb_decode_ms"],
                         "native_npy_decode_ms_excluded": sample["native_npy_decode_ms"],
                         "inference_ms_including_cpu_return": elapsed_ms,
                         "native_valid_fraction": float(native_valid.mean()),
                         "depth_reference_diagnostic": {"native_valid_pixels": int(native_valid.sum()),
                             "compared_pixels": int(compared.sum()),
                             "nonfinite_prediction_on_valid_native": int((native_valid & ~np.isfinite(predicted)).sum()),
                             "absolute_error_m": _distribution(errors)},
                         "native": {}, "rgb": {}, "agreement_with_same_arm_native_reference": {}}
                for arm in ARMS:
                    for branch, depth in (("native", native), ("rgb", predicted)):
                        result, ms = _postprocess(depth, camera, encoders[camera], arm, representation_device)
                        post_ms[branch][arm].append(ms)
                        result["postprocess_ms"] = ms
                        frame[branch][arm] = result
                    agreement = _agreement(frame["rgb"][arm]["alerts"], frame["native"][arm]["alerts"])
                    frame["agreement_with_same_arm_native_reference"][arm] = agreement
                    for key, count in agreement.items():
                        totals[arm][key] += count
                report_rows.append(frame)
                preview = (sample["bgr"], native, predicted, row["sample_index"])
                if first_preview is None:
                    first_preview = preview
                last_preview = preview
            measured_allocation_end = int(torch.cuda.memory_allocated(selected))
    finally:
        encoders.clear()
        del model
        gc.collect()
    aggregate_errors = np.concatenate(error_samples) if error_samples else np.empty(0)
    if _sha(weights) != weight_hash:
        raise RuntimeError("Weights changed during replay; retained predictions are not an identity-stable result")
    timing = {"units": "ms", "timed_frames": 11, "first_frame_warmups": 1,
              "inference_including_preprocess_resize_and_cpu_return": _distribution(inference_ms),
              "rgb_decode_excluded": _distribution([sample["rgb_decode_ms"] for sample in decoded]),
              "native_npy_decode_excluded": _distribution([sample["native_npy_decode_ms"] for sample in decoded]),
              "per_arm": {arm: {"rgb_postprocess": _distribution(post_ms["rgb"][arm]),
                  "native_postprocess": _distribution(post_ms["native"][arm]),
                  "rgb_inference_plus_postprocess": _distribution(np.asarray(inference_ms) + post_ms["rgb"][arm])}
                  for arm in ARMS},
              "excluded": ["file reading", "hashing", "image/NPY decode", "model loading", "encoder initialization",
                           "warmup", "artifact writing", "agreement scoring", "visualization"],
              "execution": "One shared inference per frame; each arm total pairs that inference with its own postprocess."}
    result = {"schema": "nearfield-rgb-native-reference-replay-v1", "frames": 11, "regions_per_frame": 9,
              "representation_device": str(representation_device),
              "source": {"model_dir": str(model_dir), "manifest_sha256": _sha(manifest_path),
                         "schema": manifest["schema_version"], "episode_id": episodes[0]["episode_id"]},
              "model": {"name": "Depth Anything V2 metric Hypersim vits", "source": source_identity,
                        "weights": str(weights), "weights_sha256": weight_hash, "weight_bytes": weights.stat().st_size,
                        "input_size": 518, "max_depth_m": 20.0, "actual_parameter_device": actual_device,
                        "backend_observation": observation, "fit_or_native_depth_calibration": "NONE"},
              "gpu_memory_bytes": {"allocated_before_model": allocation_before_model,
                  "model_resident_allocation_delta": model_resident_delta,
                  "max_allocated_during_inference": max(inference_peaks),
                  "allocated_at_measurement_end": measured_allocation_end,
                  "scope": "PyTorch process allocations; inference peak includes resident model, encoder and preexisting tensors."},
              "agreement_with_same_arm_native_reference": totals,
              "depth_reference_diagnostic": {"absolute_error_m": _distribution(aggregate_errors),
                  "native_valid_fraction_per_frame": _distribution([row["native_valid_fraction"] for row in report_rows]),
                  "mask": "finite native forward depth 0.08<d<12 m; finite predicted values only; no clipping or scale fitting",
                  "compared_pixels": int(aggregate_errors.size)},
              "timing": timing, "rows": report_rows,
              "preview": _preview(first_preview, last_preview, output),
              "limitations": ["All 11 frames are consumed synthetic Development; no cherry-picking or independent natural scenes.",
                  "Simulator intrinsics, camera height and pitch are privileged inputs in both native and RGB branches.",
                  "TP/FP/FN compare each RGB arm with the same arm's native decisions, not independent obstacle truth.",
                  "Native depth error and decision agreement are reference diagnostics, not obstacle accuracy.",
                  "No depth fitting, scale calibration, training, temporal inference, labels, actors, plans or evaluator files.",
                  "Desktop GPU timings and allocation are not phone cost, real-time deployment, category generalization or safety evidence."]}
    result = _finite(result)
    json.dumps(result, allow_nan=False)
    return result
