#!/usr/bin/env python3
"""Materialize D2 TRAIN inputs, run the frozen HTP context, and train one fixed head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    TruthReaderPolicy,
    derive_assistive_truth,
    load_manifest_frame,
    parse_trajectory,
)
from scripts.research.hftf.deployment.depthart.export_depthart_camera_external import (  # noqa: E402
    install_timm_compat,
)


PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D2_TRAIN_ONLY_BASE_OUTPUT_AND_HEAD_TRAINING"
INPUT_NAMES = ("image", "camera_prompt_4", "camera_prompt_8", "camera_prompt_16", "camera_prompt_32")
BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
FEATURE_ORDER = (
    "candidate_clearance_present",
    "candidate_clearance_m_or_zero",
    "valid_depth_fraction",
    "ground_support_fraction",
    "ground_residual_m_or_zero",
    "log1p_band_support_points",
    "log1p_intrusion_points",
    "observed_forward_m",
    "band_left",
    "band_center",
    "band_right",
)
EXPECTED_DEPTH_BYTES = 608 * 448 * 4
TASK_HORIZON_M = 2.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_raw(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    array = np.ascontiguousarray(value, dtype=np.float32)
    with temporary.open("xb") as handle:
        array.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
            "shape": list(array.shape), "dtype": "float32"}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout or ''}")
    return result


def adb(serial: str, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["adb", "-s", serial, *arguments], check=check)


def chunk_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    chunk_size = int(protocol["execution"]["chunk_size_frames"])
    require(chunk_size == 50 and 300 % chunk_size == 0, "chunk size drift")
    return [
        {"chunk_index": session_index * (300 // chunk_size) + start // chunk_size,
         "session_index": session_index, "visit_id": identity["visit_id"],
         "video_id": identity["video_id"], "frame_start": start, "frame_stop": start + chunk_size}
        for session_index, identity in enumerate(protocol["train_scope"])
        for start in range(0, 300, chunk_size)
    ]


def clearance_payload(band: dict[str, Any] | None) -> dict[str, Any]:
    band = band or {}
    value = band.get("clearance_m")
    occupied = band.get("occupied_by_horizon", {})
    if value is not None and np.isfinite(value):
        return {"valid": True, "metres": min(float(value), TASK_HORIZON_M)}
    if all(occupied.get(str(horizon)) is False for horizon in HORIZONS):
        return {"valid": True, "metres": TASK_HORIZON_M}
    return {"valid": False, "metres": None}


def compact_truth(truth: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in BANDS:
        band = truth.get("bands", {}).get(name)
        rows.append({
            "band": name,
            "clearance": clearance_payload(band),
            "occupied_by_horizon": [
                (band or {}).get("occupied_by_horizon", {}).get(str(horizon)) for horizon in HORIZONS
            ],
        })
    return rows


def candidate_features(geometry: dict[str, Any], band_name: str) -> tuple[list[float], dict[str, Any]]:
    band = geometry.get("bands", {}).get(band_name) or {}
    plane = geometry.get("ground_plane") or {}
    clearance = clearance_payload(band)
    one_hot = [1.0 if band_name == value else 0.0 for value in BANDS]
    features = [
        1.0 if clearance["valid"] else 0.0,
        float(clearance["metres"] or 0.0),
        float(geometry.get("valid_depth_fraction", 0.0)),
        float(plane.get("support_fraction", 0.0)),
        float(plane.get("median_residual_m", 0.0)),
        math.log1p(int(band.get("support_points", 0))),
        math.log1p(int(band.get("intrusion_points", 0))),
        float(band.get("observed_forward_m", 0.0)),
        *one_hot,
    ]
    require(len(features) == len(FEATURE_ORDER) and all(math.isfinite(value) for value in features),
            "non-finite candidate feature")
    evidence = {
        "ground_plane_available": geometry.get("ground_plane") is not None,
        "valid_depth_fraction": features[2],
        "ground_support_fraction": features[3],
        "band_support_points": int(band.get("support_points", 0)),
    }
    return features, evidence


class TaskHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(11, 16), nn.SiLU(), nn.Linear(16, 5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def train_head(dataset: dict[str, np.ndarray], *, steps: int = 500, seed: int = 17) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    features = torch.as_tensor(dataset["features"], dtype=torch.float64)
    known = torch.as_tensor(dataset["known"], dtype=torch.float64)
    occupied = torch.as_tensor(dataset["occupied"], dtype=torch.float64)
    raw_clearance = torch.as_tensor(dataset["raw_clearance"], dtype=torch.float64)
    truth_clearance = torch.as_tensor(dataset["truth_clearance"], dtype=torch.float64)
    paired = torch.as_tensor(dataset["clearance_paired"], dtype=torch.bool)
    require(features.ndim == 2 and features.shape[1] == 11, "feature matrix shape drift")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False)
    standardized = (features - mean) / (std + 1e-6)
    model = TaskHead().to(dtype=torch.float64, device="cpu")
    require(sum(parameter.numel() for parameter in model.parameters()) == 277, "head parameter count drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    known_count = float(known.sum().item())
    cell_count = float(known.numel())
    require(0 < known_count < cell_count, "knownness classes must both be present")
    occupied_count = float(occupied[known.bool()].sum().item())
    require(0 < occupied_count < known_count, "occupancy classes must both be present")
    occupied_weight = known_count / (2.0 * occupied_count)
    clear_weight = known_count / (2.0 * (known_count - occupied_count))

    def losses() -> tuple[torch.Tensor, dict[str, float]]:
        output = model(standardized)
        known_logits = output[:, 1:2].expand(-1, 3)
        known_loss = torch.nn.functional.binary_cross_entropy_with_logits(known_logits, known)
        occupancy_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            output[:, 2:5], occupied, reduction="none"
        )
        occupancy_weights = torch.where(occupied.bool(), occupied_weight, clear_weight)
        occupancy_loss = (occupancy_loss_raw * occupancy_weights)[known.bool()].mean()
        predicted_clearance = torch.clamp(raw_clearance + 0.5 * torch.tanh(output[:, 0]), min=0.0)
        require(bool(paired.any()), "paired clearance denominator is empty")
        clearance_loss = torch.nn.functional.huber_loss(
            predicted_clearance[paired], truth_clearance[paired], reduction="mean", delta=0.2
        )
        total = known_loss + occupancy_loss + clearance_loss
        return total, {"total": float(total.detach()), "knownness": float(known_loss.detach()),
                       "occupancy": float(occupancy_loss.detach()), "clearance": float(clearance_loss.detach())}

    _, initial = losses()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total, _ = losses()
        total.backward()
        optimizer.step()
    _, final = losses()
    state = {
        name: tensor.detach().cpu().numpy().tolist() for name, tensor in model.state_dict().items()
    }
    checkpoint = {
        "schema": "blindassist_depthart_task_preserving_d2_task_head_checkpoint_v1",
        "architecture": "Linear(11,16)-SiLU-Linear(16,5)", "parameter_count": 277,
        "dtype": "float64", "device": "cpu", "feature_order": list(FEATURE_ORDER),
        "feature_mean": mean.numpy().tolist(), "feature_std_population": std.numpy().tolist(),
        "standardization_epsilon": 1e-6, "seed": seed, "step": steps, "state_dict": state,
    }
    stats = {"initial_loss": initial, "final_loss": final, "known_cell_count": int(known_count),
             "total_cell_count": int(cell_count), "occupied_cell_count": int(occupied_count),
             "clear_cell_count": int(known_count - occupied_count),
             "paired_clearance_count": int(paired.sum().item()),
             "occupied_class_weight": occupied_weight, "clear_class_weight": clear_weight}
    return checkpoint, stats


def verify_file_binding(binding: dict[str, Any]) -> Path:
    path = Path(binding["path"]).resolve()
    require(path.is_file() and path.stat().st_size == int(binding["bytes"]), f"binding size drift: {path}")
    require(sha256(path) == binding["sha256"], f"binding SHA drift: {path}")
    return path


def load_train_videos(manifest: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    roles = [row for row in manifest["roles"] if row["role"] == "D2_TRAIN"]
    require([(row["visit_id"], row["video_id"]) for row in roles]
            == [(row["visit_id"], row["video_id"]) for row in protocol["train_scope"]], "TRAIN roster drift")
    videos = []
    for role in roles:
        checkpoint_path = Path(role["checkpoint_path"])
        require(sha256(checkpoint_path) == role["checkpoint_sha256"], "TRAIN checkpoint SHA drift")
        video = load_json(checkpoint_path)
        require(video["role"] == "D2_TRAIN" and video["visit_id"] == role["visit_id"]
                and video["video_id"] == role["video_id"], "TRAIN checkpoint role drift")
        require(len(video["selected_frame_stems"]) == 300, "TRAIN frame count drift")
        for entries in video["extracted"].values():
            require(len(entries) == 300, "TRAIN modality count drift")
            require(all("source\\train\\" in str(entry["path"]).lower() for entry in entries),
                    "non-TRAIN source path crossed execution scope")
        videos.append(video)
    return videos


def materialize_chunk(chunk: dict[str, Any], video: dict[str, Any], chunk_root: Path,
                      model: nn.Module, preprocess: Any) -> dict[str, Any]:
    receipt_path = chunk_root / "materialization-receipt.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt["chunk"] == chunk, "materialization schedule drift")
        return receipt
    require(not chunk_root.exists(), f"partial chunk without receipt: {chunk_root}")
    chunk_root.mkdir(parents=True)
    trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
    records, input_lines = [], []
    for frame_index in range(int(chunk["frame_start"]), int(chunk["frame_stop"])):
        frame = load_manifest_frame(video, frame_index, trajectory, TruthReaderPolicy())
        require(int(frame["orientation"]["rotation_index"]) in (1, 3), "non-portrait TRAIN frame")
        bgr = cv2.cvtColor(np.asarray(frame["rgb_upright"], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image, intrinsics = preprocess(bgr, np.asarray(frame["intrinsics_upright"], dtype=np.float32), 448, 448)
        require(tuple(image.shape) == (1, 3, 608, 448), "input tensor shape drift")
        image, intrinsics = image.cuda(), intrinsics.cuda()
        with torch.inference_mode():
            cameras = model.cam_embedder(intrinsics, 608, 448, "cuda")
        stem = frame["identity"]["frame_stem"]
        frame_root = chunk_root / "inputs" / stem
        arrays = {"image": image.detach().cpu().numpy(),
                  **{name: value.detach().cpu().numpy()
                     for name, value in zip(INPUT_NAMES[1:], cameras, strict=True)}}
        input_receipts = {name: atomic_raw(frame_root / f"{name}.raw", arrays[name]) for name in INPUT_NAMES}
        records.append({
            "parent_id": video["visit_id"], "session_id": video["video_id"],
            "frame_id": stem, "frame_index": frame_index,
            "timestamp_ns": int(Decimal(stem.rsplit("_", 1)[1]) * 1_000_000_000),
            "orientation_index": int(frame["orientation"]["rotation_index"]),
            "up_camera": frame["orientation"]["up_camera"],
            "intrinsics_tensor": intrinsics.detach().cpu().numpy()[0].tolist(),
            "truth_bands": compact_truth(frame["truth"]), "candidate_inputs": input_receipts,
        })
        input_lines.append(" ".join(f"{name}:=inputs/{stem}/{name}.raw" for name in INPUT_NAMES))
    input_list = chunk_root / "input-list.txt"
    input_list.write_text("\n".join(input_lines) + "\n", encoding="utf-8", newline="\n")
    receipt = {"schema": "blindassist_depthart_task_preserving_d2_train_chunk_materialization_v1",
               "chunk": chunk, "records": records,
               "input_list": {"path": str(input_list.resolve()), "bytes": input_list.stat().st_size,
                              "sha256": sha256(input_list)},
               "train_truth_accessed": True, "reference_model_output_accessed": False,
               "development_accessed": False, "r2_accessed": False}
    atomic_json(receipt_path, receipt)
    return receipt


def verify_device(protocol: dict[str, Any]) -> dict[str, str]:
    serial = protocol["device"]["serial"]
    require(adb(serial, "get-state").stdout.strip() == "device", "device is not ready")
    properties = {
        "build_fingerprint": adb(serial, "shell", "getprop", "ro.build.fingerprint").stdout.strip(),
        "model": adb(serial, "shell", "getprop", "ro.product.model").stdout.strip(),
        "device": adb(serial, "shell", "getprop", "ro.product.device").stdout.strip(),
        "soc": adb(serial, "shell", "getprop", "ro.soc.model").stdout.strip(),
        "abi": adb(serial, "shell", "getprop", "ro.product.cpu.abi").stdout.strip(),
    }
    for key, value in properties.items():
        require(value == str(protocol["device"][key]), f"device {key} drift: {value}")
    source_remote = protocol["device"]["validated_preflight_workspace"]
    for asset in protocol["device"]["remote_assets"]:
        output = adb(serial, "shell", "sha256sum", f"{source_remote}/{asset['path']}").stdout.strip().split()
        require(output and output[0].upper() == asset["sha256"], f"remote asset drift: {asset['path']}")
    return properties


def run_device_chunk(protocol: dict[str, Any], chunk: dict[str, Any], chunk_root: Path,
                     materialization: dict[str, Any]) -> dict[str, Any]:
    receipt_path = chunk_root / "device-run-receipt.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt["chunk"] == chunk, "device receipt schedule drift")
        for output in receipt["outputs"]:
            path = Path(output["path"])
            require(path.is_file() and path.stat().st_size == output["bytes"] and sha256(path) == output["sha256"],
                    f"completed output drift: {path}")
        return receipt
    properties = verify_device(protocol)
    serial = protocol["device"]["serial"]
    remote_base = protocol["execution"]["remote_root"]
    require(remote_base.startswith("/data/local/tmp/depthart_d2_train_") and ".." not in remote_base,
            "unsafe remote root")
    remote_root = f"{remote_base}/chunk-{chunk['chunk_index']:02d}"
    adb(serial, "shell", "rm", "-rf", remote_root)
    adb(serial, "shell", "mkdir", "-p", remote_root)
    adb(serial, "push", str(chunk_root / "inputs"), f"{remote_root}/inputs")
    adb(serial, "push", str(chunk_root / "input-list.txt"), f"{remote_root}/input-list.txt")
    expected = {
        record["candidate_inputs"][name]["path"].replace(str(chunk_root) + os.sep, "").replace(os.sep, "/"):
        record["candidate_inputs"][name]["sha256"]
        for record in materialization["records"] for name in INPUT_NAMES
    }
    expected["input-list.txt"] = materialization["input_list"]["sha256"]
    output = adb(serial, "shell",
                 f"cd {remote_root} && find inputs -type f -name '*.raw' -print0 | sort -z | "
                 "xargs -0 sha256sum && sha256sum input-list.txt").stdout
    observed = {}
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            observed[parts[1].lstrip("*./")] = parts[0].upper()
    require(observed == expected, "device input SHA set differs from host receipt")
    source = protocol["device"]["validated_preflight_workspace"]
    provider = "DepthArtSelectiveScanPackageInterfaceProvider"
    command = (
        f"cd {remote_root} && export LD_LIBRARY_PATH={source}/arm64 && "
        f"export ADSP_LIBRARY_PATH='{source}/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && "
        f"{source}/arm64/qnn-net-run --backend {source}/arm64/libQnnHtp.so "
        f"--retrieve_context {source}/context/depthart-d1-608x448-sm8650-v75.bin "
        f"--input_list input-list.txt --output_dir output "
        f"--op_packages libQnnDepthArtSelectiveScanPackage.so:{provider}:CPU,"
        f"libQnnDepthArtSelectiveScanPackage.so:{provider}:HTP --log_level info"
    )
    result = adb(serial, "shell", command, check=False)
    log_path = chunk_root / "device-run.log"
    log_path.write_text(result.stdout or "", encoding="utf-8", newline="\n")
    require(result.returncode == 0, f"qnn-net-run failed: {result.returncode}\n{result.stdout}")
    local_output = chunk_root / "candidate-output"
    require(not local_output.exists(), f"candidate output exists without receipt: {local_output}")
    adb(serial, "pull", f"{remote_root}/output", str(local_output))
    outputs = []
    for index, record in enumerate(materialization["records"]):
        candidates = sorted((local_output / f"Result_{index}").glob("*.raw"))
        require(len(candidates) == 1, f"expected one output for Result_{index}")
        path = candidates[0]
        values = np.fromfile(path, dtype=np.float32)
        require(path.stat().st_size == EXPECTED_DEPTH_BYTES and values.size == 608 * 448
                and np.all(np.isfinite(values)), f"candidate output invalid: {path}")
        outputs.append({"frame_id": record["frame_id"], "result_index": index,
                        "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
                        "shape": [1, 608, 448], "dtype": "float32"})
    receipt = {"schema": "blindassist_depthart_task_preserving_d2_train_device_chunk_v1",
               "chunk": chunk, "device": properties, "remote_root": remote_root,
               "remote_input_sha256_verified": True, "command": command, "exit_code": result.returncode,
               "log": {"path": str(log_path.resolve()), "bytes": log_path.stat().st_size,
                       "sha256": sha256(log_path)}, "outputs": outputs,
               "candidate_base_output_accessed": True, "performance_measured": False,
               "development_accessed": False, "r2_accessed": False}
    atomic_json(receipt_path, receipt)
    adb(serial, "shell", "rm", "-rf", remote_root)
    shutil.rmtree(chunk_root / "inputs")
    atomic_json(chunk_root / "input-cleanup-receipt.json",
                {"generated_inputs_removed_after_verified_device_pass": True,
                 "materialization_receipt_sha256": sha256(chunk_root / "materialization-receipt.json"),
                 "device_receipt_sha256": sha256(receipt_path)})
    return receipt


def build_dataset(protocol: dict[str, Any], output_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    features, known, occupied = [], [], []
    raw_clearance, truth_clearance, paired = [], [], []
    chunks = []
    for chunk in chunk_schedule(protocol):
        chunk_root = output_root / f"chunk-{chunk['chunk_index']:02d}"
        materialization_path = chunk_root / "materialization-receipt.json"
        device_path = chunk_root / "device-run-receipt.json"
        materialization, device = load_json(materialization_path), load_json(device_path)
        require(materialization["chunk"] == device["chunk"] == chunk, "chunk schedule drift")
        require(len(materialization["records"]) == len(device["outputs"]) == 50, "chunk count drift")
        for record, output in zip(materialization["records"], device["outputs"], strict=True):
            require(record["frame_id"] == output["frame_id"], "frame/output mapping drift")
            path = Path(output["path"])
            require(path.stat().st_size == output["bytes"] and sha256(path) == output["sha256"],
                    "candidate output SHA drift")
            depth = np.fromfile(path, dtype=np.float32).reshape(608, 448)
            geometry = derive_assistive_truth(
                depth, np.full(depth.shape, 2, dtype=np.uint8),
                np.asarray(record["intrinsics_tensor"], dtype=np.float64),
                np.asarray(record["up_camera"], dtype=np.float64), TruthReaderPolicy(),
            )
            for band_index, band_name in enumerate(BANDS):
                row_features, _ = candidate_features(geometry, band_name)
                truth = record["truth_bands"][band_index]
                truth_states = truth["occupied_by_horizon"]
                candidate_clear = clearance_payload(geometry.get("bands", {}).get(band_name))
                truth_clear = truth["clearance"]
                features.append(row_features)
                known.append([1.0 if value is not None else 0.0 for value in truth_states])
                occupied.append([1.0 if value is True else 0.0 for value in truth_states])
                raw_clearance.append(float(candidate_clear["metres"] or 0.0))
                truth_clearance.append(float(truth_clear["metres"] or 0.0))
                paired.append(bool(candidate_clear["valid"] and truth_clear["valid"]))
        chunks.append({"chunk_index": chunk["chunk_index"], "materialization_sha256": sha256(materialization_path),
                       "device_run_sha256": sha256(device_path)})
    arrays = {
        "features": np.asarray(features, dtype=np.float64), "known": np.asarray(known, dtype=np.float64),
        "occupied": np.asarray(occupied, dtype=np.float64),
        "raw_clearance": np.asarray(raw_clearance, dtype=np.float64),
        "truth_clearance": np.asarray(truth_clearance, dtype=np.float64),
        "clearance_paired": np.asarray(paired, dtype=bool),
    }
    require(arrays["features"].shape == (3600, 11) and arrays["known"].shape == (3600, 3),
            "final TRAIN dataset shape drift")
    return arrays, {"chunk_receipts": chunks, "band_rows": 3600, "cell_labels": 10800}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    protocol_path, activation_path = args.protocol.resolve(), args.activation_receipt.resolve()
    protocol, activation = load_json(protocol_path), load_json(activation_path)
    require(protocol["protocol_id"] == PROTOCOL_ID and protocol["status"] == "FROZEN_TRAIN_ONLY_EXECUTION",
            "execution protocol drift")
    require(activation["status"] == "TRAIN_ONLY_EXECUTION_ACTIVATED" and activation["execution_authorized"] is True,
            "TRAIN-only execution is not activated")
    require(activation["protocol_sha256"] == sha256(protocol_path), "activation/protocol drift")
    require(protocol["bindings"]["runner"]["sha256"] == sha256(Path(__file__)), "runner SHA drift")
    for binding in protocol["bindings"].values():
        if isinstance(binding, dict) and {"path", "bytes", "sha256"}.issubset(binding):
            verify_file_binding(binding)
    require(sha256(args.checkpoint.resolve()) == protocol["reference"]["checkpoint_sha256"], "checkpoint SHA drift")
    frozen_policy = json.loads(json.dumps(asdict(TruthReaderPolicy())))
    require(protocol["truth_reader_policy"] == frozen_policy, "truth reader policy drift")
    source_manifest = load_json(Path(protocol["bindings"]["source_manifest"]["path"]))
    videos = load_train_videos(source_manifest, protocol)
    output_root = args.output_root.resolve()
    attempt_path = output_root / "attempt.json"
    attempt = {"schema": "blindassist_depthart_task_preserving_d2_train_attempt_v1",
               "protocol_sha256": sha256(protocol_path), "activation_receipt_sha256": sha256(activation_path),
               "runner_sha256": sha256(Path(__file__)), "train_identity_count": 4, "frame_count": 1200,
               "development_accessed": False, "r2_accessed": False}
    if attempt_path.exists():
        require(load_json(attempt_path) == attempt, "attempt binding drift")
    else:
        require(not output_root.exists(), f"output root exists without attempt: {output_root}")
        output_root.mkdir(parents=True)
        atomic_json(attempt_path, attempt)

    chunks = chunk_schedule(protocol)
    pending_materialization = any(not (output_root / f"chunk-{c['chunk_index']:02d}" / "device-run-receipt.json").exists()
                                  for c in chunks)
    model = preprocess = None
    if pending_materialization:
        install_timm_compat()
        source = args.source_root.resolve()
        require(run(["git", "-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
                == protocol["reference"]["source_git_commit"], "DepthART source commit drift")
        require(not run(["git", "-C", str(source), "status", "--short"]).stdout.strip(),
                "DepthART source tree is dirty")
        runtime = protocol["reference"]["host_runtime"]
        require(torch.__version__ == runtime["torch"] and torch.version.cuda == runtime["cuda"]
                and cv2.__version__ == runtime["opencv"] and np.__version__ == runtime["numpy"],
                "host runtime drift")
        sys.path.insert(0, str(source / "metric"))
        sys.path.insert(0, str(source / "deploy" / "shared"))
        sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
        from common import preprocess as official_preprocess  # type: ignore
        from depthart_selective_scan import install_depthart  # type: ignore
        from model import load_model  # type: ignore
        from network import tvimblock  # type: ignore
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        model = load_model(args.checkpoint.resolve(), "S", "indoor", "cuda").eval()
        install_depthart(tvimblock)
        preprocess = official_preprocess

    for chunk in chunks:
        chunk_root = output_root / f"chunk-{chunk['chunk_index']:02d}"
        if (chunk_root / "device-run-receipt.json").exists():
            device = run_device_chunk(protocol, chunk, chunk_root, load_json(chunk_root / "materialization-receipt.json"))
            print(json.dumps({"completed": chunk["chunk_index"] + 1, "total": len(chunks),
                              "status": "RESUMED_VALID", "outputs": len(device["outputs"])}), flush=True)
            continue
        require(model is not None and preprocess is not None, "materializer runtime unavailable")
        materialization = materialize_chunk(chunk, videos[int(chunk["session_index"])], chunk_root, model, preprocess)
        device = run_device_chunk(protocol, chunk, chunk_root, materialization)
        print(json.dumps({"completed": chunk["chunk_index"] + 1, "total": len(chunks),
                          "visit_id": chunk["visit_id"], "video_id": chunk["video_id"],
                          "status": "DEVICE_PASS", "outputs": len(device["outputs"])}), flush=True)

    dataset, dataset_meta = build_dataset(protocol, output_root)
    dataset_path = output_root / "train-dataset.npz"
    require(not dataset_path.exists(), f"dataset output exists: {dataset_path}")
    np.savez_compressed(dataset_path, **dataset)
    checkpoint, training_stats = train_head(dataset, steps=500, seed=17)
    checkpoint_path = output_root / "task-head-step-500.json"
    atomic_json(checkpoint_path, checkpoint)
    result = {
        "schema": "blindassist_depthart_task_preserving_d2_train_only_result_v1",
        "status": "PASS", "terminal": "D2_TRAIN_ONLY_BASE_OUTPUT_AND_HEAD_TRAINING_PASS_HEAD_LOCKED",
        "protocol_sha256": sha256(protocol_path), "activation_receipt_sha256": sha256(activation_path),
        "attempt_sha256": sha256(attempt_path), "dataset": {"path": str(dataset_path.resolve()),
            "bytes": dataset_path.stat().st_size, "sha256": sha256(dataset_path), **dataset_meta},
        "checkpoint": {"path": str(checkpoint_path.resolve()), "bytes": checkpoint_path.stat().st_size,
                       "sha256": sha256(checkpoint_path), "step": 500, "parameter_count": 277},
        "training": training_stats, "reference_model_output_accessed": False,
        "development_accessed": False, "r2_accessed": False, "performance_measured": False,
        "next_gate": "EXPLICIT_D2_DEVELOPMENT_BASELINE_AND_FROZEN_HEAD_QUALITY_ACTIVATION",
    }
    atomic_json(output_root / "train-result.json", result)
    print(json.dumps({"status": result["status"], "terminal": result["terminal"],
                      "checkpoint_sha256": result["checkpoint"]["sha256"],
                      "result_sha256": sha256(output_root / "train-result.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
