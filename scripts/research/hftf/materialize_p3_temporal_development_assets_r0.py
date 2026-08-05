#!/usr/bin/env python3
"""Fail-closed materialization of development-only P3 temporal assets.

The A2 model import is deliberately delayed until every static input binding,
identity join and output-absence check has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

REQUEST_SCHEMA = "blindassist_p3_temporal_development_assets_r0_request"
PROTOCOL_SCHEMA = "blindassist_p3_temporal_development_screen_r0_protocol"
IDENTITY_SCHEMA = "blindassist_p3_r0_2_role_manifest"
TEACHER_SCHEMA = "blindassist_dav2_distillation_teacher_r0"
SPATIAL_SCHEMA = "blindassist_spatial_calibration_head_r1_development_cache"
MANIFEST_SCHEMA = "blindassist_p3_temporal_development_complete_manifest_r0"
CACHE_SCHEMA = "blindassist_p3_temporal_development_frozen_a2_disagreement_r0"
WEIGHTS_SCHEMA = "blindassist_p3_temporal_development_class_weights_r0"
RECEIPT_SCHEMA = "blindassist_p3_temporal_development_asset_materialization_receipt_r0"
STATES = ("CLEAR", "OCCUPIED", "UNKNOWN_GROUND")
TRANSITIONS = tuple(f"{left}_TO_{right}" for left in STATES for right in STATES)


def _bytes(value: Any, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _inside(root: Path, relative: str) -> Path:
    lexical_root = Path(os.path.abspath(root))
    target = Path(os.path.abspath(lexical_root / relative))
    _require(target.is_relative_to(lexical_root), f"path leaves repository: {relative}")
    return target


def _bind(root: Path, value: dict[str, str], label: str) -> Path:
    _require(set(value) == {"path", "sha256"}, f"{label} binding fields drift")
    path = _inside(root, value["path"])
    _require(path.is_file(), f"{label} missing")
    _require(_sha_file(path) == value["sha256"].upper(), f"{label} SHA mismatch")
    return path


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def _state(value: float, valid: bool) -> str:
    return "UNKNOWN_GROUND" if not valid else ("OCCUPIED" if value <= 1.5 else "CLEAR")


def _arkit_media_index(media: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map exact frame IDs to SHA-bound ARKit depth and intrinsics assets."""
    videos = media.get("videos")
    _require(isinstance(videos, list), "ARKit media videos missing")
    result: dict[str, dict[str, Any]] = {}
    for video in videos:
        extracted = video.get("extracted", {})
        depth = {Path(item["path"]).stem: item for item in extracted.get("lowres_depth", [])}
        intrinsics = {Path(item["path"]).stem: item for item in extracted.get("lowres_wide_intrinsics", [])}
        for frame_id in depth.keys() & intrinsics.keys():
            _require(frame_id not in result, "duplicate ARKit frame media")
            result[frame_id] = {"depth": depth[frame_id], "intrinsics": intrinsics[frame_id]}
    return result


def _intrinsics(path: Path) -> list[float]:
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    _require(len(values) == 6 and all(math.isfinite(value) for value in values), "ARKit pincam must contain six finite values")
    return values[-4:]


def _default_infer_factory(root: Path, request: dict[str, Any]) -> Callable[[dict[str, Any]], float]:
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional

    dpt_root = _inside(root, request["dav2_repo_path"])
    sys.path.insert(0, str(dpt_root / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2
    checkpoint = _bind(root, request["a2_checkpoint"], "A2 checkpoint")
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(request["device"]).eval()
    teacher_path = _bind(root, request["teacher_depth"], "teacher depth")
    teacher = np.load(teacher_path, mmap_mode="r")

    def infer(row: dict[str, Any]) -> float:
        rgb = _inside(root, row["rgb_path"])
        _require(rgb.is_file() and _sha_file(rgb) == row["rgb_sha256"], "RGB SHA mismatch")
        image = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
        _require(image is not None, "RGB decode failed")
        tensor, _ = model.image2tensor(image, 392)
        with torch.inference_mode():
            pred = model(tensor.to(request["device"]))
        truth = teacher[row["teacher_depth_index"]]
        pred = functional.interpolate(pred[:, None], size=truth.shape, mode="bilinear", align_corners=True)[0, 0].float().cpu().numpy()
        valid = np.isfinite(truth) & (truth > .1) & (truth <= 20) & np.isfinite(pred) & (pred > .1)
        _require(bool(valid.any()), "no paired depth pixels")
        return float(np.mean(np.abs(np.log(np.clip(pred[valid], .1, 20)) - np.log(np.clip(truth[valid], .1, 20)))))
    return infer


def build(root: Path, request: dict[str, Any], source_path: Path, *, infer_factory=_default_infer_factory) -> None:
    required = {"schema", "protocol", "train_identity", "validation_identity", "teacher_manifest", "teacher_depth", "spatial_manifest", "spatial_arrays", "arkit_media_manifest", "a2_checkpoint", "a2_training_receipt", "dav2_repo_path", "dav2_dpt_source", "device", "producer_sha256", "outputs"}
    _require(set(request) == required and request["schema"] == REQUEST_SCHEMA, "request schema drift")
    _require(_sha_file(source_path) == request["producer_sha256"].upper(), "producer SHA mismatch")
    bindings = {name: _bind(root, request[name], name.replace("_", " ")) for name in ("protocol", "train_identity", "validation_identity", "teacher_manifest", "teacher_depth", "spatial_manifest", "spatial_arrays", "arkit_media_manifest", "a2_checkpoint", "a2_training_receipt", "dav2_dpt_source")}
    protocol = json.loads(bindings["protocol"].read_text())
    _require(protocol.get("schema") == PROTOCOL_SCHEMA and protocol.get("status") == "FROZEN_BEFORE_MODEL_LOAD_OR_TRAINING" and protocol.get("claim_ceiling") == "DEVELOPMENT_SIGNAL_ONLY", "protocol boundary drift")
    _require("sealing-attempt-02" not in json.dumps(request, sort_keys=True), "sealed Bonn input forbidden")
    a2 = json.loads(bindings["a2_training_receipt"].read_text())
    _require(a2.get("schema") == "blindassist_dav2_392_distillation_a2_r0_training_result", "A2 receipt schema drift")
    _require(a2.get("checkpoint", {}).get("sha256") == request["a2_checkpoint"]["sha256"].upper(), "A2 selected checkpoint mismatch")
    teacher = json.loads(bindings["teacher_manifest"].read_text())
    _require(teacher.get("schema") == TEACHER_SCHEMA and teacher.get("teacher_depth", {}).get("sha256") == request["teacher_depth"]["sha256"].upper(), "teacher binding drift")
    spatial = json.loads(bindings["spatial_manifest"].read_text())
    _require(spatial.get("schema") == SPATIAL_SCHEMA and spatial.get("arrays", {}).get("sha256") == request["spatial_arrays"]["sha256"].upper(), "spatial binding drift")
    teacher_by_id = {r["frame_id"]: r for r in teacher["records"]}
    spatial_by_id = {r["frame_stem"]: (i, r) for i, r in enumerate(spatial["records"])}
    arkit_by_id = _arkit_media_index(json.loads(bindings["arkit_media_manifest"].read_text()))
    identities = []
    for role, key, expected_parents in (("train", "train_identity", 13), ("validation", "validation_identity", 3)):
        identity = json.loads(bindings[key].read_text())
        _require(identity.get("schema") == IDENTITY_SCHEMA and identity.get("role") == role, f"{role} identity schema drift")
        parents = {c["parent_id"] for c in identity["clips"]}
        _require(len(parents) == expected_parents, f"{role} clip-parent count drift")
        for clip in identity["clips"]:
            _require(len(clip["frames"]) == 4, "clip length drift")
            for frame in clip["frames"]:
                identities.append((role, clip["clip_id"], frame))
    frame_ids = [frame["frame_id"] for _, _, frame in identities]
    _require(len(frame_ids) == len(set(frame_ids)), "frame reuse across development clips")
    _require(all(fid in teacher_by_id and fid in spatial_by_id and fid in arkit_by_id for fid in frame_ids), "identity cannot join frozen assets")
    outputs = request["outputs"]
    _require(set(outputs) == {"train_manifest", "validation_manifest", "disagreement_cache", "class_weights", "activation_bindings", "receipt"}, "output schema drift")
    out_paths = {name: _inside(root, value) for name, value in outputs.items()}
    _require(len(set(out_paths.values())) == 6 and all(not p.exists() for p in out_paths.values()), "overwrite forbidden")
    # All static joins and absence checks precede this delayed model load.
    infer = infer_factory(root, request)
    disagreements: dict[str, float] = {}
    for _, _, frame in identities:
        row = teacher_by_id[frame["frame_id"]] | {"teacher_depth_index": teacher_by_id[frame["frame_id"]]["index"]}
        value = float(infer(row))
        _require(math.isfinite(value) and value >= 0, "invalid frozen disagreement")
        disagreements[frame["frame_id"]] = value
    import numpy as np
    arrays = np.load(bindings["spatial_arrays"], mmap_mode="r")
    _require(set(("truth_clearance", "truth_valid")).issubset(arrays.files), "spatial array keys drift")
    manifests: dict[str, dict[str, Any]] = {}
    counts = {name: 0 for name in TRANSITIONS}
    for role, identity_key in (("train", "train_identity"), ("validation", "validation_identity")):
        identity = json.loads(bindings[identity_key].read_text())
        clips = []
        for clip in identity["clips"]:
            frames = []
            for frame in clip["frames"]:
                idx, _ = spatial_by_id[frame["frame_id"]]
                clearance = [float(x) if bool(v) and math.isfinite(float(x)) else None for x, v in zip(arrays["truth_clearance"][idx], arrays["truth_valid"][idx])]
                valid = [bool(v) and value is not None for v, value in zip(arrays["truth_valid"][idx], clearance)]
                states = [_state(float(value or 0), ok) for value, ok in zip(clearance, valid)]
                trow = teacher_by_id[frame["frame_id"]]
                assets = arkit_by_id[frame["frame_id"]]
                depth_path = Path(assets["depth"]["path"])
                intrinsics_path = Path(assets["intrinsics"]["path"])
                _require(depth_path.is_file() and _sha_file(depth_path) == str(assets["depth"]["sha256"]).upper(), "truth depth SHA mismatch")
                _require(intrinsics_path.is_file() and _sha_file(intrinsics_path) == str(assets["intrinsics"]["sha256"]).upper(), "intrinsics SHA mismatch")
                frames.append(frame | {"teacher_depth_ref": f"npy-index:{trow['index']}", "teacher_depth_sha256": request["teacher_depth"]["sha256"].upper(), "teacher_timestamp_ns": frame["timestamp_ns"], "teacher_valid": True, "tof_valid": any(valid), "frozen_a2_mean_abs_log_depth_disagreement": disagreements[frame["frame_id"]], "clearance_m": clearance, "geometry_state": states, "geometry_target_valid": [True, True, True], "truth_depth_path": str(depth_path), "truth_depth_sha256": str(assets["depth"]["sha256"]).upper(), "truth_depth_scale_m": 0.001, "intrinsics_fx_fy_cx_cy": _intrinsics(intrinsics_path)})
            for left, right in zip(frames, frames[1:]):
                for band in range(3):
                    if left["geometry_target_valid"][band] and right["geometry_target_valid"][band]:
                        counts[f"{left['geometry_state'][band]}_TO_{right['geometry_state'][band]}"] += 1
            clips.append({"clip_id": clip["clip_id"], "video_id": clip["video_id"], "parent_id": clip["parent_id"], "frames": frames})
        manifests[role] = {"schema": MANIFEST_SCHEMA, "protocol_sha256": request["protocol"]["sha256"].upper(), "evidence_limit": "DEVELOPMENT_SIGNAL_ONLY", "role": role, "clips": clips}
    _require(all(counts[name] > 0 for name in TRANSITIONS), "all nine train transition supports must be positive")
    beta = .999
    raw = [(1 - beta) / (1 - beta ** counts[name]) for name in TRANSITIONS]
    scale = len(raw) / sum(raw)
    weights = [x * scale for x in raw]
    cache = b"".join(_bytes({"schema": CACHE_SCHEMA, "frame_id": fid, "mean_abs_log_depth_disagreement": disagreements[fid]}) + b"\n" for fid in frame_ids)
    weight_doc = {"schema": WEIGHTS_SCHEMA, "train_manifest_sha256": hashlib.sha256(_bytes(manifests["train"], True)).hexdigest().upper(), "formula": "effective number (1-beta)/(1-beta^count), beta=0.999, normalized arithmetic mean=1", "transition_counts": counts, "weights": dict(zip(TRANSITIONS, weights))}
    payloads = {"train_manifest": _bytes(manifests["train"], True), "validation_manifest": _bytes(manifests["validation"], True), "disagreement_cache": cache, "class_weights": _bytes(weight_doc, True)}
    activation = {"schema": "blindassist_p3_temporal_development_screen_r0_activation_bindings", "protocol_sha256": request["protocol"]["sha256"].upper(), "train_manifest": {"path": outputs["train_manifest"], "sha256": hashlib.sha256(payloads["train_manifest"]).hexdigest().upper()}, "validation_manifest": {"path": outputs["validation_manifest"], "sha256": hashlib.sha256(payloads["validation_manifest"]).hexdigest().upper()}, "class_weights": {"path": outputs["class_weights"], "sha256": hashlib.sha256(payloads["class_weights"]).hexdigest().upper()}, "disagreement_cache": {"path": outputs["disagreement_cache"], "sha256": hashlib.sha256(cache).hexdigest().upper()}, "runtime_state": {"bonn_sealed_bundle_read": False, "holdout_outcomes_opened": False, "p3_model_constructed": False, "optimizer_constructed": False, "training_started": False, "a2_loaded_only_for_frozen_disagreement": True}}
    payloads["activation_bindings"] = _bytes(activation, True)
    receipt = {"schema": RECEIPT_SCHEMA, "protocol_sha256": request["protocol"]["sha256"].upper(), "producer_sha256": request["producer_sha256"].upper(), "inputs": {key: request[key]["sha256"].upper() for key in bindings}, "outputs": {key: hashlib.sha256(value).hexdigest().upper() for key, value in payloads.items()}, "runtime_state": activation["runtime_state"], "terminal": "P3_TEMPORAL_DEVELOPMENT_ASSETS_MATERIALIZED_DEVELOPMENT_SIGNAL_ONLY"}
    for key in ("train_manifest", "validation_manifest", "disagreement_cache", "class_weights", "activation_bindings"):
        _write_exclusive(out_paths[key], payloads[key])
    _write_exclusive(out_paths["receipt"], _bytes(receipt, True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text()), Path(__file__).resolve())


if __name__ == "__main__":
    main()
