"""Train S1-A/S1-B on the frozen A4 development stream.

This is a development-only trainer.  S1-A learns geometry and teacher features;
S1-B is fail-closed until an explicit S1-A geometry-pass receipt exists.
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from clearance_student_mobile_s1 import (  # noqa: E402
    ClearanceStudentMobileS1, feature_distillation_loss, geometry_loss,
    normalize_bgr_batch, parameter_count, require_finite_metrics,
)
from evaluate_dav2_model_variant_gate_r0 import sha256_file  # noqa: E402


def truth_paths(record: dict[str, Any]) -> tuple[Path, Path]:
    rgb = Path(str(record["rgb_path"]))
    root, stem = rgb.parent.parent, rgb.stem
    return root / "lowres_depth" / f"{stem}.png", root / "confidence" / f"{stem}.png"


def eligible(records: list[dict[str, Any]], role: str) -> list[int]:
    out = []
    for i, r in enumerate(records):
        if r.get("role") != role:
            continue
        d, c = truth_paths(r)
        depth = cv2.imread(str(d), cv2.IMREAD_UNCHANGED)
        conf = cv2.imread(str(c), cv2.IMREAD_UNCHANGED)
        if depth is None or conf is None:
            raise OSError(f"S1 truth decode failed: {r.get('frame_id')}")
        if np.any((conf == 2) & (depth >= 250) & (depth <= 6000)):
            out.append(i)
    return out


def batch(records, indices, teacher, feature_cache, device):
    images, truths, teachers, features = [], [], [], []
    for i in indices:
        r = records[i]
        rgb = cv2.imread(str(r["rgb_path"]), cv2.IMREAD_COLOR)
        dpath, cpath = truth_paths(r)
        d = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(str(cpath), cv2.IMREAD_UNCHANGED)
        if rgb is None or d is None or c is None:
            raise OSError(f"S1 asset decode failed: {r.get('frame_id')}")
        depth = d.astype(np.float32) / 1000.0
        truths.append(np.where(np.isfinite(depth), depth, 8.0).astype(np.float32))
        images.append(torch.from_numpy(rgb.transpose(2, 0, 1).copy()))
        teachers.append(np.asarray(teacher[i], dtype=np.float32))
        features.append(np.asarray(feature_cache[i], dtype=np.float32))
    return (normalize_bgr_batch(images).to(device), torch.from_numpy(np.stack(truths)).to(device),
            torch.from_numpy(np.stack(teachers)).to(device), torch.from_numpy(np.stack(features)).to(device))


def run_epoch(model, records, indices, teacher, feature_cache, device, bs, stage, opt=None, scaler=None):
    model.train(opt is not None)
    sums: dict[str, list[float]] = {}
    for start in range(0, len(indices), bs):
        ids = indices[start:start + bs]
        x, truth, teach, tf = batch(records, ids, teacher, feature_cache, device)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(x, (192, 256))
            loss, parts = geometry_loss(pred, truth, teach, {"depth_clamp_m": [0.25, 6.0]})
            fd = feature_distillation_loss(pred["features"], tuple(tf[:, j] for j in range(4)), model.feature_projections)
            loss = loss + 0.10 * fd
            if stage == "S1-B":
                # Warm, non-collapse task head supervision uses teacher-derived
                # clearance/occupancy targets only after S1-A geometry pass.
                target_clearance = torch.clamp(torch.quantile(teach.flatten(1), 0.02, dim=1), 0.2, 6.0).unsqueeze(1).expand(-1, 3)
                loss = loss + 0.05 * F.smooth_l1_loss(pred["clearance"], target_clearance)
            parts = dict(parts, feature=fd)
        if opt is not None:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        for k, v in parts.items():
            sums.setdefault(k, []).append(float(v.detach().float().mean().cpu()))
        sums.setdefault("total", []).append(float(loss.detach().cpu()))
    return {k: float(np.mean(v)) for k, v in sums.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True); p.add_argument("--teacher-manifest", type=Path, required=True)
    p.add_argument("--teacher-features-manifest", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--stage", choices=["S1-A", "S1-B"], default="S1-A"); p.add_argument("--checkpoint", type=Path)
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.output_root.exists(): raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); manifest = json.loads(args.teacher_manifest.read_text(encoding="utf-8")); fmanifest = json.loads(args.teacher_features_manifest.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "clearance-student-mobile-s1" or manifest.get("truth_inputs_opened") is not False or fmanifest.get("truth_inputs_opened") is not False: raise ValueError("S1 protocol/teacher firewall mismatch")
    feature_path = Path(str(fmanifest["feature_path"]));
    if sha256_file(feature_path) != fmanifest["feature_sha256"]: raise ValueError("teacher feature cache hash mismatch")
    features = np.load(feature_path, mmap_mode="r"); records = manifest["records"]; teacher = np.load(Path(str(manifest["teacher_depth"]["path"])), mmap_mode="r")
    if features.shape[0] != len(records) or features.shape[1:] != (4, 384, 8, 8): raise ValueError(f"teacher feature shape mismatch: {features.shape}")
    if args.stage == "S1-B":
        if args.checkpoint is None: raise ValueError("S1-B requires --checkpoint from S1-A")
        receipt = args.checkpoint.parent / "S1-A_GEOMETRY_PASS.json"
        if not receipt.exists(): raise ValueError("S1-B blocked: missing S1-A_GEOMETRY_PASS receipt")
    seed = 20260806; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = ClearanceStudentMobileS1(False).to(device)
    if args.checkpoint: model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2); scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_ids, valid_ids = eligible(records, "train"), eligible(records, "validation"); args.output_root.mkdir(parents=True)
    best, best_epoch, history = math.inf, None, []; started = time.perf_counter()
    for epoch in range(args.epochs):
        order = [train_ids[i] for i in np.random.default_rng(seed + epoch).permutation(len(train_ids))]
        tr = run_epoch(model, records, order, teacher, features, device, args.batch_size, args.stage, opt, scaler)
        with torch.inference_mode(): va = run_epoch(model, records, valid_ids, teacher, features, device, args.batch_size, args.stage)
        row = {"epoch": epoch + 1, "train": tr, "validation": va}; history.append(row); print(json.dumps(row), flush=True)
        if va["total"] < best:
            best, best_epoch = va["total"], epoch + 1; torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output_root / "checkpoint.pth")
    result = {"schema": "blindassist_clearance_student_mobile_s1_training_result", "stage": args.stage, "protocol_sha256": sha256_file(args.protocol), "teacher_manifest_sha256": sha256_file(args.teacher_manifest), "teacher_features_manifest_sha256": sha256_file(args.teacher_features_manifest), "truth_inputs_opened": False, "parameter_count": parameter_count(model), "selected_epoch": best_epoch, "best_validation_total": best, "history": history, "terminal": f"{args.stage}_TRAINING_COMPLETE_DEVELOPMENT_ONLY", "qnn_profile_authorized": False, "production_model_replacement_authorized": False, "training_seconds": time.perf_counter() - started}
    (args.output_root / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require_finite_metrics({"best_validation_total": best}); print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))

if __name__ == "__main__": main()
