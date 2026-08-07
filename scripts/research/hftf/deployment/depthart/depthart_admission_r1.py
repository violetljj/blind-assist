#!/usr/bin/env python3
"""DepthART R1 diagnostics: preprocessing, false-block attribution and relative control."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.research.hftf import evaluate_dav2_model_variant_gate_r1 as gate_r1


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shim() -> bool:
    try:
        from timm.models.layers.helpers import to_2tuple  # type: ignore # noqa: F401
        return False
    except (ImportError, ValueError):
        from timm.layers.helpers import to_2tuple  # type: ignore
        module = types.ModuleType("timm.models.layers.helpers")
        module.to_2tuple = to_2tuple
        sys.modules["timm.models.layers.helpers"] = module
        return True


def preprocess_audit(output: Path) -> dict[str, Any]:
    cases = []
    for width, height, target_width, target_height in ((640, 480, 640, 480), (1280, 720, 640, 480), (640, 480, 448, 448)):
        scale = max(target_width / width, target_height / height)
        new_width = int(np.round(scale * width / 32) * 32)
        new_height = int(np.round(scale * height / 32) * 32)
        if new_width < target_width:
            new_width = int(np.ceil(scale * width / 32) * 32)
        if new_height < target_height:
            new_height = int(np.ceil(scale * height / 32) * 32)
        sx, sy = new_width / width, new_height / height
        K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]], dtype=np.float32)
        expected = K.copy(); expected[0, :] *= sx; expected[1, :] *= sy
        cases.append({"raw": [width, height], "target": [target_width, target_height], "resized": [new_width, new_height], "scale": [sx, sy], "K_prime": expected.tolist(), "translation": [0.0, 0.0], "passed": bool(np.allclose(expected[0, 2], K[0, 2] * sx) and np.allclose(expected[1, 2], K[1, 2] * sy))})
    result = {"schema": "blindassist_depthart_admission_r1_preprocess_audit", "official_rule": "lower_bound_resize_keep_aspect_ratio_multiple_32", "crop_or_padding": False, "cases": cases, "all_passed": all(item["passed"] for item in cases)}
    _write(output, result); return result


def false_block_audit(roster_path: Path, source_root: Path, baseline_path: Path, candidate_path: Path, output: Path, contact_dir: Path | None = None) -> dict[str, Any]:
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    rows = gate_r1.load_geometry_rows(roster, source_root, baseline_path, candidate_path)
    by_band = collections.Counter(); by_horizon = collections.Counter(); by_range = collections.Counter(); by_sequence = collections.Counter(); cases = []
    for index, (row, roster_row) in enumerate(zip(rows, roster["rows"])):
        truth, candidate = row["sensor"], row["candidate"]
        if truth.get("status") != "VALID" or candidate.get("status") != "VALID": continue
        frame_bad = False
        for band in ("left", "center", "right"):
            for horizon in ("1.0", "1.5", "2.0"):
                tv = truth["bands"][band]["occupied_by_horizon"][horizon]; cv = candidate["bands"][band]["occupied_by_horizon"][horizon]
                if tv is False and cv is True:
                    frame_bad = True; by_band[band] += 1; by_horizon[horizon] += 1
                    clearance = float(truth["bands"][band]["clearance_m"])
                    key = "0-1m" if clearance < 1 else "1-1.5m" if clearance < 1.5 else "1.5-2m" if clearance < 2 else "2-3m" if clearance < 3 else ">3m"
                    by_range[key] += 1
                    cases.append({"index": index, "frame_id": roster_row["frame_id"], "sequence_id": roster_row["sequence_id"], "band": band, "horizon_m": float(horizon), "truth_clearance_m": clearance, "candidate_clearance_m": float(candidate["bands"][band]["clearance_m"])})
        if frame_bad: by_sequence[roster_row["sequence_id"]] += 1
    if contact_dir is not None and cases:
        contact_dir.mkdir(parents=True, exist_ok=True)
        unique = sorted({case["index"] for case in cases})
        for page, start in enumerate(range(0, len(unique), 16)):
            tiles = []
            for index in unique[start:start + 16]:
                rr = roster["rows"][index]; image = cv2.imread(str(source_root / rr["sequence_root"] / rr["rgb_path"]))
                if image is None: continue
                image = cv2.resize(image, (320, 240)); cv2.putText(image, f"{index} {rr['sequence_id'].split('-')[-3]}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA); tiles.append(image)
            while len(tiles) < 16: tiles.append(np.zeros_like(tiles[0]) if tiles else np.zeros((240, 320, 3), np.uint8))
            sheet = np.vstack([np.hstack(tiles[row * 4:(row + 1) * 4]) for row in range(4)]); cv2.imwrite(str(contact_dir / f"false_block_page_{page:02d}.jpg"), sheet)
    result = {"schema": "blindassist_depthart_admission_r1_false_block_audit", "false_block_decisions": len(cases), "false_block_frames": len({case["index"] for case in cases}), "by_band": dict(by_band), "by_horizon": dict(by_horizon), "by_truth_clearance_range": dict(by_range), "by_sequence": dict(by_sequence), "classification_limits": ["ground_boundary_textureless_thin_structure_confidence_not_available_in_frozen roster"]}
    _write(output, result); return result


def materialize_relative(source_root: Path, roster_path: Path, official_source: Path, checkpoint: Path, output: Path, device: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    shim = _shim(); relative_root = official_source / "relative"; sys.path.insert(0, str(relative_root))
    try:
        from tinyvim.model.dpt import TinyVimDepth
    finally: sys.path.pop(0)
    payload = torch.load(checkpoint, map_location="cpu"); model = TinyVimDepth(encoder="S"); model.load_state_dict(payload.get("model", payload), strict=True); model.to(device).eval()
    roster = json.loads(roster_path.read_text(encoding="utf-8")); cache = np.lib.format.open_memmap(output.with_suffix(output.suffix + ".partial"), mode="w+", dtype=np.float32, shape=(len(roster["rows"]), 480, 640))
    sys.path.insert(0, str(relative_root))
    try:
        for index, row in enumerate(roster["rows"]):
            image = cv2.imread(str(source_root / row["sequence_root"] / row["rgb_path"])); rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0; rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC); mean=np.asarray((.485,.456,.406),np.float32); std=np.asarray((.229,.224,.225),np.float32); tensor=torch.from_numpy(((rgb-mean)/std).transpose(2,0,1).copy()).unsqueeze(0)
            with torch.inference_mode(): prediction=F.interpolate(model(tensor.to(device)), (480,640), mode="bilinear", align_corners=True)[0,0].detach().float().cpu().numpy()
            if not np.isfinite(prediction).all(): raise ValueError(f"non-finite relative prediction at {index}")
            cache[index]=prediction; cache.flush(); print(f"relative frame {index+1}/{len(roster['rows'])}", flush=True)
    finally: sys.path.pop(0)
    del cache; os.replace(output.with_suffix(output.suffix + ".partial"), output)
    result={"schema":"blindassist_depthart_admission_r1_relative_materialization","frames":len(roster["rows"]),"shape":[len(roster["rows"]),480,640],"dtype":"float32","timm_compat_shim_used":shim,"output_sha256":__import__("hashlib").sha256(output.read_bytes()).hexdigest().upper(),"scale_authority":"NONE_TRUTH_ALIGNED_DIAGNOSTIC_ONLY"}; _write(output.with_suffix(".receipt.json"),result); return result


def align_relative(source_root: Path, roster_path: Path, official_source: Path, raw_path: Path, output: Path) -> dict[str, Any]:
    """Truth-align relative disparity per frame; diagnostic only, never deployment scale."""
    sys.path.insert(0, str(official_source / "relative"))
    try:
        from utils.metric import align_depth_least_square  # type: ignore
    finally:
        sys.path.pop(0)
    roster = json.loads(roster_path.read_text(encoding="utf-8")); raw = np.load(raw_path, mmap_mode="r")
    aligned = np.lib.format.open_memmap(output.with_suffix(output.suffix + ".partial"), mode="w+", dtype=np.float32, shape=raw.shape)
    for index, row in enumerate(roster["rows"]):
        depth_path = source_root / row["sequence_root"] / row["depth_path"]; truth = gate_r1.tum_depth_metres(gate_r1.normalize_depth_image(cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED), depth_path)); mask = np.isfinite(truth) & (truth >= .25) & (truth <= 6.0)
        aligned[index] = align_depth_least_square(np.asarray(raw[index]), truth, mask.astype(np.uint8), depth_cap=10).astype(np.float32)
    del aligned; os.replace(output.with_suffix(output.suffix + ".partial"), output)
    result={"schema":"blindassist_depthart_admission_r1_relative_truth_aligned","output_sha256":__import__("hashlib").sha256(output.read_bytes()).hexdigest().upper(),"scale_authority":"NONE_TRUTH_ALIGNED_DIAGNOSTIC_ONLY","frames":len(roster["rows"])}; _write(output.with_suffix(".receipt.json"),result); return result


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("preprocess-audit"); p.add_argument("--output",type=Path,required=True)
    f=sub.add_parser("false-block-audit")
    for name in ("roster","source-root","baseline","candidate","output"): f.add_argument(f"--{name}",type=Path,required=True)
    f.add_argument("--contact-dir",type=Path)
    r=sub.add_parser("materialize-relative")
    for name in ("source-root","roster","official-source","checkpoint","output"): r.add_argument(f"--{name}",type=Path,required=True)
    r.add_argument("--device",default="cuda")
    a=sub.add_parser("align-relative")
    for name in ("source-root","roster","official-source","raw","output"): a.add_argument(f"--{name}",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="preprocess-audit": result=preprocess_audit(args.output)
    elif args.command=="false-block-audit": result=false_block_audit(args.roster,args.source_root,args.baseline,args.candidate,args.output,args.contact_dir)
    elif args.command=="materialize-relative": result=materialize_relative(args.source_root,args.roster,args.official_source,args.checkpoint,args.output,args.device)
    else: result=align_relative(args.source_root,args.roster,args.official_source,args.raw,args.output)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__ == "__main__": main()
