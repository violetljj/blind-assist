#!/usr/bin/env python3
"""Run a frozen HFTF student on real video frames and emit a sidecar.

This is an actual forward pass through the existing HFTF temporal student. It
does not claim that a TartanGround-trained model is calibrated on public video;
the sidecar is a domain-transfer discovery signal only. Future-field change is
computed from consecutive predicted future risk fields within each session.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from scripts.research.candidate_event_mining.pipeline import ContractError, read_jsonl, sha256_file, write_json, write_jsonl


HFTF_SIDECAR_SCHEMA = "blindassist_candidate_event_mining_hftf_sidecar_manifest_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_ROOT = REPO_ROOT / "scripts" / "research" / "hftf"
if str(HFTF_ROOT) not in sys.path:
    sys.path.insert(0, str(HFTF_ROOT))

from evaluate_stage_c_d6_sanpo_real_event_transfer import load_model  # noqa: E402
from train_stage_c_d5_tartanground_development_student import MEAN, STD  # noqa: E402


def _score(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"non-finite HFTF score: {value}")
    return max(0.0, min(1.0, value))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ContractError("HFTF sidecar input trace is empty")
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (str(row.get("source_id", "")), str(row.get("session_id", "")), int(row.get("frame_index", -1)))
        if not all(key[:2]) or key[2] < 0 or key in keys:
            raise ContractError(f"invalid or duplicate HFTF frame key: {key}")
        keys.add(key)
        if not Path(str(row.get("frame_ref", ""))).resolve().is_file():
            raise ContractError(f"HFTF frame is missing: {row.get('frame_ref')}")
    return rows


def _tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        value = tvf.resize(
            image.convert("RGB"),
            [128, 224],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    return tvf.normalize(tvf.pil_to_tensor(value).float().div_(255.0), MEAN, STD)


def _history_batch(rows: list[dict[str, Any]], index: int) -> torch.Tensor:
    start = max(0, index - 4)
    history = rows[start : index + 1]
    if len(history) < 5:
        history = [rows[0]] * (5 - len(history)) + history
    return torch.stack([_tensor(Path(str(item["frame_ref"])).resolve()) for item in history])


def _field_features(risk: np.ndarray, known: np.ndarray, previous: np.ndarray | None, scale: float) -> dict[str, float]:
    # The caller passes one frame's prediction with shape
    # [horizon, height, direction, cell, cell]; the batch dimension has
    # already been removed. Keep the horizon axis intact and exclude the
    # current-frame slot only here.
    future = risk[1:, :, :, :]
    future_known = known[1:, :, :, :]
    current = future.mean()
    known_mean = future_known.mean()
    central = future[..., 2:4].mean()
    output = {
        "hftf.future_field_risk": _score(current),
        "hftf.future_field_known": _score(known_mean),
        "hftf.future_field_central_risk": _score(central),
    }
    if previous is not None:
        delta = float(np.abs(future - previous).mean())
        positive_delta = float(np.maximum(future - previous, 0.0).mean())
        central_delta = float(np.abs(future[..., 2:4] - previous[..., 2:4]).mean())
        output.update(
            {
                "hftf.future_field_change": _score(delta / max(scale, 1e-6)),
                "hftf.future_risk_field_delta": _score(positive_delta / max(scale, 1e-6)),
                "hftf.future_field_central_change": _score(central_delta / max(scale, 1e-6)),
            }
        )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.frame_trace.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest_output.resolve()
    if output.exists() or manifest_path.exists():
        raise ContractError("refusing to overwrite HFTF sidecar outputs")
    rows = _read_trace(input_path)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["source_id"]), str(row["session_id"])), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["frame_index"]))
    model, checkpoint = load_model(args.pretrained.resolve(), args.checkpoint.resolve())
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ContractError("--device cuda requested but CUDA is unavailable")
    device = torch.device(device_name)
    model = model.to(device).eval()
    results: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for group in grouped.values():
            for start in range(0, len(group), args.batch_size):
                batch_rows = group[start : start + args.batch_size]
                batch = torch.stack([_history_batch(group, start + index) for index in range(len(batch_rows))]).to(device)
                risk_logits, known_logits = model(batch)
                risks = torch.sigmoid(risk_logits).detach().cpu().numpy()
                knowns = torch.sigmoid(known_logits).detach().cpu().numpy()
                for index, row in enumerate(batch_rows):
                    key = (str(row["source_id"]), str(row["session_id"]), int(row["frame_index"]))
                    results[key] = (risks[index], knowns[index])
    del model
    if len(results) != len(rows):
        raise ContractError(f"HFTF output count mismatch: {len(results)} != {len(rows)}")

    raw_deltas: list[float] = []
    previous_by_group: dict[tuple[str, str], np.ndarray] = {}
    for group_key, group in grouped.items():
        for row in group:
            risk, _known = results[(str(row["source_id"]), str(row["session_id"]), int(row["frame_index"]))]
            future = risk[1:, :, :, :]
            previous = previous_by_group.get(group_key)
            if previous is not None:
                raw_deltas.append(float(np.abs(future - previous).mean()))
            previous_by_group[group_key] = future
    scale = float(np.quantile(np.asarray(raw_deltas, dtype=np.float32), 0.90)) if raw_deltas else 1.0
    scale = max(scale, 1e-6)

    sidecar_rows: list[dict[str, Any]] = []
    previous_by_group.clear()
    for group_key, group in grouped.items():
        for row in group:
            key = (str(row["source_id"]), str(row["session_id"]), int(row["frame_index"]))
            risk, known = results[key]
            previous = previous_by_group.get(group_key)
            sidecar_rows.append(
                {
                    "source_id": row["source_id"],
                    "session_id": row["session_id"],
                    "frame_index": int(row["frame_index"]),
                    "signals": _field_features(risk, known, previous, scale),
                }
            )
            previous_by_group[group_key] = risk[1:, :, :, :]
    sidecar_rows.sort(key=lambda row: (str(row["source_id"]), str(row["session_id"]), int(row["frame_index"])))
    write_jsonl(output, sidecar_rows)
    manifest = {
        "schema": HFTF_SIDECAR_SCHEMA,
        "sidecar_id": "cem-real-hftf-student-domain-transfer-r0",
        "input_trace": {"path": str(input_path), "sha256": sha256_file(input_path), "frame_count": len(rows)},
        "sidecar": {"path": str(output), "sha256": sha256_file(output), "frame_count": len(sidecar_rows)},
        "model": {
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256_file(args.pretrained.resolve()),
            "architecture": checkpoint.get("architecture", "pooled"),
            "temporal_mode": checkpoint.get("temporal_mode", "joint"),
            "arm": checkpoint.get("arm", "history"),
            "device": str(device),
            "runtime": "TemporalStudent.forward",
        },
        "future_field": {
            "horizons": ["near", "far"],
            "change_definition": "mean absolute delta of predicted future risk field versus previous frame in the same source/session",
            "normalization": {"scale_quantile": 0.90, "scale": scale},
        },
        "real_hftf_model_inference": True,
        "domain_transfer_unvalidated": True,
        "proxy": False,
        "data_role": "THESIS_DEVELOPMENT_CONSUMED_DISCOVERY",
        "authorization": {"event_truth": False, "training": False, "production": False, "safety": False},
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> int:
    try:
        manifest = run(parse_args())
    except (ContractError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "frame_count": manifest["sidecar"]["frame_count"], "output": manifest["sidecar"]["path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
