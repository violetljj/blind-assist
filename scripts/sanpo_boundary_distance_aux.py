#!/usr/bin/env python3
"""Isolated boundary-distance auxiliary target contract for SANPO.

This module is intentionally not imported by the production trainer.  It lets
us inspect whether a distance-field auxiliary objective is numerically useful
without opening the blind holdout or changing the current model contract.

Class id 1 is ``boundary_step_curb``.  Targets are normalized by ``truncate``:

* unsigned: boundary pixels are 0 and other pixels approach +1;
* signed: boundary pixels are negative and non-boundary pixels are positive;
  the discrete interface lies between -1/truncate and +1/truncate.

For an empty boundary mask, both modes return an all-ones far-field target.  For
an all-boundary mask, unsigned returns zeros and signed returns all -1.  These
explicit sentinel behaviours avoid SciPy's image-edge convention leaking into
the auxiliary contract.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


BOUNDARY_CLASS_ID = 1
ALLOWED_SPLITS = {"train", "dev"}


def _validate_mask(boundary_mask: np.ndarray, truncate: float) -> np.ndarray:
    mask = np.asarray(boundary_mask)
    if mask.ndim != 2:
        raise ValueError(f"boundary mask must be 2-D, got shape {mask.shape}")
    if not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("truncate must be a finite value greater than zero")
    return mask.astype(bool, copy=False)


def boundary_distance_target(
    boundary_mask: np.ndarray,
    *,
    truncate: float = 16.0,
    signed: bool = False,
) -> np.ndarray:
    """Return a clipped float32 distance target normalized to [-1, 1] or [0, 1]."""
    mask = _validate_mask(boundary_mask, truncate)
    if not mask.any():
        return np.ones(mask.shape, dtype=np.float32)
    if mask.all():
        fill = -1.0 if signed else 0.0
        return np.full(mask.shape, fill, dtype=np.float32)

    outside = distance_transform_edt(~mask)
    if signed:
        inside = distance_transform_edt(mask)
        distance = outside - inside
        return np.clip(distance / truncate, -1.0, 1.0).astype(np.float32)
    return np.clip(outside / truncate, 0.0, 1.0).astype(np.float32)


def smooth_l1_target_and_weight(
    boundary_mask: np.ndarray,
    *,
    truncate: float = 16.0,
    signed: bool = False,
    min_weight: float = 0.05,
    boundary_weight: float = 2.0,
    empty_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Build target and per-pixel weights for a reduction='none' SmoothL1 loss.

    Weights decay linearly away from the boundary, retain ``min_weight`` for
    far-field negatives, and multiply boundary pixels by ``boundary_weight``.
    Empty masks use ``empty_weight`` everywhere so absence examples contribute
    without dominating frames that contain a boundary.
    """
    mask = _validate_mask(boundary_mask, truncate)
    for name, value in (
        ("min_weight", min_weight),
        ("boundary_weight", boundary_weight),
        ("empty_weight", empty_weight),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if boundary_weight < 1:
        raise ValueError("boundary_weight must be at least 1")

    target = boundary_distance_target(mask, truncate=truncate, signed=signed)
    if not mask.any():
        return target, np.full(mask.shape, empty_weight, dtype=np.float32)
    unsigned = (
        boundary_distance_target(mask, truncate=truncate, signed=False)
        if signed else target
    )
    weight = min_weight + (1.0 - min_weight) * (1.0 - unsigned)
    weight = weight.astype(np.float32)
    weight[mask] *= boundary_weight
    return target, weight


def _contains_blind(path: Path) -> bool:
    return any("blind" in part.lower() for part in path.parts)


def _safe_relative_path(root: Path, value: Any, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or _contains_blind(relative):
        raise ValueError(f"refusing {field} path outside train/dev contract: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"refusing escaping {field} path: {relative}") from error
    return resolved


def load_training_rows(dataset_root: Path) -> list[dict[str, Any]]:
    """Open only the canonical training manifest and reject blind rows/sessions."""
    root = dataset_root.resolve()
    if _contains_blind(dataset_root):
        raise ValueError(f"refusing blind dataset path: {dataset_root}")
    policy_path = root / "access_policy.json"
    manifest_path = root / "training_manifest.jsonl"
    if not policy_path.is_file() or not manifest_path.is_file():
        raise ValueError("dataset root must contain access_policy.json and training_manifest.jsonl")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if Path(str(policy.get("training_manifest", ""))).as_posix() != "training_manifest.jsonl":
        raise ValueError("access policy does not bind training_manifest.jsonl")
    forbidden_sessions = set(policy.get("forbidden_training_sessions", []))
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        split = str(row.get("split", ""))
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"refusing line {line_number} split {split!r}; only train/dev are allowed")
        if str(row.get("session_id", "")) in forbidden_sessions:
            raise ValueError(f"refusing forbidden training session on line {line_number}")
        _safe_relative_path(root, row.get("semantic_mask_path", ""), "semantic mask")
        rows.append(row)
    if not rows:
        raise ValueError("training manifest is empty")
    return rows


def diagnose_dataset(
    dataset_root: Path,
    *,
    truncate: float = 16.0,
    signed: bool = False,
    real_only: bool = True,
    analysis_size: int | None = None,
) -> dict[str, Any]:
    if analysis_size is not None and analysis_size <= 0:
        raise ValueError("analysis_size must be positive when provided")
    rows = load_training_rows(dataset_root)
    root = dataset_root.resolve()
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted(ALLOWED_SPLITS):
        split_rows = [row for row in rows if row["split"] == split]
        if real_only:
            split_rows = [
                row for row in split_rows
                if row.get("source", {}).get("source_id") == "sanpo_real_v0"
            ]
        quantile_samples: list[np.ndarray] = []
        total_pixels = 0
        near_boundary_pixels = 0
        total_loss_weight = 0.0
        frame_boundary_pixels: list[int] = []
        sessions: Counter[str] = Counter()
        scenes: Counter[str] = Counter()
        for row in split_rows:
            mask_path = _safe_relative_path(root, row["semantic_mask_path"], "semantic mask")
            with Image.open(mask_path) as image:
                if analysis_size is not None:
                    image = image.resize(
                        (analysis_size, analysis_size), resample=Image.Resampling.NEAREST,
                    )
                semantic = np.asarray(image)
            boundary = semantic == BOUNDARY_CLASS_ID
            target, weight = smooth_l1_target_and_weight(
                boundary, truncate=truncate, signed=signed,
            )
            flat_target = target.reshape(-1)
            flat_weight = weight.reshape(-1)
            # Full-resolution SANPO masks are 2208x1242.  Keep aggregate
            # fractions exact, but use a deterministic uniform stride for
            # quantiles so the diagnostic cannot accumulate multi-GB arrays.
            stride = max(1, flat_target.size // 4096)
            quantile_samples.append(flat_target[::stride][:4096])
            total_pixels += flat_target.size
            near_boundary_pixels += int((np.abs(flat_target) < 1.0).sum())
            total_loss_weight += float(flat_weight.sum(dtype=np.float64))
            frame_boundary_pixels.append(int(boundary.sum()))
            sessions[str(row.get("session_id", ""))] += 1
            scenes[str(row.get("scene_bucket", ""))] += 1
        if not quantile_samples:
            raise ValueError(f"no {'real-only ' if real_only else ''}{split} rows")
        sampled_values = np.concatenate(quantile_samples)
        frame_pixels = np.asarray(frame_boundary_pixels)
        by_split[split] = {
            "frame_count": len(split_rows),
            "session_count": len(sessions),
            "scene_frame_counts": dict(sorted(scenes.items())),
            "frames_with_boundary": int((frame_pixels > 0).sum()),
            "boundary_pixel_fraction": float(frame_pixels.sum() / total_pixels),
            "target_quantiles_deterministic_sample": {
                str(q): float(np.quantile(sampled_values, q)) for q in (0, 0.25, 0.5, 0.75, 0.95, 1)
            },
            "quantile_sample_pixels": int(sampled_values.size),
            "near_boundary_fraction_abs_lt_1": float(near_boundary_pixels / total_pixels),
            "mean_loss_weight": float(total_loss_weight / total_pixels),
        }
    return {
        "format": "blindassist_sanpo_boundary_distance_diagnostic_v1",
        "dataset_root": str(root),
        "contract": {
            "boundary_class_id": BOUNDARY_CLASS_ID,
            "truncate_pixels": truncate,
            "signed": signed,
            "normalized": True,
            "real_only": real_only,
            "analysis_size": analysis_size,
            "blind_access": "refused",
        },
        "splits": by_split,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--truncate", type=float, default=16.0)
    parser.add_argument("--signed", action="store_true")
    parser.add_argument(
        "--analysis-size",
        type=int,
        help="Nearest-resize masks to the square training resolution before diagnosis.",
    )
    parser.add_argument(
        "--include-non-real", action="store_true",
        help="Include procedural/pseudo-label train/dev rows; blind remains forbidden.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = diagnose_dataset(
        args.dataset_root,
        truncate=args.truncate,
        signed=args.signed,
        real_only=not args.include_non_real,
        analysis_size=args.analysis_size,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
