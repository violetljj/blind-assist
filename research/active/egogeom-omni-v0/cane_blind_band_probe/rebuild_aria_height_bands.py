"""Build exact 5 cm voxel-level Aria height-band GT across 6 height bands and 2 reference stages.

Reuses the evaluator-only FoundationStereo voxelization and local ground geometry from
`aria_surface_witness/reference.py` and heading/frame-eligibility parity from `aria_reference_heading/rebuild.py`.
Writes compact `aria_height_band_gt.npz` + `aria_height_band_gt_meta.json` under
`artifacts.local/work/egogeom-omni-v0/cane_blind_band_probe/`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
REPO = ROUTE.parents[2]
ART = REPO / "artifacts.local/work/egogeom-omni-v0"
RAW = REPO / "artifacts.local/downloads/aria-gen2-pilot-20261117"
OUT_DIR = ART / "cane_blind_band_probe"
WITNESS = ROUTE / "math_theory/m9_pgl4_frobenius_integrability/aria_surface_witness"
sys.path.insert(0, str(WITNESS))

from reference import Ground, centres, keys3, read_native  # noqa: E402

BANDS = (
    "full_ground",       # h_above_ground in [0.15, 1.90]
    "low_only_ground",   # h_above_ground in [0.15, 0.65)
    "midup_ground",      # h_above_ground in [0.65, 1.90]
    "head_ground",       # h_above_ground in [1.35, 1.90]
    "head_anchored_85",  # z_world - z_cam in [-0.85, +0.25] (zero ground dependency)
    "head_anchored_70",  # z_world - z_cam in [-0.70, +0.25] (zero ground dependency)
)
STAGES = ("centered_full", "strict_past")


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _corridor_dist_from_voxels(vox_keys: np.ndarray, pos_xy: np.ndarray, fw: np.ndarray) -> float:
    if len(vox_keys) == 0:
        return float(np.inf)
    side = np.array([fw[1], -fw[0]], dtype=np.float64)
    offset = centres(vox_keys)[:, :2] - pos_xy
    f = offset @ fw
    s = offset @ side
    z = np.sort(f[(np.abs(s) <= 0.30) & (f >= 0.30) & (f <= 4.0)])
    if len(z) < 8:
        return float(np.inf)
    hits = np.flatnonzero(np.searchsorted(z, z + 0.30, side="right") - np.arange(len(z)) >= 8)
    return float(z[hits[0]]) if len(hits) else float(np.inf)


def _eval_band_subset(
    pw_sub: np.ndarray,
    position: np.ndarray,
    fw: np.ndarray,
    ground: Ground | None,
) -> dict[str, float]:
    """Compute corridor voxel distance across all 6 height bands for pre-selected corridor points."""
    out: dict[str, float] = {}
    z_rel = pw_sub[:, 2] - position[2]
    h = ground.above(pw_sub) if ground is not None else np.full(len(pw_sub), np.nan)

    masks = {
        "full_ground": (h >= 0.15) & (h <= 1.90),
        "low_only_ground": (h >= 0.15) & (h < 0.65),
        "midup_ground": (h >= 0.65) & (h <= 1.90),
        "head_ground": (h >= 1.35) & (h <= 1.90),
        "head_anchored_85": (z_rel >= -0.85) & (z_rel <= 0.25),
        "head_anchored_70": (z_rel >= -0.70) & (z_rel <= 0.25),
    }
    pos_xy = position[:2]
    for band, m in masks.items():
        pts = pw_sub[m]
        vox = np.unique(keys3(pts)) if len(pts) > 0 else np.empty(0, dtype=np.int64)
        out[band] = _corridor_dist_from_voxels(vox, pos_xy, fw)
    return out


def rebuild_clip(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    t0 = perf_counter()
    clip = path.stem
    m_clip = re.fullmatch(r"aria_(walk_[01])_c(\d+)", clip)
    if m_clip is None:
        raise ValueError(f"Unexpected Aria clip stem: {clip}")
    walk, number = m_clip.groups()
    base = RAW / walk / "x/depth"
    params = base / "pinhole_camera_parameters.json"
    start = 150 * int(number)
    metas = json.loads(params.read_text(encoding="utf-8"))[::6][start : start + 150]
    stream = ART / "math_theory/v41_stream/aria_gen2" / path.name

    with np.load(path, allow_pickle=False) as f:
        rot = f["poses_R"]
        pos = f["poses_t"].astype(np.float64)
        allowed = f["valid_heading"].astype(bool)
        old = f["eval_rows"].astype(np.float64)
    with np.load(stream, allow_pickle=False) as f:
        strict = f["strict_fwd"][:, :2].astype(np.float64)
        np.testing.assert_allclose(rot, f["poses_R"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(pos, f["poses_t"], rtol=0, atol=1e-6)
    strict /= np.linalg.norm(strict, axis=1, keepdims=True)

    n = len(metas)
    if n != 150 or len(rot) != 150:
        raise AssertionError(f"{clip}: expected 150 frames, got {n}")
    ii = np.arange(n)
    centered_v = pos[np.minimum(n - 1, ii + 2), :2] - pos[np.maximum(0, ii - 2), :2]
    past_v = pos[np.where(ii >= 2, ii, 2), :2] - pos[np.maximum(0, ii - 2), :2]
    center_norm = np.linalg.norm(centered_v, axis=1)
    assert np.array_equal(allowed, (np.linalg.norm(past_v, axis=1) >= 0.15) & (center_norm >= 0.15))
    centered = centered_v / np.where(center_norm == 0, 1, center_norm)[:, None]
    seg = np.r_[0.0, np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)]

    rows = {
        stage: {band: np.full_like(old, np.nan) for band in BANDS}
        for stage in STAGES
    }

    cache: dict[int, np.ndarray] = {}
    for i in np.flatnonzero(allowed):
        lo, hi = max(0, i - 4), min(n, i + 5)
        for j in list(cache):
            if j < lo:
                del cache[j]
        for j in range(lo, hi):
            if j not in cache:
                cache[j] = read_native(base, metas[j], rot[j], pos[j]).points
        pw = np.concatenate([cache[j] for j in range(lo, hi)])
        n_past = sum(len(cache[j]) for j in range(lo, i + 1))

        # 1) centered_full (exact parity with frozen eval_rows)
        if np.count_nonzero(pw[:, 2] < pos[i, 2] - 0.8) >= 200:
            ground_full = Ground(pw, pos[i])
            rel = pw[:, :2] - pos[i, :2]
            fw_c = centered[i]
            side_c = np.array([fw_c[1], -fw_c[0]], dtype=np.float64)
            take_c = (np.abs(rel @ side_c) <= 0.35) & (rel @ fw_c >= 0.25) & (rel @ fw_c <= 4.05)
            d_c = _eval_band_subset(pw[take_c], pos[i], fw_c, ground_full)
            cover_full = ground_full.coverage(pos[i], fw_c, d_c["full_ground"])
            for band in BANDS:
                rows["centered_full"][band][i] = [d_c[band], cover_full, 0.0, seg[i]]

        # 2) strict_past (zero future depth leakage, causal heading)
        pw_past = pw[:n_past]
        if np.count_nonzero(pw_past[:, 2] < pos[i, 2] - 0.8) >= 200:
            ground_past = Ground(pw_past, pos[i])
            rel_p = pw_past[:, :2] - pos[i, :2]
            fw_s = strict[i]
            side_s = np.array([fw_s[1], -fw_s[0]], dtype=np.float64)
            take_p = (np.abs(rel_p @ side_s) <= 0.35) & (rel_p @ fw_s >= 0.25) & (rel_p @ fw_s <= 4.05)
            d_p = _eval_band_subset(pw_past[take_p], pos[i], fw_s, ground_past)
            cover_past = ground_past.coverage(pos[i], fw_s, d_p["full_ground"])
            for band in BANDS:
                rows["strict_past"][band][i] = [d_p[band], cover_past, 0.0, seg[i]]

    # Exact parity assertion against frozen eval_rows
    replay_full = rows["centered_full"]["full_ground"]
    errmask = np.isfinite(old) & np.isfinite(replay_full)
    max_error = float(np.max(np.abs(old[errmask] - replay_full[errmask]))) if errmask.any() else 0.0
    np.testing.assert_allclose(old, replay_full, rtol=0, atol=1e-8, equal_nan=True, err_msg=clip)

    arrays: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, int]] = {}
    for stage in STAGES:
        stats[stage] = {}
        for band in BANDS:
            arr = rows[stage][band]
            arrays[f"{clip}__{stage}__{band}"] = arr
            val = (np.isfinite(arr[:, 0]) | np.isinf(arr[:, 0])) & ~np.isnan(arr[:, 1]) & (arr[:, 1] >= 0.6)
            pos_4m = val & np.isfinite(arr[:, 0]) & (arr[:, 0] <= 4.0)
            stats[stage][band] = {"valid": int(val.sum()), "pos_4m": int(pos_4m.sum())}

    meta = {
        "clip": clip,
        "frozen_pack_sha256": sha256(path),
        "v41_stream_sha256": sha256(stream),
        "max_abs_packed_replay_error": max_error,
        "elapsed_s": round(perf_counter() - t0, 2),
        "counts": stats,
    }
    print(
        f"[aria_gt] {clip}: max_err={max_error:.2e} | "
        f"cf(valid={stats['centered_full']['full_ground']['valid']}, "
        f"full={stats['centered_full']['full_ground']['pos_4m']}, "
        f"midup={stats['centered_full']['midup_ground']['pos_4m']}, "
        f"ha85={stats['centered_full']['head_anchored_85']['pos_4m']}) | "
        f"sp(valid={stats['strict_past']['full_ground']['valid']}, "
        f"full={stats['strict_past']['full_ground']['pos_4m']}, "
        f"midup={stats['strict_past']['midup_ground']['pos_4m']}, "
        f"ha85={stats['strict_past']['head_anchored_85']['pos_4m']}) "
        f"({meta['elapsed_s']}s)",
        flush=True,
    )
    return arrays, meta


def main() -> None:
    t0 = perf_counter()
    src = sorted((ART / "zeroshot_eval/aria_g2_20clips").glob("*.npz"))
    if len(src) != 19:
        raise AssertionError(f"Expected 19 Aria clips, found {len(src)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_arrays: dict[str, np.ndarray] = {}
    clip_metas: list[dict] = []
    agg = {st: {b: {"valid": 0, "pos_4m": 0} for b in BANDS} for st in STAGES}

    with threadpool_limits(limits=1):
        for p in src:
            arrs, meta = rebuild_clip(p)
            all_arrays.update(arrs)
            clip_metas.append(meta)
            for st in STAGES:
                for b in BANDS:
                    agg[st][b]["valid"] += meta["counts"][st][b]["valid"]
                    agg[st][b]["pos_4m"] += meta["counts"][st][b]["pos_4m"]

    out_npz = OUT_DIR / "aria_height_band_gt.npz"
    out_meta = OUT_DIR / "aria_height_band_gt_meta.json"
    np.savez_compressed(out_npz, **all_arrays)
    meta_payload = {
        "n_clips": len(src),
        "bands": list(BANDS),
        "stages": list(STAGES),
        "aggregate_counts": agg,
        "output_npz": str(out_npz),
        "output_npz_sha256": sha256(out_npz),
        "output_npz_bytes": out_npz.stat().st_size,
        "total_elapsed_s": round(perf_counter() - t0, 2),
        "clips": clip_metas,
    }
    out_meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_npz} ({out_npz.stat().st_size / 1e6:.2f} MB) in {meta_payload['total_elapsed_s']}s")
    print("Aggregate counts:", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
