"""Compare causal walking directions on the consumed ADT corridor grids."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


ARMS = ("head", "past_one", "past_three")
MAX_GAP = 0.30
MIN_SPEED = 0.30
MAX_ANGLE = np.deg2rad(20)
MIN_POINTS = 38
LATERAL = np.linspace(-1.45, 1.45, 30)
FORWARD = np.linspace(0.35, 5.25, 50)
GRID_X, GRID_Z = (a.ravel() for a in np.meshgrid(LATERAL, FORWARD, indexing="ij"))


def axes(poses):
    forward = poses[:, [0, 2], 2].copy()
    if np.any(np.linalg.norm(forward, axis=1) < 1e-6):
        raise ValueError("A camera forward vector is vertical.")
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    right = np.column_stack([-forward[:, 1], forward[:, 0]])
    return forward, right


def displacement(positions, times, lag, past):
    """Return displacement direction and validity at the window endpoint or start."""
    start = slice(lag, None) if past else slice(None, -lag)
    delta = np.zeros_like(positions)
    delta[start] = positions[lag:] - positions[:-lag]
    elapsed = times[lag:] - times[:-lag]
    gaps = np.lib.stride_tricks.sliding_window_view(np.diff(times), lag)
    speed = np.linalg.norm(delta[start], axis=1) / elapsed
    valid = np.zeros(len(times), dtype=bool)
    valid[start] = (speed >= MIN_SPEED) & (gaps.max(axis=1) <= MAX_GAP)
    return delta, valid


def heading_relative_to_head(delta, forward, right):
    return np.arctan2((delta * right).sum(axis=1), (delta * forward).sum(axis=1))


def causal_heading(positions, times, forward, right, lag):
    delta, valid = displacement(positions, times, lag, past=True)
    angle = heading_relative_to_head(delta, forward, right)
    use_translation = valid & (np.abs(angle) <= MAX_ANGLE)
    return np.where(use_translation, angle, 0.0), use_translation


def corridor_count(grid, angle):
    sine, cosine = np.sin(angle[:, None]), np.cos(angle[:, None])
    x = GRID_X * cosine - GRID_Z * sine
    z = GRID_X * sine + GRID_Z * cosine
    inside = (np.abs(x) <= 0.30) & (z >= 0.30) & (z <= 2.0)
    return np.einsum("nqi,ni->nq", grid.reshape(len(angle), 2, -1), inside, dtype=np.int32)


def confirmed(raw, times):
    out = np.zeros_like(raw)
    gap = np.diff(times) <= MAX_GAP
    two_of_three = (raw[:-2].astype(np.uint8) + raw[1:-1] + raw[2:]) >= 2
    out[2:] = two_of_three & (gap[:-1] & gap[1:])[:, None]
    return out


def self_check():
    grid = np.zeros((1, 2, 30, 50), dtype=np.uint16)
    grid[0, 0, 19, 10] = 100  # x=0.45 m, z=1.35 m
    assert corridor_count(grid, np.zeros(1))[0, 0] == 0
    assert corridor_count(grid, np.array([np.arctan2(0.45, 1.35)]))[0, 0] == 100

    pos = np.column_stack([np.arange(9) * 0.01, np.arange(9) * 0.1])
    times = np.arange(9) / 5
    forward = np.tile([0.0, 1.0], (9, 1))
    right = np.tile([-1.0, 0.0], (9, 1))
    for lag in (1, 3):
        angle, use = causal_heading(pos, times, forward, right, lag)
        changed = pos.copy()
        changed[6:] *= -20
        changed_angle, changed_use = causal_heading(changed, times, forward, right, lag)
        assert np.array_equal(angle[:6], changed_angle[:6])
        assert np.array_equal(use[:6], changed_use[:6])
        assert use[lag:].all() and (angle[lag:] < 0).all()


def load_sequence(grid_file, pose_dir):
    seq = pose_dir / grid_file.stem
    meta = json.loads((seq / "meta.json").read_text(encoding="utf-8"))
    with np.load(seq / "poses.npz", allow_pickle=False) as data:
        times = (data["t_ns"] - data["t_ns"][0]) / 1e9
        poses = data["T_sc"]
    with np.load(grid_file, allow_pickle=False) as data:
        chain, reference = data["chain"], data["gt"]
    n = len(times)
    if not (
        meta["sequence"] == grid_file.stem
        and meta["frames"] == n
        and poses.shape == (n, 4, 4)
        and chain.shape == reference.shape == (n, 2, 30, 50)
        and np.isfinite(poses).all()
        and (np.diff(times) > 0).all()
    ):
        raise ValueError(f"Pose/grid contract failed for {grid_file.stem}.")
    return times, poses, chain, reference


def evaluate_sequence(grid_file, pose_dir):
    times, poses, chain, reference = load_sequence(grid_file, pose_dir)
    forward, right = axes(poses)
    positions = poses[:, [0, 2], 3]

    # Predictions have access only to the anchored depth grid and current or past poses.
    past_one, use_one = causal_heading(positions, times, forward, right, 1)
    past_three, use_three = causal_heading(positions, times, forward, right, 3)
    headings = np.stack([np.zeros_like(times), past_one, past_three])
    raw = np.stack([corridor_count(chain, angle) >= MIN_POINTS for angle in headings])
    alerts = np.stack([confirmed(arm, times) for arm in raw])

    # Reference labels are computed after the causal predictions. Future poses
    # supply a realized translation-direction proxy, not a physical collision label.
    future_delta, future_valid = displacement(positions, times, 3, past=False)
    future_angle = heading_relative_to_head(future_delta, forward, right)
    in_view = future_valid & (np.abs(future_angle) <= MAX_ANGLE)
    reference_counts = corridor_count(reference, future_angle)
    labels = np.where(reference_counts >= MIN_POINTS, 1, np.where(reference_counts == 0, 0, -1))
    labels[~in_view] = -1
    return {
        "raw": raw,
        "alerts": alerts,
        "labels": labels,
        "headings": headings,
        "future_angle": future_angle,
        "future_valid": future_valid,
        "in_view": in_view,
        "used_translation": np.stack([use_one, use_three]),
    }


def classification(labels, decisions):
    positive, negative = labels == 1, labels == 0
    for arm, predicted in zip(ARMS, decisions):
        base = decisions[0]
        yield arm, {
            "tp": int((positive & predicted).sum()),
            "fn": int((positive & ~predicted).sum()),
            "fp": int((negative & predicted).sum()),
            "tn": int((negative & ~predicted).sum()),
            "fn_fixed": int((positive & ~base & predicted).sum()),
            "fn_added": int((positive & base & ~predicted).sum()),
            "fp_fixed": int((negative & base & ~predicted).sum()),
            "fp_added": int((negative & ~base & predicted).sum()),
        }


def summarize(out):
    labels, decisions = out["labels"], out["alerts"]
    view = out["in_view"]
    angled = view & (np.abs(out["future_angle"]) >= np.deg2rad(10))
    error = out["headings"][:, view] - out["future_angle"][view]
    return {
        "frames": int(len(view)),
        "future_walking": int(out["future_valid"].sum()),
        "in_view": int(view.sum()),
        "ambiguous_queries_in_view": int((labels[view] < 0).sum()),
        "translation_used_in_view": [int(n) for n in out["used_translation"][:, view].sum(axis=1)],
        "heading_mae_deg": dict(zip(ARMS, np.degrees(np.abs(error)).mean(axis=1).round(3).tolist())),
        "confirmed": dict(classification(labels, decisions)),
        "raw": dict(classification(labels, out["raw"])),
        "head_zone": dict(classification(labels[:, 0], decisions[:, :, 0])),
        "body_zone": dict(classification(labels[:, 1], decisions[:, :, 1])),
        "angled_frames": int(angled.sum()),
        "angled_10_to_20_deg": dict(classification(labels[angled], decisions[:, angled])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=Path)
    parser.add_argument("pose_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    for filename in ("summary.json", "decisions.npz"):
        if (args.out_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite {args.out_dir / filename}.")
    self_check()
    start = perf_counter()
    files = [p for p in sorted(args.grid_dir.glob("*.npz")) if "_10s_sample_" not in p.name]
    if len(files) != 12:
        raise ValueError(f"Expected 12 known development sequences, found {len(files)}.")
    outputs = {p.stem: evaluate_sequence(p, args.pose_dir) for p in files}
    axes_by_key = {"raw": 1, "alerts": 1, "headings": 1, "used_translation": 1}
    both = {
        key: np.concatenate([out[key] for out in outputs.values()], axis=axes_by_key.get(key, 0))
        for key in next(iter(outputs.values()))
    }
    report = {
        "combined": summarize(both),
        "sequences": {name: summarize(out) for name, out in outputs.items()},
        "wall_seconds": round(perf_counter() - start, 3),
    }
    np.savez_compressed(args.out_dir / "decisions.npz", **both)
    with (args.out_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report["combined"], indent=2))
    print(f"Wall seconds: {report['wall_seconds']}")


if __name__ == "__main__":
    main()
