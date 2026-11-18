"""Verify the LIN reference from registered phone depth before joining any predictions.

From the checkout root, run python -B research/active/hardware-bringup/host/independent_low_lamar_reference_20261118.py.
The consumed LIN input, source acquisition steps and human-label ledgers are under
artifacts.local/work/independent-low-20261118/alt-native/. No network or GPU is used here.
"""
import argparse
import csv
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HOST = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST.parent / "evalkit"))
import core

DATA = HOST.parents[3] / "artifacts.local/work/independent-low-20261118/alt-native"
EXPECTED_MANIFEST = "e8e71169a773739eeb2279d051c2cf4d1431bcfe4857d52be1ead93154d8a1f6"
STAIRS = (129546927471, 15553513134, 3265202328)


def read(path):
    return json.loads(path.read_bytes())


@lru_cache
def cameras(folder, name):
    with (folder / name / "sensors.txt").open(encoding="utf-8-sig") as f:
        table = csv.reader((line for line in f if line and not line.startswith("#")), skipinitialspace=True)
        return {r[0]: tuple(map(float, r[4:10])) for r in table if len(r) >= 10 and r[3] == "PINHOLE"}


def check_plane(folder, r):
    group = r["attributes"].get("object_clip", "")
    if not group.startswith("ascending_stair_"):
        return
    anchor = STAIRS[int(group[-1])-1]
    plane = read(folder / "case_windows" / str(anchor) / "landing/plane.json")
    n = np.array(plane["world_up_plane_normal"])
    pose = np.array(r["T_local"])
    assert np.allclose(r["ground"][0], pose[:3, :3].T @ n)
    assert np.isclose(r["ground"][1], pose[:3, 3] @ n + plane["world_bias"])


def verify(folder):
    manifest_path = folder / "frozen/manifest.json"
    assert sha256(manifest_path.read_bytes()).hexdigest() == EXPECTED_MANIFEST
    records = read(manifest_path)["frames"]
    control = read(folder / "frozen/controls.json")
    cohort = read(folder / "frozen/source.json")
    control_rows = {(r["seq"], r["ts"]): r for r in control["ledger"]}
    seen = set()
    coverage = {1.5: 0, 2.5: 0}
    step_rows = 0
    for r in records:
        seq, ts = r["seq"], r["timestamp_us"]
        assert (seq, ts) not in seen and r["eye_height"] == cohort["eye_height"][seq]
        seen.add((seq, ts))
        k = np.array(r["K"])
        depth = cv2.imread(str(folder / r["depth"]), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(folder / r["confidence"]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(folder / r["rgb"]), cv2.IMREAD_REDUCED_COLOR_2)
        assert rgb is not None and depth.dtype == np.uint16 and confidence.shape == depth.shape
        assert set(np.unique(confidence)).issubset({0, 1, 2})
        h, w = depth.shape
        assert r["size"] == [w, h] and r["rgb_native_size"] == [2*rgb.shape[1], 2*rgb.shape[0]]
        for prefix, key, size in [("depth_phone", "K", r["size"]), ("cam_phone", "rgb_K", r["rgb_native_size"])]:
            width, height, fx, fy, cx, cy = cameras(folder, seq)[f"{prefix}_{ts}"]
            assert size == [width, height] and np.allclose(r[key], [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        assert np.allclose(np.asarray(r["T_local"])[:3, :3].T @ [0., -1., 0.], r["down"])
        check_plane(folder, r)
        z = depth.astype(np.float32)*.001
        z[(depth == 0) | (confidence < r["min_confidence"])] = np.nan
        n, height = r["ground"]
        frame = core.Frame("lamar_lin", seq, ts, ts/1.e6, None, k, z, "sparse_geometry", np.array(r["down"]),
                           r["eye_height"], "handheld", extra={"ground": (np.array(n), height)})
        reg = core.regions(frame, None)
        assert {b: reg["truth_"+b] for b in core.BANDS} == r["geometry_truth"]
        if (seq, ts) not in control_rows:
            step_rows += r["attributes"]["object_clip"].startswith("ascending_stair")
            continue
        reference = control_rows[seq, ts]
        assert reference["blind_route_pass"] and reference["truth"] == r["geometry_truth"]
        # Double precision matches the coverage ledger. The v1.2 band routine above uses float32 depth.
        metres = depth.astype(float)*.001
        metres[(depth == 0) | (confidence < 1)] = np.nan
        pts = core.rays(k, h, w)*metres.reshape(-1, 1)
        down, forward, right = reg["axes"]
        along, lateral = pts@forward, pts@right
        surface = np.abs(height - pts@down) <= .05
        for far in coverage:
            good = surface & (along >= .3) & (along <= far)
            counts = [int(np.sum(good & (lateral >= left) & (lateral < edge))) for left, edge in
                      zip([-.35, -.35/3, .35/3], [-.35/3, .35/3, .350001])]
            assert counts == reference["near_floor_counts"][str(far)]
            coverage[far] += min(counts) > 0 and all(reg["truth_"+b] == 0 for b in core.BANDS)
    assert len(records) == len(seen) == 243 and len(control_rows) == 351 and step_rows == 81
    assert coverage == {1.5: 56, 2.5: 127}
    return {"manifest_sha256": EXPECTED_MANIFEST, "unique_frames": len(seen),
            "original_random_frames": len(control_rows), "strict_empty_with_three_width_strips": coverage}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA)
    print(json.dumps(verify(parser.parse_args().source)))
