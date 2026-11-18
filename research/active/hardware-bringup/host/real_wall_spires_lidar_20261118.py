"""Check same-walk LiDAR coverage of the Oxford Spires TLS-empty camera corridor at native exposure times."""
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import HfFileSystem
from scipy.spatial.transform import Rotation, Slerp

import spires_insert_dev_20261117 as cam

WORK = Path("artifacts.local/work/spires-cane-reach-20261117")
DATA = Path("artifacts.local/downloads/oxford-spires-20261117")
SCREEN = Path("artifacts.local/work/real-wall-search-20261118/spires")
OUT = SCREEN / "lidar"
SOURCE = "datasets/ori-drs/oxford_spires_dataset/sequences/{}/raw/{}"
DT = np.dtype(dict(names=["x", "y", "z", "ts", "ring"], formats=["<f4"]*3+["<f8", "<u2"],
                   offsets=[0, 4, 8, 24, 32], itemsize=48))
SLICES = np.arange(3, 21) / 10.


def zip_members(fs, seq, filename, suffix):
    with fs.open(SOURCE.format(seq, filename), "rb", block_size=1 << 20) as stream, zipfile.ZipFile(stream) as arc:
        return [f.filename for f in arc.infolist() if f.filename.endswith(suffix)]


def sample(rows):
    selected = []
    for group in ("side", "none"):
        subset = [r for r in rows if r["group"] == group]
        indices = np.linspace(0, len(subset)-1, min(len(subset), 4)).round().astype(int)
        selected.extend(subset[k] for k in indices)
    return selected


def check(cloud, seq_traj, camera_traj, exposure, floor):
    header, body = cloud.split(b"DATA binary\n", 1)
    assert b"FIELDS x y z _ intensity _ timestamp ring _\n" in header
    assert b"COUNT 1 1 1 4 1 4 1 1 14\n" in header
    count = int(header.split(b"POINTS ")[1].splitlines()[0])
    points = np.frombuffer(body, dtype=DT, count=count)
    ts = points["ts"]
    if not ts.min() <= exposure <= ts.max():
        return dict(status="UNKNOWN_NO_SWEEP", dt_ms=(exposure-float(ts.min()))*1000)
    traj, interpolate_rotation = seq_traj
    if ts.min() < traj[0, 0] or ts.max() > traj[-1, 0]:
        return dict(status="UNKNOWN_NO_POSE")
    rot = interpolate_rotation(ts)
    positions = np.stack([np.interp(ts, traj[:, 0], v) for v in traj[:, 1:4].T], 1)
    origin = positions + rot.apply([0, 0, .124])
    ray = rot.apply(np.column_stack((-points["x"], -points["y"], points["z"])))
    T = camera_traj.T(exposure)[0]
    forward = cam.heading(T)
    axes = np.stack((forward, np.cross(forward, [0, 0, 1]), [0, 0, 1]))
    o = (origin - T[:3, 3]) @ axes.T
    o[:, 2] += T[2, 3] - floor
    ray = ray @ axes.T
    p = o + ray
    inside = (p[:, 0] >= .30) & (p[:, 0] <= 2) & (abs(p[:, 1]) <= .25) & (p[:, 2] >= .60) & (p[:, 2] <= 1.90)
    crossings = []
    for depth in SLICES:
        ok = (ray[:, 0] > 0) & (p[:, 0] >= depth + .05) & (o[:, 0] <= depth)
        cross = o[ok] + ray[ok] * ((depth - o[ok, 0]) / ray[ok, 0])[:, None]
        lateral = np.floor((cross[:, 1] + .25) / .10).astype(int)
        height = np.floor((cross[:, 2] - .60) / .10).astype(int)
        good = (lateral >= 0) & (lateral < 5) & (height >= 0) & (height < 13)
        crossings.append(int(np.unique(lateral[good]*13 + height[good]).size))
    status = "POTENTIAL_OCCUPIED" if inside.any() else ("COARSE_CLEAR" if min(crossings) == 65 else "UNKNOWN_COVERAGE")
    return dict(status=status, points=count, body_endpoints=int(inside.sum()), cells_of_65=crossings,
                unknown_dynamic_08m=True)


def sequence(args):
    file, heights = args
    seq = file.stem
    target = OUT / (seq + ".json")
    if target.exists():
        return json.loads(target.read_text())
    selected = sample(json.loads(file.read_bytes()))
    fs = HfFileSystem()
    cache = WORK / "insert/cam0" / (seq + ".json")
    names = json.loads(cache.read_bytes()) if cache.exists() else [
        p for p in zip_members(fs, seq, "images.zip", ".jpg") if p.startswith("cam0/")]
    rgb_times = np.sort([float(Path(p).stem) for p in names])
    gt = DATA / "seq" / seq / "gt-tum.txt"
    traj = np.loadtxt(gt)
    rotation = Slerp(traj[:, 0], Rotation.from_quat(traj[:, 4:8]))
    cameras = cam.Traj(gt)
    rows = []
    with fs.open(SOURCE.format(seq, "lidar-clouds.zip"), "rb", block_size=1 << 20) as stream, zipfile.ZipFile(stream) as arc:
        entries = sorted((float(Path(f.filename).stem), f) for f in arc.infolist() if f.filename.endswith(".pcd") and not f.is_dir())
        starts = np.array([t for t,_ in entries])
        for r in selected:
            t = r["t"]
            rgb = rgb_times[np.argmin(abs(rgb_times-t))]
            idx = np.searchsorted(starts, rgb, side="right") - 1
            decision = dict(status="UNKNOWN_NO_RGB_OR_SCAN")
            if abs(t-rgb) <= .025 and idx >= 0 and rgb-starts[idx] < .105:
                key = entries[idx][1]
                floor = np.interp(t, traj[:, 0], traj[:, 3]) - heights[r["k"]]
                decision = check(arc.read(key), (traj, rotation), cameras, rgb, floor)
                decision.update(sweep=key.filename, compressed_size=key.compress_size)
            decision.update(k=r["k"], group=r["group"], exposure=rgb, t=t)
            rows.append(decision)
    target.write_text(json.dumps(rows, indent=1))
    print(seq, len(rows), [(r["status"],r.get("cells_of_65",[None])[0]) for r in rows], flush=True)
    return rows


def main():
    labels = json.loads((WORK / "labels.json").read_bytes())
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(sequence, [(f, labels[f.stem]["dev_h"]) for f in sorted(SCREEN.glob("2024*.json"))]))
    from collections import Counter
    allrows = [r for p in parts for r in p]
    report = dict(rows=len(allrows), groups={g:dict(Counter(r["status"] for r in allrows if r["group"] == g)) for g in ("side", "none")},
                  pcd_payload_bytes=sum(r.get("compressed_size", 0) for r in allrows))
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
