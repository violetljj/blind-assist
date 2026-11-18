"""Screen Oxford Spires geometry and original RGB without using D8 scores."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import spires_insert_dev_20261117 as camera
import spires_cane_reach_dev_20261117 as geometry

WORK = Path("artifacts.local/work/spires-cane-reach-20261117")
DATA = Path("artifacts.local/downloads/oxford-spires-20261117")
OUT = Path("artifacts.local/work/real-wall-search-20261118/spires")


def screen_sequence(grid, seq, labels):
    trajectory = camera.Traj(DATA / "seq" / seq / "gt-tum.txt")
    times = np.arange(trajectory.t[0], trajectory.t[-1], 1 / geometry.HZ)
    poses = trajectory.T(times)
    files = sorted((WORK / "insert/img" / seq).glob("*.jpg"))
    rgb_times = np.array([float(p.stem) for p in files])
    available = {}
    for path, t in zip(files, rgb_times):
        k = int(round((t - times[0]) * geometry.HZ))
        if 0 <= k < len(times) and abs(times[k] - t) <= .025:
            available[k] = path
    rows = []
    for k, (pose, height, speed) in enumerate(zip(poses, labels["dev_h"], labels["speed"])):
        if height is None or speed < .30 or not np.isfinite(height):
            continue
        floor = np.interp(times[k], trajectory.t, trajectory.P[:, 2]) - height
        origin = pose[:3, 3]
        forward = camera.heading(pose)
        if forward is None:
            continue
        points = grid.c[grid.near(origin[:2], 2.5)]
        delta = points - origin
        along = delta @ forward
        above = points[:, 2] - floor
        zone = (along >= .30) & (along <= 2.) & (above >= .60) & (above <= 1.90)
        lateral = delta[zone] @ np.cross(forward, camera.A.UP)
        if (abs(lateral) <= .25).any():
            continue
        side = (abs(lateral) >= .35) & (abs(lateral) <= 1.20)
        tag = "none"
        texture = None
        if side.any():
            q = delta[zone][side] @ pose[:3, :3]
            z = q[:, 2].clip(.01)
            u = np.rint(camera.CX + camera.F*q[:, 0]/z).astype(int)
            v = np.rint(camera.CY + camera.F*q[:, 1]/z).astype(int)
            seen = (q[:, 2] > .1) & (u >= 0) & (u < camera.W) & (v >= 0) & (v < camera.H)
            h = above[zone][side]
            lat = lateral[side]
            masks = [seen & (s*lat > 0) for s in (-1, 1)]
            sufficient = [m for m in masks if len(np.unique(np.floor((h[m]-.60)/.1))) >= 10]
            if not sufficient:
                continue
            tag = "side"
            if k in available:
                mask = np.logical_or.reduce(sufficient)
                pix = np.unique(v[mask]*camera.W + u[mask])
                im = camera.pinhole_rgb(available[k])
                gray = cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),1.)
                gradient = np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
                texture = float(np.median(gradient.ravel()[pix]))
        rows.append(dict(k=k, t=float(times[k]), group=tag, rgb=k in available, texture=texture))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = json.loads((WORK / "labels.json").read_text())
    for site in geometry.SITES:
        with np.load(WORK / "vox" / f"{site}.npz") as vox:
            grid = geometry.Grid(vox["keys"], vox["counts"])
        for seq, label in labels.items():
            if geometry.site_of(seq) != site:
                continue
            rows = screen_sequence(grid, seq, label)
            (OUT / f"{seq}.json").write_text(json.dumps(rows))
            print(seq, Counter((r["group"],r["rgb"]) for r in rows), flush=True)
        del grid


if __name__ == "__main__":
    main()
